"""Unit tests for Qwen3TextEmbedding model registration, configuration, and post-processing."""

import inspect

import numpy as np
import pytest
from tokenizers import Tokenizer

from fastretrieval.common.onnx_model import OnnxOutputContext
from fastretrieval.text.qwen3_embedding import (
    DEFAULT_TASK,
    QUERY_INSTRUCTION_TEMPLATE,
    Qwen3TextEmbedding,
    Qwen3TextEmbeddingWorker,
    supported_qwen3_models,
)
from fastretrieval.text.text_embedding import TextEmbedding
from tests.onnx_runtime_stub import StubSession, install_stub_runtime, write_tokenizer_files


class TestQwen3TextEmbeddingRegistry:
    """Verify Qwen3 embedding models are properly registered."""

    def test_qwen3_in_registry(self):
        """Qwen3TextEmbedding should be in the TextEmbedding registry."""
        assert Qwen3TextEmbedding in TextEmbedding.EMBEDDINGS_REGISTRY

    def test_qwen3_models_listed(self):
        """Qwen3 models should appear in list_supported_models."""
        models = TextEmbedding.list_supported_models()
        qwen3_models = [m for m in models if "Qwen3" in m["model"]]
        assert len(qwen3_models) >= 1

    def test_qwen3_model_description(self):
        """Verify Qwen3-Embedding-0.6B model description fields."""
        desc = supported_qwen3_models[0]
        assert desc.model == "n24q02m/Qwen3-Embedding-0.6B-ONNX"
        assert desc.dim == 1024
        assert desc.license == "apache-2.0"
        assert "last-token" in desc.description.lower() or "MRL" in desc.description

    def test_default_model_is_qwen3(self):
        """TextEmbedding default model should be Qwen3."""
        sig = inspect.signature(TextEmbedding.__init__)
        default_model = sig.parameters["model_name"].default
        assert default_model == "n24q02m/Qwen3-Embedding-0.6B-ONNX"


class TestQwen3InstructionFormat:
    """Verify instruction template formatting."""

    def test_query_instruction_format(self):
        result = QUERY_INSTRUCTION_TEMPLATE.format(task="Find docs", text="test query")
        assert result == "Instruct: Find docs\nQuery: test query"

    def test_default_task_not_empty(self):
        assert len(DEFAULT_TASK) > 0
        assert "retrieve" in DEFAULT_TASK.lower()


class TestQwen3PostProcessing:
    """Test _post_process_onnx_output with mocked ONNX output."""

    def test_post_process_last_token_pooling(self):
        """Should extract last-token embeddings and normalise."""
        model_output = np.array(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.0, 0.0, 0.0, 0.0]],
                [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6], [1.7, 1.8, 1.9, 2.0]],
            ]
        )
        attention_mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
        output = OnnxOutputContext(model_output=model_output, attention_mask=attention_mask)

        result = np.array(
            list(Qwen3TextEmbedding._post_process_onnx_output(None, output))  # type: ignore
        )

        assert result.shape == (2, 4)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_post_process_mrl_truncation(self):
        """Passing dim= should truncate before normalisation."""
        model_output = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
        attention_mask = np.array([[1, 1]], dtype=np.int64)
        output = OnnxOutputContext(model_output=model_output, attention_mask=attention_mask)

        result = np.array(
            list(Qwen3TextEmbedding._post_process_onnx_output(None, output, dim=2))  # type: ignore
        )
        assert result.shape == (1, 2)
        np.testing.assert_allclose(np.linalg.norm(result[0]), 1.0, atol=1e-6)

    def test_post_process_raises_without_attention_mask(self):
        """Should raise if attention_mask is None."""
        output = OnnxOutputContext(model_output=np.zeros((1, 3, 4)), attention_mask=None)
        with pytest.raises(ValueError, match="attention_mask"):
            list(Qwen3TextEmbedding._post_process_onnx_output(None, output))  # type: ignore


_QWEN3_EMBED_MODEL = "n24q02m/Qwen3-Embedding-0.6B-ONNX"
_PROMPT_WORDS = ["alpha", "beta", "Instruct", "Query", ":", "Find"]


def _hidden_state_from_token_ids(feed: dict[str, np.ndarray]) -> np.ndarray:
    """Every token's hidden state is ``[token_id, 1, 2, 2]``, so pooling is easy to predict."""
    ids = feed["input_ids"].astype(np.float32)
    ones = np.ones_like(ids)
    return np.stack([ids, ones, 2 * ones, 2 * ones], axis=-1)


def _unit(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


@pytest.fixture
def stub_qwen3(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    vocab = write_tokenizer_files(model_dir, _PROMPT_WORDS)
    session = StubSession(_hidden_state_from_token_ids)
    # onnxruntime boundary: no CPython 3.15 wheel exists, so the session is a numpy stand-in.
    opened = install_stub_runtime(monkeypatch, session)
    model = Qwen3TextEmbedding(
        _QWEN3_EMBED_MODEL, cache_dir=str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )
    return model, session, vocab, model_dir, opened


class TestQwen3TextEmbeddingThroughRuntime:
    def test_embed_runs_one_document_per_call_and_pools_the_last_token(self, stub_qwen3):
        model, session, vocab, model_dir, opened = stub_qwen3

        embeddings = list(model.embed(["alpha beta", "beta alpha"], batch_size=64))

        assert opened == [str(model_dir / "onnx" / "model_quantized.onnx")]
        assert [feed["input_ids"].shape[0] for feed in session.feeds] == [1, 1]
        np.testing.assert_allclose(embeddings[0], _unit([vocab["beta"], 1, 2, 2]), rtol=1e-6)
        np.testing.assert_allclose(embeddings[1], _unit([vocab["alpha"], 1, 2, 2]), rtol=1e-6)

    def test_embed_truncates_to_the_requested_dimension_before_normalizing(self, stub_qwen3):
        model, _, vocab, _, _ = stub_qwen3

        (embedding,) = list(model.embed("alpha beta", dim=2))

        np.testing.assert_allclose(embedding, _unit([vocab["beta"], 1]), rtol=1e-6)

    def test_query_embed_prefixes_a_single_query_with_the_task_instruction(self, stub_qwen3):
        model, session, vocab, _, _ = stub_qwen3

        (embedding,) = list(model.query_embed("alpha", task="Find"))

        expected_ids = [vocab[w] for w in ["Instruct", ":", "Find", "Query", ":", "alpha"]]
        assert session.feeds[0]["input_ids"].tolist() == [expected_ids]
        np.testing.assert_allclose(embedding, _unit([vocab["alpha"], 1, 2, 2]), rtol=1e-6)

    def test_query_embed_prefixes_every_query_of_an_iterable(self, stub_qwen3):
        model, session, vocab, _, _ = stub_qwen3

        embeddings = list(model.query_embed(iter(["alpha", "beta"]), task="Find"))

        prefix = [vocab[w] for w in ["Instruct", ":", "Find", "Query", ":"]]
        assert [feed["input_ids"].tolist() for feed in session.feeds] == [
            [prefix + [vocab["alpha"]]],
            [prefix + [vocab["beta"]]],
        ]
        assert len(embeddings) == 2

    def test_query_embed_uses_the_default_retrieval_task(self, stub_qwen3):
        model, session, _, model_dir, _ = stub_qwen3
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

        list(model.query_embed("alpha"))

        prompt = QUERY_INSTRUCTION_TEMPLATE.format(task=DEFAULT_TASK, text="alpha")
        assert session.feeds[0]["input_ids"].tolist() == [tokenizer.encode(prompt).ids]

    def test_passage_embed_embeds_passages_without_an_instruction(self, stub_qwen3):
        model, session, vocab, _, _ = stub_qwen3

        embeddings = list(model.passage_embed(["alpha", "beta"]))

        assert [feed["input_ids"].tolist() for feed in session.feeds] == [
            [[vocab["alpha"]]],
            [[vocab["beta"]]],
        ]
        np.testing.assert_allclose(embeddings[1], _unit([vocab["beta"], 1, 2, 2]), rtol=1e-6)

    def test_worker_builds_a_single_threaded_qwen3_model(self, stub_qwen3, tmp_path):
        _, _, _, model_dir, _ = stub_qwen3

        worker = Qwen3TextEmbeddingWorker(
            _QWEN3_EMBED_MODEL, str(tmp_path / "cache"), specific_model_path=str(model_dir)
        )
        ((idx, output),) = list(worker.process([(5, ["alpha beta"])]))

        assert Qwen3TextEmbedding._get_worker_class() is Qwen3TextEmbeddingWorker
        assert type(worker.model) is Qwen3TextEmbedding
        assert worker.model.threads == 1
        assert idx == 5
        assert output.model_output.shape == (1, 2, 4)
