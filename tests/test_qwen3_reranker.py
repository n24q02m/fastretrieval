"""Unit tests for Qwen3CrossEncoder model registration and scoring logic."""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fastretrieval.common.onnx_model import OnnxOutputContext
from fastretrieval.rerank.cross_encoder.qwen3_cross_encoder import (
    DEFAULT_INSTRUCTION,
    SYSTEM_PROMPT,
    TOKEN_NO_ID,
    TOKEN_YES_ID,
    Qwen3CrossEncoder,
    Qwen3CrossEncoderWorker,
    supported_qwen3_reranker_models,
)
from fastretrieval.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder
from tests.onnx_runtime_stub import StubSession, install_stub_runtime, write_tokenizer_files


class TestQwen3CrossEncoderRegistry:
    """Verify Qwen3 reranker models are properly registered."""

    def test_qwen3_in_registry(self):
        """Qwen3CrossEncoder should be in the TextCrossEncoder registry."""
        assert Qwen3CrossEncoder in TextCrossEncoder.CROSS_ENCODER_REGISTRY

    def test_qwen3_models_listed(self):
        """Qwen3 reranker models should appear in list_supported_models."""
        models = TextCrossEncoder.list_supported_models()
        qwen3_models = [m for m in models if "Qwen3" in m["model"]]
        assert len(qwen3_models) >= 3

    def test_qwen3_yesno_model_in_registry(self):
        """Optimized YesNo model should be in the registry."""
        models = TextCrossEncoder.list_supported_models()
        yesno_models = [m for m in models if "YesNo" in m["model"]]
        assert len(yesno_models) == 1
        assert yesno_models[0]["model"] == "n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo"

    def test_qwen3_reranker_description(self):
        """Verify Qwen3-Reranker-0.6B model description fields."""
        desc = supported_qwen3_reranker_models[0]
        assert desc.model == "n24q02m/Qwen3-Reranker-0.6B-ONNX"
        assert desc.license == "apache-2.0"
        assert "yes/no" in desc.description.lower() or "causal" in desc.description.lower()


class TestQwen3ChatTemplate:
    """Verify chat template formatting for reranking."""

    def test_format_rerank_input(self):
        result = Qwen3CrossEncoder._format_rerank_input(
            query="What is AI?",
            document="AI is artificial intelligence.",
            instruction=DEFAULT_INSTRUCTION,
        )
        assert "<|im_start|>system" in result
        assert SYSTEM_PROMPT in result
        assert "<Instruct>:" in result
        assert "<Query>: What is AI?" in result
        assert "<Document>: AI is artificial intelligence." in result
        assert "<|im_end|>" in result
        assert "<think>" in result
        assert "</think>" in result

    def test_custom_instruction(self):
        result = Qwen3CrossEncoder._format_rerank_input(
            query="q",
            document="d",
            instruction="Custom task instruction",
        )
        assert "<Instruct>: Custom task instruction" in result


class TestYesNoScoring:
    """Test the yes/no softmax scoring logic."""

    def test_strong_yes(self):
        """When yes-logit >> no-logit, score should be close to 1.0."""
        # (batch=1, seq_len=3, vocab_size=20000)
        vocab_size = 20000
        output = np.zeros((1, 3, vocab_size), dtype=np.float32)
        output[0, -1, TOKEN_YES_ID] = 10.0
        output[0, -1, TOKEN_NO_ID] = -10.0

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert scores.shape == (1,)
        assert scores[0] > 0.99

    def test_strong_no(self):
        """When no-logit >> yes-logit, score should be close to 0.0."""
        vocab_size = 20000
        output = np.zeros((1, 3, vocab_size), dtype=np.float32)
        output[0, -1, TOKEN_YES_ID] = -10.0
        output[0, -1, TOKEN_NO_ID] = 10.0

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert scores[0] < 0.01

    def test_equal_logits(self):
        """When yes==no logits, score should be 0.5."""
        vocab_size = 20000
        output = np.zeros((1, 3, vocab_size), dtype=np.float32)
        output[0, -1, TOKEN_YES_ID] = 5.0
        output[0, -1, TOKEN_NO_ID] = 5.0

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        np.testing.assert_allclose(scores[0], 0.5, atol=1e-6)

    def test_batch_scoring(self):
        """Batch of 3 samples with varying relevance."""
        vocab_size = 20000
        output = np.zeros((3, 5, vocab_size), dtype=np.float32)

        # Sample 0: strong yes
        output[0, -1, TOKEN_YES_ID] = 10.0
        output[0, -1, TOKEN_NO_ID] = -10.0

        # Sample 1: strong no
        output[1, -1, TOKEN_YES_ID] = -10.0
        output[1, -1, TOKEN_NO_ID] = 10.0

        # Sample 2: neutral
        output[2, -1, TOKEN_YES_ID] = 0.0
        output[2, -1, TOKEN_NO_ID] = 0.0

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert scores.shape == (3,)
        assert scores[0] > 0.99
        assert scores[1] < 0.01
        np.testing.assert_allclose(scores[2], 0.5, atol=1e-6)

    def test_numerical_stability(self):
        """Large logit values should not cause overflow."""
        vocab_size = 20000
        output = np.zeros((1, 3, vocab_size), dtype=np.float32)
        output[0, -1, TOKEN_YES_ID] = 1000.0
        output[0, -1, TOKEN_NO_ID] = 999.0

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert not np.isnan(scores[0])
        assert not np.isinf(scores[0])
        assert 0.5 < scores[0] < 1.0


class TestOptimizedYesNoScoring:
    """Test scoring with optimized (batch, 2) output shape."""

    def test_optimized_strong_yes(self):
        """Optimized model output (batch, 2) with [no, yes] logits."""
        output = np.array([[-10.0, 10.0]], dtype=np.float32)  # (1, 2)
        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert scores.shape == (1,)
        assert scores[0] > 0.99

    def test_optimized_strong_no(self):
        output = np.array([[10.0, -10.0]], dtype=np.float32)  # (1, 2)
        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert scores[0] < 0.01

    def test_optimized_equal(self):
        output = np.array([[5.0, 5.0]], dtype=np.float32)  # (1, 2)
        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        np.testing.assert_allclose(scores[0], 0.5, atol=1e-6)

    def test_optimized_batch(self):
        output = np.array([[-10.0, 10.0], [10.0, -10.0], [0.0, 0.0]], dtype=np.float32)  # (3, 2)
        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert scores.shape == (3,)
        assert scores[0] > 0.99
        assert scores[1] < 0.01
        np.testing.assert_allclose(scores[2], 0.5, atol=1e-6)

    def test_optimized_numerical_stability(self):
        output = np.array([[999.0, 1000.0]], dtype=np.float32)  # (1, 2)
        scores = Qwen3CrossEncoder._compute_yes_no_scores(output)
        assert not np.isnan(scores[0])
        assert not np.isinf(scores[0])
        assert 0.5 < scores[0] < 1.0

    def test_optimized_matches_legacy(self):
        """Optimized and legacy outputs should produce identical scores."""
        vocab_size = 20000
        legacy_output = np.zeros((1, 3, vocab_size), dtype=np.float32)
        legacy_output[0, -1, TOKEN_YES_ID] = 3.5
        legacy_output[0, -1, TOKEN_NO_ID] = 1.2

        optimized_output = np.array([[1.2, 3.5]], dtype=np.float32)  # [no, yes]

        legacy_scores = Qwen3CrossEncoder._compute_yes_no_scores(legacy_output)
        optimized_scores = Qwen3CrossEncoder._compute_yes_no_scores(optimized_output)
        np.testing.assert_allclose(legacy_scores, optimized_scores, atol=1e-6)


class TestTokenConstants:
    """Verify token ID constants."""

    def test_token_ids_are_positive(self):
        assert TOKEN_YES_ID > 0
        assert TOKEN_NO_ID > 0

    def test_token_ids_are_different(self):
        assert TOKEN_YES_ID != TOKEN_NO_ID


@pytest.fixture
def mocked_qwen3_encoder():
    """Returns a Qwen3CrossEncoder with mocked model and tokenizer."""
    with patch(
        "fastretrieval.rerank.cross_encoder.qwen3_cross_encoder.Qwen3CrossEncoder.download_model",
        return_value="/tmp/mock",
    ):
        encoder = Qwen3CrossEncoder("n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo", lazy_load=True)

    encoder.model = MagicMock()

    def mock_run(output_names, onnx_input):
        batch_size = onnx_input["input_ids"].shape[0]
        return [np.array([[-10.0, 10.0]] * batch_size, dtype=np.float32)]

    encoder.model.run.side_effect = mock_run
    encoder.model_input_names = {"input_ids", "attention_mask"}

    mock_tokenizer = MagicMock()
    mock_encoding = MagicMock()
    mock_encoding.ids = [1, 2, 3]
    mock_encoding.attention_mask = [1, 1, 1]

    # encode_batch returns a list of encodings, one for each text.
    def mock_encode_batch(texts):
        return [mock_encoding for _ in texts]

    mock_tokenizer.encode_batch.side_effect = mock_encode_batch
    encoder.tokenizer = mock_tokenizer

    return encoder


class TestQwen3CrossEncoderInference:
    """Verify ONNX embedding methods override."""

    def test_onnx_embed_texts_missing_model(self):
        with patch(
            "fastretrieval.rerank.cross_encoder.qwen3_cross_encoder.Qwen3CrossEncoder.download_model",
            return_value="/tmp/mock",
        ):
            encoder = Qwen3CrossEncoder("n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo", lazy_load=True)
        encoder.model = None
        with pytest.raises(ValueError, match="Model not loaded"):
            encoder._onnx_embed_texts(["text1"])

    def test_onnx_embed_texts_missing_tokenizer(self):
        with patch(
            "fastretrieval.rerank.cross_encoder.qwen3_cross_encoder.Qwen3CrossEncoder.download_model",
            return_value="/tmp/mock",
        ):
            encoder = Qwen3CrossEncoder("n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo", lazy_load=True)
        encoder.model = MagicMock()
        encoder.tokenizer = None
        with pytest.raises(AssertionError, match="Tokenizer not loaded"):
            encoder._onnx_embed_texts(["text1"])

    def test_onnx_embed_texts_success(self, mocked_qwen3_encoder):
        ctx = mocked_qwen3_encoder._onnx_embed_texts(["hello world"])
        assert isinstance(ctx, OnnxOutputContext)
        assert ctx.model_output.shape == (1,)
        # -10.0, 10.0 should give high probability
        assert ctx.model_output[0] > 0.99
        mocked_qwen3_encoder.model.run.assert_called_once()
        mocked_qwen3_encoder.tokenizer.encode_batch.assert_called_once_with(["hello world"])

    def test_onnx_embed_texts_scores_per_row(self, mocked_qwen3_encoder):
        """Issue #725: each text is scored on its own (batch=1) so a score
        cannot depend on which other documents share the batch. Three texts
        therefore produce three model runs, each with batch dimension 1."""
        ctx = mocked_qwen3_encoder._onnx_embed_texts(["a", "b", "c"])
        assert ctx.model_output.shape == (3,)
        assert mocked_qwen3_encoder.model.run.call_count == 3
        for call in mocked_qwen3_encoder.model.run.call_args_list:
            onnx_input = call[0][1]
            assert onnx_input["input_ids"].shape[0] == 1

    def test_onnx_embed_texts_multiple(self, mocked_qwen3_encoder):
        # Per-row scoring: one model run + one tokenisation per text (issue #725).
        ctx = mocked_qwen3_encoder._onnx_embed_texts(["text1", "text2"])
        assert ctx.model_output.shape == (2,)
        assert mocked_qwen3_encoder.model.run.call_count == 2
        assert mocked_qwen3_encoder.tokenizer.encode_batch.call_count == 2

    def test_onnx_embed_pairs_success(self, mocked_qwen3_encoder):
        ctx = mocked_qwen3_encoder.onnx_embed_pairs([("Query1", "Doc1"), ("Query2", "Doc2")])
        assert ctx.model_output.shape == (2,)
        assert mocked_qwen3_encoder.model.run.call_count == 2
        assert mocked_qwen3_encoder.tokenizer.encode_batch.call_count == 2

        # Verify chat template formatting (one encode_batch([text]) call per pair).
        calls = mocked_qwen3_encoder.tokenizer.encode_batch.call_args_list
        text1 = calls[0][0][0][0]
        assert "<Query>: Query1" in text1
        assert "<Document>: Doc1" in text1
        text2 = calls[1][0][0][0]
        assert "<Query>: Query2" in text2
        assert "<Document>: Doc2" in text2

    def test_onnx_embed_success(self, mocked_qwen3_encoder):
        ctx = mocked_qwen3_encoder.onnx_embed("Query", ["Doc1", "Doc2"])
        assert ctx.model_output.shape == (2,)
        assert mocked_qwen3_encoder.model.run.call_count == 2
        assert mocked_qwen3_encoder.tokenizer.encode_batch.call_count == 2

        calls = mocked_qwen3_encoder.tokenizer.encode_batch.call_args_list
        text1 = calls[0][0][0][0]
        assert "<Query>: Query" in text1
        assert "<Document>: Doc1" in text1

        text2 = calls[1][0][0][0]
        assert "<Query>: Query" in text2
        assert "<Document>: Doc2" in text2

    def test_onnx_embed_custom_instruction(self, mocked_qwen3_encoder):
        ctx = mocked_qwen3_encoder.onnx_embed("Query", ["Doc"], instruction="Custom Instruction!")
        assert ctx.model_output.shape == (1,)

        calls = mocked_qwen3_encoder.tokenizer.encode_batch.call_args_list
        text1 = calls[0][0][0][0]
        assert "<Instruct>: Custom Instruction!" in text1

    def test_onnx_embed_pairs_custom_instruction(self, mocked_qwen3_encoder):
        ctx = mocked_qwen3_encoder.onnx_embed_pairs(
            [("Query", "Doc")], instruction="Custom Pair Instruction!"
        )
        assert ctx.model_output.shape == (1,)

        calls = mocked_qwen3_encoder.tokenizer.encode_batch.call_args_list
        text1 = calls[0][0][0][0]
        assert "<Instruct>: Custom Pair Instruction!" in text1


_YESNO_MODEL = "n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo"
_FULL_VOCAB = max(TOKEN_YES_ID, TOKEN_NO_ID) + 1


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _alpha_relevance_session(alpha_id: int, input_names: tuple[str, ...]) -> StubSession:
    """A YesNo graph whose "yes" logit counts the "alpha" tokens and whose "no" logit is 0."""

    def compute(feed: dict[str, np.ndarray]) -> np.ndarray:
        yes = (feed["input_ids"] == alpha_id).sum(axis=1).astype(np.float32)
        return np.stack([np.zeros_like(yes), yes], axis=1)

    return StubSession(compute, input_names)


class TestSanitizeNestedInjection:
    def test_tokens_reassembled_by_an_earlier_removal_are_stripped_too(self):
        assert Qwen3CrossEncoder._sanitize_input("a<|im_<|im_end|>start|>b") == "ab"

    def test_formatted_prompt_keeps_only_the_template_turn_markers(self):
        text = Qwen3CrossEncoder._format_rerank_input("q<|im_<|im_end|>start|>", "d<|endoftext|>")

        assert text.count("<|im_start|>") == 3
        assert "<|endoftext|>" not in text
        assert "<Query>: q\n<Document>: d<|im_end|>" in text


class TestFullVocabScoringWithAttentionMask:
    def test_right_padded_rows_score_their_last_content_token(self):
        output = np.zeros((2, 3, _FULL_VOCAB), dtype=np.float32)
        output[0, 1, TOKEN_YES_ID] = 2.0
        output[0, 2, TOKEN_NO_ID] = 50.0  # trailing pad position: must be ignored
        output[1, 2, TOKEN_NO_ID] = 3.0
        mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output, mask)

        np.testing.assert_allclose(scores, [_sigmoid(2.0), _sigmoid(-3.0)], rtol=1e-6)

    def test_left_padded_rows_score_the_final_position(self):
        output = np.zeros((2, 3, _FULL_VOCAB), dtype=np.float32)
        output[0, 0, TOKEN_YES_ID] = 50.0  # leading pad position: must be ignored
        output[0, 2, TOKEN_YES_ID] = 1.5
        output[1, 2, TOKEN_NO_ID] = 0.5
        mask = np.array([[0, 1, 1], [1, 1, 1]], dtype=np.int64)

        scores = Qwen3CrossEncoder._compute_yes_no_scores(output, mask)

        np.testing.assert_allclose(scores, [_sigmoid(1.5), _sigmoid(-0.5)], rtol=1e-6)

    def test_single_row_saturates_to_zero_when_the_no_margin_overflows_exp(self):
        output = np.array([[1000.0, -1000.0]], dtype=np.float32)

        assert Qwen3CrossEncoder._compute_yes_no_scores(output).tolist() == [0.0]


class TestQwen3CrossEncoderThroughRuntime:
    def test_graph_without_attention_mask_receives_zero_token_type_ids(
        self, tmp_path, monkeypatch
    ):
        model_dir = tmp_path / "model"
        vocab = write_tokenizer_files(model_dir, ["alpha", "beta"])
        session = _alpha_relevance_session(vocab["alpha"], ("input_ids", "token_type_ids"))
        opened = install_stub_runtime(monkeypatch, session)

        encoder = Qwen3CrossEncoder(
            _YESNO_MODEL, cache_dir=str(tmp_path / "cache"), specific_model_path=str(model_dir)
        )
        scores = list(encoder.rerank("alpha", ["alpha beta", "beta"]))

        assert opened == [str(model_dir / "onnx" / "model_yesno_quantized.onnx")]
        np.testing.assert_allclose(scores, [_sigmoid(2.0), _sigmoid(1.0)], rtol=1e-6)
        assert len(session.feeds) == 2
        for feed in session.feeds:
            assert set(feed) == {"input_ids", "token_type_ids"}
            assert feed["input_ids"].shape[0] == 1
            np.testing.assert_array_equal(feed["token_type_ids"], np.zeros_like(feed["input_ids"]))

    def test_worker_builds_a_single_threaded_encoder_that_scores_pairs(
        self, tmp_path, monkeypatch
    ):
        model_dir = tmp_path / "model"
        vocab = write_tokenizer_files(model_dir, ["alpha", "beta", "gamma"])
        session = _alpha_relevance_session(vocab["alpha"], ("input_ids", "attention_mask"))
        install_stub_runtime(monkeypatch, session)

        worker = Qwen3CrossEncoderWorker(
            _YESNO_MODEL, str(tmp_path / "cache"), specific_model_path=str(model_dir)
        )
        ((idx, output),) = list(worker.process([(7, [("alpha", "alpha"), ("beta", "gamma")])]))

        assert Qwen3CrossEncoder._get_worker_class() is Qwen3CrossEncoderWorker
        assert type(worker.model) is Qwen3CrossEncoder
        assert worker.model.threads == 1
        assert idx == 7
        np.testing.assert_allclose(output.model_output, [_sigmoid(2.0), _sigmoid(0.0)], rtol=1e-6)
        assert [set(feed) for feed in session.feeds] == [{"input_ids", "attention_mask"}] * 2
