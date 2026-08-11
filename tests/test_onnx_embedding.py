from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker

_MODEL_NAME = "test-org/test-model"
_MODEL_DESC = DenseModelDescription(
    model=_MODEL_NAME,
    sources=ModelSource(hf=_MODEL_NAME),
    model_file="model.onnx",
    description="Test model",
    license="MIT",
    size_in_GB=0.1,
    dim=4,
)


@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._select_exposed_session_options")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._get_model_description")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding.download_model")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding.load_onnx_model")
def test_onnx_text_embedding_init_lazy_load(
    mock_load_onnx_model: MagicMock,
    mock_download_model: MagicMock,
    mock_get_model_description: MagicMock,
    mock_select_exposed_session_options: MagicMock,
) -> None:
    mock_get_model_description.return_value = _MODEL_DESC
    mock_download_model.return_value = Path("/tmp/model")
    mock_select_exposed_session_options.return_value = {}

    embedding = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)

    mock_load_onnx_model.assert_not_called()
    assert embedding.lazy_load is True
    assert embedding.model_name == _MODEL_NAME
    mock_get_model_description.assert_called_once_with(_MODEL_NAME)


@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._select_exposed_session_options")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._get_model_description")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding.download_model")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding.load_onnx_model")
def test_onnx_text_embedding_init_no_lazy_load(
    mock_load_onnx_model: MagicMock,
    mock_download_model: MagicMock,
    mock_get_model_description: MagicMock,
    mock_select_exposed_session_options: MagicMock,
) -> None:
    mock_get_model_description.return_value = _MODEL_DESC
    mock_download_model.return_value = Path("/tmp/model")
    mock_select_exposed_session_options.return_value = {}

    embedding = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=False)

    mock_load_onnx_model.assert_called_once()
    assert embedding.lazy_load is False


@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._select_exposed_session_options")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._get_model_description")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding.download_model")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._embed_documents")
def test_onnx_text_embedding_embed(
    mock_embed_documents: MagicMock,
    mock_download_model: MagicMock,
    mock_get_model_description: MagicMock,
    mock_select_exposed_session_options: MagicMock,
) -> None:
    mock_get_model_description.return_value = _MODEL_DESC
    mock_download_model.return_value = Path("/tmp/model")
    mock_select_exposed_session_options.return_value = {}

    # Return empty iterator
    mock_embed_documents.return_value = iter([])

    embedding = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True, cache_dir="/tmp/cache")

    docs = ["doc1", "doc2"]
    list(embedding.embed(documents=docs, batch_size=32, parallel=4))

    mock_embed_documents.assert_called_once()
    kwargs = mock_embed_documents.call_args.kwargs
    assert kwargs["model_name"] == _MODEL_NAME
    assert (
        kwargs["cache_dir"] == str(Path("/tmp/cache").absolute())
        if Path("/tmp/cache").is_absolute()
        else str(Path("/tmp/cache").resolve())
    )
    assert kwargs["documents"] == docs
    assert kwargs["batch_size"] == 32
    assert kwargs["parallel"] == 4


@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._select_exposed_session_options")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._get_model_description")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding.download_model")
def test_onnx_text_embedding_preprocess_input(
    mock_download_model: MagicMock,
    mock_get_model_description: MagicMock,
    mock_select_exposed_session_options: MagicMock,
) -> None:
    mock_get_model_description.return_value = _MODEL_DESC
    mock_download_model.return_value = Path("/tmp/model")
    mock_select_exposed_session_options.return_value = {}

    embedding = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)

    input_dict = {"input_ids": np.array([1, 2, 3])}
    output_dict = embedding._preprocess_onnx_input(input_dict)

    assert output_dict is input_dict


@patch("qwen3_embed.text.onnx_embedding.normalize")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._select_exposed_session_options")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._get_model_description")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding.download_model")
def test_onnx_text_embedding_postprocess_2d(
    mock_download_model: MagicMock,
    mock_get_model_description: MagicMock,
    mock_select_exposed_session_options: MagicMock,
    mock_normalize: MagicMock,
) -> None:
    mock_get_model_description.return_value = _MODEL_DESC
    mock_download_model.return_value = Path("/tmp/model")
    mock_select_exposed_session_options.return_value = {}
    mock_normalize.side_effect = lambda x: x

    embedding = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)

    model_output = np.array([[1.0, 2.0], [3.0, 4.0]])
    output_context = OnnxOutputContext(model_output=model_output, attention_mask=None)

    embedding._post_process_onnx_output(output_context)

    mock_normalize.assert_called_once()
    assert np.array_equal(mock_normalize.call_args[0][0], model_output)


@patch("qwen3_embed.text.onnx_embedding.normalize")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._select_exposed_session_options")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding._get_model_description")
@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding.download_model")
def test_onnx_text_embedding_postprocess_3d(
    mock_download_model: MagicMock,
    mock_get_model_description: MagicMock,
    mock_select_exposed_session_options: MagicMock,
    mock_normalize: MagicMock,
) -> None:
    mock_get_model_description.return_value = _MODEL_DESC
    mock_download_model.return_value = Path("/tmp/model")
    mock_select_exposed_session_options.return_value = {}
    mock_normalize.side_effect = lambda x: x

    embedding = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)

    # 3D array: (batch_size, seq_len, dim)
    model_output = np.array([[[1.0, 2.0], [9.0, 9.0]], [[3.0, 4.0], [9.0, 9.0]]])
    output_context = OnnxOutputContext(model_output=model_output, attention_mask=None)

    embedding._post_process_onnx_output(output_context)

    mock_normalize.assert_called_once()
    # It should slice [:, 0]
    expected_slice = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(mock_normalize.call_args[0][0], expected_slice)


@patch("qwen3_embed.text.onnx_embedding.OnnxTextEmbedding")
def test_onnx_text_embedding_worker_init(
    mock_onnx_embedding: MagicMock,
) -> None:
    worker = OnnxTextEmbeddingWorker.__new__(OnnxTextEmbeddingWorker)
    worker.__init__ = lambda *args, **kwargs: None
    worker.init_embedding(model_name=_MODEL_NAME, cache_dir="/tmp/cache", extra="arg")

    mock_onnx_embedding.assert_called_once_with(
        model_name=_MODEL_NAME, cache_dir="/tmp/cache", threads=1, extra="arg"
    )
