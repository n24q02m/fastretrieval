import numpy as np
import pytest

from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.text.onnx_embedding import OnnxTextEmbedding


class MockOnnxTextEmbedding(OnnxTextEmbedding):
    """Mock OnnxTextEmbedding for testing without initialization overhead."""

    def __init__(self):
        # Skip super().__init__ to avoid model loading/downloading
        pass


def test_post_process_raises_on_unsupported_shape():
    """Verify that _post_process_onnx_output raises ValueError for unsupported embedding shapes."""
    # Create a mock instance
    embedding_model = MockOnnxTextEmbedding()

    # Create an output context with a 4D array (unsupported)
    # The shape (1, 2, 3, 4) is arbitrary but must be != 2 and != 3
    model_output = np.zeros((1, 2, 3, 4))
    output = OnnxOutputContext(model_output=model_output, attention_mask=None)

    # Calling the method on the instance should raise ValueError
    with pytest.raises(ValueError, match="Unsupported embedding shape"):
        # We wrap in list() in case it returns a generator, though current implementation
        # raises eagerly before returning.
        list(embedding_model._post_process_onnx_output(output))
