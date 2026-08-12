import importlib.metadata

from fastretrieval.common.custom_model import CustomModelSpec, CustomRerankerSpec
from fastretrieval.common.types import Device
from fastretrieval.models.sparse import SparseEmbedding, SparseTextEmbedding
from fastretrieval.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder
from fastretrieval.text import TextEmbedding

try:
    version = importlib.metadata.version("fastretrieval")
except importlib.metadata.PackageNotFoundError:
    version = "0.0.0"

__version__ = version
__all__ = [
    "CustomModelSpec",
    "CustomRerankerSpec",
    "Device",
    "SparseEmbedding",
    "SparseTextEmbedding",
    "TextEmbedding",
    "TextCrossEncoder",
]
