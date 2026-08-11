import importlib.metadata

from qwen3_embed.common.custom_model import CustomModelSpec, CustomRerankerSpec
from qwen3_embed.common.types import Device
from qwen3_embed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder
from qwen3_embed.text import TextEmbedding

try:
    version = importlib.metadata.version("qwen3-embed")
except importlib.metadata.PackageNotFoundError:
    version = "0.0.0"

__version__ = version
__all__ = [
    "CustomModelSpec",
    "CustomRerankerSpec",
    "Device",
    "TextEmbedding",
    "TextCrossEncoder",
]
