"""One-call registration helpers for bring-your-own (BYO) ONNX models.

``CustomModelSpec`` registers an embedding model; ``CustomRerankerSpec`` registers
a cross-encoder reranker. Both wrap the lower-level ``add_custom_model`` APIs so a
BYO model can be loaded by id without hand-building a model description.
"""

from dataclasses import dataclass, field

from qwen3_embed.common.model_description import (
    BaseModelDescription,
    CustomDenseModelDescription,
    ModelSource,
    PoolingType,
)


@dataclass
class CustomModelSpec:
    """One-call registration of a BYO ONNX embedding model.

    Only ONNX-able models with one of the four supported output shapes are
    accepted: bert-bi (``CLS``/``MEAN``), causal last-token (``LAST_TOKEN``), or
    raw 3-D output (``DISABLED``).

    Usage::

        from qwen3_embed import CustomModelSpec, TextEmbedding

        CustomModelSpec(
            model_id="Org/gte-multilingual-base-onnx",
            hf="Org/gte-multilingual-base-onnx",
            model_file="onnx/model.onnx",
            dim=768, pooling="CLS", normalization=True,
        ).register()

        model = TextEmbedding("Org/gte-multilingual-base-onnx")
    """

    model_id: str
    hf: str | None = None
    url: str | None = None
    model_file: str = "onnx/model.onnx"
    dim: int | None = None
    pooling: str | PoolingType = PoolingType.MEAN
    normalization: bool = True
    max_seq_len: int | None = None
    additional_files: list[str] = field(default_factory=list)

    def register(self) -> None:
        """Register this model with :class:`TextEmbedding` so it can be loaded by id."""
        from qwen3_embed import TextEmbedding

        if self.dim is None:
            raise ValueError("dim is required for an embedding model")

        description = CustomDenseModelDescription(
            model=self.model_id,
            dim=self.dim,
            sources=ModelSource(hf=self.hf, url=self.url),
            model_file=self.model_file,
            additional_files=self.additional_files,
        )
        TextEmbedding.add_custom_model(
            description,
            pooling=PoolingType(self.pooling),
            normalization=self.normalization,
        )


@dataclass
class CustomRerankerSpec:
    """One-call registration of a BYO ONNX cross-encoder reranker.

    Only standard ONNX cross-encoders are accepted: a single relevance logit per
    ``(query, document)`` pair (e.g. ``bge-reranker``, ``gte-reranker``, ``ms-marco``,
    ``jina-reranker``). Higher score = more relevant. Qwen3 causal yes/no rerankers
    are built in and need no registration; this path is for the bert-cross-encoder
    output shape, so — unlike :class:`CustomModelSpec` — there is no ``dim``,
    ``pooling`` or ``normalization``.

    Usage::

        from qwen3_embed import CustomRerankerSpec, TextCrossEncoder

        CustomRerankerSpec(
            model_id="onnx-community/gte-multilingual-reranker-base",
            hf="onnx-community/gte-multilingual-reranker-base",
            model_file="onnx/model_quantized.onnx",
        ).register()

        encoder = TextCrossEncoder("onnx-community/gte-multilingual-reranker-base")
        scores = list(encoder.rerank("query", ["doc a", "doc b"]))
    """

    model_id: str
    hf: str | None = None
    url: str | None = None
    model_file: str = "onnx/model.onnx"
    description: str = ""
    license: str = ""
    size_in_GB: float = 0.0
    additional_files: list[str] = field(default_factory=list)

    def register(self) -> None:
        """Register this reranker with :class:`TextCrossEncoder` so it loads by id."""
        from qwen3_embed import TextCrossEncoder

        model_description = BaseModelDescription(
            model=self.model_id,
            sources=ModelSource(hf=self.hf, url=self.url),
            model_file=self.model_file,
            description=self.description,
            license=self.license,
            size_in_GB=self.size_in_GB,
            additional_files=self.additional_files,
        )
        TextCrossEncoder.add_custom_model(model_description)
