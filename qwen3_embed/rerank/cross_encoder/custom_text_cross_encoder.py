from typing import Any

from qwen3_embed.common.model_description import BaseModelDescription
from qwen3_embed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder
from qwen3_embed.rerank.cross_encoder.onnx_text_model import TextRerankerWorker


class CustomTextCrossEncoder(OnnxTextCrossEncoder):
    SUPPORTED_MODELS: list[BaseModelDescription] = []

    @classmethod
    def _list_supported_models(cls) -> list[BaseModelDescription]:
        return cls.SUPPORTED_MODELS

    @classmethod
    def add_model(
        cls,
        model_description: BaseModelDescription,
    ) -> None:
        cls._clear_model_cache()
        cls.SUPPORTED_MODELS.append(model_description)

    @classmethod
    def _export_registry(cls) -> list[BaseModelDescription]:
        # Snapshot the runtime registry so it can be pickled into spawned
        # worker processes, which start with a fresh interpreter and an empty
        # SUPPORTED_MODELS. BaseModelDescription is a frozen dataclass (picklable).
        return list(cls.SUPPORTED_MODELS)

    @classmethod
    def _import_registry(cls, payload: list[BaseModelDescription]) -> None:
        # Re-register custom models in a worker process, idempotently (same id
        # imported twice must not create a duplicate entry).
        existing = {m.model.lower() for m in cls.SUPPORTED_MODELS}
        for desc in payload:
            if desc.model.lower() not in existing:
                cls.SUPPORTED_MODELS.append(desc)
                existing.add(desc.model.lower())

    @classmethod
    def _get_worker_class(cls) -> type["CustomTextCrossEncoderWorker"]:
        return CustomTextCrossEncoderWorker

    def _extra_worker_params(self) -> dict[str, Any]:
        # Propagate the runtime registry so spawned workers (fresh interpreters
        # with an empty SUPPORTED_MODELS) can resolve + re-register this custom
        # model instead of raising "Model ... not supported".
        return {"custom_registry": self._export_registry()}


class CustomTextCrossEncoderWorker(TextRerankerWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> CustomTextCrossEncoder:
        registry = kwargs.pop("custom_registry", None)
        if registry is not None:
            CustomTextCrossEncoder._import_registry(registry)
        return CustomTextCrossEncoder(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
