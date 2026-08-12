# Chuyển chọn từ qdrant/fastembed 0.8.0 (Apache-2.0); xem NOTICE.
import io
import os
from collections.abc import Iterable, Sequence
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any

import numpy as np

from fastretrieval.common.onnx_model import (
    EmbeddingWorker,
    OnnxModel,
    OnnxOutputContext,
    OnnxSessionConfig,
    T,
)
from fastretrieval.common.types import Device, NumpyArray, OnnxProvider
from fastretrieval.common.utils import iter_batch
from fastretrieval.contract.preprocessor import PreprocessorSpec
from fastretrieval.parallel_processor import ParallelWorkerPool, PoolConfig
from fastretrieval.preprocessing.graph import resolve_preprocessor
from fastretrieval.preprocessing.image import preprocess_images


class OnnxImageModel(OnnxModel[T]):
    """ONNX image runner nối model graph với ``PreprocessorSpec``."""

    @classmethod
    def _get_worker_class(cls) -> type["ImageEmbeddingWorker[T]"]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_output(self, output: OnnxOutputContext, **kwargs: Any) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor_spec: PreprocessorSpec | None = None

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        return onnx_input

    def _load_onnx_model(
        self,
        model_dir: Path,
        model_file: str,
        config: OnnxSessionConfig | None = None,
    ) -> tuple[Any, list[str]]:
        # Ảnh không cần tokenizer; gọi trực tiếp các bước session của base để
        # tránh load_tokenizer() bắt buộc config text.
        self.model, input_names = self._instantiate_onnx_session(
            model_path=model_dir / model_file,
            config=config,
        )
        self.model_input_names = set(input_names)
        self.static_batch_size = self._detect_static_batch_size(self.model)
        declared = getattr(self, "declared_preprocessor", None)
        resolved = resolve_preprocessor(model_dir, model_file, declared)
        if (
            resolved.kind == "text"
            and declared is None
            and not (Path(model_dir) / "preprocessor_config.json").is_file()
        ):
            resolved = PreprocessorSpec(kind="image")
        self.preprocessor_spec = resolved
        return self.model, input_names

    def load_onnx_model(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def _is_image_object(value: Any) -> bool:
        return hasattr(value, "convert") or hasattr(value, "save")

    @staticmethod
    def _raw_image_bytes(value: Any) -> bytes:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        if isinstance(value, (str, Path)):
            return Path(value).read_bytes()
        if hasattr(value, "save"):
            output = io.BytesIO()
            value.save(output, format="PNG")
            return output.getvalue()
        raise TypeError(f"unsupported raw image input: {type(value).__name__}")

    def _encode_images(self, images: list[Any]) -> NumpyArray:
        if self.preprocessor_spec is None:
            raise ValueError("Image preprocessor is not initialized")
        if self.preprocessor_spec.kind == "none":
            # Graph tự chứa decoder/processor nhận bytes, nên không được
            # chuẩn hóa lần hai ở runtime.
            return np.asarray([self._raw_image_bytes(item) for item in images], dtype=object)  # type: ignore[return-value]
        if self.preprocessor_spec.kind != "image":
            raise ValueError(
                "ImageEmbedding requires a preprocessor with kind='image' or kind='none', "
                f"got {self.preprocessor_spec.kind!r}"
            )
        return preprocess_images(images, self.preprocessor_spec)

    def _build_onnx_input(self, encoded: NumpyArray) -> dict[str, NumpyArray]:
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_onnx_model() first.")
        inputs = self.model.get_inputs()
        if not inputs:
            raise ValueError("Image model has no ONNX inputs")
        return {inputs[0].name: encoded}

    def onnx_embed(self, images: list[Any], **kwargs: Any) -> OnnxOutputContext:
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_onnx_model() first.")
        encoded = self._encode_images(images)
        onnx_input = self._build_onnx_input(encoded)
        onnx_input = self._preprocess_onnx_input(onnx_input, **kwargs)
        model_output = self.model.run(None, onnx_input)
        result = np.asarray(model_output[0])
        if result.dtype == np.float16:
            result = result.astype(np.float32)
        embeddings = result.reshape(len(images), -1)
        return OnnxOutputContext(model_output=embeddings)

    def _embed_images(
        self,
        model_name: str,
        cache_dir: str,
        images: Any,
        batch_size: int = 16,
        parallel: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        local_files_only: bool = False,
        specific_model_path: str | None = None,
        extra_session_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[T]:
        is_small = False
        if isinstance(images, (str, Path, bytes, bytearray, memoryview)) or self._is_image_object(
            images
        ):
            images = [images]
            is_small = True
        if isinstance(images, list) and len(images) < batch_size:
            is_small = True

        if parallel is None or is_small:
            if self.model is None:
                self.load_onnx_model()
            for batch in iter_batch(images, batch_size):
                yield from self._post_process_onnx_output(
                    self.onnx_embed(batch, **kwargs), **kwargs
                )
            return

        if parallel == 0:
            parallel = os.cpu_count()
        start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
        params: dict[str, Any] = {
            "model_name": model_name,
            "cache_dir": cache_dir,
            "providers": providers,
            "local_files_only": local_files_only,
            "specific_model_path": specific_model_path,
            **kwargs,
        }
        if extra_session_options is not None:
            params.update(extra_session_options)
        pool = ParallelWorkerPool(
            worker=self._get_worker_class(),
            config=PoolConfig(
                num_workers=parallel or 1,
                cuda=cuda,
                device_ids=device_ids,
                start_method=start_method,
            ),
        )
        for batch in pool.ordered_map(iter_batch(images, batch_size), **params):
            yield from self._post_process_onnx_output(batch, **kwargs)  # type: ignore[arg-type]


class ImageEmbeddingWorker(EmbeddingWorker[T]):
    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, OnnxOutputContext]]:
        for idx, batch in items:
            yield idx, self.model.onnx_embed(batch)
