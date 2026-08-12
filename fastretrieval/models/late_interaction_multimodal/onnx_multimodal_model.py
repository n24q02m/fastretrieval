# Chuyển chọn từ qdrant/fastembed 0.8.0 (Apache-2.0); xem NOTICE.
import io
import os
from collections.abc import Iterable, Sequence
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Encoding

from fastretrieval.common.onnx_model import (
    EmbeddingWorker,
    OnnxModel,
    OnnxOutputContext,
    OnnxSessionConfig,
    T,
)
from fastretrieval.common.preprocessor_utils import load_tokenizer
from fastretrieval.common.types import Device, NumpyArray, OnnxProvider
from fastretrieval.common.utils import iter_batch
from fastretrieval.contract.preprocessor import PreprocessorSpec
from fastretrieval.models.image.image_embedding_base import ImageInput
from fastretrieval.parallel_processor import ParallelWorkerPool, PoolConfig
from fastretrieval.preprocessing.graph import resolve_preprocessor
from fastretrieval.preprocessing.image import preprocess_images


class OnnxMultimodalModel(OnnxModel[T]):
    """ONNX runner dùng chung graph text/image và ``PreprocessorSpec``."""

    ONNX_OUTPUT_NAMES: list[str] | None = None

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor_spec: PreprocessorSpec | None = None

    def _preprocess_onnx_text_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        return onnx_input

    def _preprocess_onnx_image_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        return onnx_input

    @classmethod
    def _get_text_worker_class(cls) -> type["TextEmbeddingWorker[T]"]:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def _get_image_worker_class(cls) -> type["ImageEmbeddingWorker[T]"]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_text_output(self, output: OnnxOutputContext) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_image_output(self, output: OnnxOutputContext) -> Iterable[T]:
        raise NotImplementedError("Subclasses must implement this method")

    def _load_onnx_model(
        self,
        model_dir: Path,
        model_file: str,
        config: OnnxSessionConfig | None = None,
    ) -> tuple[Any, list[str]]:
        self.model, input_names = self._instantiate_onnx_session(
            model_path=model_dir / model_file,
            config=config,
        )
        self.model_input_names = set(input_names)
        self.static_batch_size = self._detect_static_batch_size(self.model)
        self.tokenizer, self.special_token_to_id = load_tokenizer(model_dir=model_dir)
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

    def tokenize(self, documents: list[str], **kwargs: Any) -> list[Encoding]:
        assert self.tokenizer is not None
        return self.tokenizer.encode_batch(documents)

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

    def _encode_images(self, images: list[ImageInput]) -> NumpyArray:
        if self.preprocessor_spec is None:
            raise ValueError("Image preprocessor is not initialized")
        if self.preprocessor_spec.kind == "none":
            return np.asarray([self._raw_image_bytes(item) for item in images], dtype=object)  # type: ignore[return-value]
        if self.preprocessor_spec.kind != "image":
            raise ValueError(
                "LateInteractionMultimodalEmbedding requires a preprocessor with kind='image' "
                f"or kind='none', got {self.preprocessor_spec.kind!r}"
            )
        return preprocess_images(images, self.preprocessor_spec)

    def onnx_embed_text(self, documents: list[str], **kwargs: Any) -> OnnxOutputContext:
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_onnx_model() first.")
        encoded = self.tokenize(documents, **kwargs)
        input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
        attention_mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
        onnx_input: dict[str, NumpyArray] = {"input_ids": input_ids}
        input_names = self.model_input_names or set()
        if "attention_mask" in input_names:
            onnx_input["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            onnx_input["token_type_ids"] = np.zeros(input_ids.shape, dtype=np.int64)
        onnx_input = self._preprocess_onnx_text_input(onnx_input, **kwargs)
        result = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)[0]
        if getattr(result, "dtype", None) == np.float16:
            result = result.astype(np.float32)
        return OnnxOutputContext(
            model_output=result,
            attention_mask=onnx_input.get("attention_mask", attention_mask),
            input_ids=onnx_input.get("input_ids", input_ids),
        )

    def onnx_embed_image(self, images: list[ImageInput], **kwargs: Any) -> OnnxOutputContext:
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_onnx_model() first.")
        onnx_input: dict[str, NumpyArray] = {"pixel_values": self._encode_images(images)}
        onnx_input = self._preprocess_onnx_image_input(onnx_input, **kwargs)
        result = self.model.run(self.ONNX_OUTPUT_NAMES, onnx_input)[0]
        if getattr(result, "dtype", None) == np.float16:
            result = result.astype(np.float32)
        return OnnxOutputContext(model_output=result)

    def _effective_batch_size(self, batch_size: int) -> int:
        static_batch_size = getattr(self, "static_batch_size", None)
        return batch_size if static_batch_size is None else min(batch_size, static_batch_size)

    def _embed_documents(
        self,
        model_name: str,
        cache_dir: str,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        local_files_only: bool = False,
        specific_model_path: str | None = None,
        extra_session_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[T]:
        is_small = isinstance(documents, str)
        if isinstance(documents, str):
            documents = [documents]
        if isinstance(documents, list) and len(documents) < batch_size:
            is_small = True
        if parallel is None or is_small:
            if self.model is None:
                self.load_onnx_model()
            for batch in iter_batch(documents, self._effective_batch_size(batch_size)):
                yield from self._post_process_onnx_text_output(
                    self.onnx_embed_text(batch, **kwargs)
                )
            return

        if parallel == 0:
            parallel = os.cpu_count()
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
        start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
        pool = ParallelWorkerPool(
            worker=self._get_text_worker_class(),
            config=PoolConfig(
                num_workers=parallel or 1,
                cuda=cuda,
                device_ids=device_ids,
                start_method=start_method,
            ),
        )
        for batch in pool.ordered_map(
            iter_batch(documents, self._effective_batch_size(batch_size)), **params
        ):
            yield from self._post_process_onnx_text_output(batch)

    def _embed_images(
        self,
        model_name: str,
        cache_dir: str,
        images: ImageInput | Iterable[ImageInput],
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
        is_small = isinstance(
            images, (str, Path, bytes, bytearray, memoryview)
        ) or self._is_image_object(images)
        if is_small:
            images = [images]
        if isinstance(images, list) and len(images) < batch_size:
            is_small = True
        if parallel is None or is_small:
            if self.model is None:
                self.load_onnx_model()
            for batch in iter_batch(images, self._effective_batch_size(batch_size)):
                yield from self._post_process_onnx_image_output(
                    self.onnx_embed_image(batch, **kwargs)
                )
            return

        if parallel == 0:
            parallel = os.cpu_count()
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
        start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
        pool = ParallelWorkerPool(
            worker=self._get_image_worker_class(),
            config=PoolConfig(
                num_workers=parallel or 1,
                cuda=cuda,
                device_ids=device_ids,
                start_method=start_method,
            ),
        )
        for batch in pool.ordered_map(
            iter_batch(images, self._effective_batch_size(batch_size)), **params
        ):
            yield from self._post_process_onnx_image_output(batch)


class TextEmbeddingWorker(EmbeddingWorker[T]):
    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, OnnxOutputContext]]:
        for idx, batch in items:
            yield idx, self.model.onnx_embed_text(batch)


class ImageEmbeddingWorker(EmbeddingWorker[T]):
    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, OnnxOutputContext]]:
        for idx, batch in items:
            yield idx, self.model.onnx_embed_image(batch)
