# Chuyển chọn từ qdrant/fastembed 0.8.0 (Apache-2.0); xem NOTICE.
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from tokenizers import Encoding

from fastretrieval.common.model_description import DenseModelDescription, ModelSource
from fastretrieval.common.onnx_model import OnnxOutputContext, OnnxSessionConfig
from fastretrieval.common.types import Device, NumpyArray, OnnxProvider
from fastretrieval.common.utils import define_cache_dir, iter_batch
from fastretrieval.models.image.image_embedding_base import ImageInput
from fastretrieval.models.late_interaction_multimodal.late_interaction_multimodal_embedding_base import (
    LateInteractionMultimodalEmbeddingBase,
)
from fastretrieval.models.late_interaction_multimodal.onnx_multimodal_model import (
    ImageEmbeddingWorker,
    OnnxMultimodalModel,
    TextEmbeddingWorker,
)

supported_colpali_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="Qdrant/colpali-v1.3-fp16",
        dim=128,
        description=(
            "Text embeddings, Multimodal (text&image), English, 50 tokens query length "
            "truncation, 2024."
        ),
        license="mit",
        size_in_GB=6.5,
        sources=ModelSource(hf="Qdrant/colpali-v1.3-fp16"),
        additional_files=["model.onnx_data"],
        model_file="model.onnx",
    ),
]


class ColPali(LateInteractionMultimodalEmbeddingBase, OnnxMultimodalModel[NumpyArray]):
    QUERY_PREFIX = "Query: "
    BOS_TOKEN = "<s>"
    PAD_TOKEN = "<pad>"
    QUERY_MARKER_TOKEN_ID = [2, 5098]
    IMAGE_PLACEHOLDER_SIZE = (3, 448, 448)
    EMPTY_TEXT_PLACEHOLDER = np.array([257152] * 1024 + [2, 50721, 573, 2416, 235265, 108])
    EVEN_ATTENTION_MASK = np.array([1] * 1030)

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        device_id: int | None = None,
        specific_model_path: str | None = None,
        **kwargs: Any,
    ):
        OnnxMultimodalModel.__init__(self)
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load
        self._extra_session_options = self._select_exposed_session_options(kwargs)
        self.device_ids = device_ids
        self.cuda = cuda
        self.device_id = (
            device_id if device_id is not None else (device_ids[0] if device_ids else None)
        )
        self.model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))
        self._specific_model_path = specific_model_path
        self._model_dir = self.download_model(
            self.model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
        )
        if not self.lazy_load:
            self.load_onnx_model()

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return supported_colpali_models

    def load_onnx_model(self) -> None:
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            config=OnnxSessionConfig(
                threads=self.threads,
                providers=self.providers,
                cuda=self.cuda,
                device_id=self.device_id,
                extra_session_options=self._extra_session_options,
            ),
        )

    def _post_process_onnx_image_output(self, output: OnnxOutputContext) -> Iterable[NumpyArray]:
        assert self.model_description.dim is not None, "Model dim is not defined"
        return output.model_output.reshape(
            output.model_output.shape[0], -1, self.model_description.dim
        )

    def _post_process_onnx_text_output(self, output: OnnxOutputContext) -> Iterable[NumpyArray]:
        return output.model_output

    def tokenize(self, documents: list[str], **kwargs: Any) -> list[Encoding]:
        assert self.tokenizer is not None
        queries = [
            self.BOS_TOKEN + self.QUERY_PREFIX + query + self.PAD_TOKEN * 10 + "\n"
            for query in documents
        ]
        return self.tokenizer.encode_batch(queries)

    def token_count(
        self,
        texts: str | Iterable[str],
        batch_size: int = 1024,
        include_extension: bool = False,
        **kwargs: Any,
    ) -> int:
        if not hasattr(self, "model") or self.model is None:
            self.load_onnx_model()
        assert self.tokenizer is not None
        texts = [texts] if isinstance(texts, str) else texts
        tokenize = self.tokenize if include_extension else self.tokenizer.encode_batch
        return sum(
            sum(encoding.attention_mask)
            for batch in iter_batch(texts, batch_size)
            for encoding in tokenize(batch)
        )

    def _preprocess_onnx_text_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        onnx_input["input_ids"] = np.asarray(
            [
                self.QUERY_MARKER_TOKEN_ID + input_ids[2:].tolist()
                for input_ids in onnx_input["input_ids"]
            ]
        )
        empty_image = np.zeros(self.IMAGE_PLACEHOLDER_SIZE, dtype=np.float32)
        onnx_input["pixel_values"] = np.asarray([empty_image for _ in onnx_input["input_ids"]])
        return onnx_input

    def _preprocess_onnx_image_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        onnx_input["input_ids"] = np.asarray(
            [self.EMPTY_TEXT_PLACEHOLDER for _ in onnx_input["pixel_values"]]
        )
        onnx_input["attention_mask"] = np.asarray(
            [self.EVEN_ATTENTION_MASK for _ in onnx_input["pixel_values"]]
        )
        return onnx_input

    def embed_text(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        yield from self._embed_documents(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            documents=documents,
            batch_size=batch_size,
            parallel=parallel,
            providers=self.providers,
            cuda=self.cuda,
            device_ids=self.device_ids,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
            extra_session_options=self._extra_session_options,
            **kwargs,
        )

    def embed_image(
        self,
        images: ImageInput | Iterable[ImageInput],
        batch_size: int = 16,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        yield from self._embed_images(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            images=images,
            batch_size=batch_size,
            parallel=parallel,
            providers=self.providers,
            cuda=self.cuda,
            device_ids=self.device_ids,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
            extra_session_options=self._extra_session_options,
            **kwargs,
        )

    @classmethod
    def _get_text_worker_class(cls) -> type[TextEmbeddingWorker[NumpyArray]]:
        return ColPaliTextEmbeddingWorker

    @classmethod
    def _get_image_worker_class(cls) -> type[ImageEmbeddingWorker[NumpyArray]]:
        return ColPaliImageEmbeddingWorker


class ColPaliTextEmbeddingWorker(TextEmbeddingWorker[NumpyArray]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> ColPali:
        return ColPali(model_name=model_name, cache_dir=cache_dir, threads=1, **kwargs)


class ColPaliImageEmbeddingWorker(ImageEmbeddingWorker[NumpyArray]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> ColPali:
        return ColPali(model_name=model_name, cache_dir=cache_dir, threads=1, **kwargs)
