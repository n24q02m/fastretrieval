# Chuyển chọn từ qdrant/fastembed 0.8.0 (Apache-2.0); xem NOTICE.
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from fastretrieval.common.model_description import DenseModelDescription, ModelSource
from fastretrieval.common.onnx_model import OnnxOutputContext, OnnxSessionConfig
from fastretrieval.common.types import Device, NumpyArray, OnnxProvider
from fastretrieval.common.utils import define_cache_dir, normalize
from fastretrieval.contract.preprocessor import PreprocessorSpec
from fastretrieval.models.image.image_embedding_base import ImageEmbeddingBase, ImageInput
from fastretrieval.models.image.onnx_image_model import ImageEmbeddingWorker, OnnxImageModel

supported_onnx_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="Qdrant/clip-ViT-B-32-vision",
        dim=512,
        description="Image embeddings, multimodal image/text model.",
        license="mit",
        size_in_GB=0.34,
        sources=ModelSource(hf="Qdrant/clip-ViT-B-32-vision"),
        model_file="model.onnx",
    ),
    DenseModelDescription(
        model="Qdrant/resnet50-onnx",
        dim=2048,
        description="Image embeddings from a ResNet-50 vision encoder.",
        license="apache-2.0",
        size_in_GB=0.1,
        sources=ModelSource(hf="Qdrant/resnet50-onnx"),
        model_file="model.onnx",
    ),
    DenseModelDescription(
        model="Qdrant/Unicom-ViT-B-16",
        dim=768,
        description="Image embeddings from a ViT-B/16 vision encoder.",
        license="apache-2.0",
        size_in_GB=0.82,
        sources=ModelSource(hf="Qdrant/Unicom-ViT-B-16"),
        model_file="model.onnx",
    ),
    DenseModelDescription(
        model="Qdrant/Unicom-ViT-B-32",
        dim=512,
        description="Image embeddings from a ViT-B/32 vision encoder.",
        license="apache-2.0",
        size_in_GB=0.48,
        sources=ModelSource(hf="Qdrant/Unicom-ViT-B-32"),
        model_file="model.onnx",
    ),
    DenseModelDescription(
        model="jinaai/jina-clip-v1",
        dim=768,
        description="Image embeddings from the Jina CLIP vision encoder.",
        license="apache-2.0",
        size_in_GB=0.34,
        sources=ModelSource(hf="jinaai/jina-clip-v1"),
        model_file="onnx/vision_model.onnx",
    ),
]


class OnnxImageEmbedding(ImageEmbeddingBase, OnnxImageModel[NumpyArray]):
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
        preprocessor: PreprocessorSpec | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load
        self._extra_session_options = self._select_exposed_session_options(kwargs)
        self.device_ids = device_ids
        self.cuda = cuda
        self.device_id = device_id
        if self.device_id is None and self.device_ids is not None:
            self.device_id = self.device_ids[0]
        self.declared_preprocessor = preprocessor
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
        return supported_onnx_models

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

    def embed(
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
            preprocessor=self.declared_preprocessor,
            **kwargs,
        )

    @classmethod
    def _get_worker_class(cls) -> type[ImageEmbeddingWorker[NumpyArray]]:
        return OnnxImageEmbeddingWorker

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        embeddings = np.asarray(output.model_output)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        elif embeddings.ndim != 2:
            embeddings = embeddings.reshape(len(embeddings), -1)
        return normalize(embeddings)


class OnnxImageEmbeddingWorker(ImageEmbeddingWorker[NumpyArray]):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxImageEmbedding:
        return OnnxImageEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
