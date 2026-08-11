from collections.abc import Iterable, Sequence
from typing import Any

from qwen3_embed.common.model_description import DenseModelDescription
from qwen3_embed.common.onnx_model import OnnxOutputContext, OnnxSessionConfig
from qwen3_embed.common.types import Device, NumpyArray, OnnxProvider
from qwen3_embed.common.utils import define_cache_dir, normalize
from qwen3_embed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase

# Base class model list kept empty — Qwen3 models are registered
# in qwen3_embedding.py. Custom models can be added at runtime
# via TextEmbedding.add_custom_model().
supported_onnx_models: list[DenseModelDescription] = []


class OnnxTextEmbedding(TextEmbeddingBase, OnnxTextModel[NumpyArray]):
    """Implementation of the Flag Embedding model."""

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        """
        Lists the supported models.

        Returns:
            list[DenseModelDescription]: A list of DenseModelDescription objects containing the model information.
        """
        return supported_onnx_models

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
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
        """
        Args:
            model_name (str): The name of the model to use.
            cache_dir (str, optional): The path to the cache directory.
                                       Can be set using the `qwen3_embed_CACHE_PATH` env variable.
                                       Defaults to `qwen3_embed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.
            providers (Optional[Sequence[OnnxProvider]], optional): The list of onnxruntime providers to use.
                Mutually exclusive with the `cuda` and `device_ids` arguments. Defaults to None.
            cuda (Union[bool, Device], optional): Whether to use cuda for inference. Mutually exclusive with `providers`
                Defaults to Device.AUTO.
            device_ids (Optional[list[int]], optional): The list of device ids to use for data parallel processing in
                workers. Should be used with `cuda` equals to `True`, `Device.AUTO` or `Device.CUDA`, mutually exclusive
                with `providers`. Defaults to None.
            lazy_load (bool, optional): Whether to load the model during class initialization or on demand.
                Should be set to True when using multiple-gpu and parallel encoding. Defaults to False.
            device_id (Optional[int], optional): The device id to use for loading the model in the worker process.
            specific_model_path (Optional[str], optional): The specific path to the onnx model dir if it should be imported from somewhere else

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-base-en.
        """
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load
        self._extra_session_options = self._select_exposed_session_options(kwargs)
        # List of device ids, that can be used for data parallel processing in workers
        self.device_ids = device_ids
        self.cuda = cuda

        # This device_id will be used if we need to load model in current process
        self.device_id: int | None = None
        if device_id is not None:
            self.device_id = device_id
        elif self.device_ids is not None:
            self.device_id = self.device_ids[0]

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

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster.
                Capped at the batch size the ONNX graph is pinned to, if it declares one.
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.

        Returns:
            List of embeddings, one per document
        """
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

    @classmethod
    def _get_worker_class(cls) -> type["TextEmbeddingWorker[NumpyArray]"]:
        return OnnxTextEmbeddingWorker

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        embeddings = output.model_output

        if embeddings.ndim == 3:  # (batch_size, seq_len, embedding_dim)
            processed_embeddings = embeddings[:, 0]
        elif embeddings.ndim == 2:  # (batch_size, embedding_dim)
            processed_embeddings = embeddings
        else:
            raise ValueError(f"Unsupported embedding shape: {embeddings.shape}")
        return normalize(processed_embeddings)

    def load_onnx_model(self) -> None:
        config = OnnxSessionConfig(
            threads=self.threads,
            providers=self.providers,
            cuda=self.cuda,
            device_id=self.device_id,
            extra_session_options=self._extra_session_options,
        )
        self._load_onnx_model(
            model_dir=self._model_dir,
            model_file=self.model_description.model_file,
            config=config,
        )

    def token_count(
        self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        return self._token_count(texts, batch_size=batch_size, **kwargs)


class OnnxTextEmbeddingWorker(TextEmbeddingWorker[NumpyArray]):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextEmbedding:
        return OnnxTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
