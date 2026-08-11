from collections.abc import Iterable, Sequence
from typing import Any

from loguru import logger

from qwen3_embed.common.model_description import BaseModelDescription
from qwen3_embed.common.onnx_model import OnnxOutputContext, OnnxSessionConfig
from qwen3_embed.common.types import Device, OnnxProvider
from qwen3_embed.common.utils import define_cache_dir
from qwen3_embed.rerank.cross_encoder.onnx_text_model import (
    OnnxCrossEncoderModel,
    TextRerankerWorker,
)
from qwen3_embed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase

# Base class model list kept empty — Qwen3 reranker is registered
# in qwen3_cross_encoder.py. Custom models can be added at runtime
# via CustomTextCrossEncoder.add_model().
supported_onnx_models: list[BaseModelDescription] = []


class OnnxTextCrossEncoder(TextCrossEncoderBase, OnnxCrossEncoderModel):
    @classmethod
    def _list_supported_models(cls) -> list[BaseModelDescription]:
        """Lists the supported models.

        Returns:
            list[BaseModelDescription]: A list of BaseModelDescription objects containing the model information.
        """
        return supported_onnx_models

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
            ValueError: If the model_name is not in the format <org>/<model> e.g. Xenova/ms-marco-MiniLM-L-6-v2.
        """
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self.providers = providers
        self.lazy_load = lazy_load
        self._extra_session_options = self._select_exposed_session_options(kwargs)

        self._setup_device(cuda, device_ids, device_id)
        self._setup_model_assets(model_name, cache_dir, specific_model_path)

        if not self.lazy_load:
            self.load_onnx_model()

    def _setup_device(
        self,
        cuda: bool | Device,
        device_ids: list[int] | None,
        device_id: int | None,
    ) -> None:
        # List of device ids, that can be used for data parallel processing in workers
        self.device_ids = device_ids
        self.cuda = cuda

        if self.device_ids is not None and len(self.device_ids) > 1:
            logger.warning(
                "Parallel execution is currently not supported for cross encoders, "
                f"only the first device will be used for inference: {self.device_ids[0]}."
            )

        # This device_id will be used if we need to load model in current process
        self.device_id = None
        if device_id is not None:
            self.device_id = device_id
        elif self.device_ids is not None:
            self.device_id = self.device_ids[0]

    def _setup_model_assets(
        self,
        model_name: str,
        cache_dir: str | None,
        specific_model_path: str | None,
    ) -> None:
        self.model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))
        self._specific_model_path = specific_model_path
        self._model_dir = self.download_model(
            self.model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
            specific_model_path=self._specific_model_path,
        )

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

    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        batch_size: int = 64,
        **kwargs: Any,
    ) -> Iterable[float]:
        """Reranks documents based on their relevance to a given query.

        Args:
            query (str): The query string to which document relevance is calculated.
            documents (Iterable[str]): Iterable of documents to be reranked.
            batch_size (int, optional): The number of documents processed in each batch. Higher batch sizes improve speed
                                        but require more memory. Default is 64.
        Returns:
            Iterable[float]: An iterable of relevance scores for each document.
        """

        yield from self._rerank_documents(
            query=query, documents=documents, batch_size=batch_size, **kwargs
        )

    def rerank_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
        batch_size: int = 64,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[float]:
        yield from self._rerank_pairs(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            pairs=pairs,
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
    def _get_worker_class(cls) -> type[TextRerankerWorker]:
        return TextCrossEncoderWorker

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[float]:
        return (float(elem) for elem in output.model_output)

    def token_count(
        self, pairs: Iterable[tuple[str, str]], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        """Returns the number of tokens in the pairs.

        Args:
            pairs: Iterable of tuples, where each tuple contains a query and a document to be tokenized
            batch_size: Batch size for tokenizing

        Returns:
            token count: overall number of tokens in the pairs
        """
        return self._token_count(pairs, batch_size=batch_size, **kwargs)


class TextCrossEncoderWorker(TextRerankerWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxTextCrossEncoder:
        return OnnxTextCrossEncoder(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
