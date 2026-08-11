import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import onnxruntime as ort
from loguru import logger
from numpy.typing import NDArray
from tokenizers import Tokenizer

from qwen3_embed.common.preprocessor_utils import load_tokenizer
from qwen3_embed.common.types import Device, NumpyArray, OnnxProvider
from qwen3_embed.parallel_processor import Worker

# Holds type of the embedding result
T = TypeVar("T")


@dataclass
class OnnxOutputContext:
    model_output: NumpyArray
    attention_mask: NDArray[np.int64] | None = None
    input_ids: NDArray[np.int64] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class OnnxSessionConfig:
    threads: int | None = None
    providers: Sequence[OnnxProvider] | None = None
    cuda: bool | Device = Device.AUTO
    device_id: int | None = None
    parallel_execution: bool = False
    extra_session_options: dict[str, Any] | None = None


class OnnxModel(Generic[T]):
    EXPOSED_SESSION_OPTIONS = ("enable_cpu_mem_arena",)

    @classmethod
    def _get_worker_class(cls) -> type["EmbeddingWorker[T]"]:
        raise NotImplementedError("Subclasses must implement this method")

    def _post_process_onnx_output(self, output: OnnxOutputContext, **kwargs: Any) -> Iterable[T]:
        """Post-process the ONNX model output to convert it into a usable format.

        Args:
            output (OnnxOutputContext): The raw output from the ONNX model.
            **kwargs: Additional keyword arguments that may be needed by specific implementations.

        Returns:
            Iterable[T]: Post-processed output as an iterable of type T.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(self) -> None:
        self.model: ort.InferenceSession | None = None
        self.model_input_names: set[str] | None = None
        self.static_batch_size: int | None = None
        self.tokenizer: Tokenizer | None = None
        self.special_token_to_id: dict[str, int] = {}

    def _preprocess_onnx_input(
        self, onnx_input: dict[str, NumpyArray], **kwargs: Any
    ) -> dict[str, NumpyArray]:
        """
        Preprocess the onnx input.
        """
        return onnx_input

    def _determine_providers(
        self,
        config: OnnxSessionConfig,
        available_providers: list[str],
    ) -> list[OnnxProvider]:
        cuda_available = "CUDAExecutionProvider" in available_providers
        explicit_cuda = config.cuda is True or config.cuda == Device.CUDA

        if explicit_cuda and config.providers is not None:
            warnings.warn(
                f"`cuda` and `providers` are mutually exclusive parameters, "
                f"cuda: {config.cuda}, providers: {config.providers}. If you'd like to use providers, cuda should be one of "
                f"[False, Device.CPU, Device.AUTO].",
                category=UserWarning,
                stacklevel=7,
            )

        dml_available = "DmlExecutionProvider" in available_providers

        onnx_providers: list[OnnxProvider]
        if config.providers is not None:
            onnx_providers = list(config.providers)
        elif explicit_cuda or (config.cuda == Device.AUTO and cuda_available):
            if config.device_id is None:
                onnx_providers = ["CUDAExecutionProvider"]
            else:
                onnx_providers = [("CUDAExecutionProvider", {"device_id": config.device_id})]
        elif config.cuda == Device.AUTO and dml_available:
            onnx_providers = ["DmlExecutionProvider"]
        else:
            onnx_providers = ["CPUExecutionProvider"]

        return onnx_providers

    def _validate_providers(
        self, onnx_providers: list[OnnxProvider], available_providers: list[str]
    ) -> list[str]:
        requested_provider_names: list[str] = []
        for provider in onnx_providers:
            # check providers available
            provider_name = provider if isinstance(provider, str) else provider[0]
            requested_provider_names.append(provider_name)
            if provider_name not in available_providers:
                raise ValueError(
                    f"Provider {provider_name} is not available. Available providers: {available_providers}"
                )
        return requested_provider_names

    def _create_session_options(
        self,
        config: OnnxSessionConfig,
    ) -> Any:
        so = ort.SessionOptions()  # type: ignore[possibly-missing-attribute]
        if config.parallel_execution:
            so.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # type: ignore[possibly-missing-attribute]
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # type: ignore[possibly-missing-attribute]
        # Disable memory pattern optimization to prevent ORT from retaining
        # peak-sized buffers across inferences with varying sequence lengths.
        so.enable_mem_pattern = False

        if config.threads is not None:
            so.intra_op_num_threads = config.threads
            so.inter_op_num_threads = config.threads

        if config.extra_session_options is not None:
            self.add_extra_session_options(so, config.extra_session_options)
        return so

    def _instantiate_onnx_session(
        self,
        model_path: Path,
        config: OnnxSessionConfig | None = None,
    ) -> tuple[ort.InferenceSession, list[str]]:
        if config is None:
            config = OnnxSessionConfig()

        # List of Execution Providers: https://onnxruntime.ai/docs/execution-providers
        available_providers = ort.get_available_providers()  # type: ignore[possibly-missing-attribute]

        onnx_providers = self._determine_providers(config, available_providers)
        requested_provider_names = self._validate_providers(onnx_providers, available_providers)
        so = self._create_session_options(config)

        session = ort.InferenceSession(str(model_path), providers=onnx_providers, sess_options=so)
        input_names = [node.name for node in session.get_inputs()]
        logger.info(f"ONNX session created with providers: {session.get_providers()}")
        if "CUDAExecutionProvider" in requested_provider_names:
            current_providers = session.get_providers()
            if "CUDAExecutionProvider" not in current_providers:
                warnings.warn(
                    f"Attempt to set CUDAExecutionProvider failed. Current providers: {current_providers}."
                    "If you are using CUDA 12.x, install onnxruntime-gpu via "
                    "`pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return session, input_names

    @staticmethod
    def _detect_static_batch_size(session: "ort.InferenceSession") -> int | None:
        """Return the batch size the graph is pinned to, or None if it is dynamic.

        ONNX declares each dimension either as an int (static) or as a symbolic
        ``dim_param`` string such as ``"batch_size"`` (dynamic). Both the inputs
        and the outputs are inspected because they can disagree: causal-LM
        exports keep the symbolic ``batch_size`` on ``input_ids`` while the graph
        body was traced with a fixed batch, and only the output declaration
        carries the literal. Feeding a larger batch to such a graph fails deep
        inside onnxruntime with a buffer shape mismatch, so the smallest literal
        found on a batch axis wins.
        """
        declared = [
            node.shape[0]
            for node in (*session.get_inputs(), *session.get_outputs())
            # Rank < 2 has no batch axis to speak of, e.g. a scalar side input
            # declared as ``[1]`` is a length, not a batch of one.
            if node.shape is not None and len(node.shape) >= 2 and isinstance(node.shape[0], int)
        ]
        return min(declared) if declared else None

    def _load_onnx_model(
        self,
        model_dir: Path,
        model_file: str,
        config: OnnxSessionConfig | None = None,
    ) -> tuple[ort.InferenceSession, list[str]]:
        model_path = model_dir / model_file
        self.model, input_names = self._instantiate_onnx_session(
            model_path=model_path,
            config=config,
        )
        self.model_input_names = set(input_names)
        self.static_batch_size = self._detect_static_batch_size(self.model)
        self.tokenizer, self.special_token_to_id = load_tokenizer(model_dir=model_dir)
        return self.model, input_names

    @classmethod
    def _select_exposed_session_options(cls, model_kwargs: dict[str, Any]) -> dict[str, Any]:
        """A convenience method to select the exposed session options in models

        Args:
            model_kwargs (dict[str, Any]): The model kwargs.

        Returns:
            dict[str, Any]: a dict with filtered exposed session options.
        """
        return {k: v for k, v in model_kwargs.items() if k in cls.EXPOSED_SESSION_OPTIONS}

    @classmethod
    def add_extra_session_options(
        cls,
        session_options: "ort.SessionOptions",
        extra_options: dict[str, Any],
    ) -> None:
        """Add extra session options to the existing options object in-place

        Args:
            session_options (ort.SessionOptions): The existing session options object.
            extra_options (dict[str, Any]): The extra session options available in cls.EXPOSED_SESSION_OPTIONS.

        Returns:
            None
        """
        for option in extra_options:
            if option not in cls.EXPOSED_SESSION_OPTIONS:
                raise ValueError(
                    f"{option} is unknown or not exposed (exposed options: {cls.EXPOSED_SESSION_OPTIONS})"
                )
        if "enable_cpu_mem_arena" in extra_options:
            session_options.enable_cpu_mem_arena = extra_options["enable_cpu_mem_arena"]

    def load_onnx_model(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def onnx_embed(self, *args: Any, **kwargs: Any) -> OnnxOutputContext:
        raise NotImplementedError("Subclasses must implement this method")


class EmbeddingWorker(Worker, Generic[T]):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> OnnxModel[T]:
        raise NotImplementedError()

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ):
        self.model = self.init_embedding(model_name, cache_dir, **kwargs)

    @classmethod
    def start(cls, model_name: str, cache_dir: str, **kwargs: Any) -> "EmbeddingWorker[T]":
        return cls(model_name=model_name, cache_dir=cache_dir, **kwargs)

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        raise NotImplementedError("Subclasses must implement this method")
