"""End-to-end converter pipelines over real ONNX graphs, tokenizers and manifests.

Every artifact is a real ONNX model built with ``onnx.helper``, saved under ``tmp_path`` and
executed by ``onnx.reference``; tokenizers are real ``tokenizers`` files and manifests are read
back from disk. Only boundaries the unit suite cannot run are replaced, and each
``install_fake_*`` helper names its boundary: onnxruntime (no CPython 3.15 wheel), torch,
transformers, optimum, modal, gguf and the llama.cpp subprocesses (heavy or native), and the
Hugging Face downloads (network).
"""

import ast
import asyncio
import contextlib
import copy
import enum
import importlib.machinery
import importlib.metadata
import json
import shutil
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import huggingface_hub
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from tokenizers import Tokenizer, models, pre_tokenizers

from fastretrieval.common.model_description import PoolingType
from fastretrieval.convert import gguf as gguf_module
from fastretrieval.convert import modal_backend
from fastretrieval.convert import onnx as onnx_module
from fastretrieval.convert.cli import main
from fastretrieval.convert.gguf import convert_gguf
from fastretrieval.convert.manifest import MANIFEST_FILENAME, load_manifest
from fastretrieval.convert.modal_backend import run_remote
from fastretrieval.convert.onnx import convert_onnx

VOCAB = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "yes": 4, "no": 5, "the": 6, "fox": 7}
HIDDEN = 4
OPSET = 21
TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
EMBEDDINGS = np.random.default_rng(7).standard_normal((len(VOCAB), HIDDEN)).astype(np.float32)
LM_HEAD = np.random.default_rng(11).standard_normal((len(VOCAB), HIDDEN)).astype(np.float32)
CLASSIFIER = np.random.default_rng(13).standard_normal((HIDDEN, 1)).astype(np.float32)
TOKEN_IDS = np.array([[2, 3, 0], [6, 7, 1]], dtype=np.int64)
TOKEN_MASK = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
ARCHITECTURES = {"bert": "BertModel", "qwen3": "Qwen3ForCausalLM"}


# ---------------------------------------------------------------------------
# Real model directories and ONNX graphs
# ---------------------------------------------------------------------------
def install_module(monkeypatch, name, **attributes):
    """Register a fake module that ``find_spec`` and ``import`` both resolve."""
    module = types.ModuleType(name)
    # find_spec() answers from sys.modules[name].__spec__, so the fake needs a spec.
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def make_tokenizer():
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def write_model_dir(path, *, model_type, skip=()):
    """Write a Hugging Face style model folder with a real word-level tokenizer."""
    config = {
        "architectures": [ARCHITECTURES[model_type]],
        "model_type": model_type,
        "hidden_size": HIDDEN,
        "max_position_embeddings": 16,
        "vocab_size": len(VOCAB),
    }
    contents = {
        "config.json": json.dumps(config),
        "tokenizer.json": make_tokenizer().to_str(),
        "tokenizer_config.json": json.dumps({"pad_token": "[PAD]", "unk_token": "[UNK]"}),
    }
    path.mkdir(parents=True, exist_ok=True)
    for name, text in contents.items():
        if name not in skip:
            (path / name).write_text(text, encoding="utf-8")
    return path


def token_inputs(names=("input_ids", "attention_mask")):
    return [
        helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "sequence"])
        for name in names
    ]


def hidden_state_graph(table=EMBEDDINGS, *, input_names=("input_ids", "attention_mask")):
    """Encoder stand-in: ``last_hidden_state = table[input_ids] * attention_mask``."""
    nodes = [
        helper.make_node("Gather", ["table", "input_ids"], ["gathered"]),
        helper.make_node("Cast", ["attention_mask"], ["mask_float"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask_float", "last_axis"], ["mask_3d"]),
        helper.make_node("Mul", ["gathered", "mask_3d"], ["last_hidden_state"]),
    ]
    initializers = [
        numpy_helper.from_array(np.asarray(table, dtype=np.float32), "table"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "last_axis"),
    ]
    output = helper.make_tensor_value_info(
        "last_hidden_state", TensorProto.FLOAT, ["batch", "sequence", HIDDEN]
    )
    return helper.make_graph(
        nodes, "hidden_state", token_inputs(input_names), [output], initializer=initializers
    )


def token_logits_graph(table, weight, *, position):
    """Logits of one token position: ``table[input_ids][:, position] @ weight``."""
    weight = np.asarray(weight, dtype=np.float32)
    nodes = [
        helper.make_node("Gather", ["table", "input_ids"], ["hidden"]),
        helper.make_node("Gather", ["hidden", "position"], ["picked"], axis=1),
        helper.make_node("MatMul", ["picked", "weight"], ["logits"]),
    ]
    initializers = [
        numpy_helper.from_array(np.asarray(table, dtype=np.float32), "table"),
        numpy_helper.from_array(np.array(position, dtype=np.int64), "position"),
        numpy_helper.from_array(weight, "weight"),
    ]
    output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", weight.shape[1]])
    return helper.make_graph(
        nodes, "token_logits", token_inputs(), [output], initializer=initializers
    )


def save_model(graph, path, *, opset=OPSET, external_data=False):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    if external_data:
        onnx.save(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{path.name}.data",
            size_threshold=0,
        )
    else:
        onnx.save(model, str(path))
    return path


def run_model(path, input_ids=TOKEN_IDS, attention_mask=TOKEN_MASK):
    evaluator = ReferenceEvaluator(onnx.load(str(path)))
    outputs = evaluator.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    assert isinstance(outputs, list)
    return outputs[0]


def relative_files(directory):
    return sorted(
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    )


def cast_targets(graph):
    return [
        attribute.i
        for node in graph.node
        if node.op_type == "Cast"
        for attribute in node.attribute
        if attribute.name == "to"
    ]


# ---------------------------------------------------------------------------
# Boundary (a): onnxruntime
# ---------------------------------------------------------------------------
def install_fake_onnxruntime(monkeypatch):
    """Sessions execute the real graph on onnx.reference; quantizers record their calls."""
    record = SimpleNamespace(sessions=[], quantize_dynamic=[], nbits=[])

    class InferenceSession:
        def __init__(self, path, providers=None):
            record.sessions.append((Path(path), providers))
            model = onnx.load(path)
            self._evaluator = ReferenceEvaluator(model)
            initializers = {tensor.name for tensor in model.graph.initializer}
            self._inputs = [
                SimpleNamespace(name=value.name)
                for value in model.graph.input
                if value.name not in initializers
            ]
            self._outputs = [SimpleNamespace(name=value.name) for value in model.graph.output]

        def get_inputs(self):
            return self._inputs

        def get_outputs(self):
            return self._outputs

        def run(self, output_names, feed):
            return self._evaluator.run(output_names, feed)

    def quantize_dynamic(*, model_input, model_output, weight_type):
        record.quantize_dynamic.append(
            {
                "model_input": model_input,
                "model_output": model_output,
                "weight_type": weight_type,
                "fp32_files": sorted(path.name for path in Path(model_input).parent.iterdir()),
            }
        )
        onnx.save(onnx.load(model_input), model_output)

    class MatMulNBitsQuantizer:
        def __init__(self, *, model, algo_config):
            record.nbits.append({"model": model, "algo_config": algo_config, "processed": False})
            self.model = SimpleNamespace(model=onnx.load(model))

        def process(self):
            record.nbits[-1]["processed"] = True

    # Boundary (a): onnxruntime ships no CPython 3.15 wheel, so the unit suite cannot import it.
    install_module(monkeypatch, "onnxruntime", InferenceSession=InferenceSession)
    install_module(
        monkeypatch,
        "onnxruntime.quantization",
        QuantType=SimpleNamespace(QInt8="QInt8"),
        quantize_dynamic=quantize_dynamic,
    )
    install_module(
        monkeypatch,
        "onnxruntime.quantization.matmul_nbits_quantizer",
        DefaultWeightOnlyQuantConfig=dict,
        MatMulNBitsQuantizer=MatMulNBitsQuantizer,
    )
    return record


# ---------------------------------------------------------------------------
# Boundary (b): torch and transformers
# ---------------------------------------------------------------------------
class FakeTensor:
    """numpy-backed stand-in for the slice of ``torch.Tensor`` fastretrieval calls."""

    device = "cpu"

    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = tuple(item.values if isinstance(item, FakeTensor) else item for item in index)
        return FakeTensor(self.values[index])

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def __sub__(self, other):
        return FakeTensor(self.values - other)

    def __truediv__(self, other):
        return FakeTensor(self.values / np.asarray(other))

    def to(self, *, dtype):
        return FakeTensor(self.values.astype(dtype))

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.values, dim))

    def squeeze(self, dim):
        return FakeTensor(np.squeeze(self.values, axis=dim))

    def sum(self, dim, keepdim=False):
        return FakeTensor(self.values.sum(axis=dim, keepdims=keepdim))

    def clamp(self, *, min):
        return FakeTensor(np.maximum(self.values, min))

    def clamp_(self, *, min):
        np.maximum(self.values, min, out=self.values)
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeModule:
    forward: Callable[..., Any]

    def __init__(self):
        pass

    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FakeLinear(FakeModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = SimpleNamespace(data=np.zeros((out_features, in_features), np.float32))
        self.bias = bias

    def forward(self, values):
        return FakeTensor(np.asarray(values) @ np.asarray(self.weight.data).T)


class FakeBody(FakeModule):
    """Transformer body: ``last_hidden_state = table[input_ids] * attention_mask``."""

    def __init__(self, table):
        super().__init__()
        self.table = table

    def forward(self, input_ids, attention_mask):
        mask = np.asarray(attention_mask)[..., None].astype(np.float32)
        hidden = self.table[np.asarray(input_ids)] * mask
        return SimpleNamespace(last_hidden_state=FakeTensor(hidden))


class FakeClassifier(FakeModule):
    """Sequence classifier: one logit from the first token's embedding."""

    def forward(self, input_ids, attention_mask):
        first = EMBEDDINGS[np.asarray(input_ids)][:, 0, :]
        return SimpleNamespace(logits=FakeTensor(first @ CLASSIFIER))


class FakeConfig:
    def __init__(self, payload):
        self.payload = payload
        self.use_cache = True

    def save_pretrained(self, directory):
        payload = {**self.payload, "use_cache": self.use_cache}
        (Path(directory) / "config.json").write_text(json.dumps(payload), encoding="utf-8")


class FakeCausalLM(FakeModule):
    def __init__(self, source):
        super().__init__()
        self.model = FakeBody(EMBEDDINGS)
        self.lm_head = FakeLinear(HIDDEN, len(VOCAB), bias=False)
        self.lm_head.weight.data = LM_HEAD
        config = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
        self.config = FakeConfig(config)


class FakeTokenizer:
    """``AutoTokenizer`` stand-in that encodes with the model folder's real tokenizer.json."""

    def __init__(self, directory, calls):
        self.directory = Path(directory)
        self._json = (self.directory / "tokenizer.json").read_text(encoding="utf-8")
        self._calls = calls
        self._backend = Tokenizer.from_str(self._json)
        self.unk_token_id = self._backend.token_to_id("[UNK]")

    def convert_tokens_to_ids(self, token):
        return self._backend.token_to_id(token)

    def __call__(self, text, **kwargs):
        self._calls.append((text, kwargs))
        backend = Tokenizer.from_str(self._json)
        if kwargs.get("truncation") and kwargs.get("max_length") is not None:
            backend.enable_truncation(max_length=kwargs["max_length"])
        if kwargs.get("padding"):
            backend.enable_padding(pad_id=VOCAB["[PAD]"], pad_token="[PAD]")
        encodings = backend.encode_batch([text] if isinstance(text, str) else list(text))
        batch = {
            "input_ids": np.array([item.ids for item in encodings], dtype=np.int64),
            "attention_mask": np.array([item.attention_mask for item in encodings], np.int64),
        }
        if kwargs["return_tensors"] == "pt":
            return {key: FakeTensor(value) for key, value in batch.items()}
        return batch

    def save_pretrained(self, directory):
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy2(self.directory / name, Path(directory) / name)


def install_fake_torch(monkeypatch):
    """The fake exporter traces the module, then writes the traced head as a real graph."""
    record = SimpleNamespace(exports=[], normalize=[])

    def normalize(tensor, *, p, dim):
        record.normalize.append({"p": p, "dim": dim})
        values = np.asarray(tensor)
        norms = np.linalg.norm(values, ord=p, axis=dim, keepdims=True)
        return FakeTensor(values / np.maximum(norms, 1e-12))

    def export(module, args, path, **options):
        traced = np.asarray(module(*args))
        record.exports.append(
            {"args": [np.asarray(arg) for arg in args], "path": Path(path), "traced": traced}
            | options
        )
        head = np.asarray(module.head.weight.data).T
        graph = token_logits_graph(module.body.table, head, position=-1)
        save_model(graph, Path(path), opset=options["opset_version"])

    # Boundary (b): torch is not a test dependency (and has no wheel for every CI row).
    install_module(
        monkeypatch,
        "torch",
        Tensor=FakeTensor,
        float32=np.float32,
        no_grad=contextlib.nullcontext,
        bmm=lambda left, right: FakeTensor(np.matmul(np.asarray(left), np.asarray(right))),
        arange=lambda count, device=None: FakeTensor(np.arange(count)),
        nn=SimpleNamespace(
            Module=FakeModule,
            Linear=FakeLinear,
            functional=SimpleNamespace(normalize=normalize),
        ),
        onnx=SimpleNamespace(export=export),
    )
    return record


def install_fake_transformers(monkeypatch):
    record = SimpleNamespace(loads=[], tokenizer_calls=[])

    def loader(kind, build):
        def from_pretrained(source, **kwargs):
            record.loads.append((kind, source, kwargs))
            return build(source)

        return SimpleNamespace(from_pretrained=from_pretrained)

    # Boundary (b): transformers is not a test dependency and would load real checkpoints.
    install_module(
        monkeypatch,
        "transformers",
        AutoTokenizer=loader(
            "AutoTokenizer", lambda source: FakeTokenizer(source, record.tokenizer_calls)
        ),
        AutoModel=loader("AutoModel", lambda source: FakeBody(EMBEDDINGS)),
        AutoModelForSequenceClassification=loader(
            "AutoModelForSequenceClassification", lambda source: FakeClassifier()
        ),
        AutoModelForCausalLM=loader("AutoModelForCausalLM", FakeCausalLM),
    )
    return record


# ---------------------------------------------------------------------------
# Boundary (b): optimum
# ---------------------------------------------------------------------------
def install_fake_optimum(monkeypatch):
    """``main_export`` writes a real graph at the export root, as recent optimum does."""
    record = SimpleNamespace(exports=[], external_data=False, version="1.27.0")

    def main_export(model_id, *, output, task, opset):
        record.exports.append(
            {"model_id": model_id, "output": Path(output), "task": task, "opset": opset}
        )
        if task == "text-classification":
            graph = token_logits_graph(EMBEDDINGS, CLASSIFIER, position=0)
        else:
            graph = hidden_state_graph()
        root = Path(output)
        save_model(graph, root / "model.onnx", opset=opset, external_data=record.external_data)
        (root / "tokenizer.json").write_text(make_tokenizer().to_str(), encoding="utf-8")

    real_version = importlib.metadata.version

    def version(name):
        if name != "optimum":
            return real_version(name)
        if record.version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return record.version

    # Boundary (b): optimum (with the torch export under it) is not a test dependency.
    install_module(monkeypatch, "optimum")
    install_module(monkeypatch, "optimum.exporters")
    install_module(monkeypatch, "optimum.exporters.onnx", main_export=main_export)
    # Boundary (b): the optimum version comes from installed package metadata.
    monkeypatch.setattr(onnx_module.importlib.metadata, "version", version)
    return record


# ---------------------------------------------------------------------------
# Boundary (c): Hugging Face downloads
# ---------------------------------------------------------------------------
class FakeHub:
    """Hub repositories are plain folders under ``root``."""

    def __init__(self, root):
        self.root = root
        self.calls = []

    def repo_dir(self, repo_id):
        return self.root / repo_id.replace("/", "--")

    def publish(self, repo_id, **model_options):
        return write_model_dir(self.repo_dir(repo_id), **model_options)

    def hf_hub_download(self, *, repo_id, filename):
        self.calls.append(("hf_hub_download", repo_id, filename))
        return str(self.repo_dir(repo_id) / filename)

    def snapshot_download(self, *, repo_id, local_dir):
        self.calls.append(("snapshot_download", repo_id, Path(local_dir)))
        shutil.copytree(self.repo_dir(repo_id), local_dir)
        return local_dir


def install_fake_hub(monkeypatch, root):
    hub = FakeHub(root)
    # Boundary (c): hub downloads need the network; the fake serves local folders.
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub.snapshot_download)
    return hub


# ---------------------------------------------------------------------------
# Boundary (b): llama.cpp and gguf
# ---------------------------------------------------------------------------
def make_llama_checkout(root):
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("# converter entry point\n", encoding="utf-8")
    (root / "build" / "bin" / "llama-quantize").write_text("quantizer\n", encoding="utf-8")
    return root


class FakeLlamaCpp:
    """Plays both llama.cpp subprocess steps by writing their output files."""

    def __init__(self, failing_step=None):
        self.failing_step = failing_step
        self.calls = []
        self.model_files = None

    def run(self, cmd, **options):
        self.calls.append((list(cmd), options))
        step = "convert" if Path(cmd[1]).name == "convert_hf_to_gguf.py" else "quantize"
        if step == self.failing_step:
            return subprocess.CompletedProcess(cmd, 1, "", f"{step} failed")
        if step == "convert":
            self.model_files = sorted(path.name for path in Path(cmd[2]).iterdir())
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF f16\n")
        else:
            Path(cmd[2]).write_bytes(Path(cmd[1]).read_bytes() + cmd[3].encode("ascii"))
        return subprocess.CompletedProcess(cmd, 0, "", "")


def install_fake_llama(monkeypatch, failing_step=None):
    llama = FakeLlamaCpp(failing_step)
    # Boundary (b): convert_hf_to_gguf.py and llama-quantize are native llama.cpp subprocesses.
    monkeypatch.setattr(gguf_module.subprocess, "run", llama.run)
    # Boundary (b): the gguf package is only probed for presence by convert_gguf.
    install_module(monkeypatch, "gguf")
    return llama


# ---------------------------------------------------------------------------
# Boundary (b): the Modal SDK and its container
# ---------------------------------------------------------------------------
class EntryType(enum.Enum):
    FILE = 1
    DIRECTORY = 2


class FakeModal:
    """The container is a local folder.

    Image mounts are copied into it, ``/mnt/...`` payload strings are mapped onto it, and
    the output Volume is its ``mnt/output`` subfolder, addressed the way the code
    addresses it today.
    """

    def __init__(self, container_root):
        self.root = container_root
        self.apps = []
        self.volumes = []

    def to_container(self, value):
        if isinstance(value, str) and value.startswith("/mnt/"):
            return self.root.joinpath(*PurePosixPath(value).parts[1:]).as_posix()
        if isinstance(value, dict):
            return {key: self.to_container(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(self.to_container(item) for item in value)
        return value

    def from_container(self, value):
        if isinstance(value, str) and Path(value).is_relative_to(self.root):
            return "/" + Path(value).relative_to(self.root).as_posix()
        if isinstance(value, dict):
            return {key: self.from_container(item) for key, item in value.items()}
        return value

    def install(self, monkeypatch):
        fake = self

        class Image:
            def __init__(self, steps):
                self.steps = steps

            @classmethod
            def debian_slim(cls, *, python_version):
                return cls((("debian_slim", python_version),))

            def uv_pip_install(self, *packages):
                return Image((*self.steps, ("uv_pip_install", *packages)))

            def add_local_python_source(self, *modules):
                return Image((*self.steps, ("add_local_python_source", *modules)))

            def add_local_dir(self, local_path, remote_path, *, copy):
                return Image((*self.steps, ("add_local_dir", local_path, remote_path, copy)))

        class Volume:
            def __init__(self):
                self.local = fake.root / "mnt" / "output"
                self.local.mkdir(parents=True, exist_ok=True)
                self.commits = 0
                self.reloads = 0

            @classmethod
            @contextlib.asynccontextmanager
            async def ephemeral(cls):
                volume = cls()
                fake.volumes.append(volume)
                yield volume

            async def commit(self):
                self.commits += 1

            async def reload(self):
                self.reloads += 1

            def _local_path(self, path):
                return self.local.joinpath(*PurePosixPath(path).relative_to("/mnt/output").parts)

            async def iterdir(self, path, *, recursive):
                base = self._local_path(path)
                for item in sorted(base.rglob("*") if recursive else base.iterdir()):
                    kind = EntryType.FILE if item.is_file() else EntryType.DIRECTORY
                    relative = item.relative_to(self.local).as_posix()
                    yield SimpleNamespace(path=f"/mnt/output/{relative}", type=kind)

            async def read_file(self, path):
                data = self._local_path(path).read_bytes()
                middle = len(data) // 2
                yield data[:middle]
                yield data[middle:]

        class Function:
            def __init__(self, app, raw, options):
                self.app = app
                self.raw = raw
                self.options = options
                self.calls = []
                self.remote = SimpleNamespace(aio=self._aio)

            async def _aio(self, *args):
                if not self.app.running:
                    raise RuntimeError("Modal functions run only inside app.run()")
                self.calls.append(copy.deepcopy(args))
                for step in self.options["image"].steps:
                    if step[0] == "add_local_dir":
                        shutil.copytree(step[1], fake.to_container(step[2]))
                result = await self.raw(*fake.to_container(args))
                return fake.from_container(result)

        class App:
            def __init__(self, name):
                self.name = name
                self.running = False
                self.functions = []
                fake.apps.append(self)

            def function(self, **options):
                def decorate(raw):
                    function = Function(self, raw, options)
                    self.functions.append(function)
                    return function

                return decorate

            @contextlib.asynccontextmanager
            async def run(self):
                self.running = True
                try:
                    yield self
                finally:
                    self.running = False

        # Boundary (b): the modal SDK is not a test dependency and would start cloud containers.
        install_module(monkeypatch, "modal", Image=Image, Volume=Volume, App=App)
        return self


@pytest.fixture
def stack(monkeypatch):
    """The ONNX conversion boundaries; each installer documents its own boundary."""
    return SimpleNamespace(
        ort=install_fake_onnxruntime(monkeypatch),
        torch=install_fake_torch(monkeypatch),
        transformers=install_fake_transformers(monkeypatch),
        optimum=install_fake_optimum(monkeypatch),
    )


def manifest_fields(directory, *keys):
    manifest = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return {key: manifest[key] for key in keys}


# ---------------------------------------------------------------------------
# convert_onnx
# ---------------------------------------------------------------------------
def test_convert_onnx_exports_quantizes_and_promotes_a_local_model(tmp_path, stack):
    source = write_model_dir(tmp_path / "tiny-bert", model_type="bert")
    stack.optimum.external_data = True
    out = tmp_path / "artifacts" / "tiny-bert-onnx"

    sizes = convert_onnx(str(source), out, pooling="mean", normalization=True)

    onnx_dir = out / "onnx"
    assert relative_files(out) == [
        "config.json",
        MANIFEST_FILENAME,
        "onnx/model_q4f16.onnx",
        "onnx/model_quantized.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert list(out.parent.iterdir()) == [out]
    assert list(sizes) == ["int8", "q4f16"]
    assert sizes == {
        "int8": pytest.approx((onnx_dir / "model_quantized.onnx").stat().st_size / 1024**2),
        "q4f16": pytest.approx((onnx_dir / "model_q4f16.onnx").stat().st_size / 1024**2),
    }
    assert manifest_fields(
        out,
        "source",
        "task",
        "model_family",
        "pooling",
        "normalization",
        "output_dim",
        "output_shape",
        "max_seq_len",
        "artifact_formats",
        "quantization",
        "exporter_version",
        "variants",
    ) == {
        "source": str(source),
        "task": "dense",
        "model_family": "bert",
        "pooling": "MEAN",
        "normalization": True,
        "output_dim": HIDDEN,
        "output_shape": [HIDDEN],
        "max_seq_len": 16,
        "artifact_formats": ["onnx"],
        "quantization": "int8,q4f16",
        "exporter_version": "optimum 1.27.0",
        "variants": ["int8", "q4f16"],
    }

    (export,) = stack.optimum.exports
    staging = export["output"]
    assert export == {
        "model_id": str(source),
        "output": staging,
        "task": "feature-extraction",
        "opset": 21,
    }
    assert staging.name == "artifact"
    assert staging.parent.name.startswith(".tiny-bert-onnx.staging-")
    assert staging.parent.parent == out.parent
    fp32 = staging / "onnx" / "model.onnx"
    assert stack.ort.quantize_dynamic == [
        {
            "model_input": str(fp32),
            "model_output": str(staging / "onnx" / "model_quantized.onnx"),
            "weight_type": "QInt8",
            "fp32_files": ["model.onnx", "model.onnx.data"],
        }
    ]
    assert stack.ort.nbits == [
        {
            "model": str(fp32),
            "algo_config": {
                "block_size": 128,
                "is_symmetric": True,
                "accuracy_level": 4,
                "bits": 4,
            },
            "processed": True,
        }
    ]
    assert [(path.name, providers) for path, providers in stack.ort.sessions] == [
        ("model_q4f16.onnx", ["CPUExecutionProvider"]),
        ("model_quantized.onnx", ["CPUExecutionProvider"]),
    ]

    expected = EMBEDDINGS[TOKEN_IDS] * TOKEN_MASK[..., None]
    np.testing.assert_allclose(run_model(onnx_dir / "model_quantized.onnx"), expected, rtol=1e-6)
    q4f16 = onnx.load(str(onnx_dir / "model_q4f16.onnx"))
    initializer_types = {tensor.name: tensor.data_type for tensor in q4f16.graph.initializer}
    assert initializer_types["table"] == TensorProto.FLOAT16
    assert cast_targets(q4f16.graph) == [TensorProto.FLOAT16]
    assert len(q4f16.graph.value_info) == 0
    half = run_model(onnx_dir / "model_q4f16.onnx")
    assert half.dtype == np.float16
    np.testing.assert_allclose(half.astype(np.float32), expected, atol=1e-2)
    for name in ("config.json", "tokenizer_config.json"):
        assert (out / name).read_bytes() == (source / name).read_bytes()
    assert Tokenizer.from_file(str(out / "tokenizer.json")).token_to_id("fox") == VOCAB["fox"]


def test_convert_onnx_fetches_missing_tokenizer_files_for_a_hub_cross_encoder(
    tmp_path, monkeypatch, stack
):
    hub = install_fake_hub(monkeypatch, tmp_path / "hub")
    repo = hub.publish("acme/tiny-cross", model_type="bert")
    stack.optimum.version = None
    out = tmp_path / "artifacts" / "tiny-cross-onnx"

    sizes = convert_onnx(
        "acme/tiny-cross",
        out,
        task="cross_encoder",
        variants=["int8"],
        pooling="cls",
        normalization=False,
        output_dim=1,
    )

    assert list(sizes) == ["int8"]
    assert relative_files(out) == [
        "config.json",
        MANIFEST_FILENAME,
        "onnx/model_quantized.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert hub.calls == [
        ("hf_hub_download", "acme/tiny-cross", "config.json"),
        ("hf_hub_download", "acme/tiny-cross", "config.json"),
        ("hf_hub_download", "acme/tiny-cross", "tokenizer_config.json"),
    ]
    (export,) = stack.optimum.exports
    assert (export["model_id"], export["task"], export["opset"]) == (
        "acme/tiny-cross",
        "text-classification",
        21,
    )
    assert stack.ort.nbits == []
    assert manifest_fields(
        out,
        "source",
        "task",
        "pooling",
        "normalization",
        "output_dim",
        "output_shape",
        "quantization",
        "exporter_version",
        "variants",
    ) == {
        "source": "acme/tiny-cross",
        "task": "cross_encoder",
        "pooling": "CLS",
        "normalization": False,
        "output_dim": 1,
        "output_shape": [1],
        "quantization": "int8",
        "exporter_version": "optimum unknown",
        "variants": ["int8"],
    }
    for name in ("config.json", "tokenizer_config.json"):
        assert (out / name).read_bytes() == (repo / name).read_bytes()
    logits = run_model(out / "onnx" / "model_quantized.onnx")
    np.testing.assert_allclose(logits, EMBEDDINGS[TOKEN_IDS][:, 0, :] @ CLASSIFIER, rtol=1e-6)


def test_convert_onnx_rejects_an_empty_variant_list(tmp_path):
    with pytest.raises(ValueError, match="at least one variant is required"):
        convert_onnx("acme/tiny-bert", tmp_path / "out", variants=[])


def test_convert_onnx_rejects_duplicate_variants(tmp_path):
    with pytest.raises(ValueError, match="variants must not contain duplicates"):
        convert_onnx("acme/tiny-bert", tmp_path / "out", variants=["int8", "int8"])


def test_convert_onnx_fails_when_a_local_tokenizer_file_is_missing(tmp_path, stack):
    source = write_model_dir(
        tmp_path / "tiny-bert", model_type="bert", skip=("tokenizer_config.json",)
    )
    out = tmp_path / "artifacts" / "tiny-bert-onnx"

    with pytest.raises(
        FileNotFoundError,
        match=r"tokenizer file required by manifest is missing: tokenizer_config\.json",
    ):
        convert_onnx(str(source), out, variants=["int8"], pooling="mean", normalization=True)

    assert list(out.parent.iterdir()) == []
    assert stack.ort.quantize_dynamic == []


def test_convert_onnx_keeps_a_non_empty_output_directory_and_removes_its_staging(tmp_path, stack):
    source = write_model_dir(tmp_path / "tiny-bert", model_type="bert")
    out = tmp_path / "artifacts" / "tiny-bert-onnx"
    out.mkdir(parents=True)
    (out / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output directory is not empty"):
        convert_onnx(str(source), out, variants=["int8"], pooling="mean", normalization=True)

    assert relative_files(out) == ["keep.txt"]
    assert list(out.parent.iterdir()) == [out]
    assert [path.name for path, _ in stack.ort.sessions] == ["model_quantized.onnx"]


def test_convert_onnx_lists_every_missing_conversion_dependency(tmp_path, monkeypatch):
    source = write_model_dir(tmp_path / "tiny-bert", model_type="bert")
    for name in ("torch", "transformers", "optimum", "onnxruntime"):
        # Boundary (a)/(b): a None entry makes find_spec report the package as not installed.
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(
        ImportError,
        match=r"Missing conversion dependencies: torch, transformers, optimum, onnxruntime\.",
    ):
        convert_onnx(str(source), tmp_path / "out", pooling="mean", normalization=True)

    assert not (tmp_path / "out").exists()


def test_convert_onnx_exports_a_yes_no_head_from_a_causal_lm(tmp_path, stack):
    source = write_model_dir(tmp_path / "tiny-qwen3", model_type="qwen3")
    out = tmp_path / "artifacts" / "tiny-qwen3-yes-no"

    sizes = convert_onnx(
        str(source),
        out,
        task="generative-reranker",
        variants=["int8"],
        pooling="last_token",
        normalization=False,
        yes_no=("yes", "no"),
    )

    assert list(sizes) == ["int8"]
    assert stack.optimum.exports == []
    head = LM_HEAD[[VOCAB["no"], VOCAB["yes"]]].T
    (export,) = stack.torch.exports
    assert export["path"].parts[-2:] == ("onnx", "model.onnx")
    assert {key: export[key] for key in export if key not in {"args", "path", "traced"}} == {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        "opset_version": 21,
        "do_constant_folding": True,
    }
    np.testing.assert_array_equal(export["args"][0], [[VOCAB["hello"], VOCAB["world"]]])
    np.testing.assert_array_equal(export["args"][1], [[1, 1]])
    np.testing.assert_allclose(export["traced"], EMBEDDINGS[[VOCAB["world"]]] @ head, rtol=1e-6)
    assert stack.transformers.loads == [
        ("AutoTokenizer", str(source), {}),
        (
            "AutoModelForCausalLM",
            str(source),
            {"torch_dtype": np.float32, "low_cpu_mem_usage": True, "device_map": "cpu"},
        ),
    ]
    assert stack.transformers.tokenizer_calls == [("hello world", {"return_tensors": "pt"})]

    assert relative_files(out) == [
        "config.json",
        MANIFEST_FILENAME,
        "onnx/model_quantized.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["use_cache"] is False
    assert manifest_fields(
        out, "task", "pooling", "normalization", "output_dim", "output_shape", "model_family"
    ) == {
        "task": "generative_reranker",
        "pooling": "LAST_TOKEN",
        "normalization": False,
        "output_dim": 2,
        "output_shape": [2],
        "model_family": "qwen3",
    }
    logits = run_model(out / "onnx" / "model_quantized.onnx")
    np.testing.assert_allclose(logits, EMBEDDINGS[TOKEN_IDS][:, -1, :] @ head, rtol=1e-6)


# ---------------------------------------------------------------------------
# convert_gguf
# ---------------------------------------------------------------------------
GGUF_OPTIONS = {"capture_output": True, "text": True, "check": False, "timeout": 3600}


def test_convert_gguf_converts_quantizes_and_promotes_a_local_model(tmp_path, monkeypatch):
    source = write_model_dir(tmp_path / "Tiny-Qwen3", model_type="qwen3")
    llama_root = make_llama_checkout(tmp_path / "llama.cpp")
    llama = install_fake_llama(monkeypatch)
    out = tmp_path / "artifacts" / "tiny-qwen3-gguf"

    result = convert_gguf(
        str(source),
        out,
        llama_cpp=str(llama_root),
        pooling="last_token",
        normalization=True,
    )

    assert result == out / "tiny-qwen3-q4-k-m.gguf"
    assert result.read_bytes() == b"GGUF f16\nQ4_K_M"
    assert relative_files(out) == [
        "config.json",
        MANIFEST_FILENAME,
        "tiny-qwen3-q4-k-m.gguf",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert list(out.parent.iterdir()) == [out]
    (convert_cmd, convert_options), (quantize_cmd, quantize_options) = llama.calls
    staged_model = Path(convert_cmd[2])
    f16 = staged_model.parent / "tiny-qwen3-f16.gguf"
    assert staged_model.name == "model"
    assert staged_model.parent.name.startswith(".tiny-qwen3-gguf.staging-")
    assert convert_cmd == [
        sys.executable,
        str(llama_root / "convert_hf_to_gguf.py"),
        str(staged_model),
        "--outfile",
        str(f16),
        "--outtype",
        "f16",
    ]
    assert quantize_cmd == [
        str(llama_root / "build" / "bin" / "llama-quantize"),
        str(f16),
        str(staged_model.parent / "artifact" / "tiny-qwen3-q4-k-m.gguf"),
        "Q4_K_M",
    ]
    assert convert_options == quantize_options == GGUF_OPTIONS
    assert llama.model_files == list(TOKENIZER_FILES)
    contract = load_manifest(out)
    assert (contract.source, contract.model_family, contract.task) == (
        str(source),
        "qwen3",
        "dense",
    )
    assert (contract.artifact_formats, contract.quantization, contract.exporter_version) == (
        ("gguf",),
        "Q4_K_M",
        "llama.cpp",
    )
    assert (contract.pooling, contract.normalization, contract.output_dim) == (
        PoolingType.LAST_TOKEN,
        True,
        HIDDEN,
    )
    for name in TOKENIZER_FILES:
        assert (out / name).read_bytes() == (source / name).read_bytes()


def test_convert_gguf_downloads_a_hub_model_into_an_existing_empty_output(tmp_path, monkeypatch):
    hub = install_fake_hub(monkeypatch, tmp_path / "hub")
    hub.publish("acme/Tiny-Qwen3", model_type="qwen3")
    llama_root = make_llama_checkout(tmp_path / "llama.cpp")
    monkeypatch.setenv("LLAMA_CPP_HOME", str(llama_root))
    llama = install_fake_llama(monkeypatch)
    out = tmp_path / "artifacts" / "tiny-qwen3-gguf"
    out.mkdir(parents=True)

    result = convert_gguf(
        "acme/Tiny-Qwen3", out, quant="Q8_0", pooling="mean", normalization=False
    )

    assert result == out / "tiny-qwen3-q8-0.gguf"
    assert result.read_bytes() == b"GGUF f16\nQ8_0"
    snapshot_dir = hub.calls[1][2]
    assert hub.calls == [
        ("hf_hub_download", "acme/Tiny-Qwen3", "config.json"),
        ("snapshot_download", "acme/Tiny-Qwen3", snapshot_dir),
    ]
    assert snapshot_dir.name == "model"
    assert llama.model_files == list(TOKENIZER_FILES)
    contract = load_manifest(out)
    assert (contract.source, contract.quantization, contract.pooling) == (
        "acme/Tiny-Qwen3",
        "Q8_0",
        PoolingType.MEAN,
    )


def test_convert_gguf_fails_when_the_model_lacks_a_tokenizer_file(tmp_path, monkeypatch):
    source = write_model_dir(
        tmp_path / "Tiny-Qwen3", model_type="qwen3", skip=("tokenizer_config.json",)
    )
    llama_root = make_llama_checkout(tmp_path / "llama.cpp")
    llama = install_fake_llama(monkeypatch)
    out = tmp_path / "artifacts" / "tiny-qwen3-gguf"

    with pytest.raises(
        FileNotFoundError,
        match=r"tokenizer file required by manifest is missing: tokenizer_config\.json",
    ):
        convert_gguf(
            str(source), out, llama_cpp=str(llama_root), pooling="mean", normalization=True
        )

    assert len(llama.calls) == 2
    assert list(out.parent.iterdir()) == []


def test_convert_gguf_stops_when_the_f16_conversion_fails(tmp_path, monkeypatch):
    source = write_model_dir(tmp_path / "Tiny-Qwen3", model_type="qwen3")
    llama_root = make_llama_checkout(tmp_path / "llama.cpp")
    llama = install_fake_llama(monkeypatch, failing_step="convert")
    out = tmp_path / "artifacts" / "tiny-qwen3-gguf"

    with pytest.raises(RuntimeError, match=r"^convert_hf_to_gguf\.py exited with code 1$"):
        convert_gguf(
            str(source), out, llama_cpp=str(llama_root), pooling="mean", normalization=True
        )

    assert len(llama.calls) == 1
    assert list(out.parent.iterdir()) == []


# ---------------------------------------------------------------------------
# CLI, local backend
# ---------------------------------------------------------------------------
def test_cli_onnx_prints_each_variant_size_in_megabytes(tmp_path, capsys, stack):
    source = write_model_dir(tmp_path / "tiny-bert", model_type="bert")
    out = tmp_path / "out"

    exit_code = main(
        ["onnx", str(source), "--out", str(out), "--pooling", "cls", "--no-normalize"]
    )

    assert exit_code == 0
    int8_mb = (out / "onnx" / "model_quantized.onnx").stat().st_size / 1024**2
    q4f16_mb = (out / "onnx" / "model_q4f16.onnx").stat().st_size / 1024**2
    assert capsys.readouterr().out == f"int8: {int8_mb:.1f} MB\nq4f16: {q4f16_mb:.1f} MB\n"
    assert manifest_fields(out, "pooling", "normalization") == {
        "pooling": "CLS",
        "normalization": False,
    }


def test_cli_gguf_prints_the_quantized_file_path(tmp_path, monkeypatch, capsys):
    source = write_model_dir(tmp_path / "Tiny-Qwen3", model_type="qwen3")
    llama_root = make_llama_checkout(tmp_path / "llama.cpp")
    llama = install_fake_llama(monkeypatch)
    out = tmp_path / "gguf-out"

    exit_code = main(
        [
            "gguf",
            str(source),
            "--out",
            str(out),
            "--quant",
            "Q5_K_M",
            "--llama-cpp",
            str(llama_root),
            "--pooling",
            "last_token",
            "--normalize",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == f"{out / 'tiny-qwen3-q5-k-m.gguf'}\n"
    assert llama.calls[-1][0][-1] == "Q5_K_M"
    assert load_manifest(out).quantization == "Q5_K_M"


# ---------------------------------------------------------------------------
# Modal backend
# ---------------------------------------------------------------------------
def test_cli_onnx_modal_backend_mounts_the_model_and_downloads_the_artifact(
    tmp_path, monkeypatch, capsys, stack
):
    source = write_model_dir(tmp_path / "tiny-bert", model_type="bert")
    modal = FakeModal(tmp_path / "container").install(monkeypatch)
    out = tmp_path / "downloaded"

    exit_code = main(
        [
            "onnx",
            str(source),
            "--out",
            str(out),
            "--pooling",
            "mean",
            "--normalize",
            "--backend",
            "modal",
        ]
    )

    assert exit_code == 0
    printed = ast.literal_eval(capsys.readouterr().out)
    files = [
        "config.json",
        MANIFEST_FILENAME,
        "onnx/model_q4f16.onnx",
        "onnx/model_quantized.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    remote_artifact = tmp_path / "container" / "mnt" / "output" / "artifact"
    assert printed == {
        "backend": "modal",
        "command": "onnx",
        "out_dir": str(out),
        "files": files,
        "result": {
            "int8": pytest.approx(
                (remote_artifact / "onnx" / "model_quantized.onnx").stat().st_size / 1024**2
            ),
            "q4f16": pytest.approx(
                (remote_artifact / "onnx" / "model_q4f16.onnx").stat().st_size / 1024**2
            ),
        },
    }
    assert relative_files(out) == files
    for relative in files:
        assert (out / relative).read_bytes() == (remote_artifact / relative).read_bytes()

    (app,) = modal.apps
    (function,) = app.functions
    (volume,) = modal.volumes
    assert app.name == "fastretrieval-convert"
    assert {key: function.options[key] for key in ("memory", "cpu", "timeout")} == {
        "memory": 32768,
        "cpu": 4.0,
        "timeout": 3600,
    }
    assert function.options["volumes"] == {"/mnt/output": volume}
    assert function.options["image"].steps == (
        ("debian_slim", "3.12"),
        ("uv_pip_install", *modal_backend._IMAGE_PACKAGES),
        ("add_local_python_source", "fastretrieval"),
        ("add_local_dir", str(source.resolve()), "/mnt/input-model", False),
    )
    assert function.calls == [
        (
            "onnx",
            {
                "source": "/mnt/input-model",
                "out_dir": "/mnt/output/artifact",
                "task": "feature-extraction",
                "modality": "text",
                "variants": ["int8", "q4f16"],
                "pooling": "mean",
                "normalization": True,
                "output_dim": None,
                "yes_no": None,
            },
        )
    ]
    assert (volume.commits, volume.reloads) == (1, 1)
    assert not app.running


def test_cli_gguf_modal_backend_mounts_llama_cpp_and_maps_the_result_path(
    tmp_path, monkeypatch, capsys
):
    source = write_model_dir(tmp_path / "Tiny-Qwen3", model_type="qwen3")
    llama_root = make_llama_checkout(tmp_path / "llama.cpp")
    llama = install_fake_llama(monkeypatch)
    modal = FakeModal(tmp_path / "container").install(monkeypatch)
    out = tmp_path / "downloaded"

    exit_code = main(
        [
            "gguf",
            str(source),
            "--out",
            str(out),
            "--llama-cpp",
            str(llama_root),
            "--pooling",
            "last_token",
            "--normalize",
            "--backend",
            "modal",
        ]
    )

    assert exit_code == 0
    printed = ast.literal_eval(capsys.readouterr().out)
    (gguf_path,) = out.glob("*.gguf")
    assert gguf_path.name.endswith("-q4-k-m.gguf")
    assert printed == {
        "backend": "modal",
        "command": "gguf",
        "out_dir": str(out),
        "files": sorted([*TOKENIZER_FILES, MANIFEST_FILENAME, gguf_path.name]),
        "result": {"path": str(gguf_path)},
    }
    assert gguf_path.read_bytes() == b"GGUF f16\nQ4_K_M"

    (function,) = modal.apps[0].functions
    assert function.options["image"].steps[-2:] == (
        ("add_local_dir", str(source.resolve()), "/mnt/input-model", False),
        ("add_local_dir", str(llama_root.resolve()), "/mnt/llama-cpp", False),
    )
    ((command, payload),) = function.calls
    assert command == "gguf"
    assert {key: payload[key] for key in ("source", "llama_cpp", "out_dir", "quant")} == {
        "source": "/mnt/input-model",
        "llama_cpp": "/mnt/llama-cpp",
        "out_dir": "/mnt/output/artifact",
        "quant": "Q4_K_M",
    }
    container_llama = tmp_path / "container" / "mnt" / "llama-cpp"
    assert Path(llama.calls[0][0][1]) == container_llama / "convert_hf_to_gguf.py"
    assert Path(llama.calls[1][0][0]) == container_llama / "build" / "bin" / "llama-quantize"
    assert llama.model_files == list(TOKENIZER_FILES)
    assert modal.volumes[0].commits == 1


def convert_hub_model_remotely(tmp_path, monkeypatch):
    hub = install_fake_hub(monkeypatch, tmp_path / "hub")
    hub.publish("acme/tiny-bert", model_type="bert")
    modal = FakeModal(tmp_path / "container").install(monkeypatch)
    out = tmp_path / "downloaded"
    result = run_remote(
        "onnx",
        source="acme/tiny-bert",
        out=str(out),
        variants=["q4f16"],
        pooling="mean",
        normalization=True,
    )
    return modal, out, result


def test_run_remote_accepts_the_out_alias_and_does_not_mount_a_hub_source(
    tmp_path, monkeypatch, stack
):
    modal, out, result = convert_hub_model_remotely(tmp_path, monkeypatch)

    files = [
        "config.json",
        MANIFEST_FILENAME,
        "onnx/model_q4f16.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert result["out_dir"] == str(out)
    assert result["files"] == files
    assert list(result["result"]) == ["q4f16"]
    assert relative_files(out) == files
    (function,) = modal.apps[0].functions
    assert function.calls == [
        (
            "onnx",
            {
                "source": "acme/tiny-bert",
                "out_dir": "/mnt/output/artifact",
                "variants": ["q4f16"],
                "pooling": "mean",
                "normalization": True,
            },
        )
    ]
    assert [step[0] for step in function.options["image"].steps] == [
        "debian_slim",
        "uv_pip_install",
        "add_local_python_source",
    ]
    assert stack.ort.quantize_dynamic == []
    assert load_manifest(out).source == "acme/tiny-bert"


def test_remote_function_rejects_an_unknown_command(tmp_path, monkeypatch, stack):
    modal, _, _ = convert_hub_model_remotely(tmp_path, monkeypatch)
    (function,) = modal.apps[0].functions

    with pytest.raises(ValueError, match="unsupported remote command 'bogus'"):
        asyncio.run(function.raw("bogus", {}))


def test_run_remote_refuses_to_block_inside_a_running_event_loop(tmp_path, monkeypatch):
    modal = FakeModal(tmp_path / "container").install(monkeypatch)

    async def call_from_a_coroutine():
        return run_remote("onnx", source="acme/tiny-bert", out_dir=str(tmp_path / "out"))

    with pytest.raises(RuntimeError, match="cannot run inside an active event loop"):
        asyncio.run(call_from_a_coroutine())

    assert modal.apps == []
    assert modal.volumes == []
