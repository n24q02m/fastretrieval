#!/usr/bin/env python3
"""Reproducible benchmark comparing fastretrieval against fastembed on CPU ONNX batches.

Run locally from the repository root:
    uv run python benchmarks/run_vs_fastembed.py

Each runtime executes in an isolated worker subprocess so OS peak RSS memory,
model initialization latency, inference latency/throughput, and output embeddings
are measured without cross-runtime memory or dependency contamination.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

DEFAULT_OUTPUT_PATH = Path("benchmarks/fastembed_comparison.json")

ENGLISH_CORPUS: list[str] = [
    "Fast multi-model retrieval runtime built on ONNX Runtime without PyTorch overhead.",
    "Semantic search maps natural language queries and passages into dense vector spaces.",
    "Mean pooling averages token embeddings across non-padding positions before normalization.",
    "Last-token pooling extracts the final causal hidden state for decoder embedding models.",
    "CLS pooling reads the leading classification token representation from encoder models.",
    "Matryoshka representation learning allows prefix truncation without retraining weights.",
    "Dynamic INT8 quantization reduces ONNX model footprint while preserving cosine ranking.",
    "Cross-encoder rerankers score query and document pairs jointly for top-k precision.",
    "Batch invariance guarantees a pair score never shifts with neighboring batch items.",
    "Declarative model contracts record pooling, normalization, and output tensor shape.",
    "Tokenizers convert raw UTF-8 strings into subword token IDs and attention masks.",
    "L2 normalization projects vectors onto the unit hypersphere for dot-product search.",
    "CPU thread pinning avoids oversubscription when multiple workers serve concurrent load.",
    "Model caching reuses downloaded Hugging Face artifacts across process restarts.",
    "Vector indexes accelerate approximate nearest neighbor lookup over millions of chunks.",
    "Hybrid retrieval combines lexical BM25 or SPLADE signals with dense semantic similarity.",
    "Late-interaction models retain per-token embeddings for fine-grained MaxSim scoring.",
    "Query instruction prefixes align asymmetric search queries with passage representations.",
    "Fixed-shape ONNX exports require batch clamping when inputs exceed static bounds.",
    "Memory pattern disabling prevents ONNX Runtime from hoarding peak sequence buffers.",
    "Lightweight runtime dependencies simplify deployment inside minimal container images.",
    "Code review graphs index symbol definitions and call edges for fast impact analysis.",
    "Persistent agent memory retrieves relevant project facts across separate work sessions.",
    "Web documentation search chunks Markdown headings before generating local embeddings.",
    "Deterministic preprocessing ensures identical token sequences across Python versions.",
    "Parity verification compares quantized ONNX embeddings against full-precision baselines.",
    "Streaming iterables yield fixed-size batches to bound peak heap usage on large corpora.",
    "Explicit model registries avoid implicit fallback defaults across retrieval modalities.",
    "FastAPI HTTP servers share one warm inference session across multiple local clients.",
    "Multiprocessing worker pools parallelize document encoding across physical CPU cores.",
    "DirectML and CUDA execution providers accelerate ONNX graphs when GPUs are present.",
    "Subword vocabulary compression balances tokenization speed and multilingual coverage.",
    "Cosine similarity between normalized vectors equals their inner product directly.",
    "Warmup inference initializes runtime arenas before collecting steady-state latency.",
    "Percentile latency tracks tail behavior better than arithmetic mean alone under load.",
    "Peak working set size captures native ONNX Runtime allocations invisible to tracemalloc.",
    "Python heap profiling isolates wrapper and tokenizer overhead from C++ tensor buffers.",
    "Short queries behave differently from multi-sentence technical paragraphs during padding.",
    "Dynamic padding aligns each batch to its longest sequence instead of global max length.",
    "Static maximum input length guards protect servers against pathological payload sizes.",
    "Environment variable overrides redirect cache storage to fast NVMe scratch volumes.",
    "Offline mode restricts model loading strictly to verified local cache directories.",
    "Custom model specifications let teams benchmark external Hugging Face ONNX weights.",
    "Model card generation records verified parity tolerances alongside artifact manifests.",
    "Structured logging surfaces active execution providers without noisy debug dumps.",
    "Unit test suites stub inference sessions so CI stays fast across three desktop OSes.",
    "Integration test suites validate real Hugging Face model weights end to end.",
    "Reproducible benchmarks pin thread count, batch size, warmup runs, and corpus order.",
    "Quantized Q4F16 weights trade slight numeric drift for lower memory bandwidth use.",
    "GGUF backends support llama.cpp inference for environments without ONNX wheels.",
    "Sparse SPLADE outputs map vocabulary IDs to positive importance weights for pruning.",
    "ColBERT token vectors preserve exact phrase matches that single-vector pooling blurs.",
    "Multimodal image encoders project visual patches into the same space as text prompts.",
    "ColPali document retrieval indexes page screenshots without fragile OCR pipelines.",
    "Trusted publishing automates wheel and sdist releases directly from GitHub Actions.",
    "Lockfile maintenance keeps transitive dependencies reproducible across developer hosts.",
    "Strict type checking catches optional dependency boundaries before runtime failures.",
    "Minimal core dependencies prevent version conflicts inside downstream MCP servers.",
    "Throughput scales with batch size until L3 cache contention saturates memory bandwidth.",
    "Latency-sensitive interactive search favors small batches and low session load overhead.",
    "Throughput-oriented indexing favors larger batches across warm persistent workers.",
    "Isolated subprocess workers give uncontaminated RSS measurements for each library.",
    "Numeric parity checks confirm embeddings stay aligned across alternative runtimes.",
    "Transparent benchmark artifacts document exact hardware, versions, and raw timings.",
]

MULTILINGUAL_CORPUS: list[str] = [
    "Retrieval runtime benchmarks measure latency, throughput, memory, and numeric parity.",
    "He thong tim kiem ngu nghia su dung vector da chieu de xep hang tai lieu chinh xac.",
    "La recherche semantique associe les requetes aux passages pertinents par similarite.",
    "Semantische Suche vergleicht dichte Vektoren ueber Kosinus-Aehnlichkeit in Echtzeit.",
    "La recuperacion multimodal combina texto e imagenes en un espacio vectorial comun.",
    "Quantized ONNX models reduce memory footprint without sacrificing retrieval accuracy.",
    "Mo hinh da ngon ngu ho tro truy van cheo giua tieng Viet, tieng Anh va tieng Phap.",
    "Les modeles legers fonctionnent efficacement sur processeur sans carte graphique.",
    "Schnelle Tokenisierung verbessert den Durchsatz bei grossen Dokumentenstapeln.",
    "Los indices vectoriales aceleran la busqueda aproximada de vecinos mas cercanos.",
    "On-device embeddings keep sensitive workspace codebases private during local search.",
    "Kiem thu hoi quy dam bao do tuong dong cosin on dinh sau khi luong hoa INT8.",
    "La normalisation L2 garantit que le produit scalaire equivaut a la similarite cosinus.",
    "Batch-Verarbeitung optimiert die Auslastung moderner Mehrkern-Prozessoren.",
    "El reordenamiento cruzado refina los mejores resultados tras la busqueda inicial.",
    "Deterministic batching avoids subtle score drift across different runtime wrappers.",
    "Bo nho dem cuc bo giam do tre khoi dong lai cho dich vu tim kiem van ban.",
    "Le decoupage par sous-mots preserve les racines morphologiques des mots composes.",
    "Praezise Metriken trennen Modellladezeit von reiner Inferenzzeit im Dauerbetrieb.",
    "Las pruebas reproducibles registran versiones exactas de bibliotecas y hardware.",
    "Compact 384-dimensional embeddings fit comfortably inside in-memory vector stores.",
    "Do tre phan vi thu 95 phan anh su on dinh cua he thong khi xu ly nhieu yeu cau.",
    "Une empreinte memoire reduite facilite le deploiement dans des conteneurs legers.",
    "Reproduzierbare Benchmarks vergleichen zwei Laufzeiten unter identischen Bedingungen.",
    "La cuantizacion dinamica acelera la multiplicacion de matrices en CPU modernas.",
    "Zero-copy tensor passing minimizes Python overhead between tokenizer and runtime.",
    "Thong luong do bang so tai lieu duoc ma hoa moi giay tren cung so luong CPU.",
    "L'isolation par sous-processus evite toute fuite memoire entre bibliotheques testees.",
    "Klare Spezifikationen definieren Pooling-Strategie und Normalisierung pro Modell.",
    "El almacenamiento en cache local evita descargas repetidas durante evaluaciones.",
    "Cross-lingual retrieval matches queries in one language against docs in another.",
    "Bao cao ket qua may doc duoc giup theo doi xu huong hieu nang qua tung phien ban.",
]


@dataclass(frozen=True)
class RuntimeTarget:
    """Model specification for one runtime within a benchmark scenario."""

    model_id: str
    custom_hf: str | None = None
    custom_model_file: str | None = None
    custom_dim: int | None = None
    custom_pooling: str | None = None
    custom_normalization: bool = True
    artifact_size_mb: float | None = None


@dataclass(frozen=True)
class BenchmarkScenario:
    """Paired fastretrieval vs fastembed comparison scenario."""

    scenario_id: str
    title: str
    comparison_kind: str
    corpus_name: str
    fastretrieval: RuntimeTarget
    fastembed: RuntimeTarget


SCENARIOS: list[BenchmarkScenario] = [
    BenchmarkScenario(
        scenario_id="minilm_l6_v2_same_onnx",
        title="all-MiniLM-L6-v2 (identical FP32 ONNX graph)",
        comparison_kind="same_weights",
        corpus_name="english",
        fastretrieval=RuntimeTarget(
            model_id="Qdrant/all-MiniLM-L6-v2-onnx",
            custom_hf="Qdrant/all-MiniLM-L6-v2-onnx",
            custom_model_file="model.onnx",
            custom_dim=384,
            custom_pooling="MEAN",
            custom_normalization=True,
            artifact_size_mb=90.0,
        ),
        fastembed=RuntimeTarget(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            artifact_size_mb=90.0,
        ),
    ),
    BenchmarkScenario(
        scenario_id="minilm_l6_v2_builtin_int8",
        title="all-MiniLM-L6-v2 (fastretrieval built-in INT8 vs fastembed default FP32)",
        comparison_kind="builtin_vs_builtin",
        corpus_name="english",
        fastretrieval=RuntimeTarget(
            model_id="n24q02m/fr-all-MiniLM-L6-v2",
            artifact_size_mb=23.0,
        ),
        fastembed=RuntimeTarget(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            artifact_size_mb=90.0,
        ),
    ),
    BenchmarkScenario(
        scenario_id="bge_small_en_v15_same_onnx",
        title="BAAI/bge-small-en-v1.5 (identical quantized ONNX graph)",
        comparison_kind="same_weights",
        corpus_name="english",
        fastretrieval=RuntimeTarget(
            model_id="Qdrant/bge-small-en-v1.5-onnx-Q",
            custom_hf="Qdrant/bge-small-en-v1.5-onnx-Q",
            custom_model_file="model_optimized.onnx",
            custom_dim=384,
            custom_pooling="CLS",
            custom_normalization=True,
            artifact_size_mb=67.0,
        ),
        fastembed=RuntimeTarget(
            model_id="BAAI/bge-small-en-v1.5",
            artifact_size_mb=67.0,
        ),
    ),
    BenchmarkScenario(
        scenario_id="multilingual_minilm_l12_v2",
        title="paraphrase-multilingual-MiniLM-L12-v2 (built-in INT8 vs fastembed quantized)",
        comparison_kind="builtin_vs_builtin",
        corpus_name="multilingual",
        fastretrieval=RuntimeTarget(
            model_id="n24q02m/fr-paraphrase-multilingual-MiniLM-L12-v2",
            artifact_size_mb=118.0,
        ),
        fastembed=RuntimeTarget(
            model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            artifact_size_mb=220.0,
        ),
    ),
]


def get_corpus(corpus_name: str) -> list[str]:
    """Return the deterministic text corpus for a named benchmark profile."""
    if corpus_name == "english":
        return list(ENGLISH_CORPUS)
    if corpus_name == "multilingual":
        return list(MULTILINGUAL_CORPUS)
    raise ValueError(f"Unknown corpus name: {corpus_name!r}")


def get_peak_rss_mb() -> float:
    """Return current process peak RSS / working set size in MiB."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        win_dll_cls = getattr(ctypes, "WinDLL", None)
        if win_dll_cls is None:
            return 0.0
        kernel32 = win_dll_cls("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        k32_get_pmc = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if k32_get_pmc is None:
            psapi = win_dll_cls("psapi", use_last_error=True)
            k32_get_pmc = psapi.GetProcessMemoryInfo
        k32_get_pmc.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        k32_get_pmc.restype = wintypes.BOOL

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = kernel32.GetCurrentProcess()
        if k32_get_pmc(handle, ctypes.byref(counters), counters.cb):
            return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
        return 0.0

    resource_mod = importlib.import_module("resource")
    ru_maxrss = float(resource_mod.getrusage(resource_mod.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return ru_maxrss / (1024.0 * 1024.0)
    return ru_maxrss / 1024.0


def summarize_latencies(
    latencies_s: list[float],
    num_docs: int,
) -> dict[str, float]:
    """Summarize per-pass wall durations (seconds) into latency and throughput stats."""
    if not latencies_s:
        raise ValueError("latencies_s must be non-empty")
    if num_docs <= 0:
        raise ValueError("num_docs must be positive")

    ms = [val * 1000.0 for val in latencies_s]
    ms_sorted = sorted(ms)
    total_s = sum(latencies_s)
    mean_ms = statistics.fmean(ms)
    median_ms = statistics.median(ms)
    p95_index = min(len(ms_sorted) - 1, max(0, int(np.ceil(0.95 * len(ms_sorted))) - 1))
    p95_ms = ms_sorted[p95_index]
    throughput = (num_docs * len(latencies_s)) / total_s if total_s > 0.0 else 0.0
    median_s = median_ms / 1000.0
    throughput_median = num_docs / median_s if median_s > 0.0 else 0.0

    return {
        "latency_ms_mean": round(mean_ms, 3),
        "latency_ms_median": round(median_ms, 3),
        "latency_ms_p95": round(p95_ms, 3),
        "latency_ms_min": round(ms_sorted[0], 3),
        "latency_ms_max": round(ms_sorted[-1], 3),
        "per_doc_ms_mean": round(mean_ms / num_docs, 4),
        "total_timed_s": round(total_s, 4),
        "throughput_docs_per_s": round(throughput, 2),
        "throughput_median_docs_per_s": round(throughput_median, 2),
    }


def compute_parity_metrics(
    vectors_a: NDArray[np.float32],
    vectors_b: NDArray[np.float32],
) -> dict[str, float]:
    """Compute vector parity and unit-norm deviation between two embedding matrices."""
    if vectors_a.shape != vectors_b.shape:
        raise ValueError(f"Shape mismatch: {vectors_a.shape} != {vectors_b.shape}")
    if vectors_a.ndim != 2 or vectors_a.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2-D matrices, got {vectors_a.shape}")

    a = vectors_a.astype(np.float64)
    b = vectors_b.astype(np.float64)
    norms_a = np.linalg.norm(a, axis=1)
    norms_b = np.linalg.norm(b, axis=1)
    denom = np.maximum(norms_a * norms_b, 1e-12)
    cosines = np.sum(a * b, axis=1) / denom
    abs_diff = np.abs(a - b)

    return {
        "cosine_similarity_mean": round(float(np.mean(cosines)), 6),
        "cosine_similarity_min": round(float(np.min(cosines)), 6),
        "max_abs_diff": round(float(np.max(abs_diff)), 6),
        "mean_abs_diff": round(float(np.mean(abs_diff)), 6),
        "fastretrieval_l2_norm_max_dev": round(float(np.max(np.abs(norms_a - 1.0))), 6),
        "fastembed_l2_norm_max_dev": round(float(np.max(np.abs(norms_b - 1.0))), 6),
    }


def _load_runtime_model(
    runtime: str,
    target: RuntimeTarget,
    threads: int,
) -> tuple[Any, str, str]:
    """Instantiate a TextEmbedding model for either fastretrieval or fastembed."""
    ort = importlib.import_module("onnxruntime")
    ort_version = str(getattr(ort, "__version__", "unknown"))

    if runtime == "fastretrieval":
        fr = importlib.import_module("fastretrieval")
        if target.custom_hf is not None:
            if target.custom_dim is None or target.custom_pooling is None:
                raise ValueError(
                    "Custom fastretrieval model requires custom_dim and custom_pooling"
                )
            fr.CustomModelSpec(
                model_id=target.model_id,
                hf=target.custom_hf,
                model_file=target.custom_model_file or "onnx/model.onnx",
                dim=target.custom_dim,
                pooling=target.custom_pooling,
                normalization=target.custom_normalization,
            ).register()
        model = fr.TextEmbedding(
            model_name=target.model_id,
            threads=threads,
            providers=["CPUExecutionProvider"],
        )
        return model, str(getattr(fr, "__version__", "unknown")), ort_version

    if runtime == "fastembed":
        fe = importlib.import_module("fastembed")
        model = fe.TextEmbedding(
            model_name=target.model_id,
            threads=threads,
            providers=["CPUExecutionProvider"],
        )
        return model, str(getattr(fe, "__version__", "unknown")), ort_version

    raise ValueError(f"Unsupported runtime: {runtime!r}")


def run_worker(
    runtime: str,
    target: RuntimeTarget,
    corpus: list[str],
    batch_size: int,
    warmup: int,
    iterations: int,
    threads: int,
    result_path: Path,
    vectors_path: Path | None,
    warm_cache_only: bool = False,
) -> dict[str, Any]:
    """Execute one isolated runtime benchmark (or cache pre-warm) and write JSON output."""
    init_start = time.perf_counter()
    model, runtime_version, ort_version = _load_runtime_model(
        runtime=runtime,
        target=target,
        threads=threads,
    )
    init_load_s = time.perf_counter() - init_start

    if warm_cache_only:
        payload: dict[str, Any] = {
            "runtime": runtime,
            "runtime_version": runtime_version,
            "onnxruntime_version": ort_version,
            "model_id": target.model_id,
            "cache_warmed": True,
            "cold_or_warm_init_s": round(init_load_s, 4),
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    warmup_start = time.perf_counter()
    last_vectors: list[NDArray[Any]] = []
    for _ in range(max(1, warmup)):
        last_vectors = list(model.embed(corpus, batch_size=batch_size))
    warmup_s = time.perf_counter() - warmup_start

    latencies_s: list[float] = []
    tracemalloc.start()
    try:
        for _ in range(iterations):
            t0 = time.perf_counter()
            last_vectors = list(model.embed(corpus, batch_size=batch_size))
            latencies_s.append(time.perf_counter() - t0)
        _, peak_traced_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    matrix = np.asarray(last_vectors, dtype=np.float32)
    if vectors_path is not None:
        np.save(vectors_path, matrix)

    timing_summary = summarize_latencies(latencies_s, num_docs=len(corpus))
    payload = {
        "runtime": runtime,
        "runtime_version": runtime_version,
        "onnxruntime_version": ort_version,
        "model_id": target.model_id,
        "artifact_size_mb": target.artifact_size_mb,
        "init_load_s": round(init_load_s, 4),
        "warmup_s": round(warmup_s, 4),
        "iterations": iterations,
        "num_docs": len(corpus),
        "batch_size": batch_size,
        "threads": threads,
        "peak_tracemalloc_mb": round(peak_traced_bytes / (1024.0 * 1024.0), 3),
        "peak_rss_mb": round(get_peak_rss_mb(), 3),
        "embedding_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        **timing_summary,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def resolve_worker_command(
    runtime: str,
    script_path: Path,
    fastembed_spec: str = "fastembed",
) -> list[str]:
    """Build the subprocess command prefix for an isolated runtime worker."""
    if runtime == "fastretrieval":
        return [sys.executable, str(script_path)]

    if runtime == "fastembed":
        if importlib.util.find_spec("fastembed") is not None:
            return [sys.executable, str(script_path)]
        uv_bin = shutil.which("uv")
        if uv_bin is None:
            raise RuntimeError(
                "fastembed is not installed in the current environment and 'uv' was not found on PATH."
            )
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        return [
            uv_bin,
            "run",
            "--no-project",
            "--python",
            py_ver,
            "--with",
            fastembed_spec,
            "python",
            str(script_path),
        ]

    raise ValueError(f"Unsupported runtime: {runtime!r}")


def _invoke_worker(
    cmd_prefix: list[str],
    runtime: str,
    target: RuntimeTarget,
    corpus_name: str,
    batch_size: int,
    warmup: int,
    iterations: int,
    threads: int,
    result_path: Path,
    vectors_path: Path | None,
    warm_cache_only: bool = False,
) -> dict[str, Any]:
    """Run one worker subprocess and load its JSON output."""
    cmd = [
        *cmd_prefix,
        "--worker",
        runtime,
        "--target-json",
        json.dumps(asdict(target)),
        "--corpus",
        corpus_name,
        "--batch-size",
        str(batch_size),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--threads",
        str(threads),
        "--result-json",
        str(result_path),
    ]
    if vectors_path is not None:
        cmd.extend(["--vectors-npy", str(vectors_path)])
    if warm_cache_only:
        cmd.append("--warm-cache-only")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed for runtime={runtime} model={target.model_id} "
            f"(exit={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


def build_comparison_summary(
    fr_metrics: dict[str, Any],
    fe_metrics: dict[str, Any],
    parity: dict[str, float],
) -> dict[str, Any]:
    """Compute relative speedup, latency ratio, and memory ratio metrics."""
    fe_throughput = float(fe_metrics["throughput_docs_per_s"])
    fr_throughput = float(fr_metrics["throughput_docs_per_s"])
    fe_latency = float(fe_metrics["latency_ms_mean"])
    fr_latency = float(fr_metrics["latency_ms_mean"])
    fe_rss = float(fe_metrics["peak_rss_mb"])
    fr_rss = float(fr_metrics["peak_rss_mb"])
    fe_init = float(fe_metrics["init_load_s"])
    fr_init = float(fr_metrics["init_load_s"])

    return {
        "throughput_speedup_vs_fastembed": (
            round(fr_throughput / fe_throughput, 3) if fe_throughput > 0 else 0.0
        ),
        "latency_ratio_vs_fastembed": (
            round(fr_latency / fe_latency, 3) if fe_latency > 0 else 0.0
        ),
        "peak_rss_ratio_vs_fastembed": round(fr_rss / fe_rss, 3) if fe_rss > 0 else 0.0,
        "init_load_ratio_vs_fastembed": round(fr_init / fe_init, 3) if fe_init > 0 else 0.0,
        **parity,
    }


def run_benchmark_suite(
    scenarios: list[BenchmarkScenario],
    batch_size: int = 32,
    warmup: int = 1,
    iterations: int = 5,
    threads: int | None = None,
    fastembed_spec: str = "fastembed",
    script_path: Path | None = None,
) -> dict[str, Any]:
    """Run all configured scenarios across isolated workers and assemble the report."""
    effective_threads = threads if threads is not None else min(4, os.cpu_count() or 1)
    resolved_script = (script_path or Path(__file__)).resolve()
    fr_prefix = resolve_worker_command("fastretrieval", resolved_script, fastembed_spec)
    fe_prefix = resolve_worker_command("fastembed", resolved_script, fastembed_spec)

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fastretrieval_bench_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for idx, scenario in enumerate(scenarios, start=1):
            print(
                f"[{idx}/{len(scenarios)}] Benchmarking {scenario.scenario_id}...",
                file=sys.stderr,
                flush=True,
            )
            fr_warm = tmp_dir / f"{scenario.scenario_id}.fr.warm.json"
            fe_warm = tmp_dir / f"{scenario.scenario_id}.fe.warm.json"
            _invoke_worker(
                cmd_prefix=fr_prefix,
                runtime="fastretrieval",
                target=scenario.fastretrieval,
                corpus_name=scenario.corpus_name,
                batch_size=batch_size,
                warmup=warmup,
                iterations=iterations,
                threads=effective_threads,
                result_path=fr_warm,
                vectors_path=None,
                warm_cache_only=True,
            )
            _invoke_worker(
                cmd_prefix=fe_prefix,
                runtime="fastembed",
                target=scenario.fastembed,
                corpus_name=scenario.corpus_name,
                batch_size=batch_size,
                warmup=warmup,
                iterations=iterations,
                threads=effective_threads,
                result_path=fe_warm,
                vectors_path=None,
                warm_cache_only=True,
            )

            fr_json = tmp_dir / f"{scenario.scenario_id}.fr.json"
            fe_json = tmp_dir / f"{scenario.scenario_id}.fe.json"
            fr_npy = tmp_dir / f"{scenario.scenario_id}.fr.npy"
            fe_npy = tmp_dir / f"{scenario.scenario_id}.fe.npy"

            fr_metrics = _invoke_worker(
                cmd_prefix=fr_prefix,
                runtime="fastretrieval",
                target=scenario.fastretrieval,
                corpus_name=scenario.corpus_name,
                batch_size=batch_size,
                warmup=warmup,
                iterations=iterations,
                threads=effective_threads,
                result_path=fr_json,
                vectors_path=fr_npy,
            )
            fe_metrics = _invoke_worker(
                cmd_prefix=fe_prefix,
                runtime="fastembed",
                target=scenario.fastembed,
                corpus_name=scenario.corpus_name,
                batch_size=batch_size,
                warmup=warmup,
                iterations=iterations,
                threads=effective_threads,
                result_path=fe_json,
                vectors_path=fe_npy,
            )

            fr_vectors = np.load(fr_npy)
            fe_vectors = np.load(fe_npy)
            parity = compute_parity_metrics(fr_vectors, fe_vectors)
            comparison = build_comparison_summary(fr_metrics, fe_metrics, parity)

            results.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "title": scenario.title,
                    "comparison_kind": scenario.comparison_kind,
                    "corpus_name": scenario.corpus_name,
                    "fastretrieval": fr_metrics,
                    "fastembed": fe_metrics,
                    "comparison": comparison,
                }
            )

    fr_versions = {item["fastretrieval"]["runtime_version"] for item in results}
    fe_versions = {item["fastembed"]["runtime_version"] for item in results}
    ort_versions = {item["fastretrieval"]["onnxruntime_version"] for item in results} | {
        item["fastembed"]["onnxruntime_version"] for item in results
    }

    return {
        "metadata": {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "command": "uv run python benchmarks/run_vs_fastembed.py",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "threads": effective_threads,
            "batch_size": batch_size,
            "warmup_passes": warmup,
            "timed_iterations": iterations,
            "provider": "CPUExecutionProvider",
            "cache_warmed_before_timing": True,
            "fastretrieval_version": sorted(fr_versions)[0] if fr_versions else "unknown",
            "fastembed_version": sorted(fe_versions)[0] if fe_versions else "unknown",
            "onnxruntime_versions": sorted(ort_versions),
        },
        "scenarios": results,
    }


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Format a compact plain-ASCII summary table from a benchmark report."""
    lines = [
        "| Scenario | fastretrieval (docs/s) | fastembed (docs/s) | Speedup | fr RSS (MiB) | fe RSS (MiB) | Mean Cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["scenarios"]:
        fr = item["fastretrieval"]
        fe = item["fastembed"]
        cmp = item["comparison"]
        lines.append(
            f"| {item['scenario_id']} "
            f"| {fr['throughput_docs_per_s']:.2f} "
            f"| {fe['throughput_docs_per_s']:.2f} "
            f"| {cmp['throughput_speedup_vs_fastembed']:.2f}x "
            f"| {fr['peak_rss_mb']:.1f} "
            f"| {fe['peak_rss_mb']:.1f} "
            f"| {cmp['cosine_similarity_mean']:.6f} |"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for both coordinator and worker modes."""
    parser = argparse.ArgumentParser(
        description="Benchmark fastretrieval against fastembed on real CPU ONNX batches."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[s.scenario_id for s in SCENARIOS],
        default=None,
        help="Subset of scenario IDs to execute (default: all scenarios).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup passes before timing.")
    parser.add_argument("--iterations", type=int, default=5, help="Timed passes per scenario.")
    parser.add_argument(
        "--threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="ONNX Runtime intra-op thread count.",
    )
    parser.add_argument(
        "--fastembed-spec",
        default="fastembed",
        help="Package spec passed to uv --with when fastembed is not installed in current env.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination path for machine-readable JSON benchmark output.",
    )

    # Internal worker mode flags (used by run_benchmark_suite subprocesses).
    parser.add_argument("--worker", choices=["fastretrieval", "fastembed"], default=None)
    parser.add_argument("--target-json", default=None)
    parser.add_argument("--corpus", default="english")
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--vectors-npy", type=Path, default=None)
    parser.add_argument("--warm-cache-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    if args.worker is not None:
        if args.target_json is None or args.result_json is None:
            raise ValueError("--worker requires --target-json and --result-json")
        target = RuntimeTarget(**json.loads(args.target_json))
        corpus = get_corpus(args.corpus)
        run_worker(
            runtime=args.worker,
            target=target,
            corpus=corpus,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            threads=args.threads,
            result_path=args.result_json,
            vectors_path=args.vectors_npy,
            warm_cache_only=args.warm_cache_only,
        )
        return 0

    selected = (
        [s for s in SCENARIOS if s.scenario_id in set(args.scenarios)]
        if args.scenarios
        else list(SCENARIOS)
    )
    report = run_benchmark_suite(
        scenarios=selected,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        threads=args.threads,
        fastembed_spec=args.fastembed_spec,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_markdown_summary(report))
    print(f"\nWrote benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
