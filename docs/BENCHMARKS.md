# Benchmarks: fastretrieval vs fastembed

This document records reproducible CPU ONNX benchmarks comparing `fastretrieval` against `fastembed`.

## Methodology

- **Script**: [`benchmarks/run_vs_fastembed.py`](../benchmarks/run_vs_fastembed.py)
- **Machine-readable output**: [`benchmarks/fastembed_comparison.json`](../benchmarks/fastembed_comparison.json)
- **Isolation**: Each runtime runs in a separate worker subprocess (`--worker fastretrieval` or `--worker fastembed`). This avoids cross-runtime ONNX Runtime arena sharing and prevents dependency conflicts (`fastembed` pins `loguru<0.8` whereas `fastretrieval` requires `loguru>=0.8.0`) while keeping core `fastretrieval` dependencies lean.
- **Metrics collected**:
  - `init_load_s`: Model and tokenizer initialization time with local cache warmed beforehand.
  - `latency_ms_mean`, `latency_ms_median`, `latency_ms_p95`: Per-pass wall latency across the full text batch.
  - `throughput_docs_per_s`: Total timed documents divided by total timed wall seconds.
  - `peak_rss_mb`: OS process peak working set / resident set size in MiB.
  - `peak_tracemalloc_mb`: Peak Python heap allocation during timed inference.
  - Vector parity: `cosine_similarity_mean`, `cosine_similarity_min`, `max_abs_diff`, `mean_abs_diff`, and L2 unit-norm deviation (`| ||v||_2 - 1.0 |`).

## Reproducible Command

```bash
uv run python benchmarks/run_vs_fastembed.py
```

Optional flags:

```bash
uv run python benchmarks/run_vs_fastembed.py \
  --scenarios minilm_l6_v2_same_onnx minilm_l6_v2_builtin_int8 \
  --batch-size 32 \
  --warmup 1 \
  --iterations 5 \
  --threads 4 \
  --output benchmarks/fastembed_comparison.json
```

When `fastembed` is not installed in the active environment, the script automatically launches the `fastembed` worker via `uv run --no-project --with fastembed`.

## Observed Results

Measured on `2026-11-26T16:19:59Z` with `fastretrieval 1.16.0`, `fastembed 0.9.1`, `onnxruntime 1.32.0`, Python `3.14.6` on `Windows-11-10.0.26300-SP0` (`Intel64 Family 6 Model 158 Stepping 13`, 6 cores, `CPUExecutionProvider`, `threads=4`, `batch_size=32`, `warmup=1`, `iterations=5`):

| Scenario | fastretrieval model | fastembed model | fr Throughput (docs/s) | fe Throughput (docs/s) | fr Mean Latency (ms) | fe Mean Latency (ms) | fr Peak RSS (MiB) | fe Peak RSS (MiB) | Mean Cosine |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `minilm_l6_v2_same_onnx` | `Qdrant/all-MiniLM-L6-v2-onnx` (FP32, 90 MB) | `sentence-transformers/all-MiniLM-L6-v2` (FP32, 90 MB) | 39.93 | 45.74 | 1602.61 | 1399.20 | 222.12 | 223.61 | 1.000000 |
| `minilm_l6_v2_builtin_int8` | `n24q02m/fr-all-MiniLM-L6-v2` (INT8, 23 MB) | `sentence-transformers/all-MiniLM-L6-v2` (FP32, 90 MB) | 14.13 | 50.82 | 4528.39 | 1259.46 | 211.16 | 223.52 | 0.957862 |
| `bge_small_en_v15_same_onnx` | `Qdrant/bge-small-en-v1.5-onnx-Q` (quantized, 67 MB) | `BAAI/bge-small-en-v1.5` (quantized, 67 MB) | 20.78 | 38.03 | 3079.74 | 1682.86 | 240.27 | 253.75 | 1.000000 |
| `multilingual_minilm_l12_v2` | `n24q02m/fr-paraphrase-multilingual-MiniLM-L12-v2` (INT8, 118 MB) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (quantized, 220 MB) | 17.90 | 13.40 | 1787.70 | 2388.61 | 480.32 | 657.07 | 0.979797 |

## Key Findings

1. **Exact numeric parity on shared ONNX graphs**: On both `Qdrant/all-MiniLM-L6-v2-onnx` (`MEAN` pooling) and `Qdrant/bge-small-en-v1.5-onnx-Q` (`CLS` pooling), `fastretrieval` via `CustomModelSpec` matches `fastembed` output vectors with `1.000000` mean and minimum cosine similarity and `0.0` max absolute difference.
2. **Lower Python heap and peak RSS footprint**: `fastretrieval` used lower `tracemalloc` peak Python memory in all four scenarios (`0.198-0.349 MiB` vs `0.282-0.426 MiB`) and lower OS peak RSS across all scenarios, notably `480.32 MiB` vs `657.07 MiB` (`0.731x`) on `paraphrase-multilingual-MiniLM-L12-v2`.
3. **Artifact size vs CPU INT8 throughput tradeoff**: `n24q02m/fr-all-MiniLM-L6-v2` reduces model artifact size by ~3.9x (`23 MB` vs `90 MB`) and keeps cosine similarity at `0.957862` (`min=0.937147`) against FP32, while dynamic INT8 `MatMulInteger` execution on non-VNNI x86 CPUs (`Family 6 Model 158`) runs slower than FP32 (`14.13` vs `50.82` docs/s). When maximum CPU throughput on older x86 hardware matters more than disk footprint, registering an FP32 ONNX artifact via `CustomModelSpec` restores full FP32 speed.
4. **Built-in L2 normalization on multilingual MiniLM**: `n24q02m/fr-paraphrase-multilingual-MiniLM-L12-v2` emits unit-normalized vectors (`max_dev = 0.0`) and achieved `1.336x` higher throughput (`17.90` vs `13.40` docs/s) than `fastembed`'s unnormalized `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (`norm_max_dev = 3.637845`, `mean cosine = 0.979797`).

## CI Coverage Note

Standard CI runs (`pytest -m "not integration"`) execute unit and schema/contract tests in [`tests/test_benchmarks.py`](../tests/test_benchmarks.py) with stubbed worker execution so CI enforces `>= 95%` coverage without downloading Hugging Face models. Live end-to-end comparison remains reproducibly runnable locally via `uv run python benchmarks/run_vs_fastembed.py`.
