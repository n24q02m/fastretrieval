"""Unit and artifact-contract tests for benchmarks/run_vs_fastembed.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import run_vs_fastembed as bench


class _StubModel:
    """Deterministic in-memory embedding model stub for worker tests."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[tuple[int, int]] = []

    def embed(self, documents: list[str], batch_size: int = 32) -> list[np.ndarray]:
        self.calls.append((len(documents), batch_size))
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((len(documents), self.dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        normalized = raw / np.maximum(norms, 1e-12)
        return [normalized[i] for i in range(len(documents))]


def test_get_corpus_profiles() -> None:
    english = bench.get_corpus("english")
    multilingual = bench.get_corpus("multilingual")
    assert len(english) >= 32
    assert len(multilingual) >= 16
    assert all(isinstance(item, str) and item.strip() for item in english)
    assert all(isinstance(item, str) and item.strip() for item in multilingual)
    with pytest.raises(ValueError, match="Unknown corpus name"):
        bench.get_corpus("nonexistent")


def test_scenarios_are_well_formed() -> None:
    ids = [s.scenario_id for s in bench.SCENARIOS]
    assert len(ids) == len(set(ids))
    assert len(bench.SCENARIOS) >= 2
    for scenario in bench.SCENARIOS:
        assert scenario.fastretrieval.model_id
        assert scenario.fastembed.model_id
        assert bench.get_corpus(scenario.corpus_name)


def test_summarize_latencies() -> None:
    summary = bench.summarize_latencies([0.1, 0.2, 0.3, 0.4, 0.5], num_docs=10)
    assert summary["latency_ms_mean"] == pytest.approx(300.0)
    assert summary["latency_ms_median"] == pytest.approx(300.0)
    assert summary["latency_ms_min"] == pytest.approx(100.0)
    assert summary["latency_ms_max"] == pytest.approx(500.0)
    assert summary["latency_ms_p95"] == pytest.approx(500.0)
    assert summary["per_doc_ms_mean"] == pytest.approx(30.0)
    assert summary["throughput_docs_per_s"] == pytest.approx(50 / 1.5, rel=1e-3)
    assert summary["throughput_median_docs_per_s"] == pytest.approx(10 / 0.3, rel=1e-3)

    with pytest.raises(ValueError, match="non-empty"):
        bench.summarize_latencies([], num_docs=10)
    with pytest.raises(ValueError, match="positive"):
        bench.summarize_latencies([0.1], num_docs=0)


def test_compute_parity_metrics() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    identical = bench.compute_parity_metrics(a, b)
    assert identical["cosine_similarity_mean"] == pytest.approx(1.0)
    assert identical["cosine_similarity_min"] == pytest.approx(1.0)
    assert identical["max_abs_diff"] == pytest.approx(0.0)
    assert identical["fastretrieval_l2_norm_max_dev"] == pytest.approx(0.0)

    c = np.array([[0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    shifted = bench.compute_parity_metrics(a, c)
    assert shifted["cosine_similarity_mean"] == pytest.approx(0.9)
    assert shifted["cosine_similarity_min"] == pytest.approx(0.8)

    with pytest.raises(ValueError, match="Shape mismatch"):
        bench.compute_parity_metrics(a, np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty 2-D"):
        bench.compute_parity_metrics(
            np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        )


def test_build_comparison_summary_handles_zero_and_normal() -> None:
    summary = bench.build_comparison_summary(
        fr_metrics={
            "throughput_docs_per_s": 100.0,
            "latency_ms_mean": 50.0,
            "peak_rss_mb": 200.0,
            "init_load_s": 1.0,
        },
        fe_metrics={
            "throughput_docs_per_s": 50.0,
            "latency_ms_mean": 100.0,
            "peak_rss_mb": 400.0,
            "init_load_s": 2.0,
        },
        parity={"cosine_similarity_mean": 0.999},
    )
    assert summary["throughput_speedup_vs_fastembed"] == pytest.approx(2.0)
    assert summary["latency_ratio_vs_fastembed"] == pytest.approx(0.5)
    assert summary["peak_rss_ratio_vs_fastembed"] == pytest.approx(0.5)
    assert summary["init_load_ratio_vs_fastembed"] == pytest.approx(0.5)
    assert summary["cosine_similarity_mean"] == pytest.approx(0.999)

    zero_summary = bench.build_comparison_summary(
        fr_metrics={
            "throughput_docs_per_s": 10.0,
            "latency_ms_mean": 10.0,
            "peak_rss_mb": 10.0,
            "init_load_s": 1.0,
        },
        fe_metrics={
            "throughput_docs_per_s": 0.0,
            "latency_ms_mean": 0.0,
            "peak_rss_mb": 0.0,
            "init_load_s": 0.0,
        },
        parity={},
    )
    assert zero_summary["throughput_speedup_vs_fastembed"] == 0.0
    assert zero_summary["peak_rss_ratio_vs_fastembed"] == 0.0


def test_get_peak_rss_mb_positive() -> None:
    rss = bench.get_peak_rss_mb()
    assert isinstance(rss, float)
    assert rss > 0.0


def test_resolve_worker_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = tmp_path / "run_vs_fastembed.py"
    assert bench.resolve_worker_command("fastretrieval", script) == [
        sys.executable,
        str(script),
    ]

    monkeypatch.setattr(bench.importlib.util, "find_spec", lambda name: object())
    assert bench.resolve_worker_command("fastembed", script) == [
        sys.executable,
        str(script),
    ]

    monkeypatch.setattr(bench.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(bench.shutil, "which", lambda name: "/usr/bin/uv")
    cmd = bench.resolve_worker_command("fastembed", script, fastembed_spec="fastembed==0.9.1")
    assert cmd[:2] == ["/usr/bin/uv", "run"]
    assert "--no-project" in cmd
    assert "fastembed==0.9.1" in cmd

    monkeypatch.setattr(bench.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="uv"):
        bench.resolve_worker_command("fastembed", script)

    with pytest.raises(ValueError, match="Unsupported runtime"):
        bench.resolve_worker_command("unknown", script)


def test_run_worker_warm_and_timed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stub = _StubModel(dim=8)
    monkeypatch.setattr(
        bench,
        "_load_runtime_model",
        lambda runtime, target, threads: (stub, "1.16.0", "1.32.0"),
    )
    target = bench.RuntimeTarget(model_id="demo/model", artifact_size_mb=10.0)
    corpus = ["alpha", "beta", "gamma", "delta"]

    warm_json = tmp_path / "warm.json"
    warm_payload = bench.run_worker(
        runtime="fastretrieval",
        target=target,
        corpus=corpus,
        batch_size=2,
        warmup=1,
        iterations=2,
        threads=2,
        result_path=warm_json,
        vectors_path=None,
        warm_cache_only=True,
    )
    assert warm_payload["cache_warmed"] is True
    assert stub.calls == []

    timed_json = tmp_path / "timed.json"
    vectors_npy = tmp_path / "vectors.npy"
    timed_payload = bench.run_worker(
        runtime="fastretrieval",
        target=target,
        corpus=corpus,
        batch_size=2,
        warmup=1,
        iterations=2,
        threads=2,
        result_path=timed_json,
        vectors_path=vectors_npy,
        warm_cache_only=False,
    )
    assert len(stub.calls) == 3  # 1 warmup + 2 timed
    assert timed_payload["embedding_shape"] == [4, 8]
    assert timed_payload["peak_rss_mb"] > 0.0
    assert json.loads(timed_json.read_text(encoding="utf-8"))["runtime"] == "fastretrieval"
    assert np.load(vectors_npy).shape == (4, 8)


def test_run_benchmark_suite_and_main_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def _fake_resolve(
        runtime: str, script_path: Path, fastembed_spec: str = "fastembed"
    ) -> list[str]:
        return [runtime]

    def _fake_invoke(
        cmd_prefix: list[str],
        runtime: str,
        target: bench.RuntimeTarget,
        corpus_name: str,
        batch_size: int,
        warmup: int,
        iterations: int,
        threads: int,
        result_path: Path,
        vectors_path: Path | None,
        warm_cache_only: bool = False,
    ) -> dict[str, Any]:
        if warm_cache_only:
            payload = {"runtime": runtime, "cache_warmed": True}
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            return payload

        docs = bench.get_corpus(corpus_name)
        vecs = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(docs), 1))
        assert vectors_path is not None
        np.save(vectors_path, vecs)
        payload = {
            "runtime": runtime,
            "runtime_version": "1.16.0" if runtime == "fastretrieval" else "0.9.1",
            "onnxruntime_version": "1.32.0",
            "model_id": target.model_id,
            "artifact_size_mb": target.artifact_size_mb,
            "init_load_s": 0.5,
            "warmup_s": 0.1,
            "iterations": iterations,
            "num_docs": len(docs),
            "batch_size": batch_size,
            "threads": threads,
            "peak_tracemalloc_mb": 0.25,
            "peak_rss_mb": 120.0 if runtime == "fastretrieval" else 150.0,
            "embedding_shape": [len(docs), 4],
            "latency_ms_mean": 100.0 if runtime == "fastretrieval" else 200.0,
            "latency_ms_median": 100.0 if runtime == "fastretrieval" else 200.0,
            "latency_ms_p95": 110.0,
            "latency_ms_min": 95.0,
            "latency_ms_max": 110.0,
            "per_doc_ms_mean": 1.5,
            "total_timed_s": 0.5 if runtime == "fastretrieval" else 1.0,
            "throughput_docs_per_s": 128.0 if runtime == "fastretrieval" else 64.0,
            "throughput_median_docs_per_s": 128.0 if runtime == "fastretrieval" else 64.0,
        }
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(bench, "resolve_worker_command", _fake_resolve)
    monkeypatch.setattr(bench, "_invoke_worker", _fake_invoke)

    out_path = tmp_path / "comparison.json"
    exit_code = bench.main(
        [
            "--scenarios",
            bench.SCENARIOS[0].scenario_id,
            "--batch-size",
            "16",
            "--iterations",
            "2",
            "--output",
            str(out_path),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert bench.SCENARIOS[0].scenario_id in captured.out

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["metadata"]["batch_size"] == 16
    assert len(report["scenarios"]) == 1
    assert report["scenarios"][0]["comparison"][
        "throughput_speedup_vs_fastembed"
    ] == pytest.approx(2.0)


def test_persisted_benchmark_artifact_contract() -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1] / "benchmarks" / "fastembed_comparison.json"
    )
    assert artifact_path.exists(), f"Missing persisted benchmark output: {artifact_path}"
    data = json.loads(artifact_path.read_text(encoding="utf-8"))

    metadata = data["metadata"]
    assert metadata["command"] == "uv run python benchmarks/run_vs_fastembed.py"
    assert metadata["fastretrieval_version"]
    assert metadata["fastembed_version"]
    assert metadata["onnxruntime_versions"]

    scenarios = data["scenarios"]
    expected_ids = {s.scenario_id for s in bench.SCENARIOS}
    assert {item["scenario_id"] for item in scenarios} == expected_ids

    for item in scenarios:
        for runtime in ("fastretrieval", "fastembed"):
            metrics = item[runtime]
            assert metrics["throughput_docs_per_s"] > 0.0
            assert metrics["latency_ms_mean"] > 0.0
            assert metrics["peak_rss_mb"] > 0.0
        parity = item["comparison"]
        assert parity["cosine_similarity_mean"] >= 0.90
        if item["comparison_kind"] == "same_weights":
            assert parity["cosine_similarity_mean"] >= 0.9999
