"""Đối chiếu artifact đã chuyển đổi với model gốc chạy bằng torch.

Lượng tử hoá hỏng không nhất thiết làm model crash: vector vẫn có thể đúng
chiều nhưng sai nội dung. Module này kiểm tra manifest trước, sau đó chạy cùng
một bộ probe qua model gốc và ONNX Runtime để bắt sai lệch trước khi publish.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from loguru import logger

from fastretrieval.contract import ModelContract
from fastretrieval.convert import require_convert_deps
from fastretrieval.convert.manifest import load_manifest

# Strict gate for full-precision (or unrecognized) artifacts.
DEFAULT_FP32_ATOL = 1e-2
# Per-variant defaults for the converter's own quantized outputs. Measured on
# sentence-transformers/all-MiniLM-L6-v2 (e2e 2026-09-26): int8 max_abs 0.0756
# (cosine 0.9496), q4f16 max_abs 0.0764 (cosine 0.9248). Defaults keep ~30%
# margin on int8 (8-bit weights drift narrowly) and ~2x margin on q4f16
# (4-bit weights drift wider across architectures).
DEFAULT_VARIANT_ATOL: dict[str, float] = {"int8": 0.1, "q4f16": 0.15}

PROBES = [
    "retrieval augmented generation",
    "the quick brown fox jumps over the lazy dog",
    "Xin chào, đây là một câu tiếng Việt có dấu.",
    "",
]


def _artifact_files(directory: Path, suffix: str) -> tuple[Path, ...]:
    """Return every artifact with ``suffix`` in deterministic order."""
    normalized = suffix.lower()
    return tuple(
        sorted(
            (
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() == normalized
            ),
            key=lambda path: path.as_posix(),
        )
    )


def _onnx_artifacts(directory: Path, contract: ModelContract) -> tuple[Path, ...]:
    if "onnx" not in contract.artifact_formats:
        raise ValueError(
            f"{contract.model_id}: ONNX artifacts are not declared; "
            f"manifest formats are {contract.artifact_formats!r}"
        )
    artifacts = _artifact_files(directory, ".onnx")
    if not artifacts:
        raise FileNotFoundError(f"{contract.model_id}: no ONNX artifact found under {directory}")
    return artifacts


def _variant_for_artifact(artifact: Path) -> str | None:
    """Map an artifact filename to its quantization variant (None = full precision).

    Follows the converter's naming convention in ``convert/onnx.py``:
    ``model_quantized.onnx`` (int8) and ``model_q4f16.onnx`` (q4f16).
    """
    name = PurePosixPath(artifact.as_posix()).name.lower()
    if "q4f16" in name:
        return "q4f16"
    if "int8" in name or "quantized" in name:
        return "int8"
    return None


def _resolve_atol(atol: float | None, variant: str | None) -> float:
    """Explicit ``atol`` always wins; otherwise pick the per-variant default."""
    if atol is not None:
        return atol
    if variant is None:
        return DEFAULT_FP32_ATOL
    return DEFAULT_VARIANT_ATOL.get(variant, DEFAULT_FP32_ATOL)


def compare_embeddings(reference: Any, candidate: Any, *, atol: float) -> dict[str, Any]:
    """So sánh hai mảng embedding.

    Raises:
        ValueError: khi hai mảng khác shape — đó là lỗi cấu hình, không phải
            sai số có thể chấm điểm.
    """
    if atol < 0:
        raise ValueError("atol must be non-negative")

    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            f"shape mismatch: reference {reference_array.shape}, got {candidate_array.shape}"
        )

    diff = np.abs(reference_array - candidate_array)
    flat_reference = reference_array.ravel()
    flat_candidate = candidate_array.ravel()
    denominator = float(np.linalg.norm(flat_reference) * np.linalg.norm(flat_candidate))
    cosine = float(flat_reference @ flat_candidate / denominator) if denominator else 1.0
    max_abs = float(diff.max()) if diff.size else 0.0
    return {
        "passed": max_abs <= atol,
        "max_abs_diff": max_abs,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "cosine": cosine,
        "atol": atol,
    }


def verify_manifest(
    converted_dir: str | Path, expected_source: str | None = None
) -> ModelContract:
    """Đọc manifest và kiểm tra artifact/tokenizer tối thiểu.

    ``verify_converted`` gọi hàm này trước khi nạp torch, transformers hoặc
    ONNX Runtime. Nhờ vậy artifact thiếu metadata, sai source hoặc không có
    format chạy được sẽ fail closed thay vì bị đoán profile khác.
    """
    directory = Path(converted_dir)
    if not directory.is_dir():
        raise ValueError(f"converted artifact directory does not exist: {directory}")

    contract = load_manifest(directory)
    if expected_source is not None and contract.source != expected_source:
        raise ValueError(
            f"manifest source mismatch: expected {expected_source!r}, got {contract.source!r}"
        )

    missing_tokenizers = [
        relative for relative in contract.tokenizer_files if not (directory / relative).is_file()
    ]
    if missing_tokenizers:
        raise ValueError(f"manifest tokenizer files are missing: {', '.join(missing_tokenizers)}")

    declared_formats = set(contract.artifact_formats)
    existing_formats: set[str] = set()
    if _artifact_files(directory, ".onnx"):
        existing_formats.add("onnx")
    if _artifact_files(directory, ".gguf"):
        existing_formats.add("gguf")
    undeclared = existing_formats - declared_formats
    if undeclared:
        raise ValueError(
            f"manifest artifact_formats does not declare existing format(s): "
            f"{', '.join(sorted(undeclared))}"
        )
    missing_declared = declared_formats - existing_formats
    if missing_declared:
        raise ValueError(
            f"manifest declares artifact format(s) with no artifact: "
            f"{', '.join(sorted(missing_declared))}"
        )
    return contract


def validate_artifacts(
    converted_dir: str | Path, expected_source: str | None = None
) -> ModelContract:
    """Validate every declared binary artifact before it is promoted.

    ONNX validation deliberately combines the format checker and an actual
    CPU Runtime session. A graph that serializes successfully but cannot load
    or expose an input/output is not a usable conversion artifact.
    """
    directory = Path(converted_dir)
    contract = verify_manifest(directory, expected_source=expected_source)
    if "onnx" in contract.artifact_formats:
        require_convert_deps("onnx", "onnxruntime")
        import onnx
        import onnxruntime as ort

        for artifact in _onnx_artifacts(directory, contract):
            try:
                model = onnx.load(str(artifact))
                onnx.checker.check_model(model)
                session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
            except Exception as exc:
                raise ValueError(
                    f"{contract.model_id}: invalid ONNX artifact {artifact}: {exc}"
                ) from exc
            if not session.get_inputs():
                raise ValueError(f"{contract.model_id}: ONNX artifact has no inputs: {artifact}")
            if not session.get_outputs():
                raise ValueError(f"{contract.model_id}: ONNX artifact has no outputs: {artifact}")
    return contract


def _pool_hidden(hidden: Any, attention_mask: Any, pooling: Any) -> Any:
    """Pool a transformer hidden-state tensor according to the contract."""
    import torch

    if hidden.ndim == 2:
        return hidden
    if hidden.ndim != 3:
        raise ValueError(
            f"expected a 2D or 3D transformer output, got shape {tuple(hidden.shape)}"
        )

    pooling_name = getattr(pooling, "value", pooling)
    pooling_name = str(pooling_name).upper()
    if pooling_name == "DISABLED":
        raise ValueError("manifest pooling=DISABLED cannot produce dense vectors from 3D output")
    if pooling_name == "CLS":
        return hidden[:, 0, :]

    if pooling_name == "MEAN":
        # ⚡ Bolt: Fast mean pooling using bmm (~4.5x faster than unsqueeze and sum)
        mask = attention_mask.to(dtype=hidden.dtype)
        sum_embeddings = torch.bmm(mask.unsqueeze(1), hidden).squeeze(1)
        # ⚡ Bolt: Fast reduction using original integer tensor sum before casting to float
        sum_mask = attention_mask.sum(dim=1, keepdim=True).to(dtype=hidden.dtype)
        sum_mask.clamp_(min=1e-9)
        return sum_embeddings / sum_mask
    if pooling_name == "LAST_TOKEN":
        # ⚡ Bolt: Fast last token indices without casting to float first
        last_indices = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_indices, last_indices, :]
    raise ValueError(f"unsupported pooling {pooling!r}")


def _pool_numpy(output: np.ndarray, attention_mask: np.ndarray, pooling: Any) -> np.ndarray:
    if output.ndim == 2:
        return output
    if output.ndim != 3:
        raise ValueError(f"expected a 2D or 3D ONNX output, got shape {output.shape}")

    pooling_name = getattr(pooling, "value", pooling)
    pooling_name = str(pooling_name).upper()
    if pooling_name == "DISABLED":
        raise ValueError("manifest pooling=DISABLED cannot produce dense vectors from 3D output")
    if pooling_name == "CLS":
        return output[:, 0, :]
    if pooling_name == "MEAN":
        # ⚡ Bolt: Fast mean pooling using np.matmul (~4.5x faster than np.expand_dims and np.sum)
        mask_cast = attention_mask.astype(output.dtype)
        sum_embeddings = np.matmul(mask_cast[:, np.newaxis, :], output).squeeze(1)
        # ⚡ Bolt: Fast reduction using original integer array sum before casting to float
        sum_mask = attention_mask.sum(axis=1, keepdims=True).astype(output.dtype, copy=False)
        np.maximum(sum_mask, 1e-9, out=sum_mask)
        return sum_embeddings / sum_mask
    if pooling_name == "LAST_TOKEN":
        # ⚡ Bolt: Fast last token indices without casting to float first
        last_indices = np.maximum(attention_mask.sum(axis=1), 1) - 1
        return output[np.arange(output.shape[0]), last_indices, :]
    raise ValueError(f"unsupported pooling {pooling!r}")


def _normalize(array: Any, enabled: bool) -> Any:
    if not enabled:
        return array
    import torch

    if isinstance(array, torch.Tensor):
        return torch.nn.functional.normalize(array, p=2, dim=-1)
    values = np.asarray(array, dtype=np.float32)
    # ⚡ Bolt: Fast L2 norm for >1D using einsum and in-place operations (~60% faster)
    norm_sq = np.einsum("...i,...i->...", values, values)
    if np.isscalar(norm_sq) or norm_sq.ndim == 0:
        norms = np.sqrt(norm_sq)
        return values / np.maximum(norms, 1e-12)
    norms = np.sqrt(norm_sq, out=norm_sq)[..., np.newaxis]
    return values / np.maximum(norms, 1e-12, out=norms)


def _validate_output_shape(array: np.ndarray, contract: ModelContract, label: str) -> None:
    if array.ndim != 2:
        raise ValueError(f"{contract.model_id}: {label} output must be 2D, got {array.shape}")
    expected_shape = contract.output_shape
    actual_shape = tuple(array.shape[1:])
    if expected_shape is not None:
        if len(expected_shape) != len(actual_shape):
            raise ValueError(
                f"{contract.model_id}: {label} output shape mismatch: "
                f"manifest {expected_shape}, got {actual_shape}"
            )
        for expected, actual in zip(expected_shape, actual_shape, strict=True):
            if expected is not None and expected != actual:
                raise ValueError(
                    f"{contract.model_id}: {label} output shape mismatch: "
                    f"manifest {expected_shape}, got {actual_shape}"
                )
    if contract.output_dim is not None and array.shape[-1] != contract.output_dim:
        raise ValueError(
            f"{contract.model_id}: {label} output dimension mismatch: "
            f"manifest {contract.output_dim}, got {array.shape[-1]}"
        )


def _reference_embeddings(source: str, contract: ModelContract) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    if contract.task == "cross_encoder":
        # Cross-encoder chuẩn ms-marco export AutoModelForSequenceClassification
        # (optimum task=text-classification); đầu ra đối chiếu là logits thô,
        # không phải hidden states — model class thực quyết định ngữ nghĩa output.
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            source, torch_dtype=torch.float32
        ).eval()
    else:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(source, torch_dtype=torch.float32).eval()
    tokenizer_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
    }
    if contract.max_seq_len is not None:
        tokenizer_kwargs["max_length"] = contract.max_seq_len
    with torch.no_grad():
        batch = tokenizer(PROBES, **tokenizer_kwargs)
        outputs = model(**batch)
        if contract.task == "cross_encoder":
            reference = np.asarray(outputs.logits.cpu(), dtype=np.float32)
        else:
            hidden = outputs.last_hidden_state
            pooled = _pool_hidden(hidden, batch["attention_mask"], contract.pooling)
            reference = _normalize(pooled, contract.normalization).cpu().numpy()
    return np.asarray(reference, dtype=np.float32)


def _onnx_embeddings(artifact: Path, contract: ModelContract, source: str) -> np.ndarray:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer_kwargs: dict[str, Any] = {
        "return_tensors": "np",
        "padding": True,
        "truncation": True,
    }
    if contract.max_seq_len is not None:
        tokenizer_kwargs["max_length"] = contract.max_seq_len
    batch = tokenizer(PROBES, **tokenizer_kwargs)

    session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    missing_inputs = sorted(input_names - set(batch))
    if missing_inputs:
        raise ValueError(
            f"{contract.model_id}: tokenizer cannot provide ONNX input(s): "
            f"{', '.join(missing_inputs)}"
        )
    feed = {name: np.asarray(batch[name]) for name in input_names}
    outputs = session.run(None, feed)
    if not outputs:
        raise ValueError(f"{contract.model_id}: ONNX session returned no outputs")
    pooled = _pool_numpy(
        np.asarray(outputs[0]), np.asarray(batch["attention_mask"]), contract.pooling
    )
    return np.asarray(_normalize(pooled, contract.normalization), dtype=np.float32)


def verify_converted(
    converted_dir: str | Path, source: str, *, atol: float | None = None
) -> dict[str, Any]:
    """So cùng bộ probe qua model gốc và artifact ONNX đã chuyển đổi.

    Args:
        converted_dir: thư mục artifact đã chuyển đổi (có manifest).
        source: model id gốc trên HuggingFace.
        atol: ngưỡng tolerance tường minh, áp dụng cho mọi variant khi được
            cung cấp (override thắng). Mặc định ``None`` chọn tolerance
            per-variant: fp32 giữ nghiêm ``DEFAULT_FP32_ATOL``, int8/q4f16 dùng
            ngưỡng rộng hơn trong ``DEFAULT_VARIANT_ATOL`` khớp sai số đo được
            của chính converter.

    Returns:
        Report dict; mỗi entry trong ``variant_reports`` mang ``atol`` hiệu lực
        của variant đó.
    """
    if atol is not None and atol < 0:
        raise ValueError("atol must be non-negative")
    contract = validate_artifacts(converted_dir, expected_source=source)
    if contract.task not in {"dense", "cross_encoder"} or contract.modality != "text":
        raise ValueError(
            f"{contract.model_id}: verify_converted supports only task='dense'/'cross_encoder', "
            f"modality='text'; got task={contract.task!r}, modality={contract.modality!r}"
        )
    if contract.task == "cross_encoder" and contract.normalization:
        raise ValueError(
            f"{contract.model_id}: cross_encoder verification requires normalization=False; "
            "classification logits are raw scores, not L2-normalized vectors"
        )
    artifacts = _onnx_artifacts(Path(converted_dir), contract)
    require_convert_deps("torch", "transformers", "onnxruntime")

    reference = _reference_embeddings(source, contract)
    _validate_output_shape(reference, contract, "reference")
    variant_reports: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        candidate = _onnx_embeddings(artifact, contract, source)
        _validate_output_shape(candidate, contract, f"converted {artifact.name}")
        variant_reports[str(artifact.relative_to(Path(converted_dir)).as_posix())] = (
            compare_embeddings(
                reference, candidate, atol=_resolve_atol(atol, _variant_for_artifact(artifact))
            )
        )

    used_atols = [item["atol"] for item in variant_reports.values()]
    report = {
        "passed": all(item["passed"] for item in variant_reports.values()),
        "max_abs_diff": max(item["max_abs_diff"] for item in variant_reports.values()),
        "mean_abs_diff": max(item["mean_abs_diff"] for item in variant_reports.values()),
        "cosine": min(item["cosine"] for item in variant_reports.values()),
        "atol": atol,
        "atol_mode": "override" if atol is not None else "per_variant",
        "atol_range": (min(used_atols), max(used_atols)) if used_atols else (atol, atol),
        "variant_reports": variant_reports,
    }
    report.update(
        {
            "probes": len(PROBES),
            "artifact": str(artifacts[0]),
            "artifacts": [str(artifact) for artifact in artifacts],
            "model_id": contract.model_id,
        }
    )
    logger.info(
        "max abs diff {:.2e}, cosine {:.6f}, {}",
        report["max_abs_diff"],
        report["cosine"],
        "PASS" if report["passed"] else "FAIL",
    )
    return report


__all__ = [
    "DEFAULT_FP32_ATOL",
    "DEFAULT_VARIANT_ATOL",
    "PROBES",
    "compare_embeddings",
    "validate_artifacts",
    "verify_converted",
    "verify_manifest",
]
