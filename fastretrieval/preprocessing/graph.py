"""Phân giải xem graph hay runtime chịu trách nhiệm tiền xử lý."""

from __future__ import annotations

from pathlib import Path

from fastretrieval.contract.preprocessor import PreprocessorSpec

EXTENSION_DOMAINS = frozenset({"ai.onnx.contrib", "com.microsoft.extensions"})


def graph_is_self_contained(model_path: Path) -> bool:
    """True khi graph khai báo custom operator của onnxruntime-extensions."""
    try:
        import onnx
    except ImportError:
        return False
    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception:
        return False
    return any(entry.domain in EXTENSION_DOMAINS for entry in model.opset_import)


def resolve_preprocessor(
    model_dir: Path,
    model_file: str,
    declared: PreprocessorSpec | None,
) -> PreprocessorSpec:
    """Chọn graph tự chứa, khai báo, config file, rồi mặc định text."""
    model_dir = Path(model_dir)
    if graph_is_self_contained(model_dir / model_file):
        return PreprocessorSpec(kind="none")
    if declared is not None:
        return declared
    from_config = PreprocessorSpec.from_model_dir(model_dir)
    if from_config is not None:
        return from_config
    return PreprocessorSpec(kind="text")
