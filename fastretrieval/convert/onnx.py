"""Export ONNX rồi lượng tử hoá xuống INT8 và Q4F16.

Bước export FP32 dùng optimum vì nó xử lý được kiến trúc tuỳ ý; hai bước
lượng tử hoá là phần optimum không làm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from fastretrieval.convert import require_convert_deps

VALID_VARIANTS = ("int8", "q4f16")


def fix_cast_nodes(graph: Any) -> None:
    """Đổi Cast(to=FLOAT) thành Cast(to=FLOAT16), đệ quy vào subgraph.

    ``convert_float_to_float16`` bỏ sót Cast nằm trong nhánh của If/Loop. Một
    node sót lại làm đồ thị FP16 lỗi kiểu lúc chạy chứ không lỗi lúc load.
    """
    import onnx

    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                    attr.i = onnx.TensorProto.FLOAT16
        for attr in node.attribute:
            if attr.g and isinstance(attr.g, onnx.GraphProto):
                fix_cast_nodes(attr.g)


def quantize_int8(fp32_path: Path, out_path: Path) -> float:
    """Lượng tử hoá động xuống INT8. Trả kích thước kết quả tính bằng MB."""
    require_convert_deps("onnxruntime")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("quantizing to INT8: {}", out_path)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )
    return out_path.stat().st_size / (1024**2)


def quantize_q4f16(fp32_path: Path, out_path: Path) -> float:
    """Trọng số INT4, kích hoạt FP16. Trả kích thước kết quả tính bằng MB."""
    require_convert_deps("onnx", "onnxconverter_common", "onnxruntime")
    import onnx
    from onnxconverter_common import float16
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("quantizing to Q4F16: {}", out_path)
    quantizer = MatMulNBitsQuantizer(
        model=str(fp32_path),
        bits=4,
        block_size=128,
        is_symmetric=True,
        accuracy_level=4,
    )
    quantizer.process()

    model = float16.convert_float_to_float16(quantizer.model.model, keep_io_types=False)
    fix_cast_nodes(model.graph)
    model.graph.ClearField("value_info")
    onnx.save(model, str(out_path))
    return out_path.stat().st_size / (1024**2)


def convert_onnx(
    source: str,
    out_dir: str | Path,
    *,
    task: str = "feature-extraction",
    modality: str = "text",
    variants: list[str] | None = None,
    pooling: str | None = None,
    normalization: bool | None = None,
    yes_no: tuple[str, str] | None = None,
) -> dict[str, float]:
    """Resolve profile, export ``source`` và ghi artifact + manifest.

    Args:
        source: model id trên HuggingFace hoặc thư mục local.
        out_dir: thư mục kết quả.
        task: task export của optimum.
        modality: modality đã khai báo trong contract.
        variants: tập con của ``("int8", "q4f16")``; None nghĩa là cả hai.
        pooling: pooling tường minh; không được tự đoán khi profile thiếu.
        normalization: trạng thái chuẩn hoá output; ``None`` chỉ hợp lệ khi profile khai báo.
        yes_no: cặp (yes_token, no_token) để rút reranker causal-LM xuống 2 logit.

    Returns:
        Ánh xạ tên variant sang kích thước MB.

    Raises:
        ValueError: khi ``variants`` chứa tên lạ hoặc không có variant nào.
    """
    variants = list(variants) if variants is not None else list(VALID_VARIANTS)
    unknown = [variant for variant in variants if variant not in VALID_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variant(s): {', '.join(unknown)}; pick from {VALID_VARIANTS}")
    if not variants:
        raise ValueError("at least one variant is required")

    require_convert_deps("torch", "transformers", "optimum", "onnx", "onnxruntime")

    out_dir = Path(out_dir)
    from fastretrieval.convert.manifest import write_manifest
    from fastretrieval.convert.profiles import resolve_profile

    profile = resolve_profile(source, task=task, modality=modality)
    contract = profile.build_contract(
        pooling=pooling,
        normalization=normalization,
        yes_no=yes_no,
    )
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if yes_no is not None:
        from fastretrieval.convert.heads import export_yes_no_head

        fp32_path = export_yes_no_head(source, out_dir, yes_token=yes_no[0], no_token=yes_no[1])
    else:
        from fastretrieval.export import export_to_onnx

        export_to_onnx(source, str(out_dir), task=task)
        fp32_path = onnx_dir / "model.onnx"
        if not fp32_path.exists():
            raise FileNotFoundError(f"optimum did not write {fp32_path}")

    sizes: dict[str, float] = {}
    if "int8" in variants:
        sizes["int8"] = quantize_int8(fp32_path, onnx_dir / "model_quantized.onnx")
    if "q4f16" in variants:
        sizes["q4f16"] = quantize_q4f16(fp32_path, onnx_dir / "model_q4f16.onnx")

    fp32_data = fp32_path.with_suffix(".onnx.data")
    fp32_path.unlink()
    if fp32_data.exists():
        fp32_data.unlink()

    write_manifest(out_dir, contract, metadata={"source": source, "variants": variants})

    return sizes
