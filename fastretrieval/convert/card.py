"""Sinh model card cho thư mục đã chuyển đổi."""

from __future__ import annotations

from fastretrieval.contract import ModelContract

KINDS = {
    "embedding": ("feature-extraction", "TextEmbedding"),
    "reranker": ("text-classification", "TextCrossEncoder"),
}

_FILENAMES = {
    "int8": "onnx/model_quantized.onnx",
    "q4f16": "onnx/model_q4f16.onnx",
}

_TEMPLATE = """\
---
tags:{tags_block}
base_model: {source}
pipeline_tag: {pipeline_tag}
model_family: {model_family}
modality: {modality}
pooling: {pooling}
normalization: {normalization}
---

# {build_title}

Converted with [fastretrieval](https://github.com/n24q02m/fastretrieval).

## Variants

| Variant | File | Size |
|---|---|---|
{rows}

## Usage

```python
from fastretrieval import {usage_class}

model = {usage_class}("<this-repo-id>")
```

## Conversion

{conversion_notes}

## License

This build inherits the license of the base model. Check
[{source}](https://huggingface.co/{source}) before use.
"""


def render_card(
    source: str,
    sizes: dict[str, float],
    *,
    kind: str,
    contract: ModelContract | None = None,
) -> str:
    """Dựng README.md phản ánh support profile và artifact thực tế."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; pick from {sorted(KINDS)}")
    if contract is None:
        raise ValueError("converted artifact manifest is required to render a model card")
    declared = {str(fmt).lower() for fmt in contract.artifact_formats}
    if not declared & {"onnx", "gguf"}:
        raise ValueError(
            f"{contract.model_id}: model cards support ONNX/GGUF artifacts only; "
            f"manifest formats are {contract.artifact_formats!r}"
        )
    unknown_variants = sorted(
        name
        for name in sizes
        if name not in _FILENAMES and not name.lower().endswith((".onnx", ".gguf"))
    )
    if unknown_variants:
        raise ValueError(f"unknown artifact variant(s): {', '.join(unknown_variants)}")
    if not sizes:
        raise ValueError(f"{contract.model_id}: at least one artifact variant is required")
    pipeline_tag, usage_class = KINDS[kind]
    rows = "\n".join(
        f"| {name} | `{_FILENAMES.get(name, name)}` | {megabytes:.0f} MB |"
        for name, megabytes in sorted(sizes.items())
    )
    if not rows:
        rows = "| - | - | - |"
    pooling = getattr(contract.pooling, "value", contract.pooling)

    onnx_notes = (
        "- ONNX opset 21\n"
        "- INT8: `onnxruntime.quantization.quantize_dynamic` (QInt8)\n"
        "- Q4F16: `MatMulNBitsQuantizer` (block_size 128, symmetric) then a float16 cast"
    )
    quant = f" ({contract.quantization})" if contract.quantization else ""
    gguf_notes = "- GGUF: llama.cpp `convert_hf_to_gguf.py` (F16), then `llama-quantize`" + quant
    if declared == {"gguf"}:
        build_title = f"GGUF build of {source}"
        tags_block = "\n  - gguf\n  - quantized"
        conversion_notes = gguf_notes
    elif declared == {"onnx"}:
        build_title = f"ONNX build of {source}"
        tags_block = "\n  - onnx\n  - quantized"
        conversion_notes = onnx_notes
    else:
        build_title = f"ONNX/GGUF build of {source}"
        tags_block = "\n  - onnx\n  - gguf\n  - quantized"
        conversion_notes = f"{onnx_notes}\n{gguf_notes}"

    return _TEMPLATE.format(
        tags_block=tags_block,
        build_title=build_title,
        source=source,
        pipeline_tag=pipeline_tag,
        usage_class=usage_class,
        model_family=contract.model_family,
        modality=contract.modality,
        pooling=pooling,
        normalization=contract.normalization,
        rows=rows,
        conversion_notes=conversion_notes,
    )


__all__ = ["KINDS", "render_card"]
