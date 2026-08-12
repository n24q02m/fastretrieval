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
tags:
  - onnx
  - quantized
base_model: {source}
pipeline_tag: {pipeline_tag}
model_family: {model_family}
modality: {modality}
pooling: {pooling}
normalization: {normalization}
---

# ONNX build of {source}

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

- ONNX opset 21
- INT8: `onnxruntime.quantization.quantize_dynamic` (QInt8)
- Q4F16: `MatMulNBitsQuantizer` (block_size 128, symmetric) then a float16 cast

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
    if "onnx" not in contract.artifact_formats:
        raise ValueError(f"{contract.model_id}: model cards currently support ONNX artifacts only")
    unknown_variants = sorted(set(sizes) - set(_FILENAMES))
    if unknown_variants:
        raise ValueError(f"unknown ONNX variant(s): {', '.join(unknown_variants)}")
    if not sizes:
        raise ValueError(f"{contract.model_id}: at least one ONNX variant is required")
    pipeline_tag, usage_class = KINDS[kind]
    rows = "\n".join(
        f"| {name} | `{_FILENAMES.get(name, name)}` | {megabytes:.0f} MB |"
        for name, megabytes in sorted(sizes.items())
    )
    if not rows:
        rows = "| - | - | - |"
    pooling = getattr(contract.pooling, "value", contract.pooling)
    return _TEMPLATE.format(
        source=source,
        pipeline_tag=pipeline_tag,
        usage_class=usage_class,
        model_family=contract.model_family,
        modality=contract.modality,
        pooling=pooling,
        normalization=contract.normalization,
        rows=rows,
    )


__all__ = ["KINDS", "render_card"]
