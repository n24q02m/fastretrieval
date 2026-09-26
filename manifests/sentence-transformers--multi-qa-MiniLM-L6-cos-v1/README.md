---
tags:
  - onnx
  - quantized
base_model: sentence-transformers/multi-qa-MiniLM-L6-cos-v1
pipeline_tag: feature-extraction
model_family: bert
modality: text
pooling: MEAN
normalization: True
---

# ONNX build of sentence-transformers/multi-qa-MiniLM-L6-cos-v1

Converted with [fastretrieval](https://github.com/n24q02m/fastretrieval).

## Variants

| Variant | File | Size |
|---|---|---|
| int8 | `onnx/model_quantized.onnx` | 22 MB |
| q4f16 | `onnx/model_q4f16.onnx` | 28 MB |

## Usage

```python
from fastretrieval import TextEmbedding

model = TextEmbedding("<this-repo-id>")
```

## Conversion

- ONNX opset 21
- INT8: `onnxruntime.quantization.quantize_dynamic` (QInt8)
- Q4F16: `MatMulNBitsQuantizer` (block_size 128, symmetric) then a float16 cast

## License

This build inherits the license of the base model. Check
[sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) before use.
