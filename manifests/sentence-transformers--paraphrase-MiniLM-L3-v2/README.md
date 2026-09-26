---
tags:
  - onnx
  - quantized
base_model: sentence-transformers/paraphrase-MiniLM-L3-v2
pipeline_tag: feature-extraction
model_family: bert
modality: text
pooling: MEAN
normalization: True
---

# ONNX build of sentence-transformers/paraphrase-MiniLM-L3-v2

Converted with [fastretrieval](https://github.com/n24q02m/fastretrieval).

## Variants

| Variant | File | Size |
|---|---|---|
| int8 | `onnx/model_quantized.onnx` | 17 MB |
| q4f16 | `onnx/model_q4f16.onnx` | 25 MB |

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
[sentence-transformers/paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2) before use.
