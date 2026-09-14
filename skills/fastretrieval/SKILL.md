---
name: fastretrieval
description: Dùng fastretrieval (thư viện Python) cho embedding/rerank local qua ONNX/GGUF — TextEmbedding, SparseEmbedding, LateInteraction, TextCrossEncoder — khi cần retrieval hạng-1 trong pipeline mà không cần MCP server.
---

# fastretrieval — embedding/rerank runtime

Thư viện hạng-1; không có CLI/server — import trực tiếp trong script/hook/pipeline.

## API chính
- `TextEmbedding(model_name)` — dense text embedding (ONNX/GGUF)
- `SparseEmbedding(model_name)` — sparse embedding
- `LateInteractionTextEmbedding` / `LateInteractionMultimodalEmbedding` — ColBERT/ColPali
- `TextCrossEncoder(model_name)` — rerank cross-encoder
- `CustomModelSpec` / `CustomRerankerSpec` — mô hình theo manifest contract
- `fastretrieval-convert` CLI — convert model theo declarative contract

## Ví dụ
```python
from fastretrieval import TextEmbedding, TextCrossEncoder

model = TextEmbedding("onnx-model")
emb = list(model.embed(["query text"]))
```

## Ghi chú
- Model cache: `define_cache_dir()` tự quản; device qua `Device`.
- MCP server surface không tồn tại ở repo này — đây là library-first member của stack.
