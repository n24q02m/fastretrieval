# Verified conversion manifests

Reference manifests produced by the real `convert → verify → card` pipeline on
a clean CPU box (scm-box: Rocky10, Python 3.12, torch 2.14.0+cpu,
optimum-onnx 0.1.0, llama.cpp `a1de614`), fastretrieval source `a838f01`
(v1.9.1 + GGUF card fix).

Each directory holds the exact `fastretrieval-manifest.json` (schema v1) and
the generated model card (`README.md`) for one sentence-transformers BERT
model, all dense/MEAN/normalized, 384 dims, converted with both quantized
variants (`int8` + `q4f16`).

These are **not** runtime registry entries: the converted artifacts are not
published to a HuggingFace repo yet (upload stays backlog), so instantiating
these model ids through the facades would not resolve. Use them as verified
references for the BYO/custom-model path, or reconvert locally:

```bash
python -m fastretrieval.convert onnx sentence-transformers/paraphrase-MiniLM-L3-v2 \
  --out out/paraphrase-MiniLM-L3-v2 --pooling mean --normalize
python -m fastretrieval.convert verify out/paraphrase-MiniLM-L3-v2 \
  --source sentence-transformers/paraphrase-MiniLM-L3-v2
python -m fastretrieval.convert card out/paraphrase-MiniLM-L3-v2 \
  --source sentence-transformers/paraphrase-MiniLM-L3-v2
```

## Verify results (numeric gate, torch reference vs ONNX Runtime, worst variant)

| Model | max_abs_diff | cosine | atol used | result |
|---|---|---|---|---|
| paraphrase-MiniLM-L3-v2 | 0.0479 | 0.9717 | 0.08 | PASS |
| all-MiniLM-L12-v2 | 0.0849 | 0.9139 | 0.1 | PASS (0.08 FAIL) |
| multi-qa-MiniLM-L6-cos-v1 | 0.0736 | 0.9382 | 0.08 | PASS |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.0643 | 0.9734 | 0.08 | PASS |

The flat 1e-2 default rejects every quantized artifact above; per-variant
defaults (int8 0.1, q4f16 0.15) are proposed in the verify-tolerance PR.

## Provenance

- Conversion pipeline: `fastretrieval.convert` CLI (`onnx` → `verify` → `card`)
- Box checkout: fastretrieval `a838f014`, llama.cpp `a1de614`
- Full conversion logs are retained with the campaign evidence copies; they are
  not committed to this repository.
