# qwen3-embed (renamed)

**This distribution has been renamed to [`fastretrieval`](https://pypi.org/project/fastretrieval/).**

`qwen3-embed` 1.14.1 is the final release. It contains no runtime of its own: installing it
installs `fastretrieval`, and the `qwen3_embed` module only re-exports the public API that
1.14.0 had (`TextEmbedding`, `TextCrossEncoder`, `CustomModelSpec`, `CustomRerankerSpec`,
`Device`) from `fastretrieval`, with a `DeprecationWarning`. Please switch to:

```bash
pip install fastretrieval        # was: pip install qwen3-embed
pip install fastretrieval[gguf]  # was: pip install qwen3-embed[gguf]
```

```python
from fastretrieval import TextEmbedding, TextCrossEncoder  # was: from qwen3_embed import ...
```

Qwen3 model identifiers are unchanged. Imports of `qwen3_embed` submodules are not
forwarded; import them from `fastretrieval` instead. The `QWEN3_EMBED_*` environment
variables are still read by `fastretrieval` as deprecated aliases.

No further releases of `qwen3-embed` are planned. Source and issues:
https://github.com/n24q02m/fastretrieval
