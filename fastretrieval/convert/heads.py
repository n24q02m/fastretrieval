"""Reduce causal-LM rerankers to two yes/no logits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from fastretrieval.convert import require_convert_deps


def resolve_yes_no_ids(
    tokenizer: Any,
    yes_token: str,
    no_token: str,
) -> tuple[int, int]:
    """Resolve distinct, known tokenizer ids for the yes/no output head."""

    unknown_id = getattr(tokenizer, "unk_token_id", None)
    resolved: dict[str, int] = {}
    for label, token in (("yes", yes_token), ("no", no_token)):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or (unknown_id is not None and token_id == unknown_id):
            raise ValueError(
                f"token {token!r} (the {label} side) is not in this model's "
                f"vocabulary; pass --{label}-token with a token this tokenizer knows"
            )
        resolved[label] = int(token_id)

    if resolved["yes"] == resolved["no"]:
        raise ValueError(
            "yes and no tokens resolve to the same token id; pass distinct "
            "tokens with --yes-token and --no-token"
        )
    return resolved["yes"], resolved["no"]


def export_yes_no_head(
    source: str,
    out_dir: str | Path,
    *,
    yes_token: str = "yes",
    no_token: str = "no",
) -> Path:
    """Export a two-logit yes/no head over the final causal hidden state."""

    require_convert_deps("torch", "transformers")

    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(out_dir)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source)
    yes_id, no_id = resolve_yes_no_ids(tokenizer, yes_token, no_token)
    logger.info("yes token id {}, no token id {}", yes_id, no_id)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.config.use_cache = False
    model.eval()

    class _YesNoHead(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.body = inner.model
            weight = inner.lm_head.weight.data
            self.head = nn.Linear(weight.shape[1], 2, bias=False)
            self.head.weight.data = weight[[no_id, yes_id], :]

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            out = self.body(input_ids=input_ids, attention_mask=attention_mask)
            return self.head(out.last_hidden_state[:, -1, :])

    wrapper = _YesNoHead(model).eval()
    dummy = tokenizer("hello world", return_tensors="pt")
    onnx_path = onnx_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=21,
            do_constant_folding=True,
        )

    tokenizer.save_pretrained(str(output_dir))
    model.config.save_pretrained(str(output_dir))
    return onnx_path
