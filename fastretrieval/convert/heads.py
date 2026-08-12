"""Export small task-specific heads for converted causal models."""

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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source)
    yes_id, no_id = resolve_yes_no_ids(tokenizer, yes_token, no_token)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()

    hidden_size = getattr(model.config, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("causal model config must expose a positive hidden_size")

    class YesNoHead(torch.nn.Module):
        def __init__(self, base_model: Any, yes_token_id: int, no_token_id: int) -> None:
            super().__init__()
            self.base_model = base_model
            self.yes_token_id = yes_token_id
            self.no_token_id = no_token_id

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden = outputs.last_hidden_state[:, -1, :]
            weight = self.base_model.get_output_embeddings().weight
            selected = weight[
                torch.tensor(
                    [self.no_token_id, self.yes_token_id],
                    device=weight.device,
                ),
                :,
            ]
            return torch.matmul(hidden, selected.transpose(0, 1))

    head = YesNoHead(model, yes_id, no_id)
    encoded = tokenizer(
        "hello world",
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    onnx_path = output_dir / "yes_no_head.onnx"
    with torch.no_grad():
        torch.onnx.export(
            head,
            (input_ids, attention_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=21,
        )

    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    logger.info("Exported yes/no head to {}", onnx_path)
    return onnx_path
