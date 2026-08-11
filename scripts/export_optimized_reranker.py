"""Export optimized Qwen3 Reranker ONNX model with 2-dim yes/no output.

Instead of outputting full vocab logits (batch, seq_len, 151669), the optimized
model outputs only (batch, 2) — [no_logit, yes_logit] at the last token position.

This reduces runtime memory from ~12GB to <1GB.

Usage:
    python scripts/export_optimized_reranker.py \
        --model-id Qwen/Qwen3-Reranker-0.6B \
        --output-dir ./optimized_reranker
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForCausalLM, AutoTokenizer

# Token IDs in the Qwen3 tokenizer vocabulary
TOKEN_YES_ID = 9693
TOKEN_NO_ID = 2152


class Qwen3RerankerYesNo(nn.Module):
    """Wrapper that extracts only yes/no logits from the last token."""

    def __init__(self, base_model: AutoModelForCausalLM):
        super().__init__()
        self.model = base_model.model  # Qwen3Model (transformer backbone)

        # Extract only the 2 relevant rows from lm_head
        lm_head_weight = base_model.lm_head.weight.data  # (vocab_size, hidden_size)
        self.yes_no_head = nn.Linear(lm_head_weight.shape[1], 2, bias=False)
        # Row order: [no, yes] to match _compute_yes_no_scores stack order
        self.yes_no_head.weight.data = lm_head_weight[[TOKEN_NO_ID, TOKEN_YES_ID], :]

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.FloatTensor:
        # Run transformer backbone
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Get last token hidden state
        last_hidden = hidden_states[:, -1, :]  # (batch, hidden_size)

        # Project to 2-dim yes/no logits
        logits = self.yes_no_head(last_hidden)  # (batch, 2)
        return logits


def verify_token_ids(tokenizer: AutoTokenizer) -> None:
    """Verify that 'yes' and 'no' tokens match the expected IDs."""
    yes_token = tokenizer.encode("yes", add_special_tokens=False)
    no_token = tokenizer.encode("no", add_special_tokens=False)
    print(f"Tokenizer 'yes' -> {yes_token}, 'no' -> {no_token}")
    assert TOKEN_YES_ID in yes_token, (
        f"TOKEN_YES_ID mismatch: expected {TOKEN_YES_ID} in {yes_token}"
    )
    assert TOKEN_NO_ID in no_token, f"TOKEN_NO_ID mismatch: expected {TOKEN_NO_ID} in {no_token}"


def verify_optimized_model_pytorch(
    tokenizer: AutoTokenizer, model: nn.Module, base_model: AutoModelForCausalLM
) -> tuple[dict, torch.Tensor]:
    """Run a quick sanity check and compare with original model."""
    dummy_text = (
        "<|im_start|>system\nJudge<|im_end|>\n"
        "<|im_start|>user\ntest<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    dummy_tokens = tokenizer(dummy_text, return_tensors="pt")
    with torch.no_grad():
        test_output = model(dummy_tokens["input_ids"], dummy_tokens["attention_mask"])
    print(f"Test output shape: {test_output.shape}")  # Should be (1, 2)
    assert test_output.shape == (1, 2), f"Unexpected output shape: {test_output.shape}"

    # Compare with original model
    with torch.no_grad():
        orig_output = base_model(
            input_ids=dummy_tokens["input_ids"],
            attention_mask=dummy_tokens["attention_mask"],
        )
        orig_logits = orig_output.logits[:, -1, :]  # (1, vocab_size)
        orig_yes_no = torch.stack(
            [orig_logits[:, TOKEN_NO_ID], orig_logits[:, TOKEN_YES_ID]], dim=1
        )  # (1, 2)

    diff = (test_output - orig_yes_no).abs().max().item()
    print(f"Max absolute difference vs original: {diff:.2e}")
    assert diff < 1e-4, f"Output mismatch too large: {diff}"

    return dummy_tokens, test_output


def export_to_onnx(model: nn.Module, dummy_tokens: dict, onnx_fp32_path: Path, opset: int) -> None:
    """Export the optimized model to ONNX format."""
    print(f"Exporting ONNX to {onnx_fp32_path}...")

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        (dummy_tokens["input_ids"], dummy_tokens["attention_mask"]),
        str(onnx_fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # Verify ONNX model (use path for >2GB models)
    onnx.checker.check_model(str(onnx_fp32_path))
    print(f"ONNX model verified. Size: {onnx_fp32_path.stat().st_size / 1e6:.1f} MB")


def quantize_onnx_model(onnx_fp32_path: Path, onnx_int8_path: Path) -> None:
    """Quantize the exported ONNX model to INT8."""
    print(f"Quantizing to INT8: {onnx_int8_path}...")
    quantize_dynamic(
        str(onnx_fp32_path),
        str(onnx_int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model size: {onnx_int8_path.stat().st_size / 1e6:.1f} MB")


def verify_onnxruntime(
    onnx_int8_path: Path, dummy_tokens: dict, test_output: torch.Tensor
) -> None:
    """Verify the quantized model with ONNXRuntime."""
    print("Verifying with onnxruntime...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_int8_path))
    ort_inputs = {
        "input_ids": dummy_tokens["input_ids"].numpy(),
        "attention_mask": dummy_tokens["attention_mask"].numpy(),
    }
    ort_output = session.run(None, ort_inputs)
    print(f"ORT output shape: {np.array(ort_output[0]).shape}")
    print(f"ORT output: {ort_output[0]}")

    # Verify numerical consistency
    ort_diff = np.abs(ort_output[0] - test_output.numpy()).max()
    print(f"ORT vs PyTorch max diff: {ort_diff:.2e}")


def export_model(model_id: str, output_dir: str, opset: int = 17) -> None:
    """Export the optimized model and tokenizer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    verify_token_ids(tokenizer)

    print("Creating optimized yes/no model...")
    model = Qwen3RerankerYesNo(base_model)
    model.eval()

    dummy_tokens, test_output = verify_optimized_model_pytorch(tokenizer, model, base_model)

    onnx_fp32_path = output_path / "model.onnx"
    export_to_onnx(model, dummy_tokens, onnx_fp32_path, opset)

    onnx_int8_path = output_path / "model_quantized.onnx"
    quantize_onnx_model(onnx_fp32_path, onnx_int8_path)

    print("Saving tokenizer...")
    tokenizer.save_pretrained(str(output_path))

    verify_onnxruntime(onnx_int8_path, dummy_tokens, test_output)

    print(f"\nExport complete! Files in {output_path}:")
    for f in sorted(output_path.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export optimized Qwen3 Reranker ONNX")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-Reranker-0.6B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        default="./optimized_reranker",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    args = parser.parse_args()
    export_model(args.model_id, args.output_dir, args.opset)
