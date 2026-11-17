import dataclasses

import pytest

from fastretrieval.contract import ModelContract, PreprocessorSpec
from fastretrieval.convert.card import render_card


def _contract():
    return ModelContract(
        model_id="acme/tiny-model",
        source="acme/tiny-model",
        task="dense",
        modality="text",
        model_family="bert",
        output_dim=384,
        output_shape=(384,),
        pooling="MEAN",
        normalization=True,
        max_seq_len=128,
        preprocessor=PreprocessorSpec(kind="text"),
        artifact_formats=("onnx",),
        tokenizer_files=("tokenizer.json",),
    )


def test_card_lists_every_variant():
    card = render_card(
        "acme/tiny-model",
        {"int8": 120.5, "q4f16": 64.25},
        kind="embedding",
        contract=_contract(),
    )
    assert "model_quantized.onnx" in card
    assert "model_q4f16.onnx" in card
    assert "120" in card and "64" in card


def test_card_declares_the_base_model():
    card = render_card(
        "acme/tiny-model",
        {"int8": 1.0},
        kind="embedding",
        contract=_contract(),
    )
    assert "base_model: acme/tiny-model" in card


def test_card_does_not_hardcode_a_license():
    card = render_card(
        "acme/tiny-model",
        {"int8": 1.0},
        kind="embedding",
        contract=_contract(),
    )
    assert "license:" not in card


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        render_card(
            "acme/tiny-model",
            {"int8": 1.0},
            kind="translation",
            contract=_contract(),
        )


def _gguf_contract():
    contract = _contract()
    return dataclasses.replace(contract, artifact_formats=("gguf",), quantization="Q4_K_M")


def test_card_supports_gguf_artifacts():
    card = render_card(
        "acme/tiny-model",
        {"tiny-model-q4-k-m.gguf": 412.4},
        kind="embedding",
        contract=_gguf_contract(),
    )
    assert "# GGUF build of acme/tiny-model" in card
    assert "`tiny-model-q4-k-m.gguf`" in card
    assert "412" in card
    assert "`llama-quantize` (Q4_K_M)" in card
    assert "  - gguf" in card


def test_card_gguf_requires_a_size_row():
    with pytest.raises(ValueError, match="at least one artifact"):
        render_card("acme/tiny-model", {}, kind="embedding", contract=_gguf_contract())


def test_card_mixed_formats_lists_both():
    contract = _contract()
    mixed = dataclasses.replace(contract, artifact_formats=("onnx", "gguf"), quantization="Q4_K_M")
    card = render_card(
        "acme/tiny-model",
        {"int8": 120.5, "acme-tiny-model-q4-k-m.gguf": 412.4},
        kind="embedding",
        contract=mixed,
    )
    assert "# ONNX/GGUF build of acme/tiny-model" in card
    assert "model_quantized.onnx" in card
    assert "acme-tiny-model-q4-k-m.gguf" in card


def test_card_rejects_unknown_variant_labels():
    with pytest.raises(ValueError, match="unknown artifact variant"):
        render_card("acme/tiny-model", {"bogus": 1.0}, kind="embedding", contract=_contract())


def test_card_requires_the_converted_manifest_contract():
    with pytest.raises(ValueError, match="converted artifact manifest is required"):
        render_card("acme/tiny-model", {"int8": 1.0}, kind="embedding")


def test_card_rejects_a_contract_without_onnx_or_gguf_formats():
    contract = _contract()
    # ModelContract validation forbids other formats, so the card's own guard is reached
    # only by bypassing the frozen dataclass; it must still fail closed.
    object.__setattr__(contract, "artifact_formats", ("safetensors",))

    with pytest.raises(ValueError, match="model cards support ONNX/GGUF artifacts only"):
        render_card("acme/tiny-model", {"int8": 1.0}, kind="embedding", contract=contract)


def test_onnx_card_documents_the_onnx_pipeline_and_contract_metadata():
    card = render_card(
        "acme/tiny-model",
        {"q4f16": 64.25, "int8": 120.5},
        kind="embedding",
        contract=_contract(),
    )

    assert "# ONNX build of acme/tiny-model" in card
    assert "tags:\n  - onnx\n  - quantized\n" in card
    assert "  - gguf" not in card
    assert "- ONNX opset 21" in card
    assert "llama-quantize" not in card
    assert "pipeline_tag: feature-extraction" in card
    assert "model = TextEmbedding(" in card
    assert "pooling: MEAN\nnormalization: True\n" in card
    assert "model_family: bert\nmodality: text\n" in card
    rows = [line for line in card.splitlines() if line.startswith(("| int8", "| q4f16"))]
    assert rows == [
        "| int8 | `onnx/model_quantized.onnx` | 120 MB |",
        "| q4f16 | `onnx/model_q4f16.onnx` | 64 MB |",
    ]


def test_reranker_card_uses_the_cross_encoder_usage_and_pipeline_tag():
    card = render_card("acme/tiny-model", {"int8": 1.0}, kind="reranker", contract=_contract())

    assert "pipeline_tag: text-classification" in card
    assert "from fastretrieval import TextCrossEncoder" in card
    assert "model = TextCrossEncoder(" in card


def test_gguf_card_without_quantization_omits_the_quant_suffix():
    contract = _contract()
    unquantized = dataclasses.replace(contract, artifact_formats=("gguf",))

    card = render_card(
        "acme/tiny-model", {"tiny-model-f16.gguf": 2.0}, kind="embedding", contract=unquantized
    )

    assert "convert_hf_to_gguf.py` (F16), then `llama-quantize`\n" in card
