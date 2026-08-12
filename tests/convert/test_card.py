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
