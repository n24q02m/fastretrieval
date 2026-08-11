"""Optional HuggingFace-id to ONNX export.

The runtime package stays onnxruntime-only; the heavy export dependencies
(torch, transformers, optimum) are imported lazily so importing this module does
not pull them in. They are NOT declared as an optional extra because the lib pins
``tokenizers``/``huggingface-hub`` versions that current ``transformers`` releases
cannot co-resolve (see pyproject #691) — installing them as an extra would make
the universal lock unsatisfiable. Install them yourself in a throwaway env::

    pip install "optimum[exporters]" torch transformers onnx
"""

from pathlib import Path

_MISSING_EXPORT_DEPS = (
    "ONNX export needs optimum + torch + transformers. Install them in a separate "
    'env: pip install "optimum[exporters]" torch transformers onnx'
)


def export_to_onnx(model_id: str, output_dir: str, *, task: str = "feature-extraction") -> str:
    """Export an HF model + tokenizer to ONNX under ``output_dir``.

    Args:
        model_id: HuggingFace model id (e.g. ``"intfloat/multilingual-e5-base"``).
        output_dir: Directory to write ``onnx/model.onnx`` + tokenizer files.
        task: optimum export task; ``"feature-extraction"`` for embeddings.

    Returns:
        The ``output_dir`` path containing the exported model.

    Raises:
        ImportError: if the ``[export]`` extra is not installed.
    """
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as e:
        raise ImportError(_MISSING_EXPORT_DEPS) from e

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    main_export(model_id, output=output_dir, task=task)
    return output_dir
