"""Built-in registry for the fastretrieval-hosted sentence-transformers ONNX models.

The five ``n24q02m/fr-*`` Hugging Face repos are produced by the converter in
``fastretrieval.convert`` (int8 + q4f16 variants, mean pooling, L2
normalized). Each description points the downloader at the int8 artifact so a
default install can resolve the short alias without a local path.
"""

from fastretrieval.common.model_description import DenseModelDescription, ModelSource

FR_MODELS: list[DenseModelDescription] = [
    DenseModelDescription(
        model="n24q02m/fr-all-MiniLM-L6-v2",
        dim=384,
        description=(
            "MiniLM-L6-v2 sentence transformer (384 dims), mean pooling with L2 "
            "normalization, INT8 dynamic quantized ONNX."
        ),
        license="apache-2.0",
        size_in_GB=0.023,
        sources=ModelSource(hf="n24q02m/fr-all-MiniLM-L6-v2"),
        model_file="onnx/model_quantized.onnx",
    ),
    DenseModelDescription(
        model="n24q02m/fr-all-MiniLM-L12-v2",
        dim=384,
        description=(
            "MiniLM-L12-v2 sentence transformer (384 dims), mean pooling with L2 "
            "normalization, INT8 dynamic quantized ONNX."
        ),
        license="apache-2.0",
        size_in_GB=0.034,
        sources=ModelSource(hf="n24q02m/fr-all-MiniLM-L12-v2"),
        model_file="onnx/model_quantized.onnx",
    ),
    DenseModelDescription(
        model="n24q02m/fr-multi-qa-MiniLM-L6-cos-v1",
        dim=384,
        description=(
            "multi-qa-MiniLM-L6-cos-v1 sentence transformer (384 dims), mean pooling "
            "with L2 normalization, INT8 dynamic quantized ONNX."
        ),
        license="apache-2.0",
        size_in_GB=0.023,
        sources=ModelSource(hf="n24q02m/fr-multi-qa-MiniLM-L6-cos-v1"),
        model_file="onnx/model_quantized.onnx",
    ),
    DenseModelDescription(
        model="n24q02m/fr-paraphrase-MiniLM-L3-v2",
        dim=384,
        description=(
            "paraphrase-MiniLM-L3-v2 sentence transformer (384 dims), mean pooling "
            "with L2 normalization, INT8 dynamic quantized ONNX."
        ),
        license="apache-2.0",
        size_in_GB=0.017,
        sources=ModelSource(hf="n24q02m/fr-paraphrase-MiniLM-L3-v2"),
        model_file="onnx/model_quantized.onnx",
    ),
    DenseModelDescription(
        model="n24q02m/fr-paraphrase-multilingual-MiniLM-L12-v2",
        dim=384,
        description=(
            "paraphrase-multilingual-MiniLM-L12-v2 sentence transformer (384 dims), "
            "mean pooling with L2 normalization, INT8 dynamic quantized ONNX."
        ),
        license="apache-2.0",
        size_in_GB=0.118,
        sources=ModelSource(hf="n24q02m/fr-paraphrase-multilingual-MiniLM-L12-v2"),
        model_file="onnx/model_quantized.onnx",
    ),
]
