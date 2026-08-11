"""Integration tests for Qwen3 ONNX-Q4F16 quantized models (embed + rerank).

These tests download ONNX-Q4F16 models from HuggingFace Hub and run real
inference.  Marked with ``pytest.mark.integration`` so they can be skipped::

    pytest -m "not integration"      # skip heavy tests
    pytest -m integration            # run only integration tests
"""

import numpy as np
import pytest

from qwen3_embed import TextCrossEncoder, TextEmbedding

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16"
RERANKER_MODEL = "n24q02m/Qwen3-Reranker-0.6B-ONNX-Q4F16"


@pytest.fixture(scope="module")
def embedding_model():
    return TextEmbedding(model_name=EMBEDDING_MODEL)


@pytest.fixture(scope="module")
def reranker_model():
    return TextCrossEncoder(model_name=RERANKER_MODEL)


# ═══════════════════════════════════════════════════════════════════════════
# Embedding: Basic operations
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestQ4F16EmbeddingBasic:
    """Core embedding functionality with Q4F16 model."""

    def test_single_document(self, embedding_model):
        embeddings = list(embedding_model.embed("Hello world"))
        assert len(embeddings) == 1
        assert embeddings[0].shape == (1024,)

    def test_multiple_documents(self, embedding_model):
        docs = ["First document.", "Second document.", "Third document."]
        embeddings = list(embedding_model.embed(docs))
        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb.shape == (1024,)

    def test_embeddings_are_normalized(self, embedding_model):
        embeddings = list(embedding_model.embed("Normalization check"))
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-3, f"Expected unit norm, got {norm}"

    def test_deterministic_output(self, embedding_model):
        text = "Deterministic test input"
        emb1 = list(embedding_model.embed(text))[0]
        emb2 = list(embedding_model.embed(text))[0]
        np.testing.assert_array_equal(emb1, emb2)


# ═══════════════════════════════════════════════════════════════════════════
# Embedding: MRL dimensions
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestQ4F16EmbeddingMRL:
    """MRL dimension truncation with Q4F16 model."""

    @pytest.mark.parametrize("dim", [32, 128, 512, 1024])
    def test_mrl_dimensions(self, embedding_model, dim):
        embeddings = list(embedding_model.embed("Test MRL dimensions", dim=dim))
        assert embeddings[0].shape == (dim,)

    @pytest.mark.parametrize("dim", [32, 128, 512, 1024])
    def test_mrl_normalized(self, embedding_model, dim):
        embeddings = list(embedding_model.embed("Normalization check", dim=dim))
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-3


# ═══════════════════════════════════════════════════════════════════════════
# Embedding: Semantic quality
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestQ4F16EmbeddingSemanticQuality:
    """Verify Q4F16 model produces semantically meaningful embeddings."""

    def test_similar_texts_closer(self, embedding_model):
        texts = [
            "The cat sat on the mat.",
            "A feline rested on the rug.",
            "The stock market crashed yesterday.",
        ]
        embeddings = list(embedding_model.embed(texts))
        sim_01 = np.dot(embeddings[0], embeddings[1])
        sim_02 = np.dot(embeddings[0], embeddings[2])
        assert sim_01 > sim_02, "Similar texts should be closer"

    def test_query_embed_relevance(self, embedding_model):
        query_emb = list(embedding_model.query_embed("What is machine learning?"))[0]
        doc_emb = list(embedding_model.embed(["Machine learning is a subset of AI."]))[0]
        irr_emb = list(embedding_model.embed(["Recipe for chocolate cake."]))[0]
        assert np.dot(query_emb, doc_emb) > np.dot(query_emb, irr_emb)


# ═══════════════════════════════════════════════════════════════════════════
# Reranker: Basic operations
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestQ4F16RerankerBasic:
    """Core reranker functionality with Q4F16 model."""

    def test_rerank_returns_scores(self, reranker_model):
        scores = list(reranker_model.rerank("What is AI?", ["AI is artificial intelligence."]))
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_rerank_multiple_docs(self, reranker_model):
        query = "What is deep learning?"
        docs = [
            "Deep learning uses neural networks with many layers.",
            "The weather forecast says rain tomorrow.",
        ]
        scores = list(reranker_model.rerank(query, docs))
        assert len(scores) == 2
        assert scores[0] > scores[1], "Relevant doc should score higher"

    def test_rerank_pairs(self, reranker_model):
        pairs = [
            ("What is Python?", "Python is a high-level programming language."),
            ("What is Python?", "The Sahara Desert is very hot."),
        ]
        scores = list(reranker_model.rerank_pairs(pairs))
        assert len(scores) == 2
        assert scores[0] > scores[1]


# ═══════════════════════════════════════════════════════════════════════════
# Reranker: Quality
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestQ4F16RerankerQuality:
    """Semantic quality checks for Q4F16 reranker."""

    def test_topic_relevance_ordering(self, reranker_model):
        query = "How does photosynthesis work?"
        docs = [
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "Plants need water and CO2 for growth.",
            "The stock exchange closes at 4 PM.",
        ]
        scores = list(reranker_model.rerank(query, docs))
        assert np.argmax(scores) == 0
        assert scores[0] > scores[2]


# ═══════════════════════════════════════════════════════════════════════════
# Cross-cutting: Embed + Rerank pipeline
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestQ4F16Pipeline:
    """End-to-end retrieval pipeline with Q4F16 models."""

    def test_full_pipeline(self, embedding_model, reranker_model):
        corpus = [
            "Python is a programming language known for its simplicity.",
            "JavaScript runs in web browsers and on servers via Node.js.",
            "Rust provides memory safety without garbage collection.",
            "The Eiffel Tower is located in Paris, France.",
        ]
        query = "Which programming language is best for beginners?"

        query_emb = list(embedding_model.query_embed(query))[0]
        doc_embs = list(embedding_model.embed(corpus))
        sims = [np.dot(query_emb, d) for d in doc_embs]
        top2_indices = np.argsort(sims)[-2:][::-1]
        top2_docs = [corpus[i] for i in top2_indices]

        for idx in top2_indices:
            assert idx < 3, f"Non-programming doc (idx={idx}) in top-2 retrieval"

        rerank_scores = list(reranker_model.rerank(query, top2_docs))
        assert len(rerank_scores) == 2
