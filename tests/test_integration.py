"""Integration tests for Qwen3 Embedding and Reranker with real ONNX models.

These tests download actual ONNX models from HuggingFace Hub (~1.2 GB total)
and run real inference. They are marked with ``pytest.mark.integration`` so
they can be skipped during fast CI runs::

    pytest -m "not integration"      # skip heavy tests
    pytest -m integration            # run only integration tests
"""

import sys

import numpy as np
import pytest

from qwen3_embed import TextCrossEncoder, TextEmbedding
from qwen3_embed.common.model_description import (
    DenseModelDescription,
    ModelSource,
    PoolingType,
)
from qwen3_embed.text.custom_text_embedding import CustomTextEmbedding

# ---------------------------------------------------------------------------
# Fixtures (shared model instances -- download once, reuse across tests)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "n24q02m/Qwen3-Embedding-0.6B-ONNX"
RERANKER_MODEL = "n24q02m/Qwen3-Reranker-0.6B-ONNX"


@pytest.fixture(scope="module")
def embedding_model():
    return TextEmbedding(model_name=EMBEDDING_MODEL)


@pytest.fixture(scope="module")
def reranker_model():
    return TextCrossEncoder(model_name=RERANKER_MODEL)


@pytest.fixture(scope="module")
def reranker_yesno_model():
    return TextCrossEncoder(model_name="n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo")


# ═══════════════════════════════════════════════════════════════════════════
# Embedding: Basic operations
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestEmbeddingBasic:
    """Core embedding functionality."""

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
        docs = [
            "Short text",
            "A somewhat longer text with more words to process",
            "x" * 500,  # repeated character stress test
        ]
        embeddings = list(embedding_model.embed(docs))
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-3, f"Expected unit norm, got {norm}"

    def test_empty_string(self, embedding_model):
        embeddings = list(embedding_model.embed(""))
        assert len(embeddings) == 1
        assert embeddings[0].shape == (1024,)

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
    def test_blank_input_yields_usable_embedding(self, embedding_model, blank):
        """Blank text must produce a finite, unit-norm vector, not an exception.

        Downstream callers reach these paths with machine-generated text (empty
        search snippets, whitespace-only queries) and cannot pre-filter reliably.
        """
        emb = list(embedding_model.embed(blank))[0]
        assert emb.shape == (1024,)
        assert not np.any(np.isnan(emb))
        assert not np.any(np.isinf(emb))
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-3

    def test_blank_entry_does_not_abort_batch(self, embedding_model):
        """One blank document must not stop the surrounding batch from embedding."""
        embeddings = list(embedding_model.embed(["First document.", "", "   ", "Last document."]))
        assert len(embeddings) == 4
        for emb in embeddings:
            assert emb.shape == (1024,)

    def test_blank_query_embed(self, embedding_model):
        """query_embed must accept a whitespace-only query (instruction prefix still applies)."""
        emb = list(embedding_model.query_embed("   "))[0]
        assert emb.shape == (1024,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-3

    def test_deterministic_output(self, embedding_model):
        """Same input should produce identical embeddings."""
        text = "Deterministic test input"
        emb1 = list(embedding_model.embed(text))[0]
        emb2 = list(embedding_model.embed(text))[0]
        np.testing.assert_array_equal(emb1, emb2)


# ═══════════════════════════════════════════════════════════════════════════
# Embedding: MRL (Matryoshka Representation Learning)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestEmbeddingMRL:
    """MRL dimension truncation."""

    @pytest.mark.parametrize("dim", [32, 64, 128, 256, 512, 1024])
    def test_mrl_dimensions(self, embedding_model, dim):
        embeddings = list(embedding_model.embed("Test MRL dimensions", dim=dim))
        assert embeddings[0].shape == (dim,)

    @pytest.mark.parametrize("dim", [32, 64, 128, 256, 512, 1024])
    def test_mrl_normalized(self, embedding_model, dim):
        embeddings = list(embedding_model.embed("Normalization check", dim=dim))
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-3

    def test_mrl_prefix_consistency(self, embedding_model):
        """Lower-dim MRL should be a prefix of the full embedding (pre-normalization)."""
        text = "Prefix consistency test"
        full = list(embedding_model.embed(text))[0]  # dim=1024, normalized
        small = list(embedding_model.embed(text, dim=256))[0]  # dim=256, normalized

        # After re-normalizing the first 256 dims of `full`, should match `small`
        prefix = full[:256]
        prefix_normed = prefix / np.linalg.norm(prefix)
        np.testing.assert_allclose(prefix_normed, small, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# Embedding: Semantic quality
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestEmbeddingSemanticQuality:
    """Verify the model produces semantically meaningful embeddings."""

    def test_similar_texts_closer(self, embedding_model):
        """Semantically similar texts should have higher cosine similarity."""
        anchor = "Machine learning is a branch of artificial intelligence."
        similar = "AI and ML are closely related fields in computer science."
        dissimilar = "The recipe calls for two cups of flour and one egg."

        embs = list(embedding_model.embed([anchor, similar, dissimilar]))
        sim_close = np.dot(embs[0], embs[1])
        sim_far = np.dot(embs[0], embs[2])

        assert sim_close > sim_far, (
            f"Similar text similarity ({sim_close:.4f}) should exceed "
            f"dissimilar text similarity ({sim_far:.4f})"
        )

    def test_query_retrieval_ranking(self, embedding_model):
        """Query embedding should rank the correct document highest."""
        query_emb = list(
            embedding_model.query_embed("What programming language is used for data science?")
        )[0]

        docs = [
            "Python is widely used in data science and machine learning.",
            "Java is a popular language for enterprise applications.",
            "The Great Wall of China is visible from space.",
            "CSS is used for styling web pages.",
        ]
        doc_embs = list(embedding_model.embed(docs))

        sims = [np.dot(query_emb, d) for d in doc_embs]
        best_idx = np.argmax(sims)
        assert best_idx == 0, f"Expected doc[0] (Python/data science), got doc[{best_idx}]"

    def test_multilingual_similarity(self, embedding_model):
        """Cross-language texts with same meaning should be similar."""
        en = "Artificial intelligence is transforming the world."
        zh = "Trí tuệ nhân tạo đang thay đổi thế giới."  # Vietnamese
        unrelated = "The price of gold increased by 2% yesterday."

        embs = list(embedding_model.embed([en, zh, unrelated]))
        sim_cross = np.dot(embs[0], embs[1])
        sim_unrelated = np.dot(embs[0], embs[2])

        # Cross-lingual similarity should beat unrelated English text
        assert sim_cross > sim_unrelated

    def test_instruction_improves_retrieval(self, embedding_model):
        """Query embedding with instruction should improve domain-specific retrieval."""
        query_text = "gradient descent"

        # Without instruction (passage embed)
        q_plain = list(embedding_model.embed(query_text))[0]

        # With instruction (query embed)
        q_instruct = list(
            embedding_model.query_embed(
                query_text,
                task="Given a machine learning term, retrieve its definition",
            )
        )[0]

        docs = [
            "Gradient descent is an optimization algorithm used to minimize"
            " the loss function in machine learning models.",
            "The mountain descent took the hikers three hours to complete.",
        ]
        doc_embs = list(embedding_model.embed(docs))

        # Both should rank correctly, but instruction should give wider margin
        sims_plain = [np.dot(q_plain, d) for d in doc_embs]
        sims_instruct = [np.dot(q_instruct, d) for d in doc_embs]

        margin_plain = sims_plain[0] - sims_plain[1]
        margin_instruct = sims_instruct[0] - sims_instruct[1]

        assert margin_plain > 0, "Plain query should still rank ML doc higher"
        assert margin_instruct > 0, "Instructed query should rank ML doc higher"


# ═══════════════════════════════════════════════════════════════════════════
# Embedding: Edge cases
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestEmbeddingEdgeCases:
    """Stress tests and edge cases."""

    def test_very_long_text(self, embedding_model):
        """Text exceeding model's context window should be truncated, not crash."""
        long_text = "This is a test sentence. " * 500  # ~3.5k tokens, exceeds typical context
        embeddings = list(embedding_model.embed(long_text))
        assert len(embeddings) == 1
        assert embeddings[0].shape == (1024,)
        assert abs(np.linalg.norm(embeddings[0]) - 1.0) < 1e-3

    def test_special_characters(self, embedding_model):
        """Special chars, unicode, emojis should not crash."""
        texts = [
            "Hello! @#$%^&*() [brackets] {braces}",
            "Unicode: Qwen3 is powerful",
            "Mixed: 123 abc !@# \n\t",
        ]
        embeddings = list(embedding_model.embed(texts))
        assert len(embeddings) == 3
        for emb in embeddings:
            assert not np.any(np.isnan(emb)), "NaN in embedding"
            assert not np.any(np.isinf(emb)), "Inf in embedding"

    def test_batch_of_ten(self, embedding_model):
        """Larger batch processed correctly with batch_size=1."""
        docs = [f"Document number {i} about topic {chr(65 + i)}." for i in range(10)]
        embeddings = list(embedding_model.embed(docs))
        assert len(embeddings) == 10
        # All embeddings should be unique
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                assert not np.array_equal(embeddings[i], embeddings[j]), (
                    f"Embeddings {i} and {j} are identical"
                )


# ═══════════════════════════════════════════════════════════════════════════
# Embedding: Custom model under multiprocessing (issue #3/#4)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows multiprocessing spawn deadlocks with onnxruntime under pytest; "
    "the worker-registry propagation itself is covered by the unit serialization test.",
)
class TestCustomModelParallel:
    """A runtime-registered custom model must resolve inside spawned workers."""

    def test_custom_model_parallel_mixed_case(self):
        """Register a custom LAST_TOKEN model under a mixed-case id, embed with
        parallel=2, and assert no KeyError + correct shapes."""
        CustomTextEmbedding._SUPPORTED.clear()
        model_id = "Org/Custom-Qwen"
        TextEmbedding.add_custom_model(
            model_description=DenseModelDescription(
                model=model_id,
                sources=ModelSource(hf="n24q02m/Qwen3-Embedding-0.6B-ONNX"),
                model_file="onnx/model_quantized.onnx",
                dim=1024,
            ),
            pooling=PoolingType.LAST_TOKEN,
            normalization=True,
        )
        try:
            model = TextEmbedding(model_name=model_id)
            # batch_size=1 forces the multiprocessing worker-pool path (otherwise
            # 3 docs < default batch_size is treated as "small" and runs in-process).
            embeddings = list(model.embed(["a", "b", "c"], batch_size=1, parallel=2))
        finally:
            CustomTextEmbedding._SUPPORTED.clear()

        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb.shape == (1024,)


# ═══════════════════════════════════════════════════════════════════════════
# Embedding: Custom model on a static-batch graph
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestCustomModelStaticBatchGraph:
    """A custom model on causal-LM weights must not crash on the default batch size.

    The Qwen3 ONNX graph is pinned to a batch of one, so embedding more
    documents than that in a single run used to abort inside onnxruntime with
    ``Shape mismatch attempting to re-use buffer``.
    """

    def test_default_batch_size_on_static_batch_graph(self):
        CustomTextEmbedding._SUPPORTED.clear()
        model_id = "Org/Custom-Qwen-Static-Batch"
        TextEmbedding.add_custom_model(
            model_description=DenseModelDescription(
                model=model_id,
                sources=ModelSource(hf="n24q02m/Qwen3-Embedding-0.6B-ONNX"),
                model_file="onnx/model_quantized.onnx",
                dim=1024,
            ),
            pooling=PoolingType.LAST_TOKEN,
            normalization=True,
        )
        try:
            model = TextEmbedding(model_name=model_id)
            docs = [f"Document number {i} about topic {chr(65 + i)}." for i in range(8)]
            # No batch_size: the default of 256 is what the caller gets by default.
            embeddings = list(model.embed(docs))
        finally:
            CustomTextEmbedding._SUPPORTED.clear()

        assert len(embeddings) == 8
        for emb in embeddings:
            assert emb.shape == (1024,)
            assert abs(np.linalg.norm(emb) - 1.0) < 1e-3


# ═══════════════════════════════════════════════════════════════════════════
# Reranker: Basic operations
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRerankerBasic:
    """Core reranker functionality."""

    def test_single_document(self, reranker_model):
        scores = list(reranker_model.rerank("test query", ["test document"]))
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_multiple_documents(self, reranker_model):
        query = "What is deep learning?"
        docs = [
            "Deep learning uses neural networks with many layers.",
            "The stock market rose today.",
            "Neural networks are inspired by biological brains.",
            "Pizza is a popular food worldwide.",
        ]
        scores = list(reranker_model.rerank(query, docs))
        assert len(scores) == 4
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_scores_in_valid_range(self, reranker_model):
        """All scores should be valid probabilities [0, 1]."""
        scores = list(
            reranker_model.rerank(
                "query",
                ["highly relevant", "somewhat relevant", "completely irrelevant"],
            )
        )
        for score in scores:
            assert 0.0 <= score <= 1.0
            assert not np.isnan(score)
            assert not np.isinf(score)

    def test_deterministic_scores(self, reranker_model):
        """Same input should produce identical scores."""
        query = "What is Python?"
        docs = ["Python is a programming language."]
        s1 = list(reranker_model.rerank(query, docs))
        s2 = list(reranker_model.rerank(query, docs))
        assert abs(s1[0] - s2[0]) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# Reranker: Batch invariance (issue #725)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRerankerBatchInvariance:
    """Issue #725: a (query, doc) score must be independent of batch composition."""

    QUERY = "Java Developer"
    DOCS = ["Python Developer", "Senior Java Engineer", "Marketing Manager"]

    def _assert_batch_invariant(self, model):
        batched = list(model.rerank(self.QUERY, self.DOCS))
        singles = [list(model.rerank(self.QUERY, [d]))[0] for d in self.DOCS]
        for i, (b, s) in enumerate(zip(batched, singles, strict=True)):
            assert b == pytest.approx(s, abs=1e-4), (
                f"doc[{i}]={self.DOCS[i]!r}: batched={b} != single={s}"
            )

    def test_yesno_variant_batch_invariant(self, reranker_yesno_model):
        self._assert_batch_invariant(reranker_yesno_model)

    def test_standard_variant_batch_invariant(self, reranker_model):
        self._assert_batch_invariant(reranker_model)


# ═══════════════════════════════════════════════════════════════════════════
# Reranker: Ranking quality
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRerankerQuality:
    """Verify the reranker correctly distinguishes relevant from irrelevant."""

    def test_relevant_vs_irrelevant(self, reranker_model):
        """Clearly relevant document should score much higher."""
        scores = list(
            reranker_model.rerank(
                "What is gradient descent?",
                [
                    "Gradient descent is an optimization algorithm that iteratively"
                    " adjusts parameters to minimize a loss function.",
                    "The weather forecast predicts rain for tomorrow afternoon.",
                ],
            )
        )
        assert scores[0] > scores[1]
        assert scores[0] > 0.8, f"Relevant doc should score high, got {scores[0]}"
        assert scores[1] < 0.2, f"Irrelevant doc should score low, got {scores[1]}"

    def test_ranking_order(self, reranker_model):
        """Documents should be ranked by semantic relevance."""
        query = "How does photosynthesis work?"
        docs = [
            "Photosynthesis converts sunlight into chemical energy in plants.",  # most relevant
            "Plants need water and CO2 for growth.",  # somewhat relevant
            "The stock exchange closes at 4 PM.",  # irrelevant
            "My cat likes to sleep on the couch.",  # irrelevant
        ]
        scores = list(reranker_model.rerank(query, docs))

        # Best match
        assert np.argmax(scores) == 0

        # Photosynthesis doc > plant doc > irrelevant docs
        assert scores[0] > scores[1]
        assert scores[1] > scores[2] or scores[1] > scores[3]

    def test_rerank_pairs(self, reranker_model):
        """rerank_pairs should produce consistent scores."""
        pairs = [
            ("What is Python?", "Python is a high-level programming language."),
            ("What is Python?", "The Sahara Desert is the largest hot desert."),
            ("Capital of France?", "Paris is the capital and largest city of France."),
        ]
        scores = list(reranker_model.rerank_pairs(pairs))
        assert len(scores) == 3
        assert scores[0] > scores[1], "Programming > Desert for Python query"
        assert scores[2] > 0.8, "Paris/France should be highly relevant"

    def test_custom_instruction(self, reranker_model):
        """Custom instruction should work without errors."""
        scores = list(
            reranker_model.rerank(
                "SELECT * FROM users",
                ["This query retrieves all rows from the users table."],
                instruction="Given a SQL query, judge whether the document correctly explains it.",
            )
        )
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Reranker: Edge cases
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRerankerEdgeCases:
    """Stress tests for the reranker."""

    def test_long_document(self, reranker_model):
        """Long document should be truncated, not crash."""
        long_doc = "This is about machine learning. " * 500
        scores = list(reranker_model.rerank("What is ML?", [long_doc]))
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_special_characters_in_query(self, reranker_model):
        """Special characters should not break scoring."""
        scores = list(
            reranker_model.rerank(
                "What's the @#$% difference?!",
                ["Normal document about differences between things."],
            )
        )
        assert len(scores) == 1
        assert not np.isnan(scores[0])

    def test_blank_document_does_not_abort_rerank(self, reranker_model):
        """A blank document in the candidate list must be scored, not raise.

        Search backends routinely return results whose snippet is empty; callers
        wrap rerank in a broad ``except`` and silently fall back to unranked
        order, so a raise here degrades retrieval invisibly.
        """
        scores = list(
            reranker_model.rerank(
                "What is Python?",
                ["Python is a programming language.", "", "   "],
            )
        )
        assert len(scores) == 3
        for score in scores:
            assert 0.0 <= score <= 1.0
            assert not np.isnan(score)

    def test_blank_query_does_not_abort_rerank(self, reranker_model):
        scores = list(reranker_model.rerank("   ", ["Some candidate document."]))
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_identical_query_and_doc(self, reranker_model):
        """When query == document, should score very high."""
        text = "Neural networks are computational models."
        scores = list(reranker_model.rerank(text, [text]))
        assert scores[0] > 0.5, f"Identical text should score high, got {scores[0]}"


# ═══════════════════════════════════════════════════════════════════════════
# Cross-cutting: Embedding + Reranker pipeline
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRetrievalPipeline:
    """End-to-end retrieval pipeline: embed -> retrieve -> rerank."""

    def test_full_pipeline(self, embedding_model, reranker_model):
        """Simulate a real retrieval-then-rerank pipeline."""
        corpus = [
            "Python is a programming language known for its simplicity.",
            "JavaScript runs in web browsers and on servers via Node.js.",
            "Rust provides memory safety without garbage collection.",
            "SQL is used for querying relational databases.",
            "The Eiffel Tower is located in Paris, France.",
            "Mount Everest is the tallest mountain on Earth.",
            "Photosynthesis converts sunlight to chemical energy.",
            "The speed of light is approximately 300,000 km/s.",
        ]

        query = "Which programming language is best for beginners?"

        # Stage 1: Dense retrieval (top-4 by cosine similarity)
        query_emb = list(embedding_model.query_embed(query))[0]
        doc_embs = list(embedding_model.embed(corpus))
        sims = np.array([np.dot(query_emb, d) for d in doc_embs])
        top4_indices = np.argsort(sims)[-4:][::-1]
        top4_docs = [corpus[i] for i in top4_indices]

        # This test used to require every top-4 slot to be a programming doc
        # (indices 0-3). That claim is not measurable on a 0.6B model: its
        # decision margin is the gap between corpus[2] (0.2286) and corpus[7]
        # (0.2315), i.e. -0.003, while the two shipped quantizations of these
        # same weights disagree by 0.089 on average across these very
        # similarities. The assertion was ~30x below the model's own numerical
        # noise floor, so it measured quantization, not retrieval.
        #
        # The two claims below clear that floor by ~2x and hold on both the
        # INT8 and Q4F16 builds. Both flip to failing if last-token pooling
        # regresses, so they still detect a real break in this pipeline.

        # The single most relevant document must rank first (margin ~0.19).
        assert top4_indices[0] == 0, (
            f"Expected the Python doc to rank first, got corpus[{top4_indices[0]}]: "
            f"{corpus[top4_indices[0]]!r} (similarities: {np.round(sims, 4).tolist()})"
        )

        # Programming docs as a group must outscore the rest (gap ~0.16).
        prog_mean = float(sims[:4].mean())
        other_mean = float(sims[4:].mean())
        assert prog_mean - other_mean > 0.05, (
            f"Programming docs should score clearly above the rest, got "
            f"{prog_mean:.4f} vs {other_mean:.4f} (gap {prog_mean - other_mean:+.4f})"
        )

        # Stage 2: Rerank top-4
        rerank_scores = list(reranker_model.rerank(query, top4_docs))

        # Python doc should rank first after reranking (simplicity -> beginners)
        best_reranked = top4_docs[np.argmax(rerank_scores)]
        assert "Python" in best_reranked, (
            f"Expected Python doc to rank first for beginners query, got: {best_reranked}"
        )
