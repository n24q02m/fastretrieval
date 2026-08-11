import pytest

import qwen3_embed.common.utils
import qwen3_embed.rerank.cross_encoder.text_cross_encoder
import qwen3_embed.text.text_embedding


@pytest.fixture(autouse=True)
def _low_max_input_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lower MAX_INPUT_LENGTH for every test in this module.

    check_input_length reads the module global at call time, so patching it
    per-test (and letting monkeypatch restore it afterwards) is sufficient --
    it does not need to be set before the modules under test are imported.
    A low limit lets the tests below exercise the length guard without
    allocating a huge string, and the patch does not leak into other test
    modules collected in the same pytest session.
    """
    monkeypatch.setattr(qwen3_embed.common.utils, "MAX_INPUT_LENGTH", 100)


def test_check_input_length():
    qwen3_embed.common.utils.check_input_length("a" * 100)
    with pytest.raises(ValueError):
        qwen3_embed.common.utils.check_input_length("a" * 101)


def test_iter_checked_texts():
    texts = ["a" * 10, "b" * 100, "c" * 5]
    assert list(qwen3_embed.common.utils.iter_checked_texts(texts)) == texts
    iterator = qwen3_embed.common.utils.iter_checked_texts(["a" * 10, "b" * 101])
    assert next(iter(iterator)) == "a" * 10
    with pytest.raises(ValueError):
        next(iter(iterator))


class MockModel:
    def embed(self, docs, *args, **kwargs):
        docs = list(docs)
        yield from [1 for _ in docs]

    def query_embed(self, q, *args, **kwargs):
        q = list(q)
        yield from [1 for _ in q]

    def passage_embed(self, texts, *args, **kwargs):
        texts = list(texts)
        yield from [1 for _ in texts]

    def rerank(self, query, docs, *args, **kwargs):
        docs = list(docs)
        yield from [1 for _ in docs]

    def rerank_pairs(self, pairs, *args, **kwargs):
        pairs = list(pairs)
        yield from [1 for _ in pairs]


def test_text_embedding_limits(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(
            qwen3_embed.text.text_embedding.TextEmbedding,
            "__init__",
            lambda self, *args, **kwargs: setattr(self, "model", MockModel()),
        )
        te = qwen3_embed.text.text_embedding.TextEmbedding("dummy")
        with pytest.raises(ValueError):
            list(te.embed("a" * 101))
        with pytest.raises(ValueError):
            list(te.embed(["a" * 50, "b" * 101]))
        assert len(list(te.embed(["a" * 10, "b" * 20]))) == 2
        with pytest.raises(ValueError):
            list(te.query_embed("a" * 101))
        with pytest.raises(ValueError):
            list(te.passage_embed(["a" * 10, "b" * 101]))


def test_text_cross_encoder_limits(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(
            qwen3_embed.rerank.cross_encoder.text_cross_encoder.TextCrossEncoder,
            "__init__",
            lambda self, *args, **kwargs: setattr(self, "model", MockModel()),
        )
        tce = qwen3_embed.rerank.cross_encoder.text_cross_encoder.TextCrossEncoder("dummy")
        with pytest.raises(ValueError):
            list(tce.rerank("a" * 101, ["doc1"]))
        with pytest.raises(ValueError):
            list(tce.rerank("query", ["doc1", "b" * 101]))
        with pytest.raises(ValueError):
            list(tce.rerank_pairs([("q1", "d1"), ("q2", "d2" * 101)]))
        assert len(list(tce.rerank("query", ["doc1", "doc2"]))) == 2
        assert len(list(tce.rerank_pairs([("q1", "d1"), ("q2", "d2")]))) == 2


def test_define_cache_dir_symlink_prevention(tmp_path):
    import os

    from qwen3_embed.common.utils import define_cache_dir

    target_dir = tmp_path / "target"
    target_dir.mkdir(mode=0o777)

    link_dir = tmp_path / "link"
    os.symlink(str(target_dir), str(link_dir))

    # define_cache_dir should not follow symlink to chmod the target
    define_cache_dir(str(link_dir))

    # Target directory permissions should remain 0o777 (or umask, but not 0o700)
    assert oct(target_dir.stat().st_mode)[-3:] != "700"
