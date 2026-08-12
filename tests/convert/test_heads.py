import pytest

from fastretrieval.convert.heads import resolve_yes_no_ids


class _FakeTokenizer:
    """Tokenizer tối giản: chỉ cần convert_tokens_to_ids và unk_token_id."""

    unk_token_id = 0

    def __init__(self, table):
        self._table = table

    def convert_tokens_to_ids(self, token):
        return self._table.get(token, self.unk_token_id)


def test_resolves_both_ids():
    tok = _FakeTokenizer({"yes": 9693, "no": 2152})
    assert resolve_yes_no_ids(tok, "yes", "no") == (9693, 2152)


def test_unknown_token_is_an_error_not_a_silent_unk():
    tok = _FakeTokenizer({"yes": 9693})
    with pytest.raises(ValueError, match="'no'"):
        resolve_yes_no_ids(tok, "yes", "no")


def test_identical_ids_are_rejected():
    tok = _FakeTokenizer({"yes": 5, "no": 5})
    with pytest.raises(ValueError, match="same token id"):
        resolve_yes_no_ids(tok, "yes", "no")
