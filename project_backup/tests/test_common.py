import pytest
from mlplo.common import normalize_text, count_words, resolve_model_reference

def test_normalize_text():
    assert normalize_text(None) == ""
    assert normalize_text(123) == "123"
    assert normalize_text(["a", "b"]) == "['a', 'b']"
    assert normalize_text("  hello \n \t world  ") == "hello world"
    assert normalize_text("a\u00a0b") == "a b"

def test_count_words():
    assert count_words(None) == 0
    assert count_words(123) == 0
    assert count_words("") == 0
    assert count_words("hello world") == 2

def test_resolve_model_reference():
    assert resolve_model_reference("my_model") == "my_model"
    assert resolve_model_reference("", fallback="fallback") == "fallback"
    with pytest.raises(ValueError):
        resolve_model_reference(None, None)
