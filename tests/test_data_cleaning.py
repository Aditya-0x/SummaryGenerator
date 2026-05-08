import pytest
from datasets import Dataset
from mlplo.data_cleaning import is_valid_example, deduplicate_split

def test_is_valid_example():
    assert is_valid_example(
        {"text": "A " * 50, "summary": "B " * 10},
        "text", "summary",
        min_document_words=10, max_document_words=100, min_summary_words=5
    )
    # Too short document
    assert not is_valid_example(
        {"text": "A " * 5, "summary": "B " * 10},
        "text", "summary",
        min_document_words=10, max_document_words=100, min_summary_words=5
    )

def test_deduplicate_split():
    data = {"text": ["A", "B", "A", "C"], "summary": ["1", "2", "3", "4"]}
    ds = Dataset.from_dict(data)
    dedup, removed = deduplicate_split(ds, "text")
    assert removed == 1
    assert len(dedup) == 3
    assert dedup["text"] == ["A", "B", "C"]
