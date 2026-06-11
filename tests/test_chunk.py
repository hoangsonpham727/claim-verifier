"""Unit test for source chunking (no API)."""
from grounding.chunk import chunk_source


SOURCE = (
    "The occupier failed to repair the broken staircase despite written notice. "
    "Under the Occupiers Liability Act 1957 s.2, an occupier owes a duty of care. "
    "The plaintiff slipped and sustained a fractured wrist. "
    "The weather on the day was overcast but dry."
)


def test_chunk_source_non_empty():
    chunks = chunk_source(SOURCE)
    assert len(chunks) >= 1
    assert all(len(c) > 0 for c in chunks)


def test_chunk_source_short_text_single_chunk():
    chunks = chunk_source("A short clause.")
    assert chunks == ["A short clause."] or len(chunks) == 1


def test_chunk_source_never_empty_for_empty_input():
    # Falls back to the original text rather than returning []
    assert chunk_source("") == [""]
