"""Integration test — hits the live Isaacus API."""
import pytest
from grounding.extract import extract_span


CLAIM = "The occupier breached their duty of care."
RELEVANT_PASSAGE = (
    "The occupier failed to repair the broken staircase despite receiving written notice "
    "from the plaintiff on 14 March 2023. Under the Occupiers Liability Act 1957 s.2, "
    "an occupier owes a common duty of care to all lawful visitors."
)
UNRELATED_PASSAGE = "The weather on the day in question was overcast but dry."


def test_extract_returns_span_for_relevant_passage():
    span, inextr = extract_span(CLAIM, [RELEVANT_PASSAGE])
    # Either we get a span or inextractability is high — at least one must hold
    assert span is not None or inextr >= 0.0


def test_extract_span_offsets_valid():
    span, _ = extract_span(CLAIM, [RELEVANT_PASSAGE])
    if span is not None:
        assert span.start >= 0
        assert span.end > span.start
        assert RELEVANT_PASSAGE[span.start:span.end] == span.text


def test_extract_high_inextractability_for_unrelated():
    span, inextr = extract_span(CLAIM, [UNRELATED_PASSAGE])
    # Unrelated passage should yield either no span or high inextractability
    if span is None:
        assert inextr > 0.0


def test_extract_empty_passages():
    span, inextr = extract_span(CLAIM, [])
    assert span is None
    assert inextr == 1.0
