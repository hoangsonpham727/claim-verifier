from grounding.segment import segment_claims


DOC = (
    "The plaintiff suffered loss as a result of the defendant's negligence. "
    "The occupier owed a duty of care under the Occupiers Liability Act 1957 s.2. "
    "There was no contributory negligence on the plaintiff's part."
)


def test_segment_returns_claims():
    claims = segment_claims(DOC)
    assert len(claims) >= 3


def test_citation_flagged():
    claims = segment_claims(DOC)
    texts = [c.text for c in claims]
    flagged = [c for c in claims if c.has_citation]
    # the sentence with "s.2" should be flagged
    assert any("s.2" in c.text for c in flagged), f"Expected s.2 sentence flagged; got {texts}"


def test_no_citation_unflagged():
    plain = "There was no contributory negligence on the plaintiff's part."
    claims = segment_claims(plain)
    assert not claims[0].has_citation


def test_range_ref_is_string_index():
    claims = segment_claims(DOC)
    for i, c in enumerate(claims):
        assert c.range_ref == str(i)
