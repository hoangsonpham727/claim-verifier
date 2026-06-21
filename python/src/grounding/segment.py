"""Segment a document into sentences; flag those carrying a citation marker."""
from __future__ import annotations

import re
from functools import lru_cache

import spacy

from .models import Claim

# Matches common legal citation markers across two document families:
#
# (A) Legal briefs / case law (footnotes, case names, statutory refs):
#     [1], s.2, art.3, (2023), Smith v Jones, see, cf., per Lord, ibid, supra, infra
#
# (B) Contracts / NDAs (section cross-references, cross-reference phrases):
#     Section 4, Article 2, Clause 3  (explicit provision references)
#     pursuant to, as defined in, as set forth in, as provided in  (cross-ref phrases)
#     (a)/(i) only when inline, NOT at sentence-start  (to avoid flagging every sub-clause header)
#
# Audit on 50 ContractNLI NDAs confirmed (A) patterns hit 0 times; (B) patterns are the
# dominant citation markers in real NDA text.
_CITATION_RE = re.compile(
    # --- (A) Legal brief / case law ---
    r"\[\d+\]"                              # footnote refs [1], [23]
    r"|s\.\s*\d+"                           # statutory: s.2, s. 47
    r"|\(\d{4}\)"                           # year in parens (2023)
    r"|\bv\b\.?\s+[A-Z]"                   # case name: Smith v Jones
    r"|\bsee\b"                             # see ...
    r"|\bcf\b\.?"                           # cf.
    r"|\bper\b\s+[A-Z]"                    # per Lord ...
    r"|\bibid\b"                            # ibid
    r"|\bsupra\b"                           # supra
    r"|\binfra\b"                           # infra
    # --- (B) Contract / NDA ---
    r"|\b(?:Section|Article|Clause|Para(?:graph)?)\s+\d+"   # Section 4, Article 2, Clause 3
    r"|\bpursuant\s+to\b"                  # pursuant to [section / agreement]
    r"|\bas\s+(?:defined|set\s+forth|provided)\s+in\b"      # as defined in, as set forth in
    r"|(?<!\A)(?<!\n)\([a-z]{1,3}\)"       # inline (a)/(i) — NOT at sentence start
,
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _nlp() -> spacy.language.Language:
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


_MIN_CLAIM_WORDS = 6  # bare "Section 1." headers have < 6 words — not verifiable claims


def segment_claims(document: str) -> list[Claim]:
    """Split document into sentence-level claims; flag those with citation markers."""
    nlp = _nlp()
    doc = nlp(document)
    claims: list[Claim] = []
    for i, sent in enumerate(doc.sents):
        text = sent.text.strip()
        if not text:
            continue
        word_count = len([t for t in sent if not t.is_space and not t.is_punct])
        has_citation = (
            word_count >= _MIN_CLAIM_WORDS
            and bool(_CITATION_RE.search(text))
        )
        claims.append(Claim(text=text, has_citation=has_citation, range_ref=str(i)))
    return claims
