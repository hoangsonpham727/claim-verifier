"""Pydantic schemas shared by the engine, the FastAPI service, and the tests.

Roughly in pipeline order: Source and Claim are inputs; Candidate, SupportingSpan
and EvidenceItem are intermediate retrieval/grounding results; ClaimResult and
VerifyResponse are what the API returns; Workspace is the per-document state the
add-in saves and restores.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    id: str
    text: str


class Claim(BaseModel):
    text: str
    has_citation: bool = False
    range_ref: Optional[str] = None  # sentence index or paragraph ref for Word
    source_id: Optional[str] = None  # which source id this claim cites


class Candidate(BaseModel):
    text: str
    source_id: str
    chunk_index: int
    score: float = Field(ge=0.0, le=1.0)


class SupportingSpan(BaseModel):
    text: str
    start: int  # char offset in source text
    end: int    # exclusive — source_text[start:end]
    score: float = Field(ge=0.0, le=1.0)


class RankedPassage(BaseModel):
    source_id: str
    score: float = Field(ge=0.0, le=1.0)  # reranker relevance score
    passage: str                          # the reranked chunk text
    line: Optional[str] = None            # exact supporting sentence (extractor)
    line_start: Optional[int] = None      # offset of `line` within `passage`
    line_end: Optional[int] = None


class EvidenceItem(BaseModel):
    """One piece of evidence gathered by the routing engine, labeled by how it
    was reached from the seed clause."""
    source_id: str
    relation: Literal["seed", "crossref", "term", "external"] = "seed"
    label: str = ""                       # e.g. "Section 4 — cross-referenced"
    text: str
    seg_type: Optional[str] = None        # ILDGS segment type (section/clause/…)
    start: Optional[int] = None           # offset of `text` in its source
    end: Optional[int] = None
    score: Optional[float] = None         # rerank/cosine score where applicable


class ClaimResult(BaseModel):
    text: str
    range_ref: Optional[str] = None
    verdict: Literal["supported", "contradicted", "unaddressed", "weak"]
    confidence: float = Field(ge=0.0, le=1.0)
    span: Optional[SupportingSpan] = None
    source_id: Optional[str] = None
    evidence: list[EvidenceItem] = Field(default_factory=list)  # routed evidence
    passages: list[RankedPassage] = Field(default_factory=list)  # deprecated shim


class VerifyRequest(BaseModel):
    document: str = ""                       # full text enriched into the graph
    claims: Optional[list[str]] = None       # explicit claims (else segmented from document)
    sources: list[Source] = Field(default_factory=list)


class VerifySummary(BaseModel):
    total_cited: int
    supported: int
    contradicted: int
    unaddressed: int
    weak: int = 0


class VerifyResponse(BaseModel):
    claims: list[ClaimResult]
    summary: VerifySummary


class Workspace(BaseModel):
    """A user's saved working state for one main document: the sources they added
    and their prior verification results, so reopening restores the review."""
    id: str
    document_hash: str = ""                 # hash of the main document at last save
    sources: list[Source] = Field(default_factory=list)
    results: dict[str, ClaimResult] = Field(default_factory=dict)  # claim text → result
    updated_at: float = 0.0
