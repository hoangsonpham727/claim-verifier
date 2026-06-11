"""ILDGS enrichment — turn each source into an Isaacus Legal Document Graph.

Wraps the `kanon-2-enricher` model (max 8 texts/call) and caches results by
content hash so unchanged sources are never re-enriched across requests.

The brand is ILDGS (Isaacus Legal Document Graph Schema); the SDK type path is
still `isaacus.types.ilgs.v1` and the schema citation reads `version="ilgs@1"`.
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from isaacus.types.ilgs.v1 import Document

from .client import get_client
from .models import Source

logger = logging.getLogger(__name__)

_ENRICH_BATCH = 8          # enricher hard limit: 8 texts per call
_CACHE_MAX = 256           # bound the in-process cache
_ENRICH_CACHE: "OrderedDict[str, Document]" = OrderedDict()


@dataclass
class EnrichedSource:
    source_id: str
    text: str                       # canonical text = Document.text when ok, else Source.text
    document: Optional[Document]    # ILDGS graph; None when enrichment failed
    ok: bool


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_get(h: str) -> Optional[Document]:
    doc = _ENRICH_CACHE.get(h)
    if doc is not None:
        _ENRICH_CACHE.move_to_end(h)
    return doc


def _cache_put(h: str, doc: Document) -> None:
    _ENRICH_CACHE[h] = doc
    _ENRICH_CACHE.move_to_end(h)
    while len(_ENRICH_CACHE) > _CACHE_MAX:
        _ENRICH_CACHE.popitem(last=False)


def _enrich_texts(texts: list[str]) -> list[Optional[Document]]:
    """Enrich a batch of ≤8 texts; returns one Document per input (None on failure)."""
    client = get_client()
    try:
        resp = client.enrichments.create(
            model="kanon-2-enricher",
            texts=texts,
            overflow_strategy="chunk",
        )
    except Exception as exc:
        logger.warning("enrichment batch failed (%d texts): %s", len(texts), exc)
        return [None] * len(texts)

    out: list[Optional[Document]] = [None] * len(texts)
    for result in resp.results:
        if 0 <= result.index < len(out):
            out[result.index] = result.document
    return out


def enrich_sources(sources: list[Source]) -> list[EnrichedSource]:
    """Enrich every source into an ILDGS Document, using and filling the cache.

    Cache misses are batched ≤8 per call. A batch failure falls back to solo
    retries; a source that still fails comes back with ok=False so the graph
    builder can fall back to plain chunking.
    """
    enriched: list[Optional[EnrichedSource]] = [None] * len(sources)
    misses: list[int] = []

    for i, src in enumerate(sources):
        cached = _cache_get(content_hash(src.text))
        if cached is not None:
            enriched[i] = EnrichedSource(src.id, cached.text, cached, ok=True)
        else:
            misses.append(i)

    for start in range(0, len(misses), _ENRICH_BATCH):
        batch_idx = misses[start:start + _ENRICH_BATCH]
        docs = _enrich_texts([sources[i].text for i in batch_idx])

        for i, doc in zip(batch_idx, docs):
            if doc is None:
                # solo retry before giving up
                doc = _enrich_texts([sources[i].text])[0]
            src = sources[i]
            if doc is None:
                logger.warning("source %r could not be enriched; using fallback", src.id)
                enriched[i] = EnrichedSource(src.id, src.text, None, ok=False)
            else:
                _cache_put(content_hash(src.text), doc)
                enriched[i] = EnrichedSource(src.id, doc.text, doc, ok=True)

    return [e for e in enriched if e is not None]
