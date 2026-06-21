"""Split a source document into passages for reranking.

The reranker (and the extractor/classifier downstream) auto-chunk internally,
but we still need *discrete* passages so the reranker can score them against
each other — that is what powers passage selection and the top-3 evidence list.
"""
from __future__ import annotations

import semchunk

# semchunk v4 requires an explicit token_counter callable.
# Word count is a fast, dependency-free approximation (~0.75 words/token for legal text).
# chunk_size=400 words ≈ 512 tokens — comfortably within every model's window.
_CHUNK_SIZE = 400
_TOKEN_COUNTER = lambda text: len(text.split())


def chunk_source(source_text: str) -> list[str]:
    """Split source text into semantic chunks."""
    chunks = semchunk.chunk(source_text, chunk_size=_CHUNK_SIZE, token_counter=_TOKEN_COUNTER)
    if isinstance(chunks, tuple):
        chunks = chunks[0]
    return chunks or [source_text]
