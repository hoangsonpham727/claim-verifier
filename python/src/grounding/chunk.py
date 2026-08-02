"""Split a source document into discrete passages for reranking.

The models auto-chunk internally, so this is not about fitting a context window.
It is about producing *separately scoreable units*: the reranker can only rank
passages against each other if we hand it a list, and those ranked passages are
what become the seed for extraction and the evidence list shown to the user.
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
