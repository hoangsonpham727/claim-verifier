"""Durable disk cache for expensive API artifacts, plus per-document workspaces.

Enrichment Documents and segment embeddings cost real API calls, so both are
persisted keyed by content hash. Reopening an unchanged document regenerates
nothing, even across restarts; editing one clause only invalidates that clause.
Workspaces store a user's sources and prior verdicts for a given document.

Layout, rooted at env CACHE_DIR (default ./cache, relative to the process CWD —
so running from python/ gives python/cache/):
    enrich/<sha256>.json     one ILDGS Document (model_dump_json)
    emb/<sha256>.npy         one L2-normalized float32 embedding vector
    workspaces/<id>.json     one Workspace record

Because keys are content hashes, the cache is never stale (changed input ⇒ new
key) and concurrent writers of the same key write identical bytes.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from isaacus.types.ilgs.v1 import Document

_CACHE_DIR = Path(os.environ.get("CACHE_DIR", "cache"))
_ENRICH_DIR = _CACHE_DIR / "enrich"
_EMB_DIR = _CACHE_DIR / "emb"
_WS_DIR = _CACHE_DIR / "workspaces"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)            # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


# ── Enrichment (ILDGS Document) ───────────────────────────────────────────────

def load_enrichment(h: str) -> Optional[Document]:
    path = _ENRICH_DIR / f"{h}.json"
    if not path.exists():
        return None
    try:
        return Document.model_validate_json(path.read_text("utf-8"))
    except Exception:
        return None                      # corrupt/partial → treat as miss


def save_enrichment(h: str, doc: Document) -> None:
    _atomic_write_text(_ENRICH_DIR / f"{h}.json", doc.model_dump_json())


# ── Embeddings (numpy vectors) ────────────────────────────────────────────────

def load_embedding(h: str) -> Optional[np.ndarray]:
    path = _EMB_DIR / f"{h}.npy"
    if not path.exists():
        return None
    try:
        return np.load(path).astype(np.float32)
    except Exception:
        return None


def save_embedding(h: str, vec: np.ndarray) -> None:
    path = _EMB_DIR / f"{h}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npy")
    os.close(fd)
    try:
        np.save(tmp, vec.astype(np.float32), allow_pickle=False)
        # np.save appends .npy to a path without that suffix
        produced = tmp if os.path.exists(tmp) else tmp + ".npy"
        os.replace(produced, path)
    finally:
        for p in (tmp, tmp + ".npy"):
            if os.path.exists(p):
                os.unlink(p)


# ── Workspaces ────────────────────────────────────────────────────────────────

def load_workspace(workspace_id: str) -> Optional[dict]:
    path = _WS_DIR / f"{_safe_id(workspace_id)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return None


def save_workspace(workspace_id: str, data: dict) -> None:
    _atomic_write_text(_WS_DIR / f"{_safe_id(workspace_id)}.json",
                       json.dumps(data, ensure_ascii=False))


def _safe_id(workspace_id: str) -> str:
    """Keep workspace ids to a safe filename charset (defends against traversal)."""
    return "".join(c for c in workspace_id if c.isalnum() or c in "-_")[:128] or "default"
