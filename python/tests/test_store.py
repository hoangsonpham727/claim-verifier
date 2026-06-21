"""Offline tests for the durable disk cache + workspace store (no API)."""
import numpy as np
import pytest
from isaacus.types.ilgs.v1 import Document

import grounding.enrich as enrich_mod
import grounding.retrieve as retrieve_mod
import grounding.store as store
from grounding.graph import SegmentNode, SessionGraph
from grounding.models import Source


def _doc(text="Section 1. Confidential Information.") -> Document:
    return Document(
        text=text, type="contract", version="ilgs@1",
        segments=[], crossreferences=[], external_documents=[], terms=[],
        persons=[], dates=[], quotes=[], locations=[], emails=[],
        websites=[], phone_numbers=[], id_numbers=[], headings=[], junk=[],
    )


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_ENRICH_DIR", tmp_path / "enrich")
    monkeypatch.setattr(store, "_EMB_DIR", tmp_path / "emb")
    monkeypatch.setattr(store, "_WS_DIR", tmp_path / "workspaces")
    enrich_mod._ENRICH_CACHE.clear()
    retrieve_mod._EMB_CACHE.clear()


# ── round-trips ───────────────────────────────────────────────────────────────

def test_enrichment_roundtrip():
    d = _doc("The Receiving Party shall keep it secret.")
    store.save_enrichment("h1", d)
    loaded = store.load_enrichment("h1")
    assert loaded is not None and loaded.text == d.text
    assert store.load_enrichment("missing") is None


def test_embedding_roundtrip():
    v = np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    store.save_embedding("e1", v)
    loaded = store.load_embedding("e1")
    assert loaded is not None and np.allclose(loaded, v) and loaded.dtype == np.float32
    assert store.load_embedding("nope") is None


def test_workspace_roundtrip():
    data = {
        "id": "ws1", "document_hash": "abc",
        "sources": [{"id": "S1", "text": "source text"}],
        "results": {}, "updated_at": 1.5,
    }
    store.save_workspace("ws1", data)
    assert store.load_workspace("ws1") == data
    assert store.load_workspace("missing") is None


def test_workspace_id_sanitized(tmp_path):
    store.save_workspace("../../etc/passwd", {"x": 1})
    # traversal stripped → stays inside the workspaces dir
    files = list((tmp_path / "workspaces").glob("*.json"))
    assert files and ".." not in files[0].name


# ── durable cache hits across an in-memory wipe (simulates a restart) ──────────

def test_durable_enrich_cache_hit(monkeypatch):
    calls = {"n": 0}

    def fake(texts):
        calls["n"] += 1
        return [_doc(t) for t in texts]

    monkeypatch.setattr(enrich_mod, "_enrich_texts", fake)
    srcs = [Source(id="a", text="alpha source text")]

    enrich_mod.enrich_sources(srcs)
    assert calls["n"] == 1

    enrich_mod._ENRICH_CACHE.clear()          # drop L1; disk (L2) must serve
    out = enrich_mod.enrich_sources(srcs)
    assert calls["n"] == 1                     # no new enricher call
    assert out[0].ok and out[0].document.text == "alpha source text"


def test_durable_embed_cache_hit(monkeypatch):
    calls = {"n": 0}

    def fake_embed(texts, task):
        calls["n"] += 1
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(retrieve_mod, "_embed", fake_embed)
    node = SegmentNode("s::1", "s", "1", "a clause of text", "unit", "section",
                       "main", None, 0, 1, True)
    g = SessionGraph(nodes={"s::1": node}, by_source={"s": ["s::1"]},
                     xref_edges={}, term_index={}, extref_index={}, source_text={"s": "…"})

    retrieve_mod.embed_segments(g)
    assert calls["n"] == 1

    retrieve_mod._EMB_CACHE.clear()            # drop L1; disk (L2) must serve
    out = retrieve_mod.embed_segments(g)
    assert calls["n"] == 1                     # no new embedder call
    assert "s::1" in out
