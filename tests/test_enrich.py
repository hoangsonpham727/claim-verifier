"""Offline tests for enrichment caching/batching (enricher call stubbed)."""
import pytest
from isaacus.types.ilgs.v1 import Document

import grounding.enrich as enrich_mod
import grounding.store as store
from grounding.enrich import enrich_sources
from grounding.models import Source


def _doc(text: str) -> Document:
    return Document(
        text=text, type="contract", version="ilgs@1",
        segments=[], crossreferences=[], external_documents=[], terms=[],
        persons=[], dates=[], quotes=[], locations=[], emails=[],
        websites=[], phone_numbers=[], id_numbers=[], headings=[], junk=[],
    )


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    # Isolate the durable disk cache so write-through doesn't touch the real cache/.
    monkeypatch.setattr(store, "_ENRICH_DIR", tmp_path / "enrich")
    monkeypatch.setattr(store, "_EMB_DIR", tmp_path / "emb")
    monkeypatch.setattr(store, "_WS_DIR", tmp_path / "workspaces")
    enrich_mod._ENRICH_CACHE.clear()


def _install_stub(monkeypatch, fail_keys=None):
    """Stub _enrich_texts: count calls/batch sizes; optionally fail given texts."""
    fail_keys = fail_keys or set()
    state = {"calls": 0, "batch_sizes": []}

    def fake(texts):
        state["calls"] += 1
        state["batch_sizes"].append(len(texts))
        return [None if t in fail_keys else _doc(t) for t in texts]

    monkeypatch.setattr(enrich_mod, "_enrich_texts", fake)
    return state


def test_cache_hit_makes_no_calls_on_repeat(monkeypatch):
    state = _install_stub(monkeypatch)
    srcs = [Source(id="a", text="alpha text")]

    first = enrich_sources(srcs)
    assert first[0].ok and first[0].document.text == "alpha text"
    assert state["calls"] == 1

    second = enrich_sources(srcs)               # identical content → cache hit
    assert second[0].ok
    assert state["calls"] == 1                  # no new enricher call


def test_batches_capped_at_eight(monkeypatch):
    state = _install_stub(monkeypatch)
    srcs = [Source(id=str(i), text=f"text-{i}") for i in range(10)]

    out = enrich_sources(srcs)
    assert len(out) == 10 and all(e.ok for e in out)
    assert max(state["batch_sizes"]) <= 8       # ≤8 per call
    assert sum(state["batch_sizes"]) == 10      # all enriched once


def test_failed_source_falls_back_to_not_ok(monkeypatch):
    _install_stub(monkeypatch, fail_keys={"bad text"})
    srcs = [Source(id="ok", text="good text"), Source(id="bad", text="bad text")]

    out = {e.source_id: e for e in enrich_sources(srcs)}
    assert out["ok"].ok is True
    assert out["bad"].ok is False and out["bad"].document is None
    assert out["bad"].text == "bad text"        # canonical text falls back to source text
