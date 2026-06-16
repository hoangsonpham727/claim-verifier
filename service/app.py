"""
Phase 5 — FastAPI /verify service.

Run locally:
    uvicorn service.app:app --reload --port 8000

Docker:
    docker build -t grounding-api .
    docker run -p 8000:8000 -e ISAACUS_API_KEY=<key> grounding-api
"""
from __future__ import annotations

import sys
import time
from pathlib import Path, PurePath

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from grounding import store
from grounding.enrich import enrich_sources
from grounding.graph import build_session_graph, retrievable_nodes
from grounding.models import Source, VerifyRequest, VerifyResponse, Workspace
from grounding.parse import count_pdf_pages, extract_text
from grounding.pipeline import verify
from grounding.retrieve import embed_segments

app = FastAPI(
    title="Claim Verifier",
    version="0.1.0",
    description="Bind cited legal claims to exact source passages and judge support.",
)

# Allow all origins for the Word add-in (Office.js runs from ms-appx:// or
# a localhost dev server; CORS must be open during development).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/verify", response_model=VerifyResponse)
async def verify_endpoint(req: VerifyRequest) -> VerifyResponse:
    """
    Verify the cited claims in `document` against `sources` AND the document
    itself (the document is enriched into the graph so same-document
    cross-references resolve; a claim is never grounded by its own clause).

    `sources` may be empty — a document can be verified against itself alone.
    """
    if not req.claims and not req.document.strip():
        raise HTTPException(status_code=422, detail="A claim or document is required.")
    try:
        return await verify(
            req.document, req.sources, claims=req.claims, include_passages=True
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/extract")
async def extract_endpoint(files: list[UploadFile] = File(...)) -> list[dict]:
    """
    Extract plain text from uploaded PDF / DOCX / TXT files.

    Returns one record per file: a default source `id` (the filename stem),
    the extracted `text`, a `chars` count, and `pages` (PDF page count or null).
    The Word add-in uses these to add sources without manual copy-paste.
    """
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required.")

    results: list[dict] = []
    for f in files:
        raw = await f.read()
        if not raw:
            raise HTTPException(status_code=422, detail=f"'{f.filename}' is empty.")
        try:
            text = extract_text(f.filename or "upload.txt", raw)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not read '{f.filename}': {exc}",
            ) from exc

        text = text.strip()
        if not text:
            raise HTTPException(
                status_code=422,
                detail=f"No readable text found in '{f.filename}'. "
                "If it is a scanned PDF, OCR it first.",
            )

        name = f.filename or "upload"
        results.append(
            {
                "id": PurePath(name).stem or name,
                "text": text,
                "chars": len(text),
                "pages": count_pdf_pages(raw) if name.lower().endswith(".pdf") else None,
            }
        )
    return results


# ── Workspaces — per-document saved state (sources + prior results) ───────────

@app.get("/workspace/{workspace_id}", response_model=Workspace)
def get_workspace(workspace_id: str) -> Workspace:
    data = store.load_workspace(workspace_id)
    if data is None:
        raise HTTPException(status_code=404, detail="No saved workspace.")
    return Workspace.model_validate(data)


@app.put("/workspace/{workspace_id}", response_model=Workspace)
def put_workspace(workspace_id: str, ws: Workspace) -> Workspace:
    ws.id = workspace_id
    ws.updated_at = time.time()
    store.save_workspace(workspace_id, ws.model_dump())
    return ws


# ── Index — pre-warm the durable cache so the first verify is instant ─────────

class IndexRequest(BaseModel):
    document: str = ""
    sources: list[Source] = []


@app.post("/index")
async def index_endpoint(req: IndexRequest) -> dict:
    """Enrich + embed the document and sources to populate the durable cache.

    Returns the number of segments indexed. Cheap once the cache is warm (disk
    hits), so the add-in can call it on open without cost on repeat visits.
    """
    graph_sources = list(req.sources)
    if req.document.strip():
        graph_sources.append(Source(id="This document", text=req.document))
    try:
        graph = build_session_graph(enrich_sources(graph_sources))
        embed_segments(graph)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"segments": len(retrievable_nodes(graph)), "sources": len(graph_sources)}
