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
from pathlib import Path, PurePath

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from grounding.models import VerifyRequest, VerifyResponse
from grounding.parse import count_pdf_pages, extract_text
from grounding.pipeline import verify

app = FastAPI(
    title="Isaacus Legal Claim Grounding",
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
