"""
Shared document-text extraction.

Single source of truth for turning an uploaded PDF / DOCX / TXT into plain
text, reused by both the Streamlit UI and the FastAPI `/extract` endpoint.
"""
from __future__ import annotations

import io


def extract_text(filename: str, raw: bytes) -> str:
    """Extract plain text from a PDF, DOCX, or text file.

    Args:
        filename: original file name (used only to pick the parser by extension).
        raw: the raw file bytes.

    Returns:
        Extracted plain text. Unknown extensions are decoded as UTF-8.
    """
    name = filename.lower()
    if name.endswith(".pdf"):
        import pymupdf

        doc = pymupdf.open(stream=raw, filetype="pdf")
        return "\n\n".join(str(page.get_text()) for page in doc)
    if name.endswith(".docx"):
        from docx import Document as DocxDoc

        docx = DocxDoc(io.BytesIO(raw))
        return "\n".join(p.text for p in docx.paragraphs if p.text.strip())
    # Fall back to raw UTF-8 (txt, md, anything else)
    return raw.decode("utf-8", errors="replace")


def count_pdf_pages(raw: bytes) -> int | None:
    """Return the page count of a PDF, or None if it isn't a readable PDF."""
    try:
        import pymupdf

        with pymupdf.open(stream=raw, filetype="pdf") as doc:
            return doc.page_count
    except Exception:
        return None
