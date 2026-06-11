"""
Phase 4 — Dev UI.

Annotated document view: paste/upload a legal document + sources,
run the pipeline, see red/amber/green per claim with span + confidence on expand.

Run: .venv/bin/streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grounding.models import Source
from grounding.parse import extract_text
from grounding.pipeline import verify

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Legal Claim Grounding",
    page_icon="⚖️",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text_from_upload(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF or DOCX file."""
    return extract_text(uploaded_file.name, uploaded_file.read())


_VERDICT_STYLE = {
    "supported":    {"bg": "#d4edda", "border": "#28a745", "emoji": "✅", "label": "Supported"},
    "contradicted": {"bg": "#f8d7da", "border": "#dc3545", "emoji": "❌", "label": "Contradicted"},
    "weak":         {"bg": "#fff3cd", "border": "#ffc107", "emoji": "⚠️", "label": "Needs review"},
    "unaddressed":  {"bg": "#f8f9fa", "border": "#adb5bd", "emoji": "🔇", "label": "Not addressed"},
}

_UNCITED_STYLE = {"bg": "#f8f9fa", "border": "#dee2e6", "emoji": "—", "label": "Not cited"}


def _claim_card(result, idx: int):
    """Render one claim result as a coloured expandable card."""
    style = _VERDICT_STYLE.get(result.verdict, _UNCITED_STYLE)
    with st.expander(
        f"{style['emoji']} **Claim {idx + 1}** — {style['label']} "
        f"(confidence: {result.confidence:.0%})",
        expanded=False,
    ):
        st.markdown(
            f"""<div style="
                background:{style['bg']};
                border-left: 4px solid {style['border']};
                padding: 10px 14px;
                border-radius: 4px;
                font-size: 0.95rem;
                margin-bottom: 8px;
            ">{result.text}</div>""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Verdict")
            st.markdown(f"**{result.verdict.upper()}**")
            st.caption("Confidence")
            st.progress(result.confidence)
            st.markdown(f"`{result.confidence:.1%}`")
        with col2:
            if result.source_id:
                st.caption("Source")
                st.markdown(f"`{result.source_id}`")
            if result.span:
                st.caption("Supporting span")
                st.markdown(
                    f"""<div style="
                        background:#fff;
                        border:1px solid #ccc;
                        padding:8px 12px;
                        border-radius:4px;
                        font-style:italic;
                        font-size:0.9rem;
                    ">"{result.span.text}"</div>""",
                    unsafe_allow_html=True,
                )
                st.caption(f"Span score: `{result.span.score:.2f}` · chars [{result.span.start}:{result.span.end}]")


def _annotated_doc_html(document: str, results: list) -> str:
    """Build colour-annotated HTML of the document with inline verdict badges."""
    result_by_text: dict[str, object] = {r.text: r for r in results}

    # Split by sentence (naive split on spaCy range_ref ordering is already done
    # in the pipeline; here we re-segment for display)
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    doc = nlp(document)

    parts: list[str] = []
    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue
        result = result_by_text.get(text)
        if result:
            style = _VERDICT_STYLE.get(result.verdict, _UNCITED_STYLE)
            badge = (
                f'<span style="'
                f'display:inline-block;'
                f'background:{style["bg"]};'
                f'border:1px solid {style["border"]};'
                f'border-radius:3px;'
                f'padding:1px 6px;'
                f'margin:2px 0;'
                f'font-size:0.85rem;'
                f'">'
                f'{style["emoji"]} {text}'
                f'</span>'
            )
            parts.append(badge)
        else:
            parts.append(f'<span style="color:#555;">{text}</span>')

    return "<p>" + " ".join(parts) + "</p>"


# ── Sidebar — sources ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚖️ Sources")
    st.caption("Add the documents your legal text cites.")

    if "sources" not in st.session_state:
        st.session_state.sources = []  # list of {"id": str, "text": str}

    with st.form("add_source_form", clear_on_submit=True):
        src_id = st.text_input("Source ID", placeholder="e.g. NDA-2023, Smith-v-Jones")
        src_upload = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
        src_text = st.text_area("…or paste text", height=160)
        add_btn = st.form_submit_button("Add source")

    if add_btn:
        if not src_id.strip():
            st.sidebar.error("Source ID is required.")
        else:
            text = ""
            if src_upload:
                text = _extract_text_from_upload(src_upload)
            elif src_text.strip():
                text = src_text.strip()
            if not text:
                st.sidebar.error("Provide either a file or pasted text.")
            else:
                st.session_state.sources.append({"id": src_id.strip(), "text": text})
                st.sidebar.success(f"Added: {src_id.strip()}")

    if st.session_state.sources:
        st.markdown("**Loaded sources:**")
        for i, s in enumerate(st.session_state.sources):
            col_a, col_b = st.columns([4, 1])
            col_a.markdown(f"`{s['id']}` — {len(s['text'])} chars")
            if col_b.button("✕", key=f"del_{i}"):
                st.session_state.sources.pop(i)
                st.rerun()
        if st.button("Clear all sources"):
            st.session_state.sources = []
            st.rerun()
    else:
        st.info("No sources loaded yet.")

# ── Main — document input ─────────────────────────────────────────────────────

st.title("Legal Claim Grounding")
st.caption("Bind every cited claim in your document to the exact passage that supports — or contradicts — it.")

doc_upload = st.file_uploader("Upload document (PDF / DOCX)", type=["pdf", "docx"], key="doc_upload")
document_text = st.text_area(
    "…or paste your legal document",
    height=220,
    placeholder="Paste a legal brief, memo, or contract here. Sentences with citation markers will be verified.",
)

if doc_upload:
    document_text = _extract_text_from_upload(doc_upload)
    st.success(f"Loaded {len(document_text)} chars from {doc_upload.name}")

verify_btn = st.button(
    "🔍 Verify claims",
    type="primary",
    disabled=not (document_text.strip() and st.session_state.sources),
)

if not document_text.strip():
    st.info("Paste or upload a document above.")
elif not st.session_state.sources:
    st.warning("Add at least one source in the sidebar before verifying.")

# ── Run pipeline ──────────────────────────────────────────────────────────────

if verify_btn and document_text.strip() and st.session_state.sources:
    sources = [Source(id=s["id"], text=s["text"]) for s in st.session_state.sources]
    with st.spinner("Verifying claims… (this may take a few seconds per claim)"):
        response = asyncio.run(verify(document_text, sources))
    st.session_state.response = response
    st.session_state.document_text = document_text

# ── Results ───────────────────────────────────────────────────────────────────

if "response" in st.session_state:
    resp = st.session_state.response
    doc = st.session_state.document_text

    st.divider()
    st.subheader("Results")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total cited claims", resp.summary.total_cited)
    m2.metric("✅ Supported", resp.summary.supported)
    m3.metric("❌ Contradicted", resp.summary.contradicted)
    m4.metric("⚠️ Needs review", resp.summary.weak)
    st.caption(f"🔇 Not addressed: {resp.summary.unaddressed}")

    st.divider()
    tab_annotated, tab_claims = st.tabs(["📄 Annotated document", "🔎 Claim details"])

    with tab_annotated:
        if resp.claims:
            st.markdown(
                _annotated_doc_html(doc, resp.claims),
                unsafe_allow_html=True,
            )
            st.caption(
                "🟢 Supported &nbsp;&nbsp; 🔴 Contradicted &nbsp;&nbsp; 🟡 Needs review &nbsp;&nbsp; 🔇 Not addressed &nbsp;&nbsp; Gray = not cited"
            )
        else:
            st.info("No cited claims found. Make sure sentences contain citation markers (Section X, see, [1], s.2, etc.).")

    with tab_claims:
        if resp.claims:
            for i, result in enumerate(resp.claims):
                _claim_card(result, i)
        else:
            st.info("No cited claims to show.")

    # Export
    st.divider()
    import json as _json
    export = _json.dumps(
        [r.model_dump() for r in resp.claims],
        indent=2,
        default=str,
    )
    st.download_button(
        "⬇️ Export cite-check report (JSON)",
        data=export,
        file_name="cite_check_report.json",
        mime="application/json",
    )
