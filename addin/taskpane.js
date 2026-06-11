/* ── Legal Claim Grounding — Word task pane ───────────────────────── */

const DEFAULT_API_URL = "https://localhost:8000";
const STORAGE_KEY_SOURCES = "lcg_sources";
const STORAGE_KEY_API_URL = "lcg_api_url";

const VERDICT_META = {
  supported:    { label: "Supported",     hl: "#c6efce", border: "#28a745" },
  contradicted: { label: "Contradicted",  hl: "#ffc7ce", border: "#dc3545" },
  weak:         { label: "Needs review",  hl: "#ffeb9c", border: "#ffc107" },
  unaddressed:  { label: "Not addressed", hl: "#d3d3d3", border: "#adb5bd" },
};

const KIND_TAG = { file: "FILE", document: "DOCUMENT", paste: "TEXT" };

// How a piece of evidence was reached in the document graph.
const RELATION_TAG = { seed: "MATCH", crossref: "CROSS-REF", term: "DEFINITION", external: "EXTERNAL" };
const DOC_SOURCE_ID = "This document";   // reserved id the backend gives the open document

/* ── State ────────────────────────────────────────────────────────── */
let sources = [];          // [{id, text, kind}]
let lastResults = [];      // ClaimResult[]
let selectedText = "";     // currently selected text in the Word document
let apiUrl = DEFAULT_API_URL;

/* ── DOM refs ────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

/* ── Init ────────────────────────────────────────────────────────── */
Office.onReady(() => {
  loadState();
  renderSources();
  renderApiUrl();
  renderClaimPill();          // show correct initial state (no "Scanning…")
  pollSelection();            // poll selection every 1s
  bindEvents();
});

function loadState() {
  try {
    sources = JSON.parse(localStorage.getItem(STORAGE_KEY_SOURCES) || "[]");
    apiUrl  = localStorage.getItem(STORAGE_KEY_API_URL) || DEFAULT_API_URL;
  } catch (_) { sources = []; }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY_SOURCES, JSON.stringify(sources));
  localStorage.setItem(STORAGE_KEY_API_URL, apiUrl);
}

/* ── Event wiring ────────────────────────────────────────────────── */
function bindEvents() {
  $("add-source-btn").addEventListener("click", handleAddSource);
  $("verify-btn").addEventListener("click", handleVerify);
  $("api-url-input").addEventListener("change", e => {
    apiUrl = e.target.value.trim().replace(/\/$/, "");
    saveState();
  });

  // Source-mode switcher
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => switchMode(btn.dataset.mode));
  });

  // File upload (click-picked or dropped)
  const fileInput = $("src-file");
  fileInput.addEventListener("change", () => handleUploadFiles(fileInput.files));
  const drop = document.querySelector(".file-drop");
  ["dragover", "dragenter"].forEach(ev =>
    drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach(ev =>
    drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", e => handleUploadFiles(e.dataTransfer.files));
}

function switchMode(mode) {
  document.querySelectorAll(".mode-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.mode === mode));
  document.querySelectorAll(".mode-panel").forEach(p =>
    p.hidden = p.dataset.panel !== mode);
  showInlineError("src-error", "");
}

/* ── Selection detection ──────────────────────────────────────────── */

async function getSelectedText() {
  return new Promise((resolve, reject) => {
    Word.run(async context => {
      const sel = context.document.getSelection();
      sel.load("text");
      await context.sync();
      resolve(sel.text.trim());
    }).catch(reject);
  });
}

function wordCount(str) {
  return str.trim().split(/\s+/).length;
}

/* Poll every second; update pill when selection changes */
async function pollSelection() {
  try {
    const text = await getSelectedText();
    if (text !== selectedText) {
      selectedText = text;
      renderClaimPill();
    }
  } catch (_) { /* not yet in a Word document */ }
  setTimeout(pollSelection, 1000);
}

function renderClaimPill() {
  const pill = $("claim-pill");
  const hasSelection = wordCount(selectedText) >= 4;
  if (!hasSelection) {
    pill.className = "claim-pill";
    pill.innerHTML = `<span class="dot"></span> Select a claim in your document`;
  } else {
    const preview = selectedText.length > 60
      ? selectedText.slice(0, 57) + "…"
      : selectedText;
    pill.className = "claim-pill has-claims";
    pill.innerHTML = `<span class="dot"></span> "${escHtml(preview)}"`;
  }
  // The open document is always a grounding source, so a selection alone is enough.
  $("verify-btn").disabled = !hasSelection;
}

/* ── Source management ────────────────────────────────────────────── */

/* Add a source with a unique id (dedupes with " (2)", " (3)", …). */
function addSource(id, text, kind) {
  const base = (id || "Source").trim();
  let unique = base, n = 2;
  while (sources.find(s => s.id === unique)) unique = `${base} (${n++})`;
  sources.push({ id: unique, text, kind });
  saveState();
  renderSources();
  renderClaimPill();
  return unique;
}

/* Paste mode — id is optional, auto-named "Source N" when blank. */
function handleAddSource() {
  const text = $("src-text").value.trim();
  if (!text) { showInlineError("src-error", "Paste the source text first."); return; }
  const id = $("src-id").value.trim() || `Source ${sources.length + 1}`;
  addSource(id, text, "paste");
  $("src-id").value   = "";
  $("src-text").value = "";
  showInlineError("src-error", "");
}

/* Upload mode — parse PDF/DOCX/TXT on the backend, one source per file. */
async function handleUploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  showInlineError("src-error", "");
  const statusEl = $("upload-status");

  for (const file of files) {
    statusEl.innerHTML = `<div class="upload-row"><span class="spinner-sm"></span> Reading ${escHtml(file.name)}…</div>`;
    try {
      const form = new FormData();
      form.append("files", file, file.name);
      const resp = await fetch(`${apiUrl}/extract`, { method: "POST", body: form });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const [rec] = await resp.json();
      const id = addSource(rec.id, rec.text, "file");
      statusEl.innerHTML = `<div class="upload-row ok">Added “${escHtml(id)}”</div>`;
    } catch (err) {
      statusEl.innerHTML = "";
      showInlineError("src-error", `Couldn't read ${file.name}: ${err.message}`);
    }
  }
  $("src-file").value = "";  // allow re-selecting the same file
  setTimeout(() => { statusEl.innerHTML = ""; }, 2500);
}

/* Read the full open Word document — auto-included as a grounding source so
   same-document cross-references resolve. */
async function readCurrentDocument() {
  return new Promise((resolve, reject) => {
    Word.run(async context => {
      const body = context.document.body;
      body.load("text");
      await context.sync();
      resolve({ text: body.text.trim(), name: currentDocumentName() });
    }).catch(reject);
  });
}

function currentDocumentName() {
  try {
    const url = Office.context.document.url || "";
    const base = decodeURIComponent(url.split(/[\\/]/).pop() || "");
    const stem = base.replace(/\.[^.]+$/, "");
    return stem || "Current document";
  } catch (_) {
    return "Current document";
  }
}

function removeSource(id) {
  sources = sources.filter(s => s.id !== id);
  saveState();
  renderSources();
  renderClaimPill();
}

function renderSources() {
  const list = $("source-list");
  if (sources.length === 0) {
    list.innerHTML = `<p class="status-msg">No external sources. The open document is checked automatically.</p>`;
    return;
  }
  list.innerHTML = sources.map(s => `
    <div class="source-item">
      <div class="source-info">
        <span class="source-kind">${KIND_TAG[s.kind] || "TEXT"}</span>
        <span class="source-id">${escHtml(s.id)}</span>
        <span class="source-meta"> · ${(s.text.length / 1000).toFixed(1)}k chars</span>
      </div>
      <button class="remove-btn" onclick="removeSource('${escHtml(s.id).replace(/'/g, "\\'")}')" title="Remove">✕</button>
    </div>
  `).join("");
}

/* ── Verify ────────────────────────────────────────────────────────── */
async function handleVerify() {
  if (!selectedText) return;

  setVerifying(true);
  clearResults();

  // Send the selected claim explicitly, plus the FULL open document so the graph
  // can resolve same-document cross-references. The backend enriches the document
  // and excludes the claim's own clause from its evidence.
  let documentText = "";
  try { documentText = (await readCurrentDocument()).text; } catch (_) { /* ignore */ }

  const payloadSources = sources.map(s => ({ id: s.id, text: s.text }));

  try {
    const resp = await fetch(`${apiUrl}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claims: [selectedText],
        document: documentText,
        sources: payloadSources,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    lastResults = data.claims || [];
    renderResults(lastResults);
    await highlightInDocument(lastResults);
  } catch (err) {
    $("results-area").innerHTML = `<div class="error-msg">Error: ${escHtml(err.message)}</div>`;
  } finally {
    setVerifying(false);
  }
}

/* ── Results rendering ────────────────────────────────────────────── */
function renderResults(claims) {
  if (!claims.length) {
    $("results-area").innerHTML = `<p class="status-msg">No claim detected in the selection.</p>`;
    return;
  }
  $("results-area").innerHTML = claims.map((r, i) => resultCard(r, i)).join("");
}

function resultCard(r, idx) {
  const m = VERDICT_META[r.verdict] || VERDICT_META.unaddressed;
  const conf = Math.round((r.confidence || 0) * 100);

  const line = (r.span && r.span.text)
    ? `<div class="supporting-line">“${escHtml(truncate(r.span.text, 240))}”</div>`
    : "";

  const evidence = (r.evidence || []);
  const evHtml = evidence.length
    ? `<div class="evidence">
         <div class="evidence-head">Evidence</div>
         ${evidence.map((e, j) => evidenceRow(e, idx, j)).join("")}
       </div>`
    : `<p class="evidence-empty">No supporting passage found.</p>`;

  return `
    <div class="result-card" style="border-left-color:${m.border}">
      <div class="verdict-row">
        <span class="verdict-dot" style="background:${m.border}"></span>
        <span class="verdict-label">${escHtml(m.label)}</span>
        <span class="verdict-conf">${conf}%</span>
      </div>
      <div class="result-claim">${escHtml(r.text)}</div>
      ${line}
      ${evHtml}
    </div>
  `;
}

/* One gathered clause, labeled by how the graph reached it (match / cross-ref /
   definition / external). Click to expand the full clause; jump to it in Word
   when it comes from the open document. */
function evidenceRow(e, idx, j) {
  const tag = RELATION_TAG[e.relation] || "MATCH";
  const score = (typeof e.score === "number") ? `${Math.round(e.score * 100)}%` : "";
  const descr = e.seg_type ? escHtml(e.seg_type) : "";
  return `
    <div class="evidence-row" id="ev-${idx}-${j}" onclick="toggleEvidence(${idx}, ${j})">
      <div class="evidence-meta">
        <span class="relation-tag rel-${escHtml(e.relation || "seed")}">${tag}</span>
        <span class="source-name">${escHtml(e.source_id)}</span>
        ${descr ? `<span class="seg-type">${descr}</span>` : ""}
        ${score ? `<span class="evidence-score">${score}</span>` : ""}
      </div>
      <div class="evidence-text">${escHtml(e.text)}</div>
    </div>
  `;
}

/* Expand/collapse a clause; jump+select in Word when it's from the open document. */
function toggleEvidence(idx, j) {
  const row = $(`ev-${idx}-${j}`);
  if (!row) return;
  const open = row.classList.toggle("open");
  const e = (lastResults[idx]?.evidence || [])[j];
  if (open && e && e.source_id === DOC_SOURCE_ID && e.text) {
    jumpToLine(e.text);
  }
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

/* Select and scroll to a line inside the open Word document. */
async function jumpToLine(line) {
  return Word.run(async context => {
    const found = context.document.body.search(line.slice(0, 255), { matchCase: false });
    found.load("items");
    await context.sync();
    if (found.items.length) {
      found.items[0].select("Select");
      found.items[0].scrollIntoView(true);
    }
    await context.sync();
  }).catch(() => {});
}

/* ── Document highlighting ────────────────────────────────────────── */
async function highlightInDocument(claims) {
  if (!claims.length) return;
  return Word.run(async context => {
    for (const r of claims) {
      const m = VERDICT_META[r.verdict] || VERDICT_META.unaddressed;
      const query = r.text.slice(0, 255); // Word search has char limit
      const found = context.document.body.search(query, { matchCase: false });
      found.load("items");
      await context.sync();
      if (found.items.length) {
        found.items[0].font.highlightColor = m.hl;
      }
    }
    await context.sync();
  }).catch(() => { /* silently ignore if doc not accessible */ });
}

/* ── UI helpers ──────────────────────────────────────────────────── */
function setVerifying(active) {
  const btn = $("verify-btn");
  btn.disabled = active;
  btn.textContent = active ? "Verifying…" : "Verify selection";
  if (active) {
    $("results-area").innerHTML = `<div style="padding:20px 0"><div class="spinner"></div></div>`;
  }
}

function clearResults() {
  $("results-area").innerHTML = "";
}

function renderApiUrl() {
  $("api-url-input").value = apiUrl;
}

function showInlineError(elemId, msg) {
  const el = $(elemId);
  el.textContent = msg;
  el.style.display = msg ? "block" : "none";
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
