# Legal Claim Grounding

Bind every cited claim in a legal document to the exact supporting passage in a source, and judge whether the source **supports**, **contradicts**, is **silent on**, or only **weakly** addresses it — each with a calibrated confidence and the precise supporting line.

Built on the [Isaacus](https://isaacus.com) legal-AI models (`kanon-2-reranker`, `kanon-universal-classifier`, `kanon-answer-extractor`). Exposed as a FastAPI `/verify` service with two clients: a **Word add-in** (select a sentence in your document, verify it against your sources) and a **Streamlit** dev UI.

---

## What it does

Given a document containing cited claims and one or more source documents, for each claim the system:

1. **Segments** the document into sentences and flags the cited ones.
2. **Chunks** each source and **reranks** the chunks against the claim to find the most relevant passages.
3. **Extracts** the exact supporting line from the best passage (char-precise span).
4. **Classifies** support with two independent classifier calls (`p_support` and `p_contra`).
5. Returns a **verdict + confidence + supporting span + top-3 evidence passages**.

The reranker is used only to pick the passage fed to the extractor — **never** for the verdict. Whether a source is silent is decided by the classifier (see [Verdict logic](#verdict-logic)).

---

## Architecture

```
document ─► segment ─► (per cited claim, fanned out concurrently)
                          ├─ chunk      (semchunk → discrete passages)
                          ├─ rerank     (kanon-2-reranker → top passages)
                          ├─ extract    (kanon-answer-extractor → exact span)
                          └─ classify   (kanon-universal-classifier ×2: claim + negation)
                       ─► ClaimResult { verdict, confidence, span, source_id, passages[] }
```

Every model call routes through `src/grounding/client.py` (the single vendor seam). The API field shapes used throughout are documented in [`NOTES_api.md`](NOTES_api.md).

### Verdict logic

Four signals per claim feed a sequential decision tree (`src/grounding/classify.py`):

| Rule | Condition | Verdict | Confidence |
|---|---|---|---|
| 1 | `max(p_support, p_contra) < τ_low` | `unaddressed` | `1 − max(p_support, p_contra)` |
| 2 | `p_contra > τ_con` and `p_contra > p_support` | `contradicted` | `p_contra × (1 − p_support)` |
| 3 | `p_support > τ_sup` and `inextract < τ_inex` | `supported` | `p_support × (1 − inextract)` |
| 4 | otherwise | `weak` | `max(p_support, p_contra)` |

`p_support` and `p_contra` are **independent** classifier calls (one on the claim, one on `"It is not the case that: {claim}"`) — they do not sum to 1. "Unaddressed" is detected when **both** are low, which means the source is genuinely silent (the classifier auto-chunks the full source, so this is robust to retrieval misses).

Shipped thresholds (calibrated on ContractNLI dev, 1037 pairs): `τ_low=0.55, τ_con=0.7, τ_sup=0.85, τ_inex=0.9`.
---

## Repository layout

```
src/grounding/        Core engine
  client.py             Isaacus / AsyncIsaacus factory (vendor seam)
  segment.py            sentence split + citation-marker detection
  chunk.py              split a source into discrete passages (semchunk)
  rerank.py             rank a source's chunks against the claim (the retrieval step)
  extract.py            extractive QA → exact supporting span (the trust anchor)
  classify.py           4-rule verdict from two classifier calls + thresholds
  parse.py              PDF / DOCX / TXT text extraction
  pipeline.py           orchestration → ClaimResult
  models.py             pydantic schemas
service/app.py        FastAPI: /health, /verify, /extract
addin/                Word task-pane add-in (HTML/CSS/JS + manifest)
ui/streamlit_app.py   Streamlit dev UI
eval/                 ContractNLI runner + threshold calibration harness
tests/                pytest suite
```

---

## Setup

Requires Python ≥ 3.11 and an Isaacus API key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm

cp .env.example .env        # then put your real key in .env
```

`.env`:

```
ISAACUS_API_KEY=your-key-here
ISAACUS_REGION=us
DEPLOY_MODE=local
```

Verify the install:

```bash
python -c "import grounding; print('ok')"
pytest -q
```

---

## Running

### FastAPI service

```bash
uvicorn service.app:app --reload --port 8000
```

The Word add-in's webview requires HTTPS. For that path, run with the Office dev certs (created by `npx office-addin-dev-certs install`):

```bash
uvicorn service.app:app --host localhost --port 8000 \
  --ssl-keyfile ~/.office-addin-dev-certs/localhost.key \
  --ssl-certfile ~/.office-addin-dev-certs/localhost.crt
```

Endpoints:

| Method | Path | Body | Returns |
|---|---|---|---|
| GET  | `/health` | — | `{"status":"ok"}` |
| POST | `/verify` | `{document, sources:[{id,text}]}` | `{claims:[ClaimResult], summary}` |
| POST | `/extract` | multipart files (PDF/DOCX/TXT) | `[{id, text, chars, pages}]` |

Example:

```bash
curl -sk -X POST https://localhost:8000/verify -H "Content-Type: application/json" -d '{
  "document": "The Receiving Party shall keep information secret for five years, pursuant to the Agreement.",
  "sources": [{"id":"NDA-1","text":"The Receiving Party agrees to keep all Confidential Information secret for a period of five (5) years from the date of disclosure."}]
}'
```

### Streamlit dev UI

```bash
streamlit run ui/streamlit_app.py
```

Paste or upload a document + sources, run the pipeline, and see each claim color-coded with its supporting span and confidence.

### Word add-in

The add-in lets you select any sentence in a Word document and verify it against sources you add (upload a file, use the current document, or paste text). Results show the verdict plus the top-3 evidence passages; clicking an evidence row opens the exact line.

1. Install the trusted local certs once:
   ```bash
   npx office-addin-dev-certs install
   ```
2. Serve the add-in files over HTTPS:
   ```bash
   python3 addin/serve_https.py        # https://localhost:3001
   ```
3. Start the FastAPI service over HTTPS (command above).
4. Sideload the manifest into Word:
   ```bash
   npm install
   npx office-addin-debugging start addin/manifest.xml desktop --app word
   ```
   Or copy `addin/manifest.xml` into `~/Library/Containers/com.microsoft.Word/Data/Documents/wef/` (Mac) and restart Word. On Windows, use **Insert → Add-ins → Upload My Add-in**.

The add-in then appears as **Verify Claims** in the Word Home ribbon.

---

## Evaluation

The pipeline is evaluated and calibrated against [ContractNLI](https://stanfordnlp.github.io/contract-nli/) (423 NDAs × 17 hypotheses). The dataset is **not** included in this repo (it has its own license/terms) — download it into `eval/contract-nli/` separately.

```bash
python eval/run_eval.py --split dev          # confusion matrix + P/R/F1 + false-green rate
```

Threshold calibration is offline and cheap once the scores are cached (`eval/calibrate.py`):

```bash
python eval/calibrate.py cache    --split dev    # one API pass → eval/scores_dev.json
python eval/calibrate.py tune     --split dev    # offline grid search over thresholds
python eval/calibrate.py diagnose --split dev    # per-class signal-separation analysis
python eval/calibrate.py report   --split dev    # apply shipped thresholds, print scorecard
```

`cache` makes the only API calls; `tune`/`diagnose`/`report` operate on the cached scores, so you can re-tune thresholds instantly without re-spending on the API. Re-cache only when models or prompts change.

---
## License

Code in this repository is provided as-is. The ContractNLI dataset referenced for evaluation is distributed under its own terms and is not included here.
