# Isaacus API Shapes — Phase 1 Contract

Confirmed from SDK type stubs (isaacus 0.21.1). These are the real field names used throughout.

The pipeline uses three models: the **reranker** (retrieval), the **universal classifier**
(verdict), and the **answer extractor** (exact span). The embedder and enricher are not used.

## 1. Rerankings

```python
client.rerankings.create(
    model="kanon-2-reranker",   # or "kanon-universal-classifier"
    query=str,
    texts=[...],        # List[str]
    top_n=int,          # optional
    scoring_method="auto",  # chunk_max | chunk_avg | chunk_min
)
# → RerankingResponse
#     .results: List[Result]  — ordered highest→lowest score
#         .index: int          # index into original texts array
#         .score: float        # 0–1 relevance
#     .usage.input_tokens: int
```

## 2. Universal Classifier

```python
client.classifications.universal.create(
    model="kanon-universal-classifier",
    query=str,           # the claim / statement to test
    texts=[...],         # List[str] — source passages
    scoring_method="auto",
)
# → UniversalClassificationResponse
#     .classifications: List[Classification]  — ordered highest→lowest score
#         .index: int
#         .score: float   # >0.5 = text SUPPORTS query; <0.5 = does not
#         .chunks: Optional[List[ClassificationChunk]]
#             .index: int
#             .start: int   # char offset in original text
#             .end: int     # char offset (exclusive) — text[start:end]
#             .score: float
#             .text: str
#     .usage.input_tokens: int
#
# NOTE: chunks are the built-in supporting spans — no separate extraction needed
# for span identification. Use extraction only for precise answer pinpointing.
```

## 3. Extractive QA

```python
client.extractions.qa.create(
    model="kanon-answer-extractor",
    query=str,
    texts=[...],              # List[str]
    top_k=int,                # optional; default = all
    ignore_inextractability=bool,  # set True when passage already confirmed relevant
)
# → AnswerExtractionResponse
#     .extractions: List[Extraction]  — ordered highest answer score (or lowest inextractability)
#         .index: int
#         .answers: List[ExtractionAnswer]   — ordered highest→lowest score
#             .text: str
#             .start: int   # char offset in original text
#             .end: int     # exclusive
#             .score: float
#         .inextractability_score: float
#             # >0.5 AND > max(answer scores) → no extractable answer → UNADDRESSED
#     .usage.input_tokens: int
```

## Per-model chunking behaviour (confirmed from docs.isaacus.com/models)

| Model | Max context | Auto-chunking? | Implication |
|---|---|---|---|
| kanon-2-reranker | 16,384 tokens | Yes (∞ with chunking) | Pass full texts; SDK chunks internally |
| kanon-universal-classifier | 512 tokens | Yes (∞ with chunking) | Pass full texts; SDK chunks internally; chunk spans are returned in response |
| kanon-answer-extractor | 512 tokens | Yes (∞ with chunking) | Pass full texts; SDK chunks internally |

**Pipeline consequence:** we still pre-chunk each source with semchunk so the reranker has *discrete* passages to rank (needed for passage selection + the top-3 evidence list). For classify/extract, pass the full source text — do not manually pre-chunk for those steps. After reranking confirms a passage is relevant, set `ignore_inextractability=True` on the extractor.

## Verdict logic — 4-rule sequential decision tree

Signals in hand after pipeline runs on one claim:
- `relevance`    — reranker score for best passage vs. claim
- `p_support`    — P(source establishes claim) — classifier call with query=claim
- `p_contra`     — P(source establishes NOT claim) — classifier call with query="It is not the case that: {claim}"
- `answer_score` — extractor confidence in the extracted span
- `inextract`    — extractor P(no answer exists in passage)

NOTE: p_support + p_contra do NOT sum to 1.  Universal classifier runs each hypothesis
independently.  Treat as two independent signals.

| Rule | Condition | Verdict | Confidence |
|---|---|---|---|
| 1 | `max(p_support, p_contra) < τ_low` | `unaddressed` | `1 − max(p_support, p_contra)` |
| 2 | `p_contra > τ_con AND p_contra > p_support` | `contradicted` | `p_contra × (1 − p_support)` |
| 3 | `p_support > τ_sup AND inextract < τ_inex` | `supported` | `p_support × (1 − inextract)` |
| 4 | else | `weak` | `max(p_support, p_contra)` |

**KEY:** UNADDRESSED is detected by the CLASSIFIER (both scores low), NOT the reranker.
The reranker relevance score conflates "source is silent" with "source supports the claim
but retrieval failed to surface the chunk" — both give relevance ≈ 0.  The classifier
auto-chunks the FULL source text, so low p_support AND low p_contra means genuinely silent.
The reranker is used only to pick the passage fed to the extractor, never for the verdict.
Confidence formulas exclude relevance for the same reason.

Thresholds are calibrated in `eval/calibrate.py` (4 subcommands):
1. `cache    --split dev`  — one API pass, writes `eval/scores_dev.json`
2. `tune     --split dev`  — offline grid search (instant, no API)
3. `diagnose --split dev`  — offline per-class signal-separation analysis
4. `report   --split dev`  — offline; applies the SHIPPED classify.py constants, prints scorecard

`tune` grid-searches (τ_low, τ_con, τ_sup, τ_inex) under a legal-tool objective: minimise
false-green (predicted supported, truth ≠ supported), then maximise macro-F1.  Re-cache only
when models/prompts change; re-tune freely.

**Shipped thresholds: τ_low=0.55, τ_con=0.7, τ_sup=0.85, τ_inex=0.9**
Dev scorecard (1037 pairs):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| supported | 91.3% | 30.4% | 0.457 |
| contradicted | 25.8% | 41.1% | 0.317 |
| unaddressed | 58.0% | 68.6% | 0.628 |

False-green rate: 2.9%.  Operating point chosen: precision-favored on `supported`
(91% precision is the trust guarantee; recall is sacrificed deliberately).

KNOWN LIMITATION — contradiction is weak (F1 0.32).  `diagnose` showed p_contra from the
negation prompt barely separates classes (supported 0.46 vs contradicted 0.55): the universal
classifier responds to topical relevance, not logical direction.  Shipping 3-way as-is; a
stronger contradiction elicitation would need a re-cache.

Rule 3's `inextract < τ_inex` guard is the entity-substitution catch: if the source discusses
the same topic (p_support high) but the extractor can't locate the specific span, the claim
may reference a party/date/amount not present in the source.

Supporting span: extractor `answers[0]` preferred (char-precise); falls back to classifier `chunks[0]`.
WEAK spans shown only when `answer_score > 0.3` (otherwise span is unreliable).
