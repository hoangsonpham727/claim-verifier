# Claim Verifier — C# / .NET port

A C# / .NET 8 reimplementation of the Claim Verifier core engine and HTTP service
(the Python original lives in `../src/grounding` and `../service`). The Python
project is left fully functional; this folder is self-contained.

## Scope

- **Ported:** core engine (segment, chunk, rerank, extract, classify, ground,
  pipeline, parse, workspace store) and the HTTP service (`/health`, `/verify`,
  `/extract`, `/workspace/{id}`, `/index`).
- **Not ported (still Python):** the pytest suite, the Streamlit dev UI, and the
  `eval/` calibration harness.
- **Unchanged:** the Word add-in (`../addin`) — the service exposes the same JSON
  contract on the same default port (8000), so the add-in works against it as-is.

## Layout

```
Grounding.sln
src/Grounding/             class library — the engine
  Isaacus/IsaacusClient.cs   hand-written typed HttpClient (Isaacus ships no .NET SDK)
  Isaacus/Dtos.cs            request/response DTOs
  Models.cs                  API records (JSON names pinned to the add-in contract)
  TextUtil.cs                sentence segmentation + helpers (spaCy replacement)
  Segment.cs Chunk.cs Classify.cs Extract.cs Ground.cs Pipeline.cs Parse.cs Store.cs
src/Grounding.Service/     ASP.NET Core minimal-API service (Kestrel)
```

## Run

```bash
# ISAACUS_API_KEY is read from the repo ../.env automatically (or set it in the env)
dotnet run --project src/Grounding.Service --urls http://localhost:8000
curl http://localhost:8000/health           # {"status":"ok"}
```

For the Word add-in (which expects `https://localhost:8000`), run behind a dev
certificate or a TLS reverse proxy.

## Notable porting decisions

- **Isaacus client:** no .NET SDK exists, so model calls go through a typed
  `HttpClient` against `https://api.isaacus.com/v1` with bearer auth and Polly retry.
- **Verdict logic:** `Classify.VerdictFromScores` is a verbatim port of the
  calibrated 4-rule (+Rule 3.5) tree (`τ_low=0.55, τ_con=0.7, τ_sup=0.85,
  τ_inex=0.9`). Verified to produce byte-identical verdicts/confidences vs Python.
- **Sentence segmentation:** spaCy is unavailable on .NET, so `TextUtil` uses a
  deterministic, abbreviation-aware rule-based splitter. Boundaries are not
  guaranteed byte-identical to spaCy; the add-in's production path sends explicit
  claims (no segmentation), which is verified at full parity.
- **ILDGS backend** (`GROUNDING_BACKEND=ildgs`: enrich → graph → embed → route) is
  **not yet ported** — only the default `semchunk` backend is. `/index` is a no-op.
- **Chunking:** `Chunk` reimplements semchunk's recursive split-then-merge
  (word-count token counter, size 400).
