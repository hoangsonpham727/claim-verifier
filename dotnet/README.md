# Claim Verifier (C# / .NET)

A C# / .NET 8 service that grounds each cited claim in a legal document to its exact
supporting source passage and classifies whether the source supports, contradicts, is
silent on, or only weakly addresses it. It is built on the Isaacus legal-AI models and
consumed by a Microsoft Word add-in over a REST contract.

## Scope

- **Core engine:** segment, chunk, rerank, extract, classify, ground, pipeline, parse,
  and a per-document workspace store.
- **HTTP service:** `/health`, `/verify`, `/extract`, `/workspace/{id}`, `/index`.
- **Word add-in (`../addin`):** the service exposes the JSON contract the add-in expects
  on the default port (8000), so the add-in works against it directly.

## Layout

```
Grounding.sln
src/Grounding/             class library, the engine
  Isaacus/IsaacusClient.cs   typed HttpClient for the Isaacus models
  Isaacus/Dtos.cs            request/response DTOs
  Models.cs                  API records (JSON names pinned to the add-in contract)
  TextUtil.cs                sentence segmentation and text helpers
  Segment.cs Chunk.cs Classify.cs Extract.cs Ground.cs Pipeline.cs Parse.cs Store.cs
src/Grounding.Service/     ASP.NET Core minimal-API service (Kestrel)
```

## Run

```bash
# ISAACUS_API_KEY is read from ../.env automatically (or set it in the env)
dotnet run --project src/Grounding.Service --urls http://localhost:8000
curl http://localhost:8000/health           # {"status":"ok"}
```

For the Word add-in (which expects `https://localhost:8000`), run behind a dev
certificate or a TLS reverse proxy.

## Notable engineering decisions

- **Isaacus client:** model calls go through a hand-written typed `HttpClient` against
  `https://api.isaacus.com/v1` with bearer auth and Polly retry policies.
- **Verdict logic:** `Classify.VerdictFromScores` implements the calibrated 4-rule
  (plus Rule 3.5) decision tree (`τ_low=0.55, τ_con=0.7, τ_sup=0.85, τ_inex=0.9`).
- **Sentence segmentation:** `TextUtil` uses a deterministic, abbreviation-aware
  rule-based splitter. The add-in's production path sends explicit claims, so it does
  not depend on document segmentation.
- **Document ingestion:** PDF and DOCX text extraction via PdfPig and the Open XML SDK.
- **Chunking:** `Chunk` performs a recursive split-then-merge with a word-count token
  counter (size 400).
- **ILDGS backend** (`GROUNDING_BACKEND=ildgs`: enrich, graph, embed, route) is not yet
  implemented; only the default `semchunk` backend is. `/index` is a no-op.
