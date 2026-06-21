# Claim Verifier

Bind every cited claim in a legal document to the exact supporting passage in a
source, and judge whether the source **supports**, **contradicts**, is **silent
on**, or only **weakly** addresses it, each with a calibrated confidence and the
precise supporting line. Built on the [Isaacus](https://isaacus.com) legal-AI
models and surfaced through a Microsoft Word add-in.

This repository holds two interchangeable backend implementations of the same
service plus the shared client and assets they have in common.

## Layout

```
python/      Python implementation (FastAPI engine, eval harness, Streamlit dev UI)
dotnet/      C# / .NET 8 implementation (ASP.NET Core engine + service)
addin/       Word task-pane add-in (HTML/CSS/JS) — the client for either backend
shared/      Language-neutral assets: API contract (NOTES_api.md), sample documents
.env         Isaacus credentials, read by both backends (see .env.example)
cache/       Runtime cache (embeddings, workspaces); override with CACHE_DIR
```

Each backend is self-contained and idiomatic to its stack, exposes the same JSON
contract on port 8000, and is documented in its own README:

- [`python/README.md`](python/README.md)
- [`dotnet/README.md`](dotnet/README.md)

## Quick start

Pick one backend; both serve the add-in identically.

```bash
# C# / .NET
dotnet run --project dotnet/src/Grounding.Service --urls http://localhost:8000

# Python
cd python && uvicorn service.app:app --reload --port 8000
```

```bash
curl http://localhost:8000/health     # {"status":"ok"}
```

## What it does

For each cited claim the system: segments the document and flags cited sentences,
chunks and reranks each source to find the most relevant passages, extracts the
exact supporting line, and classifies support with two independent classifier
calls. It returns a verdict, confidence, supporting span, and ranked evidence.
See [`shared/NOTES_api.md`](shared/NOTES_api.md) for the model API field shapes and
the calibrated verdict logic.
