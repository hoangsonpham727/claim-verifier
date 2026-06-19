using System.Text.Json.Serialization;

namespace Grounding.Isaacus;

// Request/response DTOs for the Isaacus REST API. Field names match the API JSON
// (snake_case) as documented in NOTES_api.md and the Python SDK type stubs.

// ── Rerankings: POST /rerankings ──────────────────────────────────────────────
public sealed record RerankRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("query")] string Query,
    [property: JsonPropertyName("texts")] IReadOnlyList<string> Texts);

public sealed record RerankResult(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("score")] double Score);

public sealed record RerankResponse(
    [property: JsonPropertyName("results")] List<RerankResult> Results);

// ── Universal classifier: POST /classifications/universal ─────────────────────
public sealed record ClassifyRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("query")] string Query,
    [property: JsonPropertyName("texts")] IReadOnlyList<string> Texts);

public sealed record ClassificationChunk(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("start")] int Start,
    [property: JsonPropertyName("end")] int End,
    [property: JsonPropertyName("score")] double Score,
    [property: JsonPropertyName("text")] string Text);

public sealed record Classification(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("score")] double Score,
    [property: JsonPropertyName("chunks")] List<ClassificationChunk>? Chunks);

public sealed record ClassifyResponse(
    [property: JsonPropertyName("classifications")] List<Classification> Classifications);

// ── Extractive QA: POST /extractions/qa ───────────────────────────────────────
public sealed record QaRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("query")] string Query,
    [property: JsonPropertyName("texts")] IReadOnlyList<string> Texts,
    [property: JsonPropertyName("top_k")] int TopK,
    [property: JsonPropertyName("ignore_inextractability")] bool IgnoreInextractability);

public sealed record ExtractionAnswer(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("start")] int Start,
    [property: JsonPropertyName("end")] int End,
    [property: JsonPropertyName("score")] double Score);

public sealed record Extraction(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("answers")] List<ExtractionAnswer>? Answers,
    [property: JsonPropertyName("inextractability_score")] double InextractabilityScore);

public sealed record QaResponse(
    [property: JsonPropertyName("extractions")] List<Extraction> Extractions);
