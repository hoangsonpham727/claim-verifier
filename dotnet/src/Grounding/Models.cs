using System.Text.Json.Serialization;

namespace Grounding;

// Mirrors src/grounding/models.py. JSON property names are pinned to the exact
// snake_case the Word add-in (addin/taskpane.js) and the Python service emit, so
// this service is a drop-in replacement.

public sealed record Source(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("text")] string Text);

public sealed record Claim(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("has_citation")] bool HasCitation = false,
    [property: JsonPropertyName("range_ref")] string? RangeRef = null,
    [property: JsonPropertyName("source_id")] string? SourceId = null);

public sealed record Candidate(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("source_id")] string SourceId,
    [property: JsonPropertyName("chunk_index")] int ChunkIndex,
    [property: JsonPropertyName("score")] double Score);

public sealed record SupportingSpan(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("start")] int Start,
    [property: JsonPropertyName("end")] int End,
    [property: JsonPropertyName("score")] double Score);

public sealed record RankedPassage(
    [property: JsonPropertyName("source_id")] string SourceId,
    [property: JsonPropertyName("score")] double Score,
    [property: JsonPropertyName("passage")] string Passage,
    [property: JsonPropertyName("line")] string? Line = null,
    [property: JsonPropertyName("line_start")] int? LineStart = null,
    [property: JsonPropertyName("line_end")] int? LineEnd = null);

public sealed record EvidenceItem(
    [property: JsonPropertyName("source_id")] string SourceId,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("relation")] string Relation = "seed",
    [property: JsonPropertyName("label")] string Label = "",
    [property: JsonPropertyName("seg_type")] string? SegType = null,
    [property: JsonPropertyName("start")] int? Start = null,
    [property: JsonPropertyName("end")] int? End = null,
    [property: JsonPropertyName("score")] double? Score = null);

public sealed record ClaimResult(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("verdict")] string Verdict,
    [property: JsonPropertyName("confidence")] double Confidence,
    [property: JsonPropertyName("range_ref")] string? RangeRef = null,
    [property: JsonPropertyName("span")] SupportingSpan? Span = null,
    [property: JsonPropertyName("source_id")] string? SourceId = null,
    [property: JsonPropertyName("evidence")] List<EvidenceItem>? Evidence = null,
    [property: JsonPropertyName("passages")] List<RankedPassage>? Passages = null);

public sealed record VerifyRequest(
    [property: JsonPropertyName("document")] string Document = "",
    [property: JsonPropertyName("claims")] List<string>? Claims = null,
    [property: JsonPropertyName("sources")] List<Source>? Sources = null);

public sealed record VerifySummary(
    [property: JsonPropertyName("total_cited")] int TotalCited,
    [property: JsonPropertyName("supported")] int Supported,
    [property: JsonPropertyName("contradicted")] int Contradicted,
    [property: JsonPropertyName("unaddressed")] int Unaddressed,
    [property: JsonPropertyName("weak")] int Weak = 0);

public sealed record VerifyResponse(
    [property: JsonPropertyName("claims")] List<ClaimResult> Claims,
    [property: JsonPropertyName("summary")] VerifySummary Summary);

public sealed record Workspace
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("document_hash")] public string DocumentHash { get; set; } = "";
    [JsonPropertyName("sources")] public List<Source> Sources { get; set; } = new();
    [JsonPropertyName("results")] public Dictionary<string, ClaimResult> Results { get; set; } = new();
    [JsonPropertyName("updated_at")] public double UpdatedAt { get; set; } = 0.0;
}
