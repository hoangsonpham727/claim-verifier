using Grounding.Isaacus;

namespace Grounding;

/// <summary>
/// Orchestration for claim verification. Port of src/grounding/pipeline.py
/// (semchunk backend; the opt-in ILDGS backend is deferred). A retrieval backend
/// is built once per request and shared across claims; each cited claim is
/// grounded via the unified core (retrieve → extract → classify → verdict).
/// </summary>
public sealed class Pipeline
{
    // The main document is added as a source under this reserved id so its OWN
    // clauses are available for grounding (a claim is never grounded by its own clause).
    public const string DocSourceId = "This document";

    private readonly IsaacusClient _client;
    private readonly Grounder _grounder;

    public Pipeline(IsaacusClient client, Grounder grounder)
    {
        _client = client;
        _grounder = grounder;
    }

    private IRetriever BuildRetriever(string document, IReadOnlyList<Source> sources)
    {
        var all = new List<Source>(sources);
        bool hasDoc = !string.IsNullOrWhiteSpace(document);
        if (hasDoc)
            all.Add(new Source(DocSourceId, document));
        return new SemchunkRetriever(_client, all, hasDoc ? DocSourceId : null);
    }

    /// <summary>
    /// Verify claims against the sources plus the document itself. Claims come from
    /// the explicit list when given; otherwise the document is segmented and its
    /// cited sentences are used.
    /// </summary>
    public async Task<VerifyResponse> VerifyAsync(
        string document = "", IReadOnlyList<Source>? sources = null,
        IReadOnlyList<string>? claims = null, bool includePassages = false,
        CancellationToken ct = default)
    {
        sources ??= Array.Empty<Source>();

        List<Claim> cited;
        if (claims is not null)
        {
            cited = claims
                .Select((c, i) => (c, i))
                .Where(t => !string.IsNullOrWhiteSpace(t.c))
                .Select(t => new Claim(t.c, HasCitation: true, RangeRef: t.i.ToString()))
                .ToList();
        }
        else
        {
            cited = Segment.SegmentClaims(document).Where(c => c.HasCitation).ToList();
        }

        if (cited.Count == 0)
        {
            return new VerifyResponse(
                new List<ClaimResult>(),
                new VerifySummary(0, 0, 0, 0, 0));
        }

        var retriever = BuildRetriever(document, sources);

        var tasks = cited.Select(async claim =>
        {
            var g = await _grounder.GroundAsync(
                claim.Text, retriever, Grounder.RerankK,
                ClassifierInput.Source, withLine: true, ct: ct);
            return Grounder.ToClaimResult(claim, g, includePassages);
        });
        var results = (await Task.WhenAll(tasks)).ToList();

        var summary = new VerifySummary(
            TotalCited: results.Count,
            Supported: results.Count(r => r.Verdict == "supported"),
            Contradicted: results.Count(r => r.Verdict == "contradicted"),
            Unaddressed: results.Count(r => r.Verdict == "unaddressed"),
            Weak: results.Count(r => r.Verdict == "weak"));

        return new VerifyResponse(results, summary);
    }
}
