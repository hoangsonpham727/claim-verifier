using Grounding.Isaacus;

namespace Grounding;

/// <summary>
/// Extract the exact supporting span from a passage using extractive QA.
/// Port of src/grounding/extract.py. Pass full source texts — the extractor
/// chunks internally.
/// </summary>
public sealed class Extract
{
    private readonly IsaacusClient _client;

    public Extract(IsaacusClient client) => _client = client;

    /// <summary>
    /// Return (best_span, inextractability_score) for the top passage.
    /// Set relevanceConfirmed=true once the reranker has confirmed relevance
    /// (sets ignore_inextractability=true so an answer is always returned).
    /// </summary>
    public async Task<(SupportingSpan? Span, double Inextractability)> ExtractSpanAsync(
        string claim, IReadOnlyList<string> passages, int topK = 1,
        bool relevanceConfirmed = false, CancellationToken ct = default)
    {
        if (passages.Count == 0)
            return (null, 1.0);

        var resp = await _client.ExtractQaAsync(claim, passages, topK, relevanceConfirmed, ct);
        var best = resp.Extractions.Count > 0 ? resp.Extractions[0] : null;
        if (best is null)
            return (null, 1.0);

        double inextractability = best.InextractabilityScore;
        if (best.Answers is not { Count: > 0 } answers)
            return (null, inextractability);

        var ans = answers[0];
        var span = new SupportingSpan(ans.Text, ans.Start, ans.End, TextUtil.Round4(ans.Score));
        return (span, inextractability);
    }
}
