using Grounding.Isaacus;

namespace Grounding;

/// <summary>
/// Classify whether passages support a claim; derive verdict + confidence.
/// Port of src/grounding/classify.py — the 4-rule (+Rule 3.5) sequential decision
/// tree. p_support and p_contra are INDEPENDENT classifier calls (claim and its
/// negation); they do not sum to 1. Thresholds are the shipped calibrated values.
/// </summary>
public sealed class Classify
{
    // ── Thresholds (calibrated in eval/calibrate.py against ContractNLI dev) ──
    public const double TauLow = 0.55;    // below this for BOTH scores → UNADDRESSED
    public const double TauCon = 0.7;     // p_contra threshold for CONTRADICTED
    public const double TauSup = 0.85;    // p_support threshold for SUPPORTED (high)
    public const double TauInex = 0.9;    // max inextractability for SUPPORTED
    public const double TauMargin = 0.0;  // required (p_contra − p_support) gap
    public const double TauUnaddr = 0.7;  // inextract above this (+ low support) → UNADDRESSED

    private readonly IsaacusClient _client;

    public Classify(IsaacusClient client) => _client = client;

    private static string Negate(string claim) => $"It is not the case that: {claim}";

    public readonly record struct Scores(double PSupport, double PContra, SupportingSpan? Span);

    /// <summary>
    /// Return the raw signals (p_support, p_contra, best_chunk_span) with NO
    /// verdict rules applied (mirrors classify_scores).
    /// </summary>
    public async Task<Scores> ClassifyScoresAsync(
        string claim, IReadOnlyList<string> passages, CancellationToken ct = default)
    {
        if (passages.Count == 0)
            return new Scores(0.0, 0.0, null);

        // p_support: does the source establish the claim as stated?
        var respSup = await _client.ClassifyUniversalAsync(claim, passages, ct);
        var bestSup = respSup.Classifications.Count > 0 ? respSup.Classifications[0] : null;
        double pSupport = bestSup?.Score ?? 0.0;

        // p_contra: does the source establish that the claim is false?
        var respCon = await _client.ClassifyUniversalAsync(Negate(claim), passages, ct);
        var bestCon = respCon.Classifications.Count > 0 ? respCon.Classifications[0] : null;
        double pContra = bestCon?.Score ?? 0.0;

        SupportingSpan? span = null;
        if (bestSup?.Chunks is { Count: > 0 } chunks)
        {
            var top = chunks[0];
            span = new SupportingSpan(top.Text, top.Start, top.End, TextUtil.Round4(top.Score));
        }

        return new Scores(pSupport, pContra, span);
    }

    /// <summary>
    /// Pure function: apply the decision tree to cached scores (mirrors
    /// verdict_from_scores). No API calls.
    /// </summary>
    public static (string Verdict, double Confidence) VerdictFromScores(
        double pSupport, double pContra, double inextract = 1.0,
        double tauLow = TauLow, double tauCon = TauCon, double tauSup = TauSup,
        double tauInex = TauInex, double tauMargin = TauMargin, double tauUnaddr = TauUnaddr)
    {
        // Rule 1 — Source is silent: neither support nor contradiction signal.
        if (Math.Max(pSupport, pContra) < tauLow)
            return ("unaddressed", TextUtil.Round4(1.0 - Math.Max(pSupport, pContra)));

        // Rule 2 — Contradiction: source actively says the opposite.
        if (pContra > tauCon && (pContra - pSupport) > tauMargin)
            return ("contradicted", TextUtil.Round4(pContra * (1.0 - pSupport)));

        // Rule 3 — Support: NLI says yes AND extractor can locate the span.
        if (pSupport > tauSup && inextract < tauInex)
            return ("supported", TextUtil.Round4(pSupport * (1.0 - inextract)));

        // Rule 3.5 — Silence by inextractability.
        if (inextract > tauUnaddr && pSupport < tauSup)
            return ("unaddressed", TextUtil.Round4(inextract * (1.0 - pSupport)));

        // Rule 4 — Weak: relevant source, but signal is mixed or below threshold.
        return ("weak", TextUtil.Round4(Math.Max(pSupport, pContra)));
    }
}
