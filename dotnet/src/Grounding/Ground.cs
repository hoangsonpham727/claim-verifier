using Grounding.Isaacus;

namespace Grounding;

/// <summary>What text the verdict classifier reads.</summary>
public enum ClassifierInput { Source, Candidates }

/// <summary>Retrieval backend: build once per request, reuse across claims.</summary>
public interface IRetriever
{
    Task<List<Candidate>> CandidatesAsync(string claim, int topK, CancellationToken ct);
    string SourceText(string sourceId);
}

/// <summary>
/// Chunk each source once; rerank the union of chunks per claim. Port of
/// SemchunkRetriever in src/grounding/ground.py. A claim is never grounded by the
/// chunk that restates it verbatim, nor by any later chunk of the same document.
/// </summary>
public sealed class SemchunkRetriever : IRetriever
{
    private readonly IsaacusClient _client;
    private readonly Dictionary<string, string> _text = new();
    private readonly string? _selfId;
    // (source_id, chunk_index, text) for every chunk, in document order.
    private readonly List<(string Sid, int Idx, string Text)> _chunks = new();

    public SemchunkRetriever(IsaacusClient client, IEnumerable<Source> sources,
        string? selfExcludeSourceId = null)
    {
        _client = client;
        _selfId = selfExcludeSourceId;
        foreach (var s in sources)
        {
            _text[s.Id] = s.Text;
            int i = 0;
            foreach (var ch in Chunk.ChunkSource(s.Text))
                _chunks.Add((s.Id, i++, ch));
        }
    }

    public string SourceText(string sourceId) => _text.GetValueOrDefault(sourceId, "");

    private List<(string Sid, int Idx, string Text)> Eligible(string claim)
    {
        if (_selfId is null)
            return _chunks;
        string norm = TextUtil.Norm(claim);
        int? cutoff = null;
        foreach (var (sid, idx, ch) in _chunks)
        {
            if (sid == _selfId && norm.Length > 0 && TextUtil.Norm(ch).Contains(norm))
                cutoff = cutoff is null ? idx : Math.Min(cutoff.Value, idx);
        }
        if (cutoff is null)
            return _chunks;
        return _chunks.Where(c => !(c.Sid == _selfId && c.Idx >= cutoff.Value)).ToList();
    }

    public async Task<List<Candidate>> CandidatesAsync(string claim, int topK, CancellationToken ct)
    {
        var pool = Eligible(claim);
        if (pool.Count == 0)
            return new List<Candidate>();
        var texts = pool.Select(c => c.Text).ToList();
        var resp = await _client.RerankAsync(claim, texts, ct);
        var outp = new List<Candidate>();
        foreach (var r in resp.Results.Take(topK))
        {
            var (sid, idx, ch) = pool[r.Index];
            outp.Add(new Candidate(ch, sid, idx, TextUtil.Round4(r.Score)));
        }
        return outp;
    }
}

/// <summary>Result of grounding one claim (mirrors the Grounding dataclass).</summary>
public sealed record Grounding(
    List<Candidate> Candidates,
    Candidate? Seed,
    double Relevance,
    SupportingSpan? Span,
    double Inextract,
    double AnswerScore,
    double PSupport,
    double PContra,
    SupportingSpan? ClsSpan,
    string Verdict,
    double Confidence);

/// <summary>
/// Unified per-claim grounding core: retrieve → extract → classify → verdict.
/// Port of ground.py (semchunk path; ILDGS backend deferred).
/// </summary>
public sealed class Grounder
{
    public const int RerankK = 3;

    private readonly IsaacusClient _client;
    private readonly Classify _classify;
    private readonly Extract _extract;

    public Grounder(IsaacusClient client, Classify classify, Extract extract)
    {
        _client = client;
        _classify = classify;
        _extract = extract;
    }

    /// <summary>Pick the seed sentence most relevant to the claim, via the reranker.</summary>
    private async Task<SupportingSpan?> BestLineAsync(string claim, string passage, CancellationToken ct)
    {
        var sents = TextUtil.SplitSentences(passage)
            .Select(s => s.Text.Trim())
            .Where(t => t.Length > 15)
            .ToList();
        if (sents.Count == 0)
            return null;
        var resp = await _client.RerankAsync(claim, sents, ct);
        if (resp.Results.Count == 0)
            return null;
        var top = resp.Results[0];
        string sent = sents[top.Index];
        int start = passage.IndexOf(sent, StringComparison.Ordinal);
        if (start < 0) start = 0;
        return new SupportingSpan(sent, start, start + sent.Length, TextUtil.Round4(top.Score));
    }

    public async Task<Grounding> GroundAsync(
        string claim, IRetriever retriever, int topK = RerankK,
        ClassifierInput classifierInput = ClassifierInput.Source,
        bool withLine = false, CancellationToken ct = default)
    {
        var candidates = await retriever.CandidatesAsync(claim, topK, ct);
        if (candidates.Count == 0)
            return new Grounding(new(), null, 0.0, null, 1.0, 0.0, 0.0, 0.0, null, "unaddressed", 0.0);

        var seed = candidates[0];
        double relevance = seed.Score;
        string srcText = retriever.SourceText(seed.SourceId);
        if (srcText.Length == 0) srcText = seed.Text;

        // inextract feeds the verdict (Rule 3 guard); always from the extractor.
        var (span, inextract) = await _extract.ExtractSpanAsync(
            claim, new[] { seed.Text }, relevanceConfirmed: true, ct: ct);
        double answerScore = span?.Score ?? 0.0;

        // Displayed quote: extractor span is unreliable on declarative claims, so
        // for the UI use the reranked best sentence of the seed instead.
        if (withLine)
        {
            var line = await BestLineAsync(claim, seed.Text, ct);
            if (line is not null)
            {
                span = line;
                answerScore = line.Score;
            }
        }

        var candTexts = candidates.Select(c => c.Text).ToList();
        var clsTexts = classifierInput == ClassifierInput.Source
            ? new List<string> { srcText }
            : candTexts;
        var scores = await _classify.ClassifyScoresAsync(claim, clsTexts, ct);

        var (verdict, confidence) = Classify.VerdictFromScores(
            scores.PSupport, scores.PContra, inextract);

        return new Grounding(candidates, seed, relevance, span, inextract, answerScore,
            scores.PSupport, scores.PContra, scores.Span, verdict, confidence);
    }

    /// <summary>Build the API ClaimResult from a Grounding (mirrors to_claim_result).</summary>
    public static ClaimResult ToClaimResult(Claim claim, Grounding g, bool includePassages = false)
    {
        if (g.Seed is null)
            return new ClaimResult(claim.Text, "unaddressed", 0.0, RangeRef: claim.RangeRef);

        SupportingSpan? finalSpan;
        if (g.Verdict == "unaddressed")
            finalSpan = null;
        else if (g.Verdict == "weak")
            finalSpan = g.AnswerScore > 0.3 ? g.Span : g.ClsSpan;
        else
            finalSpan = g.Span ?? g.ClsSpan;

        var evidence = new List<EvidenceItem>
        {
            new(g.Seed.SourceId, g.Seed.Text, Relation: "seed", Score: g.Seed.Score),
        };

        var passages = new List<RankedPassage>();
        if (includePassages)
        {
            for (int i = 0; i < g.Candidates.Count; i++)
            {
                var c = g.Candidates[i];
                bool isSeed = i == 0;
                passages.Add(new RankedPassage(
                    c.SourceId, c.Score, c.Text,
                    Line: isSeed && g.Span is not null ? g.Span.Text : null,
                    LineStart: isSeed && g.Span is not null ? g.Span.Start : null,
                    LineEnd: isSeed && g.Span is not null ? g.Span.End : null));
            }
        }

        return new ClaimResult(
            claim.Text, g.Verdict, g.Confidence, RangeRef: claim.RangeRef,
            Span: finalSpan, SourceId: g.Seed.SourceId, Evidence: evidence, Passages: passages);
    }
}
