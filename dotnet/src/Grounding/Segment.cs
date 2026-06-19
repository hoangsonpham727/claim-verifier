using System.Text.RegularExpressions;

namespace Grounding;

/// <summary>
/// Segment a document into sentences; flag those carrying a citation marker.
/// Port of src/grounding/segment.py. The citation regex is copied verbatim.
/// </summary>
public static partial class Segment
{
    private const int MinClaimWords = 6; // bare "Section 1." headers have < 6 words

    // Matches common legal citation markers across two document families:
    // (A) Legal briefs / case law; (B) Contracts / NDAs. See segment.py for the audit.
    [GeneratedRegex(
        @"\[\d+\]" +                               // footnote refs [1], [23]
        @"|s\.\s*\d+" +                            // statutory: s.2, s. 47
        @"|\(\d{4}\)" +                            // year in parens (2023)
        @"|\bv\b\.?\s+[A-Z]" +                     // case name: Smith v Jones
        @"|\bsee\b" +                              // see ...
        @"|\bcf\b\.?" +                            // cf.
        @"|\bper\b\s+[A-Z]" +                      // per Lord ...
        @"|\bibid\b" +                             // ibid
        @"|\bsupra\b" +                            // supra
        @"|\binfra\b" +                            // infra
        @"|\b(?:Section|Article|Clause|Para(?:graph)?)\s+\d+" + // Section 4, Article 2
        @"|\bpursuant\s+to\b" +                    // pursuant to
        @"|\bas\s+(?:defined|set\s+forth|provided)\s+in\b" +   // as defined in, ...
        @"|(?<!\A)(?<!\n)\([a-z]{1,3}\)",          // inline (a)/(i) — NOT at sentence start
        RegexOptions.IgnoreCase)]
    private static partial Regex CitationRegex();

    /// <summary>Split document into sentence-level claims; flag those with citation markers.</summary>
    public static List<Claim> SegmentClaims(string document)
    {
        var claims = new List<Claim>();
        int i = 0;
        foreach (var sent in TextUtil.SplitSentences(document))
        {
            string text = sent.Text.Trim();
            if (text.Length == 0) { i++; continue; }
            int wordCount = TextUtil.WordCount(text);
            bool hasCitation = wordCount >= MinClaimWords && CitationRegex().IsMatch(text);
            claims.Add(new Claim(Text: text, HasCitation: hasCitation, RangeRef: i.ToString()));
            i++;
        }
        return claims;
    }
}
