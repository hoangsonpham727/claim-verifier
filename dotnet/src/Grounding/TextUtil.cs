using System.Text;

namespace Grounding;

/// <summary>One detected sentence with char offsets back into the source text.</summary>
public readonly record struct Sentence(string Text, int Start, int End);

/// <summary>
/// Deterministic, dependency-free sentence segmentation + small text helpers.
///
/// Replaces spaCy (en_core_web_sm sentencizer) from the Python pipeline. spaCy is
/// not available on .NET and model downloads are not offline-deterministic, so we
/// use a rule-based, abbreviation-aware splitter. Boundaries will not be byte-
/// identical to spaCy; the citation-flagging gate (>=6 words + citation regex) and
/// the eval fixtures are the parity check (see plan Risks). The splitter sits
/// behind <see cref="SentenceSplitter"/> so a Catalyst-backed implementation can
/// be swapped in later without touching callers.
/// </summary>
public static class TextUtil
{
    public static double Round4(double x) => Math.Round(x, 4, MidpointRounding.ToEven);

    /// <summary>Collapse all runs of whitespace to single spaces and lowercase.</summary>
    public static string Norm(string text) =>
        string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)).ToLowerInvariant();

    private static readonly HashSet<string> Abbreviations = new(StringComparer.OrdinalIgnoreCase)
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "v", "etc", "inc",
        "ltd", "llc", "llp", "co", "corp", "no", "nos", "art", "arts", "sec", "secs",
        "para", "paras", "cl", "fig", "figs", "ex", "exs", "ch", "chs", "p", "pp",
        "approx", "e.g", "i.e", "cf", "al", "u.s", "u.k", "ph.d", "viz",
    };

    /// <summary>
    /// Split text into sentences with char offsets. Treats . ! ? as terminators,
    /// guarding common legal/Latin abbreviations, decimals, and enumerated refs.
    /// </summary>
    public static IReadOnlyList<Sentence> SplitSentences(string text)
    {
        var sentences = new List<Sentence>();
        if (string.IsNullOrEmpty(text))
            return sentences;

        int start = 0;
        int n = text.Length;
        for (int i = 0; i < n; i++)
        {
            char c = text[i];
            if (c != '.' && c != '!' && c != '?')
                continue;

            // Consume any run of terminators / closing quotes / brackets.
            int end = i + 1;
            while (end < n && (text[end] == '.' || text[end] == '!' || text[end] == '?'
                               || text[end] == '"' || text[end] == '\'' || text[end] == ')'
                               || text[end] == ']'))
                end++;

            if (!IsBoundary(text, i, end, n))
                continue;

            // Boundary requires following whitespace (or end of text).
            if (end < n && !char.IsWhiteSpace(text[end]))
                continue;

            AddSentence(sentences, text, start, end);
            // Skip the whitespace; next sentence starts at the next non-space.
            int next = end;
            while (next < n && char.IsWhiteSpace(text[next]))
                next++;
            start = next;
            i = next - 1;
        }

        if (start < n)
            AddSentence(sentences, text, start, n);

        return sentences;
    }

    private static bool IsBoundary(string text, int dotIdx, int end, int n)
    {
        if (text[dotIdx] != '.')
            return true; // ! and ? are unambiguous terminators here

        // Decimal number: digit . digit
        if (dotIdx > 0 && dotIdx + 1 < n && char.IsDigit(text[dotIdx - 1]) && char.IsDigit(text[dotIdx + 1]))
            return false;

        // Single capital initial: "J. Smith"
        if (dotIdx >= 1 && char.IsUpper(text[dotIdx - 1])
            && (dotIdx < 2 || !char.IsLetter(text[dotIdx - 2])))
            return false;

        // Trailing word before the dot is a known abbreviation.
        int ws = dotIdx;
        while (ws > 0 && !char.IsWhiteSpace(text[ws - 1]))
            ws--;
        string token = text.Substring(ws, dotIdx - ws).Trim('(', '[', '"', '\'');
        if (token.Length > 0 && Abbreviations.Contains(token))
            return false;

        return true;
    }

    private static void AddSentence(List<Sentence> sentences, string text, int start, int end)
    {
        // Trim leading/trailing whitespace while keeping accurate offsets.
        int s = start, e = end;
        while (s < e && char.IsWhiteSpace(text[s])) s++;
        while (e > s && char.IsWhiteSpace(text[e - 1])) e--;
        if (e > s)
            sentences.Add(new Sentence(text.Substring(s, e - s), s, e));
    }

    /// <summary>Count non-space, non-punctuation word tokens (mirrors spaCy gate).</summary>
    public static int WordCount(string text)
    {
        int count = 0;
        foreach (var tok in text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            bool hasWordChar = false;
            foreach (char ch in tok)
            {
                if (char.IsLetterOrDigit(ch)) { hasWordChar = true; break; }
            }
            if (hasWordChar) count++;
        }
        return count;
    }
}
