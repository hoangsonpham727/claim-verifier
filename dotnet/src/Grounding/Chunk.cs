namespace Grounding;

/// <summary>
/// Split a source document into passages for reranking. Port of
/// src/grounding/chunk.py (semchunk). Reimplements semchunk's recursive
/// split-on-best-separator then merge-to-fill strategy with a word-count token
/// counter and chunk_size=400 words (~512 tokens for legal text).
/// </summary>
public static class Chunk
{
    private const int ChunkSize = 400;

    // Most-semantic → least-semantic separators (semchunk hierarchy).
    private static readonly string[] Separators =
        { "\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " " };

    private static int Tokens(string s) =>
        s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;

    /// <summary>Split source text into semantic chunks (never empty).</summary>
    public static List<string> ChunkSource(string sourceText)
    {
        var chunks = Split(sourceText, ChunkSize);
        chunks = chunks.Where(c => c.Trim().Length > 0).ToList();
        return chunks.Count > 0 ? chunks : new List<string> { sourceText };
    }

    private static List<string> Split(string text, int size)
    {
        if (Tokens(text) <= size)
            return new List<string> { text };

        // Pick the most-semantic separator that actually divides this text.
        string? sep = null;
        string[]? parts = null;
        foreach (var s in Separators)
        {
            var p = text.Split(s);
            if (p.Length > 1) { sep = s; parts = p; break; }
        }

        if (sep is null || parts is null)
            return HardSplit(text, size); // single long token run

        var result = new List<string>();
        string current = "";
        for (int i = 0; i < parts.Length; i++)
        {
            string part = parts[i];
            string candidate = current.Length == 0 ? part : current + sep + part;
            if (Tokens(candidate) <= size)
            {
                current = candidate;
                continue;
            }

            if (current.Length > 0)
            {
                result.Add(current);
                current = "";
            }

            if (Tokens(part) > size)
                result.AddRange(Split(part, size)); // recurse on the oversized part
            else
                current = part;
        }
        if (current.Length > 0)
            result.Add(current);

        return result;
    }

    private static List<string> HardSplit(string text, int size)
    {
        // Fallback: split a long whitespace-free run by character budget.
        var result = new List<string>();
        int approxChars = Math.Max(1, size * 6); // ~6 chars/word
        for (int i = 0; i < text.Length; i += approxChars)
            result.Add(text.Substring(i, Math.Min(approxChars, text.Length - i)));
        return result;
    }
}
