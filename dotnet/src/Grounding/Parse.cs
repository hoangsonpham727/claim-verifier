using System.Text;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using UglyToad.PdfPig;

namespace Grounding;

/// <summary>
/// Shared document-text extraction. Port of src/grounding/parse.py — turns an
/// uploaded PDF / DOCX / TXT into plain text. PyMuPDF → PdfPig, python-docx →
/// Open XML SDK.
/// </summary>
public static class Parse
{
    /// <summary>Extract plain text from a PDF, DOCX, or text file (by extension).</summary>
    public static string ExtractText(string filename, byte[] raw)
    {
        string name = filename.ToLowerInvariant();
        if (name.EndsWith(".pdf"))
        {
            using var doc = PdfDocument.Open(raw);
            return string.Join("\n\n", doc.GetPages().Select(p => p.Text));
        }
        if (name.EndsWith(".docx"))
        {
            using var stream = new MemoryStream(raw);
            using var word = WordprocessingDocument.Open(stream, false);
            var body = word.MainDocumentPart?.Document?.Body;
            if (body is null) return "";
            var lines = body.Descendants<Paragraph>()
                .Select(p => p.InnerText)
                .Where(t => t.Trim().Length > 0);
            return string.Join("\n", lines);
        }
        // Fall back to raw UTF-8 (txt, md, anything else).
        return Encoding.UTF8.GetString(raw);
    }

    /// <summary>Return the page count of a PDF, or null if it isn't a readable PDF.</summary>
    public static int? CountPdfPages(byte[] raw)
    {
        try
        {
            using var doc = PdfDocument.Open(raw);
            return doc.NumberOfPages;
        }
        catch
        {
            return null;
        }
    }
}
