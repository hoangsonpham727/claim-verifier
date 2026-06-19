using System.Text;
using System.Text.Json;

namespace Grounding;

/// <summary>
/// Per-document workspace store. Port of the workspace portion of
/// src/grounding/store.py (the embedding/enrichment content-addressed caches
/// belong to the deferred ILDGS backend). Layout under CACHE_DIR (default
/// ./cache): workspaces/&lt;id&gt;.json.
/// </summary>
public sealed class Store
{
    private static readonly JsonSerializerOptions JsonOpts = new(JsonSerializerDefaults.Web);

    private readonly string _wsDir;

    public Store(string? cacheDir = null)
    {
        cacheDir ??= Environment.GetEnvironmentVariable("CACHE_DIR") ?? "cache";
        _wsDir = Path.Combine(cacheDir, "workspaces");
    }

    public Workspace? LoadWorkspace(string workspaceId)
    {
        var path = Path.Combine(_wsDir, SafeId(workspaceId) + ".json");
        if (!File.Exists(path))
            return null;
        try
        {
            return JsonSerializer.Deserialize<Workspace>(File.ReadAllText(path), JsonOpts);
        }
        catch
        {
            return null;
        }
    }

    public void SaveWorkspace(string workspaceId, Workspace ws)
    {
        var path = Path.Combine(_wsDir, SafeId(workspaceId) + ".json");
        AtomicWriteText(path, JsonSerializer.Serialize(ws, JsonOpts));
    }

    private static void AtomicWriteText(string path, string text)
    {
        var dir = Path.GetDirectoryName(path)!;
        Directory.CreateDirectory(dir);
        var tmp = Path.Combine(dir, Path.GetRandomFileName() + ".tmp");
        try
        {
            File.WriteAllText(tmp, text, new UTF8Encoding(false));
            File.Move(tmp, path, overwrite: true);
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    /// <summary>Keep workspace ids to a safe filename charset (defends against traversal).</summary>
    private static string SafeId(string workspaceId)
    {
        var sb = new StringBuilder();
        foreach (char c in workspaceId)
        {
            if (char.IsLetterOrDigit(c) || c == '-' || c == '_')
                sb.Append(c);
            if (sb.Length >= 128) break;
        }
        return sb.Length > 0 ? sb.ToString() : "default";
    }
}
