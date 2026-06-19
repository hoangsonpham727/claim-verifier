using System.Net.Http.Headers;
using Grounding;
using Grounding.Isaacus;
using Microsoft.AspNetCore.Http.Features;
using Polly;
using Polly.Extensions.Http;

var builder = WebApplication.CreateBuilder(args);

// Load ISAACUS_API_KEY / ISAACUS_BASE_URL from a repo .env if not already set.
LoadDotEnv(builder.Environment.ContentRootPath);

string apiKey = Environment.GetEnvironmentVariable("ISAACUS_API_KEY")
    ?? throw new InvalidOperationException("ISAACUS_API_KEY not set in environment / .env");
string baseUrl = Environment.GetEnvironmentVariable("ISAACUS_BASE_URL") ?? "https://api.isaacus.com/v1";

// ── Isaacus typed HttpClient (vendor seam) with transient-fault retry ──────────
builder.Services.AddHttpClient<IsaacusClient>(http =>
{
    http.BaseAddress = new Uri(baseUrl.EndsWith('/') ? baseUrl : baseUrl + "/");
    http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
    http.Timeout = TimeSpan.FromSeconds(120);
})
.AddPolicyHandler(HttpPolicyExtensions
    .HandleTransientHttpError()
    .WaitAndRetryAsync(3, attempt => TimeSpan.FromMilliseconds(250 * Math.Pow(2, attempt))));

builder.Services.AddSingleton<Classify>();
builder.Services.AddSingleton<Extract>();
builder.Services.AddSingleton<Grounder>();
builder.Services.AddSingleton<Pipeline>();
builder.Services.AddSingleton(new Store());

// Allow all origins for the Word add-in (Office.js runs from ms-appx:// or a
// localhost dev server; CORS must be open during development).
builder.Services.AddCors(o => o.AddDefaultPolicy(p =>
    p.AllowAnyOrigin().WithMethods("GET", "POST", "PUT").AllowAnyHeader()));

builder.Services.Configure<FormOptions>(o => o.MultipartBodyLengthLimit = 100 * 1024 * 1024);

// Same default port as the Python service (uvicorn ... --port 8000).
if (Environment.GetEnvironmentVariable("ASPNETCORE_URLS") is null && !args.Any(a => a.StartsWith("--urls")))
    builder.WebHost.UseUrls("http://localhost:8000");

var app = builder.Build();
app.UseCors();

app.MapGet("/health", () => Results.Ok(new { status = "ok" }));

// ── /verify ───────────────────────────────────────────────────────────────────
app.MapPost("/verify", async (VerifyRequest req, Pipeline pipeline, CancellationToken ct) =>
{
    bool hasClaims = req.Claims is { Count: > 0 };
    if (!hasClaims && string.IsNullOrWhiteSpace(req.Document))
        return Results.Json(new { detail = "A claim or document is required." }, statusCode: 422);
    try
    {
        var resp = await pipeline.VerifyAsync(
            req.Document, req.Sources, req.Claims, includePassages: true, ct: ct);
        return Results.Ok(resp);
    }
    catch (Exception exc)
    {
        return Results.Json(new { detail = exc.Message }, statusCode: 500);
    }
});

// ── /extract — plain text from uploaded PDF / DOCX / TXT files ─────────────────
app.MapPost("/extract", async (HttpRequest request) =>
{
    if (!request.HasFormContentType)
        return Results.Json(new { detail = "At least one file is required." }, statusCode: 422);
    var form = await request.ReadFormAsync();
    var files = form.Files;
    if (files.Count == 0)
        return Results.Json(new { detail = "At least one file is required." }, statusCode: 422);

    var results = new List<ExtractResult>();
    foreach (var f in files)
    {
        using var ms = new MemoryStream();
        await f.CopyToAsync(ms);
        var raw = ms.ToArray();
        if (raw.Length == 0)
            return Results.Json(new { detail = $"'{f.FileName}' is empty." }, statusCode: 422);

        string text;
        try { text = Parse.ExtractText(f.FileName ?? "upload.txt", raw); }
        catch (Exception exc)
        {
            return Results.Json(new { detail = $"Could not read '{f.FileName}': {exc.Message}" }, statusCode: 422);
        }

        text = text.Trim();
        if (text.Length == 0)
            return Results.Json(new
            {
                detail = $"No readable text found in '{f.FileName}'. If it is a scanned PDF, OCR it first.",
            }, statusCode: 422);

        string name = f.FileName ?? "upload";
        string stem = Path.GetFileNameWithoutExtension(name);
        results.Add(new ExtractResult(
            string.IsNullOrEmpty(stem) ? name : stem,
            text,
            text.Length,
            name.ToLowerInvariant().EndsWith(".pdf") ? Parse.CountPdfPages(raw) : null));
    }
    return Results.Ok(results);
});

// ── Workspaces — per-document saved state ─────────────────────────────────────
app.MapGet("/workspace/{workspaceId}", (string workspaceId, Store store) =>
{
    var ws = store.LoadWorkspace(workspaceId);
    return ws is null
        ? Results.Json(new { detail = "No saved workspace." }, statusCode: 404)
        : Results.Ok(ws);
});

app.MapPut("/workspace/{workspaceId}", (string workspaceId, Workspace ws, Store store) =>
{
    ws.Id = workspaceId;
    ws.UpdatedAt = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
    store.SaveWorkspace(workspaceId, ws);
    return Results.Ok(ws);
});

// ── /index — pre-warm cache (ILDGS path deferred; no-op for the semchunk backend) ──
app.MapPost("/index", (IndexRequest req) =>
{
    int sources = (req.Sources?.Count ?? 0) + (string.IsNullOrWhiteSpace(req.Document) ? 0 : 1);
    return Results.Ok(new { segments = 0, sources });
});

app.Run();

static void LoadDotEnv(string contentRoot)
{
    // Search the content root and up to a few parents for a .env file.
    var dir = new DirectoryInfo(contentRoot);
    for (int i = 0; i < 6 && dir is not null; i++, dir = dir.Parent)
    {
        var path = Path.Combine(dir.FullName, ".env");
        if (!File.Exists(path)) continue;
        foreach (var line in File.ReadAllLines(path))
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0 || trimmed.StartsWith('#')) continue;
            int eq = trimmed.IndexOf('=');
            if (eq <= 0) continue;
            var key = trimmed[..eq].Trim();
            var val = trimmed[(eq + 1)..].Trim().Trim('"', '\'');
            if (Environment.GetEnvironmentVariable(key) is null)
                Environment.SetEnvironmentVariable(key, val);
        }
        return;
    }
}

internal sealed record ExtractResult(
    [property: System.Text.Json.Serialization.JsonPropertyName("id")] string Id,
    [property: System.Text.Json.Serialization.JsonPropertyName("text")] string Text,
    [property: System.Text.Json.Serialization.JsonPropertyName("chars")] int Chars,
    [property: System.Text.Json.Serialization.JsonPropertyName("pages")] int? Pages);

internal sealed record IndexRequest(
    [property: System.Text.Json.Serialization.JsonPropertyName("document")] string Document = "",
    [property: System.Text.Json.Serialization.JsonPropertyName("sources")] List<Source>? Sources = null);
