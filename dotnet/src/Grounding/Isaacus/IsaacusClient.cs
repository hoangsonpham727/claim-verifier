using System.Net.Http.Json;
using System.Text.Json;

namespace Grounding.Isaacus;

/// <summary>
/// Vendor seam — all Isaacus model calls route through here (port of
/// src/grounding/client.py). Isaacus ships no .NET SDK, so this is a hand-written
/// typed HttpClient against the REST API.
/// </summary>
public sealed class IsaacusClient
{
    public const string RerankerModel = "kanon-2-reranker";
    public const string ClassifierModel = "kanon-universal-classifier";
    public const string ExtractorModel = "kanon-answer-extractor";

    private static readonly JsonSerializerOptions JsonOpts = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.Never,
    };

    private readonly HttpClient _http;

    public IsaacusClient(HttpClient http) => _http = http;

    public async Task<RerankResponse> RerankAsync(
        string query, IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        var req = new RerankRequest(RerankerModel, query, texts);
        return await PostAsync<RerankRequest, RerankResponse>("rerankings", req, ct);
    }

    public async Task<ClassifyResponse> ClassifyUniversalAsync(
        string query, IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        var req = new ClassifyRequest(ClassifierModel, query, texts);
        return await PostAsync<ClassifyRequest, ClassifyResponse>("classifications/universal", req, ct);
    }

    public async Task<QaResponse> ExtractQaAsync(
        string query, IReadOnlyList<string> texts, int topK,
        bool ignoreInextractability, CancellationToken ct = default)
    {
        var req = new QaRequest(ExtractorModel, query, texts, topK, ignoreInextractability);
        return await PostAsync<QaRequest, QaResponse>("extractions/qa", req, ct);
    }

    private async Task<TResp> PostAsync<TReq, TResp>(string path, TReq body, CancellationToken ct)
    {
        using var resp = await _http.PostAsJsonAsync(path, body, JsonOpts, ct);
        if (!resp.IsSuccessStatusCode)
        {
            var detail = await resp.Content.ReadAsStringAsync(ct);
            throw new IsaacusApiException(
                $"Isaacus {path} returned {(int)resp.StatusCode}: {detail}");
        }
        var parsed = await resp.Content.ReadFromJsonAsync<TResp>(JsonOpts, ct);
        return parsed ?? throw new IsaacusApiException($"Isaacus {path} returned an empty body.");
    }
}

public sealed class IsaacusApiException : Exception
{
    public IsaacusApiException(string message) : base(message) { }
}
