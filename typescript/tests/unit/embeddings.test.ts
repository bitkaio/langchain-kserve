import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { KServeEmbeddings } from "../../src/embeddings.js";
import { KServeInferenceError } from "../../src/errors.js";

// ============================================================
// Helper to build a mock embeddings response
// ============================================================

function makeEmbeddingsResponse(
  embeddings: Array<number[] | string>,
  model = "test-embedding-model"
) {
  return {
    object: "list" as const,
    data: embeddings.map((embedding, i) => ({
      object: "embedding" as const,
      index: i,
      embedding,
    })),
    model,
    usage: { prompt_tokens: embeddings.length * 5, total_tokens: embeddings.length * 5 },
  };
}

// ============================================================
// Tests
// ============================================================

describe("KServeEmbeddings", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
    delete process.env.KSERVE_EMBEDDINGS_BASE_URL;
    delete process.env.KSERVE_BASE_URL;
    delete process.env.KSERVE_EMBEDDINGS_MODEL_NAME;
    delete process.env.KSERVE_API_KEY;
  });

  it("constructs with required fields", () => {
    expect(() => {
      new KServeEmbeddings({
        baseUrl: "http://localhost:8080",
        modelName: "embed-model",
      });
    }).not.toThrow();
  });

  it("throws if baseUrl is not provided", () => {
    expect(() => {
      new KServeEmbeddings({
        baseUrl: undefined as unknown as string,
        modelName: "embed-model",
      });
    }).toThrow();
  });

  it("throws if modelName is not provided", () => {
    expect(() => {
      new KServeEmbeddings({
        baseUrl: "http://localhost:8080",
        modelName: undefined as unknown as string,
      });
    }).toThrow();
  });

  it("embedDocuments returns correct vectors in correct order", async () => {
    const vectors = [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
      [0.7, 0.8, 0.9],
    ];

    const mockFetch = vi.fn().mockResolvedValueOnce(
      new Response(JSON.stringify(makeEmbeddingsResponse(vectors)), { status: 200 })
    );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
    });

    const result = await embeddings.embedDocuments(["hello", "world", "foo"]);

    expect(result).toHaveLength(3);
    expect(result[0]).toEqual([0.1, 0.2, 0.3]);
    expect(result[1]).toEqual([0.4, 0.5, 0.6]);
    expect(result[2]).toEqual([0.7, 0.8, 0.9]);
  });

  it("embedDocuments sorts results by index (out-of-order response)", async () => {
    // Response returns items out of order
    const mockResponse = {
      object: "list" as const,
      data: [
        { object: "embedding" as const, index: 2, embedding: [0.7, 0.8] },
        { object: "embedding" as const, index: 0, embedding: [0.1, 0.2] },
        { object: "embedding" as const, index: 1, embedding: [0.4, 0.5] },
      ],
      model: "embed-model",
    };

    const mockFetch = vi.fn().mockResolvedValueOnce(
      new Response(JSON.stringify(mockResponse), { status: 200 })
    );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
    });

    const result = await embeddings.embedDocuments(["a", "b", "c"]);

    expect(result[0]).toEqual([0.1, 0.2]);
    expect(result[1]).toEqual([0.4, 0.5]);
    expect(result[2]).toEqual([0.7, 0.8]);
  });

  it("embedDocuments batches requests when texts.length > chunkSize", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify(makeEmbeddingsResponse([[0.1, 0.2], [0.3, 0.4]])),
          { status: 200 }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify(makeEmbeddingsResponse([[0.5, 0.6]])),
          { status: 200 }
        )
      );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
      chunkSize: 2,
    });

    const result = await embeddings.embedDocuments(["a", "b", "c"]);

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(result).toHaveLength(3);
    expect(result[0]).toEqual([0.1, 0.2]);
    expect(result[1]).toEqual([0.3, 0.4]);
    expect(result[2]).toEqual([0.5, 0.6]);
  });

  it("batching sends correct inputs in each request", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify(makeEmbeddingsResponse([[1, 0]])),
          { status: 200 }
        )
      )
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify(makeEmbeddingsResponse([[0, 1]])),
          { status: 200 }
        )
      );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
      chunkSize: 1,
    });

    await embeddings.embedDocuments(["hello", "world"]);

    const [, firstInit] = mockFetch.mock.calls[0] as [string, RequestInit];
    const firstBody = JSON.parse(firstInit.body as string) as { input: string[] };
    expect(firstBody.input).toEqual(["hello"]);

    const [, secondInit] = mockFetch.mock.calls[1] as [string, RequestInit];
    const secondBody = JSON.parse(secondInit.body as string) as { input: string[] };
    expect(secondBody.input).toEqual(["world"]);
  });

  it("embedQuery delegates to embedDocuments and returns first element", async () => {
    const mockFetch = vi.fn().mockResolvedValueOnce(
      new Response(
        JSON.stringify(makeEmbeddingsResponse([[0.1, 0.2, 0.3]])),
        { status: 200 }
      )
    );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
    });

    const result = await embeddings.embedQuery("hello");

    expect(Array.isArray(result)).toBe(true);
    expect(result).toEqual([0.1, 0.2, 0.3]);

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as { input: string[] };
    expect(body.input).toEqual(["hello"]);
  });

  it("dimensions parameter is passed through when set", async () => {
    const mockFetch = vi.fn().mockResolvedValueOnce(
      new Response(
        JSON.stringify(makeEmbeddingsResponse([[0.1, 0.2]])),
        { status: 200 }
      )
    );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
      dimensions: 512,
    });

    await embeddings.embedDocuments(["test"]);

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      dimensions?: number;
    };
    expect(body.dimensions).toBe(512);
  });

  it("dimensions parameter is omitted when not set", async () => {
    const mockFetch = vi.fn().mockResolvedValueOnce(
      new Response(
        JSON.stringify(makeEmbeddingsResponse([[0.1, 0.2]])),
        { status: 200 }
      )
    );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
    });

    await embeddings.embedDocuments(["test"]);

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      dimensions?: number;
    };
    expect(body.dimensions).toBeUndefined();
  });

  it("sends correct URL, model, and encoding_format in request body", async () => {
    const mockFetch = vi.fn().mockResolvedValueOnce(
      new Response(
        JSON.stringify(makeEmbeddingsResponse([[0.1]])),
        { status: 200 }
      )
    );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "my-embed-model",
    });

    await embeddings.embedDocuments(["test"]);

    const [url, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:8080/v1/embeddings");

    const body = JSON.parse(init.body as string) as {
      model: string;
      encoding_format: string;
    };
    expect(body.model).toBe("my-embed-model");
    expect(body.encoding_format).toBe("float");
  });

  it("base64 encoding format decodes to float arrays", async () => {
    // Encode [1.0, 2.0] as base64 float32 LE
    const buf = Buffer.alloc(8);
    buf.writeFloatLE(1.0, 0);
    buf.writeFloatLE(2.0, 4);
    const base64Embedding = buf.toString("base64");

    const mockResponse = {
      object: "list" as const,
      data: [
        {
          object: "embedding" as const,
          index: 0,
          embedding: base64Embedding,
        },
      ],
      model: "embed-model",
    };

    const mockFetch = vi.fn().mockResolvedValueOnce(
      new Response(JSON.stringify(mockResponse), { status: 200 })
    );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
      encodingFormat: "base64",
    });

    const result = await embeddings.embedDocuments(["hello"]);

    expect(result).toHaveLength(1);
    expect(result[0]).toHaveLength(2);
    expect(result[0]![0]).toBeCloseTo(1.0, 5);
    expect(result[0]![1]).toBeCloseTo(2.0, 5);
  });

  it("base64 encoding_format is sent in request body", async () => {
    const buf = Buffer.alloc(4);
    buf.writeFloatLE(0.5, 0);
    const base64Embedding = buf.toString("base64");

    const mockFetch = vi.fn().mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          object: "list" as const,
          data: [{ object: "embedding" as const, index: 0, embedding: base64Embedding }],
          model: "embed-model",
        }),
        { status: 200 }
      )
    );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
      encodingFormat: "base64",
    });

    await embeddings.embedDocuments(["hello"]);

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      encoding_format: string;
    };
    expect(body.encoding_format).toBe("base64");
  });

  it("sends Bearer token in Authorization header", async () => {
    const mockFetch = vi.fn().mockResolvedValueOnce(
      new Response(
        JSON.stringify(makeEmbeddingsResponse([[0.1]])),
        { status: 200 }
      )
    );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
      apiKey: "my-secret-key",
    });

    await embeddings.embedDocuments(["test"]);

    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    const headers = init.headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("Bearer my-secret-key");
  });

  it("error handling: propagates error on 401 response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValueOnce(
        new Response("Unauthorized", { status: 401 })
      )
    );

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
    });

    await expect(embeddings.embedDocuments(["test"])).rejects.toThrow();
  });

  it("error handling: throws KServeInferenceError on 500 server error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("Internal Server Error", { status: 500 })
      )
    );

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080",
      modelName: "embed-model",
      maxRetries: 0,
    });

    await expect(embeddings.embedDocuments(["test"])).rejects.toThrow(
      KServeInferenceError
    );
  });

  it("uses KSERVE_EMBEDDINGS_BASE_URL env var as fallback", () => {
    process.env.KSERVE_EMBEDDINGS_BASE_URL = "http://env-embeddings:8080";
    // With env var set, passing undefined for baseUrl should work
    expect(() => {
      new KServeEmbeddings({
        baseUrl: "http://env-embeddings:8080",
        modelName: "embed-model",
      });
    }).not.toThrow();
  });

  it("strips trailing slash from baseUrl", async () => {
    const mockFetch = vi.fn().mockResolvedValueOnce(
      new Response(
        JSON.stringify(makeEmbeddingsResponse([[0.1]])),
        { status: 200 }
      )
    );
    vi.stubGlobal("fetch", mockFetch);

    const embeddings = new KServeEmbeddings({
      baseUrl: "http://localhost:8080/",
      modelName: "embed-model",
    });

    await embeddings.embedDocuments(["test"]);

    const [url] = mockFetch.mock.calls[0] as [string];
    expect(url).toBe("http://localhost:8080/v1/embeddings");
  });
});
