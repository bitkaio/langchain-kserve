import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { parseSSE } from "../../src/client.js";
import { KServeClient } from "../../src/client.js";
import {
  KServeAuthenticationError,
  KServeModelNotFoundError,
  KServeInferenceError,
  KServeTimeoutError,
} from "../../src/errors.js";

// ============================================================
// parseSSE unit tests (standalone generator)
// ============================================================

/**
 * Helper: create a fake fetch Response from SSE text lines
 */
function makeSseResponse(lines: string[]): Response {
  const body = lines.join("\n") + "\n";
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(body));
      controller.close();
    },
  });
  return new Response(stream, { status: 200 });
}

describe("parseSSE", () => {
  it("yields data payloads from SSE lines", async () => {
    const response = makeSseResponse([
      'data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}',
      'data: {"id":"2","choices":[{"delta":{"content":" world"}}]}',
      "data: [DONE]",
    ]);

    const chunks: string[] = [];
    for await (const chunk of parseSSE(response)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(2);
    expect(chunks[0]).toContain('"Hello"');
    expect(chunks[1]).toContain('" world"');
  });

  it("stops at [DONE]", async () => {
    const response = makeSseResponse([
      "data: chunk1",
      "data: [DONE]",
      "data: chunk2", // should not be yielded
    ]);

    const chunks: string[] = [];
    for await (const chunk of parseSSE(response)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toBe("chunk1");
  });

  it("ignores non-data lines", async () => {
    const response = makeSseResponse([
      ": this is a comment",
      "event: message",
      "id: 123",
      "data: actual-data",
      "data: [DONE]",
    ]);

    const chunks: string[] = [];
    for await (const chunk of parseSSE(response)) {
      chunks.push(chunk);
    }
    expect(chunks).toEqual(["actual-data"]);
  });

  it("handles stream ending without [DONE]", async () => {
    const response = makeSseResponse(["data: only-chunk"]);

    const chunks: string[] = [];
    for await (const chunk of parseSSE(response)) {
      chunks.push(chunk);
    }
    expect(chunks).toEqual(["only-chunk"]);
  });
});

// ============================================================
// KServeClient tests (with mocked fetch)
// ============================================================

describe("KServeClient", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("makes POST request with correct headers", async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ result: "ok" }), { status: 200 })
    );
    vi.stubGlobal("fetch", mockFetch);

    const client = new KServeClient({
      baseUrl: "http://my-model.local",
      apiKey: "test-key",
    });

    await client.request("/v1/chat/completions", { model: "m" });

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://my-model.local/v1/chat/completions");
    expect((init.headers as Record<string, string>)["Authorization"]).toBe(
      "Bearer test-key"
    );
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe(
      "application/json"
    );
  });

  it("calls tokenProvider and uses returned token", async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({}), { status: 200 })
    );
    vi.stubGlobal("fetch", mockFetch);

    const tokenProvider = vi.fn().mockResolvedValue("dynamic-token-xyz");
    const client = new KServeClient({
      baseUrl: "http://my-model.local",
      tokenProvider,
    });

    await client.request("/test", {});

    expect(tokenProvider).toHaveBeenCalledOnce();
    const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect((init.headers as Record<string, string>)["Authorization"]).toBe(
      "Bearer dynamic-token-xyz"
    );
  });

  it("throws KServeAuthenticationError on 401", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("Unauthorized", { status: 401 })
      )
    );

    const client = new KServeClient({ baseUrl: "http://my-model.local" });
    await expect(client.request("/test", {})).rejects.toThrow(
      KServeAuthenticationError
    );
  });

  it("throws KServeModelNotFoundError on 404", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("Not Found", { status: 404 }))
    );

    const client = new KServeClient({ baseUrl: "http://my-model.local" });
    await expect(client.request("/test", {})).rejects.toThrow(
      KServeModelNotFoundError
    );
  });

  it("throws KServeInferenceError on 500", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("Server Error", { status: 500 })
      )
    );

    const client = new KServeClient({
      baseUrl: "http://my-model.local",
      maxRetries: 0,
    });
    await expect(client.request("/test", {})).rejects.toThrow(
      KServeInferenceError
    );
  });

  it("strips trailing slash from baseUrl", async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({}), { status: 200 })
    );
    vi.stubGlobal("fetch", mockFetch);

    const client = new KServeClient({ baseUrl: "http://my-model.local/" });
    await client.request("/v1/test", {});

    const [url] = mockFetch.mock.calls[0] as [string];
    expect(url).toBe("http://my-model.local/v1/test");
  });

  it("detects openai protocol when /v1/models returns 200", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("", { status: 200 }))
    );

    const client = new KServeClient({ baseUrl: "http://my-model.local" });
    const protocol = await client.detectProtocol();
    expect(protocol).toBe("openai");
  });

  it("detects v2 protocol when /v1/models returns non-200", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("", { status: 404 }))
    );

    const client = new KServeClient({ baseUrl: "http://my-model.local" });
    const protocol = await client.detectProtocol();
    expect(protocol).toBe("v2");
  });

  it("caches detected protocol", async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response("", { status: 200 })
    );
    vi.stubGlobal("fetch", mockFetch);

    const client = new KServeClient({ baseUrl: "http://my-model.local" });
    await client.detectProtocol();
    await client.detectProtocol();

    // /v1/models should only be called once (cached)
    expect(mockFetch).toHaveBeenCalledOnce();
  });

  it("setProtocol overrides auto-detection", async () => {
    const mockFetch = vi.fn();
    vi.stubGlobal("fetch", mockFetch);

    const client = new KServeClient({ baseUrl: "http://my-model.local" });
    client.setProtocol("v2");
    const protocol = await client.detectProtocol();
    expect(protocol).toBe("v2");
    // Should not have called fetch for detection
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("retries on transient connection failures", async () => {
    let attempts = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(async () => {
        attempts++;
        if (attempts < 3) {
          throw new Error("fetch failed: connection reset");
        }
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      })
    );

    const client = new KServeClient({
      baseUrl: "http://my-model.local",
      maxRetries: 3,
    });

    const result = await client.request<{ ok: boolean }>("/test", {});
    expect(result.ok).toBe(true);
    expect(attempts).toBe(3);
  }, 10_000);

  it("getModelInfo calls /v1/models/{model} and maps response", async () => {
    const modelResponse = {
      id: "my-model",
      object: "model",
      created: 1000,
      owned_by: "user",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(modelResponse), { status: 200 })
      )
    );

    const client = new KServeClient({ baseUrl: "http://my-model.local" });
    const info = await client.getModelInfo("my-model");

    expect(info.modelName).toBe("my-model");
    expect(info.platform).toBe("openai-compat");
    expect(info.raw).toEqual(modelResponse);
    expect(info.modelVersion).toBeUndefined();
  });

  it("getModelInfo falls back to /v2/models/{model} on failure", async () => {
    const v2ModelResponse = {
      name: "my-model",
      versions: ["1.0"],
      platform: "triton",
      inputs: [{ name: "text_input", datatype: "BYTES" }],
      outputs: [{ name: "text_output", datatype: "BYTES" }],
    };
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(new Response("Not Found", { status: 404 })) // /v1/models fails
        .mockResolvedValueOnce(
          new Response(JSON.stringify(v2ModelResponse), { status: 200 })
        )
    );

    const client = new KServeClient({ baseUrl: "http://my-model.local" });
    const info = await client.getModelInfo("my-model");

    expect(info.modelName).toBe("my-model");
    expect(info.modelVersion).toBe("1.0");
    expect(info.platform).toBe("triton");
    expect(info.inputs).toHaveLength(1);
    expect(info.outputs).toHaveLength(1);
    expect(info.raw).toEqual(v2ModelResponse);
  });
});
