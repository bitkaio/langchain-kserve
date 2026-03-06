import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { KServeLLM } from "../../src/llms.js";

// ============================================================
// SSE helper for streaming tests
// ============================================================

function makeSseStreamResponse(chunks: object[]): Response {
  const lines = chunks.map((c) => `data: ${JSON.stringify(c)}`);
  lines.push("data: [DONE]");
  const body = lines.join("\n") + "\n";
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(body));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

function makeCompletionResponse(text: string) {
  return {
    id: "cmpl-test",
    object: "text_completion",
    created: 0,
    model: "test-model",
    choices: [{ index: 0, text, finish_reason: "stop", logprobs: null }],
    usage: { prompt_tokens: 5, completion_tokens: 10, total_tokens: 15 },
  };
}

function makeV2Response(text: string) {
  return {
    id: "v2-test",
    model_name: "test-model",
    outputs: [
      { name: "text_output", shape: [1], datatype: "BYTES", data: [text] },
    ],
  };
}

describe("KServeLLM", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("constructs with required fields", () => {
    const llm = new KServeLLM({
      baseUrl: "http://localhost:8080",
      modelName: "base-model",
    });
    expect(llm._llmType()).toBe("kserve-llm");
  });

  it("generates completion via OpenAI protocol", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(new Response("", { status: 200 })) // /v1/models
      .mockResolvedValueOnce(
        new Response(JSON.stringify(makeCompletionResponse(" the sky")), {
          status: 200,
        })
      );
    vi.stubGlobal("fetch", mockFetch);

    const llm = new KServeLLM({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
    });

    const result = await llm.invoke("The color of");
    expect(result).toBe(" the sky");
  });

  it("generates completion via V2 protocol when pinned", async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(makeV2Response("42")), { status: 200 })
    );
    vi.stubGlobal("fetch", mockFetch);

    const llm = new KServeLLM({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
      protocol: "v2",
    });

    const result = await llm.invoke("The answer is");
    expect(result).toBe("42");
  });

  it("sends prompt to /v1/completions endpoint", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(new Response("", { status: 200 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify(makeCompletionResponse("output")), {
          status: 200,
        })
      );
    vi.stubGlobal("fetch", mockFetch);

    const llm = new KServeLLM({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
    });

    await llm.invoke("my prompt");

    const [url, init] = mockFetch.mock.calls[1] as [string, RequestInit];
    expect(url).toContain("/v1/completions");
    const body = JSON.parse(init.body as string) as { prompt: string };
    expect(body.prompt).toBe("my prompt");
  });

  it("includes generation params in request", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(new Response("", { status: 200 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify(makeCompletionResponse("ok")), {
          status: 200,
        })
      );
    vi.stubGlobal("fetch", mockFetch);

    const llm = new KServeLLM({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
      temperature: 0.3,
      maxTokens: 128,
    });

    await llm.invoke("prompt");

    const [, init] = mockFetch.mock.calls[1] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      temperature: number;
      max_tokens: number;
    };
    expect(body.temperature).toBe(0.3);
    expect(body.max_tokens).toBe(128);
  });

  it("handles multiple prompts via _generate", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(new Response("", { status: 200 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify(makeCompletionResponse("output1")), {
          status: 200,
        })
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(makeCompletionResponse("output2")), {
          status: 200,
        })
      );
    vi.stubGlobal("fetch", mockFetch);

    const llm = new KServeLLM({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
    });

    const result = await llm.generate(["prompt1", "prompt2"]);
    expect(result.generations).toHaveLength(2);
    expect(result.generations[0][0].text).toBe("output1");
    expect(result.generations[1][0].text).toBe("output2");
  });

  it("streaming aggregates token usage from final chunk", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(new Response("", { status: 200 })) // protocol detection
      .mockResolvedValueOnce(
        makeSseStreamResponse([
          {
            id: "cmpl-s1",
            object: "text_completion",
            created: 0,
            model: "test-model",
            choices: [{ index: 0, text: "Hello", finish_reason: null }],
          },
          {
            id: "cmpl-s2",
            object: "text_completion",
            created: 0,
            model: "test-model",
            choices: [{ index: 0, text: " world", finish_reason: "stop" }],
            usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
          },
        ])
      );
    vi.stubGlobal("fetch", mockFetch);

    const llm = new KServeLLM({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
      streaming: true,
    });

    const result = await llm.generate(["my prompt"]);
    expect(result.generations[0][0].text).toBe("Hello world");
    expect(result.llmOutput?.tokenUsage?.totalTokens).toBe(7);
    expect(result.llmOutput?.tokenUsage?.promptTokens).toBe(5);
    expect(result.llmOutput?.tokenUsage?.completionTokens).toBe(2);
  });
});
