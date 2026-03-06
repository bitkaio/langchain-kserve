import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatKServe } from "../../src/chat_models.js";

// ============================================================
// Helper to build a mock OpenAI chat response
// ============================================================

function makeChatResponse(content: string, usage = true) {
  return {
    id: "chatcmpl-test",
    object: "chat.completion",
    created: 0,
    model: "test-model",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
        logprobs: null,
      },
    ],
    ...(usage
      ? { usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } }
      : {}),
  };
}

// ============================================================
// Helper to build a mock V2 response
// ============================================================

function makeV2Response(text: string) {
  return {
    id: "v2-test",
    model_name: "test-model",
    outputs: [
      { name: "text_output", shape: [1], datatype: "BYTES", data: [text] },
    ],
  };
}

// ============================================================
// Tests
// ============================================================

describe("ChatKServe", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
    delete process.env.KSERVE_BASE_URL;
    delete process.env.KSERVE_MODEL_NAME;
    delete process.env.KSERVE_API_KEY;
  });

  it("constructs with required fields", () => {
    const model = new ChatKServe({
      baseUrl: "http://localhost:8080",
      modelName: "my-model",
    });
    expect(model._llmType()).toBe("kserve");
  });

  it("reads baseUrl and modelName from environment", () => {
    process.env.KSERVE_BASE_URL = "http://env-host:8080";
    process.env.KSERVE_MODEL_NAME = "env-model";
    const model = new ChatKServe({ baseUrl: "http://env-host:8080", modelName: "env-model" });
    expect(model._identifyingParams().base_url).toBe("http://env-host:8080");
    expect(model._identifyingParams().model_name).toBe("env-model");
  });

  it("throws if baseUrl is not provided", () => {
    // Passing undefined forces the env-var fallback path to throw
    expect(
      () =>
        new ChatKServe({
          baseUrl: undefined as unknown as string,
          modelName: "m",
        })
    ).toThrow();
  });

  it("generates a response via OpenAI protocol", async () => {
    // First fetch: protocol detection (/v1/models → 200)
    // Second fetch: actual inference
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(new Response("", { status: 200 })) // /v1/models
      .mockResolvedValueOnce(
        new Response(JSON.stringify(makeChatResponse("Hello!")), { status: 200 })
      );
    vi.stubGlobal("fetch", mockFetch);

    const model = new ChatKServe({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
    });

    const result = await model.invoke([new HumanMessage("Hi")]);
    expect(result.content).toBe("Hello!");
  });

  it("generates a response via V2 protocol when pinned", async () => {
    const mockFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(makeV2Response("V2 response")), {
        status: 200,
      })
    );
    vi.stubGlobal("fetch", mockFetch);

    const model = new ChatKServe({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
      protocol: "v2",
    });

    const result = await model.invoke([new HumanMessage("Hi")]);
    expect(result.content).toBe("V2 response");
  });

  it("falls back to V2 when /v1/models returns 404", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(new Response("", { status: 404 })) // /v1/models
      .mockResolvedValueOnce(
        new Response(JSON.stringify(makeV2Response("V2 auto")), { status: 200 })
      );
    vi.stubGlobal("fetch", mockFetch);

    const model = new ChatKServe({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
    });

    const result = await model.invoke([new HumanMessage("Hi")]);
    expect(result.content).toBe("V2 auto");
  });

  it("sends system + user messages correctly", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(new Response("", { status: 200 })) // /v1/models
      .mockResolvedValueOnce(
        new Response(JSON.stringify(makeChatResponse("response")), { status: 200 })
      );
    vi.stubGlobal("fetch", mockFetch);

    const model = new ChatKServe({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
    });

    await model.invoke([
      new SystemMessage("You are helpful"),
      new HumanMessage("Tell me a joke"),
    ]);

    const [, init] = mockFetch.mock.calls[1] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      messages: Array<{ role: string; content: string }>;
    };
    expect(body.messages[0].role).toBe("system");
    expect(body.messages[0].content).toBe("You are helpful");
    expect(body.messages[1].role).toBe("user");
    expect(body.messages[1].content).toBe("Tell me a joke");
  });

  it("includes generation params in request body", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(new Response("", { status: 200 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify(makeChatResponse("ok")), { status: 200 })
      );
    vi.stubGlobal("fetch", mockFetch);

    const model = new ChatKServe({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
      temperature: 0.7,
      maxTokens: 256,
    });

    await model.invoke([new HumanMessage("Hi")]);

    const [, init] = mockFetch.mock.calls[1] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      temperature: number;
      max_tokens: number;
    };
    expect(body.temperature).toBe(0.7);
    expect(body.max_tokens).toBe(256);
  });

  it("bindTools includes tools in request", async () => {
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(new Response("", { status: 200 }))
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            ...makeChatResponse(""),
            choices: [
              {
                index: 0,
                message: {
                  role: "assistant",
                  content: null,
                  tool_calls: [
                    {
                      id: "call_1",
                      type: "function",
                      function: { name: "search", arguments: '{"query":"test"}' },
                    },
                  ],
                },
                finish_reason: "tool_calls",
                logprobs: null,
              },
            ],
          }),
          { status: 200 }
        )
      );
    vi.stubGlobal("fetch", mockFetch);

    const model = new ChatKServe({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
    });

    const searchTool = {
      type: "function" as const,
      function: {
        name: "search",
        description: "Search",
        parameters: { type: "object", properties: { query: { type: "string" } } },
      },
    };

    const modelWithTools = model.bindTools([searchTool]);
    const result = await modelWithTools.invoke([new HumanMessage("search for kserve")]);

    // Verify tools were sent
    const [, init] = mockFetch.mock.calls[1] as [string, RequestInit];
    const body = JSON.parse(init.body as string) as { tools: unknown[] };
    expect(body.tools).toHaveLength(1);

    // Verify tool call parsed correctly
    const aiMsg = result as import("@langchain/core/messages").AIMessage;
    expect(aiMsg.tool_calls).toHaveLength(1);
    expect(aiMsg.tool_calls![0].name).toBe("search");
  });

  it("_identifyingParams returns model metadata", () => {
    const model = new ChatKServe({
      baseUrl: "http://localhost:8080",
      modelName: "my-model",
      temperature: 0.5,
    });
    const params = model._identifyingParams();
    expect(params.model_name).toBe("my-model");
    expect(params.base_url).toBe("http://localhost:8080");
    expect(params.temperature).toBe(0.5);
  });

  it("getModelInfo returns KServeModelInfo for OpenAI-compat protocol", async () => {
    const mockModelResponse = {
      id: "test-model",
      object: "model",
      created: 0,
      owned_by: "test",
    };
    const mockFetch = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(mockModelResponse), { status: 200 })
      );
    vi.stubGlobal("fetch", mockFetch);

    const model = new ChatKServe({
      baseUrl: "http://localhost:8080",
      modelName: "test-model",
      protocol: "openai",
    });

    const info = await model.getModelInfo();
    expect(info.modelName).toBe("test-model");
    expect(info.platform).toBe("openai-compat");
    expect(info.raw).toEqual(mockModelResponse);
  });
});
