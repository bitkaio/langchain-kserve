import { describe, it, expect } from "vitest";
import { HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";

import {
  buildChatRequest,
  buildCompletionRequest,
  parseChatResponse,
  parseChatStreamChunk,
  parseCompletionResponse,
  parseCompletionStreamChunk,
} from "../../src/openai-compat.js";

describe("buildChatRequest", () => {
  it("builds a basic chat request", () => {
    const messages = [new HumanMessage("Hello!")];
    const req = buildChatRequest("gpt-4", messages, {}, {}, false);

    expect(req.model).toBe("gpt-4");
    expect(req.messages).toHaveLength(1);
    expect(req.messages[0].role).toBe("user");
    expect(req.messages[0].content).toBe("Hello!");
    expect(req.stream).toBe(false);
  });

  it("converts system messages", () => {
    const messages = [
      new SystemMessage("You are helpful."),
      new HumanMessage("Hi"),
    ];
    const req = buildChatRequest("model", messages, {}, {}, false);
    expect(req.messages[0].role).toBe("system");
    expect(req.messages[0].content).toBe("You are helpful.");
  });

  it("passes generation parameters", () => {
    const messages = [new HumanMessage("test")];
    const req = buildChatRequest(
      "model",
      messages,
      { temperature: 0.5, maxTokens: 100, topP: 0.9, stop: ["END"] },
      {},
      false
    );
    expect(req.temperature).toBe(0.5);
    expect(req.max_tokens).toBe(100);
    expect(req.top_p).toBe(0.9);
    expect(req.stop).toEqual(["END"]);
  });

  it("option overrides take precedence over params", () => {
    const messages = [new HumanMessage("test")];
    const req = buildChatRequest(
      "model",
      messages,
      { temperature: 0.5, maxTokens: 100 },
      { temperature: 0.9, maxTokens: 500 },
      false
    );
    expect(req.temperature).toBe(0.9);
    expect(req.max_tokens).toBe(500);
  });

  it("includes tools when provided", () => {
    const messages = [new HumanMessage("use a tool")];
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "search",
          description: "Search the web",
          parameters: { type: "object", properties: { query: { type: "string" } } },
        },
      },
    ];
    const req = buildChatRequest("model", messages, {}, { tools }, false);
    expect(req.tools).toHaveLength(1);
    expect(req.tools![0].function.name).toBe("search");
  });

  it("adds stream_options when streaming", () => {
    const messages = [new HumanMessage("stream me")];
    const req = buildChatRequest("model", messages, {}, {}, true);
    expect(req.stream).toBe(true);
    expect(req.stream_options?.include_usage).toBe(true);
  });
});

describe("parseChatResponse", () => {
  it("parses a basic assistant response", () => {
    const response = {
      id: "chatcmpl-123",
      object: "chat.completion",
      created: 1234567890,
      model: "qwen2.5",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "Hello there!" },
          finish_reason: "stop",
          logprobs: null,
        },
      ],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    };

    const gen = parseChatResponse(response, "qwen2.5");
    expect(gen.text).toBe("Hello there!");
    expect(gen.message.content).toBe("Hello there!");
    const info = gen.generationInfo as {
      protocol: string;
      finishReason: string;
      tokenUsage: { promptTokens: number };
    };
    expect(info.protocol).toBe("openai");
    expect(info.finishReason).toBe("stop");
    expect(info.tokenUsage.promptTokens).toBe(10);
  });

  it("parses tool call responses", () => {
    const response = {
      id: "chatcmpl-456",
      object: "chat.completion",
      created: 1234567890,
      model: "qwen2.5",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_abc",
                type: "function",
                function: {
                  name: "search",
                  arguments: '{"query": "TypeScript"}',
                },
              },
            ],
          },
          finish_reason: "tool_calls",
          logprobs: null,
        },
      ],
    };

    const gen = parseChatResponse(response, "model");
    const aiMsg = gen.message as import("@langchain/core/messages").AIMessage;
    expect(aiMsg.tool_calls).toHaveLength(1);
    expect(aiMsg.tool_calls![0].name).toBe("search");
    expect(aiMsg.tool_calls![0].args).toEqual({ query: "TypeScript" });
    expect(aiMsg.tool_calls![0].id).toBe("call_abc");
  });

  it("throws when no choices", () => {
    const response = {
      id: "x",
      object: "chat.completion",
      created: 0,
      model: "m",
      choices: [],
    };
    expect(() => parseChatResponse(response, "m")).toThrow("no choices");
  });

  it("handles null content gracefully", () => {
    const response = {
      id: "x",
      object: "chat.completion",
      created: 0,
      model: "m",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: null },
          finish_reason: "stop",
          logprobs: null,
        },
      ],
    };
    const gen = parseChatResponse(response, "m");
    expect(gen.text).toBe("");
  });
});

describe("parseChatStreamChunk", () => {
  it("parses a content delta chunk", () => {
    const raw = JSON.stringify({
      id: "chatcmpl-xyz",
      object: "chat.completion.chunk",
      created: 0,
      model: "m",
      choices: [{ index: 0, delta: { content: "Hello" }, finish_reason: null }],
    });
    const chunk = parseChatStreamChunk(raw, "m");
    expect(chunk).not.toBeNull();
    expect(chunk!.text).toBe("Hello");
  });

  it("handles finish chunk with null content", () => {
    const raw = JSON.stringify({
      id: "chatcmpl-xyz",
      object: "chat.completion.chunk",
      created: 0,
      model: "m",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    });
    const chunk = parseChatStreamChunk(raw, "m");
    expect(chunk).not.toBeNull();
    const info = chunk!.generationInfo as { finishReason: string };
    expect(info.finishReason).toBe("stop");
  });

  it("returns null for invalid JSON", () => {
    expect(parseChatStreamChunk("invalid", "m")).toBeNull();
  });

  it("returns null when no choices", () => {
    const raw = JSON.stringify({ id: "x", choices: [] });
    expect(parseChatStreamChunk(raw, "m")).toBeNull();
  });

  it("includes token usage from final chunk", () => {
    const raw = JSON.stringify({
      id: "x",
      object: "chat.completion.chunk",
      created: 0,
      model: "m",
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    });
    const chunk = parseChatStreamChunk(raw, "m");
    const info = chunk!.generationInfo as { tokenUsage: { totalTokens: number } };
    expect(info.tokenUsage.totalTokens).toBe(30);
  });
});

describe("buildCompletionRequest", () => {
  it("builds a basic completion request", () => {
    const req = buildCompletionRequest("model", "Once upon a time", {}, false);
    expect(req.model).toBe("model");
    expect(req.prompt).toBe("Once upon a time");
    expect(req.stream).toBe(false);
  });

  it("passes generation params", () => {
    const req = buildCompletionRequest(
      "model",
      "prompt",
      { temperature: 0.8, maxTokens: 256 },
      false
    );
    expect(req.temperature).toBe(0.8);
    expect(req.max_tokens).toBe(256);
  });
});

describe("parseCompletionResponse", () => {
  it("parses a completion response", () => {
    const response = {
      id: "cmpl-1",
      object: "text_completion",
      created: 0,
      model: "base-model",
      choices: [
        { index: 0, text: " a time in a land far away", finish_reason: "stop", logprobs: null },
      ],
      usage: { prompt_tokens: 5, completion_tokens: 10, total_tokens: 15 },
    };
    const result = parseCompletionResponse(response, "base-model");
    expect(result.generations[0][0].text).toBe(" a time in a land far away");
    expect(result.llmOutput?.tokenUsage?.totalTokens).toBe(15);
  });
});

describe("parseCompletionStreamChunk", () => {
  it("extracts text from streaming chunk", () => {
    const raw = JSON.stringify({
      id: "cmpl-s1",
      object: "text_completion",
      created: 0,
      model: "m",
      choices: [{ index: 0, text: " hello", finish_reason: null }],
    });
    expect(parseCompletionStreamChunk(raw)).toBe(" hello");
  });

  it("returns null for invalid JSON", () => {
    expect(parseCompletionStreamChunk("not json")).toBeNull();
  });
});
