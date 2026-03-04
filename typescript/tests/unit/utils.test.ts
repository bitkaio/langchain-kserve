import { describe, it, expect } from "vitest";
import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  ToolMessage,
} from "@langchain/core/messages";

import {
  convertMessagesToOpenAI,
  convertToolToOpenAI,
  formatChatML,
  formatLlama,
  formatMessagesToPrompt,
} from "../../src/utils.js";

// ============================================================
// convertMessagesToOpenAI
// ============================================================

describe("convertMessagesToOpenAI", () => {
  it("converts HumanMessage to user role", () => {
    const msgs = convertMessagesToOpenAI([new HumanMessage("Hi")]);
    expect(msgs).toEqual([{ role: "user", content: "Hi" }]);
  });

  it("converts SystemMessage to system role", () => {
    const msgs = convertMessagesToOpenAI([new SystemMessage("You are a bot")]);
    expect(msgs).toEqual([{ role: "system", content: "You are a bot" }]);
  });

  it("converts AIMessage to assistant role", () => {
    const msgs = convertMessagesToOpenAI([new AIMessage("Hello!")]);
    expect(msgs).toEqual([{ role: "assistant", content: "Hello!" }]);
  });

  it("converts ToolMessage to tool role", () => {
    const msgs = convertMessagesToOpenAI([
      new ToolMessage({ content: "result", tool_call_id: "call_123" }),
    ]);
    expect(msgs[0].role).toBe("tool");
    expect(msgs[0].content).toBe("result");
    expect(msgs[0].tool_call_id).toBe("call_123");
  });

  it("preserves tool_calls on AIMessage", () => {
    const aiMsg = new AIMessage({
      content: "",
      tool_calls: [
        { id: "c1", name: "search", args: { query: "test" }, type: "tool_call" },
      ],
    });
    const msgs = convertMessagesToOpenAI([aiMsg]);
    expect(msgs[0].tool_calls).toHaveLength(1);
    expect(msgs[0].tool_calls![0].function.name).toBe("search");
    expect(msgs[0].tool_calls![0].id).toBe("c1");
    expect(msgs[0].content).toBeNull();
  });

  it("converts a conversation with multiple message types", () => {
    const conversation = [
      new SystemMessage("Be helpful"),
      new HumanMessage("What's the weather?"),
      new AIMessage("It's sunny"),
      new HumanMessage("Thanks!"),
    ];
    const msgs = convertMessagesToOpenAI(conversation);
    expect(msgs.map((m) => m.role)).toEqual([
      "system",
      "user",
      "assistant",
      "user",
    ]);
  });
});

// ============================================================
// convertToolToOpenAI
// ============================================================

describe("convertToolToOpenAI", () => {
  it("passes through an already-formatted OpenAI tool", () => {
    const tool = {
      type: "function" as const,
      function: {
        name: "search",
        description: "search",
        parameters: {},
      },
    };
    const result = convertToolToOpenAI(tool);
    expect(result).toEqual(tool);
  });

  it("converts a LangChain StructuredToolInterface", () => {
    const tool = {
      name: "calculator",
      description: "Do math",
      schema: {
        type: "object",
        properties: { expression: { type: "string" } },
        required: ["expression"],
      },
    };
    const result = convertToolToOpenAI(tool as never);
    expect(result.type).toBe("function");
    expect(result.function.name).toBe("calculator");
    expect(result.function.description).toBe("Do math");
    expect(result.function.parameters).toBeDefined();
  });
});

// ============================================================
// formatChatML
// ============================================================

describe("formatChatML", () => {
  it("formats a simple exchange", () => {
    const messages = [
      { role: "user", content: "Hello" },
    ];
    const result = formatChatML(messages);
    expect(result).toContain("<|im_start|>user\nHello<|im_end|>");
    expect(result).toContain("<|im_start|>assistant");
  });

  it("includes system prompt", () => {
    const messages = [
      { role: "system", content: "You are a bot" },
      { role: "user", content: "Hi" },
    ];
    const result = formatChatML(messages);
    expect(result).toContain("<|im_start|>system\nYou are a bot<|im_end|>");
    expect(result).toContain("<|im_start|>user\nHi<|im_end|>");
    expect(result).toMatch(/<\|im_start\|>assistant\s*$/);
  });

  it("includes conversation history", () => {
    const messages = [
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello!" },
      { role: "user", content: "Bye" },
    ];
    const result = formatChatML(messages);
    expect(result).toContain("<|im_start|>assistant\nHello!<|im_end|>");
    expect(result).toContain("<|im_start|>user\nBye<|im_end|>");
  });
});

// ============================================================
// formatLlama
// ============================================================

describe("formatLlama", () => {
  it("formats a user message with system prompt", () => {
    const messages = [
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Tell me a joke" },
    ];
    const result = formatLlama(messages);
    expect(result).toContain("[INST]");
    expect(result).toContain("<<SYS>>");
    expect(result).toContain("You are helpful.");
    expect(result).toContain("<</SYS>>");
    expect(result).toContain("Tell me a joke");
    expect(result).toContain("[/INST]");
  });

  it("formats multi-turn conversation", () => {
    const messages = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there!" },
      { role: "user", content: "How are you?" },
    ];
    const result = formatLlama(messages);
    expect(result).toContain("Hello");
    expect(result).toContain("Hi there!");
    expect(result).toContain("How are you?");
  });
});

// ============================================================
// formatMessagesToPrompt
// ============================================================

describe("formatMessagesToPrompt", () => {
  it("defaults to ChatML", () => {
    const messages = [new HumanMessage("test")];
    const result = formatMessagesToPrompt(messages);
    expect(result).toContain("<|im_start|>");
  });

  it("uses Llama format when specified", () => {
    const messages = [new HumanMessage("test")];
    const result = formatMessagesToPrompt(messages, "llama");
    expect(result).toContain("[INST]");
  });

  it("throws when custom format is used without a template", () => {
    const messages = [new HumanMessage("test")];
    expect(() => formatMessagesToPrompt(messages, "custom")).toThrow(
      "customChatTemplate"
    );
  });

  it("applies custom template string", () => {
    const messages = [new HumanMessage("test")];
    const template = "MESSAGES: {{messages}} END";
    const result = formatMessagesToPrompt(messages, "custom", template);
    expect(result).toContain("MESSAGES:");
    expect(result).toContain("END");
    expect(result).toContain("user");
  });
});
