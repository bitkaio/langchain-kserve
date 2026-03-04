import { describe, it, expect } from "vitest";
import { HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";

import {
  buildV2ChatRequest,
  buildV2TextRequest,
  buildV2StreamRequest,
  parseV2ChatResponse,
  parseV2TextResponse,
  parseV2ChatStreamChunk,
  parseV2TextStreamChunk,
  getV2InferPath,
} from "../../src/v2-protocol.js";

describe("buildV2TextRequest", () => {
  it("builds a valid V2 request with prompt", () => {
    const req = buildV2TextRequest("Hello world", {});
    expect(req.inputs).toHaveLength(1);
    expect(req.inputs[0].name).toBe("text_input");
    expect(req.inputs[0].data[0]).toBe("Hello world");
    expect(req.inputs[0].datatype).toBe("BYTES");
    expect(req.inputs[0].shape).toEqual([1]);
    expect(req.id).toBeTruthy();
  });

  it("includes generation parameters", () => {
    const req = buildV2TextRequest("test", {
      temperature: 0.7,
      maxTokens: 200,
      topP: 0.9,
      stop: ["<|end|>"],
    });
    expect(req.parameters?.temperature).toBe(0.7);
    expect(req.parameters?.max_tokens).toBe(200);
    expect(req.parameters?.top_p).toBe(0.9);
    expect(req.parameters?.stop).toEqual(["<|end|>"]);
  });

  it("omits parameters field when no params given", () => {
    const req = buildV2TextRequest("test", {});
    expect(req.parameters).toBeUndefined();
  });

  it("generates unique IDs per request", () => {
    const req1 = buildV2TextRequest("a", {});
    const req2 = buildV2TextRequest("b", {});
    expect(req1.id).not.toBe(req2.id);
  });
});

describe("buildV2ChatRequest", () => {
  it("formats messages using ChatML by default", () => {
    const messages = [
      new SystemMessage("You are helpful."),
      new HumanMessage("Hello!"),
    ];
    const req = buildV2ChatRequest(messages, {});
    const prompt = req.inputs[0].data[0] as string;
    expect(prompt).toContain("<|im_start|>system");
    expect(prompt).toContain("You are helpful.");
    expect(prompt).toContain("<|im_start|>user");
    expect(prompt).toContain("Hello!");
    expect(prompt).toContain("<|im_start|>assistant");
  });

  it("formats messages using Llama template", () => {
    const messages = [
      new SystemMessage("You are helpful."),
      new HumanMessage("What is 2+2?"),
    ];
    const req = buildV2ChatRequest(messages, {}, "llama");
    const prompt = req.inputs[0].data[0] as string;
    expect(prompt).toContain("[INST]");
    expect(prompt).toContain("<<SYS>>");
    expect(prompt).toContain("You are helpful.");
    expect(prompt).toContain("What is 2+2?");
    expect(prompt).toContain("[/INST]");
  });

  it("includes AI message history", () => {
    const messages = [
      new HumanMessage("Hi"),
      new AIMessage("Hello!"),
      new HumanMessage("How are you?"),
    ];
    const req = buildV2ChatRequest(messages, {});
    const prompt = req.inputs[0].data[0] as string;
    expect(prompt).toContain("Hi");
    expect(prompt).toContain("Hello!");
    expect(prompt).toContain("How are you?");
  });
});

describe("buildV2StreamRequest", () => {
  it("adds stream parameter to the request", () => {
    const base = buildV2TextRequest("test", {});
    const streaming = buildV2StreamRequest(base);
    expect(streaming.parameters?.stream).toBe(true);
  });

  it("preserves existing parameters", () => {
    const base = buildV2TextRequest("test", { temperature: 0.5 });
    const streaming = buildV2StreamRequest(base);
    expect(streaming.parameters?.temperature).toBe(0.5);
    expect(streaming.parameters?.stream).toBe(true);
  });
});

describe("parseV2ChatResponse", () => {
  it("extracts text_output and wraps in AIMessage", () => {
    const response = {
      id: "123",
      model_name: "my-model",
      outputs: [
        {
          name: "text_output",
          shape: [1],
          datatype: "BYTES",
          data: ["Hello from the model!"],
        },
      ],
    };

    const gen = parseV2ChatResponse(response, "my-model");
    expect(gen.text).toBe("Hello from the model!");
    expect(gen.message.content).toBe("Hello from the model!");
    const info = gen.generationInfo as { protocol: string; modelName: string };
    expect(info.protocol).toBe("v2");
    expect(info.modelName).toBe("my-model");
  });

  it("also accepts 'output' as the tensor name", () => {
    const response = {
      id: "456",
      model_name: "my-model",
      outputs: [
        {
          name: "output",
          shape: [1],
          datatype: "BYTES",
          data: ["Some output"],
        },
      ],
    };

    const gen = parseV2ChatResponse(response, "my-model");
    expect(gen.text).toBe("Some output");
  });

  it("throws KServeInferenceError when no text_output", () => {
    const response = {
      id: "789",
      model_name: "my-model",
      outputs: [
        {
          name: "logits",
          shape: [1, 100],
          datatype: "FP32",
          data: [],
        },
      ],
    };

    expect(() => parseV2ChatResponse(response, "my-model")).toThrow(
      "text_output"
    );
  });
});

describe("parseV2TextResponse", () => {
  it("returns LLMResult with text generation", () => {
    const response = {
      id: "abc",
      model_name: "base-model",
      outputs: [
        {
          name: "text_output",
          shape: [1],
          datatype: "BYTES",
          data: ["The answer is 42"],
        },
      ],
    };

    const result = parseV2TextResponse(response, "base-model");
    expect(result.generations).toHaveLength(1);
    expect(result.generations[0][0].text).toBe("The answer is 42");
  });
});

describe("parseV2ChatStreamChunk", () => {
  it("parses a streaming chunk with text delta", () => {
    const raw = JSON.stringify({
      id: "stream-1",
      model_name: "my-model",
      outputs: [
        {
          name: "text_output",
          shape: [1],
          datatype: "BYTES",
          data: ["Hello "],
          parameters: { sequence_end: false },
        },
      ],
    });

    const chunk = parseV2ChatStreamChunk(raw, "my-model");
    expect(chunk).not.toBeNull();
    expect(chunk!.text).toBe("Hello ");
  });

  it("marks finish reason as stop on sequence_end", () => {
    const raw = JSON.stringify({
      id: "stream-2",
      model_name: "my-model",
      outputs: [
        {
          name: "text_output",
          shape: [1],
          datatype: "BYTES",
          data: ["world"],
          parameters: { sequence_end: true },
        },
      ],
    });

    const chunk = parseV2ChatStreamChunk(raw, "my-model");
    const info = chunk!.generationInfo as { finishReason: string };
    expect(info.finishReason).toBe("stop");
  });

  it("returns null for invalid JSON", () => {
    expect(parseV2ChatStreamChunk("not json", "model")).toBeNull();
  });

  it("returns null when no output tensor found", () => {
    const raw = JSON.stringify({
      id: "stream-3",
      model_name: "my-model",
      outputs: [],
    });
    expect(parseV2ChatStreamChunk(raw, "my-model")).toBeNull();
  });
});

describe("parseV2TextStreamChunk", () => {
  it("parses text delta from streaming chunk", () => {
    const raw = JSON.stringify({
      id: "s1",
      model_name: "m",
      outputs: [{ name: "text_output", shape: [1], datatype: "BYTES", data: ["token"] }],
    });
    const chunk = parseV2TextStreamChunk(raw, "m");
    expect(chunk?.text).toBe("token");
  });
});

describe("getV2InferPath", () => {
  it("builds correct path", () => {
    expect(getV2InferPath("my-model")).toBe("/v2/models/my-model/infer");
  });

  it("URL-encodes model names with special characters", () => {
    expect(getV2InferPath("my model/v1")).toBe("/v2/models/my%20model%2Fv1/infer");
  });
});
