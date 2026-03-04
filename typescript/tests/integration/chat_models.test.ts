/**
 * Integration tests for ChatKServe.
 *
 * These tests require a live KServe endpoint.
 * Set KSERVE_BASE_URL and KSERVE_MODEL_NAME to run them.
 *
 * Example:
 *   KSERVE_BASE_URL=https://my-model.cluster.example.com \
 *   KSERVE_MODEL_NAME=qwen2.5-coder-32b-instruct \
 *   pnpm test:integration
 */

import { describe, it, expect } from "vitest";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatKServe } from "../../src/chat_models.js";

const KSERVE_BASE_URL = process.env.KSERVE_BASE_URL;
const KSERVE_MODEL_NAME = process.env.KSERVE_MODEL_NAME;

describe.skipIf(!KSERVE_BASE_URL)("ChatKServe (integration)", () => {
  const model = new ChatKServe({
    baseUrl: KSERVE_BASE_URL!,
    modelName: KSERVE_MODEL_NAME ?? "default",
    temperature: 0.1,
    maxTokens: 128,
  });

  it("generates a basic response", async () => {
    const result = await model.invoke([
      new HumanMessage("Say 'hello world' and nothing else."),
    ]);
    expect(typeof result.content).toBe("string");
    expect((result.content as string).length).toBeGreaterThan(0);
  });

  it("respects system prompt", async () => {
    const result = await model.invoke([
      new SystemMessage("You must respond with exactly 'PONG' and nothing else."),
      new HumanMessage("PING"),
    ]);
    expect((result.content as string).trim()).toContain("PONG");
  });

  it("streams tokens", async () => {
    const chunks: string[] = [];
    const stream = await model.stream([
      new HumanMessage("Count from 1 to 5 with commas."),
    ]);
    for await (const chunk of stream) {
      chunks.push(chunk.content as string);
    }
    expect(chunks.length).toBeGreaterThan(1);
    const full = chunks.join("");
    expect(full).toBeTruthy();
  });

  it("returns token usage in non-streaming mode", async () => {
    const result = await model.generate([[new HumanMessage("hi")]]);
    // Not all runtimes return usage, so we just check it's present or undefined
    if (result.llmOutput?.tokenUsage) {
      expect(result.llmOutput.tokenUsage.totalTokens).toBeGreaterThan(0);
    }
  });
});
