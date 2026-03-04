/**
 * Integration tests for KServeLLM.
 *
 * Requires a live KServe endpoint that exposes a text completion model.
 * Set KSERVE_BASE_URL and KSERVE_MODEL_NAME (and optionally KSERVE_PROTOCOL).
 */

import { describe, it, expect } from "vitest";
import { KServeLLM } from "../../src/llms.js";

const KSERVE_BASE_URL = process.env.KSERVE_BASE_URL;
const KSERVE_MODEL_NAME = process.env.KSERVE_MODEL_NAME;

describe.skipIf(!KSERVE_BASE_URL)("KServeLLM (integration)", () => {
  const llm = new KServeLLM({
    baseUrl: KSERVE_BASE_URL!,
    modelName: KSERVE_MODEL_NAME ?? "default",
    temperature: 0.1,
    maxTokens: 64,
  });

  it("generates a completion", async () => {
    const result = await llm.invoke("The capital of France is");
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
  });

  it("streams completion tokens", async () => {
    const chunks: string[] = [];
    const stream = await llm.stream("Once upon a time");
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    expect(chunks.length).toBeGreaterThan(0);
  });
});
