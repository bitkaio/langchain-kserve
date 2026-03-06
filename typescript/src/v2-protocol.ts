/**
 * V2 Inference Protocol (Open Inference Protocol) request/response mapping.
 *
 * The V2 protocol is the native KServe protocol used by runtimes like
 * Triton Inference Server. It sends/receives tensor-like data structures.
 *
 * For LLM inference:
 * - Input: text_input (BYTES, shape [1])
 * - Output: text_output (BYTES, shape [1])
 *
 * Generation parameters are passed in the top-level `parameters` field.
 */

import type { BaseMessage } from "@langchain/core/messages";
import {
  AIMessage,
  AIMessageChunk,
  isHumanMessage,
  isSystemMessage,
} from "@langchain/core/messages";
import type { ChatGeneration, Generation, LLMResult } from "@langchain/core/outputs";
import {
  ChatGenerationChunk,
  GenerationChunk,
} from "@langchain/core/outputs";
import { v4 as uuidv4 } from "uuid";

import { KServeInferenceError } from "./errors.js";
import type {
  ChatKServeCallOptions,
  KServeGenerationInfo,
  OpenAIContentBlock,
  V2InferRequest,
  V2InferResponse,
  V2StreamChunk,
} from "./types.js";
import { formatMessagesToPrompt } from "./utils.js";

// ============================================================
// Request building
// ============================================================

/**
 * Build a V2 inference request for a chat model.
 *
 * Formats messages to a prompt string using the specified chat template,
 * then wraps it in the V2 tensor format.
 *
 * Throws KServeInferenceError if tools or image_url content blocks are present,
 * as these features require the OpenAI-compatible protocol.
 *
 * @param messages       - LangChain messages to convert
 * @param params         - Generation parameters
 * @param chatTemplate   - Chat template format
 * @param customTemplate - Custom template string (if chatTemplate === "custom")
 * @param options        - Optional call options (checked for tools)
 */
export function buildV2ChatRequest(
  messages: BaseMessage[],
  params: {
    temperature?: number;
    maxTokens?: number;
    topP?: number;
    stop?: string[];
  },
  chatTemplate: "chatml" | "llama" | "custom" = "chatml",
  customTemplate?: string,
  options?: Partial<ChatKServeCallOptions>
): V2InferRequest {
  // Tool calling is not supported with V2 protocol
  if (options?.tools && options.tools.length > 0) {
    throw new KServeInferenceError(
      "Tool calling is only supported with the OpenAI-compatible protocol. " +
      "Set protocol='openai' or use a runtime that exposes the OpenAI-compatible API (e.g., vLLM)."
    );
  }

  // Vision / multimodal is not supported with V2 protocol
  for (const msg of messages) {
    if (
      (isHumanMessage(msg) || isSystemMessage(msg)) &&
      Array.isArray(msg.content)
    ) {
      const hasImageUrl = (msg.content as Array<unknown>).some(
        (block) =>
          typeof block === "object" &&
          block !== null &&
          (block as OpenAIContentBlock).type === "image_url"
      );
      if (hasImageUrl) {
        throw new KServeInferenceError(
          "Multimodal/vision messages are only supported with the OpenAI-compatible protocol. " +
          "Set protocol='openai' to use vision features."
        );
      }
    }
  }

  const prompt = formatMessagesToPrompt(messages, chatTemplate, customTemplate);
  return buildV2TextRequest(prompt, params);
}

/**
 * Build a V2 inference request for a completion (non-chat) model.
 *
 * @param prompt - Raw text prompt
 * @param params - Generation parameters
 */
export function buildV2TextRequest(
  prompt: string,
  params: {
    temperature?: number;
    maxTokens?: number;
    topP?: number;
    stop?: string[];
  }
): V2InferRequest {
  const parameters: Record<string, unknown> = {};

  if (params.temperature !== undefined) {
    parameters.temperature = params.temperature;
  }
  if (params.maxTokens !== undefined) {
    parameters.max_tokens = params.maxTokens;
    // Some runtimes use max_new_tokens
    parameters.max_new_tokens = params.maxTokens;
  }
  if (params.topP !== undefined) {
    parameters.top_p = params.topP;
  }
  if (params.stop && params.stop.length > 0) {
    parameters.stop = params.stop;
  }

  return {
    id: uuidv4(),
    inputs: [
      {
        name: "text_input",
        shape: [1],
        datatype: "BYTES",
        data: [prompt],
      },
    ],
    parameters: Object.keys(parameters).length > 0 ? parameters : undefined,
  };
}

// ============================================================
// Response parsing
// ============================================================

/**
 * Extract the text output from a V2 inference response.
 */
function extractV2OutputText(response: V2InferResponse): string {
  const output = response.outputs.find(
    (o) => o.name === "text_output" || o.name === "output"
  );
  if (!output) {
    throw new KServeInferenceError(
      `V2 response did not contain a 'text_output' tensor. ` +
        `Available outputs: ${response.outputs.map((o) => o.name).join(", ")}`
    );
  }
  const data = output.data[0];
  if (typeof data !== "string") {
    throw new KServeInferenceError(
      `V2 text_output data is not a string: ${String(data)}`
    );
  }
  return data;
}

/**
 * Parse a V2 response into a LangChain ChatGeneration.
 *
 * @param response  - Raw V2 response
 * @param modelName - Model name for metadata
 */
export function parseV2ChatResponse(
  response: V2InferResponse,
  modelName: string
): ChatGeneration {
  const text = extractV2OutputText(response);

  const generationInfo: KServeGenerationInfo = {
    modelName: response.model_name ?? modelName,
    protocol: "v2",
    // V2 protocol typically does not return token counts
    finishReason: null,
  };

  return {
    message: new AIMessage({ content: text }),
    text,
    generationInfo,
  };
}

/**
 * Parse a V2 response into an LLMResult (for KServeLLM).
 *
 * @param response  - Raw V2 response
 * @param modelName - Model name for metadata
 */
export function parseV2TextResponse(
  response: V2InferResponse,
  modelName: string
): LLMResult {
  const text = extractV2OutputText(response);

  const generationInfo: KServeGenerationInfo = {
    modelName: response.model_name ?? modelName,
    protocol: "v2",
    finishReason: null,
  };

  const generation: Generation = {
    text,
    generationInfo,
  };

  return {
    generations: [[generation]],
  };
}

// ============================================================
// Streaming parsing
// ============================================================

/**
 * Parse a V2 streaming chunk (NDJSON line) into a ChatGenerationChunk.
 *
 * Returns null if the chunk carries no text delta.
 */
export function parseV2ChatStreamChunk(
  raw: string,
  modelName: string
): ChatGenerationChunk | null {
  let chunk: V2StreamChunk;
  try {
    chunk = JSON.parse(raw) as V2StreamChunk;
  } catch {
    return null;
  }

  const output = chunk.outputs?.find(
    (o) => o.name === "text_output" || o.name === "output"
  );
  if (!output) return null;

  const text = output.data[0] ?? "";
  const isEnd = output.parameters?.sequence_end === true;

  const generationInfo: KServeGenerationInfo & Record<string, unknown> = {
    modelName: chunk.model_name ?? modelName,
    protocol: "v2",
    finishReason: isEnd ? "stop" : null,
  };

  return new ChatGenerationChunk({
    text,
    message: new AIMessageChunk({ content: text }),
    generationInfo,
  });
}

/**
 * Parse a V2 streaming chunk into a GenerationChunk (for KServeLLM).
 */
export function parseV2TextStreamChunk(
  raw: string,
  modelName: string
): GenerationChunk | null {
  let chunk: V2StreamChunk;
  try {
    chunk = JSON.parse(raw) as V2StreamChunk;
  } catch {
    return null;
  }

  const output = chunk.outputs?.find(
    (o) => o.name === "text_output" || o.name === "output"
  );
  if (!output) return null;

  const text = output.data[0] ?? "";
  const isEnd = output.parameters?.sequence_end === true;

  return new GenerationChunk({
    text,
    generationInfo: {
      modelName: chunk.model_name ?? modelName,
      protocol: "v2" as const,
      finishReason: isEnd ? "stop" : null,
    },
  });
}

/**
 * Build the streaming variant of a V2 request.
 * Adds `stream` parameter to signal streaming mode.
 */
export function buildV2StreamRequest(
  baseRequest: V2InferRequest
): V2InferRequest {
  return {
    ...baseRequest,
    parameters: {
      ...baseRequest.parameters,
      stream: true,
    },
  };
}

// ============================================================
// Path helpers
// ============================================================

/**
 * Get the V2 infer endpoint path for a model.
 *
 * @param modelName - KServe model name
 */
export function getV2InferPath(modelName: string): string {
  return `/v2/models/${encodeURIComponent(modelName)}/infer`;
}
