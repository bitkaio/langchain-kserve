/**
 * OpenAI-compatible endpoint request building and response parsing.
 *
 * Handles:
 * - /v1/chat/completions (non-streaming + streaming)
 * - /v1/completions (non-streaming + streaming)
 * - Tool call extraction from responses
 */

import type { BaseMessage } from "@langchain/core/messages";
import {
  AIMessage,
  AIMessageChunk,
} from "@langchain/core/messages";
import type { ChatGeneration, LLMResult } from "@langchain/core/outputs";
import { ChatGenerationChunk } from "@langchain/core/outputs";

import { KServeInferenceError } from "./errors.js";
import type {
  ChatKServeCallOptions,
  KServeGenerationInfo,
  OpenAIChatRequest,
  OpenAIChatResponse,
  OpenAIChatStreamChunk,
  OpenAICompletionRequest,
  OpenAICompletionResponse,
  OpenAICompletionStreamChunk,
  OpenAITool,
  OpenAIToolCall,
} from "./types.js";
import { convertMessagesToOpenAI } from "./utils.js";

// ============================================================
// Chat completions
// ============================================================

/**
 * Build an OpenAI chat completions request body.
 *
 * @param modelName - Model identifier
 * @param messages  - LangChain messages to convert
 * @param params    - Generation parameters
 * @param tools     - Optional tool definitions
 * @param stream    - Whether to request streaming
 */
export function buildChatRequest(
  modelName: string,
  messages: BaseMessage[],
  params: {
    temperature?: number;
    maxTokens?: number;
    topP?: number;
    stop?: string[];
  },
  options?: Partial<ChatKServeCallOptions>,
  stream = false
): OpenAIChatRequest {
  const request: OpenAIChatRequest = {
    model: modelName,
    messages: convertMessagesToOpenAI(messages),
    stream,
  };

  const temperature = options?.temperature ?? params.temperature;
  const maxTokens = options?.maxTokens ?? params.maxTokens;

  if (temperature !== undefined) request.temperature = temperature;
  if (maxTokens !== undefined) request.max_tokens = maxTokens;
  if (params.topP !== undefined) request.top_p = params.topP;

  const stop = options?.stop ?? params.stop;
  if (stop && stop.length > 0) request.stop = stop;

  // Tool calling
  const tools: OpenAITool[] | undefined = options?.tools;
  if (tools && tools.length > 0) {
    request.tools = tools;
    if (options?.toolChoice !== undefined) {
      // Cast needed because our union type is wider than OpenAIChatRequest expects
      request.tool_choice = options.toolChoice as typeof request.tool_choice;
    }
  }

  // Request usage in streaming chunks (vLLM supports this)
  if (stream) {
    request.stream_options = { include_usage: true };
  }

  return request;
}

/**
 * Parse an OpenAI chat completions response into a LangChain ChatGeneration.
 *
 * @param response  - Raw API response
 * @param modelName - Model name for metadata
 */
export function parseChatResponse(
  response: OpenAIChatResponse,
  modelName: string
): ChatGeneration {
  const choice = response.choices[0];
  if (!choice) {
    throw new KServeInferenceError("OpenAI response contained no choices");
  }

  const msg = choice.message;
  const toolCalls = msg.tool_calls;

  const generationInfo: KServeGenerationInfo = {
    modelName: response.model ?? modelName,
    protocol: "openai",
    finishReason: choice.finish_reason,
  };

  if (response.usage) {
    generationInfo.tokenUsage = {
      promptTokens: response.usage.prompt_tokens,
      completionTokens: response.usage.completion_tokens,
      totalTokens: response.usage.total_tokens,
    };
  }

  let aiMessage: AIMessage;

  if (toolCalls && toolCalls.length > 0) {
    aiMessage = new AIMessage({
      content: msg.content ?? "",
      tool_calls: toolCalls.map((tc: OpenAIToolCall) => ({
        id: tc.id,
        name: tc.function.name,
        args: parseToolCallArgs(tc.function.arguments),
        type: "tool_call" as const,
      })),
    });
  } else {
    aiMessage = new AIMessage({ content: msg.content ?? "" });
  }

  return {
    message: aiMessage,
    text: msg.content ?? "",
    generationInfo,
  };
}

/**
 * Parse tool call arguments string to an object.
 * Falls back to wrapping in `{ input: ... }` if JSON.parse fails.
 */
function parseToolCallArgs(argsString: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(argsString);
    if (typeof parsed === "object" && parsed !== null) {
      return parsed as Record<string, unknown>;
    }
    return { input: parsed };
  } catch {
    return { input: argsString };
  }
}

/**
 * Parse a streaming OpenAI chat chunk and return a ChatGenerationChunk.
 * Returns null if the chunk carries no useful delta.
 */
export function parseChatStreamChunk(
  raw: string,
  modelName: string
): ChatGenerationChunk | null {
  let chunk: OpenAIChatStreamChunk;
  try {
    chunk = JSON.parse(raw) as OpenAIChatStreamChunk;
  } catch {
    return null;
  }

  const choice = chunk.choices?.[0];
  if (!choice) return null;

  const delta = choice.delta;
  const content = delta.content ?? "";
  const finishReason = choice.finish_reason ?? undefined;

  const generationInfo: KServeGenerationInfo & Record<string, unknown> = {
    modelName: chunk.model ?? modelName,
    protocol: "openai",
    finishReason,
  };

  if (chunk.usage) {
    generationInfo.tokenUsage = {
      promptTokens: chunk.usage.prompt_tokens,
      completionTokens: chunk.usage.completion_tokens,
      totalTokens: chunk.usage.total_tokens,
    };
  }

  // Handle streaming tool calls
  if (delta.tool_calls && delta.tool_calls.length > 0) {
    const messageChunk = new AIMessageChunk({
      content,
      tool_call_chunks: delta.tool_calls.map((t) => ({
        index: t.index,
        id: t.id,
        name: t.function?.name,
        args: t.function?.arguments,
        type: "tool_call_chunk" as const,
      })),
    });
    return new ChatGenerationChunk({
      text: content,
      message: messageChunk,
      generationInfo,
    });
  }

  return new ChatGenerationChunk({
    text: content,
    message: new AIMessageChunk({ content }),
    generationInfo,
  });
}

// ============================================================
// Text completions
// ============================================================

/**
 * Build an OpenAI completions (non-chat) request body.
 */
export function buildCompletionRequest(
  modelName: string,
  prompt: string,
  params: {
    temperature?: number;
    maxTokens?: number;
    topP?: number;
    stop?: string[];
  },
  stream = false
): OpenAICompletionRequest {
  const request: OpenAICompletionRequest = {
    model: modelName,
    prompt,
    stream,
  };

  if (params.temperature !== undefined) request.temperature = params.temperature;
  if (params.maxTokens !== undefined) request.max_tokens = params.maxTokens;
  if (params.topP !== undefined) request.top_p = params.topP;
  if (params.stop && params.stop.length > 0) request.stop = params.stop;

  return request;
}

/**
 * Parse an OpenAI completions response into an LLMResult.
 */
export function parseCompletionResponse(
  response: OpenAICompletionResponse,
  modelName: string
): LLMResult {
  const generationInfo: KServeGenerationInfo = {
    modelName: response.model ?? modelName,
    protocol: "openai",
  };

  if (response.usage) {
    generationInfo.tokenUsage = {
      promptTokens: response.usage.prompt_tokens,
      completionTokens: response.usage.completion_tokens,
      totalTokens: response.usage.total_tokens,
    };
  }

  return {
    generations: [
      response.choices.map((choice) => ({
        text: choice.text,
        generationInfo: {
          ...generationInfo,
          finishReason: choice.finish_reason,
        },
      })),
    ],
    llmOutput: generationInfo.tokenUsage
      ? { tokenUsage: generationInfo.tokenUsage }
      : undefined,
  };
}

/**
 * Parse a streaming OpenAI completion chunk.
 * Returns the text delta, or null if nothing to emit.
 */
export function parseCompletionStreamChunk(raw: string): string | null {
  let chunk: OpenAICompletionStreamChunk;
  try {
    chunk = JSON.parse(raw) as OpenAICompletionStreamChunk;
  } catch {
    return null;
  }

  const text = chunk.choices?.[0]?.text;
  return text ?? null;
}
