/**
 * Types and interfaces for the @langchain/kserve package.
 */

import type { BaseChatModelParams } from "@langchain/core/language_models/chat_models";
import type { BaseLLMParams } from "@langchain/core/language_models/llms";

// ============================================================
// Protocol types
// ============================================================

/** Supported KServe inference protocols */
export type KServeProtocol = "openai" | "v2" | "auto";

/** Chat template format for V2 protocol message formatting */
export type ChatTemplateFormat = "chatml" | "llama" | "custom";

// ============================================================
// Shared constructor input
// ============================================================

/** Base configuration shared between ChatKServe and KServeLLM */
export interface KServeBaseInput {
  /** Base URL of the KServe inference service (e.g., https://my-model.cluster.example.com) */
  baseUrl: string;

  /** Model name as registered in KServe */
  modelName: string;

  /** Protocol to use for inference. Defaults to "auto" (auto-detect). */
  protocol?: KServeProtocol;

  /** Static API key / bearer token for authentication */
  apiKey?: string;

  /**
   * Dynamic token provider function.
   * Called before each request to obtain a fresh bearer token.
   * Useful for Kubernetes service account tokens or OAuth flows.
   */
  tokenProvider?: () => Promise<string>;

  /** Whether to verify SSL certificates. Defaults to true. */
  verifySsl?: boolean;

  /** Path to a custom CA certificate bundle (PEM format). */
  caBundle?: string;

  /** Temperature for sampling (0–2). */
  temperature?: number;

  /** Maximum number of tokens to generate. */
  maxTokens?: number;

  /** Top-p nucleus sampling parameter. */
  topP?: number;

  /** Stop sequences. */
  stop?: string[];

  /** Whether to stream the response. */
  streaming?: boolean;

  /** Request timeout in milliseconds. Defaults to 120000 (2 min for cold starts). */
  timeout?: number;

  /** Maximum number of retries on transient failures. Defaults to 3. */
  maxRetries?: number;
}

/** Constructor input for ChatKServe */
export interface ChatKServeInput extends KServeBaseInput, BaseChatModelParams {
  /**
   * Chat template format used when calling V2 protocol.
   * Defaults to "chatml" which works for Qwen models and most modern instruct models.
   */
  chatTemplate?: ChatTemplateFormat;

  /**
   * Custom Jinja-like template string (used when chatTemplate = "custom").
   * Must produce a single string from the messages array.
   */
  customChatTemplate?: string;

  /** Whether to return logprobs with each token. */
  logprobs?: boolean;

  /** Number of top logprobs to return per token (requires logprobs=true). */
  topLogprobs?: number;

  /** Whether to allow the model to make parallel tool calls. */
  parallelToolCalls?: boolean;
}

/** Constructor input for KServeLLM */
export interface KServeLLMInput extends KServeBaseInput, BaseLLMParams {}

// ============================================================
// Call options
// ============================================================

/** Per-call overrides for ChatKServe */
export interface ChatKServeCallOptions {
  /** Stop sequences (override constructor) */
  stop?: string[];
  /** Maximum tokens (override constructor) */
  maxTokens?: number;
  /** Temperature (override constructor) */
  temperature?: number;
  /** OpenAI-compatible tool definitions */
  tools?: OpenAITool[];
  /** Tool choice strategy */
  toolChoice?: string | OpenAIToolChoice;
  /** Whether to allow parallel tool calls */
  parallelToolCalls?: boolean;
  /** Whether to return logprobs */
  logprobs?: boolean;
  /** Number of top logprobs to return per token */
  topLogprobs?: number;
  /** AbortSignal for cancellation */
  signal?: AbortSignal;
}

/** Per-call overrides for KServeLLM */
export interface KServeLLMCallOptions {
  stop?: string[];
  maxTokens?: number;
  temperature?: number;
  signal?: AbortSignal;
}

// ============================================================
// V2 Inference Protocol (Open Inference Protocol)
// ============================================================

/** V2 inference request body */
export interface V2InferRequest {
  id: string;
  inputs: V2InferInput[];
  parameters?: Record<string, unknown>;
  outputs?: V2InferOutputSpec[];
}

/** A single input tensor in V2 format */
export interface V2InferInput {
  name: string;
  shape: number[];
  datatype: string;
  data: string[] | number[] | boolean[];
  parameters?: Record<string, unknown>;
}

/** Output spec in V2 request */
export interface V2InferOutputSpec {
  name: string;
  parameters?: Record<string, unknown>;
}

/** V2 inference response body */
export interface V2InferResponse {
  id: string;
  model_name: string;
  model_version?: string;
  outputs: V2InferOutput[];
}

/** A single output tensor in V2 format */
export interface V2InferOutput {
  name: string;
  shape: number[];
  datatype: string;
  data: string[] | number[] | boolean[];
  parameters?: Record<string, unknown>;
}

/** V2 streaming chunk (NDJSON) */
export interface V2StreamChunk {
  id: string;
  model_name: string;
  outputs: Array<{
    name: string;
    shape: number[];
    datatype: string;
    data: string[];
    parameters?: {
      sequence_end?: boolean;
      sequence_id?: number;
    };
  }>;
}

// ============================================================
// OpenAI-compatible types
// ============================================================

/** OpenAI chat message */
export interface OpenAIChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null | OpenAIContentBlock[];
  name?: string;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

/** OpenAI tool definition */
export interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
    strict?: boolean;
  };
}

/** OpenAI tool choice */
export type OpenAIToolChoice =
  | "none"
  | "auto"
  | "required"
  | { type: "function"; function: { name: string } };

/** OpenAI tool call */
export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

/** OpenAI chat completion request */
export interface OpenAIChatRequest {
  model: string;
  messages: OpenAIChatMessage[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  stop?: string | string[];
  stream?: boolean;
  tools?: OpenAITool[];
  tool_choice?: OpenAIToolChoice;
  parallel_tool_calls?: boolean;
  logprobs?: boolean;
  top_logprobs?: number;
  stream_options?: {
    include_usage?: boolean;
  };
}

/** OpenAI completions request (non-chat) */
export interface OpenAICompletionRequest {
  model: string;
  prompt: string | string[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  stop?: string | string[];
  stream?: boolean;
}

/** OpenAI chat completion choice */
export interface OpenAIChatChoice {
  index: number;
  message: {
    role: string;
    content: string | null;
    tool_calls?: OpenAIToolCall[];
  };
  finish_reason: string | null;
  logprobs?: OpenAILogprobs | null;
}

/** OpenAI token usage */
export interface OpenAIUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** OpenAI chat completion response */
export interface OpenAIChatResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: OpenAIChatChoice[];
  usage?: OpenAIUsage;
}

/** OpenAI streaming delta */
export interface OpenAIChatStreamDelta {
  role?: string;
  content?: string | null;
  tool_calls?: Array<{
    index: number;
    id?: string;
    type?: string;
    function?: {
      name?: string;
      arguments?: string;
    };
  }>;
}

/** OpenAI streaming chunk choice */
export interface OpenAIChatStreamChoice {
  index: number;
  delta: OpenAIChatStreamDelta;
  finish_reason: string | null;
  logprobs?: OpenAILogprobs | null;
}

/** OpenAI streaming chunk */
export interface OpenAIChatStreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: OpenAIChatStreamChoice[];
  usage?: OpenAIUsage;
}

/** OpenAI completion choice */
export interface OpenAICompletionChoice {
  index: number;
  text: string;
  finish_reason: string | null;
  logprobs: null;
}

/** OpenAI completion response */
export interface OpenAICompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: OpenAICompletionChoice[];
  usage?: OpenAIUsage;
}

/** OpenAI completion streaming chunk */
export interface OpenAICompletionStreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    text: string;
    finish_reason: string | null;
  }>;
  usage?: OpenAIUsage;
}

// ============================================================
// Internal client types
// ============================================================

/** HTTP client configuration */
export interface KServeClientConfig {
  baseUrl: string;
  apiKey?: string;
  tokenProvider?: () => Promise<string>;
  verifySsl?: boolean;
  caBundle?: string;
  timeout?: number;
  maxRetries?: number;
}

/** Options for a single HTTP request */
export interface RequestOptions {
  signal?: AbortSignal;
  headers?: Record<string, string>;
}

/** Generation metadata included in generationInfo */
export interface KServeGenerationInfo {
  modelName: string;
  protocol: "openai" | "v2";
  finishReason?: string | null;
  tokenUsage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  logprobs?: OpenAILogprobs | null;
}

// ============================================================
// Model introspection
// ============================================================

/** Metadata returned from getModelInfo() */
export interface KServeModelInfo {
  modelName: string;
  modelVersion?: string;
  platform?: string;
  inputs?: Array<Record<string, unknown>>;
  outputs?: Array<Record<string, unknown>>;
  raw: Record<string, unknown>;
}

// ============================================================
// Logprobs types
// ============================================================

/** A single token logprob entry */
export interface OpenAILogprobItem {
  token: string;
  logprob: number;
  top_logprobs: Array<{ token: string; logprob: number }>;
}

/** Logprobs for a chat completion choice */
export interface OpenAILogprobs {
  content: OpenAILogprobItem[] | null;
}

// ============================================================
// Multimodal / vision content block types
// ============================================================

/** A text content block for multimodal messages */
export interface OpenAITextContentBlock {
  type: "text";
  text: string;
}

/** An image_url content block for multimodal messages */
export interface OpenAIImageContentBlock {
  type: "image_url";
  image_url: { url: string; detail?: "low" | "high" | "auto" };
}

/** Union of supported OpenAI content block types */
export type OpenAIContentBlock = OpenAITextContentBlock | OpenAIImageContentBlock;
