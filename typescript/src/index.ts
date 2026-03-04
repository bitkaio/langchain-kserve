/**
 * @langchain/kserve — LangChain.js integration for KServe inference services.
 *
 * @packageDocumentation
 */

// Model classes
export { ChatKServe } from "./chat_models.js";
export { KServeLLM } from "./llms.js";

// Types
export type {
  ChatKServeInput,
  ChatKServeCallOptions,
  KServeLLMInput,
  KServeLLMCallOptions,
  KServeProtocol,
  ChatTemplateFormat,
  KServeBaseInput,
  KServeClientConfig,
  RequestOptions,
  KServeGenerationInfo,
  // V2 protocol types
  V2InferRequest,
  V2InferResponse,
  V2InferInput,
  V2InferOutput,
  V2StreamChunk,
  // OpenAI-compat types
  OpenAIChatMessage,
  OpenAIChatRequest,
  OpenAIChatResponse,
  OpenAITool,
  OpenAIToolCall,
  OpenAIToolChoice,
  OpenAIUsage,
} from "./types.js";

// Error classes
export {
  KServeError,
  KServeConnectionError,
  KServeAuthenticationError,
  KServeModelNotFoundError,
  KServeInferenceError,
  KServeTimeoutError,
} from "./errors.js";

// Low-level client (for advanced usage)
export { KServeClient } from "./client.js";

// Utilities (for custom integrations)
export {
  convertMessagesToOpenAI,
  convertToolToOpenAI,
  formatMessagesToPrompt,
  formatChatML,
  formatLlama,
} from "./utils.js";
