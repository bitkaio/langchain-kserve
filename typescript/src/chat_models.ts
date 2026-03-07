/**
 * ChatKServe — LangChain chat model for KServe inference services.
 *
 * Supports:
 * - OpenAI-compatible API (/v1/chat/completions)
 * - V2 Inference Protocol (/v2/models/{model}/infer)
 * - Automatic protocol detection
 * - Streaming (SSE for OpenAI, NDJSON for V2)
 * - Tool calling (OpenAI-compatible protocol only)
 * - Bearer token auth + dynamic token providers
 * - Custom TLS / CA bundle
 *
 * @example
 * ```typescript
 * import { ChatKServe } from "@langchain/kserve";
 *
 * const llm = new ChatKServe({
 *   baseUrl: "https://qwen-coder.my-cluster.example.com",
 *   modelName: "qwen2.5-coder-32b-instruct",
 *   temperature: 0.2,
 * });
 *
 * const response = await llm.invoke("Write a TypeScript binary search function.");
 * ```
 */

import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { BaseMessage } from "@langchain/core/messages";
import type { ChatGeneration, ChatResult } from "@langchain/core/outputs";
import { ChatGenerationChunk } from "@langchain/core/outputs";
import { RunnableBinding, RunnableLambda } from "@langchain/core/runnables";
import type { Runnable } from "@langchain/core/runnables";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import type { ZodType } from "zod";
import zodToJsonSchema from "zod-to-json-schema";

import { KServeClient } from "./client.js";
import {
  buildChatRequest,
  parseChatResponse,
  parseChatStreamChunk,
} from "./openai-compat.js";
import {
  buildV2ChatRequest,
  buildV2StreamRequest,
  getV2InferPath,
  parseV2ChatResponse,
  parseV2ChatStreamChunk,
} from "./v2-protocol.js";
import type {
  ChatKServeCallOptions,
  ChatKServeInput,
  KServeGenerationInfo,
  KServeModelInfo,
  OpenAIResponseFormat,
  OpenAITool,
  V2InferResponse,
} from "./types.js";
import { convertToolToOpenAI } from "./utils.js";
import { KServeInferenceError } from "./errors.js";

/**
 * ChatKServe — LangChain BaseChatModel implementation for KServe.
 */
export class ChatKServe extends BaseChatModel<ChatKServeCallOptions> {
  // LangChain serialization key
  static lc_name(): string {
    return "ChatKServe";
  }

  // Connection
  private readonly baseUrl: string;
  private readonly modelName: string;
  private readonly protocolPref: "openai" | "v2" | "auto";
  private readonly client: KServeClient;

  // Template (for V2 protocol)
  private readonly chatTemplate: "chatml" | "llama" | "custom";
  private readonly customChatTemplate?: string;

  // Generation params
  private readonly temperature?: number;
  private readonly maxTokens?: number;
  private readonly topP?: number;
  private readonly stop?: string[];
  private readonly streaming: boolean;

  // Logprobs
  private readonly logprobs?: boolean;
  private readonly topLogprobs?: number;

  // Tool calling
  private readonly parallelToolCalls?: boolean;

  // Response format (OpenAI-compatible only)
  private readonly responseFormat?: OpenAIResponseFormat;

  constructor(fields: ChatKServeInput) {
    super(fields);

    this.baseUrl =
      fields.baseUrl ??
      getEnvironmentVariable("KSERVE_BASE_URL") ??
      (() => {
        throw new Error(
          "baseUrl is required. Set it in the constructor or via KSERVE_BASE_URL env var."
        );
      })();

    this.modelName =
      fields.modelName ??
      getEnvironmentVariable("KSERVE_MODEL_NAME") ??
      (() => {
        throw new Error(
          "modelName is required. Set it in the constructor or via KSERVE_MODEL_NAME env var."
        );
      })();

    const protocolFromEnv = getEnvironmentVariable("KSERVE_PROTOCOL") as
      | "openai"
      | "v2"
      | "auto"
      | undefined;

    this.protocolPref = fields.protocol ?? protocolFromEnv ?? "auto";
    this.chatTemplate = fields.chatTemplate ?? "chatml";
    this.customChatTemplate = fields.customChatTemplate;
    this.temperature = fields.temperature;
    this.maxTokens = fields.maxTokens;
    this.topP = fields.topP;
    this.stop = fields.stop;
    this.streaming = fields.streaming ?? false;
    this.logprobs = fields.logprobs;
    this.topLogprobs = fields.topLogprobs;
    this.parallelToolCalls = fields.parallelToolCalls;

    // Validate and store responseFormat
    if (fields.responseFormat !== undefined) {
      if (fields.responseFormat.type === "json_schema") {
        const schemaVal = fields.responseFormat.json_schema?.schema;
        if (schemaVal === undefined || typeof schemaVal !== "object" || schemaVal === null) {
          throw new Error(
            'responseFormat.type "json_schema" requires responseFormat.json_schema.schema to be an object.'
          );
        }
      }
      this.responseFormat = fields.responseFormat;
    }

    this.client = new KServeClient({
      baseUrl: this.baseUrl,
      apiKey:
        fields.apiKey ?? getEnvironmentVariable("KSERVE_API_KEY"),
      tokenProvider: fields.tokenProvider,
      verifySsl: fields.verifySsl,
      caBundle: fields.caBundle ?? getEnvironmentVariable("KSERVE_CA_BUNDLE"),
      timeout: fields.timeout,
      maxRetries: fields.maxRetries,
    });

    // Pre-configure protocol if not "auto"
    if (this.protocolPref !== "auto") {
      this.client.setProtocol(this.protocolPref);
    }
  }

  /** LangChain model type identifier */
  _llmType(): string {
    return "kserve";
  }

  /** Parameters used to identify this model instance (e.g., for caching) */
  _identifyingParams(): Record<string, unknown> {
    return {
      model_name: this.modelName,
      base_url: this.baseUrl,
      protocol: this.protocolPref,
      temperature: this.temperature,
      max_tokens: this.maxTokens,
      top_p: this.topP,
    };
  }

  // --------------------------------------------------------
  // Protocol resolution
  // --------------------------------------------------------

  private async resolveProtocol(
    signal?: AbortSignal
  ): Promise<"openai" | "v2"> {
    if (this.protocolPref !== "auto") {
      return this.protocolPref;
    }
    void signal; // used implicitly inside detectProtocol
    return this.client.detectProtocol();
  }

  // --------------------------------------------------------
  // Core generation (non-streaming)
  // --------------------------------------------------------

  /**
   * Generate a response for a list of messages.
   * Called by LangChain's `invoke()`, `call()`, etc.
   */
  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    // Delegate to streaming if enabled, collect chunks
    if (this.streaming) {
      const chunks: ChatGenerationChunk[] = [];
      for await (const chunk of this._streamResponseChunks(
        messages,
        options,
        runManager
      )) {
        chunks.push(chunk);
      }

      const fullText = chunks.map((c) => c.text).join("");
      const lastInfo = chunks[chunks.length - 1]?.generationInfo as
        | (KServeGenerationInfo & Record<string, unknown>)
        | undefined;
      const tokenUsage = lastInfo?.tokenUsage as
        | KServeGenerationInfo["tokenUsage"]
        | undefined;

      const { AIMessage: AIMsg } = await import("@langchain/core/messages");
      const generation: ChatGeneration = {
        text: fullText,
        message: new AIMsg({ content: fullText }),
        generationInfo: lastInfo,
      };

      return {
        generations: [generation],
        llmOutput: tokenUsage ? { tokenUsage } : undefined,
      };
    }

    const protocol = await this.resolveProtocol(options.signal);

    const effectiveResponseFormat =
      (options as Partial<ChatKServeCallOptions>).responseFormat ?? this.responseFormat;

    if (protocol === "v2" && effectiveResponseFormat !== undefined) {
      throw new KServeInferenceError(
        "Response format constraints are only supported with the OpenAI-compatible protocol. Set protocol: 'openai'."
      );
    }

    if (protocol === "openai") {
      return this._generateOpenAI(messages, options);
    }
    return this._generateV2(messages, options);
  }

  private async _generateOpenAI(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"]
  ): Promise<ChatResult> {
    const mergedOptions: Partial<ChatKServeCallOptions> = {
      ...options,
      logprobs: (options as Partial<ChatKServeCallOptions>).logprobs ?? this.logprobs,
      topLogprobs: (options as Partial<ChatKServeCallOptions>).topLogprobs ?? this.topLogprobs,
      parallelToolCalls: (options as Partial<ChatKServeCallOptions>).parallelToolCalls ?? this.parallelToolCalls,
      responseFormat: (options as Partial<ChatKServeCallOptions>).responseFormat ?? this.responseFormat,
    };
    const request = buildChatRequest(
      this.modelName,
      messages,
      {
        temperature: this.temperature,
        maxTokens: this.maxTokens,
        topP: this.topP,
        stop: this.stop,
      },
      mergedOptions,
      false
    );

    const response = await this.client.request<
      import("./types.js").OpenAIChatResponse
    >("/v1/chat/completions", request, { signal: options.signal });

    const generation = parseChatResponse(response, this.modelName);
    const info = generation.generationInfo as KServeGenerationInfo | undefined;

    return {
      generations: [generation],
      llmOutput: info?.tokenUsage ? { tokenUsage: info.tokenUsage } : undefined,
    };
  }

  private async _generateV2(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"]
  ): Promise<ChatResult> {
    const request = buildV2ChatRequest(
      messages,
      {
        temperature: this.temperature,
        maxTokens: this.maxTokens,
        topP: this.topP,
        stop: this.stop,
      },
      this.chatTemplate,
      this.customChatTemplate
    );

    const response = await this.client.request<V2InferResponse>(
      getV2InferPath(this.modelName),
      request,
      { signal: options.signal }
    );

    const generation = parseV2ChatResponse(response, this.modelName);
    return { generations: [generation] };
  }

  // --------------------------------------------------------
  // Streaming
  // --------------------------------------------------------

  /**
   * Stream response chunks for a list of messages.
   * Called by LangChain's `stream()` method.
   */
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const protocol = await this.resolveProtocol(options.signal);

    const effectiveResponseFormat =
      (options as Partial<ChatKServeCallOptions>).responseFormat ?? this.responseFormat;

    if (protocol === "v2" && effectiveResponseFormat !== undefined) {
      throw new KServeInferenceError(
        "Response format constraints are only supported with the OpenAI-compatible protocol. Set protocol: 'openai'."
      );
    }

    if (protocol === "openai") {
      yield* this._streamOpenAI(messages, options, runManager);
    } else {
      yield* this._streamV2(messages, options, runManager);
    }
  }

  private async *_streamOpenAI(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const mergedOptions: Partial<ChatKServeCallOptions> = {
      ...options,
      logprobs: (options as Partial<ChatKServeCallOptions>).logprobs ?? this.logprobs,
      topLogprobs: (options as Partial<ChatKServeCallOptions>).topLogprobs ?? this.topLogprobs,
      parallelToolCalls: (options as Partial<ChatKServeCallOptions>).parallelToolCalls ?? this.parallelToolCalls,
      responseFormat: (options as Partial<ChatKServeCallOptions>).responseFormat ?? this.responseFormat,
    };
    const request = buildChatRequest(
      this.modelName,
      messages,
      {
        temperature: this.temperature,
        maxTokens: this.maxTokens,
        topP: this.topP,
        stop: this.stop,
      },
      mergedOptions,
      true
    );

    for await (const raw of this.client.streamRequest(
      "/v1/chat/completions",
      request,
      { signal: options.signal }
    )) {
      const chunk = parseChatStreamChunk(raw, this.modelName);
      if (!chunk) continue;
      await runManager?.handleLLMNewToken(chunk.text ?? "");
      yield chunk;
    }
  }

  private async *_streamV2(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const baseRequest = buildV2ChatRequest(
      messages,
      {
        temperature: this.temperature,
        maxTokens: this.maxTokens,
        topP: this.topP,
        stop: this.stop,
      },
      this.chatTemplate,
      this.customChatTemplate
    );

    const streamRequest = buildV2StreamRequest(baseRequest);

    let streamed = false;
    try {
      for await (const raw of this.client.streamNDJSON(
        getV2InferPath(this.modelName),
        streamRequest,
        { signal: options.signal }
      )) {
        const chunk = parseV2ChatStreamChunk(raw, this.modelName);
        if (!chunk) continue;
        streamed = true;
        await runManager?.handleLLMNewToken(chunk.text ?? "");
        yield chunk;
      }
    } catch (err) {
      if (!streamed) {
        // V2 streaming not supported — fall back to non-streaming
        console.warn(
          "[ChatKServe] V2 streaming not available, falling back to non-streaming"
        );
        const result = await this._generateV2(messages, options);
        const gen = result.generations[0];
        if (!gen) throw new KServeInferenceError("No generation in V2 response");
        yield {
          text: gen.text,
          message: gen.message,
          generationInfo: gen.generationInfo,
        } as ChatGenerationChunk;
        return;
      }
      throw err;
    }
  }

  // --------------------------------------------------------
  // Tool binding
  // --------------------------------------------------------

  /**
   * Bind tools to the model, converting them to OpenAI function-calling schema.
   *
   * Note: Tool calling requires the OpenAI-compatible protocol.
   * When protocol="v2", tool calling is not supported.
   *
   * @param tools  - LangChain tool definitions or OpenAI tool schemas
   * @param kwargs - Additional call options
   */
  bindTools(
    tools: (StructuredToolInterface | Record<string, unknown>)[],
    kwargs?: Partial<ChatKServeCallOptions>
  ): Runnable<BaseLanguageModelInput, import("@langchain/core/messages").AIMessageChunk, ChatKServeCallOptions> {
    const openAITools: OpenAITool[] = tools.map((t) =>
      convertToolToOpenAI(t)
    );

    return new RunnableBinding({
      bound: this,
      kwargs: { tools: openAITools, ...kwargs },
      config: {},
    }) as Runnable<BaseLanguageModelInput, import("@langchain/core/messages").AIMessageChunk, ChatKServeCallOptions>;
  }

  // --------------------------------------------------------
  // Structured output
  // --------------------------------------------------------

  /**
   * Return a runnable that produces structured output of type T.
   *
   * Three strategies are supported:
   * - "functionCalling" (default): binds a tool and extracts tool call args
   * - "jsonSchema": sets response_format to json_schema and parses content
   * - "jsonMode": sets response_format to json_object and parses content
   *
   * @param schema - Zod schema or plain JSON Schema object describing the output
   * @param config - Strategy options
   */
  withStructuredOutput<T = Record<string, unknown>>(
    schema: unknown,
    config?: {
      method?: string;
      includeRaw?: boolean;
      strict?: boolean;
      name?: string;
    }
  ): Runnable<BaseLanguageModelInput, T> {
    // Determine if schema is a Zod schema by checking for _def (Zod v3) or _zod (Zod v4)
    const schemaAsObj = schema as Record<string, unknown>;
    const isZodSchema =
      schema !== null &&
      typeof schema === "object" &&
      typeof schemaAsObj._def !== "undefined" &&
      schemaAsObj._def !== null;

    // Derive schema name
    let schemaName: string;
    if (config?.name) {
      schemaName = config.name;
    } else if (isZodSchema) {
      // Zod v4 stores description via globalRegistry (accessible as schema.description)
      // Zod v3 stores it in schema._def.description
      const desc =
        (schemaAsObj.description as string | undefined) ??
        ((schemaAsObj._def as Record<string, unknown> | undefined)
          ?.description as string | undefined);
      schemaName = desc ?? "output_schema";
    } else {
      schemaName = "output_schema";
    }

    // Convert to JSON Schema.
    // Uses zod-to-json-schema for Zod schemas; plain objects are used as-is.
    let jsonSchema: Record<string, unknown>;
    if (isZodSchema) {
      jsonSchema = zodToJsonSchema(schema as ZodType<unknown>) as Record<string, unknown>;
      // Strip $schema meta field — not needed in inference request payloads
      delete jsonSchema["$schema"];
    } else {
      jsonSchema = schema as Record<string, unknown>;
    }

    const method = config?.method ?? "functionCalling";

    if (method === "functionCalling") {
      // Build OpenAI tool
      const tool: OpenAITool = {
        type: "function",
        function: {
          name: schemaName,
          description: schemaName,
          parameters: jsonSchema,
        },
      };

      const modelWithTool = this.bindTools([tool as unknown as Record<string, unknown>], {
        toolChoice: { type: "function", function: { name: schemaName } },
      });

      if (config?.includeRaw) {
        const rawParser = new RunnableLambda({
          func: (message: import("@langchain/core/messages").AIMessageChunk): {
            raw: import("@langchain/core/messages").AIMessageChunk;
            parsed: T | null;
            parsingError: Error | null;
          } => {
            try {
              const aiMsg = message as import("@langchain/core/messages").AIMessage;
              const toolCalls = aiMsg.tool_calls;
              let parsed: T;
              if (toolCalls && toolCalls.length > 0) {
                const args = toolCalls[0]!.args;
                if (typeof args === "string") {
                  parsed = JSON.parse(args) as T;
                } else {
                  parsed = args as unknown as T;
                }
              } else {
                const content = typeof aiMsg.content === "string" ? aiMsg.content : "";
                parsed = JSON.parse(content) as T;
              }
              return { raw: message, parsed, parsingError: null };
            } catch (e) {
              return {
                raw: message,
                parsed: null,
                parsingError: e instanceof Error ? e : new Error(String(e)),
              };
            }
          },
        });

        return modelWithTool.pipe(rawParser) as unknown as Runnable<BaseLanguageModelInput, T>;
      }

      // Output parser: extract tool call args from AIMessage
      const outputParser = new RunnableLambda({
        func: (message: import("@langchain/core/messages").AIMessageChunk): T => {
          const aiMsg = message as import("@langchain/core/messages").AIMessage;
          const toolCalls = aiMsg.tool_calls;
          if (toolCalls && toolCalls.length > 0) {
            const args = toolCalls[0]!.args;
            if (typeof args === "string") {
              return JSON.parse(args) as T;
            }
            return args as unknown as T;
          }
          // Fallback: try to parse content as JSON
          const content = typeof aiMsg.content === "string" ? aiMsg.content : "";
          return JSON.parse(content) as T;
        },
      });

      return modelWithTool.pipe(outputParser) as unknown as Runnable<BaseLanguageModelInput, T>;
    }

    if (method === "jsonSchema") {
      const responseFormat: OpenAIResponseFormat = {
        type: "json_schema",
        json_schema: {
          name: schemaName,
          strict: config?.strict ?? true,
          schema: jsonSchema,
        },
      };

      const capturedThis = this;
      const modelWithFormat = new RunnableLambda({
        func: (input: BaseLanguageModelInput) =>
          capturedThis.invoke(input, { responseFormat } as Partial<ChatKServeCallOptions>),
      });

      if (config?.includeRaw) {
        const rawParser = new RunnableLambda({
          func: (message: import("@langchain/core/messages").AIMessageChunk): {
            raw: import("@langchain/core/messages").AIMessageChunk;
            parsed: T | null;
            parsingError: Error | null;
          } => {
            try {
              const content = typeof message.content === "string" ? message.content : JSON.stringify(message.content);
              const parsed = JSON.parse(content) as T;
              return { raw: message, parsed, parsingError: null };
            } catch (e) {
              return {
                raw: message,
                parsed: null,
                parsingError: e instanceof Error ? e : new Error(String(e)),
              };
            }
          },
        });

        return modelWithFormat.pipe(rawParser) as unknown as Runnable<BaseLanguageModelInput, T>;
      }

      const jsonParser = new JsonOutputParser<Record<string, unknown>>();
      return modelWithFormat.pipe(jsonParser) as unknown as Runnable<BaseLanguageModelInput, T>;
    }

    // jsonMode
    const responseFormat: OpenAIResponseFormat = { type: "json_object" };
    const capturedThis = this;
    const modelWithFormat = new RunnableLambda({
      func: (input: BaseLanguageModelInput) =>
        capturedThis.invoke(input, { responseFormat } as Partial<ChatKServeCallOptions>),
    });

    if (config?.includeRaw) {
      const rawParser = new RunnableLambda({
        func: (message: import("@langchain/core/messages").AIMessageChunk): {
          raw: import("@langchain/core/messages").AIMessageChunk;
          parsed: T | null;
          parsingError: Error | null;
        } => {
          try {
            const content = typeof message.content === "string" ? message.content : JSON.stringify(message.content);
            const parsed = JSON.parse(content) as T;
            return { raw: message, parsed, parsingError: null };
          } catch (e) {
            return {
              raw: message,
              parsed: null,
              parsingError: e instanceof Error ? e : new Error(String(e)),
            };
          }
        },
      });

      return modelWithFormat.pipe(rawParser) as unknown as Runnable<BaseLanguageModelInput, T>;
    }

    const jsonParser = new JsonOutputParser<Record<string, unknown>>();
    return modelWithFormat.pipe(jsonParser) as unknown as Runnable<BaseLanguageModelInput, T>;
  }

  // --------------------------------------------------------
  // Model introspection
  // --------------------------------------------------------

  /**
   * Retrieve metadata about the model from the KServe endpoint.
   *
   * Tries the OpenAI-compatible /v1/models/{model} endpoint first,
   * then falls back to the V2 /v2/models/{model} endpoint.
   */
  async getModelInfo(): Promise<KServeModelInfo> {
    return this.client.getModelInfo(this.modelName);
  }
}
