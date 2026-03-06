/**
 * KServeLLM — LangChain LLM (text completion) for KServe inference services.
 *
 * Use this for base/completion models. For chat/instruct models, use ChatKServe.
 *
 * @example
 * ```typescript
 * import { KServeLLM } from "@langchain/kserve";
 *
 * const llm = new KServeLLM({
 *   baseUrl: "https://my-model.cluster.example.com",
 *   modelName: "my-base-model",
 *   protocol: "v2", // or "openai" for vLLM-served completion models
 * });
 *
 * const response = await llm.invoke("The capital of France is");
 * ```
 */

import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { BaseLLM } from "@langchain/core/language_models/llms";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  GenerationChunk,
  LLMResult,
} from "@langchain/core/outputs";

import { KServeClient } from "./client.js";
import {
  buildCompletionRequest,
  parseCompletionResponse,
  parseCompletionStreamChunk,
} from "./openai-compat.js";
import {
  buildV2StreamRequest,
  buildV2TextRequest,
  getV2InferPath,
  parseV2TextResponse,
  parseV2TextStreamChunk,
} from "./v2-protocol.js";
import type {
  KServeLLMCallOptions,
  KServeLLMInput,
  KServeModelInfo,
  V2InferResponse,
} from "./types.js";
import { KServeInferenceError } from "./errors.js";

/**
 * KServeLLM — LangChain BaseLLM implementation for KServe text completion.
 */
export class KServeLLM extends BaseLLM<KServeLLMCallOptions> {
  static lc_name(): string {
    return "KServeLLM";
  }

  // Connection
  private readonly baseUrl: string;
  private readonly modelName: string;
  private readonly protocolPref: "openai" | "v2" | "auto";
  private readonly client: KServeClient;

  // Generation params
  private readonly temperature?: number;
  private readonly maxTokens?: number;
  private readonly topP?: number;
  private readonly stop?: string[];
  private readonly streaming: boolean;

  constructor(fields: KServeLLMInput) {
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
    this.temperature = fields.temperature;
    this.maxTokens = fields.maxTokens;
    this.topP = fields.topP;
    this.stop = fields.stop;
    this.streaming = fields.streaming ?? false;

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

    if (this.protocolPref !== "auto") {
      this.client.setProtocol(this.protocolPref);
    }
  }

  /** LangChain model type identifier */
  _llmType(): string {
    return "kserve-llm";
  }

  // --------------------------------------------------------
  // Protocol resolution
  // --------------------------------------------------------

  private async resolveProtocol(): Promise<"openai" | "v2"> {
    if (this.protocolPref !== "auto") return this.protocolPref;
    return this.client.detectProtocol();
  }

  // --------------------------------------------------------
  // Core generation
  // --------------------------------------------------------

  /**
   * Generate completions for a batch of prompts.
   * Called by LangChain's `invoke()`, `generate()`, etc.
   */
  async _generate(
    prompts: string[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): Promise<LLMResult> {
    // For streaming, process prompts one at a time and collect chunks
    if (this.streaming) {
      const allGenerations: LLMResult["generations"] = [];
      let lastUsage: import("./types.js").OpenAIUsage | undefined = undefined;

      for (const prompt of prompts) {
        const chunks: GenerationChunk[] = [];
        for await (const chunk of this._streamResponseChunks(
          prompt,
          options,
          runManager
        )) {
          chunks.push(chunk);
        }
        const text = chunks.map((c) => c.text).join("");
        const lastGenInfo = chunks[chunks.length - 1]?.generationInfo as
          | (import("./types.js").KServeGenerationInfo & Record<string, unknown>)
          | undefined;
        if (lastGenInfo?.tokenUsage) {
          const tu = lastGenInfo.tokenUsage as {
            promptTokens: number;
            completionTokens: number;
            totalTokens: number;
          };
          lastUsage = {
            prompt_tokens: tu.promptTokens,
            completion_tokens: tu.completionTokens,
            total_tokens: tu.totalTokens,
          };
        }
        allGenerations.push([{ text, generationInfo: lastGenInfo }]);
      }

      return {
        generations: allGenerations,
        llmOutput: lastUsage
          ? {
              tokenUsage: {
                promptTokens: lastUsage.prompt_tokens,
                completionTokens: lastUsage.completion_tokens,
                totalTokens: lastUsage.total_tokens,
              },
            }
          : undefined,
      };
    }

    const protocol = await this.resolveProtocol();
    const allGenerations: LLMResult["generations"] = [];

    for (const prompt of prompts) {
      if (protocol === "openai") {
        const result = await this._generateOpenAI(prompt, options);
        allGenerations.push(...result.generations);
      } else {
        const result = await this._generateV2(prompt, options);
        allGenerations.push(...result.generations);
      }
    }

    return { generations: allGenerations };
  }

  private async _generateOpenAI(
    prompt: string,
    options: this["ParsedCallOptions"]
  ): Promise<LLMResult> {
    const request = buildCompletionRequest(
      this.modelName,
      prompt,
      {
        temperature: options.temperature ?? this.temperature,
        maxTokens: options.maxTokens ?? this.maxTokens,
        topP: this.topP,
        stop: options.stop ?? this.stop,
      },
      false
    );

    const response = await this.client.request<
      import("./types.js").OpenAICompletionResponse
    >("/v1/completions", request, { signal: options.signal });

    return parseCompletionResponse(response, this.modelName);
  }

  private async _generateV2(
    prompt: string,
    options: this["ParsedCallOptions"]
  ): Promise<LLMResult> {
    const request = buildV2TextRequest(prompt, {
      temperature: options.temperature ?? this.temperature,
      maxTokens: options.maxTokens ?? this.maxTokens,
      topP: this.topP,
      stop: options.stop ?? this.stop,
    });

    const response = await this.client.request<V2InferResponse>(
      getV2InferPath(this.modelName),
      request,
      { signal: options.signal }
    );

    return parseV2TextResponse(response, this.modelName);
  }

  // --------------------------------------------------------
  // Streaming
  // --------------------------------------------------------

  /**
   * Stream response chunks for a single prompt.
   */
  async *_streamResponseChunks(
    input: string,
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<GenerationChunk> {
    const protocol = await this.resolveProtocol();

    if (protocol === "openai") {
      yield* this._streamOpenAI(input, options, runManager);
    } else {
      yield* this._streamV2(input, options, runManager);
    }
  }

  private async *_streamOpenAI(
    prompt: string,
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<GenerationChunk> {
    const request = buildCompletionRequest(
      this.modelName,
      prompt,
      {
        temperature: options.temperature ?? this.temperature,
        maxTokens: options.maxTokens ?? this.maxTokens,
        topP: this.topP,
        stop: options.stop ?? this.stop,
      },
      true
    );

    for await (const raw of this.client.streamRequest(
      "/v1/completions",
      request,
      { signal: options.signal }
    )) {
      const parsed = parseCompletionStreamChunk(raw);
      if (parsed === null) continue;
      const { text, usage } = parsed;
      const genInfo: import("./types.js").KServeGenerationInfo & Record<string, unknown> = {
        protocol: "openai",
        modelName: this.modelName,
      };
      if (usage) {
        genInfo.tokenUsage = {
          promptTokens: usage.prompt_tokens,
          completionTokens: usage.completion_tokens,
          totalTokens: usage.total_tokens,
        };
      }
      if (text) {
        await runManager?.handleLLMNewToken(text);
      }
      yield new GenerationChunk({
        text,
        generationInfo: genInfo,
      });
    }
  }

  private async *_streamV2(
    prompt: string,
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<GenerationChunk> {
    const baseRequest = buildV2TextRequest(prompt, {
      temperature: options.temperature ?? this.temperature,
      maxTokens: options.maxTokens ?? this.maxTokens,
      topP: this.topP,
      stop: options.stop ?? this.stop,
    });

    const streamRequest = buildV2StreamRequest(baseRequest);

    let streamed = false;
    try {
      for await (const raw of this.client.streamNDJSON(
        getV2InferPath(this.modelName),
        streamRequest,
        { signal: options.signal }
      )) {
        const chunk = parseV2TextStreamChunk(raw, this.modelName);
        if (!chunk) continue;
        streamed = true;
        await runManager?.handleLLMNewToken(chunk.text ?? "");
        yield chunk;
      }
    } catch (err) {
      if (!streamed) {
        console.warn(
          "[KServeLLM] V2 streaming not available, falling back to non-streaming"
        );
        const result = await this._generateV2(prompt, options);
        const gen = result.generations[0]?.[0];
        if (!gen) throw new KServeInferenceError("No generation in V2 response");
        await runManager?.handleLLMNewToken(gen.text);
        yield gen as GenerationChunk;
        return;
      }
      throw err;
    }
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
