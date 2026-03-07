/**
 * KServeEmbeddings — LangChain Embeddings for KServe embedding models.
 */

import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { Embeddings } from "@langchain/core/embeddings";
import type { EmbeddingsParams } from "@langchain/core/embeddings";

import { KServeClient } from "./client.js";

/** Request body for /v1/embeddings */
interface EmbeddingsRequest {
  model: string;
  input: string[];
  encoding_format?: "float" | "base64";
  dimensions?: number;
}

/** Single embedding item in the response */
interface EmbeddingItem {
  object: "embedding";
  index: number;
  embedding: number[] | string; // string when encoding_format=base64
}

/** Response from /v1/embeddings */
interface EmbeddingsResponse {
  object: "list";
  data: EmbeddingItem[];
  model: string;
  usage?: { prompt_tokens: number; total_tokens: number };
}

/** Constructor options for KServeEmbeddings */
export interface KServeEmbeddingsParams extends EmbeddingsParams {
  /** Base URL of the KServe embeddings service */
  baseUrl: string;

  /** Embedding model name as registered in KServe */
  modelName: string;

  /** Static bearer token */
  apiKey?: string;

  /** Dynamic token provider */
  tokenProvider?: () => Promise<string>;

  /** Whether to verify SSL certificates. Defaults to true. */
  verifySsl?: boolean;

  /** Path to custom CA bundle */
  caBundle?: string;

  /** Output embedding dimensions (for Matryoshka-capable models) */
  dimensions?: number;

  /** Encoding format. "float" returns float arrays; "base64" decodes to float arrays. */
  encodingFormat?: "float" | "base64";

  /** Request timeout in ms. Defaults to 120000. */
  timeout?: number;

  /** Max retry attempts. Defaults to 3. */
  maxRetries?: number;

  /** Max texts per API call. Defaults to 1000. */
  chunkSize?: number;
}

/**
 * KServeEmbeddings — LangChain Embeddings for KServe-hosted embedding models.
 *
 * Uses the OpenAI-compatible /v1/embeddings endpoint.
 *
 * @example
 * ```typescript
 * const embeddings = new KServeEmbeddings({
 *   baseUrl: "https://my-embedding-model.cluster.example.com",
 *   modelName: "Qwen/Qwen3-Embedding-0.6B",
 * });
 * const vectors = await embeddings.embedDocuments(["Hello world", "How are you?"]);
 * ```
 */
export class KServeEmbeddings extends Embeddings {
  private readonly modelName: string;
  private readonly client: KServeClient;
  private readonly dimensions?: number;
  private readonly encodingFormat: "float" | "base64";
  private readonly chunkSize: number;

  constructor(params: KServeEmbeddingsParams) {
    super(params);

    const baseUrl =
      params.baseUrl ??
      getEnvironmentVariable("KSERVE_EMBEDDINGS_BASE_URL") ??
      getEnvironmentVariable("KSERVE_BASE_URL") ??
      (() => {
        throw new Error(
          "baseUrl is required. Set it in the constructor or via KSERVE_EMBEDDINGS_BASE_URL env var."
        );
      })();

    this.modelName =
      params.modelName ??
      getEnvironmentVariable("KSERVE_EMBEDDINGS_MODEL_NAME") ??
      (() => {
        throw new Error(
          "modelName is required. Set it in the constructor or via KSERVE_EMBEDDINGS_MODEL_NAME env var."
        );
      })();

    this.client = new KServeClient({
      baseUrl,
      apiKey: params.apiKey ?? getEnvironmentVariable("KSERVE_API_KEY"),
      tokenProvider: params.tokenProvider,
      verifySsl: params.verifySsl,
      caBundle: params.caBundle ?? getEnvironmentVariable("KSERVE_CA_BUNDLE"),
      timeout: params.timeout,
      maxRetries: params.maxRetries,
    });

    this.dimensions = params.dimensions;
    this.encodingFormat = params.encodingFormat ?? "float";
    this.chunkSize = params.chunkSize ?? 1000;
  }

  private buildRequest(texts: string[]): EmbeddingsRequest {
    const req: EmbeddingsRequest = {
      model: this.modelName,
      input: texts,
      encoding_format: this.encodingFormat,
    };
    if (this.dimensions !== undefined) {
      req.dimensions = this.dimensions;
    }
    return req;
  }

  private decodeEmbeddings(data: EmbeddingItem[]): number[][] {
    const sorted = [...data].sort((a, b) => a.index - b.index);
    return sorted.map((item) => {
      if (this.encodingFormat === "base64" && typeof item.embedding === "string") {
        // Decode base64 float32 array
        const raw = Buffer.from(item.embedding, "base64");
        const floats: number[] = [];
        for (let i = 0; i < raw.length; i += 4) {
          floats.push(raw.readFloatLE(i));
        }
        return floats;
      }
      return item.embedding as number[];
    });
  }

  /** Embed a list of documents, batching into chunks of chunkSize. */
  async embedDocuments(documents: string[]): Promise<number[][]> {
    const results: number[][] = [];
    for (let i = 0; i < documents.length; i += this.chunkSize) {
      const batch = documents.slice(i, i + this.chunkSize);
      const response = await this.client.request<EmbeddingsResponse>(
        "/v1/embeddings",
        this.buildRequest(batch)
      );
      results.push(...this.decodeEmbeddings(response.data));
    }
    return results;
  }

  /** Embed a single query string. */
  async embedQuery(document: string): Promise<number[]> {
    const results = await this.embedDocuments([document]);
    return results[0];
  }
}
