/**
 * HTTP client for KServe inference services.
 *
 * Features:
 * - Bearer token auth (static key or dynamic provider)
 * - TLS with custom CA bundle (Node.js https.Agent)
 * - Exponential backoff with jitter on transient failures
 * - Lightweight internal SSE parser for streaming
 * - Protocol auto-detection (OpenAI-compat vs V2)
 */

import * as https from "node:https";
import * as fs from "node:fs";

import {
  KServeConnectionError,
  KServeTimeoutError,
  mapHttpErrorToKServeError,
} from "./errors.js";
import type { KServeClientConfig, KServeModelInfo, RequestOptions } from "./types.js";

// ============================================================
// SSE parser
// ============================================================

/**
 * Parse a Server-Sent Events stream.
 *
 * Yields data payloads from `data: ...` lines.
 * Stops when it encounters `data: [DONE]`.
 *
 * @param response - Fetch Response object (ReadableStream body)
 */
export async function* parseSSE(response: Response): AsyncGenerator<string> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new KServeConnectionError("Response body is not readable");
  }

  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // Keep the last (potentially incomplete) line in the buffer
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ")) {
          const data = trimmed.slice("data: ".length).trim();
          if (data === "[DONE]") return;
          if (data) yield data;
        }
        // Ignore comment lines (": ...") and event/id fields
      }
    }

    // Flush any remaining buffer content
    if (buffer.trim().startsWith("data: ")) {
      const data = buffer.trim().slice("data: ".length).trim();
      if (data && data !== "[DONE]") yield data;
    }
  } finally {
    reader.releaseLock();
  }
}

// ============================================================
// Retry helpers
// ============================================================

/** Determines whether an error is transient and warrants a retry */
function isTransientError(error: unknown): boolean {
  if (error instanceof KServeConnectionError) return true;
  if (error instanceof KServeTimeoutError) return true;
  if (error instanceof Error) {
    const msg = error.message.toLowerCase();
    return (
      msg.includes("econnreset") ||
      msg.includes("econnrefused") ||
      msg.includes("etimedout") ||
      msg.includes("network") ||
      msg.includes("fetch failed")
    );
  }
  return false;
}

/**
 * Calculate exponential backoff delay with jitter.
 *
 * @param attempt - Zero-based attempt index
 * @param baseMs  - Base delay in milliseconds (default 500)
 * @param maxMs   - Maximum delay cap in milliseconds (default 30000)
 */
function backoffDelay(
  attempt: number,
  baseMs = 500,
  maxMs = 30_000
): number {
  const exponential = baseMs * 2 ** attempt;
  const jitter = Math.random() * exponential * 0.2; // ±20% jitter
  return Math.min(exponential + jitter, maxMs);
}

// ============================================================
// KServeClient
// ============================================================

/**
 * Low-level HTTP client for KServe endpoints.
 *
 * Handles authentication, TLS, retries, and streaming.
 */
export class KServeClient {
  private readonly config: Required<
    Pick<KServeClientConfig, "baseUrl" | "timeout" | "maxRetries" | "verifySsl">
  > &
    Omit<KServeClientConfig, "baseUrl" | "timeout" | "maxRetries" | "verifySsl">;

  private detectedProtocol: "openai" | "v2" | null = null;
  private httpsAgent: https.Agent | null = null;

  constructor(config: KServeClientConfig) {
    this.config = {
      baseUrl: config.baseUrl.replace(/\/$/, ""), // strip trailing slash
      apiKey: config.apiKey,
      tokenProvider: config.tokenProvider,
      verifySsl: config.verifySsl ?? true,
      caBundle: config.caBundle,
      timeout: config.timeout ?? 120_000,
      maxRetries: config.maxRetries ?? 3,
    };

    // Build an https.Agent if we need custom TLS settings
    if (!this.config.verifySsl || this.config.caBundle) {
      const agentOptions: https.AgentOptions = {
        rejectUnauthorized: this.config.verifySsl,
      };
      if (this.config.caBundle) {
        agentOptions.ca = fs.readFileSync(this.config.caBundle, "utf8");
      }
      this.httpsAgent = new https.Agent(agentOptions);
    }
  }

  // --------------------------------------------------------
  // Auth helpers
  // --------------------------------------------------------

  private async buildAuthHeaders(): Promise<Record<string, string>> {
    if (this.config.tokenProvider) {
      const token = await this.config.tokenProvider();
      return { Authorization: `Bearer ${token}` };
    }
    if (this.config.apiKey) {
      return { Authorization: `Bearer ${this.config.apiKey}` };
    }
    return {};
  }

  // --------------------------------------------------------
  // Core fetch with timeout
  // --------------------------------------------------------

  private async fetchWithTimeout(
    url: string,
    init: RequestInit,
    signal?: AbortSignal
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(new Error("Request timed out")),
      this.config.timeout
    );

    // Chain external signal with our timeout signal
    const onAbort = () => controller.abort();
    signal?.addEventListener("abort", onAbort);

    try {
      const fetchInit: RequestInit = {
        ...init,
        signal: controller.signal,
      };

      // Attach custom https agent for Node.js (undici-compatible)
      if (this.httpsAgent) {
        // @ts-expect-error — Node.js fetch accepts `dispatcher` / agent options
        fetchInit.dispatcher = this.httpsAgent;
      }

      const response = await fetch(url, fetchInit);
      return response;
    } catch (err) {
      if (
        err instanceof Error &&
        (err.name === "AbortError" || err.message === "Request timed out")
      ) {
        if (signal?.aborted) {
          throw err; // propagate external cancellation as-is
        }
        throw new KServeTimeoutError(
          `Request to ${url} timed out after ${this.config.timeout}ms`
        );
      }
      throw new KServeConnectionError(
        `Failed to connect to ${url}: ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      clearTimeout(timeoutId);
      signal?.removeEventListener("abort", onAbort);
    }
  }

  // --------------------------------------------------------
  // Non-streaming request with retries
  // --------------------------------------------------------

  /**
   * Make a JSON POST request, returning the parsed response body.
   *
   * @param path    - URL path (appended to baseUrl)
   * @param body    - Request body (serialized to JSON)
   * @param options - Optional per-request options
   */
  async request<T>(
    path: string,
    body: unknown,
    options?: RequestOptions
  ): Promise<T> {
    const url = `${this.config.baseUrl}${path}`;
    let lastError: unknown;

    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      if (attempt > 0) {
        await new Promise((resolve) =>
          setTimeout(resolve, backoffDelay(attempt - 1))
        );
      }

      try {
        const authHeaders = await this.buildAuthHeaders();
        const response = await this.fetchWithTimeout(
          url,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "application/json",
              ...authHeaders,
              ...options?.headers,
            },
            body: JSON.stringify(body),
          },
          options?.signal
        );

        if (!response.ok) {
          const text = await response.text().catch(() => "");
          throw mapHttpErrorToKServeError(
            response.status,
            `HTTP ${response.status} from ${url}: ${text}`
          );
        }

        return (await response.json()) as T;
      } catch (err) {
        lastError = err;
        if (!isTransientError(err) || attempt === this.config.maxRetries) {
          throw err;
        }
      }
    }

    throw lastError;
  }

  // --------------------------------------------------------
  // Streaming request
  // --------------------------------------------------------

  /**
   * Make a streaming POST request, yielding raw SSE data payloads.
   *
   * @param path    - URL path (appended to baseUrl)
   * @param body    - Request body (serialized to JSON)
   * @param options - Optional per-request options
   */
  async *streamRequest(
    path: string,
    body: unknown,
    options?: RequestOptions
  ): AsyncGenerator<string> {
    const url = `${this.config.baseUrl}${path}`;
    const authHeaders = await this.buildAuthHeaders();

    const response = await this.fetchWithTimeout(
      url,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          ...authHeaders,
          ...options?.headers,
        },
        body: JSON.stringify(body),
      },
      options?.signal
    );

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw mapHttpErrorToKServeError(
        response.status,
        `HTTP ${response.status} from ${url}: ${text}`
      );
    }

    yield* parseSSE(response);
  }

  /**
   * Make a streaming POST request yielding NDJSON lines (for V2 protocol).
   *
   * @param path    - URL path (appended to baseUrl)
   * @param body    - Request body
   * @param options - Optional per-request options
   */
  async *streamNDJSON(
    path: string,
    body: unknown,
    options?: RequestOptions
  ): AsyncGenerator<string> {
    const url = `${this.config.baseUrl}${path}`;
    const authHeaders = await this.buildAuthHeaders();

    const response = await this.fetchWithTimeout(
      url,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/x-ndjson",
          ...authHeaders,
          ...options?.headers,
        },
        body: JSON.stringify(body),
      },
      options?.signal
    );

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw mapHttpErrorToKServeError(
        response.status,
        `HTTP ${response.status} from ${url}: ${text}`
      );
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new KServeConnectionError("Response body is not readable");
    }

    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed) yield trimmed;
        }
      }

      if (buffer.trim()) yield buffer.trim();
    } finally {
      reader.releaseLock();
    }
  }

  // --------------------------------------------------------
  // GET request helper (for protocol detection)
  // --------------------------------------------------------

  /**
   * Make a GET request and return the HTTP status code.
   * Does not throw on non-200 responses — just returns the code.
   */
  async getStatus(path: string, signal?: AbortSignal): Promise<number> {
    const url = `${this.config.baseUrl}${path}`;
    const authHeaders = await this.buildAuthHeaders();

    try {
      const response = await this.fetchWithTimeout(
        url,
        {
          method: "GET",
          headers: {
            Accept: "application/json",
            ...authHeaders,
          },
        },
        signal
      );
      return response.status;
    } catch {
      return 0; // connection failure
    }
  }

  // --------------------------------------------------------
  // Protocol detection
  // --------------------------------------------------------

  /**
   * Auto-detect whether the KServe endpoint exposes an OpenAI-compatible
   * API or only the V2 inference protocol.
   *
   * Detection strategy: GET /v1/models — if 200, use OpenAI. Otherwise, use V2.
   * Result is cached for the lifetime of this client instance.
   */
  async detectProtocol(): Promise<"openai" | "v2"> {
    if (this.detectedProtocol) return this.detectedProtocol;

    const status = await this.getStatus("/v1/models");
    this.detectedProtocol = status === 200 ? "openai" : "v2";
    return this.detectedProtocol;
  }

  /** Override the cached detected protocol */
  setProtocol(protocol: "openai" | "v2"): void {
    this.detectedProtocol = protocol;
  }

  // --------------------------------------------------------
  // GET request helper (for model info)
  // --------------------------------------------------------

  /**
   * Make a GET request and return the parsed JSON body.
   * Throws on non-2xx responses.
   */
  async get<T>(path: string, signal?: AbortSignal): Promise<T> {
    const url = `${this.config.baseUrl}${path}`;
    const authHeaders = await this.buildAuthHeaders();

    const response = await this.fetchWithTimeout(
      url,
      {
        method: "GET",
        headers: {
          Accept: "application/json",
          ...authHeaders,
        },
      },
      signal
    );

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw mapHttpErrorToKServeError(
        response.status,
        `HTTP ${response.status} from ${url}: ${text}`
      );
    }

    return (await response.json()) as T;
  }

  // --------------------------------------------------------
  // Model introspection
  // --------------------------------------------------------

  /**
   * Retrieve metadata about a model from the KServe endpoint.
   *
   * Tries the OpenAI-compatible GET /v1/models/{modelName} endpoint first.
   * On any failure, falls back to the V2 GET /v2/models/{modelName} endpoint.
   *
   * @param modelName - The model name to look up
   */
  async getModelInfo(modelName: string): Promise<KServeModelInfo> {
    const encoded = encodeURIComponent(modelName);

    // Try OpenAI-compat first
    try {
      const response = await this.get<Record<string, unknown>>(
        `/v1/models/${encoded}`
      );
      return {
        modelName: (response.id as string | undefined) ?? modelName,
        modelVersion: undefined,
        platform: "openai-compat",
        raw: response,
      };
    } catch {
      // Fall through to V2
    }

    // Try V2
    const response = await this.get<Record<string, unknown>>(
      `/v2/models/${encoded}`
    );
    return {
      modelName: (response.name as string | undefined) ?? modelName,
      modelVersion: (response.versions as string[] | undefined)?.[0],
      platform: response.platform as string | undefined,
      inputs: response.inputs as Array<Record<string, unknown>> | undefined,
      outputs: response.outputs as Array<Record<string, unknown>> | undefined,
      raw: response,
    };
  }
}
