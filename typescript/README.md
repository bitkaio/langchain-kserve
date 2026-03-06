# @bitkaio/langchain-kserve

LangChain.js integration for [KServe](https://kserve.github.io/website/) inference services.

Connect LangChain chains and agents to any model hosted on KServe — whether it's a vLLM-served Qwen2.5-Coder behind an Istio ingress, a Triton-served custom model, or a TGI-served Llama — with full streaming, tool calling, vision, and production-grade connection handling.

## Features

- **Two model classes**: `ChatKServe` for chat/instruct models, `KServeLLM` for base completion models
- **Dual protocol support**: OpenAI-compatible API (`/v1/chat/completions`) and V2 Inference Protocol (`/v2/models/{model}/infer`), with auto-detection
- **Full streaming**: SSE for OpenAI-compat, NDJSON for V2
- **Tool calling**: Full OpenAI function-calling format with `tool_choice`, `parallel_tool_calls`, and proper `invalid_tool_calls` handling for malformed responses
- **Vision / multimodal**: Send images alongside text via OpenAI content blocks
- **Token usage tracking**: `llmOutput` and `generationInfo` populated from vLLM responses, including streaming
- **Logprobs**: Optional per-token log-probabilities in `generationInfo`
- **Finish reason**: Always propagated in `generationInfo` and `response_metadata`
- **Model introspection**: `getModelInfo()` returns unified `KServeModelInfo` for both protocols
- **Production-ready**: TLS/CA bundles, bearer token auth, async token providers (K8s service account tokens), exponential backoff with jitter, 120s default timeout for cold starts

## Installation

```bash
npm install @bitkaio/langchain-kserve @langchain/core
# or
pnpm add @bitkaio/langchain-kserve @langchain/core
```

**Requirements**: Node.js 22.14+

## Quick Start

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";

const llm = new ChatKServe({
  baseUrl: "https://qwen-coder.my-cluster.example.com",
  modelName: "qwen2.5-coder-32b-instruct",
  temperature: 0.2,
});

const response = await llm.invoke("Write a TypeScript binary search function.");
console.log(response.content);
```

## Usage

### Basic invocation

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const llm = new ChatKServe({
  baseUrl: "https://qwen-coder.my-cluster.example.com",
  modelName: "qwen2.5-coder-32b-instruct",
});

// Single string (shorthand)
const result = await llm.invoke("Explain KServe in one sentence.");

// With messages array
const result2 = await llm.invoke([
  new SystemMessage("You are an expert TypeScript developer."),
  new HumanMessage("What is a monad?"),
]);
```

### Streaming

```typescript
const stream = await llm.stream("Implement a red-black tree in TypeScript.");

for await (const chunk of stream) {
  process.stdout.write(chunk.content as string);
}
```

### In a chain

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

const llm = new ChatKServe({
  baseUrl: "https://qwen-coder.my-cluster.example.com",
  modelName: "qwen2.5-coder-32b-instruct",
  temperature: 0.2,
  streaming: true,
});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are an expert {language} developer."],
  ["human", "{input}"],
]);

const chain = prompt.pipe(llm).pipe(new StringOutputParser());

const response = await chain.invoke({
  language: "TypeScript",
  input: "Implement a LRU cache",
});
```

### Token usage tracking

Token usage is available in both `llmOutput` (request-level) and `generationInfo` (per-generation):

```typescript
const result = await llm._generate([new HumanMessage("Hello")], {});

// Request-level
console.log(result.llmOutput);
// { tokenUsage: { promptTokens: 5, completionTokens: 12, totalTokens: 17 } }

// Per-generation
const info = result.generations[0].generationInfo;
console.log(info?.tokenUsage);    // { promptTokens: 5, completionTokens: 12, totalTokens: 17 }
console.log(info?.finishReason);  // "stop"
```

### Logprobs

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";

const llm = new ChatKServe({
  baseUrl: "https://qwen-coder.my-cluster.example.com",
  modelName: "qwen2.5-coder-32b-instruct",
  protocol: "openai",
  logprobs: true,
  topLogprobs: 5,
});

const result = await llm.invoke("Hello");
const info = result.response_metadata;
console.log(info.logprobs); // { content: [{ token: "Hi", logprob: -0.3, top_logprobs: [...] }] }
```

### Tool calling

Tool calling requires the OpenAI-compatible protocol (vLLM, TGI, etc.).

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const llm = new ChatKServe({
  baseUrl: "https://qwen-coder.my-cluster.example.com",
  modelName: "qwen2.5-coder-32b-instruct",
  protocol: "openai",
  parallelToolCalls: true,
});

const searchTool = tool(
  async ({ query }) => `Results for: ${query}`,
  {
    name: "search_codebase",
    description: "Search the codebase for relevant code",
    schema: z.object({ query: z.string() }),
  }
);

const llmWithTools = llm.bindTools([searchTool], { toolChoice: "auto" });

const result = await llmWithTools.invoke(
  "Find all usages of the AuthService class"
);

if (result.tool_calls && result.tool_calls.length > 0) {
  console.log("Tool call:", result.tool_calls[0]);
}

// Malformed arguments from the model are captured in invalid_tool_calls
if (result.invalid_tool_calls && result.invalid_tool_calls.length > 0) {
  console.log("Invalid call:", result.invalid_tool_calls[0]);
}
```

### Vision / multimodal

Send images alongside text. Works with OpenAI-compatible runtimes that support vision (e.g., vLLM with a multimodal model).

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";
import { HumanMessage } from "@langchain/core/messages";
import { readFile } from "node:fs/promises";

const llm = new ChatKServe({
  baseUrl: "https://llava.my-cluster.example.com",
  modelName: "llava-1.6",
  protocol: "openai",
});

// Base64-encoded image (preferred for cluster-internal use)
const imageData = await readFile("chart.png", { encoding: "base64" });
const message = new HumanMessage({
  content: [
    { type: "text", text: "Describe what you see in this chart:" },
    {
      type: "image_url",
      image_url: {
        url: `data:image/png;base64,${imageData}`,
        detail: "high",
      },
    },
  ],
});

const response = await llm.invoke([message]);
console.log(response.content);
```

URL-based images are also supported (the model pod fetches the image):

```typescript
const message = new HumanMessage({
  content: [
    { type: "text", text: "What's in this image?" },
    { type: "image_url", image_url: { url: "https://example.com/image.jpg" } },
  ],
});
```

### Model introspection

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";

const llm = new ChatKServe({
  baseUrl: "https://qwen-coder.my-cluster.example.com",
  modelName: "qwen2.5-coder-32b-instruct",
});

const info = await llm.getModelInfo();

console.log(info.modelName);    // "qwen2.5-coder-32b-instruct"
console.log(info.platform);     // "openai-compat" or V2 platform string
console.log(info.raw);          // full response from the endpoint
```

### V2 Inference Protocol

Use `protocol: "v2"` for runtimes that expose the native KServe V2/Open Inference Protocol (e.g., Triton Inference Server):

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";

const llm = new ChatKServe({
  baseUrl: "https://triton.my-cluster.example.com",
  modelName: "llama-3-8b-instruct",
  protocol: "v2",
  chatTemplate: "llama",
});

const result = await llm.invoke("Explain transformers in simple terms.");
```

> **Note:** Tool calling and vision are not supported on V2. Attempting either throws a `KServeInferenceError` immediately (before any HTTP call) with a clear message.

### Base (non-chat) models with KServeLLM

```typescript
import { KServeLLM } from "@bitkaio/langchain-kserve";

const llm = new KServeLLM({
  baseUrl: "https://base-model.my-cluster.example.com",
  modelName: "llama-3-base",
  protocol: "v2",
  temperature: 0.8,
  maxTokens: 256,
});

const completion = await llm.invoke("Once upon a time");

// Token usage (from streaming or non-streaming)
const result = await llm.generate(["Once upon a time"]);
console.log(result.llmOutput); // { tokenUsage: {...} } if vLLM, else undefined
```

### Environment variable configuration

All constructor options can be set via environment variables:

| Environment Variable | Constructor Field |
|----------------------|-------------------|
| `KSERVE_BASE_URL`    | `baseUrl`         |
| `KSERVE_MODEL_NAME`  | `modelName`       |
| `KSERVE_API_KEY`     | `apiKey`          |
| `KSERVE_PROTOCOL`    | `protocol`        |
| `KSERVE_CA_BUNDLE`   | `caBundle`        |

```bash
export KSERVE_BASE_URL=https://qwen-coder.my-cluster.example.com
export KSERVE_MODEL_NAME=qwen2.5-coder-32b-instruct
export KSERVE_API_KEY=my-bearer-token
```

### Authentication

**Static API key / bearer token:**

```typescript
const llm = new ChatKServe({
  baseUrl: "https://my-model.cluster.example.com",
  modelName: "my-model",
  apiKey: "my-bearer-token",
});
```

**Dynamic token provider** (e.g., Kubernetes service account tokens):

```typescript
import { readFile } from "node:fs/promises";

const llm = new ChatKServe({
  baseUrl: "https://my-model.cluster.example.com",
  modelName: "my-model",
  tokenProvider: async () => {
    const token = await readFile(
      "/var/run/secrets/kubernetes.io/serviceaccount/token",
      "utf-8"
    );
    return token.trim();
  },
});
```

### Custom TLS / CA bundle

```typescript
const llm = new ChatKServe({
  baseUrl: "https://my-model.internal.example.com",
  modelName: "my-model",
  caBundle: "/etc/ssl/certs/my-internal-ca.pem",
});
```

### Connection tuning

```typescript
const llm = new ChatKServe({
  baseUrl: "https://my-model.cluster.example.com",
  modelName: "my-model",
  timeout: 300_000,  // 5 minutes for cold starts
  maxRetries: 5,
});
```

### DeepAgents integration

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const readFile = tool(
  async ({ path }) => { /* ... */ },
  {
    name: "read_file",
    description: "Read a file from the codebase",
    schema: z.object({ path: z.string() }),
  }
);

const llm = new ChatKServe({
  baseUrl: process.env.KSERVE_BASE_URL!,
  modelName: "qwen2.5-coder-32b-instruct",
  temperature: 0.2,
  streaming: true,
});

import { Agent } from "deepagents";

const agent = new Agent({ llm, tools: [readFile] });

const result = await agent.invoke({
  input: "Refactor the authentication module to use JWT tokens",
});
```

## API Reference

### `ChatKServe`

Extends `BaseChatModel`. Use for chat/instruct models.

#### Constructor options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `baseUrl` | `string` | `KSERVE_BASE_URL` env | KServe inference service URL |
| `modelName` | `string` | `KSERVE_MODEL_NAME` env | Model name as registered in KServe |
| `protocol` | `"openai" \| "v2" \| "auto"` | `"auto"` | Inference protocol |
| `apiKey` | `string` | `KSERVE_API_KEY` env | Static bearer token |
| `tokenProvider` | `() => Promise<string>` | — | Dynamic token provider |
| `verifySsl` | `boolean` | `true` | Verify SSL certificates |
| `caBundle` | `string` | `KSERVE_CA_BUNDLE` env | Path to CA certificate bundle |
| `temperature` | `number` | — | Sampling temperature |
| `maxTokens` | `number` | — | Max tokens to generate |
| `topP` | `number` | — | Top-p nucleus sampling |
| `stop` | `string[]` | — | Stop sequences |
| `streaming` | `boolean` | `false` | Enable streaming |
| `logprobs` | `boolean` | — | Return per-token log-probabilities (OpenAI-compat only) |
| `topLogprobs` | `number` | — | Number of top logprobs per token (OpenAI-compat only) |
| `parallelToolCalls` | `boolean` | — | Allow multiple tool calls in one turn (OpenAI-compat only) |
| `timeout` | `number` | `120000` | Request timeout (ms) |
| `maxRetries` | `number` | `3` | Max retry attempts |
| `chatTemplate` | `"chatml" \| "llama" \| "custom"` | `"chatml"` | Template for V2 protocol |
| `customChatTemplate` | `string` | — | Custom template string |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `invoke(input)` | `Promise<AIMessage>` | Single-turn generation |
| `stream(input)` | `AsyncIterable<AIMessageChunk>` | Streaming generation |
| `bindTools(tools, kwargs?)` | `Runnable` | Bind tools for function calling |
| `getModelInfo()` | `Promise<KServeModelInfo>` | Fetch model metadata from endpoint |

### `KServeLLM`

Extends `BaseLLM`. Use for base/completion models. Same options as `ChatKServe`, minus `chatTemplate` and `customChatTemplate`.

### `KServeModelInfo`

```typescript
interface KServeModelInfo {
  modelName: string;
  modelVersion?: string;
  platform?: string;
  inputs?: Array<Record<string, unknown>>;
  outputs?: Array<Record<string, unknown>>;
  raw: Record<string, unknown>;
}
```

### Error classes

```typescript
import {
  KServeError,                // base class
  KServeConnectionError,      // network/DNS failures
  KServeAuthenticationError,  // 401/403
  KServeModelNotFoundError,   // 404, model not loaded
  KServeInferenceError,       // 4xx/5xx during inference, or V2 unsupported feature
  KServeTimeoutError,         // timeout exceeded
} from "@bitkaio/langchain-kserve";
```

## Protocol Capability Matrix

| Feature | OpenAI-compat | V2 |
|---|:---:|:---:|
| Text generation | ✅ | ✅ |
| Streaming | ✅ | ✅ |
| Tool calling | ✅ | ❌ |
| Vision / multimodal | ✅ | ❌ |
| Logprobs | ✅ | ❌ |
| Token usage tracking | ✅ | ❌ |
| Finish reason | ✅ | partial |

Attempting tool calling or vision with V2 throws `KServeInferenceError` immediately (before any HTTP call) with a clear message.

## Protocol auto-detection

When `protocol: "auto"` (the default), the client probes `GET /v1/models`:

- **200 response** → OpenAI-compatible protocol (`/v1/chat/completions`)
- **Non-200 / connection error** → V2 Inference Protocol (`/v2/models/{model}/infer`)

The detected protocol is cached for the lifetime of the model instance. Pin a protocol explicitly to skip detection:

```typescript
const llm = new ChatKServe({
  baseUrl: "...",
  modelName: "...",
  protocol: "openai", // or "v2"
});
```

## Development

```bash
# Install dependencies
pnpm install

# Build
pnpm build

# Run unit tests
pnpm test

# Run with watch mode
pnpm test:watch

# Type check
pnpm typecheck

# Integration tests (requires live KServe endpoint)
KSERVE_BASE_URL=https://... KSERVE_MODEL_NAME=... pnpm test:integration
```

## License

MIT
