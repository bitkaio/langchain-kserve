# @bitkaio/langchain-kserve

LangChain.js integration for [KServe](https://kserve.github.io/website/) inference services.

Connect LangChain chains and agents to any model hosted on KServe — whether it's a vLLM-served Qwen2.5-Coder behind an Istio ingress, a Triton-served custom model, or a TGI-served Llama — with full streaming, tool calling, and production-grade connection handling.

## Features

- **Two model classes**: `ChatKServe` for chat/instruct models, `KServeLLM` for base completion models
- **Dual protocol support**: OpenAI-compatible API (`/v1/chat/completions`) and V2 Inference Protocol (`/v2/models/{model}/infer`), with auto-detection
- **Full streaming**: SSE for OpenAI-compat, NDJSON for V2
- **Tool calling**: Passes LangChain tool definitions through OpenAI function-calling schema
- **Production-ready**: TLS/CA bundles, bearer token auth, async token providers (K8s service account tokens), exponential backoff with jitter, 120s default timeout for cold starts

## Installation

```bash
npm install @bitkaio/langchain-kserve @langchain/core
# or
pnpm add @bitkaio/langchain-kserve @langchain/core
```

**Requirements**: Node.js 18+ (uses native `fetch`)

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

### Tool calling

Tool calling requires the OpenAI-compatible protocol (vLLM, TGI, etc.). Models like Qwen2.5-Coder-Instruct support it natively.

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const llm = new ChatKServe({
  baseUrl: "https://qwen-coder.my-cluster.example.com",
  modelName: "qwen2.5-coder-32b-instruct",
  protocol: "openai", // pin to openai-compat for tool calling
});

const searchTool = tool(
  async ({ query }) => {
    // your implementation
    return `Results for: ${query}`;
  },
  {
    name: "search_codebase",
    description: "Search the codebase for relevant code",
    schema: z.object({
      query: z.string().describe("The search query"),
    }),
  }
);

const llmWithTools = llm.bindTools([searchTool]);

const result = await llmWithTools.invoke(
  "Find all usages of the AuthService class"
);

if (result.tool_calls && result.tool_calls.length > 0) {
  console.log("Tool call:", result.tool_calls[0]);
  // { name: 'search_codebase', args: { query: 'AuthService' }, id: '...' }
}
```

### V2 Inference Protocol

Use `protocol: "v2"` for runtimes that expose the native KServe V2/Open Inference Protocol (e.g., Triton Inference Server):

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";

const llm = new ChatKServe({
  baseUrl: "https://triton.my-cluster.example.com",
  modelName: "llama-3-8b-instruct",
  protocol: "v2",
  // Chat template format — "chatml" (default) works for Qwen models,
  // "llama" for Llama 2/3 style [INST] format
  chatTemplate: "llama",
});

const result = await llm.invoke("Explain transformers in simple terms.");
```

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

```typescript
// No constructor args needed — reads from env
const llm = new ChatKServe({
  baseUrl: process.env.KSERVE_BASE_URL!,
  modelName: process.env.KSERVE_MODEL_NAME!,
});
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
    // Read fresh token on each request
    const token = await readFile(
      "/var/run/secrets/kubernetes.io/serviceaccount/token",
      "utf-8"
    );
    return token.trim();
  },
});
```

### Custom TLS / CA bundle

For self-signed certificates or internal CAs common in Kubernetes clusters:

```typescript
const llm = new ChatKServe({
  baseUrl: "https://my-model.internal.example.com",
  modelName: "my-model",
  caBundle: "/etc/ssl/certs/my-internal-ca.pem",
  // or disable verification entirely (not recommended for production)
  // verifySsl: false,
});
```

### Connection tuning

```typescript
const llm = new ChatKServe({
  baseUrl: "https://my-model.cluster.example.com",
  modelName: "my-model",
  // Generous timeout for cold starts (KServe scales from 0)
  timeout: 300_000, // 5 minutes
  // Retries with exponential backoff + jitter
  maxRetries: 5,
});
```

### DeepAgents integration

```typescript
import { ChatKServe } from "@bitkaio/langchain-kserve";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Define your tools
const readFile = tool(
  async ({ path }) => { /* ... */ },
  {
    name: "read_file",
    description: "Read a file from the codebase",
    schema: z.object({ path: z.string() }),
  }
);

const writeFile = tool(
  async ({ path, content }) => { /* ... */ },
  {
    name: "write_file",
    description: "Write content to a file",
    schema: z.object({ path: z.string(), content: z.string() }),
  }
);

const llm = new ChatKServe({
  baseUrl: process.env.KSERVE_BASE_URL!,
  modelName: "qwen2.5-coder-32b-instruct",
  temperature: 0.2,
  streaming: true,
});

// Use with DeepAgents or any LangChain agent framework
import { Agent } from "deepagents";

const agent = new Agent({
  llm,
  tools: [readFile, writeFile],
});

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
| `timeout` | `number` | `120000` | Request timeout (ms) |
| `maxRetries` | `number` | `3` | Max retry attempts |
| `chatTemplate` | `"chatml" \| "llama" \| "custom"` | `"chatml"` | Template for V2 protocol |
| `customChatTemplate` | `string` | — | Custom template string |

### `KServeLLM`

Extends `BaseLLM`. Use for base/completion models.

Same options as `ChatKServe`, minus `chatTemplate` and `customChatTemplate`.

### Error classes

```typescript
import {
  KServeError,            // base class
  KServeConnectionError,  // network/DNS failures
  KServeAuthenticationError, // 401/403
  KServeModelNotFoundError,  // 404, model not loaded
  KServeInferenceError,   // 4xx/5xx during inference
  KServeTimeoutError,     // timeout exceeded
} from "@bitkaio/langchain-kserve";
```

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
