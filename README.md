# langchain-kserve

LangChain provider packages for models served on [KServe](https://kserve.github.io/website/) — the model inference platform for Kubernetes.

Connect any LangChain chain, agent, or multi-agent framework to self-hosted models running on Kubernetes. Works with vLLM, Triton Inference Server, TGI, and any other KServe-compatible runtime, using both the OpenAI-compatible API and the native V2 Inference Protocol.

## Packages

| Package | Language | Registry | Description |
|---------|----------|----------|-------------|
| [`langchain-kserve`](./python) | Python | [![PyPI](https://img.shields.io/pypi/v/langchain-kserve)](https://pypi.org/project/langchain-kserve/) | LangChain integration (`ChatKServe`, `KServeLLM`) |
| [`@langchain/kserve`](./typescript) | TypeScript | [![npm](https://img.shields.io/npm/v/@langchain/kserve)](https://www.npmjs.com/package/@langchain/kserve) | LangChain.js integration (`ChatKServe`, `KServeLLM`) |

## Features

- **`ChatKServe`** — `BaseChatModel` for chat/instruct models (Qwen, Llama, Mistral, …)
- **`KServeLLM`** — `BaseLLM` for base completion models
- **Dual protocol** — OpenAI-compatible (`/v1/chat/completions`) and V2 Inference Protocol (`/v2/models/{model}/infer`), auto-detected per instance
- **Full streaming** — sync and async, SSE for OpenAI-compat, chunked transfer for V2
- **Tool calling** — passes LangChain tool definitions through in OpenAI function-calling format
- **Production-grade** — custom TLS/CA bundles, static and dynamic bearer token auth (K8s service account tokens), exponential backoff retries, generous cold-start timeouts

## Quick Start

### Python

```bash
pip install langchain-kserve
```

```python
from langchain_kserve import ChatKServe

llm = ChatKServe(
    base_url="https://qwen-coder.default.svc.cluster.local",
    model_name="qwen2.5-coder-32b-instruct",
    temperature=0.2,
)

response = llm.invoke("Write a binary search in Python.")
print(response.content)
```

### TypeScript

```bash
npm install @langchain/kserve @langchain/core
```

```typescript
import { ChatKServe } from "@langchain/kserve";

const llm = new ChatKServe({
  baseUrl: "https://qwen-coder.default.svc.cluster.local",
  modelName: "qwen2.5-coder-32b-instruct",
  temperature: 0.2,
});

const response = await llm.invoke("Write a binary search in TypeScript.");
console.log(response.content);
```

## Protocol Support

Both packages auto-detect the inference protocol by probing `GET /v1/models` on the inference service:

| Runtime | Default Protocol |
|---------|-----------------|
| vLLM | OpenAI-compatible |
| TGI | OpenAI-compatible |
| Triton Inference Server | V2 Inference Protocol |
| Custom KServe runtime | Auto-detected |

Pin the protocol explicitly to skip auto-detection:

```python
# Python
llm = ChatKServe(..., protocol="openai")  # or "v2"
```

```typescript
// TypeScript
const llm = new ChatKServe({ ..., protocol: "openai" });  // or "v2"
```

## Authentication

Both packages support static bearer tokens and dynamic token providers (e.g., Kubernetes service account tokens):

```python
# Python — dynamic K8s SA token
import pathlib

llm = ChatKServe(
    base_url="https://model.my-namespace.svc.cluster.local",
    model_name="my-model",
    token_provider=lambda: pathlib.Path(
        "/var/run/secrets/kubernetes.io/serviceaccount/token"
    ).read_text().strip(),
)
```

```typescript
// TypeScript — dynamic K8s SA token
import { readFile } from "node:fs/promises";

const llm = new ChatKServe({
  baseUrl: "https://model.my-namespace.svc.cluster.local",
  modelName: "my-model",
  tokenProvider: () =>
    readFile("/var/run/secrets/kubernetes.io/serviceaccount/token", "utf-8")
      .then((t) => t.trim()),
});
```

## Environment Variables

Both packages read configuration from `KSERVE_`-prefixed environment variables:

| Variable | Description |
|----------|-------------|
| `KSERVE_BASE_URL` | Root URL of the KServe inference service |
| `KSERVE_MODEL_NAME` | Model identifier as registered in KServe |
| `KSERVE_API_KEY` | Static bearer token |
| `KSERVE_PROTOCOL` | `openai`, `v2`, or `auto` (default) |
| `KSERVE_CA_BUNDLE` | Path to custom CA certificate bundle |

## Repository Structure

```
langchain-kserve/
├── python/          # langchain-kserve Python package
│   ├── src/
│   ├── tests/
│   └── pyproject.toml
└── typescript/      # @langchain/kserve TypeScript package
    ├── src/
    ├── tests/
    └── package.json
```

See the package-specific READMEs for full API references and usage examples:

- [Python README](./python/README.md)
- [TypeScript README](./typescript/README.md)

## License

MIT — © bitkaio LLC
