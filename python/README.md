# langchain-kserve

A [LangChain](https://www.langchain.com/) provider package for models hosted on [KServe](https://kserve.github.io/website/).

Connect any LangChain chain, agent, or multi-agent framework to self-hosted models on Kubernetes — including vLLM, TGI, Triton, and custom KServe runtimes.

## Features

- **`ChatKServe`** — `BaseChatModel` for chat/instruct models (Qwen, Llama, Mistral, …)
- **`KServeLLM`** — `BaseLLM` for base completion models
- **Dual protocol** — OpenAI-compatible (`/v1/chat/completions`) and V2 Inference Protocol (`/v2/models/{model}/infer`), auto-detected
- **Full streaming** — sync and async, SSE for OpenAI-compat, chunked transfer for V2
- **Tool calling** — pass tools through in OpenAI function-calling format
- **Production-grade** — TLS/CA bundle, bearer token + dynamic token provider (K8s SA tokens), exponential backoff retries, generous cold-start timeouts

## Installation

```bash
pip install langchain-kserve
```

## Quick Start

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

## Configuration

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | — | Root URL of the KServe inference service |
| `model_name` | `str` | — | Model identifier |
| `protocol` | `"openai"` \| `"v2"` \| `"auto"` | `"auto"` | Inference protocol |
| `api_key` | `SecretStr` | `None` | Static bearer token |
| `token_provider` | `Callable[[], str]` | `None` | Dynamic token provider |
| `verify_ssl` | `bool` | `True` | Verify TLS certificates |
| `ca_bundle` | `str` | `None` | Path to custom CA cert bundle |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `None` | Max tokens to generate |
| `top_p` | `float` | `1.0` | Nucleus sampling |
| `stop` | `List[str]` | `None` | Stop sequences |
| `streaming` | `bool` | `False` | Default streaming mode |
| `timeout` | `int` | `120` | Request timeout in seconds |
| `max_retries` | `int` | `3` | Retry attempts |

### Environment variables

```bash
export KSERVE_BASE_URL="https://qwen-coder.default.svc.cluster.local"
export KSERVE_MODEL_NAME="qwen2.5-coder-32b-instruct"
export KSERVE_API_KEY="my-token"          # optional
export KSERVE_PROTOCOL="openai"           # optional, default: auto
export KSERVE_CA_BUNDLE="/path/to/ca.crt" # optional
```

## Usage Examples

### Basic invocation

```python
from langchain_kserve import ChatKServe

llm = ChatKServe(
    base_url="https://qwen-coder.default.svc.cluster.local",
    model_name="qwen2.5-coder-32b-instruct",
)

response = llm.invoke("Explain the GIL in Python.")
print(response.content)
```

### Streaming

```python
for chunk in llm.stream("Implement a red-black tree in Python."):
    print(chunk.content, end="", flush=True)
```

### Async

```python
import asyncio

async def main():
    response = await llm.ainvoke("What is KServe?")
    print(response.content)

    async for chunk in llm.astream("Explain transformers."):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

### In a LangChain chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_kserve import ChatKServe

llm = ChatKServe(
    base_url="https://qwen-coder.default.svc.cluster.local",
    model_name="qwen2.5-coder-32b-instruct",
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {language} developer."),
    ("human", "{input}"),
])

chain = prompt | llm
response = chain.invoke({"language": "Python", "input": "Implement a LRU cache."})
print(response.content)
```

### Tool calling

Tool calling works when using the OpenAI-compatible endpoint (e.g., vLLM, TGI).

```python
from langchain_core.tools import tool
from langchain_kserve import ChatKServe

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}."

llm = ChatKServe(
    base_url="https://qwen-coder.default.svc.cluster.local",
    model_name="qwen2.5-coder-32b-instruct",
    protocol="openai",
)

llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What's the weather in Berlin?")

# If the model calls the tool:
if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(f"Tool: {tool_call['name']}, Args: {tool_call['args']}")
```

### Forcing V2 Inference Protocol

```python
from langchain_kserve import ChatKServe

llm = ChatKServe(
    base_url="https://triton.default.svc.cluster.local",
    model_name="my-triton-model",
    protocol="v2",  # skip auto-detection
    temperature=0.5,
    max_tokens=512,
)

response = llm.invoke("Summarise this text in one sentence.")
print(response.content)
```

### Kubernetes service account token (dynamic auth)

```python
import pathlib
from langchain_kserve import ChatKServe

SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"

def k8s_token() -> str:
    return pathlib.Path(SA_TOKEN_PATH).read_text().strip()

llm = ChatKServe(
    base_url="https://qwen-coder.my-namespace.svc.cluster.local",
    model_name="qwen2.5-coder-32b-instruct",
    token_provider=k8s_token,  # refreshed on every request
)
```

### Custom CA certificate

```python
from langchain_kserve import ChatKServe

llm = ChatKServe(
    base_url="https://kserve-internal.my-corp.com",
    model_name="llama-3-70b",
    ca_bundle="/etc/ssl/certs/my-corp-ca.crt",
)
```

### DeepAgents integration

```python
from langchain_kserve import ChatKServe
from langchain_core.tools import tool

@tool
def code_executor(code: str) -> str:
    """Execute Python code and return the output."""
    import subprocess
    result = subprocess.run(["python", "-c", code], capture_output=True, text=True)
    return result.stdout or result.stderr

llm = ChatKServe(
    base_url="https://qwen-coder.default.svc.cluster.local",
    model_name="qwen2.5-coder-32b-instruct",
    temperature=0.1,
    max_tokens=2048,
    protocol="openai",
)

# Use with any LangChain-compatible agent framework
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant with access to a Python executor."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [code_executor], prompt)
executor = AgentExecutor(agent=agent, tools=[code_executor], verbose=True)
result = executor.invoke({"input": "Calculate the 10th Fibonacci number using code."})
print(result["output"])
```

## Protocol Auto-Detection

When `protocol="auto"` (the default), `langchain-kserve` probes the endpoint:

1. `GET /v1/models` — if it responds 200, use OpenAI-compatible protocol
2. Otherwise fall back to V2 Inference Protocol

The detected protocol is cached per instance to avoid repeated probes.

## Error Handling

```python
from langchain_kserve import ChatKServe
from langchain_kserve import (
    KServeConnectionError,
    KServeAuthenticationError,
    KServeModelNotFoundError,
    KServeInferenceError,
    KServeTimeoutError,
)

llm = ChatKServe(base_url="...", model_name="...")

try:
    response = llm.invoke("Hello")
except KServeConnectionError:
    print("Cannot reach the inference service (cold start, DNS, TLS issue)")
except KServeAuthenticationError:
    print("Invalid or missing bearer token")
except KServeModelNotFoundError:
    print("Model name does not match what is served")
except KServeTimeoutError:
    print("Inference timed out (model may be scaling up)")
except KServeInferenceError as e:
    print(f"Inference error: {e}")
```

## Base Completion Models (`KServeLLM`)

For non-chat, base completion models use `KServeLLM`:

```python
from langchain_kserve import KServeLLM

llm = KServeLLM(
    base_url="https://llama-base.default.svc.cluster.local",
    model_name="llama-3-8b",
    max_tokens=128,
)

text = llm.invoke("The quick brown fox")
print(text)

# Streaming
for chunk in llm.stream("Once upon a time"):
    print(chunk, end="", flush=True)
```

## Development

```bash
# Install dependencies
poetry install

# Run unit tests
pytest tests/unit_tests/

# Run integration tests (requires live KServe endpoint)
KSERVE_BASE_URL=https://... KSERVE_MODEL_NAME=... pytest -m integration

# Type checking
mypy --strict langchain_kserve/

# Linting
ruff check langchain_kserve/ tests/
```

## License

MIT
