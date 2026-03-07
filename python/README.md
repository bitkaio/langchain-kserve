# langchain-kserve

A [LangChain](https://www.langchain.com/) provider package for models hosted on [KServe](https://kserve.github.io/website/).

Connect any LangChain chain, agent, or multi-agent framework to self-hosted models on Kubernetes — including vLLM, TGI, Triton, and custom KServe runtimes.

## Features

- **`ChatKServe`** — `BaseChatModel` for chat/instruct models (Qwen, Llama, Mistral, …)
- **`KServeLLM`** — `BaseLLM` for base completion models
- **Dual protocol** — OpenAI-compatible (`/v1/chat/completions`) and V2 Inference Protocol (`/v2/models/{model}/infer`), auto-detected
- **Full streaming** — sync and async, SSE for OpenAI-compat, chunked transfer for V2
- **Tool calling** — full OpenAI function-calling format with `tool_choice`, `parallel_tool_calls`, and proper `invalid_tool_calls` handling
- **Vision / multimodal** — send images alongside text via OpenAI content blocks
- **Token usage tracking** — `llm_output` and `generation_info` populated from vLLM responses, including streaming
- **Logprobs** — optional per-token log-probabilities in `generation_info`
- **Finish reason** — always propagated in `generation_info`
- **Model introspection** — `get_model_info()` returns unified `KServeModelInfo` for both protocols
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
| `logprobs` | `bool` | `None` | Return per-token log-probabilities (OpenAI-compat only) |
| `top_logprobs` | `int` | `None` | Number of top logprobs per token (OpenAI-compat only) |
| `tool_choice` | `str \| dict` | `None` | Tool selection strategy: `"auto"`, `"required"`, `"none"`, or specific function |
| `parallel_tool_calls` | `bool` | `None` | Allow the model to call multiple tools in one turn |
| `response_format` | `dict` | `None` | Response format constraint (e.g. `{"type": "json_object"}`) (OpenAI-compat only) |
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

### Token usage tracking

Token usage is available in both `llm_output` (request-level) and `generation_info` (per-generation):

```python
result = llm._generate([HumanMessage(content="Hello")])

# Request-level (on ChatResult)
print(result.llm_output)
# {"token_usage": {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}, "model_name": "..."}

# Per-generation
gen_info = result.generations[0].generation_info
print(gen_info["prompt_tokens"])     # 5
print(gen_info["completion_tokens"]) # 12
print(gen_info["finish_reason"])     # "stop"
```

Token usage is also captured from the final streaming chunk (vLLM sends it when `stream_options.include_usage=True`):

```python
chunks = list(llm.stream("Hello"))
last_chunk = chunks[-1]
print(last_chunk.generation_info)  # {"token_usage": {...}}
```

### Logprobs

```python
llm = ChatKServe(
    base_url="...",
    model_name="...",
    logprobs=True,
    top_logprobs=5,
    protocol="openai",
)

response = llm.invoke("Hello")
print(response.response_metadata["logprobs"])
# {"tokens": [...], "token_logprobs": [...], "top_logprobs": [...]}
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
    tool_choice="auto",
    parallel_tool_calls=True,
)

llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What's the weather in Berlin and Paris?")

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Tool: {tc['name']}, Args: {tc['args']}")

# Malformed arguments land in invalid_tool_calls instead of crashing
if response.invalid_tool_calls:
    for itc in response.invalid_tool_calls:
        print(f"Invalid call: {itc['name']}, raw args: {itc['args']}")
```

### Vision / multimodal

Pass images alongside text using OpenAI content blocks. Works with OpenAI-compatible runtimes that support vision (e.g., vLLM with a multimodal model).

```python
from langchain_core.messages import HumanMessage
from langchain_kserve import ChatKServe
import base64, pathlib

llm = ChatKServe(
    base_url="https://llava.default.svc.cluster.local",
    model_name="llava-1.6",
    protocol="openai",
)

# Base64-encoded image (preferred for cluster-internal use)
image_data = base64.b64encode(pathlib.Path("chart.png").read_bytes()).decode()
message = HumanMessage(content=[
    {"type": "text", "text": "Describe what you see in this chart:"},
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_data}",
            "detail": "high",
        },
    },
])

response = llm.invoke([message])
print(response.content)
```

URL-based images are also supported (the model pod fetches the image):

```python
message = HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
])
```

### Model introspection

```python
import asyncio
from langchain_kserve import ChatKServe

llm = ChatKServe(base_url="...", model_name="qwen2.5-coder-32b-instruct")
info = asyncio.run(llm.get_model_info())

print(info["model_name"])    # "qwen2.5-coder-32b-instruct"
print(info["platform"])      # "openai-compat" or V2 platform string
print(info["raw"])           # full response from the endpoint
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

### JSON mode / Response format

Constrain the model output to valid JSON or a specific JSON schema (vLLM uses grammar-constrained decoding):

```python
# Force valid JSON output
llm = ChatKServe(
    base_url="...",
    model_name="...",
    protocol="openai",
    response_format={"type": "json_object"},
)

# Force output matching a specific JSON schema
llm = ChatKServe(
    base_url="...",
    model_name="...",
    protocol="openai",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
        },
    },
)
```

### Structured output

Use `with_structured_output()` for the high-level structured output API:

```python
from langchain_kserve import ChatKServe
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

llm = ChatKServe(
    base_url="...",
    model_name="...",
    protocol="openai",
)

# function_calling (default) — most reliable, uses tool calling
structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke("Extract: John is 30 years old.")
print(result.name, result.age)  # John, 30

# json_schema — uses vLLM grammar-constrained decoding
structured_llm = llm.with_structured_output(Person, method="json_schema")

# json_mode — uses json_object response format with schema instruction
structured_llm = llm.with_structured_output(Person, method="json_mode")

# include_raw=True — get both raw AIMessage and parsed output
structured_llm = llm.with_structured_output(Person, include_raw=True)
result = structured_llm.invoke("Extract: John is 30.")
print(result["parsed"])   # Person(name='John', age=30)
print(result["raw"])      # AIMessage(...)
print(result["parsing_error"])  # None
```

### Embeddings

```python
from langchain_kserve import KServeEmbeddings

embeddings = KServeEmbeddings(
    base_url="https://my-embedding-model.cluster.example.com",
    model_name="Qwen/Qwen3-Embedding-0.6B",
)

# Embed documents (batched automatically)
vectors = embeddings.embed_documents(["Hello world", "How are you?"])
print(len(vectors), len(vectors[0]))  # 2, <embedding_dim>

# Embed a single query
query_vector = embeddings.embed_query("What is KServe?")

# Async
vectors = await embeddings.aembed_documents(["Hello world", "How are you?"])
query_vector = await embeddings.aembed_query("What is KServe?")

# With dimensions (Matryoshka models)
embeddings = KServeEmbeddings(
    base_url="...",
    model_name="...",
    dimensions=512,
)
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
| JSON mode / response_format | ✅ | ❌ |
| Structured output | ✅ | ❌ |
| Embeddings | ✅ | ❌ |

Attempting tool calling, vision, or response_format with V2 raises `KServeInferenceError` immediately (before any HTTP call) with a clear message.

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
    # Also raised for unsupported features on V2 (tools, vision)
```

## Embeddings

```python
from langchain_kserve import KServeEmbeddings

embeddings = KServeEmbeddings(
    base_url="https://my-embedding-model.cluster.example.com",
    model_name="Qwen/Qwen3-Embedding-0.6B",
    chunk_size=500,      # max texts per API call (default: 1000)
    dimensions=512,      # optional, for Matryoshka models
)

vectors = embeddings.embed_documents(["Hello world", "How are you?"])
query_vector = embeddings.embed_query("What is KServe?")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | `KSERVE_EMBEDDINGS_BASE_URL` or `KSERVE_BASE_URL` | Embeddings service URL |
| `model_name` | `str` | `KSERVE_EMBEDDINGS_MODEL_NAME` | Embedding model name |
| `api_key` | `SecretStr` | `KSERVE_API_KEY` | Bearer token |
| `dimensions` | `int` | `None` | Output embedding dimensions |
| `encoding_format` | `"float" \| "base64"` | `"float"` | Wire format (base64 decoded automatically) |
| `chunk_size` | `int` | `1000` | Max texts per API call |
| `timeout` | `int` | `120` | Request timeout (seconds) |
| `max_retries` | `int` | `3` | Retry attempts |

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

# Token usage
result = llm.generate(["The quick brown fox"])
print(result.llm_output)  # {"token_usage": {...}}
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
