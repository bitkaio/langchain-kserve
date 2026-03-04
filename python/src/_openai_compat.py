"""OpenAI-compatible endpoint request/response mapping for KServe.

Handles ``POST /v1/chat/completions`` and ``POST /v1/completions`` as served
by runtimes like vLLM and TGI.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence

import httpx
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_kserve._common import (
    KServeInferenceError,
    async_request_with_retry,
    request_with_retry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message serialisation
# ---------------------------------------------------------------------------


def messages_to_openai_dicts(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    """Convert LangChain messages to the OpenAI ``messages`` list format.

    Args:
        messages: Sequence of :class:`~langchain_core.messages.BaseMessage`.

    Returns:
        List of dicts suitable for the ``messages`` field in a chat completion
        request body.
    """
    result: List[Dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": _get_text_content(msg)})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": _get_text_content(msg)})
        elif isinstance(msg, AIMessage):
            d: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(d)
        elif isinstance(msg, ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": str(msg.content),
                }
            )
        else:
            # Fallback: treat as user message
            result.append({"role": "user", "content": _get_text_content(msg)})
    return result


def _get_text_content(msg: BaseMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    # Handle list of content blocks (multimodal)
    parts: List[str] = []
    for block in msg.content:  # type: ignore[union-attr]
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def build_chat_request(
    model_name: str,
    messages: Sequence[BaseMessage],
    temperature: float,
    max_tokens: Optional[int],
    top_p: float,
    stop: Optional[List[str]],
    stream: bool,
    tools: Optional[List[Dict[str, Any]]] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a ``/v1/chat/completions`` request body.

    Args:
        model_name: The model identifier.
        messages: Conversation messages.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate (omitted if ``None``).
        top_p: Nucleus sampling parameter.
        stop: Stop sequences.
        stream: Whether to stream the response.
        tools: OpenAI-format tool schemas (for tool calling).
        extra_kwargs: Additional parameters merged into the request body.

    Returns:
        JSON-serialisable request body dict.
    """
    body: Dict[str, Any] = {
        "model": model_name,
        "messages": messages_to_openai_dicts(messages),
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if stop:
        body["stop"] = stop
    if tools:
        body["tools"] = tools
    if extra_kwargs:
        body.update(extra_kwargs)
    return body


def build_completion_request(
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    top_p: float,
    stop: Optional[List[str]],
    stream: bool,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a ``/v1/completions`` request body.

    Args:
        model_name: The model identifier.
        prompt: The text prompt.
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        top_p: Nucleus sampling.
        stop: Stop sequences.
        stream: Whether to stream.
        extra_kwargs: Extra parameters merged into the body.

    Returns:
        JSON-serialisable request body dict.
    """
    body: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if stop:
        body["stop"] = stop
    if extra_kwargs:
        body.update(extra_kwargs)
    return body


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def parse_chat_response(
    response: httpx.Response,
    model_name: str,
    protocol_used: str = "openai",
) -> ChatResult:
    """Parse a non-streaming ``/v1/chat/completions`` response.

    Args:
        response: The raw HTTP response.
        model_name: Model name for generation metadata.
        protocol_used: Protocol label for generation info.

    Returns:
        A :class:`~langchain_core.outputs.ChatResult`.

    Raises:
        KServeInferenceError: If the response format is unexpected.
    """
    try:
        data = response.json()
    except Exception as exc:
        raise KServeInferenceError(f"Failed to parse response JSON: {exc}") from exc

    choices = data.get("choices", [])
    if not choices:
        raise KServeInferenceError(f"No choices in response: {data}")

    choice = choices[0]
    message = choice.get("message", {})
    content: str = message.get("content") or ""
    finish_reason: Optional[str] = choice.get("finish_reason")

    tool_calls: List[ToolCall] = []
    raw_tool_calls = message.get("tool_calls") or []
    for tc in raw_tool_calls:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(
            ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                args=args,
            )
        )

    ai_message = AIMessage(content=content, tool_calls=tool_calls)

    usage = data.get("usage", {})
    generation_info: Dict[str, Any] = {
        "model": model_name,
        "protocol": protocol_used,
        "finish_reason": finish_reason,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }

    return ChatResult(
        generations=[ChatGeneration(message=ai_message, generation_info=generation_info)]
    )


def parse_completion_response(response: httpx.Response, model_name: str) -> str:
    """Parse a non-streaming ``/v1/completions`` response.

    Args:
        response: The raw HTTP response.
        model_name: Model name (for logging).

    Returns:
        The generated text string.

    Raises:
        KServeInferenceError: If the response format is unexpected.
    """
    try:
        data = response.json()
    except Exception as exc:
        raise KServeInferenceError(f"Failed to parse response JSON: {exc}") from exc

    choices = data.get("choices", [])
    if not choices:
        raise KServeInferenceError(f"No choices in response: {data}")

    return choices[0].get("text", "")


# ---------------------------------------------------------------------------
# Sync streaming
# ---------------------------------------------------------------------------


def stream_chat_response(
    client: httpx.Client,
    path: str,
    body: Dict[str, Any],
    max_retries: int,
    model_name: str,
    protocol_used: str = "openai",
) -> Iterator[ChatGenerationChunk]:
    """Stream ``/v1/chat/completions`` and yield :class:`ChatGenerationChunk`.

    Args:
        client: Sync HTTP client.
        path: Endpoint path (``"/v1/chat/completions"``).
        body: Request body (must have ``stream=True``).
        max_retries: Retry count for the initial connection.
        model_name: Model name for generation metadata.
        protocol_used: Protocol label.

    Yields:
        :class:`~langchain_core.outputs.ChatGenerationChunk` for each SSE token.
    """
    with client.stream("POST", path, json=body) as response:
        _raise_stream_status(response)
        for line in response.iter_lines():
            chunk = _parse_sse_chat_line(line, model_name, protocol_used)
            if chunk is not None:
                yield chunk


def stream_completion_response(
    client: httpx.Client,
    path: str,
    body: Dict[str, Any],
    model_name: str,
) -> Iterator[str]:
    """Stream ``/v1/completions`` and yield text deltas.

    Args:
        client: Sync HTTP client.
        path: Endpoint path.
        body: Request body.
        model_name: Model name.

    Yields:
        Text delta strings.
    """
    with client.stream("POST", path, json=body) as response:
        _raise_stream_status(response)
        for line in response.iter_lines():
            text = _parse_sse_completion_line(line)
            if text is not None:
                yield text


# ---------------------------------------------------------------------------
# Async streaming
# ---------------------------------------------------------------------------


async def astream_chat_response(
    client: httpx.AsyncClient,
    path: str,
    body: Dict[str, Any],
    model_name: str,
    protocol_used: str = "openai",
) -> AsyncIterator[ChatGenerationChunk]:
    """Async version of :func:`stream_chat_response`.

    Args:
        client: Async HTTP client.
        path: Endpoint path.
        body: Request body (must have ``stream=True``).
        model_name: Model name.
        protocol_used: Protocol label.

    Yields:
        :class:`~langchain_core.outputs.ChatGenerationChunk` per SSE token.
    """
    async with client.stream("POST", path, json=body) as response:
        _raise_stream_status(response)
        async for line in response.aiter_lines():
            chunk = _parse_sse_chat_line(line, model_name, protocol_used)
            if chunk is not None:
                yield chunk


async def astream_completion_response(
    client: httpx.AsyncClient,
    path: str,
    body: Dict[str, Any],
    model_name: str,
) -> AsyncIterator[str]:
    """Async version of :func:`stream_completion_response`.

    Yields:
        Text delta strings.
    """
    async with client.stream("POST", path, json=body) as response:
        _raise_stream_status(response)
        async for line in response.aiter_lines():
            text = _parse_sse_completion_line(line)
            if text is not None:
                yield text


# ---------------------------------------------------------------------------
# SSE parsing helpers
# ---------------------------------------------------------------------------


def _parse_sse_chat_line(
    line: str,
    model_name: str,
    protocol_used: str,
) -> Optional[ChatGenerationChunk]:
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    choices = data.get("choices", [])
    if not choices:
        return None
    delta = choices[0].get("delta", {})
    content: str = delta.get("content") or ""
    finish_reason = choices[0].get("finish_reason")

    # Handle tool call deltas
    tool_call_chunks = delta.get("tool_calls")

    generation_info: Dict[str, Any] = {
        "model": model_name,
        "protocol": protocol_used,
        "finish_reason": finish_reason,
    }

    return ChatGenerationChunk(
        message=AIMessageChunk(
            content=content,
            tool_call_chunks=tool_call_chunks or [],
        ),
        generation_info=generation_info,
    )


def _parse_sse_completion_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    choices = data.get("choices", [])
    if not choices:
        return None
    return choices[0].get("text") or ""


def _raise_stream_status(response: httpx.Response) -> None:
    if not response.is_success:
        raise KServeInferenceError(
            f"KServe stream request failed ({response.status_code})"
        )


# ---------------------------------------------------------------------------
# Protocol detection
# ---------------------------------------------------------------------------


def probe_openai_compat(
    client: httpx.Client,
    max_retries: int,
) -> bool:
    """Check if the endpoint exposes an OpenAI-compatible ``/v1/models`` route.

    Args:
        client: Sync HTTP client.
        max_retries: Retry count.

    Returns:
        ``True`` if the endpoint responds with HTTP 200.
    """
    try:
        response = request_with_retry(client, "GET", "/v1/models", max_retries)
        return response.status_code == 200
    except Exception:
        return False


async def async_probe_openai_compat(
    client: httpx.AsyncClient,
    max_retries: int,
) -> bool:
    """Async version of :func:`probe_openai_compat`.

    Args:
        client: Async HTTP client.
        max_retries: Retry count.

    Returns:
        ``True`` if the endpoint responds with HTTP 200.
    """
    try:
        response = await async_request_with_retry(
            client, "GET", "/v1/models", max_retries
        )
        return response.status_code == 200
    except Exception:
        return False
