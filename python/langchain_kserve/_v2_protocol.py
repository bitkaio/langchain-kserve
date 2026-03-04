"""V2 Inference Protocol (Open Inference Protocol) request/response mapping.

Handles ``POST /v2/models/{model}/infer`` as defined by the KServe / NVIDIA
Triton V2 specification.  Because the V2 protocol is tensor-based, chat
messages must first be serialised to a single prompt string using a simple
ChatML template (or the caller-supplied template).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence

import httpx
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.messages import AIMessage, AIMessageChunk

from langchain_kserve._common import KServeInferenceError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default chat template (ChatML)
# ---------------------------------------------------------------------------

_CHATML_SYSTEM = "<|im_start|>system\n{content}<|im_end|>\n"
_CHATML_USER = "<|im_start|>user\n{content}<|im_end|>\n"
_CHATML_ASSISTANT = "<|im_start|>assistant\n{content}<|im_end|>\n"
_CHATML_ASSISTANT_START = "<|im_start|>assistant\n"


def messages_to_prompt(
    messages: Sequence[BaseMessage],
    chat_template: Optional[str] = None,
) -> str:
    """Format a list of LangChain messages into a single prompt string.

    If ``chat_template`` is provided it is used as a Jinja2-compatible template
    (the caller is responsible for rendering it).  Otherwise the built-in
    ChatML format is used.

    Args:
        messages: The conversation messages to format.
        chat_template: Optional Jinja2 template string.

    Returns:
        A single formatted prompt string ready to feed to the model.
    """
    if chat_template is not None:
        # Caller is responsible for rendering Jinja2 templates externally.
        # For now fall through to ChatML if template rendering is not available.
        logger.debug(
            "Custom chat_template provided but Jinja2 rendering is not built-in; "
            "falling back to ChatML."
        )

    parts: List[str] = []
    for msg in messages:
        role = msg.type  # "human", "ai", "system", "tool"
        content = _extract_text(msg)
        if role == "system":
            parts.append(_CHATML_SYSTEM.format(content=content))
        elif role in ("human", "user"):
            parts.append(_CHATML_USER.format(content=content))
        elif role in ("ai", "assistant"):
            parts.append(_CHATML_ASSISTANT.format(content=content))
        else:
            # Tool messages and unknown roles: treat as user turn
            parts.append(_CHATML_USER.format(content=content))

    parts.append(_CHATML_ASSISTANT_START)
    return "".join(parts)


def _extract_text(msg: BaseMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    parts: List[str] = []
    for block in msg.content:  # type: ignore[union-attr]
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------


def build_v2_infer_request(
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    top_p: float,
    stop: Optional[List[str]],
    extra_kwargs: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a V2 Inference Protocol ``/infer`` request body.

    Args:
        model_name: Model identifier (used in parameters, not the path).
        prompt: Formatted prompt string.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        stop: Stop sequences.
        extra_kwargs: Additional parameters merged into ``parameters``.
        request_id: Optional request ID (UUID generated if omitted).

    Returns:
        JSON-serialisable V2 infer request body.
    """
    req_id = request_id or str(uuid.uuid4())

    parameters: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        parameters["max_tokens"] = max_tokens
    if stop:
        parameters["stop"] = stop
    if extra_kwargs:
        parameters.update(extra_kwargs)

    return {
        "id": req_id,
        "inputs": [
            {
                "name": "text_input",
                "shape": [1],
                "datatype": "BYTES",
                "data": [prompt],
            }
        ],
        "parameters": parameters,
    }


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def parse_v2_chat_response(
    response: httpx.Response,
    model_name: str,
) -> ChatResult:
    """Parse a V2 ``/infer`` response into a :class:`~langchain_core.outputs.ChatResult`.

    Args:
        response: Raw HTTP response.
        model_name: Model name for generation metadata.

    Returns:
        :class:`~langchain_core.outputs.ChatResult` with the generated text.

    Raises:
        KServeInferenceError: If the response is malformed.
    """
    text = _extract_v2_output_text(response)
    ai_message = AIMessage(content=text)
    generation_info: Dict[str, Any] = {
        "model": model_name,
        "protocol": "v2",
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    return ChatResult(
        generations=[ChatGeneration(message=ai_message, generation_info=generation_info)]
    )


def parse_v2_completion_response(response: httpx.Response) -> str:
    """Parse a V2 ``/infer`` response and return the generated text string.

    Args:
        response: Raw HTTP response.

    Returns:
        Generated text.

    Raises:
        KServeInferenceError: If the response is malformed.
    """
    return _extract_v2_output_text(response)


def _extract_v2_output_text(response: httpx.Response) -> str:
    try:
        data = response.json()
    except Exception as exc:
        raise KServeInferenceError(f"Failed to parse V2 response JSON: {exc}") from exc

    outputs = data.get("outputs", [])
    if not outputs:
        raise KServeInferenceError(f"No outputs in V2 response: {data}")

    output = outputs[0]
    raw_data = output.get("data", [])
    if not raw_data:
        raise KServeInferenceError(f"Empty data in V2 output: {output}")

    value = raw_data[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# ---------------------------------------------------------------------------
# Sync streaming for V2
# ---------------------------------------------------------------------------


def stream_v2_chat_response(
    client: httpx.Client,
    path: str,
    body: Dict[str, Any],
    model_name: str,
) -> Iterator[ChatGenerationChunk]:
    """Stream a V2 infer request via chunked transfer encoding.

    .. note::
        Not all KServe runtimes support V2 HTTP streaming.  If the runtime
        returns a non-chunked response, this function falls back to yielding
        a single chunk from the complete response.

    Args:
        client: Sync HTTP client.
        path: Inference path (``/v2/models/{model}/infer``).
        body: V2 infer request body.
        model_name: Model name for metadata.

    Yields:
        :class:`~langchain_core.outputs.ChatGenerationChunk`.
    """
    with client.stream("POST", path, json=body) as response:
        if not response.is_success:
            raise KServeInferenceError(
                f"V2 stream request failed ({response.status_code})"
            )
        chunks: List[bytes] = []
        for chunk in response.iter_bytes():
            chunks.append(chunk)

    # Try to parse as newline-delimited JSON (NDJSON) or single JSON response
    full_bytes = b"".join(chunks)
    yield from _parse_v2_stream_bytes(full_bytes, model_name)


async def astream_v2_chat_response(
    client: httpx.AsyncClient,
    path: str,
    body: Dict[str, Any],
    model_name: str,
) -> AsyncIterator[ChatGenerationChunk]:
    """Async version of :func:`stream_v2_chat_response`.

    Yields:
        :class:`~langchain_core.outputs.ChatGenerationChunk`.
    """
    async with client.stream("POST", path, json=body) as response:
        if not response.is_success:
            raise KServeInferenceError(
                f"V2 stream request failed ({response.status_code})"
            )
        chunks: List[bytes] = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)

    full_bytes = b"".join(chunks)
    for chunk in _parse_v2_stream_bytes(full_bytes, model_name):
        yield chunk


def _parse_v2_stream_bytes(
    data: bytes, model_name: str
) -> Iterator[ChatGenerationChunk]:
    """Parse raw bytes from a V2 response into :class:`ChatGenerationChunk`.

    Handles both NDJSON (one JSON object per line) and single-object responses.

    Args:
        data: Raw response bytes.
        model_name: Model name for metadata.

    Yields:
        :class:`~langchain_core.outputs.ChatGenerationChunk`.
    """
    text = data.decode("utf-8")
    lines = [l.strip() for l in text.splitlines() if l.strip()]  # noqa: E741

    yielded = False
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        outputs = obj.get("outputs", [])
        if outputs:
            value = outputs[0].get("data", [""])[0]
            token = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=token),
                generation_info={"model": model_name, "protocol": "v2"},
            )
            yielded = True

    if not yielded and text.strip():
        # Single complete response wrapped as one chunk
        try:
            obj = json.loads(text)
            outputs = obj.get("outputs", [])
            if outputs:
                value = outputs[0].get("data", [""])[0]
                token = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=token),
                    generation_info={"model": model_name, "protocol": "v2"},
                )
        except json.JSONDecodeError:
            logger.warning("Could not parse V2 stream response: %s", text[:200])


# ---------------------------------------------------------------------------
# V2 infer path helper
# ---------------------------------------------------------------------------


def infer_path(model_name: str) -> str:
    """Return the V2 infer URL path for the given model.

    Args:
        model_name: The model identifier.

    Returns:
        Path string e.g. ``"/v2/models/my-model/infer"``.
    """
    return f"/v2/models/{model_name}/infer"
