"""ChatKServe — LangChain ChatModel for KServe-hosted chat/instruct models."""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_kserve._common import (
    KServeConnectionError,
    async_request_with_retry,
    build_async_client,
    build_sync_client,
    request_with_retry,
)
from langchain_kserve._openai_compat import (
    async_probe_openai_compat,
    astream_chat_response,
    build_chat_request,
    parse_chat_response,
    probe_openai_compat,
    stream_chat_response,
)
from langchain_kserve._v2_protocol import (
    astream_v2_chat_response,
    build_v2_infer_request,
    infer_path,
    messages_to_prompt,
    parse_v2_chat_response,
    stream_v2_chat_response,
)

logger = logging.getLogger(__name__)


class ChatKServe(BaseChatModel):
    """LangChain ``BaseChatModel`` wrapper for KServe-hosted chat/instruct models.

    Supports both the OpenAI-compatible API (``/v1/chat/completions``) and the
    V2 Inference Protocol (``/v2/models/{model}/infer``).  The protocol is
    auto-detected by default.

    Example:
        .. code-block:: python

            from langchain_kserve import ChatKServe

            llm = ChatKServe(
                base_url="https://qwen-coder.default.svc.cluster.local",
                model_name="qwen2.5-coder-32b-instruct",
                temperature=0.2,
            )
            response = llm.invoke("Write a fibonacci function.")

    Environment variables:
        ``KSERVE_BASE_URL``, ``KSERVE_API_KEY``, ``KSERVE_MODEL_NAME``,
        ``KSERVE_PROTOCOL``, ``KSERVE_CA_BUNDLE``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    base_url: str = Field(
        default="",
        description="Root URL of the KServe inference service.",
    )
    model_name: str = Field(
        default="",
        description="Model identifier as served by KServe.",
    )
    protocol: Literal["openai", "v2", "auto"] = Field(
        default="auto",
        description=(
            "Which inference protocol to use. 'auto' probes the endpoint first."
        ),
    )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Static bearer token for authentication.",
    )
    token_provider: Optional[Callable[[], str]] = Field(
        default=None,
        description="Callable that returns a bearer token dynamically.",
        exclude=True,
    )

    # ------------------------------------------------------------------
    # TLS
    # ------------------------------------------------------------------
    verify_ssl: bool = Field(default=True, description="Verify TLS certificates.")
    ca_bundle: Optional[str] = Field(
        default=None,
        description="Path to a custom CA certificate bundle.",
    )

    # ------------------------------------------------------------------
    # Generation params
    # ------------------------------------------------------------------
    temperature: float = Field(default=0.7, ge=0.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop: Optional[List[str]] = Field(default=None)
    streaming: bool = Field(default=False)

    # ------------------------------------------------------------------
    # Connection behaviour
    # ------------------------------------------------------------------
    timeout: int = Field(
        default=120,
        ge=1,
        description="Request timeout in seconds (generous for cold-start).",
    )
    max_retries: int = Field(default=3, ge=0)

    # ------------------------------------------------------------------
    # Tool calling state (set by bind_tools)
    # ------------------------------------------------------------------
    _tools: Optional[List[Dict[str, Any]]] = None
    _resolved_protocol: Optional[Literal["openai", "v2"]] = None

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _populate_from_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Populate fields from environment variables if not explicitly set."""
        if not values.get("base_url"):
            values["base_url"] = get_from_env("base_url", "KSERVE_BASE_URL", default="")
        if not values.get("api_key"):
            env_key = get_from_env("api_key", "KSERVE_API_KEY", default="")
            if env_key:
                values["api_key"] = SecretStr(env_key)
        if not values.get("model_name"):
            values["model_name"] = get_from_env(
                "model_name", "KSERVE_MODEL_NAME", default=""
            )
        if values.get("protocol", "auto") == "auto":
            env_proto = get_from_env("protocol", "KSERVE_PROTOCOL", default="")
            if env_proto in ("openai", "v2"):
                values["protocol"] = env_proto
        if not values.get("ca_bundle"):
            values["ca_bundle"] = get_from_env("ca_bundle", "KSERVE_CA_BUNDLE", default="") or None
        return values

    # ------------------------------------------------------------------
    # LangChain required properties
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "kserve-chat"

    @property
    def _model_name(self) -> str:
        return self.model_name

    # ------------------------------------------------------------------
    # HTTP client helpers
    # ------------------------------------------------------------------

    def _make_sync_client(self) -> httpx.Client:
        return build_sync_client(
            self.base_url,
            self.api_key,
            self.token_provider,
            self.verify_ssl,
            self.ca_bundle,
            self.timeout,
        )

    def _make_async_client(self) -> httpx.AsyncClient:
        return build_async_client(
            self.base_url,
            self.api_key,
            self.token_provider,
            self.verify_ssl,
            self.ca_bundle,
            self.timeout,
        )

    # ------------------------------------------------------------------
    # Protocol resolution
    # ------------------------------------------------------------------

    def _resolve_protocol_sync(self, client: httpx.Client) -> Literal["openai", "v2"]:
        if self.protocol != "auto":
            return self.protocol  # type: ignore[return-value]
        if self._resolved_protocol is not None:
            return self._resolved_protocol
        use_openai = probe_openai_compat(client, self.max_retries)
        resolved: Literal["openai", "v2"] = "openai" if use_openai else "v2"
        object.__setattr__(self, "_resolved_protocol", resolved)
        logger.info("KServe auto-detected protocol: %s", resolved)
        return resolved

    async def _resolve_protocol_async(
        self, client: httpx.AsyncClient
    ) -> Literal["openai", "v2"]:
        if self.protocol != "auto":
            return self.protocol  # type: ignore[return-value]
        if self._resolved_protocol is not None:
            return self._resolved_protocol
        use_openai = await async_probe_openai_compat(client, self.max_retries)
        resolved: Literal["openai", "v2"] = "openai" if use_openai else "v2"
        object.__setattr__(self, "_resolved_protocol", resolved)
        logger.info("KServe async auto-detected protocol: %s", resolved)
        return resolved

    # ------------------------------------------------------------------
    # Tool binding
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[Any], BaseTool]],
        **kwargs: Any,
    ) -> "ChatKServe":
        """Return a copy of this model with tools bound for OpenAI function calling.

        Args:
            tools: Tools to bind. Each tool can be a :class:`~langchain_core.tools.BaseTool`,
                a Pydantic model, or an OpenAI-schema dict.
            **kwargs: Extra kwargs merged into the request body on each call.

        Returns:
            A new :class:`ChatKServe` instance with ``_tools`` set.
        """
        formatted_tools = [_format_tool(t) for t in tools]
        clone = self.model_copy(deep=True)
        object.__setattr__(clone, "_tools", formatted_tools)
        return clone

    # ------------------------------------------------------------------
    # Core generation — sync
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response synchronously.

        Args:
            messages: Conversation messages.
            stop: Optional stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Extra parameters forwarded to the request body.

        Returns:
            :class:`~langchain_core.outputs.ChatResult`.
        """
        with self._make_sync_client() as client:
            proto = self._resolve_protocol_sync(client)
            effective_stop = stop or self.stop

            if proto == "openai":
                body = build_chat_request(
                    model_name=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stop=effective_stop,
                    stream=False,
                    tools=self._tools,
                    extra_kwargs=kwargs or None,
                )
                response = request_with_retry(
                    client, "POST", "/v1/chat/completions", self.max_retries, json=body
                )
                return parse_chat_response(response, self.model_name, "openai")
            else:
                prompt = messages_to_prompt(messages)
                body_v2 = build_v2_infer_request(
                    model_name=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stop=effective_stop,
                    extra_kwargs=kwargs or None,
                )
                response = request_with_retry(
                    client,
                    "POST",
                    infer_path(self.model_name),
                    self.max_retries,
                    json=body_v2,
                )
                return parse_v2_chat_response(response, self.model_name)

    # ------------------------------------------------------------------
    # Core generation — async
    # ------------------------------------------------------------------

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response asynchronously.

        Args:
            messages: Conversation messages.
            stop: Optional stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Extra parameters forwarded to the request body.

        Returns:
            :class:`~langchain_core.outputs.ChatResult`.
        """
        async with self._make_async_client() as client:
            proto = await self._resolve_protocol_async(client)
            effective_stop = stop or self.stop

            if proto == "openai":
                body = build_chat_request(
                    model_name=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stop=effective_stop,
                    stream=False,
                    tools=self._tools,
                    extra_kwargs=kwargs or None,
                )
                response = await async_request_with_retry(
                    client, "POST", "/v1/chat/completions", self.max_retries, json=body
                )
                return parse_chat_response(response, self.model_name, "openai")
            else:
                prompt = messages_to_prompt(messages)
                body_v2 = build_v2_infer_request(
                    model_name=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stop=effective_stop,
                    extra_kwargs=kwargs or None,
                )
                response = await async_request_with_retry(
                    client,
                    "POST",
                    infer_path(self.model_name),
                    self.max_retries,
                    json=body_v2,
                )
                return parse_v2_chat_response(response, self.model_name)

    # ------------------------------------------------------------------
    # Streaming — sync
    # ------------------------------------------------------------------

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream a chat response synchronously.

        Args:
            messages: Conversation messages.
            stop: Optional stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Extra parameters.

        Yields:
            :class:`~langchain_core.outputs.ChatGenerationChunk`.
        """
        with self._make_sync_client() as client:
            proto = self._resolve_protocol_sync(client)
            effective_stop = stop or self.stop

            if proto == "openai":
                body = build_chat_request(
                    model_name=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stop=effective_stop,
                    stream=True,
                    tools=self._tools,
                    extra_kwargs=kwargs or None,
                )
                for chunk in stream_chat_response(
                    client,
                    "/v1/chat/completions",
                    body,
                    self.max_retries,
                    self.model_name,
                    "openai",
                ):
                    if run_manager and chunk.message.content:
                        run_manager.on_llm_new_token(
                            str(chunk.message.content), chunk=chunk
                        )
                    yield chunk
            else:
                logger.warning(
                    "V2 protocol streaming may not be supported by all runtimes; "
                    "falling back to V2 chunked transfer."
                )
                prompt = messages_to_prompt(messages)
                body_v2 = build_v2_infer_request(
                    model_name=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stop=effective_stop,
                    extra_kwargs=kwargs or None,
                )
                for chunk in stream_v2_chat_response(
                    client, infer_path(self.model_name), body_v2, self.model_name
                ):
                    if run_manager and chunk.message.content:
                        run_manager.on_llm_new_token(
                            str(chunk.message.content), chunk=chunk
                        )
                    yield chunk

    # ------------------------------------------------------------------
    # Streaming — async
    # ------------------------------------------------------------------

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream a chat response asynchronously.

        Args:
            messages: Conversation messages.
            stop: Optional stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Extra parameters.

        Yields:
            :class:`~langchain_core.outputs.ChatGenerationChunk`.
        """
        async with self._make_async_client() as client:
            proto = await self._resolve_protocol_async(client)
            effective_stop = stop or self.stop

            if proto == "openai":
                body = build_chat_request(
                    model_name=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stop=effective_stop,
                    stream=True,
                    tools=self._tools,
                    extra_kwargs=kwargs or None,
                )
                async for chunk in astream_chat_response(
                    client,
                    "/v1/chat/completions",
                    body,
                    self.model_name,
                    "openai",
                ):
                    if run_manager and chunk.message.content:
                        await run_manager.on_llm_new_token(
                            str(chunk.message.content), chunk=chunk
                        )
                    yield chunk
            else:
                logger.warning(
                    "V2 protocol streaming may not be supported by all runtimes."
                )
                prompt = messages_to_prompt(messages)
                body_v2 = build_v2_infer_request(
                    model_name=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stop=effective_stop,
                    extra_kwargs=kwargs or None,
                )
                async for chunk in astream_v2_chat_response(
                    client, infer_path(self.model_name), body_v2, self.model_name
                ):
                    if run_manager and chunk.message.content:
                        await run_manager.on_llm_new_token(
                            str(chunk.message.content), chunk=chunk
                        )
                    yield chunk

    # ------------------------------------------------------------------
    # Identifying params
    # ------------------------------------------------------------------

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "protocol": self.protocol,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# ---------------------------------------------------------------------------
# Tool formatting helper
# ---------------------------------------------------------------------------


def _format_tool(tool: Any) -> Dict[str, Any]:
    """Convert a LangChain tool, Pydantic model, or dict into OpenAI tool schema.

    Args:
        tool: A :class:`~langchain_core.tools.BaseTool`, a Pydantic model class,
            or an already-formatted dict.

    Returns:
        OpenAI-compatible tool schema dict.
    """
    if isinstance(tool, dict):
        return tool
    if hasattr(tool, "as_tool"):
        # Pydantic model with as_tool helper
        return tool.as_tool()  # type: ignore[union-attr]
    if hasattr(tool, "name") and hasattr(tool, "description"):
        # BaseTool
        schema = getattr(tool, "args_schema", None)
        parameters: Dict[str, Any] = {"type": "object", "properties": {}}
        if schema is not None and hasattr(schema, "model_json_schema"):
            parameters = schema.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": tool.name,  # type: ignore[union-attr]
                "description": tool.description,  # type: ignore[union-attr]
                "parameters": parameters,
            },
        }
    # Last resort
    return {"type": "function", "function": str(tool)}
