"""ChatKServe — LangChain ChatModel for KServe-hosted chat/instruct models."""

from __future__ import annotations

import json
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
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from langchain_kserve._common import (
    KServeConnectionError,
    KServeError,
    KServeInferenceError,
    KServeModelInfo,
    async_request_with_retry,
    build_async_client,
    build_sync_client,
    fetch_model_info_openai,
    fetch_model_info_v2,
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
    logprobs: Optional[bool] = Field(default=None)
    top_logprobs: Optional[int] = Field(default=None)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None)
    parallel_tool_calls: Optional[bool] = Field(default=None)
    response_format: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "OpenAI response format constraint. E.g. ``{'type': 'json_object'}`` or "
            "``{'type': 'json_schema', 'json_schema': {'name': '...', 'schema': {...}}}``."
            " Only supported with the OpenAI-compatible protocol."
        ),
    )

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

    @field_validator("response_format", mode="before")
    @classmethod
    def _validate_response_format(
        cls, v: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Validate that json_schema response_format has a proper schema dict.

        Args:
            v: The response_format value to validate.

        Returns:
            The validated response_format dict, or ``None``.

        Raises:
            ValueError: If ``type == "json_schema"`` but the nested ``schema``
                field is missing or not a dict.
        """
        if v is None:
            return v
        if v.get("type") == "json_schema":
            json_schema_block = v.get("json_schema", {})
            schema = json_schema_block.get("schema")
            if not isinstance(schema, dict):
                raise ValueError(
                    "response_format with type='json_schema' must include a "
                    "'json_schema.schema' dict. Got: "
                    f"{type(schema).__name__!r}"
                )
        return v

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
        *,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "ChatKServe":
        """Return a copy of this model with tools bound for OpenAI function calling.

        Args:
            tools: Tools to bind. Each tool can be a :class:`~langchain_core.tools.BaseTool`,
                a Pydantic model, or an OpenAI-schema dict.
            tool_choice: Override the ``tool_choice`` field on the clone. If ``None``,
                the field value on ``self`` is preserved.
            **kwargs: Extra kwargs (reserved for future use).

        Returns:
            A new :class:`ChatKServe` instance with ``_tools`` set.
        """
        formatted_tools = [convert_to_openai_tool(t) for t in tools]
        update: Dict[str, Any] = {}
        if tool_choice is not None:
            update["tool_choice"] = tool_choice
        clone = self.model_copy(update=update, deep=True)
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

            if self.response_format is not None and proto == "v2":
                raise KServeError(
                    "Response format constraints are only supported with the "
                    "OpenAI-compatible protocol. Set protocol='openai'."
                )

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
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                    tool_choice=self.tool_choice,
                    parallel_tool_calls=self.parallel_tool_calls,
                    response_format=self.response_format,
                )
                response = request_with_retry(
                    client, "POST", "/v1/chat/completions", self.max_retries, json=body
                )
                return parse_chat_response(response, self.model_name, "openai")
            else:
                if self._tools:
                    raise KServeInferenceError(
                        "Tool calling is only supported with the OpenAI-compatible protocol. "
                        "Set protocol='openai' or use a runtime that exposes the OpenAI-compatible "
                        "API (e.g., vLLM)."
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

            if self.response_format is not None and proto == "v2":
                raise KServeError(
                    "Response format constraints are only supported with the "
                    "OpenAI-compatible protocol. Set protocol='openai'."
                )

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
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                    tool_choice=self.tool_choice,
                    parallel_tool_calls=self.parallel_tool_calls,
                    response_format=self.response_format,
                )
                response = await async_request_with_retry(
                    client, "POST", "/v1/chat/completions", self.max_retries, json=body
                )
                return parse_chat_response(response, self.model_name, "openai")
            else:
                if self._tools:
                    raise KServeInferenceError(
                        "Tool calling is only supported with the OpenAI-compatible protocol. "
                        "Set protocol='openai' or use a runtime that exposes the OpenAI-compatible "
                        "API (e.g., vLLM)."
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

            if self.response_format is not None and proto == "v2":
                raise KServeError(
                    "Response format constraints are only supported with the "
                    "OpenAI-compatible protocol. Set protocol='openai'."
                )

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
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                    tool_choice=self.tool_choice,
                    parallel_tool_calls=self.parallel_tool_calls,
                    response_format=self.response_format,
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

            if self.response_format is not None and proto == "v2":
                raise KServeError(
                    "Response format constraints are only supported with the "
                    "OpenAI-compatible protocol. Set protocol='openai'."
                )

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
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                    tool_choice=self.tool_choice,
                    parallel_tool_calls=self.parallel_tool_calls,
                    response_format=self.response_format,
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
    # Structured output
    # ------------------------------------------------------------------

    def with_structured_output(
        self,
        schema: Union[Type[Any], Dict[str, Any]],
        *,
        method: Literal["function_calling", "json_schema", "json_mode"] = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable:
        """Return a chain that produces structured output conforming to ``schema``.

        Args:
            schema: A Pydantic model class or a JSON-schema dict describing the
                desired output shape.
            method: Strategy to use for structured output.

                - ``"function_calling"`` (default): forces the model to call a
                  single tool whose schema mirrors ``schema``. Parses
                  ``tool_calls[0].args`` from the response.
                - ``"json_schema"``: sets ``response_format`` to
                  ``{"type": "json_schema", ...}`` and parses JSON from the
                  message content.
                - ``"json_mode"``: sets ``response_format`` to
                  ``{"type": "json_object"}`` and uses a ``JsonOutputParser``.
            include_raw: When ``True`` the chain output is a dict with keys
                ``"raw"`` (the original :class:`~langchain_core.messages.AIMessage`),
                ``"parsed"`` (the parsed result or ``None`` on error), and
                ``"parsing_error"`` (the exception or ``None``).
            strict: Whether to enforce strict JSON schema validation (only used
                for the ``"json_schema"`` method).
            **kwargs: Additional kwargs (reserved for future use).

        Returns:
            A :class:`~langchain_core.runnables.Runnable` whose output type
            depends on ``schema`` and ``include_raw``.
        """
        # ------------------------------------------------------------------
        # Derive a name and a JSON schema from the provided schema argument
        # ------------------------------------------------------------------
        is_pydantic = isinstance(schema, type) and issubclass(schema, BaseModel)

        if is_pydantic:
            pydantic_cls = schema  # type: ignore[assignment]
            name: str = pydantic_cls.__name__
            json_schema: Dict[str, Any] = pydantic_cls.model_json_schema()
        else:
            # schema is a dict
            pydantic_cls = None  # always define so closure cells are never empty
            schema_dict: Dict[str, Any] = schema  # type: ignore[assignment]
            name = schema_dict.get("title", "output_schema")
            json_schema = schema_dict

        # ------------------------------------------------------------------
        # Build the appropriate LLM chain variant
        # ------------------------------------------------------------------

        if method == "function_calling":
            tool_dict: Dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": json_schema.get("description", ""),
                    "parameters": json_schema,
                },
            }
            forced_choice: Dict[str, Any] = {
                "type": "function",
                "function": {"name": name},
            }
            llm_with_tool = self.bind_tools([tool_dict], tool_choice=forced_choice)

            def _parse_tool_call(ai_message: AIMessage) -> Any:
                if not ai_message.tool_calls:
                    raise ValueError(
                        f"Expected tool call for '{name}' but got none. "
                        f"Message content: {ai_message.content!r}"
                    )
                args = ai_message.tool_calls[0]["args"]
                if isinstance(args, str):
                    args = json.loads(args)
                if is_pydantic:
                    return pydantic_cls.model_validate(args)  # type: ignore[union-attr]
                return args

            output_parser: Runnable = RunnableLambda(_parse_tool_call)
            chain: Runnable = llm_with_tool | output_parser  # type: ignore[operator]

        elif method == "json_schema":
            rf: Dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": strict if strict is not None else True,
                    "schema": json_schema,
                },
            }
            llm_with_rf = self.bind(response_format=rf)

            def _parse_json_schema(ai_message: AIMessage) -> Any:
                content = ai_message.content
                if isinstance(content, list):
                    # multimodal — extract text
                    content = "".join(
                        b if isinstance(b, str) else b.get("text", "")
                        for b in content  # type: ignore[union-attr]
                        if isinstance(b, (str, dict))
                    )
                parsed = json.loads(str(content))
                if is_pydantic:
                    return pydantic_cls.model_validate(parsed)  # type: ignore[union-attr]
                return parsed

            output_parser = RunnableLambda(_parse_json_schema)
            chain = llm_with_rf | output_parser  # type: ignore[operator]

        elif method == "json_mode":
            llm_with_rf = self.bind(response_format={"type": "json_object"})
            output_parser = JsonOutputParser()
            chain = llm_with_rf | output_parser  # type: ignore[operator]

        else:
            raise ValueError(
                f"Unsupported method: {method!r}. "
                "Choose from 'function_calling', 'json_schema', 'json_mode'."
            )

        # ------------------------------------------------------------------
        # Optionally wrap with include_raw passthrough
        # ------------------------------------------------------------------
        if not include_raw:
            return chain

        def _make_raw_wrapper(
            llm: Runnable, parser: Runnable
        ) -> Runnable:
            def _run_with_raw(messages: Any) -> Dict[str, Any]:
                raw = llm.invoke(messages)
                try:
                    parsed: Any = parser.invoke(raw)
                    parsing_error: Optional[Exception] = None
                except Exception as exc:
                    parsed = None
                    parsing_error = exc
                return {"raw": raw, "parsed": parsed, "parsing_error": parsing_error}

            return RunnableLambda(_run_with_raw)

        if method == "function_calling":
            return _make_raw_wrapper(llm_with_tool, output_parser)
        else:
            return _make_raw_wrapper(llm_with_rf, output_parser)

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

    # ------------------------------------------------------------------
    # Model introspection
    # ------------------------------------------------------------------

    async def get_model_info(self) -> KServeModelInfo:
        """Fetch metadata about the served model.

        Resolves the protocol (if set to ``"auto"``) and queries the appropriate
        endpoint for model metadata.

        Returns:
            :class:`~langchain_kserve._common.KServeModelInfo` with model metadata.

        Raises:
            KServeModelNotFoundError: If the model is not found.
            KServeInferenceError: On other errors.
        """
        async with self._make_async_client() as client:
            proto = await self._resolve_protocol_async(client)
            if proto == "openai":
                return await fetch_model_info_openai(
                    client, self.model_name, self.max_retries
                )
            else:
                return await fetch_model_info_v2(
                    client, self.model_name, self.max_retries
                )


