"""KServeLLM — LangChain BaseLLM wrapper for KServe-hosted base completion models."""

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
    Tuple,
)

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils import get_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_kserve._common import (
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
    astream_completion_response,
    build_completion_request,
    parse_completion_response,
    probe_openai_compat,
    stream_completion_response,
)
from langchain_kserve._v2_protocol import (
    build_v2_infer_request,
    infer_path,
    parse_v2_completion_response,
)

logger = logging.getLogger(__name__)


class KServeLLM(BaseLLM):
    """LangChain ``BaseLLM`` wrapper for KServe-hosted base/completion models.

    Use this class for raw (non-chat) models.  For instruction-tuned or chat
    models prefer :class:`~langchain_kserve.ChatKServe`.

    Supports both the OpenAI-compatible ``/v1/completions`` endpoint and the
    V2 Inference Protocol ``/v2/models/{model}/infer`` endpoint.

    Example:
        .. code-block:: python

            from langchain_kserve import KServeLLM

            llm = KServeLLM(
                base_url="https://llama.default.svc.cluster.local",
                model_name="llama-3-8b",
                max_tokens=256,
            )
            text = llm.invoke("Once upon a time")

    Environment variables:
        ``KSERVE_BASE_URL``, ``KSERVE_API_KEY``, ``KSERVE_MODEL_NAME``,
        ``KSERVE_PROTOCOL``, ``KSERVE_CA_BUNDLE``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    base_url: str = Field(default="", description="Root URL of the KServe service.")
    model_name: str = Field(default="", description="Model identifier.")
    protocol: Literal["openai", "v2", "auto"] = Field(
        default="auto",
        description="Inference protocol ('openai', 'v2', or 'auto').",
    )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    api_key: Optional[SecretStr] = Field(default=None)
    token_provider: Optional[Callable[[], str]] = Field(default=None, exclude=True)

    # ------------------------------------------------------------------
    # TLS
    # ------------------------------------------------------------------
    verify_ssl: bool = Field(default=True)
    ca_bundle: Optional[str] = Field(default=None)

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

    # ------------------------------------------------------------------
    # Connection behaviour
    # ------------------------------------------------------------------
    timeout: int = Field(default=120, ge=1)
    max_retries: int = Field(default=3, ge=0)

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
            values["ca_bundle"] = (
                get_from_env("ca_bundle", "KSERVE_CA_BUNDLE", default="") or None
            )
        return values

    # ------------------------------------------------------------------
    # Required LangChain property
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "kserve-llm"

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
        resolved: Literal["openai", "v2"] = (
            "openai" if probe_openai_compat(client, self.max_retries) else "v2"
        )
        object.__setattr__(self, "_resolved_protocol", resolved)
        return resolved

    async def _resolve_protocol_async(
        self, client: httpx.AsyncClient
    ) -> Literal["openai", "v2"]:
        if self.protocol != "auto":
            return self.protocol  # type: ignore[return-value]
        if self._resolved_protocol is not None:
            return self._resolved_protocol
        resolved: Literal["openai", "v2"] = (
            "openai"
            if await async_probe_openai_compat(client, self.max_retries)
            else "v2"
        )
        object.__setattr__(self, "_resolved_protocol", resolved)
        return resolved

    # ------------------------------------------------------------------
    # BaseLLM required: _generate
    # ------------------------------------------------------------------

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate completions for a list of prompts synchronously.

        Args:
            prompts: List of text prompts.
            stop: Optional stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Extra parameters.

        Returns:
            :class:`~langchain_core.outputs.LLMResult`.
        """
        generations: List[List[Generation]] = []
        last_usage: Optional[Dict[str, Any]] = None
        with self._make_sync_client() as client:
            proto = self._resolve_protocol_sync(client)
            effective_stop = stop or self.stop
            for prompt in prompts:
                text, usage = self._call_single(
                    client, proto, prompt, effective_stop, **kwargs
                )
                if usage is not None:
                    last_usage = usage
                generations.append([Generation(text=text)])
        llm_output: Optional[Dict[str, Any]] = (
            {"token_usage": last_usage} if last_usage is not None else None
        )
        return LLMResult(generations=generations, llm_output=llm_output)

    def _call_single(
        self,
        client: httpx.Client,
        proto: Literal["openai", "v2"],
        prompt: str,
        stop: Optional[List[str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if proto == "openai":
            body = build_completion_request(
                model_name=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop=stop,
                stream=False,
                extra_kwargs=kwargs or None,
                logprobs=self.logprobs,
                top_logprobs=self.top_logprobs,
            )
            response = request_with_retry(
                client, "POST", "/v1/completions", self.max_retries, json=body
            )
            return parse_completion_response(response, self.model_name)
        else:
            body_v2 = build_v2_infer_request(
                model_name=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop=stop,
                extra_kwargs=kwargs or None,
            )
            response = request_with_retry(
                client,
                "POST",
                infer_path(self.model_name),
                self.max_retries,
                json=body_v2,
            )
            return parse_v2_completion_response(response), None

    # ------------------------------------------------------------------
    # Streaming — sync
    # ------------------------------------------------------------------

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream completions for a single prompt synchronously.

        Args:
            prompt: Text prompt.
            stop: Optional stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Extra parameters.

        Yields:
            :class:`~langchain_core.outputs.GenerationChunk`.
        """
        with self._make_sync_client() as client:
            proto = self._resolve_protocol_sync(client)
            effective_stop = stop or self.stop

            if proto == "openai":
                body = build_completion_request(
                    model_name=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stop=effective_stop,
                    stream=True,
                    extra_kwargs=kwargs or None,
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                )
                for text in stream_completion_response(
                    client, "/v1/completions", body, self.model_name
                ):
                    chunk = GenerationChunk(text=text)
                    if run_manager:
                        run_manager.on_llm_new_token(text, chunk=chunk)
                    yield chunk
            else:
                # V2 doesn't have native streaming; do a single request and yield one chunk
                text, _usage = self._call_single(client, "v2", prompt, effective_stop, **kwargs)
                chunk = GenerationChunk(text=text)
                if run_manager:
                    run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk

    # ------------------------------------------------------------------
    # Streaming — async
    # ------------------------------------------------------------------

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Stream completions for a single prompt asynchronously.

        Args:
            prompt: Text prompt.
            stop: Optional stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Extra parameters.

        Yields:
            :class:`~langchain_core.outputs.GenerationChunk`.
        """
        async with self._make_async_client() as client:
            proto = await self._resolve_protocol_async(client)
            effective_stop = stop or self.stop

            if proto == "openai":
                body = build_completion_request(
                    model_name=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stop=effective_stop,
                    stream=True,
                    extra_kwargs=kwargs or None,
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                )
                async for text in astream_completion_response(
                    client, "/v1/completions", body, self.model_name
                ):
                    chunk = GenerationChunk(text=text)
                    if run_manager:
                        await run_manager.on_llm_new_token(text, chunk=chunk)
                    yield chunk
            else:
                response = await async_request_with_retry(
                    client,
                    "POST",
                    infer_path(self.model_name),
                    self.max_retries,
                    json=build_v2_infer_request(
                        model_name=self.model_name,
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stop=effective_stop,
                        extra_kwargs=kwargs or None,
                    ),
                )
                text = parse_v2_completion_response(response)
                chunk = GenerationChunk(text=text)
                if run_manager:
                    await run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk

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
