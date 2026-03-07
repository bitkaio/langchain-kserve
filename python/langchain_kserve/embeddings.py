"""KServeEmbeddings — LangChain Embeddings for KServe-hosted embedding models."""

from __future__ import annotations

import base64
import struct
from typing import Any, Callable, Dict, List, Literal, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from langchain_kserve._common import (
    KServeError,
    async_request_with_retry,
    build_async_client,
    build_sync_client,
    request_with_retry,
)


class KServeEmbeddings(BaseModel, Embeddings):
    """Embeddings wrapper for KServe-hosted embedding models.

    Uses the OpenAI-compatible /v1/embeddings endpoint exposed by vLLM and other
    KServe runtimes.

    Example:
        .. code-block:: python

            from langchain_kserve import KServeEmbeddings

            embeddings = KServeEmbeddings(
                base_url="https://my-embedding-model.cluster.example.com",
                model_name="Qwen/Qwen3-Embedding-0.6B",
            )
            vectors = embeddings.embed_documents(["Hello world", "How are you?"])

    Environment variables:
        ``KSERVE_EMBEDDINGS_BASE_URL`` (or ``KSERVE_BASE_URL``),
        ``KSERVE_EMBEDDINGS_MODEL_NAME``, ``KSERVE_API_KEY``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str = Field(default="", description="Root URL of the KServe embeddings service.")
    model_name: str = Field(default="", description="Embedding model identifier.")

    # Auth (same as ChatKServe)
    api_key: Optional[SecretStr] = Field(default=None, description="Static bearer token.")
    token_provider: Optional[Callable[[], str]] = Field(
        default=None, description="Dynamic token provider.", exclude=True
    )

    # TLS
    verify_ssl: bool = Field(default=True)
    ca_bundle: Optional[str] = Field(default=None)

    # Embedding params
    dimensions: Optional[int] = Field(default=None, description="Output embedding dimensions (Matryoshka).")
    encoding_format: Literal["float", "base64"] = Field(default="float")

    # Connection
    timeout: int = Field(default=120, ge=1)
    max_retries: int = Field(default=3, ge=0)

    # Batching
    chunk_size: int = Field(default=1000, ge=1, description="Max texts per API call.")

    @model_validator(mode="before")
    @classmethod
    def _populate_from_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Populate fields from environment variables if not explicitly set.

        Args:
            values: Raw field values before Pydantic validation.

        Returns:
            Updated field values dict with any env-sourced values filled in.
        """
        if not values.get("base_url"):
            values["base_url"] = (
                get_from_env("base_url", "KSERVE_EMBEDDINGS_BASE_URL", default="")
                or get_from_env("base_url", "KSERVE_BASE_URL", default="")
            )
        if not values.get("model_name"):
            values["model_name"] = get_from_env(
                "model_name", "KSERVE_EMBEDDINGS_MODEL_NAME", default=""
            )
        if not values.get("api_key"):
            env_key = get_from_env("api_key", "KSERVE_API_KEY", default="")
            if env_key:
                values["api_key"] = SecretStr(env_key)
        if not values.get("ca_bundle"):
            values["ca_bundle"] = get_from_env("ca_bundle", "KSERVE_CA_BUNDLE", default="") or None
        return values

    def _build_request_body(self, texts: List[str]) -> Dict[str, Any]:
        """Build the JSON request body for ``/v1/embeddings``.

        Args:
            texts: List of input strings to embed.

        Returns:
            JSON-serialisable request body dict.
        """
        body: Dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": self.encoding_format,
        }
        if self.dimensions is not None:
            body["dimensions"] = self.dimensions
        return body

    def _parse_response(self, data: Dict[str, Any]) -> List[List[float]]:
        """Parse the ``/v1/embeddings`` response body into a list of float vectors.

        Handles both ``"float"`` and ``"base64"`` encoding formats.  Items are
        returned in the order given by each item's ``index`` field.

        Args:
            data: Parsed JSON response body.

        Returns:
            Ordered list of embedding vectors (one per input text).

        Raises:
            KServeError: If the response does not contain a ``data`` list.
        """
        items = sorted(data["data"], key=lambda x: x["index"])
        results: List[List[float]] = []
        for item in items:
            embedding = item["embedding"]
            if self.encoding_format == "base64" and isinstance(embedding, str):
                raw = base64.b64decode(embedding)
                n = len(raw) // 4
                embedding = list(struct.unpack(f"{n}f", raw))
            results.append(embedding)
        return results

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents, batching if needed.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors, one per input document.

        Raises:
            KServeError: On connection or API errors.
        """
        results: List[List[float]] = []
        for i in range(0, len(texts), self.chunk_size):
            batch = texts[i : i + self.chunk_size]
            with build_sync_client(
                self.base_url, self.api_key, self.token_provider,
                self.verify_ssl, self.ca_bundle, self.timeout
            ) as client:
                response = request_with_retry(
                    client, "POST", "/v1/embeddings", self.max_retries,
                    json=self._build_request_body(batch)
                )
                results.extend(self._parse_response(response.json()))
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Args:
            text: The query string to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of documents, batching if needed.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors, one per input document.

        Raises:
            KServeError: On connection or API errors.
        """
        import asyncio

        tasks = []
        for i in range(0, len(texts), self.chunk_size):
            batch = texts[i : i + self.chunk_size]
            tasks.append(self._aembed_batch(batch))
        batches = await asyncio.gather(*tasks)
        results: List[List[float]] = []
        for batch_result in batches:
            results.extend(batch_result)
        return results

    async def _aembed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a single batch of texts asynchronously.

        Args:
            texts: Batch of input strings (at most ``chunk_size`` items).

        Returns:
            List of embedding vectors for the batch.
        """
        async with build_async_client(
            self.base_url, self.api_key, self.token_provider,
            self.verify_ssl, self.ca_bundle, self.timeout
        ) as client:
            response = await async_request_with_retry(
                client, "POST", "/v1/embeddings", self.max_retries,
                json=self._build_request_body(texts)
            )
            return self._parse_response(response.json())

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query string.

        Args:
            text: The query string to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        results = await self.aembed_documents([text])
        return results[0]
