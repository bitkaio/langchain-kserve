"""Unit tests for KServeEmbeddings."""

from __future__ import annotations

import base64
import json
import struct
from typing import Any, Dict, List

import httpx
import pytest
import respx

from langchain_kserve import KServeEmbeddings
from langchain_kserve._common import (
    KServeAuthenticationError,
    KServeInferenceError,
    KServeModelNotFoundError,
)

BASE_URL = "http://embedding-model.default.svc.cluster.local"
MODEL = "Qwen/Qwen3-Embedding-0.6B"


def make_embeddings(**kwargs: Any) -> KServeEmbeddings:
    """Build a KServeEmbeddings instance with test defaults."""
    defaults: Dict[str, Any] = {
        "base_url": BASE_URL,
        "model_name": MODEL,
        "max_retries": 0,
    }
    defaults.update(kwargs)
    return KServeEmbeddings(**defaults)


def _make_embedding_response(
    vectors: List[List[float]],
    model: str = MODEL,
) -> Dict[str, Any]:
    """Build a synthetic /v1/embeddings JSON response."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": vec, "index": i}
            for i, vec in enumerate(vectors)
        ],
        "model": model,
        "usage": {"prompt_tokens": len(vectors), "total_tokens": len(vectors)},
    }


# ---------------------------------------------------------------------------
# Basic embed_documents
# ---------------------------------------------------------------------------


class TestEmbedDocuments:
    @respx.mock
    def test_returns_correct_vectors(self) -> None:
        """embed_documents returns the vectors returned by the API, in order."""
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(200, json=_make_embedding_response(vectors))
        )

        emb = make_embeddings()
        result = emb.embed_documents(["hello", "world"])

        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2, 0.3])
        assert result[1] == pytest.approx([0.4, 0.5, 0.6])

    @respx.mock
    def test_order_preserved_when_api_shuffles_index(self) -> None:
        """Items are sorted by 'index', not by insertion order in the response."""
        # Swap the order in the response JSON
        shuffled_response = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.4, 0.5], "index": 1},
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
            ],
            "model": MODEL,
        }
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(200, json=shuffled_response)
        )

        emb = make_embeddings()
        result = emb.embed_documents(["first", "second"])

        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.4, 0.5])

    @respx.mock
    def test_request_body_contains_model_and_input(self) -> None:
        """The API request body includes model, input, and encoding_format."""
        route = respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(
                200, json=_make_embedding_response([[0.1, 0.2]])
            )
        )

        emb = make_embeddings()
        emb.embed_documents(["test text"])

        sent = json.loads(route.calls[0].request.content)
        assert sent["model"] == MODEL
        assert sent["input"] == ["test text"]
        assert sent["encoding_format"] == "float"

    def test_empty_input_returns_empty_list(self) -> None:
        """embed_documents([]) makes no HTTP calls and returns []."""
        emb = make_embeddings()
        result = emb.embed_documents([])
        assert result == []

    @respx.mock
    def test_dimensions_included_when_set(self) -> None:
        """The 'dimensions' field is sent when configured."""
        route = respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(
                200, json=_make_embedding_response([[0.1, 0.2]])
            )
        )

        emb = make_embeddings(dimensions=512)
        emb.embed_documents(["hi"])

        sent = json.loads(route.calls[0].request.content)
        assert sent["dimensions"] == 512

    @respx.mock
    def test_dimensions_omitted_when_none(self) -> None:
        """The 'dimensions' key is absent when dimensions=None."""
        route = respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(
                200, json=_make_embedding_response([[0.1, 0.2]])
            )
        )

        emb = make_embeddings()  # dimensions=None by default
        emb.embed_documents(["hi"])

        sent = json.loads(route.calls[0].request.content)
        assert "dimensions" not in sent


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    @respx.mock
    def test_multiple_requests_when_texts_exceed_chunk_size(self) -> None:
        """When len(texts) > chunk_size, multiple API calls are made."""
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            side_effect=[
                httpx.Response(200, json=_make_embedding_response([[0.1, 0.2]])),
                httpx.Response(200, json=_make_embedding_response([[0.3, 0.4]])),
            ]
        )

        emb = make_embeddings(chunk_size=1)
        result = emb.embed_documents(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.3, 0.4])

    @respx.mock
    def test_results_reassembled_in_original_order_across_batches(self) -> None:
        """Results from multiple batches are concatenated preserving input order."""
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_make_embedding_response([[1.0, 0.0], [0.0, 1.0]]),
                ),
                httpx.Response(
                    200,
                    json=_make_embedding_response([[0.5, 0.5], [0.9, 0.1]]),
                ),
            ]
        )

        emb = make_embeddings(chunk_size=2)
        texts = ["a", "b", "c", "d"]
        result = emb.embed_documents(texts)

        assert len(result) == 4
        assert result[0] == pytest.approx([1.0, 0.0])
        assert result[1] == pytest.approx([0.0, 1.0])
        assert result[2] == pytest.approx([0.5, 0.5])
        assert result[3] == pytest.approx([0.9, 0.1])

    @respx.mock
    def test_single_request_when_texts_equal_chunk_size(self) -> None:
        """Exactly chunk_size texts produce exactly one API call."""
        route = respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(
                200, json=_make_embedding_response([[0.1, 0.2], [0.3, 0.4]])
            )
        )

        emb = make_embeddings(chunk_size=2)
        emb.embed_documents(["x", "y"])

        assert route.call_count == 1


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    @respx.mock
    def test_embed_query_returns_single_vector(self) -> None:
        """embed_query returns a single embedding vector (not a list of lists)."""
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(
                200, json=_make_embedding_response([[0.7, 0.8, 0.9]])
            )
        )

        emb = make_embeddings()
        result = emb.embed_query("what is the capital?")

        assert isinstance(result, list)
        assert len(result) == 3
        assert result == pytest.approx([0.7, 0.8, 0.9])

    @respx.mock
    def test_embed_query_delegates_to_embed_documents(self) -> None:
        """embed_query sends input as a one-element list."""
        route = respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(
                200, json=_make_embedding_response([[0.1, 0.2]])
            )
        )

        emb = make_embeddings()
        emb.embed_query("my query")

        sent = json.loads(route.calls[0].request.content)
        assert sent["input"] == ["my query"]


# ---------------------------------------------------------------------------
# Base64 decoding
# ---------------------------------------------------------------------------


class TestBase64Encoding:
    @respx.mock
    def test_base64_encoded_embedding_decoded_correctly(self) -> None:
        """base64 encoding format is decoded from raw bytes to List[float]."""
        floats = [0.1, 0.2, 0.3, 0.4]
        raw = struct.pack(f"{len(floats)}f", *floats)
        encoded = base64.b64encode(raw).decode("utf-8")

        response_data = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": encoded, "index": 0}],
            "model": MODEL,
        }
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(200, json=response_data)
        )

        emb = make_embeddings(encoding_format="base64")
        result = emb.embed_documents(["hello"])

        assert len(result) == 1
        assert result[0] == pytest.approx(floats, abs=1e-5)

    @respx.mock
    def test_base64_encoding_format_sent_in_request(self) -> None:
        """encoding_format='base64' is passed to the API."""
        floats = [0.5, 0.6]
        raw = struct.pack(f"{len(floats)}f", *floats)
        encoded = base64.b64encode(raw).decode("utf-8")

        route = respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"object": "embedding", "embedding": encoded, "index": 0}],
                    "model": MODEL,
                },
            )
        )

        emb = make_embeddings(encoding_format="base64")
        emb.embed_documents(["hi"])

        sent = json.loads(route.calls[0].request.content)
        assert sent["encoding_format"] == "base64"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @respx.mock
    def test_401_raises_authentication_error(self) -> None:
        """A 401 response raises KServeAuthenticationError."""
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )

        emb = make_embeddings()
        with pytest.raises(KServeAuthenticationError):
            emb.embed_documents(["hi"])

    @respx.mock
    def test_404_raises_model_not_found(self) -> None:
        """A 404 response raises KServeModelNotFoundError."""
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(404, text="Not Found")
        )

        emb = make_embeddings()
        with pytest.raises(KServeModelNotFoundError):
            emb.embed_documents(["hi"])

    @respx.mock
    def test_500_raises_inference_error(self) -> None:
        """A 500 response raises KServeInferenceError."""
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        emb = make_embeddings()
        with pytest.raises(KServeInferenceError):
            emb.embed_documents(["hi"])


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


class TestAsyncEmbeddings:
    @respx.mock
    async def test_aembed_documents_returns_correct_vectors(self) -> None:
        """aembed_documents returns the same vectors as embed_documents."""
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(200, json=_make_embedding_response(vectors))
        )

        emb = make_embeddings()
        result = await emb.aembed_documents(["first", "second"])

        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.3, 0.4])

    @respx.mock
    async def test_aembed_documents_batches_in_parallel(self) -> None:
        """aembed_documents splits into batches and returns results in order."""
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            side_effect=[
                httpx.Response(200, json=_make_embedding_response([[0.1, 0.2]])),
                httpx.Response(200, json=_make_embedding_response([[0.3, 0.4]])),
            ]
        )

        emb = make_embeddings(chunk_size=1)
        result = await emb.aembed_documents(["a", "b"])

        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.3, 0.4])

    @respx.mock
    async def test_aembed_query_returns_single_vector(self) -> None:
        """aembed_query returns a flat list of floats."""
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(
                200, json=_make_embedding_response([[0.5, 0.6, 0.7]])
            )
        )

        emb = make_embeddings()
        result = await emb.aembed_query("async query")

        assert isinstance(result, list)
        assert result == pytest.approx([0.5, 0.6, 0.7])

    @respx.mock
    async def test_aembed_query_sends_single_element_input(self) -> None:
        """aembed_query sends input as a one-element list."""
        route = respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(
                200, json=_make_embedding_response([[0.1, 0.2]])
            )
        )

        emb = make_embeddings()
        await emb.aembed_query("my async query")

        sent = json.loads(route.calls[0].request.content)
        assert sent["input"] == ["my async query"]

    @respx.mock
    async def test_aembed_documents_auth_error(self) -> None:
        """Async 401 response raises KServeAuthenticationError."""
        respx.post(f"{BASE_URL}/v1/embeddings").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )

        emb = make_embeddings()
        with pytest.raises(KServeAuthenticationError):
            await emb.aembed_documents(["hi"])
