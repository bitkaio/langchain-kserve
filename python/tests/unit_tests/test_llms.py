"""Unit tests for KServeLLM."""

from __future__ import annotations

import json
from typing import Any, Dict

import httpx
import pytest
import respx

from langchain_kserve import KServeLLM
from langchain_kserve._common import KServeAuthenticationError, KServeInferenceError

BASE_URL = "http://test-kserve.default.svc.cluster.local"
MODEL = "llama-3-8b"


def make_llm(**kwargs: Any) -> KServeLLM:
    defaults: Dict[str, Any] = {
        "base_url": BASE_URL,
        "model_name": MODEL,
        "protocol": "openai",
        "max_retries": 0,
    }
    defaults.update(kwargs)
    return KServeLLM(**defaults)


class TestKServeLLMOpenAI:
    @respx.mock
    def test_generate_basic(self) -> None:
        respx.post(f"{BASE_URL}/v1/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"text": "Once upon a time…", "finish_reason": "stop"}]
                },
            )
        )

        llm = make_llm()
        result = llm._generate(["Once upon a"])
        assert result.generations[0][0].text == "Once upon a time…"

    @respx.mock
    def test_generate_multiple_prompts(self) -> None:
        respx.post(f"{BASE_URL}/v1/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"text": "response", "finish_reason": "stop"}]},
            )
        )

        llm = make_llm()
        result = llm._generate(["prompt1", "prompt2"])
        assert len(result.generations) == 2

    @respx.mock
    def test_streaming_yields_chunks(self) -> None:
        sse_payload = (
            'data: {"choices": [{"text": "Hello", "finish_reason": null}]}\n\n'
            'data: {"choices": [{"text": " world", "finish_reason": null}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post(f"{BASE_URL}/v1/completions").mock(
            return_value=httpx.Response(200, text=sse_payload)
        )

        llm = make_llm()
        chunks = list(llm._stream("Hello"))
        texts = [c.text for c in chunks if c.text]
        assert texts == ["Hello", " world"]

    @respx.mock
    def test_stop_sequences_forwarded(self) -> None:
        route = respx.post(f"{BASE_URL}/v1/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"text": "partial", "finish_reason": "stop"}]},
            )
        )

        llm = make_llm()
        llm._generate(["prompt"], stop=["END"])

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["stop"] == ["END"]


class TestKServeLLMV2:
    @respx.mock
    def test_generate_v2(self) -> None:
        respx.post(f"{BASE_URL}/v2/models/{MODEL}/infer").mock(
            return_value=httpx.Response(
                200,
                json={
                    "outputs": [
                        {
                            "name": "text_output",
                            "data": ["The answer is 42."],
                        }
                    ]
                },
            )
        )

        llm = make_llm(protocol="v2")
        result = llm._generate(["What is the answer?"])
        assert result.generations[0][0].text == "The answer is 42."


class TestKServeLLMTokenUsage:
    @respx.mock
    def test_llm_generate_llm_output_token_usage(self) -> None:
        """LLMResult.llm_output['token_usage'] is populated when usage is in response."""
        respx.post(f"{BASE_URL}/v1/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"text": "Hello world", "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    },
                },
            )
        )

        llm = make_llm()
        result = llm._generate(["Say hello"])
        assert result.llm_output is not None
        assert result.llm_output["token_usage"]["prompt_tokens"] == 4
        assert result.llm_output["token_usage"]["completion_tokens"] == 2
        assert result.llm_output["token_usage"]["total_tokens"] == 6

    @respx.mock
    def test_llm_generate_no_usage_llm_output_none(self) -> None:
        """LLMResult.llm_output is None when no usage field in response."""
        respx.post(f"{BASE_URL}/v1/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"text": "Hello world", "finish_reason": "stop"}],
                },
            )
        )

        llm = make_llm()
        result = llm._generate(["Say hello"])
        assert result.llm_output is None


class TestKServeLLMErrors:
    @respx.mock
    def test_401_raises_auth_error(self) -> None:
        respx.post(f"{BASE_URL}/v1/completions").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )

        llm = make_llm()
        with pytest.raises(KServeAuthenticationError):
            llm._generate(["prompt"])

    @respx.mock
    def test_500_raises_inference_error(self) -> None:
        respx.post(f"{BASE_URL}/v1/completions").mock(
            return_value=httpx.Response(500, text="Server Error")
        )

        llm = make_llm()
        with pytest.raises(KServeInferenceError):
            llm._generate(["prompt"])
