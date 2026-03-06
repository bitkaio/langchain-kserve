"""Unit tests for ChatKServe."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_kserve import ChatKServe, KServeModelInfo
from langchain_kserve._common import (
    KServeAuthenticationError,
    KServeInferenceError,
    KServeModelNotFoundError,
)

BASE_URL = "http://test-kserve.default.svc.cluster.local"
MODEL = "qwen2.5-coder-7b-instruct"


def make_llm(**kwargs: Any) -> ChatKServe:
    defaults: Dict[str, Any] = {
        "base_url": BASE_URL,
        "model_name": MODEL,
        "protocol": "openai",
        "max_retries": 0,
    }
    defaults.update(kwargs)
    return ChatKServe(**defaults)


# ---------------------------------------------------------------------------
# OpenAI-compatible protocol tests
# ---------------------------------------------------------------------------


class TestChatKServeOpenAI:
    @respx.mock
    def test_generate_basic(self) -> None:
        """Basic non-streaming generate with OpenAI-compat endpoint."""
        respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Hello world!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    },
                },
            )
        )

        llm = make_llm()
        result = llm._generate([HumanMessage(content="Hi")])

        assert len(result.generations) == 1
        assert isinstance(result.generations[0].message, AIMessage)
        assert result.generations[0].message.content == "Hello world!"
        assert result.generations[0].generation_info["protocol"] == "openai"
        assert result.generations[0].generation_info["prompt_tokens"] == 10

    @respx.mock
    def test_generate_with_system_message(self) -> None:
        """System message is serialised correctly."""
        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Sure!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm()
        llm._generate(
            [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Tell me a joke."),
            ]
        )

        sent_body = json.loads(route.calls[0].request.content)
        roles = [m["role"] for m in sent_body["messages"]]
        assert roles == ["system", "user"]

    @respx.mock
    def test_generate_with_stop_sequences(self) -> None:
        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "partial"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm()
        llm._generate([HumanMessage(content="Continue")], stop=["END", "\n\n"])

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["stop"] == ["END", "\n\n"]

    @respx.mock
    def test_generate_with_tool_calls(self) -> None:
        """Tool calls in the response are parsed into AIMessage.tool_calls."""
        respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city": "Berlin"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm()
        result = llm._generate([HumanMessage(content="Weather in Berlin?")])
        ai_msg = result.generations[0].message
        assert isinstance(ai_msg, AIMessage)
        assert len(ai_msg.tool_calls) == 1
        assert ai_msg.tool_calls[0]["name"] == "get_weather"
        assert ai_msg.tool_calls[0]["args"] == {"city": "Berlin"}

    @respx.mock
    def test_bind_tools_includes_tools_in_request(self) -> None:
        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        tool_schema = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }

        llm = make_llm().bind_tools([tool_schema])
        llm._generate([HumanMessage(content="Search for Python")])

        sent_body = json.loads(route.calls[0].request.content)
        assert "tools" in sent_body
        assert sent_body["tools"][0]["function"]["name"] == "search"

    @respx.mock
    def test_streaming_yields_chunks(self) -> None:
        sse_payload = (
            "data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}, \"finish_reason\": null}]}\n\n"
            "data: {\"choices\": [{\"delta\": {\"content\": \" world\"}, \"finish_reason\": null}]}\n\n"
            "data: [DONE]\n\n"
        )
        respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=sse_payload)
        )

        llm = make_llm()
        chunks = list(llm._stream([HumanMessage(content="Hi")]))
        contents = [c.message.content for c in chunks if c.message.content]
        assert contents == ["Hello", " world"]


# ---------------------------------------------------------------------------
# V2 protocol tests
# ---------------------------------------------------------------------------


class TestChatKServeV2:
    @respx.mock
    def test_generate_v2(self) -> None:
        respx.post(f"{BASE_URL}/v2/models/{MODEL}/infer").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "req-1",
                    "outputs": [
                        {
                            "name": "text_output",
                            "shape": [1],
                            "datatype": "BYTES",
                            "data": ["The answer is 42."],
                        }
                    ],
                },
            )
        )

        llm = make_llm(protocol="v2")
        result = llm._generate([HumanMessage(content="What is the answer?")])

        assert result.generations[0].message.content == "The answer is 42."
        assert result.generations[0].generation_info["protocol"] == "v2"

    @respx.mock
    def test_v2_request_contains_chatml_prompt(self) -> None:
        route = respx.post(f"{BASE_URL}/v2/models/{MODEL}/infer").mock(
            return_value=httpx.Response(
                200,
                json={
                    "outputs": [{"data": ["response"]}],
                },
            )
        )

        llm = make_llm(protocol="v2")
        llm._generate(
            [
                SystemMessage(content="Be helpful."),
                HumanMessage(content="Hello"),
            ]
        )

        sent_body = json.loads(route.calls[0].request.content)
        prompt: str = sent_body["inputs"][0]["data"][0]
        assert "<|im_start|>system" in prompt
        assert "Be helpful." in prompt
        assert "<|im_start|>user" in prompt
        assert "Hello" in prompt

    @respx.mock
    def test_v2_parameters_forwarded(self) -> None:
        route = respx.post(f"{BASE_URL}/v2/models/{MODEL}/infer").mock(
            return_value=httpx.Response(
                200,
                json={"outputs": [{"data": ["ok"]}]},
            )
        )

        llm = make_llm(protocol="v2", temperature=0.3, max_tokens=128)
        llm._generate([HumanMessage(content="test")])

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["parameters"]["temperature"] == 0.3
        assert sent_body["parameters"]["max_tokens"] == 128


# ---------------------------------------------------------------------------
# Auto-detection tests
# ---------------------------------------------------------------------------


class TestProtocolAutoDetect:
    @respx.mock
    def test_auto_detects_openai(self) -> None:
        respx.get(f"{BASE_URL}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm(protocol="auto")
        result = llm._generate([HumanMessage(content="Hello")])
        assert result.generations[0].generation_info["protocol"] == "openai"

    @respx.mock
    def test_auto_falls_back_to_v2(self) -> None:
        respx.get(f"{BASE_URL}/v1/models").mock(
            return_value=httpx.Response(404)
        )
        respx.post(f"{BASE_URL}/v2/models/{MODEL}/infer").mock(
            return_value=httpx.Response(
                200,
                json={"outputs": [{"data": ["v2 response"]}]},
            )
        )

        llm = make_llm(protocol="auto")
        result = llm._generate([HumanMessage(content="Hello")])
        assert result.generations[0].generation_info["protocol"] == "v2"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestChatKServeErrors:
    @respx.mock
    def test_401_raises_auth_error(self) -> None:
        respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )

        llm = make_llm()
        with pytest.raises(KServeAuthenticationError):
            llm._generate([HumanMessage(content="Hi")])

    @respx.mock
    def test_404_raises_model_not_found(self) -> None:
        respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(404, text="Not Found")
        )

        llm = make_llm()
        with pytest.raises(KServeModelNotFoundError):
            llm._generate([HumanMessage(content="Hi")])

    @respx.mock
    def test_500_raises_inference_error(self) -> None:
        respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        llm = make_llm()
        with pytest.raises(KServeInferenceError):
            llm._generate([HumanMessage(content="Hi")])

    @respx.mock
    def test_empty_choices_raises_inference_error(self) -> None:
        respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )

        llm = make_llm()
        with pytest.raises(KServeInferenceError):
            llm._generate([HumanMessage(content="Hi")])


# ---------------------------------------------------------------------------
# Auth header tests
# ---------------------------------------------------------------------------


class TestChatKServeNewFeatures:
    @respx.mock
    def test_v2_with_tools_raises_error(self) -> None:
        """ChatKServe with v2 protocol and bound tools raises KServeInferenceError."""
        respx.post(f"{BASE_URL}/v2/models/{MODEL}/infer").mock(
            return_value=httpx.Response(
                200,
                json={"outputs": [{"data": ["ok"]}]},
            )
        )

        tool_schema = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        llm = make_llm(protocol="v2").bind_tools([tool_schema])
        with pytest.raises(KServeInferenceError, match="Tool calling is only supported"):
            llm._generate([HumanMessage(content="Search for something")])

    @respx.mock
    def test_get_model_info_openai(self) -> None:
        """get_model_info() with OpenAI protocol fetches /v1/models/{model}."""
        respx.get(f"{BASE_URL}/v1/models/{MODEL}").mock(
            return_value=httpx.Response(
                200,
                json={"id": MODEL, "object": "model", "created": 1234567890},
            )
        )

        llm = make_llm(protocol="openai")
        info = asyncio.run(llm.get_model_info())
        assert info["model_name"] == MODEL
        assert info["raw"]["id"] == MODEL

    @respx.mock
    def test_get_model_info_v2(self) -> None:
        """get_model_info() with V2 protocol fetches /v2/models/{model}."""
        respx.get(f"{BASE_URL}/v2/models/{MODEL}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": MODEL,
                    "version": "1",
                    "platform": "onnxruntime",
                    "inputs": [{"name": "text_input"}],
                    "outputs": [{"name": "text_output"}],
                },
            )
        )

        llm = make_llm(protocol="v2")
        info = asyncio.run(llm.get_model_info())
        assert info["model_name"] == MODEL
        assert info["model_version"] == "1"
        assert info["platform"] == "onnxruntime"
        assert info["inputs"] is not None
        assert info["outputs"] is not None


class TestChatKServeAuth:
    @respx.mock
    def test_static_api_key_sent_as_bearer(self) -> None:
        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm(api_key="my-secret-token")
        llm._generate([HumanMessage(content="Hi")])

        auth_header = route.calls[0].request.headers.get("Authorization")
        assert auth_header == "Bearer my-secret-token"

    @respx.mock
    def test_token_provider_called_per_request(self) -> None:
        call_count = {"n": 0}

        def provider() -> str:
            call_count["n"] += 1
            return f"dynamic-token-{call_count['n']}"

        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm(token_provider=provider)
        llm._generate([HumanMessage(content="Hi")])

        auth_header = route.calls[0].request.headers.get("Authorization")
        assert auth_header == "Bearer dynamic-token-1"
