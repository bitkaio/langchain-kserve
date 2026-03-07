"""Unit tests for ChatKServe."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from pydantic import BaseModel

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_kserve import ChatKServe, KServeModelInfo
from langchain_kserve._common import (
    KServeAuthenticationError,
    KServeError,
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


# ---------------------------------------------------------------------------
# Feature 1: JSON Mode / Response Format
# ---------------------------------------------------------------------------


class TestResponseFormat:
    @respx.mock
    def test_response_format_included_in_request_body(self) -> None:
        """response_format is forwarded to the /v1/chat/completions request body."""
        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": '{"answer": "42"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm(response_format={"type": "json_object"})
        llm._generate([HumanMessage(content="Give me JSON")])

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["response_format"] == {"type": "json_object"}

    @respx.mock
    def test_response_format_absent_when_none(self) -> None:
        """response_format is not sent when the field is None (default)."""
        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm()  # response_format defaults to None
        llm._generate([HumanMessage(content="hi")])

        sent_body = json.loads(route.calls[0].request.content)
        assert "response_format" not in sent_body

    @respx.mock
    def test_v2_protocol_raises_kserve_error_when_response_format_set(self) -> None:
        """response_format on a V2 protocol call raises KServeError immediately."""
        respx.post(f"{BASE_URL}/v2/models/{MODEL}/infer").mock(
            return_value=httpx.Response(
                200,
                json={"outputs": [{"data": ["ok"]}]},
            )
        )

        llm = make_llm(protocol="v2", response_format={"type": "json_object"})
        with pytest.raises(KServeError, match="OpenAI-compatible protocol"):
            llm._generate([HumanMessage(content="hi")])

    def test_malformed_json_schema_response_format_raises_value_error(self) -> None:
        """json_schema response_format without a schema dict raises ValueError at construction."""
        with pytest.raises(ValueError, match="json_schema.schema"):
            make_llm(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "MyOutput",
                        # 'schema' key missing
                    },
                }
            )

    def test_malformed_json_schema_bad_schema_type_raises_value_error(self) -> None:
        """json_schema with schema as a string (not dict) raises ValueError."""
        with pytest.raises(ValueError, match="json_schema.schema"):
            make_llm(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "MyOutput",
                        "schema": "not-a-dict",
                    },
                }
            )

    def test_valid_json_schema_response_format_accepted(self) -> None:
        """json_schema response_format with a proper schema dict passes validation."""
        llm = make_llm(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "MyOutput",
                    "strict": True,
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
                },
            }
        )
        assert llm.response_format is not None
        assert llm.response_format["type"] == "json_schema"

    def test_json_object_response_format_accepted(self) -> None:
        """{"type": "json_object"} is a valid response_format (no nested schema required)."""
        llm = make_llm(response_format={"type": "json_object"})
        assert llm.response_format == {"type": "json_object"}

    @respx.mock
    def test_response_format_json_schema_included_in_request(self) -> None:
        """json_schema response_format appears verbatim in the request body."""
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "Answer",
                "strict": True,
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            },
        }
        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": '{"answer": "ok"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm(response_format=rf)
        llm._generate([HumanMessage(content="hi")])

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["response_format"]["type"] == "json_schema"
        assert sent_body["response_format"]["json_schema"]["name"] == "Answer"


# ---------------------------------------------------------------------------
# Feature 2: Structured Output via with_structured_output()
# ---------------------------------------------------------------------------


class OutputSchema(BaseModel):
    """Simple Pydantic model used in with_structured_output tests."""
    answer: str
    confidence: float


class TestWithStructuredOutputFunctionCalling:
    @respx.mock
    def test_function_calling_pydantic_model(self) -> None:
        """with_structured_output('function_calling') with a Pydantic model
        forces a tool call and parses args into the schema type."""
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
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "OutputSchema",
                                            "arguments": json.dumps(
                                                {"answer": "Paris", "confidence": 0.99}
                                            ),
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
        chain = llm.with_structured_output(OutputSchema, method="function_calling")
        result = chain.invoke([HumanMessage(content="Capital of France?")])

        assert isinstance(result, OutputSchema)
        assert result.answer == "Paris"
        assert result.confidence == pytest.approx(0.99)

    @respx.mock
    def test_function_calling_dict_schema(self) -> None:
        """with_structured_output('function_calling') with a dict schema returns a dict."""
        schema_dict: Dict[str, Any] = {
            "title": "MyOutput",
            "type": "object",
            "properties": {"value": {"type": "integer"}},
        }
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
                                        "id": "call_2",
                                        "type": "function",
                                        "function": {
                                            "name": "MyOutput",
                                            "arguments": json.dumps({"value": 7}),
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
        chain = llm.with_structured_output(schema_dict, method="function_calling")
        result = chain.invoke([HumanMessage(content="Give me a value")])

        assert isinstance(result, dict)
        assert result["value"] == 7

    @respx.mock
    def test_function_calling_tool_choice_forced(self) -> None:
        """bind_tools is called with tool_choice forcing the specific function."""
        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
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
                                        "id": "call_3",
                                        "type": "function",
                                        "function": {
                                            "name": "OutputSchema",
                                            "arguments": json.dumps(
                                                {"answer": "yes", "confidence": 1.0}
                                            ),
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
        chain = llm.with_structured_output(OutputSchema, method="function_calling")
        chain.invoke([HumanMessage(content="Are you sure?")])

        sent_body = json.loads(route.calls[0].request.content)
        assert "tool_choice" in sent_body
        assert sent_body["tool_choice"]["type"] == "function"
        assert sent_body["tool_choice"]["function"]["name"] == "OutputSchema"


class TestWithStructuredOutputJsonSchema:
    @respx.mock
    def test_json_schema_pydantic_model(self) -> None:
        """with_structured_output('json_schema') sets response_format and parses JSON."""
        respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"answer": "Rome", "confidence": 0.85}),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm()
        chain = llm.with_structured_output(OutputSchema, method="json_schema")
        result = chain.invoke([HumanMessage(content="Capital of Italy?")])

        assert isinstance(result, OutputSchema)
        assert result.answer == "Rome"

    @respx.mock
    def test_json_schema_request_body_contains_response_format(self) -> None:
        """json_schema method sets response_format.type='json_schema' in request."""
        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"answer": "Rome", "confidence": 0.85}),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm()
        chain = llm.with_structured_output(OutputSchema, method="json_schema")
        chain.invoke([HumanMessage(content="hi")])

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["response_format"]["type"] == "json_schema"
        assert sent_body["response_format"]["json_schema"]["name"] == "OutputSchema"
        assert sent_body["response_format"]["json_schema"]["strict"] is True

    @respx.mock
    def test_json_schema_strict_override(self) -> None:
        """strict=False is respected when building json_schema response_format."""
        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"answer": "x", "confidence": 0.0}),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm()
        chain = llm.with_structured_output(OutputSchema, method="json_schema", strict=False)
        chain.invoke([HumanMessage(content="hi")])

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["response_format"]["json_schema"]["strict"] is False


class TestWithStructuredOutputJsonMode:
    @respx.mock
    def test_json_mode_returns_dict(self) -> None:
        """with_structured_output('json_mode') sets json_object response_format and returns dict."""
        route = respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"city": "Berlin"}),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm()
        chain = llm.with_structured_output(
            {"type": "object", "properties": {"city": {"type": "string"}}},
            method="json_mode",
        )
        result = chain.invoke([HumanMessage(content="A city?")])

        assert isinstance(result, dict)
        assert result["city"] == "Berlin"

        sent_body = json.loads(route.calls[0].request.content)
        assert sent_body["response_format"] == {"type": "json_object"}


class TestWithStructuredOutputIncludeRaw:
    @respx.mock
    def test_include_raw_true_function_calling(self) -> None:
        """include_raw=True wraps output in {'raw', 'parsed', 'parsing_error'} dict."""
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
                                        "id": "call_x",
                                        "type": "function",
                                        "function": {
                                            "name": "OutputSchema",
                                            "arguments": json.dumps(
                                                {"answer": "yes", "confidence": 1.0}
                                            ),
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
        chain = llm.with_structured_output(
            OutputSchema, method="function_calling", include_raw=True
        )
        result = chain.invoke([HumanMessage(content="Are you certain?")])

        assert isinstance(result, dict)
        assert "raw" in result
        assert "parsed" in result
        assert "parsing_error" in result
        assert isinstance(result["raw"], AIMessage)
        assert isinstance(result["parsed"], OutputSchema)
        assert result["parsing_error"] is None

    @respx.mock
    def test_include_raw_true_json_mode(self) -> None:
        """include_raw=True with json_mode method returns correct structure."""
        respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"x": 1}),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm()
        chain = llm.with_structured_output(
            {"type": "object"},
            method="json_mode",
            include_raw=True,
        )
        result = chain.invoke([HumanMessage(content="give json")])

        assert isinstance(result, dict)
        assert set(result.keys()) == {"raw", "parsed", "parsing_error"}
        assert isinstance(result["raw"], AIMessage)
        assert result["parsed"] == {"x": 1}
        assert result["parsing_error"] is None

    @respx.mock
    def test_include_raw_true_parsing_error_captured(self) -> None:
        """When parsing fails, include_raw=True returns parsing_error and parsed=None."""
        respx.post(f"{BASE_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                # Model returns no tool_calls, causing parser to raise
                                "content": "Sorry, I cannot do that.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                },
            )
        )

        llm = make_llm()
        chain = llm.with_structured_output(
            OutputSchema, method="function_calling", include_raw=True
        )
        result = chain.invoke([HumanMessage(content="hmm")])

        assert result["raw"] is not None
        assert result["parsed"] is None
        assert result["parsing_error"] is not None
