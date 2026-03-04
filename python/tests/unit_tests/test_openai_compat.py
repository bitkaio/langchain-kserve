"""Unit tests for _openai_compat module."""

from __future__ import annotations

import json

import httpx
import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_kserve._common import KServeInferenceError
from langchain_kserve._openai_compat import (
    _parse_sse_chat_line,
    _parse_sse_completion_line,
    build_chat_request,
    build_completion_request,
    messages_to_openai_dicts,
    parse_chat_response,
    parse_completion_response,
)


class TestMessagesToOpenAIDicts:
    def test_system_message(self) -> None:
        result = messages_to_openai_dicts([SystemMessage(content="Be helpful.")])
        assert result == [{"role": "system", "content": "Be helpful."}]

    def test_human_message(self) -> None:
        result = messages_to_openai_dicts([HumanMessage(content="Hello")])
        assert result == [{"role": "user", "content": "Hello"}]

    def test_ai_message(self) -> None:
        result = messages_to_openai_dicts([AIMessage(content="Hi there!")])
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there!"

    def test_tool_message(self) -> None:
        msg = ToolMessage(content="42", tool_call_id="call_1")
        result = messages_to_openai_dicts([msg])
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "42"

    def test_mixed_conversation(self) -> None:
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="q"),
            AIMessage(content="a"),
        ]
        result = messages_to_openai_dicts(msgs)
        assert [m["role"] for m in result] == ["system", "user", "assistant"]

    def test_ai_message_with_tool_calls(self) -> None:
        from langchain_core.messages.tool import ToolCall

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                ToolCall(id="call_x", name="func", args={"k": "v"})
            ],
        )
        result = messages_to_openai_dicts([ai_msg])
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "call_x"
        assert tc["function"]["name"] == "func"
        assert json.loads(tc["function"]["arguments"]) == {"k": "v"}


class TestBuildChatRequest:
    def test_basic_fields_present(self) -> None:
        body = build_chat_request(
            model_name="my-model",
            messages=[HumanMessage(content="hi")],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop=None,
            stream=False,
        )
        assert body["model"] == "my-model"
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 100
        assert body["stream"] is False

    def test_max_tokens_omitted_when_none(self) -> None:
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
        )
        assert "max_tokens" not in body

    def test_tools_included(self) -> None:
        tool = {"type": "function", "function": {"name": "fn", "description": "d", "parameters": {}}}
        body = build_chat_request(
            model_name="m",
            messages=[],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
            tools=[tool],
        )
        assert body["tools"] == [tool]

    def test_extra_kwargs_merged(self) -> None:
        body = build_chat_request(
            model_name="m",
            messages=[],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
            extra_kwargs={"frequency_penalty": 0.5},
        )
        assert body["frequency_penalty"] == 0.5


class TestBuildCompletionRequest:
    def test_basic(self) -> None:
        body = build_completion_request(
            model_name="llama",
            prompt="Once upon",
            temperature=0.8,
            max_tokens=50,
            top_p=1.0,
            stop=["END"],
            stream=False,
        )
        assert body["prompt"] == "Once upon"
        assert body["stop"] == ["END"]


class TestParseChatResponse:
    def test_parses_content(self) -> None:
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )
        result = parse_chat_response(response, "model")
        assert result.generations[0].message.content == "Hello!"
        assert result.generations[0].generation_info["total_tokens"] == 7

    def test_empty_choices_raises(self) -> None:
        response = httpx.Response(200, json={"choices": []})
        with pytest.raises(KServeInferenceError):
            parse_chat_response(response, "model")

    def test_invalid_json_raises(self) -> None:
        response = httpx.Response(200, content=b"not json")
        with pytest.raises(KServeInferenceError):
            parse_chat_response(response, "model")


class TestParseCompletionResponse:
    def test_parses_text(self) -> None:
        response = httpx.Response(
            200,
            json={"choices": [{"text": "the output", "finish_reason": "stop"}]},
        )
        assert parse_completion_response(response, "model") == "the output"

    def test_empty_choices_raises(self) -> None:
        response = httpx.Response(200, json={"choices": []})
        with pytest.raises(KServeInferenceError):
            parse_completion_response(response, "model")


class TestSSEParsing:
    def test_parse_chat_line_content(self) -> None:
        line = 'data: {"choices": [{"delta": {"content": "hello"}, "finish_reason": null}]}'
        chunk = _parse_sse_chat_line(line, "m", "openai")
        assert chunk is not None
        assert chunk.message.content == "hello"

    def test_parse_chat_line_done(self) -> None:
        assert _parse_sse_chat_line("data: [DONE]", "m", "openai") is None

    def test_parse_chat_line_empty(self) -> None:
        assert _parse_sse_chat_line("", "m", "openai") is None

    def test_parse_completion_line(self) -> None:
        line = 'data: {"choices": [{"text": "world", "finish_reason": null}]}'
        assert _parse_sse_completion_line(line) == "world"

    def test_parse_completion_line_done(self) -> None:
        assert _parse_sse_completion_line("data: [DONE]") is None
