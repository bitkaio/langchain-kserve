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
        text, usage = parse_completion_response(response, "model")
        assert text == "the output"
        assert usage is None

    def test_parses_usage(self) -> None:
        response = httpx.Response(
            200,
            json={
                "choices": [{"text": "hello", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )
        text, usage = parse_completion_response(response, "model")
        assert text == "hello"
        assert usage is not None
        assert usage["total_tokens"] == 7

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


# ---------------------------------------------------------------------------
# New feature tests
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_parse_chat_response_llm_output_token_usage(self) -> None:
        """parse_chat_response with usage field populates llm_output["token_usage"]."""
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
        result = parse_chat_response(response, "my-model")
        assert result.llm_output is not None
        assert result.llm_output["token_usage"]["prompt_tokens"] == 10
        assert result.llm_output["token_usage"]["completion_tokens"] == 5
        assert result.llm_output["token_usage"]["total_tokens"] == 15
        assert result.llm_output["model_name"] == "my-model"

    def test_parse_chat_response_no_usage_llm_output_none(self) -> None:
        """parse_chat_response without usage field returns llm_output as None."""
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )
        result = parse_chat_response(response, "my-model")
        assert result.llm_output is None

    def test_build_chat_request_stream_options_when_streaming(self) -> None:
        """stream=True adds stream_options.include_usage=True."""
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=True,
        )
        assert body.get("stream_options") == {"include_usage": True}

    def test_build_chat_request_no_stream_options_when_not_streaming(self) -> None:
        """stream=False does not add stream_options."""
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
        )
        assert "stream_options" not in body

    def test_parse_sse_usage_only_chunk(self) -> None:
        """vLLM final usage-only chunk (choices=[]) returns a ChatGenerationChunk with token_usage."""
        import json as _json
        payload = _json.dumps({
            "choices": [],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        })
        line = f"data: {payload}"
        chunk = _parse_sse_chat_line(line, "m", "openai")
        assert chunk is not None
        assert chunk.message.content == ""
        assert chunk.generation_info is not None
        assert chunk.generation_info["token_usage"]["prompt_tokens"] == 8
        assert chunk.generation_info["token_usage"]["total_tokens"] == 11


class TestLogprobs:
    def test_build_chat_request_logprobs_params(self) -> None:
        """logprobs=True and top_logprobs=5 appear in request body."""
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
            logprobs=True,
            top_logprobs=5,
        )
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 5

    def test_build_chat_request_logprobs_omitted_when_none(self) -> None:
        """logprobs=None means logprobs key not in body."""
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
        )
        assert "logprobs" not in body

    def test_parse_chat_response_logprobs(self) -> None:
        """Response with choices[0].logprobs populates generation_info['logprobs']."""
        logprobs_data = {"tokens": ["Hi"], "token_logprobs": [-0.5]}
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                        "logprobs": logprobs_data,
                    }
                ],
                "usage": {},
            },
        )
        result = parse_chat_response(response, "m")
        assert result.generations[0].generation_info is not None
        assert result.generations[0].generation_info["logprobs"] == logprobs_data


class TestToolCallParsing:
    def test_parse_chat_response_invalid_tool_call_json(self) -> None:
        """Tool call with invalid JSON args populates AIMessage.invalid_tool_calls."""
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_bad",
                                    "type": "function",
                                    "function": {
                                        "name": "my_func",
                                        "arguments": "not valid json{",
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
        result = parse_chat_response(response, "m")
        ai_msg = result.generations[0].message
        assert isinstance(ai_msg, AIMessage)
        assert len(ai_msg.tool_calls) == 0
        assert len(ai_msg.invalid_tool_calls) == 1
        assert ai_msg.invalid_tool_calls[0]["name"] == "my_func"
        assert ai_msg.invalid_tool_calls[0]["args"] == "not valid json{"

    def test_build_chat_request_tool_choice_and_parallel(self) -> None:
        """tool_choice='required' and parallel_tool_calls=True appear in request body."""
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
            tool_choice="required",
            parallel_tool_calls=True,
        )
        assert body["tool_choice"] == "required"
        assert body["parallel_tool_calls"] is True

    def test_build_chat_request_tool_choice_omitted_when_none(self) -> None:
        """tool_choice=None means key absent from body."""
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
        )
        assert "tool_choice" not in body
        assert "parallel_tool_calls" not in body


class TestVisionContent:
    def test_messages_with_image_url_preserved(self) -> None:
        """HumanMessage with image_url block produces content as list in OpenAI dict."""
        from langchain_core.messages import HumanMessage as HM
        msg = HM(content=[
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ])
        result = messages_to_openai_dicts([msg])
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Describe this image:"}
        assert content[1] == {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}


class TestResponseFormat:
    def test_response_format_included_when_set(self) -> None:
        """response_format dict appears in the request body when provided."""
        rf = {"type": "json_object"}
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
            response_format=rf,
        )
        assert body["response_format"] == rf

    def test_response_format_omitted_when_none(self) -> None:
        """response_format key is absent when None is passed."""
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
            response_format=None,
        )
        assert "response_format" not in body

    def test_response_format_json_schema_passed_through(self) -> None:
        """json_schema response_format is passed through verbatim."""
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "MyOutput",
                "strict": True,
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            },
        }
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
            response_format=rf,
        )
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["name"] == "MyOutput"
        assert body["response_format"]["json_schema"]["schema"]["type"] == "object"

    def test_response_format_coexists_with_tools(self) -> None:
        """response_format and tools can both be present in the same request body."""
        tool = {"type": "function", "function": {"name": "fn", "description": "d", "parameters": {}}}
        rf = {"type": "json_object"}
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
            tools=[tool],
            response_format=rf,
        )
        assert "tools" in body
        assert "response_format" in body

    def test_response_format_not_overwritten_by_extra_kwargs(self) -> None:
        """extra_kwargs merging does not overwrite an explicit response_format."""
        rf = {"type": "json_object"}
        body = build_chat_request(
            model_name="m",
            messages=[HumanMessage(content="hi")],
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            stream=False,
            response_format=rf,
            extra_kwargs={"frequency_penalty": 0.1},
        )
        assert body["response_format"] == rf
        assert body["frequency_penalty"] == 0.1


class TestFinishReason:
    def test_parse_chat_response_finish_reason_always_present(self) -> None:
        """finish_reason is always in generation_info, even if None."""
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi"},
                        # no finish_reason key
                    }
                ],
                "usage": {},
            },
        )
        result = parse_chat_response(response, "m")
        assert "finish_reason" in result.generations[0].generation_info  # type: ignore[index]
        assert result.generations[0].generation_info["finish_reason"] is None  # type: ignore[index]

    def test_parse_chat_response_finish_reason_stop(self) -> None:
        """finish_reason='stop' is correctly propagated."""
        response = httpx.Response(
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
        result = parse_chat_response(response, "m")
        assert result.generations[0].generation_info["finish_reason"] == "stop"  # type: ignore[index]
