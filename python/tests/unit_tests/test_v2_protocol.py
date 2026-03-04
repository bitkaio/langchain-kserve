"""Unit tests for _v2_protocol module."""

from __future__ import annotations

import json

import httpx
import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_kserve._common import KServeInferenceError
from langchain_kserve._v2_protocol import (
    _extract_v2_output_text,
    build_v2_infer_request,
    infer_path,
    messages_to_prompt,
    parse_v2_chat_response,
    parse_v2_completion_response,
)


class TestMessagesToPrompt:
    def test_system_and_user(self) -> None:
        msgs = [
            SystemMessage(content="Be helpful."),
            HumanMessage(content="What is Python?"),
        ]
        prompt = messages_to_prompt(msgs)
        assert "<|im_start|>system\nBe helpful.<|im_end|>" in prompt
        assert "<|im_start|>user\nWhat is Python?<|im_end|>" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_user_only(self) -> None:
        msgs = [HumanMessage(content="Hello")]
        prompt = messages_to_prompt(msgs)
        assert "<|im_start|>user\nHello<|im_end|>" in prompt

    def test_ai_message_in_history(self) -> None:
        msgs = [
            HumanMessage(content="Q"),
            AIMessage(content="A"),
            HumanMessage(content="Follow up"),
        ]
        prompt = messages_to_prompt(msgs)
        assert "<|im_start|>assistant\nA<|im_end|>" in prompt

    def test_ends_with_assistant_start(self) -> None:
        msgs = [HumanMessage(content="Hi")]
        assert messages_to_prompt(msgs).endswith("<|im_start|>assistant\n")


class TestBuildV2InferRequest:
    def test_basic_structure(self) -> None:
        body = build_v2_infer_request(
            model_name="my-model",
            prompt="hello",
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
        )
        assert body["inputs"][0]["name"] == "text_input"
        assert body["inputs"][0]["data"] == ["hello"]
        assert body["inputs"][0]["datatype"] == "BYTES"
        assert body["inputs"][0]["shape"] == [1]

    def test_max_tokens_in_parameters(self) -> None:
        body = build_v2_infer_request(
            model_name="m",
            prompt="p",
            temperature=0.5,
            max_tokens=256,
            top_p=0.9,
            stop=None,
        )
        assert body["parameters"]["max_tokens"] == 256

    def test_max_tokens_omitted_when_none(self) -> None:
        body = build_v2_infer_request(
            model_name="m",
            prompt="p",
            temperature=0.5,
            max_tokens=None,
            top_p=1.0,
            stop=None,
        )
        assert "max_tokens" not in body["parameters"]

    def test_stop_in_parameters(self) -> None:
        body = build_v2_infer_request(
            model_name="m",
            prompt="p",
            temperature=0.5,
            max_tokens=None,
            top_p=1.0,
            stop=["END"],
        )
        assert body["parameters"]["stop"] == ["END"]

    def test_custom_request_id(self) -> None:
        body = build_v2_infer_request(
            model_name="m",
            prompt="p",
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            request_id="my-id",
        )
        assert body["id"] == "my-id"

    def test_extra_kwargs_in_parameters(self) -> None:
        body = build_v2_infer_request(
            model_name="m",
            prompt="p",
            temperature=0.7,
            max_tokens=None,
            top_p=1.0,
            stop=None,
            extra_kwargs={"repetition_penalty": 1.1},
        )
        assert body["parameters"]["repetition_penalty"] == 1.1


class TestParseV2Response:
    def _make_response(self, data: list) -> httpx.Response:
        return httpx.Response(
            200,
            json={"outputs": [{"name": "text_output", "data": data}]},
        )

    def test_parses_string_output(self) -> None:
        response = self._make_response(["Hello there!"])
        assert parse_v2_completion_response(response) == "Hello there!"

    def test_parses_chat_response(self) -> None:
        response = self._make_response(["Model answer"])
        result = parse_v2_chat_response(response, "my-model")
        assert result.generations[0].message.content == "Model answer"
        assert result.generations[0].generation_info["protocol"] == "v2"

    def test_empty_outputs_raises(self) -> None:
        response = httpx.Response(200, json={"outputs": []})
        with pytest.raises(KServeInferenceError):
            parse_v2_completion_response(response)

    def test_empty_data_raises(self) -> None:
        response = httpx.Response(
            200, json={"outputs": [{"name": "text_output", "data": []}]}
        )
        with pytest.raises(KServeInferenceError):
            parse_v2_completion_response(response)

    def test_invalid_json_raises(self) -> None:
        response = httpx.Response(200, content=b"not json")
        with pytest.raises(KServeInferenceError):
            parse_v2_completion_response(response)


class TestInferPath:
    def test_path_format(self) -> None:
        assert infer_path("my-model") == "/v2/models/my-model/infer"

    def test_path_with_slash_in_name(self) -> None:
        assert infer_path("org/model") == "/v2/models/org/model/infer"
