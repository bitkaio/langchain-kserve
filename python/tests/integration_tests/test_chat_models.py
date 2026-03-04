"""Integration tests for ChatKServe.

These tests require a live KServe endpoint.  Configure via environment variables:

    KSERVE_BASE_URL=https://qwen-coder.my-cluster.example.com
    KSERVE_MODEL_NAME=qwen2.5-coder-32b-instruct
    KSERVE_API_KEY=<token>          # optional
    KSERVE_PROTOCOL=openai          # optional, defaults to auto

Run with::

    pytest -m integration tests/integration_tests/
"""

from __future__ import annotations

import os

import pytest

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_kserve import ChatKServe


@pytest.fixture()
def llm() -> ChatKServe:
    return ChatKServe(
        base_url=os.environ.get("KSERVE_BASE_URL", ""),
        model_name=os.environ.get("KSERVE_MODEL_NAME", ""),
        temperature=0.1,
        max_tokens=64,
    )


@pytest.mark.integration
def test_basic_invoke(llm: ChatKServe) -> None:
    response = llm.invoke("Say 'hello world' and nothing else.")
    assert "hello" in response.content.lower()


@pytest.mark.integration
def test_streaming(llm: ChatKServe) -> None:
    chunks = list(llm.stream("Count from 1 to 5."))
    full = "".join(c.content for c in chunks if c.content)
    assert len(full) > 0


@pytest.mark.integration
def test_system_message(llm: ChatKServe) -> None:
    response = llm.invoke(
        [
            SystemMessage(content="You respond only with 'OK'."),
            HumanMessage(content="Acknowledge."),
        ]
    )
    assert response.content.strip().lower() == "ok"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_invoke(llm: ChatKServe) -> None:
    response = await llm.ainvoke("Say 'async works' and nothing else.")
    assert "async" in response.content.lower()
