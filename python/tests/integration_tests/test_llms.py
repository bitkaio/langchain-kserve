"""Integration tests for KServeLLM.

Configure via environment variables (see test_chat_models.py for details).
"""

from __future__ import annotations

import os

import pytest

from langchain_kserve import KServeLLM


@pytest.fixture()
def llm() -> KServeLLM:
    return KServeLLM(
        base_url=os.environ.get("KSERVE_BASE_URL", ""),
        model_name=os.environ.get("KSERVE_MODEL_NAME", ""),
        temperature=0.1,
        max_tokens=32,
    )


@pytest.mark.integration
def test_basic_invoke(llm: KServeLLM) -> None:
    response = llm.invoke("The capital of France is")
    assert len(response) > 0


@pytest.mark.integration
def test_streaming(llm: KServeLLM) -> None:
    chunks = list(llm.stream("Once upon a time"))
    assert len(chunks) > 0
