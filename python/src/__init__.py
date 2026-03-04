"""langchain-kserve — LangChain integration for KServe inference services."""

from langchain_kserve._common import (
    KServeAuthenticationError,
    KServeConnectionError,
    KServeError,
    KServeInferenceError,
    KServeModelNotFoundError,
    KServeTimeoutError,
)
from langchain_kserve.chat_models import ChatKServe
from langchain_kserve.llms import KServeLLM

__all__ = [
    # Model classes
    "ChatKServe",
    "KServeLLM",
    # Exceptions
    "KServeError",
    "KServeConnectionError",
    "KServeAuthenticationError",
    "KServeModelNotFoundError",
    "KServeInferenceError",
    "KServeTimeoutError",
]

__version__ = "0.1.0"
