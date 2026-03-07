"""langchain-kserve — LangChain integration for KServe inference services."""

from langchain_kserve._common import (
    KServeAuthenticationError,
    KServeConnectionError,
    KServeError,
    KServeInferenceError,
    KServeModelInfo,
    KServeModelNotFoundError,
    KServeTimeoutError,
)
from langchain_kserve.chat_models import ChatKServe
from langchain_kserve.embeddings import KServeEmbeddings
from langchain_kserve.llms import KServeLLM

__all__ = [
    # Model classes
    "ChatKServe",
    "KServeLLM",
    "KServeEmbeddings",
    # Exceptions
    "KServeError",
    "KServeConnectionError",
    "KServeAuthenticationError",
    "KServeModelNotFoundError",
    "KServeInferenceError",
    "KServeTimeoutError",
    # Metadata
    "KServeModelInfo",
]

__version__ = "0.1.0"
