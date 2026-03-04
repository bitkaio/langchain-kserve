"""Shared HTTP client, authentication, retry logic, and error hierarchy for langchain-kserve."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable, Dict, Optional

import httpx
from langchain_core.exceptions import LangChainException
from pydantic import SecretStr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class KServeError(LangChainException):
    """Base exception for all KServe errors."""


class KServeConnectionError(KServeError):
    """Cannot reach the inference service (cold start, DNS, TLS)."""


class KServeAuthenticationError(KServeError):
    """401 or 403 returned by the service."""


class KServeModelNotFoundError(KServeError):
    """Model name doesn't match what is served."""


class KServeInferenceError(KServeError):
    """Model returned an error during inference."""


class KServeTimeoutError(KServeError):
    """Inference took longer than the configured timeout."""


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


def _jitter_sleep(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Return exponential-backoff-with-jitter sleep duration (seconds)."""
    delay = min(base * (2 ** attempt), cap)
    return delay * (0.5 + random.random() * 0.5)  # noqa: S311 — not crypto


# ---------------------------------------------------------------------------
# HTTP client factory
# ---------------------------------------------------------------------------


def build_sync_client(
    base_url: str,
    api_key: Optional[SecretStr],
    token_provider: Optional[Callable[[], str]],
    verify_ssl: bool,
    ca_bundle: Optional[str],
    timeout: int,
) -> httpx.Client:
    """Create a synchronous :class:`httpx.Client` configured for KServe.

    Args:
        base_url: Root URL of the KServe inference service.
        api_key: Static bearer token.
        token_provider: Callable returning a bearer token dynamically.
        verify_ssl: Whether to verify TLS certificates.
        ca_bundle: Path to a custom CA certificate bundle.
        timeout: Request timeout in seconds.

    Returns:
        A configured :class:`httpx.Client`.
    """
    headers = _build_headers(api_key, token_provider)
    ssl_context = _build_ssl(verify_ssl, ca_bundle)
    return httpx.Client(
        base_url=base_url,
        headers=headers,
        verify=ssl_context,
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
    )


def build_async_client(
    base_url: str,
    api_key: Optional[SecretStr],
    token_provider: Optional[Callable[[], str]],
    verify_ssl: bool,
    ca_bundle: Optional[str],
    timeout: int,
) -> httpx.AsyncClient:
    """Create an asynchronous :class:`httpx.AsyncClient` configured for KServe.

    Args:
        base_url: Root URL of the KServe inference service.
        api_key: Static bearer token.
        token_provider: Callable returning a bearer token dynamically.
        verify_ssl: Whether to verify TLS certificates.
        ca_bundle: Path to a custom CA certificate bundle.
        timeout: Request timeout in seconds.

    Returns:
        A configured :class:`httpx.AsyncClient`.
    """
    headers = _build_headers(api_key, token_provider)
    ssl_context = _build_ssl(verify_ssl, ca_bundle)
    return httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        verify=ssl_context,
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
    )


def _build_headers(
    api_key: Optional[SecretStr],
    token_provider: Optional[Callable[[], str]],
) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
    elif token_provider is not None:
        headers["Authorization"] = f"Bearer {token_provider()}"
    return headers


def _build_ssl(verify_ssl: bool, ca_bundle: Optional[str]) -> Any:
    if not verify_ssl:
        return False
    if ca_bundle is not None:
        return ca_bundle
    return True


# ---------------------------------------------------------------------------
# Sync request helper with retries
# ---------------------------------------------------------------------------


def request_with_retry(
    client: httpx.Client,
    method: str,
    path: str,
    max_retries: int,
    **kwargs: Any,
) -> httpx.Response:
    """Perform an HTTP request with exponential-backoff retries.

    Args:
        client: Configured :class:`httpx.Client`.
        method: HTTP method (``"GET"``, ``"POST"``, …).
        path: URL path relative to the client's ``base_url``.
        max_retries: Maximum number of retry attempts after the first failure.
        **kwargs: Additional kwargs forwarded to ``client.request``.

    Returns:
        The successful :class:`httpx.Response`.

    Raises:
        KServeConnectionError: The service is unreachable.
        KServeAuthenticationError: 401 or 403.
        KServeModelNotFoundError: 404.
        KServeTimeoutError: Request timed out after all retries.
        KServeInferenceError: Non-retryable HTTP error.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = client.request(method, path, **kwargs)
            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                logger.warning(
                    "KServe returned %s (attempt %d/%d), retrying…",
                    response.status_code,
                    attempt + 1,
                    max_retries + 1,
                )
                import time  # noqa: PLC0415

                time.sleep(_jitter_sleep(attempt))
                continue
            _raise_for_status(response)
            return response
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < max_retries:
                import time  # noqa: PLC0415

                logger.warning(
                    "KServe connection error (attempt %d/%d): %s, retrying…",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                time.sleep(_jitter_sleep(attempt))
                continue
            raise KServeConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            last_exc = exc
            if attempt < max_retries:
                import time  # noqa: PLC0415

                logger.warning(
                    "KServe timeout (attempt %d/%d), retrying…",
                    attempt + 1,
                    max_retries + 1,
                )
                time.sleep(_jitter_sleep(attempt))
                continue
            raise KServeTimeoutError(str(exc)) from exc

    # Should not reach here, but satisfy type checker
    raise KServeConnectionError(str(last_exc))


# ---------------------------------------------------------------------------
# Async request helper with retries
# ---------------------------------------------------------------------------


async def async_request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    max_retries: int,
    **kwargs: Any,
) -> httpx.Response:
    """Async version of :func:`request_with_retry`.

    Args:
        client: Configured :class:`httpx.AsyncClient`.
        method: HTTP method.
        path: URL path relative to ``base_url``.
        max_retries: Maximum retry attempts.
        **kwargs: Forwarded to ``client.request``.

    Returns:
        The successful :class:`httpx.Response`.

    Raises:
        KServeConnectionError: The service is unreachable.
        KServeAuthenticationError: 401 or 403.
        KServeModelNotFoundError: 404.
        KServeTimeoutError: Request timed out.
        KServeInferenceError: Non-retryable HTTP error.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.request(method, path, **kwargs)
            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < max_retries:
                logger.warning(
                    "KServe returned %s (attempt %d/%d), retrying…",
                    response.status_code,
                    attempt + 1,
                    max_retries + 1,
                )
                await asyncio.sleep(_jitter_sleep(attempt))
                continue
            _raise_for_status(response)
            return response
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "KServe connection error (attempt %d/%d): %s, retrying…",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                await asyncio.sleep(_jitter_sleep(attempt))
                continue
            raise KServeConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "KServe timeout (attempt %d/%d), retrying…",
                    attempt + 1,
                    max_retries + 1,
                )
                await asyncio.sleep(_jitter_sleep(attempt))
                continue
            raise KServeTimeoutError(str(exc)) from exc

    raise KServeConnectionError(str(last_exc))


# ---------------------------------------------------------------------------
# Status code → exception mapping
# ---------------------------------------------------------------------------


def _raise_for_status(response: httpx.Response) -> None:
    """Map HTTP error status codes to :class:`KServeError` subclasses.

    Args:
        response: The HTTP response to check.

    Raises:
        KServeAuthenticationError: Status 401 or 403.
        KServeModelNotFoundError: Status 404.
        KServeInferenceError: Any other 4xx or 5xx status.
    """
    if response.is_success:
        return
    try:
        body = response.text
    except Exception:
        body = "<unreadable>"

    if response.status_code in (401, 403):
        raise KServeAuthenticationError(
            f"Authentication failed ({response.status_code}): {body}"
        )
    if response.status_code == 404:
        raise KServeModelNotFoundError(
            f"Model not found ({response.status_code}): {body}"
        )
    raise KServeInferenceError(
        f"KServe returned error ({response.status_code}): {body}"
    )
