"""Live model providers: Gemini and Groq over chat-completions APIs.

Both providers use only the standard library (urllib) and post a
chat-completions JSON body to the full endpoint URL verbatim. No
provider reads a credential file, hardcodes a key, or logs a key or
an Authorization header. Keys come only from environment variables,
and a missing key raises a clear error before any request is made.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

DEFAULT_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
DEFAULT_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
# Groq model, verified 2026-11-26. The earlier default,
# llama-3.3-70b-versatile, was retired by Groq and returns 404 on
# this account; it is not available. The chat-capable models listed
# for this account are openai/gpt-oss-120b, openai/gpt-oss-20b,
# qwen/qwen3.8-27b and allam-2-7b.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# A named client User-Agent. Groq sits behind Cloudflare, which
# blocks the default Python-urllib User-Agent as a bot (error 1010);
# a named client User-Agent passes and identifies the client
# truthfully.
USER_AGENT = (
    "llm-eval-leaderboard/1.0 "
    "(https://github.com/Sharif899/llm-eval-leaderboard)"
)

# HTTP statuses worth one retry by the runner.
RETRYABLE_STATUSES = (429, 503)


class ProviderError(Exception):
    """A provider call failed. Carries the HTTP status code when the
    provider answered with one, so the runner can decide whether to
    retry. ``status_code`` is None for transport-level failures."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class Provider(ABC):
    """One live model behind a chat-completions endpoint."""

    name: str = ""
    model: str = ""

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """Return the model's completion for one prompt."""
        raise NotImplementedError


def _require_key(env_var: str, provider_name: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise ProviderError(
            f"{provider_name}: no API key found. Set the {env_var} "
            "environment variable before running a live evaluation."
        )
    return key


def _post_chat_completions(
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Post one chat-completions request and return the reply text.

    The Authorization header is built here and never logged. Only
    the endpoint, model and status code ever appear in an error.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = ""
        raise ProviderError(
            f"Provider request failed with HTTP {exc.code} from {endpoint}"
            + (f": {detail}" if detail else ""),
            status_code=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(
            f"Provider request to {endpoint} failed: {exc.reason}",
            status_code=None,
        ) from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(
            f"Provider response from {endpoint} had no usable completion text.",
            status_code=None,
        ) from exc
    if not isinstance(content, str):
        raise ProviderError(
            f"Provider response from {endpoint} had no usable completion text.",
            status_code=None,
        )
    return content


class GeminiProvider(Provider):
    """Google Gemini through its OpenAI-compatible chat endpoint."""

    name = "gemini"

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        temperature: float = 0.0,
        api_key: str | None = None,
    ):
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.endpoint = endpoint or os.environ.get(
            "GEMINI_ENDPOINT", DEFAULT_GEMINI_ENDPOINT
        )
        self.temperature = temperature
        self._api_key = api_key if api_key is not None else _require_key(
            "GEMINI_API_KEY", "Gemini"
        )

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        return _post_chat_completions(
            self.endpoint, self._api_key, self.model,
            prompt, self.temperature, max_tokens,
        )


class GroqProvider(Provider):
    """Groq through its OpenAI-compatible chat endpoint."""

    name = "groq"

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        temperature: float = 0.0,
        api_key: str | None = None,
    ):
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.endpoint = endpoint or os.environ.get(
            "GROQ_ENDPOINT", DEFAULT_GROQ_ENDPOINT
        )
        self.temperature = temperature
        self._api_key = api_key if api_key is not None else _require_key(
            "GROQ_API_KEY", "Groq"
        )

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        return _post_chat_completions(
            self.endpoint, self._api_key, self.model,
            prompt, self.temperature, max_tokens,
        )


PROVIDERS = {"gemini": GeminiProvider, "groq": GroqProvider}


def get_provider(name: str) -> Provider:
    """Build the named provider from its environment variables."""
    try:
        factory = PROVIDERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown provider '{name}'. Choose from: {', '.join(PROVIDERS)}"
        ) from exc
    return factory()
