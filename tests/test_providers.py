"""Tests for providers: request construction, missing keys, and
error handling. All HTTP is stubbed; no network is used."""

from __future__ import annotations

import json
import urllib.error

import pytest

from llm_eval_leaderboard.providers import (
    GeminiProvider,
    GroqProvider,
    ProviderError,
    _post_chat_completions,
)


class FakeResponse:
    """A stubbed urlopen response: a context manager returning JSON bytes."""

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _completion_payload(text="Hello, this is the model speaking."):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def test_gemini_request_construction(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["auth"] = request.get_header("Authorization")
        captured["content_type"] = request.get_header("Content-type")
        captured["user_agent"] = request.get_header("User-agent")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(_completion_payload("Yoruba answer."))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    provider = GeminiProvider()
    assert provider.model == "gemini-3.5-flash-lite"
    assert provider.name == "gemini"
    assert provider.endpoint == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    text = provider.complete("Kí ni olú ìlú Nàìjíríà?")
    assert text == "Yoruba answer."
    assert captured["url"] == provider.endpoint
    assert captured["method"] == "POST"
    assert captured["auth"] == "Bearer test-key-123"
    assert captured["content_type"] == "application/json"
    assert captured["user_agent"].startswith("llm-eval-leaderboard/1.0")
    assert captured["body"]["model"] == "gemini-3.5-flash-lite"
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["messages"] == [
        {"role": "user", "content": "Kí ni olú ìlú Nàìjíríà?"}
    ]


def test_groq_request_construction(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["user_agent"] = request.get_header("User-agent")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(_completion_payload("Groq answer."))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    provider = GroqProvider()
    assert provider.model == "openai/gpt-oss-120b"
    assert provider.endpoint == "https://api.groq.com/openai/v1/chat/completions"
    assert provider.complete("Hello?") == "Groq answer."
    assert captured["url"] == provider.endpoint
    assert captured["auth"] == "Bearer test-groq-key"
    # Groq sits behind Cloudflare, which blocks the default
    # Python-urllib User-Agent (error 1010); the named client
    # User-Agent is required.
    assert captured["user_agent"].startswith("llm-eval-leaderboard/1.0")
    assert captured["body"]["model"] == "openai/gpt-oss-120b"


def test_missing_gemini_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        GeminiProvider()


def test_missing_groq_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        GroqProvider()


def test_missing_key_raises_before_any_request(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        return FakeResponse(_completion_payload())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        GeminiProvider()
    assert calls == []


def test_http_error_carries_status_code_and_no_key(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 503, "Service Unavailable", {}, None
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ProviderError) as exc_info:
        _post_chat_completions(
            "https://example.invalid/v1/chat/completions",
            "secret-key-xyz", "some-model", "prompt", 0.0, 10,
        )
    assert exc_info.value.status_code == 503
    assert "secret-key-xyz" not in str(exc_info.value)


def test_429_is_typed_as_provider_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 429, "Too Many Requests", {}, None
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ProviderError) as exc_info:
        _post_chat_completions(
            "https://example.invalid/v1/chat/completions",
            "k", "m", "prompt", 0.0, 10,
        )
    assert exc_info.value.status_code == 429


def test_malformed_response_raises(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse({"no": "choices here"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ProviderError, match="no usable completion"):
        _post_chat_completions(
            "https://example.invalid/v1/chat/completions",
            "k", "m", "prompt", 0.0, 10,
        )


def test_env_overrides_for_endpoint_and_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom")
    monkeypatch.setenv("GEMINI_ENDPOINT", "https://example.invalid/custom")
    provider = GeminiProvider()
    assert provider.model == "gemini-custom"
    assert provider.endpoint == "https://example.invalid/custom"
