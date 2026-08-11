from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest
from pydantic import ValidationError

import ragchat.telemetry as telemetry_module
from ragchat.config import Settings
from ragchat.telemetry import Telemetry


def settings(**overrides: Any) -> Settings:
    values = {
        "langsmith_tracing": False,
        "langsmith_api_key": None,
        "langsmith_endpoint": "https://api.smith.langchain.com",
        "langsmith_project": None,
        **overrides,
    }
    return Settings(
        _env_file=None,
        vllm_base_url="http://vllm.test/v1",
        vllm_model="test-model",
        qdrant_collection="test-corpus",
        **values,
    )


def test_tracing_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="LANGSMITH_API_KEY"):
        settings(langsmith_tracing=True)


def test_settings_map_langsmith_environment_and_redact_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "environment-secret")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://langsmith.test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "environment-project")

    configured = Settings(
        _env_file=None,
        model_backend="vllm",
        vllm_base_url="http://vllm.test/v1",
        vllm_model="test-model",
        qdrant_collection="test-corpus",
    )

    assert configured.langsmith_tracing is True
    assert configured.langsmith_endpoint == "https://langsmith.test"
    assert configured.langsmith_project == "environment-project"
    assert configured.langsmith_api_key is not None
    assert configured.langsmith_api_key.get_secret_value() == "environment-secret"
    assert "environment-secret" not in repr(configured)


def test_citation_checks_count_the_unverifiable_citation_rate() -> None:
    telemetry = Telemetry(settings())

    assert telemetry.unverifiable_citation_rate == 0.0

    telemetry.record_citation_check(unverifiable=False)
    telemetry.record_citation_check(unverifiable=True)

    assert telemetry.answers_checked == 2
    assert telemetry.answers_with_unverifiable_citations == 1
    assert telemetry.unverifiable_citation_rate == 0.5


def test_disabled_mode_constructs_no_client_and_explicitly_disables_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        telemetry_module,
        "Client",
        lambda **_kwargs: pytest.fail("disabled telemetry constructed a client"),
    )
    monkeypatch.setattr(
        telemetry_module,
        "tracing_context",
        lambda **kwargs: calls.append(kwargs) or nullcontext(),
    )

    telemetry = Telemetry(settings())
    with telemetry.trace_chat("session-1"):
        pass
    telemetry.close()

    assert calls == [
        {
            "project_name": None,
            "tags": ["ragchat", "chat"],
            "metadata": {"session_id": "session-1"},
            "enabled": False,
            "client": None,
        }
    ]


def test_enabled_mode_builds_context_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_calls: list[dict[str, Any]] = []
    context_calls: list[dict[str, Any]] = []

    class FakeClient:
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            client_calls.append(kwargs)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(telemetry_module, "Client", FakeClient)
    monkeypatch.setattr(
        telemetry_module,
        "tracing_context",
        lambda **kwargs: context_calls.append(kwargs) or nullcontext(),
    )

    telemetry = Telemetry(
        settings(
            langsmith_tracing=True,
            langsmith_api_key="secret",
            langsmith_endpoint="https://langsmith.test",
            langsmith_project="ragchat-test",
        )
    )
    with telemetry.trace_chat("session-2"):
        pass
    telemetry.close()

    assert client_calls == [{"api_url": "https://langsmith.test", "api_key": "secret"}]
    assert context_calls[0] == {
        "project_name": "ragchat-test",
        "tags": ["ragchat", "chat"],
        "metadata": {"session_id": "session-2"},
        "enabled": True,
        "client": telemetry._client,
    }
    assert telemetry._client.closed is True
