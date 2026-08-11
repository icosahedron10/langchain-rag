from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest
from pydantic import ValidationError

import ragchat.telemetry as telemetry_module
from ragchat.config import Settings
from ragchat.telemetry import Telemetry


class RecordingRun:
    """Stands in for the run tree the tracer keeps while the root run is open."""

    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.append(metadata)


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
    with telemetry.trace_chat("session-1") as trace:
        assert trace.run_id is None
        assert trace.tracer is None
    telemetry.close()

    assert calls == [
        {
            "project_name": None,
            "tags": ["ragchat", "chat"],
            "metadata": {
                "session_id": "session-1",
                "environment": "local",
                "model_backend": "vllm",
                "sandbox_mode": "disabled",
            },
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
    with telemetry.trace_chat("session-2") as trace:
        assert trace.run_id is not None
        assert trace.tracer is not None
        assert trace.tracer.client is telemetry._client
        assert trace.tracer.project_name == "ragchat-test"
    telemetry.close()

    assert client_calls == [
        {
            "api_url": "https://langsmith.test",
            "api_key": "secret",
            "hide_metadata": telemetry._clean_metadata,
        }
    ]
    assert context_calls[0] == {
        "project_name": "ragchat-test",
        "tags": ["ragchat", "chat"],
        "metadata": {
            "session_id": "session-2",
            "environment": "local",
            "model_backend": "vllm",
            "sandbox_mode": "disabled",
        },
        "enabled": True,
        "client": telemetry._client,
    }
    assert telemetry._client.closed is True


def test_startup_metadata_reports_the_configured_environment_and_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        telemetry_module,
        "tracing_context",
        lambda **kwargs: context_calls.append(kwargs) or nullcontext(),
    )

    telemetry = Telemetry(
        settings(
            environment="staging",
            model_backend="openai",
            openai_api_key="openai-secret",
            sandbox_mode="docker",
        )
    )
    with telemetry.trace_chat("session-3"):
        pass

    assert context_calls[0]["metadata"] == {
        "session_id": "session-3",
        "environment": "staging",
        "model_backend": "openai",
        "sandbox_mode": "docker",
    }


def test_run_metadata_drops_the_langsmith_env_vars_the_sdk_copies_in() -> None:
    telemetry = Telemetry(settings())

    cleaned = telemetry._clean_metadata(
        {
            "session_id": "session-4",
            "ls_provider": "openai",
            "LANGSMITH_ENDPOINT": "https://langsmith.test",
            "LANGSMITH_PROJECT": "ragchat",
            "LANGSMITH_TRACING": "true",
        }
    )

    assert cleaned["session_id"] == "session-4"
    assert cleaned["ls_provider"] == "openai"
    assert not [key for key in cleaned if key.startswith("LANGSMITH_")]


@pytest.mark.parametrize("dirty", [False, True])
def test_run_metadata_reports_a_commit_sha_and_a_separate_dirty_flag(
    monkeypatch: pytest.MonkeyPatch,
    dirty: bool,
) -> None:
    commit = "a" * 40
    monkeypatch.setattr(
        telemetry_module,
        "get_git_info",
        lambda: {"commit": commit, "dirty": dirty},
    )

    cleaned = Telemetry(settings())._clean_metadata({"revision_id": "c77f80b-dirty"})

    assert cleaned == {"revision_id": commit, "dirty_worktree": dirty}


def test_retrieval_metadata_reaches_the_open_root_chat_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry_module, "Client", lambda **_kwargs: object())
    monkeypatch.setattr(telemetry_module, "tracing_context", lambda **_kwargs: nullcontext())

    telemetry = Telemetry(settings(langsmith_tracing=True, langsmith_api_key="secret"))
    signals = {"answerable": True, "n_sources": 2}

    telemetry_module.record_chat_metadata(signals)
    with telemetry.trace_chat("session-5") as trace:
        assert trace.tracer is not None
        root_run = RecordingRun()
        trace.tracer.run_map[str(trace.run_id)] = root_run
        telemetry_module.record_chat_metadata(signals)
    telemetry_module.record_chat_metadata({"answerable": False})

    assert root_run.metadata == [signals]


def test_feedback_is_recorded_against_the_run_and_never_a_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feedback_calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def create_feedback(self, **kwargs: Any) -> None:
            feedback_calls.append(kwargs)

    monkeypatch.setattr(telemetry_module, "Client", FakeClient)

    telemetry = Telemetry(settings(langsmith_tracing=True, langsmith_api_key="secret"))
    telemetry.record_feedback("run-1", 0)

    assert feedback_calls == [{"run_id": "run-1", "key": "user_rating", "score": 0}]


def test_feedback_is_dropped_when_tracing_is_disabled() -> None:
    Telemetry(settings()).record_feedback("run-1", 1)
