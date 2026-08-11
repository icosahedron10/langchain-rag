from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest
from pydantic import ValidationError

import ragchat.telemetry as telemetry_module
from ragchat.config import Settings
from ragchat.telemetry import EVIDENCE_TEXT_LIMIT, Telemetry, record_retrieval_digest


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


class RecordingClient:
    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        self.updates: list[dict[str, Any]] = []

    def update_run(self, run_id: str, **kwargs: Any) -> None:
        self.updates.append({"run_id": run_id, **kwargs})

    def close(self) -> None:
        pass


def enabled_telemetry(monkeypatch: pytest.MonkeyPatch) -> Telemetry:
    monkeypatch.setattr(telemetry_module, "Client", RecordingClient)
    monkeypatch.setattr(telemetry_module, "tracing_context", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(
        telemetry_module,
        "get_current_run_tree",
        lambda: type("FakeRunTree", (), {"trace_id": "root-run-id"})(),
    )

    return Telemetry(settings(langsmith_tracing=True, langsmith_api_key="secret"))


def test_retrieval_digest_unions_every_search_corpus_call_in_one_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = enabled_telemetry(monkeypatch)

    with telemetry.trace_chat("session-3"):
        record_retrieval_digest(evidence_text="", pages=[], answerable=False)
        record_retrieval_digest(
            evidence_text="Page four wording.",
            pages=[4, None, 4],
            answerable=True,
        )
        record_retrieval_digest(
            evidence_text="Page nine wording.",
            pages=[9],
            answerable=False,
        )

    assert telemetry._client.updates == [
        {
            "run_id": "root-run-id",
            "extra": {
                "metadata": {
                    "session_id": "session-3",
                    "retrieval_digest": {
                        "evidence_text": "Page four wording.\n\nPage nine wording.",
                        "pages": [4, 9],
                        "answerable": True,
                        "search_corpus_calls": 3,
                    },
                }
            },
        }
    ]


def test_retrieval_digest_truncates_evidence_text(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = enabled_telemetry(monkeypatch)

    with telemetry.trace_chat("session-4"):
        record_retrieval_digest(
            evidence_text="e" * (EVIDENCE_TEXT_LIMIT + 500),
            pages=[1],
            answerable=True,
        )

    digest = telemetry._client.updates[0]["extra"]["metadata"]["retrieval_digest"]
    assert len(digest["evidence_text"]) == EVIDENCE_TEXT_LIMIT


def test_turns_without_retrieval_or_tracing_record_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = enabled_telemetry(monkeypatch)

    with telemetry.trace_chat("session-5"):
        pass

    assert telemetry._client.updates == []

    monkeypatch.setattr(telemetry_module, "tracing_context", lambda **_kwargs: nullcontext())
    disabled = Telemetry(settings())
    record_retrieval_digest(evidence_text="outside any turn", pages=[1], answerable=True)
    with disabled.trace_chat("session-6"):
        record_retrieval_digest(evidence_text="tracing disabled", pages=[2], answerable=True)

    assert disabled._client is None
