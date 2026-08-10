"""Offline HTTP, lifespan, validation, and SSE contract tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from litestar import Litestar
from litestar.testing import TestClient

from ragchat.app import create_app
from ragchat.config import ModelBackend, SandboxMode, Settings
from ragchat.domain import (
    ArtifactEvent,
    DomainEvent,
    DoneEvent,
    ErrorEvent,
    MessageDelta,
    ProgressEvent,
    SessionBusyError,
    SessionNotFoundError,
)


class FakeManager:
    """Small transport-level fake; no agent or external service is constructed."""

    def __init__(self) -> None:
        self.events: list[DomainEvent] = []
        self.stream_error: Exception | None = None
        self.delete_error: Exception | None = None
        self.create_calls = 0
        self.chat_calls: list[tuple[str, str]] = []
        self.delete_calls: list[str] = []
        self.shutdown_calls = 0

    def create_session(self) -> str:
        self.create_calls += 1
        return "session-1"

    async def stream_chat(
        self,
        session_id: str,
        message: str,
    ) -> AsyncIterator[DomainEvent]:
        self.chat_calls.append((session_id, message))
        if self.stream_error is not None:
            raise self.stream_error

        async def stream() -> AsyncIterator[DomainEvent]:
            for event in self.events:
                yield event

        return stream()

    async def delete_session(self, session_id: str) -> None:
        self.delete_calls.append(session_id)
        if self.delete_error is not None:
            raise self.delete_error

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        model_backend=ModelBackend.VLLM,
        vllm_base_url="http://vllm.test/v1",
        vllm_model="test-model",
        qdrant_collection="test-corpus",
        sandbox_mode=SandboxMode.DISABLED,
    )


def _test_app(
    settings: Settings,
    manager: FakeManager,
    factory_calls: list[Settings] | None = None,
) -> Litestar:
    async def manager_factory(received: Settings) -> FakeManager:
        if factory_calls is not None:
            factory_calls.append(received)
        return manager

    return create_app(settings=settings, manager_factory=manager_factory)


def test_lifespan_awaits_factory_and_shuts_down(settings: Settings) -> None:
    manager = FakeManager()
    factory_calls: list[Settings] = []
    client = TestClient(_test_app(settings, manager, factory_calls))

    assert factory_calls == []
    assert manager.shutdown_calls == 0

    with client:
        assert factory_calls == [settings]
        assert client.app.state.manager is manager
        assert manager.shutdown_calls == 0

    assert manager.shutdown_calls == 1


def test_health_create_and_delete_delegate_to_manager(settings: Settings) -> None:
    manager = FakeManager()

    with TestClient(_test_app(settings, manager)) as client:
        health = client.get("/health")
        created = client.post("/sessions")
        deleted = client.delete("/sessions/session-1")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert created.status_code == 201
    assert created.json() == {"session_id": "session-1"}
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert manager.create_calls == 1
    assert manager.delete_calls == ["session-1"]


def test_chat_rejects_an_empty_message_before_delegating(settings: Settings) -> None:
    manager = FakeManager()

    with TestClient(_test_app(settings, manager)) as client:
        response = client.post(
            "/sessions/session-1/chat",
            json={"message": ""},
        )

    assert response.status_code == 400
    assert response.json()["extra"][0]["key"] == "message"
    assert manager.chat_calls == []


def test_chat_serializes_every_domain_event_as_sse(settings: Settings) -> None:
    manager = FakeManager()
    manager.events = [
        ProgressEvent(text='Searching for: "exact query"'),
        MessageDelta(text="answer "),
        ArtifactEvent(
            name="chart.png",
            media_type="image/png",
            data="aGVsbG8=",
        ),
        DoneEvent(),
        ErrorEvent(message="The request failed."),
    ]

    with (
        TestClient(_test_app(settings, manager)) as client,
        client.stream(
            "POST",
            "/sessions/session-1/chat",
            json={"message": "question"},
        ) as response,
    ):
        lines = list(response.iter_lines())

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"

    assert manager.chat_calls == [("session-1", "question")]
    assert len(lines) == 3 * len(manager.events)
    for index, event in enumerate(manager.events):
        offset = index * 3
        assert lines[offset] == f"event: {event.type}"
        assert lines[offset + 1] == f"data: {event.model_dump_json()}"
        assert json.loads(lines[offset + 1].removeprefix("data: ")) == event.model_dump()
        assert lines[offset + 2] == ""


@pytest.mark.parametrize(
    ("error", "status_code", "body"),
    [
        (
            SessionNotFoundError("missing"),
            404,
            {
                "error": "session_not_found",
                "detail": "Unknown session: missing",
            },
        ),
        (
            SessionBusyError("session-1"),
            409,
            {
                "error": "session_busy",
                "detail": "Session session-1 already has an active request",
            },
        ),
    ],
)
def test_chat_translates_pre_stream_domain_exceptions(
    settings: Settings,
    error: Exception,
    status_code: int,
    body: dict[str, str],
) -> None:
    manager = FakeManager()
    manager.stream_error = error

    with TestClient(_test_app(settings, manager)) as client:
        response = client.post(
            "/sessions/session-1/chat",
            json={"message": "question"},
        )

    assert response.status_code == status_code
    assert response.json() == body
    assert manager.chat_calls == [("session-1", "question")]


def test_delete_uses_the_controller_exception_map(settings: Settings) -> None:
    manager = FakeManager()
    manager.delete_error = SessionBusyError("session-1")

    with TestClient(_test_app(settings, manager)) as client:
        response = client.delete("/sessions/session-1")

    assert response.status_code == 409
    assert response.json() == {
        "error": "session_busy",
        "detail": "Session session-1 already has an active request",
    }
    assert manager.delete_calls == ["session-1"]
