from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from litestar.datastructures import State

import ragchat.manager as manager_module
from conftest import FakePipeline, ScriptedChatModel
from ragchat.config import SandboxMode, Settings
from ragchat.controller import AgentController, ChatRequest
from ragchat.domain import (
    ArtifactEvent,
    DoneEvent,
    ErrorEvent,
    MessageDelta,
    ProgressEvent,
    SessionBusyError,
    SessionNotFoundError,
    StartupValidationError,
)
from ragchat.manager import STREAM_ERROR_MESSAGE, DeepAgentManager, RuntimeComponents
from ragchat.retrieval import RetrievedPassage


def settings(*, sandbox_mode: SandboxMode = SandboxMode.DISABLED) -> Settings:
    return Settings(
        vllm_base_url="http://vllm.test/v1",
        vllm_model="test-model",
        qdrant_collection="test-corpus",
        sandbox_mode=sandbox_mode,
        _env_file=None,
    )


class RecordingSaver:
    def __init__(self, failures_remaining: int = 0) -> None:
        self.deleted: list[str] = []
        self.attempted: list[str] = []
        self.failures_remaining = failures_remaining

    async def adelete_thread(self, session_id: str) -> None:
        self.attempted.append(session_id)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("checkpoint deletion failed")
        self.deleted.append(session_id)


class ScriptedOrchestrator:
    def __init__(self, items: list[Any] | None = None) -> None:
        self.items = items or []
        self.calls: list[dict[str, Any]] = []

    async def astream(
        self,
        input_value: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: list[str],
        subgraphs: bool,
    ) -> AsyncIterator[tuple[tuple[str, ...], str, Any]]:
        self.calls.append(
            {
                "input": input_value,
                "config": config,
                "stream_mode": stream_mode,
                "subgraphs": subgraphs,
            }
        )
        for item in self.items:
            if isinstance(item, Exception):
                raise item
            yield item


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, str] | str] = []

    @contextmanager
    def trace_chat(self, session_id: str) -> Any:
        self.events.append(("enter", session_id))
        try:
            yield
        finally:
            self.events.append(("exit", session_id))

    def close(self) -> None:
        self.events.append("close")


@dataclass
class RecordingSandboxSession:
    close_count: int = 0
    failures_remaining: int = 0

    async def close(self) -> None:
        self.close_count += 1
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("sandbox close failed")


@dataclass
class RecordingHandle:
    session: RecordingSandboxSession


def make_manager(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator: ScriptedOrchestrator,
    *,
    sandbox_mode: SandboxMode = SandboxMode.DISABLED,
    handle_factory: Callable[[str], Any] | None = None,
    saver: RecordingSaver | None = None,
) -> DeepAgentManager:
    model = ScriptedChatModel(messages=iter([]))
    components = RuntimeComponents(
        chat_model=model,
        pipeline=FakePipeline(),
        sandbox_handle_factory=handle_factory,
    )
    monkeypatch.setattr(manager_module, "build_search_corpus_tool", lambda *_: object())
    monkeypatch.setattr(manager_module, "build_orchestrator", lambda *_: orchestrator)
    return DeepAgentManager(
        settings(sandbox_mode=sandbox_mode),
        components,
        cast("Any", saver or RecordingSaver()),
    )


@pytest.mark.asyncio
async def test_create_uses_one_saver_and_the_same_model_for_both_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedChatModel(messages=iter([]))
    pipeline = FakePipeline()
    components = RuntimeComponents(chat_model=model, pipeline=pipeline)
    saver = RecordingSaver()
    tool = object()
    orchestrator = ScriptedOrchestrator()
    search_calls: list[tuple[object, object, object]] = []
    orchestrator_calls: list[tuple[object, object, object, object, object]] = []
    saver_builds: list[None] = []

    def build_saver() -> RecordingSaver:
        saver_builds.append(None)
        return saver

    def build_search(
        received_model: object, received_pipeline: object, corpus_description: str
    ) -> object:
        search_calls.append((received_model, received_pipeline, corpus_description))
        return tool

    def build_graph(
        received_model: object,
        received_tool: object,
        received_saver: object,
        corpus_description: str,
        sandbox_backend: object,
    ) -> ScriptedOrchestrator:
        orchestrator_calls.append(
            (received_model, received_tool, received_saver, corpus_description, sandbox_backend)
        )
        return orchestrator

    monkeypatch.setattr(manager_module, "InMemorySaver", build_saver)
    monkeypatch.setattr(manager_module, "build_search_corpus_tool", build_search)
    monkeypatch.setattr(manager_module, "build_orchestrator", build_graph)

    manager = await DeepAgentManager.create(settings(), components)

    assert manager._orchestrator is orchestrator
    assert saver_builds == [None]
    configured = settings()
    assert search_calls == [(model, pipeline, configured.corpus_description)]
    assert orchestrator_calls == [(model, tool, saver, configured.corpus_description, None)]


@pytest.mark.parametrize(
    ("sandbox_mode", "factory"),
    [
        (SandboxMode.DOCKER, None),
        (SandboxMode.DISABLED, lambda _session_id: object()),
    ],
)
def test_component_sandbox_mismatch_fails_before_graph_build(
    monkeypatch: pytest.MonkeyPatch,
    sandbox_mode: SandboxMode,
    factory: Callable[[str], object] | None,
) -> None:
    graph_builds: list[None] = []
    monkeypatch.setattr(
        manager_module,
        "build_search_corpus_tool",
        lambda *_: graph_builds.append(None),
    )
    components = RuntimeComponents(
        chat_model=cast("BaseChatModel", object()),
        pipeline=FakePipeline(),
        sandbox_handle_factory=cast("Any", factory),
    )

    with pytest.raises(StartupValidationError):
        DeepAgentManager(
            settings(sandbox_mode=sandbox_mode),
            components,
            cast("Any", RecordingSaver()),
        )

    assert graph_builds == []


@pytest.mark.asyncio
async def test_stream_uses_exact_nested_args_validates_custom_and_filters_model_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ScriptedOrchestrator(
        [
            ((), "custom", {"type": "progress", "text": "Searching"}),
            (
                ("tools:inner",),
                "messages",
                (AIMessageChunk(content="hidden retrieval text"), {}),
            ),
            ((), "messages", (HumanMessage("not an AI chunk"), {})),
            ((), "messages", (AIMessageChunk(content=""), {})),
            ((), "messages", (AIMessageChunk(content="  "), {})),
            ((), "messages", (AIMessageChunk(content="answer"), {})),
            (
                ("tools:sandbox",),
                "custom",
                {
                    "type": "artifact",
                    "name": "chart.png",
                    "media_type": "image/png",
                    "data": "cG5n",
                },
            ),
            ((), "updates", {"ignored": True}),
        ]
    )
    manager = make_manager(monkeypatch, orchestrator)
    session_id = manager.create_session()

    stream = await manager.stream_chat(session_id, "user question")
    events = [event async for event in stream]

    assert events == [
        ProgressEvent(text="Searching"),
        MessageDelta(text="  "),
        MessageDelta(text="answer"),
        ArtifactEvent(
            name="chart.png",
            media_type="image/png",
            data="cG5n",
        ),
        DoneEvent(),
    ]
    assert len(orchestrator.calls) == 1
    call = orchestrator.calls[0]
    assert call["config"] == {"configurable": {"thread_id": session_id}}
    assert call["stream_mode"] == ["messages", "custom"]
    assert call["subgraphs"] is True
    messages = call["input"]["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "user question"
    assert manager._sessions[session_id].lock.locked() is False


@pytest.mark.asyncio
async def test_lock_is_acquired_before_return_and_released_when_stream_closes_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ScriptedOrchestrator(
        [
            ((), "custom", {"type": "progress", "text": "first"}),
            ((), "custom", {"type": "progress", "text": "second"}),
        ]
    )
    manager = make_manager(monkeypatch, orchestrator)
    session_id = manager.create_session()

    stream = await manager.stream_chat(session_id, "first request")
    assert manager._sessions[session_id].lock.locked() is True
    with pytest.raises(SessionBusyError):
        await manager.stream_chat(session_id, "overlap")
    with pytest.raises(SessionBusyError):
        await manager.delete_session(session_id)

    assert await anext(stream) == ProgressEvent(text="first")
    await stream.aclose()

    assert manager._sessions[session_id].lock.locked() is False


@pytest.mark.asyncio
async def test_trace_context_starts_on_consumption_and_closes_with_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = RecordingTelemetry()
    monkeypatch.setattr(manager_module, "Telemetry", lambda _settings: telemetry)
    manager = make_manager(
        monkeypatch,
        ScriptedOrchestrator(
            [
                ((), "custom", {"type": "progress", "text": "first"}),
                ((), "custom", {"type": "progress", "text": "second"}),
            ]
        ),
    )
    session_id = manager.create_session()

    stream = await manager.stream_chat(session_id, "question")
    assert telemetry.events == []
    assert await anext(stream) == ProgressEvent(text="first")
    assert telemetry.events == [("enter", session_id)]

    await stream.aclose()

    assert telemetry.events == [("enter", session_id), ("exit", session_id)]


@pytest.mark.asyncio
async def test_unstarted_stream_close_releases_the_preacquired_session_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ScriptedOrchestrator(
        [((), "custom", {"type": "progress", "text": "never consumed"})]
    )
    manager = make_manager(monkeypatch, orchestrator)
    session_id = manager.create_session()

    stream = await manager.stream_chat(session_id, "abandoned request")
    assert manager._sessions[session_id].lock.locked() is True

    await stream.aclose()
    await stream.aclose()

    assert manager._sessions[session_id].lock.locked() is False
    assert orchestrator.calls == []


@pytest.mark.asyncio
async def test_immediate_asgi_disconnect_closes_unstarted_manager_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ScriptedOrchestrator(
        [((), "custom", {"type": "progress", "text": "never consumed"})]
    )
    manager = make_manager(monkeypatch, orchestrator)
    session_id = manager.create_session()
    response = await AgentController.chat.fn(
        None,
        State({"manager": manager}),
        session_id,
        ChatRequest(message="disconnect immediately"),
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asgi_response = response.to_asgi_response(None, cast("Any", None))
    await asgi_response(
        cast("Any", {"type": "http"}),
        cast("Any", receive),
        cast("Any", send),
    )

    assert sent[0]["type"] == "http.response.start"
    assert orchestrator.calls == []
    assert manager._sessions[session_id].lock.locked() is False


@pytest.mark.asyncio
async def test_different_sessions_can_hold_active_streams_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ScriptedOrchestrator([((), "custom", {"type": "progress", "text": "started"})])
    manager = make_manager(monkeypatch, orchestrator)
    first_id = manager.create_session()
    second_id = manager.create_session()

    first_stream = await manager.stream_chat(first_id, "first")
    second_stream = await manager.stream_chat(second_id, "second")

    assert manager._sessions[first_id].lock.locked() is True
    assert manager._sessions[second_id].lock.locked() is True
    assert await anext(first_stream) == ProgressEvent(text="started")
    assert await anext(second_stream) == ProgressEvent(text="started")
    await first_stream.aclose()
    await second_stream.aclose()

    assert manager._sessions[first_id].lock.locked() is False
    assert manager._sessions[second_id].lock.locked() is False


@pytest.mark.asyncio
async def test_missing_sessions_fail_before_a_stream_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = make_manager(monkeypatch, ScriptedOrchestrator())

    with pytest.raises(SessionNotFoundError):
        await manager.stream_chat("missing", "hello")
    with pytest.raises(SessionNotFoundError):
        await manager.delete_session("missing")


@pytest.mark.asyncio
async def test_stream_exception_yields_only_fixed_generic_error_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = ScriptedOrchestrator(
        [RuntimeError("secret-api-key must never leave the server")]
    )
    manager = make_manager(monkeypatch, orchestrator)
    session_id = manager.create_session()

    events = [event async for event in await manager.stream_chat(session_id, "trigger failure")]

    assert events == [ErrorEvent(message=STREAM_ERROR_MESSAGE)]
    assert "secret-api-key" not in events[0].message
    assert manager._sessions[session_id].lock.locked() is False


@pytest.mark.asyncio
async def test_invalid_custom_payload_becomes_generic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = make_manager(
        monkeypatch,
        ScriptedOrchestrator([((), "custom", {"type": "progress"})]),
    )
    session_id = manager.create_session()

    events = [event async for event in await manager.stream_chat(session_id, "hello")]

    assert events == [ErrorEvent(message=STREAM_ERROR_MESSAGE)]
    assert manager._sessions[session_id].lock.locked() is False


@pytest.mark.asyncio
async def test_delete_closes_owned_sandbox_and_deletes_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handles: dict[str, RecordingHandle] = {}

    def factory(session_id: str) -> RecordingHandle:
        handle = RecordingHandle(RecordingSandboxSession())
        handles[session_id] = handle
        return handle

    saver = RecordingSaver()
    manager = make_manager(
        monkeypatch,
        ScriptedOrchestrator(),
        sandbox_mode=SandboxMode.DOCKER,
        handle_factory=factory,
        saver=saver,
    )
    session_id = manager.create_session()

    await manager.delete_session(session_id)

    assert handles[session_id].session.close_count == 1
    assert saver.deleted == [session_id]
    assert session_id not in manager._sessions
    with pytest.raises(SessionNotFoundError):
        await manager.delete_session(session_id)


@pytest.mark.asyncio
async def test_failed_checkpoint_deletion_retains_busy_session_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handles: dict[str, RecordingHandle] = {}

    def factory(session_id: str) -> RecordingHandle:
        handle = RecordingHandle(RecordingSandboxSession())
        handles[session_id] = handle
        return handle

    saver = RecordingSaver(failures_remaining=1)
    manager = make_manager(
        monkeypatch,
        ScriptedOrchestrator(),
        sandbox_mode=SandboxMode.DOCKER,
        handle_factory=factory,
        saver=saver,
    )
    session_id = manager.create_session()

    with pytest.raises(RuntimeError, match="checkpoint deletion failed"):
        await manager.delete_session(session_id)

    retained = manager._sessions[session_id]
    assert retained.deleting is True
    assert retained.sandbox_closed is True
    assert retained.checkpoint_deleted is False
    assert retained.lock.locked() is False
    assert handles[session_id].session.close_count == 1
    assert saver.attempted == [session_id]
    with pytest.raises(SessionBusyError):
        await manager.stream_chat(session_id, "must not route")

    await manager.delete_session(session_id)

    assert handles[session_id].session.close_count == 1
    assert saver.attempted == [session_id, session_id]
    assert saver.deleted == [session_id]
    assert session_id not in manager._sessions


@pytest.mark.asyncio
async def test_failed_sandbox_close_retains_session_after_checkpoint_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handles: dict[str, RecordingHandle] = {}

    def factory(session_id: str) -> RecordingHandle:
        handle = RecordingHandle(RecordingSandboxSession(failures_remaining=1))
        handles[session_id] = handle
        return handle

    saver = RecordingSaver()
    manager = make_manager(
        monkeypatch,
        ScriptedOrchestrator(),
        sandbox_mode=SandboxMode.DOCKER,
        handle_factory=factory,
        saver=saver,
    )
    session_id = manager.create_session()

    with pytest.raises(RuntimeError, match="sandbox close failed"):
        await manager.delete_session(session_id)

    retained = manager._sessions[session_id]
    assert retained.sandbox_closed is False
    assert retained.checkpoint_deleted is True
    assert saver.attempted == [session_id]

    await manager.delete_session(session_id)

    assert handles[session_id].session.close_count == 2
    assert saver.attempted == [session_id]
    assert session_id not in manager._sessions


@pytest.mark.asyncio
async def test_shutdown_closes_and_forgets_every_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handles: dict[str, RecordingHandle] = {}

    def factory(session_id: str) -> RecordingHandle:
        handle = RecordingHandle(RecordingSandboxSession())
        handles[session_id] = handle
        return handle

    saver = RecordingSaver()
    manager = make_manager(
        monkeypatch,
        ScriptedOrchestrator(),
        sandbox_mode=SandboxMode.DOCKER,
        handle_factory=factory,
        saver=saver,
    )
    session_ids = [manager.create_session() for _ in range(3)]

    await manager.shutdown()

    assert manager._sessions == {}
    assert saver.deleted == session_ids
    assert all(handles[session_id].session.close_count == 1 for session_id in session_ids)


@pytest.mark.asyncio
async def test_shutdown_closes_telemetry_when_session_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = RecordingTelemetry()
    monkeypatch.setattr(manager_module, "Telemetry", lambda _settings: telemetry)
    manager = make_manager(
        monkeypatch,
        ScriptedOrchestrator(),
        saver=RecordingSaver(failures_remaining=1),
    )
    manager.create_session()

    with pytest.raises(ExceptionGroup, match="Failed to fully shut down"):
        await manager.shutdown()

    assert telemetry.events == ["close"]


@pytest.mark.asyncio
async def test_shutdown_retains_failed_sessions_and_retries_only_unfinished_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handles: dict[str, RecordingHandle] = {}

    def factory(session_id: str) -> RecordingHandle:
        handle = RecordingHandle(RecordingSandboxSession())
        handles[session_id] = handle
        return handle

    saver = RecordingSaver(failures_remaining=1)
    manager = make_manager(
        monkeypatch,
        ScriptedOrchestrator(),
        sandbox_mode=SandboxMode.DOCKER,
        handle_factory=factory,
        saver=saver,
    )
    first_id, second_id = (manager.create_session() for _ in range(2))

    with pytest.raises(ExceptionGroup, match="Failed to fully shut down"):
        await manager.shutdown()

    assert list(manager._sessions) == [first_id]
    assert manager._sessions[first_id].deleting is True
    assert handles[first_id].session.close_count == 1
    assert handles[second_id].session.close_count == 1
    assert saver.deleted == [second_id]

    await manager.shutdown()

    assert manager._sessions == {}
    assert handles[first_id].session.close_count == 1
    assert saver.deleted == [second_id, first_id]


def test_session_ids_are_unique_uuid_hex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = make_manager(monkeypatch, ScriptedOrchestrator())

    session_ids = {manager.create_session() for _ in range(10)}

    assert len(session_ids) == 10
    assert all(len(session_id) == 32 for session_id in session_ids)
    assert all(int(session_id, 16) >= 0 for session_id in session_ids)


def tool_call(name: str, arguments: dict[str, Any], call_id: str, content: str = "") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": arguments, "id": call_id}],
    )


@pytest.mark.asyncio
async def test_full_nested_flow_streams_only_root_text_and_keeps_inner_state_transient() -> None:
    model = ScriptedChatModel(
        messages=iter(
            [
                tool_call(
                    "search_corpus",
                    {"question": "What does the policy say?"},
                    "outer-search",
                ),
                tool_call(
                    "qdrant_hybrid_search",
                    {"query": "policy exact wording"},
                    "inner-search",
                    content="hidden retrieval reasoning",
                ),
                tool_call(
                    "RetrievalResult",
                    {
                        "answerable": True,
                        "summary": "The policy requires review.",
                        "selected_point_ids": ["point-1"],
                        "gaps": [],
                    },
                    "inner-result",
                    content="hidden selection reasoning",
                ),
                AIMessage(content="final answer text"),
            ]
        )
    )
    pipeline = FakePipeline([RetrievedPassage("point-1", "policy.pdf", 7, "Review is required.")])
    manager = await DeepAgentManager.create(
        settings(),
        RuntimeComponents(chat_model=model, pipeline=pipeline),
    )
    session_id = manager.create_session()

    events = [
        event
        async for event in await manager.stream_chat(
            session_id,
            "What does the policy say?",
        )
    ]

    assert pipeline.queries == ["policy exact wording"]
    assert [event.text for event in events if isinstance(event, ProgressEvent)] == [
        'Searching for: "policy exact wording"',
        "Reviewing the retrieved passages…",
        "Preparing an evidence-grounded answer…",
    ]
    assert "".join(event.text for event in events if isinstance(event, MessageDelta)) == (
        "final answer text"
    )
    assert "hidden retrieval reasoning" not in repr(events)
    assert "hidden selection reasoning" not in repr(events)
    assert isinstance(events[-1], DoneEvent)

    config = {"configurable": {"thread_id": session_id}}
    snapshot = await manager._orchestrator.aget_state(config)
    persisted_messages = snapshot.values["messages"]
    persisted_tool_names = [
        call["name"]
        for message in persisted_messages
        for call in getattr(message, "tool_calls", [])
    ]
    persisted_text = "\n".join(str(message.content) for message in persisted_messages)
    assert persisted_tool_names == ["search_corpus"]
    assert "Searching for:" not in persisted_text
    assert "hidden retrieval reasoning" not in persisted_text
    assert "hidden selection reasoning" not in persisted_text

    await manager.shutdown()
