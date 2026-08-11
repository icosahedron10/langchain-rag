"""Framework-free orchestration, session lifecycle, and streaming policy."""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import TypeAdapter

from ragchat.agents.orchestrator import build_orchestrator
from ragchat.agents.retrieval import build_search_corpus_tool
from ragchat.config import SandboxMode, Settings
from ragchat.domain import (
    DomainEvent,
    DoneEvent,
    ErrorEvent,
    MessageDelta,
    SessionBusyError,
    SessionNotFoundError,
    StartupValidationError,
)
from ragchat.providers import build_chat_model
from ragchat.retrieval import HybridSearchPipeline, SearchPipeline
from ragchat.retrieval.qdrant import (
    build_qdrant_client,
    build_reranker,
    build_vector_store,
    validate_corpus,
)
from ragchat.telemetry import Telemetry

if TYPE_CHECKING:
    from ragchat.sandbox.backend import SessionSandboxHandle


STREAM_ERROR_MESSAGE = "An unexpected error occurred while processing your request."

_DOMAIN_EVENT_ADAPTER: TypeAdapter[DomainEvent] = TypeAdapter(DomainEvent)


@dataclass(frozen=True, slots=True)
class RuntimeComponents:
    """Startup-built dependencies, injectable as one seam in offline tests."""

    chat_model: BaseChatModel
    pipeline: SearchPipeline
    sandbox_handle_factory: Callable[[str], SessionSandboxHandle] | None = None


async def build_components(settings: Settings) -> RuntimeComponents:
    """Build the one provider and the external read-only retrieval pipeline."""

    chat_model = build_chat_model(settings)
    qdrant_client = build_qdrant_client(settings)
    validate_corpus(qdrant_client, settings)
    vector_store = build_vector_store(settings, qdrant_client)
    reranker = build_reranker(settings)
    pipeline = HybridSearchPipeline(vector_store, reranker)

    sandbox_handle_factory: Callable[[str], SessionSandboxHandle] | None = None
    if settings.sandbox_mode is SandboxMode.DOCKER:
        # Docker and deepagents' filesystem backend remain wholly optional.
        from deepagents.backends import FilesystemBackend

        from ragchat.sandbox.backend import SessionSandboxHandle
        from ragchat.sandbox.docker_session import (
            DockerSandboxSession,
            assert_docker_available,
        )

        await assert_docker_available()
        workspace_root = Path(tempfile.gettempdir()) / "ragchat-workspaces"

        def build_sandbox_handle(session_id: str) -> SessionSandboxHandle:
            workspace = workspace_root / session_id
            workspace.mkdir(parents=True, exist_ok=True)
            return SessionSandboxHandle(
                session=DockerSandboxSession(settings, session_id, workspace),
                files=FilesystemBackend(root_dir=workspace, virtual_mode=True),
                settings=settings,
            )

        sandbox_handle_factory = build_sandbox_handle

    return RuntimeComponents(
        chat_model=chat_model,
        pipeline=pipeline,
        sandbox_handle_factory=sandbox_handle_factory,
    )


@dataclass(slots=True)
class Session:
    """Resources and concurrency state owned by one API session."""

    id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    handle: SessionSandboxHandle | None = None
    deleting: bool = False
    sandbox_closed: bool = False
    checkpoint_deleted: bool = False


class SessionEventStream(AsyncIterator[DomainEvent]):
    """A closeable stream that owns a session lock from construction onward."""

    def __init__(
        self,
        source: AsyncGenerator[DomainEvent, None],
        session_lock: asyncio.Lock,
    ) -> None:
        self._source = source
        self._session_lock = session_lock
        self._close_lock = asyncio.Lock()
        self._closed = False

    def __aiter__(self) -> SessionEventStream:
        return self

    async def __anext__(self) -> DomainEvent:
        if self._closed:
            raise StopAsyncIteration

        try:
            return await anext(self._source)
        except StopAsyncIteration:
            await self.aclose()
            raise
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        """Close the source and release the owned lock, even before iteration."""

        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                await self._source.aclose()
            finally:
                self._session_lock.release()


class DeepAgentManager:
    """Own the agent graph, checkpoints, and all live chat sessions."""

    def __init__(
        self,
        settings: Settings,
        components: RuntimeComponents,
        checkpointer: InMemorySaver,
    ) -> None:
        self._settings = settings
        self._components = components
        self._checkpointer = checkpointer
        self._sessions: dict[str, Session] = {}
        self._telemetry = Telemetry(settings)

        factory = components.sandbox_handle_factory
        if settings.sandbox_mode is SandboxMode.DOCKER and factory is None:
            raise StartupValidationError("SANDBOX_MODE=docker requires a sandbox handle factory.")
        if settings.sandbox_mode is SandboxMode.DISABLED and factory is not None:
            raise StartupValidationError(
                "SANDBOX_MODE=disabled cannot use a sandbox handle factory."
            )

        search_corpus = build_search_corpus_tool(
            components.chat_model,
            components.pipeline,
            settings.corpus_description,
        )
        sandbox_backend = None
        if settings.sandbox_mode is SandboxMode.DOCKER:
            from ragchat.sandbox.backend import SessionRoutingBackend

            sandbox_backend = SessionRoutingBackend(self._resolve_handle)

        self._orchestrator = build_orchestrator(
            components.chat_model,
            search_corpus,
            checkpointer,
            settings.corpus_description,
            sandbox_backend,
        )

    @classmethod
    async def create(
        cls,
        settings: Settings,
        components: RuntimeComponents | None = None,
    ) -> DeepAgentManager:
        """Build one shared runtime and one in-memory checkpoint saver."""

        runtime = components if components is not None else await build_components(settings)
        return cls(settings, runtime, InMemorySaver())

    def create_session(self) -> str:
        """Create a conversation and its cheap, still-lazy sandbox handle."""

        session_id = uuid.uuid4().hex
        factory = self._components.sandbox_handle_factory
        handle = factory(session_id) if factory is not None else None
        self._sessions[session_id] = Session(id=session_id, handle=handle)
        return session_id

    async def stream_chat(
        self,
        session_id: str,
        message: str,
    ) -> SessionEventStream:
        """Acquire the session immediately and return its event stream."""

        session = self._require_session(session_id)
        if session.deleting or session.lock.locked():
            raise SessionBusyError(session_id)
        await session.lock.acquire()

        async def event_stream() -> AsyncGenerator[DomainEvent, None]:
            try:
                with self._telemetry.trace_chat(session_id):
                    async for namespace, mode, payload in self._orchestrator.astream(
                        {"messages": [HumanMessage(message)]},
                        config={"configurable": {"thread_id": session_id}},
                        stream_mode=["messages", "custom"],
                        subgraphs=True,
                    ):
                        if mode == "custom":
                            yield _DOMAIN_EVENT_ADAPTER.validate_python(payload)
                            continue

                        if mode != "messages" or namespace != ():
                            continue
                        chunk, _metadata = payload
                        if isinstance(chunk, AIMessageChunk) and chunk.text:
                            yield MessageDelta(text=chunk.text)

                    yield DoneEvent()
            except Exception:
                yield ErrorEvent(message=STREAM_ERROR_MESSAGE)

        return SessionEventStream(event_stream(), session.lock)

    async def delete_session(self, session_id: str) -> None:
        """Destroy a non-busy session and remove all of its checkpoints."""

        session = self._require_session(session_id)
        if session.lock.locked():
            raise SessionBusyError(session_id)

        session.deleting = True
        await session.lock.acquire()
        try:
            await self._cleanup_session(session)
            self._sessions.pop(session_id)
        finally:
            session.lock.release()

    async def shutdown(self) -> None:
        """Clean up every session, retaining any whose cleanup needs a retry."""

        failures: list[Exception] = []
        for session_id in list(self._sessions):
            try:
                await self.delete_session(session_id)
            except Exception as exc:
                failures.append(exc)

        try:
            self._telemetry.close()
        except Exception as exc:
            failures.append(exc)

        if failures:
            raise ExceptionGroup("Failed to fully shut down the agent manager", failures)

    async def _cleanup_session(self, session: Session) -> None:
        failures: list[Exception] = []

        if not session.sandbox_closed:
            try:
                if session.handle is not None:
                    await session.handle.session.close()
            except Exception as exc:
                failures.append(exc)
            else:
                session.sandbox_closed = True

        if not session.checkpoint_deleted:
            try:
                await self._checkpointer.adelete_thread(session.id)
            except Exception as exc:
                failures.append(exc)
            else:
                session.checkpoint_deleted = True

        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup(f"Failed to delete session {session.id}", failures)

    def _require_session(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def _resolve_handle(self, session_id: str) -> SessionSandboxHandle:
        session = self._require_session(session_id)
        if session.deleting:
            raise SessionBusyError(session_id)
        if session.handle is None:
            raise RuntimeError(f"Session {session_id} has no sandbox handle")
        return session.handle
