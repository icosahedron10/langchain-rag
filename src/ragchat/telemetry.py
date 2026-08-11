"""Optional LangSmith tracing lifecycle."""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_core.tracers import LangChainTracer
from langsmith import Client
from langsmith.env import get_git_info
from langsmith.run_helpers import tracing_context

from ragchat.config import Settings

# Deployment plumbing that the LangSmith SDK copies out of the environment onto
# every run it ships. It describes where traces go, never what a turn did.
ENV_METADATA_NOISE_KEYS = frozenset(
    {"LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT", "LANGSMITH_TRACING"}
)


@dataclass(frozen=True, slots=True)
class RootRun:
    """Late-bound handle to the root chat run, whose tracer owns the live tree."""

    tracer: LangChainTracer
    run_id: uuid.UUID

    def add_metadata(self, metadata: Mapping[str, Any]) -> None:
        run = self.tracer.run_map.get(str(self.run_id))
        if run is not None:
            run.add_metadata(dict(metadata))


@dataclass(frozen=True, slots=True)
class ChatTrace:
    """One traced chat turn: the root run id to rate, and the tracer that owns it."""

    run_id: uuid.UUID | None = None
    tracer: LangChainTracer | None = None


_ROOT_RUN: ContextVar[RootRun | None] = ContextVar("ragchat_root_run", default=None)


def record_chat_metadata(metadata: Mapping[str, Any]) -> None:
    """Attach per-turn metadata to the traced root chat run, when one is active."""

    root_run = _ROOT_RUN.get()
    if root_run is not None:
        root_run.add_metadata(metadata)


class Telemetry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._closed = False
        git_info = get_git_info()
        self._commit: str | None = git_info.get("commit")
        self._dirty_worktree = bool(git_info.get("dirty"))
        self._client = (
            Client(
                api_url=settings.langsmith_endpoint,
                api_key=settings.langsmith_api_key.get_secret_value(),
                hide_metadata=self._clean_metadata,
            )
            if settings.langsmith_tracing and settings.langsmith_api_key is not None
            else None
        )

    @contextmanager
    def trace_chat(self, session_id: str) -> Iterator[ChatTrace]:
        """Trace one chat turn under a root run the caller can rate afterwards."""

        trace = (
            ChatTrace(
                run_id=uuid.uuid4(),
                tracer=LangChainTracer(
                    client=self._client,
                    project_name=self._settings.langsmith_project,
                ),
            )
            if self._client is not None
            else ChatTrace()
        )
        # The tracer keeps the root run open for the whole turn, so tools can
        # still add per-turn metadata to it long after it started.
        token = (
            _ROOT_RUN.set(RootRun(trace.tracer, trace.run_id))
            if trace.tracer is not None and trace.run_id is not None
            else None
        )
        try:
            with tracing_context(
                project_name=self._settings.langsmith_project,
                tags=["ragchat", "chat"],
                metadata={
                    "session_id": session_id,
                    "environment": self._settings.environment,
                    "model_backend": self._settings.model_backend.value,
                    "sandbox_mode": self._settings.sandbox_mode.value,
                },
                enabled=self._client is not None,
                client=self._client,
            ):
                yield trace
        finally:
            if token is not None:
                _ROOT_RUN.reset(token)

    def record_feedback(self, run_id: str, score: int) -> None:
        """Rate one traced root chat run; feedback always targets a run, never a trace."""

        if self._client is not None:
            self._client.create_feedback(run_id=run_id, key="user_rating", score=score)

    def close(self) -> None:
        if self._client is not None and not self._closed:
            self._client.close()
            self._closed = True

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        cleaned = {
            key: value for key, value in metadata.items() if key not in ENV_METADATA_NOISE_KEYS
        }
        if self._commit is not None:
            # The SDK derives revision_id from `git describe --dirty`, which pins
            # traces to no commit at all; report the commit and the dirt apart.
            cleaned["revision_id"] = self._commit
            cleaned["dirty_worktree"] = self._dirty_worktree
        return cleaned
