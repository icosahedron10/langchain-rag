"""Optional LangSmith tracing lifecycle."""

from __future__ import annotations

from contextlib import AbstractContextManager

from langsmith import Client
from langsmith.run_helpers import tracing_context

from ragchat.config import Settings


class Telemetry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._closed = False
        self._client = (
            Client(
                api_url=settings.langsmith_endpoint,
                api_key=settings.langsmith_api_key.get_secret_value(),
            )
            if settings.langsmith_tracing and settings.langsmith_api_key is not None
            else None
        )

    def trace_chat(self, session_id: str) -> AbstractContextManager[None]:
        return tracing_context(
            project_name=self._settings.langsmith_project,
            tags=["ragchat", "chat"],
            metadata={"session_id": session_id},
            enabled=self._client is not None,
            client=self._client,
        )

    def close(self) -> None:
        if self._client is not None and not self._closed:
            self._client.close()
            self._closed = True
