"""Domain events and exceptions shared between the manager and the HTTP layer.

This module (and everything the manager imports) must stay free of HTTP
framework types. The controller translates these into SSE frames and status
codes.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ProgressEvent(BaseModel):
    """Transient, human-readable progress. Never persisted or replayed."""

    type: Literal["progress"] = "progress"
    text: str


class MessageDelta(BaseModel):
    """An incremental chunk of assistant answer text."""

    type: Literal["message"] = "message"
    text: str


class ArtifactEvent(BaseModel):
    """An inline displayable image produced in the session workspace."""

    type: Literal["artifact"] = "artifact"
    name: str
    media_type: str
    data: str  # base64-encoded file content


class RetrievedSourcesEvent(BaseModel):
    """Document/page pairs resolved for one turn. Consumed by the manager only."""

    type: Literal["retrieved_sources"] = "retrieved_sources"
    pages: list[tuple[str, int]]


class UnverifiableCitationEvent(BaseModel):
    """Answer citations whose page appears in no source retrieved this turn."""

    type: Literal["unverifiable_citation"] = "unverifiable_citation"
    citations: list[str]


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


DomainEvent = Annotated[
    ProgressEvent
    | MessageDelta
    | ArtifactEvent
    | RetrievedSourcesEvent
    | UnverifiableCitationEvent
    | DoneEvent
    | ErrorEvent,
    Field(discriminator="type"),
]


class RagChatError(Exception):
    """Base class for domain errors."""


class SessionNotFoundError(RagChatError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Unknown session: {session_id}")
        self.session_id = session_id


class SessionBusyError(RagChatError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session {session_id} already has an active request")
        self.session_id = session_id


class StartupValidationError(RagChatError):
    """Raised when configuration or external dependencies are unusable."""


class SandboxUnavailableError(RagChatError):
    """Raised at startup when SANDBOX_MODE=docker but Docker is unusable."""
