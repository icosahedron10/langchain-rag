"""HTTP routing, validation, and SSE serialization for ragchat."""

from __future__ import annotations

from collections.abc import AsyncIterator

from litestar import Controller, Request, Response, delete, get, post
from litestar.datastructures import State
from litestar.response import ServerSentEvent, ServerSentEventMessage
from litestar.status_codes import HTTP_200_OK, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT
from pydantic import BaseModel, Field

from ragchat.domain import SessionBusyError, SessionNotFoundError


class ChatRequest(BaseModel):
    """Validated request body for one chat turn."""

    message: str = Field(min_length=1)


def _handle_session_not_found(
    _: Request,
    exc: SessionNotFoundError,
) -> Response[dict[str, str]]:
    return Response(
        content={"error": "session_not_found", "detail": str(exc)},
        status_code=HTTP_404_NOT_FOUND,
    )


def _handle_session_busy(
    _: Request,
    exc: SessionBusyError,
) -> Response[dict[str, str]]:
    return Response(
        content={"error": "session_busy", "detail": str(exc)},
        status_code=HTTP_409_CONFLICT,
    )


class AgentController(Controller):
    """The application's single HTTP controller."""

    path = ""
    exception_handlers = {  # noqa: RUF012 - Litestar controller configuration
        SessionNotFoundError: _handle_session_not_found,
        SessionBusyError: _handle_session_busy,
    }

    @get("/health", sync_to_thread=False)
    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    @post("/sessions", sync_to_thread=False)
    def create_session(self, state: State) -> dict[str, str]:
        return {"session_id": state.manager.create_session()}

    @post("/sessions/{session_id:str}/chat", status_code=HTTP_200_OK)
    async def chat(
        self,
        state: State,
        session_id: str,
        data: ChatRequest,
    ) -> ServerSentEvent:
        events = await state.manager.stream_chat(session_id, data.message)

        async def messages() -> AsyncIterator[ServerSentEventMessage]:
            async for event in events:
                yield ServerSentEventMessage(
                    data=event.model_dump_json(),
                    event=event.type,
                )

        return ServerSentEvent(messages())

    @delete("/sessions/{session_id:str}")
    async def delete_session(self, state: State, session_id: str) -> None:
        await state.manager.delete_session(session_id)
