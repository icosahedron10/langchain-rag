"""Litestar application factory and manager lifespan."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from litestar import Litestar

from ragchat.config import Settings
from ragchat.controller import AgentController
from ragchat.manager import DeepAgentManager

ManagerFactory = Callable[[Settings], Awaitable[DeepAgentManager]]


def create_app(
    settings: Settings | None = None,
    manager_factory: ManagerFactory | None = None,
) -> Litestar:
    """Create the API, building and shutting down its manager in lifespan."""

    app_settings = settings if settings is not None else Settings()
    factory = manager_factory if manager_factory is not None else DeepAgentManager.create

    @asynccontextmanager
    async def lifespan(app: Litestar) -> AsyncIterator[None]:
        manager = await factory(app_settings)
        app.state.manager = manager
        try:
            yield
        finally:
            await manager.shutdown()

    return Litestar(route_handlers=[AgentController], lifespan=[lifespan])
