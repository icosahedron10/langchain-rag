"""Composition of the user-facing orchestrator agent."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ragchat.prompts import orchestrator_prompt


def build_orchestrator(
    model: BaseChatModel,
    search_corpus_tool: BaseTool,
    checkpointer: Any,
    sandbox_backend: Any | None = None,
) -> Any:
    """Build the same corpus orchestrator with optional sandbox middleware."""

    if sandbox_backend is None:
        return create_agent(
            model,
            name="ragchat_orchestrator",
            tools=[search_corpus_tool],
            system_prompt=orchestrator_prompt(False),
            checkpointer=checkpointer,
        )

    from deepagents import FilesystemMiddleware

    return create_agent(
        model,
        name="ragchat_orchestrator",
        tools=[search_corpus_tool],
        system_prompt=orchestrator_prompt(True),
        middleware=[FilesystemMiddleware(backend=sandbox_backend)],
        checkpointer=checkpointer,
    )
