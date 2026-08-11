"""Reusable offline fakes for agent and manager tests."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from typing import Any, cast

import pytest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from pydantic import Field

from ragchat.retrieval import RetrievedPassage


@pytest.fixture(autouse=True)
def disable_ambient_langsmith_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGCHAIN_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)


class ScriptedChatModel(GenericFakeChatModel):
    """Tool-capable fake that preserves tool calls while streaming."""

    bound_tools: list[list[Any]] = Field(default_factory=list, exclude=True)

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ScriptedChatModel:
        del tool_choice, kwargs
        self.bound_tools.append(list(tools))
        return self

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        chat_result = self._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        message = cast("AIMessage", chat_result.generations[0].message)

        if message.tool_calls:
            tool_call_chunks = [
                {
                    "name": call["name"],
                    "args": json.dumps(call["args"]),
                    "id": call["id"],
                    "index": index,
                }
                for index, call in enumerate(message.tool_calls)
            ]
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=message.content,
                    id=message.id,
                    tool_call_chunks=tool_call_chunks,
                    chunk_position="last",
                )
            )
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk
            return

        content = message.content
        if not content:
            return
        if not isinstance(content, str):
            raise ValueError("Expected scripted message content to be a string.")

        content_chunks = cast("list[str]", re.split(r"(\s)", content))
        for index, token in enumerate(content_chunks):
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=token,
                    id=message.id,
                    chunk_position="last" if index == len(content_chunks) - 1 else None,
                )
            )
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk


class FakePipeline:
    """Deterministic search pipeline that records exact model-chosen queries."""

    def __init__(self, passages: list[RetrievedPassage] | None = None) -> None:
        self.passages = passages or []
        self.queries: list[str] = []

    async def search(self, query: str) -> list[RetrievedPassage]:
        self.queries.append(query)
        return list(self.passages)
