from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

import ragchat.agents.orchestrator as orchestrator_module
from conftest import ScriptedChatModel
from ragchat.agents.orchestrator import build_orchestrator
from ragchat.sandbox.backend import SessionRoutingBackend, SessionSandboxHandle

CORPUS_DESCRIPTION = "the tabletop rulebook corpus"


@tool
async def search_corpus(question: str) -> str:
    """Search the corpus for evidence."""
    return question


def test_disabled_mode_wires_only_search_without_importing_deepagents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    graph = object()

    def fake_create_agent(_model: Any, **kwargs: Any) -> object:
        calls.append(kwargs)
        return graph

    monkeypatch.setattr(orchestrator_module, "create_agent", fake_create_agent)
    model = ScriptedChatModel(messages=iter([]))
    checkpointer = object()

    result = build_orchestrator(model, search_corpus, checkpointer, CORPUS_DESCRIPTION)

    assert result is graph
    assert calls == [
        {
            "name": "ragchat_orchestrator",
            "tools": [search_corpus],
            "system_prompt": orchestrator_module.orchestrator_prompt(False, CORPUS_DESCRIPTION),
            "checkpointer": checkpointer,
        }
    ]
    assert "middleware" not in calls[0]


def test_deepagents_is_not_imported_at_orchestrator_module_scope() -> None:
    module_path = Path(orchestrator_module.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert "deepagents" not in imported_roots


def test_sandbox_mode_adds_one_filesystem_middleware_around_the_same_corpus_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    middleware_backends: list[object] = []

    class FakeFilesystemMiddleware:
        def __init__(self, *, backend: object) -> None:
            middleware_backends.append(backend)

    deepagents = ModuleType("deepagents")
    deepagents.FilesystemMiddleware = FakeFilesystemMiddleware  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepagents", deepagents)
    monkeypatch.setattr(
        orchestrator_module,
        "create_agent",
        lambda _model, **kwargs: calls.append(kwargs) or object(),
    )
    backend = object()

    build_orchestrator(
        ScriptedChatModel(messages=iter([])),
        search_corpus,
        False,
        CORPUS_DESCRIPTION,
        backend,
    )

    assert middleware_backends == [backend]
    assert calls[0]["tools"] == [search_corpus]
    assert calls[0]["name"] == "ragchat_orchestrator"
    assert calls[0]["system_prompt"] == orchestrator_module.orchestrator_prompt(
        True, CORPUS_DESCRIPTION
    )
    assert calls[0]["checkpointer"] is False
    assert len(calls[0]["middleware"]) == 1


@pytest.mark.asyncio
async def test_disabled_mode_exposes_exactly_one_tool_to_the_model() -> None:
    model = ScriptedChatModel(messages=iter([AIMessage(content="hello")]))
    graph = build_orchestrator(
        model, search_corpus, checkpointer=False, corpus_description=CORPUS_DESCRIPTION
    )

    await graph.ainvoke({"messages": [HumanMessage("hi")]})

    assert model.bound_tools
    assert [Counter(tool.name for tool in bound) for bound in model.bound_tools] == [
        Counter({"search_corpus": 1})
    ]


@pytest.mark.asyncio
async def test_sandbox_mode_exposes_filesystem_tools_but_no_second_corpus_tool() -> None:
    model = ScriptedChatModel(messages=iter([AIMessage(content="hello")]))
    backend = SessionRoutingBackend(lambda _session_id: cast("SessionSandboxHandle", object()))
    graph = build_orchestrator(
        model,
        search_corpus,
        checkpointer=False,
        corpus_description=CORPUS_DESCRIPTION,
        sandbox_backend=backend,
    )

    await graph.ainvoke({"messages": [HumanMessage("hi")]})

    assert all(isinstance(tool, BaseTool) for binding in model.bound_tools for tool in binding)
    assert [Counter(tool.name for tool in binding) for binding in model.bound_tools] == [
        Counter(
            {
                "search_corpus": 1,
                "ls": 1,
                "read_file": 1,
                "write_file": 1,
                "edit_file": 1,
                "delete": 1,
                "glob": 1,
                "grep": 1,
                "execute": 1,
            }
        )
    ]
