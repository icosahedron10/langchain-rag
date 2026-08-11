"""Offline tests for the genuinely agentic retrieval boundary."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage

import ragchat.agents.retrieval as retrieval_module
from conftest import FakePipeline, ScriptedChatModel
from ragchat.agents.retrieval import (
    MAX_SEARCHES,
    CorpusEvidence,
    RetrievalResult,
    build_search_corpus_tool,
)
from ragchat.retrieval import RetrievedPassage


def tool_call(name: str, arguments: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id}],
    )


def retrieval_result(
    *,
    selected_point_ids: list[str] | None = None,
    gaps: list[str] | None = None,
) -> AIMessage:
    return tool_call(
        "RetrievalResult",
        {
            "answerable": True,
            "summary": "The model's evidence synthesis.",
            "selected_point_ids": selected_point_ids or [],
            "gaps": gaps or [],
        },
        "structured-result",
    )


@pytest.mark.asyncio
async def test_model_chosen_query_emits_progress_before_search_and_builds_fresh_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline: list[Any] = []
    passage = RetrievedPassage("point-1", "manual.pdf", 4, "Exact corpus wording.")

    class OrderingPipeline(FakePipeline):
        async def search(self, query: str) -> list[RetrievedPassage]:
            timeline.append(("search", query))
            return await super().search(query)

    pipeline = OrderingPipeline([passage])
    model = ScriptedChatModel(
        messages=iter(
            [
                tool_call(
                    "qdrant_hybrid_search",
                    {"query": "model-authored embedding query"},
                    "search-1",
                ),
                retrieval_result(selected_point_ids=["point-1"]),
            ]
        )
    )
    monkeypatch.setattr(retrieval_module, "get_stream_writer", lambda: timeline.append)
    real_create_agent = retrieval_module.create_agent
    create_calls: list[dict[str, Any]] = []

    def recording_create_agent(*args: Any, **kwargs: Any) -> Any:
        create_calls.append(kwargs)
        return real_create_agent(*args, **kwargs)

    monkeypatch.setattr(retrieval_module, "create_agent", recording_create_agent)

    raw_evidence = await build_search_corpus_tool(model, pipeline).ainvoke(
        {"question": "original user wording"}
    )

    evidence = CorpusEvidence.model_validate_json(raw_evidence)
    assert pipeline.queries == ["model-authored embedding query"]
    assert timeline == [
        {
            "type": "progress",
            "text": 'Searching for: "model-authored embedding query"',
        },
        ("search", "model-authored embedding query"),
        {"type": "progress", "text": "Reviewing the retrieved passages…"},
        {"type": "progress", "text": "Preparing an evidence-grounded answer…"},
    ]
    assert evidence.sources[0].evidence == "Exact corpus wording."
    assert len(create_calls) == 1
    assert create_calls[0]["name"] == "ragchat_retrieval"
    assert create_calls[0]["checkpointer"] is False
    assert isinstance(create_calls[0]["response_format"], ToolStrategy)
    assert [tool.name for tool in create_calls[0]["tools"]] == ["qdrant_hybrid_search"]


@pytest.mark.asyncio
async def test_hard_search_cap_refuses_a_fourth_model_requested_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, str]] = []
    queries = [f"query-{index}" for index in range(1, MAX_SEARCHES + 2)]
    messages = [
        tool_call("qdrant_hybrid_search", {"query": query}, f"search-{index}")
        for index, query in enumerate(queries, start=1)
    ]
    messages.append(retrieval_result())
    model = ScriptedChatModel(messages=iter(messages))
    pipeline = FakePipeline()
    monkeypatch.setattr(retrieval_module, "get_stream_writer", lambda: events.append)

    await build_search_corpus_tool(model, pipeline).ainvoke({"question": "research this"})

    assert pipeline.queries == queries[:MAX_SEARCHES]
    searching_events = [event for event in events if event["text"].startswith("Searching for:")]
    assert len(searching_events) == MAX_SEARCHES
    assert all(queries[-1] not in event["text"] for event in searching_events)


@pytest.mark.asyncio
async def test_search_budget_counter_is_fresh_per_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, str]] = []
    messages: list[AIMessage] = []
    for invocation in (1, 2):
        messages.extend(
            [
                tool_call(
                    "qdrant_hybrid_search",
                    {"query": f"invocation-{invocation}-query-{index}"},
                    f"invocation-{invocation}-search-{index}",
                )
                for index in range(1, MAX_SEARCHES + 1)
            ]
        )
        messages.append(retrieval_result())

    model = ScriptedChatModel(messages=iter(messages))
    pipeline = FakePipeline()
    monkeypatch.setattr(retrieval_module, "get_stream_writer", lambda: events.append)
    real_create_agent = retrieval_module.create_agent
    agent_builds = 0

    def recording_create_agent(*args: Any, **kwargs: Any) -> Any:
        nonlocal agent_builds
        agent_builds += 1
        return real_create_agent(*args, **kwargs)

    monkeypatch.setattr(retrieval_module, "create_agent", recording_create_agent)
    search_tool = build_search_corpus_tool(model, pipeline)

    await search_tool.ainvoke({"question": "first"})
    await search_tool.ainvoke({"question": "second"})

    assert pipeline.queries == [
        f"invocation-{invocation}-query-{index}"
        for invocation in (1, 2)
        for index in range(1, MAX_SEARCHES + 1)
    ]
    assert agent_builds == 2
    assert model.bound_tools


@pytest.mark.asyncio
async def test_observed_passages_do_not_leak_between_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passage = RetrievedPassage("first-only", "first.pdf", 1, "First-run evidence.")

    class FirstRunOnlyPipeline(FakePipeline):
        async def search(self, query: str) -> list[RetrievedPassage]:
            self.queries.append(query)
            return [passage] if len(self.queries) == 1 else []

    model = ScriptedChatModel(
        messages=iter(
            [
                tool_call(
                    "qdrant_hybrid_search",
                    {"query": "first invocation query"},
                    "first-search",
                ),
                retrieval_result(selected_point_ids=["first-only"]),
                retrieval_result(selected_point_ids=["first-only"]),
            ]
        )
    )
    monkeypatch.setattr(retrieval_module, "get_stream_writer", lambda: lambda _: None)
    search_tool = build_search_corpus_tool(model, FirstRunOnlyPipeline())

    first = CorpusEvidence.model_validate_json(await search_tool.ainvoke({"question": "first"}))
    second = CorpusEvidence.model_validate_json(await search_tool.ainvoke({"question": "second"}))

    assert [source.point_id for source in first.sources] == ["first-only"]
    assert second.sources == []
    assert "first-only" in second.gaps[-1]


@pytest.mark.asyncio
async def test_evidence_resolution_is_verbatim_and_unknown_ids_become_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passage = RetrievedPassage(
        point_id="known-id",
        document="policy.pdf",
        page=12,
        content="Verbatim evidence; punctuation stays exactly here.",
    )
    model = ScriptedChatModel(
        messages=iter(
            [
                tool_call(
                    "qdrant_hybrid_search",
                    {"query": "policy terminology"},
                    "search",
                ),
                retrieval_result(
                    selected_point_ids=["known-id", "invented-id"],
                    gaps=["The date remains unclear."],
                ),
            ]
        )
    )
    monkeypatch.setattr(retrieval_module, "get_stream_writer", lambda: lambda _: None)

    raw_evidence = await build_search_corpus_tool(model, FakePipeline([passage])).ainvoke(
        {"question": "What is the policy?"}
    )

    evidence = CorpusEvidence.model_validate_json(raw_evidence)
    assert evidence.answerable is True
    assert evidence.summary == "The model's evidence synthesis."
    assert [source.model_dump() for source in evidence.sources] == [
        {
            "point_id": "known-id",
            "document": "policy.pdf",
            "page": 12,
            "evidence": "Verbatim evidence; punctuation stays exactly here.",
        }
    ]
    assert evidence.gaps[0] == "The date remains unclear."
    assert "invented-id" in evidence.gaps[1]
    assert "invented-id" not in {source.point_id for source in evidence.sources}


@pytest.mark.asyncio
async def test_agent_exception_propagates_without_a_local_catch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingAgent:
        async def ainvoke(self, _input: Any) -> Any:
            raise RuntimeError("model runtime failed")

    monkeypatch.setattr(retrieval_module, "create_agent", lambda *args, **kwargs: ExplodingAgent())
    monkeypatch.setattr(retrieval_module, "get_stream_writer", lambda: lambda _: None)
    model = ScriptedChatModel(messages=iter([]))

    with pytest.raises(RuntimeError, match="model runtime failed"):
        await build_search_corpus_tool(model, FakePipeline()).ainvoke({"question": "question"})


@pytest.mark.asyncio
async def test_scripted_chat_model_preserves_streamed_tool_calls_and_records_bindings() -> None:
    message = tool_call("search_corpus", {"question": "chosen by model"}, "outer-search")
    model = ScriptedChatModel(messages=iter([message]))
    bound_marker = object()

    assert model.bind_tools([bound_marker], tool_choice="any") is model
    chunks = [chunk async for chunk in model.astream([HumanMessage("hello")])]

    assert model.bound_tools == [[bound_marker]]
    streamed_calls = [call for chunk in chunks for call in chunk.tool_calls]
    assert streamed_calls == message.tool_calls


def test_retrieval_result_list_defaults_are_not_shared() -> None:
    first = RetrievalResult(answerable=False, summary="first")
    second = RetrievalResult(answerable=False, summary="second")

    first.gaps.append("only first")
    first.selected_point_ids.append("point")

    assert second.gaps == []
    assert second.selected_point_ids == []
