"""Agentic corpus search with an application-enforced evidence boundary."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from ragchat.prompts import retrieval_prompt
from ragchat.retrieval import RetrievedPassage, SearchPipeline
from ragchat.telemetry import record_retrieval_digest

MAX_SEARCHES = 3


class RetrievalResult(BaseModel):
    """The retrieval agent's interpretation and evidence selections."""

    answerable: bool
    summary: str
    selected_point_ids: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class EvidenceSource(BaseModel):
    """A verbatim passage resolved by application code."""

    point_id: str
    document: str
    page: int | None
    evidence: str


class CorpusEvidence(BaseModel):
    """Evidence returned to the orchestrator after point-id resolution."""

    answerable: bool
    summary: str
    sources: list[EvidenceSource]
    gaps: list[str]


def build_search_corpus_tool(model: BaseChatModel, pipeline: SearchPipeline) -> BaseTool:
    """Build the orchestrator's sole corpus-search tool."""

    @tool
    async def search_corpus(question: str) -> str:
        """Research the question and return grounded evidence.

        Use concise 3-8 word search queries for better hits.
        """
        observed: dict[str, RetrievedPassage] = {}
        search_count = 0

        @tool
        async def qdrant_hybrid_search(query: str) -> str:
            """Run one hybrid corpus search using a focused 3-8 word query, not a full question."""
            nonlocal search_count

            if search_count >= MAX_SEARCHES:
                return f"Search budget exhausted: at most {MAX_SEARCHES} searches are allowed."

            search_count += 1
            writer = get_stream_writer()
            writer({"type": "progress", "text": f'Searching for: "{query}"'})
            passages = await pipeline.search(query)
            for passage in passages:
                observed[passage.point_id] = passage

            writer({"type": "progress", "text": "Reviewing the retrieved passages…"})
            return "\n".join(
                f"[{passage.point_id}] {passage.document} p.{passage.page}: {passage.content}"
                for passage in passages
            )

        agent = create_agent(
            model,
            name="ragchat_retrieval",
            tools=[qdrant_hybrid_search],
            system_prompt=retrieval_prompt(),
            response_format=ToolStrategy(RetrievalResult),
            checkpointer=False,
        )
        result = await agent.ainvoke({"messages": [HumanMessage(question)]})
        structured = RetrievalResult.model_validate(result["structured_response"])

        get_stream_writer()({"type": "progress", "text": "Preparing an evidence-grounded answer…"})

        sources: list[EvidenceSource] = []
        unknown_ids: list[str] = []
        for point_id in structured.selected_point_ids:
            passage = observed.get(point_id)
            if passage is None:
                unknown_ids.append(point_id)
                continue
            sources.append(
                EvidenceSource(
                    point_id=passage.point_id,
                    document=passage.document,
                    page=passage.page,
                    evidence=passage.content,
                )
            )

        gaps = list(structured.gaps)
        if unknown_ids:
            unresolved = ", ".join(dict.fromkeys(unknown_ids))
            gaps.append(f"Unresolvable citations: point IDs were not observed: {unresolved}")

        evidence = CorpusEvidence(
            answerable=structured.answerable,
            summary=structured.summary,
            sources=sources,
            gaps=gaps,
        )
        record_retrieval_digest(
            evidence_text="\n\n".join(source.evidence for source in evidence.sources),
            pages=[source.page for source in evidence.sources],
            answerable=evidence.answerable,
        )
        return evidence.model_dump_json()

    return search_corpus
