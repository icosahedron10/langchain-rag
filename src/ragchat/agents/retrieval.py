"""Agentic corpus search with an application-enforced evidence boundary."""

from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from ragchat.prompts import retrieval_prompt
from ragchat.retrieval import RetrievedPassage, SearchPipeline

logger = logging.getLogger(__name__)

MAX_SEARCHES = 3
MIN_PREFIX_LENGTH = 8


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


def _resolve_selection(
    raw: str,
    observed: dict[str, RetrievedPassage],
    ordinals: dict[str, str],
) -> RetrievedPassage | None:
    """Resolve a model-written citation label to an observed passage, tolerating slips."""
    candidate = raw.strip().strip("[]").strip().rstrip("?").strip()
    if not candidate:
        return None
    if candidate.isdigit() and candidate in ordinals:
        return observed.get(ordinals[candidate])
    passage = observed.get(candidate)
    if passage is not None:
        return passage
    if len(candidate) >= MIN_PREFIX_LENGTH:
        matches = [key for key in observed if key.startswith(candidate)]
        if len(matches) == 1:
            return observed[matches[0]]
    return None


def build_search_corpus_tool(model: BaseChatModel, pipeline: SearchPipeline) -> BaseTool:
    """Build the orchestrator's sole corpus-search tool."""

    @tool
    async def search_corpus(question: str) -> str:
        """Research the question and return grounded evidence.

        Use concise 3-8 word search queries for better hits.
        """
        observed: dict[str, RetrievedPassage] = {}
        ordinals: dict[str, str] = {}
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
            labelled = {point_id: label for label, point_id in ordinals.items()}
            lines: list[str] = []
            for passage in passages:
                observed[passage.point_id] = passage
                label = labelled.get(passage.point_id)
                if label is None:
                    label = str(len(ordinals) + 1)
                    ordinals[label] = passage.point_id
                    labelled[passage.point_id] = label
                lines.append(f"[{label}] {passage.document} p.{passage.page}: {passage.content}")

            writer({"type": "progress", "text": "Reviewing the retrieved passages…"})
            return "\n".join(lines)

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
        for selected in structured.selected_point_ids:
            passage = _resolve_selection(selected, observed, ordinals)
            if passage is None:
                unknown_ids.append(selected)
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
            gaps.append(
                f"Unresolvable citations ({len(unknown_ids)} unresolved, {len(sources)} "
                f"resolved): point IDs were not observed: {unresolved}"
            )
            logger.warning(
                "search_corpus dropped %d selected passages (%d resolved): %s",
                len(unknown_ids),
                len(sources),
                unresolved,
            )

        evidence = CorpusEvidence(
            answerable=structured.answerable,
            summary=structured.summary,
            sources=sources,
            gaps=gaps,
        )
        return evidence.model_dump_json()

    return search_corpus
