"""Hybrid retrieval followed by point-id deduplication and reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    """A passage whose evidence can be resolved without trusting model output."""

    point_id: str
    document: str
    page: int | None
    content: str


class SearchPipeline(Protocol):
    """Narrow interface consumed by the retrieval agent."""

    async def search(self, query: str) -> list[RetrievedPassage]: ...


class HybridSearchPipeline:
    """Retrieve a fused candidate set, deduplicate it, then rerank it."""

    def __init__(
        self,
        vector_store: Any,
        reranker: Any,
        k_retrieve: int = 20,
        k_final: int = 10,
    ) -> None:
        self._store = vector_store
        self._reranker = reranker
        self._k_retrieve = k_retrieve
        self._k_final = k_final

    async def search(self, query: str) -> list[RetrievedPassage]:
        documents = await self._store.asimilarity_search(query, k=self._k_retrieve)

        unique: dict[str, Any] = {}
        for document in documents:
            point_id = str(document.metadata["_id"])
            unique.setdefault(point_id, document)

        reranked = await self._reranker.acompress_documents(list(unique.values()), query)
        return [
            RetrievedPassage(
                point_id=str(document.metadata["_id"]),
                document=str(document.metadata.get("source", "unknown")),
                page=document.metadata.get("page"),
                content=document.page_content,
            )
            for document in reranked[: self._k_final]
        ]
