from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ragchat.retrieval.pipeline import HybridSearchPipeline


class RecordingStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, int]] = []

    async def asimilarity_search(self, query: str, *, k: int) -> list[Document]:
        self.calls.append((query, k))
        return self.documents


class RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Document], str]] = []

    async def acompress_documents(self, documents: list[Document], query: str) -> list[Document]:
        self.calls.append((documents, query))
        return list(reversed(documents))


@pytest.mark.asyncio
async def test_pipeline_retrieves_20_deduplicates_by_point_id_then_reranks() -> None:
    first = Document(
        page_content="first verbatim passage",
        metadata={"_id": "p1", "source": "guide.pdf", "page": 7},
    )
    duplicate = Document(
        page_content="duplicate should be discarded",
        metadata={"_id": "p1", "source": "other.pdf", "page": 99},
    )
    second = Document(page_content="second", metadata={"_id": 2, "source": "notes.md"})
    store = RecordingStore([first, duplicate, second])
    reranker = RecordingReranker()

    passages = await HybridSearchPipeline(store, reranker).search("exact query")

    assert store.calls == [("exact query", 20)]
    reranked_documents, reranker_query = reranker.calls[0]
    assert reranked_documents == [first, second]
    assert reranker_query == "exact query"
    assert [(item.point_id, item.document, item.page, item.content) for item in passages] == [
        ("2", "notes.md", None, "second"),
        ("p1", "guide.pdf", 7, "first verbatim passage"),
    ]


@pytest.mark.asyncio
async def test_pipeline_enforces_final_result_limit() -> None:
    documents = [Document(page_content=str(index), metadata={"_id": index}) for index in range(12)]
    store = RecordingStore(documents)
    reranker = RecordingReranker()

    passages = await HybridSearchPipeline(store, reranker).search("query")

    assert len(passages) == 10
    assert passages[0].document == "unknown"


@pytest.mark.asyncio
async def test_pipeline_fails_hard_when_qdrant_point_id_is_missing() -> None:
    store = RecordingStore([Document(page_content="bad", metadata={})])

    with pytest.raises(KeyError, match="_id"):
        await HybridSearchPipeline(store, RecordingReranker()).search("query")
