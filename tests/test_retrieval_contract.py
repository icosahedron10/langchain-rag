from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
    SparseEmbeddings,
    SparseVector,
)
from qdrant_client.http import models

from ragchat.config import Settings
from ragchat.retrieval.pipeline import HybridSearchPipeline
from ragchat.retrieval.qdrant import validate_corpus


class FakeDenseEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.document_inputs: list[list[str]] = []
        self.query_inputs: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_inputs.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_inputs.append(text)
        return [0.3, 0.2, 0.1]


class FakeSparseEmbeddings(SparseEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1, 7], values=[0.4, 0.6]) for _ in texts]

    def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1, 7], values=[0.4, 0.6])


class ReadOnlyQdrantClient:
    def __init__(self) -> None:
        self.trace: list[tuple[str, dict[str, Any]]] = []

    def collection_exists(self, **kwargs: Any) -> bool:
        self.trace.append(("collection_exists", kwargs))
        return True

    def scroll(self, **kwargs: Any) -> tuple[list[Any], None]:
        self.trace.append(("scroll", kwargs))
        point = SimpleNamespace(
            payload={
                "page_content": "startup sample",
                "metadata": {"source": "manual.pdf", "page": 1},
            }
        )
        return [point], None

    def get_collection(self, **kwargs: Any) -> Any:
        self.trace.append(("get_collection", kwargs))
        params = SimpleNamespace(
            vectors={"dense": SimpleNamespace(size=3, distance=models.Distance.COSINE)},
            sparse_vectors={"sparse": SimpleNamespace()},
        )
        return SimpleNamespace(config=SimpleNamespace(params=params))

    def query_points(self, **kwargs: Any) -> Any:
        self.trace.append(("query_points", kwargs))
        point = SimpleNamespace(
            id="point-42",
            score=0.91,
            payload={
                "page_content": "retrieved verbatim passage",
                "metadata": {"source": "manual.pdf", "page": 9},
            },
        )
        return SimpleNamespace(points=[point])

    def _fail_mutation(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the retrieval client is read-only")

    upsert = _fail_mutation
    create_collection = _fail_mutation
    delete_collection = _fail_mutation
    delete = _fail_mutation


def test_real_qdrant_hybrid_search_uses_one_named_rrf_query_and_stays_read_only() -> None:
    client = ReadOnlyQdrantClient()
    settings = Settings(
        _env_file=None,
        vllm_base_url="http://vllm.test/v1",
        vllm_model="local-model",
        qdrant_collection="corpus",
    )
    validate_corpus(client, settings)

    dense = FakeDenseEmbeddings()
    store = QdrantVectorStore(
        client=client,  # type: ignore[arg-type]
        collection_name="corpus",
        embedding=dense,
        sparse_embedding=FakeSparseEmbeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
        validate_collection_config=True,
    )

    documents = store.similarity_search("gearbox vibration", k=20)

    assert dense.document_inputs == [["dummy_text"]]
    assert dense.query_inputs == ["gearbox vibration"]
    assert [name for name, _ in client.trace] == [
        "collection_exists",
        "scroll",
        "get_collection",
        "get_collection",
        "query_points",
    ]
    query_calls = [kwargs for name, kwargs in client.trace if name == "query_points"]
    assert len(query_calls) == 1
    query_call = query_calls[0]
    dense_prefetch, sparse_prefetch = query_call["prefetch"]
    assert (dense_prefetch.using, dense_prefetch.limit) == ("dense", 20)
    assert (sparse_prefetch.using, sparse_prefetch.limit) == ("sparse", 20)
    assert dense_prefetch.query == [0.3, 0.2, 0.1]
    assert sparse_prefetch.query == models.SparseVector(indices=[1, 7], values=[0.4, 0.6])
    assert query_call["limit"] == 20
    assert query_call["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
    assert documents == [
        Document(
            page_content="retrieved verbatim passage",
            metadata={
                "source": "manual.pdf",
                "page": 9,
                "_id": "point-42",
                "_collection_name": "corpus",
            },
        )
    ]

    with pytest.raises(AssertionError, match="read-only"):
        store.add_texts(["must not be written"])
    with pytest.raises(AssertionError, match="read-only"):
        store.delete(ids=["point-42"])


class RecordingCandidateStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, int]] = []

    async def asimilarity_search(self, query: str, *, k: int) -> list[Document]:
        self.calls.append((query, k))
        return self.documents


class RecordingCrossEncoder(BaseCrossEncoder):
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(text_pairs)
        return [self.scores[passage] for _, passage in text_pairs]


@pytest.mark.asyncio
async def test_real_cross_encoder_scores_every_deduped_pair_and_returns_raw_top_ten() -> None:
    raw_scores = [0.25, -1.0, 8.0, 1.5, 1.1, 6.2, 0.0, 3.3, 4.4, 2.2, 7.7, 5.5]
    documents = [
        Document(
            page_content=f"passage-{index}",
            metadata={"_id": index, "source": "manual.pdf", "page": index},
        )
        for index in range(len(raw_scores))
    ]
    documents.append(
        Document(
            page_content="duplicate-must-not-be-scored",
            metadata={"_id": "4", "source": "duplicate.pdf", "page": 99},
        )
    )
    store = RecordingCandidateStore(documents)
    model = RecordingCrossEncoder(
        {f"passage-{index}": score for index, score in enumerate(raw_scores)}
    )
    reranker = CrossEncoderReranker(model=model, top_n=10)

    passages = await HybridSearchPipeline(store, reranker).search("bearing failure")

    assert store.calls == [("bearing failure", 20)]
    assert model.calls == [
        [("bearing failure", f"passage-{index}") for index in range(len(raw_scores))]
    ]
    expected_indices = sorted(range(len(raw_scores)), key=raw_scores.__getitem__, reverse=True)[:10]
    assert [passage.point_id for passage in passages] == [str(index) for index in expected_indices]
    assert [raw_scores[int(passage.point_id)] for passage in passages] == sorted(
        raw_scores, reverse=True
    )[:10]


def test_importing_qdrant_module_does_not_load_model_runtimes() -> None:
    script = """
import json
import sys

import ragchat.retrieval.qdrant

forbidden = (
    "torch",
    "fastembed",
    "sentence_transformers",
    "langchain_huggingface",
    "langchain_qdrant",
    "langchain_community.cross_encoders",
    "langchain_classic.retrievers.document_compressors",
    "langchain_core.cross_encoders",
)
loaded = sorted(
    module
    for module in sys.modules
    if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden)
)
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
