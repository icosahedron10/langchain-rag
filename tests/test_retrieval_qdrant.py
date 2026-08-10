from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from ragchat.config import Settings
from ragchat.domain import StartupValidationError
from ragchat.retrieval.qdrant import (
    build_qdrant_client,
    build_reranker,
    build_vector_store,
    validate_corpus,
)


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "vllm_base_url": "http://vllm.test/v1",
        "vllm_model": "local-model",
        "qdrant_collection": "corpus",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class ReadOnlyClient:
    def __init__(self, *, exists: bool = True, payload: Any = None) -> None:
        self.exists = exists
        self.payload = (
            payload
            if payload is not None
            else {
                "page_content": "verbatim",
                "metadata": {"source": "manual.pdf", "page": 4},
            }
        )
        self.calls: list[tuple[str, Any]] = []

    def collection_exists(self, *, collection_name: str) -> bool:
        self.calls.append(("collection_exists", collection_name))
        return self.exists

    def scroll(self, **kwargs: Any) -> tuple[list[Any], None]:
        self.calls.append(("scroll", kwargs))
        return [SimpleNamespace(payload=self.payload)], None

    def upsert(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("retrieval validation must never mutate Qdrant")

    create_collection = upsert
    delete_collection = upsert
    delete = upsert


def test_validate_corpus_uses_read_only_calls_and_checks_payload_contract() -> None:
    client = ReadOnlyClient()

    validate_corpus(client, make_settings())

    assert client.calls == [
        ("collection_exists", "corpus"),
        (
            "scroll",
            {
                "collection_name": "corpus",
                "limit": 1,
                "with_payload": True,
                "with_vectors": False,
            },
        ),
    ]


def test_validate_corpus_rejects_a_missing_collection_without_scrolling() -> None:
    client = ReadOnlyClient(exists=False)

    with pytest.raises(StartupValidationError, match="does not exist"):
        validate_corpus(client, make_settings())

    assert client.calls == [("collection_exists", "corpus")]


def test_validate_corpus_rejects_an_empty_collection() -> None:
    class EmptyClient(ReadOnlyClient):
        def scroll(self, **kwargs: Any) -> tuple[list[Any], None]:
            self.calls.append(("scroll", kwargs))
            return [], None

    with pytest.raises(StartupValidationError, match="is empty"):
        validate_corpus(EmptyClient(), make_settings())


def test_validate_corpus_accepts_a_present_page_with_a_null_value() -> None:
    payload = {
        "page_content": "verbatim",
        "metadata": {"source": "manual.pdf", "page": None},
    }

    validate_corpus(ReadOnlyClient(payload=payload), make_settings())


@pytest.mark.parametrize(
    ("payload", "missing"),
    [
        ({"metadata": {"source": "x", "page": 1}}, "page_content"),
        ({"page_content": "x", "metadata": {"page": 1}}, "metadata.source"),
        ({"page_content": "x", "metadata": {"source": "x"}}, "metadata.page"),
        ({"page_content": "x", "metadata": None}, "metadata.source"),
    ],
)
def test_validate_corpus_rejects_malformed_payloads(payload: Any, missing: str) -> None:
    with pytest.raises(StartupValidationError, match=missing):
        validate_corpus(ReadOnlyClient(payload=payload), make_settings())


def test_validate_corpus_wraps_connectivity_errors() -> None:
    class BrokenClient:
        def collection_exists(self, *, collection_name: str) -> bool:
            raise OSError("connection refused")

    with pytest.raises(StartupValidationError, match="connection refused"):
        validate_corpus(BrokenClient(), make_settings())


def test_build_qdrant_client_passes_only_configured_connection_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_client(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    import qdrant_client

    monkeypatch.setattr(qdrant_client, "QdrantClient", fake_client)
    settings = make_settings(qdrant_url="http://qdrant.test", qdrant_api_key=SecretStr("secret"))

    build_qdrant_client(settings)

    assert calls == [{"url": "http://qdrant.test", "api_key": "secret"}]


def test_build_vector_store_uses_verified_hybrid_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs: Any) -> None:
            calls["dense"] = self
            calls["dense_kwargs"] = kwargs

    class FakeSparse:
        def __init__(self, **kwargs: Any) -> None:
            calls["sparse"] = self
            calls["sparse_kwargs"] = kwargs

    class FakeVectorStore:
        def __init__(self, **kwargs: Any) -> None:
            calls["store"] = kwargs

    huggingface = ModuleType("langchain_huggingface")
    huggingface.HuggingFaceEmbeddings = FakeEmbeddings  # type: ignore[attr-defined]
    qdrant = ModuleType("langchain_qdrant")
    qdrant.FastEmbedSparse = FakeSparse  # type: ignore[attr-defined]
    qdrant.QdrantVectorStore = FakeVectorStore  # type: ignore[attr-defined]
    qdrant.RetrievalMode = SimpleNamespace(HYBRID="hybrid")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_huggingface", huggingface)
    monkeypatch.setitem(sys.modules, "langchain_qdrant", qdrant)
    client = object()

    result = build_vector_store(make_settings(), client)

    assert isinstance(result, FakeVectorStore)
    assert calls["dense_kwargs"] == {"model_name": "sentence-transformers/all-mpnet-base-v2"}
    assert calls["sparse_kwargs"] == {"model_name": "Qdrant/bm25"}
    assert calls["store"] == {
        "client": client,
        "collection_name": "corpus",
        "embedding": calls["dense"],
        "sparse_embedding": calls["sparse"],
        "retrieval_mode": "hybrid",
        "vector_name": "dense",
        "sparse_vector_name": "sparse",
        "validate_collection_config": True,
    }


def test_build_reranker_loads_configured_model_and_returns_top_ten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeCrossEncoder:
        def __init__(self, **kwargs: Any) -> None:
            calls["model"] = kwargs

    class FakeReranker:
        def __init__(self, **kwargs: Any) -> None:
            calls["reranker"] = kwargs

    encoders = ModuleType("langchain_community.cross_encoders")
    encoders.HuggingFaceCrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
    compressors = ModuleType("langchain_classic.retrievers.document_compressors")
    compressors.CrossEncoderReranker = FakeReranker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_community.cross_encoders", encoders)
    monkeypatch.setitem(
        sys.modules, "langchain_classic.retrievers.document_compressors", compressors
    )

    result = build_reranker(make_settings(reranker_model="custom-reranker"))

    assert isinstance(result, FakeReranker)
    assert calls["model"] == {"model_name": "custom-reranker"}
    assert calls["reranker"]["top_n"] == 10
    assert isinstance(calls["reranker"]["model"], FakeCrossEncoder)
