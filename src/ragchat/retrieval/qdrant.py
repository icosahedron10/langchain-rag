"""Builders and startup validation for the external read-only Qdrant corpus.

Model-backed imports stay inside builder functions so importing ``ragchat`` in
offline tests never initializes torch, sentence-transformers, or fastembed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragchat.config import Settings
from ragchat.domain import StartupValidationError


def _collection_name(settings: Settings) -> str:
    collection = settings.qdrant_collection
    if collection is None:
        raise StartupValidationError("QDRANT_COLLECTION must name an existing collection.")
    return collection


def build_qdrant_client(settings: Settings) -> Any:
    """Create the synchronous client required by ``QdrantVectorStore``."""

    from qdrant_client import QdrantClient

    api_key = (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key is not None else None
    )
    return QdrantClient(url=settings.qdrant_url, api_key=api_key)


def validate_corpus(client: Any, settings: Settings) -> None:
    """Validate corpus existence and the payload contract without mutating it."""

    collection = _collection_name(settings)
    try:
        exists = client.collection_exists(collection_name=collection)
    except Exception as exc:
        raise StartupValidationError(
            f"Could not validate Qdrant collection {collection!r}: {exc}"
        ) from exc

    if not exists:
        raise StartupValidationError(
            f"Qdrant collection {collection!r} does not exist; populate it externally first."
        )

    try:
        points, _ = client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        raise StartupValidationError(
            f"Could not inspect Qdrant collection {collection!r}: {exc}"
        ) from exc

    if not points:
        raise StartupValidationError(
            f"Qdrant collection {collection!r} is empty; populate it externally first."
        )

    payload = points[0].payload
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    missing: list[str] = []
    if not isinstance(payload, Mapping) or "page_content" not in payload:
        missing.append("page_content")
    if not isinstance(metadata, Mapping) or "source" not in metadata:
        missing.append("metadata.source")
    if not isinstance(metadata, Mapping) or "page" not in metadata:
        missing.append("metadata.page")
    if missing:
        fields = ", ".join(missing)
        raise StartupValidationError(
            f"Qdrant collection {collection!r} has an invalid payload; missing: {fields}."
        )


def build_vector_store(settings: Settings, client: Any) -> Any:
    """Build the verified dense+sparse RRF vector-store configuration."""

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

    dense = HuggingFaceEmbeddings(model_name=settings.dense_embedding_model)
    sparse = FastEmbedSparse(model_name=settings.sparse_embedding_model)
    return QdrantVectorStore(
        client=client,
        collection_name=_collection_name(settings),
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=settings.qdrant_dense_vector_name,
        sparse_vector_name=settings.qdrant_sparse_vector_name,
        validate_collection_config=True,
    )


def build_reranker(settings: Settings) -> Any:
    """Load the shared BGE cross-encoder reranker, configured for ten results."""

    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    model = HuggingFaceCrossEncoder(model_name=settings.reranker_model)
    return CrossEncoderReranker(model=model, top_n=10)
