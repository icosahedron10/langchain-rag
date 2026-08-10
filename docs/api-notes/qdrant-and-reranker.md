All verified from installed source at `C:/Users/madse/Documents/langchain-rag/.venv/Lib/site-packages`.

# 1. QdrantVectorStore.__init__

Import: `from langchain_qdrant import QdrantVectorStore, RetrievalMode` (defined in `langchain_qdrant/qdrant.py`; `langchain_qdrant/vectorstores.py` holds only the legacy `Qdrant` class).

```python
class QdrantVectorStore(VectorStore):
    CONTENT_KEY: str = "page_content"
    METADATA_KEY: str = "metadata"
    VECTOR_NAME: str = ""                          # default/unnamed dense vector
    SPARSE_VECTOR_NAME: str = "langchain-sparse"

    def __init__(
        self,
        client: QdrantClient,                      # SYNC only. AsyncQdrantClient NOT accepted/used anywhere in the package
        collection_name: str,
        embedding: Embeddings | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,   # DENSE="dense", SPARSE="sparse", HYBRID="hybrid"
        vector_name: str = VECTOR_NAME,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        distance: models.Distance = models.Distance.COSINE,
        sparse_embedding: SparseEmbeddings | None = None,
        sparse_vector_name: str = SPARSE_VECTOR_NAME,
        validate_embeddings: bool = True,
        validate_collection_config: bool = True,
    ) -> None:
```

Validation:
- `_validate_embeddings` (if `validate_embeddings`): raises `ValueError` — DENSE: `"'embedding' cannot be None when retrieval mode is 'dense'"`; SPARSE: same for `sparse_embedding`; HYBRID: `"Both 'embedding' and 'sparse_embedding' cannot be None when retrieval mode is 'hybrid'"`.
- `_validate_collection_config` (if `validate_collection_config`): DENSE → `_validate_collection_for_dense`; SPARSE → `_validate_collection_for_sparse`; HYBRID → both. Each calls `client.get_collection(collection_name=collection_name)` (so HYBRID calls it twice). `__init__` does NOT call `collection_exists` — a missing collection surfaces as whatever `get_collection` raises (qdrant_client error), not a langchain error.
- Dense check reads `collection_info.config.params.vectors`. If it's a `dict` and `vector_name not in vector_config` → `QdrantVectorStoreError("Existing Qdrant collection {name} does not contain dense vector named {vector_name}. Did you mean one of the existing vectors: ...? If you want to recreate the collection, set `force_recreate` parameter to `True`.")`. If it's a bare `VectorParams` (unnamed vector) but `vector_name != ""` → `QdrantVectorStoreError(... "is built with unnamed dense vector ... set `vector_name` to ''(empty string)" ...)`. Then dims: `vector_size = len(embedding.embed_documents(["dummy_text"])[0])`; `vector_config.size != vector_size` → `QdrantVectorStoreError("...configured for dense vectors with {size} dimensions. Selected embeddings are {n}-dimensional...")`. Distance mismatch → `QdrantVectorStoreError("...configured for {X} similarity, but requested {Y}...")`.
- Sparse check reads `collection_info.config.params.sparse_vectors`; if `None` or `sparse_vector_name not in sparse_vector_config` → `QdrantVectorStoreError("Existing Qdrant collection {name} does not contain sparse vectors named {sparse_vector_name}...")`.
- `QdrantVectorStoreError` is importable from `langchain_qdrant.qdrant`.

```python
vs = QdrantVectorStore(client=client, collection_name="docs", embedding=emb,
                       retrieval_mode=RetrievalMode.HYBRID, sparse_embedding=FastEmbedSparse(),
                       vector_name="", sparse_vector_name="langchain-sparse")
```

# 2. HYBRID retrieval trace (k=20)

Call `vs.similarity_search(query, k=20)` (or `similarity_search_with_score`). Extra kwargs: `filter: models.Filter | None`, `search_params: models.SearchParams | None`, `offset: int = 0`, `score_threshold: float | None`, `consistency`, `hybrid_fusion: models.FusionQuery | None`.

HYBRID branch of `similarity_search_with_score` (qdrant.py:601-627): computes `embeddings.embed_query(query)` and `sparse_embeddings.embed_query(query)`, then ONE call:

```python
results = self.client.query_points(
    prefetch=[
        models.Prefetch(using=self.vector_name, query=query_dense_embedding,
                        filter=filter, limit=k, params=search_params),
        models.Prefetch(using=self.sparse_vector_name,
                        query=models.SparseVector(indices=..., values=...),
                        filter=filter, limit=k, params=search_params),
    ],
    query=hybrid_fusion or models.FusionQuery(fusion=models.Fusion.RRF),
    # plus query_options: collection_name, query_filter=filter, search_params,
    # limit=k, offset=0, with_payload=True, with_vectors=False, score_threshold, consistency
).points
```

Each prefetch leg limit = k (20) AND final fused limit = k (20). No over-fetch multiplier.

Async: NOT native. `grep 'async def|AsyncQdrantClient'` over langchain_qdrant → zero matches. `asimilarity_search`/`asimilarity_search_with_score` come from `langchain_core.vectorstores.base.VectorStore` and are `run_in_executor(None, self.similarity_search..., ...)` wrappers (base.py:431-448, 606-622).

# 3. Point → Document conversion

Point id goes ONLY to `metadata["_id"]`; `Document.id` is NOT set. Verbatim (qdrant.py:1022-1036):

```python
@classmethod
def _document_from_point(cls, scored_point: Any, collection_name: str,
                         content_payload_key: str, metadata_payload_key: str) -> Document:
    metadata = scored_point.payload.get(metadata_payload_key) or {}
    metadata["_id"] = scored_point.id
    metadata["_collection_name"] = collection_name
    return Document(page_content=scored_point.payload.get(content_payload_key, ""), metadata=metadata)
```

Payload written at index time: `{content_payload_key: text, metadata_payload_key: metadata_or_None}` (defaults `"page_content"` / `"metadata"`).

# 4. FastEmbedSparse

`from langchain_qdrant import FastEmbedSparse` (langchain_qdrant/fastembed_sparse.py).

```python
def __init__(self, model_name: str = "Qdrant/bm25", batch_size: int = 256,
             cache_dir: str | None = None, threads: int | None = None,
             providers: Sequence[Any] | None = None, parallel: int | None = None,
             **kwargs: Any) -> None
```

Requires `fastembed` installed (raises `ValueError` otherwise). `embed_query(text: str) -> SparseVector` where `SparseVector` is a pydantic model (`langchain_qdrant.sparse_embeddings.SparseVector`) with `indices: list[int]`, `values: list[float]`. `embed_documents(texts) -> list[SparseVector]`. ABC `SparseEmbeddings` has sync abstract methods + `aembed_*` run_in_executor wrappers.

# 5. qdrant_client calls to fake for tests

`__init__` (HYBRID, both validations on):
- `client.get_collection(collection_name: str) -> CollectionInfo` — called twice. Fake must expose `.config.params.vectors` = `dict[str, models.VectorParams]` keyed by `vector_name` (or bare `VectorParams` only if `vector_name==""`), with `.size` matching `len(embedding.embed_documents(["dummy_text"])[0])` and `.distance == models.Distance.COSINE`; and `.config.params.sparse_vectors` = dict containing `sparse_vector_name` key.
- (dense validation also calls `embedding.embed_documents(["dummy_text"])` — one call per dense validation.)

Hybrid `similarity_search(query, k)`:
- `client.query_points(collection_name=str, prefetch=list[models.Prefetch(2)], query=models.FusionQuery(fusion=models.Fusion.RRF), query_filter=None, search_params=None, limit=k, offset=0, with_payload=True, with_vectors=False, score_threshold=None, consistency=None) -> models.QueryResponse` — return `models.QueryResponse(points=[models.ScoredPoint(id="...", version=0, score=0.9, payload={"page_content": "...", "metadata": {...}})])` (qdrant_client 1.19.0: `QueryResponse.points: List[ScoredPoint]`; `ScoredPoint` requires `id, version, score`; real signature at qdrant_client/qdrant_client.py:269).

Other paths (not `__init__`/hybrid-search, for completeness): `upsert(collection_name, points=list[PointStruct])` (add_texts), `delete(collection_name, points_selector=ids)`, `retrieve(collection_name, ids, with_payload=True)` (get_by_ids), `collection_exists(name)`/`create_collection(...)`/`delete_collection(name)` (only in `from_texts`/`construct_instance`).

# 6. Reranking

`HuggingFaceCrossEncoder` — EXISTS in langchain_community 0.4.2: `from langchain_community.cross_encoders import HuggingFaceCrossEncoder` (huggingface.py). Does NOT exist in langchain_huggingface 1.2.2 (modules: chat_models, embeddings, llms, utils only).

```python
class HuggingFaceCrossEncoder(BaseModel, BaseCrossEncoder):
    client: Any = None
    model_name: str = "BAAI/bge-reranker-base"     # DEFAULT_MODEL_NAME
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    def __init__(self, **kwargs: Any):             # pydantic kwargs-only; builds sentence_transformers.CrossEncoder(self.model_name, **self.model_kwargs); ImportError if sentence-transformers missing
    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        scores = self.client.predict(text_pairs)
        if len(scores.shape) > 1:                  # 2-logit models: take relevant column
            scores = map(lambda x: x[1], scores)
        return scores
```

`BaseCrossEncoder` ABC actually lives in `langchain_core/cross_encoders.py` (langchain_community.cross_encoders.base and langchain_classic both re-export it):

```python
class BaseCrossEncoder(ABC):
    @abstractmethod
    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]: ...
```

`CrossEncoderReranker` — EXISTS: `from langchain_classic.retrievers.document_compressors import CrossEncoderReranker` (cross_encoder_rerank.py). Full class:

```python
class CrossEncoderReranker(BaseDocumentCompressor):
    model: BaseCrossEncoder
    top_n: int = 3
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @override
    def compress_documents(self, documents: Sequence[Document], query: str,
                           callbacks: Callbacks | None = None) -> Sequence[Document]:
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores, strict=False))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        return [doc for doc, _ in result[: self.top_n]]
```

Ordering: descending score, truncated to `top_n` (default 3). Usage: `CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base"), top_n=5)`.

`BaseDocumentCompressor` — `from langchain_core.documents import BaseDocumentCompressor` (defined in `langchain_core/documents/compressor.py`, lazy-exported via `langchain_core/documents/__init__.py`): `class BaseDocumentCompressor(BaseModel, ABC)` with abstract `compress_documents(documents: Sequence[Document], query: str, callbacks: Callbacks | None = None) -> Sequence[Document]` and default `acompress_documents` via `run_in_executor`.