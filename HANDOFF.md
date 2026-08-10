# HANDOFF — North Star for finishing `ragchat`

You are picking up a partially built project. This document is the single source
of truth for **what is being built, what is already done, every decision already
made, and the exact blueprint for the rest**. The most expensive work — verifying
the actual APIs of the installed (post-training-cutoff) library versions — is
already done and preserved in `docs/api-notes/`. **Do not re-derive it, and do
not trust your memory of these libraries; they changed. When in doubt, read
`docs/api-notes/*.md` first, then the installed source in `.venv`.**

---

## 1. Mission

A local, single-user proof-of-concept chat app answering questions from an
**externally populated, read-only** Qdrant corpus:

```
Streamlit (ui/) ── HTTP+SSE ──> Litestar AgentController (one controller)
                                      │ delegates everything
                                      v
                               DeepAgentManager (no HTTP imports)
                                ├─ Orchestrator agent (LangChain create_agent)
                                │    ├─ search_corpus tool ──> Retrieval agent (genuinely agentic)
                                │    │                            └─ Qdrant hybrid search (≤3 per invocation)
                                │    └─ optional sandbox tools (deepagents FilesystemMiddleware + Docker backend)
                                ├─ InMemorySaver checkpoints (thread_id == session_id)
                                ├─ per-session asyncio.Lock (409 on overlap, no queue)
                                └─ model runtime: vLLM via AsyncOpenAI (default) | OpenAI (removable)
```

Full binding requirements are digested in §8 (test list and completion standard
are near-verbatim from the spec — treat them as acceptance criteria).

## 2. Non-negotiable decisions already made

1. **Poetry, not uv.** The user explicitly rejected uv mid-session: *"use Poetry
   and assume a linux deployment."* Deliverable is a synchronized `poetry.lock`.
   Poetry 2.3.3, PEP 621 `[project]` table + `[tool.poetry]` packages. Venv is
   in-project (`.venv/`, configured via `poetry.toml`). Dev machine is Windows;
   **target is Linux** — pathlib everywhere, `/bin/sh` in the sandbox, no
   Windows accommodations in shipped code.
2. **Python 3.12** (`>=3.12,<3.13`). Package: `src/ragchat/`.
3. **Resolved versions (installed, verified):** deepagents 0.7.5, langchain
   1.3.14, langchain-core 1.5.3, langgraph 1.2.10, langchain-openai 1.4.2,
   langchain-qdrant 1.1.0, langchain-community 0.4.2, langchain-classic 1.0.8
   (transitive — must be declared, see §5 step 0), litestar 2.24.0, openai
   2.53.0, qdrant-client 1.19.0, fastembed 0.8.0, sentence-transformers 5.7.0,
   streamlit 1.61.1, httpx-sse 0.4.3.
4. **Orchestrator uses `langchain.agents.create_agent` in BOTH modes, not
   `create_deep_agent`.** Verified reason: in deepagents 0.7.5,
   `FilesystemMiddleware` (tools: ls/read_file/write_file/edit_file/delete/
   glob/grep/execute) and `SubAgentMiddleware` (task tool) are *protected* and
   cannot be removed from `create_deep_agent` — but RAG-only mode must expose
   **no** filesystem/execution tools. Manual composition with deepagents'
   exported `FilesystemMiddleware` + a custom `SandboxBackendProtocol` backend
   (sandbox mode only) keeps both modes symmetric and still satisfies the
   "Deep Agents backend and filesystem middleware abstractions" requirement.
5. **SSE model:** event names `progress | message | artifact | done | error`;
   data = JSON of the pydantic models in `src/ragchat/domain.py`.
6. **Session concurrency:** check `lock.locked()` → raise `SessionBusyError`
   (→409), else acquire; release in the stream generator's `finally`. No queue.
7. **Artifact size cap:** `ARTIFACT_MAX_BYTES` (default 5 MB raw, pre-base64);
   oversized images are skipped and reported by name via a progress event.
8. Default API port **8080** (vLLM commonly owns 8000).

## 3. Current state

### Done (written, never yet executed — expect only trivial fixes)
| Path | Status |
|---|---|
| `pyproject.toml`, `poetry.lock`, `poetry.toml`, `.venv/` | Installed with `--all-groups`. **Missing dep: `langchain-classic` (§5 step 0)** |
| `src/ragchat/config.py` | Strict `Settings` (one model), cross-field validation, actionable errors |
| `src/ragchat/domain.py` | Domain events (SSE payloads) + exceptions; zero framework imports |
| `src/ragchat/prompts/` | Complete: `orchestrator.py`, `retrieval.py`, `citations.py`, `runtime.py`, composed via `orchestrator_prompt(sandbox_enabled)` / `retrieval_prompt()` |
| `src/ragchat/sandbox/docker_session.py` | `DockerSandboxSession` (async, docker-CLI subprocess, lazy container, hardened flags, fakeable `runner` seam), `assert_docker_available` |
| `src/ragchat/sandbox/artifacts.py` | Workspace image snapshot/diff → `ArtifactEvent`s + skipped names |
| `ui/streamlit_app.py` | Complete API-only client (httpx + httpx-sse), progress/status, incremental text, inline images, busy-disabled input, clear-chat → DELETE |
| `.env.example`, `sandbox/Dockerfile` | Complete |
| `docs/api-notes/*.md` | **Verified API references — read before coding** |

### Not started
`src/ragchat/providers.py`, `openai_backend.py`, `retrieval/` (empty dir, no
`__init__.py`), `agents/` (same), `sandbox/backend.py`, `manager.py`,
`controller.py`, `app.py`, `__main__.py`, all of `tests/`, `README.md`
(currently a 65-byte stub), `docs/architecture.md`.

## 4. Verified API facts you must not re-learn the hard way

Full details with line numbers in `docs/api-notes/`. The load-bearing ones:

1. **ChatOpenAI + injected AsyncOpenAI** (`chatopenai-async-client.md`): the
   guard is `if not self.async_client:` — passing `root_async_client` alone gets
   **overwritten**. Pass **both** `async_client=client.chat.completions` and
   `root_async_client=client`, plus non-empty `api_key=` (e.g. `"EMPTY"`) so the
   sync-client default construction doesn't hit env fallback. Async request
   path (`_astream`/`_agenerate`) never touches the sync client.
2. **Structured output** (`langchain-create-agent.md`):
   `from langchain.agents.structured_output import ToolStrategy`. Synthetic tool
   name == schema class `__name__` (so a fake model must emit a tool call named
   `"RetrievalResult"`). Parsed object lands in `result["structured_response"]`.
3. **Nested streaming** (`langgraph-streaming.md`) — the subtlest part:
   - Outer `astream(..., stream_mode=["messages", "custom"], subgraphs=True)`
     is **required** for custom events and LLM tokens from the *inner* retrieval
     agent (invoked inside a tool) to surface. Without `subgraphs=True` the
     inner stream writer is silently a no-op.
   - With `subgraphs=True`, yields are 3-tuples `(ns, mode, payload)`.
   - **Filter assistant tokens by namespace:** the orchestrator's own model runs
     have `ns == ()`; the inner retrieval agent's LLM has `ns == ("tools:<id>",)`
     — never forward those (hidden reasoning). `payload` for `messages` is
     `(AIMessageChunk, metadata)`.
   - `get_stream_writer()` (`from langgraph.config import get_stream_writer`)
     works inside `@tool` bodies (and in code they call); tools may instead take
     a `runtime: ToolRuntime` param (`from langchain.tools import ToolRuntime`).
   - An inner compiled graph invoked inside a tool **inherits the outer
     checkpointer** unless created with `checkpointer=False`
     (`Checkpointer = None | bool | BaseCheckpointSaver` — verified). Use
     `create_agent(..., checkpointer=False)` for the retrieval agent so its
     internal messages are never persisted.
   - Parent config auto-propagates into `inner.ainvoke()` with no config arg
     (contextvar; Python ≥3.11).
4. **Qdrant** (`qdrant-and-reranker.md`): `QdrantVectorStore` takes a **sync**
   `QdrantClient` only; async methods are `run_in_executor` wrappers (fine).
   HYBRID mode = one `client.query_points(prefetch=[dense Prefetch, sparse
   Prefetch], query=FusionQuery(fusion=Fusion.RRF), limit=k)` — each prefetch
   leg limit == k == final limit, so `similarity_search(query, k=20)` satisfies
   "retrieve 20 points". Point id appears **only** in `doc.metadata["_id"]`.
   `__init__` with `validate_collection_config=True` calls `get_collection`
   (twice for HYBRID) and validates named dense vector (name/size/distance) and
   sparse vector name — but **not** collection existence; call
   `client.collection_exists()` yourself first for a clear error. Fake clients
   for tests need: `get_collection`, `collection_exists`, `query_points`,
   `scroll` (see api-note §5 for exact shapes).
5. **Reranking** (`qdrant-and-reranker.md` §6):
   `from langchain_community.cross_encoders import HuggingFaceCrossEncoder`
   (wraps `sentence_transformers.CrossEncoder`; handles 2-logit models) +
   `from langchain_classic.retrievers.document_compressors import
   CrossEncoderReranker` — `CrossEncoderReranker(model=..., top_n=10)`
   `.compress_documents(docs, query)` scores every pair, sorts desc, truncates
   to top_n. Exactly the specified pipeline. `acompress_documents` = executor
   wrapper. `BaseCrossEncoder` ABC lives in `langchain_core.cross_encoders`.
6. **deepagents sandbox seam** (`deepagents.md`): implement a backend
   subclassing `SandboxBackendProtocol` (from `deepagents.backends.protocol`,
   as is `ExecuteResponse(output, exit_code=None, truncated=False)` at
   protocol.py:754). `BackendProtocol` methods are **non-abstract** (default
   `raise NotImplementedError`; async `a*` default to `to_thread` of sync) —
   partial implementations are safe. `FilesystemMiddleware`'s `execute` tool
   calls **`await backend.aexecute(command, timeout=...)`** (filesystem.py:2958)
   and catches `NotImplementedError` into an error ToolMessage. The middleware
   shows the `execute` tool only when `isinstance(backend,
   SandboxBackendProtocol)`. Backend **factories were removed** — the backend is
   one instance fixed at graph build time, hence the session-routing design in
   §5. `FilesystemBackend(root_dir=..., virtual_mode=True)` is the shipped
   host-dir file-ops backend.
7. **Litestar** (`litestar.md`): `ServerSentEvent(async_gen)` +
   `ServerSentEventMessage(data=..., event=...)`; `@post` defaults to 201,
   `@delete` to 204 and its handler **must** be annotated `-> None`; path params
   **require** a type (`{session_id:str}`); controller-level
   `exception_handlers = {Exc: handler(request, exc) -> Response}`;
   `Litestar(lifespan=[asynccontextmanager taking app])`; `app.state` mutable in
   lifespan, handlers take reserved kwarg `state: State`. **TestClient runs
   lifespan only as a context manager** (`with TestClient(app) as c:`); consume
   SSE via `with client.stream("POST", url, json=...) as r: r.iter_lines()`
   (separator `\r\n`).
8. **Fake chat models** (`langchain-create-agent.md` §4):
   `GenericFakeChatModel` (a) has **no `bind_tools`** (base raises
   NotImplementedError — `create_agent` always calls it) and (b) its `_stream`
   **drops `tool_calls`**. Tests must subclass: `bind_tools` records the tool
   list and returns `self`; `_stream` yields scripted tool-call messages as a
   single `AIMessageChunk(tool_call_chunks=[{name, args=json.dumps(args), id,
   index}])` and word-splits plain content (for incremental-text assertions).

## 5. Implementation blueprint (remaining work, in order)

### Step 0 — dependency fix
```
poetry add "langchain-classic>=1.0"
```
(Already installed transitively via deepagents; must be declared because we
import it directly. Keeps `poetry.lock` synchronized.)

### Step 1 — `src/ragchat/retrieval/` (`__init__.py`, `qdrant.py`, `pipeline.py`)
`qdrant.py` — heavy imports **inside functions** (offline tests must not need
torch/fastembed):
- `build_qdrant_client(settings)`; `validate_corpus(client, settings)`:
  `collection_exists` → clear `StartupValidationError` if missing; `scroll`
  limit=1 → fail if empty, or if payload lacks `page_content` or
  `metadata.source`/`metadata.page`. Read-only ops only (`get_collection`,
  `collection_exists`, `scroll`, `query_points`) — never upsert/create/delete.
- `build_vector_store(settings, client)`: `HuggingFaceEmbeddings(model_name=
  settings.dense_embedding_model)` + `FastEmbedSparse(model_name=
  settings.sparse_embedding_model)` + `QdrantVectorStore(client=client,
  collection_name=..., retrieval_mode=RetrievalMode.HYBRID, vector_name=
  settings.qdrant_dense_vector_name, sparse_vector_name=
  settings.qdrant_sparse_vector_name, validate_collection_config=True)`.
- `build_reranker(settings)`: `CrossEncoderReranker(model=
  HuggingFaceCrossEncoder(model_name=settings.reranker_model), top_n=10)`.

`pipeline.py` — no heavy imports:
```python
@dataclass
class RetrievedPassage: point_id: str; document: str; page: int | None; content: str

class HybridSearchPipeline:                      # constructor: (vector_store, reranker, k_retrieve=20, k_final=10)
    async def search(self, query: str) -> list[RetrievedPassage]:
        docs = await self._store.asimilarity_search(query, k=self._k_retrieve)   # 20, RRF-fused hybrid
        unique = {}                                                              # dedup by stable point id
        for d in docs: unique.setdefault(str(d.metadata["_id"]), d)
        top = await self._reranker.acompress_documents(list(unique.values()), query)  # BGE, top 10, desc
        return [RetrievedPassage(str(d.metadata["_id"]), str(d.metadata.get("source", "unknown")),
                                 d.metadata.get("page"), d.page_content) for d in top]
```
Define a `SearchPipeline` Protocol (just `async search`) for the fakes.

### Step 2 — `providers.py` + `openai_backend.py`
```python
# providers.py
def build_chat_model(settings) -> BaseChatModel:
    if settings.model_backend is ModelBackend.OPENAI:
        # Temporary workaround; delete openai_backend.py and this branch to remove.
        from ragchat.openai_backend import build_openai_chat_model
        return build_openai_chat_model(settings)
    return _build_vllm_chat_model(settings)

def _build_vllm_chat_model(settings):
    from openai import AsyncOpenAI
    from langchain_openai import ChatOpenAI
    key = settings.vllm_api_key.get_secret_value()
    client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key=key)
    return ChatOpenAI(model=settings.vllm_model, base_url=settings.vllm_base_url, api_key=key,
                      async_client=client.chat.completions, root_async_client=client)  # BOTH — see §4.1
```
`openai_backend.py` is the whole OpenAI branch (one `build_openai_chat_model`).
Lazy import keeps `sys.modules` clean in vllm mode (testable) and deletion
trivial. No runtime fallback anywhere.

### Step 3 — `agents/retrieval.py`
Pydantic models: `RetrievalResult(answerable, summary, selected_point_ids=[],
gaps=[])`, `EvidenceSource(point_id, document, page, evidence)`,
`CorpusEvidence(answerable, summary, sources, gaps)`. `MAX_SEARCHES = 3`.

`build_search_corpus_tool(model, pipeline) -> BaseTool`: returns
`@tool async def search_corpus(question: str) -> str` which **per invocation**:
1. Fresh closure state: `observed: dict[str, RetrievedPassage]`, search counter.
2. Defines inner `@tool async def qdrant_hybrid_search(query: str) -> str`:
   - if counter ≥ 3 → return budget-exhausted notice **without searching**;
   - `get_stream_writer()({"type": "progress", "text": f'Searching for: "{query}"'})`
     — must be the exact embedding-query text, emitted *before* the search;
   - `passages = await pipeline.search(query)`; record into `observed`;
   - emit `"Reviewing the retrieved passages…"`; return passages formatted as
     `[point_id] document p.page: content`.
3. Builds the retrieval agent **per invocation** (cheap; keeps state honest):
   `create_agent(model, tools=[qdrant_hybrid_search], system_prompt=
   retrieval_prompt(), response_format=ToolStrategy(RetrievalResult),
   checkpointer=False)` ← False is critical (§4.3).
4. `result = await agent.ainvoke({"messages": [HumanMessage(question)]})`;
   `structured = result["structured_response"]`.
5. Emit `"Preparing an evidence-grounded answer…"`; resolve
   `selected_point_ids` **only against `observed`** → `CorpusEvidence` with
   verbatim `evidence=passage.content`; drop unknown ids and append a gap noting
   unresolvable citations; return `evidence.model_dump_json()`.

No sufficiency heuristics. No try/except around the agent — `create_agent`'s
ToolNode default error handling turns tool exceptions into error ToolMessages
for the orchestrator (framework-first requirement).

### Step 4 — `sandbox/backend.py` (imported lazily, docker mode only)
```python
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

@dataclass
class SessionSandboxHandle:            # one per session, created by the manager
    session: DockerSandboxSession      # container lifecycle + async execute (exists already)
    files: FilesystemBackend           # FilesystemBackend(root_dir=workspace, virtual_mode=True)
    settings: Settings

class SessionRoutingBackend(SandboxBackendProtocol):
    """One instance at graph build time; routes per-run via thread_id (== session_id)."""
    def __init__(self, resolve: Callable[[str], SessionSandboxHandle]): ...
    def _handle(self):
        from langgraph.config import get_config
        return self._resolve(get_config()["configurable"]["thread_id"])
    # file ops: pure delegation → self._handle().files.ls/read/grep/glob/write/edit/delete(...)
    # upload_files/download_files: delegate to files backend too
    @property
    def id(self): return "ragchat-docker"
    def execute(self, command, *, timeout=None):
        raise NotImplementedError  # middleware calls aexecute (verified) and catches this
    async def aexecute(self, command, *, timeout=None) -> ExecuteResponse:
        handle = self._handle()
        before = snapshot_workspace(handle.session.workspace)          # sandbox/artifacts.py
        result = await handle.session.execute(command)                 # lazy container start inside
        artifacts, skipped = collect_image_artifacts(handle.session.workspace, before,
                                                     handle.settings.artifact_max_bytes)
        writer = get_stream_writer()
        for a in artifacts: writer(a.model_dump())                     # inline, immediate
        for name in skipped:
            writer({"type": "progress", "text": f"Skipped large image {name} (> size cap)"})
        return ExecuteResponse(output=result.output, exit_code=result.exit_code)
```
Sync file-op delegation is enough — the protocol's default `a*` methods wrap
them in `to_thread`, and `get_config()` still works there (contextvars copy).
Note the model-facing path mapping: file tools address the workspace as `/…`
(virtual root), the shell sees the same files under `/workspace` — the
`SANDBOX_RUNTIME` prompt section should gain one line stating this.

### Step 5 — `agents/orchestrator.py`
```python
def build_orchestrator(model, search_corpus_tool, checkpointer, sandbox_backend=None):
    if sandbox_backend is None:
        return create_agent(model, tools=[search_corpus_tool],
                            system_prompt=orchestrator_prompt(False), checkpointer=checkpointer)
    from deepagents import FilesystemMiddleware
    return create_agent(model, tools=[search_corpus_tool],
                        system_prompt=orchestrator_prompt(True),
                        middleware=[FilesystemMiddleware(backend=sandbox_backend)],
                        checkpointer=checkpointer)
```
`search_corpus` is the only corpus tool in both modes; disabled mode has no
other tools at all.

### Step 6 — `manager.py`
```python
@dataclass
class RuntimeComponents:               # the DI seam for offline tests
    chat_model: BaseChatModel          # ONE instance shared by both agents
    pipeline: SearchPipeline
    sandbox_handle_factory: Callable[[str], SessionSandboxHandle] | None  # None unless docker mode

async def build_components(settings) -> RuntimeComponents:
    # real path: build_chat_model; qdrant client + validate_corpus + vector store +
    # reranker + HybridSearchPipeline (heavy imports live in retrieval/qdrant.py);
    # docker mode ONLY: import ragchat.sandbox.*, await assert_docker_available(),
    # and build the handle factory (workspace under tempdir/ragchat-workspaces/<sid>).
    # disabled mode: no sandbox import, no docker check, factory None.
```
`DeepAgentManager`:
- `create(settings, components=None)` classmethod → builds components if None;
  `InMemorySaver()`; `search_corpus = build_search_corpus_tool(...)`;
  `SessionRoutingBackend(self._resolve_handle)` if docker mode;
  `build_orchestrator(...)`. No litestar imports anywhere in this module.
- Sessions: `dict[str, Session]`, `Session(id, lock=asyncio.Lock(),
  handle: SessionSandboxHandle | None)`. `create_session()` → uuid4 hex; in
  docker mode create the handle now (cheap — **container still lazy**).
- `stream_chat(session_id, message)`: session lookup → `SessionNotFoundError`;
  `if lock.locked(): raise SessionBusyError`; `await lock.acquire()`
  (uncontended, no suspension between check and acquire); return async
  generator that: `try:` iterate
  `self._orchestrator.astream({"messages": [HumanMessage(message)]},
  config={"configurable": {"thread_id": session_id}},
  stream_mode=["messages", "custom"], subgraphs=True)` unpacking
  `(ns, mode, payload)`:
  - `custom` → payload is a domain-event dict (progress/artifact) — validate via
    `TypeAdapter(DomainEvent)` and yield;
  - `messages` → `(chunk, meta)`; yield `MessageDelta(text=chunk.text)` only if
    `ns == ()` (root = orchestrator model, §4.3) and the chunk is an
    AIMessageChunk with non-empty text;
  - end → `DoneEvent`; `except Exception` → `ErrorEvent(message=...)` (no
    internals leaked); `finally:` release lock.
- `delete_session`: busy → `SessionBusyError`; pop; `await handle.session.close()`
  (removes container + workspace); `await saver.adelete_thread(session_id)`.
- `shutdown()`: close all session sandboxes, clear dict.

### Step 7 — `controller.py`, `app.py`, `__main__.py`
Controller (the ONLY HTTP module): `class AgentController(Controller)` with
`GET /health`, `POST /sessions` (201, `{"session_id": ...}`),
`POST /sessions/{session_id:str}/chat` (body `ChatRequest{message: str,
min_length=1}`) → `await manager.stream_chat(...)` **before** constructing
`ServerSentEvent` (so Busy/NotFound raise pre-stream) → wrap events as
`ServerSentEventMessage(data=ev.model_dump_json(), event=ev.type)`;
`DELETE /sessions/{session_id:str}` annotated `-> None`. Controller-level
`exception_handlers = {SessionNotFoundError: →404, SessionBusyError: →409}`.
Manager comes from `state: State` (reserved kwarg). Nothing else lives here.

`app.py`: `create_app(settings=None, manager_factory=None) -> Litestar` with an
`@asynccontextmanager` lifespan that awaits the factory (default
`DeepAgentManager.create`), sets `app.state.manager`, and calls
`manager.shutdown()` on exit. `__main__.py`: read `Settings()`, `uvicorn.run`
the app on `settings.api_host:settings.api_port`.

### Step 8 — `tests/` (all offline; see §6)
### Step 9 — `poetry run ruff format . && ruff check . && mypy src && pytest` until green
### Step 10 — rewrite `README.md` (concise: purpose, prerequisites incl. external
Qdrant contract, setup, run commands, config table, sandbox notes, artifact cap,
Linux deployment note about torch CUDA wheels) + `docs/architecture.md` (short:
the diagram, boundaries, key decisions from §2/§4, why create_agent over
create_deep_agent, streaming design, evidence boundary). Keep `docs/api-notes/`.

## 6. Testing strategy

`tests/conftest.py` fakes (no network, no model downloads, no docker, no qdrant):
- **`ScriptedChatModel(GenericFakeChatModel)`** — see §4.8. Records every
  `bind_tools` tool list (assert exposed toolsets per mode) and preserves
  scripted `tool_calls` under streaming. For a full manager-level run the ONE
  shared instance consumes messages in this exact order:
  `[AI(tool_call search_corpus), AI(tool_call qdrant_hybrid_search),
  AI(tool_call RetrievalResult), AI("final answer text")]`.
- **FakePipeline** implementing `SearchPipeline` — records queries, returns
  canned `RetrievedPassage`s.
- **Fake docker runner** (`DockerRunner` seam in `docker_session.py`) recording
  argv lists; assert `--network none`, exactly one `--volume` (the workspace),
  no Qdrant/env leakage, lazy `run`, `rm --force` on close.
- **FakeQdrantClient** for `pipeline`/read-only tests (shapes in
  `docs/api-notes/qdrant-and-reranker.md` §5); raise on any mutation method.
- Litestar tests: `create_test_client`/`TestClient` **as context manager**, fake
  manager injected via `create_app(manager_factory=...)`; SSE via
  `client.stream(...)` + `iter_lines()`.
- Import-isolation tests are AST/`sys.modules` checks: manager module tree never
  imports `litestar`; `ui/streamlit_app.py` never imports `ragchat*`; building
  vllm components leaves `ragchat.openai_backend` out of `sys.modules`; disabled
  sandbox mode never imports `ragchat.sandbox.backend` nor invokes docker.
- Checkpointer hygiene: after a full fake run, `saver.aget_tuple(cfg)` /
  `graph.aget_state` messages contain no `"Searching for"` text (progress events
  are transient); retrieval-agent internal messages absent (inner
  `checkpointer=False`).

**The spec's required test list (acceptance criteria):** controller delegates to
manager; manager has no HTTP deps; vLLM is default backend; one startup provider
used by both agents; missing OpenAI creds don't affect vllm mode; OpenAI branch
isolated/removable; orchestrator has exactly one corpus-search tool; retrieval
is genuinely agentic (model-chosen queries, model-driven loop); three-search cap
enforced mechanically; exact query strings produce immediate progress events;
progress events not checkpointed; hybrid retrieval requests 20 points; dedup by
point id before reranking; reranker returns top 10; selected point ids resolve
to verbatim evidence; qdrant access never mutates; docker never checked in
disabled mode; disabled mode exposes no dead sandbox tools; containers are
session-owned + lazy in enabled mode; container gets only the workspace mount;
no corpus mount or qdrant credentials reach the sandbox; one active request per
session (409); SSE streams progress/text/inline images/completion/errors;
Streamlit talks only through the API.

## 7. Footguns (ranked by cost if forgotten)

1. `subgraphs=True` or inner progress events vanish silently; 3-tuple unpack.
2. `checkpointer=False` on the inner agent or it inherits the outer saver.
3. `async_client` + `root_async_client` both, or your vLLM client is discarded.
4. Fakes: no `bind_tools` on `GenericFakeChatModel`; `_stream` drops tool_calls.
5. Litestar: lifespan only under `with TestClient(...)`; `{id:str}` typed path
   params; `@delete` handler must return `None`; SSE lines use `\r\n`.
6. Point id only in `metadata["_id"]`; `collection_exists` is NOT checked by
   `QdrantVectorStore.__init__` — check it yourself first.
7. Keep torch/sentence-transformers/fastembed imports inside builder functions —
   offline unit tests must import `ragchat.*` without touching them.
8. `poetry add langchain-classic` before importing `CrossEncoderReranker`.
9. Emit **no** raw scores, embeddings, credentials, prompts, or reasoning via
   SSE; `ErrorEvent.message` must be generic.
10. Everything in `src/ragchat/manager.py`'s import graph stays litestar-free.

## 8. Requirements digest (binding, from the original spec)

- Endpoints: `GET /health`, `POST /sessions`, `POST /sessions/{id}/chat` (SSE),
  `DELETE /sessions/{id}`. One controller. Controller owns only routing/
  validation/SSE serialization/status codes/domain-exception translation.
- No auth, no accounts, no durable sessions, no multi-user. Bind 127.0.0.1
  (configurable). In-memory checkpoints only; conversations die with the API.
- One active request per session → 409, no queue.
- `MODEL_BACKEND=vllm|openai` chosen once at startup; never via HTTP, never in
  sessions, no fallback; OpenAI branch trivially deletable; both agents share
  the one selected runtime; async client in the request path.
- Orchestrator: only corpus tool is `search_corpus`; no web search, no direct
  Qdrant tools, no alternate retrievers. Prompted to search first, use only
  returned evidence, verify claims against verbatim passages, cite document+page,
  treat summaries as interpretation, report gaps, avoid general knowledge for
  corpus claims. RAG-only mode: conversational + RAG, zero exec/fs tools.
- Retrieval agent: exactly one internal tool (Qdrant hybrid search); model
  decides query formulation/sufficiency/re-query/distillation; the ONLY
  mechanical limit is ≤3 searches; no sufficiency heuristics.
- Qdrant: externally populated, read-only. NO ingestion/upsert/create/delete/
  collection management anywhere. Validate connectivity, collection existence,
  named-vector compatibility, payload fields; fail clearly, never repair.
- Pipeline (exact): mpnet dense (768) via LangChain embeddings + FastEmbedSparse
  `Qdrant/bm25` → QdrantVectorStore HYBRID with named vectors + RRF, retrieve
  20 → dedup by point id → `BAAI/bge-reranker-base` (loaded once, shared) scores
  every pair → sort by raw score → top 10.
- Evidence boundary: agent returns `{answerable, summary, selected_point_ids,
  gaps}`; application code resolves ids to verbatim `page_content` + source/page
  observed during that invocation. Never ask the model to reproduce passages.
- SSE: sparse human-readable progress (`Searching for: "<exact query>"`,
  `Reviewing the retrieved passages…`, `Preparing an evidence-grounded
  answer…`); progress transient (not persisted/replayed); stream text
  incrementally; no hidden CoT/credentials/scores/prompts.
- Sandbox: default disabled; docker mode fails clearly if unavailable; one lazy
  container per session; separate `docker exec <c> /bin/sh -lc "<cmd>"` per
  command starting in `/workspace`; only workspace mounted rw; no corpus mount,
  no qdrant creds; no network; read-only root; cap-drop; no-new-privileges;
  CPU/mem/pids/output/timeout limits; destroyed on delete/shutdown.
- Artifacts: PNG/JPEG/WebP created-or-modified in workspace → inline SSE
  `artifact` events, base64, size-capped, no physical paths. No artifact
  endpoints.
- Streamlit: API-only, never imports app code, renders progress/stream/images,
  input disabled while active, clear-chat deletes API session.
- Prompts: all in `prompts/` package (done), composed from named sections;
  RAG-only prompt must not mention sandbox tools.
- Completion standard: architecture boundaries hold; normal path is AsyncOpenAI
  → vLLM; retrieval agent is the only search mechanism; Qdrant read-only;
  hybrid+rerank implemented; progress and tokens stream; Docker fully optional;
  Streamlit renders via API; prompts localized; offline tests + static checks
  pass. Favor the simplest implementation; do not build a platform.

## 9. Commands

```bash
poetry install --all-groups          # deps (torch CPU on win; CUDA wheels on linux — note in README)
poetry run pytest                    # offline tests
poetry run ruff format .             # format
poetry run ruff check .              # lint  (line-length 100, rules E,F,I,UP,B,SIM,RUF)
poetry run mypy src                  # types (lenient: ignore_missing_imports)
poetry run python -m ragchat         # API (after __main__.py exists)
poetry run streamlit run ui/streamlit_app.py
docker build -t ragchat-sandbox:latest sandbox/   # optional, for SANDBOX_MODE=docker
```

## 10. Final words

Work in this order: §5 steps 0→10; write tests alongside each step rather than
at the end. Trust the api-notes over instinct — every signature in them was
copied from the installed source. The design above was chosen so that each
spec bullet in §6/§8 has an obvious home; if you find yourself adding a queue,
an event bus, an ingestion path, or a second corpus tool, stop — it's
explicitly out of scope. Keep it small; it's a proof of concept.
