# ragchat

`ragchat` is a local, single-user proof-of-concept chat application for an
externally populated Qdrant corpus. A Litestar API streams agent progress,
answer text, and optional image artifacts over server-sent events (SSE) to a
Streamlit UI.

The retrieval agent chooses up to three searches per question. Each search
combines MPNet dense vectors with Qdrant BM25 sparse vectors using reciprocal
rank fusion, retrieves 20 candidates, deduplicates them by Qdrant point ID, and
scores every deduplicated candidate with `BAAI/bge-reranker-base` before keeping
the top 10. Application code resolves the agent's selected IDs back to observed
verbatim passages before the orchestrator answers with document-and-page
citations.

This repository does **not** ingest, create, repair, update, or delete Qdrant
data. The configured collection is an external, read-only dependency.

## Prerequisites

- Python 3.12 (the project requires `>=3.12,<3.13`)
- [Poetry](https://python-poetry.org/)
- A populated Qdrant collection reachable over HTTP
- One chat-model runtime:
  - an OpenAI-compatible vLLM server (default), or
  - an OpenAI API key with `MODEL_BACKEND=openai`
- Docker and a built sandbox image only when `SANDBOX_MODE=docker`

The first API startup may download the configured embedding, sparse-embedding,
and reranker models unless they are already cached.

## External Qdrant contract

Populate the collection outside this application. With the default settings,
every point must use these named vectors:

| Name | Kind | Contract |
|---|---|---|
| `dense` | dense | 768 dimensions, cosine distance, produced by `sentence-transformers/all-mpnet-base-v2` |
| `sparse` | sparse | produced by FastEmbed model `Qdrant/bm25` |

The names are configurable through `QDRANT_DENSE_VECTOR_NAME` and
`QDRANT_SPARSE_VECTOR_NAME`, but the configured names and embedding models must
match the existing collection. Each point must also have this payload shape:

```json
{
  "page_content": "The verbatim passage text.",
  "metadata": {
    "source": "documents/guide.pdf",
    "page": 12
  }
}
```

`page_content`, `metadata.source`, and `metadata.page` must all be present.
Qdrant supplies the stable point ID; it is not a payload field. At startup,
`ragchat` checks connectivity, collection existence and non-emptiness, payload
fields, and named-vector compatibility, then fails with an actionable error if
the contract does not match. It never attempts to fix the collection.

## Setup

Copy the example configuration and edit it for your model runtime and existing
Qdrant collection:

```bash
cp .env.example .env
poetry install --all-groups
```

PowerShell equivalent:

```powershell
Copy-Item .env.example .env
poetry install --all-groups
```

For the default vLLM backend, `VLLM_BASE_URL`, `VLLM_MODEL`, and
`QDRANT_COLLECTION` are required. The example file targets vLLM on port 8000,
Qdrant on port 6333, and the ragchat API on port 8080.

Start the API and UI in separate terminals:

```bash
poetry run python -m ragchat
poetry run streamlit run ui/streamlit_app.py
```

The API listens at `http://127.0.0.1:8080` by default. Streamlit normally opens
the UI at `http://localhost:8501`; it communicates only with the API configured
by `STREAMLIT_API_URL`.

Build the Python package with:

```bash
poetry build
```

For optional sandbox tools, build the image before starting the API with
`SANDBOX_MODE=docker`:

```bash
docker build -t ragchat-sandbox:latest sandbox/
```

## API

| Method and path | Behavior |
|---|---|
| `GET /health` | Returns `{"status":"ok"}`. |
| `POST /sessions` | Creates an in-memory conversation and returns `{"session_id":"..."}` (201). |
| `POST /sessions/{session_id}/chat` | Accepts `{"message":"..."}` and streams `progress`, `message`, `artifact`, `done`, or `error` SSE events. |
| `DELETE /sessions/{session_id}` | Deletes the conversation and its sandbox, if any (204). |

Only one chat request may run per session; overlapping requests receive 409
rather than being queued. Sessions and checkpoints are in memory and disappear
when the API stops.

## Configuration

Settings are read from environment variables and `.env`. See `.env.example`
for the complete list.

| Variable | Default | Purpose |
|---|---|---|
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8080` | Litestar bind address. Keep the unauthenticated single-user API on a trusted interface. |
| `STREAMLIT_API_URL` | `http://127.0.0.1:8080` | API base URL used by Streamlit. |
| `MODEL_BACKEND` | `vllm` | Selects `vllm` or `openai` once at startup; there is no runtime fallback. |
| `VLLM_BASE_URL` | required for vLLM | OpenAI-compatible endpoint, typically `http://127.0.0.1:8000/v1`. |
| `VLLM_API_KEY` | `EMPTY` | Key sent to the vLLM-compatible API. |
| `VLLM_MODEL` | required for vLLM | Model name exposed by the vLLM server. |
| `OPENAI_API_KEY` | required for OpenAI | Credential used only with `MODEL_BACKEND=openai`. |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name. |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Existing Qdrant service. |
| `QDRANT_API_KEY` | unset | Optional Qdrant credential. It is never passed into the sandbox. |
| `QDRANT_COLLECTION` | required | Existing, externally populated collection name. |
| `QDRANT_DENSE_VECTOR_NAME` / `QDRANT_SPARSE_VECTOR_NAME` | `dense` / `sparse` | Existing named vectors used for hybrid retrieval. |
| `DENSE_EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Dense query embedding model; collection vectors must be compatible (768 dimensions). |
| `SPARSE_EMBEDDING_MODEL` | `Qdrant/bm25` | Sparse query embedding model. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder used after hybrid retrieval. |
| `SANDBOX_MODE` | `disabled` | Set to `docker` to expose isolated filesystem and execution tools. |
| `SANDBOX_IMAGE` | `ragchat-sandbox:latest` | Docker image used for session sandboxes. |
| `ARTIFACT_MAX_BYTES` | `5000000` | Maximum raw, pre-base64 size of each inline PNG, JPEG, or WebP artifact. |

Docker resource and output limits are also configurable through
`SANDBOX_COMMAND_TIMEOUT_SECONDS`, `SANDBOX_CPU_LIMIT`,
`SANDBOX_MEMORY_LIMIT`, `SANDBOX_PIDS_LIMIT`, and
`SANDBOX_OUTPUT_LIMIT_CHARS`.

## Sandbox isolation and artifacts

Sandboxing is disabled by default and Docker is not checked in that mode. In
Docker mode, each API session gets one container, created lazily on its first
execution and removed when the session is deleted or the API shuts down. The
container has no network, a read-only root filesystem, dropped capabilities,
`no-new-privileges`, CPU/memory/PID limits, and only that session's workspace
mounted read/write at `/workspace`. The corpus, Qdrant credentials, and model
credentials are never mounted or injected.

On Linux, run the API as a dedicated non-root host user with Docker access.
Docker runs both the container and each command with that workspace owner's
numeric UID:GID, which keeps the bind mount writable without granting the
sandbox broader host-file access. Sandbox launch refuses a root-owned workspace
instead of overriding the image's non-root user with `--user 0:0`. Each command
is wrapped in an in-container GNU `timeout` process group; the Docker CLI gets
only a small cleanup grace, and stdout/stderr are drained continuously while
retained output remains bounded by the configured cap.

New or modified PNG, JPEG, and WebP files in the workspace stream inline as
SSE artifacts. Files larger than `ARTIFACT_MAX_BYTES` raw bytes are skipped and
reported in progress; no filesystem paths or artifact-download endpoints are
exposed.

## Development and Linux deployment

Run the offline test and static-check suite with:

```bash
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy src
poetry run pytest
```

The deployment target is Linux. On a GPU host, select PyTorch wheels that match
the deployed CUDA and driver combination; do not assume the default wheel
chosen by Poetry is appropriate for that machine. Use the current PyTorch
installation guidance for the target CUDA runtime, then verify that Poetry's
resolved environment retains that build. CPU deployments can use CPU wheels.
