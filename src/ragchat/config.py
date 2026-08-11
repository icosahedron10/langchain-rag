"""Application configuration.

One strict Pydantic Settings model. Contradictory or incomplete configuration
is rejected at startup with actionable errors; nothing is silently repaired.
"""

from __future__ import annotations

import enum

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelBackend(enum.StrEnum):
    VLLM = "vllm"
    OPENAI = "openai"


class SandboxMode(enum.StrEnum):
    DISABLED = "disabled"
    DOCKER = "docker"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    # Litestar bind address. Local single-user app: default to loopback.
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8081, ge=1, le=65535)

    # Streamlit client -> API base URL.
    streamlit_api_url: str = "http://127.0.0.1:8081"

    # Model runtime. vLLM is the long-term default; OpenAI is a temporary,
    # removable startup option selected once before the API starts.
    model_backend: ModelBackend = ModelBackend.VLLM
    vllm_base_url: str | None = None
    vllm_api_key: SecretStr = SecretStr("EMPTY")
    vllm_model: str | None = None
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-luna"

    # Qdrant: externally populated, read-only dependency.
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str | None = None
    qdrant_dense_vector_name: str = "sentence-transformers/all-mpnet-base-v2"
    qdrant_sparse_vector_name: str = "Qdrant/bm25"

    # Retrieval models (must match the contract used to populate the collection).
    dense_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    sparse_embedding_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Citation validation. Flagging is always on; strict mode also re-prompts the
    # orchestrator once, naming the citations it could not verify.
    citation_strict_mode: bool = False

    # Optional LangSmith tracing.
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str | None = None

    # Optional Docker sandbox.
    sandbox_mode: SandboxMode = SandboxMode.DISABLED
    sandbox_image: str = "ragchat-sandbox:latest"
    sandbox_command_timeout_seconds: int = Field(default=60, ge=1)
    sandbox_cpu_limit: float = Field(default=1.0, gt=0)
    sandbox_memory_limit: str = "512m"
    sandbox_pids_limit: int = Field(default=128, ge=1)
    sandbox_output_limit_chars: int = Field(default=20_000, ge=1)
    # Inline image artifacts larger than this (raw bytes, pre-base64) are skipped.
    artifact_max_bytes: int = Field(default=5_000_000, ge=1)

    @model_validator(mode="after")
    def _validate_completeness(self) -> Settings:
        problems: list[str] = []
        if self.model_backend is ModelBackend.VLLM:
            if not self.vllm_base_url:
                problems.append(
                    "MODEL_BACKEND=vllm requires VLLM_BASE_URL (e.g. http://127.0.0.1:8000/v1)."
                )
            if not self.vllm_model:
                problems.append(
                    "MODEL_BACKEND=vllm requires VLLM_MODEL "
                    "(the model name served by your vLLM instance)."
                )
        elif self.model_backend is ModelBackend.OPENAI:
            if self.openai_api_key is None or not self.openai_api_key.get_secret_value():
                problems.append("MODEL_BACKEND=openai requires OPENAI_API_KEY.")
        if self.langsmith_tracing and (
            self.langsmith_api_key is None or not self.langsmith_api_key.get_secret_value()
        ):
            problems.append("LANGSMITH_TRACING=true requires LANGSMITH_API_KEY.")
        if not self.qdrant_collection:
            problems.append(
                "QDRANT_COLLECTION is required and must name an existing, "
                "externally populated collection."
            )
        if problems:
            raise ValueError("Invalid configuration:\n" + "\n".join(f"  - {p}" for p in problems))
        return self
