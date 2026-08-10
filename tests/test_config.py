from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragchat.config import ModelBackend, SandboxMode, Settings


def test_local_vllm_and_disabled_sandbox_are_the_defaults() -> None:
    configured = Settings(
        _env_file=None,
        vllm_base_url="http://vllm.test/v1",
        vllm_model="test-model",
        qdrant_collection="corpus",
    )

    assert configured.api_host == "127.0.0.1"
    assert configured.api_port == 8080
    assert configured.model_backend is ModelBackend.VLLM
    assert configured.sandbox_mode is SandboxMode.DISABLED


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"vllm_base_url": None, "vllm_model": "model"},
            "VLLM_BASE_URL",
        ),
        (
            {"vllm_base_url": "http://vllm.test/v1", "vllm_model": None},
            "VLLM_MODEL",
        ),
    ],
)
def test_vllm_branch_requires_only_its_runtime_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        Settings(
            _env_file=None,
            qdrant_collection="corpus",
            openai_api_key=None,
            **overrides,
        )


def test_openai_branch_requires_its_key_but_not_vllm_configuration() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(
            _env_file=None,
            model_backend=ModelBackend.OPENAI,
            openai_api_key=None,
            vllm_base_url=None,
            vllm_model=None,
            qdrant_collection="corpus",
        )

    configured = Settings(
        _env_file=None,
        model_backend=ModelBackend.OPENAI,
        openai_api_key="official-secret",
        vllm_base_url=None,
        vllm_model=None,
        qdrant_collection="corpus",
    )

    assert configured.model_backend is ModelBackend.OPENAI


def test_external_qdrant_collection_is_always_required() -> None:
    with pytest.raises(ValidationError, match="QDRANT_COLLECTION"):
        Settings(
            _env_file=None,
            vllm_base_url="http://vllm.test/v1",
            vllm_model="test-model",
            qdrant_collection=None,
        )
