from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

from ragchat import providers
from ragchat.config import ModelBackend, Settings


class FakeAsyncOpenAI:
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        self.completions = object()
        self.chat = SimpleNamespace(completions=self.completions)


class FakeChatOpenAI:
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def reset_provider_fakes() -> None:
    FakeAsyncOpenAI.calls.clear()
    FakeChatOpenAI.calls.clear()


@pytest.fixture
def fake_provider_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    openai = ModuleType("openai")
    openai.AsyncOpenAI = FakeAsyncOpenAI  # type: ignore[attr-defined]
    langchain_openai = ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = FakeChatOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai)


def vllm_settings() -> Settings:
    return Settings(
        vllm_base_url="http://vllm.test:8000/v1",
        vllm_api_key="EMPTY",
        vllm_model="local-model",
        qdrant_collection="corpus",
        _env_file=None,
    )


def openai_settings() -> Settings:
    return Settings(
        model_backend=ModelBackend.OPENAI,
        openai_api_key="official-secret",
        openai_model="official-model",
        qdrant_collection="corpus",
        _env_file=None,
    )


def test_vllm_is_default_and_does_not_require_openai_environment_key(
    monkeypatch: pytest.MonkeyPatch,
    fake_provider_modules: None,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = vllm_settings()

    model = providers.build_chat_model(settings)

    assert settings.model_backend is ModelBackend.VLLM
    assert isinstance(model, FakeChatOpenAI)
    assert FakeAsyncOpenAI.calls == [{"base_url": "http://vllm.test:8000/v1", "api_key": "EMPTY"}]


def test_vllm_injects_both_async_client_handles(
    fake_provider_modules: None,
) -> None:
    model = providers.build_chat_model(vllm_settings())

    assert isinstance(model, FakeChatOpenAI)
    client_kwargs = model.kwargs
    root_client = client_kwargs["root_async_client"]
    assert isinstance(root_client, FakeAsyncOpenAI)
    assert client_kwargs == {
        "model": "local-model",
        "base_url": "http://vllm.test:8000/v1",
        "api_key": "EMPTY",
        "async_client": root_client.chat.completions,
        "root_async_client": root_client,
        "max_tokens": 32_768,
        "timeout": 90.0,
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def test_vllm_branch_does_not_import_removable_openai_backend(
    monkeypatch: pytest.MonkeyPatch,
    fake_provider_modules: None,
) -> None:
    monkeypatch.delitem(sys.modules, "ragchat.openai_backend", raising=False)

    providers.build_chat_model(vllm_settings())

    assert "ragchat.openai_backend" not in sys.modules


def test_build_chat_model_selects_only_the_configured_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[tuple[str, Settings]] = []
    vllm_model = object()
    openai_model = object()

    def fake_vllm(settings: Settings) -> object:
        selected.append(("vllm", settings))
        return vllm_model

    def fake_openai(settings: Settings) -> object:
        selected.append(("openai", settings))
        return openai_model

    openai_backend = ModuleType("ragchat.openai_backend")
    openai_backend.build_openai_chat_model = fake_openai  # type: ignore[attr-defined]
    monkeypatch.setattr(providers, "_build_vllm_chat_model", fake_vllm)
    monkeypatch.setitem(sys.modules, "ragchat.openai_backend", openai_backend)

    vllm_config = vllm_settings()
    assert providers.build_chat_model(vllm_config) is vllm_model
    assert selected == [("vllm", vllm_config)]

    selected.clear()
    openai_config = openai_settings()
    assert providers.build_chat_model(openai_config) is openai_model
    assert selected == [("openai", openai_config)]


def test_openai_backend_constructs_only_official_model_configuration(
    fake_provider_modules: None,
) -> None:
    from ragchat.openai_backend import build_openai_chat_model

    model = build_openai_chat_model(openai_settings())

    assert isinstance(model, FakeChatOpenAI)
    assert model.kwargs == {
        "model": "official-model",
        "api_key": "official-secret",
        "reasoning": {"effort": "low"},
        "timeout": 90.0,
        "max_tokens": 32_768,
        "max_retries": 2,
    }
    assert FakeAsyncOpenAI.calls == []


def test_configured_request_bounds_reach_both_backends(
    fake_provider_modules: None,
) -> None:
    from ragchat.openai_backend import build_openai_chat_model

    vllm = providers.build_chat_model(
        vllm_settings().model_copy(update={"model_request_timeout_seconds": 12.5})
    )
    official = build_openai_chat_model(
        openai_settings().model_copy(
            update={"model_request_timeout_seconds": 12.5, "model_max_output_tokens": 4_096}
        )
    )

    assert isinstance(vllm, FakeChatOpenAI)
    assert isinstance(official, FakeChatOpenAI)
    assert vllm.kwargs["timeout"] == 12.5
    assert official.kwargs["timeout"] == 12.5
    assert official.kwargs["max_tokens"] == 4_096
