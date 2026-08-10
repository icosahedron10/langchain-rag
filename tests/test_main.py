"""Offline tests for the ``python -m ragchat`` wiring."""

from __future__ import annotations

import pytest

import ragchat.__main__ as main_module
from ragchat.config import ModelBackend, Settings


def test_main_passes_the_configured_app_and_bind_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        api_host="127.0.0.42",
        api_port=8765,
        model_backend=ModelBackend.VLLM,
        vllm_base_url="http://vllm.test/v1",
        vllm_model="test-model",
        qdrant_collection="test-corpus",
    )
    app = object()
    created_with: list[Settings] = []
    run_calls: list[tuple[object, str, int]] = []

    def build_settings() -> Settings:
        return settings

    def build_app(received: Settings) -> object:
        created_with.append(received)
        return app

    def run(received_app: object, *, host: str, port: int) -> None:
        run_calls.append((received_app, host, port))

    monkeypatch.setattr(main_module, "Settings", build_settings)
    monkeypatch.setattr(main_module, "create_app", build_app)
    monkeypatch.setattr(main_module.uvicorn, "run", run)

    main_module.main()

    assert created_with == [settings]
    assert run_calls == [(app, "127.0.0.42", 8765)]
