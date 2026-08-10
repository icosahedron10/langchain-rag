"""Startup selection for the configured chat-model provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragchat.config import ModelBackend

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from ragchat.config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Build exactly the provider selected at startup, without fallback."""
    if settings.model_backend is ModelBackend.OPENAI:
        # Temporary workaround: deleting this branch and openai_backend.py removes
        # the OpenAI provider without affecting the vLLM implementation.
        from ragchat.openai_backend import build_openai_chat_model

        return build_openai_chat_model(settings)

    return _build_vllm_chat_model(settings)


def _build_vllm_chat_model(settings: Settings) -> BaseChatModel:
    """Build ChatOpenAI over the configured vLLM OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI
    from openai import AsyncOpenAI

    base_url = settings.vllm_base_url
    model = settings.vllm_model
    if base_url is None or model is None:
        raise ValueError("vLLM backend requires a base URL and model name.")

    key = settings.vllm_api_key.get_secret_value()
    client = AsyncOpenAI(base_url=base_url, api_key=key)
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=key,  # type: ignore[arg-type]  # Pydantic accepts and coerces str.
        async_client=client.chat.completions,
        root_async_client=client,
    )
