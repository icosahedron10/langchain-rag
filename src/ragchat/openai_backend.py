"""Temporary, removable support for the official OpenAI model backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from ragchat.config import Settings


def build_openai_chat_model(settings: Settings) -> BaseChatModel:
    """Build the configured official OpenAI chat model."""
    from langchain_openai import ChatOpenAI

    if settings.openai_api_key is None:
        raise ValueError("OpenAI backend requires an API key.")

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),  # type: ignore[arg-type]
        reasoning={"effort": "low"},
        timeout=settings.model_request_timeout_seconds,
        max_tokens=settings.model_max_output_tokens,
        max_retries=2,
    )
