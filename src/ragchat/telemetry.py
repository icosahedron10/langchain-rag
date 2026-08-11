"""Optional LangSmith tracing lifecycle."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Any

from langsmith import Client
from langsmith.run_helpers import get_current_run_tree, tracing_context

from ragchat.config import Settings

EVIDENCE_TEXT_LIMIT = 20_000

_RETRIEVAL_DIGEST: ContextVar[dict[str, Any] | None] = ContextVar(
    "ragchat_retrieval_digest",
    default=None,
)


def record_retrieval_digest(
    *,
    evidence_text: str,
    pages: Sequence[int | None],
    answerable: bool,
) -> None:
    """Fold one search_corpus result into the current turn's retrieval digest."""

    digest = _RETRIEVAL_DIGEST.get()
    if digest is None:
        return

    digest["search_corpus_calls"] += 1
    digest["answerable"] = digest["answerable"] or answerable
    if evidence_text:
        separator = "\n\n" if digest["evidence_text"] else ""
        digest["evidence_text"] = (digest["evidence_text"] + separator + evidence_text)[
            :EVIDENCE_TEXT_LIMIT
        ]
    for page in pages:
        if page is not None and page not in digest["pages"]:
            digest["pages"].append(page)

    run_tree = get_current_run_tree()
    if run_tree is not None:
        digest["root_run_id"] = str(run_tree.trace_id)


class Telemetry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._closed = False
        self._client = (
            Client(
                api_url=settings.langsmith_endpoint,
                api_key=settings.langsmith_api_key.get_secret_value(),
            )
            if settings.langsmith_tracing and settings.langsmith_api_key is not None
            else None
        )

    @contextmanager
    def trace_chat(self, session_id: str) -> Iterator[None]:
        digest: dict[str, Any] = {
            "evidence_text": "",
            "pages": [],
            "answerable": False,
            "search_corpus_calls": 0,
            "root_run_id": None,
        }
        # The digest is only known after retrieval runs, so the turn accumulates into
        # this mutable holder and it is flushed onto the root run as the trace closes.
        token = _RETRIEVAL_DIGEST.set(digest)
        try:
            with tracing_context(
                project_name=self._settings.langsmith_project,
                tags=["ragchat", "chat"],
                metadata={"session_id": session_id},
                enabled=self._client is not None,
                client=self._client,
            ):
                yield
        finally:
            with suppress(ValueError):
                # A turn resumed in another context cannot reset its own token.
                _RETRIEVAL_DIGEST.reset(token)
            self._flush_retrieval_digest(session_id, digest)

    def _flush_retrieval_digest(self, session_id: str, digest: dict[str, Any]) -> None:
        root_run_id = digest.pop("root_run_id")
        if self._client is None or root_run_id is None or not digest["search_corpus_calls"]:
            return

        with suppress(Exception):
            self._client.update_run(
                root_run_id,
                extra={
                    "metadata": {
                        "session_id": session_id,
                        "retrieval_digest": digest,
                    }
                },
            )

    def close(self) -> None:
        if self._client is not None and not self._closed:
            self._client.close()
            self._closed = True
