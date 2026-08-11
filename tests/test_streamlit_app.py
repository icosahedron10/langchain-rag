from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import httpx_sse
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"


class FakeSSEStream:
    """Replays scripted SSE frames in place of a live API connection."""

    def __init__(self, events: list[httpx_sse.ServerSentEvent]) -> None:
        self._events = events
        self.response = httpx.Response(
            200,
            request=httpx.Request("POST", "http://api.test/sessions/session-123/chat"),
        )

    def __enter__(self) -> FakeSSEStream:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def iter_sse(self) -> Iterator[httpx_sse.ServerSentEvent]:
        return iter(self._events)


def _app_with_chat() -> AppTest:
    app = AppTest.from_file(str(APP)).run()
    app.session_state["api_session_id"] = "session-123"
    app.session_state["messages"] = [{"role": "user", "text": "Keep this message", "images": []}]
    return app


def _response(status_code: int) -> httpx.Response:
    request = httpx.Request("DELETE", "http://api.test/sessions/session-123")
    return httpx.Response(status_code, request=request)


def test_run_started_event_is_remembered_and_rated_against_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        httpx_sse.ServerSentEvent(event="run_started", data=json.dumps({"run_id": "run-42"})),
        httpx_sse.ServerSentEvent(event="message", data=json.dumps({"text": "answer"})),
        httpx_sse.ServerSentEvent(event="done", data="{}"),
    ]
    posts: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(
        httpx_sse,
        "connect_sse",
        lambda *_args, **_kwargs: FakeSSEStream(events),
    )
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, *, json, timeout: posts.append((url, json)) or _response(204),
    )

    app = _app_with_chat()
    app.session_state["pending"] = "question"
    app.session_state["busy"] = True
    app.run()

    assert app.session_state["run_id"] == "run-42"

    app.button(key="rate_up").click().run()

    assert [url.endswith("/sessions/session-123/feedback") for url, _ in posts] == [True]
    assert [body for _, body in posts] == [{"run_id": "run-42", "score": 1}]
    assert app.session_state["rated_run_id"] == "run-42"
    assert len(app.error) == 0


def test_thumbs_down_rates_the_run_zero_and_surfaces_a_failed_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts: list[dict[str, Any]] = []

    def post(url: str, *, json: dict[str, Any], timeout: int) -> httpx.Response:
        del url, timeout
        posts.append(json)
        return _response(503)

    monkeypatch.setattr(httpx, "post", post)

    app = _app_with_chat()
    app.session_state["run_id"] = "run-42"
    app.run()

    app.button(key="rate_down").click().run()

    assert posts == [{"run_id": "run-42", "score": 0}]
    assert app.session_state["rated_run_id"] is None
    assert app.error[0].value.startswith("Could not send feedback:")


@pytest.mark.parametrize("status_code", [204, 404])
def test_clear_chat_clears_local_state_after_api_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    app = _app_with_chat()
    calls: list[tuple[str, int]] = []

    def delete(url: str, *, timeout: int) -> httpx.Response:
        calls.append((url, timeout))
        return _response(status_code)

    monkeypatch.setattr(httpx, "delete", delete)

    app.button[0].click().run()

    assert calls == [("http://127.0.0.1:8080/sessions/session-123", 10)]
    assert app.session_state["api_session_id"] is None
    assert app.session_state["messages"] == []
    assert len(app.error) == 0


@pytest.mark.parametrize(
    "failure",
    [
        lambda: _response(409),
        lambda: _response(503),
        lambda: httpx.ConnectError(
            "API unavailable",
            request=httpx.Request("DELETE", "http://api.test/sessions/session-123"),
        ),
    ],
    ids=["session-busy", "server-error", "network-error"],
)
def test_clear_chat_retains_local_state_and_surfaces_delete_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure: Callable[[], httpx.Response | httpx.HTTPError],
) -> None:
    app = _app_with_chat()
    outcome = failure()

    def delete(url: str, *, timeout: int) -> httpx.Response:
        del url, timeout
        if isinstance(outcome, httpx.HTTPError):
            raise outcome
        return outcome

    monkeypatch.setattr(httpx, "delete", delete)

    app.button[0].click().run()

    assert app.session_state["api_session_id"] == "session-123"
    assert app.session_state["messages"] == [
        {"role": "user", "text": "Keep this message", "images": []}
    ]
    assert len(app.error) == 1
    assert app.error[0].value.startswith("Could not clear chat:")
