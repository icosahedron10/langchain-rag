from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"


def _app_with_chat() -> AppTest:
    app = AppTest.from_file(str(APP)).run()
    app.session_state["api_session_id"] = "session-123"
    app.session_state["messages"] = [{"role": "user", "text": "Keep this message", "images": []}]
    return app


def _response(status_code: int) -> httpx.Response:
    request = httpx.Request("DELETE", "http://api.test/sessions/session-123")
    return httpx.Response(status_code, request=request)


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
