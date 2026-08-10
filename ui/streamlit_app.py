"""Streamlit client for the ragchat API.

A pure HTTP/SSE client: it talks only to the Litestar API and never imports
the manager, agents, or retrieval code, and never touches session workspaces.

Run:  poetry run streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import base64
import binascii
import json
import os

import httpx
import streamlit as st
from httpx_sse import ServerSentEvent, connect_sse

API_URL = os.environ.get("STREAMLIT_API_URL", "http://127.0.0.1:8080").rstrip("/")

st.set_page_config(page_title="Corpus chat", page_icon="📚")


def _init_state() -> None:
    st.session_state.setdefault("api_session_id", None)
    st.session_state.setdefault("messages", [])  # {role, text, images: [(name, bytes)]}
    st.session_state.setdefault("busy", False)
    st.session_state.setdefault("pending", None)


def _ensure_session(client: httpx.Client) -> str:
    if st.session_state.api_session_id is None:
        resp = client.post(f"{API_URL}/sessions")
        resp.raise_for_status()
        st.session_state.api_session_id = resp.json()["session_id"]
    return st.session_state.api_session_id


def _reset_chat() -> None:
    session_id = st.session_state.api_session_id
    if session_id is not None:
        try:
            response = httpx.delete(f"{API_URL}/sessions/{session_id}", timeout=10)
            if response.status_code != 404:
                response.raise_for_status()
        except httpx.HTTPError as exc:
            st.error(f"Could not clear chat: {exc}")
            return
    st.session_state.api_session_id = None
    st.session_state.messages = []
    st.session_state.busy = False
    st.session_state.pending = None


def _decode_artifact(payload: dict) -> tuple[str, bytes] | None:
    try:
        return payload.get("name", "artifact"), base64.b64decode(payload["data"])
    except (KeyError, binascii.Error, ValueError):
        return None


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["text"]:
                st.markdown(message["text"])
            for name, image in message.get("images", []):
                st.image(image, caption=name)


def _stream_reply(prompt: str) -> None:
    """POST the chat message and render SSE events as they arrive."""
    answer = ""
    images: list[tuple[str, bytes]] = []
    terminal: str | None = None
    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=True)
        placeholder = st.empty()
        try:
            with httpx.Client(timeout=httpx.Timeout(10, read=None)) as client:
                session_id = _ensure_session(client)
                with connect_sse(
                    client,
                    "POST",
                    f"{API_URL}/sessions/{session_id}/chat",
                    json={"message": prompt},
                ) as events:
                    if events.response.status_code == 409:
                        st.warning("A request is already running for this session.")
                        status.update(label="Session busy", state="error")
                        return
                    events.response.raise_for_status()
                    for event in events.iter_sse():
                        answer, terminal = _handle_event(event, status, placeholder, answer, images)
                        if terminal is not None:
                            break
            if terminal == "done":
                status.update(label="Done", state="complete", expanded=False)
            elif terminal == "error":
                status.update(label="Failed", state="error")
            else:
                status.update(label="Stream ended unexpectedly", state="error")
        except httpx.HTTPError as exc:
            status.update(label="API request failed", state="error")
            st.error(f"API request failed: {exc}")
    if answer or images:
        st.session_state.messages.append({"role": "assistant", "text": answer, "images": images})


def _handle_event(
    event: ServerSentEvent,
    status,
    placeholder,
    answer: str,
    images: list[tuple[str, bytes]],
) -> tuple[str, str | None]:
    """Apply one SSE event; return answer and its optional terminal outcome."""
    try:
        payload = json.loads(event.data) if event.data else {}
    except json.JSONDecodeError:
        return answer, None
    if event.event == "progress":
        status.write(payload.get("text", ""))
    elif event.event == "message":
        answer += payload.get("text", "")
        placeholder.markdown(answer)
    elif event.event == "artifact":
        decoded = _decode_artifact(payload)
        if decoded is not None:
            images.append(decoded)
            st.image(decoded[1], caption=decoded[0])
    elif event.event == "error":
        st.error(payload.get("message", "The request failed."))
        return answer, "error"
    elif event.event == "done":
        return answer, "done"
    return answer, None


_init_state()

st.title("📚 Corpus chat")
st.caption("Answers come from the configured document corpus.")

with st.sidebar:
    st.button(
        "Clear chat",
        on_click=_reset_chat,
        disabled=st.session_state.busy,
        use_container_width=True,
    )

_render_history()

user_input = st.chat_input("Ask about the corpus…", disabled=st.session_state.busy)
if st.session_state.busy and st.session_state.pending is not None:
    prompt = st.session_state.pending
    st.session_state.pending = None
    try:
        _stream_reply(prompt)
    finally:
        st.session_state.busy = False
    st.rerun()
elif user_input and not st.session_state.busy:
    st.session_state.messages.append({"role": "user", "text": user_input, "images": []})
    st.session_state.pending = user_input
    st.session_state.busy = True
    st.rerun()
