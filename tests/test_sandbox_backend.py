from __future__ import annotations

import base64
from pathlib import Path
from typing import cast
from unittest.mock import create_autospec

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

import ragchat.sandbox.backend as backend_module
from ragchat.config import Settings
from ragchat.sandbox.backend import SessionRoutingBackend, SessionSandboxHandle
from ragchat.sandbox.docker_session import DockerSandboxSession, ExecResult


def settings(
    *,
    artifact_max_bytes: int = 5_000_000,
    sandbox_command_timeout_seconds: int = 60,
) -> Settings:
    return Settings(
        vllm_base_url="http://vllm.test/v1",
        vllm_model="test-model",
        qdrant_collection="test-corpus",
        artifact_max_bytes=artifact_max_bytes,
        sandbox_command_timeout_seconds=sandbox_command_timeout_seconds,
        _env_file=None,
    )


def handle_with_files(files: FilesystemBackend) -> SessionSandboxHandle:
    return SessionSandboxHandle(
        session=cast("DockerSandboxSession", object()),
        files=files,
        settings=settings(),
    )


def set_thread_id(monkeypatch: pytest.MonkeyPatch, active: dict[str, str]) -> None:
    import langgraph.config

    monkeypatch.setattr(
        langgraph.config,
        "get_config",
        lambda: {"configurable": {"thread_id": active["thread_id"]}},
    )


def test_delegates_all_file_operations_and_preserves_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = create_autospec(FilesystemBackend, instance=True)
    results = {
        name: object()
        for name in (
            "ls",
            "read",
            "grep",
            "glob",
            "write",
            "edit",
            "delete",
            "upload_files",
            "download_files",
        )
    }
    for name, result in results.items():
        getattr(files, name).return_value = result

    resolved: list[str] = []
    active = {"thread_id": "session-one"}
    handle = handle_with_files(files)

    def resolve(session_id: str) -> SessionSandboxHandle:
        resolved.append(session_id)
        return handle

    set_thread_id(monkeypatch, active)
    backend = SessionRoutingBackend(resolve)

    uploads = [("/input.bin", b"input")]
    downloads = ["/one.txt", "/two.txt"]
    assert backend.ls("/docs") is results["ls"]
    assert backend.read("/docs/a.txt", offset=7, limit=11) is results["read"]
    assert (
        backend.grep(
            "needle",
            path="/src",
            glob="*.py",
            max_count=13,
            context_lines=2,
        )
        is results["grep"]
    )
    assert backend.glob("**/*.png", path="/charts") is results["glob"]
    assert backend.write("/notes.txt", "content") is results["write"]
    assert backend.edit("/notes.txt", "old", "new", replace_all=True) is results["edit"]
    assert backend.delete("/obsolete") is results["delete"]
    assert backend.upload_files(uploads) is results["upload_files"]
    assert backend.download_files(downloads) is results["download_files"]

    files.ls.assert_called_once_with("/docs")
    files.read.assert_called_once_with("/docs/a.txt", offset=7, limit=11)
    files.grep.assert_called_once_with(
        "needle",
        path="/src",
        glob="*.py",
        max_count=13,
        context_lines=2,
    )
    files.glob.assert_called_once_with("**/*.png", path="/charts")
    files.write.assert_called_once_with("/notes.txt", "content")
    files.edit.assert_called_once_with("/notes.txt", "old", "new", replace_all=True)
    files.delete.assert_called_once_with("/obsolete")
    files.upload_files.assert_called_once_with(uploads)
    files.download_files.assert_called_once_with(downloads)
    assert resolved == ["session-one"] * 9


def test_routes_each_call_using_the_current_thread_id(monkeypatch: pytest.MonkeyPatch) -> None:
    first_files = create_autospec(FilesystemBackend, instance=True)
    second_files = create_autospec(FilesystemBackend, instance=True)
    handles = {
        "first": handle_with_files(first_files),
        "second": handle_with_files(second_files),
    }
    active = {"thread_id": "first"}
    set_thread_id(monkeypatch, active)
    backend = SessionRoutingBackend(handles.__getitem__)

    backend.write("/answer.txt", "one")
    active["thread_id"] = "second"
    backend.write("/answer.txt", "two")

    first_files.write.assert_called_once_with("/answer.txt", "one")
    second_files.write.assert_called_once_with("/answer.txt", "two")


@pytest.mark.asyncio
async def test_inherited_async_file_operation_routes_through_sync_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = create_autospec(FilesystemBackend, instance=True)
    expected = object()
    files.read.return_value = expected
    active = {"thread_id": "async-session"}
    set_thread_id(monkeypatch, active)
    backend = SessionRoutingBackend(lambda _: handle_with_files(files))

    result = await backend.aread("/async.txt", offset=3, limit=4)

    assert result is expected
    files.read.assert_called_once_with("/async.txt", offset=3, limit=4)


def test_is_sandbox_protocol_with_constant_id_and_no_sync_execution() -> None:
    backend = SessionRoutingBackend(lambda _: cast("SessionSandboxHandle", object()))

    assert isinstance(backend, SandboxBackendProtocol)
    assert backend.id == "ragchat-docker"
    with pytest.raises(NotImplementedError):
        backend.execute("pwd", timeout=3)


class ArtifactProducingSession:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.commands: list[tuple[str, int | None]] = []

    async def execute(self, command: str, *, timeout: int | None = None) -> ExecResult:
        self.commands.append((command, timeout))
        (self.workspace / "changed.png").write_bytes(b"new!")
        chart_dir = self.workspace / "charts"
        chart_dir.mkdir()
        (chart_dir / "new.webp").write_bytes(b"webp")
        (self.workspace / "large.jpg").write_bytes(b"123456")
        return ExecResult(exit_code=9, output="combined output")


class RecordingExecSession:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.timeouts: list[int | None] = []

    async def execute(self, _command: str, *, timeout: int | None = None) -> ExecResult:
        self.timeouts.append(timeout)
        return ExecResult(exit_code=0, output="bounded", truncated=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(("requested", "expected"), [(None, 7), (3, 3), (99, 7)])
async def test_aexecute_propagates_positive_timeout_capped_by_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    requested: int | None,
    expected: int,
) -> None:
    session = RecordingExecSession(tmp_path)
    handle = SessionSandboxHandle(
        session=cast("DockerSandboxSession", session),
        files=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        settings=settings(sandbox_command_timeout_seconds=7),
    )
    set_thread_id(monkeypatch, {"thread_id": "timeout-session"})
    monkeypatch.setattr(backend_module, "get_stream_writer", lambda: lambda _event: None)
    backend = SessionRoutingBackend(lambda _: handle)

    response = await backend.aexecute("bounded command", timeout=requested)

    assert session.timeouts == [expected]
    assert response == ExecuteResponse(output="bounded", exit_code=0, truncated=True)


@pytest.mark.asyncio
async def test_aexecute_rejects_non_positive_timeout_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session = RecordingExecSession(tmp_path)
    handle = SessionSandboxHandle(
        session=cast("DockerSandboxSession", session),
        files=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        settings=settings(sandbox_command_timeout_seconds=7),
    )
    set_thread_id(monkeypatch, {"thread_id": "timeout-session"})
    backend = SessionRoutingBackend(lambda _: handle)

    with pytest.raises(ValueError, match="must be positive"):
        await backend.aexecute("invalid", timeout=0)

    assert session.timeouts == []


@pytest.mark.asyncio
async def test_aexecute_streams_changed_artifacts_and_skipped_progress_without_host_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "changed.png").write_bytes(b"old")
    (tmp_path / "unchanged.png").write_bytes(b"same")
    session = ArtifactProducingSession(tmp_path)
    handle = SessionSandboxHandle(
        session=cast("DockerSandboxSession", session),
        files=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        settings=settings(artifact_max_bytes=5),
    )
    active = {"thread_id": "artifact-session"}
    set_thread_id(monkeypatch, active)
    events: list[dict[str, str]] = []
    monkeypatch.setattr(backend_module, "get_stream_writer", lambda: events.append)
    backend = SessionRoutingBackend(lambda _: handle)

    response = await backend.aexecute("make images", timeout=17)

    assert response == ExecuteResponse(output="combined output", exit_code=9)
    assert session.commands == [("make images", 17)]
    assert events == [
        {
            "type": "artifact",
            "name": "changed.png",
            "media_type": "image/png",
            "data": base64.b64encode(b"new!").decode("ascii"),
        },
        {
            "type": "artifact",
            "name": "charts/new.webp",
            "media_type": "image/webp",
            "data": base64.b64encode(b"webp").decode("ascii"),
        },
        {
            "type": "progress",
            "text": (
                "Skipped image artifact large.jpg (collection limit or concurrent file change)"
            ),
        },
    ]
    assert "unchanged.png" not in repr(events)
    assert str(tmp_path) not in repr(events)
