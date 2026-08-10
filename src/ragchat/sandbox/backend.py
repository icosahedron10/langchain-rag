"""Per-run routing between deepagents filesystem tools and session sandboxes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from langgraph.config import get_stream_writer

from ragchat.config import Settings
from ragchat.sandbox.artifacts import collect_image_artifacts, snapshot_workspace
from ragchat.sandbox.docker_session import DockerSandboxSession


@dataclass
class SessionSandboxHandle:
    """The filesystem and container resources owned by one chat session."""

    session: DockerSandboxSession
    files: FilesystemBackend
    settings: Settings


class SessionRoutingBackend(SandboxBackendProtocol):
    """Route one graph-level backend to the current session's resources."""

    def __init__(self, resolve: Callable[[str], SessionSandboxHandle]) -> None:
        self._resolve = resolve

    def _handle(self) -> SessionSandboxHandle:
        from langgraph.config import get_config

        return self._resolve(get_config()["configurable"]["thread_id"])

    def ls(self, path: str) -> LsResult:
        return self._handle().files.ls(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._handle().files.read(file_path, offset=offset, limit=limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
        context_lines: int = 0,
    ) -> GrepResult:
        return self._handle().files.grep(
            pattern,
            path=path,
            glob=glob,
            max_count=max_count,
            context_lines=context_lines,
        )

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self._handle().files.glob(pattern, path=path)

    def write(self, file_path: str, content: str) -> WriteResult:
        return self._handle().files.write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return self._handle().files.edit(
            file_path,
            old_string,
            new_string,
            replace_all=replace_all,
        )

    def delete(self, file_path: str) -> DeleteResult:
        return self._handle().files.delete(file_path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._handle().files.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._handle().files.download_files(paths)

    @property
    def id(self) -> str:
        return "ragchat-docker"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        raise NotImplementedError

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        handle = self._handle()
        configured_timeout = handle.settings.sandbox_command_timeout_seconds
        if timeout is not None and timeout < 1:
            raise ValueError("sandbox command timeout must be positive")
        effective_timeout = min(timeout or configured_timeout, configured_timeout)
        before = snapshot_workspace(handle.session.workspace)
        result = await handle.session.execute(command, timeout=effective_timeout)
        artifacts, skipped = collect_image_artifacts(
            handle.session.workspace,
            before,
            handle.settings.artifact_max_bytes,
        )

        writer = get_stream_writer()
        for artifact in artifacts:
            writer(artifact.model_dump())
        for name in skipped:
            writer(
                {
                    "type": "progress",
                    "text": (
                        f"Skipped image artifact {name} "
                        "(collection limit or concurrent file change)"
                    ),
                }
            )

        return ExecuteResponse(
            output=result.output,
            exit_code=result.exit_code,
            truncated=result.truncated,
        )
