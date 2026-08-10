"""Per-session Docker sandbox driven through the docker CLI.

Each application session owns at most one container, created lazily on the
first execution request and destroyed on session deletion or shutdown. Every
command runs as a separate `docker exec <container> /bin/sh -lc "<command>"`
process, so shell state does not persist between commands; only /workspace
files and container-level state do.

The `runner` seam (an async callable executing a docker CLI invocation)
exists so tests can fake Docker entirely.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragchat.config import Settings
from ragchat.domain import SandboxUnavailableError

# (args after "docker", outer timeout seconds, retained bytes per output pipe)
# -> (exit_code, stdout, stderr)
DockerRunner = Callable[[list[str], float, int], Awaitable[tuple[int, bytes, bytes]]]

_EXIT_TIMEOUT = 124  # conventional coreutils timeout exit code
_CONTROL_CAPTURE_LIMIT_BYTES = 16_384
_DOCKER_EXEC_GRACE_SECONDS = 5.0
_CLEANUP_TIMEOUT_SECONDS = 5.0
_FALLBACK_CONTAINER_UID = 10001
_FALLBACK_CONTAINER_GID = 10001
_PIPE_CHUNK_BYTES = 64 * 1024
_PIPE_TRUNCATION_MARKER = b"\n... docker CLI output capture truncated"

_CLEANUP_SCRIPT = """
import os
import signal
import sys

marker = b"RAGCHAT_EXEC_TOKEN=" + sys.argv[1].encode() + b"\\0"
current = os.getpid()
for entry in os.listdir("/proc"):
    if not entry.isdigit():
        continue
    pid = int(entry)
    if pid in (1, current):
        continue
    try:
        with open(f"/proc/{pid}/environ", "rb") as stream:
            environment = stream.read()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if marker not in environment:
        continue
    try:
        os.kill(pid, signal.SIGKILL)
    except (PermissionError, ProcessLookupError):
        pass
""".strip()


def _numeric_workspace_owner(workspace: Path) -> str:
    """Return the bind-mount owner's numeric identity on Linux."""
    if os.name == "posix":
        stat = workspace.stat()
        return _format_workspace_owner(stat.st_uid, stat.st_gid)
    # Docker Desktop mediates host bind-mount permissions. Keep the image's
    # explicit non-root identity rather than accidentally selecting root from
    # the synthetic uid/gid values returned by Windows stat().
    return f"{_FALLBACK_CONTAINER_UID}:{_FALLBACK_CONTAINER_GID}"


def _format_workspace_owner(uid: int, gid: int) -> str:
    if uid == 0 or gid == 0:
        raise SandboxUnavailableError(
            "Refusing to launch a Docker sandbox with a root UID or GID. "
            "Run the API as a dedicated non-root host user with Docker access."
        )
    return f"{uid}:{gid}"


def _execution_capture_limit(output_limit_chars: int) -> int:
    # Four bytes covers one UTF-8 code point. Leave room for the pipe marker so
    # byte-level truncation is still visible after character-level truncation.
    return (output_limit_chars + len(_PIPE_TRUNCATION_MARKER)) * 4


async def _drain_pipe(
    stream: asyncio.StreamReader,
    max_output_bytes: int,
) -> bytes:
    """Continuously drain a pipe while retaining a bounded prefix."""
    retained = bytearray()
    truncated = False
    while chunk := await stream.read(_PIPE_CHUNK_BYTES):
        remaining = max_output_bytes - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True

    if not truncated:
        return bytes(retained)
    if max_output_bytes <= len(_PIPE_TRUNCATION_MARKER):
        return _PIPE_TRUNCATION_MARKER[:max_output_bytes]
    prefix_bytes = max_output_bytes - len(_PIPE_TRUNCATION_MARKER)
    return bytes(retained[:prefix_bytes]) + _PIPE_TRUNCATION_MARKER


def _kill_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    with suppress(ProcessLookupError):
        proc.kill()


async def _finish_future_despite_cancellation(future: asyncio.Future[Any]) -> bool:
    """Wait for a bounded cleanup future, deferring cancellation until it finishes."""
    interrupted = False
    current = asyncio.current_task()
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            interrupted = True
            if current is not None:
                current.uncancel()
    await future
    return interrupted


async def _settle_cancelled_process(
    proc: asyncio.subprocess.Process,
    tasks: tuple[asyncio.Future[Any], ...],
) -> None:
    """Kill a cancelled CLI process and let its pipe drainers reach EOF."""
    _kill_process(proc)
    settle_task = asyncio.gather(*tasks, return_exceptions=True)
    await _finish_future_despite_cancellation(settle_task)


async def run_docker_cli(
    args: list[str],
    timeout: float,
    max_output_bytes: int,
) -> tuple[int, bytes, bytes]:
    """Run one docker CLI invocation with bounded, continuously drained pipes."""
    if max_output_bytes < 1:
        raise ValueError("max_output_bytes must be positive")
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    wait_task = asyncio.create_task(proc.wait())
    stdout_task = asyncio.create_task(_drain_pipe(proc.stdout, max_output_bytes))
    stderr_task = asyncio.create_task(_drain_pipe(proc.stderr, max_output_bytes))
    tasks: tuple[asyncio.Future[Any], ...] = (wait_task, stdout_task, stderr_task)
    timed_out = False
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout)
    except TimeoutError:
        timed_out = True
        _kill_process(proc)
    except asyncio.CancelledError:
        await _settle_cancelled_process(proc, tasks)
        raise

    try:
        await wait_task
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
    except asyncio.CancelledError:
        await _settle_cancelled_process(proc, tasks)
        raise

    if timed_out:
        timeout_message = b"docker CLI exceeded its outer timeout"
        if len(stderr) + len(timeout_message) + 1 <= max_output_bytes:
            stderr += b"\n" + timeout_message
        return _EXIT_TIMEOUT, stdout, stderr
    return proc.returncode or 0, stdout, stderr


async def assert_docker_available(runner: DockerRunner = run_docker_cli) -> None:
    """Fail clearly at startup when SANDBOX_MODE=docker but Docker is unusable."""
    try:
        code, _, stderr = await runner(
            ["version", "--format", "{{.Server.Version}}"],
            15.0,
            _CONTROL_CAPTURE_LIMIT_BYTES,
        )
    except (OSError, TimeoutError) as exc:
        raise SandboxUnavailableError(
            "SANDBOX_MODE=docker but the docker CLI could not be run. "
            "Install Docker or set SANDBOX_MODE=disabled."
        ) from exc
    if code != 0:
        raise SandboxUnavailableError(
            "SANDBOX_MODE=docker but the Docker daemon is not reachable: "
            f"{stderr.decode(errors='replace').strip()}"
        )


@dataclass
class ExecResult:
    exit_code: int
    output: str  # merged stdout + stderr, truncated to the configured limit
    truncated: bool = False


class DockerSandboxSession:
    """One sandbox container owned by one application session."""

    def __init__(
        self,
        settings: Settings,
        session_id: str,
        workspace: Path,
        runner: DockerRunner = run_docker_cli,
    ) -> None:
        self._settings = settings
        self._workspace = workspace
        self._runner = runner
        self._container = f"ragchat-sandbox-{session_id}"
        self._container_user: str | None = None
        self._started = False
        self._start_lock = asyncio.Lock()

    @property
    def workspace(self) -> Path:
        return self._workspace

    async def _ensure_started(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            self._workspace.mkdir(parents=True, exist_ok=True)
            container_user = _numeric_workspace_owner(self._workspace)
            settings = self._settings
            args = [
                "run",
                "--detach",
                "--name",
                self._container,
                "--init",
                "--user",
                container_user,
                # Isolation: no network, immutable root, no capabilities,
                # no privilege escalation. Only the session workspace is
                # writable (plus a small tmpfs /tmp for scratch files).
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,size=64m",
                "--memory",
                settings.sandbox_memory_limit,
                "--cpus",
                str(settings.sandbox_cpu_limit),
                "--pids-limit",
                str(settings.sandbox_pids_limit),
                "--volume",
                f"{self._workspace.resolve().as_posix()}:/workspace:rw",
                "--workdir",
                "/workspace",
                settings.sandbox_image,
                "sleep",
                "infinity",
            ]
            code, _, stderr = await self._runner(
                args,
                60.0,
                _CONTROL_CAPTURE_LIMIT_BYTES,
            )
            if code != 0:
                raise SandboxUnavailableError(
                    f"Failed to start sandbox container from image "
                    f"'{settings.sandbox_image}': {stderr.decode(errors='replace').strip()}"
                )
            self._container_user = container_user
            self._started = True

    async def execute(self, command: str, *, timeout: int | None = None) -> ExecResult:
        """Run one command in a fresh /bin/sh process inside the container."""
        await self._ensure_started()
        configured_timeout = self._settings.sandbox_command_timeout_seconds
        if timeout is not None and timeout < 1:
            raise ValueError("sandbox command timeout must be positive")
        effective_timeout = min(timeout or configured_timeout, configured_timeout)
        token = secrets.token_hex(16)
        assert self._container_user is not None
        try:
            code, stdout, stderr = await self._runner(
                [
                    "exec",
                    "--user",
                    self._container_user,
                    "--env",
                    f"RAGCHAT_EXEC_TOKEN={token}",
                    self._container,
                    # GNU timeout creates a separate process group by default.
                    # TERM followed by KILL bounds descendants that honor
                    # neither the shell exit nor the first signal.
                    "/usr/bin/timeout",
                    "--signal=TERM",
                    "--kill-after=1s",
                    f"{effective_timeout}s",
                    "/bin/sh",
                    "-lc",
                    command,
                ],
                float(effective_timeout) + _DOCKER_EXEC_GRACE_SECONDS,
                _execution_capture_limit(self._settings.sandbox_output_limit_chars),
            )
        finally:
            cleanup_task = asyncio.create_task(self._cleanup_exec_processes(token))
            cleanup_interrupted = await _finish_future_despite_cancellation(cleanup_task)
            if cleanup_interrupted:
                raise asyncio.CancelledError
        runner_truncated = _PIPE_TRUNCATION_MARKER in stdout or _PIPE_TRUNCATION_MARKER in stderr
        merged = (stdout + stderr).decode(errors="replace")
        limit = self._settings.sandbox_output_limit_chars
        if len(merged) > limit:
            runner_truncated = True
            merged = merged[:limit] + f"\n… output truncated at {limit} characters"
        if code == _EXIT_TIMEOUT:
            merged = (f"Command timed out after {effective_timeout}s.\n" + merged).strip()
        return ExecResult(exit_code=code, output=merged, truncated=runner_truncated)

    async def _cleanup_exec_processes(self, token: str) -> None:
        """Best-effort kill of token-tagged processes that escaped the command group."""
        assert self._container_user is not None
        # Cleanup supplements the timeout process group. Container removal
        # remains the hard session boundary if the daemon becomes unusable.
        with suppress(OSError, TimeoutError):
            await self._runner(
                [
                    "exec",
                    "--user",
                    self._container_user,
                    self._container,
                    "python",
                    "-c",
                    _CLEANUP_SCRIPT,
                    token,
                ],
                _CLEANUP_TIMEOUT_SECONDS,
                _CONTROL_CAPTURE_LIMIT_BYTES,
            )

    async def close(self) -> None:
        """Destroy the container and the session workspace."""
        container_error: SandboxUnavailableError | None = None
        container_cause: BaseException | None = None

        if self._started:
            try:
                code, _, stderr = await self._runner(
                    ["rm", "--force", self._container],
                    30.0,
                    _CONTROL_CAPTURE_LIMIT_BYTES,
                )
            except (OSError, TimeoutError) as exc:
                container_cause = exc
                container_error = SandboxUnavailableError(
                    f"Failed to remove sandbox container '{self._container}': "
                    f"the docker CLI could not be run: {exc}"
                )
            else:
                if code == 0:
                    self._started = False
                    self._container_user = None
                else:
                    detail = stderr.decode(errors="replace").strip() or "no error output"
                    container_error = SandboxUnavailableError(
                        f"Failed to remove sandbox container '{self._container}' "
                        f"(docker rm exited with code {code}): {detail}"
                    )

        workspace_error: RuntimeError | None = None
        workspace_cause: OSError | None = None
        try:
            shutil.rmtree(self._workspace)
        except FileNotFoundError:
            pass
        except OSError as exc:
            workspace_cause = exc
            workspace_error = RuntimeError(
                f"Failed to remove sandbox workspace '{self._workspace}': {exc}"
            )

        if container_error is not None and workspace_error is not None:
            raise SandboxUnavailableError(f"{container_error}; {workspace_error}") from (
                container_cause or workspace_cause
            )
        if container_error is not None:
            raise container_error from container_cause
        if workspace_error is not None:
            raise workspace_error from workspace_cause
