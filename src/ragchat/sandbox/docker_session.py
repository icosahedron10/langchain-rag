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
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from ragchat.config import Settings
from ragchat.domain import SandboxUnavailableError

# (args after "docker", timeout seconds) -> (exit_code, stdout, stderr)
DockerRunner = Callable[[list[str], float], Awaitable[tuple[int, bytes, bytes]]]

_EXIT_TIMEOUT = 124  # conventional coreutils timeout exit code


async def run_docker_cli(args: list[str], timeout: float) -> tuple[int, bytes, bytes]:
    """Run one docker CLI invocation in a separate subprocess."""
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return _EXIT_TIMEOUT, b"", b"command timed out"
    return proc.returncode or 0, stdout, stderr


async def assert_docker_available(runner: DockerRunner = run_docker_cli) -> None:
    """Fail clearly at startup when SANDBOX_MODE=docker but Docker is unusable."""
    try:
        code, _, stderr = await runner(["version", "--format", "{{.Server.Version}}"], 15.0)
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
            settings = self._settings
            args = [
                "run",
                "--detach",
                "--name",
                self._container,
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
            code, _, stderr = await self._runner(args, 60.0)
            if code != 0:
                raise SandboxUnavailableError(
                    f"Failed to start sandbox container from image "
                    f"'{settings.sandbox_image}': {stderr.decode(errors='replace').strip()}"
                )
            self._started = True

    async def execute(self, command: str) -> ExecResult:
        """Run one command in a fresh /bin/sh process inside the container."""
        await self._ensure_started()
        code, stdout, stderr = await self._runner(
            ["exec", self._container, "/bin/sh", "-lc", command],
            float(self._settings.sandbox_command_timeout_seconds),
        )
        merged = (stdout + stderr).decode(errors="replace")
        limit = self._settings.sandbox_output_limit_chars
        if len(merged) > limit:
            merged = merged[:limit] + f"\n… output truncated at {limit} characters"
        if code == _EXIT_TIMEOUT:
            merged = (
                f"Command timed out after "
                f"{self._settings.sandbox_command_timeout_seconds}s.\n" + merged
            ).strip()
        return ExecResult(exit_code=code, output=merged)

    async def close(self) -> None:
        """Destroy the container and the session workspace."""
        container_error: SandboxUnavailableError | None = None
        container_cause: BaseException | None = None

        if self._started:
            try:
                code, _, stderr = await self._runner(
                    ["rm", "--force", self._container],
                    30.0,
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
