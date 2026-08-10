from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import ragchat.sandbox.docker_session as docker_session_module
from ragchat.config import SandboxMode, Settings
from ragchat.domain import SandboxUnavailableError
from ragchat.sandbox.docker_session import (
    DockerSandboxSession,
    assert_docker_available,
)


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "vllm_base_url": "http://vllm.test/v1",
        "vllm_api_key": "vllm-secret",
        "vllm_model": "test-model",
        "qdrant_url": "http://qdrant.internal:6333",
        "qdrant_api_key": "qdrant-secret",
        "qdrant_collection": "private-corpus",
        "sandbox_mode": SandboxMode.DOCKER,
        "sandbox_image": "ragchat-sandbox:test",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], float]] = []
        self.exec_result: tuple[int, bytes, bytes] = (0, b"stdout\n", b"stderr")
        self.run_result: tuple[int, bytes, bytes] = (0, b"container-id", b"")
        self.rm_result: tuple[int, bytes, bytes] = (0, b"", b"")
        self.rm_error: OSError | TimeoutError | None = None

    async def __call__(
        self,
        args: list[str],
        timeout: float,
    ) -> tuple[int, bytes, bytes]:
        self.calls.append((args, timeout))
        if args[0] == "run":
            return self.run_result
        if args[0] == "exec":
            return self.exec_result
        if args[0] == "rm":
            if self.rm_error is not None:
                raise self.rm_error
            return self.rm_result
        return 0, b"", b""


@pytest.mark.asyncio
async def test_container_is_lazy_hardened_and_receives_only_the_workspace_mount(
    tmp_path: Path,
) -> None:
    configured = settings()
    runner = RecordingRunner()
    workspace = tmp_path / "session-workspace"
    session = DockerSandboxSession(configured, "abc123", workspace, runner=runner)

    assert runner.calls == []
    assert not workspace.exists()

    first = await session.execute("printf first")
    second = await session.execute("printf second")

    assert first.exit_code == 0
    assert first.output == "stdout\nstderr"
    assert second.output == "stdout\nstderr"
    assert workspace.is_dir()
    assert [args[0] for args, _ in runner.calls] == ["run", "exec", "exec"]

    run_args, run_timeout = runner.calls[0]
    assert run_timeout == 60.0
    assert run_args[run_args.index("--network") + 1] == "none"
    assert "--read-only" in run_args
    assert run_args[run_args.index("--cap-drop") + 1] == "ALL"
    assert run_args[run_args.index("--security-opt") + 1] == "no-new-privileges"
    assert run_args[run_args.index("--memory") + 1] == configured.sandbox_memory_limit
    assert run_args[run_args.index("--cpus") + 1] == str(configured.sandbox_cpu_limit)
    assert run_args[run_args.index("--pids-limit") + 1] == str(configured.sandbox_pids_limit)
    assert run_args.count("--volume") == 1
    assert run_args[run_args.index("--volume") + 1] == (
        f"{workspace.resolve().as_posix()}:/workspace:rw"
    )
    assert run_args[run_args.index("--workdir") + 1] == "/workspace"
    assert run_args[-3:] == [configured.sandbox_image, "sleep", "infinity"]
    assert "--env" not in run_args
    serialized_args = " ".join(run_args)
    assert configured.qdrant_url not in serialized_args
    assert configured.qdrant_collection not in serialized_args
    assert "qdrant-secret" not in serialized_args
    assert "vllm-secret" not in serialized_args

    assert runner.calls[1] == (
        [
            "exec",
            "ragchat-sandbox-abc123",
            "/bin/sh",
            "-lc",
            "printf first",
        ],
        float(configured.sandbox_command_timeout_seconds),
    )
    assert runner.calls[2][0][-1] == "printf second"

    await session.close()
    await session.close()

    assert [args[0] for args, _ in runner.calls] == ["run", "exec", "exec", "rm"]
    assert runner.calls[-1] == (
        ["rm", "--force", "ragchat-sandbox-abc123"],
        30.0,
    )
    assert not workspace.exists()


@pytest.mark.asyncio
async def test_failed_container_removal_is_reported_and_can_be_retried(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner()
    workspace = tmp_path / "work"
    session = DockerSandboxSession(settings(), "retry-rm", workspace, runner=runner)
    await session.execute("create container")
    runner.rm_result = (1, b"", b"daemon rejected removal")

    with pytest.raises(
        SandboxUnavailableError,
        match=r"docker rm exited with code 1.*daemon rejected removal",
    ):
        await session.close()

    assert not workspace.exists()
    assert [args[0] for args, _ in runner.calls].count("rm") == 1

    runner.rm_result = (0, b"", b"")
    await session.close()
    await session.close()

    assert [args[0] for args, _ in runner.calls].count("rm") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rm_error",
    [OSError("docker executable vanished"), TimeoutError("docker rm timed out")],
)
async def test_container_removal_runner_errors_are_reported_and_retryable(
    tmp_path: Path,
    rm_error: OSError | TimeoutError,
) -> None:
    runner = RecordingRunner()
    workspace = tmp_path / "work"
    session = DockerSandboxSession(settings(), "runner-error", workspace, runner=runner)
    await session.execute("create container")
    runner.rm_error = rm_error

    with pytest.raises(SandboxUnavailableError, match="docker CLI could not be run"):
        await session.close()

    assert not workspace.exists()

    runner.rm_error = None
    await session.close()

    assert [args[0] for args, _ in runner.calls].count("rm") == 2


@pytest.mark.asyncio
async def test_workspace_cleanup_failure_is_reported_and_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner()
    workspace = tmp_path / "work"
    session = DockerSandboxSession(settings(), "workspace-error", workspace, runner=runner)
    await session.execute("create container")
    original_rmtree = docker_session_module.shutil.rmtree

    def reject_workspace_removal(path: str | Path) -> None:
        if Path(path) == workspace:
            raise PermissionError("workspace is locked")
        original_rmtree(path)

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(
            docker_session_module.shutil,
            "rmtree",
            reject_workspace_removal,
        )
        with pytest.raises(RuntimeError, match=r"workspace.*workspace is locked"):
            await session.close()

    assert workspace.exists()
    assert [args[0] for args, _ in runner.calls].count("rm") == 1

    await session.close()

    assert not workspace.exists()
    assert [args[0] for args, _ in runner.calls].count("rm") == 1


@pytest.mark.asyncio
async def test_close_before_start_removes_an_existing_workspace_and_is_idempotent(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner()
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "artifact.txt").write_text("temporary", encoding="utf-8")
    session = DockerSandboxSession(settings(), "never-started", workspace, runner=runner)

    await session.close()
    await session.close()

    assert not workspace.exists()
    assert runner.calls == []


@pytest.mark.asyncio
async def test_execute_reports_timeout_and_output_truncation(tmp_path: Path) -> None:
    configured = settings(
        sandbox_command_timeout_seconds=7,
        sandbox_output_limit_chars=8,
    )
    runner = RecordingRunner()
    runner.exec_result = (124, b"0123456789", b"abcdefghij")
    session = DockerSandboxSession(configured, "timeout", tmp_path / "work", runner=runner)

    result = await session.execute("long command")

    assert result.exit_code == 124
    assert result.output.startswith("Command timed out after 7s.")
    assert "output truncated at 8 characters" in result.output
    assert runner.calls[1][1] == 7.0
    await session.close()


@pytest.mark.asyncio
async def test_failed_container_start_raises_clear_error_without_exec(tmp_path: Path) -> None:
    runner = RecordingRunner()
    runner.run_result = (1, b"", b"image missing")
    session = DockerSandboxSession(settings(), "failed", tmp_path / "work", runner=runner)

    with pytest.raises(SandboxUnavailableError, match="image missing"):
        await session.execute("never runs")

    assert [args[0] for args, _ in runner.calls] == ["run"]
    await session.close()


@pytest.mark.asyncio
async def test_docker_availability_probe_accepts_success_and_rejects_failures() -> None:
    calls: list[tuple[list[str], float]] = []

    async def success(args: list[str], timeout: float) -> tuple[int, bytes, bytes]:
        calls.append((args, timeout))
        return 0, b"27.0", b""

    await assert_docker_available(success)

    assert calls == [(["version", "--format", "{{.Server.Version}}"], 15.0)]

    async def daemon_down(_args: list[str], _timeout: float) -> tuple[int, bytes, bytes]:
        return 1, b"", b"daemon unavailable"

    with pytest.raises(SandboxUnavailableError, match="daemon unavailable"):
        await assert_docker_available(daemon_down)

    async def cli_missing(_args: list[str], _timeout: float) -> tuple[int, bytes, bytes]:
        raise OSError("docker not found")

    with pytest.raises(SandboxUnavailableError, match="docker CLI could not be run"):
        await assert_docker_available(cli_missing)
