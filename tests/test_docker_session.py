from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import ragchat.sandbox.docker_session as docker_session_module
from ragchat.config import SandboxMode, Settings
from ragchat.domain import SandboxUnavailableError
from ragchat.sandbox.docker_session import (
    DockerSandboxSession,
    assert_docker_available,
    run_docker_cli,
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
        self.calls: list[tuple[list[str], float, int]] = []
        self.exec_result: tuple[int, bytes, bytes] = (0, b"stdout\n", b"stderr")
        self.cleanup_result: tuple[int, bytes, bytes] = (0, b"", b"")
        self.run_result: tuple[int, bytes, bytes] = (0, b"container-id", b"")
        self.rm_result: tuple[int, bytes, bytes] = (0, b"", b"")
        self.rm_error: OSError | TimeoutError | None = None

    async def __call__(
        self,
        args: list[str],
        timeout: float,
        max_output_bytes: int,
    ) -> tuple[int, bytes, bytes]:
        self.calls.append((args, timeout, max_output_bytes))
        if args[0] == "run":
            return self.run_result
        if args[0] == "exec":
            if "--env" not in args:
                return self.cleanup_result
            return self.exec_result
        if args[0] == "rm":
            if self.rm_error is not None:
                raise self.rm_error
            return self.rm_result
        return 0, b"", b""


def command_calls(runner: RecordingRunner) -> list[tuple[list[str], float, int]]:
    return [call for call in runner.calls if call[0][0] == "exec" and "--env" in call[0]]


def cleanup_calls(runner: RecordingRunner) -> list[tuple[list[str], float, int]]:
    return [call for call in runner.calls if call[0][0] == "exec" and "--env" not in call[0]]


class FakeDockerProcess:
    def __init__(self, stdout: bytes, stderr: bytes, *, running: bool = False) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stderr.feed_data(stderr)
        self.returncode: int | None = None if running else 0
        self.killed = False
        self._done = asyncio.Event()
        if running:
            return
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self._done.set()


@pytest.mark.asyncio
async def test_docker_cli_continuously_drains_but_retains_bounded_pipe_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeDockerProcess(b"a" * 1_000_000, b"b" * 1_000_000)

    async def create_subprocess(*_args: object, **_kwargs: object) -> FakeDockerProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    code, stdout, stderr = await run_docker_cli(["version"], 1.0, 128)

    assert code == 0
    assert len(stdout) == 128
    assert len(stderr) == 128
    assert stdout.endswith(docker_session_module._PIPE_TRUNCATION_MARKER)
    assert stderr.endswith(docker_session_module._PIPE_TRUNCATION_MARKER)


@pytest.mark.asyncio
async def test_cancelling_docker_cli_kills_process_and_finishes_pipe_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeDockerProcess(b"partial stdout", b"partial stderr", running=True)
    created = asyncio.Event()

    async def create_subprocess(*_args: object, **_kwargs: object) -> FakeDockerProcess:
        created.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    task = asyncio.create_task(run_docker_cli(["exec"], 60.0, 128))
    await created.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed is True


@pytest.mark.parametrize("uid, gid", [(0, 1000), (1000, 0)])
def test_root_workspace_identity_is_rejected(uid: int, gid: int) -> None:
    with pytest.raises(SandboxUnavailableError, match="root UID or GID"):
        docker_session_module._format_workspace_owner(uid, gid)


def test_non_root_workspace_identity_is_numeric() -> None:
    assert docker_session_module._format_workspace_owner(1234, 5678) == "1234:5678"


@pytest.mark.asyncio
async def test_container_is_lazy_hardened_and_receives_only_the_workspace_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        docker_session_module,
        "_numeric_workspace_owner",
        lambda _workspace: "1234:5678",
    )
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
    assert [args[0] for args, _, _ in runner.calls] == [
        "run",
        "exec",
        "exec",
        "exec",
        "exec",
    ]

    run_args, run_timeout, run_capture_limit = runner.calls[0]
    assert run_timeout == 60.0
    assert run_capture_limit > 0
    assert "--init" in run_args
    assert run_args[run_args.index("--user") + 1] == "1234:5678"
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

    first_args, first_timeout, first_capture_limit = command_calls(runner)[0]
    assert first_args[:5] == ["exec", "--user", "1234:5678", "--env", first_args[4]]
    token = first_args[4].removeprefix("RAGCHAT_EXEC_TOKEN=")
    assert len(token) == 32
    assert first_args[5:12] == [
        "ragchat-sandbox-abc123",
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=1s",
        f"{configured.sandbox_command_timeout_seconds}s",
        "/bin/sh",
        "-lc",
    ]
    assert first_args[-1] == "printf first"
    assert first_timeout == configured.sandbox_command_timeout_seconds + 5.0
    assert first_capture_limit >= configured.sandbox_output_limit_chars * 4
    assert command_calls(runner)[1][0][-1] == "printf second"

    first_cleanup_args = cleanup_calls(runner)[0][0]
    assert first_cleanup_args[:4] == [
        "exec",
        "--user",
        "1234:5678",
        "ragchat-sandbox-abc123",
    ]
    assert first_cleanup_args[-1] == token
    assert "pid in (1, current)" in first_cleanup_args[-2]

    await session.close()
    await session.close()

    assert [args[0] for args, _, _ in runner.calls] == [
        "run",
        "exec",
        "exec",
        "exec",
        "exec",
        "rm",
    ]
    assert runner.calls[-1] == (
        ["rm", "--force", "ragchat-sandbox-abc123"],
        30.0,
        runner.calls[0][2],
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
    assert [args[0] for args, _, _ in runner.calls].count("rm") == 1

    runner.rm_result = (0, b"", b"")
    await session.close()
    await session.close()

    assert [args[0] for args, _, _ in runner.calls].count("rm") == 2


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

    assert [args[0] for args, _, _ in runner.calls].count("rm") == 2


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
    assert [args[0] for args, _, _ in runner.calls].count("rm") == 1

    await session.close()

    assert not workspace.exists()
    assert [args[0] for args, _, _ in runner.calls].count("rm") == 1


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
    assert result.truncated is True
    assert result.output.startswith("Command timed out after 7s.")
    assert "output truncated at 8 characters" in result.output
    assert command_calls(runner)[0][1] == 12.0
    assert "7s" in command_calls(runner)[0][0]
    await session.close()


@pytest.mark.asyncio
async def test_requested_timeout_is_positive_and_capped_by_configuration(
    tmp_path: Path,
) -> None:
    configured = settings(sandbox_command_timeout_seconds=7)
    runner = RecordingRunner()
    session = DockerSandboxSession(configured, "bounded-timeout", tmp_path / "work", runner)

    await session.execute("shorter", timeout=3)
    await session.execute("capped", timeout=99)

    first_args, first_outer_timeout, _ = command_calls(runner)[0]
    second_args, second_outer_timeout, _ = command_calls(runner)[1]
    assert "3s" in first_args
    assert first_outer_timeout == 8.0
    assert "7s" in second_args
    assert second_outer_timeout == 12.0
    with pytest.raises(ValueError, match="must be positive"):
        await session.execute("invalid", timeout=0)
    await session.close()


@pytest.mark.asyncio
async def test_cancelling_command_waits_for_token_scoped_container_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        docker_session_module,
        "_numeric_workspace_owner",
        lambda _workspace: "1234:5678",
    )
    main_started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    calls: list[list[str]] = []

    async def blocking_runner(
        args: list[str], _timeout: float, _max_output_bytes: int
    ) -> tuple[int, bytes, bytes]:
        calls.append(args)
        if args[0] == "run":
            return 0, b"container", b""
        if args[0] == "exec" and "--env" in args:
            main_started.set()
            await asyncio.Future()
        if args[0] == "exec":
            await asyncio.sleep(0)
            cleanup_finished.set()
            return 0, b"", b""
        return 0, b"", b""

    session = DockerSandboxSession(
        settings(),
        "cancelled",
        tmp_path / "work",
        runner=blocking_runner,
    )
    task = asyncio.create_task(session.execute("long-running"))
    await main_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleanup_finished.is_set()
    command_args = next(args for args in calls if "--env" in args)
    cleanup_args = next(args for args in calls if args[0] == "exec" and "--env" not in args)
    assert cleanup_args[-1] == command_args[4].removeprefix("RAGCHAT_EXEC_TOKEN=")
    await session.close()


@pytest.mark.asyncio
async def test_failed_container_start_raises_clear_error_without_exec(tmp_path: Path) -> None:
    runner = RecordingRunner()
    runner.run_result = (1, b"", b"image missing")
    session = DockerSandboxSession(settings(), "failed", tmp_path / "work", runner=runner)

    with pytest.raises(SandboxUnavailableError, match="image missing"):
        await session.execute("never runs")

    assert [args[0] for args, _, _ in runner.calls] == ["run"]
    await session.close()


@pytest.mark.asyncio
async def test_docker_availability_probe_accepts_success_and_rejects_failures() -> None:
    calls: list[tuple[list[str], float, int]] = []

    async def success(
        args: list[str], timeout: float, max_output_bytes: int
    ) -> tuple[int, bytes, bytes]:
        calls.append((args, timeout, max_output_bytes))
        return 0, b"27.0", b""

    await assert_docker_available(success)

    assert calls[0][:2] == (["version", "--format", "{{.Server.Version}}"], 15.0)
    assert calls[0][2] > 0

    async def daemon_down(
        _args: list[str], _timeout: float, _max_output_bytes: int
    ) -> tuple[int, bytes, bytes]:
        return 1, b"", b"daemon unavailable"

    with pytest.raises(SandboxUnavailableError, match="daemon unavailable"):
        await assert_docker_available(daemon_down)

    async def cli_missing(
        _args: list[str], _timeout: float, _max_output_bytes: int
    ) -> tuple[int, bytes, bytes]:
        raise OSError("docker not found")

    with pytest.raises(SandboxUnavailableError, match="docker CLI could not be run"):
        await assert_docker_available(cli_missing)
