from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import ragchat.manager as manager_module
from ragchat.config import SandboxMode, Settings
from ragchat.manager import build_components


def settings(*, sandbox_mode: SandboxMode = SandboxMode.DISABLED) -> Settings:
    return Settings(
        vllm_base_url="http://vllm.test/v1",
        vllm_model="test-model",
        qdrant_collection="test-corpus",
        sandbox_mode=sandbox_mode,
        _env_file=None,
    )


def install_lightweight_builders(
    monkeypatch: pytest.MonkeyPatch,
    timeline: list[tuple[str, tuple[Any, ...]]],
) -> dict[str, object]:
    objects = {
        "model": object(),
        "client": object(),
        "store": object(),
        "reranker": object(),
        "pipeline": object(),
    }

    def build_model(*args: Any) -> object:
        timeline.append(("model", args))
        return objects["model"]

    def build_client(*args: Any) -> object:
        timeline.append(("client", args))
        return objects["client"]

    def validate(*args: Any) -> None:
        timeline.append(("validate", args))

    def build_store(*args: Any) -> object:
        timeline.append(("store", args))
        return objects["store"]

    def build_ranker(*args: Any) -> object:
        timeline.append(("reranker", args))
        return objects["reranker"]

    def build_pipeline(*args: Any) -> object:
        timeline.append(("pipeline", args))
        return objects["pipeline"]

    monkeypatch.setattr(manager_module, "build_chat_model", build_model)
    monkeypatch.setattr(manager_module, "build_qdrant_client", build_client)
    monkeypatch.setattr(manager_module, "validate_corpus", validate)
    monkeypatch.setattr(manager_module, "build_vector_store", build_store)
    monkeypatch.setattr(manager_module, "build_reranker", build_ranker)
    monkeypatch.setattr(manager_module, "HybridSearchPipeline", build_pipeline)
    return objects


@pytest.mark.asyncio
async def test_build_components_wires_one_provider_and_read_only_pipeline_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline: list[tuple[str, tuple[Any, ...]]] = []
    objects = install_lightweight_builders(monkeypatch, timeline)
    configured = settings()

    components = await build_components(configured)

    assert [name for name, _ in timeline] == [
        "model",
        "client",
        "validate",
        "store",
        "reranker",
        "pipeline",
    ]
    assert timeline[0][1] == (configured,)
    assert timeline[1][1] == (configured,)
    assert timeline[2][1] == (objects["client"], configured)
    assert timeline[3][1] == (configured, objects["client"])
    assert timeline[4][1] == (configured,)
    assert timeline[5][1] == (objects["store"], objects["reranker"])
    assert components.chat_model is objects["model"]
    assert components.pipeline is objects["pipeline"]
    assert components.sandbox_handle_factory is None


@pytest.mark.asyncio
async def test_disabled_mode_neither_imports_sandbox_modules_nor_checks_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_lightweight_builders(monkeypatch, [])
    monkeypatch.delitem(sys.modules, "ragchat.sandbox.backend", raising=False)
    monkeypatch.delitem(sys.modules, "ragchat.sandbox.docker_session", raising=False)

    components = await build_components(settings())

    assert components.sandbox_handle_factory is None
    assert "ragchat.sandbox.backend" not in sys.modules
    assert "ragchat.sandbox.docker_session" not in sys.modules


@pytest.mark.asyncio
async def test_docker_mode_checks_availability_and_builds_session_workspace_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_lightweight_builders(monkeypatch, [])

    import deepagents.backends

    import ragchat.sandbox.docker_session as docker_module

    docker_checks: list[None] = []
    docker_sessions: list[FakeDockerSession] = []
    filesystem_backends: list[FakeFilesystemBackend] = []

    async def check_docker() -> None:
        docker_checks.append(None)

    class FakeDockerSession:
        def __init__(
            self,
            configured: Settings,
            session_id: str,
            workspace: Path,
        ) -> None:
            self.settings = configured
            self.session_id = session_id
            self.workspace = workspace
            self.executed = False
            docker_sessions.append(self)

    class FakeFilesystemBackend:
        def __init__(self, *, root_dir: Path, virtual_mode: bool) -> None:
            self.root_dir = root_dir
            self.virtual_mode = virtual_mode
            filesystem_backends.append(self)

    monkeypatch.setattr(docker_module, "assert_docker_available", check_docker)
    monkeypatch.setattr(docker_module, "DockerSandboxSession", FakeDockerSession)
    monkeypatch.setattr(deepagents.backends, "FilesystemBackend", FakeFilesystemBackend)
    monkeypatch.setattr(manager_module.tempfile, "gettempdir", lambda: str(tmp_path))
    configured = settings(sandbox_mode=SandboxMode.DOCKER)

    components = await build_components(configured)

    assert docker_checks == [None]
    assert components.sandbox_handle_factory is not None
    handle = components.sandbox_handle_factory("session-123")
    expected_workspace = tmp_path / "ragchat-workspaces" / "session-123"
    assert expected_workspace.is_dir()
    assert handle.settings is configured
    assert handle.session is docker_sessions[0]
    assert docker_sessions[0].session_id == "session-123"
    assert docker_sessions[0].workspace == expected_workspace
    assert docker_sessions[0].executed is False
    assert handle.files is filesystem_backends[0]
    assert filesystem_backends[0].root_dir == expected_workspace
    assert filesystem_backends[0].virtual_mode is True


def test_manager_has_no_http_or_eager_sandbox_imports() -> None:
    tree = ast.parse(Path(manager_module.__file__).read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any(name == "litestar" or name.startswith("litestar.") for name in imported_modules)
    assert not any(name.startswith("ragchat.sandbox") for name in imported_modules)


@pytest.mark.asyncio
async def test_building_vllm_components_never_imports_removable_openai_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            self.chat = SimpleNamespace(completions=object())

    class FakeChatOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    openai = ModuleType("openai")
    openai.AsyncOpenAI = FakeAsyncOpenAI  # type: ignore[attr-defined]
    langchain_openai = ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = FakeChatOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai)
    monkeypatch.delitem(sys.modules, "ragchat.openai_backend", raising=False)
    client = object()
    store = object()
    reranker = object()
    monkeypatch.setattr(manager_module, "build_qdrant_client", lambda _settings: client)
    monkeypatch.setattr(manager_module, "validate_corpus", lambda *_args: None)
    monkeypatch.setattr(manager_module, "build_vector_store", lambda *_args: store)
    monkeypatch.setattr(manager_module, "build_reranker", lambda _settings: reranker)
    monkeypatch.setattr(manager_module, "HybridSearchPipeline", lambda *_args: object())

    components = await build_components(settings())

    assert isinstance(components.chat_model, FakeChatOpenAI)
    assert "ragchat.openai_backend" not in sys.modules
