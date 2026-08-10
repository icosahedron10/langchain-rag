from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ragchat"
UI = ROOT / "ui" / "streamlit_app.py"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_importing_manager_and_its_runtime_tree_does_not_import_litestar() -> None:
    script = (
        "import sys; import ragchat.manager; "
        "assert not any(name == 'litestar' or name.startswith('litestar.') "
        "for name in sys.modules), sorted(name for name in sys.modules "
        "if name == 'litestar' or name.startswith('litestar.'))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_streamlit_client_imports_no_ragchat_application_modules() -> None:
    imports = _imported_modules(UI)

    assert not any(name == "ragchat" or name.startswith("ragchat.") for name in imports)


def test_streamlit_renders_disabled_input_before_starting_blocking_stream() -> None:
    source = UI.read_text(encoding="utf-8")

    input_position = source.index("user_input = st.chat_input")
    stream_position = source.index("if st.session_state.busy and st.session_state.pending")

    assert input_position < stream_position


def test_streamlit_does_not_mark_error_events_as_complete() -> None:
    source = UI.read_text(encoding="utf-8")

    assert 'return answer, "error"' in source
    assert 'if terminal == "done"' in source
    assert 'elif terminal == "error"' in source


def test_only_controller_and_app_composition_root_import_litestar() -> None:
    allowed = {SRC / "controller.py", SRC / "app.py"}
    offenders: list[Path] = []
    for path in SRC.rglob("*.py"):
        imports = _imported_modules(path)
        if path not in allowed and any(
            name == "litestar" or name.startswith("litestar.") for name in imports
        ):
            offenders.append(path)

    assert offenders == []


def test_application_defines_exactly_one_http_controller() -> None:
    controller_classes: list[tuple[Path, str]] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(isinstance(base, ast.Name) and base.id == "Controller" for base in node.bases):
                controller_classes.append((path, node.name))

    assert controller_classes == [(SRC / "controller.py", "AgentController")]
