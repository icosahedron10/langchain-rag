from __future__ import annotations

import base64
from pathlib import Path

import pytest

import ragchat.sandbox.artifacts as artifacts_module
from ragchat.sandbox.artifacts import collect_image_artifacts, snapshot_workspace


def test_collects_created_and_changed_images_but_not_unchanged(tmp_path: Path) -> None:
    changed = tmp_path / "changed.png"
    unchanged = tmp_path / "unchanged.jpg"
    changed.write_bytes(b"old")
    unchanged.write_bytes(b"same")
    before = snapshot_workspace(tmp_path)

    changed.write_bytes(b"new!")
    chart_dir = tmp_path / "charts"
    chart_dir.mkdir()
    (chart_dir / "created.WEBP").write_bytes(b"chart")

    artifacts, skipped = collect_image_artifacts(tmp_path, before, max_bytes=100)

    assert skipped == []
    assert [artifact.name for artifact in artifacts] == ["changed.png", "charts/created.WEBP"]
    assert [artifact.media_type for artifact in artifacts] == ["image/png", "image/webp"]
    assert [base64.b64decode(artifact.data) for artifact in artifacts] == [b"new!", b"chart"]
    assert "unchanged.jpg" not in {artifact.name for artifact in artifacts}


def test_omits_final_symlink_to_image_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_image = tmp_path / "outside.png"
    outside_image.write_bytes(b"secret")
    link = workspace / "leak.png"
    _make_symlink_or_skip(link, outside_image)

    assert snapshot_workspace(workspace) == {}
    assert collect_image_artifacts(workspace, {}, max_bytes=100) == ([], [])


def test_omits_images_beneath_symlinked_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "leak.png").write_bytes(b"secret")
    _make_symlink_or_skip(workspace / "linked", outside_dir, target_is_directory=True)

    assert snapshot_workspace(workspace) == {}
    assert collect_image_artifacts(workspace, {}, max_bytes=100) == ([], [])


def test_file_growing_past_cap_during_bounded_read_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "growing.png"
    image.write_bytes(b"1234")
    original_read_bounded = artifacts_module._read_bounded
    observed_limits: list[int] = []

    def grow_then_read(file_fd: int, limit: int) -> bytes:
        observed_limits.append(limit)
        with image.open("ab") as growing_file:
            growing_file.write(b"56")
        return original_read_bounded(file_fd, limit)

    monkeypatch.setattr(artifacts_module, "_read_bounded", grow_then_read)

    artifacts, skipped = collect_image_artifacts(tmp_path, {}, max_bytes=5)

    assert artifacts == []
    assert skipped == ["growing.png"]
    assert observed_limits == [6]


def test_aggregate_count_budget_skips_excess_images(tmp_path: Path) -> None:
    for name in ("one.png", "three.png", "two.png"):
        (tmp_path / name).write_bytes(b"x")

    artifacts, skipped = collect_image_artifacts(
        tmp_path,
        {},
        max_bytes=10,
        max_count=2,
        max_total_bytes=100,
    )

    assert [artifact.name for artifact in artifacts] == ["one.png", "three.png"]
    assert skipped == ["two.png"]


def test_aggregate_byte_budget_bounds_individually_under_cap_images(tmp_path: Path) -> None:
    (tmp_path / "one.png").write_bytes(b"1234")
    (tmp_path / "three.png").write_bytes(b"12")
    (tmp_path / "two.png").write_bytes(b"5678")

    artifacts, skipped = collect_image_artifacts(
        tmp_path,
        {},
        max_bytes=5,
        max_count=10,
        max_total_bytes=6,
    )

    assert [artifact.name for artifact in artifacts] == ["one.png", "three.png"]
    assert [base64.b64decode(artifact.data) for artifact in artifacts] == [b"1234", b"12"]
    assert skipped == ["two.png"]


def _make_symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool = False,
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Symlinks are unavailable: {exc}")
