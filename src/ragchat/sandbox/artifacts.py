"""Detect displayable images created or modified in a session workspace."""

from __future__ import annotations

import base64
import os
import stat
from pathlib import Path, PurePosixPath

from ragchat.domain import ArtifactEvent

IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

# Relative path -> (mtime_ns, size) fingerprint.
WorkspaceSnapshot = dict[str, tuple[int, int]]

_READ_CHUNK_BYTES = 64 * 1024
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_SECURE_DIR_FDS = (
    hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
    and os.scandir in os.supports_fd
)


def snapshot_workspace(workspace: Path) -> WorkspaceSnapshot:
    """Fingerprint every displayable regular image without following links."""
    if _SECURE_DIR_FDS:
        return _snapshot_with_dir_fds(workspace)
    return _snapshot_with_lstat(workspace)


def collect_image_artifacts(
    workspace: Path,
    before: WorkspaceSnapshot,
    max_bytes: int,
) -> tuple[list[ArtifactEvent], list[str]]:
    """Return changed images and names of oversized or raced regular files.

    Unsafe and unreadable paths are omitted. Artifact names are always
    workspace-relative; physical host paths are never exposed.
    """
    artifacts: list[ArtifactEvent] = []
    skipped: list[str] = []
    for name, fingerprint in snapshot_workspace(workspace).items():
        if before.get(name) == fingerprint:
            continue
        if _SECURE_DIR_FDS:
            data, was_skipped = _read_with_dir_fds(workspace, name, fingerprint, max_bytes)
        else:
            data, was_skipped = _read_with_lstat(workspace, name, fingerprint, max_bytes)
        if data is not None:
            artifacts.append(
                ArtifactEvent(
                    name=name,
                    media_type=IMAGE_MEDIA_TYPES[_suffix(name)],
                    data=base64.b64encode(data).decode("ascii"),
                )
            )
        elif was_skipped:
            skipped.append(name)
    return artifacts, skipped


def _snapshot_with_dir_fds(workspace: Path) -> WorkspaceSnapshot:
    try:
        root_fd = os.open(workspace, _DIRECTORY_FLAGS)
    except OSError:
        return {}
    try:
        result: WorkspaceSnapshot = {}
        if stat.S_ISDIR(os.fstat(root_fd).st_mode):
            _scan_dir_fd(root_fd, (), result)
        return dict(sorted(result.items()))
    finally:
        os.close(root_fd)


def _scan_dir_fd(
    directory_fd: int,
    prefix: tuple[str, ...],
    result: WorkspaceSnapshot,
) -> None:
    try:
        with os.scandir(directory_fd) as entries:
            names = sorted(entry.name for entry in entries)
    except OSError:
        return

    for entry_name in names:
        try:
            info = os.stat(entry_name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            continue
        if _is_link(info):
            continue
        if stat.S_ISDIR(info.st_mode):
            try:
                child_fd = os.open(entry_name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            except OSError:
                continue
            try:
                if stat.S_ISDIR(os.fstat(child_fd).st_mode):
                    _scan_dir_fd(child_fd, (*prefix, entry_name), result)
            finally:
                os.close(child_fd)
            continue
        name = "/".join((*prefix, entry_name))
        if stat.S_ISREG(info.st_mode) and _suffix(name) in IMAGE_MEDIA_TYPES:
            result[name] = _fingerprint(info)


def _snapshot_with_lstat(workspace: Path) -> WorkspaceSnapshot:
    root, _ = _checked_lstat(workspace, directory=True)
    if root is None:
        return {}
    result: WorkspaceSnapshot = {}
    try:
        walker = os.walk(workspace, topdown=True, followlinks=False)
        for directory, directory_names, file_names in walker:
            directory_path = Path(directory)
            current, _ = _checked_lstat(directory_path, directory=True)
            if current is None:
                directory_names.clear()
                continue
            directory_names[:] = [
                name
                for name in sorted(directory_names)
                if _checked_lstat(directory_path / name, directory=True)[0] is not None
            ]
            for file_name in sorted(file_names):
                path = directory_path / file_name
                info, _ = _checked_lstat(path, directory=False)
                name = path.relative_to(workspace).as_posix()
                if info is not None and _suffix(name) in IMAGE_MEDIA_TYPES:
                    result[name] = _fingerprint(info)
    except OSError:
        pass
    return dict(sorted(result.items()))


def _read_with_dir_fds(
    workspace: Path,
    name: str,
    expected: tuple[int, int],
    max_bytes: int,
) -> tuple[bytes | None, bool]:
    parts = _safe_parts(name)
    if parts is None:
        return None, False
    directory_fds: list[int] = []
    file_fd: int | None = None
    try:
        try:
            current_fd = os.open(workspace, _DIRECTORY_FLAGS)
        except FileNotFoundError:
            return None, True
        except OSError:
            return None, False
        directory_fds.append(current_fd)

        for component in parts[:-1]:
            try:
                current_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            except FileNotFoundError:
                return None, True
            except OSError:
                return None, False
            directory_fds.append(current_fd)
            if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
                return None, False

        try:
            file_fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=current_fd)
        except FileNotFoundError:
            return None, True
        except OSError:
            return None, False
        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode):
            return None, False
        return _read_checked(file_fd, opened, expected, max_bytes)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def _read_with_lstat(
    workspace: Path,
    name: str,
    expected: tuple[int, int],
    max_bytes: int,
) -> tuple[bytes | None, bool]:
    parts = _safe_parts(name)
    if parts is None:
        return None, False
    checked, was_skipped = _lstat_image_path(workspace, parts)
    if checked is None:
        return None, was_skipped
    path, before_open = checked
    try:
        file_fd = os.open(path, _FILE_FLAGS)
    except FileNotFoundError:
        return None, True
    except OSError:
        return None, False
    try:
        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode):
            return None, False
        if _identity(opened) != _identity(before_open):
            return None, True
        data, was_skipped = _read_checked(file_fd, opened, expected, max_bytes)
        if data is None:
            return None, was_skipped
        rechecked, was_skipped = _lstat_image_path(workspace, parts)
        if rechecked is None:
            return None, was_skipped
        if _identity(rechecked[1]) != _identity(opened):
            return None, True
        return data, False
    finally:
        os.close(file_fd)


def _lstat_image_path(
    workspace: Path,
    parts: tuple[str, ...],
) -> tuple[tuple[Path, os.stat_result] | None, bool]:
    current = workspace
    root, was_skipped = _checked_lstat(current, directory=True)
    if root is None:
        return None, was_skipped

    for component in parts[:-1]:
        current /= component
        info, was_skipped = _checked_lstat(current, directory=True)
        if info is None:
            return None, was_skipped

    current /= parts[-1]
    info, was_skipped = _checked_lstat(current, directory=False)
    if info is None:
        return None, was_skipped
    return (current, info), False


def _checked_lstat(path: Path, *, directory: bool) -> tuple[os.stat_result | None, bool]:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None, True
    except OSError:
        return None, False
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if _is_link(info) or not expected_type(info.st_mode):
        return None, False
    return info, False


def _read_checked(
    file_fd: int,
    opened: os.stat_result,
    expected: tuple[int, int],
    max_bytes: int,
) -> tuple[bytes | None, bool]:
    if _fingerprint(opened) != expected or opened.st_size > max_bytes:
        return None, True
    try:
        data = _read_bounded(file_fd, max_bytes + 1)
        final = os.fstat(file_fd)
    except OSError:
        return None, False
    if not stat.S_ISREG(final.st_mode):
        return None, False
    if _fingerprint(final) != expected or final.st_size != len(data) or len(data) > max_bytes:
        return None, True
    return data, False


def _read_bounded(file_fd: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    while limit > 0:
        chunk = os.read(file_fd, min(limit, _READ_CHUNK_BYTES))
        if not chunk:
            break
        chunks.append(chunk)
        limit -= len(chunk)
    return b"".join(chunks)


def _safe_parts(name: str) -> tuple[str, ...] | None:
    parts = tuple(name.split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    return parts


def _suffix(name: str) -> str:
    return PurePosixPath(name).suffix.lower()


def _is_link(info: os.stat_result) -> bool:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(info.st_mode) or bool(reparse & getattr(info, "st_file_attributes", 0))


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _fingerprint(info: os.stat_result) -> tuple[int, int]:
    return info.st_mtime_ns, info.st_size
