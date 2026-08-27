"""Exclusive ownership of one local MindBridge data directory."""

from __future__ import annotations

import errno
import os
from pathlib import Path


class DataDirectoryInUseError(RuntimeError):
    """Raised when another store already owns a local data directory."""


class DataDirectoryLock:
    """Hold an operating-system lock for the lifetime of a local store."""

    def __init__(self, data_dir: Path) -> None:
        try:
            data_dir.mkdir(mode=0o700, parents=True)
        except FileExistsError:
            if not data_dir.is_dir():
                raise
        if os.name != "nt":
            os.chmod(data_dir, 0o700)
        self._path = data_dir / ".mindbridge.lock"
        self._descriptor: int | None = None
        self._acquire()

    def _acquire(self) -> None:
        descriptor = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            _lock_descriptor(descriptor)
        except OSError as error:
            os.close(descriptor)
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise DataDirectoryInUseError(
                    f"MindBridge data directory is already in use: {self._path.parent}"
                ) from None
            raise
        self._descriptor = descriptor
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)

    def close(self) -> None:
        """Release ownership; repeated calls are harmless."""
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            _unlock_descriptor(descriptor)
        finally:
            os.close(descriptor)


if os.name == "nt":

    def _lock_descriptor(descriptor: int) -> None:
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]

    def _unlock_descriptor(descriptor: int) -> None:
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]

else:

    def _lock_descriptor(descriptor: int) -> None:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_descriptor(descriptor: int) -> None:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)
