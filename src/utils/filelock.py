import asyncio
import os
import time

try:
    import fcntl          # Unix (macOS / Linux)
except ImportError:
    fcntl = None
try:
    import msvcrt         # Windows
except ImportError:
    msvcrt = None

RETRY_INTERVAL_SEC = 0.05


class FileLock:
    """Cross-process exclusive lock held on a separate lock file; async context manager.

    Unix uses fcntl.flock, Windows msvcrt.locking; both are tried non-blocking and retried
    every RETRY_INTERVAL_SEC until `timeout`. With neither available the lock is a no-op
    (writes still happen, without cross-process protection), as P5 3.6 A allows.
    One instance guards one critical section; create a new FileLock per use.
    """

    def __init__(self, path: str, timeout: float = 10.0):
        self.path = str(path)
        self.timeout = timeout
        self._fd = None

    def _try_lock(self) -> bool:
        if fcntl is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                return False
        if msvcrt is not None:
            try:
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        return True

    def _unlock(self) -> None:
        if fcntl is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        elif msvcrt is not None:
            os.lseek(self._fd, 0, os.SEEK_SET)
            msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)

    async def __aenter__(self) -> "FileLock":
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.timeout
        while not self._try_lock():
            if time.monotonic() >= deadline:
                os.close(self._fd)
                self._fd = None
                raise TimeoutError(f"could not lock {self.path} within {self.timeout}s")
            await asyncio.sleep(RETRY_INTERVAL_SEC)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            self._unlock()
        finally:
            os.close(self._fd)
            self._fd = None
