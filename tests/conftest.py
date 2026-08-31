# tests/conftest.py
# Shared fixtures for the P1 acceptance suite.
# Automates the manual verification procedure in docs/P1.md section 5.

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# make `import main` / `import src...` work no matter where pytest is invoked from
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# the port main.py binds by default, and the one the `health` CLI command hardcodes
API_PORT = 8080
SERVER_BOOT_TIMEOUT_S = 30.0

# P1.md 5.4: the process must stay alive for at least 30 seconds
LIVENESS_WINDOW_S = 30.0


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes several seconds of wall clock time")


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) != 0


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def real_config_path(project_root) -> Path:
    path = project_root / "config" / "config.yaml"
    if not path.is_file():
        pytest.skip(f"{path} missing, run `python main.py setup` first")
    return path


@pytest.fixture(scope="session")
def platform_instance(real_config_path):
    """A LLMRouterPlatform built from the real config, no server started."""
    from main import LLMRouterPlatform

    return LLMRouterPlatform(str(real_config_path))


@pytest.fixture(scope="session")
def api_client(platform_instance):
    """In-process client for the FastAPI app, used for the section 5.5 endpoint checks."""
    from fastapi.testclient import TestClient

    return TestClient(platform_instance._create_fastapi_app())


@pytest.fixture(scope="session")
def live_server(project_root, tmp_path_factory):
    """Start `python main.py start` as a real subprocess.

    Needed for the checks that cannot be done in-process: the startup log lines
    (5.4) and the `health` CLI command, which talks HTTP to localhost (5.6).
    """
    import httpx

    if not _port_is_free(API_PORT):
        pytest.skip(f"port {API_PORT} already in use, stop the running server first")

    log_path = tmp_path_factory.mktemp("server") / "stdout.log"
    log_file = log_path.open("w", encoding="utf-8")

    started_at = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "main.py", "start"],
        cwd=str(project_root),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        deadline = time.monotonic() + SERVER_BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log_file.flush()
                raise RuntimeError(
                    f"server exited early with code {proc.returncode}\n"
                    f"{log_path.read_text(encoding='utf-8')}"
                )
            try:
                if httpx.get(f"http://localhost:{API_PORT}/health", timeout=1.0).status_code == 200:
                    break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError(f"server did not become ready within {SERVER_BOOT_TIMEOUT_S}s")

        log_file.flush()
        yield {
            "proc": proc,
            "port": API_PORT,
            "log_path": log_path,
            "started_at": started_at,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_file.close()


def run_cli(project_root: Path, *args, timeout: int = 30):
    """Invoke `python main.py <args>` and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def parse_json_stdout(result) -> dict:
    """Parse a CLI command's stdout as JSON, with the raw output in the failure message."""
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"stdout is not JSON: {exc}\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
        ) from exc
