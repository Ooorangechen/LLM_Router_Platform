# tests/test_p1_service.py
# P1.md 5.4 verification point 3 and 5.6 verification point 5.
#
# These need a real `python main.py start` process: the startup log lines only
# exist on the server's stdout, and the `health` CLI command talks HTTP to
# localhost. Everything else is covered in process by test_p1_endpoints.py.

import time

import httpx
import pytest

from tests.conftest import LIVENESS_WINDOW_S, parse_json_stdout, run_cli

# 5.4: the three lines that must appear on startup
EXPECTED_STARTUP_LOGS = [
    "Initializing LLM Router Platform services...",
    "All services initialized successfully",
    "Uvicorn running on http://0.0.0.0:8080",
]


def read_server_log(live_server) -> str:
    return live_server["log_path"].read_text(encoding="utf-8")


@pytest.mark.slow
@pytest.mark.parametrize("expected", EXPECTED_STARTUP_LOGS)
def test_startup_log_line_is_emitted(live_server, expected):
    """5.4: initialisation runs even though there are no services in P1."""
    assert expected in read_server_log(live_server)


@pytest.mark.slow
def test_server_stays_alive_for_thirty_seconds(live_server):
    """5.4: the process must not exit within 30 seconds."""
    remaining = LIVENESS_WINDOW_S - (time.monotonic() - live_server["started_at"])
    if remaining > 0:
        time.sleep(remaining)

    assert live_server["proc"].poll() is None, (
        f"server exited early with code {live_server['proc'].returncode}\n"
        f"{read_server_log(live_server)}"
    )


@pytest.mark.slow
def test_server_answers_over_the_network(live_server):
    """The in-process TestClient cannot prove the port was actually bound."""
    response = httpx.get(f"http://localhost:{live_server['port']}/health", timeout=5.0)

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.slow
def test_cli_health_reports_healthy(live_server, project_root):
    """5.6: `python main.py health` prints JSON containing status=healthy."""
    result = run_cli(project_root, "health")

    assert result.returncode == 0, result.stderr
    assert parse_json_stdout(result)["status"] == "healthy"


@pytest.mark.slow
def test_cli_health_for_an_unregistered_service(live_server, project_root):
    """5.6: no inference service exists in P1, so --service reports not found."""
    result = run_cli(project_root, "health", "--service", "inference")

    assert result.returncode == 0, result.stderr
    assert parse_json_stdout(result) == {"error": "not found"}


@pytest.mark.slow
def test_cli_health_fails_with_exit_code_1_when_nothing_is_running(project_root):
    """3.6.3: on failure the command writes to stderr and exits non zero, which is
    what a supervisor or CI step keys off."""
    result = run_cli(project_root, "health", "--port", "9", timeout=30)

    if result.returncode == 2 and "no such option" in result.stderr.lower():
        pytest.skip("this build of the health command has no --port option")

    assert result.returncode == 1
    assert "Health check failed" in result.stderr


def test_cli_exposes_the_four_subcommands(project_root):
    """3.6.3: setup / start / health / deploy are all reachable."""
    result = run_cli(project_root, "--help")

    assert result.returncode == 0, result.stderr
    for command in ("setup", "start", "health", "deploy"):
        assert command in result.stdout, f"{command} missing from --help"


def test_deploy_creates_the_output_scaffold(project_root, tmp_path):
    """3.6.3: the P1 deploy stub creates its output directories."""
    target = tmp_path / "deploy"
    result = run_cli(project_root, "deploy", "--output-dir", str(target))

    assert result.returncode == 0, result.stderr
    assert target.is_dir()
    assert (target / "docker").is_dir()


def test_main_py_does_not_import_business_modules(project_root):
    """3.6 core constraint: P1's main.py may not import any Part1-Part9 module."""
    source = (project_root / "main.py").read_text(encoding="utf-8")

    banned = ["ModelRouter", "InferenceEngine", "langgraph", "LangGraph"]
    found = [name for name in banned if name in source]
    assert not found, f"main.py references business modules: {found}"
