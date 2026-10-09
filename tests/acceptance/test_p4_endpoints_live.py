"""P4 M5 (§5.6): /health, /status, /analytics on a running service, and the `main.py health` CLI."""

import json
import subprocess
import sys

import pytest

from tests.acceptance.server import MONITORING_ON, free_port, wait_until


EXIT_BY_SCORE = {0: 0, 1: 2, 2: 3}


@pytest.fixture(scope="module")
def server(server_factory):
    return server_factory(overrides=MONITORING_ON)


def _get(server, path, **params):
    with server.client() as client:
        return client.get(path, params=params).json()


def _cli(project_root, port, *args):
    return subprocess.run(
        [sys.executable, "main.py", "health", "--host", "localhost", "--port", str(port), *args],
        cwd=project_root, capture_output=True, text=True, timeout=30)


def test_health_lists_every_service_with_required_fields(server):
    body = _get(server, "/health")

    assert body["status"] in ("healthy", "degraded", "unhealthy")
    assert {"router", "inference", "monitor", "alert"} <= set(body["services"])
    for name, entry in body["services"].items():
        assert {"status", "message", "last_check_at"} <= set(entry), name


def test_status_reports_resources_after_first_collection(server):
    required = {"timestamp", "uptime_seconds", "router_mode", "models_count", "providers_ready",
                "pipeline", "monitoring", "resource", "traffic_recent_1min"}

    def collected():
        # The collector runs every 2s; cpu_percent -1 means no snapshot yet.
        body = _get(server, "/status")
        return body if body["resource"]["cpu_percent"] >= 0 else None

    body = wait_until(collected, timeout=15)

    assert body, "resource collector produced no snapshot"
    assert required <= set(body)
    assert body["monitoring"]["enabled"] is True
    assert body["monitoring"]["resource_collector_running"] is True
    for key in ("cpu_percent", "memory_percent", "disk_percent"):
        assert 0 <= body["resource"][key] <= 100


def test_analytics_returns_window_source_and_row_schema(server):
    body = _get(server, "/analytics", window_minutes=5)

    assert body["window_minutes"] == 5
    assert body["data_source"] in ("clickhouse", "memory", "none")
    for row in body["rows"]:
        assert set(row) >= {"model_name", "request_count", "success_rate",
                            "p95_latency_ms", "avg_cost_usd"}


def test_cli_text_exit_code_matches_health(server, project_root):
    score = _get(server, "/health")["overall_score"]
    result = _cli(project_root, server.port, "--format", "text")

    assert result.returncode == EXIT_BY_SCORE[score], result.stdout + result.stderr
    assert "LLM Router Platform Health" in result.stdout
    for name in ("router", "inference", "monitor", "alert"):
        assert name in result.stdout


def test_cli_json_output_is_valid(server, project_root):
    result = _cli(project_root, server.port, "--format", "json")

    assert json.loads(result.stdout)["status"] in ("healthy", "degraded", "unhealthy")


def test_cli_unreachable_service_exits_3(project_root):
    assert _cli(project_root, free_port(), "--timeout", "2").returncode == 3
