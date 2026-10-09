"""P4 M5 / task 3.6: /health aggregation, /status and /analytics shapes, /admin/alerts fallbacks."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.utils.schema import HealthStatus


def _service(status, message="ok"):
    async def get_health_status():
        return HealthStatus(service_name="svc", status=status, message=message,
                            last_check_at=datetime.now(timezone.utc))
    return SimpleNamespace(get_health_status=get_health_status)


def _client(make_platform, services=None, config=None):
    platform = make_platform(config=config, services=services)
    return TestClient(platform._create_fastapi_app())


# ---------- /health ----------

@pytest.mark.parametrize("statuses, overall, score, http_code", [
    ([], "healthy", 0, 200),
    (["healthy", "healthy"], "healthy", 0, 200),
    (["healthy", "degraded"], "degraded", 1, 200),
    (["degraded", "unhealthy"], "unhealthy", 2, 503),
])
def test_health_aggregates_worst_status(make_platform, statuses, overall, score, http_code):
    services = {f"svc{i}": _service(s) for i, s in enumerate(statuses)}
    result = _client(make_platform, services).get("/health")

    assert result.status_code == http_code
    body = result.json()
    assert body["status"] == overall
    assert body["overall_score"] == score
    assert {"checked_at", "uptime_seconds", "services"} <= set(body)
    for entry in body["services"].values():
        assert set(entry) == {"status", "message", "last_check_at"}


def test_health_probe_exception_marks_service_unhealthy_not_500(make_platform):
    async def broken():
        raise RuntimeError("probe exploded")

    services = {"router": _service("healthy"),
                "inference": SimpleNamespace(get_health_status=broken)}
    result = _client(make_platform, services).get("/health")

    assert result.status_code == 503
    entry = result.json()["services"]["inference"]
    assert entry["status"] == "unhealthy"
    assert "probe exploded" in entry["message"]


def test_health_service_without_probe_counts_as_registered(make_platform):
    result = _client(make_platform, {"cache": SimpleNamespace()}).get("/health")

    assert result.status_code == 200
    assert result.json()["services"]["cache"]["message"] == "registered"


# ---------- /status ----------

STATUS_KEYS = {"timestamp", "uptime_seconds", "router_mode", "models_count",
               "providers_ready", "pipeline", "monitoring", "resource",
               "traffic_recent_1min"}


def test_status_with_monitoring_off_returns_full_shape_with_zeros(make_platform):
    config = {"api": {"cors_origins": ["*"]}, "monitoring": {"enabled": False}}
    body = _client(make_platform, config=config).get("/status").json()

    assert set(body) == STATUS_KEYS
    assert body["router_mode"] == "modular"
    assert body["pipeline"] == {"enabled": False, "kafka_ok": False, "clickhouse_ok": False}
    assert body["monitoring"] == {"enabled": False, "resource_collector_running": False,
                                  "alert_enabled": False, "active_alerts_count": 0}
    assert body["traffic_recent_1min"] == {"requests": 0, "error_rate": 0.0}


def test_status_lists_healthy_providers(make_platform):
    async def get_health_status():
        return HealthStatus(
            service_name="inference", status="healthy", message="1 provider ready",
            last_check_at=datetime.now(timezone.utc),
            metadata={"providers": {"openai": {"status": "healthy"},
                                    "vllm": {"status": "unhealthy"}}})

    services = {"inference": SimpleNamespace(get_health_status=get_health_status)}
    body = _client(make_platform, services).get("/status").json()

    assert body["providers_ready"] == ["openai"]


# ---------- /analytics ----------

def test_analytics_without_any_backend_returns_none_not_error(make_platform):
    result = _client(make_platform).get("/analytics?window_minutes=99999")

    assert result.status_code == 200
    body = result.json()
    assert body["data_source"] == "none"
    assert body["rows"] == []
    assert body["window_minutes"] == 1440  # clamped to 24h


def _router_with_stats():
    return SimpleNamespace(model_stats={
        "gpt-a": {"total_requests": 3, "success_rate": 1.0},
        "idle": {"total_requests": 0, "success_rate": 0.0},
    })


def test_analytics_falls_back_to_memory_when_clickhouse_query_fails(make_platform):
    client = Mock()
    client.query.side_effect = RuntimeError("clickhouse down")
    services = {
        "router": _router_with_stats(),
        "pipeline": SimpleNamespace(ch_writer=SimpleNamespace(enabled=True, client=client)),
    }
    body = _client(make_platform, services).get("/analytics").json()

    assert body["data_source"] == "memory"
    assert [row["model_name"] for row in body["rows"]] == ["gpt-a"]
    for row in body["rows"]:
        assert set(row) == {"model_name", "request_count", "success_rate",
                            "p95_latency_ms", "avg_cost_usd"}


def test_analytics_reads_clickhouse_when_available(make_platform):
    client = Mock()
    client.query.return_value = SimpleNamespace(result_rows=[("gpt-a", 10, 0.9, 812.0, 0.002)])
    services = {"pipeline": SimpleNamespace(ch_writer=SimpleNamespace(enabled=True, client=client))}
    body = _client(make_platform, services).get("/analytics?window_minutes=5").json()

    assert body["data_source"] == "clickhouse"
    assert body["rows"] == [{"model_name": "gpt-a", "request_count": 10, "success_rate": 0.9,
                             "p95_latency_ms": 812.0, "avg_cost_usd": 0.002}]
    assert "INTERVAL 5 MINUTE" in client.query.call_args.args[0]


# ---------- /admin/alerts ----------

@pytest.mark.parametrize("services", [{}, {"alert": SimpleNamespace(_running=False)}])
@pytest.mark.parametrize("path", ["/admin/alerts/active", "/admin/alerts/history"])
def test_admin_alerts_empty_when_alert_manager_absent_or_stopped(make_platform, services, path):
    result = _client(make_platform, services).get(path)

    assert result.status_code == 200
    assert result.json() == []
