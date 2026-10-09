"""P4 M3 (§5.4): Prometheus scrapes the service, Grafana imports the dashboard, stat panels show data.

Needs the monitoring stack (bash scripts/start_monitoring_stack.sh) and port 8080 free:
monitoring/prometheus.yml scrapes llm-router-api at localhost:8080, so this
server must listen there.
"""

import json

import httpx
import pytest

from tests.acceptance.server import MONITORING_ON, route_payload, wait_until


PROMETHEUS = "http://localhost:9090"
GRAFANA = "http://localhost:3000"
GRAFANA_AUTH = ("admin", "admin")


@pytest.fixture(scope="module")
def server(server_factory):
    try:
        httpx.get(f"{PROMETHEUS}/-/ready", timeout=3).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.fail(f"Prometheus not reachable at {PROMETHEUS} "
                    f"(run scripts/start_monitoring_stack.sh): {exc}")
    return server_factory(overrides=MONITORING_ON, port=8080)


@pytest.fixture(scope="module")
def dashboard(project_root):
    return json.loads((project_root / "monitoring/grafana/dashboard.json").read_text(encoding="utf-8"))


def _target_health():
    targets = httpx.get(f"{PROMETHEUS}/api/v1/targets", timeout=5).json()["data"]["activeTargets"]
    return {t["labels"]["job"]: (t["health"], t.get("lastError", "")) for t in targets}


def test_prometheus_targets_api_and_node_exporter_up(server):
    # llm-router-api scrapes every 5s; allow a few rounds.
    wait_until(lambda: _target_health().get("llm-router-api", ("",))[0] == "up", timeout=30)
    health = _target_health()

    must_up = ["llm-router-api"] + (["node-exporter"] if "node-exporter" in health else [])
    down = {job: health.get(job) for job in must_up if health.get(job, ("",))[0] != "up"}
    assert not down, f"targets not up: {down}"


def test_grafana_imports_dashboard(dashboard):
    with httpx.Client(base_url=GRAFANA, auth=GRAFANA_AUTH, timeout=30) as client:
        assert client.get("/api/health").status_code == 200
        if client.get("/api/datasources/name/Prometheus").status_code == 404:
            created = client.post("/api/datasources", json={
                "name": "Prometheus", "type": "prometheus", "access": "proxy",
                "url": "http://host.docker.internal:9090", "isDefault": True})
            assert created.status_code < 300, created.text
        imported = client.post("/api/dashboards/db", json={"dashboard": dashboard, "overwrite": True})
        assert imported.status_code < 300, imported.text

        stored = client.get(f"/api/dashboards/uid/{dashboard['uid']}").json()["dashboard"]
    assert stored["refresh"] == "30s"
    assert stored["time"] == {"from": "now-6h", "to": "now"}


def _panel_query(expr):
    # Grafana template variables, resolved the way an "All / Last 6 hours" view would.
    for var, value in {"$model": ".*", "$user_tier": ".*", "$__range": "6h",
                       "$__rate_interval": "1m"}.items():
        expr = expr.replace(var, value)
    result = httpx.get(f"{PROMETHEUS}/api/v1/query", params={"query": expr}, timeout=5).json()
    values = [float(r["value"][1]) for r in result["data"]["result"]]
    return values[0] if values else 0.0


@pytest.mark.real_provider
def test_stat_panels_non_zero_after_traffic(server, dashboard):
    with server.client() as client:
        for i in range(20):
            client.post("/route", json=route_payload(i, user_id="dash"))

    stats = {p["title"]: p["targets"][0]["expr"] for p in dashboard["panels"] if p["type"] == "stat"}
    assert len(stats) >= 4

    def all_non_zero():
        values = {title: _panel_query(expr) for title, expr in stats.items()}
        return values if all(v > 0 for v in values.values()) else None

    # Two 5s scrapes plus the 1m rate window filling in.
    assert wait_until(all_non_zero, timeout=90, interval=5), {
        title: _panel_query(expr) for title, expr in stats.items()}
