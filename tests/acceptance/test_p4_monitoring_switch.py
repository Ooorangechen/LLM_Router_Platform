"""P4 M1 (§5.2, §5.7): the monitoring master switch.

OFF: P2 base endpoints and /route behave as before; the log has no monitoring lines.
ON without Prometheus/Grafana running: the service starts, /route still serves, and
monitoring degradation is logged at info level only.

The full P2 §5.3~§5.9 rerun needs the P2 acceptance suite; this file covers §5.3
(base endpoints) and §5.4 (/route contract) under each switch position.
"""

import re
import time
import uuid

import pytest

from tests.acceptance.server import MONITORING_OFF, MONITORING_ON, deep_merge, free_port


# §5.2 and §5.7 grep patterns combined.
MONITORING_LOG = re.compile(r"monitor|alert|prometheus|grafana|resource_collector", re.I)
ROUTE_KEYS = {"query_id", "response", "model_name", "tokens", "cost_usd", "latency_ms", "cached"}


@pytest.fixture(scope="module")
def off_server(server_factory):
    return server_factory(overrides=MONITORING_OFF)


@pytest.fixture(scope="module")
def on_server(server_factory):
    # Nothing listens on this port: Prometheus behaves as if the stack were not started.
    no_stack = {"monitoring": {"prometheus": {"server_url": f"http://127.0.0.1:{free_port()}"}}}
    return server_factory(overrides=deep_merge(MONITORING_ON, no_stack))


def _route_once(server):
    payload = {"query": f"Say hello. ({uuid.uuid4()})", "user_id": "p4-switch",
               "max_tokens": 32, "temperature": 1.0}
    with server.client() as client:
        result = client.post("/route", json=payload)
    assert result.status_code == 200, result.text
    body = result.json()
    assert set(body) == ROUTE_KEYS
    assert body["response"].strip()
    assert body["tokens"]["total"] == body["tokens"]["input"] + body["tokens"]["output"]
    assert body["cached"] is False


# ---------- OFF ----------

def test_off_base_endpoints_unchanged(off_server):
    with off_server.client() as client:
        health = client.get("/health").json()
        services = client.get("/admin/services").json()
        reload = client.post("/admin/reload-config")

    assert health["services"]["router"]["status"] == "healthy"
    assert health["services"]["inference"]["status"] == "healthy"
    assert {"router", "inference"} <= set(services["services"])
    assert not {"monitor", "alert"} & set(services["services"])
    assert reload.status_code == 200 and reload.json() == {"status": "config_reloaded"}


@pytest.mark.real_provider
def test_off_route_contract_unchanged(off_server):
    _route_once(off_server)


def test_off_log_has_no_monitoring_lines(off_server):
    # Runs after the tests above in this module, so the log includes their requests.
    hits = [line for line in off_server.log_text().splitlines() if MONITORING_LOG.search(line)]
    assert hits == [], f"{len(hits)} monitoring lines, first: {hits[:3]}"


# ---------- ON, no monitoring stack ----------

def test_on_without_stack_starts_with_monitoring_services(on_server):
    with on_server.client() as client:
        result = client.get("/health")
        services = client.get("/admin/services").json()["services"]

    assert result.status_code == 200
    assert {"router", "inference", "monitor", "alert"} <= set(services)


@pytest.mark.real_provider
def test_on_without_stack_route_still_serves(on_server):
    _route_once(on_server)


def test_on_without_stack_logs_degradation_at_info_only(on_server):
    time.sleep(5)  # let the alert loop (2s) evaluate its Prometheus-backed rule a few times
    noisy = [line for line in on_server.log_text().splitlines()
             if MONITORING_LOG.search(line) and (" - WARNING - " in line or " - ERROR - " in line)]
    assert noisy == [], f"{len(noisy)} warning/error monitoring lines, first: {noisy[:3]}"
