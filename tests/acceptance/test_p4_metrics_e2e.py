"""P4 M2 (§5.3): /metrics exposes every metric category and counts /route traffic."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.acceptance.server import MONITORING_ON, metric_total, route_payload


# M2: at least one line per category; checked by the # HELP line of each metric.
CATEGORY_METRICS = {
    "SYSTEM": ["llm_router_requests_total", "llm_router_errors_total"],
    "ROUTER": ["llm_router_router_routing_decisions_total"],
    "INFERENCE": ["llm_router_inference_requests_total", "llm_router_inference_cache_hits_total"],
    "PIPELINE": ["llm_router_pipeline_kafka_produce_total",
                 "llm_router_pipeline_clickhouse_write_total"],
    "RESOURCE": ["llm_router_resource_cpu_percent", "llm_router_resource_memory_percent"],
    "HEALTH": ["llm_router_health_service_health_info"],
}


@pytest.fixture(scope="module")
def server(server_factory):
    return server_factory(overrides=MONITORING_ON)


def _metrics(server):
    with server.client() as client:
        result = client.get("/metrics", follow_redirects=False)
    assert result.status_code == 200
    return result.text


def test_metrics_exposes_every_category(server):
    text = _metrics(server)
    lines = [line for line in text.splitlines() if line.strip()]
    helped = {line.split()[2] for line in lines if line.startswith("# HELP ")}

    assert len(lines) >= 30
    missing = {cat: [n for n in names if n not in helped] for cat, names in CATEGORY_METRICS.items()}
    assert not any(missing.values()), missing


@pytest.mark.real_provider
def test_100_routes_increment_requests_total_by_95_to_100(server):
    def route_200_count():
        return metric_total(_metrics(server), "llm_router_requests_total",
                            endpoint="/route", method="POST", status="200")

    before = route_200_count()
    with server.client() as client, ThreadPoolExecutor(max_workers=5) as pool:  # ab -c 5
        codes = list(pool.map(lambda i: client.post("/route", json=route_payload(i)).status_code,
                              range(100)))
    delta = route_200_count() - before

    assert 95 <= delta <= 100, f"delta={delta}, 2xx={sum(200 <= c < 300 for c in codes)}"
