# tests/test_p1_endpoints.py
# P1.md 5.5 verification point 4: the five base endpoints plus the OpenAPI schema.
#
# These run in process through fastapi TestClient. The checks that need a real
# server process live in test_p1_service.py.

import numbers

import pytest

# 3.6.2: the endpoints P1 implements, and the business endpoints it must not
EXPECTED_PATHS = {
    "/health",
    "/status",
    "/analytics",
    "/admin/reload-config",
    "/admin/services",
}
FORBIDDEN_PATHS = {
    "/route",
    "/route/advanced",
    "/feedback",
    "/quality/dashboard",
    "/adapters/canary/status",
}


def test_health_returns_healthy_with_no_services(api_client):
    """5.5 V4-1: status healthy, services {}, timestamp is a number."""
    response = api_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["services"] == {}
    assert isinstance(body["timestamp"], numbers.Number)


def test_admin_services_is_empty(api_client):
    """5.5 V4-2."""
    response = api_client.get("/admin/services")

    assert response.status_code == 200
    assert response.json() == {"services": [], "count": 0}


def test_reload_config_succeeds(api_client):
    """5.5 V4-3."""
    response = api_client.post("/admin/reload-config")

    assert response.status_code == 200
    assert response.json() == {"status": "config_reloaded"}


def test_status_exposes_router_mode(api_client):
    """5.5 V4-4: 200, carries router_mode, no exception."""
    response = api_client.get("/status")

    assert response.status_code == 200
    body = response.json()
    assert "router_mode" in body
    assert "system" in body


def test_analytics_returns_base_structure(api_client):
    response = api_client.get("/analytics")

    assert response.status_code == 200
    assert response.json() == {"system": {}}


def test_openapi_title_version_and_paths(api_client):
    """5.5 V4-5: /docs shows "LLM Router & Execution Platform v2.0.0" and lists
    the five endpoints."""
    schema = api_client.get("/openapi.json").json()

    assert schema["info"]["title"] == "LLM Router & Execution Platform"
    assert schema["info"]["version"] == "2.0.0"
    assert EXPECTED_PATHS <= set(schema["paths"])


def test_business_endpoints_are_not_implemented_yet(api_client):
    """3.6.2: /route and friends belong to later phases."""
    schema = api_client.get("/openapi.json").json()

    leaked = FORBIDDEN_PATHS & set(schema["paths"])
    assert not leaked, f"endpoints that P1 should not implement: {leaked}"


@pytest.mark.parametrize("path", sorted(EXPECTED_PATHS - {"/admin/reload-config"}))
def test_endpoints_return_json(api_client, path):
    """3.6 core constraint: every endpoint answers with application/json."""
    response = api_client.get(path)

    assert response.headers["content-type"].startswith("application/json")


def test_health_reports_degraded_instead_of_raising(platform_instance):
    """3.6 core constraint: /health must never return 500, even when a service
    blows up while being probed."""
    from fastapi.testclient import TestClient

    class ExplodingService:
        def is_healthy(self):
            raise RuntimeError("boom")

    original = platform_instance.services
    platform_instance.services = {"exploding": ExplodingService()}
    try:
        response = TestClient(platform_instance._create_fastapi_app()).get("/health")
    finally:
        platform_instance.services = original

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "boom" in body["error"]


def test_health_marks_the_whole_platform_degraded_when_one_service_is_down(platform_instance):
    """3.6.2: all_ok is "every service healthy". A service reporting
    {"healthy": False} has to drag the overall status down."""
    from fastapi.testclient import TestClient

    class DownService:
        def get_health_status(self):
            return {"healthy": False}

    class UnknownService:
        pass

    original = platform_instance.services
    platform_instance.services = {"down": DownService(), "unknown": UnknownService()}
    try:
        response = TestClient(platform_instance._create_fastapi_app()).get("/health")
    finally:
        platform_instance.services = original

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "degraded", body
    assert body["services"]["unknown"] == {"healthy": True}
