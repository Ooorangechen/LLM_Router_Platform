"""POST /route HTTP-boundary tests without external providers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import LLMRouterPlatform
from src.utils.schema import InferenceResponse, UserTier


def _client(response: InferenceResponse):
    platform = object.__new__(LLMRouterPlatform)
    platform.config = {"api": {"cors_origins": ["*"]}}
    process_query = AsyncMock(return_value=response)
    platform.services = {"inference": SimpleNamespace(process_query=process_query)}
    return TestClient(platform._create_fastapi_app()), process_query


def test_route_maps_request_and_response_contract():
    response = InferenceResponse(
        response_text="hello", model_name="fallback-model", provider="test",
        token_count_input=4, token_count_output=2, latency_ms=12,
        tokens_per_second=10.0, cost_usd=0.001, cached=True,
    )
    client, process_query = _client(response)

    with patch("main.src.utils.metrics.SYSTEM_METRICS") as metrics:
        result = client.post("/route", json={"query": "Hi", "user_id": "user-1"})

    assert result.status_code == 200
    assert result.json() == {
        "query_id": process_query.await_args.args[0].request_id,
        "response": "hello",
        "model_name": "fallback-model",
        "tokens": {"input": 4, "output": 2, "total": 6},
        "cost_usd": 0.001,
        "latency_ms": 12,
        "cached": True,
    }
    request = process_query.await_args.args[0]
    assert request.user_tier == UserTier.FREE
    assert request.max_tokens == 512
    assert request.temperature == 1.0
    metrics.requests_total.labels.assert_called_once_with(
        endpoint="/route", method="POST", status="200"
    )


def test_route_returns_500_for_terminal_inference_error():
    response = InferenceResponse(
        response_text="", model_name="unknown", provider="error",
        token_count_input=0, token_count_output=0, latency_ms=1,
        tokens_per_second=0.0, cost_usd=0.0, error="No inference providers are available",
    )
    client, _ = _client(response)

    with patch("main.src.utils.metrics.SYSTEM_METRICS") as metrics:
        result = client.post("/route", json={"query": "Hi", "user_id": "user-1"})

    assert result.status_code == 500
    assert result.json() == {"detail": "No inference providers are available"}
    metrics.errors_total.labels.assert_called_once_with(
        component="api", error_type="RuntimeError"
    )
