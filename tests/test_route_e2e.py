"""Real POST /route E2E acceptance test.

This test uses a real provider and Redis, so it is skipped by default. Run it
from the project root with output enabled:

RUN_ROUTE_E2E_TEST=1 venv/bin/python -m pytest tests/test_route_e2e.py -q -s
"""

import os

import httpx
import pytest

from main import LLMRouterPlatform
from src.utils.schema import QueryRequest


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_ROUTE_E2E_TEST") != "1",
        reason="Real /route E2E test requires explicit opt-in",
    ),
]


def status(step: int, message: str) -> None:
    print(f"[{step}] {message}", flush=True)


async def test_real_route_e2e():
    platform = LLMRouterPlatform()
    cache_keys = []

    try:
        status(1, "Initializing ModelRouter and InferenceEngine")
        await platform._initialize_services()
        assert set(platform.services) >= {"router", "inference"}

        router = platform.services["router"]
        inference = platform.services["inference"]
        status(2, f"Services ready: {list(platform.services)}")

        provider_names = list(inference.providers)
        status(3, f"Available providers: {provider_names}")
        assert provider_names, "No real provider initialized; check .env API keys"

        health = inference.get_health_status()
        status(4, f"Inference health: healthy={health['healthy']}, providers={health['providers']}")
        assert health["healthy"] is True

        status(5, f"Redis cache ready: enabled={inference.cache.enabled}")
        assert inference.cache.enabled and inference.cache.redis_client is not None, (
            "Redis cache is unavailable; start Redis and verify inference.cache configuration"
        )

        payload = {
            "query": (
                "Write a quick sort function in Python. The function name must be "
                "quick_sort, the input must be list[int], and it must return the sorted list."
            ),
            "user_id": "route-e2e-test",
            "user_tier": "premium",
            "max_tokens": 512,
            "temperature": 1.0,
        }

        request = QueryRequest(**payload)
        cache_keys = [
            inference.cache.generate_cache_key(request, model_name)
            for model_name in router.models
        ]
        await inference.cache.redis_client.delete(*cache_keys)
        status(6, f"Cache cleared for configured models: {list(router.models)}")

        transport = httpx.ASGITransport(app=platform._create_fastapi_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            status(7, "Sending first POST /route request; a real provider call will occur")
            first = await client.post("/route", json=payload)
            status(8, f"First response: HTTP {first.status_code}")
            if first.status_code != 200:
                status(8, f"Failure body: {first.text}")
            assert first.status_code == 200

            first_body = first.json()
            status(
                9,
                "First result: "
                f"model={first_body.get('model_name')}, "
                f"tokens={first_body.get('tokens')}, "
                f"cost=${first_body.get('cost_usd')}, "
                f"latency_ms={first_body.get('latency_ms')}, "
                f"cached={first_body.get('cached')}",
            )

            expected_fields = {
                "query_id", "response", "model_name", "tokens",
                "cost_usd", "latency_ms", "cached",
            }
            assert set(first_body) == expected_fields
            assert first_body["response"].strip()
            assert "def quick_sort" in first_body["response"].lower()
            assert first_body["model_name"] in router.models
            assert first_body["tokens"]["input"] >= 0
            assert first_body["tokens"]["output"] >= 0
            assert first_body["tokens"]["total"] == (
                first_body["tokens"]["input"] + first_body["tokens"]["output"]
            )
            assert isinstance(first_body["cost_usd"], (int, float))
            assert first_body["cost_usd"] >= 0
            assert isinstance(first_body["latency_ms"], int)
            assert first_body["latency_ms"] >= 0
            assert first_body["cached"] is False
            status(10, "First response contract passed")

            status(11, "Sending identical POST /route request; Redis hit expected")
            second = await client.post("/route", json=payload)
            status(12, f"Second response: HTTP {second.status_code}")
            if second.status_code != 200:
                status(12, f"Failure body: {second.text}")
            assert second.status_code == 200

            second_body = second.json()
            status(
                13,
                "Second result: "
                f"model={second_body.get('model_name')}, "
                f"latency_ms={second_body.get('latency_ms')}, "
                f"cached={second_body.get('cached')}",
            )
            assert second_body["cached"] is True
            assert second_body["response"] == first_body["response"]
            assert second_body["model_name"] == first_body["model_name"]
            assert second_body["tokens"] == first_body["tokens"]
            assert second_body["latency_ms"] < first_body["latency_ms"]
            status(14, "Redis cache hit and reduced latency passed")

        status(15, "Real POST /route E2E acceptance passed")
    finally:
        if cache_keys:
            inference = platform.services.get("inference")
            if inference is not None and inference.cache.redis_client is not None:
                await inference.cache.redis_client.delete(*cache_keys)
                status(16, "Removed E2E Redis cache entry")
        await platform._shutdown_services()
        status(17, "Platform services shut down")
