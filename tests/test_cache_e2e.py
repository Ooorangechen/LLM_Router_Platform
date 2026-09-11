"""Real Redis cache-only E2E test.

No model provider is called. Run from the project root with:

RUN_CACHE_E2E_TEST=1 venv/bin/python -m pytest tests/test_cache_e2e.py -q -s
"""

import os
from pathlib import Path

import pytest
import yaml

from src.llm_router_part2_inference import ResponseCache
from src.utils.schema import InferenceResponse, QueryRequest


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_CACHE_E2E_TEST") != "1",
        reason="Real Redis cache E2E test requires explicit opt-in",
    ),
]


def status(step: int, message: str) -> None:
    print(f"[{step}] {message}", flush=True)


async def test_real_redis_cache_e2e():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config/config.yaml").read_text(encoding="utf-8"))
    cache_config = {**config["inference"]["cache"], "enabled": True, "ttl": 60}
    cache = ResponseCache(cache_config)

    request = QueryRequest(
        query="cache-only E2E request",
        user_id="cache-user-a",
        context="stable context",
        max_tokens=64,
        temperature=1.0,
    )
    same_inputs = request.model_copy(update={"user_id": "cache-user-b"})
    changed_context = request.model_copy(update={"context": "stable context "})
    model_name = "cache-e2e-model"
    key = cache.generate_cache_key(request, model_name)
    changed_key = cache.generate_cache_key(changed_context, model_name)

    try:
        status(1, "Connecting to real Redis")
        await cache.initialize()
        assert cache.enabled and cache.redis_client is not None, (
            "Redis is unavailable; start Redis and verify inference.cache configuration"
        )
        status(2, "Redis connection and PING passed")

        status(3, "Checking cache-key identity rules")
        assert cache.generate_cache_key(same_inputs, model_name) == key
        assert changed_key != key
        status(4, "Same generation inputs share a key; changed context has a different key")

        await cache.redis_client.delete(key, changed_key)
        status(5, "Removed stale entries for this E2E test")

        status(6, "Reading an empty key; cache miss expected")
        assert await cache.get_cached_response(key) is None
        status(7, "Initial cache miss passed")

        response = InferenceResponse(
            response_text="cached response",
            model_name=model_name,
            provider="cache-test",
            token_count_input=4,
            token_count_output=2,
            latency_ms=250,
            tokens_per_second=8.0,
            cost_usd=0.001,
        )
        response_data = response.model_dump(mode="json")

        status(8, "Writing a successful response with Redis TTL")
        await cache.cache_response(key, response_data)
        ttl = await cache.redis_client.ttl(key)
        assert 0 < ttl <= cache.ttl
        status(9, f"Cache write passed; remaining TTL={ttl}s")

        status(10, "Reading the same key; cache hit expected")
        cached = await cache.get_cached_response(key)
        assert cached == response_data
        restored = InferenceResponse(**{**cached, "cached": True})
        assert restored.cached is True
        assert restored.response_text == response.response_text
        assert restored.total_tokens == response.total_tokens
        status(11, "Cache hit, JSON round trip, and response restoration passed")

        status(12, "Reading the changed-context key; cache miss expected")
        assert await cache.get_cached_response(changed_key) is None
        status(13, "Changed-context cache miss passed")

        status(14, "Verifying error responses are not cached")
        error_data = {**response_data, "error": "provider failure"}
        await cache.cache_response(changed_key, error_data)
        assert await cache.redis_client.exists(changed_key) == 0
        status(15, "Error-response exclusion passed")

        status(16, "Real Redis cache-only E2E acceptance passed")
    finally:
        if cache.redis_client is not None:
            if cache.enabled:
                await cache.redis_client.delete(key, changed_key)
            await cache.redis_client.aclose()
            status(17, "Redis test entries removed and connection closed")
