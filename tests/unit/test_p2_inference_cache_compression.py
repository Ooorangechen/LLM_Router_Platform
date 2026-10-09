"""Offline regression checks for contextual caching and compression budgets."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.llm_router_part1_router import TokenCounter
from src.llm_router_part2_inference import ContextCompressor, InferenceEngine, ResponseCache
from src.utils.schema import InferenceResponse, QueryRequest


@pytest.fixture
def counter():
    with patch.object(TokenCounter, "_initialize_encoders"):
        return TokenCounter()


@pytest.mark.parametrize("method", ["semantic_graph", "sketch_based", "attention_based", "sliding_window"])
@pytest.mark.asyncio
async def test_token_budget_and_character_fallback(counter, method, caplog):
    # Deliberately different from character count to verify token-based triggering.
    counter.encoders["default"] = lambda text: len(text.encode("utf-8"))
    compressor = ContextCompressor(
        dict(enabled=True, max_context_tokens=40, compression_ratio=0.5, method=method), counter,
    )
    context = "重要内容。" * 5
    assert len(context) < 40 < counter.count_tokens(context)
    result = await compressor.compress_context(context, "default")
    assert result != context
    assert 0 < counter.count_tokens(result) <= 37
    result = await compressor.compress_context(context, "default", target_tokens=10)
    assert 0 < counter.count_tokens(result) <= 10

    with patch.object(counter, "count_tokens", side_effect=RuntimeError("tokenizer unavailable")):
        result = await compressor.compress_context(context, "default", target_tokens=10)
    assert 0 < len(result) <= 10
    assert "character count" in caplog.text


def test_cache_key_exact_inputs():
    cache = ResponseCache({})
    request = QueryRequest(query="hello", user_id="a", context="上下文" * 10000)
    key = cache.generate_cache_key(request, "model")
    assert len(key) == 64
    assert cache.generate_cache_key(request.model_copy(update={"user_id": "b"}), "model") == key
    for field, value in [("context", request.context + " "), ("context", None),
                         ("query", "other"), ("temperature", 0.2), ("max_tokens", 8)]:
        assert cache.generate_cache_key(request.model_copy(update={field: value}), "model") != key
    assert cache.generate_cache_key(request, "other-model") != key


@pytest.mark.asyncio
async def test_cache_uses_original_context_before_compression(counter):
    router = SimpleNamespace(
        token_counter=counter,
        route_query=AsyncMock(return_value=SimpleNamespace(selected_model="model")),
        update_model_stats=Mock(),
    )
    engine = InferenceEngine({"cache": {"enabled": True}, "compression": {"enabled": True}}, router)
    provider = SimpleNamespace(generate_response=AsyncMock(side_effect=lambda *args: InferenceResponse(
        response_text="answer", model_name="model", provider="test", token_count_input=1,
        token_count_output=1, latency_ms=1, tokens_per_second=1, cost_usd=0,
    )))
    engine._resolve_available_model = Mock(return_value=("model", provider))
    engine.context_compressor.compress_context = AsyncMock(return_value="same compressed text")
    stored = {}

    async def setex(key, ttl, value):
        stored[key] = value

    engine.cache.redis_client = SimpleNamespace(get=AsyncMock(side_effect=stored.get), setex=setex)
    request = QueryRequest(query="hello", user_id="a", context="original context")
    first = await engine.process_query(request)
    second = await engine.process_query(request.model_copy(update={"user_id": "b"}))
    changed = await engine.process_query(request.model_copy(update={"context": "original context "}))
    assert not first.cached and first.compressed_context
    assert second.cached and second.compressed_context
    assert not changed.cached
    assert provider.generate_response.await_count == 2
    assert request.context == "original context"
    assert len(stored) == 2
