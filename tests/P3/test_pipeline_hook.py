"""Focused tests for the P2-to-P3 internal telemetry hook."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.llm_router_part2_inference import InferenceEngine
from src.llm_router_part3_pipeline import PipelineManager
from src.utils.schema import (
    InferenceResponse,
    QueryRequest,
    QueryType,
    RoutingDecision,
)


def _decision() -> RoutingDecision:
    return RoutingDecision(
        selected_model="model-a",
        query_type=QueryType.GENERAL,
        routing_reason="test",
        token_count=3,
        estimated_cost=0.01,
        routing_time_ms=2,
        confidence=0.9,
    )


def _response(*, error=None) -> InferenceResponse:
    return InferenceResponse(
        response_text="" if error else "answer",
        model_name="model-a",
        provider="test" if error is None else "error",
        token_count_input=3,
        token_count_output=2,
        latency_ms=4,
        tokens_per_second=1.0,
        cost_usd=0.01,
        error=error,
    )


def _engine(route_result, hook):
    router = SimpleNamespace(
        token_counter=SimpleNamespace(count_tokens=lambda text, model: len(text)),
        route_query=AsyncMock(side_effect=route_result),
        update_model_stats=Mock(),
    )
    engine = InferenceEngine({}, router, pipeline_hook=hook)
    provider = SimpleNamespace(generate_response=AsyncMock(return_value=_response()))
    engine._resolve_available_model = Mock(return_value=("model-a", provider))
    return engine


@pytest.mark.asyncio
async def test_process_query_keeps_response_contract_and_publishes_decision():
    hook = AsyncMock()
    decision = _decision()
    engine = _engine(lambda request: decision, hook)
    request = QueryRequest(query="hello", user_id="u1")

    response = await engine.process_query(request)
    await asyncio.sleep(0)

    assert isinstance(response, InferenceResponse)
    hook.assert_awaited_once()
    args = hook.await_args.args
    assert args[0] is request
    assert args[1] is decision
    assert args[2] is response
    assert args[3].tzinfo == timezone.utc
    assert args[4].tzinfo == timezone.utc
    assert args[3] <= args[4]
    assert args[5] is None


@pytest.mark.asyncio
async def test_cache_hit_publishes_without_calling_provider():
    hook = AsyncMock()
    decision = _decision()
    engine = _engine(lambda request: decision, hook)
    cached = _response().model_dump(mode="json")
    engine.cache.enabled = True
    engine.cache.get_cached_response = AsyncMock(return_value=cached)
    request = QueryRequest(query="hello", user_id="u1")

    response = await engine.process_query(request)
    await asyncio.sleep(0)

    assert response.cached is True
    hook.assert_awaited_once()
    assert hook.await_args.args[1] is decision
    engine._resolve_available_model.return_value[1].generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_route_failure_publishes_error_without_fake_decision():
    hook = AsyncMock(side_effect=RuntimeError("hook failure"))
    engine = _engine(RuntimeError("routing failed"), hook)

    response = await engine.process_query(QueryRequest(query="hello", user_id="u1"))
    await asyncio.sleep(0)

    assert response.error == "routing failed"
    assert hook.await_args.args[1] is None
    assert isinstance(hook.await_args.args[5], RuntimeError)


@pytest.mark.asyncio
async def test_hook_is_non_blocking_and_shutdown_waits_for_it():
    started = asyncio.Event()
    release = asyncio.Event()

    async def hook(*args):
        started.set()
        await release.wait()

    engine = _engine(lambda request: _decision(), hook)
    response = await engine.process_query(QueryRequest(query="hello", user_id="u1"))
    await started.wait()

    assert response.response_text == "answer"
    shutdown = asyncio.create_task(engine.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    release.set()
    await shutdown


@pytest.mark.asyncio
async def test_pipeline_manager_builds_business_events_and_error_event(tmp_path):
    config = {
        "pipeline": {"enabled": True, "dlq_local_dir": str(tmp_path)},
        "kafka": {"topics": {}},
        "clickhouse": {},
    }
    manager = PipelineManager(config)
    manager.producer.produce_batch = AsyncMock(return_value=(9, 0))
    request = QueryRequest(query="hello", user_id="u1")
    error = RuntimeError("provider failed")
    response = _response(error=str(error))
    received_at = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    completed_at = datetime(2026, 9, 24, 12, 0, 1, tzinfo=timezone.utc)

    await manager.publish_pipeline_events(
        request, _decision(), response, received_at, completed_at, error)

    records = manager.producer.produce_batch.await_args.args[0]
    topics = [record[0] for record in records]
    assert topics.count("llm-queries") == 1
    assert topics.count("llm-responses") == 1
    assert topics.count("llm-metrics") == 6
    assert topics.count("llm-errors") == 1
    query_event = next(record[2] for record in records if record[0] == "llm-queries")
    response_event = next(
        record[2] for record in records if record[0] == "llm-responses")
    assert query_event.request_received_at == received_at
    assert response_event.response_completed_at == completed_at


@pytest.mark.asyncio
async def test_pipeline_manager_route_failure_only_publishes_error(tmp_path):
    config = {
        "pipeline": {"enabled": True, "dlq_local_dir": str(tmp_path)},
        "kafka": {"topics": {}},
        "clickhouse": {},
    }
    manager = PipelineManager(config)
    manager.producer.produce_batch = AsyncMock(return_value=(1, 0))
    request = QueryRequest(query="hello", user_id="u1")

    await manager.publish_pipeline_events(
        request, None, _response(error="routing failed"),
        datetime.now(timezone.utc), datetime.now(timezone.utc),
        RuntimeError("routing failed"))

    records = manager.producer.produce_batch.await_args.args[0]
    assert [record[0] for record in records] == ["llm-errors"]
