"""P3 M1 / task 3.7: the pipeline switch, startup degradation and the non-blocking /route hook."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

import main
import src.llm_router_part3_pipeline as pipeline_module
from src.llm_router_part3_pipeline import KafkaProducerManager, PipelineManager
from src.utils.schema import InferenceResponse, QueryRequest, QueryType, RoutingDecision


pytestmark = pytest.mark.asyncio


class FakeRouter:
    def __init__(self, config):
        self.config = config

    async def initialize(self):
        pass

    async def route_query(self, request):
        return DECISION


class FakeEngine:
    def __init__(self, config, router):
        self.router = router
        self.cache = SimpleNamespace()

    async def initialize(self):
        pass


DECISION = RoutingDecision(
    selected_model="gpt-a", query_type=QueryType.GENERAL, routing_reason="test",
    token_count=1, estimated_cost=0.0, routing_time_ms=1, confidence=0.5,
)
RESPONSE = InferenceResponse(
    response_text="hello", model_name="gpt-a", provider="test",
    token_count_input=1, token_count_output=1, latency_ms=5,
    tokens_per_second=1.0, cost_usd=0.0,
)


def _config(tmp_path, pipeline_enabled):
    return {
        "api": {"cors_origins": ["*"]},
        "router": {}, "inference": {},
        "pipeline": {"enabled": pipeline_enabled, "dlq_local_dir": str(tmp_path / "dlq")},
        "monitoring": {"enabled": False},
    }


@pytest.fixture
def fake_core():
    with patch.object(main, "ModelRouter", FakeRouter), \
            patch.object(main, "InferenceEngine", FakeEngine):
        yield


# ---------- startup: switch and degradation ----------

async def test_pipeline_disabled_never_builds_pipeline_or_wraps_router(
        make_platform, fake_core, tmp_path):
    platform = make_platform(_config(tmp_path, pipeline_enabled=False))
    with patch.object(pipeline_module, "PipelineManager") as manager:
        await platform._initialize_services()

    manager.assert_not_called()
    assert set(platform.services) == {"router", "inference", "cache"}
    # The P2 router is left untouched: no ContextVar capture wrapper.
    assert platform.services["router"].route_query.__func__ is FakeRouter.route_query


async def test_pipeline_init_exception_does_not_block_startup(make_platform, fake_core, tmp_path):
    platform = make_platform(_config(tmp_path, pipeline_enabled=True))
    failing = Mock(return_value=SimpleNamespace(
        initialize=AsyncMock(side_effect=RuntimeError("kafka exploded"))))
    with patch.object(pipeline_module, "PipelineManager", failing):
        await platform._initialize_services()

    assert "pipeline" not in platform.services
    assert {"router", "inference"} <= set(platform.services)


async def test_pipeline_with_unreachable_middleware_starts_degraded(
        make_platform, fake_core, tmp_path, monkeypatch):
    def refuse(**kwargs):
        raise ConnectionError("clickhouse unreachable")

    monkeypatch.setattr(pipeline_module, "clickhouse_connect", SimpleNamespace(get_client=refuse))
    monkeypatch.setattr(KafkaProducerManager, "_connect",
                        AsyncMock(side_effect=ConnectionError("kafka unreachable")))
    platform = make_platform(_config(tmp_path, pipeline_enabled=True))
    await platform._initialize_services()

    pipeline = platform.services["pipeline"]
    assert not pipeline.producer.enabled
    assert not pipeline.ch_writer.enabled
    assert not pipeline.consumer.enabled
    health = await pipeline.get_health_status()
    assert health.status == "degraded"
    # With the producer disabled, publishing is a no-op and must not raise.
    request = QueryRequest(query="hi", user_id="u")
    await pipeline.publish_pipeline_events(request, DECISION, RESPONSE, RESPONSE.timestamp)


async def test_publish_swallows_producer_exceptions(tmp_path):
    manager = PipelineManager({"pipeline": {"enabled": True, "dlq_local_dir": str(tmp_path)}})
    manager.producer.produce = AsyncMock(side_effect=RuntimeError("send failed"))
    request = QueryRequest(query="hi", user_id="u")

    await manager.publish_pipeline_events(request, DECISION, RESPONSE, RESPONSE.timestamp)

    manager.producer.produce.assert_awaited()


# ---------- /route hook ----------

def _route_platform(make_platform, publish, enabled=True, decision=DECISION):
    async def process_query(request):
        # Stands in for the ContextVar wrapper _initialize_services puts on route_query.
        main._routing_decision.set(decision)
        return RESPONSE

    services = {
        "inference": SimpleNamespace(process_query=process_query),
        "pipeline": SimpleNamespace(enabled=enabled, publish_pipeline_events=publish),
    }
    return make_platform(services=services)


async def _post_route(platform):
    transport = httpx.ASGITransport(app=platform._create_fastapi_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/route", json={"query": "hi", "user_id": "u1"})


async def test_route_returns_before_publish_finishes(make_platform):
    release = asyncio.Event()
    published = {}

    async def publish(**kwargs):
        await release.wait()
        published.update(kwargs)

    # If /route awaited the hook it would block on `release` forever; fail instead of hanging.
    result = await asyncio.wait_for(_post_route(_route_platform(make_platform, publish)), 2)

    # The response is back while publish is still blocked: the hook was not awaited.
    assert result.status_code == 200
    assert result.json()["response"] == "hello"
    assert published == {}

    release.set()
    for _ in range(5):
        await asyncio.sleep(0)
    assert published["routing_decision"] is DECISION
    assert published["inference_response"] is RESPONSE
    assert str(published["request"].request_id) == result.json()["query_id"]


@pytest.mark.parametrize("enabled, decision", [(False, DECISION), (True, None)])
async def test_route_skips_publish_without_enabled_pipeline_or_decision(
        make_platform, enabled, decision):
    publish = AsyncMock()

    result = await _post_route(_route_platform(make_platform, publish, enabled, decision))
    await asyncio.sleep(0)

    assert result.status_code == 200
    publish.assert_not_called()
