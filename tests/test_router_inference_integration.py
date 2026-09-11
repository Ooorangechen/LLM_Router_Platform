"""Router 与 inference 的真实集成检查。

需要：根目录 .env 中的 provider API key，以及 localhost:6379 的 Redis。
测试默认跳过，避免普通 pytest 产生 API 费用或依赖外部服务。

运行全部测试：
RUN_ROUTER_INFERENCE_TEST=1 .venv/bin/python -m pytest \
    tests/test_router_inference_integration.py -q -s

仅运行最终真实 inference：
RUN_ROUTER_INFERENCE_TEST=1 .venv/bin/python -m pytest \
    tests/test_router_inference_integration.py::test_real_inference_pipeline -q -s
"""

import asyncio
import os
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.llm_router_part1_router import ModelRouter
from src.llm_router_part2_inference import ContextCompressor, InferenceEngine, ResponseCache
from src.utils.schema import QueryRequest, QueryType


pytestmark = [
    pytest.mark.skipif(
        os.getenv("RUN_ROUTER_INFERENCE_TEST") != "1",
        reason="真实集成测试需显式设置 RUN_ROUTER_INFERENCE_TEST=1",
    ),
]


class TracedRouter(ModelRouter):
    """只为测试打印调用边界；routing 逻辑仍由 ModelRouter 执行。"""

    async def route_query(self, request):
        print(
            f"\n[1] Router received: request_id={request.request_id}"
            f", user_tier={request.user_tier.value}, query={request.query!r}"
        )
        decision = await super().route_query(request)
        print(
            f"[4] Routing decision: model={decision.selected_model}"
            f", type={decision.query_type.value}, reason={decision.routing_reason}"
            f", fallbacks={decision.fallback_models}"
        )
        return decision

    async def _build_query_context(self, request):
        context = await super()._build_query_context(request)
        print(
            f"[2] Query context: type={context['query_type']}"
            f", confidence={context['classification_confidence']:.3f}"
            f", token_counts={context['token_counts']}"
        )
        return context

    def _get_available_models(self, context):
        available = super()._get_available_models(context)
        print(f"[3] Access + token-limit eligible models: {available}")
        return available


class TracedInferenceEngine(InferenceEngine):
    """打印 provider 初始化和最终执行模型，不改变 inference 行为。"""

    async def initialize(self):
        print("[5] Checking configured inference providers...")
        await super().initialize()
        print(f"[6] Available providers: {list(self.providers)}")

    def _resolve_available_model(self, requested):
        print(f"[7] Resolving routed model={requested} against available providers")
        model_name, provider = super()._resolve_available_model(requested)
        provider_name = self.router.models[model_name].provider
        print(f"[8] Final execution model={model_name}, provider={provider_name}")
        return model_name, provider


@pytest.fixture(scope="module")
def app_config():
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    return yaml.safe_load((root / "config/config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def router(app_config):
    result = TracedRouter(app_config["router"])
    asyncio.run(result.initialize())
    return result


def test_request_classifier(router):
    """检查中英文请求分类，并打印类别与置信度。"""
    examples = {
        "用 Python 写一个排序函数": QueryType.CODE_GENERATION,
        "Translate this sentence into Chinese": QueryType.TRANSLATION,
    }
    for query, expected in examples.items():
        query_type, confidence = router.classifier.classify_query(query)
        print(f"Classifier: {query!r} -> {query_type.value}, confidence={confidence:.3f}")
        assert query_type == expected
        assert 0.0 <= confidence <= 1.0


def test_token_counter(router):
    """检查各模型选择对应 tokenizer family，并得到正数 token count。"""
    text = "Token counting test with 中文内容。"
    for model_name in router.models:
        family = router.token_counter._get_encoder_key(model_name)
        count = router.token_counter.count_tokens(text, model_name)
        encoder = router.token_counter.encoder_names.get(family, "word-count approximation")
        print(f"TokenCounter: model={model_name}, family={family}, encoder={encoder}, tokens={count}")
        assert count > 0


@pytest.mark.asyncio
async def test_router_query_and_fallback(router):
    """打印 intelligent decision，并验证显式 fallback strategy。"""
    request = QueryRequest(
        query="用 Python 实现一个二分搜索函数",
        user_id="router-integration-test",
        user_tier="enterprise",
        max_tokens=128,
        temperature=1.0,
    )
    decision = await router.route_query(request)
    assert decision.selected_model in router.models
    assert decision.query_type == QueryType.CODE_GENERATION

    original_strategy = router.routing_strategy
    try:
        router.routing_strategy = "fallback"
        fallback = await router.route_query(request)
    finally:
        router.routing_strategy = original_strategy

    print(f"Fallback test: selected={fallback.selected_model}, reason={fallback.routing_reason}")
    assert fallback.selected_model == router.default_model
    assert fallback.routing_strategy == "fallback"


@pytest.mark.asyncio
async def test_context_compressor(router):
    """只测试 compressor 内部 token-budget 行为，不调用 provider。"""
    compressor = ContextCompressor(
        {
            "enabled": True,
            "max_context_tokens": 30,
            "compression_ratio": 0.3,
            "method": "sliding_window",
        },
        router.token_counter,
    )
    context = " ".join(f"Important context sentence number {index}." for index in range(80))
    model_name = "gpt-5.6-terra"
    before = router.token_counter.count_tokens(context, model_name)
    compressed = await compressor.compress_context(context, model_name)
    after = router.token_counter.count_tokens(compressed, model_name)

    print(f"Compressor: model={model_name}, before={before}, after={after}, changed={compressed != context}")
    assert compressed != context
    assert 0 < after <= compressor.max_context_tokens


@pytest.mark.asyncio
async def test_real_redis_cache(app_config):
    """连接真实 Redis，验证 miss -> write -> hit，并删除本测试的 key。"""
    cache_config = {**app_config["inference"]["cache"], "enabled": True, "ttl": 60}
    cache = ResponseCache(cache_config)
    await cache.initialize()
    assert cache.enabled and cache.redis_client is not None, "Redis 未在配置地址运行"

    request = QueryRequest(
        query="redis integration cache check",
        user_id="redis-integration-test",
        max_tokens=32,
        temperature=1.0,
    )
    key = cache.generate_cache_key(request, "gpt-5.6-terra")
    value = {"response_text": "cached integration response", "error": None}
    try:
        await cache.redis_client.delete(key)
        assert await cache.get_cached_response(key) is None
        await cache.cache_response(key, value)
        assert await cache.get_cached_response(key) == value
        print(f"Redis: miss -> write -> hit, key={key}, ttl={await cache.redis_client.ttl(key)}s")
    finally:
        await cache.redis_client.delete(key)
        await cache.redis_client.aclose()


@pytest.mark.asyncio
async def test_real_inference_pipeline(app_config, router):
    """真实执行 Router -> provider availability -> final model -> API response。"""
    inference_config = deepcopy(app_config["inference"])
    inference_config["cache"]["enabled"] = False
    inference_config["compression"]["enabled"] = False
    engine = TracedInferenceEngine(inference_config, router)
    request = QueryRequest(
        query="Reply with one short greeting.",
        user_id="inference-integration-test",
        user_tier="enterprise",
        max_tokens=128,
        temperature=1.0,
    )

    try:
        await engine.initialize()
        assert engine.providers, "没有可用 provider；请检查 .env API keys 和服务地址"
        response = await engine.process_query(request)
        print(
            f"[9] Provider response: model={response.model_name}, provider={response.provider}"
            f", input_tokens={response.token_count_input}, output_tokens={response.token_count_output}"
            f", latency_ms={response.latency_ms}, cost=${response.cost_usd:.6f}"
        )
        assert response.error is None
        assert response.response_text.strip()
        assert response.provider in engine.providers
        assert response.model_name in router.models
        assert response.token_count_input > 0 and response.token_count_output > 0
    finally:
        await engine.shutdown()
