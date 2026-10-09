"""Router 在真实 config/config.yaml 与真实 tokenizer 上的集成检查。

不调用 provider，但 tokenizer 初始化可能需要网络下载。
不加 --run-external 时默认跳过。

venv/bin/python -m pytest tests/integration/test_p2_router_real_config.py --run-external -q -s
"""

import pytest

from src.llm_router_part2_inference import ContextCompressor
from src.utils.schema import QueryRequest, QueryType


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
