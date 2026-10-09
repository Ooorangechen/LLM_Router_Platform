"""Router -> 真实 provider 的 inference 集成检查。

需要：根目录 .env 中的 provider API key。会产生 API 费用；cache 与压缩在本测试中关闭。
不加 --run-external 时默认跳过。

venv/bin/python -m pytest tests/integration/test_p2_inference_real.py --run-external -q -s
"""

from copy import deepcopy

import pytest

from src.llm_router_part2_inference import InferenceEngine
from src.utils.schema import QueryRequest


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
