"""Fixtures shared by integration tests that run the router on the real config."""

import asyncio

import pytest

from src.llm_router_part1_router import ModelRouter


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


@pytest.fixture(scope="module")
def router(app_config):
    result = TracedRouter(app_config["router"])
    asyncio.run(result.initialize())
    return result
