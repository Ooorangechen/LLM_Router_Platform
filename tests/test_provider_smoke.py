"""真实 provider smoke tests（每项含一次普通调用和一次流式调用，可能产生费用）。

从项目根目录运行，例如仅测试 OpenAI：
RUN_PROVIDER_SMOKE=1 .venv/bin/python -m pytest tests/test_provider_smoke.py::test_openai -q -s

去掉 ::test_openai 可测试全部三个 provider；普通 pytest 默认跳过本文件。
读取 config/config.yaml，每个 provider 选择其第一个配置模型。
自动加载根目录 .env，不覆盖已有环境变量。vLLM 需要事先启动对应服务。
这里只测 provider，不创建 Router，也不加载 tokenizer、缓存或压缩器。
"""

import os
from pathlib import Path

import pytest
import yaml

from src.llm_router_part2_inference import OpenAIProvider, AnthropicProvider, vLLMProvider
from src.utils.schema import ModelConfig, QueryRequest


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_PROVIDER_SMOKE") != "1",
        reason="真实请求需显式设置 RUN_PROVIDER_SMOKE=1",
    ),
]


@pytest.fixture
def config():
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    return yaml.safe_load((root / "config/config.yaml").read_text(encoding="utf-8"))


def models_for(config, provider_name):
    return {
        name: ModelConfig(name=name, **settings)
        for name, settings in config["router"]["models"].items()
        if settings["provider"] == provider_name
    }


@pytest.fixture
def request_data():
    return QueryRequest(
        query="Reply with one short greeting.",
        user_id="provider-smoke-test",
        max_tokens=512,
        # 当前默认 reasoning 模式使用 temperature=1；schema 参数重构留待后续。
        temperature=1.0,
    )


async def test_openai(config, request_data):
    """验证 OpenAI 凭证、模型调用、usage、成本和流式文本。"""
    models = models_for(config, "openai")
    model_name = next(iter(models))
    provider = OpenAIProvider(config["inference"]["openai"], models)

    try:
        assert await provider.initialize(), "请填写 OPENAI_API_KEY"
        response = await provider.generate_response(request_data, model_name)
        assert response.error is None
        assert response.response_text.strip()
        assert response.provider == "openai" and response.model_name == model_name
        assert response.token_count_input > 0 and response.token_count_output > 0
        price = models[model_name]
        assert response.cost_usd == pytest.approx(
            response.token_count_input * price.cost_input_token
            + response.token_count_output * price.cost_output_token
        )
        print(f"OpenAI: {model_name}, tokens={response.total_tokens}, cost=${response.cost_usd:.6f}")

        chunks = [text async for text in provider.stream_response(request_data, model_name)]
        assert "".join(chunks).strip(), "流式调用未返回文本"
        print(f"OpenAI stream: {''.join(chunks)}")
    finally:
        if provider.client is not None:
            await provider.client.close()


async def test_anthropic(config, request_data):
    """验证 Anthropic 凭证、模型调用、usage、成本和流式文本。"""
    models = models_for(config, "anthropic")
    model_name = next(iter(models))
    provider = AnthropicProvider(config["inference"]["anthropic"], models)

    try:
        assert await provider.initialize(), "请填写 ANTHROPIC_API_KEY"
        response = await provider.generate_response(request_data, model_name)
        assert response.error is None
        assert response.response_text.strip()
        assert response.provider == "anthropic" and response.model_name == model_name
        assert response.token_count_input > 0 and response.token_count_output > 0
        price = models[model_name]
        assert response.cost_usd == pytest.approx(
            response.token_count_input * price.cost_input_token
            + response.token_count_output * price.cost_output_token
        )
        print(f"Anthropic: {model_name}, tokens={response.total_tokens}, cost=${response.cost_usd:.6f}")

        chunks = [text async for text in provider.stream_response(request_data, model_name)]
        assert "".join(chunks).strip(), "流式调用未返回文本"
        print(f"Anthropic stream: {''.join(chunks)}")
    finally:
        if provider.client is not None:
            await provider.client.close()


async def test_vllm(config, request_data):
    """验证本地 vLLM health、served model name、普通和 SSE 流式调用。"""
    models = models_for(config, "vllm")
    model_name = next(iter(models))
    provider = vLLMProvider(config["inference"]["vllm"], models)

    try:
        assert await provider.initialize()
        response = await provider.generate_response(request_data, model_name)
        assert response.error is None
        assert response.response_text.strip()
        assert response.provider == "vllm" and response.model_name == model_name
        # P2 允许 vLLM 不返回 usage，此时计数为 0。
        assert response.token_count_input >= 0 and response.token_count_output >= 0
        assert response.cost_usd == 0.0
        print(f"vLLM: {model_name}, tokens={response.total_tokens}, cost=$0")

        chunks = [text async for text in provider.stream_response(request_data, model_name)]
        assert "".join(chunks).strip(), "流式调用未返回文本"
        print(f"vLLM stream: {''.join(chunks)}")
    finally:
        if provider.http_client is not None:
            await provider.http_client.aclose()
