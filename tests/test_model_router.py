import asyncio
import sys
from types import SimpleNamespace
from copy import deepcopy
from unittest.mock import patch

import pytest

from src.llm_router_part1_router import ModelCandidate, ModelRouter, QueryClassifier, TokenCounter
from src.utils.schema import QueryRequest, QueryType, RoutingDecision


def test_chinese_code_pattern_handles_lowercased_language_names():
    classifier = QueryClassifier()
    for language in ("Python", "JavaScript", "SQL", "Rust"):
        query_type, _ = classifier.classify_query(f"用{language}实现")
        assert query_type == QueryType.CODE_GENERATION


@pytest.mark.parametrize("failed_encoding", ["cl100k_base", "o200k_base"])
def test_tiktoken_initialization_isolates_encoding_failures(failed_encoding):
    calls = []

    def get_encoding(name):
        calls.append(name)
        if name == failed_encoding:
            raise RuntimeError("Simulated unavailable encoding")
        return SimpleNamespace(encode=lambda text, **kwargs: [1, 2])

    def unavailable_hf(*args, **kwargs):
        raise RuntimeError("No downloads in unit tests")

    with patch.dict(sys.modules, {
        "tiktoken": SimpleNamespace(get_encoding=get_encoding),
        "transformers": SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=unavailable_hf)
        ),
    }):
        counter = TokenCounter()

    assert calls == ["cl100k_base", "o200k_base", "o200k_base"]
    if failed_encoding == "cl100k_base":
        assert counter.count_tokens("example", "gpt-test") == 2
        assert counter.count_tokens("example", "claude-test") == 2
    else:
        assert counter.count_tokens("example", "default") == 2


@pytest.fixture
def router():
    config = {
        "default_model": "mistral-free",
        "models": {
            "mistral-free": dict(provider="vllm", max_tokens=100,
                                 priority=3, capabilities=["general"],
                                 cost_input_token=0, cost_output_token=0),
            "gpt-a": dict(provider="openai", max_tokens=100,
                          priority=2, capabilities=["coding", "analysis"],
                          cost_input_token=0.001, cost_output_token=0.002),
            "gpt-b": dict(provider="openai", max_tokens=100,
                          priority=2, capabilities=["coding", "analysis"],
                          cost_input_token=0.001, cost_output_token=0.002),
            "claude-a": dict(provider="anthropic", max_tokens=100,
                             priority=1, capabilities=["writing"],
                             cost_input_token=0.001, cost_output_token=0.003),
        },
        "routing_rules": [dict(name="code", condition="query_type == 'code_generation'",
                               models=["gpt-a", "gpt-b"], fallback="mistral-free")],
    }
    with patch.object(TokenCounter, "_initialize_encoders"):
        result = ModelRouter(config)
    result.token_counter.encoders = {
        "mistral": lambda text: 120, "gpt": lambda text: 40,
        "claude": lambda text: 80,
    }
    asyncio.run(result.initialize())
    return result


def run(router, **kwargs):
    return asyncio.run(router.route_query(QueryRequest(
        query="Write a function in Python", user_id="test", **kwargs
    )))


def test_rule_selection_uses_model_counts_cost_and_preserves_request(router):
    request = QueryRequest(query="Write a function in Python", user_id="u",
                           user_tier="premium", context="Additional context", max_tokens=10)
    before = request.model_dump()
    with patch.object(router.token_counter, "count_tokens",
                      wraps=router.token_counter.count_tokens) as count:
        decision = asyncio.run(router.route_query(request))
    assert count.call_count == 3  # Four models, three tokenizer families.
    assert all("Additional context" in call.args[0] for call in count.call_args_list)
    assert isinstance(decision, RoutingDecision)
    assert decision.selected_model == "gpt-a"  # Tie preserves rule order.
    assert decision.token_count == 40
    assert decision.estimated_cost == pytest.approx(0.06)
    assert decision.fallback_models == ["gpt-b"]
    assert request.model_dump() == before


@pytest.mark.parametrize("strategy", ["intelligent", "round_robin", "weighted"])
def test_common_pool_enforces_permissions_and_token_limits(router, strategy):
    router.routing_strategy = strategy
    assert run(router, user_tier="premium").selected_model in {"gpt-a", "gpt-b"}
    # No eligible free model: P2 explicitly returns the default nomination.
    assert run(router, user_tier="free").selected_model == "mistral-free"


def test_round_robin_and_weighted(router):
    router.routing_strategy = "round_robin"
    assert [run(router, user_tier="premium").selected_model for _ in range(3)] == [
        "gpt-a", "gpt-b", "gpt-a"
    ]
    router.routing_strategy = "weighted"
    with patch("src.llm_router_part1_router.random.choices", return_value=["claude-a"]) as choices:
        assert run(router, user_tier="enterprise").selected_model == "claude-a"
    assert choices.call_args.kwargs["weights"] == [0.25, 0.25, 0.5]
    assert router.routing_rules[0].weight == 1.0


def test_model_specific_rule_threshold_and_capability_fallthrough(router):
    rule = router.routing_rules[0]
    rule.condition = "token_count >= 60"
    rule.models = ["gpt-a", "claude-a"]
    decision = run(router, user_tier="enterprise")
    assert decision.selected_model == "claude-a"
    assert decision.token_count == 80
    rule.condition = "unknown_field > 1"
    assert run(router, user_tier="premium").selected_model == "gpt-a"


def test_rule_fallback_and_explicit_fallback(router):
    router.token_counter.encoders["mistral"] = lambda text: 10
    assert run(router, user_tier="free").selected_model == "mistral-free"
    router.routing_strategy = "fallback"
    assert run(router, user_tier="enterprise").selected_model == "mistral-free"


def test_exception_fallback_and_metrics_isolation(router):
    with patch.object(router, "_select_best_model", side_effect=ValueError("score failed")):
        decision = run(router, user_tier="premium")
    assert decision.selected_model == router.default_model
    assert decision.query_type == QueryType.GENERAL
    assert decision.confidence == 0.0
    assert "Fallback due to error" in decision.routing_reason
    with patch("src.llm_router_part1_router.ROUTER_METRICS") as metrics:
        metrics.routing_duration.observe.side_effect = RuntimeError("metrics failed")
        decision = run(router, user_tier="premium")
        assert decision.selected_model == "gpt-a"
        metrics.routing_decisions.labels.assert_called_once_with(
            model="gpt-a", query_type="code_generation"
        )
        assert 0 <= metrics.routing_duration.observe.call_args.args[0] < 1


def test_stats_influence_score_and_bad_updates_do_not_break_requests(router):
    router.update_model_stats("gpt-a", False, 1000)
    router.update_model_stats("gpt-a", True, 2000)
    assert router.model_stats["gpt-a"]["success_rate"] == 0.5
    assert router.model_stats["gpt-a"]["avg_latency"] == 1500
    assert run(router, user_tier="premium").selected_model == "gpt-b"
    before = deepcopy(router.model_stats)
    router.update_model_stats("unknown", True, 0)
    router.update_model_stats("gpt-a", True, -1)
    assert router.model_stats == before
    asyncio.run(router.initialize())
    assert router.model_stats == before


def test_classification_failure_and_zero_cost(router):
    router.token_counter.encoders["mistral"] = lambda text: 10
    with patch.object(router.classifier, "classify_query", side_effect=ValueError("bad classifier")):
        decision = run(router)
    assert decision.query_type == QueryType.GENERAL
    assert decision.confidence == 0.0
    assert decision.estimated_cost == 0.0


def test_context_boundary_and_missing_model_info(router):
    router.token_counter.encoders["gpt"] = lambda text: 100
    assert run(router, user_tier="premium").selected_model == "gpt-a"
    router.token_counter.encoders["gpt"] = lambda text: 101
    assert run(router, user_tier="premium").selected_model == router.default_model
    assert router.get_model_info("missing") is None
    assert router.get_model_info("gpt-a")["config"]["provider"] == "openai"
