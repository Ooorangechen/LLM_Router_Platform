# tests/test_p1_schema.py
# P1.md 5.8 verification point 7: the pydantic layer actually rejects bad input.
#
# Each V7 check below asserts on the *field* that failed, not merely that some
# ValidationError was raised. QueryRequest declares session_id and conversation_id
# without defaults, so a call that omits them raises no matter what the other
# arguments look like, and a bare pytest.raises would pass for the wrong reason.

import pytest
from pydantic import ValidationError

from src.utils.schema import (
    ModelConfig,
    QueryRequest,
    QueryType,
    SystemMetric,
    UserTier,
)

# the two metadata fields QueryRequest requires but 5.8's snippet omits
QUERY_METADATA = {"session_id": "s1", "conversation_id": "c1"}


def failed_fields(exc_info) -> set:
    return {error["loc"][0] for error in exc_info.value.errors()}


def test_v7_1_blank_query_is_rejected():
    """5.8 V7-1: a whitespace only query must not pass."""
    with pytest.raises(ValidationError) as exc_info:
        QueryRequest(query="   ", user_id="u1", **QUERY_METADATA)

    assert "query" in failed_fields(exc_info)


def test_query_is_stripped_when_valid():
    """3.2.3: the validator strips surrounding whitespace rather than rejecting it."""
    request = QueryRequest(query="  hello  ", user_id="u1", **QUERY_METADATA)

    assert request.query == "hello"


def test_v7_2_unknown_tier_is_rejected():
    """5.8 V7-2: user_tier only accepts the UserTier values."""
    with pytest.raises(ValidationError) as exc_info:
        QueryRequest(query="hi", user_id="u1", user_tier="bad-tier", **QUERY_METADATA)

    assert "user_tier" in failed_fields(exc_info)


def test_v7_3_unknown_capability_is_rejected():
    """5.8 V7-3."""
    with pytest.raises(ValidationError) as exc_info:
        ModelConfig(
            name="m", provider="p", max_tokens=100, cost_per_token=0,
            priority=1, capabilities=["bad-cap"],
        )

    assert "capabilities" in failed_fields(exc_info)


def test_v7_4_metric_name_charset_is_enforced():
    """5.8 V7-4: metric names are alphanumeric plus _ and . only."""
    with pytest.raises(ValidationError) as exc_info:
        SystemMetric(name="metric name with space!!!", value=1.0, labels={})

    assert "name" in failed_fields(exc_info)


def test_valid_metric_name_is_accepted():
    metric = SystemMetric(name="llm_router.requests_total", value=1.0, labels={"a": "b"})

    assert metric.name == "llm_router.requests_total"


def test_v7_5_zero_max_tokens_is_rejected():
    """5.8 V7-5: max_tokens has a ge=1 bound."""
    with pytest.raises(ValidationError) as exc_info:
        ModelConfig(
            name="m", provider="p", max_tokens=0, cost_per_token=0,
            priority=1, capabilities=["general"],
        )

    assert "max_tokens" in failed_fields(exc_info)


def test_v7_6_query_type_enum_is_complete():
    """5.8 V7-6: at least 12 query types."""
    assert len(list(QueryType)) >= 12


def test_enum_values_are_lowercase():
    """3.2 core constraint: every enum value is lower case for clean JSON."""
    for enum_cls in (UserTier, QueryType):
        for member in enum_cls:
            assert member.value == member.value.lower(), member


def test_valid_query_request_round_trips():
    """A well formed request keeps its defaults, so the rejections above are not
    the model refusing everything."""
    request = QueryRequest(query="hello", user_id="u1", **QUERY_METADATA)

    assert request.user_tier is UserTier.FREE
    assert request.max_tokens == 512
    assert request.temperature == pytest.approx(0.7)
    assert request.priority == 1
    assert request.attachments == []
    assert request.request_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", 0),
        ("max_tokens", 8193),
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("priority", 0),
        ("priority", 6),
    ],
)
def test_numeric_bounds_are_enforced(field, value):
    """3.2.3: the documented ranges for max_tokens, temperature and priority."""
    with pytest.raises(ValidationError) as exc_info:
        QueryRequest(query="hi", user_id="u1", **{field: value}, **QUERY_METADATA)

    assert field in failed_fields(exc_info)


def test_model_config_accepts_every_documented_capability():
    """3.2.6: the valid capability set."""
    capabilities = [
        "reasoning", "coding", "analysis", "writing",
        "creative", "general", "math", "translation",
    ]
    config = ModelConfig(
        name="m", provider="p", max_tokens=100, cost_per_token=0.0,
        priority=1, capabilities=capabilities,
    )

    assert config.capabilities == capabilities
