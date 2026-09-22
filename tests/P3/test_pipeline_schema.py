"""Tests for P3 pipeline event schemas and factory functions."""

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from uuid import UUID

from src.llm_router_part3_pipeline import (
    build_error_entry,
    build_metric_entries,
    build_query_log_entry,
    build_response_log_entry,
)
from src.utils.schema import (
    InferenceResponse,
    QueryRequest,
    QueryType,
    RoutingDecision,
    UserTier,
)


def test_pipeline_module_imports_from_project_root():
    result = subprocess.run(
        [sys.executable, "-c", "import src.llm_router_part3_pipeline"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _request() -> QueryRequest:
    return QueryRequest(
        request_id="12345678-1234-5678-1234-567812345678",
        query="Explain Kafka ordering",
        user_id="user-42",
        user_tier=UserTier.PREMIUM,
        context="Previous discussion",
        max_tokens=256,
        temperature=0.2,
        metadata={"source": "api"},
    )


def _decision() -> RoutingDecision:
    return RoutingDecision(
        selected_model="gpt-5.6-terra",
        query_type=QueryType.ANALYSIS,
        routing_reason="Best analysis model",
        token_count=37,
        estimated_cost=0.001,
        routing_time_ms=12,
        confidence=0.91,
        routing_strategy="intelligent",
        user_tier=UserTier.PREMIUM,
    )


def _response(*, error=None) -> InferenceResponse:
    return InferenceResponse(
        response_text="Kafka preserves order within a partition.",
        model_name="gpt-5.6-terra",
        provider="openai",
        token_count_input=41,
        token_count_output=9,
        latency_ms=315,
        tokens_per_second=28.5,
        cost_usd=0.0042,
        cached=False,
        compressed_context=True,
        error=error,
    )


def test_build_query_log_entry_maps_request_and_decision_to_json_safe_event():
    entry = build_query_log_entry(
        _request(),
        _decision(),
        datetime(2026, 9, 20, 14, 30, 0),
    )

    assert entry.query_id == UUID("12345678-1234-5678-1234-567812345678")
    assert entry.user_id == "user-42"
    assert entry.user_tier == "premium"
    assert entry.query_text == "Explain Kafka ordering"
    assert entry.query_type == "analysis"
    assert entry.selected_model == "gpt-5.6-terra"
    assert entry.routing_strategy == "intelligent"
    assert entry.routing_confidence == 0.91
    assert entry.token_count_input == 37
    assert entry.temperature == 0.2
    assert entry.max_tokens == 256
    assert entry.has_context is True
    assert entry.has_attachments is False
    assert entry.request_received_at == datetime(2026, 9, 20, 14, 30, tzinfo=timezone.utc)
    assert entry.status == "received"
    assert entry.extra_labels == {}
    assert '"query_id":"12345678-1234-5678-1234-567812345678"' in entry.model_dump_json()
    assert '"request_received_at":"2026-09-20T14:30:00Z"' in entry.model_dump_json()


def test_build_response_log_entry_maps_response_and_normalizes_utc():
    eastern = timezone(timedelta(hours=-4))
    entry = build_response_log_entry(
        _request(),
        _decision(),
        _response(),
        datetime(2026, 9, 20, 10, 30, 1, tzinfo=eastern),
    )

    assert entry.query_id == UUID("12345678-1234-5678-1234-567812345678")
    assert entry.user_id == "user-42"
    assert entry.model_name == "gpt-5.6-terra"
    assert entry.provider == "openai"
    assert entry.token_count_input == 41
    assert entry.token_count_output == 9
    assert entry.total_tokens == 50
    assert entry.cost_usd == 0.0042
    assert entry.latency_ms == 315
    assert entry.routing_time_ms == 12
    assert entry.cached is False
    assert entry.compressed_context is True
    assert entry.error is None
    assert entry.response_completed_at == datetime(2026, 9, 20, 14, 30, 1, tzinfo=timezone.utc)
    assert entry.status == "success"


def test_build_response_log_entry_marks_failed_inference_as_error():
    entry = build_response_log_entry(
        _request(),
        _decision(),
        _response(error="provider unavailable"),
        datetime.now(timezone.utc),
    )

    assert entry.status == "error"
    assert entry.error == "provider unavailable"


def test_build_metric_entries_emits_required_metrics_and_dimensions():
    entries = build_metric_entries(_request(), _decision(), _response())

    assert {entry.metric_name: entry.value for entry in entries} == {
        "request_count": 1.0,
        "tokens_input": 41.0,
        "tokens_output": 9.0,
        "cost_usd": 0.0042,
        "inference_latency_ms": 315.0,
        "routing_time_ms": 12.0,
    }
    assert all(entry.service == "llm-router" for entry in entries)
    assert all(entry.timestamp.tzinfo == timezone.utc for entry in entries)
    assert all(entry.labels == {
        "model": "gpt-5.6-terra",
        "provider": "openai",
        "user_tier": "premium",
        "query_type": "analysis",
        "status": "success",
    } for entry in entries)


def test_build_error_entry_records_uuid_context_and_traceback():
    query_id = UUID("12345678-1234-5678-1234-567812345678")
    try:
        raise ValueError("bad pipeline message")
    except ValueError as exc:
        entry = build_error_entry(
            exc,
            component="pipeline",
            query_id=query_id,
            extra={"topic": "llm-queries"},
        )

    assert isinstance(entry.error_id, UUID)
    assert entry.query_id == query_id
    assert entry.error_type == "ValueError"
    assert entry.error_message == "bad pipeline message"
    assert "ValueError: bad pipeline message" in entry.stacktrace
    assert entry.component == "pipeline"
    assert entry.severity == "error"
    assert entry.timestamp.tzinfo == timezone.utc
    assert entry.extra == {"topic": "llm-queries"}
