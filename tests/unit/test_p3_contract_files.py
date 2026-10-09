"""P3 cross-file contracts: topics.json, constants.py, schema.sql and the rows the consumer writes.

A mismatch here raises nothing at runtime: an extra key fails the whole
JSONEachRow batch into the DLQ, and a missing key is silently filled with the
column default. These tests compare the files directly.
"""

import json
import re
from datetime import datetime, timezone

import pytest

from src.llm_router_part3_pipeline import (
    ClickHouseWriter,
    KafkaConsumerEngine,
    KafkaProducerManager,
    build_error_entry,
    build_metric_entries,
    build_query_log_entry,
    build_response_log_entry,
)
from src.utils.constants import ClickHouseTables, KafkaTopics
from src.utils.schema import InferenceResponse, QueryRequest, QueryType, RoutingDecision


def _schema_sql(project_root):
    return (project_root / "clickhouse/schema.sql").read_text(encoding="utf-8")


def _insertable_columns(schema_sql):
    """{table: set of columns an INSERT may name}; MATERIALIZED/ALIAS columns are excluded."""
    tables = {}
    for name, body in re.findall(
            r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\)\s*ENGINE", schema_sql, re.S):
        tables[name] = {
            line.split()[0]
            for line in (raw.strip() for raw in body.splitlines())
            if line and not line.startswith("--")
            and "MATERIALIZED" not in line and "ALIAS" not in line
        }
    return tables


@pytest.fixture
def events():
    request = QueryRequest(
        query="Write a quick sort", user_id="contract-user", user_tier="premium",
        context="some context", max_tokens=64, temperature=0.2,
    )
    decision = RoutingDecision(
        selected_model="gpt-a", query_type=QueryType.CODE_GENERATION,
        routing_reason="rule", token_count=5, estimated_cost=0.001,
        routing_time_ms=3, confidence=0.8,
    )
    response = InferenceResponse(
        response_text="def quick_sort(xs): ...", model_name="gpt-a", provider="openai",
        token_count_input=5, token_count_output=7, latency_ms=120,
        tokens_per_second=58.0, cost_usd=0.0004,
    )
    now = datetime.now(timezone.utc)
    return request, decision, response, now


@pytest.fixture
def consumer(tmp_path):
    config = {"pipeline": {"enabled": True, "dlq_local_dir": str(tmp_path / "dlq")}}

    class Writer(ClickHouseWriter):
        async def buffer_write(self, table, row):
            self.rows.append((table, row))
            return (0, 0)

    writer = Writer(config)
    writer.rows = []
    engine = KafkaConsumerEngine(config, writer, KafkaProducerManager(config))
    engine._current = (None, 0)  # set by the poll loop; the query half records it
    return engine


def _rows_for(consumer, table):
    return [row for name, row in consumer.clickhouse_writer.rows if name == table]


def test_topics_json_matches_kafka_topics_enum(project_root):
    data = json.loads((project_root / "kafka/topics.json").read_text(encoding="utf-8"))
    assert {topic["name"] for topic in data["topics"]} == {t.value for t in KafkaTopics}


def test_every_clickhouse_table_constant_is_created_by_schema_sql(project_root):
    created = set(re.findall(
        r"CREATE (?:TABLE|VIEW|MATERIALIZED VIEW) IF NOT EXISTS (\w+)", _schema_sql(project_root)))
    missing = {t.value for t in ClickHouseTables} - created
    assert not missing, f"schema.sql does not create: {missing}"


@pytest.mark.asyncio
async def test_query_and_response_merge_into_query_logs_and_user_analytics_columns(
        consumer, events, project_root):
    request, decision, response, now = events
    columns = _insertable_columns(_schema_sql(project_root))

    query = build_query_log_entry(request, decision, now)
    reply = build_response_log_entry(request, decision, response, now)
    assert await consumer._handle_message(
        KafkaTopics.QUERIES.value, query.model_dump_json().encode())
    assert consumer.clickhouse_writer.rows == []  # half a row waits for its partner
    assert await consumer._handle_message(
        KafkaTopics.RESPONSES.value, reply.model_dump_json().encode())

    [query_row] = _rows_for(consumer, ClickHouseTables.QUERY_LOGS.value)
    [user_row] = _rows_for(consumer, ClickHouseTables.USER_ANALYTICS.value)
    assert set(query_row) == columns[ClickHouseTables.QUERY_LOGS.value]
    assert set(user_row) == columns[ClickHouseTables.USER_ANALYTICS.value]


@pytest.mark.asyncio
async def test_metric_and_error_events_match_system_metrics_columns(
        consumer, events, project_root):
    request, decision, response, _ = events
    expected = _insertable_columns(_schema_sql(project_root))[
        ClickHouseTables.SYSTEM_METRICS.value]

    for entry in build_metric_entries(request, decision, response):
        assert await consumer._handle_message(
            KafkaTopics.METRICS.value, entry.model_dump_json().encode())
    error = build_error_entry(
        RuntimeError("boom"), "inference", request.request_id, {"model": "gpt-a"})
    assert await consumer._handle_message(
        KafkaTopics.ERRORS.value, error.model_dump_json().encode())

    rows = _rows_for(consumer, ClickHouseTables.SYSTEM_METRICS.value)
    assert len(rows) >= 2
    for row in rows:
        assert set(row) == expected, row["metric_name"]
