import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import src.llm_router_part3_pipeline as pipeline


def _config(tmp_path, *, batch_size=2, retry_max=3):
    return {
        "pipeline": {
            "enabled": True,
            "dlq_local_dir": str(tmp_path / "dlq"),
        },
        "kafka": {
            "bootstrap_servers": "broker:9092",
            "topics": {
                "queries": "llm-queries",
                "responses": "llm-responses",
                "metrics": "llm-metrics",
                "errors": "llm-errors",
                "dead_letter": "llm-dead-letter",
            },
            "consumer": {
                "group_id": "test-consumer",
                "auto_offset_reset": "earliest",
                "max_poll_records": 10,
                "max_poll_interval_ms": 1000,
                "fetch_min_bytes": 1,
                "fetch_max_wait_ms": 1,
            },
        },
        "clickhouse": {
            "schema_file": str(tmp_path / "schema.sql"),
            "batch_size": batch_size,
            "retry_max": retry_max,
            "retry_backoff_base_ms": 0,
        },
    }


class RecordingWriter:
    def __init__(self):
        self.enabled = True
        self.rows = []
        self.dlq = []

    async def buffer_write(self, table, row):
        self.rows.append((table, row))

    async def _write_to_dlq(self, table, rows, reason):
        self.dlq.append((table, rows, reason))


@pytest.mark.asyncio
async def test_consumer_initializes_starts_and_stops(monkeypatch, tmp_path):
    instances = []

    class FakeConsumer:
        def __init__(self, *topics, **options):
            self.topics = topics
            self.options = options
            self.started = False
            self.stopped = False
            instances.append(self)

        async def start(self):
            self.started = True

        async def stop(self):
            self.stopped = True

    monkeypatch.setattr(pipeline, "AIOKafkaConsumer", FakeConsumer)
    engine = pipeline.KafkaConsumerEngine(_config(tmp_path), RecordingWriter())
    engine._consume_loop = AsyncMock()

    await engine.initialize()
    await engine.start()
    await engine._consume_task
    await engine.stop()

    assert instances[0].started is True
    assert instances[0].stopped is True
    assert instances[0].options["enable_auto_commit"] is False
    assert engine.consumer is None
    assert engine.running is False


@pytest.mark.asyncio
async def test_consumer_merges_query_response_and_routes_metrics(tmp_path):
    writer = RecordingWriter()
    engine = pipeline.KafkaConsumerEngine(_config(tmp_path), writer)
    query = {
        "query_id": "12345678-1234-5678-1234-567812345678",
        "user_id": "u1",
        "user_tier": "premium",
        "query_text": "hello",
        "query_type": "general",
        "selected_model": "model-a",
        "routing_strategy": "intelligent",
        "routing_confidence": 0.9,
        "token_count_input": 4,
        "temperature": 0.2,
        "max_tokens": 64,
        "has_context": False,
        "has_attachments": False,
        "request_received_at": "2026-09-22T12:00:00Z",
        "status": "received",
        "extra_labels": {},
    }
    response = {
        "query_id": query["query_id"],
        "user_id": "u1",
        "model_name": "model-a",
        "provider": "openai",
        "response_text": "world",
        "token_count_input": 5,
        "token_count_output": 2,
        "total_tokens": 7,
        "cost_usd": 0.01,
        "latency_ms": 20,
        "routing_time_ms": 2,
        "cached": False,
        "compressed_context": False,
        "error": None,
        "response_completed_at": "2026-09-22T12:00:01Z",
        "status": "success",
    }
    metric = {
        "timestamp": "2026-09-22T12:00:01Z",
        "service": "llm-router",
        "metric_name": "request_count",
        "value": 1.0,
        "labels": {"model": "model-a"},
    }

    assert await engine._handle_message(
        "llm-queries", json.dumps(query).encode()) is True
    assert writer.rows == []
    assert await engine._handle_message(
        "llm-responses", json.dumps(response).encode()) is True
    assert await engine._handle_message(
        "llm-metrics", json.dumps(metric).encode()) is True

    table, merged = writer.rows[0]
    assert table == "query_logs"
    assert merged["query_id"] == query["query_id"]
    assert merged["token_count_input"] == 5
    assert merged["status"] == "success"
    assert "model_name" not in merged
    assert "extra_labels" not in merged
    assert writer.rows[1] == ("system_metrics", metric)


@pytest.mark.asyncio
async def test_consumer_invalid_json_goes_to_local_dlq(tmp_path):
    writer = RecordingWriter()
    engine = pipeline.KafkaConsumerEngine(_config(tmp_path), writer)

    assert await engine._handle_message("llm-metrics", b"not-json") is False
    assert writer.dlq[0][0] == "kafka"
    assert writer.dlq[0][1][0]["topic"] == "llm-metrics"


@pytest.mark.asyncio
async def test_writer_buffers_flushes_and_retries(tmp_path):
    writer = pipeline.ClickHouseWriter(_config(tmp_path))

    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.inserted = None

        def insert(self, table, data, column_names):
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("temporary")
            self.inserted = (table, data, column_names)

    writer.client = FakeClient()
    await writer.buffer_write("system_metrics", {"metric_name": "a", "value": 1})
    await writer.buffer_write("system_metrics", {"metric_name": "b", "value": 2})

    assert writer.client.calls == 3
    assert writer.client.inserted == (
        "system_metrics",
        [["a", 1], ["b", 2]],
        ["metric_name", "value"],
    )
    assert writer._buffers["system_metrics"] == []


@pytest.mark.asyncio
async def test_writer_failed_batch_is_persisted_and_replayed(tmp_path):
    writer = pipeline.ClickHouseWriter(_config(tmp_path, batch_size=10, retry_max=1))
    writer.client = SimpleNamespace(insert=lambda *args, **kwargs: None)
    writer._execute_insert = AsyncMock(side_effect=ConnectionError("down"))
    await writer.buffer_write("system_metrics", {"metric_name": "a", "value": 1})

    assert await writer.flush_table("system_metrics") == (0, 1)
    dlq_file = next((tmp_path / "dlq").glob("*.jsonl"))
    record = json.loads(dlq_file.read_text().strip())
    assert record["table"] == "system_metrics"
    assert record["reason"] == "down"

    writer._execute_insert = AsyncMock(return_value=None)
    assert await writer.replay_dlq() == (1, 0)
    assert not dlq_file.exists()
    assert (tmp_path / "dlq" / "archived" / dlq_file.name).exists()


@pytest.mark.asyncio
async def test_writer_initialize_applies_schema_and_shutdown(monkeypatch, tmp_path):
    schema = tmp_path / "schema.sql"
    schema.write_text(
        "-- comment\nCREATE TABLE a (x UInt8);\nCREATE TABLE b (x UInt8);"
    )
    config = _config(tmp_path)
    commands = []

    class FakeClient:
        def command(self, sql):
            commands.append(sql)

        def close(self):
            commands.append("closed")

    import clickhouse_connect
    monkeypatch.setattr(clickhouse_connect, "get_client", lambda **kwargs: FakeClient())
    writer = pipeline.ClickHouseWriter(config)

    await writer.initialize()
    await writer.shutdown()

    assert commands == [
        "CREATE TABLE a (x UInt8)",
        "CREATE TABLE b (x UInt8)",
        "SELECT 1",
        "closed",
    ] or commands == [
        "SELECT 1",
        "CREATE TABLE a (x UInt8)",
        "CREATE TABLE b (x UInt8)",
        "closed",
    ]
    assert writer.enabled is False
    assert writer.client is None
