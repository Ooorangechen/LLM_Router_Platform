"""P3 tasks 3.1/3.3/3.4: pipeline components resolve topics, tables and secrets.

Topic names come only from kafka/topics.json and table names only from
src/utils/constants.py (P3 §3.1 single source); kafka.topics and
clickhouse.tables in config are not read.
"""

import pytest

from src.llm_router_part3_pipeline import (
    ClickHouseWriter,
    KafkaConsumerEngine,
    KafkaProducerManager,
)
from src.utils.constants import ClickHouseTables, KafkaTopics


@pytest.mark.asyncio
async def test_pipeline_resolves_topics_file_tables_and_password(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_CLICKHOUSE_PASSWORD", "resolved-secret")
    schema_path = tmp_path / "schema.sql"
    schema_path.write_text(
        "CREATE TABLE query_logs (id UInt8);\n"
        "CREATE TABLE system_metrics (id UInt8);\n",
        encoding="utf-8",
    )
    config = {
        "pipeline": {"enabled": True, "dlq_local_dir": str(tmp_path / "dlq")},
        "kafka": {
            "producer": {"max_in_flight": 4},
            "consumer": {},
        },
        "clickhouse": {
            "password_env": "TEST_CLICKHOUSE_PASSWORD",
            "schema_file": str(schema_path),
        },
    }

    class Writer(ClickHouseWriter):
        async def buffer_write(self, table, row):
            self.recorded = (table, row)
            return (0, 0)

    writer = Writer(config)
    producer = KafkaProducerManager(config)
    consumer = KafkaConsumerEngine(config, writer, producer)
    result = await consumer._handle_message(
        KafkaTopics.METRICS.value,
        b'{"timestamp":"2026-09-24T12:00:00Z","metric_name":"requests","value":1}',
    )

    assert set(producer.topics) == {topic.value for topic in KafkaTopics}
    assert producer.max_in_flight == 4
    # The dead-letter topic is produced to but never consumed, or failures would loop.
    assert consumer.topics == [
        KafkaTopics.QUERIES.value, KafkaTopics.RESPONSES.value,
        KafkaTopics.METRICS.value, KafkaTopics.ERRORS.value,
    ]
    assert writer.password == "resolved-secret"
    assert result is True
    assert writer.recorded[0] == ClickHouseTables.SYSTEM_METRICS.value

    commands = []
    writer.client = type("Client", (), {"command": lambda self, sql: commands.append(sql)})()
    await writer._create_tables_if_not_exists()
    assert commands == [
        "CREATE TABLE query_logs (id UInt8)",
        "CREATE TABLE system_metrics (id UInt8)",
    ]
