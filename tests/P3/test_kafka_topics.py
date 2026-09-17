"""Topic provisioning tests; Kafka network operations are replaced by a fake broker."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from aiokafka.protocol.admin import CreateTopicsResponse_v0, CreateTopicsResponse_v3

import main


@pytest.fixture
def topic_config(tmp_path):
    metadata = json.loads(Path("kafka/topics.json").read_text())
    path = tmp_path / "topics.json"
    path.write_text(json.dumps(metadata))
    return {"kafka": {"bootstrap_servers": "broker:9092", "topics_file": str(path)}}


@pytest.fixture
def broker(monkeypatch):
    class FakeAdmin:
        existing = set()
        created = []
        instances = []
        error_code = 0
        response_version = 3
        start_error = None

        def __init__(self, **kwargs):
            self.closed = False
            self.options = kwargs
            self.instances.append(self)

        async def start(self):
            if self.start_error:
                raise self.start_error

        async def list_topics(self):
            return list(self.existing)

        async def create_topics(self, new_topics):
            self.created.extend(new_topics)
            if self.error_code in (0, 36):
                self.existing.update(topic.name for topic in new_topics)
            if self.response_version == 0:
                return CreateTopicsResponse_v0(
                    topic_errors=[(topic.name, self.error_code) for topic in new_topics]
                )
            return CreateTopicsResponse_v3(
                throttle_time_ms=0,
                topic_errors=[
                    (topic.name, self.error_code, "broker result") for topic in new_topics
                ],
            )

        async def close(self):
            self.closed = True

    monkeypatch.setattr("aiokafka.admin.AIOKafkaAdminClient", FakeAdmin)
    return FakeAdmin


@pytest.mark.asyncio
async def test_creates_metadata_topics_and_repeated_run_skips_existing(topic_config, broker):
    assert await main._init_kafka_topics(topic_config) == (5, 0)
    assert await main._init_kafka_topics(topic_config) == (0, 5)
    assert [(t.name, t.num_partitions, t.replication_factor) for t in broker.created] == [
        ("llm-queries", 6, 1), ("llm-responses", 6, 1),
        ("llm-metrics", 4, 1), ("llm-errors", 4, 1), ("llm-dead-letter", 2, 1),
    ]
    assert broker.created[0].topic_configs == {
        "retention.ms": "604800000", "cleanup.policy": "delete",
    }
    assert all(instance.closed for instance in broker.instances)


@pytest.mark.asyncio
async def test_partial_existing_and_replication_override(topic_config, broker):
    broker.existing.add("llm-queries")
    topic_config["kafka"]["replication_factor"] = 3
    assert await main._init_kafka_topics(topic_config) == (4, 1)
    assert {topic.replication_factor for topic in broker.created} == {3}
    assert "llm-queries" not in {topic.name for topic in broker.created}


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [0, 3])
async def test_concurrent_creation_already_exists_is_success(topic_config, broker, version):
    broker.error_code = 36
    broker.response_version = version
    assert await main._init_kafka_topics(topic_config) == (0, 5)


@pytest.mark.asyncio
async def test_broker_error_is_not_reported_as_success(topic_config, broker):
    broker.error_code = 29  # TopicAuthorizationFailed
    with pytest.raises(RuntimeError, match="llm-queries"):
        await main._init_kafka_topics(topic_config)
    assert broker.instances[0].closed


@pytest.mark.asyncio
async def test_connection_failure_closes_client(topic_config, broker):
    broker.start_error = ConnectionError("broker unavailable")
    with pytest.raises(ConnectionError, match="broker unavailable"):
        await main._init_kafka_topics(topic_config)
    assert broker.instances[0].closed


def test_cli_success_and_failure_exit_codes(topic_config, broker, monkeypatch):
    class ConfigOnlyPlatform:
        def __init__(self, config_path):
            self.config = topic_config

    monkeypatch.setattr(main, "LLMRouterPlatform", ConfigOnlyPlatform)
    runner = CliRunner()
    result = runner.invoke(main.cli, ["init-kafka-topics"])
    assert result.exit_code == 0, result.output
    assert "Created 5 topics" in result.output
    broker.existing.clear()
    broker.start_error = ConnectionError("broker unavailable")
    result = runner.invoke(main.cli, ["init-kafka-topics"])
    assert result.exit_code == 1
    assert "broker unavailable" in result.output
