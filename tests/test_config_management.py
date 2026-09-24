"""Focused contracts for the canonical configuration design."""

import inspect
import json
from pathlib import Path

import pytest
import yaml

import main
from src.llm_router_part0_setup import (
    CONFIG_REL_PATH,
    CONFIG_TEMPLATE,
    ProjectSetup,
)
from src.llm_router_part3_pipeline import (
    ClickHouseWriter,
    KafkaConsumerEngine,
    KafkaProducerManager,
)
from src.utils.constants import ClickHouseTables, KafkaTopics


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_config_and_partial_override_contract(monkeypatch, tmp_path):
    canonical = yaml.safe_load((ROOT / CONFIG_REL_PATH).read_text(encoding="utf-8"))
    assert canonical["router"]["routing_strategy"] == "intelligent"
    assert canonical["inference"]["compression"]["max_context_tokens"] == 100000
    assert canonical["inference"]["batching"]["enabled"] is True
    assert canonical["pipeline"]["async_publish"] is True
    assert canonical["kafka"]["consumer"]["enable_auto_commit"] is False
    assert canonical["kafka"]["producer"]["max_in_flight"] == 5
    assert canonical["clickhouse"]["password_env"] == "CLICKHOUSE_PASSWORD"
    assert "password" not in canonical["clickhouse"]
    assert not (ROOT / "config/defaults.yaml").exists()

    override = tmp_path / "override.yaml"
    override.write_text(
        "api:\n  port: 9099\n  cors_origins: [https://example.test]\n"
        "router:\n  default_model: null\n",
        encoding="utf-8",
    )
    platform = main.LLMRouterPlatform(override)
    assert platform.config["api"] == {
        **canonical["api"],
        "port": 9099,
        "cors_origins": ["https://example.test"],
    }
    assert platform.config["router"]["default_model"] is None
    assert platform.config["router"]["models"] == canonical["router"]["models"]
    assert set(inspect.signature(main.LLMRouterPlatform).parameters) == {"config_path"}

    monkeypatch.chdir(tmp_path)
    assert main.LLMRouterPlatform().config["api"]["port"] == 8080


def test_setup_generates_valid_templates_without_a_defaults_file(tmp_path):
    setup = ProjectSetup(project_root=str(tmp_path))
    setup.setup_project_environment(install_deps=False)

    generated_path = tmp_path / CONFIG_REL_PATH
    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
    assert generated_path.read_text(encoding="utf-8") == CONFIG_TEMPLATE
    assert not (tmp_path / "config/defaults.yaml").exists()
    json.loads((tmp_path / "kafka/topics.json").read_text(encoding="utf-8"))
    json.loads((tmp_path / "monitoring/grafana/dashboard.json").read_text(encoding="utf-8"))
    yaml.safe_load((tmp_path / "monitoring/prometheus.yml").read_text(encoding="utf-8"))
    yaml.safe_load((tmp_path / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.9/3.10: syntax check remains optional.
        tomllib = None
    if tomllib is not None:
        tomllib.loads(
            (tmp_path / "streamlit_ui/config.toml").read_text(encoding="utf-8"))
    assert "CREATE TABLE" in (tmp_path / "clickhouse/schema.sql").read_text(encoding="utf-8")

    requirements = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
    assert (tmp_path / "docker/requirements.txt").read_text(encoding="utf-8") == requirements
    for dependency in ("python-dotenv", "tiktoken", "openai", "anthropic", "tenacity"):
        assert dependency in requirements

    edited = {**generated, "api": {**generated["api"], "port": 7777}}
    generated_path.write_text(
        yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")
    setup.setup_project_environment(install_deps=False)
    assert yaml.safe_load(generated_path.read_text(encoding="utf-8"))["api"]["port"] == 7777


@pytest.mark.parametrize("text", ["- not-a-mapping\n", "api: [unclosed\n"])
def test_custom_config_rejects_invalid_yaml_documents(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit):
        main.LLMRouterPlatform(path)


@pytest.mark.asyncio
async def test_pipeline_resolves_configured_topics_tables_and_password(monkeypatch, tmp_path):
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
            "topics": {
                "queries": "tenant-queries", "responses": "tenant-responses",
                "metrics": "tenant-metrics", "errors": "tenant-errors",
                "dead_letter": "tenant-dlq",
            },
            "producer": {"max_in_flight": 4},
            "consumer": {},
        },
        "clickhouse": {
            "password_env": "TEST_CLICKHOUSE_PASSWORD",
            "schema_file": str(schema_path),
            "tables": {
                "query_logs": "tenant_query_logs",
                "system_metrics": "tenant_system_metrics",
                "model_performance": "tenant_model_performance",
                "user_analytics": "tenant_user_analytics",
            },
        },
    }

    class Writer(ClickHouseWriter):
        async def buffer_write(self, table, row):
            self.recorded = (table, row)

    writer = Writer(config)
    producer = KafkaProducerManager(config)
    consumer = KafkaConsumerEngine(config, writer)
    result = await consumer._handle_message(
        "tenant-metrics",
        b'{"timestamp":"2026-09-24T12:00:00Z","metric_name":"requests","value":1}',
    )

    assert KafkaTopics.QUERIES.value == "llm-queries"
    assert ClickHouseTables.QUERY_LOGS.value == "query_logs"
    assert producer.topics["dead_letter"] == "tenant-dlq"
    assert producer.max_in_flight == 4
    assert consumer.topics == [
        "tenant-queries", "tenant-responses", "tenant-metrics", "tenant-errors",
    ]
    assert writer.password == "resolved-secret"
    assert result == "tenant_system_metrics"
    assert writer.recorded[0] == "tenant_system_metrics"

    commands = []
    writer.client = type("Client", (), {"command": lambda self, sql: commands.append(sql)})()
    await writer._create_tables_if_not_exists()
    assert commands == [
        "CREATE TABLE tenant_query_logs (id UInt8)",
        "CREATE TABLE tenant_system_metrics (id UInt8)",
    ]
