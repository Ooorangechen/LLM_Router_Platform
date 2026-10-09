"""P3 task 3.7: canonical config.yaml deep-merged with a partial custom override."""

import inspect

import yaml

import main
from src.llm_router_part0_setup import CONFIG_REL_PATH


def test_canonical_config_and_partial_override_contract(monkeypatch, tmp_path, project_root):
    canonical = yaml.safe_load((project_root / CONFIG_REL_PATH).read_text(encoding="utf-8"))
    assert canonical["router"]["routing_strategy"] == "intelligent"
    assert canonical["inference"]["compression"]["max_context_tokens"] == 100000
    assert canonical["inference"]["batching"]["enabled"] is True
    assert canonical["pipeline"]["async_publish"] is True
    assert canonical["kafka"]["consumer"]["enable_auto_commit"] is False
    assert canonical["kafka"]["producer"]["max_in_flight"] == 5
    assert canonical["clickhouse"]["password_env"] == "CLICKHOUSE_PASSWORD"
    assert "password" not in canonical["clickhouse"]
    assert not (project_root / "config/defaults.yaml").exists()

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
