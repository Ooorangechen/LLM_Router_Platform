"""P1 task 3.1: ProjectSetup scaffolding and generated templates."""

import json

import yaml

from src.llm_router_part0_setup import CONFIG_REL_PATH, CONFIG_TEMPLATE, ProjectSetup


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
