# tests/test_p1_setup.py
# P1.md 5.2 verification point 1: the setup command builds the full scaffold.
#
# Everything runs against a throwaway directory, the real project tree is never
# touched. Dependency installation is disabled: 5.3 covers `pip install` and
# creating a venv inside a unit test would take minutes.

import json

import pytest
import yaml

from src.llm_router_part0_setup import (
    CONFIG_REL_PATH,
    PACKAGE_DIRS,
    REQUIRED_DIRS,
    ProjectSetup,
    setup_project_environment,
)

# 5.2 lists these explicitly, README.md is the tenth from task 3.1
TEMPLATE_FILES = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "docker/requirements.txt",
    "kafka/topics.json",
    "clickhouse/schema.sql",
    "monitoring/prometheus.yml",
    "monitoring/grafana/dashboard.json",
    "streamlit_ui/config.toml",
    ".github/workflows/ci.yml",
]


@pytest.fixture(scope="module")
def scaffold(tmp_path_factory):
    """Run the scaffold once into an empty directory and reuse the result."""
    root = tmp_path_factory.mktemp("scaffold")
    setup_project_environment(project_root=str(root), install_deps=False)
    return root


def test_seventeen_directories_exist(scaffold):
    assert len(REQUIRED_DIRS) == 17, "5.2 requires 17 directories"
    missing = [name for name in REQUIRED_DIRS if not (scaffold / name).is_dir()]
    assert not missing, f"directories not created: {missing}"


def test_package_dirs_have_init_files(scaffold):
    assert set(PACKAGE_DIRS) == {"src", "src/models", "src/utils", "tests"}
    missing = [name for name in PACKAGE_DIRS if not (scaffold / name / "__init__.py").is_file()]
    assert not missing, f"__init__.py missing in: {missing}"


def test_ten_template_files_exist_and_are_non_empty(scaffold):
    assert len(TEMPLATE_FILES) == 10, "5.2 requires 10 template files"
    problems = []
    for name in TEMPLATE_FILES:
        path = scaffold / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif path.stat().st_size == 0:
            problems.append(f"{name}: empty")
    assert not problems, problems


def test_config_yaml_exists_and_parses(scaffold):
    path = scaffold / CONFIG_REL_PATH
    assert path.is_file(), f"{CONFIG_REL_PATH} was not generated"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(config, dict) and config, "config.yaml parsed to an empty document"


def test_generated_config_has_every_section_main_py_reads(scaffold):
    """3.5 lists the sections main.py._load_config injects defaults for, plus the
    five that 3.1.4 validates. All of them should already be in the template."""
    config = yaml.safe_load((scaffold / CONFIG_REL_PATH).read_text(encoding="utf-8"))
    expected = [
        "api", "logging", "router", "inference", "kafka", "clickhouse",
        "monitoring", "slack", "streamlit", "flink", "security", "performance",
        "development", "features", "pipeline", "adapters", "policies",
        "optimization", "quality", "router_mode",
    ]
    missing = [name for name in expected if name not in config]
    assert not missing, f"sections missing from the generated config: {missing}"


@pytest.mark.parametrize("name", ["kafka/topics.json", "monitoring/grafana/dashboard.json"])
def test_json_templates_are_valid_json(scaffold, name):
    json.loads((scaffold / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["monitoring/prometheus.yml", ".github/workflows/ci.yml"])
def test_yaml_templates_are_valid_yaml(scaffold, name):
    assert yaml.safe_load((scaffold / name).read_text(encoding="utf-8"))


def test_setup_is_idempotent(scaffold):
    """3.1 core constraint: running setup twice must not raise."""
    before = sorted(p.name for p in scaffold.rglob("*"))
    setup_project_environment(project_root=str(scaffold), install_deps=False)
    after = sorted(p.name for p in scaffold.rglob("*"))
    assert before == after, "second setup run changed the tree"


def test_existing_files_are_not_overwritten(tmp_path):
    """3.1.2 write policy: existing files are skipped, user edits survive."""
    marker = "# edited by hand\n"
    (tmp_path / ".gitignore").write_text(marker, encoding="utf-8")

    setup_project_environment(project_root=str(tmp_path), install_deps=False)

    assert (tmp_path / ".gitignore").read_text(encoding="utf-8") == marker


def test_validation_rejects_a_missing_required_directory(tmp_path):
    """3.1.4: a missing directory must raise FileNotFoundError."""
    setup = ProjectSetup(project_root=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        setup._validate_environment()


def test_validation_rejects_broken_yaml(tmp_path):
    """3.1.4: config.yaml that is not valid YAML must raise."""
    setup = ProjectSetup(project_root=str(tmp_path))
    config_path = tmp_path / CONFIG_REL_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("api:\n  port: [unclosed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid YAML"):
        setup._validate_config()


def test_no_plaintext_secrets_in_templates(scaffold):
    """3.1 core constraint: templates must never carry a real key, only *_env names."""
    banned = ["sk-", "xoxb-", "ghp_", "AKIA"]
    offenders = []
    for name in TEMPLATE_FILES + [CONFIG_REL_PATH]:
        text = (scaffold / name).read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                offenders.append(f"{name}: contains {token!r}")
    assert not offenders, offenders


def test_config_template_matches_the_shipped_config(project_root):
    """The embedded template and config/config.yaml are the same artifact (D6).

    Editing one without the other is exactly how the quality.feedback defaults
    drifted, so this pins them together: a fresh `setup` must reproduce the file
    the project actually ships.
    """
    from src.llm_router_part0_setup import CONFIG_TEMPLATE

    shipped = (project_root / CONFIG_REL_PATH).read_text(encoding="utf-8")

    assert CONFIG_TEMPLATE == shipped, (
        "config/config.yaml and part0_setup.CONFIG_TEMPLATE have diverged, "
        "re-sync the template after editing the config"
    )
