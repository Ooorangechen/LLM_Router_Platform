"""Shared test setup.

Layout:
  unit/         no network, docker or API keys; runs by default
  integration/  real Redis / Kafka / ClickHouse / provider APIs
  acceptance/   docs/P*.md §5 acceptance checks

integration/ and acceptance/ are skipped unless --run-external is given:
  venv/bin/python -m pytest                    # unit only
  venv/bin/python -m pytest --run-external     # everything
  venv/bin/python -m pytest --run-external -m "not real_provider"   # skip paid API calls
"""

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIRS = {"integration", "acceptance"}


def pytest_addoption(parser):
    parser.addoption(
        "--run-external", action="store_true",
        help="run integration/ and acceptance/ tests (may need docker, API keys, cost money)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_provider: calls real OpenAI/Anthropic APIs (costs money)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-external"):
        return
    skip = pytest.mark.skip(reason="needs --run-external")
    for item in items:
        if EXTERNAL_DIRS & set(item.path.relative_to(ROOT).parts):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def app_config():
    """Canonical config/config.yaml, with root .env loaded without overriding the shell."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    return yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
