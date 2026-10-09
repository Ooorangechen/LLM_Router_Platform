"""P4 §5.1 step 1: required runtime dependencies are installed in this environment."""

import importlib.util


REQUIRED = [
    "fastapi", "uvicorn", "pydantic", "click", "yaml", "structlog",
    "prometheus_client", "httpx", "psutil", "aiokafka", "clickhouse_connect",
]


def test_p4_dependencies_importable():
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    assert not missing, f"MISSING: {missing}"
