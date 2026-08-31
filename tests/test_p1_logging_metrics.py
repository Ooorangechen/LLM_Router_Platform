# tests/test_p1_logging_metrics.py
# P1.md 5.9 verification point 8: logger writes to both channels and the four
# metric singletons can be incremented without raising.

import json
import logging

import pytest

from src.utils.logger import NAMESPACE, get_logger, setup_logging
from src.utils.metrics import (
    INFERENCE_METRICS,
    PIPELINE_METRICS,
    ROUTER_METRICS,
    SYSTEM_METRICS,
)


@pytest.fixture
def isolated_log(tmp_path):
    """Point the logging stack at a temp file, then restore the platform config.

    setup_logging reconfigures the shared "llm_router" logger, so the teardown
    puts the default configuration back for the rest of the session.
    """
    log_file = tmp_path / "llm_router.log"
    setup_logging(
        log_level="DEBUG",
        log_file=str(log_file),
        console_output=True,
        structured_logs=True,
    )
    yield log_file

    for handler in list(logging.getLogger(NAMESPACE).handlers):
        handler.close()
    setup_logging()


def test_text_channel_receives_records(isolated_log):
    """5.9: the plain text log file has content after logging."""
    log = get_logger(__name__)
    log.info("P1 logger test message")
    log.warning("P1 logger warning")
    logging.getLogger(NAMESPACE).handlers[0].flush()

    assert isolated_log.exists()
    assert isolated_log.stat().st_size > 0

    text = isolated_log.read_text(encoding="utf-8")
    assert "P1 logger test message" in text
    assert "P1 logger warning" in text


def test_json_channel_receives_parseable_records(isolated_log):
    """3.3: the structured channel writes one JSON object per line."""
    get_logger(__name__).info("structured channel check")
    for handler in logging.getLogger(NAMESPACE).handlers:
        handler.flush()

    json_log = isolated_log.with_suffix(isolated_log.suffix + ".jsonl")
    assert json_log.exists(), "the .jsonl channel was not created"

    records = [json.loads(line) for line in json_log.read_text(encoding="utf-8").splitlines() if line]
    assert records, "no structured records were written"
    assert any(record.get("message") == "structured channel check" for record in records)


def test_logger_names_are_namespaced(isolated_log):
    """3.3: get_logger returns a child of the llm_router namespace."""
    assert get_logger("some.module").name == f"{NAMESPACE}.some.module"


def test_namespace_logger_does_not_propagate(isolated_log):
    """3.3: propagate is off so records are not emitted twice."""
    assert logging.getLogger(NAMESPACE).propagate is False


@pytest.mark.parametrize(
    "metric,labels",
    [
        ("SYSTEM_METRICS", {"endpoint": "/health", "method": "GET", "status": "200"}),
        ("ROUTER_METRICS", {"model": "m", "query_type": "general"}),
        ("INFERENCE_METRICS", {"model": "m", "provider": "p"}),
        ("PIPELINE_METRICS", {"topic": "t"}),
    ],
)
def test_counter_increment_does_not_raise(metric, labels):
    """5.9: incrementing one counter on each of the four singletons works."""
    holders = {
        "SYSTEM_METRICS": (SYSTEM_METRICS, "requests_total"),
        "ROUTER_METRICS": (ROUTER_METRICS, "routing_decisions"),
        "INFERENCE_METRICS": (INFERENCE_METRICS, "requests_total"),
        "PIPELINE_METRICS": (PIPELINE_METRICS, "messages_produced"),
    }
    holder, attribute = holders[metric]

    counter = getattr(holder, attribute)
    counter.labels(**labels).inc()

    assert counter.labels(**labels)._value.get() >= 1


def test_metric_names_follow_the_naming_rules():
    """3.4 core constraint: llm_router_ prefix, snake_case, Counter ends in _total.

    Read straight from the source: prometheus_client strips the _total suffix off a
    Counter's family name, so the registry cannot tell you how it was declared.
    """
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src" / "utils" / "metrics.py"
    text = source.read_text(encoding="utf-8")

    declared = re.findall(r"(Counter|Gauge|Histogram)\(\s*\"([^\"]+)\"", text)
    assert declared, "no metrics found in metrics.py"

    offenders = []
    for kind, name in declared:
        if not name.startswith("llm_router"):
            offenders.append(f"{name}: missing llm_router prefix")
        if name != name.lower():
            offenders.append(f"{name}: not snake_case")
        if kind == "Counter" and not name.endswith("_total"):
            offenders.append(f"{name}: Counter must end with _total")

    assert not offenders, offenders


def test_all_four_singletons_are_distinct_instances():
    """3.4: four singletons created once at import, shared by every module."""
    instances = [SYSTEM_METRICS, ROUTER_METRICS, INFERENCE_METRICS, PIPELINE_METRICS]

    assert len({id(instance) for instance in instances}) == 4
