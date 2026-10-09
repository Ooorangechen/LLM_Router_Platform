"""P4 M4 / task 3.5: AlertManager firing, dedup, resolve, history, suppression and notifiers.

Rules are driven through _evaluate_rule / _record_event directly so no eval loop or sleep runs.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.llm_router_part4_monitor import AlertManager, AlertRule


pytestmark = pytest.mark.asyncio

INTERVAL = 10


def _manager(services=None, **alert_manager_cfg):
    config = {"monitoring": {
        "alert_enabled": True,
        "prometheus": {"server_url": "http://127.0.0.1:9"},  # nothing listens; refused at once
        "alert_manager": {
            "eval_interval_sec": INTERVAL,
            "notifiers": {"stdout": {"enabled": False}},
            **alert_manager_cfg,
        },
    }}
    return AlertManager(config, services if services is not None else {})


def _rule(duration_seconds=30, name="HighMemoryPercent", threshold=90.0):
    # duration 30s / interval 10s -> 3 consecutive hits to fire, 3 misses to resolve.
    return AlertRule(name=name, expr_lambda_src="", threshold=threshold,
                     duration_seconds=duration_seconds, severity="warning",
                     description="test rule", enabled=True)


def _feed(manager, rule, pattern):
    """Record a sequence of hits (True) / misses (False); return the non-None results."""
    results = [manager._record_event(rule, hit, 95.0 if hit else 10.0, {}) for hit in pattern]
    return [r for r in results if r is not None]


# ---------- rules ----------

async def test_initialize_loads_four_default_rules():
    manager = _manager()
    await manager.initialize()

    assert {(r.name, r.threshold, r.severity) for r in manager.rules} == {
        ("HighErrorRate", 0.05, "critical"),
        ("HighLatencyP95Sec", 5.0, "warning"),
        ("HighMemoryPercent", 90.0, "warning"),
        ("HighDiskPercent", 80.0, "warning"),
    }


# ---------- firing / dedup / resolve ----------

async def test_fires_only_after_duration_worth_of_consecutive_hits():
    manager, rule = _manager(), _rule()

    assert _feed(manager, rule, [True, True]) == []
    [(record, action)] = _feed(manager, rule, [True])

    assert action == "firing"
    assert record.status == "firing"
    assert record.rule_name == rule.name and record.value == 95.0
    assert manager.get_active_alerts() == [record]


async def test_a_single_miss_resets_the_hit_streak():
    manager, rule = _manager(), _rule()

    assert _feed(manager, rule, [True, True, False, True, True]) == []
    assert manager.get_active_alerts() == []


async def test_repeated_hits_while_firing_are_deduplicated():
    manager, rule = _manager(), _rule()
    _feed(manager, rule, [True] * 3)

    assert _feed(manager, rule, [True] * 100) == []
    assert len(manager.get_active_alerts()) == 1
    assert len(manager.get_history()) == 1


async def test_resolves_after_duration_worth_of_consecutive_misses():
    manager, rule = _manager(), _rule()
    _feed(manager, rule, [True] * 3)

    assert _feed(manager, rule, [False, False]) == []
    [(record, action)] = _feed(manager, rule, [False])

    assert action == "resolved"
    assert record.status == "resolved" and record.resolved_at is not None
    assert manager.get_active_alerts() == []


async def test_history_keeps_the_firing_event_after_resolve():
    manager, rule = _manager(), _rule()
    _feed(manager, rule, [True] * 3 + [False] * 3)

    assert sorted(r.status for r in manager.get_history()) == ["firing", "resolved"]


async def test_history_is_bounded_and_newest_first():
    manager, rule = _manager(history_max_size=5), _rule(duration_seconds=INTERVAL)
    _feed(manager, rule, [True, False] * 10)  # need=1: every hit fires, every miss resolves

    history = manager.get_history()
    assert len(history) == 5
    assert [r.fired_at for r in history] == sorted((r.fired_at for r in history), reverse=True)
    assert len(manager.get_history(limit=2)) == 2


# ---------- notification ----------

async def test_firing_notifications_are_suppressed_within_window_until_resolved():
    manager, rule = _manager(suppress_duplicate_seconds=300), _rule(duration_seconds=INTERVAL)
    [(first, _)] = _feed(manager, rule, [True])
    assert manager._should_notify(first, "firing")
    assert not manager._should_notify(first, "firing")

    [(resolved, action)] = _feed(manager, rule, [False])
    assert manager._should_notify(resolved, action)
    [(refired, _)] = _feed(manager, rule, [True])
    assert manager._should_notify(refired, "firing")  # resolve clears the suppression window


async def test_one_failing_notifier_does_not_stop_the_others():
    manager, rule = _manager(), _rule(duration_seconds=INTERVAL)
    broken = SimpleNamespace(send_alert=AsyncMock(side_effect=RuntimeError("slack down")))
    healthy = SimpleNamespace(send_alert=AsyncMock(return_value=True))
    manager._notifiers = {"slack": broken, "stdout": healthy}
    [(record, action)] = _feed(manager, rule, [True])

    await manager._notify_all(record, action)

    healthy.send_alert.assert_awaited_once_with(record, "firing")


# ---------- rule value extraction ----------

def _monitor(cpu=10.0, memory=95.0, disk=50.0):
    snapshot = SimpleNamespace(cpu_percent=cpu, memory_percent=memory, disk_percent=disk)
    return SimpleNamespace(get_latest_snapshot=lambda: snapshot)


@pytest.mark.parametrize("services, expected", [
    ({"monitor": _monitor(memory=95.0)}, (True, 95.0)),
    ({"monitor": _monitor(memory=50.0)}, (False, 50.0)),
    ({"monitor": _monitor(cpu=-1, memory=-1)}, (False, 0.0)),  # not collected yet
    ({}, (False, 0.0)),                                          # monitoring service absent
])
async def test_memory_rule_reads_the_resource_snapshot(services, expected):
    manager = _manager(services)
    assert await manager._evaluate_rule(_rule()) == expected


async def test_error_rate_rule_uses_the_delta_since_the_window_start():
    manager = _manager()
    rule = _rule(name="HighErrorRate", threshold=0.05)
    with patch.object(manager, "_inference_request_totals",
                      side_effect=[(100.0, 0.0), (100.0, 0.0), (200.0, 10.0)]):
        assert await manager._evaluate_rule(rule) == (False, 0.0)  # first sample: no delta
        assert await manager._evaluate_rule(rule) == (False, 0.0)  # no new requests
        assert await manager._evaluate_rule(rule) == (True, 0.1)   # 10 errors / 100 requests


async def test_latency_rule_without_prometheus_does_not_trigger():
    manager = _manager()
    assert await manager._evaluate_rule(_rule(name="HighLatencyP95Sec", threshold=5.0)) == (False, 0.0)


async def test_prometheus_outage_logs_info_once_then_debug_and_resets_on_recovery():
    manager = _manager()
    manager.logger = Mock()
    rule = _rule(name="HighLatencyP95Sec", threshold=5.0)

    await manager._evaluate_rule(rule)
    await manager._evaluate_rule(rule)
    assert manager.logger.info.call_count == 1
    assert manager.logger.debug.call_count == 1
    manager.logger.warning.assert_not_called()

    # Prometheus back, but no latency samples yet: no error, and the outage flag clears.
    empty = Mock(status_code=200, json=Mock(return_value={"status": "success",
                                                         "data": {"result": []}}))
    client = AsyncMock()
    client.__aenter__.return_value.get = AsyncMock(return_value=empty)
    with patch("src.llm_router_part4_monitor.httpx.AsyncClient", return_value=client):
        assert await manager._evaluate_rule(rule) == (False, 0.0)
    assert manager._prometheus_down is False
    manager.logger.warning.assert_not_called()


async def test_unknown_rule_never_triggers():
    assert await _manager()._evaluate_rule(_rule(name="NoSuchRule")) == (False, 0.0)


# ---------- switch ----------

@pytest.mark.parametrize("alert_enabled, running, status", [
    (False, False, "healthy"),
    (True, False, "degraded"),
    (True, True, "healthy"),
])
async def test_health_reflects_switch_and_loop_state(alert_enabled, running, status):
    manager = _manager()
    manager.enabled = alert_enabled
    manager._running = running

    assert (await manager.get_health_status()).status == status
