"""P4 M4 (§5.5): a real error spike fires HighErrorRate, notifies once, dedups, and is recorded.

No provider has a key (NO_PROVIDERS), so every /route is an inference error and no API
is called. Intervals are shortened as §5.5 B allows: eval every 1s, fire after 2s.
The Slack notifier posts to a local HTTP server, which counts the notifications.
"""

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tests.acceptance.server import MONITORING_ON, NO_PROVIDERS, deep_merge, route_payload, wait_until


SLACK_ENV = "P4_ACCEPTANCE_SLACK_WEBHOOK"
ALERT_CONFIG = deep_merge(deep_merge(MONITORING_ON, NO_PROVIDERS), {"monitoring": {"alert_manager": {
    "eval_interval_sec": 1,
    "suppress_duplicate_seconds": 300,
    "rules_override": [{
        "name": "HighErrorRate", "expr_lambda_src": "inference error_rate(5m) > 0.05",
        "threshold": 0.05, "duration_seconds": 2, "severity": "critical",
        "description": "Inference error rate over last 5m exceeds 5%", "enabled": True,
    }],
    "notifiers": {"stdout": {"enabled": True},
                  "slack": {"enabled": True, "webhook_url_env": SLACK_ENV}},
}}})


@pytest.fixture(scope="module")
def slack():
    """A stand-in Slack webhook; .messages collects the posted texts."""
    messages = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            messages.append(json.loads(body)["text"])
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    httpd.messages = messages
    httpd.url = f"http://127.0.0.1:{httpd.server_port}/hook"
    yield httpd
    httpd.shutdown()


@pytest.fixture(scope="module")
def server(server_factory, slack):
    return server_factory(overrides=ALERT_CONFIG, env={SLACK_ENV: slack.url})


def _send_failing(server, count):
    with server.client() as client, ThreadPoolExecutor(max_workers=10) as pool:
        return list(pool.map(
            lambda i: client.post("/route", json=route_payload(i, user_id=f"e{i}")).status_code,
            range(count)))


def _active(server, **params):
    with server.client() as client:
        return client.get("/admin/alerts/active", params=params).json()


def _history(server):
    with server.client() as client:
        return client.get("/admin/alerts/history", params={"limit": 200}).json()


@pytest.fixture(scope="module")
def fired(server):
    codes = _send_failing(server, 30)
    assert set(codes) == {500}, codes  # P2: no provider -> HTTP 500 with detail

    def firing():
        return [a for a in _active(server)
                if a["rule_name"] == "HighErrorRate" and a["status"] == "firing"]

    records = wait_until(firing, timeout=30)
    assert records, f"HighErrorRate never fired; active={_active(server)}"
    return records[0]


def test_high_error_rate_fires_with_complete_fields(fired):
    assert fired["severity"] == "critical"
    assert fired["fired_at"]
    assert fired["value"] > fired["threshold"] == 0.05


def test_active_alerts_filter_by_severity(server, fired):
    assert [a["rule_name"] for a in _active(server, severity="critical")] == ["HighErrorRate"]
    assert _active(server, severity="warning") == []


def test_slack_webhook_notified_once(slack, fired):
    assert wait_until(lambda: slack.messages, timeout=10)
    assert len(slack.messages) == 1
    assert "[FIRING] HighErrorRate" in slack.messages[0]


def test_dense_spike_is_deduplicated(server, slack, fired):
    _send_failing(server, 100)
    time.sleep(4)  # let ~4 eval rounds see the spike

    firing = Counter(a["rule_name"] for a in _history(server) if a["status"] == "firing")
    assert firing["HighErrorRate"] == 1
    assert len(slack.messages) == 1


def test_history_is_bounded_and_newest_first(server, fired):
    history = _history(server)
    assert 1 <= len(history) <= 1000
    fired_at = [a["fired_at"] for a in history]
    assert fired_at == sorted(fired_at, reverse=True)


def test_stdout_notifier_writes_alert_to_log(server, fired):
    assert wait_until(lambda: "[ALERT FIRING] HighErrorRate" in server.log_text(), timeout=10)
