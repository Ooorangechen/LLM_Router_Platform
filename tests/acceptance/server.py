"""Run `python main.py start` in a subprocess, the way the docs/P*.md §5 acceptance does.

Each server gets its own override config, port and log file, so log greps only
see that server's output and a developer server already on 8080 is left alone.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from prometheus_client.parser import text_string_to_metric_families


# Point the real providers at an env var that is never set: no key, no API call,
# every /route ends as an inference error. (.env is loaded by main.py, so unsetting
# OPENAI_API_KEY in the environment is not enough.)
NO_PROVIDERS = {"inference": {
    "openai": {"api_key_env": "P4_ACCEPTANCE_UNSET_KEY"},
    "anthropic": {"api_key_env": "P4_ACCEPTANCE_UNSET_KEY"},
}}

MONITORING_ON = {"monitoring": {
    "enabled": True,
    "alert_enabled": True,
    "resource_collector": {"enabled": True, "interval_sec": 2},
    "alert_manager": {"eval_interval_sec": 2},
}}

MONITORING_OFF = {"monitoring": {"enabled": False}, "pipeline": {"enabled": False}}


def deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def port_in_use(port):
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def metric_total(text, name, **labels):
    """Sum every sample called `name` whose labels include `labels`."""
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items())
    )


def route_payload(i, **extra):
    return {"query": f"mon test {i}", "user_id": f"u{i}", "max_tokens": 32,
            "temperature": 1.0, **extra}


@dataclass
class Server:
    port: int
    log_file: Path
    stdout_file: Path
    proc: subprocess.Popen

    @property
    def base_url(self):
        return f"http://localhost:{self.port}"

    def client(self, timeout=60):
        return httpx.Client(base_url=self.base_url, timeout=timeout, follow_redirects=True)

    def log_text(self):
        return self.log_file.read_text(encoding="utf-8") if self.log_file.exists() else ""

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def start_server(root, workdir, overrides=None, port=None, env=None, timeout=120):
    """Start main.py with canonical config + `overrides`; return once /health answers."""
    port = port or free_port()
    if port_in_use(port):
        raise RuntimeError(f"port {port} is already in use; stop the process listening on it")

    log_file = workdir / f"llm_router_{port}.log"
    config = deep_merge(
        {"api": {"port": port}, "logging": {"file": str(log_file)}}, overrides or {})
    config_file = workdir / f"config_{port}.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    child_env = dict(os.environ)
    for key, value in (env or {}).items():
        if value is None:
            child_env.pop(key, None)
        else:
            child_env[key] = value

    stdout_file = workdir / f"stdout_{port}.txt"
    with stdout_file.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            [sys.executable, "main.py", "start", "--config", str(config_file)],
            cwd=root, env=child_env, stdout=out, stderr=subprocess.STDOUT)
    server = Server(port, log_file, stdout_file, proc)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited with code {proc.returncode}:\n"
                + stdout_file.read_text(encoding="utf-8")[-3000:])
        try:
            httpx.get(f"{server.base_url}/health", timeout=2)
            return server
        except httpx.HTTPError:
            time.sleep(0.5)
    server.stop()
    raise RuntimeError(f"server on :{port} not ready after {timeout}s; see {stdout_file}")


def wait_until(predicate, timeout, interval=0.5):
    """Poll `predicate` until it returns a truthy value; return that value or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    return None
