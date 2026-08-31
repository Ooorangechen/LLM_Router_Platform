# tests/test_p1_config.py
# P1.md 5.7 verification point 6: the platform still starts when config sections
# are missing, plus the 3.5 rules around _load_config and the reload endpoint.

import pytest
import yaml

# 3.5.2: the seven sections _load_config injects defaults for
DEFAULTED_SECTIONS = [
    "api", "router", "adapters", "optimization", "quality", "policies", "router_mode",
]


@pytest.fixture
def minimal_config(tmp_path):
    """The 5.7 scenario: a config that only sets api.port."""
    path = tmp_path / "config.yaml"
    path.write_text("api:\n  port: 8081\n", encoding="utf-8")
    return path


@pytest.fixture
def minimal_platform(minimal_config):
    from main import LLMRouterPlatform

    return LLMRouterPlatform(str(minimal_config))


def test_platform_builds_from_a_nearly_empty_config(minimal_platform):
    """5.7: adapters / optimization / quality / policies default injection works,
    the platform does not crash on the missing sections."""
    assert minimal_platform.config["api"]["port"] == 8081


@pytest.mark.parametrize("section", DEFAULTED_SECTIONS)
def test_every_defaulted_section_is_present(minimal_platform, section):
    assert section in minimal_platform.config


def test_router_mode_defaults_to_modular_pipeline(minimal_platform):
    """3.5.2: router_mode.use_integrated_router defaults to False."""
    assert minimal_platform.config["router_mode"]["use_integrated_router"] is False


def test_tier_policy_defaults(minimal_platform):
    """3.5.2: quota / sla / budget defaults for the three tiers."""
    policies = minimal_platform.config["policies"]

    assert policies["quota"]["tier_quotas"]["free"] == {"daily": 100, "hourly": 10}
    assert policies["quota"]["tier_quotas"]["premium"] == {"daily": 1000, "hourly": 100}
    assert policies["quota"]["tier_quotas"]["enterprise"] == {"daily": 10000, "hourly": 1000}
    assert policies["sla"]["latency_slas"] == {
        "free": "10s", "premium": "5s", "enterprise": "2s",
    }
    assert policies["budget"]["cost_budgets"] == {
        "free": 0.01, "premium": 0.10, "enterprise": 1.00,
    }


def test_health_is_healthy_under_the_minimal_config(minimal_platform):
    """5.7 step 4: /health still reports healthy on the stripped down config."""
    from fastapi.testclient import TestClient

    response = TestClient(minimal_platform._create_fastapi_app()).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_load_config_has_no_side_effects(minimal_platform):
    """3.5 core constraint: _load_config may run at startup and again at runtime,
    and must not initialise any service."""
    before = dict(minimal_platform.services)

    first = minimal_platform._load_config()
    second = minimal_platform._load_config()

    assert first == second
    assert minimal_platform.services == before


def test_reload_config_returns_500_when_the_file_disappeared(minimal_platform, minimal_config):
    """3.5.3: the reload endpoint answers 500 on a bad config instead of killing
    the running process."""
    from fastapi.testclient import TestClient

    client = TestClient(minimal_platform._create_fastapi_app(), raise_server_exceptions=False)
    minimal_config.unlink()

    response = client.post("/admin/reload-config")

    assert response.status_code == 500
    # the process survived: a follow up request still succeeds
    assert client.get("/health").status_code == 200


def test_reload_config_returns_500_on_broken_yaml(minimal_platform, minimal_config):
    from fastapi.testclient import TestClient

    client = TestClient(minimal_platform._create_fastapi_app(), raise_server_exceptions=False)
    minimal_config.write_text("api:\n  port: [unclosed\n", encoding="utf-8")

    response = client.post("/admin/reload-config")

    assert response.status_code == 500
    assert client.get("/health").status_code == 200


def test_reload_config_picks_up_a_changed_value(minimal_platform, minimal_config):
    """3.5.3: reload replaces self.config, later requests observe the new values."""
    from fastapi.testclient import TestClient

    client = TestClient(minimal_platform._create_fastapi_app())
    minimal_config.write_text(
        "api:\n  port: 9099\nrouter_mode:\n  use_integrated_router: true\n",
        encoding="utf-8",
    )

    assert client.post("/admin/reload-config").json() == {"status": "config_reloaded"}
    assert minimal_platform.config["api"]["port"] == 9099
    assert client.get("/status").json()["router_mode"]["use_integrated_router"] is True


# exact key names that would mean a credential is stored inline. Substring matching
# is wrong here: max_tokens, cost_per_token and security.api_keys are all innocent.
BARE_SECRET_KEYS = {
    "password", "secret", "api_key", "apikey", "token", "bot_token",
    "app_token", "signing_secret", "admin_password", "private_key", "access_key",
}

# prefixes of real credentials from the providers this platform talks to
CREDENTIAL_PREFIXES = ("sk-", "xoxb-", "xoxp-", "xapp-", "ghp_", "AKIA")


def _walk_config(node, trail=""):
    """Yield (dotted_path, key, value) for every scalar in the document."""
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{trail}.{key}" if trail else str(key)
            yield path, str(key), value
            yield from _walk_config(value, path)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk_config(item, f"{trail}[{index}]")


def test_shipped_config_stores_credentials_as_env_var_names(real_config_path):
    """3.5 core constraint: a secret field may only hold an env variable NAME."""
    config = yaml.safe_load(real_config_path.read_text(encoding="utf-8"))

    inline_secrets = []
    bad_env_values = []

    for path, key, value in _walk_config(config):
        if key.lower() in BARE_SECRET_KEYS:
            inline_secrets.append(path)
        if key.lower().endswith("_env"):
            if not isinstance(value, str) or not value.isupper():
                bad_env_values.append(f"{path}={value!r}")

    assert not inline_secrets, f"secret fields not using the *_env convention: {inline_secrets}"
    assert not bad_env_values, f"*_env fields must hold an uppercase variable name: {bad_env_values}"


def test_shipped_config_contains_no_literal_credentials(real_config_path):
    """3.1 / 5.1 core constraint: no provider key may ever be committed."""
    text = real_config_path.read_text(encoding="utf-8")

    found = [prefix for prefix in CREDENTIAL_PREFIXES if prefix in text]
    assert not found, f"config.yaml contains what looks like a real credential: {found}"


def test_every_model_declares_an_api_key_env(real_config_path):
    """3.5.1: each router model references its key by variable name."""
    config = yaml.safe_load(real_config_path.read_text(encoding="utf-8"))
    models = config.get("router", {}).get("models", {})
    assert models, "router.models is empty"

    missing = [name for name, spec in models.items() if not spec.get("api_key_env")]
    assert not missing, f"models without api_key_env: {missing}"
