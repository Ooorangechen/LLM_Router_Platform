# src/llm_router_part0_setup.py
# public class: ProjectSetup
# public function: setup_project_environment()
# P1 scaffold: create directories, write template files, create venv,
# install dependencies, then validate the resulting environment.
# Every step is idempotent: running setup twice must never raise.

import subprocess
import sys
from pathlib import Path

import yaml

# logger.py imports python-json-logger, which is not guaranteed to be installed when
# `main.py setup` runs (P1 5.2 only installs pyyaml + click beforehand). Degrade to a
# print-based logger instead of crashing, same optional-dependency pattern as the
# prometheus_client import in main.py.
try:
    from src.utils.logger import setup_logging, get_logger
    _LOGGER_AVAILABLE = True
except Exception:
    _LOGGER_AVAILABLE = False


# P1 3.1 requires 17 directories but enumerates only 15. Two are implied elsewhere:
# "src" because 3.1 also asks for an __init__.py inside it, and "data" because it is
# the parent that the 5.1 clean-up (`rm -rf data`) removes.
REQUIRED_DIRS = [
    "config",
    "data",
    "data/queries",
    "data/processed/routed",
    "docker",
    ".github/workflows",
    "flink",
    "kafka",
    "clickhouse/data",
    "monitoring/grafana",
    "slack/credentials",
    "streamlit_ui",
    "logs",
    "src",
    "src/models",
    "src/utils",
    "tests",
]

# python package directories that additionally need an empty __init__.py
PACKAGE_DIRS = ["src", "src/models", "src/utils", "tests"]

CONFIG_REL_PATH = "config/config.yaml"

# 3.1.4 key files that must exist once setup finishes
KEY_FILES = [CONFIG_REL_PATH, "requirements.txt"]

# 3.1.4 sections looked for inside config.yaml, missing ones are only reported
# because main.py injects defaults for them at load time
EXPECTED_SECTIONS = ["api", "router", "inference", "kafka", "monitoring"]


REQUIREMENTS_TEMPLATE = """\
# LLM Router Platform — 依赖清单
# Python 要求：>= 3.9（开发/生产推荐 3.11）

# Web 框架
fastapi>=0.104
uvicorn[standard]>=0.24

# 数据校验 / Schema
pydantic>=2.5
pydantic-settings>=2.1

# 配置文件格式
pyyaml>=6.0

# 结构化日志
structlog>=23.2
python-json-logger>=2.0
loguru>=0.7

# 指标采集
prometheus-client>=0.19

# CLI 命令框架
click>=8.1

# HTTP Client（后续阶段使用）
httpx>=0.25
aiohttp>=3.9

# 消息队列（后续阶段使用）
aiokafka>=0.8
kafka-python>=2.0

# 分析数据库（后续阶段使用）
clickhouse-connect>=0.6

# 缓存（后续阶段使用）
redis>=5.0


# 前端控制台（后续阶段使用）
streamlit>=1.28
plotly>=5.18
pandas>=2.0

# 异步编排（后续阶段使用）
langgraph>=0.0.20
langchain-core>=0.1

# 模型微调（后续阶段使用）
peft>=0.7
transformers>=4.36
datasets>=2.16
torch>=2.1

# 代码质量（可选）
pytest>=7.4
pytest-asyncio>=0.21
black>=23.0
flake8>=6.0
mypy>=1.7
"""


CONFIG_TEMPLATE = """\
## 全平台配置文件，按section分层
## api / logging / router / ...
api:
  host: "0.0.0.0"
  port: 8080
  log_level: info
  cors_origins:
    - "*"
  rate_limiting:
    enabled: false
    rpm: 60
    burst_size: 10

logging:
  level: info
  file: logs/llm_router.log
  max_bytes: 10485760
  backup_count: 5
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  json_format: "%(asctime)s %(name)s %(levelname)s %(message)s %(process)d %(thread)d %(module)s %(lineno)d"
  console_output: true
  structured_logs: true

router:
  default_model: mistral-7b
  routing_strategy: intelligent
  models: 
    mistral-7b:
      provider: vllm
      api_key_env: VLLM_API_KEY
      max_tokens: 8192
      cost_per_token: 0.0
      priority: 1
      capabilities:
        - general
        - math
      gpu_memory_gb: 16
    gpt-4-turbo:
      provider: openai
      api_key_env: OPENAI_API_KEY
      max_tokens: 4096
      cost_per_token: 1.5e-05
      priority: 2
      capabilities:
        - coding
        - reasoning
        - analysis
        - general
    claude-3.5-sonnet:
      provider: anthropic
      api_key_env: ANTHROPIC_API_KEY
      max_tokens: 8192
      cost_per_token: 6.0e-06
      priority: 2
      capabilities:
        - writing
        - creative
        - analysis
        - reasoning
        - general
    llama-3.1-70b: 
      provider: vllm
      api_key_env: VLLM_API_KEY
      max_tokens: 8192
      cost_per_token: 0.0
      priority: 1
      capabilities:
        - reasoning
        - analysis
        - general
        - translation
      gpu_memory_gb: 160

  routing_rules:
    - name: code_generation
      condition: "query_type == 'code_generation'"
      target_model: gpt-4-turbo
    - name: long_context_analysis
      condition: "query_type == 'analysis' and context_length > 20000"
      target_model: claude-3.5-sonnet
    - name: premium_tier
      condition: "user_tier == 'premium'"
      target_model: claude-3.5-sonnet
    - name: free_tier
      condition: "user_tier == 'free'"
      target_model: mistral-7b

inference:
  vllm:
    host: localhost
    port: 8001
    base_url: http://localhost:8001/v1
    timeout: 60
    retries: 3
  openai:
    host: api.openai.com
    port: 443
    base_url: https://api.openai.com/v1
    timeout: 30
    retries: 3
  anthropic:
    host: api.anthropic.com
    port: 443
    base_url: https://api.anthropic.com/v1
    timeout: 30
    retries: 3
  compression:
    enabled: false
    max_context_tokens: 100000
    compression_ratio: 0.5
    method: summarization
  cache:
    enabled: false
    backend: redis
    ttl: 3600
    max_size: 10000
  batching:
    enabled: false
    max_batch_size: 32
    max_wait_time_ms: 100
  
kafka:
  enabled: false
  bootstrap_servers: localhost:9092

  topics:
    queries: llm-router-queries
    responses: llm-router-responses
    metrics: llm-router-metrics
    errors: llm-router-errors
  
  producer:
    acks: all
    retries: 3
    batch_size: 16384
    linger_ms: 10
    compression_type: gzip
  
  consumer:
    group_id: llm-router-consumer-group
    auto_offset_reset: earliest
    enable_auto_commit: true
    max_poll_records: 500

clickhouse:
  enabled: false
  host: localhost
  port: 8123
  database: llm_router
  username: default
  password_env: CLICKHOUSE_PASSWORD

  tables:
    query_logs: query_logs
    system_metrics: system_metrics
    model_performance: model_performance
    user_analytics: user_analytics

monitoring:
  enabled: false
  prometheus_port: 8000

  prometheus:
    scrape_interval: 15s
  
  grafana:
    port: 3000
    admin_user: admin
    admin_password_env: GRAFANA_ADMIN_PASSWORD
  
  alerts:
    error_rate_threshold: 0.05
    latency_p95_threshold_ms: 2000
    cpu_usage_threshold: 0.85
    memory_usage_threshold: 0.85
  
  health_checks:
    interval: 30s
    timeout: 5s
  
slack:
  enabled: false
  bot_token_env: SLACK_BOT_TOKEN
  app_token_env: SLACK_APP_TOKEN
  signing_secret_env: SLACK_SIGNING_SECRET

  channels:
    - general
    - llm-router-alerts
  
  response_settings:
    max_response_length: 3000
    thread_replies: true
    typing_indicator: true
  
  rate_limiting:
    enabled: true
    rpm: 20

streamlit:
  enabled: true
  port: 8501
  host: "0.0.0.0"

  theme:
    mode: dark
    primary_color: "#FF6B6B"
    background_color: "#0E1117"
  
  dashboard:
    refresh_interval_seconds: 10
    default_time_range_hours: 24
  
flink:
  enabled: false
  job_manager:
    host: localhost
    port: 8081
  parallelism: 2
  checkpointing:
    enabled: true
    interval_ms: 60000
    mode: exactly_once

security:
  api_keys:
    enabled: false
    header_name: X-API-Key
  jwt: 
    enabled: false
    secret_env: JWT_SECRET
    algorithm: HS256
    expiration_hours: 24
  cors:
    allow_credentials: true
    allow_methods:
      - GET
      - POST
      - PUT
      - DELETE
    allow_headers:
      - "*"

performance:
  connection_pools:
    database:
      min_size: 2
      max_size: 10
    http:
      min_size: 5
      max_size: 50
  workers:
    api: 4
    inference: 2
    pipeline: 2
  memory:
    heap_size_mb: 2048
    gc_threshold: 0.8

development: 
  debug: false
  auto_reload: false
  profiling: false
  mock_external_apis: false

features:
  context_compression: false
  semantic_caching: false
  batch_processing: false
  multi_modal: false
  function_calling: false
  streaming_responses: false

pipeline:
  enabled: false

adapters:
  enabled: false
  registry_path: data/adapters/registry.json

  selection:
    strategy: static
    canary:
      enabled: false
      stages: 
        - 5
        - 20
        - 100
  training:
    base_model: mistral-7b
    method: lora
    learning_rate: 0.0002
    epochs: 3
    batch_size: 8

policies: 
  quota:
    tier_quotas:
      free: 
        daily: 100
        hourly: 10
      premium:
        daily: 1000
        hourly: 100
      enterprise:
        daily: 10000
        hourly: 1000

  sla:
    latency_slas:
      free: 10s
      premium: 5s
      enterprise: 2s
  
  budget: 
    cost_budgets:
      free: 0.01
      premium: 0.10
      enterprise: 1.00
  
  circuit_breaker: 
    enabled: true
    failure_threshold: 5
    recovery_timeout_s: 30
  
optimization:
  enabled: false
  kv_cache_size_gb: 8
  max_batch_size: 32
  max_wait_ms: 100
  flash_attn: true
  tensorrt: false

quality:
  monitor:
    enabled: false
    window_size: 100
    window_duration_minutes: 60

  slo_targets:
    availability: 0.999
    latency_p95_ms: 2000
    error_rate_max: 0.01
  
  feedback:
    storage_path: data/feedback
  
  health_check_interval_s: 30

router_mode:
  use_integrated_router: false"""


class _PrintLogger:
    """Fallback logger used when src.utils.logger cannot be imported yet."""

    def info(self, message):
        print(f"[setup] {message}")

    def debug(self, message):
        print(f"[setup] {message}")

    def warning(self, message):
        print(f"[setup] WARNING: {message}")


class ProjectSetup:
    """P1 project scaffold: directories, template files, venv, dependencies, validation."""

    def __init__(self, project_root: str = ".", logger=None):
        self.project_root = Path(project_root).resolve()
        self.logger = logger or _PrintLogger()

        self.required_dirs = list(REQUIRED_DIRS)
        self.package_dirs = list(PACKAGE_DIRS)

        # 3.1.2 the 10 template files: relative path -> content builder
        self.required_files = {
            ".gitignore": self._template_gitignore,
            "README.md": self._template_readme,
            "requirements.txt": self._template_requirements,
            "docker/requirements.txt": self._template_docker_requirements,
            "kafka/topics.json": self._template_kafka_topics,
            "clickhouse/schema.sql": self._template_clickhouse_schema,
            "monitoring/prometheus.yml": self._template_prometheus,
            "monitoring/grafana/dashboard.json": self._template_grafana_dashboard,
            "streamlit_ui/config.toml": self._template_streamlit_config,
            ".github/workflows/ci.yml": self._template_ci_workflow,
        }

    # ---------------------------------------------------------------- entry

    def setup_project_environment(self, install_deps: bool = True) -> None:
        """Scaffold entry point: directories -> templates -> config -> venv -> validation."""
        self.logger.info(f"Starting project environment setup at {self.project_root}")

        self._create_directories()
        self._create_files()
        self._create_config_file()

        if install_deps:
            self._setup_python_environment()
        else:
            self.logger.info("Skipping dependency install (install_deps=False)")

        self._validate_environment()
        self._validate_config()

        self.logger.info("Project environment setup finished successfully")

    # ------------------------------------------------------ 1. directories

    def _create_directories(self) -> None:
        for rel_dir in self.required_dirs:
            dir_path = self.project_root / rel_dir
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Directory ensured: {rel_dir}")

        for rel_dir in self.package_dirs:
            init_file = self.project_root / rel_dir / "__init__.py"
            if init_file.exists():
                self.logger.debug(f"Package marker already exists: {rel_dir}/__init__.py")
                continue
            init_file.touch()
            self.logger.debug(f"Created package marker: {rel_dir}/__init__.py")

        self.logger.info(f"{len(self.required_dirs)} directories ready")

    # -------------------------------------------------- 2. template files

    def _create_files(self) -> None:
        created = 0
        for rel_path, build_content in self.required_files.items():
            file_path = self.project_root / rel_path

            # 3.1.2 write when absent, skip when present, never overwrite user edits
            if file_path.exists():
                self.logger.debug(f"File already exists, skip: {rel_path}")
                continue

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(build_content(), encoding="utf-8")
            created += 1
            self.logger.debug(f"Template file created: {rel_path}")

        self.logger.info(f"{len(self.required_files)} template files ready ({created} newly created)")

    # --------------------------------------------------------- 3. config

    def _create_config_file(self) -> None:
        """config.yaml is not one of the 10 templates but 5.2 still expects it to exist
        after a clean setup, and 3.1.4 validates it right afterwards."""
        config_path = self.project_root / CONFIG_REL_PATH

        if config_path.exists():
            self.logger.debug(f"File already exists, skip: {CONFIG_REL_PATH}")
            return

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(self._template_config_yaml(), encoding="utf-8")
        self.logger.info(f"Config template created: {CONFIG_REL_PATH}")

    # ------------------------------------------------ 4. python environment

    def _setup_python_environment(self) -> None:
        venv_path = self.project_root / "venv"

        if venv_path.exists():
            self.logger.info(f"venv already exists at {venv_path}, skip creation")
        else:
            self.logger.info("Creating virtual environment (venv)...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

        pip_path = self._venv_pip_path(venv_path)
        requirements_path = self.project_root / "requirements.txt"

        if not pip_path.exists() or not requirements_path.exists():
            self.logger.warning(
                f"Skipping dependency install: pip exists={pip_path.exists()}, "
                f"requirements.txt exists={requirements_path.exists()}"
            )
            return

        self.logger.info("Installing dependencies from requirements.txt, this may take a while...")
        subprocess.run([str(pip_path), "install", "-r", str(requirements_path)], check=True)
        self.logger.info("Dependencies installed successfully")

    @staticmethod
    def _venv_pip_path(venv_path: Path) -> Path:
        # posix puts pip under bin/, windows under Scripts/
        if sys.platform.startswith("win"):
            return venv_path / "Scripts" / "pip.exe"
        return venv_path / "bin" / "pip"

    # ----------------------------------------------------- 5. validation

    def _validate_environment(self) -> None:
        for rel_dir in self.required_dirs:
            dir_path = self.project_root / rel_dir
            if not dir_path.is_dir():
                raise FileNotFoundError(f"Required directory missing: {dir_path}")

        for rel_file in KEY_FILES:
            file_path = self.project_root / rel_file
            if not file_path.is_file():
                raise FileNotFoundError(f"Required file missing: {file_path}")

        self.logger.info("Directory and key file validation passed")

    def _validate_config(self) -> None:
        config_path = self.project_root / CONFIG_REL_PATH

        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"config.yaml is not valid YAML: {exc}") from exc

        # 3.1.4 only the YAML syntax is a hard requirement here, a missing section is
        # reported but not fatal because main.py._load_config injects defaults for it
        missing = [name for name in EXPECTED_SECTIONS if name not in config]
        if missing:
            self.logger.warning(
                f"config.yaml is missing sections {missing}, "
                f"main.py will inject defaults at load time"
            )

        self.logger.info("config.yaml structure validation passed")

    # ------------------------------------------------------- templates

    @staticmethod
    def _template_gitignore() -> str:
        return """\
# --- Python ---
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.venv/
venv/

# --- Env / Secrets ---
.env
.env.*
*.pem
*.key
secrets/
slack/credentials/

# --- Logs and runtime data ---
logs/
*.log
*.log.jsonl
data/queries/
data/processed/
data/feedback/

# --- Model weights ---
*.bin
*.safetensors
*.pt
*.gguf
data/adapters/

# --- Service volumes ---
clickhouse/data/
docker/volumes/

# --- IDE ---
.vscode/
.idea/
.DS_Store

# --- Tests / coverage ---
.pytest_cache/
.coverage
htmlcov/
"""

    @staticmethod
    def _template_readme() -> str:
        return """\
# LLM Router & Execution Platform

A multi-model LLM routing and execution platform. It exposes one entry point for
OpenAI, Anthropic and self-hosted vLLM backends, and adds routing decisions,
end-to-end observability and multi-tenant governance on top.

## Quick start

```bash
python main.py setup    # create directories, templates and dependencies
python main.py start    # start the service, default http://localhost:8080
python main.py health   # check service health
```

## CLI commands

| Command | Purpose |
|---|---|
| `setup` | Create the project scaffold |
| `start` | Run the platform, add `--dev` for auto reload |
| `health` | Query `/health` of a running instance |
| `deploy` | Generate deployment artifacts (P1 stub) |

## Layout

See section 4.2 of `docs/P1.md` for the module breakdown.

This file is a P1 placeholder and will be expanded in later phases.
"""

    @staticmethod
    def _template_requirements() -> str:
        return REQUIREMENTS_TEMPLATE

    @staticmethod
    def _template_docker_requirements() -> str:
        # kept identical to requirements.txt to match the current project artifact,
        # trimming it to a runtime-only subset is a later-phase optimisation
        return REQUIREMENTS_TEMPLATE

    @staticmethod
    def _template_kafka_topics() -> str:
        return """\
{
  "topics": [
    {
      "name": "llm-router-queries",
      "partitions": 3,
      "replication_factor": 1,
      "retention_ms": 604800000,
      "compression_type": "gzip"
    },
    {
      "name": "llm-router-responses",
      "partitions": 3,
      "replication_factor": 1,
      "retention_ms": 604800000,
      "compression_type": "gzip"
    },
    {
      "name": "llm-router-metrics",
      "partitions": 3,
      "replication_factor": 1,
      "retention_ms": 259200000,
      "compression_type": "gzip"
    },
    {
      "name": "llm-router-errors",
      "partitions": 3,
      "replication_factor": 1,
      "retention_ms": 1209600000,
      "compression_type": "gzip"
    }
  ]
}
"""

    @staticmethod
    def _template_clickhouse_schema() -> str:
        return """\
-- P1 template, activated in P3

CREATE TABLE IF NOT EXISTS query_logs (
    request_id String,
    user_id String,
    user_tier String,
    query_type String,
    model_name String,
    provider String,
    input_tokens UInt32,
    output_tokens UInt32,
    latency_ms Float64,
    cost_usd Float64,
    status String,
    timestamp DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, user_id);

CREATE TABLE IF NOT EXISTS system_metrics (
    name String,
    value Float64,
    labels String,
    timestamp DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, name);

CREATE TABLE IF NOT EXISTS model_performance (
    model_name String,
    provider String,
    tokens_per_second Float64,
    quality_score Float64,
    error_rate Float64,
    timestamp DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, model_name);

CREATE TABLE IF NOT EXISTS user_analytics (
    user_id String,
    user_tier String,
    total_requests UInt64,
    total_tokens UInt64,
    total_cost_usd Float64,
    date Date
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_metrics
ENGINE = SummingMergeTree()
ORDER BY (hour, model_name)
AS
SELECT
    toStartOfHour(timestamp) AS hour,
    model_name,
    count() AS request_count,
    sum(cost_usd) AS total_cost_usd,
    sum(input_tokens + output_tokens) AS total_tokens
FROM query_logs
GROUP BY hour, model_name;

-- bloom filter indexes for the three highest cardinality lookup columns
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS idx_user_id user_id TYPE bloom_filter GRANULARITY 4;
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS idx_model_name model_name TYPE bloom_filter GRANULARITY 4;
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS idx_query_type query_type TYPE bloom_filter GRANULARITY 4;
"""

    @staticmethod
    def _template_prometheus() -> str:
        return """\
# P1 template, activated in P4
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["localhost:9093"]

scrape_configs:
  - job_name: "api"
    static_configs:
      - targets: ["localhost:8000"]

  - job_name: "inference"
    static_configs:
      - targets: ["localhost:8001"]

  - job_name: "vllm"
    static_configs:
      - targets: ["localhost:8002"]

  - job_name: "kafka-exporter"
    static_configs:
      - targets: ["localhost:9308"]

  - job_name: "clickhouse-exporter"
    static_configs:
      - targets: ["localhost:9116"]

  - job_name: "node-exporter"
    static_configs:
      - targets: ["localhost:9100"]

  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
"""

    @staticmethod
    def _template_grafana_dashboard() -> str:
        return """\
{
  "title": "LLM Router Platform - Overview",
  "timezone": "browser",
  "refresh": "10s",
  "time": {
    "from": "now-1h",
    "to": "now"
  },
  "panels": [
    {
      "id": 1,
      "title": "Requests Per Second",
      "type": "timeseries",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
      "targets": [
        {"expr": "sum(rate(llm_router_requests_total[1m]))", "legendFormat": "rps"}
      ]
    },
    {
      "id": 2,
      "title": "P95 Latency",
      "type": "timeseries",
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum by (le) (rate(llm_router_request_duration_seconds_bucket[5m])))",
          "legendFormat": "p95"
        }
      ]
    },
    {
      "id": 3,
      "title": "Model Distribution",
      "type": "piechart",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
      "targets": [
        {"expr": "sum by (model) (llm_router_router_routing_decisions_total)", "legendFormat": "{{model}}"}
      ]
    }
  ]
}
"""

    @staticmethod
    def _template_streamlit_config() -> str:
        return """\
[server]
port = 8501
address = "0.0.0.0"
headless = true
# internal network deployment, no browser origin checks needed
enableCORS = false
enableXsrfProtection = false

[theme]
base = "dark"
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"

[browser]
gatherUsageStats = false
"""

    @staticmethod
    def _template_ci_workflow() -> str:
        return """\
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      kafka:
        image: bitnami/kafka:3.6
        ports:
          - 9092:9092
        env:
          KAFKA_CFG_NODE_ID: 0
          KAFKA_CFG_PROCESS_ROLES: controller,broker
          KAFKA_CFG_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
          KAFKA_CFG_CONTROLLER_QUORUM_VOTERS: 0@localhost:9093
          KAFKA_CFG_CONTROLLER_LISTENER_NAMES: CONTROLLER
      clickhouse:
        image: clickhouse/clickhouse-server:24.3
        ports:
          - 8123:8123
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v

  security-scan:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: security scan placeholder
        run: echo "wire up pip-audit / bandit here"

  docker-build:
    runs-on: ubuntu-latest
    needs: security-scan
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: docker build placeholder
        run: echo "wire up docker/build-push-action here"

  deploy:
    runs-on: ubuntu-latest
    needs: docker-build
    if: github.ref == 'refs/heads/main'
    strategy:
      matrix:
        environment: [staging, production]
    steps:
      - name: deploy placeholder
        run: echo "deploy to ${{ matrix.environment }}"
"""

    @staticmethod
    def _template_config_yaml() -> str:
        return CONFIG_TEMPLATE


def setup_project_environment(project_root: str = ".", install_deps: bool = True) -> None:
    """P1 public entry point, called by the `setup` CLI command in main.py.

    Order follows 3.1.5: set up logging, build ProjectSetup, run the scaffold.
    """
    if _LOGGER_AVAILABLE:
        setup_logging()
        logger = get_logger(__name__)
    else:
        logger = _PrintLogger()
        logger.warning("src.utils.logger unavailable, falling back to print output")

    project_setup = ProjectSetup(project_root=project_root, logger=logger)
    project_setup.setup_project_environment(install_deps=install_deps)


if __name__ == "__main__":
    print(f"required directories ({len(REQUIRED_DIRS)}):")
    for name in REQUIRED_DIRS:
        print(f"  {name}")

    setup = ProjectSetup()
    print(f"template files ({len(setup.required_files)}):")
    for name in setup.required_files:
        print(f"  {name}")

    print(f"config template: {CONFIG_REL_PATH}")
    print("Dry run only, call setup_project_environment() to apply.")
