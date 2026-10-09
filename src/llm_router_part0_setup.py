# src/llm_router_part0_setup.py
# public class: ProjectSetup
# public function: setup_project_environment()
# P1–P4 scaffold: create directories, write template files, create venv,
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
    "data/prometheus",
    "data/grafana",
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
KEY_FILES = [
    CONFIG_REL_PATH,
    "requirements.txt",
]

# Sections required in the canonical platform configuration.
EXPECTED_SECTIONS = [
    "api", "logging", "router", "inference", "kafka", "clickhouse",
    "monitoring", "slack", "streamlit", "flink", "security", "performance",
    "development", "features", "pipeline", "adapters", "policies",
    "optimization", "quality", "router_mode",
]


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
python-dotenv>=1.0

# 结构化日志
structlog>=23.2
python-json-logger>=2.0
loguru>=0.7

# 指标采集
prometheus-client>=0.19
# P4 resource collection (GPU support is optional)
psutil>=5.9
GPUtil>=1.4.0; sys_platform != 'darwin'

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

# Encoders
tiktoken>=0.14.0

# LLM Provider SDK
openai>=1.12
anthropic>=0.18

# 重试策略
tenacity>=8.2
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
      max_tokens: 32768
      cost_input_token: 0.0
      cost_output_token: 0.0
      priority: 3
      capabilities:
        - general
        - math
        - coding
      gpu_memory_gb: 16
      model_path: null
    gpt-5.6-terra:
      provider: openai
      api_key_env: OPENAI_API_KEY
      max_tokens: 5000
      cost_input_token: 2.0e-06
      cost_output_token: 1.2e-05
      priority: 2
      capabilities:
        - coding
        - reasoning
        - analysis
        - math
        - general
        - creative
      model_path: null
    claude-sonnet-5:
      provider: anthropic
      api_key_env: ANTHROPIC_API_KEY
      max_tokens: 5000
      cost_input_token: 2.0e-06
      cost_output_token: 1.0e-05
      priority: 2
      capabilities:
        - writing
        - creative
        - analysis
        - reasoning
        - general
        - coding
      model_path: null
    llama-3.1-70b:
      provider: vllm
      api_key_env: VLLM_API_KEY
      max_tokens: 131072
      cost_input_token: 0.0
      cost_output_token: 0.0
      priority: 3
      capabilities:
        - reasoning
        - analysis
        - general
        - translation
        - writing
        - coding
      gpu_memory_gb: 160
      model_path: null

  routing_rules:
    - name: code_generation
      condition: "query_type == 'code_generation'"
      models:
        - gpt-5.6-terra
        - claude-sonnet-5
      fallback: mistral-7b
    - name: long_context_analysis
      condition: "query_type == 'analysis' and token_count >= 5000"
      models:
        - claude-sonnet-5
        - gpt-5.6-terra
      fallback: llama-3.1-70b
    - name: premium_tier
      condition: "user_tier == 'premium'"
      models:
        - gpt-5.6-terra
        - claude-sonnet-5
      fallback: mistral-7b
    - name: free_tier
      condition: "user_tier == 'free'"
      models:
        - mistral-7b
        - llama-3.1-70b
      fallback: mistral-7b

inference:
  vllm:
    host: localhost
    port: 8001
    base_url: http://localhost:8001/v1
    api_key_env: VLLM_API_KEY
    timeout: 300
    retries: 3
  openai:
    host: api.openai.com
    port: 443
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    timeout: 60
    retries: 3
  anthropic:
    host: api.anthropic.com
    port: 443
    base_url: https://api.anthropic.com
    api_key_env: ANTHROPIC_API_KEY
    timeout: 60
    retries: 3
  compression:
    enabled: true
    max_context_tokens: 100000
    compression_ratio: 0.3
    method: semantic_graph
  cache:
    enabled: true
    backend: redis
    host: localhost
    port: 6379
    db: 0
    ttl: 3600
    max_size: 10000
  batching:
    enabled: true
    max_batch_size: 32
    max_wait_time_ms: 50

kafka:
  bootstrap_servers: localhost:9092
  topics_file: kafka/topics.json
  replication_factor: 1

  topics:
    queries: llm-queries
    responses: llm-responses
    metrics: llm-metrics
    errors: llm-errors
    dead_letter: llm-dead-letter

  producer:
    acks: all
    retries: 3
    batch_size: 16384
    linger_ms: 5
    compression_type: gzip
    request_timeout_ms: 30000
    max_in_flight: 5
    enable_idempotence: true

  consumer:
    group_id: llm-router-clickhouse-consumer
    auto_offset_reset: earliest
    max_poll_records: 500
    max_poll_interval_ms: 300000
    enable_auto_commit: false
    fetch_min_bytes: 1024
    fetch_max_wait_ms: 500

clickhouse:
  host: localhost
  port: 8123
  native_port: 9000
  username: default
  password_env: CLICKHOUSE_PASSWORD
  database: default
  schema_file: clickhouse/schema.sql
  batch_size: 200
  retry_max: 3
  retry_backoff_base_ms: 1000
  connection_timeout_sec: 10
  send_receive_timeout_sec: 300

  tables:
    query_logs: query_logs
    system_metrics: system_metrics
    model_performance: model_performance
    user_analytics: user_analytics

monitoring:
  enabled: false
  alert_enabled: true
  resource_collector:
    enabled: true
    interval_sec: 15
    disk_path: /
    net_iface: null # null means sum of all interfaces
    gpu_enabled: true

  prometheus_server:
    enabled: false # disabled by default, use mode A /metrics
    port: 9101
    addr: 0.0.0.0
  metrics_expose:
    use_fastapi_mount: true # app.mount("/metrics")

  prometheus:
    server_url: http://localhost:9090
    alert_rules_file: monitoring/alert_rules.yml
    config_file: monitoring/prometheus.yml
    scrape_interval_seconds: 15

  grafana:
    port: 3000
    admin_user: admin
    admin_password_env: GRAFANA_ADMIN_PASSWORD

  alerts:
    error_rate_threshold: 0.05
    latency_p95_threshold_seconds: 2.0
    cpu_usage_threshold: 0.85
    memory_usage_threshold: 0.85

  health_checks:
    interval_seconds: 30
    timeout_seconds: 5

  alert_manager:
    eval_interval_sec: 15
    rules_override: []
    suppress_duplicate_seconds: 300
    history_max_size: 1000
    notifiers:
      stdout: { enabled: true }
      slack:
        enabled: false
        webhook_url_env: SLACK_ALERT_WEBHOOK_URL
        channel: "#alerts"
        mention: "@oncall"
      email: { enabled: false, smtp_host: "", smtp_port: 587, username_env: "", password_env: "", from_addr: "", to_addrs: [] }
      pagerduty: { enabled: false, routing_key_env: PAGERDUTY_ROUTING_KEY }

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
  job_name: LLM Router Analytics
  checkpoint_dir: data/flink-checkpoints
  job_manager:
    host: localhost
    port: 8081
  parallelism: 2
  checkpointing:
    enabled: true
    interval_seconds: 60
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
  async_publish: true
  dlq_local_dir: data/dlq
  flush_interval_ms: 5000
  metrics_report_interval_ms: 10000

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
    latency_sla_seconds:
      free: 10.0
      premium: 5.0
      enterprise: 2.0

  budget:
    cost_budgets:
      free: 0.01
      premium: 0.10
      enterprise: 1.00

  circuit_breaker:
    enabled: false
    failure_threshold: 5
    recovery_timeout_seconds: 30

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
    window_duration_seconds: 3600

  slo_targets:
    availability: 0.999
    latency_p95_seconds: 2.0
    error_rate_max: 0.01

  feedback:
    storage_path: data/feedback

  health_check_interval_seconds: 30

router_mode:
  use_integrated_router: false
"""


class _PrintLogger:
    """Fallback logger used when src.utils.logger cannot be imported yet."""

    def info(self, message):
        print(f"[setup] {message}")

    def debug(self, message):
        print(f"[setup] {message}")

    def warning(self, message):
        print(f"[setup] WARNING: {message}")


class ProjectSetup:
    """P1–P4 project scaffold: directories, templates, venv, dependencies, validation."""

    def __init__(self, project_root: str = ".", logger=None):
        self.project_root = Path(project_root).resolve()
        self.logger = logger or _PrintLogger()

        self.required_dirs = list(REQUIRED_DIRS)
        self.package_dirs = list(PACKAGE_DIRS)

        # P1–P4 template files: relative path -> content builder
        self.required_files = {
            ".gitignore": self._template_gitignore,
            "README.md": self._template_readme,
            "requirements.txt": self._template_requirements,
            "docker/requirements.txt": self._template_docker_requirements,
            "kafka/topics.json": self._template_kafka_topics,
            "clickhouse/schema.sql": self._template_clickhouse_schema,
            "monitoring/prometheus.yml": self._template_prometheus,
            "monitoring/alert_rules.yml": self._template_alert_rules,
            "monitoring/grafana/dashboard.json": self._template_grafana_dashboard,
            "streamlit_ui/config.toml": self._template_streamlit_config,
            ".github/workflows/ci.yml": self._template_ci_workflow,
        }

    # ---------------------------------------------------------------- entry

    def setup_project_environment(self, install_deps: bool = True) -> None:
        """Scaffold entry point: directories -> templates -> configs -> venv -> validation."""
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
        """Initialize the canonical config without overwriting user edits."""
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
            raise ValueError(f"{CONFIG_REL_PATH} is not valid YAML: {exc}") from exc

        if not isinstance(config, dict):
            raise ValueError(
                f"{CONFIG_REL_PATH} must contain a top-level YAML mapping")

        missing = [name for name in EXPECTED_SECTIONS if name not in config]
        if missing:
            raise ValueError(
                f"{CONFIG_REL_PATH} is missing required sections: {missing}")

        self.logger.info(
            f"Configuration structure validation passed: {CONFIG_REL_PATH}")

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
data/prometheus/
data/grafana/

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

A multi-model routing and inference platform for OpenAI, Anthropic and
self-hosted vLLM backends, with optional Kafka and ClickHouse persistence.

## Quick start

```bash
python main.py setup
python main.py start
python main.py health
```

`config/config.yaml` is the complete platform configuration. Pass a partial
override with `python main.py start --config path/to/override.yaml`.

## CLI commands

| Command | Purpose |
|---|---|
| `setup` | Create the project scaffold |
| `start` | Run the platform, add `--dev` for auto reload |
| `health` | Query `/health` of a running instance |
| `deploy` | Generate deployment artifacts (P1 stub) |
| `init-kafka-topics` | Create missing Kafka topics from metadata |

## Layout

See `docs/P1.md`, `docs/P2.md`, `docs/P3.md`, and `docs/P4.md`
for the phased architecture and monitoring setup.
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
      "name": "llm-queries",
      "partitions": 6,
      "replication_factor": 1,
      "retention_ms": 604800000,
      "cleanup_policy": "delete",
      "description": "Query request entry log, generated 1 per POST /route"
    },
    {
      "name": "llm-responses",
      "partitions": 6,
      "replication_factor": 1,
      "retention_ms": 604800000,
      "cleanup_policy": "delete",
      "description": "Inference response details, including response_text and token/cost details"
    },
    {
      "name": "llm-metrics",
      "partitions": 4,
      "replication_factor": 1,
      "retention_ms": 259200000,
      "cleanup_policy": "delete",
      "description": "System/model metric events, granularity 1 request generates 5~10 entries"
    },
    {
      "name": "llm-errors",
      "partitions": 4,
      "replication_factor": 1,
      "retention_ms": 2592000000,
      "cleanup_policy": "delete",
      "description": "Error and exception events, including stacktrace, error_type"
    },
    {
      "name": "llm-dead-letter",
      "partitions": 2,
      "replication_factor": 1,
      "retention_ms": 2592000000,
      "cleanup_policy": "delete",
      "description": "Messages that failed ClickHouse write or consumption exceptions, for offline replay"
    }
  ]
}
"""

    @staticmethod
    def _template_clickhouse_schema() -> str:
        schema_path = Path(__file__).resolve().parents[1] / "clickhouse/schema.sql"
        return schema_path.read_text(encoding="utf-8")

    @staticmethod
    def _template_prometheus() -> str:
        return r"""global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s
  external_labels:
    cluster: llm-router-local
    environment: dev

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["localhost:9093"]

scrape_configs:
  - job_name: llm-router-api
    metrics_path: /metrics
    static_configs:
      - targets: ["localhost:8080"]
        labels: { tier: "api" }
    scrape_interval: 5s

  - job_name: llm-router-inference
    metrics_path: /metrics
    params: { component: ["inference"] }
    static_configs:
      - targets: ["localhost:8080"]
    scrape_interval: 10s

  - job_name: vllm-server
    metrics_path: /metrics
    static_configs:
      - targets: ["localhost:8000"]
    scrape_interval: 10s
    honor_labels: true

  - job_name: kafka-exporter
    static_configs:
      - targets: ["localhost:9308"]
    scrape_interval: 30s

  - job_name: clickhouse-exporter
    static_configs:
      - targets: ["localhost:9116"]
    scrape_interval: 30s

  - job_name: node-exporter
    static_configs:
      - targets: ["localhost:9100"]
    scrape_interval: 15s

  - job_name: prometheus-self
    static_configs:
      - targets: ["localhost:9090"]
    scrape_interval: 15s
"""

    @staticmethod
    def _template_alert_rules() -> str:
        return r"""groups:
  - name: llm-router.rules
    rules:
      - alert: HighErrorRate
        expr: |
          sum by (model_name) (rate(
            {__name__=~"llm_router_inference_requests_total|inference_requests_total", status="error"}[5m]
          ))
          /
          sum by (model_name) (rate(
            {__name__=~"llm_router_inference_requests_total|inference_requests_total"}[5m]
          ))
          > 0.05
        for: 2m
        labels:
          severity: critical
          team: llm-platform
        annotations:
          summary: "High error rate (>5%)"
          description: "model {{ $labels.model_name }} error rate {{ $value | humanizePercentage }}"

      - alert: HighLatencyP95
        expr: histogram_quantile(0.95, sum by (le,model_name) (rate(llm_router_inference_request_duration_seconds_bucket[5m]))) > 5
        for: 5m
        labels: { severity: warning }
        annotations:
          summary: "P95 latency > 5s per model"
          description: "model {{ $labels.model_name }} p95={{ $value }}s"

      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / on (job) group_left() machine_memory_bytes > 0.9
              or llm_router_resource_memory_percent > 90
        for: 2m
        labels: { severity: warning }
        annotations: { summary: "Memory usage > 90%" }

      - alert: HighDiskUsage
        expr: llm_router_resource_disk_percent > 80
        for: 5m
        labels: { severity: warning }
        annotations: { summary: "Disk usage > 80%" }

      - alert: PipelineDLQAccumulating
        expr: increase(llm_router_pipeline_dead_letter_total[10m]) > 10
        for: 3m
        labels: { severity: critical }
        annotations: { summary: "Pipeline dead-letter accumulating" }
"""

    @staticmethod
    def _template_grafana_dashboard() -> str:
        return r"""{
  "id": null,
  "uid": "llm-router-main",
  "title": "LLM Router Platform — Main Overview",
  "description": "P4 main overview: RPS / P95 latency / success rate / cost stats, model distribution, per-model P95 trend, recent error queries (ClickHouse).",
  "tags": ["llm-router", "platform", "mvp"],
  "timezone": "utc",
  "schemaVersion": 38,
  "version": 1,
  "editable": true,
  "graphTooltip": 1,
  "refresh": "30s",
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {
    "refresh_intervals": ["10s", "30s", "1m", "5m", "15m", "1h"]
  },
  "annotations": {
    "list": []
  },
  "links": [],
  "templating": {
    "list": [
      {
        "name": "datasource",
        "label": "Prometheus",
        "type": "datasource",
        "query": "prometheus",
        "current": {"text": "Prometheus", "value": "Prometheus"},
        "hide": 0,
        "refresh": 1,
        "regex": "",
        "options": []
      },
      {
        "name": "model",
        "label": "Model",
        "type": "query",
        "datasource": {"type": "prometheus", "uid": "$datasource"},
        "definition": "label_values(llm_router_inference_requests_total, model_name)",
        "query": {
          "query": "label_values(llm_router_inference_requests_total, model_name)",
          "refId": "PrometheusVariableQueryEditor-VariableQuery"
        },
        "includeAll": true,
        "multi": true,
        "allValue": ".*",
        "current": {"text": ["All"], "value": ["$__all"]},
        "refresh": 2,
        "sort": 1,
        "hide": 0,
        "regex": "",
        "options": []
      },
      {
        "name": "user_tier",
        "label": "User Tier",
        "description": "Queried from the user_tier label on llm_router_inference_requests_by_tier_total (P4 adjustment, parallel tier metric). Filters the per-tier latency panel and the ClickHouse error table.",
        "type": "query",
        "datasource": {"type": "prometheus", "uid": "$datasource"},
        "definition": "label_values(llm_router_inference_requests_by_tier_total, user_tier)",
        "query": {
          "query": "label_values(llm_router_inference_requests_by_tier_total, user_tier)",
          "refId": "PrometheusVariableQueryEditor-VariableQuery"
        },
        "includeAll": true,
        "multi": true,
        "allValue": ".*",
        "current": {"text": ["All"], "value": ["$__all"]},
        "refresh": 2,
        "sort": 1,
        "hide": 0,
        "regex": "",
        "options": []
      },
      {
        "name": "env",
        "label": "Env",
        "description": "Informational only: prometheus.yml external_labels are not attached to locally stored series, so no query filters on env.",
        "type": "custom",
        "query": "dev,stg,prod",
        "includeAll": false,
        "multi": false,
        "current": {"text": "dev", "value": "dev"},
        "hide": 0,
        "options": [
          {"text": "dev", "value": "dev", "selected": true},
          {"text": "stg", "value": "stg", "selected": false},
          {"text": "prod", "value": "prod", "selected": false}
        ]
      },
      {
        "name": "ch_datasource",
        "label": "ClickHouse",
        "type": "datasource",
        "query": "grafana-clickhouse-datasource",
        "current": {},
        "hide": 0,
        "refresh": 1,
        "regex": "",
        "options": []
      }
    ]
  },
  "panels": [
    {
      "id": 2,
      "title": "Requests Per Second (RPS)",
      "type": "stat",
      "gridPos": {"x": 0, "y": 1, "w": 6, "h": 3},
      "datasource": {"type": "prometheus", "uid": "$datasource"},
      "targets": [
        {
          "refId": "A",
          "datasource": {"type": "prometheus", "uid": "$datasource"},
          "expr": "sum(rate(llm_router_inference_requests_total{model_name=~\"$model\"}[1m]))",
          "instant": false,
          "range": true,
          "legendFormat": "rps"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "reqps",
          "decimals": 2,
          "color": {"mode": "fixed", "fixedColor": "green"},
          "thresholds": {
            "mode": "absolute",
            "steps": [{"color": "green", "value": null}]
          }
        },
        "overrides": []
      },
      "options": {
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false},
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "textMode": "auto"
      }
    },
    {
      "id": 3,
      "title": "P95 Response Latency (ms)",
      "type": "stat",
      "gridPos": {"x": 6, "y": 1, "w": 6, "h": 3},
      "datasource": {"type": "prometheus", "uid": "$datasource"},
      "targets": [
        {
          "refId": "A",
          "datasource": {"type": "prometheus", "uid": "$datasource"},
          "expr": "1000 * histogram_quantile(0.95, sum by (le) (rate(llm_router_inference_request_duration_seconds_bucket{model_name=~\"$model\"}[5m])))",
          "instant": false,
          "range": true,
          "legendFormat": "p95"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "ms",
          "decimals": 0,
          "color": {"mode": "thresholds"},
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {"color": "green", "value": null},
              {"color": "yellow", "value": 2000},
              {"color": "red", "value": 5000}
            ]
          }
        },
        "overrides": []
      },
      "options": {
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false},
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "textMode": "auto"
      }
    },
    {
      "id": 4,
      "title": "Success Rate (%)",
      "type": "stat",
      "gridPos": {"x": 12, "y": 1, "w": 6, "h": 3},
      "datasource": {"type": "prometheus", "uid": "$datasource"},
      "targets": [
        {
          "refId": "A",
          "datasource": {"type": "prometheus", "uid": "$datasource"},
          "expr": "100 * sum(rate(llm_router_inference_requests_total{status=\"success\", model_name=~\"$model\"}[5m])) / sum(rate(llm_router_inference_requests_total{model_name=~\"$model\"}[5m]))",
          "instant": false,
          "range": true,
          "legendFormat": "success %"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "percent",
          "decimals": 2,
          "min": 0,
          "max": 100,
          "color": {"mode": "thresholds"},
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {"color": "red", "value": null},
              {"color": "yellow", "value": 95},
              {"color": "green", "value": 99}
            ]
          }
        },
        "overrides": []
      },
      "options": {
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false},
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "textMode": "auto"
      }
    },
    {
      "id": 5,
      "title": "Cumulative Cost (Window)",
      "description": "increase() over the selected time range ($__range, default 6h).",
      "type": "stat",
      "gridPos": {"x": 18, "y": 1, "w": 6, "h": 3},
      "datasource": {"type": "prometheus", "uid": "$datasource"},
      "targets": [
        {
          "refId": "A",
          "datasource": {"type": "prometheus", "uid": "$datasource"},
          "expr": "sum(increase(llm_router_inference_cost_usd_total{model_name=~\"$model\"}[$__range]))",
          "instant": true,
          "range": false,
          "legendFormat": "cost"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "currencyUSD",
          "decimals": 4,
          "color": {"mode": "fixed", "fixedColor": "blue"},
          "thresholds": {
            "mode": "absolute",
            "steps": [{"color": "blue", "value": null}]
          }
        },
        "overrides": []
      },
      "options": {
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false},
        "colorMode": "value",
        "graphMode": "none",
        "justifyMode": "auto",
        "orientation": "auto",
        "textMode": "auto"
      }
    },
    {
      "id": 6,
      "title": "Model Distribution (by Requests)",
      "type": "piechart",
      "gridPos": {"x": 0, "y": 4, "w": 8, "h": 8},
      "datasource": {"type": "prometheus", "uid": "$datasource"},
      "targets": [
        {
          "refId": "A",
          "datasource": {"type": "prometheus", "uid": "$datasource"},
          "expr": "sum by (model_name) (increase(llm_router_inference_requests_total{model_name=~\"$model\"}[$__range]))",
          "instant": true,
          "range": false,
          "legendFormat": "{{model_name}}"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "decimals": 0,
          "color": {"mode": "palette-classic"}
        },
        "overrides": []
      },
      "options": {
        "pieType": "donut",
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": false},
        "displayLabels": ["percent"],
        "legend": {
          "show": true,
          "displayMode": "table",
          "placement": "right",
          "values": ["value", "percent"]
        },
        "tooltip": {"mode": "single", "sort": "none"}
      }
    },
    {
      "id": 7,
      "title": "P95 Latency Trend by Model x User Tier (ms)",
      "description": "One curve per model_name x user_tier, from the parallel tier metric added in the P4 adjustment (docs/p4_adjustment.md). Filtered by the $model and $user_tier variables.",
      "type": "timeseries",
      "gridPos": {"x": 8, "y": 4, "w": 16, "h": 8},
      "datasource": {"type": "prometheus", "uid": "$datasource"},
      "targets": [
        {
          "refId": "A",
          "datasource": {"type": "prometheus", "uid": "$datasource"},
          "expr": "1000 * histogram_quantile(0.95, sum by (le, model_name, user_tier) (rate(llm_router_inference_request_duration_by_tier_seconds_bucket{model_name=~\"$model\", user_tier=~\"$user_tier\"}[$__rate_interval])))",
          "instant": false,
          "range": true,
          "legendFormat": "{{model_name}} / {{user_tier}}"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "ms",
          "color": {"mode": "palette-classic"},
          "custom": {
            "drawStyle": "line",
            "lineInterpolation": "smooth",
            "lineWidth": 2,
            "fillOpacity": 10,
            "showPoints": "never",
            "spanNulls": true,
            "thresholdsStyle": {"mode": "dashed"}
          },
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {"color": "transparent", "value": null},
              {"color": "yellow", "value": 2000},
              {"color": "red", "value": 5000}
            ]
          }
        },
        "overrides": []
      },
      "options": {
        "legend": {
          "showLegend": true,
          "displayMode": "table",
          "placement": "bottom",
          "calcs": ["mean", "max", "lastNotNull"]
        },
        "tooltip": {"mode": "multi", "sort": "desc"}
      }
    },
    {
      "id": 9,
      "title": "ClickHouse Details",
      "type": "row",
      "collapsed": true,
      "gridPos": {"x": 0, "y": 12, "w": 24, "h": 1},
      "panels": [
        {
          "id": 8,
          "title": "Top 10 Recent Error Queries",
          "description": "Requires the grafana-clickhouse-datasource plugin and a ClickHouse datasource selected in the ClickHouse variable.",
          "type": "table",
          "gridPos": {"x": 0, "y": 13, "w": 24, "h": 8},
          "datasource": {"type": "grafana-clickhouse-datasource", "uid": "$ch_datasource"},
          "targets": [
            {
              "refId": "A",
              "datasource": {"type": "grafana-clickhouse-datasource", "uid": "$ch_datasource"},
              "editorType": "sql",
              "queryType": "table",
              "format": 1,
              "rawSql": "SELECT request_received_at, query_id, user_id, user_tier, selected_model AS model_name, latency_ms, error FROM query_logs WHERE status = 'error' AND $__timeFilter(request_received_at) AND user_tier IN (${user_tier:singlequote}) ORDER BY request_received_at DESC LIMIT 10"
            }
          ],
          "fieldConfig": {
            "defaults": {
              "custom": {
                "align": "auto",
                "cellOptions": {"type": "auto"}
              }
            },
            "overrides": [
              {
                "matcher": {"id": "byName", "options": "error"},
                "properties": [
                  {"id": "color", "value": {"mode": "fixed", "fixedColor": "red"}},
                  {"id": "custom.cellOptions", "value": {"type": "color-text"}}
                ]
              },
              {
                "matcher": {"id": "byName", "options": "latency_ms"},
                "properties": [
                  {"id": "unit", "value": "ms"},
                  {"id": "color", "value": {"mode": "thresholds"}},
                  {"id": "thresholds", "value": {"mode": "absolute", "steps": [
                    {"color": "green", "value": null},
                    {"color": "yellow", "value": 2000},
                    {"color": "red", "value": 5000}
                  ]}},
                  {"id": "custom.cellOptions", "value": {"type": "color-text"}}
                ]
              }
            ]
          },
          "options": {
            "showHeader": true,
            "cellHeight": "sm",
            "footer": {"show": false, "reducer": ["sum"], "fields": ""}
          }
        }
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

    print(f"platform config template: {CONFIG_REL_PATH}")
    print("Dry run only, call setup_project_environment() to apply.")
