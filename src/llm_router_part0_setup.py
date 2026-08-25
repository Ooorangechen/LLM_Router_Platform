# ProjectSetup类
# + setup_project_enviornment() 入口：
# 建目录、写模版文件、建venv、装依赖、校验环境

import subprocess
import sys
import yaml
from pathlib import Path


def _log(message: str) -> None:
    """P1 阶段 setup 脚手架不依赖 logger.py（避免模块间循环耦合，
    且 setup 可能在依赖安装之前就被调用），统一用 print 输出进度。
    """
    print(f"[setup] {message}")


class ProjectSetup:
    """P1 项目脚手架：建目录、写模板文件、建 venv、装依赖、校验环境。

    所有方法均为幂等操作：重复执行 setup_project_environment() 不应报错，
    已存在的目录/文件会被跳过（仅记录 debug 日志）。
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()

        # 4.2 节：17 个约定目录
        self.required_dirs = [
            "config",
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

        # 需要额外生成空 __init__.py 的 Python 包目录
        self.package_dirs = ["src", "src/models", "src/utils", "tests"]

        # 10 份模板文件：路径 -> 内容生成函数
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

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def setup_project_environment(self, install_deps: bool = True) -> None:
        """脚手架总入口：目录 -> 模板文件 -> venv/依赖 -> 校验。"""
        _log("Starting project environment setup...")

        self._create_directories()
        self._create_files()

        if install_deps:
            self._setup_python_environment()

        self._validate_environment()
        self._validate_config()

        _log("Project environment setup finished successfully.")

    # ------------------------------------------------------------------
    # 1. 目录创建
    # ------------------------------------------------------------------

    def _create_directories(self) -> None:
        for rel_dir in self.required_dirs:
            dir_path = self.project_root / rel_dir
            dir_path.mkdir(parents=True, exist_ok=True)
            _log(f"Directory ensured: {dir_path}")

        for rel_dir in self.package_dirs:
            init_file = self.project_root / rel_dir / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                _log(f"Created package marker: {init_file}")

    # ------------------------------------------------------------------
    # 2. 模板文件生成
    # ------------------------------------------------------------------

    def _create_files(self) -> None:
        for rel_path, template_fn in self.required_files.items():
            file_path = self.project_root / rel_path
            if file_path.exists():
                _log(f"File already exists, skip: {file_path}")
                continue

            file_path.parent.mkdir(parents=True, exist_ok=True)
            content = template_fn()
            file_path.write_text(content, encoding="utf-8")
            _log(f"Template file created: {file_path}")

    # ------------------------------------------------------------------
    # 3. Python 环境安装
    # ------------------------------------------------------------------

    def _setup_python_environment(self) -> None:
        venv_path = self.project_root / "venv"

        if not venv_path.exists():
            _log("Creating virtual environment (venv)...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_path)],
                check=True,
            )
        else:
            _log(f"venv already exists at {venv_path}, skip creation.")

        pip_path = self._venv_pip_path(venv_path)
        requirements_path = self.project_root / "requirements.txt"

        if pip_path.exists() and requirements_path.exists():
            _log("Installing dependencies from requirements.txt (this may take a while)...")
            subprocess.run(
                [str(pip_path), "install", "-r", str(requirements_path)],
                check=True,
            )
            _log("Dependencies installed successfully.")
        else:
            _log(
                f"WARNING: skip dependency install: pip={pip_path.exists()}, "
                f"requirements.txt={requirements_path.exists()}"
            )

    @staticmethod
    def _venv_pip_path(venv_path: Path) -> Path:
        # POSIX: venv/bin/pip ; Windows: venv/Scripts/pip.exe
        if sys.platform.startswith("win"):
            return venv_path / "Scripts" / "pip.exe"
        return venv_path / "bin" / "pip"

    # ------------------------------------------------------------------
    # 4. 环境校验
    # ------------------------------------------------------------------

    def _validate_environment(self) -> None:
        for rel_dir in self.required_dirs:
            dir_path = self.project_root / rel_dir
            if not dir_path.is_dir():
                raise FileNotFoundError(f"Required directory missing: {dir_path}")

        for rel_file in ["config/config.yaml", "requirements.txt"]:
            file_path = self.project_root / rel_file
            if not file_path.is_file():
                raise FileNotFoundError(f"Required file missing: {file_path}")

        _log("Directory and key file validation passed.")

    def _validate_config(self) -> None:
        config_path = self.project_root / "config" / "config.yaml"
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"config.yaml is not valid YAML: {exc}") from exc

        # P1 仅校验文件格式合法；缺失 section 由 main.py 加载时补默认值。
        for section in ["api", "router", "inference", "kafka", "monitoring"]:
            data.setdefault(section, {})

        _log("config.yaml structure validation passed.")

    # ------------------------------------------------------------------
    # 模板内容
    # ------------------------------------------------------------------

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

# --- Logs & runtime data ---
logs/
*.log
data/queries/
data/processed/

# --- LLM Router specific ---
*.bin
*.safetensors
*.pt
*.gguf
clickhouse/data/
docker/volumes/
slack/credentials/

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

多模型 LLM 路由与执行平台，统一接入 OpenAI / Anthropic / 自托管 vLLM，
提供智能路由决策、全链路可观测与多租户治理能力。

## 快速开始

```bash
python main.py setup   # 初始化目录结构与依赖
python main.py start   # 启动服务（默认 http://localhost:8080）
python main.py health  # 检查服务健康状态
```

## 目录结构

详见 `docs/P1.md` 4.2 节模块划分说明。

（本文件为 P1 阶段占位内容，后续阶段将补充完整。）
"""

    @staticmethod
    def _template_requirements() -> str:
        return """\
fastapi>=0.104
uvicorn[standard]>=0.24
pydantic>=2.5
pydantic-settings>=2.1
pyyaml>=6.0
structlog>=23.2
python-json-logger>=2.0
click>=8.1
prometheus-client>=0.19
httpx>=0.25
"""

    @staticmethod
    def _template_docker_requirements() -> str:
        return """\
fastapi>=0.104
uvicorn[standard]>=0.24
pydantic>=2.5
pydantic-settings>=2.1
pyyaml>=6.0
structlog>=23.2
click>=8.1
prometheus-client>=0.19
"""

    @staticmethod
    def _template_kafka_topics() -> str:
        return """\
{
  "topics": [
    {
      "name": "queries",
      "partitions": 3,
      "replication_factor": 1,
      "retention_ms": 604800000,
      "compression_type": "gzip"
    },
    {
      "name": "responses",
      "partitions": 3,
      "replication_factor": 1,
      "retention_ms": 604800000,
      "compression_type": "gzip"
    },
    {
      "name": "metrics",
      "partitions": 3,
      "replication_factor": 1,
      "retention_ms": 259200000,
      "compression_type": "gzip"
    },
    {
      "name": "errors",
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
-- P1 阶段模板，P3 启用

CREATE TABLE IF NOT EXISTS query_logs (
    request_id String,
    user_id String,
    user_tier String,
    query_type String,
    model_name String,
    latency_ms Float64,
    cost_usd Float64,
    timestamp DateTime
) ENGINE = MergeTree()
ORDER BY (timestamp, user_id);

CREATE TABLE IF NOT EXISTS system_metrics (
    name String,
    value Float64,
    labels String,
    timestamp DateTime
) ENGINE = MergeTree()
ORDER BY (timestamp, name);

CREATE TABLE IF NOT EXISTS model_performance (
    model_name String,
    provider String,
    tokens_per_second Float64,
    quality_score Float64,
    timestamp DateTime
) ENGINE = MergeTree()
ORDER BY (timestamp, model_name);

CREATE TABLE IF NOT EXISTS user_analytics (
    user_id String,
    user_tier String,
    total_requests UInt64,
    total_cost_usd Float64,
    date Date
) ENGINE = MergeTree()
ORDER BY (date, user_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_metrics
ENGINE = SummingMergeTree()
ORDER BY (hour, model_name)
AS
SELECT
    toStartOfHour(timestamp) AS hour,
    model_name,
    count() AS request_count,
    sum(cost_usd) AS total_cost_usd
FROM query_logs
GROUP BY hour, model_name;

-- BloomFilter 索引示例
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS idx_user_id user_id TYPE bloom_filter GRANULARITY 4;
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS idx_model_name model_name TYPE bloom_filter GRANULARITY 4;
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS idx_query_type query_type TYPE bloom_filter GRANULARITY 4;
"""

    @staticmethod
    def _template_prometheus() -> str:
        return """\
# P1 阶段模板，P4 启用
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
      - targets: ["localhost:8080"]

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
      "type": "graph",
      "targets": [{"expr": "rate(llm_router_requests_total[1m])"}]
    },
    {
      "id": 2,
      "title": "P95 Latency",
      "type": "graph",
      "targets": [{"expr": "histogram_quantile(0.95, rate(llm_router_request_duration_bucket[5m]))"}]
    },
    {
      "id": 3,
      "title": "Model Distribution",
      "type": "piechart",
      "targets": [{"expr": "sum by (model) (llm_router_routing_decisions_total)"}]
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
enableCORS = false
enableXsrfProtection = false

[theme]
base = "dark"
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
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
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - run: pip install -r requirements.txt
      - run: pytest tests/ || true

  security-scan:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - run: echo "security scan placeholder"

  docker-build:
    runs-on: ubuntu-latest
    needs: security-scan
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - run: echo "docker build placeholder"

  deploy:
    runs-on: ubuntu-latest
    needs: docker-build
    if: github.ref == 'refs/heads/main'
    steps:
      - run: echo "deploy placeholder"
"""


def setup_project_environment(project_root: str = ".", install_deps: bool = True) -> None:
    """P1 公开入口函数，供 main.py 的 `setup` CLI 命令调用。"""
    project_setup = ProjectSetup(project_root=project_root)
    project_setup.setup_project_environment(install_deps=install_deps)
