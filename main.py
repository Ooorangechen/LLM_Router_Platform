import os
import sys
import json
import time
import signal
import asyncio
from pathlib import Path
from copy import deepcopy
import click
import yaml

from src.llm_router_part0_setup import setup_project_environment
from src.utils.logger import setup_logging, get_logger
import src.utils.metrics  

DEFAULTS_CONFIG_PATH = "config/defaults.yaml"
CONFIG_PATH = "config/config.yaml"


try:
    from prometheus_client import start_http_server
    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False


class LLMRouterPlatform:
    """
    P1: 
        load config, 
        initialize logging, 
        initializeing service structure
    """

    def __init__(self, config_path: str = CONFIG_PATH, defaults_config_path: str = DEFAULTS_CONFIG_PATH,):
        self.config_path = Path(config_path)
        self.defaults_config_path = Path(defaults_config_path)
        self.config = self._load_config()
        self.services = {}
        self._setup_logging()
        self.logger = get_logger(__name__)

    def _load_config(self) -> dict:
        """
        Load canonical defaults, then recursively apply user overrides.
        """
        defaults = self._read_yaml_mapping(
            self.defaults_config_path
        )
        overrides = self._read_yaml_mapping(
            self.config_path
        )

        return self._deep_merge(defaults, overrides)

    @staticmethod
    def _read_yaml_mapping(path: Path) -> dict:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {path}")
            sys.exit(1)
        except yaml.YAMLError as exc:
            print(f"Invalid YAML in config file {path}: {exc}")
            sys.exit(1)

        if data is None:
            data = {}

        if not isinstance(data, dict):
            print(f"Config file must contain a YAML mapping: {path}")
            sys.exit(1)

        return data

    @staticmethod
    def _deep_merge(defaults: dict, overrides: dict) -> dict:
        result = deepcopy(defaults)

        for key, override_value in overrides.items():
            default_value = result.get(key)

            if isinstance(default_value, dict) and isinstance(
                override_value, dict
            ):
                result[key] = LLMRouterPlatform._deep_merge(
                    default_value,
                    override_value,
                )
            else:
                result[key] = deepcopy(override_value)

        return result

    def _setup_logging(self):
        log_cfg = self.config.get("logging", {})
        kwargs = {}
        if "level" in log_cfg:
            kwargs["log_level"] = log_cfg["level"]
        if "file" in log_cfg:
            kwargs["log_file"] = log_cfg["file"]
        if "max_bytes" in log_cfg:
            kwargs["max_bytes"] = log_cfg["max_bytes"]
        if "backup_count" in log_cfg:
            kwargs["backup_count"] = log_cfg["backup_count"]
        if "console_output" in log_cfg:
            kwargs["console_output"] = log_cfg["console_output"]
        if "structured_logs" in log_cfg:
            kwargs["structured_logs"] = log_cfg["structured_logs"]
        if "format" in log_cfg:
            kwargs["log_format"] = log_cfg["format"]
        if "json_format" in log_cfg:
            kwargs["json_format"] = log_cfg["json_format"]
        setup_logging(**kwargs)

    async def _initialize_services(self):
        """ P1 phase only do the log"""
        self.logger.info("Initializing LLM Router Platform services...")
        self.logger.info("All services initialized successfully")

    async def _start_services(self):
        import uvicorn
        app = self._create_fastapi_app()

        if _PROM_AVAILABLE:
            try:
                prom_port = self.config.get("monitoring", {}).get("prometheus_port", 8000)
                start_http_server(prom_port)
            except Exception as e:
                self.logger.warning(f"Prometheus metrics server failed to start: {e} ")

        api_cfg = self.config.get("api", {})
        host = api_cfg.get("host", "0.0.0.0")
        port = api_cfg.get("port", 8080)

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        self.logger.info(f"Uvicorn running on http://{host}:{port}")
        await server.serve()

    async def _shutdown_services(self):
        """ 
        Shut down services in reversed order.
        Only logging in P1
        """
        self.logger.info(f"shutting down services...")
        for name in reversed(list(self.services.keys())):
            self.logger.info(f"Stopping services: {name}")
        self.logger.info("Shutdown complete")

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}. Shutting down...")
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._shutdown_services())
        except RuntimeError:
            pass
        sys.exit(0)

    def _create_fastapi_app(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        app = FastAPI(
            title="LLM Router & Execution Platform",
            description="Production-grade multi-model deployment system with adaptive routing",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        cors_origins = self.config.get("api", {}).get("cors_origins", ["*"])
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/health")
        async def health():
            try:
                service_status = {}
                for name, service in self.services.items():
                    if hasattr(service, "is_healthy"):
                        service_status[name] = service.is_healthy()
                    elif hasattr(service, "get_health_status"):
                        service_status[name] = service.get_health_status()
                    else:
                        service_status[name] = {"healthy": True}

                # §3.6.2 requires all_ok = all services healthy
                # （bool / dict）,  all(values()) will return True when empty dict 
                # even {"healthy": False} will count as healthy.
                def _is_ok(v):
                    return v.get("healthy", False) if isinstance(v, dict) else bool(v)

                all_ok = all(_is_ok(v) for v in service_status.values()) \
                    if service_status else True
                return {
                    "status": "healthy" if all_ok else "degraded",
                    "services": service_status,
                    "timestamp": time.time(),
                }
            except Exception as exc:
                return {
                    "status": "degraded",
                    "error": str(exc),
                    "timestamp": time.time(),
                }

        @app.get("/status")
        async def status():
            return {
                "system": {},
                "router_mode": self.config.get("router_mode", {}),
            }

        @app.get("/analytics")
        async def analytics():
            return {"system": {}}

        @app.post("/admin/reload-config")
        async def reload_config():
            try:
                self.config = self._load_config()
                return {"status": "config_reloaded"}
            except BaseException as exc:
                # §3.5 要求 _load_config 遇到坏配置时 sys.exit(1)，而 sys.exit 抛的是
                # SystemExit —— 它继承 BaseException 而非 Exception，用 except Exception
                # 抓不住，进程会被直接杀掉。这里用 BaseException 才能满足「异常 500」。
                raise HTTPException(status_code=500, detail=str(exc))

        @app.get("/admin/services")
        async def admin_services():
            return {
                "services": list(self.services.keys()),
                "count": len(self.services),
            }

        return app
        
    async def run(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        await self._initialize_services()
        await self._start_services()

app = None

if os.getenv("LLM_ROUTER_DEV_MODE") == "true":
    _dev_platform = LLMRouterPlatform(
        config_path=os.getenv("LLM_ROUTER_CONFIG", CONFIG_PATH)
    )
    app = _dev_platform._create_fastapi_app()

    @app.on_event("startup")
    async def _dev_startup():
        await _dev_platform._initialize_services()

@click.group()
def cli():
    pass

@cli.command()
def setup():
    """初始化项目目录结构与模板文件。"""
    try:
        setup_project_environment()
        click.echo("Setup completed.")
    except Exception as exc:
        click.echo(f"Setup failed: {exc}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--config", "config_path", default=CONFIG_PATH,
              show_default=True, help="Config path")
@click.option("--dev", is_flag=True, default=False,
              help="Dev: auto restart after code changes")
def start(config_path, dev):
    if dev:
        import uvicorn
        os.environ["LLM_ROUTER_DEV_MODE"] = "true"
        os.environ["LLM_ROUTER_CONFIG"] = config_path

        platform = LLMRouterPlatform(config_path)
        port = platform.config.get("api", {}).get("port", 8080)
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
        return

    platform = LLMRouterPlatform(config_path)
    try:
        asyncio.run(platform.run())
    except KeyboardInterrupt:
        click.echo("Shutting down gracefully...")


@cli.command()
@click.option("--service", default=None, help="Checking the health status of one service")
def health(service):
    import httpx
    try:
        resp = httpx.get("http://localhost:8080/health", timeout=5.0)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        click.echo(f"Health check failed: {exc}", err=True)
        sys.exit(1)

    if service:
        result = payload.get("services", {}).get(service, {"error": "not found"})
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(json.dumps(payload, indent=2))


@cli.command()
@click.option("--output-dir", "output_path", default="deploy",
              show_default=True, help="Deploy the required templates.")
def deploy(output_path):
    try:
        base = Path(output_path)
        base.mkdir(parents=True, exist_ok=True)
        (base / "docker").mkdir(exist_ok=True)
        (base / "k8s").mkdir(exist_ok=True)
        click.echo(f"Deploy scaffold created under {base}/ (P1 stub).")
    except Exception as exc:
        click.echo(f"Deploy failed: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()