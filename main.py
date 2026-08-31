### CLI 入口 (click group)
### + FastAPI应用创建 + 服务生命周期编排
### _setup_logging, _load_config, _create_fastapi_app
### _signal_handler, _initialize, _start, _shutdown 
import sys
import time
import asyncio

import click
import yaml
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.llm_router_part0_setup import setup_project_environment
from src.utils.logger import setup_logging, get_logger

DEFAULT_CONFIG_PATH = "config/config.yaml"

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

    def __init__(self, config_path:str=DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()
        self.services = {}
        self._setup_logging()
        self.logger = get_logger(__name__)

    def _load_config(self) -> dict:
        """
        Read from config.yaml,
        Use default values when missing sections. 
        """
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Invalid YAML in config file: {e}")
            sys.exit(1)

        config.setdefault("api", {
            "host": "0.0.0.0",
            "port": 8080,
            "log_level": "info",
            "cors_origins": ["*"],
        })
        config.setdefault("router", {
            "default_model": "mistral-7b",
            "routing_strategy": "intelligent",
            "models": {},
            "routing_rules": [],
        })
        config.setdefault("adapters", {
            "enabled": False,
            "registry_path": "data/adapters/registry.json",
        })
        config.setdefault("optimization", {
            "enabled": False,
            "kv_cache_size_gb": 8,
            "max_batch_size": 32,
            "max_wait_ms": 100,
            "flash_attn": True,
            "tensorrt": False,
        })
        config.setdefault("quality", {
            "monitor": {"enabled": False, "window_size": 100},
            "slo_targets": {
                "availability": 0.999,
                "latency_p95_ms": 2000,
                "error_rate_max": 0.01,
            },
        })
        config.setdefault("policies", {
            "quota": {
                "tier_quotas": {
                    "free": {"daily": 100, "hourly": 10},
                    "premium": {"daily": 1000, "hourly": 100},
                    "enterprise": {"daily": 10000, "hourly": 1000},
                }
            },
            "sla": {
                "latency_slas": {
                    "free": "10s",
                    "premium": "5s",
                    "enterprise": "2s",
                }
            },
            "budget": {
                "cost_budgets": {
                    "free": 0.01,
                    "premium": 0.10,
                    "enterprise": 1.00,
                }
            },
        })
        config.setdefault("router_mode", {
            "use_integrated_router": False,
        })

        return config

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
        setup_logging(**kwargs)

    async def _initialize_services(self):
        """ P1 phase only do the log"""
        self.logger.info("Initializing LLM Router Platform services...")
        self.logger.info("All services initialized successfully")

    async def _start_services(self):
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
        sys.exit(0)


    #########
    def _create_fastapi_app(self):
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
                        service_status[name] = True

                all_ok = all(service_status.values()) if service_status else True
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
            except Exception as exc:
                return {"status": "error", "error": str(exc)}

        @app.get("/admin/services")
        async def admin_services():
            return {
                "services": list(self.services.keys()),
                "count": len(self.services),
            }

        return app
