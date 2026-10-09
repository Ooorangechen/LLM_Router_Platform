import os
import sys
import json
import time
import signal
import asyncio
import functools
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy
from typing import Optional, Union, Dict, Tuple
import click
import yaml
from dotenv import load_dotenv
from collections import deque
from src.llm_router_part0_setup import setup_project_environment
from src.llm_router_part1_router import ModelRouter
from src.llm_router_part2_inference import InferenceEngine
from src.utils.logger import setup_logging, get_logger
from src.utils.schema import QueryRequest, RoutingDecision, UserTier
import src.utils.metrics

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = str(PROJECT_ROOT / "config/config.yaml")

# Decision: capture the RoutingDecision through a ContextVar wrapper around
# router.route_query instead of changing InferenceEngine.process_query's return
# type (P3 §I.7: connect via Hook, P2 signatures stay frozen). Each HTTP
# request runs in its own asyncio task with its own context copy, so
# concurrent requests never see each other's decision.

_routing_decision: ContextVar[Optional[RoutingDecision]] = ContextVar(
    "routing_decision", default=None)
def _capture_routing_decision(route_query):
    @functools.wraps(route_query)
    async def wrapper(request):
        decision = await route_query(request)
        _routing_decision.set(decision)
        return decision
    return wrapper


try:
    from prometheus_client import make_asgi_app, start_http_server
    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False


class LLMRouterPlatform:
    """
    load config, initialize logging,  initializeing service structure
    """
    def __init__(self, config_path: Union[str, Path] = CONFIG_PATH):
        self.config_path = Path(config_path).resolve()
        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
        self.config = self._load_config()
        self.services = {}
        self._prom_server = None
        self._start_time = time.time()
        self._traffic_window: deque = deque(maxlen=100_000)
        self._setup_logging()
        self.logger = get_logger(__name__)

    def _load_config(self) -> dict:
        """
        Load the platform config, then recursively apply an optional override.
        """
        canonical_path = Path(CONFIG_PATH).resolve()
        canonical = self._read_yaml_mapping(canonical_path)
        if self.config_path == canonical_path:
            return canonical
        overrides = self._read_yaml_mapping(self.config_path)
        return self._deep_merge(canonical, overrides)

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

    def _should_start_prom_server(self, monitoring_cfg, prom_server_cfg) -> bool:
        '''helper function to simply the prom server start'''
        return (
        _PROM_AVAILABLE
        and monitoring_cfg.get("enabled", False)
        and prom_server_cfg.get("enabled", False)
        and self._prom_server is None)
    
    async def _initialize_services(self):
        """Initialize router, optional P3 pipeline, then inference."""
        self.logger.info("Initializing LLM Router Platform services...")

        router = ModelRouter(self.config["router"])
        await router.initialize()
        self.services["router"] = router

        if self.config.get("pipeline", {}).get("enabled", False):
            self.logger.info("Initializing PipelineManager...")
            # pipeline.enabled=False leaves the P2 router object untouched.
            router.route_query = _capture_routing_decision(router.route_query)
            try:
                from src.llm_router_part3_pipeline import PipelineManager
                pipeline = PipelineManager(self.config)
                await pipeline.initialize()
                await pipeline.start_consumer()
                self.services["pipeline"] = pipeline
            except Exception as exc:
                self.logger.warning("Pipeline initialization skipped: %s", exc)
        else:
            self.logger.info("Pipeline disabled by config")

        inference = InferenceEngine(self.config["inference"], router=router)
        await inference.initialize()
        self.services["inference"] = inference
        self.services["cache"] = inference.cache

        # P4: resource collector + alert manager, only when monitoring is on.
        monitoring_cfg = self.config.get("monitoring", {})
        if monitoring_cfg.get("enabled", False):
            from src.llm_router_part4_monitor import (
                SystemResourceCollector, AlertManager)

            collector = SystemResourceCollector(self.config)
            await collector.initialize()
            if collector.enabled:            
                await collector.start()

            self.services["monitor"] = collector
            alert = AlertManager(self.config, self.services)
            await alert.initialize()
            if alert.enabled:                
                await alert.start()
            self.services["alert"] = alert

        # P4 Mode B: standalone Prometheus HTTP server
        monitoring_cfg = self.config.get("monitoring", {})
        prom_server_cfg = monitoring_cfg.get("prometheus_server", {})
        if self._should_start_prom_server(monitoring_cfg, prom_server_cfg):
            port = prom_server_cfg.get("port", 9101)
            addr = prom_server_cfg.get("addr", "0.0.0.0")
            try:
                self._prom_server, _ = start_http_server(port=port, addr=addr)
                self.logger.info("Prometheus HTTP server started on :%d", port)
            except Exception as exc:
                self.logger.warning(
                    "Prometheus HTTP server not started on :%d: %s", port, exc)

        self.logger.info("All services initialized successfully")

    async def _build_health_status(self, services) -> Tuple[str, int, Dict]:
        now = datetime.now(timezone.utc).isoformat()
        svc: Dict[str, Dict] = {}
        for name, service in services.items():
            try:
                probe = getattr(service, "get_health_status", None)
                if probe is None:
                    entry = {"status": "healthy", "message": "registered",
                             "last_check_at": now}
                else:
                    result = await probe()
                    d = result.model_dump(mode="json")
                    entry = {"status": d["status"], "message": d["message"],
                             "last_check_at": d["last_check_at"]}
            except Exception as exc:
                entry = {"status": "unhealthy",
                         "message": f"health check error: {exc}",
                         "last_check_at": now}
            svc[name] = entry

        statuses = [v["status"] for v in svc.values()]
        if any(s == "unhealthy" for s in statuses):
            overall, score = "unhealthy", 2
        elif any(s == "degraded" for s in statuses):
            overall, score = "degraded", 1
        else:
            overall, score = "healthy", 0        # empty services -> healthy

        try:                                     # metric write must not break /health
            src.utils.metrics.HEALTH_METRICS.overall_health_status.set(score)
            for n, v in svc.items():
                src.utils.metrics.HEALTH_METRICS.service_health_info.labels(
                    service_name=n, status=v["status"], message=v["message"]
                ).info({"last_check_at": v["last_check_at"]})
        except Exception as exc:
            self.logger.debug("health metric write failed: %s", exc)

        return overall, score, svc

    def _record_traffic(self, is_error: bool) -> None:
        """record one /route outcome into 60s window"""
        now = time.time()
        self._traffic_window.append((now, is_error))
        cutoff = now - 60.0
        w = self._traffic_window
        while w and w[0][0] < cutoff:
            w.popleft()

    def _traffic_recent_1min(self) -> Dict[str, float]:
        """aggregate 60s window: request count + error rate"""
        cutoff = time.time() - 60.0
        recent = [e for e in self._traffic_window if e[0] >= cutoff]
        requests = len(recent)
        errors = sum(1 for _, is_err in recent if is_err)
        return {
            "requests": requests,
            "error_rate": (errors / requests) if requests else 0.0,
        }

    async def _start_services(self):
        import uvicorn
        app = self._create_fastapi_app()

        api_cfg = self.config.get("api", {})
        host = api_cfg.get("host", "0.0.0.0")
        port = api_cfg.get("port", 8080)

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        self.logger.info(f"Uvicorn running on http://{host}:{port}")
        await server.serve()

    async def _shutdown_services(self):
        """Shut down initialized services in reverse order."""
        self.logger.info(f"shutting down services...")
        for name in reversed(list(self.services.keys())):
            self.logger.info(f"Stopping services: {name}")
            service = self.services[name]
            if name == "pipeline":
                # stop consuming, flush buffers, then close clients.
                await service.stop_consumer()
                await service.flush_all()
                await service.shutdown()
            elif name in ("monitor", "alert"):
                await service.stop()
            elif hasattr(service, "shutdown"):
                await service.shutdown()
        self.services.clear()

        if self._prom_server is not None:
            self._prom_server.shutdown()
            self._prom_server.server_close()
            self._prom_server = None
            self.logger.info("Prometheus HTTP server stopped")
            
        self.logger.info("Shutdown complete")

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}. Shutting down...")
        sys.exit(0)

    def _create_fastapi_app(self):
        from fastapi import FastAPI, HTTPException, Request
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

        # P4 Mode A: mount /metrics on the API port (default on).
        metrics_expose_cfg = self.config.get("monitoring", {}).get("metrics_expose", {})
        if _PROM_AVAILABLE and metrics_expose_cfg.get("use_fastapi_mount", True):
            app.mount("/metrics", make_asgi_app())
            self.logger.info("Mounted /metrics via make_asgi_app")

        @app.get("/health")
        async def health():
            from fastapi.responses import JSONResponse
            overall, score, svc = await self._build_health_status(self.services)
            body = {
                "status": overall,
                "overall_score": score,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": time.time() - self._start_time,
                "services": svc,
            }
            # healthy/degraded -> 200 (monitor scrapes treat 2xx as alive),
            # unhealthy -> 503.
            return JSONResponse(content=body,
                                status_code=503 if score >= 2 else 200)

        @app.get("/status")
        async def status():
            monitoring_cfg = self.config.get("monitoring", {})
            mon_enabled = monitoring_cfg.get("enabled", False)

            router = self.services.get("router")
            inference = self.services.get("inference")
            pipeline = self.services.get("pipeline")
            collector = self.services.get("monitor")
            alert = self.services.get("alert")
            use_integrated = self.config.get("router_mode", {}).get(
                "use_integrated_router", False)
            router_mode = "integrated" if use_integrated else "modular"
            models_count = len(router.models) if router is not None else 0

            providers_ready = []
            if inference is not None:
                try:
                    health = await inference.get_health_status()
                    providers_ready = [
                        name for name, s in health.metadata.get("providers", {}).items()
                        if s.get("status") == "healthy"
                    ]
                except Exception as exc:
                    self.logger.debug("providers_ready probe failed: %s", exc)

            if pipeline is not None:
                pipeline_block = {
                    "enabled": pipeline.enabled,
                    "kafka_ok": pipeline.producer.enabled,
                    "clickhouse_ok": pipeline.ch_writer.enabled,
                }
            else:
                pipeline_block = {
                    "enabled": False, "kafka_ok": False, "clickhouse_ok": False}
                
            if mon_enabled:
                monitoring_block = {
                    "enabled": True,
                    "resource_collector_running": bool(
                        getattr(collector, "_running", False)),
                    "alert_enabled": monitoring_cfg.get("alert_enabled", False),
                    "active_alerts_count": (
                        len(alert.get_active_alerts()) if alert is not None else 0),
                }
            else:
                monitoring_block = {
                    "enabled": False, "resource_collector_running": False,
                    "alert_enabled": False, "active_alerts_count": 0}

            if collector is not None:
                s = collector.get_latest_snapshot()
                resource_block = {
                    "cpu_percent": s.cpu_percent,
                    "memory_percent": s.memory_percent,
                    "disk_percent": s.disk_percent,
                    "gpu_count": s.gpu_count,
                }
            else:
                resource_block = {
                    "cpu_percent": -1, "memory_percent": -1,
                    "disk_percent": 0, "gpu_count": 0}

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": time.time() - self._start_time,
                "router_mode": router_mode,
                "models_count": models_count,
                "providers_ready": providers_ready,
                "pipeline": pipeline_block,
                "monitoring": monitoring_block,
                "resource": resource_block,
                "traffic_recent_1min": self._traffic_recent_1min(),
            }


        @app.get("/analytics")
        async def analytics(window_minutes: int = 5, group_by: str = "model"):
            window_minutes = max(1, min(int(window_minutes), 1440))

            pipeline = self.services.get("pipeline")
            ch_writer = getattr(pipeline, "ch_writer", None) if pipeline else None
            ch_available = (
                ch_writer is not None
                and getattr(ch_writer, "enabled", False)
                and getattr(ch_writer, "client", None) is not None
            )

            if ch_available:
                sql = f"""
                    SELECT
                        selected_model                          AS model_name,
                        count()                                 AS request_count,
                        countIf(status='success') * 1.0 / count() AS success_rate,
                        quantileExact(0.95)(latency_ms)         AS p95_latency_ms,
                        avg(cost_usd)                           AS avg_cost_usd
                    FROM query_logs
                    WHERE request_received_at >= now() - INTERVAL {window_minutes} MINUTE
                    GROUP BY selected_model
                    ORDER BY request_count DESC
                """
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(ch_writer.client.query, sql),
                        timeout=2.0,
                    )
                    rows = [
                        {
                            "model_name": model_name,
                            "request_count": int(request_count),
                            "success_rate": float(success_rate),
                            "p95_latency_ms": float(p95),
                            "avg_cost_usd": float(avg_cost),  # Decimal/Float -> float
                        }
                        for (model_name, request_count, success_rate, p95, avg_cost)
                        in result.result_rows
                    ]
                    return {
                        "window_minutes": window_minutes,
                        "data_source": "clickhouse",
                        "rows": rows,
                    }
                except Exception as exc:
                    self.logger.warning(
                        "Analytics ClickHouse query failed, falling back to "
                        "memory: %s", exc)

            # fallback（router.model_stats）only have request_count / success_rate 
            # p95、per-model cost is placeholder for now
            router = self.services.get("router")
            if router is not None:
                rows = [
                    {
                        "model_name": name,
                        "request_count": stats.get("total_requests", 0),
                        "success_rate": stats.get("success_rate", 0.0),
                        "p95_latency_ms": 0.0,   # placeholder: no latency samples in memory
                        "avg_cost_usd": 0.0,     # placeholder: per-model cost not tracked
                    }
                    for name, stats in router.model_stats.items()
                    if stats.get("total_requests", 0) > 0
                ]
                if rows:
                    return {
                        "window_minutes": window_minutes,
                        "data_source": "memory",
                        "rows": rows,
                    }
                
            return {
                "window_minutes": window_minutes,
                "data_source": "none",
                "rows": [],
                "error": "no analytics backend available",
            }

        @app.post("/admin/reload-config")
        async def reload_config():
            try:
                self.config = self._load_config()
                await self._shutdown_services()
                await self._initialize_services()
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

        @app.get('/admin/alerts/active')
        async def admin_alerts_active(severity: Optional[str] = None):
            alert_manager = self.services.get("alert", None)
            if alert_manager is None or not alert_manager._running:
                return []
            return alert_manager.get_active_alerts(severity=severity)

        @app.get('/admin/alerts/history')
        async def admin_alerts_history(limit: Optional[int] = 100):
            alert_manager = self.services.get("alert", None)
            if alert_manager is None or not alert_manager._running:
                return []
            return alert_manager.get_history(limit)
    

        @app.post("/route")
        async def route_query(request: Request):
            try:
                request_received_at = datetime.now(timezone.utc)
                payload = await request.json()
                query_request = QueryRequest(
                    query=payload.get("query"),
                    user_id=payload.get("user_id"),
                    user_tier=UserTier(payload.get("user_tier", "free")),
                    context=payload.get("context"),
                    max_tokens=payload.get("max_tokens", 512),
                    temperature=payload.get("temperature", 1.0),
                )
                _routing_decision.set(None)
                resp = await self.services["inference"].process_query(
                    query_request)
                routing_decision = _routing_decision.get()

                # P3 Pipeline Hook (non-blocking)
                if (
                    routing_decision is not None
                    and "pipeline" in self.services
                    and self.services["pipeline"].enabled
                ):
                    asyncio.create_task(
                        self.services["pipeline"].publish_pipeline_events(
                            request=query_request,
                            routing_decision=routing_decision,
                            inference_response=resp,
                            received_at=request_received_at,
                        )
                    )
                # End P3 Hook

                if resp.error:
                    raise RuntimeError(resp.error)

                try:
                    src.utils.metrics.SYSTEM_METRICS.requests_total.labels(
                        endpoint="/route", method="POST", status="200"
                    ).inc()
                except Exception as metric_exc:
                    self.logger.debug("requests_total write failed: %s", metric_exc)
                self._record_traffic(is_error=False)
                return {
                    "query_id": str(query_request.request_id),
                    "response": resp.response_text,
                    "model_name": resp.model_name,
                    "tokens": {
                        "input": resp.token_count_input,
                        "output": resp.token_count_output,
                        "total": resp.total_tokens,
                    },
                    "cost_usd": resp.cost_usd,
                    "latency_ms": resp.latency_ms,
                    "cached": resp.cached,
                }
            except Exception as exc:
                try:
                    src.utils.metrics.SYSTEM_METRICS.errors_total.labels(
                        component="api", error_type=type(exc).__name__
                    ).inc()
                except Exception as metric_exc:
                    self.logger.debug("errors_total write failed: %s", metric_exc)
                self._record_traffic(is_error=True)
                raise HTTPException(status_code=500, detail=str(exc))

        return app
        
    async def run(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            await self._initialize_services()
            await self._start_services()
        finally:
            await self._shutdown_services()

app = None

if os.getenv("LLM_ROUTER_DEV_MODE") == "true":
    _dev_platform = LLMRouterPlatform(
        config_path=os.getenv("LLM_ROUTER_CONFIG", CONFIG_PATH)
    )
    app = _dev_platform._create_fastapi_app()

    @app.on_event("startup")
    async def _dev_startup():
        await _dev_platform._initialize_services()

    @app.on_event("shutdown")
    async def _dev_shutdown():
        await _dev_platform._shutdown_services()

@click.group()
def cli():
    pass

@cli.command()
def setup():
    """initialize project enviornment and templates."""
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
    """Start the llm router"""
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


_STATUS_COLORS = {"healthy": "\033[32m", "degraded": "\033[33m", "unhealthy": "\033[31m"}
_COLOR_RESET = "\033[0m"
_EXIT_BY_SCORE = {0: 0, 1: 2, 2: 3}   # healthy=0, degraded=2, unhealthy=3


def _format_uptime(seconds) -> str:
    s = int(seconds or 0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _render_health_text(payload, use_color) -> str:
    status = payload.get("status", "unknown")
    score = payload.get("overall_score", 2)
    lines = [
        "LLM Router Platform Health",
        "==========================",
        f"Overall  : {status.upper():<11} (exit {_EXIT_BY_SCORE.get(score, 3)})",
        f"Uptime   : {_format_uptime(payload.get('uptime_seconds'))}",
        f"Checked  : {payload.get('checked_at', '')}",
        "-------- Services -------",
    ]
    for name, obj in payload.get("services", {}).items():
        st = obj.get("status", "unknown")
        tag = f"[{st.upper()}]".ljust(11)          # pad before coloring to keep alignment
        if use_color and st in _STATUS_COLORS:
            tag = f"{_STATUS_COLORS[st]}{tag}{_COLOR_RESET}"
        lines.append(f"{name:<12} {tag} {obj.get('message', '')}")
    return "\n".join(lines)


@cli.command(name="health")
@click.option("--host", default="localhost", show_default=True)
@click.option("--port", default=8080, show_default=True, type=int)
@click.option("--timeout", default=10.0, show_default=True, type=float)
@click.option("--format", "fmt", type=click.Choice(["text", "json"]),
              default="text", show_default=True)
@click.option("--service", default=None, help="Show only one service")
def health(host, port, timeout, fmt, service):
    """exit 0=healthy / 2=degraded / 3=unhealthy."""
    import httpx
    try:
        resp = httpx.get(f"http://{host}:{port}/health", timeout=timeout)
        payload = resp.json()
    except Exception as exc:
        click.echo(f"Health check failed: {exc}", err=True)
        sys.exit(3)

    if service:
        services = payload.get("services", {})
        payload = {**payload, "services": {
            service: services.get(service,
                                  {"status": "unhealthy", "message": "not found"})}}

    if fmt == "json":
        click.echo(json.dumps(payload, indent=2))
    else:
        click.echo(_render_health_text(payload, use_color=(os.name != "nt")))

    sys.exit(_EXIT_BY_SCORE.get(payload.get("overall_score", 2), 3))


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

async def _init_kafka_topics(config: dict) -> tuple[int, int]:
    """Create missing metadata topics; return (created, already_existing)."""
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from aiokafka.errors import TopicAlreadyExistsError
    kafka_config = config["kafka"]
    topics_path = PROJECT_ROOT / kafka_config["topics_file"]
    with topics_path.open(encoding="utf-8") as file:
        topics = json.load(file)["topics"]

    definitions = [
        NewTopic(
            name=item["name"],
            num_partitions=item["partitions"],
            replication_factor=kafka_config.get("replication_factor", item["replication_factor"]),
            topic_configs={
                "retention.ms": str(item["retention_ms"]),
                "cleanup.policy": item["cleanup_policy"],
            },
        )
        for item in topics
    ]

    admin = AIOKafkaAdminClient(
        bootstrap_servers=kafka_config["bootstrap_servers"],
        request_timeout_ms=10000,
    )
    try:
        await admin.start()
        existing_names = set(await admin.list_topics())
        missing = [topic for topic in definitions if topic.name not in existing_names]
        existing_count = len(definitions) - len(missing)
        if not missing:
            return 0, existing_count

        response = await admin.create_topics(missing)
        created_count = 0
        for result in response.topic_errors:
            name, error_code = result[:2]
            if error_code == 0:
                created_count += 1
            elif error_code == TopicAlreadyExistsError.errno:
                # Another process may have created it after list_topics().
                existing_count += 1
            else:
                raise RuntimeError(f"Kafka topic creation failed: {name}, error_code={error_code}")
        return created_count, existing_count
    finally:
        await admin.close()


@cli.command(name="init-kafka-topics")
def init_kafka_topics():
    """Create Kafka topics defined in kafka/topics.json."""
    try:
        platform = LLMRouterPlatform()
        created, existing = asyncio.run(
            _init_kafka_topics(platform.config)
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    
    if created:
        click.echo(f"Created {created} topics")
    if existing:
        click.echo(f"{existing} topics already exist")


def _clickhouse_writer():
    """Build a ClickHouseWriter for the CLI commands.

    The writer only connects when pipeline.enabled is true, which defaults to
    false; these commands target ClickHouse explicitly, so enable it here.
    """
    from src.llm_router_part3_pipeline import ClickHouseWriter
    config = deepcopy(LLMRouterPlatform().config)
    config.setdefault("pipeline", {})["enabled"] = True
    return ClickHouseWriter(config)


async def _replay_dlq(since):
    writer = _clickhouse_writer()
    await writer.initialize()
    if not writer.enabled:
        raise RuntimeError("ClickHouse not available")
    try:
        return await writer.replay_dlq(since)
    finally:
        await writer.shutdown()


@cli.command(name="replay-dlq")
@click.option("--since", default=None, help="Replay DLQ entries since ISO date (e.g. 2026-08-01)")
def replay_dlq(since):
    """Replay local dead-letter queue to ClickHouse."""
    try:
        since_dt = datetime.fromisoformat(since) if since else None
        succeeded, failed = asyncio.run(_replay_dlq(since_dt))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Replayed {succeeded + failed} entries, {succeeded} succeeded, {failed} failed")


async def _init_clickhouse_schema():
    # initialize() connects and runs schema.sql once; it keeps the counts.
    writer = _clickhouse_writer()
    await writer.initialize()
    if not writer.enabled:
        raise RuntimeError("ClickHouse not available")
    try:
        return writer.schema_result
    finally:
        await writer.shutdown()


@cli.command(name="init-clickhouse-schema")
def init_clickhouse_schema():
    """Execute clickhouse/schema.sql DDL."""
    try:
        succeeded, failed = asyncio.run(_init_clickhouse_schema())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if failed:
        raise click.ClickException(
            f"Executed {succeeded + failed} DDL statements, {failed} errors")
    click.echo(f"Executed {succeeded} DDL statements, no errors")

if __name__ == "__main__":
    cli()
