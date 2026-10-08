from pydantic import BaseModel, Field
from datetime import datetime, timezone
from uuid import UUID, uuid4
import time
from typing import List, Dict, Any, Optional, Tuple, Deque
import asyncio
from collections import deque
from src.utils.logger import get_logger
from src.utils.metrics import INFERENCE_METRICS, ALERT_METRICS
import os
import platform # consider both windows / linux, mac
from abc import abstractmethod, ABC
import math 
import httpx

try:
    import psutil
except Exception:
    psutil = None


class ResourceSnapshot(BaseModel):
    '''Resource snapshot, returned directly by /status'''
    timestamp: datetime # in UTC
    cpu_percent: float 
    memory_percent: float
    memory_used_bytes: int
    memory_total_bytes: int
    disk_percent: float 
    disk_used_bytes: int
    disk_total_bytes: int
    gpu_count: int
    gpu_utilization_percent: List[float]
    gpu_memory_percent: List[float] 
    gpu_memory_used_mb: List[float]

    net_io_recv_bytes_per_sec: float 
    net_io_send_bytes_per_sec: float 

    process_count: int
    open_fds_count: int 
    uptime_seconds: float 

class HealthStatus(BaseModel):
    """Sub-service health status, reused by /health"""
    service_name: str
    status: str # healthy, degraded, unhealthy
    message: str
    last_check_at: datetime # UTC
    metadata: Dict[str, Any] = {}


class SystemResourceCollector:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("monitor")
        self.snapshot: Optional[ResourceSnapshot] = None
        self._running: bool = False

        monitor_config = config.get("monitoring",{})
        resource_collector_config = monitor_config.get("resource_collector", {})
        self.enabled: bool = resource_collector_config.get("enabled", False)
        self.interval_sec: int = resource_collector_config.get("interval_sec", 15)
        self.gpu_enabled: bool = resource_collector_config.get("gpu_enabled", False)
        self.disk_path: str = resource_collector_config.get("disk_path", "/")
        self.net_iface: Optional[str] = resource_collector_config.get("net_iface", None)

        self._task = None
        self._process_start_time = None
        self._last_net_io = None 

        self.systemm = platform.system()
        self._gputil_module = None

    async def initialize(self) -> None:
        if psutil is None:
            self.enabled = False
            self.logger.info("Failed on import psutil, collector turned off")
            return

        if self.gpu_enabled:
            try:
                import GPUtil as GPU
                self._gputil_module = GPU
            except Exception:
                self._gputil_module = None

        self._process_start_time = time.time()

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._collect_loop())

    async def _collect_loop(self) -> None:
        while self._running:
            self._collect_once()
            await asyncio.sleep(self.interval_sec)

    def _collect_once(self) -> ResourceSnapshot:
        # any item failure fills 0 (or empty list) and must not interrupt overall collection
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_used_bytes = 0
        memory_total_bytes = 0
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            memory_percent = ram.percent
            memory_used_bytes = ram.used
            memory_total_bytes = ram.total
        except Exception as e:
            self.logger.warning("Resource collector item failed (cpu/memory): %s", e)
                                                            
        disk_percent = 0.0
        disk_used_bytes = 0
        disk_total_bytes = 0
        try:
            disk = psutil.disk_usage(self.disk_path)
            disk_percent = disk.percent
            disk_used_bytes = disk.used
            disk_total_bytes = disk.total
        except Exception as e:
            self.logger.warning("Resource collector item failed (disk): %s", e)

        gpu_count = 0
        gpu_utilization_percent: List[float] = []
        gpu_memory_used_mb: List[float] = []
        gpu_memory_percent: List[float] = []
        try:
            if self._gputil_module is not None:
                gpus = self._gputil_module.getGPUs()
                gpu_count = len(gpus)
                for gpu in gpus:
                    gpu_utilization_percent.append(gpu.load * 100)
                    gpu_memory_used_mb.append(gpu.memoryUsed)
                    gpu_memory_percent.append(gpu.memoryUtil * 100)
        except Exception as e:
            self.logger.warning("Resource collector item failed (gpu): %s", e)

        net_io_recv_bytes_per_sec = 0.0
        net_io_send_bytes_per_sec = 0.0
        try:
            if self.net_iface:
                counters = psutil.net_io_counters(pernic=True)
                now = counters[self.net_iface]
            else:
                now = psutil.net_io_counters()

            if self._last_net_io is not None:
                net_io_recv_bytes_per_sec = (now.bytes_recv - self._last_net_io.bytes_recv) / self.interval_sec
                net_io_send_bytes_per_sec = (now.bytes_sent - self._last_net_io.bytes_sent) / self.interval_sec
            self._last_net_io = now
        except Exception as e:
            self.logger.warning("Resource collector item failed (net_io): %s", e)

        process_count = 0
        open_fds_count = 0
        try:
            p = psutil.Process(os.getpid())
            process_count = len(p.children(recursive=True))
            if self.systemm == "Windows":
                open_fds_count = p.num_handles()
            else:
                open_fds_count = p.num_fds()
        except Exception as e:
            self.logger.warning("Resource collector item failed (process): %s", e)

        uptime_seconds = 0.0
        try:
            uptime_seconds = time.time() - self._process_start_time
        except Exception as e:
            self.logger.warning("Resource collector item failed (uptime): %s", e)

        snapshot = ResourceSnapshot(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_bytes=memory_used_bytes,
            memory_total_bytes=memory_total_bytes,
            disk_percent=disk_percent,
            disk_used_bytes=disk_used_bytes,
            disk_total_bytes=disk_total_bytes,
            gpu_count=gpu_count,
            gpu_utilization_percent=gpu_utilization_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_percent=gpu_memory_percent,
            net_io_send_bytes_per_sec=net_io_send_bytes_per_sec,
            net_io_recv_bytes_per_sec=net_io_recv_bytes_per_sec,
            process_count=process_count,
            open_fds_count=open_fds_count,
            uptime_seconds=uptime_seconds,
        )
        self.snapshot = snapshot
        return snapshot


    def get_latest_snapshot(self) -> ResourceSnapshot:
        # return the most recent snapshot, if not collected, returns an empty snapshot with CPU = -1. memory = -1
        if self.snapshot is not None:
            return self.snapshot

        return ResourceSnapshot(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=-1,
            memory_percent=-1,
            memory_used_bytes=0,
            memory_total_bytes=0,
            disk_percent=0,
            disk_used_bytes=0,
            disk_total_bytes=0,
            gpu_count=0,
            gpu_utilization_percent=[],
            gpu_memory_used_mb=[],
            gpu_memory_percent=[],
            net_io_recv_bytes_per_sec=0,
            net_io_send_bytes_per_sec=0,
            process_count=0,
            open_fds_count=0,
            uptime_seconds=0,
        )
    
    async def get_health_status(self) -> HealthStatus:
        """Operational health infomation"""
        now = datetime.now(timezone.utc)
        meta: Dict[str, Any] = {
            "enabled": self.enabled,
            "running": self._running,
            "interval_sec": self.interval_sec
        }

        if psutil is None:
            status, message = "degraded", "psutil unavailable, resource metrics off"
        elif not self.enabled:
            status, message = "healthy", "resource collector disabled by config"
        elif not self._running:
            status, message = "degraded", "collector not started"
        elif self.snapshot is None:
            status, message = "degraded", "awaiting first sample"
        else:
            age = (now - self.snapshot.timestamp).total_seconds()
            meta["snapshot_age_sec"] = round(age, 1)
            meta["cpu_percent"] = self.snapshot.cpu_percent
            meta["memory_percent"] = self.snapshot.memory_percent
            if age > 3 * self.interval_sec:
                status = "degraded"
                message = f"collector stalled, last sample {age:.0f}s ago"
            else:
                status = "healthy"
                message = (f"running (cpu={self.snapshot.cpu_percent:.1f}%, "
                       f"mem={self.snapshot.memory_percent:.1f}%)")
        return HealthStatus(service_name="monitor", status=status, message=message, last_check_at=now,metadata=meta)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task


## Alert Manager

class AlertRule(BaseModel):
    name: str
    expr_lambda_src: str
    threshold: float
    duration_seconds: int
    severity: str
    description: str
    enabled: bool
    extra_labels: Dict[str, str] = {}

class AlertRecord(BaseModel):
    alert_id: UUID
    rule_name: str
    severity: str
    status: str
    value: float
    threshold: float
    fired_at: datetime
    resolved_at: Optional[datetime] = None  
    description: str
    labels: Dict[str, str] = {}            

class AlertManager:
    def __init__(self, config: Dict, services_ref: Dict[str, Any]) -> None:
        self.logger = get_logger("alert")
        self.services_ref = services_ref

        monitor_config = config.get("monitoring", {})
        alert_manager_config = monitor_config.get("alert_manager", {})
        self.enabled: bool = monitor_config.get("alert_enabled", False)

        self._interval_sec: int = alert_manager_config.get("eval_interval_sec", 15)
        self._suppress_duplicate_seconds: int = alert_manager_config.get("suppress_duplicate_seconds", 300)
        self._history_max_size: int = alert_manager_config.get("history_max_size", 1000)
        self._rules_override: List[Any] = alert_manager_config.get("rules_override", []) or []
        self._notifier_config: Dict[str, Any] = alert_manager_config.get("notifiers", {})

        # error rate 5 min window: ts, total_cumulative, errors_cumulative
        self._error_window_sec: int = alert_manager_config.get("error_rate_window_sec", 300)
        self._error_samples: Deque[Tuple[float, float, float]] = deque()

        self.rules: List[AlertRule] = []
        self._active: Dict[str, AlertRecord] = {}  # key = rule.name
        self._history: Deque[AlertRecord] = deque(maxlen=self._history_max_size)
        self._notifiers: Dict[str, BaseNotifier] = {}

        # record event per-rule streak hitting counter
        self._hit_strek: Dict[str, int] = {}
        self._miss_strek: Dict[str, int] = {}
        
        self._last_notify_at: Dict[str, float] = {}

        self._running: bool = False
        self._task = None

        self._prometheus_url = monitor_config.get("prometheus", {}).get("server_url", "http://localhost:9090")


    async def initialize(self) -> None:
        if self._rules_override:
            self.rules = [
                r if isinstance(r, AlertRule) else AlertRule(**r)
                for r in self._rules_override
            ]
        else:
            self.rules = self._load_default_rules()

        cfg = self._notifier_config or {}

        if cfg.get("stdout", {}).get("enabled", True):
            self._notifiers["stdout"] = StdoutNotifier()

        slack = SlackWebhookNotifier(cfg.get("slack", {}))
        if slack.enabled:
            self._notifiers["slack"] = slack

        email = EmailNotifier(cfg.get("email", {}))
        if email.enabled:
            self._notifiers["email"] = email

        pagerduty = PagerDutyNotifier(cfg.get("pagerduty", {}))
        if pagerduty.enabled:
            self._notifiers["pagerduty"] = pagerduty

        self.logger.info(
            "AlertManager initialized (%d rules, notifiers=%s, alert_enabled=%s)",
            len(self.rules), list(self._notifiers.keys()), self.enabled,
        )

    def _load_default_rules(self) -> List[AlertRule]:
        return [
            AlertRule(
                name="HighErrorRate",
                expr_lambda_src="inference error_rate(5m) > 0.05",
                threshold=0.05,
                duration_seconds=120,
                severity="critical",
                description="Inference error rate over last 5m exceeds 5%",
                enabled=True,
            ),
            AlertRule(
                name="HighLatencyP95Sec",
                expr_lambda_src="inference p95 latency > 5s",
                threshold=5.0,
                duration_seconds=300,
                severity="warning",
                description="Inference P95 latency exceeds 5 seconds",
                enabled=True,
            ),
            AlertRule(
                name="HighMemoryPercent",
                expr_lambda_src="snapshot.memory_percent > 90",
                threshold=90.0,
                duration_seconds=120,
                severity="warning",
                description="System memory usage exceeds 90%",
                enabled=True,
            ),
            AlertRule(
                name="HighDiskPercent",
                expr_lambda_src="snapshot.disk_percent > 80",
                threshold=80.0,
                duration_seconds=300,
                severity="warning",
                description="System disk usage exceeds 80%",
                enabled=True,
            ),
        ]

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._eval_loop())

    async def _eval_loop(self) -> None:
        while self._running:
            for rule in self.rules:
                if not rule.enabled:
                    continue
                try:
                    triggered, value = await self._evaluate_rule(rule)
                    result = self._record_event(rule, triggered, value, rule.extra_labels)
                    if result is not None:
                        record, action = result
                        if self._should_notify(record, action):
                            await self._notify_all(record, action)
                except Exception as e:
                    self.logger.warning("Alert rule '%s' evaluation failed: %s", rule.name, e)
            await asyncio.sleep(self._interval_sec)

    async def _evaluate_rule(self, rule: AlertRule) -> Tuple[bool, float]:
        name = rule.name
        try:
            if name == "HighErrorRate":
                value = self._approx_error_rate()
            elif name == "HighLatencyP95Sec":
                value = await self._p95_latency_seconds()
            elif name == "HighMemoryPercent":
                value = self._snapshot_value("memory_percent")
            elif name == "HighDiskPercent":
                value = self._snapshot_value("disk_percent")
            else:
                return (False, 0.0)
        except Exception as e:
            self.logger.warning("Rule '%s' value extraction failed: %s", name, e)
            return (False, 0.0)

        if value is None:
            return (False, 0.0)
        return (value > rule.threshold, value)

    def _inference_request_totals(self) -> Tuple[float, float]:
        total = errors = 0.0
        try:
            for family in INFERENCE_METRICS.requests_total.collect():
                for s in family.samples:
                    if not s.name.endswith("_total"):   # skip _created
                        continue
                    total += s.value
                    if s.labels.get("status") == "error":
                        errors += s.value
        except Exception as e:
            self.logger.warning("read INFERENCE_METRICS failed: %s", e)
        return (total, errors)
    
    def _approx_error_rate(self) -> Optional[float]:
        total, errors = self._inference_request_totals()

        now = time.time()
        self._error_samples.append((now,total,errors))
        cutoff = now - self._error_window_sec
        while self._error_samples and self._error_samples[0][0] < cutoff:
            self._error_samples.popleft()

        _, base_total, base_errors = self._error_samples[0]
        d_total = total - base_total
        d_errors = errors - base_errors
        if d_total <= 0:
            return None
        return d_errors / d_total

    async def _p95_latency_seconds(self) -> Optional[float]:
        query = ("histogram_quantile(0.95, sum by (le) (rate(llm_router_inference_request_duration_seconds_bucket[5m])))")
        try:
            async with  httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self._prometheus_url}/api/v1/query",
                params={"query": query})
            if resp.status_code != 200:
                return None
            payload = resp.json()
            if payload.get("status") != "success":
                return None
            result = payload["data"]["result"]
            value = float(result[0]["value"][1])
            if value != value:
                return None
            return value
        except Exception as e:
            self.logger.warning("Prometheus P95 query failed: %s", e)
            return None 
        
    def _snapshot_value(self, field:str) -> Optional[float]:
        collector = self.services_ref.get("monitor")
        if collector is None:
            return None
        snap = collector.get_latest_snapshot()
        if snap is None or snap.cpu_percent < 0 or snap.memory_percent < 0:
            return None
        val = getattr(snap, field, None)
        return None if val is None else float(val)

    def _record_event(self, rule, firing: bool, value, labels):
        rule_name = rule.name
        now = datetime.now(timezone.utc)
        need = max(1, math.ceil(rule.duration_seconds / self._interval_sec)) # the needed counter
        active = self._active.get(rule_name)

        if firing: 
            self._hit_strek[rule_name] = self._hit_strek.get(rule_name, 0) + 1
            self._miss_strek[rule_name] = 0 

            if active is not None: 
                # already active, duplicate hit, no notifification
                ALERT_METRICS.alerts_total.labels(rule_name=rule_name, severity=rule.severity, action="deduplicated").inc()
                return None
                
            if self._hit_strek[rule_name] < need:
                return None 

            # when first firing
            record = AlertRecord(
                alert_id=uuid4(),rule_name=rule_name, severity=rule.severity,
                status="firing", value=value, threshold= rule.threshold, 
                fired_at=now, description=rule.description, labels=labels
            )
            self._active[rule_name] = record
            self._history.append(record)
            ALERT_METRICS.alerts_total.labels(rule_name=rule_name, severity=rule.severity, action="fired").inc()
            ALERT_METRICS.active_alerts.labels(severity=rule.severity).inc()
            return (record, "firing")

        else:
            self._miss_strek[rule_name] = self._miss_strek.get(rule_name, 0) + 1
            self._hit_strek[rule_name] = 0

            if active is None:
                return None

            if self._miss_strek[rule_name] < need:
                return None 

            # active and missing conditions continuously within duration, resolved
            record = self._active.pop(rule_name)
            record.status = "resolved"
            record.resolved_at = now
            self._history.append(record)
            # recovered -> clear the suppression window so a new fire notifies at once
            self._last_notify_at.pop(rule_name, None)
            ALERT_METRICS.alerts_total.labels(
                rule_name=rule_name, severity=rule.severity, action="resolved").inc()
            ALERT_METRICS.active_alerts.labels(severity=record.severity).dec()
            return (record, "resolved")

    def _should_notify(self, record: AlertRecord, action: str) -> bool:
        if action != "firing":
            return True
        now = time.monotonic()
        last = self._last_notify_at.get(record.rule_name)
        if last is not None and (now - last) < self._suppress_duplicate_seconds:
            return False
        self._last_notify_at[record.rule_name] = now
        return True

    async def _notify_all(self, record:AlertRecord, action:str):
        for channel, notifier in self._notifiers.items():
            try:
                ok = await notifier.send_alert(record, action)
            except Exception as e:
                ok = False
                self.logger.warning("Notifier '%s' failed on alert '%s': %s", channel, record.rule_name, e)
            status = 'success' if ok else 'failure'
            try:
                ALERT_METRICS.notifications_total.labels(channel=channel, status=status).inc()
            except Exception:
                pass

    def get_active_alerts(self, severity: Optional[str] = None) -> List[AlertRecord]:
        records = list(self._active.values())
        if severity:
            records = [r for r in records if r.severity == severity]
        return records 

    def get_history(self, limit=200) -> List[AlertRecord]:
        orderd = sorted(self._history, key = lambda r: r.fired_at, reverse=True)
        return orderd[:limit]

    async def get_health_status(self) -> HealthStatus:
        from collections import Counter
        now = datetime.now(timezone.utc)
        active_by_sev = dict(Counter(r.severity for r in self._active.values()))
        meta: Dict[str, Any] = {
            "enabled": self.enabled,
            "running": self._running,
            "rules_count": len(self.rules),
            "active_count": len(self._active),
            "active_by_severity": active_by_sev,
        }

        if not self.enabled:                                  
            status, message = "healthy", "alerting disabled by config"
        elif not self._running:
            status, message = "degraded", "enabled but eval loop not running"
        else:
            status = "healthy"
            message = f"{len(self.rules)} rules loaded, {len(self._active)} active"
    
        return HealthStatus(service_name="alert", status=status, message=message, last_check_at=now,metadata=meta)

        
    async def stop(self):
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass




class BaseNotifier(ABC):

    @abstractmethod
    async def send_alert(self, record: AlertRecord, action: str) -> bool:
        '''Send one alert. action is "firing" / "resolved" / "deduplicated".
        Returns True on success, False on failure.'''
        ...


class StdoutNotifier(BaseNotifier):
    '''Always enabled by default; prints each alert to logger.info.'''
    def __init__(self):
        super().__init__()
        self.enabled = True
        self.logger = get_logger("stdout_notifier")

    async def send_alert(self, record: AlertRecord, action: str) -> bool:
        try:
            self.logger.info(
                "[ALERT %s] %s severity=%s value=%s threshold=%s :: %s",
                action.upper(),
                record.rule_name,
                record.severity,
                record.value,
                record.threshold,
                record.description,
            )
            return True
        except Exception as e:
            self.logger.warning("StdoutNotifier failed: %s", e)
            return False


class SlackWebhookNotifier(BaseNotifier):
    '''Posts alerts to a Slack Incoming Webhook.'''

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = get_logger("slack_notifier")
        # config load notifiers.slack 
        env_name = config.get("webhook_url_env", "")
        url = os.environ.get(env_name, "") if env_name else ""
        self.webhook_url = url
        self.channel = config.get("channel", "")
        self.mention = config.get("mention", "")
        self.enabled = bool(config.get("enabled", False)) and bool(self.webhook_url)

    async def send_alert(self, record: AlertRecord, action: str) -> bool:
        if not self.enabled or not self.webhook_url:
            return True
        emoji = ":red_circle:" if action.lower() == "firing" else ":large_green_circle:"
        text = (
            f"{emoji} [{action.upper()}] {record.rule_name} "
            f"(severity={record.severity}) value={record.value} "
            f"threshold={record.threshold} :: {record.description}"
        )
        if self.mention:
            text = f"{self.mention} {text}"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(self.webhook_url, json={"text": text})
            if resp.status_code >= 400:
                self.logger.warning("Slack webhook returned HTTP %s", resp.status_code)
                return False
            return True
        except Exception as e:
            self.logger.warning("Slack webhook failed: %s", e)
            return False

class EmailNotifier(BaseNotifier):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or {}
        self.enabled = bool(config.get("enabled", False))

    async def send_alert(self, record: AlertRecord, action: str) -> bool:
        return True

class PagerDutyNotifier(BaseNotifier):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        config = config or {}
        self.enabled = bool(config.get("enabled", False))

    async def send_alert(self, record: AlertRecord, action: str) -> bool:
        return True

