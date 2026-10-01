from pydantic import BaseModel, Field
from datetime import datetime, timezone
import time
from typing import List, Dict, Any, Optional
import asyncio
from src.utils.logger import get_logger
import os
import platform # consider both windows / linux, mac

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

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
