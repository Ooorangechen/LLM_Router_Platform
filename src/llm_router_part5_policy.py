import asyncio
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from src.llm_router_part5_langgraph import PolicyResult, RequestState
from src.utils.logger import get_logger
from src.utils.metrics import POLICY_METRICS

logger = get_logger("policy")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TIERS: Dict[str, Dict[str, Any]] = {
    "FREE":       {"daily_quota": 100,   "hourly_quota": 10,   "sla_target_ms": 10000, "cost_budget_usd": 0.01},
    "PREMIUM":    {"daily_quota": 1000,  "hourly_quota": 100,  "sla_target_ms": 5000,  "cost_budget_usd": 0.10},
    "ENTERPRISE": {"daily_quota": 10000, "hourly_quota": 1000, "sla_target_ms": 2000,  "cost_budget_usd": 1.00},
}

CB_CLOSED = "CLOSED"
CB_OPEN = "OPEN"
CB_HALF_OPEN = "HALF_OPEN"

# time format: YYYY-MM-DD / YYYYMMDDHH
DAILY_KEY_FMT = "%Y-%m-%d"
HOURLY_KEY_FMT = "%Y%m%d%H"


################# A. QuotaStore #################

class QuotaStore(ABC):
    """Quota State Persistence Abstraction Layer, stored the consumed quota"""
    @abstractmethod
    async def get_daily_quota(self, user_id: str, user_tier: str, date_str: str) -> int:
        ...

    @abstractmethod
    async def set_daily_quota(self, user_id: str, tier: str, date_str: str, value: int) -> None:
        ...

    @abstractmethod
    async def incr_daily(self, user_id: str, tier: str, date_str: str, delta: int) -> int:
        ...

    @abstractmethod
    async def get_hourly_quota(self, user_id: str, tier: str, date_hour_str: str) -> int:
        ...

    @abstractmethod
    async def incr_hourly(self, user_id: str, tier: str, date_hour_str: str, delta: int) -> int:
        ...

    @abstractmethod
    async def snapshot_dump(self) -> Dict:
        ...

    @abstractmethod
    async def snapshot_load(self, data: Dict) -> None:
        ...

    @abstractmethod
    async def flush_snapshot(self) -> None:
        ...

    async def close(self) -> None:
        return None


class MemoryQuotaStore(QuotaStore):
    """in-memory quota store (default)"""

    def __init__(self, snapshot_path: Optional[str] = None):
        # key by (user_id, date_str), no tier
        self._daily: Dict[Tuple[str, str], int] = {}
        self._hourly: Dict[Tuple[str, str], int] = {}
        self._lock = asyncio.Lock()
        self._snapshot_path = Path(snapshot_path) if snapshot_path else None

    async def _incr(self, bucket: Dict[Tuple[str, str], int], key: Tuple[str, str], delta: int) -> int:
        async with self._lock:
            if delta == 0: # read only
                return bucket.get(key, 0)
            new = max(0, bucket.get(key, 0) + delta)
            bucket[key] = new
            return new

    async def get_daily_quota(self, user_id: str, user_tier: str, date_str: str) -> int:
        async with self._lock:
            return self._daily.get((user_id, date_str), 0)

    async def set_daily_quota(self, user_id: str, tier: str, date_str: str, value: int) -> None:
        async with self._lock:
            self._daily[(user_id, date_str)] = max(0, value)

    async def incr_daily(self, user_id: str, tier: str, date_str: str, delta: int) -> int:
        return await self._incr(self._daily, (user_id, date_str), delta)

    async def get_hourly_quota(self, user_id: str, tier: str, date_hour_str: str) -> int:
        async with self._lock:
            return self._hourly.get((user_id, date_hour_str), 0)

    async def incr_hourly(self, user_id: str, tier: str, date_hour_str: str, delta: int) -> int:
        return await self._incr(self._hourly, (user_id, date_hour_str), delta)

    async def snapshot_dump(self) -> Dict:
        now = datetime.now(timezone.utc)
        today, this_hour = now.strftime(DAILY_KEY_FMT), now.strftime(HOURLY_KEY_FMT)
        async with self._lock:
            self._daily = {k: v for k, v in self._daily.items() if k[1] >= today}
            self._hourly = {k: v for k, v in self._hourly.items() if k[1] >= this_hour}
            return {
                "daily": {f"{u}|{b}": v for (u, b), v in self._daily.items()},
                "hourly": {f"{u}|{b}": v for (u, b), v in self._hourly.items()},
                "dumped_at": now.isoformat(),
            }

    async def snapshot_load(self, data: Dict) -> None:
        if not isinstance(data, dict):
            logger.warning("Quota snapshot ignored: expected a dict, got %s", type(data).__name__)
            return
        daily: Dict[Tuple[str, str], int] = {}
        hourly: Dict[Tuple[str, str], int] = {}
        for section, target in (("daily", daily), ("hourly", hourly)):
            raw = data.get(section) or {}
            if not isinstance(raw, dict):
                logger.warning("Quota snapshot section %r ignored: not a dict", section)
                continue
            for key, value in raw.items():
                try:
                    user_id, bucket = key.rsplit("|", 1)
                    target[(user_id, bucket)] = max(0, int(value))
                except (AttributeError, ValueError, TypeError):
                    logger.warning("Quota snapshot entry skipped: %r", key)
        async with self._lock:
            self._daily, self._hourly = daily, hourly
        logger.info("Quota snapshot loaded (daily=%d, hourly=%d)", len(daily), len(hourly))

    async def flush_snapshot(self) -> None:
        if self._snapshot_path is None:
            return
        data = await self.snapshot_dump()
        try:
            await asyncio.to_thread(self._write_snapshot, data)
        except Exception as exc:
            logger.warning("Quota snapshot flush failed: %s", exc)

    def _write_snapshot(self, data: Dict) -> None:
        path = self._snapshot_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

_INCR_CLAMP_LUA = """
local v = redis.call('INCRBY', KEYS[1], ARGV[1])
if v < 0 then
  redis.call('SET', KEYS[1], 0)
  v = 0
end
redis.call('EXPIREAT', KEYS[1], ARGV[2])
return v
"""


class RedisQuotaStore(QuotaStore):
    """redis quota store (optional)"""

    def __init__(self, redis_config: Dict[str, Any]):
        self._config = redis_config
        self._client = None
        self._incr_script = None
        self._key_prefix: str = redis_config.get("key_prefix", "llm-router:policy:")

    async def initialize(self) -> None:
        from redis.asyncio import BlockingConnectionPool, Redis
        timeout = self._config.get("timeout_sec", 2)
        pool = BlockingConnectionPool(
            host=self._config.get("host", "localhost"),
            port=self._config.get("port", 6379),
            db=self._config.get("db", 1),
            decode_responses=True,
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
            max_connections=self._config.get("max_connections", 50),
            timeout=timeout,
        )
        client = Redis.from_pool(pool)
        try:
            await client.ping()
        except Exception:
            await client.aclose()
            raise
        self._client = client
        self._incr_script = client.register_script(_INCR_CLAMP_LUA)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _daily_key(self, user_id: str, date_str: str) -> str:
        return f"{self._key_prefix}user:{user_id}:d:{date_str}"

    def _hourly_key(self, user_id: str, date_hour_str: str) -> str:
        return f"{self._key_prefix}user:{user_id}:h:{date_hour_str}"

    @staticmethod
    def _day_end_ts(date_str: str) -> int:
        start = datetime.strptime(date_str, DAILY_KEY_FMT).replace(tzinfo=timezone.utc)
        return int((start + timedelta(days=1)).timestamp())

    @staticmethod
    def _hour_end_ts(date_hour_str: str) -> int:
        start = datetime.strptime(date_hour_str, HOURLY_KEY_FMT).replace(tzinfo=timezone.utc)
        return int((start + timedelta(hours=1)).timestamp())

    async def _get(self, key: str) -> int:
        value = await self._client.get(key)
        return int(value) if value is not None else 0

    async def _incr(self, key: str, delta: int, expire_at: int) -> int:
        if delta == 0:
            return await self._get(key)
        return int(await self._incr_script(keys=[key], args=[delta, expire_at]))

    async def get_daily_quota(self, user_id: str, user_tier: str, date_str: str) -> int:
        return await self._get(self._daily_key(user_id, date_str))

    async def set_daily_quota(self, user_id: str, tier: str, date_str: str, value: int) -> None:
        key = self._daily_key(user_id, date_str)
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.set(key, max(0, value))
            pipe.expireat(key, self._day_end_ts(date_str))
            await pipe.execute()

    async def incr_daily(self, user_id: str, tier: str, date_str: str, delta: int) -> int:
        return await self._incr(self._daily_key(user_id, date_str), delta, self._day_end_ts(date_str))

    async def get_hourly_quota(self, user_id: str, tier: str, date_hour_str: str) -> int:
        return await self._get(self._hourly_key(user_id, date_hour_str))

    async def incr_hourly(self, user_id: str, tier: str, date_hour_str: str, delta: int) -> int:
        return await self._incr(self._hourly_key(user_id, date_hour_str), delta,
                                self._hour_end_ts(date_hour_str))

    async def snapshot_dump(self) -> Dict:
        return {}

    async def snapshot_load(self, data: Dict) -> None:
        return None

    async def flush_snapshot(self) -> None:
        return None


################# B. CircuitBreaker #################

class CircuitBreaker:
    """Per-model (or global) circuit breaker: CLOSED -> OPEN -> HALF_OPEN -> CLOSED/OPEN"""

    def __init__(self, name: str, failure_threshold: int = 5, half_open_after: int = 30):
        self.name = name
        self.state = CB_CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.half_open_after = half_open_after          # config.circuit_breaker.half_open_after_sec
        self.last_failure_at: Optional[datetime] = None  # UTC
        self._lock = asyncio.Lock()
        # Decision: 文档没定义 HALF_OPEN 时第二个请求怎么办。M3 要求"只放 1 个探测请求"，
        # 所以记录探测开始时间：有探测在跑时，其他请求的 acquire() 返回 False。
        # 用时间而不是 bool：探测请求如果没回报结果（节点崩了、没调 record_*），
        # 超过 half_open_after 秒就允许再放一个探测，熔断器不会永远卡在 HALF_OPEN。
        self._probe_started_at: Optional[datetime] = None
        self._export_state()

    def _export_state(self) -> None:
        for state in (CB_CLOSED, CB_OPEN, CB_HALF_OPEN):
            POLICY_METRICS.circuit_breaker_state_info.labels(
                name=self.name, state=state).set(1 if state == self.state else 0)

    def _transition(self, new_state: str) -> None:
        if new_state == self.state:
            return
        logger.info("Circuit breaker %s: %s -> %s", self.name, self.state, new_state)
        if new_state == CB_OPEN:
            POLICY_METRICS.circuit_breaker_opened_total.labels(name=self.name).inc()
        self.state = new_state
        self._export_state()

    def _seconds_since(self, t: Optional[datetime], now: datetime) -> float:
        return (now - t).total_seconds() if t is not None else float("inf")

    async def acquire(self) -> bool:
        async with self._lock:
            now = datetime.now(timezone.utc)
            if self.state == CB_CLOSED:
                return True
            if self.state == CB_OPEN:
                if self._seconds_since(self.last_failure_at, now) < self.half_open_after:
                    return False
                self._transition(CB_HALF_OPEN)
                self._probe_started_at = now
                return True
            if self._seconds_since(self._probe_started_at, now) < self.half_open_after:
                return False
            self._probe_started_at = now
            return True

    async def record_success(self) -> None:
        async with self._lock:
            if self.state == CB_HALF_OPEN:
                self._transition(CB_CLOSED)
                self._probe_started_at = None
                self.failure_count = 0
            elif self.state == CB_CLOSED:
                self.failure_count = 0

    async def record_failure(self) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            if self.state == CB_HALF_OPEN:
                self.failure_count += 1
                self.last_failure_at = now
                self._probe_started_at = None
                self._transition(CB_OPEN)
            elif self.state == CB_CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.last_failure_at = now
                    self._transition(CB_OPEN)

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at else None,
        }


################# C. ExecutionPolicy #################

class ExecutionPolicy:
    """Quota / SLA / budget checks + circuit breaker registry, the only entry used by LangGraph nodes"""

    def __init__(self, config: Dict, quota_store: Optional[QuotaStore] = None,
                 cb_registry: Optional[Dict[str, CircuitBreaker]] = None):
        self.config = config
        self.enabled: bool = bool(config.get("enabled", False))
        self.default_tiers: Dict[str, Dict] = DEFAULT_TIERS
        self.quota_store: Optional[QuotaStore] = quota_store   # 外部传入优先（测试注入）
        self._cb_registry: Dict[str, CircuitBreaker] = cb_registry if cb_registry is not None else {}
        self._cb_global: Optional[CircuitBreaker] = None
        self._flush_task: Optional[asyncio.Task] = None
        self.backend: str = "injected" if quota_store is not None else "uninitialized"

        cb_cfg = config.get("circuit_breaker", {}) or {}
        self.cb_enabled: bool = bool(cb_cfg.get("enabled", True))
        self._cb_per_model: bool = bool(cb_cfg.get("per_model", True))

        self._cb_threshold: int = int(cb_cfg.get("failure_threshold", 5)) if self.cb_enabled else 2**31
        self._cb_half_open_after: int = int(cb_cfg.get("half_open_after_sec", 30))

        snap_cfg = config.get("snapshot", {}) or {}
        self._snapshot_enabled: bool = bool(snap_cfg.get("enabled", True))
        self._snapshot_path: Path = PROJECT_ROOT / snap_cfg.get("file", "data/policy_state/snapshot.json")
        self._flush_interval: float = float(snap_cfg.get("flush_interval_sec", 60))

        budget_cfg = config.get("cost_budget", {}) or {}
        self._budget_enabled: bool = bool(budget_cfg.get("enabled", True))
        self._refund_on_failure: bool = bool(config.get("refund_on_failure", True))

    async def initialize(self) -> None:
        # 1. store selection
        if self.quota_store is None:
            if self.config.get("backend", "memory") == "redis":
                redis_store = RedisQuotaStore(self.config.get("redis", {}) or {})
                try:
                    await redis_store.initialize()
                    self.quota_store, self.backend = redis_store, "redis"
                except Exception as exc:
                    logger.info("RedisQuotaStore unavailable (%s: %s), falling back to memory",
                                type(exc).__name__, exc)
            if self.quota_store is None:
                self.quota_store = MemoryQuotaStore(
                    str(self._snapshot_path) if self._snapshot_enabled else None)
                self.backend = "memory"
                if self._snapshot_enabled:
                    await self._load_snapshot()

        # 2. global circuit
        self._cb_global = self.circuit_breaker_for("global")

        if self._snapshot_enabled and self.backend == "memory":
            self._flush_task = asyncio.create_task(self._snapshot_flush_loop())

        logger.info("ExecutionPolicy initialized (backend=%s, %d tiers loaded)",
                    self.backend, len(self.default_tiers))

    async def _load_snapshot(self) -> None:
        if not self._snapshot_path.exists():
            return
        try:
            data = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Quota snapshot unreadable, starting empty: %s", exc)
            return
        await self.quota_store.snapshot_load(data)

    async def _snapshot_flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            await self.quota_store.flush_snapshot()

    async def shutdown(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        if self.quota_store is not None:
            await self.quota_store.flush_snapshot()
            await self.quota_store.close()
        logger.info("ExecutionPolicy shutdown")

    def _tier_key(self, tier: Any) -> str:
        name = str(getattr(tier, "value", tier)).upper()
        return name

    def _tier_config(self, tier: str) -> Dict:
        key = self._tier_key(tier)
        overrides = {str(k).upper(): v for k, v in (self.config.get("tiers") or {}).items()}
        return {**self.default_tiers[key], **(overrides.get(key) or {})}

    @staticmethod
    def _time_keys() -> Tuple[str, str]:
        now = datetime.now(timezone.utc)
        return now.strftime(DAILY_KEY_FMT), now.strftime(HOURLY_KEY_FMT)

    async def check(self, state: RequestState) -> PolicyResult:
        tier_key = self._tier_key(state.input.user_tier)
        tier = self._tier_config(tier_key)
        label = tier_key.lower()
        user_id = state.input.user_id
        daily_total = int(tier["daily_quota"])
        hourly_total = int(tier["hourly_quota"])
        sla_target_ms = int(tier["sla_target_ms"])
        cost_budget_usd = float(tier["cost_budget_usd"])
        POLICY_METRICS.quota_checked_total.labels(user_tier=label).inc()

        date_str, hour_str = self._time_keys()
        try:
            used_d = await self.quota_store.incr_daily(user_id, tier_key, date_str, 0)
            used_h = await self.quota_store.incr_hourly(user_id, tier_key, hour_str, 0)
        except Exception as exc:
            logger.warning("Quota store error in check, allowing request: %s", exc)
            return PolicyResult(passed=True, daily_quota_total=daily_total, hourly_quota_total=hourly_total,
                                sla_target_ms=sla_target_ms, cost_budget_usd=cost_budget_usd,
                                message=f"policy_error: quota store unavailable ({type(exc).__name__})")

        daily_remaining = max(0, daily_total - used_d)
        hourly_remaining = max(0, hourly_total - used_h)

        break_reason, metric_reason = None, None
        if daily_remaining <= 0:
            break_reason, metric_reason = "daily_quota_exceeded", "daily"
        elif hourly_remaining <= 0:
            break_reason, metric_reason = "hourly_quota_exceeded", "hourly"
        elif self._budget_enabled:
            estimated = state.metadata.extra.get("estimated_cost_usd")
            if estimated is not None and not float(estimated) <= cost_budget_usd:
                break_reason, metric_reason = "cost_budget_exceeded", "budget"

        if break_reason is not None:
            POLICY_METRICS.quota_rejected_total.labels(user_tier=label, reason=metric_reason).inc()

        return PolicyResult(
            passed=break_reason is None,
            break_reason=break_reason,
            daily_quota_total=daily_total,
            hourly_quota_total=hourly_total,
            daily_remaining_before=daily_remaining,
            daily_remaining_after=daily_remaining,
            hourly_remaining_before=hourly_remaining,
            hourly_remaining_after=hourly_remaining,
            sla_target_ms=sla_target_ms,
            cost_budget_usd=cost_budget_usd,
            message="ok" if break_reason is None else break_reason,
        )

    async def consume(self, state: RequestState, policy_result: PolicyResult) -> PolicyResult:
        if not policy_result.passed:
            return policy_result

        tier_key = self._tier_key(state.input.user_tier)
        tier = self._tier_config(tier_key)
        label = tier_key.lower()
        user_id = state.input.user_id
        daily_quota, hourly_quota = int(tier["daily_quota"]), int(tier["hourly_quota"])
        date_str, hour_str = self._time_keys()

        try:
            used_d = await self.quota_store.incr_daily(user_id, tier_key, date_str, 1)
            used_h = await self.quota_store.incr_hourly(user_id, tier_key, hour_str, 1)
        except Exception as exc:
            logger.warning("Quota store error in consume, allowing request: %s", exc)
            return policy_result.model_copy(update={
                "message": f"policy_error: quota store unavailable ({type(exc).__name__})"})

        break_reason, metric_reason = None, None
        if used_d > daily_quota:
            break_reason, metric_reason = "daily_quota_exceeded", "daily"
        elif used_h > hourly_quota:
            break_reason, metric_reason = "hourly_quota_exceeded", "hourly"
        if break_reason is not None:
            try:
                await self.quota_store.incr_daily(user_id, tier_key, date_str, -1)
                await self.quota_store.incr_hourly(user_id, tier_key, hour_str, -1)
            except Exception as exc:
                logger.warning("Quota rollback failed for %s: %s", user_id, exc)
            POLICY_METRICS.quota_rejected_total.labels(user_tier=label, reason=metric_reason).inc()
            return policy_result.model_copy(update={
                "passed": False,
                "break_reason": break_reason,
                "daily_remaining_before": max(0, daily_quota - used_d + 1),
                "daily_remaining_after": max(0, daily_quota - used_d + 1),
                "hourly_remaining_before": max(0, hourly_quota - used_h + 1),
                "hourly_remaining_after": max(0, hourly_quota - used_h + 1),
                "message": break_reason,
            })

        state.metadata.extra["policy_date_key"] = date_str
        state.metadata.extra["policy_hour_key"] = hour_str
        POLICY_METRICS.quota_consumed_total.labels(user_tier=label).inc()


        return policy_result.model_copy(update={
            "daily_remaining_before": daily_quota - used_d + 1,
            "daily_remaining_after": daily_quota - used_d,
            "hourly_remaining_before": hourly_quota - used_h + 1,
            "hourly_remaining_after": hourly_quota - used_h,
            "message": "consumed",
        })

    async def refund(self, state: RequestState, reason: str = "execution_failure") -> None:
        if not self._refund_on_failure:
            return
        
        date_str = state.metadata.extra.pop("policy_date_key", None)
        hour_str = state.metadata.extra.pop("policy_hour_key", None)
        if date_str is None or hour_str is None:
            return

        tier_key = self._tier_key(state.input.user_tier)
        user_id = state.input.user_id
        try:
            await self.quota_store.incr_daily(user_id, tier_key, date_str, -1)
            await self.quota_store.incr_hourly(user_id, tier_key, hour_str, -1)
        except Exception as exc:
            logger.warning("Quota refund failed for %s: %s", user_id, exc)
            return
        POLICY_METRICS.quota_refunded_total.labels(user_tier=tier_key.lower(), reason=reason).inc()

    def circuit_breaker_for(self, key: str = "global") -> CircuitBreaker:
        if not self._cb_per_model:
            key = "global"
        cb = self._cb_registry.get(key)
        if cb is None:
            cb = CircuitBreaker(key, self._cb_threshold, self._cb_half_open_after)
            self._cb_registry[key] = cb
        return cb
