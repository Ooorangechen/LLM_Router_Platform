import asyncio
import json
import os
import threading
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from uuid import UUID
from src.llm_router_part5_schema import FeedbackRecord, QualityRecord, QualitySnapshot, RequestState
from src.utils.filelock import FileLock
from src.utils.logger import get_logger
from src.utils.metrics import QUALITY_METRICS

logger = get_logger("quality")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUALITY_START_KEY = "__quality"   # node_started_at key written by start_request
CACHE_CLEAN_INTERVAL_SEC = 30

DEFAULT_FEEDBACK_PATH = "data/feedback/feedback.jsonl"
ANONYMOUS_USER = "anonymous"
MAX_SCAN_LINES = 20000       # 3.6: cap the JSONL scan so the dashboard stays under 500ms
MAX_WINDOW_MINUTES = 1440    # 3.6: window_minutes upper limit

VIOLATION_ERROR_RATE = "error_rate_threshold"
VIOLATION_LATENCY_P95 = "latency_p95_threshold"
# metric label values for QUALITY_METRICS.slo_violations_total (3.5 metric list)
VIOLATION_LABELS = {VIOLATION_ERROR_RATE: "error_rate", VIOLATION_LATENCY_P95: "latency_p95"}


def _percentile(sorted_values: List[int], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default); 0.0 for no data."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo))


################# Task 3.5: QualityMonitor #################


class QualityMonitor:
    """Sliding-window SLO + hotspot detection over recorded /route/advanced requests."""

    def __init__(self, config: Dict[str, Any], alert_manager_ref: Optional[Any] = None):
        self.config = config or {}
        self.enabled: bool = bool(self.config.get("enabled", False))
        # Decision: kept for the 3.5 signature only; alerting is wired by main.py's hook on AlertManager.
        self.alert_manager_ref = alert_manager_ref

        self._window_minutes: int = int(self.config.get("rolling_window_minutes", 5))
        self._rolling_window: Deque[QualityRecord] = deque(
            maxlen=int(self.config.get("rolling_window_max_records", 10000)))
        # Decision: threading.Lock, because start/end_request and get_snapshot are sync (3.5 A)
        self._lock = threading.Lock()

        self._cache_ttl: float = float(self.config.get("cache_ttl_sec", 10))
        self._stats_cache: Dict[int, Tuple[float, QualitySnapshot]] = {}  # window -> (expire monotonic ts, snapshot)
        self._cleaner_task: Optional[asyncio.Task] = None

        slo_cfg = self.config.get("slo", {}) or {}
        self._slo_targets: Dict[str, float] = {
            "error_rate": float(slo_cfg.get("error_rate_threshold", 0.05)),
            "latency_p95_ms": float(slo_cfg.get("latency_p95_threshold_ms", 3000)),
        }
        hotspot_cfg = self.config.get("hotspot", {}) or {}
        self._hotspot_threshold: Dict[str, float] = {
            "min_samples": int(hotspot_cfg.get("min_samples", 10)),
            "avg_latency_ms": float(hotspot_cfg.get("avg_latency_threshold_ms", 5000)),
        }

        # metric dedupe state: violations active in the SLO window, hotspots already counted this hour
        self._active_violations: Set[str] = set()
        self._hotspot_hour: Optional[str] = None
        self._hotspots_counted: Set[str] = set()

    async def initialize(self) -> None:
        self._cleaner_task = asyncio.create_task(self._stats_cache_cleaner())
        logger.info("QualityMonitor initialized (window=%dmin, max_records=%d)",
                    self._window_minutes, self._rolling_window.maxlen)

    async def shutdown(self) -> None:
        if self._cleaner_task is not None:
            self._cleaner_task.cancel()
            try:
                await self._cleaner_task
            except asyncio.CancelledError:
                pass
            self._cleaner_task = None

    async def _stats_cache_cleaner(self) -> None:
        while True:
            await asyncio.sleep(CACHE_CLEAN_INTERVAL_SEC)
            now = monotonic()
            for key in [k for k, (expire, _) in self._stats_cache.items() if expire <= now]:
                self._stats_cache.pop(key, None)

    ######## telemetry ########

    def start_request(self, state: RequestState) -> None:
        state.mark_node_start(QUALITY_START_KEY, datetime.now(timezone.utc))

    def end_request(self, state: RequestState) -> None:
        # Decision: policy rejections are business decisions, not service quality; they are not recorded.
        if state.policy.policy_check_passed is False:
            return
        now = datetime.now(timezone.utc)
        result = state.result
        model = (result.fallback_model if result.fallback_used else state.routing.selected_model) or "unknown"
        tier = state.input.user_tier.value
        # success None means the graph never finished execute: counted as a failure (3.5 constraint)
        success = result.success is True
        # Decision: model latency (result.latency_ms), matching per-model hotspots; falls back to the
        # time since start_request when execute never produced one.
        latency_ms = result.latency_ms
        if latency_ms is None:
            started = state.metadata.node_started_at.get(QUALITY_START_KEY)
            latency_ms = int((now - started).total_seconds() * 1000) if started else 0

        record = QualityRecord(
            request_id=state.input.request_id,
            user_id=state.input.user_id,
            model=model,
            tier=tier,
            success=success,
            latency_ms=latency_ms,
            cost_usd=result.cost_usd or 0.0,
            tokens_total=result.total_tokens or 0,
            error_type=result.error_type,
            ts=now,
        )
        with self._lock:
            self._rolling_window.append(record)  # maxlen drops the oldest

        QUALITY_METRICS.requests_total.labels(model=model, user_tier=tier,
                                              status="success" if success else "error").inc()
        QUALITY_METRICS.request_duration_seconds.labels(model=model, user_tier=tier).observe(latency_ms / 1000)
        QUALITY_METRICS.tokens_total.labels(model=model).inc(record.tokens_total)
        QUALITY_METRICS.cost_usd_total.labels(model=model).inc(record.cost_usd)

    ######## snapshot ########

    def _rolling_records(self, window_minutes: int) -> List[QualityRecord]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        with self._lock:
            return [r for r in self._rolling_window if r.ts >= cutoff]

    def _compute_snapshot(self, window_minutes: int) -> QualitySnapshot:
        recs = self._rolling_records(window_minutes)
        n = len(recs)
        success_rate = sum(r.success for r in recs) / n if n else 1.0
        error_rate = 1.0 - success_rate
        latencies = sorted(r.latency_ms for r in recs)
        p95 = _percentile(latencies, 0.95)

        violations: List[str] = []
        if error_rate > self._slo_targets["error_rate"]:
            violations.append(VIOLATION_ERROR_RATE)
        if p95 > self._slo_targets["latency_p95_ms"]:
            violations.append(VIOLATION_LATENCY_P95)
        # Decision: the SLO is defined over rolling_window_minutes; snapshots of other windows
        # (dashboard, hotspots) report violations but do not count them again.
        if window_minutes == self._window_minutes:
            self._count_new_violations(violations)

        return QualitySnapshot(
            window_minutes=window_minutes,
            request_count=n,
            success_rate=success_rate,
            error_rate=error_rate,
            avg_latency_ms=sum(latencies) / n if n else 0.0,
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=p95,
            p99_latency_ms=_percentile(latencies, 0.99),
            total_tokens=sum(r.tokens_total for r in recs),
            total_cost_usd=sum(r.cost_usd for r in recs),
            slo_ok=not violations,
            slo_violations=violations,
            hotspot_models=self._detect_hotspots(recs),
        )

    def _count_new_violations(self, violations: List[str]) -> None:
        # "inc if triggered for the first time": count when a violation starts, not on every snapshot
        current = set(violations)
        for v in current - self._active_violations:
            QUALITY_METRICS.slo_violations_total.labels(violation=VIOLATION_LABELS[v]).inc()
        self._active_violations = current

    def _detect_hotspots(self, recs: List[QualityRecord]) -> List[str]:
        by_model: Dict[str, List[int]] = defaultdict(list)
        for r in recs:
            by_model[r.model].append(r.latency_ms)
        hotspots = sorted(
            model for model, lat in by_model.items()
            if len(lat) >= self._hotspot_threshold["min_samples"]
            and sum(lat) / len(lat) > self._hotspot_threshold["avg_latency_ms"]
        )
        hour = datetime.now(timezone.utc).strftime("%Y%m%d%H")
        if hour != self._hotspot_hour:
            self._hotspot_hour, self._hotspots_counted = hour, set()
        for model in hotspots:
            if model not in self._hotspots_counted:  # deduplicated by hour
                QUALITY_METRICS.hotspot_detected_total.labels(model=model).inc()
                self._hotspots_counted.add(model)
        return hotspots

    def get_snapshot(self, window_minutes: Optional[int] = None, use_cache: bool = True) -> QualitySnapshot:
        # Decision: default window is config rolling_window_minutes (doc default 5 == config default)
        window_minutes = window_minutes or self._window_minutes
        now = monotonic()
        cached = self._stats_cache.get(window_minutes)
        if use_cache and cached is not None and cached[0] > now:
            return cached[1]
        snapshot = self._compute_snapshot(window_minutes)
        self._stats_cache[window_minutes] = (now + self._cache_ttl, snapshot)
        return snapshot

    def get_slo_status(self, window_minutes: Optional[int] = None) -> Tuple[bool, List[str]]:
        snapshot = self.get_snapshot(window_minutes)
        return snapshot.slo_ok, snapshot.slo_violations

    def get_hotspot_models(self, window_minutes: int = 15) -> List[str]:
        return self.get_snapshot(window_minutes).hotspot_models

    def models_by_request(self) -> Dict[UUID, str]:
        """request_id -> model of the recorded requests; FeedbackCollector's fallback model lookup."""
        with self._lock:
            return {r.request_id: r.model for r in self._rolling_window}


################# Task 3.6: FeedbackCollector #################


def _score(record: FeedbackRecord) -> Optional[int]:
    # Decision: like counts as 5 and dislike as 1, so they sort together with ratings
    if record.rating is not None:
        return record.rating
    return {1: 5, -1: 1}.get(record.like_value)


class FeedbackCollector:
    """User feedback persisted as JSONL (asyncio.Lock + cross-process file lock) + dashboard aggregation."""

    def __init__(self, config: Dict[str, Any], storage_path: Optional[str] = None,
                 quality_monitor: Optional[QualityMonitor] = None, pipeline: Optional[Any] = None):
        self.config = config or {}
        self.enabled: bool = bool(self.config.get("enabled", False))
        self.path: Path = PROJECT_ROOT / (storage_path or self.config.get("storage_path", DEFAULT_FEEDBACK_PATH))
        # Single-process (one uvicorn worker): asyncio.Lock alone orders the writes and FileLock never
        # contends. FileLock only matters when several processes append (--workers N, external scripts).
        self._lock = asyncio.Lock()
        self._lock_timeout: float = float(self.config.get("lock_timeout_sec", 10))
        self._comment_max_length: int = int(self.config.get("comment_max_length", 2000))
        self._allow_anonymous: bool = bool(self.config.get("allow_anonymous", True))

        dashboard_cfg = self.config.get("dashboard", {}) or {}
        self.default_window_minutes: int = int(dashboard_cfg.get("default_window_minutes", 60))
        self.default_limit: int = int(dashboard_cfg.get("default_limit", 20))
        self._join_clickhouse: bool = bool(dashboard_cfg.get("join_clickhouse_model", True))

        # Decision: quality_monitor / pipeline are not in the 3.6 C signature; they feed the model lookup
        # (ClickHouse query_logs first, QualityMonitor memory second) and the dashboard snapshot.
        self.quality_monitor = quality_monitor
        self.pipeline = pipeline

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        logger.info("FeedbackCollector initialized (path=%s)", self.path)

    async def submit(self, record: FeedbackRecord) -> FeedbackRecord:
        if record.comment is not None and len(record.comment) > self._comment_max_length:
            raise ValueError(f"comment longer than {self._comment_max_length} characters")
        if not record.user_id:
            if not self._allow_anonymous:
                raise ValueError("user_id is required (feedback.allow_anonymous=false)")
            record.user_id = ANONYMOUS_USER
        record.created_at = datetime.now(timezone.utc)
        # json.dumps escapes newlines, so one record is always exactly one line
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"

        async with self._lock:
            async with FileLock(str(self.path) + ".lock", timeout=self._lock_timeout):
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())

        QUALITY_METRICS.feedback_submitted_total.labels(
            type=record.feedback_type, rating=str(record.rating) if record.rating is not None else "none").inc()
        return record

    def _read_records(self, window_minutes: int) -> List[FeedbackRecord]:
        """Newest first. Only the last MAX_SCAN_LINES lines are scanned (3.6 500ms budget)."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        with open(self.path, encoding="utf-8") as f:
            lines = deque(f, maxlen=MAX_SCAN_LINES)
        records = []
        for line in reversed(lines):
            try:
                record = FeedbackRecord.model_validate_json(line)
            except ValueError:   # corrupted line (bad JSON or schema) is skipped (3.6 constraint)
                continue
            if record.created_at >= cutoff:
                records.append(record)
        return records

    async def _models_for(self, query_ids: List[UUID]) -> Dict[UUID, str]:
        models: Dict[UUID, str] = {}
        writer = getattr(self.pipeline, "ch_writer", None) if self._join_clickhouse and self.pipeline else None
        if query_ids and writer is not None and writer.enabled and writer.client is not None:
            # UUID objects format as plain hex, so they are safe to inline
            ids = ", ".join(f"toUUID('{q}')" for q in set(query_ids))
            sql = f"SELECT query_id, any(selected_model) FROM query_logs WHERE query_id IN ({ids}) GROUP BY query_id"
            try:
                result = await asyncio.wait_for(asyncio.to_thread(writer.client.query, sql), timeout=2.0)
                models = {UUID(str(qid)): model for qid, model in result.result_rows}
            except Exception as exc:
                logger.warning("Feedback model lookup in ClickHouse failed, using QualityMonitor: %s", exc)
        missing = set(query_ids) - models.keys()
        if missing and self.quality_monitor is not None:
            recorded = self.quality_monitor.models_by_request()
            models.update({q: recorded[q] for q in missing if q in recorded})
        return models

    async def query(self, window_minutes: int = 60, user_id: Optional[str] = None, model: Optional[str] = None,
                    min_rating: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        records = await asyncio.to_thread(self._read_records, window_minutes)
        if user_id is not None:
            records = [r for r in records if r.user_id == user_id]
        if min_rating is not None:
            records = [r for r in records if r.rating is not None and r.rating >= min_rating]
        if model is None:
            records = records[:limit]   # resolve models only for what is returned
        models = await self._models_for([r.query_id for r in records])
        rows = [{**r.model_dump(mode="json"), "model": models.get(r.query_id)} for r in records]
        if model is not None:
            rows = [row for row in rows if row["model"] == model]
        return rows[:limit]

    async def get_dashboard_data(self, window_minutes: Optional[int] = None, limit: Optional[int] = None,
                                 quality_monitor: Optional[QualityMonitor] = None) -> Dict[str, Any]:
        window_minutes = min(window_minutes or self.default_window_minutes, MAX_WINDOW_MINUTES)
        limit = limit or self.default_limit
        quality_monitor = quality_monitor or self.quality_monitor
        records = await asyncio.to_thread(self._read_records, window_minutes)

        scored = [(s, r) for r in records if (s := _score(r)) is not None]
        # Decision: positive = score >= 4, negative = score <= 2; rating 3 and comment-only feedback
        # appear in neither list. Sorts are stable, so equal scores stay newest first.
        positive = sorted((x for x in scored if x[0] >= 4), key=lambda x: x[0], reverse=True)[:limit]
        negative = sorted((x for x in scored if x[0] <= 2), key=lambda x: x[0])[:limit]
        models = await self._models_for([r.query_id for _, r in positive + negative])

        def row(record: FeedbackRecord) -> Dict[str, Any]:
            return {
                "query_id": str(record.query_id),
                "feedback_type": record.feedback_type,
                "rating": record.rating,
                "like_value": record.like_value,
                "comment": record.comment,
                "model": models.get(record.query_id),
                "user_id": record.user_id,
                "created_at": record.created_at.isoformat(),
            }

        ratings = Counter(r.rating for r in records if r.rating is not None)
        snapshot = quality_monitor.get_snapshot(window_minutes) if quality_monitor is not None else None
        return {
            "window_minutes": window_minutes,
            "feedback_total": len(records),
            "top_positive": [row(r) for _, r in positive],
            "top_negative": [row(r) for _, r in negative],
            "rating_distribution": {str(k): ratings[k] for k in sorted(ratings)},
            "snapshot": snapshot.model_dump(mode="json") if snapshot is not None else None,
        }
