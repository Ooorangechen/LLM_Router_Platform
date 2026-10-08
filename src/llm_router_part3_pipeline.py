import json
import os
import time
import traceback
from pathlib import Path
from uuid import UUID, uuid4
from pydantic import AfterValidator, BaseModel, Field
from typing import Annotated, Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
from src.utils.schema import RoutingDecision, InferenceResponse, QueryRequest
from src.utils.metrics import PIPELINE_METRICS
from src.llm_router_part4_monitor import HealthStatus
import asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from src.utils.constants import ClickHouseTables, KafkaTopics
from src.utils.logger import get_logger
from tenacity import retry, stop_after_attempt

try:
    import clickhouse_connect
    from clickhouse_connect.driver import httputil
except ImportError:
    clickhouse_connect = None
    httputil = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_topic_metadata(kafka_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Read topics_file, the single Topic source (P3 §3.1), as {name: metadata}."""
    topics_path = PROJECT_ROOT / kafka_config.get(
        "topics_file", "kafka/topics.json")
    with topics_path.open(encoding="utf-8") as file:
        return {item["name"]: item for item in json.load(file)["topics"]}

# TASK 3.2

##### New Pydantic Models

def _as_utc(value: datetime) -> datetime:
    """Normalize one datetime at an external data boundary."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


UTCDateTime = Annotated[datetime, AfterValidator(_as_utc)]

class QueryLogEntry(BaseModel):
    query_id: UUID
    user_id: str = Field(min_length=1)
    user_tier: str
    query_text: str
    query_type: str

    selected_model: str = Field(min_length=1)
    routing_strategy: str
    routing_confidence: float = Field(ge=0.0, le=1.0)

    token_count_input: int = Field(ge=0)

    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(ge=1)
    has_context: bool 
    has_attachments: bool
    request_received_at: UTCDateTime

    status: str = "received"
    extra_labels: Dict[str, str] = Field(default_factory=dict)

class ResponseLogEntry(BaseModel):
    query_id: UUID
    user_id: str = Field(min_length=1)

    model_name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    response_text: str

    token_count_input: int = Field(ge=0)
    token_count_output: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)

    routing_time_ms: int = Field(ge=0)
    
    cached: bool # InferenceResponse
    compressed_context: bool # InferenceResponse

    error: Optional[str] = None

    response_completed_at: UTCDateTime

    status: str = "success"

class MetricEntry(BaseModel):
    timestamp: UTCDateTime
    service: str = "llm-router"
    metric_name: str = Field(min_length=1)
    value: float 
    labels: Dict[str, str]

class ErrorEntry(BaseModel):
    error_id: UUID = Field(default_factory=uuid4)
    query_id: Optional[UUID] = None
    error_type: str = Field(min_length=1)
    error_message: str
    stacktrace: str
    component: str = Field(min_length=1)
    severity: str = "error"
    timestamp: UTCDateTime
    extra: Dict[str, Any] = Field(default_factory=dict)

class DeadLetterEntry(BaseModel):
    """encapsulates messages that failed to write"""

    dlq_id: UUID = Field(default_factory=uuid4)
    original_topic: str
    original_message: str
    failure_reason: str
    failure_count: int = Field(ge=1)
    first_failed_at: UTCDateTime
    last_failed_at: UTCDateTime


#### Factory functions
def build_query_log_entry(
        request: QueryRequest, decision: RoutingDecision, 
        received_at: datetime) -> QueryLogEntry:
    """Build the query-side pipeline event from routing inputs."""
    return QueryLogEntry(
        query_id=UUID(request.request_id),
        user_id=request.user_id,
        user_tier=request.user_tier.value,
        query_text=request.query,
        query_type=decision.query_type.value,
        selected_model=decision.selected_model,
        routing_strategy=decision.routing_strategy,
        routing_confidence=decision.confidence,
        token_count_input=decision.token_count,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        has_context=bool(request.context),
        has_attachments=bool(request.attachments),
        request_received_at=received_at,
    )

def build_response_log_entry(
        request: QueryRequest, decision: RoutingDecision, 
        response: InferenceResponse, completed_at: datetime) -> ResponseLogEntry:
    """Build the response-side pipeline event from inference output."""
    return ResponseLogEntry(
        query_id=UUID(request.request_id),
        user_id=request.user_id,
        model_name=response.model_name,
        provider=response.provider,
        response_text=response.response_text,
        token_count_input=response.token_count_input,
        token_count_output=response.token_count_output,
        total_tokens=response.total_tokens,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        routing_time_ms=decision.routing_time_ms,
        cached=response.cached,
        compressed_context=response.compressed_context,
        error=response.error,
        response_completed_at=completed_at,
        status="error" if response.error else "success",
    )

def build_metric_entries(
        request: QueryRequest, decision: RoutingDecision,
        response: InferenceResponse) -> List[MetricEntry]:
    """
    generates multiple entries per request, covering 
    request_count, tokens_input, tokens_output, cost, latency, routing_time
    """
    timestamp = datetime.now(timezone.utc)
    labels = {
        "model": response.model_name,
        "provider": response.provider,
        "user_tier": request.user_tier.value,
        "query_type": decision.query_type.value,
        "status": "error" if response.error else "success",
    }
    values = (
        ("request_count", 1.0),
        ("tokens_input", float(response.token_count_input)),
        ("tokens_output", float(response.token_count_output)),
        ("cost_usd", float(response.cost_usd)),
        ("inference_latency_ms", float(response.latency_ms)),
        ("routing_time_ms", float(decision.routing_time_ms)),
    )
    return [
        MetricEntry(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value,
            labels=labels.copy(),
        )
        for metric_name, value in values
    ]


def build_error_entry(
        exc: Exception, component: str, 
        query_id: Optional[UUID], extra: Dict) -> ErrorEntry:
    """Build an error event, including a traceback even outside except blocks."""
    stacktrace = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    return ErrorEntry(
        query_id=query_id,
        error_type=type(exc).__name__,
        error_message=str(exc),
        stacktrace=stacktrace,
        component=component,
        timestamp=datetime.now(timezone.utc),
        extra=dict(extra),
    )




# TASK 3.3: Kafka Producer Manager

class KafkaProducerManager:
    """Publish pipeline events to Kafka, degrading to a no-op when unreachable."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("kafka")
        # pipeline.enabled=False keeps the P2 chain byte-for-byte unchanged.
        self.enabled = config.get("pipeline", {}).get("enabled", False)

        kafka_config = config.get("kafka", {})
        producer_config = kafka_config.get("producer", {})
        # Topic metadata comes only from topics_file, the same source that
        # init-kafka-topics creates topics from (P3 §3.1).
        self.topics = _load_topic_metadata(kafka_config)
        self.bootstrap_servers = kafka_config.get(
            "bootstrap_servers", "localhost:9092")
        # aiokafka does not expose a max-in-flight constructor argument. Keep
        # the resolved P3 setting for producer backends that support it.
        self.max_in_flight = int(producer_config.get("max_in_flight", 5))
        self.producer_options = {
            "bootstrap_servers": self.bootstrap_servers,
            "acks": producer_config.get("acks", "all"),
            "max_batch_size": producer_config.get("batch_size", 16384),
            "linger_ms": producer_config.get("linger_ms", 5),
            "compression_type": producer_config.get("compression_type", "gzip"),
            "request_timeout_ms": producer_config.get("request_timeout_ms", 30000),
            "enable_idempotence": producer_config.get("enable_idempotence", True),
        }
        self.producer: Optional[AIOKafkaProducer] = None

    async def initialize(self) -> None:
        """Connect to Kafka; on failure degrade to disabled without raising."""
        if not self.enabled:
            return

        try:
            await self._connect()
            self.logger.info(
                "Kafka producer connected to %s", self.bootstrap_servers)
        except Exception as e:
            self.enabled = False
            self.logger.warning(
                "Kafka producer disabled: connection failed: %s", e)

    # No wait between the 3 attempts; reraise hands initialize() the last error.
    @retry(stop=stop_after_attempt(3), reraise=True)
    async def _connect(self) -> None:
        producer: Optional[AIOKafkaProducer] = None
        try:
            producer = AIOKafkaProducer(**self.producer_options)
            await producer.start()
        except Exception:
            if producer is not None:
                try:
                    await producer.stop()   # release a half-open connection
                except Exception:
                    pass
            raise
        self.producer = producer

    # skipping optional async def _ensure_topics_exist(self):

    async def produce(
            self, topic: str, key: Optional[str], value: BaseModel,
            headers: Optional[Dict[str, str]] = None) -> bool:
        """Send one event. Never raises, so the main chain cannot be blocked."""
        # Disabled means Kafka is unreachable, which is not the caller's failure:
        # produce() reports no-op success instead of propagating a degradation.
        if not self.enabled or self.producer is None:
            return True

        try:
            headers = {**(headers or {}), "query_id": key or "",
                       "produced_at": datetime.utcnow().isoformat(),
                       "schema_version": "1.0"}
            await self.producer.send_and_wait(
                topic,
                value.model_dump_json().encode("utf-8"),
                # Keying by query_id keeps one query's events on one partition.
                key=key.encode("utf-8") if key else None,
                headers=[(name, text.encode("utf-8"))
                         for name, text in headers.items()])
            PIPELINE_METRICS.kafka_produce_total.labels(
                topic=topic, status="success").inc()
            return True
        except Exception as e:
            PIPELINE_METRICS.kafka_produce_total.labels(
                topic=topic, status="failure").inc()
            self.logger.error(
                "Kafka produce failed: topic=%s, error=%s", topic, e)
            return False

    async def produce_batch(
            self, records: List[Tuple[str, Optional[str], BaseModel]]
    ) -> Tuple[int, int]:
        """Send several events concurrently and return (success, failure)."""
        results = await asyncio.gather(
            *(self.produce(topic, key, value) for topic, key, value in records))
        success = sum(results)
        return success, len(results) - success

    async def flush(self) -> None:
        if self.producer is None:
            return

        try:
            await self.producer.flush()
        except Exception as e:
            self.logger.error("Kafka producer flush failed: %s", e)

    async def shutdown(self) -> None:
        if self.producer is None:
            return

        try:
            await self.flush()
            await self.producer.stop()
            self.logger.info("Kafka producer stopped")
        except Exception as e:
            self.logger.error("Kafka producer shutdown failed: %s", e)
        finally:
            self.producer = None
            self.enabled = False


# TASK 3.4: Kafka Consumer Engine + ClickHouse Writer

def _ch_datetime(value: Any) -> str:
    """Render a datetime (or ISO string) as a ClickHouse DateTime64(3,'UTC')."""
    # Decision: format as "YYYY-MM-DD HH:MM:SS.mmm" instead of passing
    # .isoformat() through. JSONEachRow parses that shape on every ClickHouse
    # version, while an ISO string carrying a "+00:00" offset does not.
    if isinstance(value, datetime):
        dt = _as_utc(value)
    else:
        dt = _as_utc(datetime.fromisoformat(str(value)))
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


class KafkaConsumerEngine:
    """Consume pipeline events and dispatch them to ClickHouseWriter."""

    def __init__(
            self, config: Dict[str, Any],
            clickhouse_writer: "ClickHouseWriter",
            producer: KafkaProducerManager):
        self.logger = get_logger("kafka")
        pipeline_config = config.get("pipeline", {})
        self.enabled = pipeline_config.get("enabled", False)
        self.clickhouse_writer = clickhouse_writer
        # Decision: P3 §3.4 gives KafkaConsumerEngine(config, clickhouse_writer);
        # the producer is added so failed messages can reach llm-dead-letter.
        self.producer = producer

        kafka_config = config.get("kafka", {})
        consumer_config = kafka_config.get("consumer", {})

        self.bootstrap_servers = kafka_config.get(
            "bootstrap_servers", "localhost:9092")
        self.group_id = consumer_config.get(
            "group_id", "llm-router-clickhouse-consumer")
        self.auto_offset_reset = consumer_config.get(
            "auto_offset_reset", "earliest")
        self.max_poll_records = consumer_config.get("max_poll_records", 500)
        self.max_poll_interval_ms = consumer_config.get(
            "max_poll_interval_ms", 300000)
        self.fetch_min_bytes = consumer_config.get("fetch_min_bytes", 1024)
        self.fetch_max_wait_ms = consumer_config.get("fetch_max_wait_ms", 500)
        self.flush_interval_sec = pipeline_config.get(
            "flush_interval_ms", 5000) / 1000

        # Names come from topics_file like the producer; KafkaTopics only says
        # which event kind a name carries. The DLQ topic is not consumed, or
        # failures would loop back here.
        self.topics = [
            name for name in _load_topic_metadata(kafka_config)
            if name != KafkaTopics.DEAD_LETTER.value
        ]
        # A query and its response arrive as separate events; hold the first
        # half until its partner lands so query_logs gets one complete row.
        # Value: (topic, payload, partition, offset) of the waiting half.
        self._pending: Dict[
            str, Tuple[str, Dict[str, Any], TopicPartition, int]] = {}
        self._dlq_local_path = PROJECT_ROOT / pipeline_config.get(
            "dlq_local_dir", "data/dlq")

        # Manual-commit bookkeeping per partition: the offset a rewind returns
        # to (last commit, or the first offset seen) and the next unread offset.
        self._committed: Dict[TopicPartition, int] = {}
        self._positions: Dict[TopicPartition, int] = {}
        self._current: Optional[Tuple[TopicPartition, int]] = None
        # Serializes message handling with the periodic flush-and-commit.
        self._lock = asyncio.Lock()

        self.consumer: Optional[AIOKafkaConsumer] = None
        self.running = False
        self._consume_task: Optional[asyncio.Task[Any]] = None
        self._flush_task: Optional[asyncio.Task[Any]] = None

    async def initialize(self) -> None:
        """Create and start the Kafka consumer with graceful degradation."""
        if not self.enabled:
            return

        try:
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                # Manual commit is a P3 invariant, so the config value is ignored.
                enable_auto_commit=False,
                auto_offset_reset=self.auto_offset_reset,
                max_poll_records=self.max_poll_records,
                max_poll_interval_ms=self.max_poll_interval_ms,
                fetch_min_bytes=self.fetch_min_bytes,
                fetch_max_wait_ms=self.fetch_max_wait_ms,
            )
            await self.consumer.start()
            self.logger.info(
                "Kafka consumer started (group_id=%s)", self.group_id)
        except Exception as e:
            self.consumer = None
            self.enabled = False
            self.logger.warning(
                "Kafka consumer disabled: connection failed: %s", e)

    async def start(self) -> None:
        """Start the consume loop as one background asyncio task."""
        if not self.enabled or self.consumer is None or self.running:
            return

        self.running = True
        self._consume_task = asyncio.create_task(self._consume_loop())
        # pipeline.flush_interval_ms: flush and commit even when batch_size is
        # not reached, since `async for` blocks while no message arrives.
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def _consume_loop(self) -> None:
        """Consume messages one by one; offsets are committed by _flush_and_commit."""
        while self.running:
            try:
                async for msg in self.consumer:
                    tp = TopicPartition(msg.topic, msg.partition)
                    async with self._lock:
                        self._committed.setdefault(tp, msg.offset)
                        self._current = (tp, msg.offset)
                        ok = await self._handle_message(msg.topic, msg.value)
                        self._positions[tp] = msg.offset + 1
                        if ok:
                            PIPELINE_METRICS.messages_consumed.labels(
                                topic=msg.topic, group_id=self.group_id).inc()
                        else:
                            self._rewind()
            except Exception as e:
                PIPELINE_METRICS.consumer_errors.inc()
                self.logger.error("Consume loop failed: %s", e)
                await asyncio.sleep(1)

    async def _flush_loop(self) -> None:
        """Flush and commit every flush_interval_ms."""
        while self.running:
            await asyncio.sleep(self.flush_interval_sec)
            try:
                async with self._lock:
                    await self._flush_and_commit()
            except Exception as e:
                PIPELINE_METRICS.consumer_errors.inc()
                self.logger.error("Periodic flush failed: %s", e)

    async def _flush_and_commit(self) -> None:
        """Write every buffered row; commit only if every row is durable
        (in ClickHouse or the local DLQ)."""
        failed = 0
        for table in self.clickhouse_writer.tables:
            _, table_failed = await self.clickhouse_writer.flush_table(table)
            failed += table_failed
        if failed:
            self._rewind()
            return

        # A half row waiting in _pending is not in ClickHouse yet, so its
        # partition commits no further than that message.
        offsets: Dict[TopicPartition, int] = {}
        for tp, position in self._positions.items():
            waiting = [offset for _, _, pending_tp, offset
                       in self._pending.values() if pending_tp == tp]
            offset = min([position, *waiting])
            if offset > self._committed[tp]:
                offsets[tp] = offset
        if offsets:
            await self.consumer.commit(offsets)
            self._committed.update(offsets)

    def _rewind(self) -> None:
        """Rows lost to both ClickHouse and the DLQ are not committed and are
        consumed again."""
        # Every waiting half sits at or after its committed offset, so the
        # rewind replays it too; drop the stale copies.
        for tp, offset in self._committed.items():
            self.consumer.seek(tp, offset)
        self._positions = dict(self._committed)
        self._pending.clear()
        self.logger.warning(
            "Rows not written or dead-lettered; consumer rewound to "
            "committed offsets")

    async def _handle_message(self, topic: str, msg_value: bytes) -> bool:
        """Buffer one message; False only when the flush it triggered lost rows.

        A message that cannot be parsed or routed is dead-lettered and counts
        as handled, so one bad payload cannot stall its partition.
        """
        try:
            payload = json.loads(msg_value.decode("utf-8"))

            # (table, row) pairs this message adds to the ClickHouse buffers.
            writes: List[Tuple[str, Dict[str, Any]]] = []

            if topic == KafkaTopics.METRICS.value:
                writes.append((ClickHouseTables.SYSTEM_METRICS.value, {
                    **payload,
                    "timestamp": _ch_datetime(payload["timestamp"])}))

            elif topic == KafkaTopics.ERRORS.value:
                # schema.sql has no error table, so an error becomes a metric row.
                writes.append((ClickHouseTables.SYSTEM_METRICS.value, {
                    "timestamp": _ch_datetime(payload["timestamp"]),
                    "service": "llm-router", "metric_name": "error_event",
                    "value": 1.0,
                    "labels": {"query_id": str(payload.get("query_id") or ""),
                               "error_type": payload["error_type"],
                               "component": payload["component"],
                               "severity": payload["severity"]}}))

            elif topic in (
                    KafkaTopics.QUERIES.value, KafkaTopics.RESPONSES.value):
                query_id = payload["query_id"]
                other = self._pending.get(query_id)
                # A redelivered half of the same topic replaces the waiting one.
                if other is None or other[0] == topic:
                    tp, offset = self._current
                    self._pending[query_id] = (topic, payload, tp, offset)
                    return True                      # half a row; wait
                del self._pending[query_id]
                query, response = ((payload, other[1])
                                   if topic == KafkaTopics.QUERIES.value
                                   else (other[1], payload))
                # Entry field names are the query_logs column names, so the two
                # halves merge directly; response wins on status and on
                # token_count_input because it carries the actual token count.
                row = {**query, **response}
                row.pop("extra_labels", None)        # not a column
                row.pop("model_name", None)          # the column is selected_model
                for key in ("has_context", "has_attachments",
                            "cached", "compressed_context"):
                    row[key] = int(row[key])         # the columns are UInt8
                received_at = _as_utc(
                    datetime.fromisoformat(row["request_received_at"]))
                for key in ("request_received_at", "response_completed_at"):
                    row[key] = _ch_datetime(row[key])
                writes.append((ClickHouseTables.QUERY_LOGS.value, row))

                # One completed request adds one user_analytics row; the
                # SummingMergeTree sums the counters per (day, user_id, user_tier).
                writes.append((ClickHouseTables.USER_ANALYTICS.value, {
                    "day": received_at.date().isoformat(),
                    "user_id": row["user_id"],
                    "user_tier": row["user_tier"],
                    "request_count": 1,
                    "total_tokens_input": row["token_count_input"],
                    "total_tokens_output": row["token_count_output"],
                    "total_cost_usd": row["cost_usd"],
                    "error_count": int(row["status"] == "error"),
                    "avg_latency_ms": row["latency_ms"],
                }))
            else:
                raise ValueError(f"unknown topic: {topic}")
        except Exception as e:
            # P3 §3.4 point 5: the original message goes to llm-dead-letter,
            # with a local JSONL copy as double insurance. The local files keep
            # their own kafka_ prefix because replay_dlq() only replays batches
            # that name a ClickHouse table.
            now = datetime.now(timezone.utc)
            entry = DeadLetterEntry(
                original_topic=topic,
                original_message=msg_value.decode("utf-8", errors="replace"),
                failure_reason=str(e), failure_count=1,
                first_failed_at=now, last_failed_at=now)
            await self.producer.produce(
                KafkaTopics.DEAD_LETTER.value, None, entry)
            try:
                path = (self._dlq_local_path
                        / f"kafka_{now.strftime('%Y%m%d_%H')}.jsonl")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(entry.model_dump_json() + "\n")
            except Exception as write_error:
                self.logger.error("Local DLQ write failed: %s", write_error)
            PIPELINE_METRICS.dead_letter_total.labels(
                source="kafka", reason="handler_error").inc()
            self.logger.error("Message sent to DLQ: topic=%s, error=%s", topic, e)
            return True

        failed = 0
        for table, row in writes:
            _, table_failed = await self.clickhouse_writer.buffer_write(
                table, row)
            failed += table_failed
        return failed == 0

    async def stop(self) -> None:
        """Stop the background loops and close the Kafka consumer."""
        self.running = False
        tasks = [task for task in (self._consume_task, self._flush_task)
                 if task is not None]
        try:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.consumer is not None:
                await self.consumer.stop()
                self.logger.info("Kafka consumer stopped")
        except Exception as e:
            self.logger.error("Kafka consumer shutdown failed: %s", e)
        finally:
            self._consume_task = None
            self._flush_task = None
            self.consumer = None


class ClickHouseWriter:
    """Buffer pipeline rows, batch insert them, and persist failed batches."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("clickhouse")
        pipeline_config = config.get("pipeline", {})
        clickhouse_config = config.get("clickhouse", {})

        self.enabled = pipeline_config.get("enabled", False)
        self.host = clickhouse_config.get("host", "localhost")
        self.port = clickhouse_config.get("port", 8123)
        self.username = clickhouse_config.get("username", "default")
        password_env = clickhouse_config.get(
            "password_env", "CLICKHOUSE_PASSWORD")
        self.password = os.getenv(password_env, "")
        self.database = clickhouse_config.get("database", "default")
        self.schema_file = PROJECT_ROOT / clickhouse_config.get(
            "schema_file", "clickhouse/schema.sql")
        self.batch_size = clickhouse_config.get("batch_size", 200)
        self.retry_max = clickhouse_config.get("retry_max", 3)
        self.retry_backoff_base_ms = clickhouse_config.get(
            "retry_backoff_base_ms", 1000)
        self.connection_timeout_sec = clickhouse_config.get(
            "connection_timeout_sec", 10)
        self.send_receive_timeout_sec = clickhouse_config.get(
            "send_receive_timeout_sec", 300)

        # The tables clickhouse/schema.sql creates. A table name is interpolated
        # into SQL, so it must never come from message content.
        self.tables = {table.value for table in ClickHouseTables}

        self.client: Optional[Any] = None
        self.schema_result: Tuple[int, int] = (0, 0)
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._dlq_local_path = PROJECT_ROOT / pipeline_config.get(
            "dlq_local_dir", "data/dlq")

    async def initialize(self) -> None:
        """Connect to ClickHouse, prepare the DLQ directory, and apply DDL."""
        if not self.enabled:
            return

        try:
            if clickhouse_connect is None:
                raise ImportError("clickhouse_connect is not installed")

            self._dlq_local_path.mkdir(parents=True, exist_ok=True)
            (self._dlq_local_path / "archived").mkdir(
                parents=True, exist_ok=True)

            # HTTP client (port 8123) with max_connections=10 (P3 §3.4); every
            # clickhouse_connect call is blocking, so it runs in a thread.
            self.client = await asyncio.to_thread(
                clickhouse_connect.get_client,
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                database=self.database,
                connect_timeout=self.connection_timeout_sec,
                send_receive_timeout=self.send_receive_timeout_sec,
                pool_mgr=httputil.get_pool_manager(maxsize=10),
            )
            await asyncio.to_thread(self.client.command, "SELECT 1")
            self.logger.info(
                "ClickHouse connected: %s:%s", self.host, self.port)

            # Kept for init-clickhouse-schema, which reports these counts.
            self.schema_result = await self._create_tables_if_not_exists()
        except Exception as e:
            self.client = None
            self.enabled = False
            self.logger.warning(
                "ClickHouse writer disabled: connection failed: %s", e)

    async def _create_tables_if_not_exists(self) -> Tuple[int, int]:
        """Execute clickhouse/schema.sql; return (succeeded, failed) statements."""
        try:
            sql_text = self.schema_file.read_text(encoding="utf-8")
        except Exception as e:
            self.logger.warning("ClickHouse schema file unreadable: %s", e)
            return (0, 1)

        # Drop comment lines before splitting: a ";" inside a comment would
        # otherwise glue the comment's tail onto the next statement.
        sql_body = "\n".join(
            line for line in sql_text.splitlines()
            if line.strip() and not line.strip().startswith("--")
        )
        statements = [
            part.strip() for part in sql_body.split(";") if part.strip()
        ]

        failed = 0
        for statement in statements:
            try:
                await asyncio.to_thread(self.client.command, statement)
            except Exception as e:
                # DDL failure degrades ClickHouse; it must not crash startup.
                failed += 1
                self.logger.warning(
                    "ClickHouse DDL failed: %s | statement=%s", e, statement)
        return (len(statements) - failed, failed)

    async def buffer_write(
            self, table: str, row: Dict[str, Any]) -> Tuple[int, int]:
        """Append one row; at batch_size, flush and return (written, failed)."""
        # Returning the flush result lets the consumer see a failed write, per
        # P3 §3.4 "at most return failure count".
        if not self.enabled:
            return (0, 0)
        if table not in self.tables:
            self.logger.error("Unknown ClickHouse table: %s", table)
            return (0, 0)

        async with self._lock:
            buffer = self._buffers.setdefault(table, [])
            buffer.append(dict(row))
            should_flush = len(buffer) >= self.batch_size

        if should_flush:
            return await self.flush_table(table)
        return (0, 0)

    async def flush_table(self, table: str) -> Tuple[int, int]:
        """Write one table buffer and return (written, failed)."""
        async with self._lock:
            rows = self._buffers.get(table) or []
            if not rows:
                return (0, 0)
            # Detach the batch under the lock so new rows keep buffering.
            self._buffers[table] = []

        try:
            await self._execute_insert(table, rows)
            PIPELINE_METRICS.clickhouse_write_total.labels(
                table=table, status="success").inc()
            return (len(rows), 0)
        except Exception as e:
            PIPELINE_METRICS.clickhouse_write_total.labels(
                table=table, status="failure").inc()
            self.logger.error(
                "ClickHouse insert failed, batch sent to DLQ: table=%s, "
                "rows=%d, error=%s", table, len(rows), e)
            # Decision: `failed` counts rows stored nowhere durable (neither
            # ClickHouse nor the local DLQ). A batch persisted to the DLQ is
            # recoverable via replay-dlq, so the consumer may commit past it
            # instead of rewinding into an endless retry-and-dead-letter loop
            # while ClickHouse is down. The ClickHouse failure itself is still
            # counted by clickhouse_write_total{status="failure"} and
            # dead_letter_total.
            persisted = await self._write_to_dlq(table, rows, str(e))
            return (0, 0 if persisted else len(rows))

    async def flush_all(self) -> None:
        """Flush every non-empty table buffer independently."""
        for table in list(self._buffers.keys()):
            # flush_table never raises, so one table entering DLQ does not
            # stop the remaining tables from being written.
            await self.flush_table(table)

    async def _execute_insert(
            self, table: str, rows: List[Dict[str, Any]]) -> None:
        """Insert rows with exponential-backoff retry."""
        if self.client is None:
            raise RuntimeError("ClickHouse client is not connected")

        # Decision: raw_insert with fmt="JSONEachRow" instead of client.insert().
        # P3 §3.4 specifies INSERT ... FORMAT JSONEachRow with list[dict] rows,
        # and raw_insert takes the serialized rows without a column list.
        block = "\n".join(json.dumps(row, default=str) for row in rows)
        last_error: Optional[Exception] = None

        for attempt in range(self.retry_max):
            started = time.perf_counter()
            try:
                await asyncio.to_thread(
                    self.client.raw_insert,
                    table,
                    insert_block=block,
                    fmt="JSONEachRow",
                )
                PIPELINE_METRICS.clickhouse_write_latency_seconds.labels(
                    table=table).observe(time.perf_counter() - started)
                return
            except Exception as e:
                PIPELINE_METRICS.clickhouse_write_latency_seconds.labels(
                    table=table).observe(time.perf_counter() - started)
                last_error = e
                if attempt < self.retry_max - 1:
                    PIPELINE_METRICS.clickhouse_write_total.labels(
                        table=table, status="retry").inc()
                    # retry_max=3 attempts: 2 retries after 1s, then 2s.
                    await asyncio.sleep(
                        self.retry_backoff_base_ms / 1000 * (2 ** attempt))

        # flush_table() owns the DLQ fallback.
        raise last_error

    async def _write_to_dlq(
            self, table: str, rows: List[Dict[str, Any]],
            reason: str) -> bool:
        """Append one failed ClickHouse batch to an hourly JSONL file.

        Returns whether the batch reached disk.
        """
        now = datetime.now(timezone.utc)
        path = self._dlq_local_path / f"{now.strftime('%Y%m%d_%H')}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Append, so concurrent failures in the same hour are all kept.
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "table": table, "rows": rows, "reason": reason,
                    "timestamp": now.isoformat()}, default=str) + "\n")
            PIPELINE_METRICS.dead_letter_total.labels(
                source="clickhouse", reason="clickhouse_write_failed").inc()
            return True
        except Exception as e:
            self.logger.error("Local DLQ write failed: %s", e)
            return False

    async def replay_dlq(
            self, since: Optional[datetime] = None) -> Tuple[int, int]:
        """Replay eligible local DLQ files and return (succeeded, failed)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        if since is not None:
            cutoff = max(cutoff, _as_utc(since))

        success_count = 0
        failure_count = 0
        archived_dir = self._dlq_local_path / "archived"
        archived_dir.mkdir(parents=True, exist_ok=True)

        # Only the ClickHouse batch files (YYYYMMDD_HH.jsonl) are replayable;
        # kafka_*.jsonl holds raw payloads with no target table.
        for path in sorted(self._dlq_local_path.glob("[0-9]*.jsonl")):
            try:
                file_time = datetime.strptime(
                    path.stem, "%Y%m%d_%H").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            # The file covers one hour, so keep it when its hour ends after
            # the cutoff.
            if file_time + timedelta(hours=1) <= cutoff:
                continue

            file_ok = True
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    table = entry["table"]
                    rows = entry["rows"]
                    if table not in self.tables:
                        raise ValueError(f"unknown table: {table}")
                except Exception as e:
                    # An unreadable line counts as one failed entry.
                    file_ok = False
                    failure_count += 1
                    self.logger.error("DLQ entry unreadable: %s | %s", path, e)
                    continue

                try:
                    await self._execute_insert(table, rows)
                    success_count += len(rows)
                except Exception as e:
                    file_ok = False
                    failure_count += len(rows)
                    self.logger.error("DLQ replay failed: %s | %s", path, e)

            # Archive only a fully replayed file; partial failures stay active.
            if file_ok:
                path.replace(archived_dir / path.name)

        return (success_count, failure_count)

    async def shutdown(self) -> None:
        """Flush pending rows and close the ClickHouse client."""
        try:
            await self.flush_all()
            if self.client is not None:
                await asyncio.to_thread(self.client.close)
                self.logger.info("ClickHouse client closed")
        except Exception as e:
            self.logger.error("ClickHouse shutdown failed: %s", e)
        finally:
            self.client = None
            self.enabled = False


class PipelineManager:
    """Compose KafkaProducerManager, KafkaConsumerEngine and ClickHouseWriter."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = get_logger("pipeline")
        self.enabled = config.get("pipeline", {}).get("enabled", False)
        self.producer = KafkaProducerManager(config)
        self.ch_writer = ClickHouseWriter(config)
        self.consumer = KafkaConsumerEngine(
            config, self.ch_writer, self.producer)
        for source, reason in (
            ("clickhouse", "clickhouse_write_failed"),
            ("kafka", "handler_error"),
        ):
            PIPELINE_METRICS.dead_letter_total.labels(source=source, reason=reason)

    async def initialize(self) -> None:
        if not self.enabled:
            return
        await self.ch_writer.initialize()
        await self.producer.initialize()
        # Without ClickHouse there is nowhere to write, so do not consume.
        if self.ch_writer.enabled:
            await self.consumer.initialize()
        else:
            self.consumer.enabled = False
            self.logger.warning(
                "Kafka consumer skipped because ClickHouse is unavailable")

        if self.producer.enabled and self.ch_writer.enabled:
            self.logger.info("PipelineManager initialized")
        else:
            self.logger.info("Pipeline skipped: Kafka/ClickHouse not available")

    async def get_health_status(self) -> HealthStatus:

        kafka_ok = self.producer.enabled
        ch_ok = self.ch_writer.enabled

        if not self.enabled:
            status, message = "healthy", "pipeline disabled"
        elif kafka_ok and ch_ok:
            status, message = "healthy", "kafka and clickhouse both enabled"
        else:
            status = "degraded"
            message = (f"pipeline.enabled=True but "
                       f"kafka={kafka_ok}, clickhouse={ch_ok}")
        return HealthStatus(
            service_name="pipeline",
            status=status,
            message=message,
            last_check_at=datetime.now(timezone.utc),
            metadata={"kafka_ok": kafka_ok, "clickhouse_ok": ch_ok},
        )

    async def start_consumer(self) -> None:
        await self.consumer.start()

    async def stop_consumer(self) -> None:
        await self.consumer.stop()

    async def publish_pipeline_events(
        self,
        request: QueryRequest,
        routing_decision: RoutingDecision,
        inference_response: InferenceResponse,
        received_at: datetime,
    ) -> None:
        """Build and publish all events available for one inference request."""
        if not self.enabled:
            return

        try:
            key = str(request.request_id)
            events: List[Tuple[str, BaseModel]] = [
                (KafkaTopics.QUERIES.value,
                 build_query_log_entry(request, routing_decision, received_at)),
                # The hook passes no completion time; the task starts right
                # after /route has the response, so now() stands in for it.
                (KafkaTopics.RESPONSES.value,
                 build_response_log_entry(
                     request, routing_decision, inference_response,
                     datetime.now(timezone.utc))),
            ]
            events.extend(
                (KafkaTopics.METRICS.value, metric)
                for metric in build_metric_entries(
                    request, routing_decision, inference_response)
            )

            if inference_response.error:
                # P2 turns the exception into response.error, so the original
                # traceback is gone; the entry carries this one frame only.
                event_error = RuntimeError(inference_response.error)
                try:
                    query_id = UUID(request.request_id)
                except ValueError:
                    query_id = None
                events.append((
                    KafkaTopics.ERRORS.value,
                    build_error_entry(
                        event_error,
                        component="inference",
                        query_id=query_id,
                        extra={"model_name": inference_response.model_name},
                    ),
                ))

            # P3 §3.7 point 2: one producer.produce() call per event.
            results = [
                (topic, await self.producer.produce(topic, key, event))
                for topic, event in events
            ]

            # P3 §3.7 point 3: instrument once every produce has completed.
            # A disabled producer returns True without sending, so skip it.
            if self.producer.enabled:
                for topic, ok in results:
                    if ok:
                        PIPELINE_METRICS.messages_produced.labels(
                            topic=topic).inc()
                    else:
                        PIPELINE_METRICS.producer_errors.inc()
        except Exception as exc:
            self.logger.error("Pipeline event publication failed: %s", exc)

    async def flush_all(self) -> None:
        await self.producer.flush()
        await self.ch_writer.flush_all()

    async def shutdown(self) -> None:
        """Close all three components; main runs stop_consumer and flush_all first."""
        await self.producer.shutdown()
        await self.consumer.stop()      # no-op once stop_consumer() has run
        await self.ch_writer.shutdown()
        self.enabled = False
