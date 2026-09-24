import json
import os
import re
import time
import traceback
from pathlib import Path
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Tuple, Literal
from datetime import datetime, timezone, timedelta
from src.utils.schema import RoutingDecision, InferenceResponse, QueryRequest
from src.utils.metrics import PIPELINE_METRICS
import asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from src.utils.constants import ClickHouseTables, KafkaTopics
from src.utils.logger import get_logger

# Decision: import clickhouse_connect optionally. P3 lists it as a hard
# dependency (D2), but a missing driver must degrade the writer the same way a
# refused connection does instead of breaking `import` for pipeline.enabled=False.
try:
    import clickhouse_connect
except ImportError:
    clickhouse_connect = None

# TASK 3.2

##### New Pydantic Models

def _as_utc(value: datetime) -> datetime:
    """Return an aware UTC datetime, treating legacy naive values as UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

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
    request_received_at: datetime

    status: Literal["received"] = "received"
    extra_labels: Dict[str, str] = Field(default_factory=dict)

    @field_validator("request_received_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

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

    response_completed_at: datetime

    status: Literal["success", "error"]

    @field_validator("response_completed_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

class MetricEntry(BaseModel):
    timestamp: datetime
    service: str = "llm-router"
    metric_name: str = Field(min_length=1)
    value: float 
    labels: Dict[str, str]

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

class ErrorEntry(BaseModel):
    error_id: UUID = Field(default_factory=uuid4)
    query_id: Optional[UUID] = None
    error_type: str = Field(min_length=1)
    error_message: str
    stacktrace: str
    component: str = Field(min_length=1)
    severity: Literal["warning", "error", "critical"] = "error"
    timestamp: datetime
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value)

class DeadLetterEntry(BaseModel):
    """encapsulates messages that failed to write"""

    dlq_id: UUID = Field(default_factory=uuid4)
    original_topic: str
    original_message: str
    failure_reason: str
    failure_count: int = Field(ge=1)
    first_failed_at: datetime
    last_failed_at: datetime

    @field_validator("first_failed_at", "last_failed_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        return _as_utc(value)


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
    total_tokens = response.total_tokens
    if total_tokens is None:
        total_tokens = response.token_count_input + response.token_count_output

    return ResponseLogEntry(
        query_id=UUID(request.request_id),
        user_id=request.user_id,
        model_name=response.model_name,
        provider=response.provider,
        response_text=response.response_text,
        token_count_input=response.token_count_input,
        token_count_output=response.token_count_output,
        total_tokens=total_tokens,
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
        configured_topics = kafka_config.get("topics", {})
        self.topics = {
            topic.name.lower(): configured_topics.get(
                topic.name.lower(), topic.value
            )
            for topic in KafkaTopics
        }
        self.bootstrap_servers = kafka_config.get(
            "bootstrap_servers", "localhost:9092")
        # aiokafka has no `retries` option, so P3's kafka.producer.retries is
        # spent on connection attempts in initialize() instead.
        self.max_attempts = int(producer_config.get("retries", 3))
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

        last_error: Optional[Exception] = None
        for _ in range(self.max_attempts):
            producer = AIOKafkaProducer(**self.producer_options)
            try:
                await producer.start()
                self.producer = producer
                self.logger.info(
                    "Kafka producer connected to %s", self.bootstrap_servers)
                return
            except Exception as e:
                last_error = e
                try:
                    await producer.stop()   # release a half-open connection
                except Exception:
                    pass

        self.enabled = False
        self.logger.warning(
            "Kafka producer disabled: connection failed: %s", last_error)

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
                       "produced_at": datetime.now(timezone.utc).isoformat(),
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
            clickhouse_writer: "ClickHouseWriter"):
        self.logger = get_logger("kafka")
        self.enabled = config.get("pipeline", {}).get("enabled", False)
        self.clickhouse_writer = clickhouse_writer

        kafka_config = config.get("kafka", {})
        consumer_config = kafka_config.get("consumer", {})
        configured_topics = kafka_config.get("topics", {})
        self.topic_names = {
            topic.name.lower(): configured_topics.get(
                topic.name.lower(), topic.value
            )
            for topic in KafkaTopics
        }

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

        # Business topics only: consuming the DLQ topic loops failures back here.
        self.topics = [
            self.topic_names["queries"],
            self.topic_names["responses"],
            self.topic_names["metrics"],
            self.topic_names["errors"],
        ]
        configured_tables = config.get("clickhouse", {}).get("tables", {})
        self.table_names = {
            table.name.lower(): configured_tables.get(
                table.name.lower(), table.value
            )
            for table in ClickHouseTables
        }
        # A query and its response arrive as separate events; hold the first
        # half until its partner lands so query_logs gets one complete row.
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._dlq_local_path = Path(
            config.get("pipeline", {}).get("dlq_local_dir", "data/dlq"))

        self.consumer: Optional[AIOKafkaConsumer] = None
        self.running = False
        self._consume_task: Optional[asyncio.Task[Any]] = None

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

    async def _consume_loop(self) -> None:
        """Poll, dispatch, flush, and manually commit Kafka messages."""
        while self.running:
            try:
                batches = await self.consumer.getmany(
                    timeout_ms=self.fetch_max_wait_ms,
                    max_records=self.max_poll_records)

                affected = set()
                for records in batches.values():
                    for record in records:
                        table = await self._handle_message(
                            record.topic, record.value)
                        if table is not None:
                            affected.add(table)
                            PIPELINE_METRICS.messages_consumed.labels(
                                topic=record.topic, group_id=self.group_id).inc()

                failed_total = 0
                for table in affected:
                    _, failed = await self.clickhouse_writer.flush_table(table)
                    failed_total += failed

                # Offsets advance only once the batch is durable. A failed batch
                # stays uncommitted and is redelivered; ReplacingMergeTree folds
                # away the duplicate rows that redelivery produces.
                if affected and not failed_total:
                    await self.consumer.commit()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Poll, dispatch, flush and commit all degrade the same way:
                # log it, keep the offset, keep consuming.
                PIPELINE_METRICS.consumer_errors.inc()
                self.logger.error("Consume batch failed: %s", e)
                await asyncio.sleep(1)

    async def _handle_message(
            self, topic: str, msg_value: bytes) -> Optional[str]:
        """Buffer one message and return the ClickHouse table it touched."""
        try:
            payload = json.loads(msg_value.decode("utf-8"))

            if topic == self.topic_names["metrics"]:
                table = self.table_names["system_metrics"]
                row = {**payload,
                       "timestamp": _ch_datetime(payload["timestamp"])}

            elif topic == self.topic_names["errors"]:
                # schema.sql has no error table, so an error becomes a metric row.
                table = self.table_names["system_metrics"]
                row = {"timestamp": _ch_datetime(payload["timestamp"]),
                       "service": "llm-router", "metric_name": "error_event",
                       "value": 1.0,
                       "labels": {"query_id": str(payload.get("query_id") or ""),
                                  "error_type": payload["error_type"],
                                  "component": payload["component"],
                                  "severity": payload["severity"]}}

            elif topic in (
                    self.topic_names["queries"],
                    self.topic_names["responses"]):
                table = self.table_names["query_logs"]
                other = self._pending.pop(payload["query_id"], None)
                if other is None:
                    self._pending[payload["query_id"]] = payload
                    return None                      # half a row; wait
                query, response = ((payload, other)
                                   if topic == self.topic_names["queries"]
                                   else (other, payload))
                # Entry field names are the query_logs column names, so the two
                # halves merge directly; response wins on status and on
                # token_count_input because it carries the actual token count.
                row = {**query, **response}
                row.pop("extra_labels", None)        # not a column
                row.pop("model_name", None)          # the column is selected_model
                for key in ("has_context", "has_attachments",
                            "cached", "compressed_context"):
                    row[key] = int(row[key])         # the columns are UInt8
                for key in ("request_received_at", "response_completed_at"):
                    row[key] = _ch_datetime(row[key])
            else:
                raise ValueError(f"unknown topic: {topic}")

            await self.clickhouse_writer.buffer_write(table, row)
            return table
        except Exception as e:
            # P3 §3.4: a message we cannot dispatch is persisted as-is and
            # skipped, so one bad payload cannot stall its partition forever.
            # These files keep their own kafka_ prefix because replay_dlq()
            # only replays batches that name a ClickHouse table.
            now = datetime.now(timezone.utc)
            path = self._dlq_local_path / f"kafka_{now.strftime('%Y%m%d_%H')}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(DeadLetterEntry(
                    original_topic=topic,
                    original_message=msg_value.decode("utf-8", errors="replace"),
                    failure_reason=str(e), failure_count=1,
                    first_failed_at=now,
                    last_failed_at=now).model_dump_json() + "\n")
            PIPELINE_METRICS.dead_letter_total.labels(source="kafka").inc()
            self.logger.error("Message sent to DLQ: topic=%s, error=%s", topic, e)
            return None

    async def stop(self) -> None:
        """Stop the background loop and close the Kafka consumer."""
        self.running = False
        try:
            if self._consume_task is not None:
                self._consume_task.cancel()
                await asyncio.gather(self._consume_task, return_exceptions=True)
            if self.consumer is not None:
                await self.consumer.stop()
                self.logger.info("Kafka consumer stopped")
        except Exception as e:
            self.logger.error("Kafka consumer shutdown failed: %s", e)
        finally:
            self._consume_task = None
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
        self.schema_file = Path(clickhouse_config.get(
            "schema_file", "clickhouse/schema.sql"))
        self.batch_size = clickhouse_config.get("batch_size", 200)
        self.retry_max = clickhouse_config.get("retry_max", 3)
        self.retry_backoff_base_ms = clickhouse_config.get(
            "retry_backoff_base_ms", 1000)
        self.connection_timeout_sec = clickhouse_config.get(
            "connection_timeout_sec", 10)
        self.send_receive_timeout_sec = clickhouse_config.get(
            "send_receive_timeout_sec", 300)

        # Allowed tables come from config; the defaults are the four tables in
        # clickhouse/schema.sql. The table name is interpolated into SQL, so it
        # must never come from message content.
        configured_tables = clickhouse_config.get("tables", {})
        self.table_names = {
            table.name.lower(): configured_tables.get(
                table.name.lower(), table.value)
            for table in ClickHouseTables
        }
        invalid_tables = [
            name for name in self.table_names.values()
            if not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
        ]
        if invalid_tables:
            raise ValueError(
                f"Invalid ClickHouse table names: {invalid_tables}")
        self.tables = set(self.table_names.values())
        if len(self.tables) != len(self.table_names):
            raise ValueError("ClickHouse table names must be unique")

        self.client: Optional[Any] = None
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._dlq_local_path = Path(pipeline_config.get(
            "dlq_local_dir", "data/dlq"))

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

            # HTTP client (port 8123); every clickhouse_connect call is
            # blocking, so it runs through asyncio.to_thread().
            self.client = await asyncio.to_thread(
                clickhouse_connect.get_client,
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                database=self.database,
                connect_timeout=self.connection_timeout_sec,
                send_receive_timeout=self.send_receive_timeout_sec,
            )
            await asyncio.to_thread(self.client.command, "SELECT 1")
            self.logger.info(
                "ClickHouse connected: %s:%s", self.host, self.port)

            await self._create_tables_if_not_exists()
        except Exception as e:
            self.client = None
            self.enabled = False
            self.logger.warning(
                "ClickHouse writer disabled: connection failed: %s", e)

    async def _create_tables_if_not_exists(self) -> None:
        """Execute every idempotent statement in clickhouse/schema.sql."""
        try:
            sql_text = self.schema_file.read_text(encoding="utf-8")
        except Exception as e:
            self.logger.warning("ClickHouse schema file unreadable: %s", e)
            return

        # The bundled schema uses enum defaults. Rewrite identifiers so table
        # overrides also apply to CREATE TABLE and materialized-view references.
        for table in ClickHouseTables:
            resolved_name = self.table_names[table.name.lower()]
            sql_text = re.sub(
                rf"\b{re.escape(table.value)}\b",
                resolved_name,
                sql_text,
            )

        statements = []
        for part in sql_text.split(";"):
            body = "\n".join(
                line for line in part.splitlines()
                if line.strip() and not line.strip().startswith("--")
            ).strip()
            if body:
                statements.append(body)

        for statement in statements:
            try:
                await asyncio.to_thread(self.client.command, statement)
            except Exception as e:
                # DDL failure degrades ClickHouse; it must not crash startup.
                self.logger.warning(
                    "ClickHouse DDL failed: %s | statement=%s", e, statement)

    async def buffer_write(
            self, table: str, row: Dict[str, Any]) -> None:
        """Append one row and flush its table when batch_size is reached."""
        if not self.enabled:
            return
        if table not in self.tables:
            self.logger.error("Unknown ClickHouse table: %s", table)
            return

        async with self._lock:
            buffer = self._buffers.setdefault(table, [])
            buffer.append(dict(row))
            should_flush = len(buffer) >= self.batch_size

        if should_flush:
            await self.flush_table(table)

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
            await self._write_to_dlq(table, rows, str(e))
            return (0, len(rows))

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
                    # 1s, 2s, 4s by default.
                    await asyncio.sleep(
                        self.retry_backoff_base_ms / 1000 * (2 ** attempt))

        # flush_table() owns the DLQ fallback.
        raise last_error

    async def _write_to_dlq(
            self, table: str, rows: List[Dict[str, Any]],
            reason: str) -> None:
        """Append one failed ClickHouse batch to an hourly JSONL file."""
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
                source="clickhouse").inc()
        except Exception as e:
            self.logger.error("Local DLQ write failed: %s", e)

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
