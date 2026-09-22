import traceback
from pathlib import Path
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Tuple, Literal
from datetime import datetime, timezone
from src.utils.schema import RoutingDecision, InferenceResponse, QueryRequest
from src.utils.metrics import PIPELINE_METRICS
import asyncio 
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from src.utils.logger import get_logger

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

    def __init__(self, config:Dict[str, Any]):
        # unified logging system requirement
        self.logger = get_logger("kafka")

        # when `pipeline.enabled=False`, 
        # the main P2 chain behavior is completely consistent with P2 acceptance criteria
        self._pipeline_enabled = config.get("pipeline", {}).get('enabled', False) 
        self.enabled = self._pipeline_enabled

        kafka_config = config.get("kafka", {})
        self.bootstrap_servers = kafka_config.get("bootstrap_servers","localhost:9092")
        # read from config.yaml directly, ignoring topics_file for now. 
        self.topics : Dict[str, Any] = dict(kafka_config.get("topics", {}))

        producer_config = kafka_config.get("producer", {})
        self.max_attempts = int(producer_config.get("retries", 3))
        self.producer_options = {
            "bootstrap_servers": self.bootstrap_servers,
            "acks": producer_config.get("acks", "all"),
            "max_batch_size": producer_config.get("batch_size",16384),
            "linger_ms": producer_config.get("linger_ms",5),
            "compression_type": producer_config.get("compression_type","gzip"),
            "request_timeout_ms": producer_config.get("request_timeout_ms",30000),
            "enable_idempotence": producer_config.get("enable_idempotence",True),
        }
        self.producer: Optional[AIOKafkaProducer] = None

    async def initialize(self):
        # producer configuration read from config.yaml ovveriadable with default setting based on p3
        """
        `initialize()` internally `try` to connect Kafka (`@retry` 3 times), 
        all fail → `self.enabled=False` + `logger.warning("Kafka producer disabled: connection failed")`, 
        **does not throw exceptions upward**, service starts normally;
        """
        if not self.enabled:
            return 
        last_error: Optional[Exception] = None

        for _ in range(self.max_attempts):
            producer: Optional[AIOKafkaProducer] = None
            try:
                producer = AIOKafkaProducer(**self.producer_options)
                await producer.start()

                self.producer = producer
                self.logger.info("Kafka producer connected to %s", self.bootstrap_servers)
                return 
            except Exception as e:
                last_error = e
                if producer is not None:
                    try:
                        await producer.stop()
                    except Exception:
                        pass

        self.enabled = False
        self.producer = None
        self.logger.warning("Kafka producer disabled: connection failed: %s", last_error,)
    
    # skipping optional async def _ensure_topics_exist(self):

    async def produce(
            self, topic:str, key:Optional[str], 
            value:BaseModel, headers:Optional[Dict[str,str]] = None) -> bool:
        """
        If `pipeline.enabled=False`, main.py does not instantiate KafkaProducerManager at all;
        if `pipeline.enabled=True` but Kafka cannot connect, 
        instantiate but with enabled=False, 
        produce() directly returns True (no-op success), does not block the main chain.
        """
        if not self.enabled or self.producer is None:
            return True

        try:
            message_headers = dict(headers or {})
            message_headers.update({
                "query_id": key or "",
                "produced_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": "1.0"
            })

            kafka_headers = [
                (name, header_value.encode('utf-8')) 
                for name, header_value in message_headers.items()
            ]
        
            kafka_key = (key.encode('utf-8') if key is not None else None)

            await self.producer.send_and_wait(
                topic=topic,
                key=kafka_key,
                value=value.model_dump_json().encode("utf-8"),
                headers=kafka_headers
            )

            PIPELINE_METRICS.kafka_produce_total.labels(
                topic = topic, 
                status="success"
            ).inc()

            return True

        except Exception as e:
            PIPELINE_METRICS.kafka_produce_total.labels(
                topic=topic,
                status="failure"
            ).inc()

            self.logger.error(
                "Kafka produce failed: topic=%s, error=%s",
                topic,
                e,
            )

            return False
    
    async def produce_batch(
            self, records: List[Tuple[str, Optional[str], BaseModel]]) -> Tuple[int, int]:
        results = await asyncio.gather(
            *[
                self.produce(topic, key, value)
                for topic, key, value in records
            ]
        )

        success_count = sum(results)
        failure_count = len(results) - success_count
        return success_count, failure_count

    async def flush(self) -> None:  
        if self.producer is None:
            return

        try:
            await self.producer.flush()
        except Exception as e:
            self.logger.error(
                "Kafka producer flush failed: %s", e
            )

    async def shutdown(self) -> None:
        if self.producer is None:
            return 

        try:
            await self.producer.flush()
            await self.producer.stop()
            self.logger.info("Kafka producer stopped")
        except Exception as e:
            self.logger.error("" \
            "kafka producer shutdown failed: %s", e)
        finally:
            self.producer = None 
            self.enabled = False


# TASK 3.4: Kafka Consumer Engine + ClickHouse Writer

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
        topics = kafka_config.get("topics", {})

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

        # Subscribe only to business topics. The DLQ topic must not be consumed
        # by this engine, otherwise a failed DLQ message can create a loop.
        self.topics = [
            topics.get("queries", "llm-queries"),
            topics.get("responses", "llm-responses"),
            topics.get("metrics", "llm-metrics"),
            topics.get("errors", "llm-errors"),
        ]
        self.dead_letter_topic = topics.get(
            "dead_letter", "llm-dead-letter")

        self.consumer: Optional[AIOKafkaConsumer] = None
        self.running = False
        self._consume_task: Optional[asyncio.Task[Any]] = None

    async def initialize(self) -> None:
        """Create and start the Kafka consumer with graceful degradation."""
        # TODO:
        # 1. Return immediately when self.enabled is False.
        # 2. Construct AIOKafkaConsumer with self.topics and the configured
        #    bootstrap/group/poll/fetch options.
        # 3. Force enable_auto_commit=False; manual commit is a P3 invariant.
        # 4. Await consumer.start() and assign the connected consumer.
        # 5. On connection failure, set consumer=None and enabled=False, log a
        #    warning, and do not propagate the exception to application startup.
        if not self.enabled:
            return
        raise NotImplementedError

    async def start(self) -> None:
        """Start the consume loop as one background asyncio task."""
        # TODO:
        # 1. Return when disabled or when initialize() produced no consumer.
        # 2. Return when a consume task is already running.
        # 3. Set running=True.
        # 4. Store asyncio.create_task(self._consume_loop()) so stop() can wait
        #    for or cancel the same task during application shutdown.
        raise NotImplementedError

    async def _consume_loop(self) -> None:
        """Poll, dispatch, flush, and manually commit Kafka messages."""
        # TODO:
        # 1. Poll at most self.max_poll_records messages per batch. getmany()
        #    is the recommended implementation because commit is batch-based.
        # 2. Call _handle_message() for every record and collect the ClickHouse
        #    tables affected by the batch.
        # 3. Flush each affected table after dispatching the batch.
        # 4. Commit Kafka offsets only when every required ClickHouse flush
        #    succeeds. Never commit before durable persistence.
        # 5. On parsing/dispatch/write failure, leave the offset uncommitted,
        #    persist the original message to the DLQ, record pipeline metrics,
        #    and continue consuming rather than terminating this loop.
        # 6. Exit cleanly when self.running becomes False.
        raise NotImplementedError

    async def _handle_message(
            self, topic: str, msg_value: bytes) -> bool:
        """Decode one Kafka message and route it to the correct table buffer."""
        # TODO:
        # 1. Decode msg_value as UTF-8 and parse its JSON object.
        # 2. Route llm-metrics to the system_metrics buffer.
        # 3. Convert llm-errors to a system_metrics row whose metric_name is
        #    "error_event", unless a dedicated error table is added later.
        # 4. Route llm-queries and llm-responses toward query_logs. These two
        #    events contain partial rows, so pair them by query_id before asking
        #    ClickHouseWriter to buffer one complete query_logs row.
        # 5. Return True only when the message was accepted by the writer.
        # 6. On unknown topic or invalid JSON, write the original payload to the
        #    local/Kafka DLQ path and return False without raising upward.
        raise NotImplementedError

    async def stop(self) -> None:
        """Stop the background loop and close the Kafka consumer."""
        # TODO:
        # 1. Set running=False so _consume_loop can finish.
        # 2. Await or cancel self._consume_task and clear the task reference.
        # 3. Await consumer.stop() to leave the group and close connections.
        # 4. Clear self.consumer. Log shutdown errors instead of propagating
        #    them into the application shutdown sequence.
        raise NotImplementedError


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
        self.password = clickhouse_config.get("password", "")
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

        self.client: Optional[Any] = None
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._dlq_local_path = Path(pipeline_config.get(
            "dlq_local_dir", "data/dlq"))

    async def initialize(self) -> None:
        """Connect to ClickHouse, prepare the DLQ directory, and apply DDL."""
        # TODO:
        # 1. Return immediately when self.enabled is False.
        # 2. Create self._dlq_local_path and its archived/ child directory.
        # 3. Build a clickhouse_connect HTTP client from the configured host,
        #    port, credentials, database, and timeout values.
        # 4. Run blocking clickhouse_connect calls through asyncio.to_thread().
        # 5. Verify the connection, then call _create_tables_if_not_exists().
        # 6. On failure, set client=None and enabled=False, log a warning, and
        #    do not propagate the exception to application startup.
        raise NotImplementedError

    async def _create_tables_if_not_exists(self) -> None:
        """Execute every idempotent statement in clickhouse/schema.sql."""
        # TODO:
        # 1. Resolve and read self.schema_file as UTF-8 text.
        # 2. Split the file on semicolons and discard empty/comment-only parts.
        # 3. Execute each DDL statement sequentially through asyncio.to_thread.
        # 4. Log the failing statement and warning if DDL execution fails; table
        #    creation failure must degrade ClickHouse rather than crash startup.
        raise NotImplementedError

    async def buffer_write(
            self, table: str, row: Dict[str, Any]) -> None:
        """Append one row and flush its table when batch_size is reached."""
        # TODO:
        # 1. Return as a no-op when the writer is disabled.
        # 2. Validate table against the configured/allowed ClickHouse tables.
        # 3. Append a copy of row to self._buffers[table].
        # 4. Call flush_table(table) when the buffer reaches self.batch_size.
        # 5. Protect buffer mutation with an asyncio lock in the implementation
        #    because consumer and periodic flush tasks may run concurrently.
        raise NotImplementedError

    async def flush_table(self, table: str) -> Tuple[int, int]:
        """Write one table buffer and return (written, failed)."""
        # TODO:
        # 1. Return (0, 0) when the table buffer is empty.
        # 2. Atomically detach the current batch so new rows can keep buffering.
        # 3. Call _execute_insert(table, rows).
        # 4. On success, increment ClickHouse success/latency metrics and return
        #    (len(rows), 0).
        # 5. After final failure, call _write_to_dlq(), increment failure/DLQ
        #    metrics, and return (0, len(rows)); never raise to the consumer.
        raise NotImplementedError

    async def flush_all(self) -> None:
        """Flush every non-empty table buffer independently."""
        # TODO:
        # 1. Snapshot the current table names.
        # 2. Call flush_table() for every table with buffered rows.
        # 3. Continue with other tables when one table fails and enters DLQ.
        raise NotImplementedError

    async def _execute_insert(
            self, table: str, rows: List[Dict[str, Any]]) -> None:
        """Insert rows with exponential-backoff retry."""
        # TODO:
        # 1. Build the batch INSERT for the trusted table name using the
        #    JSONEachRow-compatible dictionaries.
        # 2. Execute the blocking client call with asyncio.to_thread().
        # 3. Record pipeline_clickhouse_write_latency_seconds per attempt.
        # 4. On failure, retry up to self.retry_max with exponential delays
        #    derived from self.retry_backoff_base_ms (1s, 2s, 4s by default).
        # 5. Raise the final exception to flush_table(), which owns DLQ fallback.
        raise NotImplementedError

    async def _write_to_dlq(
            self, table: str, rows: List[Dict[str, Any]],
            reason: str) -> None:
        """Append one failed ClickHouse batch to an hourly JSONL file."""
        # TODO:
        # 1. Select data/dlq/YYYYMMDD_HH.jsonl using the current UTC time.
        # 2. Serialize table, rows, reason, and timestamp as one JSON line.
        # 3. Append rather than overwrite so concurrent failures are retained.
        # 4. Increment pipeline_dead_letter_total{source="clickhouse"}.
        # 5. Log local-DLQ write failure without interrupting the consumer loop.
        raise NotImplementedError

    async def replay_dlq(
            self, since: Optional[datetime] = None) -> Tuple[int, int]:
        """Replay eligible local DLQ files and return (succeeded, failed)."""
        # TODO:
        # 1. Scan JSONL files from the last seven days, additionally applying
        #    the optional UTC since boundary.
        # 2. Parse every stored batch and retry it through _execute_insert().
        # 3. Count successful and failed rows/batches consistently.
        # 4. Move a source file to data/dlq/archived/ only when every entry in
        #    that file was replayed successfully; keep partial failures active.
        # 5. Return aggregate (success_count, failure_count).
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Flush pending rows and close the ClickHouse client."""
        # TODO:
        # 1. Await flush_all() before closing the client.
        # 2. Close the blocking client through asyncio.to_thread().
        # 3. Clear self.client and set enabled=False.
        # 4. Log shutdown failures without propagating them.
        raise NotImplementedError
