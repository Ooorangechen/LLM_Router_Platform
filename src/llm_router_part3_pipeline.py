import traceback
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Tuple, Literal
from datetime import datetime, timezone
from src.utils.schema import RoutingDecision, InferenceResponse, QueryRequest
from src.utils.metrics import PIPELINE_METRICS
import asyncio 
from aiokafka import AIOKafkaProducer
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
    
