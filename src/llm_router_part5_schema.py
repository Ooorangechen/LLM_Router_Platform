from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4
from pydantic import AfterValidator, AwareDatetime, BaseModel, Field, model_validator
from src.utils.schema import UserTier


################# Task 3.1: RequestState #################


def _to_utc(v: datetime) -> datetime:
    return v.astimezone(timezone.utc)


class InputState(BaseModel):
    request_id: UUID
    user_id: str
    user_tier: UserTier
    tenant_id: Optional[str] = None
    domain: Optional[str] = None
    query_text: str
    context_text: Optional[str] = None
    attachments_count: int
    temperature: float
    max_tokens: int
    # Decision: naive datetimes are rejected rather than guessed (P2 timestamps are naive local time);
    # aware ones are normalized to UTC (3.1 core constraint).
    received_at: Annotated[AwareDatetime, AfterValidator(_to_utc)] = Field(
        default_factory=lambda: datetime.now(timezone.utc))

class RoutingState(BaseModel):
    query_type: Optional[str] = None
    query_type_confidence: Optional[float] = None
    selected_model: Optional[str] = None
    selected_provider: Optional[str] = None
    routing_strategy: Optional[str] = None
    routing_reason: Optional[str] = None
    adapter_name: Optional[str] = None
    routing_time_ms: Optional[int] = None

class PolicyState(BaseModel):
    policy_applied: bool = False
    policy_check_passed: Optional[bool] = None
    daily_quota_total: Optional[int] = None
    daily_quota_remaining_before: Optional[int] = None
    daily_quota_remaining_after: Optional[int] = None
    hourly_quota_total: Optional[int] = None
    hourly_quota_remaining_before: Optional[int] = None
    hourly_quota_remaining_after: Optional[int] = None
    sla_target_ms: Optional[int] = None
    cost_budget_usd: Optional[float] = None
    cost_consumed_usd: Optional[float] = None
    use_cache: Optional[bool] = None
    batch_id: Optional[str] = None
    break_reason: Optional[str] = None
    policy_error: Optional[str] = None

class ResultState(BaseModel):
    success: Optional[bool] = None
    response_text: Optional[str] = None
    token_count_input: Optional[int] = None
    token_count_output: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    cached_response: Optional[bool] = None
    compressed_context: Optional[bool] = None
    provider: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    fallback_used: bool = False
    fallback_model: Optional[str] = None

class MetadataState(BaseModel):
    execution_path: List[str] = []
    circuit_breaker_state: Optional[str] = None # in CLOSED, OPEN, HALF_OPEN
    node_started_at: Dict[str, datetime] = {}
    node_finished_at: Dict[str, datetime] = {}
    node_errors: Dict[str, str] = {}
    total_elapsed_ms: Optional[int] = None
    server_version: str = "P5-v1.0"
    extra: Dict[str, Any] = {}

class PolicyResult(BaseModel):
    passed: bool
    break_reason: Optional[str] = None
    # Decision: 文档的 PolicyResult 没有总量字段，加上后 policy_check 节点可以直接回填 PolicyState 的两个 *_total
    daily_quota_total: Optional[int] = None
    hourly_quota_total: Optional[int] = None
    daily_remaining_before: Optional[int] = None
    daily_remaining_after: Optional[int] = None
    hourly_remaining_before: Optional[int] = None
    hourly_remaining_after: Optional[int] = None
    sla_target_ms: Optional[int] = None
    cost_budget_usd: Optional[float] = None
    message: str

class QualitySnapshot(BaseModel):
    window_minutes: int
    request_count: int
    success_rate: float
    error_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    slo_ok: bool
    slo_violations: List[str]
    hotspot_models: List[str]
    snapshot_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RequestState(BaseModel):
    input: InputState
    routing: RoutingState = Field(default_factory=RoutingState)
    policy: PolicyState = Field(default_factory=PolicyState)
    result: ResultState = Field(default_factory=ResultState)
    metadata: MetadataState = Field(default_factory=MetadataState)

    def append_execution_step(self, node_name: str) -> None:
        path = self.metadata.execution_path
        if not path or path[-1] != node_name:
            path.append(node_name)

    # no tz normalization here. Callers pass datetime.now(timezone.utc),
    # and dict item assignment would bypass pydantic validators anyway.
    def mark_node_start(self, node: str, t: datetime) -> None:
        self.metadata.node_started_at[node] = t

    def mark_node_end(self, node: str, t: datetime) -> None:
        self.metadata.node_finished_at[node] = t

    def to_audit_log(self) -> Dict[str, Any]:
        return {
            "input": self.input.model_dump(mode="json"),
            "policy": self.policy.model_dump(mode="json", include={
                "daily_quota_remaining_before",
                "daily_quota_remaining_after",
                "hourly_quota_remaining_before",
                "hourly_quota_remaining_after",
            }),
            "result": self.result.model_dump(mode="json", include={
                "success", "latency_ms", "cost_usd"}),
            "execution_path": list(self.metadata.execution_path),
        }


################# Task 3.5: QualityMonitor record #################


class QualityRecord(BaseModel):
    request_id: UUID
    user_id: str
    model: str
    tier: str
    success: bool
    latency_ms: int
    cost_usd: float
    tokens_total: int
    error_type: Optional[str] = None
    ts: datetime


################# Task 3.6: FeedbackCollector record #################


LIKE_VALUES = {"like": 1, "dislike": -1}


class FeedbackRecord(BaseModel):
    feedback_id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    user_id: str = ""   # empty -> FeedbackCollector.submit applies feedback.allow_anonymous
    feedback_type: Literal["like", "dislike", "rating", "comment"]
    like_value: Optional[int] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=2000)
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    created_at: Annotated[AwareDatetime, AfterValidator(_to_utc)] = Field(
        default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _check_type_fields(self) -> "FeedbackRecord":
        if self.feedback_type == "rating" and self.rating is None:
            raise ValueError("rating is required for feedback_type=rating")
        # Decision: like_value is derived from feedback_type (like=+1 / dislike=-1 / others None),
        # so a client cannot send a "like" carrying -1.
        self.like_value = LIKE_VALUES.get(self.feedback_type)
        return self


################# Task 3.4: POST /route/advanced response #################


class AdvancedRouteResponse(BaseModel):
    # first 7 keys match POST /route: query_id / response / model_name / tokens / cost_usd / latency_ms / cached
    query_id: UUID
    user_id: str
    success: bool
    response: Optional[str] = None
    model_name: Optional[str] = None
    provider: Optional[str] = None
    tokens: Optional[Dict[str, Optional[int]]] = None   # input / output / total
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    cached: Optional[bool] = None
    compressed_context: Optional[bool] = None
    execution_path: List[str]
    policy_applied: bool
    policy_rejected: bool
    break_reason: Optional[str] = None
    quota_remaining: Dict[str, Any]   # daily / hourly / sla_target_ms / cost_budget_usd
    circuit_breaker_state: Optional[str] = None   # global breaker
    error: Optional[str] = None
    fallback_used: bool
    total_elapsed_ms: int
