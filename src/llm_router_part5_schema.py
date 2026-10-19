from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from uuid import UUID
from src.utils.schema import UserTier

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
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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
    # Only input is known at ingress; the other partitions are filled node by node.
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
        # Decision: mode="json" so the dict goes straight to logger / Kafka;
        # UUID -> str, datetime -> ISO 8601 (UTC rendered as "Z", same as P2/P3).
        # "policy.remaining" is read as the four *_remaining_before/after fields.
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
