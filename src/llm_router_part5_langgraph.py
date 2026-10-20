import asyncio
import json
from pydantic import AfterValidator, AwareDatetime, BaseModel, Field
from typing import TYPE_CHECKING, Annotated, Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypedDict
from datetime import datetime, timezone
from time import perf_counter
from uuid import UUID
from src.utils.schema import InferenceResponse, QueryRequest, RoutingDecision, UserTier
from src.utils.logger import get_logger
from src.utils.metrics import INFERENCE_METRICS

if TYPE_CHECKING:
    # 只用于类型标注，运行时不 import：
    # - policy 模块在运行时 import 本模块的 RequestState / PolicyResult，这里再运行时 import 它会循环；
    # - router / inference 由构造函数注入（文档 3.3 D），本模块不依赖 P1/P2 模块本身。
    from src.llm_router_part1_router import ModelRouter
    from src.llm_router_part2_inference import InferenceEngine
    from src.llm_router_part5_policy import ExecutionPolicy
    from src.llm_router_part5_quality import QualityMonitor

logger = get_logger("langgraph")

try:
    from langgraph.graph import StateGraph, END, START
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    START, END = "__start__", "__end__"

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


################# Task 3.3: LangGraph State Graph Orchestration #################


NODE_CLASSIFY = "classify"
NODE_POLICY_CHECK = "policy_check"
NODE_COMPRESS = "compress"
NODE_SELECT_ADAPTER = "select_adapter"
NODE_EXECUTE = "execute"
NODE_FALLBACK = "fallback"
NODE_LOG = "log"

PATH_START, PATH_END = "START", "END"

ROUTE_PASS, ROUTE_REJECT = "pass", "reject"
ROUTE_SUCCESS, ROUTE_FAILURE = "success", "failure"

BREAK_POLICY_SKIP = "policy_skip"
BREAK_CIRCUIT_OPEN = "circuit_breaker_open"


def _merge_partition(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge: a node returns only the fields it changed inside a partition."""
    return {**(old or {}), **(new or {})}


def _apply_update(state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """The merge LangGraph applies through the GraphState reducers, for code paths outside
    the compiled graph (SimpleStateGraph, the forced log run in ainvoke, combining updates)."""
    merged = dict(state)
    for key, value in (update or {}).items():
        merged[key] = _merge_partition(merged.get(key, {}), value)
    return merged


class GraphState(TypedDict):
    input: Annotated[Dict[str, Any], _merge_partition]
    routing: Annotated[Dict[str, Any], _merge_partition]
    policy: Annotated[Dict[str, Any], _merge_partition]
    result: Annotated[Dict[str, Any], _merge_partition]
    metadata: Annotated[Dict[str, Any], _merge_partition]

class PolicyRejectedError(Exception):
    """Raised by execute when the quota gate rejects the request."""

    def __init__(self, break_reason: str, updates: Optional[Dict[str, Any]] = None):
        super().__init__(break_reason)
        self.break_reason = break_reason
        self.updates = updates or {}


class CircuitOpenError(Exception):
    """Raised by execute / fallback when a circuit breaker refuses the call."""

    def __init__(self, breaker_name: str, updates: Optional[Dict[str, Any]] = None):
        super().__init__(f"circuit breaker '{breaker_name}' is OPEN")
        self.breaker_name = breaker_name
        self.break_reason = BREAK_CIRCUIT_OPEN
        self.updates = updates or {}



class SimpleStateGraph:
    """Minimal implementation when LangGraph is not available."""

    def __init__(self, state_schema: Any = None):
        # state_schema 只为了和 StateGraph(GraphState) 的调用方式保持一致，不使用
        self._nodes: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}
        self._edges: Dict[str, str] = {}
        self._conditional: Dict[str, Tuple[Callable[[Dict[str, Any]], str], Dict[str, str]]] = {}

    def add_node(self, name: str, fn: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]) -> None:
        self._nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self._edges[src] = dst

    def add_conditional_edges(self, src: str, path_fn: Callable[[Dict[str, Any]], str],
                              path_map: Dict[str, str]) -> None:
        self._conditional[src] = (path_fn, path_map)

    def compile(self) -> "SimpleStateGraph":
        # 不需要编译，返回 self，让调用方式和 langgraph 一致
        return self

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        node = self._edges[START]
        while node != END:
            # same per-partition merge as the GraphState reducers
            state = _apply_update(state, await self._nodes[node](state))
            if node in self._conditional:
                path_fn, path_map = self._conditional[node]
                node = path_map[path_fn(state)]
            else:
                node = self._edges[node]
        return state


class LangGraphRouterOrchestrator:
    """7-node state graph: classify -> policy_check -> compress -> select_adapter -> execute -> [fallback] -> log"""

    def __init__(self, config: Dict[str, Any], router: "ModelRouter", inference: "InferenceEngine",
                 policy: Optional["ExecutionPolicy"] = None, quality_monitor: Optional["QualityMonitor"] = None,
                 pipeline: Optional[Any] = None):
        self.config = config or {}
        self.enabled: bool = bool(self.config.get("enabled", False))
        self.graph: Any = None  # compiled StateGraph / SimpleStateGraph, set by initialize()

        self.router = router
        self.inference = inference
        self.policy = policy
        self.quality_monitor = quality_monitor
        # Decision: `pipeline` kwarg is not in the 3.3 D signature; log needs it to publish P3 events.
        self.pipeline = pipeline

        # main.py only builds ExecutionPolicy / QualityMonitor when their section is enabled,
        # so None is the "disabled" switch itself.
        self._policy_enabled: bool = policy is not None
        self._quality_enabled: bool = quality_monitor is not None

        nodes_cfg = self.config.get("nodes", {}) or {}
        self._classify_enabled: bool = bool(nodes_cfg.get("classify_enabled", True))
        self._compress_enabled: bool = bool(nodes_cfg.get("compress_enabled", True))
        self._fallback_enabled: bool = bool(nodes_cfg.get("fallback_enabled", True))
        # Decision: max_node_retries not read (a retried execute would consume quota twice; fallback is the retry).
        self._use_simple_fallback: bool = bool(self.config.get("use_simple_graph_fallback", True))
        self._force_log: bool = bool(self.config.get("force_log_node_always", True))
        self._timeout_sec: float = float(self.config.get("timeout_sec", 60))

    async def initialize(self) -> None:
        """Build and compile the graph; nodes are bound methods, so dependencies come from self."""
        if not self.enabled:
            return
        if LANGGRAPH_AVAILABLE:
            builder, backend = StateGraph(GraphState), "langgraph"
        elif self._use_simple_fallback:
            builder, backend = SimpleStateGraph(GraphState), "simple"
        else:
            logger.warning("langgraph not installed and use_simple_graph_fallback=false; orchestrator disabled")
            self.enabled = False
            return

        nodes = {
            NODE_CLASSIFY: self.classify_node,
            NODE_POLICY_CHECK: self.policy_check_node,
            NODE_COMPRESS: self.compress_node,
            NODE_SELECT_ADAPTER: self.select_adapter_node,
            NODE_EXECUTE: self.execute_node,
            NODE_LOG: self.log_node,
        }
        if self._fallback_enabled:
            nodes[NODE_FALLBACK] = self.fallback_node
        for name, fn in nodes.items():
            builder.add_node(name, self._node_inject_exception_handler(name, fn))

        builder.add_edge(START, NODE_CLASSIFY)
        builder.add_edge(NODE_CLASSIFY, NODE_POLICY_CHECK)
        # rejections still pass through log for the audit trail
        builder.add_conditional_edges(NODE_POLICY_CHECK, self._route_after_policy,
                                      {ROUTE_PASS: NODE_COMPRESS, ROUTE_REJECT: NODE_LOG})
        builder.add_edge(NODE_COMPRESS, NODE_SELECT_ADAPTER)
        builder.add_edge(NODE_SELECT_ADAPTER, NODE_EXECUTE)
        builder.add_conditional_edges(NODE_EXECUTE, self._route_after_execute, {
            ROUTE_SUCCESS: NODE_LOG,
            ROUTE_FAILURE: NODE_FALLBACK if self._fallback_enabled else NODE_LOG,
            ROUTE_REJECT: NODE_LOG,
        })
        if self._fallback_enabled:
            builder.add_edge(NODE_FALLBACK, NODE_LOG)
        builder.add_edge(NODE_LOG, END)

        self.graph = builder.compile()
        logger.info("LangGraphRouterOrchestrator compiled (backend=%s)", backend)

    async def ainvoke(self, state: RequestState) -> RequestState:
        if not state.metadata.execution_path:
            # /route/advanced seeds ["START"] (3.4 B.4); seeded here too for direct callers
            state.append_execution_step(PATH_START)
        initial = state.model_dump()
        final = initial
        try:
            final = await asyncio.wait_for(self.graph.ainvoke(initial), timeout=self._timeout_sec)
        except Exception as exc:
            # timeout or graph-level failure (node exceptions are absorbed by the wrapper). Node updates
            # made so far are lost, so a quota consumed before the timeout is not refunded.
            reason = "timeout" if isinstance(exc, asyncio.TimeoutError) else f"{type(exc).__name__}: {exc}"
            logger.error("LangGraph run failed for %s: %s", state.input.request_id, reason)
            final = _apply_update(initial, {
                "result": {"success": False, "error_type": type(exc).__name__, "error_message": reason},
                "metadata": {"node_errors": {**initial["metadata"]["node_errors"], "__graph__": reason}},
            })
        if self._force_log and NODE_LOG not in final["metadata"]["execution_path"]:
            log_node = self._node_inject_exception_handler(NODE_LOG, self.log_node)
            final = _apply_update(final, await log_node(final))

        result = RequestState.model_validate(final)
        result.append_execution_step(PATH_END)
        return result

    def _node_inject_exception_handler(
        self, name: str, node_fn: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    ) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
        """Wrap a node: exceptions become state updates instead of aborting the graph."""
        async def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
            started = datetime.now(timezone.utc)
            error: Optional[str] = None
            try:
                update = await node_fn(state) or {}
            except (PolicyRejectedError, CircuitOpenError) as exc:
                policy_update: Dict[str, Any] = {"break_reason": exc.break_reason}
                if isinstance(exc, PolicyRejectedError):
                    # policy_check_passed=False sends execute's edge to log, not fallback
                    policy_update["policy_check_passed"] = False
                update = _apply_update(exc.updates, {
                    "policy": policy_update,
                    "result": {"success": False, "error_type": type(exc).__name__, "error_message": str(exc)},
                })
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("Node %s failed: %s", name, error)
                # Decision: other nodes failing only degrade (default model / fail-open / original context).
                update = {}
                if name in (NODE_EXECUTE, NODE_FALLBACK):
                    update["result"] = {"success": False, "error_type": type(exc).__name__,
                                        "error_message": str(exc)}
            return self._with_audit(name, state, update, started, error)
        return wrapped

    @staticmethod
    def _with_audit(name: str, state: Dict[str, Any], update: Dict[str, Any],
                    started: datetime, error: Optional[str]) -> Dict[str, Any]:
        # Decision: path step + timings recorded here once, not inside each node; failed nodes still appear.
        old = state["metadata"]
        meta = dict(update.get("metadata") or {})
        path = list(meta.get("execution_path", old["execution_path"]))
        if not path or path[-1] != name:
            path.append(name)
        meta["execution_path"] = path
        meta["node_started_at"] = {**old["node_started_at"], name: started}
        meta["node_finished_at"] = {**old["node_finished_at"], name: datetime.now(timezone.utc)}
        if error is not None:
            meta["node_errors"] = {**old["node_errors"], name: error}
        return {**update, "metadata": meta}

    @staticmethod
    def _route_after_policy(state: Dict[str, Any]) -> str:
        policy = state["policy"]
        if policy.get("policy_check_passed") is False and policy.get("break_reason") not in (BREAK_POLICY_SKIP, None):
            return ROUTE_REJECT
        return ROUTE_PASS

    @staticmethod
    def _route_after_execute(state: Dict[str, Any]) -> str:
        # Decision: extra "reject" branch; consume() may still reject under concurrency, which must skip fallback.
        if state["policy"].get("policy_check_passed") is False:
            return ROUTE_REJECT
        return ROUTE_SUCCESS if state["result"].get("success") in (True, None) else ROUTE_FAILURE

    ######## nodes ########

    async def classify_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self._classify_enabled:
            return {}
        # Decision: route_query (runs classify_query inside) instead of the classifier alone, so later
        # nodes get selected_model / fallback_models / estimated_cost; routing happens once per request.
        decision = await self.router.route_query(self._to_request(state["input"]))
        model_cfg = self.router.models.get(decision.selected_model)
        return {
            "routing": {
                "query_type": decision.query_type.value,
                "query_type_confidence": decision.confidence,
                "selected_model": decision.selected_model,
                "selected_provider": model_cfg.provider if model_cfg is not None else None,
                "routing_strategy": decision.routing_strategy,
                "routing_reason": decision.routing_reason,
                "routing_time_ms": decision.routing_time_ms,
            },
            "metadata": {"extra": {
                **state["metadata"]["extra"],
                "estimated_cost_usd": decision.estimated_cost,
                # kept for fallback (fallback_models) and the P3 pipeline publish in log
                "routing_decision": decision.model_dump(),
            }},
        }

    async def policy_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self._policy_enabled:
            return {"policy": {"policy_check_passed": True, "break_reason": BREAK_POLICY_SKIP}}
        result = await self.policy.check(RequestState.model_validate(state))
        # check() fails open on quota-store errors and reports it through message
        policy_error = result.message if result.message.startswith("policy_error") else None
        # quota is not consumed here; execute consumes it (3.3 B, policy_check step 5)
        return {"policy": {
            **self._policy_fields(result),
            "policy_applied": True,
            "policy_check_passed": result.passed,
            "break_reason": result.break_reason,
            "policy_error": policy_error,
        }}

    async def compress_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        compressor = self.inference.context_compressor
        context = state["input"].get("context_text")
        if not self._compress_enabled or not compressor.enabled or not context:
            return {"result": {"compressed_context": False}}
        model_name = state["routing"].get("selected_model") or self.router.default_model
        compressed = await compressor.compress_context(context, model_name)
        # below max_context_tokens compress_context returns the input unchanged
        if compressed == context:
            return {"result": {"compressed_context": False}}
        # recorded by process_query in P2, which execute bypasses
        INFERENCE_METRICS.compressions_total.labels(method=compressor.method).inc()
        return {"input": {"context_text": compressed}, "result": {"compressed_context": True}}

    async def select_adapter_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # P5 placeholder; AdapterManager arrives in P6
        return {"routing": {"adapter_name": None}}

    async def execute_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        policy = state["policy"]
        if policy.get("policy_check_passed") is False:
            raise PolicyRejectedError(policy.get("break_reason") or "policy_rejected")

        updates: Dict[str, Any] = {}
        if self._policy_enabled:
            state_obj = RequestState.model_validate(state)
            consumed = await self.policy.consume(state_obj, PolicyResult(passed=True, message="ok"))
            # only the remaining counts come back non-None; consume() also stores refund keys in extra
            updates = {"policy": {k: v for k, v in self._policy_fields(consumed).items() if v is not None},
                       "metadata": {"extra": state_obj.metadata.extra}}
            if not consumed.passed:
                # check() is a read-only pre-filter; consume()'s incr + recheck is the real gate
                raise PolicyRejectedError(consumed.break_reason, updates)

        model_name = state["routing"].get("selected_model") or self.router.default_model
        return await self._attempt(state, model_name, updates)

    async def fallback_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state["result"].get("fallback_used"):
            return {}  # one fallback per request
        decision = state["metadata"]["extra"].get("routing_decision") or {}
        fallback_models = decision.get("fallback_models") or []
        fallback_model = fallback_models[0] if fallback_models else self.router.default_model
        # no consume here: the quota was taken once by execute
        marker = {"result": {"fallback_used": True, "fallback_model": fallback_model}}
        return await self._attempt(state, fallback_model, marker)

    async def log_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state_obj = RequestState.model_validate(state)
        if self._policy_enabled and state_obj.result.success is not True:
            # Decision: refund here, not in execute (3.3 B step 6): a fallback success must stay charged (M2).
            # refund() is a no-op when nothing was consumed.
            reason = BREAK_CIRCUIT_OPEN if state_obj.policy.break_reason == BREAK_CIRCUIT_OPEN else "execution_failure"
            await self.policy.refund(state_obj, reason=reason)

        received_at = state_obj.input.received_at
        state_obj.metadata.total_elapsed_ms = int((datetime.now(timezone.utc) - received_at).total_seconds() * 1000)
        # the wrapper appends "log" after this node returns; add it now so the audit line includes it
        state_obj.append_execution_step(NODE_LOG)
        logger.info(json.dumps(state_obj.to_audit_log(), ensure_ascii=False))
        if self._quality_enabled:
            self.quality_monitor.end_request(state_obj)
        self._publish_pipeline_events(state_obj)
        return {"metadata": {"extra": state_obj.metadata.extra,
                             "total_elapsed_ms": state_obj.metadata.total_elapsed_ms}}

    ######## execution helpers (execute + fallback) ########

    async def _attempt(self, state: Dict[str, Any], model_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Breaker gate -> generate -> breaker report; `updates` is merged into the returned update."""
        circuits = self._circuits_for(model_name) if self._policy_enabled else []
        for cb in circuits:
            if not await cb.acquire():
                # circuits[-1] is the global breaker, which AdvancedRouteResponse reports (3.4 A)
                raise CircuitOpenError(cb.name, _apply_update(
                    updates, {"metadata": {"circuit_breaker_state": circuits[-1].state}}))

        response, error_type = await self._generate(state, model_name)
        failed = bool(response.error)
        extra = (updates.get("metadata") or {}).get("extra", state["metadata"]["extra"])
        more: Dict[str, Any] = {
            "result": {
                "success": not failed,
                "response_text": response.response_text,
                "token_count_input": response.token_count_input,
                "token_count_output": response.token_count_output,
                "total_tokens": response.total_tokens,
                "latency_ms": response.latency_ms,
                "cost_usd": response.cost_usd,
                "cached_response": response.cached,
                "provider": response.provider,
                # explicit None so a successful fallback clears execute's error
                "error_type": (error_type or "InferenceError") if failed else None,
                "error_message": response.error if failed else None,
            },
            "policy": {
                "cost_consumed_usd": (state["policy"].get("cost_consumed_usd") or 0.0) + response.cost_usd,
                "use_cache": self.inference.cache.enabled,
            },
            # kept for the P3 pipeline publish in log; fallback overwrites execute's response
            "metadata": {"extra": {**extra, "inference_response": response.model_dump()}},
        }
        for cb in circuits:
            await (cb.record_failure() if failed else cb.record_success())
        if circuits:
            more["metadata"]["circuit_breaker_state"] = circuits[-1].state
        return _apply_update(updates, more)

    async def _generate(self, state: Dict[str, Any], model_name: str) -> Tuple[InferenceResponse, Optional[str]]:
        """P2 process_query minus routing and compression (done by classify / compress):
        resolve provider -> cache lookup -> batch_processor -> cache write -> stats / metrics.
        Errors become InferenceResponse(error=...), as in P2."""
        start = perf_counter()
        request = self._to_request(state["input"])
        cache = self.inference.cache
        error_type: Optional[str] = None
        try:
            model_name, provider = self.inference._resolve_available_model(model_name)
            # Decision: key hashes the compressed context, so cache entries are not shared with /route.
            cache_key = cache.generate_cache_key(request, model_name) if cache.enabled else None
            if cache_key is not None:
                cached = await cache.get_cached_response(cache_key, model_name)
                if cached is not None:
                    # as in P2, cache hits return before stats / metrics
                    return InferenceResponse(**{
                        **cached,
                        "cached": True,
                        "latency_ms": int((perf_counter() - start) * 1000),
                    }), None
            response = await self.inference.batch_processor.add_request(request, provider, model_name)
            response.compressed_context = bool(state["result"].get("compressed_context"))
            if cache_key is not None:
                await cache.cache_response(cache_key, response.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("Inference failed for model '%s': %s", model_name, exc)
            error_type = type(exc).__name__
            response = InferenceResponse(
                response_text="", model_name=model_name, provider="error", error=str(exc),
                token_count_input=0, token_count_output=0, latency_ms=int((perf_counter() - start) * 1000),
                tokens_per_second=0.0, cost_usd=0.0,
            )
        await self._record_inference(request, response, error_type, start)
        return response, error_type

    async def _record_inference(self, request: QueryRequest, response: InferenceResponse,
                                error_type: Optional[str], start: float) -> None:
        # Same bookkeeping as the tail of InferenceEngine.process_query, which execute bypasses
        try:
            await self.inference._update_stats(response)
            model_name = response.model_name
            status = "error" if response.error else "success"
            duration = perf_counter() - start
            user_tier = request.user_tier.value
            self.router.update_model_stats(model_name, success=not response.error, latency_ms=response.latency_ms)
            if error_type:
                INFERENCE_METRICS.errors_total.labels(model_name=model_name, error_type=error_type).inc()
            INFERENCE_METRICS.requests_total.labels(model_name=model_name, provider=response.provider, status=status).inc()
            INFERENCE_METRICS.request_duration_seconds.labels(model_name=model_name, provider=response.provider).observe(duration)
            INFERENCE_METRICS.requests_by_tier_total.labels(model_name=model_name, provider=response.provider, status=status, user_tier=user_tier).inc()
            INFERENCE_METRICS.request_duration_by_tier_seconds.labels(model_name=model_name, provider=response.provider, user_tier=user_tier).observe(duration)
        except Exception as exc:
            logger.warning("Inference statistics update failed: %s", exc)

    def _circuits_for(self, model_name: str) -> List[Any]:
        # Decision: gate on global + per-model breakers; without an acquire() the global one could never leave OPEN.
        # Model first: a refusal there must not use up the global breaker's single HALF_OPEN probe.
        # With per_model=false both lookups return the same global breaker.
        global_cb = self.policy.circuit_breaker_for("global")
        model_cb = self.policy.circuit_breaker_for(model_name)
        return [global_cb] if model_cb is global_cb else [model_cb, global_cb]

    def _publish_pipeline_events(self, state_obj: RequestState) -> None:
        if self.pipeline is None or not self.pipeline.enabled:
            return
        decision = state_obj.metadata.extra.get("routing_decision")
        response = state_obj.metadata.extra.get("inference_response")
        if decision is None or response is None:
            return  # rejected before execute: there is no inference event to publish
        # non-blocking, same as the P3 hook in POST /route
        asyncio.create_task(self.pipeline.publish_pipeline_events(
            request=self._to_request(state_obj.input.model_dump()),
            routing_decision=RoutingDecision.model_validate(decision),
            inference_response=InferenceResponse.model_validate(response),
            received_at=state_obj.input.received_at,
        ))

    ######## state conversion ########

    @staticmethod
    def _to_request(inp: Dict[str, Any]) -> QueryRequest:
        # RequestState keeps only attachments_count (3.1), so attachments are not forwarded
        return QueryRequest(
            query=inp["query_text"],
            user_id=inp["user_id"],
            user_tier=inp["user_tier"],
            request_id=str(inp["request_id"]),
            context=inp.get("context_text"),
            max_tokens=inp["max_tokens"],
            temperature=inp["temperature"],
        )

    @staticmethod
    def _policy_fields(result: PolicyResult) -> Dict[str, Any]:
        return {
            "daily_quota_total": result.daily_quota_total,
            "daily_quota_remaining_before": result.daily_remaining_before,
            "daily_quota_remaining_after": result.daily_remaining_after,
            "hourly_quota_total": result.hourly_quota_total,
            "hourly_quota_remaining_before": result.hourly_remaining_before,
            "hourly_quota_remaining_after": result.hourly_remaining_after,
            "sla_target_ms": result.sla_target_ms,
            "cost_budget_usd": result.cost_budget_usd,
        }


class IntegratedLLMRouter:
    """P5 semi-finished entry wrapper; refined in P6"""

    def __init__(self, router: "ModelRouter", inference: "InferenceEngine",
                 policy: Optional["ExecutionPolicy"] = None,
                 orchestrator: Optional[LangGraphRouterOrchestrator] = None,
                 quality_monitor: Optional[Any] = None):
        self.router = router
        self.inference = inference
        self.policy = policy
        self.orchestrator = orchestrator
        self.quality_monitor = quality_monitor

    async def process_query_advanced(self, request: QueryRequest) -> Dict[str, Any]:
        # Decision: returns a dict keyed like AdvancedRouteResponse; 3.4 adds the Pydantic model and wraps it.
        state = RequestState(input=InputState(
            request_id=request.request_id,
            user_id=request.user_id,
            user_tier=request.user_tier,
            query_text=request.query,
            context_text=request.context,
            attachments_count=len(request.attachments),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ))
        state.append_execution_step(PATH_START)
        if self.quality_monitor is not None:
            self.quality_monitor.start_request(state)
        state = await self.orchestrator.ainvoke(state)

        inp, policy, result, meta = state.input, state.policy, state.result, state.metadata
        # Decision: -1 marks "unknown" (policy disabled) in quota_remaining, distinct from 0 remaining.
        unknown = -1
        return {
            "query_id": str(inp.request_id),
            "user_id": inp.user_id,
            "success": result.success is True,
            "response": result.response_text if result.success else None,
            "model_name": result.fallback_model if result.fallback_used else state.routing.selected_model,
            "provider": result.provider,
            "tokens": {
                "input": result.token_count_input,
                "output": result.token_count_output,
                "total": result.total_tokens,
            },
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "cached": result.cached_response,
            "compressed_context": result.compressed_context,
            "execution_path": meta.execution_path,
            "policy_applied": policy.policy_applied,
            "policy_rejected": policy.policy_check_passed is False,
            "break_reason": policy.break_reason,
            "quota_remaining": {
                "daily": policy.daily_quota_remaining_after if policy.daily_quota_remaining_after is not None else unknown,
                "hourly": policy.hourly_quota_remaining_after if policy.hourly_quota_remaining_after is not None else unknown,
                "sla_target_ms": policy.sla_target_ms if policy.sla_target_ms is not None else unknown,
                "cost_budget_usd": policy.cost_budget_usd if policy.cost_budget_usd is not None else float(unknown),
            },
            "circuit_breaker_state": meta.circuit_breaker_state,
            "error": result.error_message,
            "fallback_used": result.fallback_used,
            "total_elapsed_ms": meta.total_elapsed_ms or 0,
        }