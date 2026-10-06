# P4 Adjustments Log

Deviations from `docs/P4.md` made during P4 implementation, with rationale and scope.

---

## ADJ-001: Add `user_tier` label to inference metrics for per-tier SLO monitoring

**Date**: 2026-10-06

**Decision**: Approved (plan A over the ClickHouse-only alternative).

### What changed

| File | Change |
|---|---|
| `src/utils/metrics.py` | `InferenceMetrics.requests_total` labels: `[model_name, provider, status]` → `[model_name, provider, status, user_tier]`. `InferenceMetrics.request_duration_seconds` labels: `[model_name, provider]` → `[model_name, provider, user_tier]`. |
| `src/llm_router_part2_inference.py` (~L595) | Inference instrumentation now passes `user_tier=request.user_tier.value` on both `.labels(...)` calls. |
| `monitoring/grafana/dashboard.json` | Panel 7 retargeted to `sum by (le, model_name, user_tier)` with a `user_tier=~"$user_tier"` filter and legend `{{model_name}} / {{user_tier}}`; title → "P95 Latency Trend by Model x User Tier (ms)". The `user_tier` template variable changed from a hardcoded `custom` list to a Prometheus `query` variable: `label_values(llm_router_inference_requests_total, user_tier)`. |

### Why

P4 task 3.4 requires Panel 7 ("P95 Latency Trend faceted by model × user_tier") to break latency down by user tier, and this tier dimension is a mandatory part of the SLO. The P2 Prometheus inference metrics carried no `user_tier` label, so tier faceting was impossible from the metrics backend — only from ClickHouse `query_logs`, which depends on `pipeline.enabled=true` and ClickHouse being online.

Plan A (add the label) was chosen over the ClickHouse-only route so that per-tier latency is:
- available whenever `monitoring.enabled=true`, independent of the data pipeline;
- served from the same backend (Prometheus) that drives alerting, so tier SLOs can later be expressed as alert rules.

### Deviation from P4.md

- **P4.md §I.1.7** states the P2 Routing Inference MVP has "no internal changes" and that P4 "reads directly via instrumentation already present". Adding a label to the inference histogram/counter and populating it in the inference path extends that instrumentation, which goes beyond "no internal changes".
- **P4.md §3.2** metric tables list `requests_total` with labels `model_name, provider, status` and `request_duration_seconds` with `model_name, provider`. The `user_tier` label is an addition on top of those.

### Compatibility / risk

- Backward compatible for existing queries: RPS / success-rate / cost panels use `sum(...)` and the P95 panels aggregate with `sum by (le, model_name)`, all of which aggregate over the new label without change. `monitoring/alert_rules.yml` (HighErrorRate, HighLatencyP95) likewise sums over it.
- Cardinality impact is bounded: `user_tier` has 3 values (free / premium / enterprise), multiplied across model × provider × buckets.
- Only one instrumentation call site exists for each metric (`src/llm_router_part2_inference.py`), so there is no risk of a partially-labelled call raising `LabelError`.

### Follow-up / not done here

- Per-tier alert rules (tier-specific latency SLO thresholds) are not yet added to `monitoring/alert_rules.yml`.
- The M3 / §5.4 "Total Requests (Window)" stat panel discrepancy is tracked separately and intentionally left as-is pending confirmation (see conversation notes), not part of this adjustment.
