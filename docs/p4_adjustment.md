# P4 Adjustments Log

Deviations from `docs/P4.md` made during P4 implementation, with rationale and scope.

---

## ADJ-001: Add parallel `user_tier`-labelled inference metrics for per-tier SLO monitoring

**Date**: 2026-10-06 (revised 2026-10-07)

**Decision**: Approved. Revised 2026-10-07: the frozen P2 metrics must not change signature, so the tier dimension is carried by **new parallel metrics** rather than by adding a label to the existing ones.

### What changed

| File | Change |
|---|---|
| `src/utils/metrics.py` | Added two new metrics, leaving the frozen `requests_total` / `request_duration_seconds` untouched: `requests_by_tier_total` → `llm_router_inference_requests_by_tier_total`, labels `[model_name, provider, status, user_tier]`; `request_duration_by_tier_seconds` → `llm_router_inference_request_duration_by_tier_seconds`, labels `[model_name, provider, user_tier]` (same buckets as the frozen histogram). |
| `src/llm_router_part2_inference.py` (~L595) | Inference instrumentation keeps the original two `.inc()` / `.observe()` calls unchanged and adds two parallel calls that pass `user_tier=request.user_tier.value`, reusing the request object already in scope. Emitted at the inference layer, so data exists even when `pipeline.enabled=false`. |
| `monitoring/grafana/dashboard.json` | Panel 7 expr now reads `llm_router_inference_request_duration_by_tier_seconds_bucket`; the `user_tier` template variable now queries `label_values(llm_router_inference_requests_by_tier_total, user_tier)`. Panel 9 error table is unchanged (it reads the ClickHouse `query_logs.user_tier` column, not a Prometheus label). |

### Why

P4 task 3.4 requires Panel 7 ("P95 Latency Trend faceted by model × user_tier") to break latency down by user tier, and this tier dimension is a mandatory part of the SLO. The P2 Prometheus inference metrics carried no `user_tier` label, so tier faceting was impossible from the metrics backend — only from ClickHouse `query_logs`, which depends on `pipeline.enabled=true` and ClickHouse being online.

The Prometheus route was chosen over the ClickHouse-only one so that per-tier latency is available whenever `monitoring.enabled=true` (independent of the data pipeline) and served from the same backend that drives alerting. To get this **without** breaking the cross-stage freeze contract on the P2 metrics, the tier dimension is carried by new, separately-named metrics instead of a label added to the frozen ones.

### Deviation from P4.md

- **P4.md §I.1.7** states the P2 Routing Inference MVP has "no internal changes" and that P4 "reads directly via instrumentation already present". Adding new metric instrumentation to the inference path extends that instrumentation, which goes beyond "no internal changes" — but the existing metrics keep their exact signatures.
- **P4.md §3.2** metric tables list `requests_total` and `request_duration_seconds` with their original labels; these are unchanged. The tier-faceted series come from the two added metrics listed above.

### Compatibility / risk

- The frozen metrics are byte-for-byte unchanged, so every existing query, panel and alert rule (`monitoring/alert_rules.yml`: HighErrorRate, HighLatencyP95) is unaffected.
- The new metrics duplicate the request/duration counts along a tier breakdown, so **the paired metrics must not be summed together** (double counting). Treat `*_by_tier_*` as the tier view only.
- Cardinality impact is bounded: `user_tier` has 3 values (free / premium / enterprise), multiplied across model × provider × (buckets) on the new series only.
- Each new metric has a single instrumentation call site (`src/llm_router_part2_inference.py`), co-located with the frozen calls, so there is no risk of a partially-labelled call raising `LabelError`.

### Follow-up / not done here

- Per-tier alert rules (tier-specific latency SLO thresholds) are not yet added to `monitoring/alert_rules.yml`.
- The M3 / §5.4 "Total Requests (Window)" stat panel discrepancy is tracked separately and intentionally left as-is pending confirmation (see conversation notes), not part of this adjustment.
