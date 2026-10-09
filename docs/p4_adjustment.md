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

---

## ADJ-002: Scrape host-side targets via `host.docker.internal`

**Date**: 2026-10-08

**Decision**: Approved.

### What changed

| File | Change |
|---|---|
| `monitoring/prometheus.yml` | Targets of `llm-router-api`, `llm-router-inference`, `vllm-server`, `kafka-exporter`, `clickhouse-exporter`, `node-exporter` changed from `localhost:<port>` to `host.docker.internal:<port>`. `prometheus-self` stays `localhost:9090`; the `alerting` target (`localhost:9093`) is unchanged since no Alertmanager runs locally. Job names, ports and scrape intervals are unchanged. |

### Why

Prometheus runs in a Docker container (`scripts/start_monitoring_stack.sh`, P4 §5.1). Inside the container `localhost` is the container itself, so every target except `prometheus-self` was DOWN, including `llm-router-api`, which M3 requires to be UP. The service runs on the host and the exporters publish their ports on the host, so the container reaches all of them through `host.docker.internal`.

### Deviation from P4.md

- **P4.md §3.3** lists the scrape targets as `localhost:<port>`. The 7 job names, ports and intervals required by §3.3 are unchanged; only the host part differs.

### Compatibility / risk

- `host.docker.internal` resolves by default on Docker Desktop (macOS/Windows). On Linux the Prometheus container needs `--add-host=host.docker.internal:host-gateway`; the start script does not add it yet.
- A Prometheus binary run directly on the host (not in Docker) cannot resolve `host.docker.internal`; it would need the `localhost` targets back.
- The `prometheus.yml` template in `src/llm_router_part0_setup.py` still uses `localhost`. `setup` only writes the file when it is missing, so the repository copy is not affected.

### Follow-up / not done here

- Add `--add-host=host.docker.internal:host-gateway` to the Prometheus `docker run` in `scripts/start_monitoring_stack.sh` if the stack must run on Linux.
- The existing `ne-p4` container was created without `-p 9100:9100` and had to be recreated once (`docker rm -f ne-p4`, then rerun the start script), since the script reuses existing containers.
