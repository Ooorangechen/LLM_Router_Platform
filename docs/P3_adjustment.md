# P3 Implementation Adjustments

This document tracks implementation changes that intentionally differ from
`docs/P3.md`. The original P3 document remains unchanged.

| P3 design | Adjustment | Reason | Files | Verification |
|---|---|---|---|---|
| P2 documents `process_query()` as returning only `InferenceResponse`. | It now returns `(Optional[RoutingDecision], InferenceResponse)` to the internal `/route` orchestration layer. | Task 3.7 requires `/route` to publish the exact decision and response produced by one inference execution. | `src/llm_router_part2_inference.py`, `main.py` | P2 tuple-return and route-hook tests |
| Task 3.7 shows a decision is available to the route hook. | Routing failures return `(None, error_response)` and skip the business-event hook rather than fabricating routing data. | Kafka routing fields must describe a real decision. | `src/llm_router_part2_inference.py`, `main.py` | Routing-failure test |
| Each P3 event model declares its own UTC validator. | All P3 timestamp fields share one `UTCDateTime` annotation; conversion to ClickHouse text happens once at the Kafka-consumer boundary. | Preserves the UTC contract without repeating validators or reconverting inside the writer. | `src/llm_router_part3_pipeline.py` | Kafka JSON and ClickHouse UTC tests |

## Compatibility

- `InferenceEngine.process_query()` is an internal tuple-returning interface.
- `src/utils/schema.py` and the external `/route` response body are unchanged.
- With `pipeline.enabled: false`, P3 is not imported or initialized by the
  service bootstrap path.
- Kafka publication is scheduled only by `/route` with `asyncio.create_task()`.
