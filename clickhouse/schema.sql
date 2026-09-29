-- Read this SQL file in ClickHouseWriter.initialize(), split by semicolon,
-- then execute each non-empty statement.
-- The hourly_metrics materialized view writes query_logs aggregate states into
-- model_performance_agg and is enabled by default; model_performance is the
-- view that merges those states.

-- ============================================================
-- 1. query_logs: Query + Response merged detail table
--    MergeTree partitioned by month, TTL 90 days; BloomFilter index
-- ============================================================
CREATE TABLE IF NOT EXISTS query_logs
(
    query_id            UUID,
    user_id             String,
    user_tier           LowCardinality(String),
    query_text          String,
    query_type          LowCardinality(String),
    selected_model      LowCardinality(String),
    routing_strategy    LowCardinality(String),
    routing_confidence  Float32,
    token_count_input   UInt32,
    token_count_output  UInt32,
    total_tokens        UInt32,
    temperature         Float32,
    max_tokens          UInt32,
    has_context         UInt8,
    has_attachments     UInt8,
    provider            LowCardinality(String),
    response_text       String CODEC(ZSTD(1)),
    cost_usd            Decimal(18, 8),
    latency_ms          UInt32,
    routing_time_ms     UInt32,
    cached              UInt8,
    compressed_context  UInt8,
    status              LowCardinality(String),
    error               Nullable(String),
    request_received_at DateTime64(3, 'UTC'),
    response_completed_at DateTime64(3, 'UTC'),
    day                 Date MATERIALIZED toDate(request_received_at)
)
ENGINE = ReplacingMergeTree(response_completed_at)
PARTITION BY toYYYYMM(request_received_at)
ORDER BY (query_id, request_received_at)
PRIMARY KEY query_id
-- Decision: toDateTime() around the DateTime64 column. ClickHouse 23.8 (the
-- P3 D4 version) rejects a TTL whose result is DateTime64 (BAD_TTL_EXPRESSION).
TTL toDateTime(request_received_at) + INTERVAL 90 DAY
SETTINGS index_granularity = 8192;

-- BloomFilter index (optional: execute ALTER TABLE after table creation or inline directly)
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS bf_user_id user_id TYPE bloom_filter GRANULARITY 1;
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS bf_model   selected_model TYPE bloom_filter GRANULARITY 1;
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS bf_status  status TYPE bloom_filter GRANULARITY 1;
ALTER TABLE query_logs MATERIALIZE INDEX IF EXISTS bf_user_id;
ALTER TABLE query_logs MATERIALIZE INDEX IF EXISTS bf_model;
ALTER TABLE query_logs MATERIALIZE INDEX IF EXISTS bf_status;

-- ============================================================
-- 2. system_metrics: Metric details
--    ReplacingMergeTree, partitioned by day, TTL 30 days
-- ============================================================
CREATE TABLE IF NOT EXISTS system_metrics
(
    timestamp    DateTime64(3, 'UTC'),
    service      LowCardinality(String),
    metric_name  LowCardinality(String),
    value        Float64,
    labels       Map(LowCardinality(String), String),
    day          Date MATERIALIZED toDate(timestamp)
)
ENGINE = ReplacingMergeTree(timestamp)
PARTITION BY toYYYYMMDD(timestamp)
ORDER BY (service, metric_name, labels, timestamp)
-- toDateTime() for the same ClickHouse 23.8 DateTime64 TTL restriction.
TTL toDateTime(timestamp) + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- ============================================================
-- 3. model_performance_agg: Model dimension aggregate states
--    AggregatingMergeTree, partitioned by month, TTL 60 days
--    Written by hourly_metrics. Each insert block adds partial states that
--    merge correctly, so avg/p95/p99 stay exact across blocks.
-- ============================================================
CREATE TABLE IF NOT EXISTS model_performance_agg
(
    day                 Date,
    model_name          LowCardinality(String),
    provider            SimpleAggregateFunction(any, String),
    request_count       SimpleAggregateFunction(sum, UInt64),
    success_count       SimpleAggregateFunction(sum, UInt64),
    error_count         SimpleAggregateFunction(sum, UInt64),
    total_tokens_input  SimpleAggregateFunction(sum, UInt64),
    total_tokens_output SimpleAggregateFunction(sum, UInt64),
    total_cost_usd      SimpleAggregateFunction(sum, Decimal(38, 8)),
    avg_latency_ms      AggregateFunction(avg, UInt32),
    p95_latency_ms      AggregateFunction(quantileExact(0.95), UInt32),
    p99_latency_ms      AggregateFunction(quantileExact(0.99), UInt32),
    cache_hit_count     SimpleAggregateFunction(sum, UInt64),
    updated_at          SimpleAggregateFunction(max, DateTime64(3, 'UTC'))
)
ENGINE = AggregatingMergeTree
PARTITION BY toYYYYMM(day)
ORDER BY (day, model_name)
TTL day + INTERVAL 60 DAY
SETTINGS index_granularity = 8192;

-- ============================================================
-- 3b. model_performance: Model dimension aggregation (view)
--     Same columns as the P3 table; merges model_performance_agg states
-- ============================================================
CREATE VIEW IF NOT EXISTS model_performance
AS
SELECT
    day,
    model_name,
    any(provider)                                                             AS provider,
    sum(request_count)                                                        AS request_count,
    sum(success_count)                                                        AS success_count,
    sum(error_count)                                                          AS error_count,
    sum(total_tokens_input)                                                   AS total_tokens_input,
    sum(total_tokens_output)                                                  AS total_tokens_output,
    sum(total_cost_usd)                                                       AS total_cost_usd,
    avgMerge(avg_latency_ms)                                                  AS avg_latency_ms,
    quantileExactMerge(0.95)(p95_latency_ms)                                  AS p95_latency_ms,
    quantileExactMerge(0.99)(p99_latency_ms)                                  AS p99_latency_ms,
    sum(cache_hit_count)                                                      AS cache_hit_count,
    max(updated_at)                                                           AS updated_at
FROM model_performance_agg
GROUP BY day, model_name;

-- ============================================================
-- 4. user_analytics: User dimension aggregation table
--    SummingMergeTree, partitioned by month, TTL 180 days
-- ============================================================
CREATE TABLE IF NOT EXISTS user_analytics
(
    day                 Date,
    user_id             String,
    user_tier           LowCardinality(String),
    request_count       UInt64,
    total_tokens_input  UInt64,
    total_tokens_output UInt64,
    total_cost_usd      Decimal(32, 8),
    error_count         UInt64,
    avg_latency_ms      Float64
)
ENGINE = SummingMergeTree((request_count, total_tokens_input, total_tokens_output, total_cost_usd, error_count))
PARTITION BY toYYYYMM(day)
ORDER BY (day, user_id, user_tier)
TTL day + INTERVAL 180 DAY
SETTINGS index_granularity = 8192;

-- ============================================================
-- 5. hourly_metrics Materialized View: Real-time aggregation by model dimension per hour
--    Automatically triggered when writing from query_logs
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_metrics
TO model_performance_agg
AS
SELECT
    toDate(request_received_at)                                               AS day,
    selected_model                                                            AS model_name,
    any(provider)                                                             AS provider,
    count()                                                                   AS request_count,
    countIf(status = 'success')                                               AS success_count,
    countIf(status = 'error')                                                 AS error_count,
    sum(token_count_input)                                                    AS total_tokens_input,
    sum(token_count_output)                                                   AS total_tokens_output,
    sum(cost_usd)                                                             AS total_cost_usd,
    avgState(latency_ms)                                                      AS avg_latency_ms,
    quantileExactState(0.95)(latency_ms)                                      AS p95_latency_ms,
    quantileExactState(0.99)(latency_ms)                                      AS p99_latency_ms,
    sum(cached)                                                               AS cache_hit_count,
    now64(3, 'UTC')                                                           AS updated_at
FROM query_logs
GROUP BY
    toStartOfHour(request_received_at),
    toDate(request_received_at),
    selected_model
SETTINGS allow_experimental_analyzer = 1;
