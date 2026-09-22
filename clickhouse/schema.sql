-- Read this SQL file in ClickHouseWriter.initialize(), split by semicolon,
-- then execute each non-empty statement.
-- The hourly_metrics materialized view writes query_logs aggregates into
-- model_performance and is enabled by default.

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
TTL request_received_at + INTERVAL 90 DAY
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
TTL timestamp + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- ============================================================
-- 3. model_performance: Model dimension aggregation table
--    ReplacingMergeTree, partitioned by month, TTL 60 days
--    Written by consumer engine or Flink job
-- ============================================================
CREATE TABLE IF NOT EXISTS model_performance
(
    day                 Date,
    model_name          LowCardinality(String),
    provider            LowCardinality(String),
    request_count       UInt64,
    success_count       UInt64,
    error_count         UInt64,
    total_tokens_input  UInt64,
    total_tokens_output UInt64,
    total_cost_usd      Decimal(32, 8),
    avg_latency_ms      Float64,
    p95_latency_ms      Float64,
    p99_latency_ms      Float64,
    cache_hit_count     UInt64,
    updated_at          DateTime64(3, 'UTC') DEFAULT now64()
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(day)
ORDER BY (day, model_name)
TTL day + INTERVAL 60 DAY
SETTINGS index_granularity = 8192;

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
TO model_performance
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
    avg(latency_ms)                                                           AS avg_latency_ms,
    quantileExact(0.95)(latency_ms)                                           AS p95_latency_ms,
    quantileExact(0.99)(latency_ms)                                           AS p99_latency_ms,
    sum(cached)                                                               AS cache_hit_count,
    now64()                                                                   AS updated_at
FROM query_logs
WHERE request_received_at >= now() - INTERVAL 2 HOUR
GROUP BY
    toStartOfHour(request_received_at),
    toDate(request_received_at),
    selected_model
SETTINGS allow_experimental_analyzer = 1;
