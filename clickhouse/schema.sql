-- P1 template, activated in P3

CREATE TABLE IF NOT EXISTS query_logs (
    request_id String,
    user_id String,
    user_tier String,
    query_type String,
    model_name String,
    provider String,
    input_tokens UInt32,
    output_tokens UInt32,
    latency_ms Float64,
    cost_usd Float64,
    status String,
    timestamp DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, user_id);

CREATE TABLE IF NOT EXISTS system_metrics (
    name String,
    value Float64,
    labels String,
    timestamp DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, name);

CREATE TABLE IF NOT EXISTS model_performance (
    model_name String,
    provider String,
    tokens_per_second Float64,
    quality_score Float64,
    error_rate Float64,
    timestamp DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, model_name);

CREATE TABLE IF NOT EXISTS user_analytics (
    user_id String,
    user_tier String,
    total_requests UInt64,
    total_tokens UInt64,
    total_cost_usd Float64,
    date Date
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_metrics
ENGINE = SummingMergeTree()
ORDER BY (hour, model_name)
AS
SELECT
    toStartOfHour(timestamp) AS hour,
    model_name,
    count() AS request_count,
    sum(cost_usd) AS total_cost_usd,
    sum(input_tokens + output_tokens) AS total_tokens
FROM query_logs
GROUP BY hour, model_name;

-- bloom filter indexes for the three highest cardinality lookup columns
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS idx_user_id user_id TYPE bloom_filter GRANULARITY 4;
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS idx_model_name model_name TYPE bloom_filter GRANULARITY 4;
ALTER TABLE query_logs ADD INDEX IF NOT EXISTS idx_query_type query_type TYPE bloom_filter GRANULARITY 4;
