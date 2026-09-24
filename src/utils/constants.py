from enum import Enum


class KafkaTopics(str, Enum):
    QUERIES = "llm-queries"
    RESPONSES = "llm-responses"
    METRICS = "llm-metrics"
    ERRORS = "llm-errors"
    DEAD_LETTER = "llm-dead-letter"


class ClickHouseTables(str, Enum):
    QUERY_LOGS = "query_logs"
    SYSTEM_METRICS = "system_metrics"
    MODEL_PERFORMANCE = "model_performance"
    USER_ANALYTICS = "user_analytics"
