from enum import Enum


class KafkaTopics(str, Enum):
    QUERIES = "llm-queries"
    RESPONSES = "llm-responses"
    METRICS = "llm-metrics"
    ERRORS = "llm-errors"
    DEAD_LETTER = "llm-dead-letter"
