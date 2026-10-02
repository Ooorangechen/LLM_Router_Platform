# Prometheus metrics
# SystemMetrics / RouterMetrics / InferenceMetrics / PipelineMetrics

try:
    from prometheus_client import Counter, Gauge, Histogram, Info, Enum
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False

    class _NoOpMetric:
        """
        _noop returns self when prometheus_client not avaiable 
        ROUTER_METRICS.routing_decisions.labels(model=m).inc() simply does nothing.
        """
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return self._noop

        def _noop(self, *args, **kwargs):
            return self
        
    Counter = Gauge = Histogram = Info = Enum = _NoOpMetric

class SystemMetrics:
    """
    System metrics used for main.py / Monitoring.
    """
    def __init__(self):
        self.requests_total = Counter("llm_router_requests_total", 
                                      "total requests received", 
                                      ["endpoint", "method", "status"])
        self.request_duration_seconds = Histogram("llm_router_request_duration_seconds", 
                                          "HTTP request duration",
                                          labelnames=["endpoint", "method"],
                                          buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0,float("inf")],
                                          )

        self.active_requests = Gauge("llm_router_active_requests", 
                                    "number of requests currently being processed",
                                    labelnames=["endpoint"])

        self.errors_total = Counter("llm_router_errors_total",
                                    "total number of errors",
                                    labelnames=["component", "error_type"])

        self.cpu_usage = Gauge("llm_router_cpu_usage",
                               "Current cpu use in percentage")
        self.memory_usage = Gauge("llm_router_memory_usage",
                                  "current memory usage in bytes")
        self.memory_usage_percent = Gauge("llm_router_memory_usage_percent",
                                          "current memory usage in percentage")
        self.disk_usage = Gauge("llm_router_disk_usage",
                                "current disk usage",
                                labelnames=["mount_point"])

        self.database_connections = Gauge("llm_router_database_connections",
                                          "number of database connections",
                                          labelnames=["database", "state"])

        self.http_connections = Gauge("llm_router_http_connections",
                                      "number of outbound http connections",
                                      labelnames=["target", "state"])

        self.info = Info("llm_router", "LLM router platform build/version information")

        self.health_status = Enum("llm_router_health_status",
                                  "overall platform health status",
                                  states=["healthy", "degraded", "unhealthy"])
class RouterMetrics:
    """
    Routing Metrics used for routing module. 
    """
    def __init__(self):
        self.routing_decisions = Counter("llm_router_router_routing_decisions_total", 
                                         "number of routing decision made",
                                         labelnames=["selected_model", "query_type", "strategy"])

        self.routing_duration_seconds = Histogram(
            "llm_router_router_routing_duration_seconds",
            "Routing decision duration in seconds",
            buckets=[
                0.0005,
                0.001,
                0.0025,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                float("inf"),
            ],
        )

        self.routing_latency_seconds = Histogram("llm_router_router_routing_latency_seconds",
                                                 "routing latency seconds",
                                                 labelnames=['strategy'])

        self.routing_fallbacks_total = Counter("llm_router_router_routing_fallbacks_total",
                                               "total number of routing fallbacks",
                                               labelnames=['cause'])
        
        self.routing_confidence = Histogram("llm_router_router_routing_confidence",
                                            "routing confidence level", 
                                            labelnames=["model", "query_type"],
                                            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        self.model_availability = Gauge("llm_router_router_model_availability",
                                        "number of avaibale models",
                                        labelnames=["model", "provider"])

        self.query_classifications = Counter("llm_router_router_query_classifications_total",
                                             "total number of queries in each type", 
                                             labelnames=["query_type", "confidence_bucket"])

        self.fallback_usage = Counter("llm_router_router_fallback_usage_total", 
                                      "total router fallback usage",
                                      labelnames=["original_model", "fallback_model", "reason"])


class InferenceMetrics:
    """
    Inference Metrics used for inference.py
    """

    def __init__(self):
        self.requests_total = Counter("llm_router_inference_requests_total",
                                      "total number of inference requests",
                                      labelnames=["model_name", "provider", "status"])
        
        self.request_duration_seconds = Histogram("llm_router_inference_request_duration_seconds", 
                                           "inference requests duration",
                                           labelnames=["model_name","provider"],
                                           buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, float("inf")]
                                           )
        
        self.tokens_input_total =  Counter("llm_router_inference_tokens_input_total",
                                  "total number of input tokens used in inference",
                                  labelnames=["model_name"])

        self.tokens_output_total =  Counter("llm_router_inference_tokens_output_total",
                                  "total number of output tokens used in inference",
                                  labelnames=["model_name"])

        self.tokens_total = Counter("llm_router_inference_tokens_total",
                                  "total number of all tokens used in inference",
                                  labelnames=["model_name"])

        self.cost_usd_total = Counter("llm_router_inference_cost_usd_total",
                                       "total inference cost in usd",
                                       labelnames=["model_name"])

        self.cache_hits = Counter("llm_router_inference_cache_hits_total",
                                  "total number of inference cache hits",
                                  labelnames=["model_name"])
        
        self.cache_misses = Counter("llm_router_inference_cache_misses_total",
                                    "total number of inference cache misses",
                                    labelnames=["model_name"])

        self.compressions_total = Counter("llm_router_inference_compressions_total",
                                          "total number of inference compressions",
                                          labelnames=["method"])

        self.errors_total = Counter("llm_router_inference_errors_total",
                                    "total number of inference errors",
                                    labelnames=["model_name","error_type"])        

        self.batch_sizes = Histogram("llm_router_inference_batch_sizes",
                                     "inference batch sizes",
                                     buckets=[1,2,4,8,16,32,64,128,float("inf")])
        # what's the proper design on batch size buckets? 

class PipelineMetrics:
    """
    Pipeline Metrics 
    """
    def __init__(self):
        self.messages_produced = Counter("llm_router_pipeline_messages_produced_total",
                                         "total number of messages produced in pipeline",
                                         labelnames=["topic"])
        self.messages_consumed = Counter("llm_router_pipeline_messages_consumed_total",
                                         "total number of messages consumed in pipeline",
                                         labelnames=["topic", "group_id"])

        self.producer_errors = Counter("llm_router_pipeline_producer_errors_total",
                                       "total number of producer errors") 
        self.consumer_errors = Counter("llm_router_pipeline_consumer_errors_total",
                                       "total number of consumer errors")

        self.db_writes_total = Counter("llm_router_pipeline_db_writes_total",
                                       "total number of databse writes",
                                       labelnames=["table", "status"])

        self.db_write_duration = Histogram(
            "llm_router_pipeline_db_write_duration_seconds",
            "Database write duration in seconds",
            labelnames=["table"],
            buckets=[
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                float("inf"),
            ],
        )

        self.consumer_lag = Gauge("llm_router_pipeline_consumer_lag",
                                  "Pipeline consumer lag",
                                  labelnames=["topic", "partition"])

        self.kafka_produce_total = Counter(
            "llm_router_pipeline_kafka_produce_total",
            "Kafka produce attempts by topic and status",
            labelnames=["topic", "status"],)

        self.kafka_consume_total = Counter(
            "llm_router_pipeline_kafka_consume_total",
            "Kafka consume attempts by topic and status",
            labelnames=["topic", "status"],)

        self.clickhouse_write_total = Counter(
            "llm_router_pipeline_clickhouse_write_total", 
            "ClickHouse batch write outcomes",
            labelnames=["table", "status"],
        )

        self.clickhouse_write_latency_seconds = Histogram(
            "llm_router_pipeline_clickhouse_write_latency_seconds", 
            "ClickHouse attempt latency",
            labelnames=["table"],
        )

        # keeps the "pipeline_" segment like the rest of this class; P4 §3.3 sample
        # llm_router_dead_letter_total is rewritten to this name in alert_rules.yml
        self.dead_letter_total = Counter(
            "llm_router_pipeline_dead_letter_total",
            "Dead-letter events", 
            labelnames=["source", "reason"],
        )


class ResourceMetrics:
    """Resource collector metrics"""
    # all names use llm_router_resource_<field>, matching the llm_router_<category>_<field> convention;
    # P4 §3.3 samples (llm_router_memory_percent / llm_router_disk_percent) are rewritten in alert_rules.yml

    def __init__(self):
        self.cpu_percent = Gauge(
            "llm_router_resource_cpu_percent",
            "Current cpu percentage"
        )
        self.memory_percent = Gauge(
            "llm_router_resource_memory_percent",
            "current ram percentage"
        )
        self.memory_used_bytes = Gauge(
            "llm_router_resource_memory_used_bytes",
            "current used ram in bytes"
        )
        self.memory_total_bytes = Gauge(
            "llm_router_resource_memory_total_bytes",
            "current total ram in bytes"
        )

        self.disk_percent = Gauge(
            "llm_router_resource_disk_percent",
            "current disk percentage"
        )
        self.disk_used_bytes = Gauge(
            "llm_router_resource_disk_used_bytes",
            "current disk used in bytes"
        )

        self.disk_total_bytes = Gauge(
            "llm_router_resource_disk_total_bytes",
            "current total disk in bytes"
        )

        self.gpu_count = Gauge(
            "llm_router_resource_gpu_count",
            "current gpu count numbers"
        )
        self.gpu_utilization_percent = Gauge(
            "llm_router_resource_gpu_utilization_percent",
            "gpu utilization percentage by gpu id",
            labelnames=["gpu_id"]
        )
        self.gpu_memory_percent = Gauge(
            "llm_router_resource_gpu_memory_percent",
            "gpu memory percentage by gpu id",
            labelnames=["gpu_id"]
        )

        self.net_recv_bytes_per_sec = Gauge(
            "llm_router_resource_net_recv_bytes_per_sec",
            "average net receving in bytes per seconds"
        )
        self.net_send_bytes_per_sec = Gauge(
            "llm_router_resource_net_send_bytes_per_sec",
            "average net sent in bytes per seconds"
        )

        self.process_count = Gauge(
            "llm_router_resource_process_count",
            "current process numbers"
        )
        self.open_fds_count = Gauge(
            "llm_router_resource_open_fds_count",
            "current open fds numbers"
        )

        self.uptime_seconds = Gauge(
            "llm_router_resource_uptime_seconds",
            "resrouce collector uptimes in seconds"
        )


class AlertMetrics:
    def __init__(self):
        self.alerts_total = Counter(
            "llm_router_alert_alerts_total",
            "total alert events",
            labelnames=["rule_name", "severity", "action"]
        )
        self.active_alerts = Gauge(
            "llm_router_alert_active_alerts",
            "current active alerts number",
            labelnames=["severity"]
        )
        self.notifications_total = Counter(
            "llm_router_alert_notifications_total",
            "total notification number",
            labelnames=["channel", "status"]
        )


class HealthMetrics:
    def __init__(self):
        # Info appends "_info" on export -> llm_router_health_service_health_info
        self.service_health_info = Info(
            "llm_router_health_service_health",
            "service health infomation",
            labelnames=["service_name", "status", "message"]
        )
        self.overall_health_status = Gauge(
            "llm_router_health_overall_health_status",
            "current overall health status"
        )



SYSTEM_METRICS = SystemMetrics()
ROUTER_METRICS = RouterMetrics()
INFERENCE_METRICS = InferenceMetrics()
PIPELINE_METRICS = PipelineMetrics()

# P4 added, resource colelctor & alert module
RESOURCE_METRICS = ResourceMetrics()
HEALTH_METRICS = HealthMetrics()
ALERT_METRICS = AlertMetrics()

# create once, then prometheus will register to the global.
# load once at creation, then all the other modules share the same default registry

if __name__ == "__main__":
    print("SystemMetrics:", SYSTEM_METRICS)
    print("RouterMetrics:", ROUTER_METRICS)
    print("InferenceMetrics:", INFERENCE_METRICS)
    print("PipelineMetrics:", PIPELINE_METRICS)
    print("")
    print("All metrics instantiated without error.")
