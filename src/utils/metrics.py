# Prometheus 指标类
# SystemMetrics / RouterMetrics / InferenceMetrics / PipelineMetrics四件套
# P1 只做到实例化，定义类型，先不进行业务埋点

from prometheus_client import Counter, Gauge, Histogram, Info, Enum

class SystemMetrics:
    """
    System metrics used for main.py / Monitoring.
    """
    def __init__(self):
        self.requests_total = Counter("llm_router_requests_total", 
                                      "total requests received", 
                                      ["endpoint", "method", "status"])
        self.request_duration = Histogram("llm_router_request_duration_seconds", 
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
                                         labelnames=["model", "query_type"])

        self.routing_latency = Histogram("llm_router_router_routing_latency_ms",
                                         "routing decision latency in ms", 
                                         buckets=[1,5,10,25,50,100,250,500, 1000, float("inf")])

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
                                      labelnames=["model", "provider"])
        self.request_duration = Histogram("llm_router_inference_request_duration_seconds", 
                                           "inference requests duration",
                                           labelnames=["model","provider"],
                                           buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, float("inf")]
                                           )

        self.tokens_total = Counter("llm_router_inference_tokens_total",
                                  "total number of tokens used in inference",
                                  labelnames=["model","direction"])

        self.cost_usd_total = Counter("llm_router_inference_cost_usd_total",
                                       "total inference cost in usd",
                                       labelnames=["model"])

        self.cache_hits = Counter("llm_router_inference_cache_hits_total",
                                  "total number of inference cache hits")
        self.cache_misses = Counter("llm_router_inference_cache_misses_total",
                                    "total number of inference cache misses")

        self.compressions_total = Counter("llm_router_inference_compressions_total",
                                          "total number of inference compressions",
                                          labelnames=["method"])

        self.errors_total = Counter("llm_router_inference_errors_total",
                                    "total number of inference errors",
                                    labelnames=["model","error_type"])        

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

        self.db_write_latency = Histogram("llm_router_pipeline_db_write_latency_ms",
                                        "latency of database writes in ms",
                                        labelnames=["table"],
                                        buckets=[1,5,10,25,50,100,250,500, 1000, float("inf")])

        self.consumer_lag = Gauge("llm_router_pipeline_consumer_lag",
                                  "Pipeline consumer lag",
                                  labelnames=["topic", "partition"])


SYSTEM_METRICS = SystemMetrics()
ROUTER_METRICS = RouterMetrics()
INFERENCE_METRICS = InferenceMetrics()
PIPELINE_METRICS = PipelineMetrics()
# create once, then prometheus will register to the global. 
# load once at creation, then all the other modules share the same,
# use .lables()...inc() ot update isntead of creating new 

if __name__ == "__main__":
    print("SystemMetrics:", SYSTEM_METRICS)
    print("RouterMetrics:", ROUTER_METRICS)
    print("InferenceMetrics:", INFERENCE_METRICS)
    print("PipelineMetrics:", PIPELINE_METRICS)
    print("All metrics instantiated without error.")
