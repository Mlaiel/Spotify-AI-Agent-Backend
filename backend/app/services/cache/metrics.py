import prometheus_client
import logging
import time
from app.utils.metrics_manager import get_counter, get_histogram, get_gauge

logger = logging.getLogger("cache_metrics")

class CacheMetrics:
    """
    Exporte les métriques Prometheus pour le cache (hits, misses, sets, latency, partition, erreurs).
    Utilisable pour dashboard Grafana, alerting, etc.
    """
    def __init__(self):
        self.cache_hits = get_counter('cache_hits', 'Nombre de hits cache')
        self.cache_misses = get_counter('cache_misses', 'Nombre de misses cache')
        self.cache_sets = get_counter('cache_sets', 'Nombre de sets cache')
        self.cache_latency = get_histogram('cache_latency_seconds', 'Latence des opérations cache')
        self.cache_errors = get_counter('cache_errors', 'Nombre d\'erreurs cache')
        self.cache_partition = get_gauge('cache_partition', 'Partition courante du cache')
        
    def inc_hit(self):
        self.cache_hits.inc()
    """
    Exporte les métriques Prometheus pour le cache (hits, misses, sets, latency, partition, erreurs).
    Utilisable pour dashboard Grafana, alerting, etc.
    """
    def __init__(self):
        self.cache_hits = prometheus_client.Counter('cache_hits', 'Nombre de hits cache')
        self.cache_misses = prometheus_client.Counter('cache_misses', 'Nombre de misses cache')
        self.cache_sets = prometheus_client.Counter('cache_sets', 'Nombre de sets cache')
        self.cache_latency = prometheus_client.Histogram('cache_latency_seconds', 'Latence des opérations cache')
        self.cache_errors = prometheus_client.Counter('cache_errors', 'Nombre d’erreurs cache')
        self.cache_partition = prometheus_client.Gauge('cache_partition', 'Partition courante du cache')
    def inc_hit(self):
        self.cache_hits.inc()
    def inc_miss(self):
        self.cache_misses.inc()
    def inc_set(self):
        self.cache_sets.inc()
    def inc_get(self):
        pass  # Pour extension future
    def observe_latency(self, seconds: float):
        self.cache_latency.observe(seconds)
    def inc_error(self):
        self.cache_errors.inc()
    def set_partition(self, partition: str):
        # Pour dashboard multi-tenant
        self.cache_partition.set(hash(partition) % 10000)
    def expose_metrics(self, port: int = 8001):
        prometheus_client.start_http_server(port)
        logger.info(f"Prometheus metrics exposées sur le port {port}")
