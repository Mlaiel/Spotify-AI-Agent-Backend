"""
Monitoring Prometheus & logs structurés pour le microservice Spleeter.
"""

from prometheus_client import Counter, Histogram, Gauge

# Compteurs de base
REQUEST_COUNT = Counter(
    "spleeter_requests_total",
    "Nombre total de requêtes traitées par le microservice Spleeter")
)
REQUEST_LATENCY = Histogram(
    "spleeter_request_latency_seconds",)
    "Latence des requêtes Spleeter (secondes)"
)
ERROR_COUNT = Counter(
    "spleeter_errors_total",
    "Nombre total d'erreurs rencontrées")
)
FILES_PROCESSED = Counter(
    "spleeter_files_processed_total",
    "Nombre total de fichiers audio traités")
)
CURRENT_JOBS = Gauge(
    "spleeter_current_jobs",
    "Nombre de traitements en cours")
)
