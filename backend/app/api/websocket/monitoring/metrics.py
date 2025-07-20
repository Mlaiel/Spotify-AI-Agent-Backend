from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time

# Compteurs de connexions WebSocket
ws_connections_total = Counter('ws_connections_total', 'Nombre total de connexions WebSocket')
ws_disconnections_total = Counter('ws_disconnections_total', 'Nombre total de déconnexions WebSocket')
ws_messages_total = Counter('ws_messages_total', 'Nombre total de messages WebSocket')
ws_errors_total = Counter('ws_errors_total', 'Nombre total d\'erreurs WebSocket')
ws_latency_seconds = Histogram('ws_latency_seconds', 'Latence des messages WebSocket (s)')

# Gauge pour le nombre de connexions actives
ws_active_connections = Gauge('ws_active_connections', 'Nombre de connexions WebSocket actives')

def start_metrics_server(port: int = 8001):
    """
    Démarre un serveur HTTP Prometheus pour exposer les métriques.
    """
    start_http_server(port)

# Exemple d'utilisation dans un handler :
# ws_connections_total.inc()
# ws_active_connections.inc()
# ws_active_connections.dec()
# ws_messages_total.inc()
# with ws_latency_seconds.time():
#     ... traitement ...
# ws_errors_total.inc()
