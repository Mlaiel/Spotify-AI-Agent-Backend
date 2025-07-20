# Monitoring & Observierbarkeit für WebSocket

Dieses Verzeichnis enthält Tools und Skripte für fortschrittliches Monitoring des WebSocket-Moduls:

- **metrics.py**: Prometheus-Metriken (Verbindungen, Nachrichten, Fehler, Latenz, etc.)
- **logging_config.py**: Zentrales Logging, Rotation, erweitertes Format, kompatibel mit ELK/Loki
- **alerting.py**: Sentry- und Prometheus-Alerting (Webhooks, Echtzeitüberwachung)
- **opentelemetry_tracing.py**: Verteiltes Tracing mit OpenTelemetry (Jaeger, Tempo, Tempo Cloud, etc.)
- **docker-compose.monitoring.yml**: Kompletter Stack Prometheus, Grafana, Loki, ELK, Sentry
- **prometheus.yml**: Prometheus-Konfiguration zum Scrapen der Metriken

## Empfehlungen
- Prometheus-Metrikserver im Python-App starten (`start_metrics_server()`)
- Logging initialisieren (`setup_logging()`)
- Alerting für kritische Fehler nutzen (`SentryAlerter`, `PrometheusAlerter`)
- Kritische Nachrichten mit OpenTelemetry tracen
- Metriken und Logs in Grafana/Kibana visualisieren
- Alerts für kritische Fehler in Sentry/Prometheus konfigurieren

## Integrationsbeispiel
```python
from monitoring.metrics import start_metrics_server
from monitoring.logging_config import setup_logging
from monitoring.alerting import SentryAlerter, PrometheusAlerter
from monitoring.opentelemetry_tracing import tracer

setup_logging()
start_metrics_server(port=8001)
sentry = SentryAlerter(sentry_dsn="...")
prom = PrometheusAlerter(alertmanager_url="...")

with tracer.start_as_current_span("websocket_message"):
    # ... Verarbeitung ...
```

## Sicherheit & Compliance
- Zugriff auf Dashboards einschränken (Authentifizierung, VPN)
- Sensible Metriken nicht öffentlich exponieren
- Nutzer-Logs anonymisieren, Einwilligungen protokollieren, DSGVO-Konformität

## Erweiterbarkeit
- Alerting an weitere Tools (PagerDuty, Slack, etc.) anbinden
- Traces an Jaeger, Tempo oder beliebige OpenTelemetry-Backends exportieren
- Eigene Grafana-Dashboards ergänzen
