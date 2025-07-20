# WebSocket Monitoring & Observability

This folder contains tools and scripts for advanced monitoring of the WebSocket module:

- **metrics.py**: Prometheus metrics (connections, messages, errors, latency, etc.)
- **logging_config.py**: Centralized logging, rotation, enriched format, ELK/Loki compatible
- **alerting.py**: Sentry and Prometheus Alertmanager alerting (webhook, real-time supervision)
- **opentelemetry_tracing.py**: Distributed tracing with OpenTelemetry (Jaeger, Tempo, Tempo Cloud, etc.)
- **docker-compose.monitoring.yml**: Full stack for Prometheus, Grafana, Loki, ELK, Sentry
- **prometheus.yml**: Prometheus configuration to scrape metrics

## Recommendations
- Start the Prometheus metrics server in your Python app (`start_metrics_server()`)
- Initialize logging (`setup_logging()`)
- Use alerting for any critical error (`SentryAlerter`, `PrometheusAlerter`)
- Trace each critical message with OpenTelemetry
- Visualize metrics and logs in Grafana/Kibana
- Configure Sentry/Prometheus alerts for critical errors

## Integration Example
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
    # ... processing ...
```

## Security & Compliance
- Restrict access to dashboards (authentication, VPN)
- Never expose sensitive metrics publicly
- Anonymize user logs, record consents, ensure GDPR compliance

## Extensibility
- Connect alerting to other tools (PagerDuty, Slack, etc.)
- Export traces to Jaeger, Tempo, or any OpenTelemetry backend
- Add custom Grafana dashboards
