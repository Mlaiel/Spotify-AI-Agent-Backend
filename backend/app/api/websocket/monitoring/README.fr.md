# Monitoring & Observabilité WebSocket

Ce dossier contient les outils et scripts pour le monitoring avancé du module WebSocket :

- **metrics.py** : exposition de métriques Prometheus (connexions, messages, erreurs, latence, etc.)
- **logging_config.py** : configuration de logging centralisé, rotation, format enrichi, compatible ELK/Loki
- **alerting.py** : alerting Sentry et Prometheus Alertmanager (webhook, supervision temps réel)
- **opentelemetry_tracing.py** : traçabilité distribuée OpenTelemetry (Jaeger, Tempo, Tempo Cloud, etc.)
- **docker-compose.monitoring.yml** : stack complète Prometheus, Grafana, Loki, ELK, Sentry
- **prometheus.yml** : configuration Prometheus pour scrapper les métriques

## Recommandations
- Démarrer le serveur de métriques Prometheus dans l’app Python (`start_metrics_server()`)
- Initialiser le logging (`setup_logging()`)
- Utiliser l’alerting pour toute erreur critique (`SentryAlerter`, `PrometheusAlerter`)
- Tracer chaque message critique avec OpenTelemetry
- Visualiser les métriques et logs dans Grafana/Kibana
- Configurer des alertes Sentry/Prometheus pour les erreurs critiques

## Exemple d’intégration
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
    # ... traitement ...
```

## Sécurité & conformité
- Restreindre l’accès aux dashboards (authentification, VPN)
- Ne pas exposer les métriques sensibles publiquement
- Anonymiser les logs utilisateurs, journaliser les consentements, conformité RGPD

## Extensibilité
- Brancher l’alerting sur d’autres outils (PagerDuty, Slack, etc.)
- Exporter les traces vers Jaeger, Tempo, ou tout backend OpenTelemetry
- Ajouter des dashboards Grafana personnalisés
