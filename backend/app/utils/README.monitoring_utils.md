# Monitoring Utils - Documentation Enterprise

## Vue d'ensemble

Le module `monitoring_utils.py` fournit l'écosystème d'observabilité complet pour Spotify AI Agent, incluant métriques avancées, alerting intelligent, APM (Application Performance Monitoring), et analytics temps réel. Développé par l'équipe observabilité enterprise sous la direction de **Fahed Mlaiel**.

## Équipe d'Experts Observabilité

- **Lead Developer + Observability Architect** : Architecture monitoring et métriques
- **SRE Engineer Senior** : Reliability, alerting, et incident response
- **Performance Engineer** : APM, profiling, et optimisation
- **Data Engineer** : Analytics, dashboards, et data pipeline
- **Security Engineer** : Security monitoring et audit logging

## Architecture Monitoring Enterprise

### Composants Principaux

#### MetricsCollector
Collecteur de métriques haute performance avec support multi-backend et custom metrics.

**Types de Métriques :**
- **System Metrics** : CPU, mémoire, disque, réseau
- **Application Metrics** : Latence, throughput, erreurs
- **Business Metrics** : KPIs métier, conversion, engagement
- **ML Metrics** : Accuracy, drift, performance modèles
- **Custom Metrics** : Métriques spécifiques domaine

```python
# Collecteur métriques enterprise
metrics_collector = MetricsCollector()

# Configuration multi-backend
collector_config = {
    'backends': {
        'prometheus': {
            'enabled': True,
            'endpoint': 'http://prometheus:9090',
            'push_gateway': 'http://pushgateway:9091',
            'scrape_interval': 15  # secondes
        },
        'influxdb': {
            'enabled': True,
            'url': 'http://influxdb:8086',
            'database': 'spotify_metrics',
            'retention_policy': '7d'
        },
        'datadog': {
            'enabled': True,
            'api_key': '${DATADOG_API_KEY}',
            'app_key': '${DATADOG_APP_KEY}',
            'tags': ['env:production', 'service:spotify-ai']
        }
    },
    'collection_settings': {
        'high_frequency_metrics': 1,    # 1 seconde
        'standard_metrics': 15,         # 15 secondes
        'business_metrics': 60,         # 1 minute
        'batch_size': 1000,
        'compression': 'gzip'
    },
    'custom_metrics': {
        'audio_processing_latency': {
            'type': 'histogram',
            'buckets': [10, 50, 100, 500, 1000, 5000],
            'labels': ['model_type', 'audio_format']
        },
        'recommendation_accuracy': {
            'type': 'gauge',
            'labels': ['user_segment', 'algorithm_version']
        },
        'user_engagement_score': {
            'type': 'counter',
            'labels': ['playlist_type', 'device_type']
        }
    }
}

# Initialisation collecteur
await metrics_collector.configure(collector_config)

# Collection métriques custom
await metrics_collector.record_metric(
    name='audio_processing_latency',
    value=145.7,  # ms
    labels={
        'model_type': 'neural_classifier',
        'audio_format': 'mp3_320kbps'
    },
    timestamp=datetime.utcnow()
)

# Métriques business
await metrics_collector.record_business_metric(
    name='user_engagement_score',
    value=8.7,
    user_id='user_12345',
    context={
        'session_duration': 3600,
        'tracks_played': 47,
        'skips_ratio': 0.12,
        'playlist_completion': 0.89
    }
)
```

#### HealthChecker
Vérificateur de santé complet avec checks custom et dépendances externes.

**Health Checks Enterprise :**
- **Service Health** : Santé services internes
- **Database Health** : État bases de données
- **External Dependencies** : APIs externes, CDN
- **ML Models Health** : Performance modèles ML
- **Infrastructure Health** : Kubernetes, containers

```python
# Health checker enterprise
health_checker = HealthChecker()

# Configuration health checks
health_config = {
    'checks': {
        'database_postgresql': {
            'type': 'database',
            'connection_string': '${POSTGRES_URL}',
            'timeout_seconds': 5,
            'critical': True,
            'interval_seconds': 30
        },
        'redis_cache': {
            'type': 'redis',
            'host': 'redis',
            'port': 6379,
            'timeout_seconds': 2,
            'critical': True,
            'interval_seconds': 15
        },
        'ml_recommendation_model': {
            'type': 'custom',
            'check_function': 'check_ml_model_health',
            'parameters': {
                'model_name': 'recommendation_v3',
                'max_inference_time_ms': 100,
                'min_accuracy': 0.85
            },
            'timeout_seconds': 10,
            'critical': False,
            'interval_seconds': 60
        },
        'spotify_web_api': {
            'type': 'http',
            'url': 'https://api.spotify.com/v1/me',
            'headers': {'Authorization': 'Bearer ${SPOTIFY_TOKEN}'},
            'timeout_seconds': 5,
            'critical': False,
            'interval_seconds': 60
        },
        'kubernetes_cluster': {
            'type': 'kubernetes',
            'namespace': 'spotify-ai-agent',
            'resources': ['deployments', 'services', 'pods'],
            'critical': True,
            'interval_seconds': 30
        }
    },
    'alerting': {
        'critical_failure_threshold': 1,   # 1 check critique fail
        'warning_failure_threshold': 3,    # 3 checks non-critiques fail
        'notification_channels': ['slack', 'pagerduty', 'email']
    }
}

# Exécution health checks
health_status = await health_checker.run_all_checks()

# Résultat health check :
{
    'overall_status': 'healthy',
    'timestamp': '2024-01-15T10:30:00Z',
    'checks': {
        'database_postgresql': {
            'status': 'healthy',
            'response_time_ms': 23,
            'details': {'connections': 45, 'slow_queries': 0}
        },
        'ml_recommendation_model': {
            'status': 'warning', 
            'response_time_ms': 87,
            'details': {
                'inference_time_ms': 87,
                'accuracy': 0.86,
                'memory_usage_mb': 512,
                'warning': 'inference_time_near_limit'
            }
        }
    },
    'summary': {
        'total_checks': 5,
        'healthy': 4,
        'warning': 1,
        'critical': 0,
        'failed': 0
    }
}
```

#### AlertManager
Gestionnaire d'alertes intelligent avec machine learning et escalade automatique.

**Alerting Enterprise :**
- **Smart Thresholds** : Seuils adaptatifs via ML
- **Anomaly Detection** : Détection anomalies automatique
- **Alert Correlation** : Corrélation alertes liées
- **Escalation Rules** : Escalade automatique selon gravité
- **Notification Channels** : Multi-canaux (Slack, email, SMS, PagerDuty)

```python
# Alert manager intelligent
alert_manager = AlertManager()

# Configuration alerting avancée
alerting_config = {
    'smart_thresholds': {
        'enabled': True,
        'ml_model': 'anomaly_detection_v2',
        'adaptation_window_hours': 24,
        'confidence_threshold': 0.95
    },
    'alert_rules': {
        'high_api_latency': {
            'metric': 'api_response_time_p95',
            'threshold_type': 'adaptive',
            'base_threshold': 500,  # ms
            'severity': 'warning',
            'duration': '5m',
            'labels': {'team': 'backend', 'service': 'api'}
        },
        'ml_model_accuracy_drop': {
            'metric': 'ml_model_accuracy',
            'threshold_type': 'static',
            'threshold': 0.80,
            'severity': 'critical',
            'duration': '1m',
            'labels': {'team': 'ml', 'model': 'recommendation'}
        },
        'user_engagement_anomaly': {
            'metric': 'user_engagement_score',
            'threshold_type': 'anomaly_detection',
            'anomaly_sensitivity': 'high',
            'severity': 'warning',
            'labels': {'team': 'product', 'feature': 'recommendations'}
        }
    },
    'notification_channels': {
        'slack': {
            'webhook_url': '${SLACK_WEBHOOK_URL}',
            'channel': '#alerts-production',
            'mention_on_critical': ['@here']
        },
        'pagerduty': {
            'integration_key': '${PAGERDUTY_KEY}',
            'escalation_policy': 'production-escalation'
        },
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'recipients': ['oncall@spotify-ai.com']
        }
    },
    'escalation_rules': {
        'warning': ['slack'],
        'critical': ['slack', 'pagerduty', 'email'],
        'timeout_minutes': 15,
        'max_escalations': 3
    }
}

# Déclenchement alerte
alert = await alert_manager.create_alert(
    name='high_recommendation_latency',
    severity='warning',
    message='Recommendation API latency above threshold',
    metrics={
        'current_latency_ms': 847,
        'threshold_ms': 500,
        'p95_latency_ms': 923
    },
    context={
        'service': 'recommendation-api',
        'endpoint': '/api/v1/recommendations',
        'affected_users': 1247
    }
)

# Corrélation alertes automatique
correlated_alerts = await alert_manager.correlate_alerts(
    time_window_minutes=10,
    correlation_threshold=0.8
)
```

#### APMTracer
Tracer APM pour observabilité distribuée avec OpenTelemetry et profiling avancé.

**APM Features :**
- **Distributed Tracing** : Traçage requêtes cross-services
- **Performance Profiling** : Profiling CPU, mémoire, I/O
- **Span Analytics** : Analytics détaillées sur spans
- **Error Tracking** : Tracking erreurs avec contexte
- **Dependency Mapping** : Cartographie dépendances automatique

```python
# APM Tracer enterprise
apm_tracer = APMTracer()

# Configuration OpenTelemetry
tracer_config = {
    'service_name': 'spotify-ai-agent',
    'service_version': '2.0.0',
    'environment': 'production',
    'exporters': {
        'jaeger': {
            'endpoint': 'http://jaeger:14268/api/traces',
            'batch_export': True,
            'max_batch_size': 512
        },
        'datadog': {
            'api_key': '${DATADOG_API_KEY}',
            'service_mapping': {
                'recommendation-api': 'recommendation',
                'ml-inference': 'ml-models'
            }
        }
    },
    'sampling': {
        'strategy': 'probabilistic',
        'rate': 0.1,  # 10% sampling
        'head_sampling': True
    },
    'instrumentation': {
        'auto_instrument': ['flask', 'requests', 'sqlalchemy'],
        'custom_spans': True,
        'async_context': True
    }
}

# Traçage automatique avec context
@apm_tracer.trace('recommendation.generate')
async def generate_recommendations(user_id: str, context: dict):
    """Génère recommandations avec traçage automatique."""
    
    # Span automatique pour cette fonction
    with apm_tracer.start_span('user.profile.load') as span:
        span.set_attribute('user.id', user_id)
        user_profile = await load_user_profile(user_id)
        span.set_attribute('user.profile.size', len(user_profile))
    
    # Span ML inference
    with apm_tracer.start_span('ml.inference') as span:
        span.set_attribute('model.name', 'recommendation_v3')
        span.set_attribute('model.version', '3.2.1')
        
        start_time = time.time()
        recommendations = await ml_model.predict(user_profile)
        inference_time = time.time() - start_time
        
        span.set_attribute('ml.inference_time_ms', inference_time * 1000)
        span.set_attribute('ml.predictions_count', len(recommendations))
    
    # Métriques custom
    apm_tracer.record_metric('recommendation.generation.duration', 
                           inference_time, 
                           tags={'user_segment': user_profile.segment})
    
    return recommendations

# Analytics spans avancées
span_analytics = await apm_tracer.analyze_spans(
    time_range='1h',
    service='recommendation-api',
    operation='generate_recommendations'
)

# Résultats analytics :
{
    'total_spans': 15420,
    'average_duration_ms': 234.7,
    'p95_duration_ms': 890.2,
    'error_rate': 0.003,
    'throughput_per_minute': 257,
    'slowest_spans': [
        {'trace_id': 'abc123', 'duration_ms': 2340, 'error': 'ml_timeout'},
        {'trace_id': 'def456', 'duration_ms': 1890, 'error': None}
    ],
    'dependency_map': {
        'database': {'calls': 45230, 'avg_duration_ms': 23},
        'redis': {'calls': 18940, 'avg_duration_ms': 3},
        'ml-service': {'calls': 15420, 'avg_duration_ms': 187}
    }
}
```

#### LogAnalyzer
Analyseur de logs intelligent avec ML pour détection anomalies et insights.

**Log Analytics :**
- **Structured Logging** : Logs JSON structurés
- **Log Aggregation** : Agrégation multi-sources
- **Anomaly Detection** : Détection anomalies dans logs
- **Pattern Recognition** : Reconnaissance patterns d'erreurs
- **Correlation Analysis** : Corrélation logs avec métriques

```python
# Log analyzer intelligent
log_analyzer = LogAnalyzer()

# Configuration logging enterprise
logging_config = {
    'structured_logging': {
        'format': 'json',
        'fields': ['timestamp', 'level', 'service', 'trace_id', 'user_id', 'message'],
        'correlation_id': True,
        'request_id': True
    },
    'log_sources': {
        'application_logs': {
            'path': '/var/log/spotify-ai/*.log',
            'format': 'json',
            'parser': 'fluentd'
        },
        'nginx_logs': {
            'path': '/var/log/nginx/access.log',
            'format': 'combined',
            'parser': 'nginx'
        },
        'kubernetes_logs': {
            'namespace': 'spotify-ai-agent',
            'container': 'all',
            'parser': 'kubernetes'
        }
    },
    'analytics': {
        'anomaly_detection': {
            'enabled': True,
            'algorithm': 'isolation_forest',
            'sensitivity': 'medium'
        },
        'pattern_recognition': {
            'enabled': True,
            'min_pattern_frequency': 10,
            'clustering_algorithm': 'dbscan'
        },
        'correlation_analysis': {
            'metrics_correlation': True,
            'trace_correlation': True,
            'time_window_minutes': 5
        }
    }
}

# Analyse logs temps réel
log_insights = await log_analyzer.analyze_logs(
    time_range='1h',
    filters={
        'level': ['ERROR', 'WARN'],
        'service': 'recommendation-api'
    },
    analysis_type='comprehensive'
)

# Insights générés :
{
    'summary': {
        'total_logs': 45230,
        'error_logs': 127,
        'warning_logs': 890,
        'unique_error_patterns': 8
    },
    'anomalies_detected': [
        {
            'type': 'error_spike',
            'message': 'Sudden increase in ML timeout errors',
            'time_window': '14:30-14:45',
            'affected_traces': 23,
            'severity': 'high'
        }
    ],
    'error_patterns': [
        {
            'pattern': 'ML model inference timeout',
            'frequency': 45,
            'first_seen': '14:32:15',
            'services_affected': ['recommendation-api', 'ml-inference'],
            'suggested_action': 'Scale ML inference service'
        }
    ],
    'correlations': [
        {
            'metric': 'api_latency_p95',
            'correlation_score': 0.87,
            'description': 'High correlation between error spikes and API latency'
        }
    ]
}
```

## Dashboards et Visualisations

### Dashboards Grafana Enterprise
```python
GRAFANA_DASHBOARDS = {
    'system_overview': {
        'panels': [
            'cpu_usage_by_service',
            'memory_usage_trends', 
            'network_io_patterns',
            'disk_io_patterns'
        ],
        'refresh_interval': '15s'
    },
    'application_performance': {
        'panels': [
            'api_latency_percentiles',
            'throughput_by_endpoint',
            'error_rate_by_service',
            'database_query_performance'
        ],
        'refresh_interval': '10s'
    },
    'ml_monitoring': {
        'panels': [
            'model_inference_latency',
            'model_accuracy_trends',
            'feature_drift_detection',
            'ml_resource_utilization'
        ],
        'refresh_interval': '30s'
    },
    'business_metrics': {
        'panels': [
            'user_engagement_score',
            'recommendation_click_through_rate',
            'user_retention_cohorts',
            'revenue_impact_metrics'
        ],
        'refresh_interval': '5m'
    }
}
```

### Alerting Rules Prometheus
```yaml
# prometheus_alerts.yml
groups:
  - name: spotify-ai-agent.rules
    rules:
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "High API latency detected"
          description: "95th percentile latency is {{ $value }}s"

      - alert: MLModelAccuracyDrop
        expr: ml_model_accuracy < 0.80
        for: 1m
        labels:
          severity: critical
          team: ml
        annotations:
          summary: "ML model accuracy dropped below threshold"
          description: "Model {{ $labels.model_name }} accuracy is {{ $value }}"
```

## Observabilité Cloud Native

### Configuration Kubernetes
```yaml
# monitoring-stack.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: monitoring-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    scrape_configs:
      - job_name: 'spotify-ai-agent'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            action: keep
            regex: spotify-ai-agent
```

### Service Mesh Integration
```python
# Istio/Envoy métriques
ISTIO_METRICS = {
    'request_total': 'Nombre total requêtes',
    'request_duration_milliseconds': 'Latence requêtes',
    'request_bytes': 'Taille requêtes',
    'response_bytes': 'Taille réponses',
    'tcp_opened_total': 'Connexions TCP ouvertes',
    'tcp_closed_total': 'Connexions TCP fermées'
}

# Configuration Envoy
envoy_config = {
    'stats_sinks': [
        {
            'name': 'envoy.stat_sinks.prometheus',
            'config': {
                'stats_endpoint': '/metrics',
                'stats_tags': ['service', 'version', 'environment']
            }
        }
    ],
    'tracing': {
        'http': {
            'name': 'envoy.tracers.jaeger',
            'config': {
                'collector_cluster': 'jaeger',
                'collector_endpoint': '/api/traces'
            }
        }
    }
}
```

## Configuration Production

### Variables d'Environnement
```bash
# Monitoring Core
MONITORING_UTILS_METRICS_ENABLED=true
MONITORING_UTILS_METRICS_INTERVAL=15
MONITORING_UTILS_HEALTH_CHECKS_ENABLED=true
MONITORING_UTILS_ALERTING_ENABLED=true

# Backends
MONITORING_UTILS_PROMETHEUS_URL=http://prometheus:9090
MONITORING_UTILS_INFLUXDB_URL=http://influxdb:8086
MONITORING_UTILS_JAEGER_URL=http://jaeger:14268

# Alerting
MONITORING_UTILS_SLACK_WEBHOOK=${SLACK_WEBHOOK_URL}
MONITORING_UTILS_PAGERDUTY_KEY=${PAGERDUTY_INTEGRATION_KEY}
MONITORING_UTILS_EMAIL_SMTP=smtp.gmail.com

# APM
MONITORING_UTILS_TRACING_ENABLED=true
MONITORING_UTILS_SAMPLING_RATE=0.1
MONITORING_UTILS_SPAN_ANALYTICS=true
```

## Tests et Validation

### Tests Monitoring
```bash
# Tests métriques
pytest tests/monitoring/test_metrics.py --with-prometheus

# Tests health checks  
pytest tests/monitoring/test_health.py --check-dependencies

# Tests alerting
pytest tests/monitoring/test_alerts.py --simulate-alerts

# Tests APM
pytest tests/monitoring/test_apm.py --with-tracing
```

## Roadmap Observabilité

### Version 2.1 (Q1 2024)
- [ ] **AIOps Integration** : IA pour operations automatisées
- [ ] **Predictive Alerting** : Alertes prédictives ML
- [ ] **Auto-remediation** : Correction automatique incidents
- [ ] **Chaos Engineering** : Monitoring chaos testing

### Version 2.2 (Q2 2024)
- [ ] **Digital Experience Monitoring** : Real User Monitoring
- [ ] **Business Process Monitoring** : KPIs métier temps réel
- [ ] **Security Monitoring** : SIEM intégré
- [ ] **Carbon Footprint Monitoring** : Métriques empreinte carbone

---

**Développé par l'équipe Observabilité Spotify AI Agent Expert**  
**Dirigé par Fahed Mlaiel**  
**Monitoring Utils v2.0.0 - Full Observability Ready**
