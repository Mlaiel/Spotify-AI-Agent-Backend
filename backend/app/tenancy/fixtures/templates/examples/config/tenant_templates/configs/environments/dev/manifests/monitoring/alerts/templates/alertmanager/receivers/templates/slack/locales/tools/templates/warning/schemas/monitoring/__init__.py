"""
Monitoring Schemas Module - Ultra Advanced Industrial Solution
===============================================================

Ce module fournit une architecture complète de monitoring industrialisé
pour les systèmes distribués avec support multi-tenant.

Architectures supportées:
- Prometheus/Grafana/AlertManager
- ELK Stack (Elasticsearch/Logstash/Kibana)
- Datadog/New Relic
- Custom metrics & alerting

Fonctionnalités avancées:
- Auto-scaling monitoring
- Anomaly detection ML-based
- Multi-dimensional metrics
- Real-time alerting with smart routing
- Security monitoring & compliance
- Performance profiling & optimization
- Cost monitoring & optimization

Version: 2.0.0
Dernière mise à jour: Juillet 2025
"""

from .metric_schemas import *
from .alert_schemas import *
from .dashboard_schemas import *
from .tenant_monitoring import *
from .compliance_monitoring import *
from .ml_monitoring import *
from .security_monitoring import *
from .performance_monitoring import *

__version__ = "2.0.0"
__author__ = "Advanced Monitoring Team"

__all__ = [
    "MetricSchema",
    "AlertConfigSchema",
    "DashboardSchema",
    "TenantMonitoringConfig",
    "ComplianceMetrics",
    "MLModelMonitoring",
    "SecurityEventSchema",
    "PerformanceMetrics",
]
