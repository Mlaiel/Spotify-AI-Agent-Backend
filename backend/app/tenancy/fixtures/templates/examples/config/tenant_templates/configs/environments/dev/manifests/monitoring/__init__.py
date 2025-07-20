# -*- coding: utf-8 -*-
"""
Spotify AI Agent - Module de Monitoring Multi-Tenant
=====================================================

Module ultra-avancé de monitoring pour l'architecture multi-tenant de l'agent IA Spotify.
Surveillance complète des performances, métriques, alertes et observabilité.

Architecture:
- Prometheus pour la collecte de métriques
- Grafana pour la visualisation
- AlertManager pour la gestion d'alertes  
- Jaeger pour le tracing distribué
- ELK Stack pour la centralisation des logs
- Health checks automatisés
- SLA monitoring
- Cost tracking
- Security monitoring

Auteur: Développé par l'équipe d'experts dirigée par Fahed Mlaiel
Version: 2.0.0
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Expert Team - Fahed Mlaiel"
__email__ = "fahed.mlaiel@spotifyai.com"

# Modules disponibles
__all__ = [
    'MetricsCollector',
    'AlertManager', 
    'HealthChecker',
    'PerformanceMonitor',
    'SecurityMonitor',
    'CostTracker',
    'SLAMonitor',
    'DashboardManager'
]

# Configuration par défaut
DEFAULT_CONFIG = {
    'monitoring': {
        'enabled': True,
        'retention_days': 30,
        'alert_cooldown': 300,
        'health_check_interval': 30,
        'metrics_scrape_interval': 15,
        'log_level': 'INFO'
    },
    'prometheus': {
        'port': 9090,
        'scrape_interval': '15s',
        'evaluation_interval': '15s',
        'retention': '30d'
    },
    'grafana': {
        'port': 3000,
        'admin_user': 'admin',
        'enable_alerting': True
    },
    'jaeger': {
        'port': 14268,
        'ui_port': 16686,
        'sampling_rate': 0.1
    },
    'elasticsearch': {
        'port': 9200,
        'kibana_port': 5601,
        'retention_days': 7
    }
}

# Importation des classes principales
try:
    from .core.metrics_collector import MetricsCollector
    from .core.alert_manager import AlertManager
    from .core.health_checker import HealthChecker
    from .core.performance_monitor import PerformanceMonitor
    from .core.security_monitor import SecurityMonitor
    from .core.cost_tracker import CostTracker
    from .core.sla_monitor import SLAMonitor
    from .core.dashboard_manager import DashboardManager
except ImportError as e:
    # Logging de base en cas d'erreur d'import
    import logging
    logging.warning(f"Certains modules de monitoring ne sont pas disponibles: {e}")
