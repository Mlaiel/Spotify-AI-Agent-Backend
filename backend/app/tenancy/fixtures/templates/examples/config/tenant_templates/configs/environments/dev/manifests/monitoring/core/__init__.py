# -*- coding: utf-8 -*-
"""
Core Monitoring Modules - Spotify AI Agent
==========================================

Modules centraux pour le système de monitoring ultra-avancé.
Ces modules fournissent toutes les fonctionnalités de base pour la surveillance,
l'alerting et l'observabilité de l'architecture multi-tenant.

Modules disponibles:
- MetricsCollector : Collecte et agrégation des métriques
- AlertManager : Gestion intelligente des alertes
- HealthChecker : Vérifications de santé système
- PerformanceMonitor : Surveillance des performances
- SecurityMonitor : Monitoring de sécurité
- CostTracker : Suivi des coûts et optimisation
- SLAMonitor : Surveillance des SLAs
- DashboardManager : Gestion des dashboards

Version: 2.0.0
Auteur: Expert Team - Fahed Mlaiel
"""

__version__ = "2.0.0"
__author__ = "Expert Team - Fahed Mlaiel"

# Configuration des modules core
CORE_CONFIG = {
    'metrics': {
        'collection_interval': 15,
        'retention_policy': '30d',
        'batch_size': 1000,
        'compression': True
    },
    'alerts': {
        'cooldown_period': 300,
        'max_retries': 3,
        'notification_channels': ['slack', 'email', 'sms'],
        'severity_levels': ['info', 'warning', 'error', 'critical']
    },
    'health': {
        'check_interval': 30,
        'timeout': 10,
        'retries': 3,
        'endpoints': [
            '/health',
            '/ready',
            '/metrics'
        ]
    },
    'performance': {
        'latency_thresholds': {
            'p50': 100,
            'p95': 500,
            'p99': 1000
        },
        'throughput_min': 1000,
        'error_rate_max': 0.01
    },
    'security': {
        'anomaly_detection': True,
        'threat_intelligence': True,
        'compliance_checks': True,
        'audit_logging': True
    },
    'cost': {
        'tracking_enabled': True,
        'optimization_alerts': True,
        'budget_thresholds': {
            'warning': 0.8,
            'critical': 0.95
        }
    },
    'sla': {
        'uptime_target': 99.9,
        'response_time_target': 200,
        'error_rate_target': 0.001,
        'reporting_interval': 3600
    }
}

# Importation des modules
try:
    from .metrics_collector import MetricsCollector
    from .alert_manager import AlertManager
    from .health_checker import HealthChecker
    from .performance_monitor import PerformanceMonitor
    from .security_monitor import SecurityMonitor
    from .cost_tracker import CostTracker
    from .sla_monitor import SLAMonitor
    from .dashboard_manager import DashboardManager
    
    __all__ = [
        'MetricsCollector',
        'AlertManager',
        'HealthChecker', 
        'PerformanceMonitor',
        'SecurityMonitor',
        'CostTracker',
        'SLAMonitor',
        'DashboardManager',
        'CORE_CONFIG'
    ]
    
except ImportError as e:
    import logging
    logging.warning(f"Certains modules core ne sont pas disponibles: {e}")
    __all__ = ['CORE_CONFIG']
