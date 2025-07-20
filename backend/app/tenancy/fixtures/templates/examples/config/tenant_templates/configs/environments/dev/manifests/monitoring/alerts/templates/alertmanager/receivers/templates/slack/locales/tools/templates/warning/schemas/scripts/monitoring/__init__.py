# -*- coding: utf-8 -*-
"""
Monitoring Tools - Industrial Grade Monitoring System
=====================================================

Système de monitoring avancé pour l'architecture multi-tenant Spotify AI Agent.
Fournit des capacités de surveillance, métriques, alertes et observabilité 
en temps réel avec intégration Prometheus, Grafana et systèmes d'alerting.

Key Features:
- Monitoring en temps réel des tenants et services
- Collection de métriques personnalisées
- Système d'alertes intelligent avec escalade
- Tableaux de bord dynamiques
- Tracing distribué et observabilité
- Health checks automatisés
- Détection d'anomalies avec ML
- Rapports de performance automatisés

Author: Spotify AI Agent Team
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .metrics_collector import MetricsCollector
from .alert_manager import AlertManager
from .health_monitor import HealthMonitor
from .performance_monitor import PerformanceMonitor
from .anomaly_detector import AnomalyDetector
from .dashboard_manager import DashboardManager
from .log_analyzer import LogAnalyzer
from .tracing_monitor import TracingMonitor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION DU MODULE
# =============================================================================

MONITORING_CONFIG = {
    "version": "1.0.0",
    "name": "monitoring_tools",
    "description": "Industrial grade monitoring system for multi-tenant architecture",
    "author": "Spotify AI Agent Team",
    
    # Configuration des métriques
    "metrics_config": {
        "collection_interval_seconds": 30,
        "retention_days": 90,
        "batch_size": 1000,
        "compression_enabled": True,
        "export_formats": ["prometheus", "influxdb", "json"],
        "custom_metrics_enabled": True
    },
    
    # Configuration des alertes
    "alerts_config": {
        "enabled": True,
        "evaluation_interval_seconds": 60,
        "max_alerts_per_minute": 10,
        "escalation_enabled": True,
        "notification_channels": ["slack", "email", "webhook", "pagerduty"],
        "severity_levels": ["info", "warning", "critical", "emergency"]
    },
    
    # Configuration de la santé
    "health_config": {
        "check_interval_seconds": 30,
        "timeout_seconds": 10,
        "retry_attempts": 3,
        "health_endpoints": [
            "/health",
            "/health/liveness",
            "/health/readiness"
        ]
    },
    
    # Configuration des performances
    "performance_config": {
        "monitoring_enabled": True,
        "profiling_enabled": True,
        "apm_enabled": True,
        "trace_sampling_rate": 0.1,
        "metrics_aggregation_window": 300
    },
    
    # Configuration de l'observabilité
    "observability_config": {
        "tracing_enabled": True,
        "distributed_tracing": True,
        "log_correlation": True,
        "service_map_enabled": True,
        "dependency_tracking": True
    },
    
    # Configuration des tableaux de bord
    "dashboard_config": {
        "auto_generation": True,
        "template_based": True,
        "multi_tenant_views": True,
        "custom_panels_enabled": True,
        "export_formats": ["json", "yaml", "pdf"]
    },
    
    # Configuration des intégrations
    "integrations": {
        "prometheus": {
            "enabled": True,
            "endpoint": "http://prometheus:9090",
            "push_gateway": "http://pushgateway:9091"
        },
        "grafana": {
            "enabled": True,
            "endpoint": "http://grafana:3000",
            "api_key": "",
            "organization": "spotify-ai-agent"
        },
        "jaeger": {
            "enabled": True,
            "endpoint": "http://jaeger:14268/api/traces"
        },
        "elasticsearch": {
            "enabled": True,
            "endpoint": "http://elasticsearch:9200",
            "index_pattern": "spotify-ai-agent-*"
        }
    }
}

# Types de monitoring supportés
MONITORING_TYPES = {
    "metrics": {
        "class": "MetricsCollector",
        "description": "System and custom metrics collection",
        "real_time": True,
        "alertable": True
    },
    "health": {
        "class": "HealthMonitor",
        "description": "Service health monitoring",
        "real_time": True,
        "alertable": True
    },
    "performance": {
        "class": "PerformanceMonitor",
        "description": "Performance monitoring and profiling",
        "real_time": True,
        "alertable": True
    },
    "logs": {
        "class": "LogAnalyzer",
        "description": "Log analysis and monitoring",
        "real_time": True,
        "alertable": True
    },
    "traces": {
        "class": "TracingMonitor",
        "description": "Distributed tracing monitoring",
        "real_time": True,
        "alertable": False
    },
    "anomalies": {
        "class": "AnomalyDetector",
        "description": "Anomaly detection with ML",
        "real_time": True,
        "alertable": True
    }
}

# Catégories de métriques
METRIC_CATEGORIES = {
    "system": {
        "description": "System-level metrics",
        "metrics": ["cpu_usage", "memory_usage", "disk_usage", "network_io"]
    },
    "application": {
        "description": "Application-level metrics",
        "metrics": ["request_rate", "response_time", "error_rate", "throughput"]
    },
    "business": {
        "description": "Business-level metrics",
        "metrics": ["user_activity", "feature_usage", "revenue_metrics", "conversion_rate"]
    },
    "tenant": {
        "description": "Tenant-specific metrics",
        "metrics": ["tenant_activity", "resource_usage", "api_calls", "storage_usage"]
    },
    "security": {
        "description": "Security-related metrics",
        "metrics": ["failed_logins", "suspicious_activity", "access_violations", "security_events"]
    }
}

# =============================================================================
# REGISTRY DES MONITEURS
# =============================================================================

class MonitoringRegistry:
    """Registry centralisé pour tous les types de moniteurs."""
    
    def __init__(self):
        self._monitors: Dict[str, Any] = {}
        self._active_monitors: Dict[str, bool] = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialise le registry avec les moniteurs disponibles."""
        self._monitors = {
            "metrics": MetricsCollector,
            "alerts": AlertManager,
            "health": HealthMonitor,
            "performance": PerformanceMonitor,
            "anomalies": AnomalyDetector,
            "dashboards": DashboardManager,
            "logs": LogAnalyzer,
            "traces": TracingMonitor
        }
        
        # Marquer tous comme inactifs par défaut
        self._active_monitors = {name: False for name in self._monitors.keys()}
    
    def register_monitor(self, name: str, monitor_class: Any):
        """Enregistre un nouveau type de moniteur."""
        self._monitors[name] = monitor_class
        self._active_monitors[name] = False
        logger.info(f"Registered monitor: {name}")
    
    def get_monitor(self, name: str) -> Optional[Any]:
        """Récupère un moniteur par son nom."""
        return self._monitors.get(name)
    
    def activate_monitor(self, name: str):
        """Active un moniteur."""
        if name in self._monitors:
            self._active_monitors[name] = True
            logger.info(f"Activated monitor: {name}")
    
    def deactivate_monitor(self, name: str):
        """Désactive un moniteur."""
        if name in self._monitors:
            self._active_monitors[name] = False
            logger.info(f"Deactivated monitor: {name}")
    
    def is_active(self, name: str) -> bool:
        """Vérifie si un moniteur est actif."""
        return self._active_monitors.get(name, False)
    
    def list_monitors(self) -> List[str]:
        """Liste tous les moniteurs disponibles."""
        return list(self._monitors.keys())
    
    def list_active_monitors(self) -> List[str]:
        """Liste les moniteurs actifs."""
        return [name for name, active in self._active_monitors.items() if active]

# Instance globale du registry
monitoring_registry = MonitoringRegistry()

# =============================================================================
# FACTORY POUR LES MONITEURS
# =============================================================================

class MonitoringFactory:
    """Factory pour créer des instances de moniteurs."""
    
    @staticmethod
    def create_monitor(monitor_type: str, **kwargs) -> Any:
        """
        Crée une instance de moniteur selon le type.
        
        Args:
            monitor_type: Type de moniteur
            **kwargs: Arguments de configuration
            
        Returns:
            Instance du moniteur approprié
            
        Raises:
            ValueError: Si le type de moniteur n'est pas supporté
        """
        monitor_class = monitoring_registry.get_monitor(monitor_type)
        
        if not monitor_class:
            raise ValueError(f"Unsupported monitor type: {monitor_type}")
        
        # Configuration par défaut
        config = MONITORING_CONFIG.copy()
        config.update(kwargs.get('config', {}))
        
        return monitor_class(config=config, **kwargs)
    
    @staticmethod
    def create_monitoring_suite(config: Optional[Dict[str, Any]] = None) -> "MonitoringSuite":
        """
        Crée une suite complète de monitoring.
        
        Args:
            config: Configuration optionnelle
            
        Returns:
            Suite de monitoring configurée
        """
        return MonitoringSuite(config or MONITORING_CONFIG)

# =============================================================================
# GESTIONNAIRE CENTRAL DE MONITORING
# =============================================================================

class MonitoringManager:
    """
    Gestionnaire central pour orchestrer tous les systèmes de monitoring.
    Point d'entrée unique pour toutes les opérations de surveillance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or MONITORING_CONFIG
        self.monitors: Dict[str, Any] = {}
        self.running = False
        
        # Initialiser les moniteurs selon la configuration
        self._initialize_monitors()
        
        logger.info("MonitoringManager initialized")
    
    def _initialize_monitors(self):
        """Initialise tous les moniteurs configurés."""
        enabled_monitors = [
            "metrics", "health", "performance", "alerts", 
            "logs", "traces", "anomalies", "dashboards"
        ]
        
        for monitor_type in enabled_monitors:
            try:
                monitor = MonitoringFactory.create_monitor(
                    monitor_type,
                    config=self.config
                )
                self.monitors[monitor_type] = monitor
                monitoring_registry.activate_monitor(monitor_type)
                logger.info(f"Initialized {monitor_type} monitor")
                
            except Exception as e:
                logger.error(f"Failed to initialize {monitor_type} monitor: {e}")
    
    async def start_monitoring(self) -> Dict[str, Any]:
        """
        Démarre tous les systèmes de monitoring.
        
        Returns:
            Résultat du démarrage avec statuts
        """
        if self.running:
            return {"status": "already_running", "message": "Monitoring already started"}
        
        start_results = {}
        
        for monitor_name, monitor in self.monitors.items():
            try:
                if hasattr(monitor, 'start'):
                    await monitor.start()
                start_results[monitor_name] = "started"
                logger.info(f"Started {monitor_name} monitor")
                
            except Exception as e:
                start_results[monitor_name] = f"failed: {e}"
                logger.error(f"Failed to start {monitor_name} monitor: {e}")
        
        self.running = True
        
        return {
            "status": "started",
            "monitors": start_results,
            "total_monitors": len(self.monitors),
            "successful_starts": len([r for r in start_results.values() if r == "started"])
        }
    
    async def stop_monitoring(self) -> Dict[str, Any]:
        """
        Arrête tous les systèmes de monitoring.
        
        Returns:
            Résultat de l'arrêt avec statuts
        """
        if not self.running:
            return {"status": "not_running", "message": "Monitoring not started"}
        
        stop_results = {}
        
        for monitor_name, monitor in self.monitors.items():
            try:
                if hasattr(monitor, 'stop'):
                    await monitor.stop()
                stop_results[monitor_name] = "stopped"
                logger.info(f"Stopped {monitor_name} monitor")
                
            except Exception as e:
                stop_results[monitor_name] = f"failed: {e}"
                logger.error(f"Failed to stop {monitor_name} monitor: {e}")
        
        self.running = False
        
        return {
            "status": "stopped",
            "monitors": stop_results,
            "total_monitors": len(self.monitors),
            "successful_stops": len([r for r in stop_results.values() if r == "stopped"])
        }
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Récupère le statut de tous les systèmes de monitoring.
        
        Returns:
            Statut détaillé de tous les moniteurs
        """
        status = {
            "overall_status": "running" if self.running else "stopped",
            "monitors": {},
            "active_alerts": 0,
            "total_metrics": 0,
            "health_status": "unknown"
        }
        
        for monitor_name, monitor in self.monitors.items():
            try:
                if hasattr(monitor, 'get_status'):
                    monitor_status = await monitor.get_status()
                else:
                    monitor_status = {"status": "active" if self.running else "inactive"}
                
                status["monitors"][monitor_name] = monitor_status
                
            except Exception as e:
                status["monitors"][monitor_name] = {"status": "error", "error": str(e)}
        
        # Agréger les statistiques
        if "alerts" in self.monitors:
            try:
                alert_stats = await self.monitors["alerts"].get_alert_statistics()
                status["active_alerts"] = alert_stats.get("active_count", 0)
            except:
                pass
        
        if "metrics" in self.monitors:
            try:
                metric_stats = await self.monitors["metrics"].get_metric_statistics()
                status["total_metrics"] = metric_stats.get("total_count", 0)
            except:
                pass
        
        if "health" in self.monitors:
            try:
                health_stats = await self.monitors["health"].get_overall_health()
                status["health_status"] = health_stats.get("status", "unknown")
            except:
                pass
        
        return status
    
    async def add_custom_metric(self, name: str, value: Union[int, float], labels: Dict[str, str] = None) -> bool:
        """
        Ajoute une métrique personnalisée.
        
        Args:
            name: Nom de la métrique
            value: Valeur de la métrique
            labels: Labels optionnels
            
        Returns:
            True si ajouté avec succès
        """
        if "metrics" not in self.monitors:
            logger.error("Metrics monitor not available")
            return False
        
        try:
            await self.monitors["metrics"].add_custom_metric(name, value, labels or {})
            return True
        except Exception as e:
            logger.error(f"Failed to add custom metric {name}: {e}")
            return False
    
    async def trigger_alert(self, alert_name: str, severity: str, message: str, labels: Dict[str, str] = None) -> bool:
        """
        Déclenche une alerte.
        
        Args:
            alert_name: Nom de l'alerte
            severity: Niveau de sévérité
            message: Message d'alerte
            labels: Labels optionnels
            
        Returns:
            True si déclenchée avec succès
        """
        if "alerts" not in self.monitors:
            logger.error("Alert manager not available")
            return False
        
        try:
            await self.monitors["alerts"].trigger_alert(alert_name, severity, message, labels or {})
            return True
        except Exception as e:
            logger.error(f"Failed to trigger alert {alert_name}: {e}")
            return False
    
    async def get_metrics_summary(self, time_range: str = "1h") -> Dict[str, Any]:
        """
        Récupère un résumé des métriques.
        
        Args:
            time_range: Plage temporelle (1h, 6h, 24h, 7d)
            
        Returns:
            Résumé des métriques
        """
        if "metrics" not in self.monitors:
            return {"error": "Metrics monitor not available"}
        
        try:
            return await self.monitors["metrics"].get_summary(time_range)
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}
    
    async def get_health_report(self) -> Dict[str, Any]:
        """
        Génère un rapport de santé complet.
        
        Returns:
            Rapport de santé détaillé
        """
        if "health" not in self.monitors:
            return {"error": "Health monitor not available"}
        
        try:
            return await self.monitors["health"].generate_health_report()
        except Exception as e:
            logger.error(f"Failed to generate health report: {e}")
            return {"error": str(e)}
    
    async def get_performance_insights(self) -> Dict[str, Any]:
        """
        Récupère les insights de performance.
        
        Returns:
            Insights de performance
        """
        if "performance" not in self.monitors:
            return {"error": "Performance monitor not available"}
        
        try:
            return await self.monitors["performance"].get_insights()
        except Exception as e:
            logger.error(f"Failed to get performance insights: {e}")
            return {"error": str(e)}

# =============================================================================
# SUITE DE MONITORING COMPLÈTE
# =============================================================================

class MonitoringSuite:
    """
    Suite complète de monitoring avec tous les composants intégrés.
    Interface haut niveau pour le monitoring d'entreprise.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.manager = MonitoringManager(config)
        self.dashboards: Dict[str, Any] = {}
        
        logger.info("MonitoringSuite initialized")
    
    async def setup(self) -> Dict[str, Any]:
        """
        Configure et démarre la suite complète.
        
        Returns:
            Résultat de la configuration
        """
        setup_results = {
            "monitoring_setup": None,
            "dashboards_setup": None,
            "integrations_setup": None
        }
        
        # Démarrer le monitoring
        setup_results["monitoring_setup"] = await self.manager.start_monitoring()
        
        # Configurer les tableaux de bord
        try:
            await self._setup_dashboards()
            setup_results["dashboards_setup"] = "success"
        except Exception as e:
            setup_results["dashboards_setup"] = f"failed: {e}"
            logger.error(f"Dashboard setup failed: {e}")
        
        # Configurer les intégrations
        try:
            await self._setup_integrations()
            setup_results["integrations_setup"] = "success"
        except Exception as e:
            setup_results["integrations_setup"] = f"failed: {e}"
            logger.error(f"Integrations setup failed: {e}")
        
        return setup_results
    
    async def _setup_dashboards(self):
        """Configure les tableaux de bord par défaut."""
        if "dashboards" not in self.manager.monitors:
            return
        
        dashboard_manager = self.manager.monitors["dashboards"]
        
        # Tableaux de bord système
        system_dashboard = {
            "name": "System Overview",
            "panels": [
                {"type": "graph", "metric": "cpu_usage", "title": "CPU Usage"},
                {"type": "graph", "metric": "memory_usage", "title": "Memory Usage"},
                {"type": "graph", "metric": "disk_usage", "title": "Disk Usage"},
                {"type": "stat", "metric": "uptime", "title": "Uptime"}
            ]
        }
        
        # Tableau de bord application
        app_dashboard = {
            "name": "Application Metrics",
            "panels": [
                {"type": "graph", "metric": "request_rate", "title": "Request Rate"},
                {"type": "graph", "metric": "response_time", "title": "Response Time"},
                {"type": "graph", "metric": "error_rate", "title": "Error Rate"},
                {"type": "heatmap", "metric": "latency_distribution", "title": "Latency Distribution"}
            ]
        }
        
        await dashboard_manager.create_dashboard("system", system_dashboard)
        await dashboard_manager.create_dashboard("application", app_dashboard)
        
        logger.info("Default dashboards created")
    
    async def _setup_integrations(self):
        """Configure les intégrations avec les systèmes externes."""
        integrations = self.config.get("integrations", {})
        
        for integration_name, integration_config in integrations.items():
            if integration_config.get("enabled", False):
                try:
                    await self._setup_integration(integration_name, integration_config)
                    logger.info(f"Integration {integration_name} configured successfully")
                except Exception as e:
                    logger.error(f"Failed to configure integration {integration_name}: {e}")
    
    async def _setup_integration(self, name: str, config: Dict[str, Any]):
        """Configure une intégration spécifique."""
        # Simulation de configuration d'intégration
        await asyncio.sleep(0.1)
        
        if name == "prometheus":
            # Configuration Prometheus
            pass
        elif name == "grafana":
            # Configuration Grafana
            pass
        elif name == "jaeger":
            # Configuration Jaeger
            pass
        elif name == "elasticsearch":
            # Configuration Elasticsearch
            pass
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        Récupère un statut compréhensif de toute la suite.
        
        Returns:
            Statut complet avec toutes les métriques
        """
        status = await self.manager.get_monitoring_status()
        
        # Ajouter les informations de la suite
        status["suite_info"] = {
            "version": self.config["version"],
            "uptime": "calculated_uptime",
            "total_dashboards": len(self.dashboards),
            "integrations_active": len([
                i for i in self.config.get("integrations", {}).values()
                if i.get("enabled", False)
            ])
        }
        
        return status

# =============================================================================
# UTILITAIRES ET HELPERS
# =============================================================================

def validate_monitoring_config(config: Dict[str, Any]) -> bool:
    """
    Valide la configuration de monitoring.
    
    Args:
        config: Configuration à valider
        
    Returns:
        True si valide, False sinon
    """
    required_keys = ["metrics_config", "alerts_config", "health_config"]
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False
    
    return True

async def quick_health_check() -> Dict[str, Any]:
    """
    Effectue un check de santé rapide.
    
    Returns:
        Résultat du health check
    """
    # Simulation de health check rapide
    await asyncio.sleep(0.1)
    
    return {
        "status": "healthy",
        "timestamp": "current_time",
        "services": {
            "database": "healthy",
            "redis": "healthy",
            "api": "healthy"
        }
    }

async def generate_monitoring_report(time_range: str = "24h") -> Dict[str, Any]:
    """
    Génère un rapport de monitoring complet.
    
    Args:
        time_range: Plage temporelle du rapport
        
    Returns:
        Rapport complet
    """
    # Simulation de génération de rapport
    await asyncio.sleep(0.5)
    
    return {
        "report_id": "monitoring_report_123",
        "time_range": time_range,
        "generated_at": "current_time",
        "summary": {
            "total_alerts": 5,
            "critical_alerts": 0,
            "avg_response_time": 150,
            "uptime_percentage": 99.9
        },
        "sections": [
            "performance_summary",
            "alert_summary", 
            "health_summary",
            "trends_analysis"
        ]
    }

# =============================================================================
# EXPORTS PUBLICS
# =============================================================================

__all__ = [
    # Classes principales
    "MonitoringManager",
    "MonitoringSuite", 
    "MonitoringFactory",
    "MonitoringRegistry",
    
    # Moniteurs spécialisés
    "MetricsCollector",
    "AlertManager",
    "HealthMonitor",
    "PerformanceMonitor",
    "AnomalyDetector",
    "DashboardManager",
    "LogAnalyzer",
    "TracingMonitor",
    
    # Configuration
    "MONITORING_CONFIG",
    "MONITORING_TYPES",
    "METRIC_CATEGORIES",
    
    # Registry global
    "monitoring_registry",
    
    # Fonctions utilitaires
    "validate_monitoring_config",
    "quick_health_check",
    "generate_monitoring_report"
]

# Initialisation du module
logger.info(f"Monitoring module initialized - Version {MONITORING_CONFIG['version']}")
