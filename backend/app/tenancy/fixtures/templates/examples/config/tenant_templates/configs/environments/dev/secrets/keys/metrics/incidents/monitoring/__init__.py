# =============================================================================
# Monitoring & Observability - Main Integration Module Enterprise
# =============================================================================
# 
# Module principal d'intégration du système de monitoring et observabilité
# enterprise avec Prometheus, Grafana, alerting intelligent et IA.
#
# Architecture Microservices & Cloud-Native:
# - Multi-tenant monitoring avec isolation complète
# - Auto-scaling metrics et adaptive thresholds
# - Real-time streaming analytics avec Kafka/Redis
# - ML-powered anomaly detection
# - Enterprise security et compliance (SOC2, ISO27001)
#
# Développé par l'équipe d'experts techniques:
# - Lead Developer + AI Architect (Architecture monitoring complète)
# - Backend Senior Developer (Python/FastAPI/Django)
# - Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
# - DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
# - Spécialiste Sécurité Backend (Enterprise Security)
# - Architecte Microservices (Cloud-Native Architecture)
#
# Direction Technique: Fahed Mlaiel
# =============================================================================

import asyncio
import logging
import signal
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import uuid
import json
import yaml
import hashlib
import hmac
import jwt
from pathlib import Path
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

# Imports asyncio avancés
import aiofiles
import aioredis
import asyncpg
from aiohttp import web, ClientSession
import aiokafka

# Imports pour monitoring enterprise
import psutil
import platform
import socket
from prometheus_client import CollectorRegistry, REGISTRY, push_to_gateway
import structlog

# Imports des modules de monitoring
from .prometheus_metrics import (
    EnterpriseMetricsRegistry, 
    EnterpriseAlertManager,
    MetricType,
    AlertSeverity,
    AlertStatus,
    MetricAggregator,
    TimeSeriesAnalyzer
)
from .grafana_dashboards import (
    GrafanaDashboardGenerator,
    GrafanaExporter,
    VisualizationType,
    TimeRange,
    DashboardTemplate,
    PanelConfiguration
)
from .alerting_system import (
    IntelligentAlertManager,
    NotificationType,
    EscalationAction,
    AlertCorrelationEngine,
    AnomalyDetector
)
from .observability_engine import (
    ObservabilityEngine,
    TraceCollector,
    LogAggregator,
    MetricsCorrelator
)

# Configuration logging structuré
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# =============================================================================
# MODÈLES D'INTÉGRATION ENTERPRISE
# =============================================================================

class MonitoringComponentStatus(Enum):
    """Statuts des composants de monitoring"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    RECOVERING = "recovering"

class MonitoringTier(IntEnum):
    """Niveaux de monitoring enterprise"""
    BASIC = 1
    STANDARD = 2
    PREMIUM = 3
    ENTERPRISE = 4

class DataRetentionPolicy(Enum):
    """Politiques de rétention des données"""
    SHORT_TERM = "short_term"  # 7 jours
    MEDIUM_TERM = "medium_term"  # 30 jours
    LONG_TERM = "long_term"  # 90 jours
    COMPLIANCE = "compliance"  # 7 ans
    CUSTOM = "custom"

@dataclass
class MonitoringConfig:
    """Configuration complète du monitoring enterprise"""
    # Configuration Prometheus
    prometheus_config: Dict[str, Any] = field(default_factory=lambda: {
        "host": "localhost",
        "port": 9090,
        "scrape_interval": "15s",
        "evaluation_interval": "15s",
        "external_labels": {},
        "rule_files": [],
        "scrape_configs": [],
        "alerting": {
            "alertmanagers": []
        },
        "storage": {
            "tsdb": {
                "retention": "30d",
                "max_block_duration": "2h"
            }
        }
    })
    
    # Configuration Grafana
    grafana_config: Dict[str, Any] = field(default_factory=lambda: {
        "host": "localhost",
        "port": 3000,
        "admin_user": "admin",
        "admin_password": "",
        "api_key": "",
        "org_id": 1,
        "datasources": [],
        "dashboards": [],
        "plugins": ["grafana-piechart-panel", "grafana-clock-panel"]
    })
    
    # Configuration Alerting
    alerting_config: Dict[str, Any] = field(default_factory=lambda: {
        "smtp": {
            "host": "localhost",
            "port": 587,
            "username": "",
            "password": "",
            "from_address": "alerts@monitoring.local"
        },
        "slack": {
            "webhook_url": "",
            "channel": "#alerts",
            "username": "MonitoringBot"
        },
        "pagerduty": {
            "integration_key": "",
            "severity_mapping": {}
        },
        "webhook": {
            "url": "",
            "headers": {},
            "timeout": 30
        }
    })
    
    # Configuration de sécurité
    security_config: Dict[str, Any] = field(default_factory=lambda: {
        "encryption_key": "",
        "jwt_secret": "",
        "api_keys": {},
        "rbac_enabled": True,
        "audit_logging": True,
        "rate_limiting": {
            "enabled": True,
            "requests_per_minute": 100
        }
    })
    
    # Configuration générale
    tier: MonitoringTier = MonitoringTier.STANDARD
    enable_system_metrics: bool = True
    enable_ml_metrics: bool = True
    enable_business_metrics: bool = True
    enable_security_metrics: bool = True
    enable_compliance_metrics: bool = True
    
    # Intervalles de collecte
    metrics_collection_interval: int = 30
    dashboard_refresh_interval: str = "30s"
    alert_evaluation_interval: int = 30
    health_check_interval: int = 60
    
    # Rétention des données
    retention_policy: DataRetentionPolicy = DataRetentionPolicy.MEDIUM_TERM
    metrics_retention_days: int = 30
    alerts_retention_days: int = 90
    logs_retention_days: int = 30
    traces_retention_days: int = 7
    
    # Configuration des seuils par défaut
    default_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "cpu": {"warning": 70.0, "critical": 90.0},
        "memory": {"warning": 80.0, "critical": 95.0},
        "disk": {"warning": 85.0, "critical": 95.0},
        "network": {"warning": 80.0, "critical": 95.0},
        "response_time": {"warning": 1000.0, "critical": 5000.0},
        "error_rate": {"warning": 5.0, "critical": 10.0}
    })
    
    # Configuration multi-tenant
    tenant_isolation: bool = True
    tenant_resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_metrics": 10000,
        "max_dashboards": 100,
        "max_alerts": 500,
        "max_retention_days": 90
    })
    
    # Configuration ML/AI
    ml_config: Dict[str, Any] = field(default_factory=lambda: {
        "anomaly_detection": {
            "enabled": True,
            "model_type": "isolation_forest",
            "sensitivity": 0.1,
            "training_window": "7d"
        },
        "predictive_scaling": {
            "enabled": True,
            "horizon": "1h",
            "confidence": 0.95
        },
        "pattern_recognition": {
            "enabled": True,
            "min_pattern_length": 10
        }
    })

@dataclass
class MonitoringHealth:
    """État de santé du système de monitoring"""
    overall_status: MonitoringComponentStatus
    components: Dict[str, MonitoringComponentStatus] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)
    metrics_collected: int = 0
    alerts_active: int = 0
    alerts_resolved: int = 0
    dashboards_active: int = 0
    
    # Métriques de performance
    collection_latency_ms: float = 0.0
    query_latency_ms: float = 0.0
    storage_utilization_percent: float = 0.0
    
    # Métriques de fiabilité
    uptime_percent: float = 100.0
    error_rate_percent: float = 0.0
    availability_sla: float = 99.9
    
    # Ressources système
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_bandwidth_mbps: float = 0.0
    
    # État des intégrations
    prometheus_healthy: bool = True
    grafana_healthy: bool = True
    alertmanager_healthy: bool = True
    storage_healthy: bool = True

@dataclass
class TenantMonitoringContext:
    """Contexte de monitoring pour un tenant"""
    tenant_id: str
    tier: MonitoringTier
    config: MonitoringConfig
    limits: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, int] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.limits:
            self.limits = {
                "max_metrics": 1000 if self.tier == MonitoringTier.BASIC else 10000,
                "max_dashboards": 10 if self.tier == MonitoringTier.BASIC else 100,
                "max_alerts": 50 if self.tier == MonitoringTier.BASIC else 500,
                "max_retention_days": 7 if self.tier == MonitoringTier.BASIC else 90
            }
        
        if not self.usage:
            self.usage = {
                "metrics_count": 0,
                "dashboards_count": 0,
                "alerts_count": 0,
                "storage_mb": 0
            }
    dashboards_deployed: int = 0
    errors: List[str] = field(default_factory=list)

# =============================================================================
# ORCHESTRATEUR DE MONITORING ENTERPRISE
# =============================================================================

class EnterpriseMonitoringOrchestrator:
    """
    Orchestrateur principal du système de monitoring enterprise.
    Coordonne Prometheus, Grafana et l'alerting intelligent.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Composants de monitoring
        self.metrics_registry: Optional[EnterpriseMetricsRegistry] = None
        self.dashboard_generator: Optional[GrafanaDashboardGenerator] = None
        self.alert_manager: Optional[IntelligentAlertManager] = None
        self.grafana_exporter: Optional[GrafanaExporter] = None
        
        # État du système
        self.health = MonitoringHealth(overall_status=MonitoringComponentStatus.INACTIVE)
        self.running = False
        
        # Métriques internes
        self.internal_metrics = {
            "component_startup_time": {},
            "metrics_collection_rate": 0,
            "dashboard_generation_time": 0,
            "alert_processing_time": 0
        }
        
        logger.info("EnterpriseMonitoringOrchestrator initialisé")

    async def initialize(self):
        """Initialisation complète du système de monitoring"""
        try:
            self.health.overall_status = MonitoringComponentStatus.INITIALIZING
            logger.info("Démarrage de l'initialisation du monitoring enterprise...")
            
            # 1. Initialisation du registry de métriques
            await self._initialize_metrics_registry()
            
            # 2. Initialisation du générateur de dashboards
            await self._initialize_dashboard_generator()
            
            # 3. Initialisation du gestionnaire d'alertes
            await self._initialize_alert_manager()
            
            # 4. Initialisation de l'exporteur Grafana
            await self._initialize_grafana_exporter()
            
            # 5. Configuration des intégrations
            await self._setup_integrations()
            
            # 6. Démarrage des tâches de monitoring
            await self._start_monitoring_tasks()
            
            # 7. Génération des dashboards par défaut
            await self._generate_default_dashboards()
            
            # 8. Vérification de santé initiale
            await self._perform_health_check()
            
            self.running = True
            self.health.overall_status = MonitoringComponentStatus.ACTIVE
            
            logger.info("Système de monitoring enterprise initialisé avec succès")
            
        except Exception as e:
            self.health.overall_status = MonitoringComponentStatus.ERROR
            self.health.errors.append(f"Erreur initialisation: {e}")
            logger.error(f"Erreur initialisation monitoring: {e}")
            raise

    async def _initialize_metrics_registry(self):
        """Initialisation du registry de métriques Prometheus"""
        start_time = datetime.utcnow()
        
        try:
            logger.info("Initialisation du registry de métriques...")
            
            prometheus_config = self.config.prometheus_config.copy()
            prometheus_config.update({
                'enable_system_metrics': self.config.enable_system_metrics,
                'enable_ml_metrics': self.config.enable_ml_metrics
            })
            
            self.metrics_registry = EnterpriseMetricsRegistry(prometheus_config)
            await self.metrics_registry.initialize()
            
            self.health.components['metrics_registry'] = MonitoringComponentStatus.ACTIVE
            self.internal_metrics["component_startup_time"]["metrics_registry"] = (
                datetime.utcnow() - start_time
            ).total_seconds()
            
            logger.info("Registry de métriques initialisé")
            
        except Exception as e:
            self.health.components['metrics_registry'] = MonitoringComponentStatus.ERROR
            raise

    async def _initialize_dashboard_generator(self):
        """Initialisation du générateur de dashboards Grafana"""
        start_time = datetime.utcnow()
        
        try:
            logger.info("Initialisation du générateur de dashboards...")
            
            grafana_config = self.config.grafana_config.copy()
            grafana_config.setdefault('refresh_interval', self.config.dashboard_refresh_interval)
            
            self.dashboard_generator = GrafanaDashboardGenerator(grafana_config)
            await self.dashboard_generator.initialize()
            
            self.health.components['dashboard_generator'] = MonitoringComponentStatus.ACTIVE
            self.internal_metrics["component_startup_time"]["dashboard_generator"] = (
                datetime.utcnow() - start_time
            ).total_seconds()
            
            logger.info("Générateur de dashboards initialisé")
            
        except Exception as e:
            self.health.components['dashboard_generator'] = MonitoringComponentStatus.ERROR
            raise

    async def _initialize_alert_manager(self):
        """Initialisation du gestionnaire d'alertes"""
        start_time = datetime.utcnow()
        
        try:
            logger.info("Initialisation du gestionnaire d'alertes...")
            
            alerting_config = self.config.alerting_config.copy()
            alerting_config.setdefault('evaluation_interval', self.config.alert_evaluation_interval)
            
            self.alert_manager = IntelligentAlertManager(alerting_config)
            await self.alert_manager.initialize()
            
            self.health.components['alert_manager'] = MonitoringComponentStatus.ACTIVE
            self.internal_metrics["component_startup_time"]["alert_manager"] = (
                datetime.utcnow() - start_time
            ).total_seconds()
            
            logger.info("Gestionnaire d'alertes initialisé")
            
        except Exception as e:
            self.health.components['alert_manager'] = MonitoringComponentStatus.ERROR
            raise

    async def _initialize_grafana_exporter(self):
        """Initialisation de l'exporteur Grafana"""
        start_time = datetime.utcnow()
        
        try:
            logger.info("Initialisation de l'exporteur Grafana...")
            
            self.grafana_exporter = GrafanaExporter(self.config.grafana_config)
            
            self.health.components['grafana_exporter'] = MonitoringComponentStatus.ACTIVE
            self.internal_metrics["component_startup_time"]["grafana_exporter"] = (
                datetime.utcnow() - start_time
            ).total_seconds()
            
            logger.info("Exporteur Grafana initialisé")
            
        except Exception as e:
            self.health.components['grafana_exporter'] = MonitoringComponentStatus.ERROR
            raise

    async def _setup_integrations(self):
        """Configuration des intégrations entre composants"""
        try:
            logger.info("Configuration des intégrations...")
            
            # Intégration métriques -> alertes
            if self.metrics_registry and self.alert_manager:
                # Configuration des seuils par défaut
                await self._configure_default_alert_rules()
            
            # Intégration alertes -> dashboards
            if self.alert_manager and self.dashboard_generator:
                # Création de dashboards d'alerting
                await self._create_alerting_dashboards()
            
            logger.info("Intégrations configurées")
            
        except Exception as e:
            logger.error(f"Erreur configuration intégrations: {e}")
            raise

    async def _configure_default_alert_rules(self):
        """Configuration des règles d'alertes par défaut"""
        
        default_rules = [
            {
                "name": "high_incident_rate",
                "query": "rate(incidents_total[5m]) > 0.1",
                "for_duration": "2m",
                "severity": "critical",
                "description": "Taux d'incidents élevé détecté"
            },
            {
                "name": "api_high_latency",
                "query": "histogram_quantile(0.95, api_request_duration_seconds) > 2",
                "for_duration": "5m",
                "severity": "warning",
                "description": "Latence API élevée"
            },
            {
                "name": "system_cpu_high",
                "query": "system_cpu_usage_percent > 85",
                "for_duration": "5m",
                "severity": "warning",
                "description": "Utilisation CPU élevée"
            },
            {
                "name": "ml_model_accuracy_low",
                "query": "ml_model_accuracy < 0.8",
                "for_duration": "10m",
                "severity": "warning",
                "description": "Précision du modèle ML faible"
            }
        ]
        
        for rule_config in default_rules:
            # Les règles sont déjà créées dans le gestionnaire d'alertes
            logger.info(f"Règle configurée: {rule_config['name']}")

    async def _create_alerting_dashboards(self):
        """Création des dashboards d'alerting"""
        
        # Dashboard spécialisé pour l'alerting est déjà créé
        # dans le générateur de dashboards
        logger.info("Dashboards d'alerting configurés")

    async def _start_monitoring_tasks(self):
        """Démarrage des tâches de monitoring en arrière-plan"""
        
        # Tâche de collecte de métriques
        asyncio.create_task(self._metrics_collection_loop())
        
        # Tâche de surveillance de santé
        asyncio.create_task(self._health_monitoring_loop())
        
        # Tâche de maintenance
        asyncio.create_task(self._maintenance_loop())
        
        logger.info("Tâches de monitoring démarrées")

    async def _generate_default_dashboards(self):
        """Génération des dashboards par défaut"""
        try:
            if self.dashboard_generator:
                # Les dashboards par défaut sont créés lors de l'initialisation
                # du générateur
                
                # Déploiement vers Grafana si configuré
                if self.grafana_exporter:
                    deployed_count = await self.grafana_exporter.deploy_all_dashboards(
                        self.dashboard_generator.output_dir
                    )
                    self.health.dashboards_deployed = deployed_count
                    logger.info(f"Déployé {deployed_count} dashboards")
                
        except Exception as e:
            logger.error(f"Erreur génération dashboards: {e}")

    async def _metrics_collection_loop(self):
        """Boucle de collecte de métriques"""
        while self.running:
            try:
                start_time = datetime.utcnow()
                
                # Collecte de métriques internes
                await self._collect_internal_metrics()
                
                # Mise à jour du taux de collecte
                collection_time = (datetime.utcnow() - start_time).total_seconds()
                self.internal_metrics["metrics_collection_rate"] = 1.0 / collection_time if collection_time > 0 else 0
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Erreur collecte métriques: {e}")
                await asyncio.sleep(5)

    async def _collect_internal_metrics(self):
        """Collecte de métriques internes du système de monitoring"""
        
        if not self.metrics_registry:
            return
        
        # Métriques de santé des composants
        for component, status in self.health.components.items():
            self.metrics_registry.get_metric('system_metrics')
            # Enregistrement de l'état des composants
        
        # Métriques d'alertes actives
        if self.alert_manager:
            summary = await self.alert_manager.get_active_alerts_summary()
            self.health.alerts_active = summary['total_active']
            
            # Enregistrement des métriques d'alertes
            for severity, count in summary['by_severity'].items():
                self.metrics_registry.set_active_incidents("monitoring", severity, count)

    async def _health_monitoring_loop(self):
        """Boucle de surveillance de santé"""
        while self.running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(60)  # Vérification chaque minute
                
            except Exception as e:
                logger.error(f"Erreur surveillance santé: {e}")
                await asyncio.sleep(30)

    async def _perform_health_check(self):
        """Vérification de santé du système"""
        
        self.health.last_check = datetime.utcnow()
        errors = []
        
        # Vérification de chaque composant
        components_healthy = 0
        total_components = len(self.health.components)
        
        for component, status in self.health.components.items():
            if status == MonitoringComponentStatus.ACTIVE:
                components_healthy += 1
            elif status == MonitoringComponentStatus.ERROR:
                errors.append(f"Composant {component} en erreur")
        
        # Détermination du statut global
        if components_healthy == total_components:
            self.health.overall_status = MonitoringComponentStatus.ACTIVE
        elif components_healthy > total_components / 2:
            self.health.overall_status = MonitoringComponentStatus.MAINTENANCE
        else:
            self.health.overall_status = MonitoringComponentStatus.ERROR
        
        self.health.errors = errors
        
        # Génération d'alertes de santé si nécessaire
        if self.health.overall_status == MonitoringComponentStatus.ERROR:
            await self._generate_health_alert()

    async def _generate_health_alert(self):
        """Génération d'alerte de santé du système de monitoring"""
        
        if self.alert_manager:
            health_alert = {
                "rule_name": "monitoring_system_health",
                "severity": "critical",
                "title": "Monitoring System Health Issue",
                "description": f"Monitoring system health degraded: {', '.join(self.health.errors)}",
                "labels": {"component": "monitoring", "type": "health"},
                "tenant_id": "system"
            }
            
            await self.alert_manager.process_alert(health_alert)

    async def _maintenance_loop(self):
        """Boucle de maintenance"""
        while self.running:
            try:
                await self._perform_maintenance()
                await asyncio.sleep(3600)  # Maintenance chaque heure
                
            except Exception as e:
                logger.error(f"Erreur maintenance: {e}")
                await asyncio.sleep(300)

    async def _perform_maintenance(self):
        """Opérations de maintenance"""
        
        # Nettoyage des anciennes métriques
        await self._cleanup_old_metrics()
        
        # Nettoyage des anciennes alertes
        await self._cleanup_old_alerts()
        
        # Optimisation des performances
        await self._optimize_performance()

    async def _cleanup_old_metrics(self):
        """Nettoyage des anciennes métriques"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.metrics_retention_days)
        logger.info(f"Nettoyage métriques antérieures à {cutoff_date}")
        
        # Implémentation du nettoyage

    async def _cleanup_old_alerts(self):
        """Nettoyage des anciennes alertes"""
        
        if self.alert_manager:
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.alerts_retention_days)
            
            # Nettoyage de l'historique des alertes
            old_alerts = [
                alert for alert in self.alert_manager.alert_history
                if alert.resolved_at and alert.resolved_at < cutoff_date
            ]
            
            for alert in old_alerts:
                self.alert_manager.alert_history.remove(alert)
            
            if old_alerts:
                logger.info(f"Nettoyé {len(old_alerts)} anciennes alertes")

    async def _optimize_performance(self):
        """Optimisation des performances"""
        
        # Optimisation des métriques
        if self.metrics_registry:
            # Logique d'optimisation
            pass
        
        # Optimisation des alertes
        if self.alert_manager:
            # Consolidation des alertes similaires
            pass

    # =============================================================================
    # API PUBLIQUE
    # =============================================================================

    async def record_incident(self, tenant_id: str, severity: str, category: str, 
                            source: str = "manual", metadata: Dict[str, Any] = None):
        """Enregistrement d'un incident avec métriques automatiques"""
        
        if self.metrics_registry:
            self.metrics_registry.record_incident(tenant_id, severity, category, source)
            
            # Génération d'alerte si seuil dépassé
            if severity in ["critical", "emergency"] and self.alert_manager:
                alert_data = {
                    "rule_name": "incident_created",
                    "severity": severity,
                    "title": f"New {severity} incident",
                    "description": f"Incident created in category {category}",
                    "labels": {"category": category, "source": source},
                    "tenant_id": tenant_id,
                    "metadata": metadata or {}
                }
                
                await self.alert_manager.process_alert(alert_data)

    async def record_api_metrics(self, method: str, endpoint: str, status_code: int, 
                               tenant_id: str, duration: float):
        """Enregistrement de métriques API"""
        
        if self.metrics_registry:
            self.metrics_registry.record_api_request(
                method, endpoint, status_code, tenant_id, duration
            )

    async def record_ml_metrics(self, model_name: str, model_version: str, 
                              tenant_id: str, result: str, duration: float, 
                              accuracy: Optional[float] = None):
        """Enregistrement de métriques ML"""
        
        if self.metrics_registry:
            self.metrics_registry.record_ml_prediction(
                model_name, model_version, tenant_id, result, duration, accuracy
            )

    async def get_dashboard_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Récupération d'un dashboard par nom"""
        
        if self.dashboard_generator:
            return await self.dashboard_generator.get_dashboard_by_name(name)
        
        return None

    async def create_custom_dashboard(self, name: str, tenant_id: str, 
                                    config: Dict[str, Any]) -> str:
        """Création d'un dashboard personnalisé"""
        
        if self.dashboard_generator:
            dashboard_id = await self.dashboard_generator.create_custom_dashboard(
                name, tenant_id, config
            )
            
            # Déploiement immédiat si Grafana configuré
            if self.grafana_exporter:
                await self.grafana_exporter.deploy_dashboard(dashboard_id)
            
            return dashboard_id
        
        raise ValueError("Dashboard generator non disponible")

    async def get_system_health(self) -> MonitoringHealth:
        """Récupération de l'état de santé du système"""
        return self.health

    async def get_metrics_summary(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Récupération d'un résumé des métriques"""
        
        summary = {
            "collection_rate": self.internal_metrics.get("metrics_collection_rate", 0),
            "total_metrics": 0,
            "active_alerts": self.health.alerts_active,
            "resolved_alerts": self.health.alerts_resolved,
            "system_health": self.health.overall_status.value
        }
        
        if self.metrics_registry:
            summary.update(await self.metrics_registry.get_metrics_summary(tenant_id))
        
        return summary

    async def export_configuration(self) -> Dict[str, Any]:
        """Export de la configuration complète"""
        
        config_export = {
            "monitoring_config": asdict(self.config),
            "health": asdict(self.health),
            "components": {}
        }
        
        # Export des configurations des composants
        if self.metrics_registry:
            config_export["components"]["metrics"] = await self.metrics_registry.export_config()
        
        if self.dashboard_generator:
            config_export["components"]["dashboards"] = await self.dashboard_generator.export_config()
        
        if self.alert_manager:
            config_export["components"]["alerts"] = await self.alert_manager.export_config()
        
        return config_export

    async def import_configuration(self, config_data: Dict[str, Any]):
        """Import d'une configuration"""
        
        # Validation de la configuration
        if "monitoring_config" not in config_data:
            raise ValueError("Configuration monitoring manquante")
        
        # Application de la nouvelle configuration
        self.config = MonitoringConfig(**config_data["monitoring_config"])
        
        # Redémarrage des composants avec la nouvelle configuration
        await self.shutdown()
        await self.initialize()

    async def shutdown(self):
        """Arrêt propre du système de monitoring"""
        
        logger.info("Arrêt du système de monitoring...")
        
        self.running = False
        
        # Arrêt des composants
        if self.alert_manager:
            await self.alert_manager.shutdown()
        
        if self.metrics_registry:
            await self.metrics_registry.shutdown()
        
        if self.dashboard_generator:
            await self.dashboard_generator.shutdown()
        
        self.health.overall_status = MonitoringComponentStatus.INACTIVE
        
        logger.info("Système de monitoring arrêté")

# =============================================================================
# GESTIONNAIRE MULTI-TENANT
# =============================================================================

class MultiTenantMonitoringManager:
    """
    Gestionnaire de monitoring multi-tenant avec isolation complète
    et gestion des ressources par tenant.
    """
    
    def __init__(self, global_config: MonitoringConfig):
        self.global_config = global_config
        self.tenant_orchestrators: Dict[str, EnterpriseMonitoringOrchestrator] = {}
        self.tenant_contexts: Dict[str, TenantMonitoringContext] = {}
        
        # Métriques globales
        self.global_metrics = {
            "total_tenants": 0,
            "active_tenants": 0,
            "resource_utilization": {}
        }
        
        logger.info("MultiTenantMonitoringManager initialisé")

    async def create_tenant_monitoring(self, tenant_id: str, tier: MonitoringTier, 
                                     custom_config: Optional[Dict[str, Any]] = None) -> str:
        """Création d'un environnement de monitoring pour un tenant"""
        
        try:
            # Création du contexte tenant
            tenant_config = self._create_tenant_config(tier, custom_config)
            context = TenantMonitoringContext(
                tenant_id=tenant_id,
                tier=tier,
                config=tenant_config
            )
            
            # Création de l'orchestrateur tenant
            orchestrator = EnterpriseMonitoringOrchestrator(tenant_config)
            await orchestrator.initialize()
            
            # Enregistrement
            self.tenant_orchestrators[tenant_id] = orchestrator
            self.tenant_contexts[tenant_id] = context
            
            self.global_metrics["total_tenants"] += 1
            self.global_metrics["active_tenants"] += 1
            
            logger.info(f"Monitoring créé pour tenant {tenant_id} (tier: {tier.name})")
            
            return tenant_id
            
        except Exception as e:
            logger.error(f"Erreur création monitoring tenant {tenant_id}: {e}")
            raise

    def _create_tenant_config(self, tier: MonitoringTier, 
                            custom_config: Optional[Dict[str, Any]] = None) -> MonitoringConfig:
        """Création d'une configuration tenant basée sur le tier"""
        
        # Configuration de base
        config = MonitoringConfig()
        
        # Adaptation selon le tier
        if tier == MonitoringTier.BASIC:
            config.metrics_retention_days = 7
            config.enable_ml_metrics = False
            config.enable_security_metrics = False
        elif tier == MonitoringTier.STANDARD:
            config.metrics_retention_days = 30
            config.enable_ml_metrics = True
            config.enable_security_metrics = False
        elif tier == MonitoringTier.PREMIUM:
            config.metrics_retention_days = 90
            config.enable_ml_metrics = True
            config.enable_security_metrics = True
        elif tier == MonitoringTier.ENTERPRISE:
            config.metrics_retention_days = 365
            config.enable_ml_metrics = True
            config.enable_security_metrics = True
            config.enable_compliance_metrics = True
        
        # Application de la configuration personnalisée
        if custom_config:
            # Merge avec validation des limites par tier
            pass
        
        return config

    async def get_tenant_orchestrator(self, tenant_id: str) -> Optional[EnterpriseMonitoringOrchestrator]:
        """Récupération de l'orchestrateur d'un tenant"""
        return self.tenant_orchestrators.get(tenant_id)

    async def record_tenant_incident(self, tenant_id: str, severity: str, 
                                   category: str, source: str = "manual", 
                                   metadata: Dict[str, Any] = None):
        """Enregistrement d'incident pour un tenant spécifique"""
        
        orchestrator = await self.get_tenant_orchestrator(tenant_id)
        if orchestrator:
            await orchestrator.record_incident(
                tenant_id, severity, category, source, metadata
            )
            
            # Mise à jour du contexte tenant
            if tenant_id in self.tenant_contexts:
                self.tenant_contexts[tenant_id].last_activity = datetime.utcnow()
                self.tenant_contexts[tenant_id].usage["incidents_count"] = (
                    self.tenant_contexts[tenant_id].usage.get("incidents_count", 0) + 1
                )

    async def get_global_health(self) -> Dict[str, Any]:
        """État de santé global du système multi-tenant"""
        
        tenant_health = {}
        overall_healthy = 0
        
        for tenant_id, orchestrator in self.tenant_orchestrators.items():
            health = await orchestrator.get_system_health()
            tenant_health[tenant_id] = {
                "status": health.overall_status.value,
                "alerts_active": health.alerts_active,
                "last_check": health.last_check.isoformat()
            }
            
            if health.overall_status == MonitoringComponentStatus.ACTIVE:
                overall_healthy += 1
        
        return {
            "total_tenants": self.global_metrics["total_tenants"],
            "active_tenants": self.global_metrics["active_tenants"],
            "healthy_tenants": overall_healthy,
            "tenant_health": tenant_health,
            "global_status": "healthy" if overall_healthy == len(self.tenant_orchestrators) else "degraded"
        }

    async def cleanup_inactive_tenants(self, inactive_threshold_days: int = 30):
        """Nettoyage des tenants inactifs"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=inactive_threshold_days)
        inactive_tenants = []
        
        for tenant_id, context in self.tenant_contexts.items():
            if context.last_activity < cutoff_date:
                inactive_tenants.append(tenant_id)
        
        for tenant_id in inactive_tenants:
            await self.remove_tenant_monitoring(tenant_id)
            logger.info(f"Tenant inactif supprimé: {tenant_id}")

    async def remove_tenant_monitoring(self, tenant_id: str):
        """Suppression du monitoring pour un tenant"""
        
        if tenant_id in self.tenant_orchestrators:
            await self.tenant_orchestrators[tenant_id].shutdown()
            del self.tenant_orchestrators[tenant_id]
        
        if tenant_id in self.tenant_contexts:
            del self.tenant_contexts[tenant_id]
        
        self.global_metrics["active_tenants"] = len(self.tenant_orchestrators)

    async def shutdown_all(self):
        """Arrêt de tous les systèmes de monitoring"""
        
        logger.info("Arrêt de tous les systèmes de monitoring...")
        
        shutdown_tasks = []
        for orchestrator in self.tenant_orchestrators.values():
            shutdown_tasks.append(orchestrator.shutdown())
        
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.tenant_orchestrators.clear()
        self.tenant_contexts.clear()
        
        logger.info("Tous les systèmes de monitoring arrêtés")

# =============================================================================
# FACTORY ET UTILITAIRES
# =============================================================================

class MonitoringFactory:
    """Factory pour création d'instances de monitoring"""
    
    @staticmethod
    def create_single_tenant_monitoring(config: MonitoringConfig) -> EnterpriseMonitoringOrchestrator:
        """Création d'un monitoring single-tenant"""
        return EnterpriseMonitoringOrchestrator(config)
    
    @staticmethod
    def create_multi_tenant_monitoring(config: MonitoringConfig) -> MultiTenantMonitoringManager:
        """Création d'un monitoring multi-tenant"""
        return MultiTenantMonitoringManager(config)
    
    @staticmethod
    def create_default_config(tier: MonitoringTier = MonitoringTier.STANDARD) -> MonitoringConfig:
        """Création d'une configuration par défaut"""
        config = MonitoringConfig()
        config.tier = tier
        return config

# =============================================================================
# POINTS D'ENTRÉE PRINCIPAUX
# =============================================================================

# Instance globale pour usage simple
_global_orchestrator: Optional[EnterpriseMonitoringOrchestrator] = None
_global_multi_tenant_manager: Optional[MultiTenantMonitoringManager] = None

async def initialize_monitoring(config: Optional[MonitoringConfig] = None, 
                              multi_tenant: bool = False) -> Union[EnterpriseMonitoringOrchestrator, MultiTenantMonitoringManager]:
    """
    Point d'entrée principal pour l'initialisation du monitoring.
    
    Args:
        config: Configuration optionnelle (utilise la config par défaut si None)
        multi_tenant: Active le mode multi-tenant
    
    Returns:
        L'orchestrateur ou le gestionnaire multi-tenant initialisé
    """
    global _global_orchestrator, _global_multi_tenant_manager
    
    if config is None:
        config = MonitoringFactory.create_default_config()
    
    if multi_tenant:
        _global_multi_tenant_manager = MonitoringFactory.create_multi_tenant_monitoring(config)
        return _global_multi_tenant_manager
    else:
        _global_orchestrator = MonitoringFactory.create_single_tenant_monitoring(config)
        await _global_orchestrator.initialize()
        return _global_orchestrator

async def get_monitoring() -> Optional[Union[EnterpriseMonitoringOrchestrator, MultiTenantMonitoringManager]]:
    """Récupération de l'instance de monitoring globale"""
    return _global_orchestrator or _global_multi_tenant_manager

async def shutdown_monitoring():
    """Arrêt du monitoring global"""
    global _global_orchestrator, _global_multi_tenant_manager
    
    if _global_orchestrator:
        await _global_orchestrator.shutdown()
        _global_orchestrator = None
    
    if _global_multi_tenant_manager:
        await _global_multi_tenant_manager.shutdown_all()
        _global_multi_tenant_manager = None

# =============================================================================
# SIGNAL HANDLERS POUR ARRÊT PROPRE
# =============================================================================

def _setup_signal_handlers():
    """Configuration des gestionnaires de signaux pour arrêt propre"""
    
    def signal_handler(signum, frame):
        logger.info(f"Signal {signum} reçu, arrêt du monitoring...")
        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(shutdown_monitoring())
        else:
            asyncio.run(shutdown_monitoring())
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# Configuration automatique des signal handlers
_setup_signal_handlers()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes principales
    'EnterpriseMonitoringOrchestrator',
    'MultiTenantMonitoringManager',
    'MonitoringFactory',
    
    # Modèles de données
    'MonitoringConfig',
    'MonitoringHealth',
    'TenantMonitoringContext',
    'MonitoringComponentStatus',
    'MonitoringTier',
    'DataRetentionPolicy',
    
    # Fonctions utilitaires
    'initialize_monitoring',
    'get_monitoring',
    'shutdown_monitoring',
    
    # Imports re-exportés
    'EnterpriseMetricsRegistry',
    'GrafanaDashboardGenerator',
    'IntelligentAlertManager',
    'MetricType',
    'AlertSeverity',
    'AlertStatus',
    'VisualizationType',
    'TimeRange',
    'NotificationType',
    'EscalationAction'
]

# =============================================================================
# CONFIGURATION PAR DÉFAUT POUR DÉVELOPPEMENT
# =============================================================================

if __name__ == "__main__":
    # Script de test et développement
    async def main():
        logger.info("Démarrage du système de monitoring en mode développement...")
        
        config = MonitoringFactory.create_default_config(MonitoringTier.ENTERPRISE)
        orchestrator = await initialize_monitoring(config, multi_tenant=False)
        
        # Test d'enregistrement d'incident
        await orchestrator.record_incident(
            tenant_id="dev-tenant",
            severity="warning",
            category="test",
            source="development",
            metadata={"test": "true"}
        )
        
        # Affichage de la santé du système
        health = await orchestrator.get_system_health()
        logger.info(f"État du système: {health.overall_status}")
        
        # Arrêt après 10 secondes
        await asyncio.sleep(10)
        await shutdown_monitoring()
    
    asyncio.run(main())
        
        if self.metrics_registry:
            self.metrics_registry.record_ml_prediction(
                model_name, model_version, tenant_id, result, duration
            )
            
            if accuracy is not None:
                self.metrics_registry.ml_metrics['ml_model_accuracy'].labels(
                    model_name=model_name,
                    model_version=model_version,
                    dataset="production"
                ).set(accuracy)

    async def create_custom_dashboard(self, tenant_id: str, 
                                    dashboard_config: Dict[str, Any]) -> bool:
        """Création d'un dashboard personnalisé"""
        
        if not self.dashboard_generator:
            return False
        
        try:
            # Génération du dashboard tenant
            await self.dashboard_generator.generate_tenant_dashboards(tenant_id)
            
            # Déploiement si configuré
            if self.grafana_exporter:
                dashboard_file = (
                    self.dashboard_generator.output_dir / 
                    f"tenant_{tenant_id}_overview.json"
                )
                
                if dashboard_file.exists():
                    with open(dashboard_file, 'r') as f:
                        dashboard_json = json.load(f)
                    
                    return await self.grafana_exporter.deploy_dashboard(dashboard_json)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur création dashboard {tenant_id}: {e}")
            return False

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Accusé de réception d'une alerte"""
        
        if self.alert_manager:
            return await self.alert_manager.acknowledge_alert(alert_id, acknowledged_by)
        
        return False

    async def resolve_alert(self, alert_id: str, resolved_by: str = "manual") -> bool:
        """Résolution d'une alerte"""
        
        if self.alert_manager:
            return await self.alert_manager.resolve_alert(alert_id, resolved_by)
        
        return False

    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Récupération d'un résumé complet du monitoring"""
        
        summary = {
            "health": {
                "overall_status": self.health.overall_status.value,
                "components": {k: v.value for k, v in self.health.components.items()},
                "last_check": self.health.last_check.isoformat(),
                "errors": self.health.errors
            },
            "metrics": {
                "collection_rate": self.internal_metrics["metrics_collection_rate"],
                "registry_active": self.metrics_registry is not None
            },
            "dashboards": {
                "deployed": self.health.dashboards_deployed,
                "generator_active": self.dashboard_generator is not None
            },
            "alerts": {
                "active": self.health.alerts_active,
                "manager_active": self.alert_manager is not None
            }
        }
        
        # Ajout du résumé des alertes si disponible
        if self.alert_manager:
            alerts_summary = await self.alert_manager.get_active_alerts_summary()
            summary["alerts"].update(alerts_summary)
        
        return summary

    async def export_prometheus_metrics(self) -> tuple[str, str]:
        """Export des métriques Prometheus"""
        
        if self.metrics_registry:
            return self.metrics_registry.export_metrics()
        
        return "", "text/plain"

    async def shutdown(self):
        """Arrêt propre du système de monitoring"""
        try:
            logger.info("Arrêt du système de monitoring...")
            
            self.running = False
            
            # Arrêt des composants
            if self.alert_manager:
                await self.alert_manager.shutdown()
            
            # Mise à jour de l'état
            self.health.overall_status = MonitoringComponentStatus.INACTIVE
            for component in self.health.components:
                self.health.components[component] = MonitoringComponentStatus.INACTIVE
            
            logger.info("Système de monitoring arrêté")
            
        except Exception as e:
            logger.error(f"Erreur arrêt monitoring: {e}")

# =============================================================================
# UTILITAIRES ET HELPERS
# =============================================================================

def create_default_monitoring_config() -> MonitoringConfig:
    """Création d'une configuration de monitoring par défaut"""
    
    return MonitoringConfig(
        prometheus_config={
            'namespace': 'incidents',
            'enable_system_metrics': True,
            'enable_ml_metrics': True
        },
        grafana_config={
            'templates_dir': './grafana_templates',
            'output_dir': './grafana_dashboards',
            'datasource_name': 'Prometheus',
            'grafana_url': 'http://localhost:3000',
            'api_key': '',
            'org_id': 1
        },
        alerting_config={
            'redis_url': 'redis://localhost:6379/0',
            'database_url': 'postgresql+asyncpg://user:pass@localhost/monitoring',
            'evaluation_interval': 30,
            'deduplication_window_minutes': 5,
            'grouping_interval_seconds': 30
        },
        enable_system_metrics=True,
        enable_ml_metrics=True,
        enable_business_metrics=True,
        metrics_collection_interval=30,
        dashboard_refresh_interval="30s",
        alert_evaluation_interval=30,
        metrics_retention_days=30,
        alerts_retention_days=90,
        default_thresholds={
            "cpu_usage": {"warning": 80.0, "critical": 90.0},
            "memory_usage": {"warning": 85.0, "critical": 95.0},
            "api_latency": {"warning": 1.0, "critical": 2.0},
            "error_rate": {"warning": 0.01, "critical": 0.05}
        }
    )

async def setup_enterprise_monitoring(config: Optional[MonitoringConfig] = None) -> EnterpriseMonitoringOrchestrator:
    """Configuration et démarrage du monitoring enterprise"""
    
    if config is None:
        config = create_default_monitoring_config()
    
    orchestrator = EnterpriseMonitoringOrchestrator(config)
    await orchestrator.initialize()
    
    return orchestrator

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

async def main():
    """Exemple d'utilisation du système de monitoring complet"""
    
    print("=== Démonstration du système de monitoring enterprise ===")
    
    # Configuration
    config = create_default_monitoring_config()
    
    # Initialisation
    monitoring = await setup_enterprise_monitoring(config)
    
    try:
        # Simulation d'activité
        print("\nSimulation d'activité du système...")
        
        # Enregistrement d'incidents
        await monitoring.record_incident("tenant_123", "critical", "database", "monitoring")
        await monitoring.record_incident("tenant_456", "warning", "api", "external")
        
        # Enregistrement de métriques API
        await monitoring.record_api_metrics("GET", "/api/incidents", 200, "tenant_123", 0.5)
        await monitoring.record_api_metrics("POST", "/api/incidents", 201, "tenant_123", 1.2)
        
        # Enregistrement de métriques ML
        await monitoring.record_ml_metrics(
            "anomaly_detector", "v1.0", "tenant_123", "normal", 0.15, 0.92
        )
        
        # Création d'un dashboard personnalisé
        dashboard_created = await monitoring.create_custom_dashboard(
            "tenant_123", {"custom_metrics": True}
        )
        print(f"Dashboard personnalisé créé: {dashboard_created}")
        
        # Attente pour voir les alertes
        await asyncio.sleep(10)
        
        # Récupération du résumé
        summary = await monitoring.get_monitoring_summary()
        print(f"\n=== Résumé du monitoring ===")
        print(f"Statut global: {summary['health']['overall_status']}")
        print(f"Alertes actives: {summary['alerts']['active']}")
        print(f"Dashboards déployés: {summary['dashboards']['deployed']}")
        
        # Export des métriques
        metrics_output, content_type = await monitoring.export_prometheus_metrics()
        print(f"Métriques exportées: {len(metrics_output)} caractères")
        
        print("\n=== Démonstration terminée avec succès ===")
        
    finally:
        await monitoring.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
