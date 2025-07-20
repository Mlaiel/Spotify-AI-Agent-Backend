"""
Module de métriques avancées pour Alertmanager Receivers

Ce module collecte, analyse et expose des métriques détaillées
sur les performances, la disponibilité et l'utilisation des receivers.

Author: Spotify AI Agent Team
Maintainer: Fahed Mlaiel - DBA & Data Engineer
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import statistics

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types de métriques disponibles"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertStatus(Enum):
    """Statuts des alertes"""
    SENT = "sent"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"

@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    unit: str = ""
    buckets: Optional[List[float]] = None

@dataclass
class AlertMetrics:
    """Métriques spécifiques à une alerte"""
    alert_id: str
    tenant: str
    service: str
    severity: str
    timestamp: datetime
    processing_time: float
    delivery_attempts: int
    successful_deliveries: int
    failed_deliveries: int
    integration_results: Dict[str, bool] = field(default_factory=dict)

class PrometheusMetrics:
    """Gestionnaire des métriques Prometheus"""
    
    def __init__(self):
        # Métriques principales
        self.alerts_total = Counter(
            'alertmanager_alerts_total',
            'Total number of alerts processed',
            ['tenant', 'service', 'severity', 'status']
        )
        
        self.alert_processing_duration = Histogram(
            'alertmanager_alert_processing_duration_seconds',
            'Time spent processing alerts',
            ['tenant', 'service'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, float('inf')]
        )
        
        self.integration_requests_total = Counter(
            'alertmanager_integration_requests_total',
            'Total integration requests',
            ['integration', 'tenant', 'status']
        )
        
        self.integration_response_time = Histogram(
            'alertmanager_integration_response_time_seconds',
            'Integration response time',
            ['integration', 'tenant'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
        )
        
        self.active_alerts = Gauge(
            'alertmanager_active_alerts',
            'Number of active alerts',
            ['tenant', 'severity']
        )
        
        self.receiver_health = Gauge(
            'alertmanager_receiver_health',
            'Health status of receivers (1=healthy, 0=unhealthy)',
            ['receiver', 'tenant']
        )
        
        self.config_reloads_total = Counter(
            'alertmanager_config_reloads_total',
            'Total configuration reloads',
            ['status']
        )
        
        self.escalation_events_total = Counter(
            'alertmanager_escalation_events_total',
            'Total escalation events',
            ['tenant', 'policy', 'level']
        )
        
        # Métriques de performance
        self.memory_usage = Gauge(
            'alertmanager_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'alertmanager_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # Métriques de sécurité
        self.authentication_attempts = Counter(
            'alertmanager_auth_attempts_total',
            'Authentication attempts',
            ['tenant', 'method', 'status']
        )
        
        self.rate_limit_hits = Counter(
            'alertmanager_rate_limit_hits_total',
            'Rate limit hits',
            ['tenant', 'endpoint']
        )

class MetricsCollector:
    """Collecteur de métriques avancé"""
    
    def __init__(self):
        self.prometheus = PrometheusMetrics()
        self.alert_history: deque = deque(maxlen=10000)
        self.tenant_metrics: Dict[str, Dict] = defaultdict(dict)
        self.service_metrics: Dict[str, Dict] = defaultdict(dict)
        self.integration_metrics: Dict[str, Dict] = defaultdict(dict)
        self.performance_buffer: deque = deque(maxlen=1000)
        
    def record_alert_processed(self, alert_metrics: AlertMetrics):
        """Enregistre le traitement d'une alerte"""
        # Métriques Prometheus
        self.prometheus.alerts_total.labels(
            tenant=alert_metrics.tenant,
            service=alert_metrics.service,
            severity=alert_metrics.severity,
            status=AlertStatus.SENT.value if alert_metrics.successful_deliveries > 0 else AlertStatus.FAILED.value
        ).inc()
        
        self.prometheus.alert_processing_duration.labels(
            tenant=alert_metrics.tenant,
            service=alert_metrics.service
        ).observe(alert_metrics.processing_time)
        
        # Historique local
        self.alert_history.append({
            "timestamp": alert_metrics.timestamp.isoformat(),
            "alert_id": alert_metrics.alert_id,
            "tenant": alert_metrics.tenant,
            "service": alert_metrics.service,
            "severity": alert_metrics.severity,
            "processing_time": alert_metrics.processing_time,
            "delivery_attempts": alert_metrics.delivery_attempts,
            "successful_deliveries": alert_metrics.successful_deliveries,
            "integration_results": alert_metrics.integration_results
        })
        
        # Mise à jour des métriques par tenant
        self._update_tenant_metrics(alert_metrics)
        
        # Mise à jour des métriques par service
        self._update_service_metrics(alert_metrics)
    
    def _update_tenant_metrics(self, alert_metrics: AlertMetrics):
        """Met à jour les métriques par tenant"""
        tenant = alert_metrics.tenant
        
        if tenant not in self.tenant_metrics:
            self.tenant_metrics[tenant] = {
                "total_alerts": 0,
                "successful_alerts": 0,
                "failed_alerts": 0,
                "average_processing_time": 0.0,
                "by_severity": defaultdict(int),
                "by_service": defaultdict(int),
                "last_alert": None
            }
        
        metrics = self.tenant_metrics[tenant]
        metrics["total_alerts"] += 1
        metrics["last_alert"] = alert_metrics.timestamp.isoformat()
        
        if alert_metrics.successful_deliveries > 0:
            metrics["successful_alerts"] += 1
        else:
            metrics["failed_alerts"] += 1
        
        # Calcul de la moyenne mobile du temps de traitement
        old_avg = metrics["average_processing_time"]
        total = metrics["total_alerts"]
        metrics["average_processing_time"] = (old_avg * (total - 1) + alert_metrics.processing_time) / total
        
        # Répartition par sévérité
        metrics["by_severity"][alert_metrics.severity] += 1
        
        # Répartition par service
        metrics["by_service"][alert_metrics.service] += 1
    
    def _update_service_metrics(self, alert_metrics: AlertMetrics):
        """Met à jour les métriques par service"""
        service = alert_metrics.service
        
        if service not in self.service_metrics:
            self.service_metrics[service] = {
                "total_alerts": 0,
                "successful_alerts": 0,
                "failed_alerts": 0,
                "by_tenant": defaultdict(int),
                "by_severity": defaultdict(int),
                "last_alert": None,
                "average_processing_time": 0.0
            }
        
        metrics = self.service_metrics[service]
        metrics["total_alerts"] += 1
        metrics["last_alert"] = alert_metrics.timestamp.isoformat()
        
        if alert_metrics.successful_deliveries > 0:
            metrics["successful_alerts"] += 1
        else:
            metrics["failed_alerts"] += 1
        
        # Calcul de la moyenne mobile
        old_avg = metrics["average_processing_time"]
        total = metrics["total_alerts"]
        metrics["average_processing_time"] = (old_avg * (total - 1) + alert_metrics.processing_time) / total
        
        metrics["by_tenant"][alert_metrics.tenant] += 1
        metrics["by_severity"][alert_metrics.severity] += 1
    
    def record_integration_request(
        self, 
        integration: str, 
        tenant: str, 
        success: bool, 
        response_time: float
    ):
        """Enregistre une requête d'intégration"""
        status = "success" if success else "error"
        
        # Métriques Prometheus
        self.prometheus.integration_requests_total.labels(
            integration=integration,
            tenant=tenant,
            status=status
        ).inc()
        
        self.prometheus.integration_response_time.labels(
            integration=integration,
            tenant=tenant
        ).observe(response_time)
        
        # Métriques locales
        if integration not in self.integration_metrics:
            self.integration_metrics[integration] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time": 0.0,
                "by_tenant": defaultdict(lambda: {"success": 0, "error": 0}),
                "last_request": None
            }
        
        metrics = self.integration_metrics[integration]
        metrics["total_requests"] += 1
        metrics["last_request"] = datetime.utcnow().isoformat()
        
        if success:
            metrics["successful_requests"] += 1
            metrics["by_tenant"][tenant]["success"] += 1
        else:
            metrics["failed_requests"] += 1
            metrics["by_tenant"][tenant]["error"] += 1
        
        # Calcul de la moyenne mobile du temps de réponse
        old_avg = metrics["average_response_time"]
        total = metrics["total_requests"]
        metrics["average_response_time"] = (old_avg * (total - 1) + response_time) / total
    
    def record_escalation_event(self, tenant: str, policy: str, level: str):
        """Enregistre un événement d'escalade"""
        self.prometheus.escalation_events_total.labels(
            tenant=tenant,
            policy=policy,
            level=level
        ).inc()
    
    def record_authentication_attempt(
        self, 
        tenant: str, 
        method: str, 
        success: bool
    ):
        """Enregistre une tentative d'authentification"""
        status = "success" if success else "failure"
        
        self.prometheus.authentication_attempts.labels(
            tenant=tenant,
            method=method,
            status=status
        ).inc()
    
    def record_rate_limit_hit(self, tenant: str, endpoint: str):
        """Enregistre un hit de rate limit"""
        self.prometheus.rate_limit_hits.labels(
            tenant=tenant,
            endpoint=endpoint
        ).inc()
    
    def update_active_alerts_count(self, tenant: str, severity: str, count: int):
        """Met à jour le nombre d'alertes actives"""
        self.prometheus.active_alerts.labels(
            tenant=tenant,
            severity=severity
        ).set(count)
    
    def update_receiver_health(self, receiver: str, tenant: str, healthy: bool):
        """Met à jour l'état de santé d'un receiver"""
        self.prometheus.receiver_health.labels(
            receiver=receiver,
            tenant=tenant
        ).set(1 if healthy else 0)
    
    def record_config_reload(self, success: bool):
        """Enregistre un rechargement de configuration"""
        status = "success" if success else "error"
        self.prometheus.config_reloads_total.labels(status=status).inc()
    
    def record_performance_metrics(self, cpu_percent: float, memory_bytes: int):
        """Enregistre les métriques de performance système"""
        self.prometheus.cpu_usage.set(cpu_percent)
        self.prometheus.memory_usage.set(memory_bytes)
        
        # Buffer pour analyse de tendance
        self.performance_buffer.append({
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_bytes": memory_bytes
        })

class MetricsAnalyzer:
    """Analyseur de métriques avancé"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        
    def get_tenant_summary(self, tenant: str) -> Dict[str, Any]:
        """Récupère un résumé des métriques pour un tenant"""
        if tenant not in self.collector.tenant_metrics:
            return {}
        
        metrics = self.collector.tenant_metrics[tenant]
        
        # Calcul du taux de succès
        total = metrics["total_alerts"]
        success_rate = (metrics["successful_alerts"] / total * 100) if total > 0 else 0
        
        # Top services par nombre d'alertes
        top_services = sorted(
            metrics["by_service"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Répartition par sévérité (en pourcentage)
        severity_distribution = {}
        for severity, count in metrics["by_severity"].items():
            severity_distribution[severity] = (count / total * 100) if total > 0 else 0
        
        return {
            "tenant": tenant,
            "total_alerts": total,
            "success_rate": round(success_rate, 2),
            "average_processing_time": round(metrics["average_processing_time"], 3),
            "last_alert": metrics["last_alert"],
            "top_services": top_services,
            "severity_distribution": severity_distribution
        }
    
    def get_service_summary(self, service: str) -> Dict[str, Any]:
        """Récupère un résumé des métriques pour un service"""
        if service not in self.collector.service_metrics:
            return {}
        
        metrics = self.collector.service_metrics[service]
        
        total = metrics["total_alerts"]
        success_rate = (metrics["successful_alerts"] / total * 100) if total > 0 else 0
        
        # Top tenants pour ce service
        top_tenants = sorted(
            metrics["by_tenant"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "service": service,
            "total_alerts": total,
            "success_rate": round(success_rate, 2),
            "average_processing_time": round(metrics["average_processing_time"], 3),
            "last_alert": metrics["last_alert"],
            "top_tenants": top_tenants
        }
    
    def get_integration_summary(self, integration: str) -> Dict[str, Any]:
        """Récupère un résumé des métriques pour une intégration"""
        if integration not in self.collector.integration_metrics:
            return {}
        
        metrics = self.collector.integration_metrics[integration]
        
        total = metrics["total_requests"]
        success_rate = (metrics["successful_requests"] / total * 100) if total > 0 else 0
        
        # Performance par tenant
        tenant_performance = {}
        for tenant, stats in metrics["by_tenant"].items():
            tenant_total = stats["success"] + stats["error"]
            tenant_success_rate = (stats["success"] / tenant_total * 100) if tenant_total > 0 else 0
            tenant_performance[tenant] = {
                "total_requests": tenant_total,
                "success_rate": round(tenant_success_rate, 2)
            }
        
        return {
            "integration": integration,
            "total_requests": total,
            "success_rate": round(success_rate, 2),
            "average_response_time": round(metrics["average_response_time"], 3),
            "last_request": metrics["last_request"],
            "tenant_performance": tenant_performance
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Récupère un résumé de la santé du système"""
        # Analyse des performances récentes
        recent_performance = list(self.collector.performance_buffer)[-100:]  # 100 derniers points
        
        if recent_performance:
            cpu_values = [p["cpu_percent"] for p in recent_performance]
            memory_values = [p["memory_bytes"] for p in recent_performance]
            
            cpu_avg = statistics.mean(cpu_values)
            cpu_max = max(cpu_values)
            memory_avg = statistics.mean(memory_values)
            memory_max = max(memory_values)
        else:
            cpu_avg = cpu_max = memory_avg = memory_max = 0
        
        # Calcul des tendances (dernière heure)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_alerts = [
            alert for alert in self.collector.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > one_hour_ago
        ]
        
        alert_rate = len(recent_alerts)  # Alertes par heure
        
        # Taux de succès global
        if recent_alerts:
            successful_alerts = sum(1 for alert in recent_alerts if alert["successful_deliveries"] > 0)
            global_success_rate = (successful_alerts / len(recent_alerts) * 100)
        else:
            global_success_rate = 0
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_rate_per_hour": alert_rate,
            "global_success_rate": round(global_success_rate, 2),
            "system_performance": {
                "cpu_average": round(cpu_avg, 2),
                "cpu_peak": round(cpu_max, 2),
                "memory_average_mb": round(memory_avg / 1024 / 1024, 2),
                "memory_peak_mb": round(memory_max / 1024 / 1024, 2)
            },
            "total_tenants": len(self.collector.tenant_metrics),
            "total_services": len(self.collector.service_metrics),
            "active_integrations": len(self.collector.integration_metrics)
        }
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Détecte les anomalies dans les métriques"""
        anomalies = []
        
        # Détection d'anomalies par tenant
        for tenant, metrics in self.collector.tenant_metrics.items():
            total = metrics["total_alerts"]
            if total > 0:
                success_rate = metrics["successful_alerts"] / total * 100
                
                # Taux de succès anormalement bas
                if success_rate < 80 and total > 10:
                    anomalies.append({
                        "type": "low_success_rate",
                        "tenant": tenant,
                        "success_rate": round(success_rate, 2),
                        "total_alerts": total,
                        "severity": "high" if success_rate < 50 else "medium"
                    })
                
                # Temps de traitement anormalement élevé
                if metrics["average_processing_time"] > 10:  # 10 secondes
                    anomalies.append({
                        "type": "high_processing_time",
                        "tenant": tenant,
                        "processing_time": round(metrics["average_processing_time"], 3),
                        "severity": "medium"
                    })
        
        # Détection d'anomalies par intégration
        for integration, metrics in self.collector.integration_metrics.items():
            total = metrics["total_requests"]
            if total > 0:
                success_rate = metrics["successful_requests"] / total * 100
                
                if success_rate < 90 and total > 5:
                    anomalies.append({
                        "type": "integration_failure",
                        "integration": integration,
                        "success_rate": round(success_rate, 2),
                        "severity": "high" if success_rate < 70 else "medium"
                    })
        
        return anomalies

class MetricsConfigManager:
    """Gestionnaire principal des métriques"""
    
    def __init__(self, prometheus_port: int = 8090):
        self.collector = MetricsCollector()
        self.analyzer = MetricsAnalyzer(self.collector)
        self.prometheus_port = prometheus_port
        self.monitoring_tasks: List[asyncio.Task] = []
        
    async def initialize_metrics(self) -> bool:
        """Initialise le système de métriques"""
        try:
            logger.info("Initializing metrics configuration manager")
            
            # Démarrage du serveur Prometheus
            start_http_server(self.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            
            # Démarrage des tâches de monitoring
            await self._start_monitoring_tasks()
            
            logger.info("Metrics system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            return False
    
    async def _start_monitoring_tasks(self):
        """Démarre les tâches de monitoring"""
        
        # Tâche de collecte des métriques système
        system_metrics_task = asyncio.create_task(self._collect_system_metrics())
        self.monitoring_tasks.append(system_metrics_task)
        
        # Tâche de détection d'anomalies
        anomaly_detection_task = asyncio.create_task(self._monitor_anomalies())
        self.monitoring_tasks.append(anomaly_detection_task)
        
        # Tâche de nettoyage des métriques anciennes
        cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
        self.monitoring_tasks.append(cleanup_task)
    
    async def _collect_system_metrics(self):
        """Collecte les métriques système en continu"""
        while True:
            try:
                await asyncio.sleep(30)  # Collecte toutes les 30 secondes
                
                # Simulation de métriques système (en prod: utiliser psutil)
                import random
                cpu_percent = random.uniform(20, 80)
                memory_bytes = random.randint(512 * 1024 * 1024, 2 * 1024 * 1024 * 1024)
                
                self.collector.record_performance_metrics(cpu_percent, memory_bytes)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_anomalies(self):
        """Surveille les anomalies en continu"""
        while True:
            try:
                await asyncio.sleep(300)  # Vérification toutes les 5 minutes
                
                anomalies = self.analyzer.detect_anomalies()
                
                if anomalies:
                    logger.warning(f"Detected {len(anomalies)} anomalies")
                    for anomaly in anomalies:
                        logger.warning(f"Anomaly: {anomaly}")
                
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_old_metrics(self):
        """Nettoie les anciennes métriques"""
        while True:
            try:
                await asyncio.sleep(3600)  # Nettoyage toutes les heures
                
                # Nettoyage de l'historique des alertes (garder 24h)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                old_count = len(self.collector.alert_history)
                
                # Filtrage des alertes récentes
                recent_alerts = deque()
                for alert in self.collector.alert_history:
                    alert_time = datetime.fromisoformat(alert["timestamp"])
                    if alert_time > cutoff_time:
                        recent_alerts.append(alert)
                
                self.collector.alert_history = recent_alerts
                cleaned_count = old_count - len(recent_alerts)
                
                if cleaned_count > 0:
                    logger.info(f"Cleaned {cleaned_count} old alert records")
                
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
                await asyncio.sleep(3600)
    
    def record_alert(self, alert_metrics: AlertMetrics):
        """Enregistre une nouvelle alerte"""
        self.collector.record_alert_processed(alert_metrics)
    
    def record_integration_call(
        self, 
        integration: str, 
        tenant: str, 
        success: bool, 
        response_time: float
    ):
        """Enregistre un appel d'intégration"""
        self.collector.record_integration_request(integration, tenant, success, response_time)
    
    def get_tenant_metrics(self, tenant: str) -> Dict[str, Any]:
        """Récupère les métriques d'un tenant"""
        return self.analyzer.get_tenant_summary(tenant)
    
    def get_service_metrics(self, service: str) -> Dict[str, Any]:
        """Récupère les métriques d'un service"""
        return self.analyzer.get_service_summary(service)
    
    def get_integration_metrics(self, integration: str) -> Dict[str, Any]:
        """Récupère les métriques d'une intégration"""
        return self.analyzer.get_integration_summary(integration)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Récupère la santé du système"""
        return self.analyzer.get_system_health_summary()
    
    def get_all_anomalies(self) -> List[Dict[str, Any]]:
        """Récupère toutes les anomalies détectées"""
        return self.analyzer.detect_anomalies()
    
    async def shutdown(self):
        """Arrête proprement le système de métriques"""
        logger.info("Shutting down metrics system")
        
        # Arrêt des tâches de monitoring
        for task in self.monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics system shutdown completed")

# Instance singleton
metrics_manager = MetricsConfigManager()
