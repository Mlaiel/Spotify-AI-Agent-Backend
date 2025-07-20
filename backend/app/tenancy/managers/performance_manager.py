"""
⚡ Tenant Performance Manager - Gestionnaire Performance Multi-Tenant
===================================================================

Gestionnaire avancé de performance et optimisation pour l'architecture multi-tenant.
Gère le monitoring, l'optimisation automatique et la montée en charge.

Features:
- Monitoring performance en temps réel
- Optimisation automatique des ressources
- Gestion de la montée en charge
- Détection des goulots d'étranglement
- Prédiction de charge avec ML
- Optimisation des requêtes SQL
- Cache intelligent adaptatif
- Load balancing dynamique
- Alertes de performance
- Rapports de performance détaillés

Author: Architecte IA + ML Engineer + Backend Senior Developer
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import time
import statistics
import numpy as np
from collections import deque, defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert, delete, text
from fastapi import HTTPException
from pydantic import BaseModel, validator
import redis.asyncio as redis
import psutil
import asyncpg

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class PerformanceMetric(str, Enum):
    """Métriques de performance"""
    RESPONSE_TIME = "response_time"          # Temps de réponse
    THROUGHPUT = "throughput"                # Débit
    CPU_USAGE = "cpu_usage"                  # Utilisation CPU
    MEMORY_USAGE = "memory_usage"            # Utilisation mémoire
    DISK_IO = "disk_io"                     # I/O disque
    NETWORK_IO = "network_io"               # I/O réseau
    DATABASE_CONNECTIONS = "db_connections"  # Connexions DB
    CACHE_HIT_RATE = "cache_hit_rate"       # Taux de cache hit
    ERROR_RATE = "error_rate"               # Taux d'erreur
    QUEUE_SIZE = "queue_size"               # Taille des files


class OptimizationStrategy(str, Enum):
    """Stratégies d'optimisation"""
    AUTO_SCALING = "auto_scaling"            # Montée en charge automatique
    CACHE_OPTIMIZATION = "cache_optimization" # Optimisation cache
    QUERY_OPTIMIZATION = "query_optimization" # Optimisation requêtes
    LOAD_BALANCING = "load_balancing"        # Équilibrage de charge
    RESOURCE_ALLOCATION = "resource_allocation" # Allocation ressources
    CONNECTION_POOLING = "connection_pooling" # Pool de connexions


class PerformanceStatus(str, Enum):
    """États de performance"""
    OPTIMAL = "optimal"                      # Performance optimale
    GOOD = "good"                           # Bonne performance
    DEGRADED = "degraded"                   # Performance dégradée
    CRITICAL = "critical"                   # Performance critique
    FAILING = "failing"                     # Défaillance


class AlertSeverity(str, Enum):
    """Sévérité des alertes"""
    INFO = "info"                           # Information
    WARNING = "warning"                     # Avertissement
    CRITICAL = "critical"                   # Critique
    EMERGENCY = "emergency"                 # Urgence


@dataclass
class PerformanceThreshold:
    """Seuils de performance"""
    metric: PerformanceMetric
    warning_threshold: float
    critical_threshold: float
    unit: str
    direction: str = "above"  # "above" ou "below"


@dataclass
class PerformanceData:
    """Données de performance"""
    tenant_id: str
    metric: PerformanceMetric
    value: float
    timestamp: datetime
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationAction:
    """Action d'optimisation"""
    action_id: str
    tenant_id: str
    strategy: OptimizationStrategy
    description: str
    parameters: Dict[str, Any]
    estimated_impact: float  # Impact estimé en %
    execution_time: Optional[datetime] = None
    success: Optional[bool] = None
    actual_impact: Optional[float] = None


@dataclass
class PerformanceAlert:
    """Alerte de performance"""
    alert_id: str
    tenant_id: str
    metric: PerformanceMetric
    severity: AlertSeverity
    threshold_value: float
    actual_value: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class LoadPrediction:
    """Prédiction de charge"""
    tenant_id: str
    predicted_load: float
    confidence: float
    prediction_horizon: timedelta
    predicted_at: datetime
    factors: Dict[str, float] = field(default_factory=dict)


class PerformanceConfig(BaseModel):
    """Configuration de performance"""
    tenant_id: str
    auto_optimization_enabled: bool = True
    monitoring_interval: int = 60  # secondes
    thresholds: Dict[str, Dict[str, float]] = {}
    optimization_strategies: List[OptimizationStrategy] = []
    max_auto_actions_per_hour: int = 5


class TenantPerformanceManager:
    """
    Gestionnaire de performance multi-tenant avancé.
    
    Responsabilités:
    - Monitoring performance en temps réel
    - Détection proactive des problèmes
    - Optimisation automatique
    - Prédiction de charge ML
    - Gestion des alertes
    - Génération de rapports
    """

    def __init__(self):
        self._redis_client: Optional[redis.Redis] = None
        
        # Stockage des données
        self.performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24h à 1min
        self.active_alerts: Dict[str, List[PerformanceAlert]] = defaultdict(list)
        self.optimization_history: Dict[str, List[OptimizationAction]] = defaultdict(list)
        self.tenant_configs: Dict[str, PerformanceConfig] = {}
        
        # Seuils par défaut
        self.default_thresholds = {
            PerformanceMetric.RESPONSE_TIME: PerformanceThreshold(
                metric=PerformanceMetric.RESPONSE_TIME,
                warning_threshold=1000.0,    # 1s
                critical_threshold=5000.0,   # 5s
                unit="ms",
                direction="above"
            ),
            PerformanceMetric.CPU_USAGE: PerformanceThreshold(
                metric=PerformanceMetric.CPU_USAGE,
                warning_threshold=70.0,      # 70%
                critical_threshold=90.0,     # 90%
                unit="%",
                direction="above"
            ),
            PerformanceMetric.MEMORY_USAGE: PerformanceThreshold(
                metric=PerformanceMetric.MEMORY_USAGE,
                warning_threshold=80.0,      # 80%
                critical_threshold=95.0,     # 95%
                unit="%",
                direction="above"
            ),
            PerformanceMetric.ERROR_RATE: PerformanceThreshold(
                metric=PerformanceMetric.ERROR_RATE,
                warning_threshold=1.0,       # 1%
                critical_threshold=5.0,      # 5%
                unit="%",
                direction="above"
            ),
            PerformanceMetric.CACHE_HIT_RATE: PerformanceThreshold(
                metric=PerformanceMetric.CACHE_HIT_RATE,
                warning_threshold=80.0,      # 80%
                critical_threshold=60.0,     # 60%
                unit="%",
                direction="below"
            )
        }
        
        # Configuration ML
        self.ml_models = {}
        self.prediction_cache = {}
        
        # Monitoring actif
        self.monitoring_tasks = {}
        
        # Configuration
        self.config = {
            "metrics_retention_hours": 24,
            "prediction_update_interval": 300,  # 5 minutes
            "auto_optimization_enabled": True,
            "max_concurrent_optimizations": 3,
            "performance_sampling_rate": 60,    # secondes
        }

    async def get_redis_client(self) -> redis.Redis:
        """Obtenir le client Redis"""
        if not self._redis_client:
            self._redis_client = await get_redis_client()
        return self._redis_client

    async def start_monitoring(
        self,
        tenant_id: str,
        config: Optional[PerformanceConfig] = None
    ):
        """
        Démarrer le monitoring pour un tenant.
        
        Args:
            tenant_id: ID du tenant
            config: Configuration de monitoring (optionnelle)
        """
        try:
            # Configuration par défaut
            if not config:
                config = PerformanceConfig(tenant_id=tenant_id)
            
            self.tenant_configs[tenant_id] = config
            
            # Démarrage de la tâche de monitoring
            if tenant_id not in self.monitoring_tasks:
                task = asyncio.create_task(
                    self._monitoring_loop(tenant_id, config)
                )
                self.monitoring_tasks[tenant_id] = task

            logger.info(f"Monitoring démarré pour tenant {tenant_id}")

        except Exception as e:
            logger.error(f"Erreur démarrage monitoring: {str(e)}")
            raise

    async def stop_monitoring(self, tenant_id: str):
        """
        Arrêter le monitoring pour un tenant.
        
        Args:
            tenant_id: ID du tenant
        """
        try:
            if tenant_id in self.monitoring_tasks:
                self.monitoring_tasks[tenant_id].cancel()
                del self.monitoring_tasks[tenant_id]

            logger.info(f"Monitoring arrêté pour tenant {tenant_id}")

        except Exception as e:
            logger.error(f"Erreur arrêt monitoring: {str(e)}")

    async def record_performance_metric(
        self,
        tenant_id: str,
        metric: PerformanceMetric,
        value: float,
        metadata: Dict[str, Any] = None
    ):
        """
        Enregistrer une métrique de performance.
        
        Args:
            tenant_id: ID du tenant
            metric: Type de métrique
            value: Valeur de la métrique
            metadata: Métadonnées additionnelles
        """
        try:
            data_point = PerformanceData(
                tenant_id=tenant_id,
                metric=metric,
                value=value,
                timestamp=datetime.utcnow(),
                unit=self.default_thresholds.get(metric, PerformanceThreshold(
                    metric=metric, warning_threshold=0, critical_threshold=0, unit=""
                )).unit,
                metadata=metadata or {}
            )

            # Stockage en mémoire
            key = f"{tenant_id}:{metric}"
            self.performance_data[key].append(data_point)

            # Stockage dans Redis pour persistance
            await self._store_metric_in_redis(data_point)

            # Vérification des seuils
            await self._check_thresholds(tenant_id, metric, value)

            # Déclenchement d'optimisations si nécessaire
            if self.tenant_configs.get(tenant_id, PerformanceConfig(tenant_id=tenant_id)).auto_optimization_enabled:
                await self._trigger_auto_optimization(tenant_id, metric, value)

        except Exception as e:
            logger.error(f"Erreur enregistrement métrique: {str(e)}")

    async def get_performance_metrics(
        self,
        tenant_id: str,
        metrics: List[PerformanceMetric] = None,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Récupérer les métriques de performance.
        
        Args:
            tenant_id: ID du tenant
            metrics: Liste des métriques (toutes si None)
            time_range: Plage temporelle (dernière heure par défaut)
            
        Returns:
            Dictionnaire des métriques
        """
        try:
            if not metrics:
                metrics = list(PerformanceMetric)
            
            if not time_range:
                time_range = timedelta(hours=1)

            cutoff_time = datetime.utcnow() - time_range
            result = {}

            for metric in metrics:
                key = f"{tenant_id}:{metric}"
                data_points = self.performance_data.get(key, deque())
                
                # Filtrage par temps
                filtered_points = [
                    {
                        "timestamp": dp.timestamp.isoformat(),
                        "value": dp.value,
                        "unit": dp.unit,
                        "metadata": dp.metadata
                    }
                    for dp in data_points
                    if dp.timestamp >= cutoff_time
                ]
                
                result[metric] = filtered_points

            return result

        except Exception as e:
            logger.error(f"Erreur récupération métriques: {str(e)}")
            return {}

    async def get_performance_summary(
        self,
        tenant_id: str,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Obtenir un résumé de performance.
        
        Args:
            tenant_id: ID du tenant
            time_range: Plage temporelle
            
        Returns:
            Résumé de performance
        """
        try:
            if not time_range:
                time_range = timedelta(hours=1)

            metrics_data = await self.get_performance_metrics(tenant_id, time_range=time_range)
            summary = {
                "tenant_id": tenant_id,
                "time_range": str(time_range),
                "overall_status": PerformanceStatus.OPTIMAL,
                "metrics_summary": {},
                "active_alerts": len(self.active_alerts.get(tenant_id, [])),
                "recent_optimizations": len([
                    opt for opt in self.optimization_history.get(tenant_id, [])
                    if opt.execution_time and 
                    opt.execution_time >= datetime.utcnow() - time_range
                ])
            }

            worst_status = PerformanceStatus.OPTIMAL
            
            for metric, data_points in metrics_data.items():
                if not data_points:
                    continue
                
                values = [dp["value"] for dp in data_points]
                threshold = self.default_thresholds.get(metric)
                
                metric_summary = {
                    "current_value": values[-1] if values else 0,
                    "average": statistics.mean(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "trend": self._calculate_trend(values),
                    "status": self._evaluate_metric_status(values[-1] if values else 0, threshold)
                }
                
                summary["metrics_summary"][metric] = metric_summary
                
                # Mise à jour du statut global
                metric_status = metric_summary["status"]
                if self._status_priority(metric_status) > self._status_priority(worst_status):
                    worst_status = metric_status

            summary["overall_status"] = worst_status
            return summary

        except Exception as e:
            logger.error(f"Erreur résumé performance: {str(e)}")
            return {"tenant_id": tenant_id, "error": str(e)}

    async def predict_performance_load(
        self,
        tenant_id: str,
        horizon: timedelta = timedelta(hours=1)
    ) -> LoadPrediction:
        """
        Prédire la charge de performance future.
        
        Args:
            tenant_id: ID du tenant
            horizon: Horizon de prédiction
            
        Returns:
            Prédiction de charge
        """
        try:
            # Récupération des données historiques
            historical_data = await self.get_performance_metrics(
                tenant_id,
                time_range=timedelta(days=7)
            )

            # Analyse des patterns
            cpu_data = [dp["value"] for dp in historical_data.get(PerformanceMetric.CPU_USAGE, [])]
            memory_data = [dp["value"] for dp in historical_data.get(PerformanceMetric.MEMORY_USAGE, [])]
            
            if not cpu_data or not memory_data:
                return LoadPrediction(
                    tenant_id=tenant_id,
                    predicted_load=50.0,  # Valeur par défaut
                    confidence=0.5,
                    prediction_horizon=horizon,
                    predicted_at=datetime.utcnow()
                )

            # Calcul de tendances simples (en production, utiliser des modèles ML plus sophistiqués)
            cpu_trend = self._calculate_trend(cpu_data[-60:])  # Dernière heure
            memory_trend = self._calculate_trend(memory_data[-60:])
            
            # Prédiction basée sur les tendances
            current_cpu = cpu_data[-1] if cpu_data else 50.0
            current_memory = memory_data[-1] if memory_data else 50.0
            
            # Extrapolation simple
            horizon_minutes = horizon.total_seconds() / 60
            predicted_cpu = max(0, min(100, current_cpu + (cpu_trend * horizon_minutes)))
            predicted_memory = max(0, min(100, current_memory + (memory_trend * horizon_minutes)))
            
            # Charge globale prédite
            predicted_load = (predicted_cpu + predicted_memory) / 2
            
            # Calcul de confiance basé sur la stabilité des données
            cpu_variance = statistics.variance(cpu_data[-60:]) if len(cpu_data) >= 2 else 0
            memory_variance = statistics.variance(memory_data[-60:]) if len(memory_data) >= 2 else 0
            confidence = max(0.1, 1.0 - (cpu_variance + memory_variance) / 10000)

            prediction = LoadPrediction(
                tenant_id=tenant_id,
                predicted_load=predicted_load,
                confidence=confidence,
                prediction_horizon=horizon,
                predicted_at=datetime.utcnow(),
                factors={
                    "cpu_trend": cpu_trend,
                    "memory_trend": memory_trend,
                    "current_cpu": current_cpu,
                    "current_memory": current_memory
                }
            )

            # Cache de la prédiction
            self.prediction_cache[tenant_id] = prediction

            return prediction

        except Exception as e:
            logger.error(f"Erreur prédiction charge: {str(e)}")
            return LoadPrediction(
                tenant_id=tenant_id,
                predicted_load=50.0,
                confidence=0.1,
                prediction_horizon=horizon,
                predicted_at=datetime.utcnow()
            )

    async def optimize_tenant_performance(
        self,
        tenant_id: str,
        strategies: List[OptimizationStrategy] = None
    ) -> List[str]:
        """
        Optimiser la performance d'un tenant.
        
        Args:
            tenant_id: ID du tenant
            strategies: Stratégies d'optimisation (toutes par défaut)
            
        Returns:
            Liste des IDs d'actions d'optimisation
        """
        try:
            if not strategies:
                strategies = list(OptimizationStrategy)

            executed_actions = []
            
            for strategy in strategies:
                action_id = await self._execute_optimization_strategy(tenant_id, strategy)
                if action_id:
                    executed_actions.append(action_id)

            return executed_actions

        except Exception as e:
            logger.error(f"Erreur optimisation performance: {str(e)}")
            return []

    async def get_active_alerts(
        self,
        tenant_id: str,
        severity: Optional[AlertSeverity] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupérer les alertes actives.
        
        Args:
            tenant_id: ID du tenant
            severity: Filtre par sévérité
            
        Returns:
            Liste des alertes actives
        """
        try:
            alerts = self.active_alerts.get(tenant_id, [])
            
            # Filtrage par sévérité si spécifié
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            
            # Filtrage des alertes non résolues
            active_alerts = [alert for alert in alerts if not alert.resolved]
            
            return [
                {
                    "alert_id": alert.alert_id,
                    "metric": alert.metric,
                    "severity": alert.severity,
                    "threshold_value": alert.threshold_value,
                    "actual_value": alert.actual_value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ]

        except Exception as e:
            logger.error(f"Erreur récupération alertes: {str(e)}")
            return []

    # Méthodes privées

    async def _monitoring_loop(
        self,
        tenant_id: str,
        config: PerformanceConfig
    ):
        """Boucle de monitoring principale"""
        try:
            while True:
                # Collecte des métriques système
                await self._collect_system_metrics(tenant_id)
                
                # Collecte des métriques applicatives
                await self._collect_application_metrics(tenant_id)
                
                # Collecte des métriques de base de données
                await self._collect_database_metrics(tenant_id)
                
                # Mise à jour des prédictions
                if datetime.utcnow().timestamp() % self.config["prediction_update_interval"] == 0:
                    await self.predict_performance_load(tenant_id)

                await asyncio.sleep(config.monitoring_interval)

        except asyncio.CancelledError:
            logger.info(f"Monitoring arrêté pour tenant {tenant_id}")
        except Exception as e:
            logger.error(f"Erreur boucle monitoring: {str(e)}")

    async def _collect_system_metrics(self, tenant_id: str):
        """Collecter les métriques système"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.record_performance_metric(
                tenant_id, PerformanceMetric.CPU_USAGE, cpu_percent
            )

            # Mémoire
            memory = psutil.virtual_memory()
            await self.record_performance_metric(
                tenant_id, PerformanceMetric.MEMORY_USAGE, memory.percent
            )

            # Disque I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                await self.record_performance_metric(
                    tenant_id, PerformanceMetric.DISK_IO, 
                    disk_io.read_bytes + disk_io.write_bytes,
                    {"read_bytes": disk_io.read_bytes, "write_bytes": disk_io.write_bytes}
                )

        except Exception as e:
            logger.error(f"Erreur collecte métriques système: {str(e)}")

    async def _collect_application_metrics(self, tenant_id: str):
        """Collecter les métriques applicatives"""
        try:
            # Simulation de métriques applicatives
            # En production, intégrer avec l'application réelle
            
            # Temps de réponse moyen (simulation)
            response_time = np.random.normal(500, 100)  # 500ms ± 100ms
            await self.record_performance_metric(
                tenant_id, PerformanceMetric.RESPONSE_TIME, response_time
            )

            # Taux d'erreur (simulation)
            error_rate = max(0, np.random.normal(0.5, 0.2))  # 0.5% ± 0.2%
            await self.record_performance_metric(
                tenant_id, PerformanceMetric.ERROR_RATE, error_rate
            )

        except Exception as e:
            logger.error(f"Erreur collecte métriques applicatives: {str(e)}")

    async def _collect_database_metrics(self, tenant_id: str):
        """Collecter les métriques de base de données"""
        try:
            # Simulation de métriques DB
            # En production, se connecter à la vraie DB
            
            # Connexions actives
            db_connections = np.random.randint(5, 50)
            await self.record_performance_metric(
                tenant_id, PerformanceMetric.DATABASE_CONNECTIONS, db_connections
            )

            # Taux de cache hit
            cache_hit_rate = max(60, min(99, np.random.normal(85, 5)))  # 85% ± 5%
            await self.record_performance_metric(
                tenant_id, PerformanceMetric.CACHE_HIT_RATE, cache_hit_rate
            )

        except Exception as e:
            logger.error(f"Erreur collecte métriques DB: {str(e)}")

    async def _check_thresholds(
        self,
        tenant_id: str,
        metric: PerformanceMetric,
        value: float
    ):
        """Vérifier les seuils et générer des alertes"""
        try:
            threshold = self.default_thresholds.get(metric)
            if not threshold:
                return

            severity = None
            threshold_value = None

            # Détermination de la sévérité
            if threshold.direction == "above":
                if value >= threshold.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif value >= threshold.warning_threshold:
                    severity = AlertSeverity.WARNING
                    threshold_value = threshold.warning_threshold
            else:  # direction == "below"
                if value <= threshold.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif value <= threshold.warning_threshold:
                    severity = AlertSeverity.WARNING
                    threshold_value = threshold.warning_threshold

            # Génération d'alerte si nécessaire
            if severity:
                await self._create_alert(tenant_id, metric, severity, threshold_value, value)

        except Exception as e:
            logger.error(f"Erreur vérification seuils: {str(e)}")

    async def _create_alert(
        self,
        tenant_id: str,
        metric: PerformanceMetric,
        severity: AlertSeverity,
        threshold_value: float,
        actual_value: float
    ):
        """Créer une alerte de performance"""
        try:
            # Vérifier si une alerte similaire existe déjà
            existing_alerts = [
                alert for alert in self.active_alerts.get(tenant_id, [])
                if alert.metric == metric and not alert.resolved
            ]
            
            if existing_alerts:
                return  # Éviter la duplication d'alertes

            alert = PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                metric=metric,
                severity=severity,
                threshold_value=threshold_value,
                actual_value=actual_value,
                message=f"{metric} {severity}: {actual_value} (seuil: {threshold_value})",
                timestamp=datetime.utcnow()
            )

            if tenant_id not in self.active_alerts:
                self.active_alerts[tenant_id] = []
            self.active_alerts[tenant_id].append(alert)

            # Notification (en production, utiliser un système de notification)
            logger.warning(f"Alerte performance: {alert.message}")

        except Exception as e:
            logger.error(f"Erreur création alerte: {str(e)}")

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculer la tendance d'une série de valeurs"""
        if len(values) < 2:
            return 0.0
        
        # Régression linéaire simple
        n = len(values)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(i * v for i, v in enumerate(values))
        sum_x2 = sum(i * i for i in x)
        
        # Pente de la droite de régression
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope

    def _evaluate_metric_status(
        self,
        value: float,
        threshold: Optional[PerformanceThreshold]
    ) -> PerformanceStatus:
        """Évaluer le statut d'une métrique"""
        if not threshold:
            return PerformanceStatus.OPTIMAL

        if threshold.direction == "above":
            if value >= threshold.critical_threshold:
                return PerformanceStatus.CRITICAL
            elif value >= threshold.warning_threshold:
                return PerformanceStatus.DEGRADED
            else:
                return PerformanceStatus.OPTIMAL
        else:  # direction == "below"
            if value <= threshold.critical_threshold:
                return PerformanceStatus.CRITICAL
            elif value <= threshold.warning_threshold:
                return PerformanceStatus.DEGRADED
            else:
                return PerformanceStatus.OPTIMAL

    def _status_priority(self, status: PerformanceStatus) -> int:
        """Obtenir la priorité d'un statut"""
        priority_map = {
            PerformanceStatus.OPTIMAL: 0,
            PerformanceStatus.GOOD: 1,
            PerformanceStatus.DEGRADED: 2,
            PerformanceStatus.CRITICAL: 3,
            PerformanceStatus.FAILING: 4
        }
        return priority_map.get(status, 0)

    async def _store_metric_in_redis(self, data_point: PerformanceData):
        """Stocker une métrique dans Redis"""
        try:
            redis_client = await self.get_redis_client()
            key = f"metrics:{data_point.tenant_id}:{data_point.metric}"
            
            # Stockage avec TTL
            await redis_client.zadd(
                key,
                {
                    json.dumps({
                        "value": data_point.value,
                        "unit": data_point.unit,
                        "metadata": data_point.metadata
                    }): data_point.timestamp.timestamp()
                }
            )
            
            # Nettoyage des anciennes données
            cutoff = (datetime.utcnow() - timedelta(hours=self.config["metrics_retention_hours"])).timestamp()
            await redis_client.zremrangebyscore(key, 0, cutoff)

        except Exception as e:
            logger.error(f"Erreur stockage métrique Redis: {str(e)}")


# Instance globale du gestionnaire de performance
tenant_performance_manager = TenantPerformanceManager()
