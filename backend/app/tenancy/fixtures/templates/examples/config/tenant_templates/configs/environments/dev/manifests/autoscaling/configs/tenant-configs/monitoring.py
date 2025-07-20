"""
Advanced Monitoring & Analytics System
=====================================

Système de surveillance et d'analytiques avancé pour les tenants avec ML intégré.
Collecte, analyse et prédit les métriques de performance en temps réel.

Fonctionnalités:
- Collecte de métriques temps réel multi-dimensionnelles
- Analyse prédictive avec ML
- Détection d'anomalies automatisée
- Optimisation continue des performances
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import statistics

# Configuration logging
logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types de métriques supportées."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Metric:
    """Représentation d'une métrique."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    unit: str = ""
    description: str = ""


@dataclass
class Alert:
    """Représentation d'une alerte."""
    tenant_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceProfile:
    """Profil de performance d'un tenant."""
    tenant_id: str
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_latency: float
    peak_rps: float
    error_rate: float
    availability: float
    efficiency_score: float
    cost_per_request: float
    updated_at: datetime = field(default_factory=datetime.utcnow)


class TenantMetricsCollector:
    """
    Collecteur de métriques avancé pour les tenants.
    
    Fonctionnalités:
    - Collecte temps réel multi-source
    - Agrégation intelligente
    - Stockage optimisé avec rétention
    - Export vers systèmes externes
    """
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.tenant_sessions: Dict[str, Dict] = {}
        self.active_collectors: Dict[str, bool] = {}
        
        # Configuration de stockage
        self.storage_backend = None
        self.export_handlers = []
        
        # Cache pour performances
        self.metrics_cache: Dict[str, Dict] = {}
        self.cache_ttl = 60  # 1 minute
        
        logger.info("TenantMetricsCollector initialized", retention_days=retention_days)
    
    async def start_collection(self, tenant_id: str, config: Dict = None):
        """Démarre la collecte de métriques pour un tenant."""
        try:
            if tenant_id in self.active_collectors:
                logger.warning("Metrics collection already active", tenant_id=tenant_id)
                return
            
            # Configuration de la collecte
            collection_config = config or {
                "interval": 30,  # seconds
                "metrics": [
                    "cpu_usage", "memory_usage", "network_io",
                    "disk_io", "request_rate", "response_time",
                    "error_rate", "active_connections"
                ],
                "aggregation_window": 300,  # 5 minutes
            }
            
            # Initialiser la session de collecte
            self.tenant_sessions[tenant_id] = {
                "started_at": datetime.utcnow(),
                "config": collection_config,
                "last_collection": None,
                "total_metrics": 0,
                "errors": 0,
            }
            
            self.active_collectors[tenant_id] = True
            
            # Démarrer la boucle de collecte
            asyncio.create_task(self._collection_loop(tenant_id))
            
            logger.info("Metrics collection started", tenant_id=tenant_id)
            
        except Exception as e:
            logger.error(
                "Failed to start metrics collection",
                tenant_id=tenant_id,
                error=str(e)
            )
            raise
    
    async def stop_collection(self, tenant_id: str):
        """Arrête la collecte de métriques pour un tenant."""
        if tenant_id in self.active_collectors:
            self.active_collectors[tenant_id] = False
            
            if tenant_id in self.tenant_sessions:
                session = self.tenant_sessions[tenant_id]
                session["stopped_at"] = datetime.utcnow()
                session["duration"] = session["stopped_at"] - session["started_at"]
            
            logger.info("Metrics collection stopped", tenant_id=tenant_id)
    
    async def collect_metric(
        self,
        tenant_id: str,
        metric: Metric
    ):
        """Collecte une métrique individuelle."""
        try:
            # Valider la métrique
            if not await self._validate_metric(metric):
                return
            
            # Ajouter au buffer
            metric_key = f"{tenant_id}:{metric.name}"
            self.metrics_buffer[metric_key].append(metric)
            
            # Mettre à jour les statistiques de session
            if tenant_id in self.tenant_sessions:
                self.tenant_sessions[tenant_id]["total_metrics"] += 1
                self.tenant_sessions[tenant_id]["last_collection"] = datetime.utcnow()
            
            # Stocker si configuré
            if self.storage_backend:
                await self.storage_backend.store_metric(tenant_id, metric)
            
            # Vérifier les seuils d'alerte
            await self._check_alert_thresholds(tenant_id, metric)
            
        except Exception as e:
            logger.error(
                "Failed to collect metric",
                tenant_id=tenant_id,
                metric_name=metric.name,
                error=str(e)
            )
            
            if tenant_id in self.tenant_sessions:
                self.tenant_sessions[tenant_id]["errors"] += 1
    
    async def get_tenant_metrics(
        self,
        tenant_id: str,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[Metric]]:
        """Récupère les métriques d'un tenant."""
        try:
            # Vérifier le cache
            cache_key = f"{tenant_id}:{hash(str(metric_names))}"
            if cache_key in self.metrics_cache:
                cache_entry = self.metrics_cache[cache_key]
                if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                    return cache_entry["data"]
            
            # Récupérer depuis le buffer/stockage
            result = {}
            
            if metric_names is None:
                # Récupérer toutes les métriques du tenant
                metric_names = [
                    key.split(":", 1)[1] for key in self.metrics_buffer.keys()
                    if key.startswith(f"{tenant_id}:")
                ]
            
            for metric_name in metric_names:
                metric_key = f"{tenant_id}:{metric_name}"
                metrics = list(self.metrics_buffer.get(metric_key, []))
                
                # Filtrer par temps si spécifié
                if start_time or end_time:
                    metrics = [
                        m for m in metrics
                        if (start_time is None or m.timestamp >= start_time) and
                           (end_time is None or m.timestamp <= end_time)
                    ]
                
                result[metric_name] = metrics
            
            # Mettre en cache
            self.metrics_cache[cache_key] = {
                "data": result,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to get tenant metrics",
                tenant_id=tenant_id,
                error=str(e)
            )
            return {}
    
    async def get_aggregated_metrics(
        self,
        tenant_id: str,
        metric_name: str,
        aggregation: str = "avg",
        window_size: int = 300  # 5 minutes
    ) -> Optional[float]:
        """Calcule des métriques agrégées."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(seconds=window_size)
            
            metrics_data = await self.get_tenant_metrics(
                tenant_id, [metric_name], start_time, end_time
            )
            
            if metric_name not in metrics_data or not metrics_data[metric_name]:
                return None
            
            values = [m.value for m in metrics_data[metric_name]]
            
            if aggregation == "avg":
                return statistics.mean(values)
            elif aggregation == "max":
                return max(values)
            elif aggregation == "min":
                return min(values)
            elif aggregation == "sum":
                return sum(values)
            elif aggregation == "median":
                return statistics.median(values)
            elif aggregation == "p95":
                return np.percentile(values, 95)
            elif aggregation == "p99":
                return np.percentile(values, 99)
            else:
                return statistics.mean(values)
                
        except Exception as e:
            logger.error(
                "Failed to calculate aggregated metrics",
                tenant_id=tenant_id,
                metric_name=metric_name,
                error=str(e)
            )
            return None
    
    async def _collection_loop(self, tenant_id: str):
        """Boucle principale de collecte de métriques."""
        session = self.tenant_sessions.get(tenant_id)
        if not session:
            return
        
        interval = session["config"]["interval"]
        
        while self.active_collectors.get(tenant_id, False):
            try:
                # Collecter les métriques système
                await self._collect_system_metrics(tenant_id)
                
                # Collecter les métriques application
                await self._collect_application_metrics(tenant_id)
                
                # Collecter les métriques business
                await self._collect_business_metrics(tenant_id)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(
                    "Error in metrics collection loop",
                    tenant_id=tenant_id,
                    error=str(e)
                )
                await asyncio.sleep(interval * 2)  # Wait longer on error
    
    async def _collect_system_metrics(self, tenant_id: str):
        """Collecte les métriques système."""
        try:
            # CPU Usage
            cpu_usage = await self._get_cpu_usage(tenant_id)
            if cpu_usage is not None:
                await self.collect_metric(tenant_id, Metric(
                    name="cpu_usage",
                    value=cpu_usage,
                    timestamp=datetime.utcnow(),
                    labels={"type": "system"},
                    unit="percent"
                ))
            
            # Memory Usage
            memory_usage = await self._get_memory_usage(tenant_id)
            if memory_usage is not None:
                await self.collect_metric(tenant_id, Metric(
                    name="memory_usage",
                    value=memory_usage,
                    timestamp=datetime.utcnow(),
                    labels={"type": "system"},
                    unit="percent"
                ))
            
            # Network I/O
            network_io = await self._get_network_io(tenant_id)
            if network_io:
                for direction, value in network_io.items():
                    await self.collect_metric(tenant_id, Metric(
                        name="network_io",
                        value=value,
                        timestamp=datetime.utcnow(),
                        labels={"type": "system", "direction": direction},
                        unit="bytes/sec"
                    ))
                    
        except Exception as e:
            logger.error(
                "Failed to collect system metrics",
                tenant_id=tenant_id,
                error=str(e)
            )
    
    async def _collect_application_metrics(self, tenant_id: str):
        """Collecte les métriques application."""
        try:
            # Request Rate
            request_rate = await self._get_request_rate(tenant_id)
            if request_rate is not None:
                await self.collect_metric(tenant_id, Metric(
                    name="request_rate",
                    value=request_rate,
                    timestamp=datetime.utcnow(),
                    labels={"type": "application"},
                    unit="req/sec"
                ))
            
            # Response Time
            response_time = await self._get_response_time(tenant_id)
            if response_time is not None:
                await self.collect_metric(tenant_id, Metric(
                    name="response_time",
                    value=response_time,
                    timestamp=datetime.utcnow(),
                    labels={"type": "application"},
                    unit="ms"
                ))
            
            # Error Rate
            error_rate = await self._get_error_rate(tenant_id)
            if error_rate is not None:
                await self.collect_metric(tenant_id, Metric(
                    name="error_rate",
                    value=error_rate,
                    timestamp=datetime.utcnow(),
                    labels={"type": "application"},
                    unit="percent"
                ))
                
        except Exception as e:
            logger.error(
                "Failed to collect application metrics",
                tenant_id=tenant_id,
                error=str(e)
            )
    
    async def _collect_business_metrics(self, tenant_id: str):
        """Collecte les métriques business."""
        try:
            # Active Users
            active_users = await self._get_active_users(tenant_id)
            if active_users is not None:
                await self.collect_metric(tenant_id, Metric(
                    name="active_users",
                    value=active_users,
                    timestamp=datetime.utcnow(),
                    labels={"type": "business"},
                    unit="count"
                ))
            
            # Revenue per Hour
            revenue = await self._get_revenue_per_hour(tenant_id)
            if revenue is not None:
                await self.collect_metric(tenant_id, Metric(
                    name="revenue_per_hour",
                    value=revenue,
                    timestamp=datetime.utcnow(),
                    labels={"type": "business"},
                    unit="currency"
                ))
                
        except Exception as e:
            logger.error(
                "Failed to collect business metrics",
                tenant_id=tenant_id,
                error=str(e)
            )
    
    # Méthodes de collecte spécialisées (à implémenter selon l'infrastructure)
    
    async def _get_cpu_usage(self, tenant_id: str) -> Optional[float]:
        """Récupère l'utilisation CPU du tenant."""
        # Implémentation selon l'infrastructure (Kubernetes, Docker, etc.)
        return None
    
    async def _get_memory_usage(self, tenant_id: str) -> Optional[float]:
        """Récupère l'utilisation mémoire du tenant."""
        return None
    
    async def _get_network_io(self, tenant_id: str) -> Optional[Dict[str, float]]:
        """Récupère les I/O réseau du tenant."""
        return None
    
    async def _get_request_rate(self, tenant_id: str) -> Optional[float]:
        """Récupère le taux de requêtes du tenant."""
        return None
    
    async def _get_response_time(self, tenant_id: str) -> Optional[float]:
        """Récupère le temps de réponse moyen du tenant."""
        return None
    
    async def _get_error_rate(self, tenant_id: str) -> Optional[float]:
        """Récupère le taux d'erreur du tenant."""
        return None
    
    async def _get_active_users(self, tenant_id: str) -> Optional[int]:
        """Récupère le nombre d'utilisateurs actifs du tenant."""
        return None
    
    async def _get_revenue_per_hour(self, tenant_id: str) -> Optional[float]:
        """Récupère le revenu par heure du tenant."""
        return None
    
    async def _validate_metric(self, metric: Metric) -> bool:
        """Valide une métrique."""
        if not metric.name:
            return False
        if metric.value is None:
            return False
        if not isinstance(metric.timestamp, datetime):
            return False
        return True
    
    async def _check_alert_thresholds(self, tenant_id: str, metric: Metric):
        """Vérifie les seuils d'alerte pour une métrique."""
        # Implémentation de la logique d'alerte
        pass


class PerformanceAnalyzer:
    """
    Analyseur de performance avancé avec ML intégré.
    
    Fonctionnalités:
    - Analyse de tendances et patterns
    - Détection d'anomalies automatisée
    - Recommandations d'optimisation
    - Prédictions de performance
    """
    
    def __init__(self, metrics_collector: TenantMetricsCollector):
        self.metrics_collector = metrics_collector
        self.ml_models = {}
        self.analysis_cache = {}
        self.anomaly_detectors = {}
        
        logger.info("PerformanceAnalyzer initialized")
    
    async def analyze_tenant_performance(self, tenant_id: str) -> PerformanceProfile:
        """Analyse complète des performances d'un tenant."""
        try:
            # Récupérer les métriques des dernières 24h
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            metrics = await self.metrics_collector.get_tenant_metrics(
                tenant_id, start_time=start_time, end_time=end_time
            )
            
            # Calculer les métriques de performance
            profile = PerformanceProfile(
                tenant_id=tenant_id,
                avg_cpu_usage=await self._calculate_avg_metric(metrics, "cpu_usage"),
                avg_memory_usage=await self._calculate_avg_metric(metrics, "memory_usage"),
                avg_latency=await self._calculate_avg_metric(metrics, "response_time"),
                peak_rps=await self._calculate_peak_metric(metrics, "request_rate"),
                error_rate=await self._calculate_avg_metric(metrics, "error_rate"),
                availability=await self._calculate_availability(metrics),
                efficiency_score=0.0,  # À calculer
                cost_per_request=0.0,  # À calculer
            )
            
            # Calculer le score d'efficacité
            profile.efficiency_score = await self._calculate_efficiency_score(profile)
            
            return profile
            
        except Exception as e:
            logger.error(
                "Failed to analyze tenant performance",
                tenant_id=tenant_id,
                error=str(e)
            )
            raise
    
    async def detect_anomalies(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Détecte les anomalies dans les métriques d'un tenant."""
        try:
            anomalies = []
            
            # Récupérer les métriques récentes
            metrics = await self.metrics_collector.get_tenant_metrics(tenant_id)
            
            for metric_name, metric_data in metrics.items():
                if not metric_data:
                    continue
                
                # Analyser les anomalies pour cette métrique
                metric_anomalies = await self._detect_metric_anomalies(
                    tenant_id, metric_name, metric_data
                )
                anomalies.extend(metric_anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(
                "Failed to detect anomalies",
                tenant_id=tenant_id,
                error=str(e)
            )
            return []
    
    async def analyze_and_optimize(
        self,
        tenant_id: str,
        config: Any,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyse et propose des optimisations."""
        try:
            optimizations = {}
            
            # Analyser les tendances de performance
            performance_profile = await self.analyze_tenant_performance(tenant_id)
            
            # Recommandations basées sur l'utilisation CPU
            if performance_profile.avg_cpu_usage > 80:
                optimizations["autoscaling.max_replicas"] = min(
                    config.autoscaling.max_replicas * 1.5,
                    100
                )
            elif performance_profile.avg_cpu_usage < 30:
                optimizations["autoscaling.max_replicas"] = max(
                    config.autoscaling.max_replicas * 0.8,
                    config.autoscaling.min_replicas
                )
            
            # Recommandations basées sur la latence
            if performance_profile.avg_latency > 1000:  # 1 seconde
                optimizations["autoscaling.metrics.latency_threshold"] = min(
                    config.autoscaling.metrics.latency_threshold * 0.8,
                    500
                )
            
            # Recommandations basées sur le taux d'erreur
            if performance_profile.error_rate > 5:
                optimizations["circuit_breaker_enabled"] = True
                optimizations["autoscaling.scale_up_cooldown"] = max(
                    config.autoscaling.scale_up_cooldown * 0.7,
                    60
                )
            
            return optimizations
            
        except Exception as e:
            logger.error(
                "Failed to analyze and optimize",
                tenant_id=tenant_id,
                error=str(e)
            )
            return {}
    
    # Méthodes d'aide pour les calculs
    
    async def _calculate_avg_metric(self, metrics: Dict, metric_name: str) -> float:
        """Calcule la moyenne d'une métrique."""
        if metric_name not in metrics or not metrics[metric_name]:
            return 0.0
        
        values = [m.value for m in metrics[metric_name]]
        return statistics.mean(values) if values else 0.0
    
    async def _calculate_peak_metric(self, metrics: Dict, metric_name: str) -> float:
        """Calcule le pic d'une métrique."""
        if metric_name not in metrics or not metrics[metric_name]:
            return 0.0
        
        values = [m.value for m in metrics[metric_name]]
        return max(values) if values else 0.0
    
    async def _calculate_availability(self, metrics: Dict) -> float:
        """Calcule la disponibilité basée sur les métriques."""
        # Logique de calcul de disponibilité
        return 99.9  # Placeholder
    
    async def _calculate_efficiency_score(self, profile: PerformanceProfile) -> float:
        """Calcule un score d'efficacité composite."""
        # Algorithme de calcul du score d'efficacité
        cpu_score = max(0, 100 - profile.avg_cpu_usage) / 100
        latency_score = max(0, (1000 - profile.avg_latency) / 1000)
        error_score = max(0, (100 - profile.error_rate) / 100)
        
        return (cpu_score + latency_score + error_score) / 3 * 100
    
    async def _detect_metric_anomalies(
        self,
        tenant_id: str,
        metric_name: str,
        metric_data: List[Metric]
    ) -> List[Dict[str, Any]]:
        """Détecte les anomalies dans une métrique spécifique."""
        anomalies = []
        
        if len(metric_data) < 10:  # Pas assez de données
            return anomalies
        
        values = [m.value for m in metric_data]
        
        # Calcul statistique simple pour la détection d'anomalies
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        # Seuil d'anomalie (3 sigma)
        threshold = mean_val + (3 * std_val)
        
        for metric in metric_data[-10:]:  # Vérifier les 10 dernières valeurs
            if metric.value > threshold and std_val > 0:
                anomalies.append({
                    "metric_name": metric_name,
                    "value": metric.value,
                    "expected_range": f"{mean_val - std_val:.2f} - {mean_val + std_val:.2f}",
                    "deviation": abs(metric.value - mean_val) / std_val if std_val > 0 else 0,
                    "timestamp": metric.timestamp,
                    "severity": "high" if abs(metric.value - mean_val) > 5 * std_val else "medium"
                })
        
        return anomalies
