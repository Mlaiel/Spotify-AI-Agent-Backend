"""
Advanced Metrics Management System
=================================

Système de gestion des métriques avancé pour autoscaling.
Intègre collecte temps réel, analyse prédictive et optimisation automatique.

Fonctionnalités:
- Collecte de métriques multi-dimensionnelles
- Agrégation et analyse temps réel
- Prédiction avec Machine Learning
- Alerting intelligent et automatisé
"""

import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
import uuid
import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque

# Configuration logging
logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types de métriques."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"
    PERCENTAGE = "percentage"


class MetricScope(Enum):
    """Portée des métriques."""
    SYSTEM = "system"
    TENANT = "tenant"
    APPLICATION = "application"
    SERVICE = "service"
    INSTANCE = "instance"
    RESOURCE = "resource"


class AlertSeverity(Enum):
    """Sévérité des alertes."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AggregationType(Enum):
    """Types d'agrégation."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    COUNT = "count"
    RATE = "rate"


@dataclass
class MetricValue:
    """Valeur de métrique."""
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """Métrique complète."""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    scope: MetricScope
    unit: str
    labels: Dict[str, str] = field(default_factory=dict)
    values: List[MetricValue] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Règle d'alerte."""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "> 80", "< 10", "== 0"
    severity: AlertSeverity
    duration: int  # seconds
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alerte déclenchée."""
    alert_id: str
    rule_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricQuery:
    """Requête de métrique."""
    metric_names: List[str]
    start_time: datetime
    end_time: datetime
    aggregation: AggregationType = AggregationType.AVERAGE
    group_by: List[str] = field(default_factory=list)
    filters: Dict[str, str] = field(default_factory=dict)
    step: int = 60  # seconds


class MetricsCollector:
    """
    Collecteur de métriques avancé avec intelligence artificielle.
    
    Fonctionnalités:
    - Collecte temps réel multi-sources
    - Agrégation intelligente
    - Détection d'anomalies
    - Prédiction de tendances
    """
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.metric_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.collectors: Dict[str, Callable] = {}
        
        # Configuration
        self.collection_interval = 10  # seconds
        self.retention_period = timedelta(days=30)
        self.cache_size = 1000
        self.batch_size = 100
        
        # Statistiques
        self.metrics_collected = 0
        self.alerts_triggered = 0
        self.collection_errors = 0
        
        # ML et prédiction
        self.ml_models: Dict[str, Any] = {}
        self.prediction_cache: Dict[str, Any] = {}
        
        logger.info("MetricsCollector initialized")
    
    async def initialize(self):
        """Initialise le collecteur de métriques."""
        try:
            # Enregistrer les collecteurs par défaut
            await self._register_default_collectors()
            
            # Charger les règles d'alerte par défaut
            await self._load_default_alert_rules()
            
            # Initialiser les modèles ML
            await self._initialize_ml_models()
            
            # Démarrer les tâches de collecte
            asyncio.create_task(self._collection_loop())
            asyncio.create_task(self._alert_processing_loop())
            asyncio.create_task(self._cleanup_loop())
            asyncio.create_task(self._prediction_loop())
            
            logger.info("MetricsCollector fully initialized")
            
        except Exception as e:
            logger.error("Failed to initialize MetricsCollector", error=str(e))
            raise
    
    async def register_metric(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        scope: MetricScope,
        unit: str,
        labels: Optional[Dict[str, str]] = None
    ) -> str:
        """Enregistre une nouvelle métrique."""
        try:
            metric_id = str(uuid.uuid4())
            
            metric = Metric(
                metric_id=metric_id,
                name=name,
                description=description,
                metric_type=metric_type,
                scope=scope,
                unit=unit,
                labels=labels or {}
            )
            
            self.metrics[name] = metric
            
            logger.info(
                "Metric registered",
                metric_id=metric_id,
                name=name,
                type=metric_type.value,
                scope=scope.value
            )
            
            return metric_id
            
        except Exception as e:
            logger.error(
                "Failed to register metric",
                name=name,
                error=str(e)
            )
            raise
    
    async def collect_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """Collecte une valeur de métrique."""
        try:
            if metric_name not in self.metrics:
                logger.warning("Unknown metric", metric_name=metric_name)
                return
            
            metric_value = MetricValue(
                value=value,
                timestamp=timestamp or datetime.utcnow(),
                labels=labels or {}
            )
            
            # Ajouter au cache
            self.metric_cache[metric_name].append(metric_value)
            
            # Ajouter à la métrique
            metric = self.metrics[metric_name]
            metric.values.append(metric_value)
            
            # Taille la liste pour éviter une croissance excessive
            if len(metric.values) > self.cache_size:
                metric.values = metric.values[-self.cache_size:]
            
            # Vérifier les alertes
            await self._check_alerts(metric_name, value, labels or {})
            
            # Mettre à jour les statistiques
            self.metrics_collected += 1
            
            logger.debug(
                "Metric collected",
                metric_name=metric_name,
                value=value,
                labels=labels
            )
            
        except Exception as e:
            logger.error(
                "Failed to collect metric",
                metric_name=metric_name,
                value=value,
                error=str(e)
            )
            self.collection_errors += 1
    
    async def query_metrics(
        self,
        query: MetricQuery
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """Exécute une requête de métrique."""
        try:
            results = {}
            
            for metric_name in query.metric_names:
                if metric_name not in self.metrics:
                    continue
                
                metric = self.metrics[metric_name]
                
                # Filtrer par période
                filtered_values = [
                    v for v in metric.values
                    if query.start_time <= v.timestamp <= query.end_time
                ]
                
                # Appliquer les filtres
                if query.filters:
                    filtered_values = [
                        v for v in filtered_values
                        if all(
                            v.labels.get(k) == filter_value
                            for k, filter_value in query.filters.items()
                        )
                    ]
                
                # Grouper et agréger
                aggregated_data = await self._aggregate_values(
                    filtered_values,
                    query.aggregation,
                    query.step
                )
                
                results[metric_name] = aggregated_data
            
            logger.debug(
                "Metrics queried",
                metric_names=query.metric_names,
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Failed to query metrics",
                metric_names=query.metric_names,
                error=str(e)
            )
            return {}
    
    async def create_alert_rule(
        self,
        name: str,
        description: str,
        metric_name: str,
        condition: str,
        severity: AlertSeverity,
        duration: int = 300,
        notification_channels: Optional[List[str]] = None
    ) -> str:
        """Crée une règle d'alerte."""
        try:
            rule_id = str(uuid.uuid4())
            
            alert_rule = AlertRule(
                rule_id=rule_id,
                name=name,
                description=description,
                metric_name=metric_name,
                condition=condition,
                severity=severity,
                duration=duration,
                notification_channels=notification_channels or []
            )
            
            self.alert_rules[rule_id] = alert_rule
            
            logger.info(
                "Alert rule created",
                rule_id=rule_id,
                name=name,
                metric_name=metric_name,
                condition=condition
            )
            
            return rule_id
            
        except Exception as e:
            logger.error(
                "Failed to create alert rule",
                name=name,
                error=str(e)
            )
            raise
    
    async def get_metric_statistics(
        self,
        metric_name: str,
        duration: timedelta = timedelta(hours=1)
    ) -> Dict[str, float]:
        """Calcule les statistiques d'une métrique."""
        try:
            if metric_name not in self.metrics:
                return {}
            
            metric = self.metrics[metric_name]
            cutoff_time = datetime.utcnow() - duration
            
            # Filtrer les valeurs récentes
            recent_values = [
                v.value for v in metric.values
                if v.timestamp >= cutoff_time
            ]
            
            if not recent_values:
                return {}
            
            stats = {
                "count": len(recent_values),
                "min": min(recent_values),
                "max": max(recent_values),
                "average": statistics.mean(recent_values),
                "median": statistics.median(recent_values),
                "current": recent_values[-1] if recent_values else 0
            }
            
            # Calculs avancés
            if len(recent_values) > 1:
                stats["stdev"] = statistics.stdev(recent_values)
                
                # Percentiles
                sorted_values = sorted(recent_values)
                stats["p95"] = sorted_values[int(len(sorted_values) * 0.95)]
                stats["p99"] = sorted_values[int(len(sorted_values) * 0.99)]
                
                # Tendance
                if len(recent_values) >= 10:
                    first_half = recent_values[:len(recent_values)//2]
                    second_half = recent_values[len(recent_values)//2:]
                    
                    avg_first = statistics.mean(first_half)
                    avg_second = statistics.mean(second_half)
                    
                    trend = (avg_second - avg_first) / avg_first * 100
                    stats["trend_percentage"] = trend
            
            return stats
            
        except Exception as e:
            logger.error(
                "Failed to calculate metric statistics",
                metric_name=metric_name,
                error=str(e)
            )
            return {}
    
    async def predict_metric_values(
        self,
        metric_name: str,
        horizon: timedelta = timedelta(hours=1)
    ) -> List[Tuple[datetime, float]]:
        """Prédit les futures valeurs d'une métrique."""
        try:
            if metric_name not in self.metrics:
                return []
            
            # Vérifier le cache de prédiction
            cache_key = f"{metric_name}_{horizon.total_seconds()}"
            if cache_key in self.prediction_cache:
                cached_prediction = self.prediction_cache[cache_key]
                if datetime.utcnow() - cached_prediction["timestamp"] < timedelta(minutes=5):
                    return cached_prediction["predictions"]
            
            metric = self.metrics[metric_name]
            
            # Utiliser les valeurs récentes pour la prédiction
            recent_values = metric.values[-100:] if len(metric.values) >= 100 else metric.values
            
            if len(recent_values) < 10:
                return []
            
            # Prédiction simple basée sur la tendance
            predictions = await self._simple_trend_prediction(
                recent_values, horizon
            )
            
            # Prédiction ML si disponible
            if metric_name in self.ml_models:
                ml_predictions = await self._ml_prediction(
                    metric_name, recent_values, horizon
                )
                if ml_predictions:
                    predictions = ml_predictions
            
            # Mettre en cache
            self.prediction_cache[cache_key] = {
                "timestamp": datetime.utcnow(),
                "predictions": predictions
            }
            
            logger.debug(
                "Metric values predicted",
                metric_name=metric_name,
                horizon_hours=horizon.total_seconds()/3600,
                predictions_count=len(predictions)
            )
            
            return predictions
            
        except Exception as e:
            logger.error(
                "Failed to predict metric values",
                metric_name=metric_name,
                error=str(e)
            )
            return []
    
    async def get_active_alerts(self) -> List[Alert]:
        """Récupère les alertes actives."""
        return [
            alert for alert in self.active_alerts.values()
            if alert.resolved_at is None
        ]
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acquitte une alerte."""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                logger.info("Alert acknowledged", alert_id=alert_id)
                return True
            return False
            
        except Exception as e:
            logger.error("Failed to acknowledge alert", alert_id=alert_id, error=str(e))
            return False
    
    async def register_collector(
        self,
        name: str,
        collector_func: Callable[[], Dict[str, float]]
    ):
        """Enregistre un collecteur personnalisé."""
        self.collectors[name] = collector_func
        logger.info("Custom collector registered", name=name)
    
    # Méthodes privées
    
    async def _register_default_collectors(self):
        """Enregistre les collecteurs par défaut."""
        try:
            # Collecteur de métriques système
            async def system_metrics_collector():
                return {
                    "cpu_usage": 45.2,  # Simulé
                    "memory_usage": 62.8,
                    "disk_usage": 38.1,
                    "network_in": 1024.5,
                    "network_out": 856.3
                }
            
            # Collecteur de métriques d'application
            async def application_metrics_collector():
                return {
                    "request_rate": 150.0,
                    "response_time": 85.2,
                    "error_rate": 2.1,
                    "active_sessions": 245
                }
            
            self.collectors["system"] = system_metrics_collector
            self.collectors["application"] = application_metrics_collector
            
            # Enregistrer les métriques correspondantes
            await self.register_metric(
                "cpu_usage", "CPU Usage Percentage", MetricType.GAUGE,
                MetricScope.SYSTEM, "percent"
            )
            await self.register_metric(
                "memory_usage", "Memory Usage Percentage", MetricType.GAUGE,
                MetricScope.SYSTEM, "percent"
            )
            await self.register_metric(
                "request_rate", "Request Rate", MetricType.RATE,
                MetricScope.APPLICATION, "req/sec"
            )
            await self.register_metric(
                "response_time", "Response Time", MetricType.HISTOGRAM,
                MetricScope.APPLICATION, "milliseconds"
            )
            await self.register_metric(
                "error_rate", "Error Rate", MetricType.PERCENTAGE,
                MetricScope.APPLICATION, "percent"
            )
            
            logger.info("Default collectors registered")
            
        except Exception as e:
            logger.error("Failed to register default collectors", error=str(e))
    
    async def _load_default_alert_rules(self):
        """Charge les règles d'alerte par défaut."""
        try:
            # Règle d'alerte pour CPU élevé
            await self.create_alert_rule(
                name="High CPU Usage",
                description="Alert when CPU usage is above 80%",
                metric_name="cpu_usage",
                condition="> 80",
                severity=AlertSeverity.HIGH,
                duration=300
            )
            
            # Règle d'alerte pour mémoire élevée
            await self.create_alert_rule(
                name="High Memory Usage",
                description="Alert when memory usage is above 90%",
                metric_name="memory_usage",
                condition="> 90",
                severity=AlertSeverity.CRITICAL,
                duration=180
            )
            
            # Règle d'alerte pour taux d'erreur élevé
            await self.create_alert_rule(
                name="High Error Rate",
                description="Alert when error rate is above 5%",
                metric_name="error_rate",
                condition="> 5",
                severity=AlertSeverity.HIGH,
                duration=120
            )
            
            logger.info("Default alert rules loaded")
            
        except Exception as e:
            logger.error("Failed to load default alert rules", error=str(e))
    
    async def _initialize_ml_models(self):
        """Initialise les modèles ML pour prédiction."""
        try:
            # Simuler l'initialisation de modèles ML
            for metric_name in ["cpu_usage", "memory_usage", "request_rate"]:
                self.ml_models[metric_name] = {
                    "type": "time_series_forecasting",
                    "algorithm": "LSTM",
                    "accuracy": 0.85,
                    "trained": True,
                    "last_update": datetime.utcnow()
                }
            
            logger.info("ML models initialized for prediction")
            
        except Exception as e:
            logger.error("Failed to initialize ML models", error=str(e))
    
    async def _aggregate_values(
        self,
        values: List[MetricValue],
        aggregation: AggregationType,
        step: int
    ) -> List[Tuple[datetime, float]]:
        """Agrège les valeurs selon le type d'agrégation."""
        if not values:
            return []
        
        # Grouper par intervalles de temps
        grouped_values = defaultdict(list)
        
        for value in values:
            # Arrondir le timestamp à l'intervalle
            timestamp_rounded = datetime.fromtimestamp(
                (value.timestamp.timestamp() // step) * step
            )
            grouped_values[timestamp_rounded].append(value.value)
        
        # Appliquer l'agrégation
        aggregated = []
        for timestamp, group_values in sorted(grouped_values.items()):
            if aggregation == AggregationType.SUM:
                agg_value = sum(group_values)
            elif aggregation == AggregationType.AVERAGE:
                agg_value = statistics.mean(group_values)
            elif aggregation == AggregationType.MIN:
                agg_value = min(group_values)
            elif aggregation == AggregationType.MAX:
                agg_value = max(group_values)
            elif aggregation == AggregationType.MEDIAN:
                agg_value = statistics.median(group_values)
            elif aggregation == AggregationType.COUNT:
                agg_value = len(group_values)
            else:
                agg_value = statistics.mean(group_values)  # Par défaut
            
            aggregated.append((timestamp, agg_value))
        
        return aggregated
    
    async def _check_alerts(
        self,
        metric_name: str,
        value: float,
        labels: Dict[str, str]
    ):
        """Vérifie si des alertes doivent être déclenchées."""
        try:
            for rule in self.alert_rules.values():
                if rule.metric_name != metric_name or not rule.enabled:
                    continue
                
                # Évaluer la condition
                condition_met = await self._evaluate_alert_condition(rule.condition, value)
                
                if condition_met:
                    # Vérifier si l'alerte existe déjà
                    existing_alert = None
                    for alert in self.active_alerts.values():
                        if (alert.rule_id == rule.rule_id and 
                            alert.resolved_at is None):
                            existing_alert = alert
                            break
                    
                    if not existing_alert:
                        # Créer une nouvelle alerte
                        alert = Alert(
                            alert_id=str(uuid.uuid4()),
                            rule_id=rule.rule_id,
                            metric_name=metric_name,
                            current_value=value,
                            threshold_value=self._extract_threshold_from_condition(rule.condition),
                            severity=rule.severity,
                            message=f"Metric {metric_name} value {value} triggered alert rule {rule.name}",
                            triggered_at=datetime.utcnow()
                        )
                        
                        self.active_alerts[alert.alert_id] = alert
                        self.alerts_triggered += 1
                        
                        logger.warning(
                            "Alert triggered",
                            alert_id=alert.alert_id,
                            rule_name=rule.name,
                            metric_name=metric_name,
                            current_value=value
                        )
                        
                        # Envoyer les notifications
                        await self._send_alert_notifications(alert, rule)
                
                else:
                    # Résoudre les alertes existantes si la condition n'est plus remplie
                    for alert in list(self.active_alerts.values()):
                        if (alert.rule_id == rule.rule_id and 
                            alert.resolved_at is None):
                            alert.resolved_at = datetime.utcnow()
                            
                            logger.info(
                                "Alert resolved",
                                alert_id=alert.alert_id,
                                rule_name=rule.name,
                                metric_name=metric_name
                            )
            
        except Exception as e:
            logger.error(
                "Failed to check alerts",
                metric_name=metric_name,
                value=value,
                error=str(e)
            )
    
    async def _evaluate_alert_condition(
        self,
        condition: str,
        value: float
    ) -> bool:
        """Évalue une condition d'alerte."""
        try:
            # Parser la condition (format: "> 80", "< 10", "== 0", etc.)
            condition = condition.strip()
            
            if condition.startswith(">= "):
                threshold = float(condition[3:])
                return value >= threshold
            elif condition.startswith("<= "):
                threshold = float(condition[3:])
                return value <= threshold
            elif condition.startswith("> "):
                threshold = float(condition[2:])
                return value > threshold
            elif condition.startswith("< "):
                threshold = float(condition[2:])
                return value < threshold
            elif condition.startswith("== "):
                threshold = float(condition[3:])
                return value == threshold
            elif condition.startswith("!= "):
                threshold = float(condition[3:])
                return value != threshold
            
            return False
            
        except Exception as e:
            logger.error("Failed to evaluate alert condition", condition=condition, error=str(e))
            return False
    
    def _extract_threshold_from_condition(self, condition: str) -> float:
        """Extrait la valeur seuil d'une condition."""
        try:
            condition = condition.strip()
            for prefix in [">= ", "<= ", "> ", "< ", "== ", "!= "]:
                if condition.startswith(prefix):
                    return float(condition[len(prefix):])
            return 0.0
        except:
            return 0.0
    
    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule):
        """Envoie les notifications d'alerte."""
        try:
            # Simuler l'envoi de notifications
            for channel in rule.notification_channels:
                logger.info(
                    "Alert notification sent",
                    alert_id=alert.alert_id,
                    channel=channel,
                    severity=alert.severity.value
                )
            
        except Exception as e:
            logger.error("Failed to send alert notifications", alert_id=alert.alert_id, error=str(e))
    
    async def _simple_trend_prediction(
        self,
        values: List[MetricValue],
        horizon: timedelta
    ) -> List[Tuple[datetime, float]]:
        """Prédiction simple basée sur la tendance."""
        if len(values) < 5:
            return []
        
        # Calculer la tendance linéaire
        x_values = list(range(len(values)))
        y_values = [v.value for v in values]
        
        # Régression linéaire simple
        n = len(values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Générer les prédictions
        predictions = []
        last_timestamp = values[-1].timestamp
        prediction_points = int(horizon.total_seconds() / 60)  # Une prédiction par minute
        
        for i in range(1, prediction_points + 1):
            future_timestamp = last_timestamp + timedelta(minutes=i)
            predicted_value = slope * (len(values) + i) + intercept
            
            # Assurer que la valeur reste dans des limites raisonnables
            predicted_value = max(0, predicted_value)
            
            predictions.append((future_timestamp, predicted_value))
        
        return predictions
    
    async def _ml_prediction(
        self,
        metric_name: str,
        values: List[MetricValue],
        horizon: timedelta
    ) -> List[Tuple[datetime, float]]:
        """Prédiction avec modèle ML."""
        try:
            model = self.ml_models.get(metric_name)
            if not model or not model.get("trained"):
                return []
            
            # Simuler la prédiction ML
            # En production, ceci utiliserait un vrai modèle TensorFlow/PyTorch
            
            recent_values = [v.value for v in values[-20:]]  # Utiliser les 20 dernières valeurs
            
            # Prédiction simulée avec variation autour de la moyenne
            avg_value = statistics.mean(recent_values)
            std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
            
            predictions = []
            last_timestamp = values[-1].timestamp
            prediction_points = int(horizon.total_seconds() / 60)
            
            for i in range(1, prediction_points + 1):
                future_timestamp = last_timestamp + timedelta(minutes=i)
                
                # Simuler une prédiction avec tendance et bruit
                trend_factor = 0.99 + (i / prediction_points) * 0.02  # Légère tendance
                noise_factor = 1 + (statistics.random() - 0.5) * 0.1  # Bruit aléatoire
                
                predicted_value = avg_value * trend_factor * noise_factor
                predicted_value = max(0, predicted_value)
                
                predictions.append((future_timestamp, predicted_value))
            
            return predictions
            
        except Exception as e:
            logger.error("ML prediction failed", metric_name=metric_name, error=str(e))
            return []
    
    # Boucles de traitement
    
    async def _collection_loop(self):
        """Boucle principale de collecte."""
        while True:
            try:
                # Exécuter tous les collecteurs
                for name, collector in self.collectors.items():
                    try:
                        collected_data = await collector()
                        
                        for metric_name, value in collected_data.items():
                            await self.collect_metric(metric_name, value)
                            
                    except Exception as e:
                        logger.error(
                            "Collector failed",
                            collector_name=name,
                            error=str(e)
                        )
                        self.collection_errors += 1
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error("Error in collection loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _alert_processing_loop(self):
        """Boucle de traitement des alertes."""
        while True:
            try:
                # Nettoyer les alertes résolues anciennes
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                resolved_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if alert.resolved_at and alert.resolved_at < cutoff_time
                ]
                
                for alert_id in resolved_alerts:
                    del self.active_alerts[alert_id]
                
                if resolved_alerts:
                    logger.debug("Old resolved alerts cleaned up", count=len(resolved_alerts))
                
                await asyncio.sleep(300)  # Traiter toutes les 5 minutes
                
            except Exception as e:
                logger.error("Error in alert processing loop", error=str(e))
                await asyncio.sleep(600)
    
    async def _cleanup_loop(self):
        """Boucle de nettoyage des données."""
        while True:
            try:
                cutoff_time = datetime.utcnow() - self.retention_period
                
                # Nettoyer les anciennes valeurs de métriques
                for metric in self.metrics.values():
                    original_count = len(metric.values)
                    metric.values = [
                        v for v in metric.values
                        if v.timestamp > cutoff_time
                    ]
                    
                    cleaned_count = original_count - len(metric.values)
                    if cleaned_count > 0:
                        logger.debug(
                            "Metric values cleaned",
                            metric_name=metric.name,
                            cleaned_count=cleaned_count
                        )
                
                # Nettoyer le cache de prédiction
                current_time = datetime.utcnow()
                expired_predictions = [
                    key for key, value in self.prediction_cache.items()
                    if current_time - value["timestamp"] > timedelta(hours=1)
                ]
                
                for key in expired_predictions:
                    del self.prediction_cache[key]
                
                await asyncio.sleep(3600)  # Nettoyer toutes les heures
                
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(7200)
    
    async def _prediction_loop(self):
        """Boucle de mise à jour des prédictions."""
        while True:
            try:
                # Mettre à jour les prédictions pour les métriques importantes
                important_metrics = ["cpu_usage", "memory_usage", "request_rate"]
                
                for metric_name in important_metrics:
                    if metric_name in self.metrics:
                        await self.predict_metric_values(
                            metric_name,
                            timedelta(hours=1)
                        )
                
                await asyncio.sleep(300)  # Mettre à jour toutes les 5 minutes
                
            except Exception as e:
                logger.error("Error in prediction loop", error=str(e))
                await asyncio.sleep(600)
