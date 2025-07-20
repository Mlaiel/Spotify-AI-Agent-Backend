"""
Advanced Analytics & Predictive Intelligence System
==================================================

Système d'analytiques avancé avec intelligence prédictive pour optimisation multi-tenant.
Intègre ML/AI pour prédictions, optimisations et insights en temps réel.

Fonctionnalités:
- Analytics multi-dimensionnels en temps réel
- Prédictions ML pour scaling et performance
- Détection d'anomalies automatisée
- Optimisation intelligente des ressources
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
import statistics
from collections import defaultdict, deque
import uuid

# Configuration logging
logger = structlog.get_logger(__name__)


class AnalyticsType(Enum):
    """Types d'analytiques."""
    PERFORMANCE = "performance"
    USAGE = "usage"
    COST = "cost"
    SECURITY = "security"
    BUSINESS = "business"
    PREDICTION = "prediction"


class PredictionHorizon(Enum):
    """Horizons de prédiction."""
    SHORT_TERM = "1h"      # 1 heure
    MEDIUM_TERM = "24h"    # 24 heures
    LONG_TERM = "7d"       # 7 jours
    STRATEGIC = "30d"      # 30 jours


class AlertLevel(Enum):
    """Niveaux d'alerte."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AnalyticsMetric:
    """Métrique d'analytiques."""
    metric_id: str
    tenant_id: str
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class Prediction:
    """Prédiction analytique."""
    prediction_id: str
    tenant_id: str
    metric_name: str
    predicted_value: float
    confidence_score: float
    horizon: PredictionHorizon
    factors: List[str]
    model_version: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    valid_until: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=1))


@dataclass
class Anomaly:
    """Anomalie détectée."""
    anomaly_id: str
    tenant_id: str
    metric_name: str
    actual_value: float
    expected_value: float
    deviation_score: float
    severity: str
    description: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False


@dataclass
class OptimizationRecommendation:
    """Recommandation d'optimisation."""
    recommendation_id: str
    tenant_id: str
    category: str  # performance, cost, security
    title: str
    description: str
    impact: str  # low, medium, high
    effort: str  # low, medium, high
    estimated_savings: Optional[float]
    implementation_steps: List[str]
    priority_score: float
    confidence: float
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TenantInsight:
    """Insight sur un tenant."""
    insight_id: str
    tenant_id: str
    title: str
    description: str
    category: str
    importance: str
    data_points: List[Dict[str, Any]]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)


class TenantAnalytics:
    """
    Système d'analytiques avancé pour tenants avec IA intégrée.
    
    Fonctionnalités:
    - Collection et agrégation de métriques multi-source
    - Calculs d'analytiques en temps réel
    - Génération d'insights automatisée
    - Tableaux de bord dynamiques
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.insights_cache: List[TenantInsight] = []
        self.anomalies: List[Anomaly] = []
        
        # Configuration
        self.buffer_size = 10000
        self.aggregation_intervals = [60, 300, 3600, 86400]  # 1m, 5m, 1h, 1d
        self.ml_enabled = True
        
        logger.info("TenantAnalytics initialized", tenant_id=tenant_id)
    
    async def ingest_metric(self, metric: AnalyticsMetric):
        """Ingère une métrique dans le système d'analytiques."""
        try:
            # Ajouter au buffer
            self.metrics_buffer[metric.name].append(metric)
            
            # Calculer les agrégations en temps réel
            await self._update_real_time_aggregations(metric)
            
            # Détecter les anomalies
            if self.ml_enabled:
                await self._detect_anomalies(metric)
            
            logger.debug(
                "Metric ingested",
                tenant_id=self.tenant_id,
                metric_name=metric.name,
                value=metric.value
            )
            
        except Exception as e:
            logger.error(
                "Failed to ingest metric",
                tenant_id=self.tenant_id,
                metric_name=metric.name,
                error=str(e)
            )
    
    async def get_analytics_summary(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Génère un résumé analytique complet."""
        try:
            if not time_range:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=24)
                time_range = (start_time, end_time)
            
            summary = {
                "tenant_id": self.tenant_id,
                "time_range": {
                    "start": time_range[0].isoformat(),
                    "end": time_range[1].isoformat()
                },
                "performance_metrics": await self._calculate_performance_metrics(time_range),
                "usage_metrics": await self._calculate_usage_metrics(time_range),
                "cost_metrics": await self._calculate_cost_metrics(time_range),
                "security_metrics": await self._calculate_security_metrics(time_range),
                "trends": await self._calculate_trends(time_range),
                "anomalies": await self._get_recent_anomalies(time_range),
                "top_insights": await self._get_top_insights(5),
                "health_score": await self._calculate_health_score(),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(
                "Failed to generate analytics summary",
                tenant_id=self.tenant_id,
                error=str(e)
            )
            return {}
    
    async def generate_insights(self) -> List[TenantInsight]:
        """Génère des insights automatisés pour le tenant."""
        try:
            insights = []
            
            # Insights de performance
            performance_insights = await self._analyze_performance_patterns()
            insights.extend(performance_insights)
            
            # Insights de coûts
            cost_insights = await self._analyze_cost_patterns()
            insights.extend(cost_insights)
            
            # Insights de sécurité
            security_insights = await self._analyze_security_patterns()
            insights.extend(security_insights)
            
            # Insights d'usage
            usage_insights = await self._analyze_usage_patterns()
            insights.extend(usage_insights)
            
            # Mettre en cache
            self.insights_cache = insights
            
            logger.info(
                "Insights generated",
                tenant_id=self.tenant_id,
                count=len(insights)
            )
            
            return insights
            
        except Exception as e:
            logger.error(
                "Failed to generate insights",
                tenant_id=self.tenant_id,
                error=str(e)
            )
            return []
    
    async def predict_resource_needs(
        self,
        horizon: PredictionHorizon = PredictionHorizon.MEDIUM_TERM
    ) -> Dict[str, Prediction]:
        """Prédit les besoins en ressources."""
        try:
            predictions = {}
            
            # Prédire CPU
            cpu_prediction = await self._predict_metric("cpu_usage", horizon)
            if cpu_prediction:
                predictions["cpu"] = cpu_prediction
            
            # Prédire Memory
            memory_prediction = await self._predict_metric("memory_usage", horizon)
            if memory_prediction:
                predictions["memory"] = memory_prediction
            
            # Prédire Network
            network_prediction = await self._predict_metric("network_io", horizon)
            if network_prediction:
                predictions["network"] = network_prediction
            
            # Prédire Storage
            storage_prediction = await self._predict_metric("storage_usage", horizon)
            if storage_prediction:
                predictions["storage"] = storage_prediction
            
            logger.info(
                "Resource predictions generated",
                tenant_id=self.tenant_id,
                horizon=horizon.value,
                predictions_count=len(predictions)
            )
            
            return predictions
            
        except Exception as e:
            logger.error(
                "Failed to predict resource needs",
                tenant_id=self.tenant_id,
                error=str(e)
            )
            return {}
    
    async def get_performance_baseline(self) -> Dict[str, float]:
        """Calcule les baselines de performance."""
        try:
            baseline = {}
            
            # Calculer les baselines sur les 7 derniers jours
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)
            
            key_metrics = ["cpu_usage", "memory_usage", "response_time", "error_rate"]
            
            for metric_name in key_metrics:
                values = await self._get_metric_values(metric_name, start_time, end_time)
                if values:
                    baseline[metric_name] = {
                        "avg": statistics.mean(values),
                        "p50": np.percentile(values, 50),
                        "p95": np.percentile(values, 95),
                        "p99": np.percentile(values, 99),
                        "std": statistics.stdev(values) if len(values) > 1 else 0
                    }
            
            return baseline
            
        except Exception as e:
            logger.error(
                "Failed to calculate performance baseline",
                tenant_id=self.tenant_id,
                error=str(e)
            )
            return {}
    
    # Méthodes privées
    
    async def _update_real_time_aggregations(self, metric: AnalyticsMetric):
        """Met à jour les agrégations en temps réel."""
        # Implémentation des agrégations temps réel
        pass
    
    async def _detect_anomalies(self, metric: AnalyticsMetric):
        """Détecte les anomalies dans les métriques."""
        try:
            metric_history = list(self.metrics_buffer[metric.name])
            
            if len(metric_history) < 10:  # Pas assez d'historique
                return
            
            # Calculer la valeur attendue et la déviation
            recent_values = [m.value for m in metric_history[-20:]]
            mean_value = statistics.mean(recent_values)
            std_value = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
            
            # Détecter l'anomalie (3-sigma rule)
            if std_value > 0:
                z_score = abs(metric.value - mean_value) / std_value
                if z_score > 3:
                    anomaly = Anomaly(
                        anomaly_id=str(uuid.uuid4()),
                        tenant_id=self.tenant_id,
                        metric_name=metric.name,
                        actual_value=metric.value,
                        expected_value=mean_value,
                        deviation_score=z_score,
                        severity="high" if z_score > 5 else "medium",
                        description=f"Anomalous {metric.name}: {metric.value:.2f} (expected ~{mean_value:.2f})"
                    )
                    
                    self.anomalies.append(anomaly)
                    
                    logger.warning(
                        "Anomaly detected",
                        tenant_id=self.tenant_id,
                        metric_name=metric.name,
                        z_score=z_score,
                        actual_value=metric.value,
                        expected_value=mean_value
                    )
            
        except Exception as e:
            logger.error(
                "Failed to detect anomalies",
                tenant_id=self.tenant_id,
                metric_name=metric.name,
                error=str(e)
            )
    
    async def _calculate_performance_metrics(
        self,
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Calcule les métriques de performance."""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_values = await self._get_metric_values("cpu_usage", *time_range)
            if cpu_values:
                metrics["cpu"] = {
                    "avg": statistics.mean(cpu_values),
                    "max": max(cpu_values),
                    "min": min(cpu_values),
                    "p95": np.percentile(cpu_values, 95)
                }
            
            # Memory metrics
            memory_values = await self._get_metric_values("memory_usage", *time_range)
            if memory_values:
                metrics["memory"] = {
                    "avg": statistics.mean(memory_values),
                    "max": max(memory_values),
                    "min": min(memory_values),
                    "p95": np.percentile(memory_values, 95)
                }
            
            # Response time metrics
            response_time_values = await self._get_metric_values("response_time", *time_range)
            if response_time_values:
                metrics["response_time"] = {
                    "avg": statistics.mean(response_time_values),
                    "max": max(response_time_values),
                    "min": min(response_time_values),
                    "p95": np.percentile(response_time_values, 95),
                    "p99": np.percentile(response_time_values, 99)
                }
            
        except Exception as e:
            logger.error(
                "Failed to calculate performance metrics",
                tenant_id=self.tenant_id,
                error=str(e)
            )
        
        return metrics
    
    async def _calculate_usage_metrics(
        self,
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Calcule les métriques d'usage."""
        metrics = {}
        
        try:
            # Request rate
            request_values = await self._get_metric_values("request_rate", *time_range)
            if request_values:
                metrics["requests"] = {
                    "total": sum(request_values),
                    "avg_rate": statistics.mean(request_values),
                    "peak_rate": max(request_values)
                }
            
            # Active users
            user_values = await self._get_metric_values("active_users", *time_range)
            if user_values:
                metrics["users"] = {
                    "avg": statistics.mean(user_values),
                    "peak": max(user_values)
                }
            
        except Exception as e:
            logger.error(
                "Failed to calculate usage metrics",
                tenant_id=self.tenant_id,
                error=str(e)
            )
        
        return metrics
    
    async def _calculate_cost_metrics(
        self,
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Calcule les métriques de coût."""
        metrics = {}
        
        try:
            # Coût par heure
            cost_values = await self._get_metric_values("cost_per_hour", *time_range)
            if cost_values:
                duration_hours = (time_range[1] - time_range[0]).total_seconds() / 3600
                metrics["total_cost"] = sum(cost_values)
                metrics["avg_cost_per_hour"] = statistics.mean(cost_values)
                metrics["projected_monthly"] = statistics.mean(cost_values) * 24 * 30
            
        except Exception as e:
            logger.error(
                "Failed to calculate cost metrics",
                tenant_id=self.tenant_id,
                error=str(e)
            )
        
        return metrics
    
    async def _calculate_security_metrics(
        self,
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Calcule les métriques de sécurité."""
        metrics = {}
        
        try:
            # Taux d'erreur
            error_values = await self._get_metric_values("error_rate", *time_range)
            if error_values:
                metrics["error_rate"] = {
                    "avg": statistics.mean(error_values),
                    "max": max(error_values)
                }
            
            # Violations de sécurité
            security_violations = [a for a in self.anomalies if "security" in a.metric_name.lower()]
            metrics["security_violations"] = len(security_violations)
            
        except Exception as e:
            logger.error(
                "Failed to calculate security metrics",
                tenant_id=self.tenant_id,
                error=str(e)
            )
        
        return metrics
    
    async def _calculate_trends(
        self,
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Calcule les tendances."""
        trends = {}
        
        try:
            key_metrics = ["cpu_usage", "memory_usage", "request_rate", "response_time"]
            
            for metric_name in key_metrics:
                values = await self._get_metric_values(metric_name, *time_range)
                if values and len(values) > 1:
                    # Calculer la tendance simple (pente)
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    
                    trends[metric_name] = {
                        "trend": "increasing" if slope > 0 else "decreasing",
                        "slope": slope,
                        "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                    }
            
        except Exception as e:
            logger.error(
                "Failed to calculate trends",
                tenant_id=self.tenant_id,
                error=str(e)
            )
        
        return trends
    
    async def _get_recent_anomalies(
        self,
        time_range: Tuple[datetime, datetime]
    ) -> List[Dict[str, Any]]:
        """Récupère les anomalies récentes."""
        recent_anomalies = [
            {
                "anomaly_id": a.anomaly_id,
                "metric_name": a.metric_name,
                "actual_value": a.actual_value,
                "expected_value": a.expected_value,
                "deviation_score": a.deviation_score,
                "severity": a.severity,
                "detected_at": a.detected_at.isoformat()
            }
            for a in self.anomalies
            if time_range[0] <= a.detected_at <= time_range[1] and not a.resolved
        ]
        
        return recent_anomalies
    
    async def _get_top_insights(self, limit: int) -> List[Dict[str, Any]]:
        """Récupère les top insights."""
        # Trier par importance et récence
        sorted_insights = sorted(
            self.insights_cache,
            key=lambda x: (x.importance == "high", x.created_at),
            reverse=True
        )
        
        return [
            {
                "insight_id": i.insight_id,
                "title": i.title,
                "description": i.description,
                "category": i.category,
                "importance": i.importance,
                "created_at": i.created_at.isoformat()
            }
            for i in sorted_insights[:limit]
        ]
    
    async def _calculate_health_score(self) -> float:
        """Calcule un score de santé global."""
        try:
            score = 100.0
            
            # Pénalités pour anomalies
            active_anomalies = [a for a in self.anomalies if not a.resolved]
            for anomaly in active_anomalies:
                if anomaly.severity == "high":
                    score -= 20
                elif anomaly.severity == "medium":
                    score -= 10
                else:
                    score -= 5
            
            # Bonus pour performance stable
            baseline = await self.get_performance_baseline()
            if baseline:
                # Logique de bonus basée sur la stabilité
                pass
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(
                "Failed to calculate health score",
                tenant_id=self.tenant_id,
                error=str(e)
            )
            return 50.0  # Score neutre en cas d'erreur
    
    async def _get_metric_values(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[float]:
        """Récupère les valeurs d'une métrique dans une plage de temps."""
        if metric_name not in self.metrics_buffer:
            return []
        
        metrics = list(self.metrics_buffer[metric_name])
        filtered_metrics = [
            m.value for m in metrics
            if start_time <= m.timestamp <= end_time
        ]
        
        return filtered_metrics
    
    async def _predict_metric(
        self,
        metric_name: str,
        horizon: PredictionHorizon
    ) -> Optional[Prediction]:
        """Prédit une métrique."""
        try:
            # Récupérer l'historique
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)  # 7 jours d'historique
            
            values = await self._get_metric_values(metric_name, start_time, end_time)
            
            if len(values) < 10:  # Pas assez de données
                return None
            
            # Prédiction simple basée sur la tendance (remplacer par ML en production)
            x = np.arange(len(values))
            
            # Régression linéaire simple
            slope, intercept = np.polyfit(x, values, 1)
            
            # Prédire pour l'horizon donné
            horizon_hours = {
                PredictionHorizon.SHORT_TERM: 1,
                PredictionHorizon.MEDIUM_TERM: 24,
                PredictionHorizon.LONG_TERM: 168,  # 7 jours
                PredictionHorizon.STRATEGIC: 720   # 30 jours
            }
            
            future_point = len(values) + horizon_hours[horizon]
            predicted_value = slope * future_point + intercept
            
            # Calculer la confiance basée sur la variance
            residuals = [values[i] - (slope * i + intercept) for i in range(len(values))]
            mse = statistics.mean([r**2 for r in residuals])
            confidence = max(0.1, min(0.9, 1.0 - (mse / statistics.variance(values))))
            
            prediction = Prediction(
                prediction_id=str(uuid.uuid4()),
                tenant_id=self.tenant_id,
                metric_name=metric_name,
                predicted_value=max(0, predicted_value),  # Pas de valeurs négatives
                confidence_score=confidence,
                horizon=horizon,
                factors=["historical_trend", "linear_regression"],
                model_version="simple_v1.0",
                valid_until=datetime.utcnow() + timedelta(hours=horizon_hours[horizon])
            )
            
            return prediction
            
        except Exception as e:
            logger.error(
                "Failed to predict metric",
                tenant_id=self.tenant_id,
                metric_name=metric_name,
                error=str(e)
            )
            return None
    
    async def _analyze_performance_patterns(self) -> List[TenantInsight]:
        """Analyse les patterns de performance."""
        insights = []
        
        try:
            # Analyser les pics de CPU
            cpu_values = await self._get_metric_values(
                "cpu_usage",
                datetime.utcnow() - timedelta(days=7),
                datetime.utcnow()
            )
            
            if cpu_values and statistics.mean(cpu_values) > 80:
                insights.append(TenantInsight(
                    insight_id=str(uuid.uuid4()),
                    tenant_id=self.tenant_id,
                    title="High CPU Usage Detected",
                    description=f"Average CPU usage is {statistics.mean(cpu_values):.1f}%, which is above the recommended threshold of 80%",
                    category="performance",
                    importance="high",
                    data_points=[{"metric": "cpu_usage", "avg": statistics.mean(cpu_values)}],
                    recommendations=[
                        "Consider scaling up resources",
                        "Optimize application code",
                        "Review resource allocation"
                    ]
                ))
            
        except Exception as e:
            logger.error(
                "Failed to analyze performance patterns",
                tenant_id=self.tenant_id,
                error=str(e)
            )
        
        return insights
    
    async def _analyze_cost_patterns(self) -> List[TenantInsight]:
        """Analyse les patterns de coût."""
        insights = []
        
        try:
            # Analyser l'évolution des coûts
            cost_values = await self._get_metric_values(
                "cost_per_hour",
                datetime.utcnow() - timedelta(days=30),
                datetime.utcnow()
            )
            
            if cost_values and len(cost_values) > 7:
                recent_avg = statistics.mean(cost_values[-7:])
                older_avg = statistics.mean(cost_values[:7])
                
                if recent_avg > older_avg * 1.2:  # 20% d'augmentation
                    insights.append(TenantInsight(
                        insight_id=str(uuid.uuid4()),
                        tenant_id=self.tenant_id,
                        title="Cost Increase Detected",
                        description=f"Costs have increased by {((recent_avg - older_avg) / older_avg * 100):.1f}% in the last week",
                        category="cost",
                        importance="medium",
                        data_points=[
                            {"metric": "recent_cost", "value": recent_avg},
                            {"metric": "previous_cost", "value": older_avg}
                        ],
                        recommendations=[
                            "Review resource utilization",
                            "Optimize autoscaling settings",
                            "Consider reserved instances"
                        ]
                    ))
            
        except Exception as e:
            logger.error(
                "Failed to analyze cost patterns",
                tenant_id=self.tenant_id,
                error=str(e)
            )
        
        return insights
    
    async def _analyze_security_patterns(self) -> List[TenantInsight]:
        """Analyse les patterns de sécurité."""
        insights = []
        
        try:
            # Analyser les anomalies de sécurité
            security_anomalies = [
                a for a in self.anomalies
                if "security" in a.metric_name.lower() or "error" in a.metric_name.lower()
            ]
            
            if len(security_anomalies) > 5:  # Plus de 5 anomalies de sécurité
                insights.append(TenantInsight(
                    insight_id=str(uuid.uuid4()),
                    tenant_id=self.tenant_id,
                    title="Security Anomalies Detected",
                    description=f"Detected {len(security_anomalies)} security-related anomalies in recent activity",
                    category="security",
                    importance="high",
                    data_points=[{"anomalies_count": len(security_anomalies)}],
                    recommendations=[
                        "Review access logs",
                        "Check for suspicious activities",
                        "Verify security policies"
                    ]
                ))
            
        except Exception as e:
            logger.error(
                "Failed to analyze security patterns",
                tenant_id=self.tenant_id,
                error=str(e)
            )
        
        return insights
    
    async def _analyze_usage_patterns(self) -> List[TenantInsight]:
        """Analyse les patterns d'usage."""
        insights = []
        
        try:
            # Analyser les patterns d'utilisation
            request_values = await self._get_metric_values(
                "request_rate",
                datetime.utcnow() - timedelta(days=7),
                datetime.utcnow()
            )
            
            if request_values:
                # Détecter les pics d'usage
                avg_requests = statistics.mean(request_values)
                max_requests = max(request_values)
                
                if max_requests > avg_requests * 3:  # Pic 3x supérieur à la moyenne
                    insights.append(TenantInsight(
                        insight_id=str(uuid.uuid4()),
                        tenant_id=self.tenant_id,
                        title="Traffic Spike Pattern Detected",
                        description=f"Peak traffic ({max_requests:.1f} req/s) is {(max_requests/avg_requests):.1f}x higher than average ({avg_requests:.1f} req/s)",
                        category="usage",
                        importance="medium",
                        data_points=[
                            {"metric": "avg_requests", "value": avg_requests},
                            {"metric": "peak_requests", "value": max_requests}
                        ],
                        recommendations=[
                            "Consider burst scaling configuration",
                            "Implement rate limiting",
                            "Optimize for peak loads"
                        ]
                    ))
            
        except Exception as e:
            logger.error(
                "Failed to analyze usage patterns",
                tenant_id=self.tenant_id,
                error=str(e)
            )
        
        return insights


class PredictiveScaler:
    """
    Scaler prédictif avec ML pour anticipation des besoins.
    
    Fonctionnalités:
    - Prédictions de charge multi-horizon
    - Modèles ML adaptatifs
    - Scaling proactif intelligent
    - Optimisation continue
    """
    
    def __init__(self, tenant_analytics: TenantAnalytics):
        self.tenant_analytics = tenant_analytics
        self.ml_models = {}
        self.prediction_cache = {}
        self.scaling_history = []
        
        # Configuration
        self.prediction_accuracy_threshold = 0.7
        self.scaling_confidence_threshold = 0.8
        self.model_retrain_interval = timedelta(days=7)
        
        logger.info("PredictiveScaler initialized", tenant_id=tenant_analytics.tenant_id)
    
    async def predict_scaling_needs(
        self,
        horizon: PredictionHorizon = PredictionHorizon.MEDIUM_TERM
    ) -> Dict[str, Any]:
        """Prédit les besoins de scaling."""
        try:
            tenant_id = self.tenant_analytics.tenant_id
            
            # Prédire les ressources nécessaires
            resource_predictions = await self.tenant_analytics.predict_resource_needs(horizon)
            
            # Analyser les prédictions pour déterminer les actions de scaling
            scaling_recommendations = {}
            
            for resource_type, prediction in resource_predictions.items():
                if prediction.confidence_score >= self.scaling_confidence_threshold:
                    current_usage = await self._get_current_resource_usage(resource_type)
                    
                    if current_usage:
                        scaling_factor = prediction.predicted_value / current_usage
                        
                        if scaling_factor > 1.2:  # Besoin d'augmenter de 20%+
                            scaling_recommendations[resource_type] = {
                                "action": "scale_up",
                                "current_usage": current_usage,
                                "predicted_usage": prediction.predicted_value,
                                "scaling_factor": scaling_factor,
                                "confidence": prediction.confidence_score,
                                "recommendation": f"Scale up {resource_type} by {(scaling_factor-1)*100:.1f}%"
                            }
                        elif scaling_factor < 0.8:  # Peut diminuer de 20%+
                            scaling_recommendations[resource_type] = {
                                "action": "scale_down",
                                "current_usage": current_usage,
                                "predicted_usage": prediction.predicted_value,
                                "scaling_factor": scaling_factor,
                                "confidence": prediction.confidence_score,
                                "recommendation": f"Scale down {resource_type} by {(1-scaling_factor)*100:.1f}%"
                            }
            
            result = {
                "tenant_id": tenant_id,
                "horizon": horizon.value,
                "predictions": resource_predictions,
                "scaling_recommendations": scaling_recommendations,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(
                "Scaling needs predicted",
                tenant_id=tenant_id,
                horizon=horizon.value,
                recommendations_count=len(scaling_recommendations)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to predict scaling needs",
                tenant_id=self.tenant_analytics.tenant_id,
                error=str(e)
            )
            return {}
    
    async def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Génère des recommandations d'optimisation."""
        try:
            recommendations = []
            tenant_id = self.tenant_analytics.tenant_id
            
            # Analyser l'historique des métriques
            baseline = await self.tenant_analytics.get_performance_baseline()
            
            # Recommandations de performance
            if "cpu_usage" in baseline:
                cpu_baseline = baseline["cpu_usage"]
                if cpu_baseline["avg"] > 80:
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        tenant_id=tenant_id,
                        category="performance",
                        title="High CPU Usage Optimization",
                        description=f"Average CPU usage ({cpu_baseline['avg']:.1f}%) is above optimal range",
                        impact="high",
                        effort="medium",
                        estimated_savings=None,
                        implementation_steps=[
                            "Scale up CPU resources",
                            "Optimize application code",
                            "Implement CPU-efficient algorithms",
                            "Consider distributed processing"
                        ],
                        priority_score=8.5,
                        confidence=0.85
                    ))
            
            # Recommandations de coût
            scaling_predictions = await self.predict_scaling_needs(PredictionHorizon.LONG_TERM)
            if scaling_predictions.get("scaling_recommendations"):
                for resource, rec in scaling_predictions["scaling_recommendations"].items():
                    if rec["action"] == "scale_down":
                        estimated_savings = rec["current_usage"] * 0.1 * 24 * 30  # Estimation
                        recommendations.append(OptimizationRecommendation(
                            recommendation_id=str(uuid.uuid4()),
                            tenant_id=tenant_id,
                            category="cost",
                            title=f"Cost Optimization for {resource.upper()}",
                            description=f"Predicted opportunity to reduce {resource} usage by {(1-rec['scaling_factor'])*100:.1f}%",
                            impact="medium",
                            effort="low",
                            estimated_savings=estimated_savings,
                            implementation_steps=[
                                f"Implement gradual {resource} scaling down",
                                "Monitor performance impact",
                                "Set up automated scaling policies"
                            ],
                            priority_score=7.0,
                            confidence=rec["confidence"]
                        ))
            
            # Trier par score de priorité
            recommendations.sort(key=lambda x: x.priority_score, reverse=True)
            
            logger.info(
                "Optimization recommendations generated",
                tenant_id=tenant_id,
                count=len(recommendations)
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(
                "Failed to generate optimization recommendations",
                tenant_id=self.tenant_analytics.tenant_id,
                error=str(e)
            )
            return []
    
    async def _get_current_resource_usage(self, resource_type: str) -> Optional[float]:
        """Récupère l'usage actuel d'une ressource."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)  # Dernière heure
            
            metric_name = f"{resource_type}_usage"
            values = await self.tenant_analytics._get_metric_values(metric_name, start_time, end_time)
            
            if values:
                return statistics.mean(values)
            
            return None
            
        except Exception as e:
            logger.error(
                "Failed to get current resource usage",
                resource_type=resource_type,
                error=str(e)
            )
            return None
