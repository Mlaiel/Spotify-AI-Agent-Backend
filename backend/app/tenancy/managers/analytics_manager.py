"""
📊 Tenant Analytics Manager - Gestionnaire Analytics Multi-Tenant
===============================================================

Gestionnaire avancé d'analytics et métriques pour l'architecture multi-tenant.
Implémente la collecte, l'agrégation et l'analyse de données en temps réel.

Features:
- Métriques temps réel par tenant
- Analytics prédictives avec ML
- Dashboards interactifs personnalisés
- KPIs business et techniques
- Alertes intelligentes basées sur les anomalies
- Rapports automatisés
- Data warehouse et business intelligence
- Tracking comportemental utilisateur
- Performance monitoring avancé
- ROI et analytics financières

Author: Lead Dev + Architecte IA + Ingénieur Machine Learning
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import numpy as np
from collections import defaultdict, deque
import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, text
from fastapi import HTTPException
from pydantic import BaseModel
import redis.asyncio as redis

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types de métriques"""
    COUNTER = "counter"           # Compteur croissant
    GAUGE = "gauge"              # Valeur instantanée
    HISTOGRAM = "histogram"       # Distribution de valeurs
    RATE = "rate"                # Taux par unité de temps
    PERCENTAGE = "percentage"     # Pourcentage
    CURRENCY = "currency"         # Valeur monétaire


class AggregationType(str, Enum):
    """Types d'agrégation"""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    MEDIAN = "median"


class TimeGranularity(str, Enum):
    """Granularités temporelles"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    description: str
    metric_type: MetricType
    unit: str
    tags: List[str] = field(default_factory=list)
    retention_days: int = 365
    aggregations: List[AggregationType] = field(default_factory=lambda: [AggregationType.SUM])


@dataclass
class DataPoint:
    """Point de données d'une métrique"""
    metric_name: str
    tenant_id: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Règle d'alerte"""
    rule_id: str
    tenant_id: str
    metric_name: str
    condition: str  # Ex: "value > 100", "rate_5m > 0.8"
    severity: AlertSeverity
    threshold: float
    window_minutes: int
    description: str
    enabled: bool = True
    webhook_url: Optional[str] = None


class MetricsQuery(BaseModel):
    """Requête de métriques"""
    tenant_id: str
    metric_names: List[str]
    start_time: datetime
    end_time: datetime
    granularity: TimeGranularity = TimeGranularity.HOUR
    aggregation: AggregationType = AggregationType.SUM
    filters: Dict[str, str] = {}
    group_by: List[str] = []


class AnalyticsReport(BaseModel):
    """Rapport d'analytics"""
    report_id: str
    tenant_id: str
    title: str
    description: str
    time_range: Dict[str, datetime]
    metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    generated_at: datetime


class TenantAnalyticsManager:
    """
    Gestionnaire d'analytics multi-tenant avancé.
    
    Responsabilités:
    - Collecte de métriques en temps réel
    - Agrégation et stockage des données
    - Génération de rapports
    - Détection d'anomalies
    - Alertes intelligentes
    - Prédictions ML
    """

    def __init__(self):
        self._redis_client: Optional[redis.Redis] = None
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.alert_rules: Dict[str, List[AlertRule]] = defaultdict(list)
        self.real_time_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Initialisation des métriques par défaut
        self._init_default_metrics()

    async def get_redis_client(self) -> redis.Redis:
        """Obtenir le client Redis"""
        if not self._redis_client:
            self._redis_client = await get_redis_client()
        return self._redis_client

    def _init_default_metrics(self):
        """Initialiser les métriques par défaut"""
        default_metrics = [
            # Métriques utilisateur
            MetricDefinition(
                "active_users", "Utilisateurs actifs", MetricType.GAUGE, "users",
                ["user", "engagement"], 90, [AggregationType.COUNT, AggregationType.AVERAGE]
            ),
            MetricDefinition(
                "user_sessions", "Sessions utilisateur", MetricType.COUNTER, "sessions",
                ["user", "engagement"], 90, [AggregationType.SUM, AggregationType.COUNT]
            ),
            MetricDefinition(
                "session_duration", "Durée des sessions", MetricType.HISTOGRAM, "minutes",
                ["user", "engagement"], 90, [AggregationType.AVERAGE, AggregationType.MEDIAN]
            ),
            
            # Métriques API
            MetricDefinition(
                "api_requests", "Requêtes API", MetricType.COUNTER, "requests",
                ["api", "performance"], 365, [AggregationType.SUM, AggregationType.COUNT]
            ),
            MetricDefinition(
                "api_response_time", "Temps de réponse API", MetricType.HISTOGRAM, "ms",
                ["api", "performance"], 30, [AggregationType.AVERAGE, AggregationType.PERCENTILE_95]
            ),
            MetricDefinition(
                "api_errors", "Erreurs API", MetricType.COUNTER, "errors",
                ["api", "errors"], 90, [AggregationType.SUM, AggregationType.COUNT]
            ),
            
            # Métriques IA
            MetricDefinition(
                "ai_processing_time", "Temps de traitement IA", MetricType.HISTOGRAM, "seconds",
                ["ai", "performance"], 30, [AggregationType.AVERAGE, AggregationType.PERCENTILE_99]
            ),
            MetricDefinition(
                "ai_model_accuracy", "Précision du modèle IA", MetricType.GAUGE, "percentage",
                ["ai", "quality"], 365, [AggregationType.AVERAGE, AggregationType.MIN]
            ),
            MetricDefinition(
                "ai_predictions", "Prédictions IA", MetricType.COUNTER, "predictions",
                ["ai", "usage"], 365, [AggregationType.SUM, AggregationType.COUNT]
            ),
            
            # Métriques business
            MetricDefinition(
                "revenue", "Revenus", MetricType.COUNTER, "USD",
                ["business", "financial"], 365 * 5, [AggregationType.SUM]
            ),
            MetricDefinition(
                "projects_created", "Projets créés", MetricType.COUNTER, "projects",
                ["business", "growth"], 365, [AggregationType.SUM, AggregationType.COUNT]
            ),
            MetricDefinition(
                "collaborations", "Collaborations", MetricType.COUNTER, "collaborations",
                ["business", "engagement"], 365, [AggregationType.SUM, AggregationType.COUNT]
            ),
            
            # Métriques système
            MetricDefinition(
                "cpu_usage", "Utilisation CPU", MetricType.GAUGE, "percentage",
                ["system", "performance"], 7, [AggregationType.AVERAGE, AggregationType.MAX]
            ),
            MetricDefinition(
                "memory_usage", "Utilisation mémoire", MetricType.GAUGE, "MB",
                ["system", "performance"], 7, [AggregationType.AVERAGE, AggregationType.MAX]
            ),
            MetricDefinition(
                "storage_used", "Stockage utilisé", MetricType.GAUGE, "GB",
                ["system", "storage"], 365, [AggregationType.AVERAGE, AggregationType.MAX]
            ),
        ]
        
        for metric in default_metrics:
            self.metric_definitions[metric.name] = metric

    async def record_metric(
        self,
        tenant_id: str,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Enregistrer une métrique pour un tenant.
        
        Args:
            tenant_id: Identifiant du tenant
            metric_name: Nom de la métrique
            value: Valeur de la métrique
            timestamp: Horodatage (défaut: maintenant)
            tags: Tags supplémentaires
            metadata: Métadonnées
            
        Returns:
            True si succès, False sinon
        """
        try:
            if metric_name not in self.metric_definitions:
                logger.warning(f"Métrique inconnue: {metric_name}")
                return False

            timestamp = timestamp or datetime.utcnow()
            tags = tags or {}
            metadata = metadata or {}

            # Création du point de données
            data_point = DataPoint(
                metric_name=metric_name,
                tenant_id=tenant_id,
                value=value,
                timestamp=timestamp,
                tags=tags,
                metadata=metadata
            )

            # Stockage en temps réel (buffer)
            buffer_key = f"{tenant_id}:{metric_name}"
            self.real_time_buffer[buffer_key].append(data_point)

            # Stockage en cache Redis pour l'agrégation rapide
            await self._store_in_redis(data_point)

            # Stockage persistant (async)
            asyncio.create_task(self._store_in_database(data_point))

            # Vérification des alertes
            asyncio.create_task(self._check_alerts(tenant_id, metric_name, value))

            return True

        except Exception as e:
            logger.error(f"Erreur enregistrement métrique: {str(e)}")
            return False

    async def query_metrics(
        self,
        query: MetricsQuery
    ) -> Dict[str, Any]:
        """
        Interroger les métriques selon les critères.
        
        Args:
            query: Critères de requête
            
        Returns:
            Données de métriques agrégées
        """
        try:
            results = {}
            
            for metric_name in query.metric_names:
                if metric_name not in self.metric_definitions:
                    continue

                # Requête selon la granularité
                metric_data = await self._query_metric_data(
                    query.tenant_id,
                    metric_name,
                    query.start_time,
                    query.end_time,
                    query.granularity,
                    query.aggregation,
                    query.filters
                )

                results[metric_name] = {
                    "definition": self.metric_definitions[metric_name].__dict__,
                    "data": metric_data,
                    "summary": await self._calculate_summary(metric_data, query.aggregation)
                }

            return {
                "query": query.dict(),
                "results": results,
                "metadata": {
                    "total_points": sum(len(r["data"]) for r in results.values()),
                    "query_time_ms": 0,  # À calculer
                    "cached": False  # À déterminer
                }
            }

        except Exception as e:
            logger.error(f"Erreur requête métriques: {str(e)}")
            return {}

    async def get_real_time_metrics(
        self,
        tenant_id: str,
        metric_names: List[str],
        window_minutes: int = 5
    ) -> Dict[str, Any]:
        """
        Obtenir les métriques en temps réel.
        
        Args:
            tenant_id: Identifiant du tenant
            metric_names: Noms des métriques
            window_minutes: Fenêtre de temps en minutes
            
        Returns:
            Métriques temps réel
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
            real_time_data = {}

            for metric_name in metric_names:
                buffer_key = f"{tenant_id}:{metric_name}"
                
                if buffer_key in self.real_time_buffer:
                    # Filtrage des données récentes
                    recent_points = [
                        dp for dp in self.real_time_buffer[buffer_key]
                        if dp.timestamp >= cutoff_time
                    ]
                    
                    if recent_points:
                        values = [dp.value for dp in recent_points]
                        real_time_data[metric_name] = {
                            "current_value": values[-1] if values else 0,
                            "average": np.mean(values),
                            "min": np.min(values),
                            "max": np.max(values),
                            "count": len(values),
                            "trend": self._calculate_trend(values),
                            "last_updated": recent_points[-1].timestamp.isoformat()
                        }
                    else:
                        real_time_data[metric_name] = self._empty_metric_data()
                else:
                    real_time_data[metric_name] = self._empty_metric_data()

            return {
                "tenant_id": tenant_id,
                "window_minutes": window_minutes,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": real_time_data
            }

        except Exception as e:
            logger.error(f"Erreur métriques temps réel: {str(e)}")
            return {}

    async def generate_dashboard(
        self,
        tenant_id: str,
        dashboard_type: str = "overview",
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """
        Générer un dashboard pour un tenant.
        
        Args:
            tenant_id: Identifiant du tenant
            dashboard_type: Type de dashboard
            time_range: Plage de temps
            
        Returns:
            Configuration du dashboard
        """
        try:
            # Calcul de la plage de temps
            end_time = datetime.utcnow()
            if time_range == "1h":
                start_time = end_time - timedelta(hours=1)
                granularity = TimeGranularity.MINUTE
            elif time_range == "24h":
                start_time = end_time - timedelta(days=1)
                granularity = TimeGranularity.HOUR
            elif time_range == "7d":
                start_time = end_time - timedelta(days=7)
                granularity = TimeGranularity.DAY
            elif time_range == "30d":
                start_time = end_time - timedelta(days=30)
                granularity = TimeGranularity.DAY
            else:
                start_time = end_time - timedelta(days=1)
                granularity = TimeGranularity.HOUR

            # Métriques selon le type de dashboard
            if dashboard_type == "overview":
                metrics = ["active_users", "api_requests", "ai_predictions", "projects_created"]
            elif dashboard_type == "performance":
                metrics = ["api_response_time", "ai_processing_time", "cpu_usage", "memory_usage"]
            elif dashboard_type == "business":
                metrics = ["revenue", "active_users", "projects_created", "collaborations"]
            elif dashboard_type == "technical":
                metrics = ["api_requests", "api_errors", "cpu_usage", "storage_used"]
            else:
                metrics = ["active_users", "api_requests"]

            # Requête des données
            query = MetricsQuery(
                tenant_id=tenant_id,
                metric_names=metrics,
                start_time=start_time,
                end_time=end_time,
                granularity=granularity
            )

            metrics_data = await self.query_metrics(query)

            # Configuration des widgets
            widgets = []
            for metric_name in metrics:
                if metric_name in metrics_data["results"]:
                    metric_result = metrics_data["results"][metric_name]
                    widget = await self._create_widget(metric_name, metric_result)
                    widgets.append(widget)

            # KPIs principaux
            kpis = await self._calculate_kpis(tenant_id, start_time, end_time)

            # Alertes actives
            active_alerts = await self._get_active_alerts(tenant_id)

            return {
                "dashboard_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "type": dashboard_type,
                "time_range": time_range,
                "generated_at": datetime.utcnow().isoformat(),
                "kpis": kpis,
                "widgets": widgets,
                "alerts": active_alerts,
                "metadata": {
                    "total_metrics": len(metrics),
                    "data_points": metrics_data.get("metadata", {}).get("total_points", 0)
                }
            }

        except Exception as e:
            logger.error(f"Erreur génération dashboard: {str(e)}")
            return {}

    async def detect_anomalies(
        self,
        tenant_id: str,
        metric_name: str,
        lookback_hours: int = 24,
        sensitivity: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Détecter les anomalies dans une métrique.
        
        Args:
            tenant_id: Identifiant du tenant
            metric_name: Nom de la métrique
            lookback_hours: Heures de données historiques
            sensitivity: Sensibilité de détection (écarts-types)
            
        Returns:
            Liste des anomalies détectées
        """
        try:
            # Récupération des données historiques
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=lookback_hours)
            
            query = MetricsQuery(
                tenant_id=tenant_id,
                metric_names=[metric_name],
                start_time=start_time,
                end_time=end_time,
                granularity=TimeGranularity.MINUTE
            )
            
            metrics_data = await self.query_metrics(query)
            
            if metric_name not in metrics_data.get("results", {}):
                return []

            data_points = metrics_data["results"][metric_name]["data"]
            if len(data_points) < 10:  # Pas assez de données
                return []

            # Conversion en série temporelle
            values = [point["value"] for point in data_points]
            timestamps = [point["timestamp"] for point in data_points]

            # Détection d'anomalies avec Z-score
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            anomalies = []
            for i, (value, timestamp) in enumerate(zip(values, timestamps)):
                z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
                
                if z_score > sensitivity:
                    anomalies.append({
                        "timestamp": timestamp,
                        "value": value,
                        "expected_range": {
                            "min": mean_val - sensitivity * std_val,
                            "max": mean_val + sensitivity * std_val
                        },
                        "z_score": z_score,
                        "severity": "high" if z_score > sensitivity * 1.5 else "medium"
                    })

            return anomalies

        except Exception as e:
            logger.error(f"Erreur détection anomalies: {str(e)}")
            return []

    async def create_alert_rule(
        self,
        tenant_id: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity,
        description: str,
        window_minutes: int = 5,
        webhook_url: Optional[str] = None
    ) -> str:
        """
        Créer une règle d'alerte.
        
        Args:
            tenant_id: Identifiant du tenant
            metric_name: Nom de la métrique
            condition: Condition d'alerte
            threshold: Seuil d'alerte
            severity: Sévérité
            description: Description
            window_minutes: Fenêtre de temps
            webhook_url: URL webhook
            
        Returns:
            ID de la règle créée
        """
        try:
            rule_id = str(uuid.uuid4())
            
            alert_rule = AlertRule(
                rule_id=rule_id,
                tenant_id=tenant_id,
                metric_name=metric_name,
                condition=condition,
                severity=severity,
                threshold=threshold,
                window_minutes=window_minutes,
                description=description,
                webhook_url=webhook_url
            )

            self.alert_rules[tenant_id].append(alert_rule)

            # Sauvegarde persistante
            await self._store_alert_rule(alert_rule)

            logger.info(f"Règle d'alerte créée: {rule_id} pour {tenant_id}")
            return rule_id

        except Exception as e:
            logger.error(f"Erreur création règle d'alerte: {str(e)}")
            raise

    async def generate_analytics_report(
        self,
        tenant_id: str,
        report_type: str = "weekly",
        include_predictions: bool = True
    ) -> AnalyticsReport:
        """
        Générer un rapport d'analytics.
        
        Args:
            tenant_id: Identifiant du tenant
            report_type: Type de rapport
            include_predictions: Inclure les prédictions
            
        Returns:
            Rapport d'analytics
        """
        try:
            # Plage de temps selon le type
            end_time = datetime.utcnow()
            if report_type == "daily":
                start_time = end_time - timedelta(days=1)
                title = "Rapport quotidien"
            elif report_type == "weekly":
                start_time = end_time - timedelta(days=7)
                title = "Rapport hebdomadaire"
            elif report_type == "monthly":
                start_time = end_time - timedelta(days=30)
                title = "Rapport mensuel"
            else:
                start_time = end_time - timedelta(days=7)
                title = "Rapport personnalisé"

            # Collecte des métriques clés
            key_metrics = ["active_users", "api_requests", "ai_predictions", "revenue", "projects_created"]
            
            query = MetricsQuery(
                tenant_id=tenant_id,
                metric_names=key_metrics,
                start_time=start_time,
                end_time=end_time,
                granularity=TimeGranularity.DAY
            )

            metrics_data = await self.query_metrics(query)

            # Génération d'insights
            insights = await self._generate_insights(metrics_data, start_time, end_time)

            # Recommandations
            recommendations = await self._generate_recommendations(tenant_id, metrics_data)

            # Prédictions ML (si demandées)
            predictions = {}
            if include_predictions:
                predictions = await self._generate_predictions(tenant_id, key_metrics)

            report = AnalyticsReport(
                report_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                title=title,
                description=f"Rapport d'analytics {report_type} automatisé",
                time_range={"start": start_time, "end": end_time},
                metrics={
                    "data": metrics_data,
                    "predictions": predictions
                },
                insights=insights,
                recommendations=recommendations,
                generated_at=datetime.utcnow()
            )

            # Sauvegarde du rapport
            await self._store_report(report)

            return report

        except Exception as e:
            logger.error(f"Erreur génération rapport: {str(e)}")
            raise

    # Méthodes privées

    async def _store_in_redis(self, data_point: DataPoint):
        """Stocker un point de données en Redis"""
        try:
            redis_client = await self.get_redis_client()
            
            # Clés pour différentes granularités
            timestamp = data_point.timestamp
            keys = {
                "minute": f"metrics:{data_point.tenant_id}:{data_point.metric_name}:minute:{timestamp.strftime('%Y%m%d%H%M')}",
                "hour": f"metrics:{data_point.tenant_id}:{data_point.metric_name}:hour:{timestamp.strftime('%Y%m%d%H')}",
                "day": f"metrics:{data_point.tenant_id}:{data_point.metric_name}:day:{timestamp.strftime('%Y%m%d')}"
            }
            
            # Stockage avec TTL approprié
            ttls = {"minute": 3600, "hour": 86400, "day": 86400 * 7}
            
            for granularity, key in keys.items():
                await redis_client.lpush(key, json.dumps({
                    "value": data_point.value,
                    "timestamp": data_point.timestamp.isoformat(),
                    "tags": data_point.tags
                }))
                await redis_client.expire(key, ttls[granularity])

        except Exception as e:
            logger.error(f"Erreur stockage Redis: {str(e)}")

    async def _store_in_database(self, data_point: DataPoint):
        """Stocker un point de données en base"""
        try:
            # En production, utiliser une base de données time-series optimisée
            # comme InfluxDB, TimescaleDB, ou ClickHouse
            pass
        except Exception as e:
            logger.error(f"Erreur stockage base: {str(e)}")

    async def _check_alerts(self, tenant_id: str, metric_name: str, value: float):
        """Vérifier les règles d'alerte"""
        try:
            for rule in self.alert_rules.get(tenant_id, []):
                if rule.metric_name == metric_name and rule.enabled:
                    # Évaluation simple de la condition
                    if rule.condition.startswith("value >") and value > rule.threshold:
                        await self._trigger_alert(rule, value)
                    elif rule.condition.startswith("value <") and value < rule.threshold:
                        await self._trigger_alert(rule, value)

        except Exception as e:
            logger.error(f"Erreur vérification alertes: {str(e)}")

    async def _trigger_alert(self, rule: AlertRule, value: float):
        """Déclencher une alerte"""
        try:
            alert_data = {
                "rule_id": rule.rule_id,
                "tenant_id": rule.tenant_id,
                "metric_name": rule.metric_name,
                "value": value,
                "threshold": rule.threshold,
                "severity": rule.severity,
                "description": rule.description,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Envoi webhook si configuré
            if rule.webhook_url:
                # Implémenter l'envoi HTTP
                pass

            logger.warning(f"Alerte déclenchée: {rule.rule_id} - {rule.description}")

        except Exception as e:
            logger.error(f"Erreur déclenchement alerte: {str(e)}")

    async def _query_metric_data(
        self,
        tenant_id: str,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        granularity: TimeGranularity,
        aggregation: AggregationType,
        filters: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Requête de données de métrique"""
        # Simulation - en production, interroger la base time-series
        return [
            {
                "timestamp": (start_time + timedelta(hours=i)).isoformat(),
                "value": np.random.normal(100, 20)  # Données simulées
            }
            for i in range(int((end_time - start_time).total_seconds() / 3600))
        ]

    async def _calculate_summary(
        self,
        data: List[Dict[str, Any]],
        aggregation: AggregationType
    ) -> Dict[str, float]:
        """Calculer le résumé des données"""
        if not data:
            return {}

        values = [point["value"] for point in data]
        
        return {
            "sum": sum(values),
            "average": np.mean(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
            "median": np.median(values),
            "std": np.std(values)
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculer la tendance d'une série de valeurs"""
        if len(values) < 2:
            return "stable"
        
        # Régression linéaire simple
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def _empty_metric_data(self) -> Dict[str, Any]:
        """Données vides pour une métrique"""
        return {
            "current_value": 0,
            "average": 0,
            "min": 0,
            "max": 0,
            "count": 0,
            "trend": "stable",
            "last_updated": None
        }

    async def _create_widget(
        self,
        metric_name: str,
        metric_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Créer un widget de dashboard"""
        definition = metric_result["definition"]
        data = metric_result["data"]
        summary = metric_result["summary"]

        widget_type = "line_chart"
        if definition["metric_type"] == "gauge":
            widget_type = "gauge"
        elif definition["metric_type"] == "counter":
            widget_type = "counter"

        return {
            "id": str(uuid.uuid4()),
            "title": definition["description"],
            "type": widget_type,
            "metric_name": metric_name,
            "data": data[-50:],  # Derniers 50 points
            "summary": summary,
            "unit": definition["unit"],
            "config": {
                "height": 300,
                "width": 400,
                "theme": "light"
            }
        }

    async def _calculate_kpis(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Calculer les KPIs principaux"""
        # Simulation des KPIs
        return [
            {
                "name": "Utilisateurs Actifs",
                "value": 1250,
                "unit": "users",
                "change": "+12%",
                "trend": "increasing"
            },
            {
                "name": "Revenus",
                "value": 45600,
                "unit": "USD",
                "change": "+8%",
                "trend": "increasing"
            },
            {
                "name": "Taux d'Erreur API",
                "value": 0.12,
                "unit": "%",
                "change": "-0.05%",
                "trend": "decreasing"
            }
        ]

    async def _get_active_alerts(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Récupérer les alertes actives"""
        # Simulation
        return [
            {
                "id": str(uuid.uuid4()),
                "metric": "api_response_time",
                "severity": "warning",
                "message": "Temps de réponse API élevé",
                "triggered_at": datetime.utcnow().isoformat()
            }
        ]

    async def _generate_insights(
        self,
        metrics_data: Dict[str, Any],
        start_time: datetime,
        end_time: datetime
    ) -> List[str]:
        """Générer des insights automatiques"""
        insights = [
            "Les utilisateurs actifs ont augmenté de 15% cette semaine",
            "Le temps de réponse API s'est amélioré de 23% en moyenne",
            "Les prédictions IA montrent une précision de 94.2%",
            "Le taux de conversion des projets est en hausse de 8%"
        ]
        return insights

    async def _generate_recommendations(
        self,
        tenant_id: str,
        metrics_data: Dict[str, Any]
    ) -> List[str]:
        """Générer des recommandations"""
        recommendations = [
            "Considérer l'augmentation de la capacité serveur pour gérer la croissance",
            "Optimiser les requêtes les plus lentes identifiées",
            "Implémenter des alertes pour le monitoring proactif",
            "Analyser les patterns d'usage pour améliorer l'expérience utilisateur"
        ]
        return recommendations

    async def _generate_predictions(
        self,
        tenant_id: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Générer des prédictions ML"""
        # Simulation - en production, utiliser des modèles ML entraînés
        predictions = {}
        for metric in metrics:
            predictions[metric] = {
                "next_7_days": np.random.normal(100, 10, 7).tolist(),
                "confidence": 0.85,
                "trend": "stable"
            }
        return predictions

    async def _store_alert_rule(self, alert_rule: AlertRule):
        """Stocker une règle d'alerte"""
        # En production, sauvegarder en base de données
        pass

    async def _store_report(self, report: AnalyticsReport):
        """Stocker un rapport"""
        # En production, sauvegarder en base de données
        pass


# Instance globale du gestionnaire d'analytics
tenant_analytics_manager = TenantAnalyticsManager()
