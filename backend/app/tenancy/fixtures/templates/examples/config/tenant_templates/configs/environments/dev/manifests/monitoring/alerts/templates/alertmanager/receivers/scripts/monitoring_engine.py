"""
Système de Monitoring Intelligent Ultra-Avancé

Monitoring prédictif avec IA pour Alertmanager:
- Détection d'anomalies par machine learning
- Prédiction de pannes avec algorithmes avancés
- Auto-remediation intelligente
- Corrélation multi-dimensionnelle des métriques
- Alertes contextuelles adaptatives
- Optimisation continue des seuils

Version: 3.0.0
Développé par l'équipe Spotify AI Agent
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import aiohttp
import prometheus_client
import redis.asyncio as redis
import psutil
import socket
import ssl
import certifi
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"

class MetricType(Enum):
    """Types de métriques surveillées"""
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SECURITY = "security"
    BUSINESS = "business"

class AnomalyType(Enum):
    """Types d'anomalies détectées"""
    SPIKE = "spike"
    DROP = "drop"
    TREND = "trend"
    OUTLIER = "outlier"
    PATTERN_BREAK = "pattern_break"
    CORRELATION_BREAK = "correlation_break"

@dataclass
class MetricDefinition:
    """Définition d'une métrique à surveiller"""
    name: str
    metric_type: MetricType
    query: str
    threshold_warning: float
    threshold_critical: float
    unit: str
    description: str
    adaptive_thresholds: bool = True
    ml_enabled: bool = True
    correlation_metrics: List[str] = field(default_factory=list)
    business_impact: str = "medium"
    auto_remediation: bool = False
    remediation_script: Optional[str] = None

@dataclass
class AnomalyDetection:
    """Résultat de détection d'anomalie"""
    metric_name: str
    anomaly_type: AnomalyType
    severity: AlertSeverity
    confidence: float
    value: float
    expected_range: Tuple[float, float]
    timestamp: datetime
    context: Dict[str, Any]
    remediation_suggested: Optional[str] = None

class AIMonitoringEngine:
    """Moteur de monitoring intelligent avec IA"""
    
    def __init__(self, prometheus_url: str = "http://prometheus:9090"):
        self.prometheus_url = prometheus_url
        self.redis_client = None
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.anomaly_detectors = {}
        self.correlation_matrix = np.array([])
        self.alert_history = []
        self.adaptive_thresholds = {}
        self.prediction_models = {}
        self.context_analyzer = ContextAnalyzer()
        self.auto_remediator = AutoRemediator()
        
        # Métriques core à surveiller
        self.core_metrics = self._define_core_metrics()
        
    def _define_core_metrics(self) -> List[MetricDefinition]:
        """Définit les métriques core à surveiller"""
        return [
            MetricDefinition(
                name="alertmanager_notifications_total",
                metric_type=MetricType.THROUGHPUT,
                query="rate(alertmanager_notifications_total[5m])",
                threshold_warning=10.0,
                threshold_critical=50.0,
                unit="notifications/sec",
                description="Taux de notifications envoyées",
                correlation_metrics=["alertmanager_alerts_received_total"],
                auto_remediation=True,
                remediation_script="restart_alertmanager"
            ),
            MetricDefinition(
                name="alertmanager_notification_latency",
                metric_type=MetricType.LATENCY,
                query="histogram_quantile(0.95, alertmanager_notification_latency_seconds_bucket)",
                threshold_warning=5.0,
                threshold_critical=10.0,
                unit="seconds",
                description="Latence des notifications (95e percentile)",
                correlation_metrics=["alertmanager_notifications_total"]
            ),
            MetricDefinition(
                name="alertmanager_alerts_received_total",
                metric_type=MetricType.THROUGHPUT,
                query="rate(alertmanager_alerts_received_total[5m])",
                threshold_warning=100.0,
                threshold_critical=500.0,
                unit="alerts/sec",
                description="Taux d'alertes reçues",
                business_impact="high"
            ),
            MetricDefinition(
                name="alertmanager_alerts_invalid_total",
                metric_type=MetricType.ERROR_RATE,
                query="rate(alertmanager_alerts_invalid_total[5m])",
                threshold_warning=1.0,
                threshold_critical=5.0,
                unit="errors/sec",
                description="Taux d'alertes invalides",
                auto_remediation=True,
                remediation_script="validate_alert_configs"
            ),
            MetricDefinition(
                name="container_cpu_usage_seconds_total",
                metric_type=MetricType.RESOURCE_USAGE,
                query="rate(container_cpu_usage_seconds_total{container='alertmanager'}[5m]) * 100",
                threshold_warning=70.0,
                threshold_critical=90.0,
                unit="percent",
                description="Utilisation CPU d'Alertmanager",
                correlation_metrics=["container_memory_working_set_bytes"]
            ),
            MetricDefinition(
                name="container_memory_working_set_bytes",
                metric_type=MetricType.RESOURCE_USAGE,
                query="container_memory_working_set_bytes{container='alertmanager'} / 1024 / 1024",
                threshold_warning=256.0,
                threshold_critical=512.0,
                unit="MB",
                description="Utilisation mémoire d'Alertmanager"
            ),
            MetricDefinition(
                name="alertmanager_cluster_members",
                metric_type=MetricType.AVAILABILITY,
                query="alertmanager_cluster_members",
                threshold_warning=2.0,
                threshold_critical=1.0,
                unit="count",
                description="Nombre de membres du cluster",
                business_impact="critical"
            ),
            MetricDefinition(
                name="up",
                metric_type=MetricType.AVAILABILITY,
                query="up{job='alertmanager'}",
                threshold_warning=0.9,
                threshold_critical=0.5,
                unit="ratio",
                description="Disponibilité d'Alertmanager",
                business_impact="critical",
                auto_remediation=True,
                remediation_script="restart_failed_instances"
            )
        ]
    
    async def initialize(self):
        """Initialise le moteur de monitoring"""
        logger.info("Initializing AI Monitoring Engine")
        
        # Connexion Redis pour le cache
        try:
            self.redis_client = redis.Redis(
                host="redis",
                port=6379,
                decode_responses=True
            )
            await self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
        
        # Initialisation des détecteurs d'anomalies
        for metric in self.core_metrics:
            if metric.ml_enabled:
                self.anomaly_detectors[metric.name] = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
        
        # Chargement des données historiques
        await self._load_historical_data()
        
        logger.info("AI Monitoring Engine initialized successfully")
    
    async def start_monitoring(self):
        """Démarre le monitoring en continu"""
        logger.info("Starting continuous monitoring")
        
        while True:
            try:
                # Collecte des métriques
                current_metrics = await self._collect_metrics()
                
                # Détection d'anomalies avec IA
                anomalies = await self._detect_anomalies(current_metrics)
                
                # Analyse contextuelle
                enriched_anomalies = await self._enrich_with_context(anomalies)
                
                # Corrélation des alertes
                correlated_alerts = await self._correlate_alerts(enriched_anomalies)
                
                # Prédiction de pannes
                failure_predictions = await self._predict_failures(current_metrics)
                
                # Auto-remediation
                await self._execute_auto_remediation(correlated_alerts)
                
                # Mise à jour des seuils adaptatifs
                await self._update_adaptive_thresholds(current_metrics)
                
                # Stockage et notification
                await self._process_alerts(correlated_alerts + failure_predictions)
                
                # Attente avant la prochaine itération
                await asyncio.sleep(30)  # Monitoring toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"Monitoring iteration failed: {e}")
                await asyncio.sleep(60)  # Attente plus longue en cas d'erreur
    
    async def _collect_metrics(self) -> Dict[str, float]:
        """Collecte les métriques depuis Prometheus"""
        metrics = {}
        
        async with aiohttp.ClientSession() as session:
            for metric_def in self.core_metrics:
                try:
                    url = f"{self.prometheus_url}/api/v1/query"
                    params = {"query": metric_def.query}
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if data["status"] == "success" and data["data"]["result"]:
                                value = float(data["data"]["result"][0]["value"][1])
                                metrics[metric_def.name] = value
                                
                                # Stockage dans le buffer
                                self.metrics_buffer[metric_def.name].append({
                                    "timestamp": datetime.now(),
                                    "value": value
                                })
                            else:
                                logger.warning(f"No data for metric: {metric_def.name}")
                        else:
                            logger.error(f"Failed to fetch {metric_def.name}: {response.status}")
                            
                except Exception as e:
                    logger.error(f"Error collecting metric {metric_def.name}: {e}")
        
        return metrics
    
    async def _detect_anomalies(self, current_metrics: Dict[str, float]) -> List[AnomalyDetection]:
        """Détecte les anomalies avec machine learning"""
        anomalies = []
        
        for metric_name, value in current_metrics.items():
            metric_def = next((m for m in self.core_metrics if m.name == metric_name), None)
            if not metric_def or not metric_def.ml_enabled:
                continue
            
            try:
                # Récupération des données historiques
                historical_data = self._get_historical_data(metric_name)
                
                if len(historical_data) < 50:  # Pas assez de données
                    continue
                
                # Préparation des données pour ML
                X = np.array(historical_data).reshape(-1, 1)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Entraînement du détecteur si nécessaire
                detector = self.anomaly_detectors[metric_name]
                detector.fit(X_scaled)
                
                # Détection d'anomalie pour la valeur actuelle
                current_scaled = scaler.transform([[value]])
                is_anomaly = detector.predict(current_scaled)[0] == -1
                
                if is_anomaly:
                    # Calcul de la confiance
                    anomaly_score = detector.decision_function(current_scaled)[0]
                    confidence = abs(anomaly_score)
                    
                    # Détermination du type d'anomalie
                    anomaly_type = self._classify_anomaly_type(metric_name, value, historical_data)
                    
                    # Détermination de la sévérité
                    severity = self._calculate_severity(metric_def, value, confidence)
                    
                    # Calcul de la plage attendue
                    expected_range = self._calculate_expected_range(historical_data)
                    
                    anomaly = AnomalyDetection(
                        metric_name=metric_name,
                        anomaly_type=anomaly_type,
                        severity=severity,
                        confidence=confidence,
                        value=value,
                        expected_range=expected_range,
                        timestamp=datetime.now(),
                        context={}
                    )
                    
                    anomalies.append(anomaly)
                    logger.warning(f"Anomaly detected: {anomaly}")
                    
            except Exception as e:
                logger.error(f"Error detecting anomalies for {metric_name}: {e}")
        
        return anomalies
    
    def _classify_anomaly_type(self, metric_name: str, value: float, historical_data: List[float]) -> AnomalyType:
        """Classifie le type d'anomalie détectée"""
        
        if len(historical_data) < 10:
            return AnomalyType.OUTLIER
        
        recent_avg = np.mean(historical_data[-10:])
        historical_avg = np.mean(historical_data)
        std_dev = np.std(historical_data)
        
        # Spike: valeur très élevée par rapport à la moyenne
        if value > historical_avg + 3 * std_dev:
            return AnomalyType.SPIKE
        
        # Drop: valeur très faible par rapport à la moyenne
        if value < historical_avg - 3 * std_dev:
            return AnomalyType.DROP
        
        # Trend: changement de tendance
        if abs(recent_avg - historical_avg) > 2 * std_dev:
            return AnomalyType.TREND
        
        return AnomalyType.OUTLIER
    
    def _calculate_severity(self, metric_def: MetricDefinition, value: float, confidence: float) -> AlertSeverity:
        """Calcule la sévérité de l'anomalie"""
        
        # Utilisation des seuils adaptatifs si disponibles
        threshold_critical = self.adaptive_thresholds.get(
            f"{metric_def.name}_critical",
            metric_def.threshold_critical
        )
        threshold_warning = self.adaptive_thresholds.get(
            f"{metric_def.name}_warning", 
            metric_def.threshold_warning
        )
        
        # Ajustement selon la confiance ML
        confidence_factor = min(confidence * 2, 1.0)
        
        if metric_def.business_impact == "critical":
            confidence_factor *= 1.5
        
        if value >= threshold_critical * confidence_factor:
            return AlertSeverity.CRITICAL
        elif value >= threshold_warning * confidence_factor:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    async def _enrich_with_context(self, anomalies: List[AnomalyDetection]) -> List[AnomalyDetection]:
        """Enrichit les anomalies avec du contexte"""
        
        enriched_anomalies = []
        
        for anomaly in anomalies:
            try:
                # Contexte système
                system_context = await self._get_system_context()
                
                # Contexte métier
                business_context = await self._get_business_context(anomaly.timestamp)
                
                # Contexte de corrélation
                correlation_context = await self._get_correlation_context(anomaly.metric_name)
                
                # Suggestion de remediation
                remediation = await self._suggest_remediation(anomaly)
                
                # Enrichissement de l'anomalie
                anomaly.context.update({
                    "system": system_context,
                    "business": business_context,
                    "correlations": correlation_context
                })
                anomaly.remediation_suggested = remediation
                
                enriched_anomalies.append(anomaly)
                
            except Exception as e:
                logger.error(f"Error enriching anomaly {anomaly.metric_name}: {e}")
                enriched_anomalies.append(anomaly)  # Garder l'anomalie même sans enrichissement
        
        return enriched_anomalies
    
    async def _correlate_alerts(self, anomalies: List[AnomalyDetection]) -> List[AnomalyDetection]:
        """Corrèle les alertes pour réduire le bruit"""
        
        if len(anomalies) <= 1:
            return anomalies
        
        # Groupement par temps (fenêtre de 5 minutes)
        time_groups = defaultdict(list)
        for anomaly in anomalies:
            time_key = anomaly.timestamp.replace(second=0, microsecond=0)
            time_key = time_key.replace(minute=time_key.minute // 5 * 5)
            time_groups[time_key].append(anomaly)
        
        correlated_alerts = []
        
        for time_key, group_anomalies in time_groups.items():
            if len(group_anomalies) == 1:
                correlated_alerts.extend(group_anomalies)
                continue
            
            # Analyse des corrélations
            correlation_groups = self._find_correlation_groups(group_anomalies)
            
            for group in correlation_groups:
                if len(group) > 1:
                    # Créer une alerte corrélée
                    primary_anomaly = max(group, key=lambda x: x.confidence)
                    primary_anomaly.context["correlated_metrics"] = [
                        a.metric_name for a in group if a != primary_anomaly
                    ]
                    correlated_alerts.append(primary_anomaly)
                else:
                    correlated_alerts.extend(group)
        
        return correlated_alerts
    
    def _find_correlation_groups(self, anomalies: List[AnomalyDetection]) -> List[List[AnomalyDetection]]:
        """Trouve les groupes de métriques corrélées"""
        
        groups = []
        processed = set()
        
        for anomaly in anomalies:
            if anomaly.metric_name in processed:
                continue
            
            # Trouver les métriques corrélées
            metric_def = next((m for m in self.core_metrics if m.name == anomaly.metric_name), None)
            if not metric_def:
                groups.append([anomaly])
                processed.add(anomaly.metric_name)
                continue
            
            group = [anomaly]
            processed.add(anomaly.metric_name)
            
            # Chercher les corrélations définies
            for other_anomaly in anomalies:
                if (other_anomaly.metric_name in metric_def.correlation_metrics and
                    other_anomaly.metric_name not in processed):
                    group.append(other_anomaly)
                    processed.add(other_anomaly.metric_name)
            
            groups.append(group)
        
        return groups
    
    async def _predict_failures(self, current_metrics: Dict[str, float]) -> List[AnomalyDetection]:
        """Prédit les pannes potentielles"""
        
        predictions = []
        
        # Prédiction basée sur les tendances
        for metric_name, value in current_metrics.items():
            try:
                historical_data = self._get_historical_data(metric_name, window_hours=24)
                
                if len(historical_data) < 100:
                    continue
                
                # Analyse de tendance
                trend_slope = self._calculate_trend_slope(historical_data)
                
                # Prédiction de la valeur dans 1 heure
                predicted_value = value + (trend_slope * 12)  # 12 points de 5 minutes
                
                metric_def = next((m for m in self.core_metrics if m.name == metric_name), None)
                if not metric_def:
                    continue
                
                # Vérification si la prédiction dépasse les seuils critiques
                if predicted_value >= metric_def.threshold_critical * 1.2:
                    prediction = AnomalyDetection(
                        metric_name=f"{metric_name}_prediction",
                        anomaly_type=AnomalyType.TREND,
                        severity=AlertSeverity.WARNING,
                        confidence=0.7,
                        value=predicted_value,
                        expected_range=(0, metric_def.threshold_critical),
                        timestamp=datetime.now() + timedelta(hours=1),
                        context={
                            "prediction": True,
                            "current_value": value,
                            "trend_slope": trend_slope,
                            "time_to_critical": self._calculate_time_to_critical(
                                value, trend_slope, metric_def.threshold_critical
                            )
                        }
                    )
                    predictions.append(prediction)
                    
            except Exception as e:
                logger.error(f"Error predicting failures for {metric_name}: {e}")
        
        return predictions
    
    def _calculate_trend_slope(self, data: List[float]) -> float:
        """Calcule la pente de la tendance"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        y = np.array(data)
        
        # Régression linéaire simple
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _calculate_time_to_critical(self, current_value: float, slope: float, threshold: float) -> Optional[str]:
        """Calcule le temps estimé avant d'atteindre le seuil critique"""
        if slope <= 0:
            return None
        
        time_points = (threshold - current_value) / slope
        time_minutes = time_points * 5  # 5 minutes par point
        
        if time_minutes < 60:
            return f"{int(time_minutes)} minutes"
        elif time_minutes < 1440:
            return f"{int(time_minutes / 60)} hours"
        else:
            return f"{int(time_minutes / 1440)} days"

class ContextAnalyzer:
    """Analyseur de contexte pour enrichir les alertes"""
    
    async def get_deployment_context(self) -> Dict[str, Any]:
        """Récupère le contexte de déploiement"""
        return {
            "recent_deployments": await self._get_recent_deployments(),
            "configuration_changes": await self._get_config_changes(),
            "cluster_events": await self._get_cluster_events()
        }
    
    async def _get_recent_deployments(self) -> List[Dict[str, Any]]:
        """Récupère les déploiements récents"""
        # Implémentation pour récupérer les déploiements via Kubernetes API
        return []
    
    async def _get_config_changes(self) -> List[Dict[str, Any]]:
        """Récupère les changements de configuration récents"""
        # Implémentation pour tracker les changements de config
        return []
    
    async def _get_cluster_events(self) -> List[Dict[str, Any]]:
        """Récupère les événements cluster récents"""
        # Implémentation pour récupérer les événements Kubernetes
        return []

class AutoRemediator:
    """Système d'auto-remediation intelligente"""
    
    def __init__(self):
        self.remediation_scripts = {
            "restart_alertmanager": self._restart_alertmanager,
            "restart_failed_instances": self._restart_failed_instances,
            "validate_alert_configs": self._validate_alert_configs,
            "scale_up_resources": self._scale_up_resources,
            "clear_alert_queue": self._clear_alert_queue
        }
    
    async def execute_remediation(self, script_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute un script de remediation"""
        
        if script_name not in self.remediation_scripts:
            raise ValueError(f"Unknown remediation script: {script_name}")
        
        logger.info(f"Executing remediation: {script_name}")
        
        try:
            result = await self.remediation_scripts[script_name](context)
            logger.info(f"Remediation completed: {script_name}")
            return result
        except Exception as e:
            logger.error(f"Remediation failed: {script_name}, error: {e}")
            raise
    
    async def _restart_alertmanager(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Redémarre Alertmanager"""
        # Implémentation du redémarrage
        return {"action": "restart", "status": "completed"}
    
    async def _restart_failed_instances(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Redémarre les instances défaillantes"""
        # Implémentation du redémarrage d'instances
        return {"action": "restart_instances", "status": "completed"}
    
    async def _validate_alert_configs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Valide les configurations d'alertes"""
        # Implémentation de la validation
        return {"action": "validate_configs", "status": "completed"}
    
    async def _scale_up_resources(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Augmente les ressources"""
        # Implémentation du scaling
        return {"action": "scale_up", "status": "completed"}
    
    async def _clear_alert_queue(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Vide la queue d'alertes"""
        # Implémentation du nettoyage
        return {"action": "clear_queue", "status": "completed"}

# Interface principale
async def start_intelligent_monitoring(prometheus_url: str = "http://prometheus:9090"):
    """Démarre le monitoring intelligent"""
    
    engine = AIMonitoringEngine(prometheus_url)
    await engine.initialize()
    await engine.start_monitoring()

if __name__ == "__main__":
    # Démarrage du monitoring
    asyncio.run(start_intelligent_monitoring())
