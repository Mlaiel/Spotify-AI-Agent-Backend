"""
Moteur principal d'analytics pour les alertes - Spotify AI Agent
Développé par l'équipe d'experts sous la direction de Fahed Mlaiel

Ce module implémente le cœur de l'intelligence artificielle pour l'analyse 
des alertes en temps réel avec des capacités avancées de ML.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import aioredis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Niveaux de gravité des alertes"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertStatus(Enum):
    """États des alertes"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"

@dataclass
class AlertEvent:
    """Événement d'alerte enrichi"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    status: AlertStatus
    source: str
    service: str
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    fingerprint: Optional[str] = None
    confidence_score: float = 0.0

@dataclass
class AnalyticsResult:
    """Résultat d'analyse d'alerte"""
    alert_id: str
    analysis_timestamp: datetime
    anomaly_score: float
    predicted_impact: str
    recommended_actions: List[str]
    correlation_events: List[str]
    risk_assessment: Dict[str, float]
    confidence_level: float

class AlertAnalyticsEngine:
    """
    Moteur principal d'analytics pour les alertes
    
    Fonctionnalités:
    - Analyse en temps réel des patterns d'alertes
    - Détection d'anomalies avec ML avancé
    - Corrélation intelligente d'événements
    - Prédiction d'impact et recommandations
    - Optimisation continue des modèles
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client: Optional[aioredis.Redis] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        self.isolation_forest = IsolationForest(
            contamination=config.get('contamination_rate', 0.1),
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_initialized = False
        
        # Métriques Prometheus
        self.alerts_processed = Counter(
            'alerts_analytics_processed_total',
            'Total des alertes analysées',
            ['severity', 'source']
        )
        self.analysis_duration = Histogram(
            'alerts_analytics_duration_seconds',
            'Durée d\'analyse des alertes'
        )
        self.anomaly_score_gauge = Gauge(
            'alerts_anomaly_score',
            'Score d\'anomalie des alertes',
            ['alert_id']
        )
        
    async def initialize(self):
        """Initialisation du moteur d'analytics"""
        try:
            # Connexion Redis pour cache
            self.redis_client = await aioredis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                encoding='utf-8',
                decode_responses=True
            )
            
            # Pool de connexions PostgreSQL
            self.db_pool = await asyncpg.create_pool(
                self.config.get('database_url'),
                min_size=5,
                max_size=20
            )
            
            # Préparation des modèles ML
            await self._prepare_ml_models()
            
            self.is_initialized = True
            logger.info("Moteur d'analytics initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def _prepare_ml_models(self):
        """Préparation et entraînement des modèles ML"""
        try:
            # Chargement des données historiques pour l'entraînement
            historical_data = await self._load_historical_alerts()
            
            if len(historical_data) > 100:  # Minimum requis
                # Préparation des features
                features = self._extract_features(historical_data)
                
                # Normalisation
                features_scaled = self.scaler.fit_transform(features)
                
                # Entraînement Isolation Forest
                self.isolation_forest.fit(features_scaled)
                
                logger.info(f"Modèle ML entraîné sur {len(historical_data)} échantillons")
            else:
                logger.warning("Données insuffisantes pour l'entraînement ML")
                
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des modèles: {e}")
    
    async def analyze_alert(self, alert_event: AlertEvent) -> AnalyticsResult:
        """
        Analyse complète d'une alerte
        
        Args:
            alert_event: Événement d'alerte à analyser
            
        Returns:
            AnalyticsResult: Résultat complet de l'analyse
        """
        start_time = datetime.now()
        
        try:
            # Mise à jour des métriques
            self.alerts_processed.labels(
                severity=alert_event.severity.value,
                source=alert_event.source
            ).inc()
            
            # Analyse d'anomalie
            anomaly_score = await self._detect_anomaly(alert_event)
            
            # Corrélation avec autres événements
            correlated_events = await self._find_correlations(alert_event)
            
            # Prédiction d'impact
            predicted_impact = await self._predict_impact(alert_event, anomaly_score)
            
            # Génération de recommandations
            recommendations = await self._generate_recommendations(
                alert_event, anomaly_score, predicted_impact
            )
            
            # Évaluation des risques
            risk_assessment = await self._assess_risks(alert_event, correlated_events)
            
            # Calcul du niveau de confiance
            confidence_level = await self._calculate_confidence(
                alert_event, anomaly_score, len(correlated_events)
            )
            
            result = AnalyticsResult(
                alert_id=alert_event.id,
                analysis_timestamp=datetime.now(),
                anomaly_score=anomaly_score,
                predicted_impact=predicted_impact,
                recommended_actions=recommendations,
                correlation_events=[e.id for e in correlated_events],
                risk_assessment=risk_assessment,
                confidence_level=confidence_level
            )
            
            # Sauvegarde du résultat
            await self._save_analysis_result(result)
            
            # Mise à jour des métriques
            duration = (datetime.now() - start_time).total_seconds()
            self.analysis_duration.observe(duration)
            self.anomaly_score_gauge.labels(alert_id=alert_event.id).set(anomaly_score)
            
            logger.info(f"Analyse terminée pour l'alerte {alert_event.id} en {duration:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de l'alerte {alert_event.id}: {e}")
            raise
    
    async def _detect_anomaly(self, alert_event: AlertEvent) -> float:
        """Détection d'anomalie avec ML"""
        try:
            # Extraction des features de l'alerte
            features = self._extract_alert_features(alert_event)
            
            # Normalisation
            features_scaled = self.scaler.transform([features])
            
            # Prédiction d'anomalie
            anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
            
            # Normalisation du score entre 0 et 1
            normalized_score = max(0, min(1, (anomaly_score + 0.5) * 2))
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Erreur détection anomalie: {e}")
            return 0.5  # Score neutre en cas d'erreur
    
    async def _find_correlations(self, alert_event: AlertEvent) -> List[AlertEvent]:
        """Recherche d'événements corrélés"""
        try:
            # Fenêtre temporelle pour la corrélation
            window_start = alert_event.timestamp - timedelta(minutes=30)
            window_end = alert_event.timestamp + timedelta(minutes=5)
            
            # Requête des alertes dans la fenêtre temporelle
            query = """
                SELECT * FROM alerts 
                WHERE timestamp BETWEEN $1 AND $2 
                AND service = $3 
                AND id != $4
                ORDER BY timestamp DESC
                LIMIT 50
            """
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    query, window_start, window_end, 
                    alert_event.service, alert_event.id
                )
            
            correlated_events = []
            for row in rows:
                # Calcul de la similarité
                similarity = self._calculate_similarity(alert_event, row)
                if similarity > 0.7:  # Seuil de corrélation
                    correlated_events.append(self._row_to_alert_event(row))
            
            return correlated_events
            
        except Exception as e:
            logger.error(f"Erreur recherche corrélations: {e}")
            return []
    
    async def _predict_impact(self, alert_event: AlertEvent, anomaly_score: float) -> str:
        """Prédiction de l'impact de l'alerte"""
        try:
            # Logique de prédiction basée sur multiple facteurs
            impact_score = 0
            
            # Facteur gravité
            severity_weights = {
                AlertSeverity.CRITICAL: 0.4,
                AlertSeverity.HIGH: 0.3,
                AlertSeverity.MEDIUM: 0.2,
                AlertSeverity.LOW: 0.1,
                AlertSeverity.INFO: 0.05
            }
            impact_score += severity_weights.get(alert_event.severity, 0.1)
            
            # Facteur anomalie
            impact_score += anomaly_score * 0.3
            
            # Facteur service critique
            critical_services = self.config.get('critical_services', [])
            if alert_event.service in critical_services:
                impact_score += 0.2
            
            # Historique du service
            historical_impact = await self._get_historical_impact(alert_event.service)
            impact_score += historical_impact * 0.1
            
            # Classification de l'impact
            if impact_score >= 0.8:
                return "CRITICAL_BUSINESS_IMPACT"
            elif impact_score >= 0.6:
                return "HIGH_USER_IMPACT"
            elif impact_score >= 0.4:
                return "MODERATE_IMPACT"
            elif impact_score >= 0.2:
                return "LOW_IMPACT"
            else:
                return "MINIMAL_IMPACT"
                
        except Exception as e:
            logger.error(f"Erreur prédiction impact: {e}")
            return "UNKNOWN_IMPACT"
    
    async def _generate_recommendations(
        self, 
        alert_event: AlertEvent, 
        anomaly_score: float, 
        predicted_impact: str
    ) -> List[str]:
        """Génération de recommandations d'actions"""
        recommendations = []
        
        try:
            # Recommandations basées sur la gravité
            if alert_event.severity == AlertSeverity.CRITICAL:
                recommendations.extend([
                    "Escalader immédiatement à l'équipe d'astreinte",
                    "Déclencher la procédure d'incident majeur",
                    "Notifier le management technique"
                ])
            
            # Recommandations basées sur l'anomalie
            if anomaly_score > 0.8:
                recommendations.extend([
                    "Investiguer les patterns inhabituels détectés",
                    "Comparer avec l'historique des 30 derniers jours",
                    "Vérifier les déploiements récents"
                ])
            
            # Recommandations basées sur l'impact
            if "CRITICAL" in predicted_impact:
                recommendations.extend([
                    "Activer le plan de continuité d'activité",
                    "Préparer une communication client",
                    "Mobiliser l'équipe de gestion de crise"
                ])
            
            # Recommandations spécifiques au service
            service_recommendations = await self._get_service_recommendations(
                alert_event.service
            )
            recommendations.extend(service_recommendations)
            
            # Déduplication et priorisation
            unique_recommendations = list(dict.fromkeys(recommendations))
            return unique_recommendations[:10]  # Maximum 10 recommandations
            
        except Exception as e:
            logger.error(f"Erreur génération recommandations: {e}")
            return ["Analyser manuellement l'alerte", "Contacter l'équipe de support"]
    
    async def _assess_risks(
        self, 
        alert_event: AlertEvent, 
        correlated_events: List[AlertEvent]
    ) -> Dict[str, float]:
        """Évaluation des risques associés"""
        try:
            risks = {
                "service_degradation": 0.0,
                "data_loss": 0.0,
                "security_breach": 0.0,
                "financial_impact": 0.0,
                "reputation_damage": 0.0
            }
            
            # Analyse des patterns de risque
            if alert_event.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                risks["service_degradation"] = 0.8
            
            # Risque basé sur les corrélations
            if len(correlated_events) > 5:
                risks["service_degradation"] = min(1.0, risks["service_degradation"] + 0.3)
            
            # Analyse des labels pour risques spécifiques
            if "security" in alert_event.labels.get("category", "").lower():
                risks["security_breach"] = 0.7
            
            if "database" in alert_event.service.lower():
                risks["data_loss"] = 0.5
            
            # Risque financier pour services critiques
            critical_services = self.config.get('critical_services', [])
            if alert_event.service in critical_services:
                risks["financial_impact"] = 0.6
                risks["reputation_damage"] = 0.4
            
            return risks
            
        except Exception as e:
            logger.error(f"Erreur évaluation risques: {e}")
            return {"unknown_risk": 0.5}
    
    async def _calculate_confidence(
        self, 
        alert_event: AlertEvent, 
        anomaly_score: float, 
        correlation_count: int
    ) -> float:
        """Calcul du niveau de confiance de l'analyse"""
        try:
            confidence = 0.5  # Base confidence
            
            # Facteur qualité des données
            if alert_event.metrics:
                confidence += 0.2
            
            # Facteur corrélation
            if correlation_count > 0:
                confidence += min(0.3, correlation_count * 0.05)
            
            # Facteur historique
            historical_accuracy = await self._get_model_accuracy()
            confidence += historical_accuracy * 0.3
            
            # Facteur anomalie (scores extrêmes = moins de confiance)
            if 0.3 <= anomaly_score <= 0.7:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Erreur calcul confiance: {e}")
            return 0.5
    
    # Méthodes utilitaires et helpers
    
    def _extract_features(self, alerts_data: List[Dict]) -> np.ndarray:
        """Extraction de features pour ML"""
        features = []
        for alert in alerts_data:
            feature_vector = [
                self._severity_to_numeric(alert.get('severity')),
                len(alert.get('labels', {})),
                len(alert.get('message', '')),
                alert.get('response_time', 0),
                alert.get('error_rate', 0)
            ]
            features.append(feature_vector)
        return np.array(features)
    
    def _extract_alert_features(self, alert_event: AlertEvent) -> List[float]:
        """Extraction de features d'une alerte"""
        return [
            self._severity_to_numeric(alert_event.severity.value),
            len(alert_event.labels),
            len(alert_event.message),
            alert_event.metrics.get('response_time', 0),
            alert_event.metrics.get('error_rate', 0)
        ]
    
    def _severity_to_numeric(self, severity: str) -> float:
        """Conversion gravité en valeur numérique"""
        mapping = {
            'critical': 1.0,
            'high': 0.75,
            'medium': 0.5,
            'low': 0.25,
            'info': 0.1
        }
        return mapping.get(severity.lower(), 0.5)
    
    def _calculate_similarity(self, alert1: AlertEvent, alert2_row: dict) -> float:
        """Calcul de similarité entre alertes"""
        similarity = 0.0
        
        # Similarité service
        if alert1.service == alert2_row.get('service'):
            similarity += 0.4
        
        # Similarité gravité
        if alert1.severity.value == alert2_row.get('severity'):
            similarity += 0.3
        
        # Similarité temporelle
        time_diff = abs((alert1.timestamp - alert2_row.get('timestamp')).total_seconds())
        if time_diff < 300:  # 5 minutes
            similarity += 0.3
        
        return similarity
    
    def _row_to_alert_event(self, row: dict) -> AlertEvent:
        """Conversion row DB vers AlertEvent"""
        return AlertEvent(
            id=row['id'],
            timestamp=row['timestamp'],
            severity=AlertSeverity(row['severity']),
            status=AlertStatus(row['status']),
            source=row['source'],
            service=row['service'],
            message=row['message'],
            labels=row.get('labels', {}),
            annotations=row.get('annotations', {}),
            metrics=row.get('metrics', {})
        )
    
    async def _load_historical_alerts(self) -> List[Dict]:
        """Chargement des alertes historiques"""
        try:
            query = """
                SELECT * FROM alerts 
                WHERE timestamp > NOW() - INTERVAL '30 days'
                ORDER BY timestamp DESC
                LIMIT 10000
            """
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Erreur chargement historique: {e}")
            return []
    
    async def _get_historical_impact(self, service: str) -> float:
        """Récupération de l'impact historique d'un service"""
        try:
            # Cache Redis
            cache_key = f"historical_impact:{service}"
            cached_value = await self.redis_client.get(cache_key)
            
            if cached_value:
                return float(cached_value)
            
            # Calcul depuis la DB
            query = """
                SELECT AVG(impact_score) as avg_impact
                FROM alert_impacts 
                WHERE service = $1 
                AND timestamp > NOW() - INTERVAL '7 days'
            """
            
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, service)
                impact = result or 0.3
            
            # Cache pour 1 heure
            await self.redis_client.setex(cache_key, 3600, str(impact))
            return impact
            
        except Exception as e:
            logger.error(f"Erreur récupération impact historique: {e}")
            return 0.3
    
    async def _get_service_recommendations(self, service: str) -> List[str]:
        """Recommandations spécifiques au service"""
        service_recommendations = {
            "database": [
                "Vérifier l'état des connexions DB",
                "Analyser les requêtes lentes",
                "Contrôler l'espace disque"
            ],
            "api": [
                "Vérifier les limites de rate limiting",
                "Analyser les logs d'erreur",
                "Contrôler la latence réseau"
            ],
            "auth": [
                "Vérifier les certificats SSL",
                "Analyser les tentatives de connexion",
                "Contrôler les sessions utilisateur"
            ]
        }
        
        for key, recommendations in service_recommendations.items():
            if key in service.lower():
                return recommendations
        
        return ["Consulter la documentation du service"]
    
    async def _get_model_accuracy(self) -> float:
        """Récupération de la précision du modèle"""
        try:
            cache_key = "model_accuracy"
            cached_value = await self.redis_client.get(cache_key)
            
            if cached_value:
                return float(cached_value)
            
            # Calcul de précision basé sur feedback
            query = """
                SELECT 
                    COUNT(CASE WHEN feedback = 'correct' THEN 1 END)::float / 
                    COUNT(*)::float as accuracy
                FROM analysis_feedback 
                WHERE timestamp > NOW() - INTERVAL '7 days'
            """
            
            async with self.db_pool.acquire() as conn:
                accuracy = await conn.fetchval(query) or 0.75
            
            # Cache pour 4 heures
            await self.redis_client.setex(cache_key, 14400, str(accuracy))
            return accuracy
            
        except Exception as e:
            logger.error(f"Erreur récupération précision modèle: {e}")
            return 0.75
    
    async def _save_analysis_result(self, result: AnalyticsResult):
        """Sauvegarde du résultat d'analyse"""
        try:
            query = """
                INSERT INTO alert_analytics (
                    alert_id, analysis_timestamp, anomaly_score,
                    predicted_impact, recommended_actions, correlation_events,
                    risk_assessment, confidence_level
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    result.alert_id,
                    result.analysis_timestamp,
                    result.anomaly_score,
                    result.predicted_impact,
                    result.recommended_actions,
                    result.correlation_events,
                    result.risk_assessment,
                    result.confidence_level
                )
            
            logger.debug(f"Résultat d'analyse sauvegardé pour {result.alert_id}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde résultat: {e}")
    
    async def get_analytics_summary(self, timeframe: str = "24h") -> Dict[str, Any]:
        """Récupération du résumé analytics"""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_alerts,
                    AVG(anomaly_score) as avg_anomaly_score,
                    AVG(confidence_level) as avg_confidence,
                    COUNT(DISTINCT service) as services_affected
                FROM alert_analytics 
                WHERE analysis_timestamp > NOW() - INTERVAL %s
            """
            
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow(query.replace('%s', f"'{timeframe}'"))
            
            return {
                "timeframe": timeframe,
                "total_alerts_analyzed": result['total_alerts'],
                "average_anomaly_score": float(result['avg_anomaly_score'] or 0),
                "average_confidence": float(result['avg_confidence'] or 0),
                "services_affected": result['services_affected'],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur récupération résumé: {e}")
            return {}
    
    async def close(self):
        """Fermeture propre des connexions"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.db_pool:
                await self.db_pool.close()
            
            logger.info("Connexions fermées proprement")
            
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture: {e}")


# Factory pour création d'instance
async def create_analytics_engine(config: Dict[str, Any]) -> AlertAnalyticsEngine:
    """Factory pour créer et initialiser le moteur d'analytics"""
    engine = AlertAnalyticsEngine(config)
    await engine.initialize()
    return engine
