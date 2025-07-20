"""
📊 Advanced Analytics Engine - Production-Ready System
====================================================

Moteur d'analytics ultra-avancé pour métriques temps réel, prédictions ML
et optimisation automatique des performances de contenu.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import uuid

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types de métriques collectées"""
    ENGAGEMENT = "engagement"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    SENTIMENT = "sentiment"
    CONVERSION = "conversion"
    RETENTION = "retention"
    USAGE = "usage"
    ERROR = "error"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class AnalyticsEvent(Enum):
    """Types d'événements analytics"""
    CONTENT_GENERATED = "content_generated"
    CONTENT_VIEWED = "content_viewed"
    CONTENT_CLICKED = "content_clicked"
    CONTENT_SHARED = "content_shared"
    CONTENT_RATED = "content_rated"
    CONTENT_CONVERTED = "content_converted"
    ERROR_OCCURRED = "error_occurred"
    VALIDATION_FAILED = "validation_failed"
    OPTIMIZATION_APPLIED = "optimization_applied"


class AggregationPeriod(Enum):
    """Périodes d'agrégation"""
    REAL_TIME = "real_time"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class MetricEvent:
    """Événement métrique"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    user_id: Optional[str] = None
    content_id: Optional[str] = None
    event_type: AnalyticsEvent = AnalyticsEvent.CONTENT_GENERATED
    metric_type: MetricType = MetricType.ENGAGEMENT
    
    # Données de l'événement
    value: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Métadonnées temporelles
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    
    # Contexte
    channel: str = "default"
    template_type: str = "unknown"
    language: str = "en"
    
    # Géolocalisation
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None


@dataclass
class AggregatedMetric:
    """Métrique agrégée"""
    metric_type: MetricType
    aggregation_period: AggregationPeriod
    period_start: datetime
    period_end: datetime
    
    # Statistiques
    count: int = 0
    sum: float = 0.0
    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0
    
    # Données détaillées
    values: List[float] = field(default_factory=list)
    percentiles: Dict[int, float] = field(default_factory=dict)
    
    # Segmentation
    by_tenant: Dict[str, float] = field(default_factory=dict)
    by_channel: Dict[str, float] = field(default_factory=dict)
    by_template: Dict[str, float] = field(default_factory=dict)


class RealTimeMetricsCollector:
    """Collecteur de métriques temps réel"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Buffers temps réel
        self.event_buffer: deque = deque(maxlen=config.get('buffer_size', 10000))
        self.metric_cache: Dict[str, Any] = {}
        
        # Configuration
        self.batch_size = config.get('batch_size', 100)
        self.flush_interval = config.get('flush_interval', 30)  # secondes
        self.retention_hours = config.get('retention_hours', 24)
        
        # Processus en arrière-plan
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Métriques système
        self.system_metrics = {
            'events_processed': 0,
            'events_lost': 0,
            'processing_errors': 0,
            'avg_processing_time': 0.0
        }
    
    async def start(self):
        """Démarrage du collecteur"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events_loop())
        self.logger.info("Real-time metrics collector started")
    
    async def stop(self):
        """Arrêt du collecteur"""
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Flush final
        await self._flush_buffer()
        self.logger.info("Real-time metrics collector stopped")
    
    async def collect_event(self, event: MetricEvent):
        """Collection d'un événement"""
        try:
            # Enrichissement de l'événement
            enriched_event = await self._enrich_event(event)
            
            # Ajout au buffer
            if len(self.event_buffer) >= self.event_buffer.maxlen:
                self.system_metrics['events_lost'] += 1
                self.logger.warning("Event buffer full, dropping event")
            else:
                self.event_buffer.append(enriched_event)
                self.system_metrics['events_processed'] += 1
            
            # Mise à jour des métriques temps réel
            await self._update_realtime_metrics(enriched_event)
            
        except Exception as e:
            self.system_metrics['processing_errors'] += 1
            self.logger.error(f"Failed to collect event: {str(e)}")
    
    async def _enrich_event(self, event: MetricEvent) -> MetricEvent:
        """Enrichissement d'un événement"""
        
        # Ajout de métadonnées système
        event.properties.update({
            'collector_version': '1.0.0',
            'processing_timestamp': datetime.utcnow().isoformat(),
            'server_id': self.config.get('server_id', 'unknown')
        })
        
        # Enrichissement géographique (simulation)
        if not event.country:
            event.country = await self._detect_country(event)
        
        # Enrichissement de session
        if not event.session_id:
            event.session_id = await self._get_or_create_session(event)
        
        return event
    
    async def _detect_country(self, event: MetricEvent) -> str:
        """Détection du pays (simulation)"""
        # Dans un vrai système, on utiliserait une API de géolocalisation
        return "US"  # Valeur par défaut
    
    async def _get_or_create_session(self, event: MetricEvent) -> str:
        """Obtention ou création d'une session"""
        # Logique de session simplifiée
        session_key = f"{event.tenant_id}_{event.user_id}"
        return f"session_{hash(session_key)}"
    
    async def _update_realtime_metrics(self, event: MetricEvent):
        """Mise à jour des métriques temps réel"""
        
        # Clés de métriques
        base_key = f"{event.metric_type.value}_{event.tenant_id}"
        channel_key = f"{base_key}_{event.channel}"
        template_key = f"{base_key}_{event.template_type}"
        
        # Mise à jour des compteurs
        self.metric_cache[base_key] = self.metric_cache.get(base_key, 0) + event.value
        self.metric_cache[channel_key] = self.metric_cache.get(channel_key, 0) + event.value
        self.metric_cache[template_key] = self.metric_cache.get(template_key, 0) + event.value
        
        # Métriques par période (dernière heure)
        hour_key = f"{base_key}_hour_{datetime.utcnow().hour}"
        self.metric_cache[hour_key] = self.metric_cache.get(hour_key, 0) + event.value
    
    async def _process_events_loop(self):
        """Boucle de traitement des événements"""
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Event processing loop error: {str(e)}")
    
    async def _flush_buffer(self):
        """Vidage du buffer vers le stockage"""
        if not self.event_buffer:
            return
        
        start_time = datetime.utcnow()
        
        # Extraction des événements
        events_to_process = []
        while self.event_buffer and len(events_to_process) < self.batch_size:
            events_to_process.append(self.event_buffer.popleft())
        
        # Traitement par lot
        await self._process_event_batch(events_to_process)
        
        # Mise à jour des métriques de performance
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        self._update_processing_metrics(processing_time, len(events_to_process))
        
        self.logger.debug(f"Flushed {len(events_to_process)} events in {processing_time:.2f}s")
    
    async def _process_event_batch(self, events: List[MetricEvent]):
        """Traitement d'un lot d'événements"""
        
        # Agrégation par période
        aggregations = self._aggregate_events(events)
        
        # Stockage (simulation - dans un vrai système, on sauvegarderait en DB)
        for aggregation in aggregations:
            self.logger.debug(f"Aggregated metric: {aggregation.metric_type.value} = {aggregation.avg}")
        
        # Déclenchement d'alertes si nécessaire
        await self._check_alerts(aggregations)
    
    def _aggregate_events(self, events: List[MetricEvent]) -> List[AggregatedMetric]:
        """Agrégation des événements"""
        
        # Groupement par type de métrique et période
        groups = defaultdict(list)
        
        for event in events:
            # Arrondi à l'heure
            hour_start = event.timestamp.replace(minute=0, second=0, microsecond=0)
            key = (event.metric_type, hour_start)
            groups[key].append(event)
        
        # Création des agrégations
        aggregations = []
        for (metric_type, hour_start), group_events in groups.items():
            
            values = [e.value for e in group_events]
            
            aggregation = AggregatedMetric(
                metric_type=metric_type,
                aggregation_period=AggregationPeriod.HOUR,
                period_start=hour_start,
                period_end=hour_start + timedelta(hours=1),
                count=len(values),
                sum=sum(values),
                avg=np.mean(values),
                min=min(values),
                max=max(values),
                std=np.std(values),
                values=values,
                percentiles={
                    25: np.percentile(values, 25),
                    50: np.percentile(values, 50),
                    75: np.percentile(values, 75),
                    95: np.percentile(values, 95),
                    99: np.percentile(values, 99)
                }
            )
            
            # Segmentation
            aggregation.by_tenant = self._segment_by_field(group_events, 'tenant_id')
            aggregation.by_channel = self._segment_by_field(group_events, 'channel')
            aggregation.by_template = self._segment_by_field(group_events, 'template_type')
            
            aggregations.append(aggregation)
        
        return aggregations
    
    def _segment_by_field(self, events: List[MetricEvent], field: str) -> Dict[str, float]:
        """Segmentation par champ"""
        segments = defaultdict(list)
        
        for event in events:
            value = getattr(event, field)
            segments[value].append(event.value)
        
        return {k: np.mean(v) for k, v in segments.items()}
    
    async def _check_alerts(self, aggregations: List[AggregatedMetric]):
        """Vérification des alertes"""
        
        for aggregation in aggregations:
            # Seuils d'alerte (configurables)
            alert_thresholds = {
                MetricType.ERROR: 0.05,  # 5% d'erreurs
                MetricType.PERFORMANCE: 1000,  # 1s de latence
                MetricType.ENGAGEMENT: 0.1  # 10% d'engagement minimum
            }
            
            threshold = alert_thresholds.get(aggregation.metric_type)
            if threshold and aggregation.avg > threshold:
                await self._trigger_alert(aggregation, threshold)
    
    async def _trigger_alert(self, aggregation: AggregatedMetric, threshold: float):
        """Déclenchement d'alerte"""
        
        alert_data = {
            'metric_type': aggregation.metric_type.value,
            'current_value': aggregation.avg,
            'threshold': threshold,
            'period': aggregation.period_start.isoformat(),
            'severity': 'warning' if aggregation.avg < threshold * 2 else 'critical'
        }
        
        self.logger.warning(f"Alert triggered: {alert_data}")
        
        # Dans un vrai système, on enverrait une notification
    
    def _update_processing_metrics(self, processing_time: float, event_count: int):
        """Mise à jour des métriques de traitement"""
        
        # Moyenne mobile du temps de traitement
        current_avg = self.system_metrics['avg_processing_time']
        total_processed = self.system_metrics['events_processed']
        
        if total_processed > 0:
            new_avg = (current_avg * (total_processed - event_count) + processing_time) / total_processed
            self.system_metrics['avg_processing_time'] = new_avg
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Obtention des métriques temps réel"""
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system_metrics': self.system_metrics.copy(),
            'cache_size': len(self.metric_cache),
            'buffer_size': len(self.event_buffer),
            'top_metrics': dict(sorted(
                self.metric_cache.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])
        }


class EngagementAnalyzer:
    """Analyseur d'engagement avec ML"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Modèles ML
        self.engagement_model = None
        self.churn_model = None
        self.scaler = StandardScaler()
        
        # Données d'entraînement
        self.training_data = []
        self.feature_columns = [
            'content_length', 'sentiment_score', 'readability_score',
            'hour_of_day', 'day_of_week', 'channel_numeric',
            'template_type_numeric', 'personalization_score'
        ]
        
        # Cache des prédictions
        self.prediction_cache: Dict[str, float] = {}
        
        # Initialisation
        self._init_models()
    
    def _init_models(self):
        """Initialisation des modèles ML"""
        
        try:
            # Tentative de chargement des modèles pré-entraînés
            model_path = self.config.get('model_path', './ml_models/')
            
            try:
                with open(f"{model_path}/engagement_model.pkl", 'rb') as f:
                    self.engagement_model = pickle.load(f)
                with open(f"{model_path}/churn_model.pkl", 'rb') as f:
                    self.churn_model = pickle.load(f)
                with open(f"{model_path}/scaler.pkl", 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.logger.info("Pre-trained models loaded successfully")
                
            except FileNotFoundError:
                # Création de nouveaux modèles
                self.engagement_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
                self.churn_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
                
                self.logger.info("New models initialized")
                
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
    
    async def analyze_engagement(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        historical_data: List[MetricEvent]
    ) -> Dict[str, Any]:
        """Analyse de l'engagement"""
        
        try:
            # Extraction des features
            features = await self._extract_features(content, metadata, historical_data)
            
            # Prédiction d'engagement
            engagement_prediction = await self._predict_engagement(features)
            
            # Prédiction de churn
            churn_prediction = await self._predict_churn(features, historical_data)
            
            # Recommandations d'optimisation
            recommendations = await self._generate_optimization_recommendations(
                features, engagement_prediction, churn_prediction
            )
            
            # Score global d'engagement
            engagement_score = self._calculate_engagement_score(
                engagement_prediction, churn_prediction, features
            )
            
            return {
                'engagement_score': engagement_score,
                'engagement_prediction': engagement_prediction,
                'churn_risk': churn_prediction,
                'recommendations': recommendations,
                'features': features,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Engagement analysis failed: {str(e)}")
            return {
                'engagement_score': 0.5,
                'error': str(e),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    async def _extract_features(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        historical_data: List[MetricEvent]
    ) -> Dict[str, float]:
        """Extraction des features pour ML"""
        
        features = {}
        
        # Features de contenu
        features['content_length'] = len(content)
        features['sentiment_score'] = metadata.get('sentiment_score', 0.5)
        features['readability_score'] = metadata.get('readability_score', 0.5)
        features['personalization_score'] = metadata.get('personalization_score', 0.5)
        
        # Features temporelles
        now = datetime.utcnow()
        features['hour_of_day'] = now.hour
        features['day_of_week'] = now.weekday()
        
        # Features de canal et template
        channel_mapping = {'slack': 1, 'email': 2, 'teams': 3, 'web': 4}
        features['channel_numeric'] = channel_mapping.get(metadata.get('channel', 'web'), 4)
        
        template_mapping = {'welcome': 1, 'alert': 2, 'recommendation': 3, 'billing': 4}
        features['template_type_numeric'] = template_mapping.get(
            metadata.get('template_type', 'welcome'), 1
        )
        
        # Features historiques
        if historical_data:
            recent_events = [e for e in historical_data if 
                           (now - e.timestamp).days <= 7]  # 7 derniers jours
            
            if recent_events:
                engagement_events = [e for e in recent_events if 
                                   e.metric_type == MetricType.ENGAGEMENT]
                features['avg_historical_engagement'] = np.mean([e.value for e in engagement_events]) if engagement_events else 0.5
                
                features['event_frequency'] = len(recent_events) / 7  # événements par jour
            else:
                features['avg_historical_engagement'] = 0.5
                features['event_frequency'] = 0.0
        else:
            features['avg_historical_engagement'] = 0.5
            features['event_frequency'] = 0.0
        
        return features
    
    async def _predict_engagement(self, features: Dict[str, float]) -> float:
        """Prédiction d'engagement"""
        
        if not self.engagement_model:
            return 0.5  # Valeur par défaut
        
        try:
            # Préparation des features
            feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Prédiction
            if hasattr(self.engagement_model, 'predict_proba'):
                # Classification probabiliste
                proba = self.engagement_model.predict_proba(feature_vector_scaled)[0]
                return proba[1] if len(proba) > 1 else proba[0]
            else:
                # Régression
                prediction = self.engagement_model.predict(feature_vector_scaled)[0]
                return max(0.0, min(1.0, prediction))
                
        except Exception as e:
            self.logger.error(f"Engagement prediction failed: {str(e)}")
            return 0.5
    
    async def _predict_churn(
        self, 
        features: Dict[str, float], 
        historical_data: List[MetricEvent]
    ) -> float:
        """Prédiction de churn"""
        
        if not self.churn_model:
            return 0.1  # Risque faible par défaut
        
        try:
            # Features spécifiques au churn
            churn_features = features.copy()
            
            # Analyse de l'activité récente
            now = datetime.utcnow()
            last_week_events = [e for e in historical_data if 
                              (now - e.timestamp).days <= 7]
            last_month_events = [e for e in historical_data if 
                               (now - e.timestamp).days <= 30]
            
            churn_features['events_last_week'] = len(last_week_events)
            churn_features['events_last_month'] = len(last_month_events)
            
            # Tendance d'activité
            if len(last_month_events) > 0:
                week1_events = len([e for e in last_month_events if (now - e.timestamp).days <= 7])
                week2_events = len([e for e in last_month_events if 7 < (now - e.timestamp).days <= 14])
                week3_events = len([e for e in last_month_events if 14 < (now - e.timestamp).days <= 21])
                week4_events = len([e for e in last_month_events if 21 < (now - e.timestamp).days <= 30])
                
                # Tendance (slope)
                weeks = [1, 2, 3, 4]
                activity = [week1_events, week2_events, week3_events, week4_events]
                churn_features['activity_trend'] = np.polyfit(weeks, activity, 1)[0] if len(activity) > 1 else 0
            else:
                churn_features['activity_trend'] = 0
            
            # Préparation et prédiction
            feature_vector = [churn_features.get(col, 0.0) for col in self.feature_columns + ['events_last_week', 'events_last_month', 'activity_trend']]
            feature_vector_scaled = self.scaler.transform([feature_vector[:len(self.feature_columns)]])  # Adapter à la taille attendue
            
            if hasattr(self.churn_model, 'predict_proba'):
                proba = self.churn_model.predict_proba(feature_vector_scaled)[0]
                return proba[1] if len(proba) > 1 else proba[0]
            else:
                prediction = self.churn_model.predict(feature_vector_scaled)[0]
                return max(0.0, min(1.0, prediction))
                
        except Exception as e:
            self.logger.error(f"Churn prediction failed: {str(e)}")
            return 0.1
    
    async def _generate_optimization_recommendations(
        self, 
        features: Dict[str, float],
        engagement_prediction: float,
        churn_prediction: float
    ) -> List[str]:
        """Génération de recommandations d'optimisation"""
        
        recommendations = []
        
        # Recommandations basées sur l'engagement
        if engagement_prediction < 0.3:
            recommendations.append("Low engagement predicted - consider personalizing content")
            
            if features.get('content_length', 0) > 200:
                recommendations.append("Content may be too long - consider shortening")
            
            if features.get('sentiment_score', 0.5) < 0.4:
                recommendations.append("Sentiment is negative - review tone and messaging")
            
            if features.get('personalization_score', 0.5) < 0.3:
                recommendations.append("Increase personalization based on user preferences")
        
        elif engagement_prediction > 0.7:
            recommendations.append("High engagement predicted - maintain current approach")
        
        # Recommandations basées sur le churn
        if churn_prediction > 0.3:
            recommendations.append("High churn risk - implement retention strategies")
            recommendations.append("Consider offering incentives or special content")
            
            if features.get('event_frequency', 0) < 1:
                recommendations.append("Low activity detected - send re-engagement campaigns")
        
        # Recommandations temporelles
        hour = features.get('hour_of_day', 12)
        if hour < 9 or hour > 17:
            recommendations.append("Consider optimal sending time for better engagement")
        
        # Recommandations de canal
        if features.get('channel_numeric', 4) == 2:  # Email
            recommendations.append("Email channel - consider mobile-friendly formatting")
        elif features.get('channel_numeric', 4) == 1:  # Slack
            recommendations.append("Slack channel - use interactive elements for better engagement")
        
        return recommendations
    
    def _calculate_engagement_score(
        self, 
        engagement_prediction: float,
        churn_prediction: float,
        features: Dict[str, float]
    ) -> float:
        """Calcul du score global d'engagement"""
        
        # Score de base
        base_score = engagement_prediction
        
        # Pénalité pour risque de churn
        churn_penalty = churn_prediction * 0.3
        
        # Bonus pour personnalisation
        personalization_bonus = features.get('personalization_score', 0.5) * 0.1
        
        # Bonus pour qualité
        quality_bonus = (
            features.get('sentiment_score', 0.5) + 
            features.get('readability_score', 0.5)
        ) * 0.05
        
        # Score final
        final_score = base_score - churn_penalty + personalization_bonus + quality_bonus
        
        return max(0.0, min(1.0, final_score))
    
    async def update_models(self, training_events: List[MetricEvent]):
        """Mise à jour des modèles avec de nouvelles données"""
        
        if len(training_events) < 100:  # Pas assez de données
            return
        
        try:
            # Préparation des données d'entraînement
            X, y_engagement, y_churn = await self._prepare_training_data(training_events)
            
            if len(X) < 50:  # Minimum pour l'entraînement
                return
            
            # Division train/test
            X_train, X_test, y_eng_train, y_eng_test = train_test_split(
                X, y_engagement, test_size=0.2, random_state=42
            )
            
            # Normalisation
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Entraînement du modèle d'engagement
            self.engagement_model.fit(X_train_scaled, y_eng_train)
            
            # Évaluation
            y_pred = self.engagement_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_eng_test, y_pred)
            
            self.logger.info(f"Engagement model updated - Accuracy: {accuracy:.3f}")
            
            # Sauvegarde des modèles
            await self._save_models()
            
        except Exception as e:
            self.logger.error(f"Model update failed: {str(e)}")
    
    async def _prepare_training_data(
        self, 
        events: List[MetricEvent]
    ) -> Tuple[List[List[float]], List[int], List[int]]:
        """Préparation des données d'entraînement"""
        
        X = []
        y_engagement = []
        y_churn = []
        
        # Groupement par session
        sessions = defaultdict(list)
        for event in events:
            sessions[event.session_id].append(event)
        
        for session_id, session_events in sessions.items():
            if len(session_events) < 3:  # Sessions trop courtes
                continue
            
            # Features de session
            features = await self._extract_session_features(session_events)
            
            # Labels
            engagement_label = 1 if self._is_engaged_session(session_events) else 0
            churn_label = 1 if self._is_churned_session(session_events) else 0
            
            X.append([features.get(col, 0.0) for col in self.feature_columns])
            y_engagement.append(engagement_label)
            y_churn.append(churn_label)
        
        return X, y_engagement, y_churn
    
    async def _extract_session_features(self, events: List[MetricEvent]) -> Dict[str, float]:
        """Extraction des features de session"""
        
        features = {}
        
        # Agrégation des événements
        if events:
            first_event = min(events, key=lambda e: e.timestamp)
            last_event = max(events, key=lambda e: e.timestamp)
            
            # Features basiques
            features['content_length'] = np.mean([len(e.properties.get('content', '')) for e in events])
            features['sentiment_score'] = np.mean([e.properties.get('sentiment_score', 0.5) for e in events])
            features['readability_score'] = np.mean([e.properties.get('readability_score', 0.5) for e in events])
            features['personalization_score'] = np.mean([e.properties.get('personalization_score', 0.5) for e in events])
            
            # Features temporelles
            features['hour_of_day'] = first_event.timestamp.hour
            features['day_of_week'] = first_event.timestamp.weekday()
            
            # Features de canal
            channels = [e.channel for e in events]
            channel_mapping = {'slack': 1, 'email': 2, 'teams': 3, 'web': 4}
            features['channel_numeric'] = channel_mapping.get(max(set(channels), key=channels.count), 4)
            
            # Features de template
            templates = [e.template_type for e in events]
            template_mapping = {'welcome': 1, 'alert': 2, 'recommendation': 3, 'billing': 4}
            features['template_type_numeric'] = template_mapping.get(max(set(templates), key=templates.count), 1)
            
            # Features d'activité
            features['avg_historical_engagement'] = np.mean([e.value for e in events if e.metric_type == MetricType.ENGAGEMENT])
            features['event_frequency'] = len(events) / max(1, (last_event.timestamp - first_event.timestamp).days or 1)
        
        return features
    
    def _is_engaged_session(self, events: List[MetricEvent]) -> bool:
        """Détermination si une session est engagée"""
        
        engagement_events = [e for e in events if e.metric_type == MetricType.ENGAGEMENT]
        if not engagement_events:
            return False
        
        avg_engagement = np.mean([e.value for e in engagement_events])
        return avg_engagement > 0.5
    
    def _is_churned_session(self, events: List[MetricEvent]) -> bool:
        """Détermination si une session indique un churn"""
        
        # Logique simplifiée : session avec beaucoup d'erreurs ou faible engagement
        error_events = [e for e in events if e.metric_type == MetricType.ERROR]
        engagement_events = [e for e in events if e.metric_type == MetricType.ENGAGEMENT]
        
        error_rate = len(error_events) / len(events)
        avg_engagement = np.mean([e.value for e in engagement_events]) if engagement_events else 0
        
        return error_rate > 0.2 or avg_engagement < 0.2
    
    async def _save_models(self):
        """Sauvegarde des modèles"""
        
        try:
            model_path = self.config.get('model_path', './ml_models/')
            
            with open(f"{model_path}/engagement_model.pkl", 'wb') as f:
                pickle.dump(self.engagement_model, f)
            with open(f"{model_path}/churn_model.pkl", 'wb') as f:
                pickle.dump(self.churn_model, f)
            with open(f"{model_path}/scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Model save failed: {str(e)}")


class InfoAnalytics:
    """Système d'analytics principal pour le module info"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Composants
        self.metrics_collector = RealTimeMetricsCollector(config)
        self.engagement_analyzer = EngagementAnalyzer(config)
        
        # Configuration
        self.analytics_enabled = config.get('analytics_enabled', True)
        self.real_time_enabled = config.get('real_time_enabled', True)
        
        # Dashboards et rapports
        self.dashboard_data: Dict[str, Any] = {}
        self.report_cache: Dict[str, Any] = {}
    
    async def start(self):
        """Démarrage du système d'analytics"""
        if not self.analytics_enabled:
            return
        
        if self.real_time_enabled:
            await self.metrics_collector.start()
        
        self.logger.info("Info analytics system started")
    
    async def stop(self):
        """Arrêt du système d'analytics"""
        if self.real_time_enabled:
            await self.metrics_collector.stop()
        
        self.logger.info("Info analytics system stopped")
    
    async def track_event(
        self, 
        event_type: AnalyticsEvent,
        content: str,
        metadata: Dict[str, Any]
    ):
        """Tracking d'un événement"""
        
        if not self.analytics_enabled:
            return
        
        # Création de l'événement
        event = MetricEvent(
            tenant_id=metadata.get('tenant_id', 'unknown'),
            user_id=metadata.get('user_id'),
            content_id=metadata.get('content_id'),
            event_type=event_type,
            metric_type=self._map_event_to_metric(event_type),
            value=self._calculate_event_value(event_type, content, metadata),
            properties={
                'content_length': len(content),
                'content_hash': hash(content),
                **metadata
            },
            channel=metadata.get('channel', 'default'),
            template_type=metadata.get('template_type', 'unknown'),
            language=metadata.get('language', 'en')
        )
        
        # Collection de l'événement
        if self.real_time_enabled:
            await self.metrics_collector.collect_event(event)
        
        # Mise à jour du dashboard
        await self._update_dashboard(event)
    
    def _map_event_to_metric(self, event_type: AnalyticsEvent) -> MetricType:
        """Mapping événement -> métrique"""
        
        mapping = {
            AnalyticsEvent.CONTENT_GENERATED: MetricType.PERFORMANCE,
            AnalyticsEvent.CONTENT_VIEWED: MetricType.ENGAGEMENT,
            AnalyticsEvent.CONTENT_CLICKED: MetricType.ENGAGEMENT,
            AnalyticsEvent.CONTENT_SHARED: MetricType.ENGAGEMENT,
            AnalyticsEvent.CONTENT_RATED: MetricType.QUALITY,
            AnalyticsEvent.CONTENT_CONVERTED: MetricType.CONVERSION,
            AnalyticsEvent.ERROR_OCCURRED: MetricType.ERROR,
            AnalyticsEvent.VALIDATION_FAILED: MetricType.QUALITY
        }
        
        return mapping.get(event_type, MetricType.USAGE)
    
    def _calculate_event_value(
        self, 
        event_type: AnalyticsEvent, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> float:
        """Calcul de la valeur de l'événement"""
        
        if event_type == AnalyticsEvent.CONTENT_GENERATED:
            return 1.0  # Compteur simple
        elif event_type == AnalyticsEvent.CONTENT_VIEWED:
            return metadata.get('view_duration', 1.0)
        elif event_type == AnalyticsEvent.CONTENT_CLICKED:
            return 1.0  # Click = engagement
        elif event_type == AnalyticsEvent.CONTENT_SHARED:
            return 2.0  # Partage = fort engagement
        elif event_type == AnalyticsEvent.CONTENT_RATED:
            return metadata.get('rating', 3.0)
        elif event_type == AnalyticsEvent.CONTENT_CONVERTED:
            return metadata.get('conversion_value', 5.0)
        elif event_type == AnalyticsEvent.ERROR_OCCURRED:
            return 1.0  # Compteur d'erreurs
        else:
            return 1.0
    
    async def _update_dashboard(self, event: MetricEvent):
        """Mise à jour du dashboard temps réel"""
        
        # Mise à jour des métriques clés
        key = f"{event.metric_type.value}_{event.tenant_id}"
        
        if key not in self.dashboard_data:
            self.dashboard_data[key] = {
                'count': 0,
                'total_value': 0.0,
                'avg_value': 0.0,
                'last_update': None
            }
        
        # Mise à jour
        data = self.dashboard_data[key]
        data['count'] += 1
        data['total_value'] += event.value
        data['avg_value'] = data['total_value'] / data['count']
        data['last_update'] = event.timestamp.isoformat()
    
    async def get_analytics_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Obtention du dashboard analytics"""
        
        # Métriques temps réel
        real_time_metrics = {}
        if self.real_time_enabled:
            real_time_metrics = self.metrics_collector.get_realtime_metrics()
        
        # Métriques par tenant
        tenant_metrics = {
            k: v for k, v in self.dashboard_data.items() 
            if k.endswith(f"_{tenant_id}")
        }
        
        # Analyse d'engagement
        engagement_data = await self._get_engagement_summary(tenant_id)
        
        # KPIs principaux
        kpis = await self._calculate_kpis(tenant_id)
        
        return {
            'tenant_id': tenant_id,
            'timestamp': datetime.utcnow().isoformat(),
            'real_time_metrics': real_time_metrics,
            'tenant_metrics': tenant_metrics,
            'engagement_analysis': engagement_data,
            'kpis': kpis,
            'status': 'active' if self.analytics_enabled else 'disabled'
        }
    
    async def _get_engagement_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Résumé de l'analyse d'engagement"""
        
        # Simulation de données historiques
        historical_events = []  # En réalité, on récupérerait de la DB
        
        if self.engagement_analyzer:
            # Analyse avec données factices pour démo
            dummy_content = "Sample content for analysis"
            dummy_metadata = {
                'tenant_id': tenant_id,
                'channel': 'web',
                'template_type': 'welcome'
            }
            
            analysis = await self.engagement_analyzer.analyze_engagement(
                dummy_content, dummy_metadata, historical_events
            )
            
            return analysis
        
        return {'engagement_score': 0.5, 'status': 'analyzer_not_available'}
    
    async def _calculate_kpis(self, tenant_id: str) -> Dict[str, float]:
        """Calcul des KPIs principaux"""
        
        kpis = {
            'total_content_generated': 0,
            'avg_engagement_rate': 0.0,
            'conversion_rate': 0.0,
            'error_rate': 0.0,
            'avg_response_time': 0.0
        }
        
        # Calcul basé sur les métriques dashboard
        for key, data in self.dashboard_data.items():
            if key.endswith(f"_{tenant_id}"):
                metric_type = key.split('_')[0]
                
                if metric_type == 'performance':
                    kpis['total_content_generated'] += data['count']
                    kpis['avg_response_time'] = data['avg_value']
                elif metric_type == 'engagement':
                    kpis['avg_engagement_rate'] = data['avg_value']
                elif metric_type == 'conversion':
                    kpis['conversion_rate'] = data['avg_value']
                elif metric_type == 'error':
                    kpis['error_rate'] = data['avg_value']
        
        return kpis


class EngagementTracker:
    """Tracker spécialisé pour l'engagement utilisateur"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Sessions actives
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Métriques d'engagement
        self.engagement_metrics = {
            'total_sessions': 0,
            'active_sessions': 0,
            'avg_session_duration': 0.0,
            'avg_interactions_per_session': 0.0
        }
    
    async def start_session(self, user_id: str, tenant_id: str, metadata: Dict[str, Any]) -> str:
        """Démarrage d'une session de tracking"""
        
        session_id = f"session_{user_id}_{int(datetime.utcnow().timestamp())}"
        
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'tenant_id': tenant_id,
            'start_time': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'interactions': [],
            'content_views': [],
            'metadata': metadata
        }
        
        self.engagement_metrics['total_sessions'] += 1
        self.engagement_metrics['active_sessions'] = len(self.active_sessions)
        
        self.logger.debug(f"Started tracking session: {session_id}")
        return session_id
    
    async def track_interaction(
        self, 
        session_id: str, 
        interaction_type: str, 
        content_id: str,
        interaction_data: Dict[str, Any]
    ):
        """Tracking d'une interaction"""
        
        if session_id not in self.active_sessions:
            self.logger.warning(f"Session not found: {session_id}")
            return
        
        session = self.active_sessions[session_id]
        session['last_activity'] = datetime.utcnow()
        
        interaction = {
            'type': interaction_type,
            'content_id': content_id,
            'timestamp': datetime.utcnow(),
            'data': interaction_data
        }
        
        session['interactions'].append(interaction)
        
        # Calcul de l'engagement en temps réel
        await self._update_session_engagement(session_id)
    
    async def track_content_view(
        self, 
        session_id: str, 
        content_id: str,
        view_duration: float,
        view_data: Dict[str, Any]
    ):
        """Tracking d'une vue de contenu"""
        
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session['last_activity'] = datetime.utcnow()
        
        view = {
            'content_id': content_id,
            'duration': view_duration,
            'timestamp': datetime.utcnow(),
            'data': view_data
        }
        
        session['content_views'].append(view)
    
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """Fin d'une session de tracking"""
        
        if session_id not in self.active_sessions:
            return {}
        
        session = self.active_sessions[session_id]
        end_time = datetime.utcnow()
        
        # Calcul des métriques de session
        session_duration = (end_time - session['start_time']).total_seconds()
        num_interactions = len(session['interactions'])
        num_views = len(session['content_views'])
        
        # Engagement score de la session
        engagement_score = await self._calculate_session_engagement_score(session)
        
        # Résumé de la session
        session_summary = {
            'session_id': session_id,
            'user_id': session['user_id'],
            'tenant_id': session['tenant_id'],
            'duration_seconds': session_duration,
            'num_interactions': num_interactions,
            'num_content_views': num_views,
            'engagement_score': engagement_score,
            'start_time': session['start_time'].isoformat(),
            'end_time': end_time.isoformat()
        }
        
        # Mise à jour des métriques globales
        await self._update_global_metrics(session_duration, num_interactions)
        
        # Suppression de la session active
        del self.active_sessions[session_id]
        self.engagement_metrics['active_sessions'] = len(self.active_sessions)
        
        self.logger.debug(f"Ended tracking session: {session_id} - Score: {engagement_score:.2f}")
        
        return session_summary
    
    async def _update_session_engagement(self, session_id: str):
        """Mise à jour de l'engagement de session"""
        
        session = self.active_sessions[session_id]
        engagement_score = await self._calculate_session_engagement_score(session)
        session['current_engagement_score'] = engagement_score
    
    async def _calculate_session_engagement_score(self, session: Dict[str, Any]) -> float:
        """Calcul du score d'engagement de session"""
        
        score = 0.0
        
        # Score basé sur la durée
        duration = (session['last_activity'] - session['start_time']).total_seconds()
        duration_score = min(1.0, duration / 300)  # Max score à 5 minutes
        score += duration_score * 0.3
        
        # Score basé sur les interactions
        num_interactions = len(session['interactions'])
        interaction_score = min(1.0, num_interactions / 10)  # Max score à 10 interactions
        score += interaction_score * 0.4
        
        # Score basé sur les vues de contenu
        num_views = len(session['content_views'])
        view_score = min(1.0, num_views / 5)  # Max score à 5 vues
        score += view_score * 0.2
        
        # Score basé sur la diversité des interactions
        interaction_types = set(i['type'] for i in session['interactions'])
        diversity_score = min(1.0, len(interaction_types) / 3)  # Max 3 types différents
        score += diversity_score * 0.1
        
        return score
    
    async def _update_global_metrics(self, session_duration: float, num_interactions: int):
        """Mise à jour des métriques globales"""
        
        # Moyenne mobile de la durée de session
        current_avg = self.engagement_metrics['avg_session_duration']
        total_sessions = self.engagement_metrics['total_sessions']
        
        new_avg = (current_avg * (total_sessions - 1) + session_duration) / total_sessions
        self.engagement_metrics['avg_session_duration'] = new_avg
        
        # Moyenne mobile des interactions par session
        current_avg_interactions = self.engagement_metrics['avg_interactions_per_session']
        new_avg_interactions = (current_avg_interactions * (total_sessions - 1) + num_interactions) / total_sessions
        self.engagement_metrics['avg_interactions_per_session'] = new_avg_interactions
    
    def get_engagement_metrics(self) -> Dict[str, Any]:
        """Obtention des métriques d'engagement"""
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': self.engagement_metrics.copy(),
            'active_sessions_count': len(self.active_sessions),
            'session_details': {
                session_id: {
                    'user_id': session['user_id'],
                    'tenant_id': session['tenant_id'],
                    'duration': (datetime.utcnow() - session['start_time']).total_seconds(),
                    'interactions': len(session['interactions']),
                    'engagement_score': session.get('current_engagement_score', 0.0)
                }
                for session_id, session in self.active_sessions.items()
            }
        }
