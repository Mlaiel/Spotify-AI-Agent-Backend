"""
Module d'analyse comportementale avancée pour le monitoring intelligent.

Ce module implémente des algorithmes sophistiqués pour :
- Profiling comportemental des utilisateurs et systèmes
- Détection de déviations par rapport aux patterns normaux
- Analyse d'anomalies comportementales
- Modélisation de patterns d'usage temporels
- Clustering comportemental et segmentation

Utilise des techniques de ML avancées pour la caractérisation comportementale.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import json
import asyncio
from enum import Enum
from collections import defaultdict, deque
import hashlib

# Machine Learning imports
from sklearn.cluster import KMeans, DBSCAN, GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, ICA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
from sklearn.mixture import BayesianGaussianMixture

# Analyse statistique
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
import networkx as nx

# Deep Learning pour l'analyse comportementale
import tensorflow as tf
from tensorflow.keras import layers, Model

# Monitoring et métriques
from prometheus_client import Counter, Histogram, Gauge, Summary
import redis

logger = logging.getLogger(__name__)

# Métriques Prometheus
BEHAVIOR_PROFILES_CREATED = Counter('behavior_profiles_created_total', 'Total behavior profiles created')
BEHAVIOR_ANALYSIS_LATENCY = Histogram('behavior_analysis_duration_seconds', 'Time spent analyzing behavior')
ACTIVE_BEHAVIOR_CLUSTERS = Gauge('active_behavior_clusters_count', 'Number of active behavior clusters')
DEVIATION_DETECTIONS = Counter('behavior_deviations_detected_total', 'Total behavior deviations detected')

class BehaviorType(Enum):
    """Types de comportements analysés."""
    USER_INTERACTION = "user_interaction"
    SYSTEM_PERFORMANCE = "system_performance"
    RESOURCE_USAGE = "resource_usage"
    API_CONSUMPTION = "api_consumption"
    ERROR_PATTERNS = "error_patterns"
    SECURITY_BEHAVIOR = "security_behavior"

class DeviationType(Enum):
    """Types de déviations comportementales."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    PATTERN_SHIFT = "pattern_shift"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    TEMPORAL_DEVIATION = "temporal_deviation"
    VOLUME_SPIKE = "volume_spike"
    SEQUENTIAL_ANOMALY = "sequential_anomaly"

@dataclass
class BehaviorEvent:
    """Événement comportemental."""
    timestamp: datetime
    entity_id: str  # user_id, system_id, etc.
    event_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None

@dataclass
class BehaviorProfile:
    """Profil comportemental d'une entité."""
    entity_id: str
    behavior_type: BehaviorType
    creation_time: datetime
    last_updated: datetime
    
    # Caractéristiques statistiques
    activity_patterns: Dict[str, Any] = field(default_factory=dict)
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    frequency_distributions: Dict[str, Any] = field(default_factory=dict)
    
    # Modèles appris
    normal_behavior_model: Optional[Any] = None
    anomaly_threshold: float = 0.95
    
    # Métriques de qualité
    confidence_score: float = 0.0
    sample_count: int = 0
    last_deviation_time: Optional[datetime] = None

@dataclass
class BehaviorDeviation:
    """Déviation comportementale détectée."""
    deviation_id: str
    entity_id: str
    detection_time: datetime
    deviation_type: DeviationType
    severity_score: float
    confidence: float
    
    # Détails de la déviation
    observed_behavior: Dict[str, Any]
    expected_behavior: Dict[str, Any]
    deviation_metrics: Dict[str, float]
    
    # Contexte
    contributing_factors: List[str] = field(default_factory=list)
    similar_historical_deviations: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

@dataclass
class BehaviorCluster:
    """Cluster de comportements similaires."""
    cluster_id: str
    cluster_type: BehaviorType
    entities: Set[str]
    centroid_profile: Dict[str, Any]
    cohesion_score: float
    creation_time: datetime
    last_updated: datetime

class BehaviorAnalysisEngine:
    """Moteur principal d'analyse comportementale."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.entity_profiles = {}
        self.event_streams = defaultdict(deque)
        self.behavior_clusters = {}
        self.deviation_history = []
        
        # Modèles d'analyse
        self.anomaly_detectors = {}
        self.clustering_models = {}
        self.pattern_models = {}
        
        # Cache et persistence
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
    def _default_config(self) -> Dict[str, Any]:
        """Configuration par défaut."""
        return {
            'window_size': timedelta(hours=24),
            'min_events_for_profile': 50,
            'anomaly_threshold': 0.95,
            'clustering_update_interval': timedelta(hours=6),
            'profile_update_frequency': timedelta(minutes=30),
            'max_event_history': 10000,
            'deviation_sensitivity': 0.8
        }
    
    def add_behavior_event(self, event: BehaviorEvent) -> None:
        """Ajoute un événement comportemental."""
        
        entity_id = event.entity_id
        
        # Ajout à l'historique des événements
        self.event_streams[entity_id].append(event)
        
        # Limitation de la taille de l'historique
        max_history = self.config['max_event_history']
        if len(self.event_streams[entity_id]) > max_history:
            self.event_streams[entity_id].popleft()
        
        # Mise à jour du profil comportemental
        self._update_behavior_profile(entity_id)
        
        # Détection de déviations en temps réel
        deviation = self._detect_real_time_deviation(event)
        if deviation:
            self.deviation_history.append(deviation)
            DEVIATION_DETECTIONS.inc()
            
            logger.info(f"Behavior deviation detected for {entity_id}: {deviation.deviation_type}")
    
    def _update_behavior_profile(self, entity_id: str) -> None:
        """Met à jour le profil comportemental d'une entité."""
        
        events = list(self.event_streams[entity_id])
        
        if len(events) < self.config['min_events_for_profile']:
            return
        
        current_time = datetime.now()
        
        if entity_id in self.entity_profiles:
            profile = self.entity_profiles[entity_id]
            
            # Vérification si mise à jour nécessaire
            time_since_update = current_time - profile.last_updated
            if time_since_update < self.config['profile_update_frequency']:
                return
        else:
            # Création d'un nouveau profil
            profile = BehaviorProfile(
                entity_id=entity_id,
                behavior_type=BehaviorType.USER_INTERACTION,  # Default
                creation_time=current_time,
                last_updated=current_time
            )
        
        # Analyse des patterns d'activité
        profile.activity_patterns = self._analyze_activity_patterns(events)
        
        # Analyse des patterns temporels
        profile.temporal_patterns = self._analyze_temporal_patterns(events)
        
        # Analyse des distributions de fréquence
        profile.frequency_distributions = self._analyze_frequency_distributions(events)
        
        # Entraînement du modèle de comportement normal
        profile.normal_behavior_model = self._train_normal_behavior_model(events)
        
        # Mise à jour des métriques
        profile.sample_count = len(events)
        profile.confidence_score = self._calculate_profile_confidence(events, profile)
        profile.last_updated = current_time
        
        self.entity_profiles[entity_id] = profile
        
        BEHAVIOR_PROFILES_CREATED.inc()
        
        # Sauvegarde dans Redis
        self._cache_profile(profile)
    
    def _analyze_activity_patterns(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyse les patterns d'activité."""
        
        patterns = {}
        
        # Analyse de volume d'activité
        event_counts_by_hour = defaultdict(int)
        event_types_frequency = defaultdict(int)
        
        for event in events:
            hour = event.timestamp.hour
            event_counts_by_hour[hour] += 1
            event_types_frequency[event.event_type] += 1
        
        # Statistiques de volume
        hourly_counts = list(event_counts_by_hour.values())
        patterns['volume_stats'] = {
            'mean_events_per_hour': np.mean(hourly_counts) if hourly_counts else 0,
            'std_events_per_hour': np.std(hourly_counts) if hourly_counts else 0,
            'peak_activity_hour': max(event_counts_by_hour, key=event_counts_by_hour.get) if event_counts_by_hour else 0,
            'total_events': len(events)
        }
        
        # Distribution des types d'événements
        total_events = len(events)
        patterns['event_type_distribution'] = {
            event_type: count / total_events 
            for event_type, count in event_types_frequency.items()
        }
        
        # Patterns de burst (pics d'activité)
        patterns['burst_analysis'] = self._analyze_burst_patterns(events)
        
        return patterns
    
    def _analyze_temporal_patterns(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyse les patterns temporels."""
        
        patterns = {}
        
        # Analyse par jour de la semaine
        weekday_activity = defaultdict(int)
        hour_activity = defaultdict(int)
        
        for event in events:
            weekday_activity[event.timestamp.weekday()] += 1
            hour_activity[event.timestamp.hour] += 1
        
        patterns['weekday_distribution'] = dict(weekday_activity)
        patterns['hourly_distribution'] = dict(hour_activity)
        
        # Détection de saisonnalité
        if len(events) >= 168:  # Au moins une semaine de données
            patterns['seasonality'] = self._detect_temporal_seasonality(events)
        
        # Analyse des intervalles entre événements
        if len(events) >= 2:
            intervals = []
            for i in range(1, len(events)):
                interval = (events[i].timestamp - events[i-1].timestamp).total_seconds()
                intervals.append(interval)
            
            patterns['inter_event_intervals'] = {
                'mean_interval': np.mean(intervals),
                'std_interval': np.std(intervals),
                'median_interval': np.median(intervals),
                'percentile_95': np.percentile(intervals, 95)
            }
        
        return patterns
    
    def _analyze_frequency_distributions(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyse les distributions de fréquence."""
        
        distributions = {}
        
        # Distribution des attributs d'événements
        attribute_distributions = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            for attr_name, attr_value in event.attributes.items():
                if isinstance(attr_value, (int, float, str)):
                    attribute_distributions[attr_name][str(attr_value)] += 1
        
        # Conversion en probabilités
        for attr_name, value_counts in attribute_distributions.items():
            total = sum(value_counts.values())
            distributions[attr_name] = {
                value: count / total 
                for value, count in value_counts.items()
            }
        
        return distributions
    
    def _analyze_burst_patterns(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyse les patterns de pics d'activité."""
        
        # Groupement par fenêtres de temps
        window_size = timedelta(minutes=5)
        event_windows = defaultdict(int)
        
        for event in events:
            window_key = event.timestamp.replace(
                minute=(event.timestamp.minute // 5) * 5,
                second=0,
                microsecond=0
            )
            event_windows[window_key] += 1
        
        window_counts = list(event_windows.values())
        
        if not window_counts:
            return {'burst_detected': False}
        
        # Détection de bursts (utilisation de z-score)
        mean_count = np.mean(window_counts)
        std_count = np.std(window_counts)
        
        burst_threshold = mean_count + 2 * std_count
        burst_windows = [count for count in window_counts if count > burst_threshold]
        
        return {
            'burst_detected': len(burst_windows) > 0,
            'burst_frequency': len(burst_windows) / len(window_counts),
            'avg_burst_intensity': np.mean(burst_windows) if burst_windows else 0,
            'burst_threshold': burst_threshold
        }
    
    def _detect_temporal_seasonality(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Détecte la saisonnalité temporelle."""
        
        # Agrégation par heure
        hourly_counts = defaultdict(int)
        for event in events:
            hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1
        
        # Série temporelle
        sorted_hours = sorted(hourly_counts.keys())
        counts = [hourly_counts[hour] for hour in sorted_hours]
        
        # Analyse de Fourier simple pour détecter périodicités
        if len(counts) >= 24:
            try:
                fft_result = np.fft.fft(counts)
                frequencies = np.fft.fftfreq(len(counts))
                
                # Identification des fréquences dominantes
                power_spectrum = np.abs(fft_result) ** 2
                dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                dominant_period = 1 / abs(frequencies[dominant_freq_idx]) if frequencies[dominant_freq_idx] != 0 else 0
                
                return {
                    'seasonality_detected': True,
                    'dominant_period_hours': dominant_period,
                    'seasonality_strength': power_spectrum[dominant_freq_idx] / np.sum(power_spectrum)
                }
                
            except:
                pass
        
        return {'seasonality_detected': False}
    
    def _train_normal_behavior_model(self, events: List[BehaviorEvent]) -> Optional[Any]:
        """Entraîne un modèle de comportement normal."""
        
        try:
            # Extraction de features pour le modèle
            features = self._extract_behavior_features(events)
            
            if len(features) < 10:
                return None
            
            # Entraînement d'un modèle Isolation Forest
            model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(features)
            
            return model
            
        except Exception as e:
            logger.warning(f"Failed to train normal behavior model: {e}")
            return None
    
    def _extract_behavior_features(self, events: List[BehaviorEvent]) -> np.ndarray:
        """Extrait les features comportementales."""
        
        features = []
        
        # Features basées sur des fenêtres temporelles
        window_size = timedelta(hours=1)
        
        # Groupement par fenêtres
        windows = defaultdict(list)
        for event in events:
            window_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
            windows[window_key].append(event)
        
        for window_time, window_events in windows.items():
            if len(window_events) == 0:
                continue
            
            # Features temporelles
            hour = window_time.hour
            day_of_week = window_time.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Features d'activité
            event_count = len(window_events)
            unique_event_types = len(set(e.event_type for e in window_events))
            
            # Features d'intervalles
            if len(window_events) > 1:
                intervals = []
                for i in range(1, len(window_events)):
                    interval = (window_events[i].timestamp - window_events[i-1].timestamp).total_seconds()
                    intervals.append(interval)
                
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)
            else:
                avg_interval = 0
                std_interval = 0
            
            # Assemblage des features
            feature_vector = [
                hour, day_of_week, is_weekend,
                event_count, unique_event_types,
                avg_interval, std_interval
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _calculate_profile_confidence(self, events: List[BehaviorEvent], 
                                    profile: BehaviorProfile) -> float:
        """Calcule le score de confiance du profil."""
        
        # Facteurs de confiance
        sample_size_factor = min(1.0, len(events) / 1000)  # Plus d'échantillons = plus de confiance
        
        # Consistance temporelle
        time_span = (events[-1].timestamp - events[0].timestamp).total_seconds() / 86400  # jours
        temporal_factor = min(1.0, time_span / 30)  # 30 jours pour confiance maximale
        
        # Diversité des événements
        unique_event_types = len(set(e.event_type for e in events))
        diversity_factor = min(1.0, unique_event_types / 10)
        
        # Score composite
        confidence = (sample_size_factor * 0.4 + temporal_factor * 0.3 + diversity_factor * 0.3)
        
        return confidence
    
    def _cache_profile(self, profile: BehaviorProfile) -> None:
        """Met en cache le profil dans Redis."""
        
        try:
            cache_key = f"behavior_profile:{profile.entity_id}"
            cache_data = {
                'entity_id': profile.entity_id,
                'behavior_type': profile.behavior_type.value,
                'creation_time': profile.creation_time.isoformat(),
                'last_updated': profile.last_updated.isoformat(),
                'activity_patterns': profile.activity_patterns,
                'temporal_patterns': profile.temporal_patterns,
                'confidence_score': profile.confidence_score,
                'sample_count': profile.sample_count
            }
            
            self.redis_client.setex(
                cache_key,
                timedelta(days=7),
                json.dumps(cache_data, default=str)
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache profile: {e}")
    
    def _detect_real_time_deviation(self, event: BehaviorEvent) -> Optional[BehaviorDeviation]:
        """Détecte les déviations comportementales en temps réel."""
        
        entity_id = event.entity_id
        
        if entity_id not in self.entity_profiles:
            return None
        
        profile = self.entity_profiles[entity_id]
        
        if profile.normal_behavior_model is None:
            return None
        
        try:
            # Extraction des features pour l'événement actuel
            recent_events = list(self.event_streams[entity_id])[-10:]  # 10 derniers événements
            features = self._extract_behavior_features(recent_events)
            
            if len(features) == 0:
                return None
            
            # Prédiction d'anomalie
            anomaly_scores = profile.normal_behavior_model.decision_function(features)
            is_anomaly = profile.normal_behavior_model.predict(features)
            
            latest_score = anomaly_scores[-1]
            latest_prediction = is_anomaly[-1]
            
            if latest_prediction == -1:  # Anomalie détectée
                # Détermination du type de déviation
                deviation_type = self._classify_deviation_type(event, profile, latest_score)
                
                # Calcul de la sévérité
                severity_score = self._calculate_deviation_severity(latest_score, profile)
                
                # Analyse des facteurs contributeurs
                contributing_factors = self._analyze_contributing_factors(event, profile)
                
                deviation = BehaviorDeviation(
                    deviation_id=f"{entity_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    entity_id=entity_id,
                    detection_time=event.timestamp,
                    deviation_type=deviation_type,
                    severity_score=severity_score,
                    confidence=abs(latest_score),
                    observed_behavior={'latest_event': event.__dict__},
                    expected_behavior=profile.activity_patterns,
                    deviation_metrics={'anomaly_score': latest_score},
                    contributing_factors=contributing_factors,
                    recommended_actions=self._generate_deviation_recommendations(deviation_type, severity_score)
                )
                
                return deviation
        
        except Exception as e:
            logger.warning(f"Real-time deviation detection failed: {e}")
        
        return None
    
    def _classify_deviation_type(self, event: BehaviorEvent, profile: BehaviorProfile, 
                               anomaly_score: float) -> DeviationType:
        """Classifie le type de déviation."""
        
        # Analyse de volume
        recent_events = list(self.event_streams[event.entity_id])[-60:]  # Dernière heure
        current_volume = len([e for e in recent_events if (event.timestamp - e.timestamp).total_seconds() <= 3600])
        expected_volume = profile.activity_patterns.get('volume_stats', {}).get('mean_events_per_hour', 0)
        
        if current_volume > expected_volume * 2:
            return DeviationType.VOLUME_SPIKE
        
        # Analyse temporelle
        current_hour = event.timestamp.hour
        expected_hourly_activity = profile.temporal_patterns.get('hourly_distribution', {})
        
        if current_hour in expected_hourly_activity:
            if expected_hourly_activity[current_hour] == 0:
                return DeviationType.TEMPORAL_DEVIATION
        
        # Analyse de fréquence
        event_type_freq = profile.frequency_distributions.get(event.event_type, {})
        if not event_type_freq:
            return DeviationType.PATTERN_SHIFT
        
        # Par défaut, outlier statistique
        return DeviationType.STATISTICAL_OUTLIER
    
    def _calculate_deviation_severity(self, anomaly_score: float, profile: BehaviorProfile) -> float:
        """Calcule la sévérité de la déviation."""
        
        # Normalisation du score d'anomalie
        base_severity = abs(anomaly_score)
        
        # Ajustement basé sur la confiance du profil
        confidence_adjustment = profile.confidence_score * 0.2
        
        # Ajustement basé sur l'historique de déviations
        if profile.last_deviation_time:
            time_since_last = datetime.now() - profile.last_deviation_time
            if time_since_last < timedelta(hours=1):
                confidence_adjustment += 0.3  # Déviations répétées = plus sévère
        
        severity = min(1.0, base_severity + confidence_adjustment)
        
        return severity
    
    def _analyze_contributing_factors(self, event: BehaviorEvent, 
                                    profile: BehaviorProfile) -> List[str]:
        """Analyse les facteurs contribuant à la déviation."""
        
        factors = []
        
        # Facteurs temporels
        if event.timestamp.hour not in profile.temporal_patterns.get('hourly_distribution', {}):
            factors.append("unusual_time_of_activity")
        
        if event.timestamp.weekday() >= 5:  # Weekend
            weekday_activity = sum(profile.temporal_patterns.get('weekday_distribution', {}).values())
            weekend_activity = profile.temporal_patterns.get('weekday_distribution', {}).get(5, 0) + \
                             profile.temporal_patterns.get('weekday_distribution', {}).get(6, 0)
            
            if weekend_activity < weekday_activity * 0.1:
                factors.append("weekend_activity_unusual")
        
        # Facteurs de type d'événement
        event_type_prob = profile.frequency_distributions.get(event.event_type, {})
        if not event_type_prob or max(event_type_prob.values()) < 0.1:
            factors.append("rare_event_type")
        
        # Facteurs de volume
        recent_count = len([e for e in self.event_streams[event.entity_id] 
                           if (event.timestamp - e.timestamp).total_seconds() <= 3600])
        expected_hourly = profile.activity_patterns.get('volume_stats', {}).get('mean_events_per_hour', 0)
        
        if recent_count > expected_hourly * 1.5:
            factors.append("high_activity_volume")
        elif recent_count < expected_hourly * 0.5:
            factors.append("low_activity_volume")
        
        return factors
    
    def _generate_deviation_recommendations(self, deviation_type: DeviationType, 
                                          severity_score: float) -> List[str]:
        """Génère des recommandations basées sur le type de déviation."""
        
        recommendations = []
        
        # Recommandations basées sur le type
        if deviation_type == DeviationType.VOLUME_SPIKE:
            recommendations.extend([
                "Monitor system resources for potential overload",
                "Check for DDoS or bot activity",
                "Review rate limiting policies"
            ])
        elif deviation_type == DeviationType.TEMPORAL_DEVIATION:
            recommendations.extend([
                "Investigate unusual after-hours activity",
                "Check for automated processes or batch jobs",
                "Review access control logs"
            ])
        elif deviation_type == DeviationType.PATTERN_SHIFT:
            recommendations.extend([
                "Analyze recent system changes",
                "Review user behavior trends",
                "Check for new feature rollouts"
            ])
        
        # Recommandations basées sur la sévérité
        if severity_score > 0.8:
            recommendations.extend([
                "Escalate to security team",
                "Consider temporary access restrictions",
                "Activate incident response procedures"
            ])
        elif severity_score > 0.6:
            recommendations.extend([
                "Increase monitoring frequency",
                "Notify relevant team leads",
                "Review recent configuration changes"
            ])
        
        return recommendations

class UserBehaviorAnalyzer(BehaviorAnalysisEngine):
    """Analyseur spécialisé pour le comportement des utilisateurs."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.session_analyzer = UserSessionAnalyzer()
        self.journey_analyzer = UserJourneyAnalyzer()
        
    def analyze_user_session(self, user_id: str, session_events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyse une session utilisateur complète."""
        
        return self.session_analyzer.analyze_session(user_id, session_events)
    
    def analyze_user_journey(self, user_id: str, timeframe: timedelta) -> Dict[str, Any]:
        """Analyse le parcours utilisateur sur une période."""
        
        cutoff_time = datetime.now() - timeframe
        user_events = [
            event for event in self.event_streams[user_id]
            if event.timestamp > cutoff_time
        ]
        
        return self.journey_analyzer.analyze_journey(user_id, user_events)

class SystemBehaviorProfiler(BehaviorAnalysisEngine):
    """Profileur spécialisé pour le comportement des systèmes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.performance_analyzer = SystemPerformanceAnalyzer()
        self.resource_analyzer = ResourceUsageAnalyzer()
        
    def analyze_system_performance(self, system_id: str) -> Dict[str, Any]:
        """Analyse les performances d'un système."""
        
        system_events = list(self.event_streams[system_id])
        return self.performance_analyzer.analyze_performance(system_id, system_events)
    
    def analyze_resource_usage(self, system_id: str) -> Dict[str, Any]:
        """Analyse l'usage des ressources d'un système."""
        
        system_events = list(self.event_streams[system_id])
        return self.resource_analyzer.analyze_usage(system_id, system_events)

class DeviationDetector:
    """Détecteur de déviations comportementales avancé."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'sensitivity': 0.8,
            'min_baseline_samples': 100,
            'detection_methods': ['statistical', 'ml', 'ensemble']
        }
        
        self.baseline_models = {}
        self.detection_history = []
        
    def build_baseline(self, entity_id: str, historical_events: List[BehaviorEvent]) -> None:
        """Construit une baseline comportementale."""
        
        if len(historical_events) < self.config['min_baseline_samples']:
            logger.warning(f"Insufficient samples for baseline: {len(historical_events)}")
            return
        
        # Modèle statistique
        statistical_model = self._build_statistical_baseline(historical_events)
        
        # Modèle ML
        ml_model = self._build_ml_baseline(historical_events)
        
        self.baseline_models[entity_id] = {
            'statistical': statistical_model,
            'ml': ml_model,
            'created_at': datetime.now(),
            'sample_count': len(historical_events)
        }
        
        logger.info(f"Baseline built for entity {entity_id} with {len(historical_events)} samples")
    
    def _build_statistical_baseline(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Construit une baseline statistique."""
        
        # Extraction de métriques statistiques
        event_intervals = []
        hourly_counts = defaultdict(int)
        event_types = defaultdict(int)
        
        for i, event in enumerate(events):
            if i > 0:
                interval = (event.timestamp - events[i-1].timestamp).total_seconds()
                event_intervals.append(interval)
            
            hourly_counts[event.timestamp.hour] += 1
            event_types[event.event_type] += 1
        
        return {
            'interval_stats': {
                'mean': np.mean(event_intervals) if event_intervals else 0,
                'std': np.std(event_intervals) if event_intervals else 0,
                'percentiles': {
                    50: np.percentile(event_intervals, 50) if event_intervals else 0,
                    95: np.percentile(event_intervals, 95) if event_intervals else 0,
                    99: np.percentile(event_intervals, 99) if event_intervals else 0
                }
            },
            'hourly_distribution': dict(hourly_counts),
            'event_type_distribution': dict(event_types),
            'total_events': len(events)
        }
    
    def _build_ml_baseline(self, events: List[BehaviorEvent]) -> Optional[Any]:
        """Construit une baseline ML."""
        
        try:
            # Extraction de features
            features = self._extract_ml_features(events)
            
            if len(features) < 10:
                return None
            
            # Modèle d'auto-encodeur simple pour détection d'anomalies
            autoencoder = self._create_autoencoder(features.shape[1])
            
            # Normalisation
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features)
            
            # Entraînement
            autoencoder.fit(
                normalized_features, normalized_features,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            return {
                'model': autoencoder,
                'scaler': scaler,
                'input_shape': features.shape[1]
            }
            
        except Exception as e:
            logger.warning(f"ML baseline creation failed: {e}")
            return None
    
    def _extract_ml_features(self, events: List[BehaviorEvent]) -> np.ndarray:
        """Extrait les features ML des événements."""
        
        # Features similaires à celles de BehaviorAnalysisEngine
        engine = BehaviorAnalysisEngine()
        return engine._extract_behavior_features(events)
    
    def _create_autoencoder(self, input_dim: int) -> Model:
        """Crée un autoencodeur pour la détection d'anomalies."""
        
        encoding_dim = max(2, input_dim // 2)
        
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def detect_deviation(self, entity_id: str, current_events: List[BehaviorEvent]) -> Optional[BehaviorDeviation]:
        """Détecte une déviation comportementale."""
        
        if entity_id not in self.baseline_models:
            return None
        
        baseline = self.baseline_models[entity_id]
        
        # Détection statistique
        statistical_deviation = self._detect_statistical_deviation(current_events, baseline['statistical'])
        
        # Détection ML
        ml_deviation = None
        if baseline['ml']:
            ml_deviation = self._detect_ml_deviation(current_events, baseline['ml'])
        
        # Combinaison des résultats
        return self._combine_detection_results(entity_id, current_events, statistical_deviation, ml_deviation)
    
    def _detect_statistical_deviation(self, events: List[BehaviorEvent], 
                                    baseline: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Détection statistique de déviation."""
        
        if not events:
            return None
        
        # Calcul des métriques actuelles
        current_intervals = []
        current_hourly = defaultdict(int)
        
        for i, event in enumerate(events):
            if i > 0:
                interval = (event.timestamp - events[i-1].timestamp).total_seconds()
                current_intervals.append(interval)
            current_hourly[event.timestamp.hour] += 1
        
        deviations = {}
        
        # Test des intervalles
        if current_intervals and baseline['interval_stats']['std'] > 0:
            current_mean = np.mean(current_intervals)
            baseline_mean = baseline['interval_stats']['mean']
            baseline_std = baseline['interval_stats']['std']
            
            z_score = abs(current_mean - baseline_mean) / baseline_std
            if z_score > 2:  # Déviation significative
                deviations['interval_deviation'] = {
                    'z_score': z_score,
                    'current_mean': current_mean,
                    'baseline_mean': baseline_mean
                }
        
        # Test de distribution horaire
        baseline_hourly = baseline['hourly_distribution']
        for hour, count in current_hourly.items():
            expected_count = baseline_hourly.get(hour, 0)
            if expected_count > 0:
                ratio = count / expected_count
                if ratio > 2 or ratio < 0.5:  # Déviation de plus de 50%
                    deviations[f'hourly_deviation_{hour}'] = {
                        'ratio': ratio,
                        'current_count': count,
                        'expected_count': expected_count
                    }
        
        return deviations if deviations else None
    
    def _detect_ml_deviation(self, events: List[BehaviorEvent], 
                           ml_baseline: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Détection ML de déviation."""
        
        try:
            model = ml_baseline['model']
            scaler = ml_baseline['scaler']
            
            # Extraction des features actuelles
            current_features = self._extract_ml_features(events)
            
            if len(current_features) == 0:
                return None
            
            # Normalisation
            normalized_features = scaler.transform(current_features)
            
            # Prédiction (reconstruction)
            reconstructed = model.predict(normalized_features)
            
            # Calcul de l'erreur de reconstruction
            reconstruction_errors = np.mean(np.square(normalized_features - reconstructed), axis=1)
            
            # Seuil d'anomalie (95ème percentile des erreurs d'entraînement)
            threshold = np.percentile(reconstruction_errors, 95)
            
            max_error = np.max(reconstruction_errors)
            
            if max_error > threshold:
                return {
                    'max_reconstruction_error': max_error,
                    'threshold': threshold,
                    'anomaly_score': max_error / threshold,
                    'anomalous_samples': np.sum(reconstruction_errors > threshold)
                }
        
        except Exception as e:
            logger.warning(f"ML deviation detection failed: {e}")
        
        return None
    
    def _combine_detection_results(self, entity_id: str, events: List[BehaviorEvent],
                                 statistical_result: Optional[Dict[str, Any]],
                                 ml_result: Optional[Dict[str, Any]]) -> Optional[BehaviorDeviation]:
        """Combine les résultats de détection."""
        
        if not statistical_result and not ml_result:
            return None
        
        # Calcul du score de confiance composite
        confidence_score = 0.0
        deviation_metrics = {}
        
        if statistical_result:
            # Score basé sur les z-scores statistiques
            max_z_score = max(
                item.get('z_score', 0) for item in statistical_result.values() 
                if isinstance(item, dict) and 'z_score' in item
            )
            confidence_score += min(1.0, max_z_score / 3) * 0.5
            deviation_metrics.update(statistical_result)
        
        if ml_result:
            # Score basé sur l'erreur de reconstruction
            anomaly_score = ml_result.get('anomaly_score', 0)
            confidence_score += min(1.0, anomaly_score) * 0.5
            deviation_metrics.update(ml_result)
        
        # Détermination du type de déviation
        deviation_type = DeviationType.STATISTICAL_OUTLIER
        if 'interval_deviation' in deviation_metrics:
            deviation_type = DeviationType.FREQUENCY_ANOMALY
        elif any('hourly_deviation' in k for k in deviation_metrics.keys()):
            deviation_type = DeviationType.TEMPORAL_DEVIATION
        
        return BehaviorDeviation(
            deviation_id=f"{entity_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            entity_id=entity_id,
            detection_time=datetime.now(),
            deviation_type=deviation_type,
            severity_score=confidence_score,
            confidence=confidence_score,
            observed_behavior={'recent_events_count': len(events)},
            expected_behavior={'baseline_sample_count': self.baseline_models[entity_id]['sample_count']},
            deviation_metrics=deviation_metrics
        )

# Classes d'analyse spécialisées

class UserSessionAnalyzer:
    """Analyseur de sessions utilisateur."""
    
    def analyze_session(self, user_id: str, session_events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyse une session utilisateur."""
        
        if not session_events:
            return {}
        
        session_start = session_events[0].timestamp
        session_end = session_events[-1].timestamp
        session_duration = (session_end - session_start).total_seconds()
        
        # Analyse des actions
        action_sequence = [event.event_type for event in session_events]
        unique_actions = len(set(action_sequence))
        
        # Analyse des patterns de navigation
        page_views = [event for event in session_events if event.event_type == 'page_view']
        
        return {
            'session_duration_seconds': session_duration,
            'total_events': len(session_events),
            'unique_actions': unique_actions,
            'action_sequence': action_sequence,
            'page_views_count': len(page_views),
            'session_quality_score': self._calculate_session_quality(session_events)
        }
    
    def _calculate_session_quality(self, events: List[BehaviorEvent]) -> float:
        """Calcule un score de qualité de session."""
        
        # Facteurs de qualité
        duration_factor = min(1.0, len(events) / 20)  # Plus d'événements = meilleure qualité
        diversity_factor = len(set(e.event_type for e in events)) / len(events) if events else 0
        
        return (duration_factor + diversity_factor) / 2

class UserJourneyAnalyzer:
    """Analyseur de parcours utilisateur."""
    
    def analyze_journey(self, user_id: str, journey_events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyse le parcours utilisateur."""
        
        if not journey_events:
            return {}
        
        # Segmentation en sessions
        sessions = self._segment_into_sessions(journey_events)
        
        # Analyse des funnel de conversion
        conversion_analysis = self._analyze_conversion_funnel(journey_events)
        
        # Analyse des patterns de retour
        return_patterns = self._analyze_return_patterns(journey_events)
        
        return {
            'total_sessions': len(sessions),
            'avg_session_length': np.mean([len(session) for session in sessions]) if sessions else 0,
            'conversion_analysis': conversion_analysis,
            'return_patterns': return_patterns,
            'journey_quality_score': self._calculate_journey_quality(journey_events)
        }
    
    def _segment_into_sessions(self, events: List[BehaviorEvent]) -> List[List[BehaviorEvent]]:
        """Segmente les événements en sessions."""
        
        sessions = []
        current_session = []
        session_timeout = timedelta(minutes=30)
        
        for event in events:
            if current_session and (event.timestamp - current_session[-1].timestamp) > session_timeout:
                # Nouvelle session
                sessions.append(current_session)
                current_session = [event]
            else:
                current_session.append(event)
        
        if current_session:
            sessions.append(current_session)
        
        return sessions
    
    def _analyze_conversion_funnel(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyse le funnel de conversion."""
        
        funnel_steps = ['page_view', 'product_view', 'add_to_cart', 'purchase']
        funnel_counts = {step: 0 for step in funnel_steps}
        
        for event in events:
            if event.event_type in funnel_counts:
                funnel_counts[event.event_type] += 1
        
        # Calcul des taux de conversion
        conversion_rates = {}
        for i in range(1, len(funnel_steps)):
            current_step = funnel_steps[i]
            previous_step = funnel_steps[i-1]
            
            if funnel_counts[previous_step] > 0:
                conversion_rates[f"{previous_step}_to_{current_step}"] = (
                    funnel_counts[current_step] / funnel_counts[previous_step]
                )
            else:
                conversion_rates[f"{previous_step}_to_{current_step}"] = 0
        
        return {
            'funnel_counts': funnel_counts,
            'conversion_rates': conversion_rates
        }
    
    def _analyze_return_patterns(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyse les patterns de retour."""
        
        # Groupement par jour
        daily_activity = defaultdict(int)
        for event in events:
            day_key = event.timestamp.date()
            daily_activity[day_key] += 1
        
        active_days = len(daily_activity)
        avg_daily_activity = np.mean(list(daily_activity.values())) if daily_activity else 0
        
        return {
            'active_days': active_days,
            'avg_daily_activity': avg_daily_activity,
            'return_frequency': 'high' if active_days > 7 else 'medium' if active_days > 3 else 'low'
        }
    
    def _calculate_journey_quality(self, events: List[BehaviorEvent]) -> float:
        """Calcule un score de qualité de parcours."""
        
        # Diversité des événements
        event_types = set(e.event_type for e in events)
        diversity_score = min(1.0, len(event_types) / 10)
        
        # Engagement (nombre d'événements)
        engagement_score = min(1.0, len(events) / 100)
        
        # Span temporel
        if len(events) > 1:
            time_span = (events[-1].timestamp - events[0].timestamp).total_seconds() / 86400  # jours
            temporal_score = min(1.0, time_span / 30)
        else:
            temporal_score = 0
        
        return (diversity_score + engagement_score + temporal_score) / 3

class SystemPerformanceAnalyzer:
    """Analyseur de performance système."""
    
    def analyze_performance(self, system_id: str, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyse les performances d'un système."""
        
        # Extraction des métriques de performance
        response_times = []
        error_counts = defaultdict(int)
        throughput_data = []
        
        for event in events:
            if 'response_time' in event.attributes:
                response_times.append(event.attributes['response_time'])
            
            if event.event_type == 'error':
                error_type = event.attributes.get('error_type', 'unknown')
                error_counts[error_type] += 1
            
            if 'requests_per_second' in event.attributes:
                throughput_data.append(event.attributes['requests_per_second'])
        
        # Analyse statistique
        performance_stats = {}
        
        if response_times:
            performance_stats['response_time'] = {
                'mean': np.mean(response_times),
                'median': np.median(response_times),
                'p95': np.percentile(response_times, 95),
                'p99': np.percentile(response_times, 99),
                'std': np.std(response_times)
            }
        
        if throughput_data:
            performance_stats['throughput'] = {
                'mean_rps': np.mean(throughput_data),
                'max_rps': np.max(throughput_data),
                'min_rps': np.min(throughput_data)
            }
        
        # Calcul du score de santé
        health_score = self._calculate_system_health_score(response_times, error_counts, throughput_data)
        
        return {
            'performance_stats': performance_stats,
            'error_distribution': dict(error_counts),
            'total_errors': sum(error_counts.values()),
            'health_score': health_score,
            'analysis_period': {
                'start': events[0].timestamp.isoformat() if events else None,
                'end': events[-1].timestamp.isoformat() if events else None,
                'event_count': len(events)
            }
        }
    
    def _calculate_system_health_score(self, response_times: List[float], 
                                     error_counts: Dict[str, int],
                                     throughput_data: List[float]) -> float:
        """Calcule un score de santé du système."""
        
        health_factors = []
        
        # Facteur de temps de réponse
        if response_times:
            avg_response_time = np.mean(response_times)
            response_factor = max(0, 1 - (avg_response_time / 1000))  # 1s = score 0
            health_factors.append(response_factor)
        
        # Facteur d'erreurs
        total_errors = sum(error_counts.values())
        error_rate = total_errors / len(response_times) if response_times else 0
        error_factor = max(0, 1 - error_rate * 10)  # 10% d'erreurs = score 0
        health_factors.append(error_factor)
        
        # Facteur de débit
        if throughput_data:
            throughput_stability = 1 - (np.std(throughput_data) / np.mean(throughput_data))
            health_factors.append(max(0, throughput_stability))
        
        return np.mean(health_factors) if health_factors else 0.5

class ResourceUsageAnalyzer:
    """Analyseur d'usage des ressources."""
    
    def analyze_usage(self, system_id: str, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyse l'usage des ressources."""
        
        # Extraction des métriques de ressources
        cpu_usage = []
        memory_usage = []
        disk_usage = []
        network_usage = []
        
        for event in events:
            attrs = event.attributes
            
            if 'cpu_percent' in attrs:
                cpu_usage.append(attrs['cpu_percent'])
            if 'memory_percent' in attrs:
                memory_usage.append(attrs['memory_percent'])
            if 'disk_percent' in attrs:
                disk_usage.append(attrs['disk_percent'])
            if 'network_mbps' in attrs:
                network_usage.append(attrs['network_mbps'])
        
        # Analyse des tendances
        trends = {}
        for resource_name, usage_data in [
            ('cpu', cpu_usage), ('memory', memory_usage),
            ('disk', disk_usage), ('network', network_usage)
        ]:
            if usage_data:
                trends[resource_name] = self._analyze_resource_trend(usage_data)
        
        # Détection de saturation
        saturation_alerts = self._detect_resource_saturation({
            'cpu': cpu_usage,
            'memory': memory_usage,
            'disk': disk_usage,
            'network': network_usage
        })
        
        return {
            'resource_trends': trends,
            'saturation_alerts': saturation_alerts,
            'overall_usage_score': self._calculate_overall_usage_score(cpu_usage, memory_usage, disk_usage)
        }
    
    def _analyze_resource_trend(self, usage_data: List[float]) -> Dict[str, Any]:
        """Analyse la tendance d'une ressource."""
        
        if len(usage_data) < 2:
            return {'trend': 'insufficient_data'}
        
        # Régression linéaire simple
        x = np.arange(len(usage_data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, usage_data)
        
        # Classification de la tendance
        if abs(slope) < 0.1:
            trend_type = 'stable'
        elif slope > 0:
            trend_type = 'increasing'
        else:
            trend_type = 'decreasing'
        
        return {
            'trend': trend_type,
            'slope': slope,
            'r_squared': r_value ** 2,
            'mean_usage': np.mean(usage_data),
            'max_usage': np.max(usage_data),
            'min_usage': np.min(usage_data),
            'volatility': np.std(usage_data)
        }
    
    def _detect_resource_saturation(self, resource_data: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Détecte la saturation des ressources."""
        
        alerts = []
        thresholds = {
            'cpu': 80,
            'memory': 85,
            'disk': 90,
            'network': 80
        }
        
        for resource_name, usage_data in resource_data.items():
            if not usage_data:
                continue
            
            threshold = thresholds.get(resource_name, 80)
            max_usage = max(usage_data)
            avg_usage = np.mean(usage_data)
            
            if max_usage > threshold:
                severity = 'critical' if max_usage > threshold + 10 else 'warning'
                
                alerts.append({
                    'resource': resource_name,
                    'severity': severity,
                    'max_usage': max_usage,
                    'avg_usage': avg_usage,
                    'threshold': threshold,
                    'recommendation': f"Scale {resource_name} resources" if severity == 'critical' 
                                    else f"Monitor {resource_name} usage closely"
                })
        
        return alerts
    
    def _calculate_overall_usage_score(self, cpu_usage: List[float], 
                                     memory_usage: List[float], 
                                     disk_usage: List[float]) -> float:
        """Calcule un score global d'usage des ressources."""
        
        scores = []
        
        for usage_data in [cpu_usage, memory_usage, disk_usage]:
            if usage_data:
                avg_usage = np.mean(usage_data)
                # Score inversement proportionnel à l'usage (100% = score 0)
                score = max(0, 1 - (avg_usage / 100))
                scores.append(score)
        
        return np.mean(scores) if scores else 0.5
