# 🎵 Spotify AI Agent - Real-Time Analytics Engine
# =================================================
# 
# Moteur d'analytics temps réel avec traitement de flux,
# métriques avancées et insights intelligents.
#
# 🎖️ Expert: Data Engineer + ML Engineer + Analytics Specialist
#
# 👨‍💻 Développé par: Fahed Mlaiel
# =================================================

"""
📊 Real-Time Analytics Engine
============================

Enterprise real-time analytics system providing:
- Stream processing and real-time aggregations
- User behavior analytics and pattern detection
- Music recommendation analytics and A/B testing
- Performance monitoring and business metrics
- Predictive analytics and trend analysis
- Real-time dashboards and alerting
- Data pipeline orchestration
- GDPR-compliant analytics with privacy controls
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Union, Callable, Tuple
import statistics
import math

# Data processing
import numpy as np
import pandas as pd
from scipy import stats
import aioredis

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Time series
import asyncio
from collections import OrderedDict

# Monitoring
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types de métriques"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    SET = "set"


class AggregationType(Enum):
    """Types d'agrégation"""
    SUM = "sum"
    COUNT = "count"
    AVERAGE = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    STDDEV = "stddev"
    RATE = "rate"
    UNIQUE = "unique"


class EventCategory(Enum):
    """Catégories d'événements analytiques"""
    USER_BEHAVIOR = "user_behavior"
    MUSIC_INTERACTION = "music_interaction"
    SYSTEM_PERFORMANCE = "system_performance"
    BUSINESS_METRIC = "business_metric"
    SECURITY = "security"
    ERROR = "error"


@dataclass
class AnalyticsEvent:
    """Événement analytique"""
    event_id: str
    category: EventCategory
    event_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    
    # Données de l'événement
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Contexte
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tenant_id: str = "default"
    device_type: Optional[str] = None
    platform: Optional[str] = None
    app_version: Optional[str] = None
    
    # Géolocalisation
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    
    # Privacy
    is_anonymous: bool = False
    consent_given: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "event_id": self.event_id,
            "category": self.category.value,
            "event_type": self.event_type,
            "user_id": self.user_id if self.consent_given else None,
            "session_id": self.session_id,
            "properties": self.properties,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "device_type": self.device_type,
            "platform": self.platform,
            "app_version": self.app_version,
            "country": self.country,
            "region": self.region,
            "city": self.city,
            "is_anonymous": self.is_anonymous,
            "consent_given": self.consent_given
        }


@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    type: MetricType
    aggregation: AggregationType
    dimensions: List[str] = field(default_factory=list)
    window_size: int = 300  # 5 minutes par défaut
    retention_days: int = 7
    sampling_rate: float = 1.0
    
    # Alertes
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    # Privacy
    requires_consent: bool = True
    anonymizable: bool = True


@dataclass
class TimeSeries:
    """Série temporelle pour métriques"""
    metric_name: str
    dimension_values: Dict[str, str] = field(default_factory=dict)
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_point(self, timestamp: datetime, value: float):
        """Ajoute un point de données"""
        self.data_points.append((timestamp, value))
    
    def get_values(self, start_time: Optional[datetime] = None) -> List[Tuple[datetime, float]]:
        """Récupère les valeurs dans une plage de temps"""
        if start_time is None:
            return list(self.data_points)
        
        return [(ts, val) for ts, val in self.data_points if ts >= start_time]
    
    def calculate_aggregation(self, 
                            aggregation: AggregationType,
                            start_time: Optional[datetime] = None) -> Optional[float]:
        """Calcule une agrégation sur la série"""
        values = [val for ts, val in self.get_values(start_time)]
        
        if not values:
            return None
        
        if aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.COUNT:
            return len(values)
        elif aggregation == AggregationType.AVERAGE:
            return statistics.mean(values)
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)
        elif aggregation == AggregationType.MEDIAN:
            return statistics.median(values)
        elif aggregation == AggregationType.STDDEV:
            return statistics.stdev(values) if len(values) > 1 else 0
        elif aggregation == AggregationType.RATE:
            # Calculer le taux par seconde
            if len(values) < 2:
                return 0
            time_span = (self.data_points[-1][0] - self.data_points[0][0]).total_seconds()
            return sum(values) / max(time_span, 1)
        
        return None


class StreamProcessor:
    """Processeur de flux temps réel"""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.event_windows: Dict[str, deque] = defaultdict(lambda: deque())
        self.processors: List[Callable] = []
        
        # Métriques de performance
        self.processed_events = 0
        self.processing_times = deque(maxlen=1000)
        
    def register_processor(self, processor: Callable[[List[AnalyticsEvent]], Dict[str, Any]]):
        """Enregistre un processeur d'événements"""
        self.processors.append(processor)
    
    async def process_event(self, event: AnalyticsEvent) -> Dict[str, Any]:
        """Traite un événement"""
        start_time = time.time()
        
        # Ajouter à la fenêtre
        window_key = f"{event.category.value}:{event.event_type}"
        self.event_windows[window_key].append(event)
        
        # Nettoyer les anciens événements
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.window_size)
        while (self.event_windows[window_key] and 
               self.event_windows[window_key][0].timestamp < cutoff_time):
            self.event_windows[window_key].popleft()
        
        # Exécuter les processeurs
        results = {}
        window_events = list(self.event_windows[window_key])
        
        for i, processor in enumerate(self.processors):
            try:
                processor_result = processor(window_events)
                results[f"processor_{i}"] = processor_result
            except Exception as e:
                logger.error("Processor error", processor_index=i, error=str(e))
        
        # Métriques de performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.processed_events += 1
        
        return results
    
    def get_window_stats(self, window_key: str) -> Dict[str, Any]:
        """Statistiques d'une fenêtre"""
        events = list(self.event_windows[window_key])
        
        if not events:
            return {"count": 0}
        
        return {
            "count": len(events),
            "oldest_event": events[0].timestamp.isoformat(),
            "newest_event": events[-1].timestamp.isoformat(),
            "unique_users": len(set(e.user_id for e in events if e.user_id)),
            "unique_sessions": len(set(e.session_id for e in events if e.session_id))
        }


class UserBehaviorAnalyzer:
    """Analyseur de comportement utilisateur"""
    
    def __init__(self):
        self.user_sessions: Dict[str, Dict] = {}
        self.user_metrics: Dict[str, Dict] = defaultdict(dict)
        
        # ML Models
        self.engagement_model = None
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.user_clusters = KMeans(n_clusters=5)
        
    async def analyze_user_event(self, event: AnalyticsEvent) -> Dict[str, Any]:
        """Analyse un événement utilisateur"""
        if not event.user_id or not event.consent_given:
            return {}
        
        user_id = event.user_id
        
        # Mettre à jour la session
        if event.session_id:
            await self._update_user_session(user_id, event)
        
        # Calculer les métriques d'engagement
        engagement_score = await self._calculate_engagement_score(user_id, event)
        
        # Détecter les anomalies
        is_anomaly = await self._detect_anomaly(user_id, event)
        
        # Segmentation utilisateur
        user_segment = await self._get_user_segment(user_id)
        
        return {
            "user_id": user_id,
            "engagement_score": engagement_score,
            "is_anomaly": is_anomaly,
            "user_segment": user_segment,
            "session_duration": self._get_session_duration(user_id),
            "events_in_session": self._get_session_event_count(user_id)
        }
    
    async def _update_user_session(self, user_id: str, event: AnalyticsEvent):
        """Met à jour la session utilisateur"""
        session_id = event.session_id
        
        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = {
                "user_id": user_id,
                "start_time": event.timestamp,
                "last_activity": event.timestamp,
                "events": [],
                "pages_visited": set(),
                "actions_performed": []
            }
        
        session = self.user_sessions[session_id]
        session["last_activity"] = event.timestamp
        session["events"].append(event)
        
        # Analyser le type d'action
        if event.event_type in ["page_view", "screen_view"]:
            session["pages_visited"].add(event.properties.get("page_name", "unknown"))
        elif event.event_type in ["click", "tap", "play", "pause"]:
            session["actions_performed"].append(event.event_type)
    
    async def _calculate_engagement_score(self, user_id: str, event: AnalyticsEvent) -> float:
        """Calcule le score d'engagement"""
        # Score basé sur différents facteurs
        score = 0.0
        
        # Type d'événement (pondération)
        event_weights = {
            "music_play": 1.0,
            "track_complete": 2.0,
            "playlist_create": 1.5,
            "social_share": 2.0,
            "comment": 1.8,
            "like": 1.2,
            "page_view": 0.3
        }
        
        score += event_weights.get(event.event_type, 0.5)
        
        # Durée de session
        session_duration = self._get_session_duration(user_id)
        if session_duration:
            # Score augmente avec la durée, mais avec rendements décroissants
            score += min(2.0, math.log(session_duration + 1) / 3)
        
        # Fréquence d'activité
        recent_events = self._get_recent_user_events(user_id, hours=24)
        if recent_events:
            # Plus d'événements récents = plus d'engagement
            score += min(1.0, len(recent_events) / 20)
        
        # Normaliser entre 0 et 1
        return min(1.0, score / 5.0)
    
    async def _detect_anomaly(self, user_id: str, event: AnalyticsEvent) -> bool:
        """Détecte des anomalies dans le comportement"""
        # Features pour la détection d'anomalie
        features = self._extract_behavior_features(user_id, event)
        
        if len(features) < 5:  # Pas assez de données
            return False
        
        # Utiliser le modèle d'isolation forest
        try:
            prediction = self.anomaly_detector.predict([features])
            return prediction[0] == -1  # -1 = anomalie
        except:
            return False
    
    def _extract_behavior_features(self, user_id: str, event: AnalyticsEvent) -> List[float]:
        """Extrait les features de comportement"""
        # Heure de la journée
        hour_of_day = event.timestamp.hour
        
        # Jour de la semaine
        day_of_week = event.timestamp.weekday()
        
        # Durée depuis la dernière activité
        last_activity = self.user_metrics[user_id].get("last_activity")
        time_since_last = 0
        if last_activity:
            time_since_last = (event.timestamp - last_activity).total_seconds() / 3600
        
        # Nombre d'événements dans la dernière heure
        recent_events = self._get_recent_user_events(user_id, hours=1)
        events_last_hour = len(recent_events) if recent_events else 0
        
        # Type d'événement encodé
        event_type_encoded = hash(event.event_type) % 100
        
        return [
            hour_of_day,
            day_of_week,
            time_since_last,
            events_last_hour,
            event_type_encoded
        ]
    
    async def _get_user_segment(self, user_id: str) -> str:
        """Détermine le segment utilisateur"""
        user_features = self._get_user_features(user_id)
        
        if not user_features:
            return "new_user"
        
        # Segmentation basique
        engagement = user_features.get("avg_engagement", 0)
        activity_frequency = user_features.get("daily_events", 0)
        
        if engagement > 0.7 and activity_frequency > 20:
            return "power_user"
        elif engagement > 0.5 and activity_frequency > 10:
            return "regular_user"
        elif engagement > 0.3:
            return "casual_user"
        else:
            return "inactive_user"
    
    def _get_user_features(self, user_id: str) -> Dict[str, float]:
        """Récupère les features utilisateur"""
        return self.user_metrics.get(user_id, {})
    
    def _get_session_duration(self, user_id: str) -> Optional[float]:
        """Récupère la durée de session"""
        # Trouver la session active
        for session in self.user_sessions.values():
            if session["user_id"] == user_id:
                duration = (session["last_activity"] - session["start_time"]).total_seconds()
                return duration
        return None
    
    def _get_session_event_count(self, user_id: str) -> int:
        """Compte les événements dans la session"""
        for session in self.user_sessions.values():
            if session["user_id"] == user_id:
                return len(session["events"])
        return 0
    
    def _get_recent_user_events(self, user_id: str, hours: int = 24) -> List[AnalyticsEvent]:
        """Récupère les événements récents d'un utilisateur"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_events = []
        
        for session in self.user_sessions.values():
            if session["user_id"] == user_id:
                for event in session["events"]:
                    if event.timestamp >= cutoff:
                        recent_events.append(event)
        
        return recent_events


class MusicAnalytics:
    """Analytics spécifiques à la musique"""
    
    def __init__(self):
        self.track_stats: Dict[str, Dict] = defaultdict(dict)
        self.playlist_stats: Dict[str, Dict] = defaultdict(dict)
        self.genre_trends: Dict[str, List] = defaultdict(list)
        
    async def analyze_music_event(self, event: AnalyticsEvent) -> Dict[str, Any]:
        """Analyse un événement musical"""
        analysis = {}
        
        if event.event_type == "track_play":
            analysis.update(await self._analyze_track_play(event))
        elif event.event_type == "track_skip":
            analysis.update(await self._analyze_track_skip(event))
        elif event.event_type == "playlist_play":
            analysis.update(await self._analyze_playlist_play(event))
        elif event.event_type == "track_complete":
            analysis.update(await self._analyze_track_complete(event))
        
        return analysis
    
    async def _analyze_track_play(self, event: AnalyticsEvent) -> Dict[str, Any]:
        """Analyse la lecture d'un track"""
        track_id = event.properties.get("track_id")
        if not track_id:
            return {}
        
        # Mettre à jour les stats du track
        stats = self.track_stats[track_id]
        stats["total_plays"] = stats.get("total_plays", 0) + 1
        stats["last_played"] = event.timestamp
        
        # Analyser l'heure de la journée
        hour = event.timestamp.hour
        stats.setdefault("hourly_plays", {})
        stats["hourly_plays"][str(hour)] = stats["hourly_plays"].get(str(hour), 0) + 1
        
        # Analyser la plateforme
        if event.platform:
            stats.setdefault("platform_plays", {})
            stats["platform_plays"][event.platform] = stats["platform_plays"].get(event.platform, 0) + 1
        
        # Calculer la popularité
        popularity_score = self._calculate_track_popularity(track_id)
        
        return {
            "track_id": track_id,
            "total_plays": stats["total_plays"],
            "popularity_score": popularity_score,
            "trending": self._is_track_trending(track_id)
        }
    
    async def _analyze_track_skip(self, event: AnalyticsEvent) -> Dict[str, Any]:
        """Analyse le skip d'un track"""
        track_id = event.properties.get("track_id")
        skip_position = event.properties.get("position", 0)  # Position en secondes
        
        if not track_id:
            return {}
        
        stats = self.track_stats[track_id]
        stats["total_skips"] = stats.get("total_skips", 0) + 1
        
        # Analyser la position de skip
        stats.setdefault("skip_positions", [])
        stats["skip_positions"].append(skip_position)
        
        # Calculer le taux de skip
        total_plays = stats.get("total_plays", 0)
        skip_rate = stats["total_skips"] / max(total_plays, 1)
        
        return {
            "track_id": track_id,
            "skip_rate": skip_rate,
            "avg_skip_position": statistics.mean(stats["skip_positions"]) if stats["skip_positions"] else 0
        }
    
    async def _analyze_track_complete(self, event: AnalyticsEvent) -> Dict[str, Any]:
        """Analyse la fin d'un track"""
        track_id = event.properties.get("track_id")
        duration_listened = event.properties.get("duration_listened", 0)
        
        if not track_id:
            return {}
        
        stats = self.track_stats[track_id]
        stats["total_completes"] = stats.get("total_completes", 0) + 1
        
        # Calculer le taux de completion
        total_plays = stats.get("total_plays", 0)
        completion_rate = stats["total_completes"] / max(total_plays, 1)
        
        # Analyser la durée d'écoute
        stats.setdefault("listen_durations", [])
        stats["listen_durations"].append(duration_listened)
        
        return {
            "track_id": track_id,
            "completion_rate": completion_rate,
            "avg_listen_duration": statistics.mean(stats["listen_durations"])
        }
    
    async def _analyze_playlist_play(self, event: AnalyticsEvent) -> Dict[str, Any]:
        """Analyse la lecture d'une playlist"""
        playlist_id = event.properties.get("playlist_id")
        if not playlist_id:
            return {}
        
        stats = self.playlist_stats[playlist_id]
        stats["total_plays"] = stats.get("total_plays", 0) + 1
        stats["last_played"] = event.timestamp
        
        return {
            "playlist_id": playlist_id,
            "total_plays": stats["total_plays"]
        }
    
    def _calculate_track_popularity(self, track_id: str) -> float:
        """Calcule le score de popularité d'un track"""
        stats = self.track_stats[track_id]
        
        # Facteurs de popularité
        plays = stats.get("total_plays", 0)
        completes = stats.get("total_completes", 0)
        skips = stats.get("total_skips", 0)
        
        # Score basé sur le ratio completion/skip
        completion_rate = completes / max(plays, 1)
        skip_rate = skips / max(plays, 1)
        
        # Pondération temporelle (récence)
        last_played = stats.get("last_played")
        recency_factor = 1.0
        if last_played:
            days_ago = (datetime.utcnow() - last_played).days
            recency_factor = max(0.1, 1.0 - (days_ago / 30))  # Décroissance sur 30 jours
        
        # Score final
        popularity = (completion_rate * 0.6 - skip_rate * 0.4) * math.log(plays + 1) * recency_factor
        
        return max(0.0, min(1.0, popularity))
    
    def _is_track_trending(self, track_id: str) -> bool:
        """Détermine si un track est trending"""
        stats = self.track_stats[track_id]
        
        # Un track est trending s'il a beaucoup de plays récentes
        last_played = stats.get("last_played")
        if not last_played:
            return False
        
        # Vérifier l'activité des dernières 24h
        recent_threshold = datetime.utcnow() - timedelta(hours=24)
        if last_played < recent_threshold:
            return False
        
        # Comparer avec la moyenne des plays
        total_plays = stats.get("total_plays", 0)
        
        # Logique simplifiée: trending si plus de 100 plays et récent
        return total_plays > 100 and last_played >= recent_threshold


class PerformanceMonitor:
    """Moniteur de performance système"""
    
    def __init__(self):
        self.metrics: Dict[str, TimeSeries] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        # Seuils d'alerte
        self.thresholds = {
            "response_time": {"warning": 1.0, "critical": 3.0},
            "error_rate": {"warning": 0.05, "critical": 0.1},
            "cpu_usage": {"warning": 0.8, "critical": 0.95},
            "memory_usage": {"warning": 0.8, "critical": 0.95}
        }
    
    async def record_metric(self, 
                          metric_name: str,
                          value: float,
                          dimensions: Dict[str, str] = None) -> Dict[str, Any]:
        """Enregistre une métrique de performance"""
        dimensions = dimensions or {}
        
        # Créer la clé unique pour cette série
        series_key = f"{metric_name}:" + ":".join(f"{k}={v}" for k, v in sorted(dimensions.items()))
        
        # Créer la série si elle n'existe pas
        if series_key not in self.metrics:
            self.metrics[series_key] = TimeSeries(metric_name, dimensions)
        
        # Ajouter le point de données
        self.metrics[series_key].add_point(datetime.utcnow(), value)
        
        # Vérifier les seuils d'alerte
        alert = await self._check_thresholds(metric_name, value, dimensions)
        
        return {
            "metric_name": metric_name,
            "value": value,
            "dimensions": dimensions,
            "alert": alert
        }
    
    async def _check_thresholds(self, 
                              metric_name: str,
                              value: float,
                              dimensions: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Vérifie les seuils d'alerte"""
        thresholds = self.thresholds.get(metric_name)
        if not thresholds:
            return None
        
        alert_level = None
        if value >= thresholds.get("critical", float('inf')):
            alert_level = "critical"
        elif value >= thresholds.get("warning", float('inf')):
            alert_level = "warning"
        
        if alert_level:
            alert = {
                "id": str(uuid.uuid4()),
                "level": alert_level,
                "metric_name": metric_name,
                "value": value,
                "threshold": thresholds[alert_level],
                "dimensions": dimensions,
                "timestamp": datetime.utcnow(),
                "message": f"{metric_name} is {alert_level}: {value} >= {thresholds[alert_level]}"
            }
            
            self.alerts.append(alert)
            
            logger.warning("Performance alert", 
                          level=alert_level,
                          metric=metric_name,
                          value=value)
            
            return alert
        
        return None
    
    def get_metric_stats(self, 
                        metric_name: str,
                        dimensions: Dict[str, str] = None,
                        start_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Récupère les statistiques d'une métrique"""
        dimensions = dimensions or {}
        series_key = f"{metric_name}:" + ":".join(f"{k}={v}" for k, v in sorted(dimensions.items()))
        
        if series_key not in self.metrics:
            return {}
        
        series = self.metrics[series_key]
        values = [val for ts, val in series.get_values(start_time)]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0,
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }
    
    def get_active_alerts(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère les alertes actives"""
        if level:
            return [alert for alert in self.alerts if alert["level"] == level]
        return self.alerts.copy()


class RealTimeAnalyticsEngine:
    """Moteur principal d'analytics temps réel"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Composants d'analyse
        self.stream_processor = StreamProcessor()
        self.user_analyzer = UserBehaviorAnalyzer()
        self.music_analytics = MusicAnalytics()
        self.performance_monitor = PerformanceMonitor()
        
        # Métriques et configuration
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.time_series: Dict[str, TimeSeries] = {}
        
        # Queues et buffers
        self.event_queue = asyncio.Queue(maxsize=10000)
        self.batch_buffer: List[AnalyticsEvent] = []
        self.batch_size = 100
        self.flush_interval = 30  # secondes
        
        # Tâches en arrière-plan
        self.background_tasks: List[asyncio.Task] = []
        
        # Métriques Prometheus
        self.prometheus_metrics = {
            "events_processed": Counter("analytics_events_processed_total", "Total events processed", ["category", "type"]),
            "processing_latency": Histogram("analytics_processing_latency_seconds", "Processing latency"),
            "queue_size": Gauge("analytics_queue_size", "Queue size"),
            "batch_size": Histogram("analytics_batch_size", "Batch processing size")
        }
    
    async def initialize(self):
        """Initialise le moteur d'analytics"""
        try:
            # Connexion Redis
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Enregistrer les métriques par défaut
            self._register_default_metrics()
            
            # Démarrer les workers
            await self._start_background_workers()
            
            logger.info("Real-time analytics engine initialized")
            
        except Exception as e:
            logger.error("Failed to initialize analytics engine", error=str(e))
            raise
    
    def _register_default_metrics(self):
        """Enregistre les métriques par défaut"""
        default_metrics = [
            MetricDefinition(
                name="user_engagement",
                type=MetricType.GAUGE,
                aggregation=AggregationType.AVERAGE,
                dimensions=["user_segment", "platform"]
            ),
            MetricDefinition(
                name="track_plays",
                type=MetricType.COUNTER,
                aggregation=AggregationType.SUM,
                dimensions=["track_id", "genre", "platform"]
            ),
            MetricDefinition(
                name="session_duration",
                type=MetricType.HISTOGRAM,
                aggregation=AggregationType.AVERAGE,
                dimensions=["user_segment", "device_type"]
            ),
            MetricDefinition(
                name="response_time",
                type=MetricType.HISTOGRAM,
                aggregation=AggregationType.PERCENTILE,
                dimensions=["endpoint", "method"],
                threshold_warning=1.0,
                threshold_critical=3.0
            ),
            MetricDefinition(
                name="error_rate",
                type=MetricType.GAUGE,
                aggregation=AggregationType.RATE,
                dimensions=["service", "error_type"],
                threshold_warning=0.05,
                threshold_critical=0.1
            )
        ]
        
        for metric_def in default_metrics:
            self.metric_definitions[metric_def.name] = metric_def
    
    async def track_event(self, event: AnalyticsEvent) -> Dict[str, Any]:
        """Enregistre un événement d'analytics"""
        try:
            # Ajouter à la queue pour traitement
            await self.event_queue.put(event)
            
            # Métriques Prometheus
            self.prometheus_metrics["events_processed"].labels(
                category=event.category.value,
                type=event.event_type
            ).inc()
            
            self.prometheus_metrics["queue_size"].set(self.event_queue.qsize())
            
            logger.debug("Event tracked", 
                        event_id=event.event_id,
                        category=event.category.value,
                        type=event.event_type)
            
            return {"status": "queued", "event_id": event.event_id}
            
        except Exception as e:
            logger.error("Failed to track event", event_id=event.event_id, error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def _start_background_workers(self):
        """Démarre les workers en arrière-plan"""
        # Worker principal de traitement
        main_worker = asyncio.create_task(self._event_processing_worker())
        self.background_tasks.append(main_worker)
        
        # Worker de flush des batches
        batch_worker = asyncio.create_task(self._batch_flush_worker())
        self.background_tasks.append(batch_worker)
        
        # Worker de nettoyage
        cleanup_worker = asyncio.create_task(self._cleanup_worker())
        self.background_tasks.append(cleanup_worker)
        
        # Worker de calcul des métriques agrégées
        aggregation_worker = asyncio.create_task(self._aggregation_worker())
        self.background_tasks.append(aggregation_worker)
        
        logger.info("Analytics background workers started")
    
    async def _event_processing_worker(self):
        """Worker principal de traitement d'événements"""
        while True:
            try:
                # Récupérer un événement
                event = await self.event_queue.get()
                
                start_time = time.time()
                
                # Traiter l'événement
                results = await self._process_single_event(event)
                
                # Ajouter au batch pour persistance
                self.batch_buffer.append(event)
                
                # Flush si batch plein
                if len(self.batch_buffer) >= self.batch_size:
                    await self._flush_batch()
                
                # Métriques de performance
                processing_time = time.time() - start_time
                self.prometheus_metrics["processing_latency"].observe(processing_time)
                
                # Marquer comme terminé
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error("Event processing worker error", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_single_event(self, event: AnalyticsEvent) -> Dict[str, Any]:
        """Traite un événement unique"""
        results = {}
        
        try:
            # Traitement par le stream processor
            stream_results = await self.stream_processor.process_event(event)
            results["stream"] = stream_results
            
            # Analyse comportementale
            if event.category == EventCategory.USER_BEHAVIOR:
                behavior_results = await self.user_analyzer.analyze_user_event(event)
                results["behavior"] = behavior_results
            
            # Analytics musicales
            if event.category == EventCategory.MUSIC_INTERACTION:
                music_results = await self.music_analytics.analyze_music_event(event)
                results["music"] = music_results
            
            # Monitoring de performance
            if event.category == EventCategory.SYSTEM_PERFORMANCE:
                for metric_name, value in event.metrics.items():
                    perf_results = await self.performance_monitor.record_metric(
                        metric_name, value, event.properties
                    )
                    results["performance"] = perf_results
            
            # Mettre à jour les time series
            await self._update_time_series(event)
            
        except Exception as e:
            logger.error("Event processing error", event_id=event.event_id, error=str(e))
            results["error"] = str(e)
        
        return results
    
    async def _update_time_series(self, event: AnalyticsEvent):
        """Met à jour les séries temporelles"""
        # Pour chaque métrique de l'événement
        for metric_name, value in event.metrics.items():
            metric_def = self.metric_definitions.get(metric_name)
            if not metric_def:
                continue
            
            # Créer la clé de série avec dimensions
            dimensions = {}
            for dim in metric_def.dimensions:
                if dim in event.properties:
                    dimensions[dim] = str(event.properties[dim])
            
            series_key = f"{metric_name}:" + ":".join(f"{k}={v}" for k, v in sorted(dimensions.items()))
            
            # Créer ou récupérer la série
            if series_key not in self.time_series:
                self.time_series[series_key] = TimeSeries(metric_name, dimensions)
            
            # Ajouter le point
            self.time_series[series_key].add_point(event.timestamp, value)
    
    async def _batch_flush_worker(self):
        """Worker de flush périodique des batches"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                
                if self.batch_buffer:
                    await self._flush_batch()
                
            except Exception as e:
                logger.error("Batch flush worker error", error=str(e))
    
    async def _flush_batch(self):
        """Flush un batch d'événements vers le stockage"""
        if not self.batch_buffer:
            return
        
        try:
            batch_size = len(self.batch_buffer)
            
            # Sérialiser les événements
            events_data = [event.to_dict() for event in self.batch_buffer]
            
            # Stocker dans Redis
            if self.redis_client:
                pipe = self.redis_client.pipeline()
                
                for event_data in events_data:
                    # Stocker l'événement individuel
                    event_key = f"event:{event_data['event_id']}"
                    pipe.setex(event_key, 86400 * 7, json.dumps(event_data))  # 7 jours
                    
                    # Ajouter aux index
                    if event_data['user_id']:
                        pipe.lpush(f"user_events:{event_data['user_id']}", event_data['event_id'])
                        pipe.expire(f"user_events:{event_data['user_id']}", 86400 * 30)  # 30 jours
                    
                    # Index par catégorie
                    pipe.lpush(f"events:{event_data['category']}", event_data['event_id'])
                    pipe.expire(f"events:{event_data['category']}", 86400 * 7)  # 7 jours
                
                await pipe.execute()
            
            # Vider le buffer
            self.batch_buffer.clear()
            
            # Métriques
            self.prometheus_metrics["batch_size"].observe(batch_size)
            
            logger.debug("Batch flushed", batch_size=batch_size)
            
        except Exception as e:
            logger.error("Batch flush error", error=str(e))
    
    async def _cleanup_worker(self):
        """Worker de nettoyage des données expirées"""
        while True:
            try:
                await asyncio.sleep(3600)  # Toutes les heures
                
                # Nettoyer les anciennes données des time series
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                
                for series in self.time_series.values():
                    # Supprimer les anciens points
                    while (series.data_points and 
                           series.data_points[0][0] < cutoff_time):
                        series.data_points.popleft()
                
                logger.debug("Cleanup completed")
                
            except Exception as e:
                logger.error("Cleanup worker error", error=str(e))
    
    async def _aggregation_worker(self):
        """Worker de calcul des agrégations"""
        while True:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                # Calculer les agrégations pour chaque métrique
                for metric_name, metric_def in self.metric_definitions.items():
                    await self._calculate_aggregations(metric_name, metric_def)
                
            except Exception as e:
                logger.error("Aggregation worker error", error=str(e))
    
    async def _calculate_aggregations(self, metric_name: str, metric_def: MetricDefinition):
        """Calcule les agrégations pour une métrique"""
        try:
            # Trouver toutes les séries pour cette métrique
            metric_series = [
                series for key, series in self.time_series.items()
                if series.metric_name == metric_name
            ]
            
            for series in metric_series:
                # Calculer l'agrégation
                aggregated_value = series.calculate_aggregation(metric_def.aggregation)
                
                if aggregated_value is not None:
                    # Stocker l'agrégation dans Redis
                    agg_key = f"agg:{metric_name}:" + ":".join(
                        f"{k}={v}" for k, v in sorted(series.dimension_values.items())
                    )
                    
                    agg_data = {
                        "value": aggregated_value,
                        "timestamp": datetime.utcnow().isoformat(),
                        "aggregation": metric_def.aggregation.value,
                        "window_size": metric_def.window_size
                    }
                    
                    if self.redis_client:
                        await self.redis_client.setex(
                            agg_key, 
                            metric_def.window_size * 2,  # TTL double de la fenêtre
                            json.dumps(agg_data)
                        )
            
        except Exception as e:
            logger.error("Aggregation calculation error", 
                        metric_name=metric_name, 
                        error=str(e))
    
    async def get_metric_value(self, 
                             metric_name: str,
                             dimensions: Dict[str, str] = None,
                             aggregation: Optional[AggregationType] = None) -> Optional[float]:
        """Récupère la valeur d'une métrique"""
        dimensions = dimensions or {}
        
        # Chercher dans les agrégations stockées
        agg_key = f"agg:{metric_name}:" + ":".join(f"{k}={v}" for k, v in sorted(dimensions.items()))
        
        if self.redis_client:
            agg_data = await self.redis_client.get(agg_key)
            if agg_data:
                return json.loads(agg_data)["value"]
        
        # Sinon calculer en temps réel
        series_key = f"{metric_name}:" + ":".join(f"{k}={v}" for k, v in sorted(dimensions.items()))
        
        if series_key in self.time_series:
            series = self.time_series[series_key]
            metric_def = self.metric_definitions.get(metric_name)
            
            if metric_def:
                return series.calculate_aggregation(aggregation or metric_def.aggregation)
        
        return None
    
    async def get_dashboard_data(self, dashboard_name: str) -> Dict[str, Any]:
        """Récupère les données pour un dashboard"""
        if dashboard_name == "overview":
            return await self._get_overview_dashboard()
        elif dashboard_name == "music":
            return await self._get_music_dashboard()
        elif dashboard_name == "users":
            return await self._get_users_dashboard()
        elif dashboard_name == "performance":
            return await self._get_performance_dashboard()
        else:
            return {}
    
    async def _get_overview_dashboard(self) -> Dict[str, Any]:
        """Données du dashboard overview"""
        return {
            "active_users": await self.get_metric_value("active_users"),
            "total_events": await self.get_metric_value("total_events"),
            "avg_session_duration": await self.get_metric_value("session_duration"),
            "top_tracks": await self._get_top_tracks(limit=10),
            "user_segments": await self._get_user_segments(),
            "real_time_activity": await self._get_real_time_activity()
        }
    
    async def _get_music_dashboard(self) -> Dict[str, Any]:
        """Données du dashboard musical"""
        return {
            "total_plays": await self.get_metric_value("track_plays"),
            "top_genres": await self._get_top_genres(),
            "trending_tracks": await self._get_trending_tracks(),
            "playlist_stats": await self._get_playlist_stats(),
            "skip_rate": await self.get_metric_value("skip_rate"),
            "completion_rate": await self.get_metric_value("completion_rate")
        }
    
    async def _get_users_dashboard(self) -> Dict[str, Any]:
        """Données du dashboard utilisateurs"""
        return {
            "total_users": await self.get_metric_value("total_users"),
            "new_users": await self.get_metric_value("new_users"),
            "user_engagement": await self.get_metric_value("user_engagement"),
            "retention_rate": await self.get_metric_value("retention_rate"),
            "user_segments": await self._get_user_segments(),
            "geographic_distribution": await self._get_geographic_distribution()
        }
    
    async def _get_performance_dashboard(self) -> Dict[str, Any]:
        """Données du dashboard performance"""
        return {
            "response_time": self.performance_monitor.get_metric_stats("response_time"),
            "error_rate": await self.get_metric_value("error_rate"),
            "throughput": await self.get_metric_value("throughput"),
            "active_alerts": self.performance_monitor.get_active_alerts(),
            "system_health": await self._get_system_health()
        }
    
    async def _get_top_tracks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère les tracks les plus populaires"""
        # Utiliser les stats du music analytics
        tracks = []
        for track_id, stats in self.music_analytics.track_stats.items():
            tracks.append({
                "track_id": track_id,
                "total_plays": stats.get("total_plays", 0),
                "popularity_score": self.music_analytics._calculate_track_popularity(track_id)
            })
        
        # Trier par popularité
        tracks.sort(key=lambda x: x["popularity_score"], reverse=True)
        return tracks[:limit]
    
    async def _get_user_segments(self) -> Dict[str, int]:
        """Récupère la distribution des segments utilisateurs"""
        # Analyser les segments depuis le user analyzer
        segments = Counter()
        
        for user_id in self.user_analyzer.user_metrics.keys():
            segment = await self.user_analyzer._get_user_segment(user_id)
            segments[segment] += 1
        
        return dict(segments)
    
    async def _get_real_time_activity(self) -> Dict[str, Any]:
        """Récupère l'activité temps réel"""
        # Compter les événements des dernières minutes
        now = datetime.utcnow()
        activity = {}
        
        for window_key, events in self.stream_processor.event_windows.items():
            # Compter les événements de la dernière minute
            recent_events = [
                e for e in events 
                if (now - e.timestamp).total_seconds() <= 60
            ]
            activity[window_key] = len(recent_events)
        
        return activity
    
    async def shutdown(self):
        """Arrête le moteur d'analytics"""
        # Arrêter les workers
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Flush final
        if self.batch_buffer:
            await self._flush_batch()
        
        # Fermer les connexions
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Analytics engine shutdown completed")


# Factory function
async def create_analytics_engine(redis_url: str = "redis://localhost:6379") -> RealTimeAnalyticsEngine:
    """Crée et initialise le moteur d'analytics"""
    engine = RealTimeAnalyticsEngine(redis_url)
    await engine.initialize()
    return engine


# Instance globale
analytics_engine: Optional[RealTimeAnalyticsEngine] = None


# Fonctions utilitaires
async def track_user_event(user_id: str,
                         event_type: str,
                         properties: Dict[str, Any] = None,
                         session_id: Optional[str] = None) -> str:
    """Fonction utilitaire pour tracker un événement utilisateur"""
    
    if not analytics_engine:
        raise RuntimeError("Analytics engine not initialized")
    
    event = AnalyticsEvent(
        event_id=str(uuid.uuid4()),
        category=EventCategory.USER_BEHAVIOR,
        event_type=event_type,
        user_id=user_id,
        session_id=session_id,
        properties=properties or {}
    )
    
    result = await analytics_engine.track_event(event)
    return result.get("event_id", "")


async def track_music_event(user_id: str,
                          event_type: str,
                          track_id: str,
                          properties: Dict[str, Any] = None) -> str:
    """Fonction utilitaire pour tracker un événement musical"""
    
    if not analytics_engine:
        raise RuntimeError("Analytics engine not initialized")
    
    props = properties or {}
    props["track_id"] = track_id
    
    event = AnalyticsEvent(
        event_id=str(uuid.uuid4()),
        category=EventCategory.MUSIC_INTERACTION,
        event_type=event_type,
        user_id=user_id,
        properties=props
    )
    
    result = await analytics_engine.track_event(event)
    return result.get("event_id", "")


# Export des classes principales
__all__ = [
    "RealTimeAnalyticsEngine",
    "AnalyticsEvent",
    "MetricDefinition",
    "TimeSeries",
    "StreamProcessor",
    "UserBehaviorAnalyzer",
    "MusicAnalytics",
    "PerformanceMonitor",
    "EventCategory",
    "MetricType",
    "AggregationType",
    "track_user_event",
    "track_music_event",
    "create_analytics_engine"
]
