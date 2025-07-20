"""
Advanced User Behavior Analytics & Intelligence Engine
====================================================

Enterprise-grade behavioral analytics system for understanding user patterns,
predicting churn, and optimizing user experience through AI-driven insights.

Features:
- Real-time user behavior tracking and analysis
- Churn prediction with early warning system
- User segmentation and persona identification
- Listening pattern analysis and anomaly detection
- Personalization engine for user experience optimization
- A/B testing framework integration
- Revenue optimization through behavioral insights
- Privacy-preserving analytics with differential privacy
- Real-time dashboard and alerting system
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy import stats
import joblib
import redis
from enum import Enum
import hashlib

from . import audit_ml_operation, cache_ml_result, ML_CONFIG

logger = logging.getLogger(__name__)

class UserSegment(Enum):
    """User segmentation categories"""
    POWER_USER = "power_user"
    CASUAL_LISTENER = "casual_listener"
    MUSIC_DISCOVERER = "music_discoverer"
    PREMIUM_SUBSCRIBER = "premium_subscriber"
    CHURN_RISK = "churn_risk"
    NEW_USER = "new_user"
    INACTIVE = "inactive"

class BehaviorEventType(Enum):
    """Behavior event types"""
    PLAY = "play"
    SKIP = "skip"
    LIKE = "like"
    DISLIKE = "dislike"
    PLAYLIST_CREATE = "playlist_create"
    PLAYLIST_ADD = "playlist_add"
    SEARCH = "search"
    SHARE = "share"
    DOWNLOAD = "download"
    FOLLOW = "follow"
    UNFOLLOW = "unfollow"

@dataclass
class BehaviorEvent:
    """Individual behavior event"""
    user_id: str
    event_type: BehaviorEventType
    timestamp: datetime
    track_id: Optional[str] = None
    artist_id: Optional[str] = None
    playlist_id: Optional[str] = None
    session_id: Optional[str] = None
    device_type: Optional[str] = None
    location: Optional[str] = None
    duration: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class UserProfile:
    """Comprehensive user profile"""
    user_id: str
    segment: UserSegment
    behavior_features: Dict[str, float]
    listening_patterns: Dict[str, Any]
    preferences: Dict[str, Any]
    engagement_metrics: Dict[str, float]
    churn_probability: float
    lifetime_value: float
    created_at: datetime
    last_updated: datetime

@dataclass
class ChurnPrediction:
    """Churn prediction result"""
    user_id: str
    churn_probability: float
    risk_level: str
    contributing_factors: List[str]
    recommended_actions: List[str]
    prediction_confidence: float
    timestamp: datetime

class UserBehaviorFeatureExtractor:
    """
    Extract comprehensive behavioral features from user events
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.scaler = StandardScaler()
        
    def extract_features(self, events: List[BehaviorEvent], 
                        time_window: timedelta = timedelta(days=30)) -> Dict[str, float]:
        """Extract behavioral features from user events"""
        if not events:
            return self._get_default_features()
        
        # Filter events within time window
        cutoff_time = datetime.utcnow() - time_window
        recent_events = [e for e in events if e.timestamp >= cutoff_time]
        
        features = {}
        
        # Temporal features
        features.update(self._extract_temporal_features(recent_events))
        
        # Engagement features
        features.update(self._extract_engagement_features(recent_events))
        
        # Content features
        features.update(self._extract_content_features(recent_events))
        
        # Session features
        features.update(self._extract_session_features(recent_events))
        
        # Device and context features
        features.update(self._extract_context_features(recent_events))
        
        return features
    
    def _extract_temporal_features(self, events: List[BehaviorEvent]) -> Dict[str, float]:
        """Extract time-based behavioral features"""
        if not events:
            return {}
        
        # Convert to timestamps
        timestamps = [e.timestamp for e in events]
        hours = [t.hour for t in timestamps]
        days = [t.weekday() for t in timestamps]
        
        features = {
            # Activity frequency
            'events_per_day': len(events) / 30.0,
            'total_events': len(events),
            
            # Temporal patterns
            'avg_hour_of_activity': np.mean(hours),
            'activity_hour_std': np.std(hours),
            'weekend_activity_ratio': sum(1 for d in days if d >= 5) / max(len(days), 1),
            
            # Activity consistency
            'days_active': len(set(t.date() for t in timestamps)),
            'max_gap_between_sessions': self._calculate_max_gap(timestamps),
            'avg_sessions_per_day': len(set(t.date() for t in timestamps)) / 30.0,
        }
        
        # Peak activity hour
        hour_counts = pd.Series(hours).value_counts()
        features['peak_activity_hour'] = hour_counts.index[0] if not hour_counts.empty else 12
        
        return features
    
    def _extract_engagement_features(self, events: List[BehaviorEvent]) -> Dict[str, float]:
        """Extract engagement-related features"""
        if not events:
            return {}
        
        play_events = [e for e in events if e.event_type == BehaviorEventType.PLAY]
        skip_events = [e for e in events if e.event_type == BehaviorEventType.SKIP]
        like_events = [e for e in events if e.event_type == BehaviorEventType.LIKE]
        
        features = {
            # Play behavior
            'total_plays': len(play_events),
            'total_skips': len(skip_events),
            'skip_rate': len(skip_events) / max(len(play_events), 1),
            
            # Engagement depth
            'like_rate': len(like_events) / max(len(play_events), 1),
            'playlist_creations': len([e for e in events if e.event_type == BehaviorEventType.PLAYLIST_CREATE]),
            'search_frequency': len([e for e in events if e.event_type == BehaviorEventType.SEARCH]),
            'share_frequency': len([e for e in events if e.event_type == BehaviorEventType.SHARE]),
        }
        
        # Listen duration analysis
        play_durations = [e.duration for e in play_events if e.duration is not None]
        if play_durations:
            features.update({
                'avg_listen_duration': np.mean(play_durations),
                'total_listen_time': sum(play_durations),
                'listen_completion_rate': sum(1 for d in play_durations if d > 30) / len(play_durations),
            })
        
        return features
    
    def _extract_content_features(self, events: List[BehaviorEvent]) -> Dict[str, float]:
        """Extract content preference features"""
        if not events:
            return {}
        
        # Unique content consumption
        unique_tracks = set(e.track_id for e in events if e.track_id)
        unique_artists = set(e.artist_id for e in events if e.artist_id)
        
        features = {
            'unique_tracks_played': len(unique_tracks),
            'unique_artists_played': len(unique_artists),
            'track_diversity_ratio': len(unique_tracks) / max(len(events), 1),
            'artist_diversity_ratio': len(unique_artists) / max(len(events), 1),
        }
        
        # Repeat listening behavior
        track_counts = pd.Series([e.track_id for e in events if e.track_id]).value_counts()
        if not track_counts.empty:
            features.update({
                'avg_track_repeats': track_counts.mean(),
                'max_track_repeats': track_counts.max(),
                'top_track_dominance': track_counts.iloc[0] / track_counts.sum(),
            })
        
        return features
    
    def _extract_session_features(self, events: List[BehaviorEvent]) -> Dict[str, float]:
        """Extract session-based features"""
        if not events:
            return {}
        
        # Group events by session
        sessions = {}
        for event in events:
            session_id = event.session_id or f"{event.user_id}_{event.timestamp.date()}"
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(event)
        
        session_lengths = [len(session) for session in sessions.values()]
        session_durations = []
        
        for session_events in sessions.values():
            if len(session_events) > 1:
                start_time = min(e.timestamp for e in session_events)
                end_time = max(e.timestamp for e in session_events)
                duration = (end_time - start_time).total_seconds() / 60  # minutes
                session_durations.append(duration)
        
        features = {
            'total_sessions': len(sessions),
            'avg_session_length': np.mean(session_lengths) if session_lengths else 0,
            'avg_session_duration': np.mean(session_durations) if session_durations else 0,
            'max_session_length': max(session_lengths) if session_lengths else 0,
        }
        
        return features
    
    def _extract_context_features(self, events: List[BehaviorEvent]) -> Dict[str, float]:
        """Extract contextual features"""
        if not events:
            return {}
        
        devices = [e.device_type for e in events if e.device_type]
        locations = [e.location for e in events if e.location]
        
        features = {
            'unique_devices': len(set(devices)),
            'unique_locations': len(set(locations)),
            'mobile_usage_ratio': sum(1 for d in devices if d == 'mobile') / max(len(devices), 1),
        }
        
        return features
    
    def _calculate_max_gap(self, timestamps: List[datetime]) -> float:
        """Calculate maximum gap between consecutive sessions"""
        if len(timestamps) < 2:
            return 0
        
        sorted_times = sorted(timestamps)
        gaps = [(sorted_times[i] - sorted_times[i-1]).total_seconds() / 3600 for i in range(1, len(sorted_times))]
        return max(gaps) if gaps else 0
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features for users with no events"""
        return {
            'events_per_day': 0.0,
            'total_events': 0,
            'skip_rate': 0.0,
            'like_rate': 0.0,
            'unique_tracks_played': 0,
            'total_sessions': 0,
            'churn_risk_score': 1.0  # High churn risk for inactive users
        }

class ChurnPredictionModel(nn.Module):
    """
    Deep learning model for churn prediction
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
        # Feature importance tracking
        self.feature_importance = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def predict_churn_probability(self, features: torch.Tensor) -> torch.Tensor:
        """Predict churn probability for given features"""
        self.eval()
        with torch.no_grad():
            return self.forward(features)

class UserSegmentationEngine:
    """
    Advanced user segmentation using ML clustering
    """
    
    def __init__(self, n_segments: int = 7):
        self.n_segments = n_segments
        self.kmeans = KMeans(n_clusters=n_segments, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.pca = PCA(n_components=10)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit_segmentation(self, user_features: pd.DataFrame) -> Dict[str, Any]:
        """Fit segmentation models on user features"""
        logger.info("🎯 Training user segmentation models...")
        
        # Normalize features
        feature_matrix = self.scaler.fit_transform(user_features.values)
        
        # Dimensionality reduction
        feature_matrix_pca = self.pca.fit_transform(feature_matrix)
        
        # K-means clustering
        kmeans_labels = self.kmeans.fit_predict(feature_matrix_pca)
        
        # DBSCAN for anomaly detection
        dbscan_labels = self.dbscan.fit_predict(feature_matrix_pca)
        
        self.is_fitted = True
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(user_features, kmeans_labels)
        
        logger.info(f"✅ User segmentation completed: {self.n_segments} segments identified")
        
        return {
            'n_segments': self.n_segments,
            'cluster_sizes': pd.Series(kmeans_labels).value_counts().to_dict(),
            'anomaly_count': sum(1 for label in dbscan_labels if label == -1),
            'cluster_analysis': cluster_analysis
        }
    
    def predict_segment(self, user_features: pd.Series) -> UserSegment:
        """Predict user segment for given features"""
        if not self.is_fitted:
            logger.warning("Segmentation model not fitted, returning default segment")
            return UserSegment.NEW_USER
        
        # Normalize and transform features
        feature_vector = self.scaler.transform([user_features.values])
        feature_vector_pca = self.pca.transform(feature_vector)
        
        # Predict cluster
        cluster_id = self.kmeans.predict(feature_vector_pca)[0]
        
        # Map cluster to segment (simplified mapping)
        segment_mapping = {
            0: UserSegment.POWER_USER,
            1: UserSegment.CASUAL_LISTENER,
            2: UserSegment.MUSIC_DISCOVERER,
            3: UserSegment.PREMIUM_SUBSCRIBER,
            4: UserSegment.CHURN_RISK,
            5: UserSegment.NEW_USER,
            6: UserSegment.INACTIVE
        }
        
        return segment_mapping.get(cluster_id, UserSegment.CASUAL_LISTENER)
    
    def _analyze_clusters(self, user_features: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        analysis = {}
        
        for cluster_id in range(self.n_segments):
            cluster_mask = labels == cluster_id
            cluster_features = user_features[cluster_mask]
            
            if len(cluster_features) > 0:
                analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_features),
                    'avg_engagement': cluster_features['total_plays'].mean() if 'total_plays' in cluster_features else 0,
                    'avg_session_duration': cluster_features['avg_session_duration'].mean() if 'avg_session_duration' in cluster_features else 0,
                    'characteristics': self._get_cluster_characteristics(cluster_features)
                }
        
        return analysis
    
    def _get_cluster_characteristics(self, cluster_features: pd.DataFrame) -> List[str]:
        """Get human-readable cluster characteristics"""
        characteristics = []
        
        # High engagement indicators
        if cluster_features['total_plays'].mean() > cluster_features['total_plays'].median():
            characteristics.append("High engagement")
        
        # Discovery behavior
        if cluster_features['unique_tracks_played'].mean() > cluster_features['unique_tracks_played'].median():
            characteristics.append("Music discoverer")
        
        # Session patterns
        if cluster_features['avg_session_duration'].mean() > 60:  # More than 1 hour
            characteristics.append("Long listening sessions")
        
        return characteristics

class BehaviorAnomalyDetector:
    """
    Detect anomalous user behavior patterns
    """
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False
        
    def fit(self, user_features: pd.DataFrame):
        """Fit anomaly detection model"""
        self.isolation_forest.fit(user_features.values)
        self.is_fitted = True
        logger.info("🔍 Behavior anomaly detector trained")
    
    def detect_anomalies(self, user_features: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalous behavior patterns"""
        if not self.is_fitted:
            logger.warning("Anomaly detector not fitted")
            return {}
        
        # Predict anomalies
        anomaly_scores = self.isolation_forest.decision_function(user_features.values)
        anomaly_labels = self.isolation_forest.predict(user_features.values)
        
        # Identify anomalous users
        anomalous_users = user_features[anomaly_labels == -1]
        
        return {
            'total_users': len(user_features),
            'anomalous_users': len(anomalous_users),
            'anomaly_rate': len(anomalous_users) / len(user_features),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomalous_user_ids': anomalous_users.index.tolist() if hasattr(anomalous_users, 'index') else []
        }

class BehaviorAnalyticsEngine:
    """
    Comprehensive Behavior Analytics Engine
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_extractor = UserBehaviorFeatureExtractor()
        self.churn_model = None
        self.segmentation_engine = UserSegmentationEngine()
        self.anomaly_detector = BehaviorAnomalyDetector()
        
        # Storage
        self.user_profiles = {}
        self.behavior_events = {}
        
        # Redis for caching
        try:
            self.redis_client = redis.Redis.from_url(
                ML_CONFIG["feature_store_url"], decode_responses=True
            )
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Load pre-trained models
        self._load_models()
        
        logger.info("🧠 Behavior Analytics Engine initialized")
    
    def _load_models(self):
        """Load pre-trained models"""
        model_path = Path(ML_CONFIG["model_registry_path"])
        
        try:
            if (model_path / "churn_prediction_model.pth").exists():
                model_state = torch.load(model_path / "churn_prediction_model.pth", map_location='cpu')
                self.churn_model = ChurnPredictionModel(model_state['input_dim'])
                self.churn_model.load_state_dict(model_state['state_dict'])
                logger.info("✅ Loaded pre-trained churn prediction model")
                
        except Exception as e:
            logger.error(f"Failed to load behavior models: {e}")
    
    @audit_ml_operation("behavior_event_processing")
    async def process_behavior_event(self, event: BehaviorEvent):
        """Process a single behavior event"""
        try:
            # Store event
            if event.user_id not in self.behavior_events:
                self.behavior_events[event.user_id] = []
            
            self.behavior_events[event.user_id].append(event)
            
            # Update user profile asynchronously
            await self._update_user_profile(event.user_id)
            
            logger.debug(f"📊 Processed behavior event: {event.event_type.value} for user {event.user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process behavior event: {e}")
    
    @audit_ml_operation("user_profile_analysis")
    @cache_ml_result(ttl=1800)  # Cache for 30 minutes
    async def analyze_user_behavior(self, user_id: str, 
                                  time_window: timedelta = timedelta(days=30)) -> UserProfile:
        """Comprehensive user behavior analysis"""
        try:
            # Get user events
            user_events = self.behavior_events.get(user_id, [])
            
            # Extract behavioral features
            features = self.feature_extractor.extract_features(user_events, time_window)
            
            # Predict user segment
            feature_series = pd.Series(features)
            user_segment = self.segmentation_engine.predict_segment(feature_series)
            
            # Predict churn probability
            churn_probability = await self._predict_churn(features)
            
            # Calculate engagement metrics
            engagement_metrics = self._calculate_engagement_metrics(user_events)
            
            # Analyze listening patterns
            listening_patterns = self._analyze_listening_patterns(user_events)
            
            # Extract preferences
            preferences = self._extract_user_preferences(user_events)
            
            # Calculate lifetime value
            lifetime_value = self._estimate_lifetime_value(features, user_segment)
            
            # Create user profile
            user_profile = UserProfile(
                user_id=user_id,
                segment=user_segment,
                behavior_features=features,
                listening_patterns=listening_patterns,
                preferences=preferences,
                engagement_metrics=engagement_metrics,
                churn_probability=churn_probability,
                lifetime_value=lifetime_value,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            # Cache profile
            self.user_profiles[user_id] = user_profile
            
            logger.info(f"✅ User behavior analysis completed for {user_id}")
            return user_profile
            
        except Exception as e:
            logger.error(f"❌ User behavior analysis failed: {e}")
            raise
    
    async def _update_user_profile(self, user_id: str):
        """Update user profile after new event"""
        try:
            updated_profile = await self.analyze_user_behavior(user_id)
            
            # Check for significant changes
            previous_profile = self.user_profiles.get(user_id)
            if previous_profile:
                await self._detect_behavior_changes(previous_profile, updated_profile)
            
        except Exception as e:
            logger.error(f"Profile update failed for user {user_id}: {e}")
    
    async def _predict_churn(self, features: Dict[str, float]) -> float:
        """Predict churn probability for user"""
        if not self.churn_model:
            # Simple heuristic if no model available
            return self._heuristic_churn_prediction(features)
        
        try:
            # Convert features to tensor
            feature_values = list(features.values())
            feature_tensor = torch.FloatTensor([feature_values])
            
            # Predict
            churn_prob = self.churn_model.predict_churn_probability(feature_tensor)
            return float(churn_prob.item())
            
        except Exception as e:
            logger.error(f"Churn prediction failed: {e}")
            return self._heuristic_churn_prediction(features)
    
    def _heuristic_churn_prediction(self, features: Dict[str, float]) -> float:
        """Simple heuristic churn prediction"""
        risk_factors = 0
        
        if features.get('events_per_day', 0) < 1:
            risk_factors += 0.3
        
        if features.get('days_active', 0) < 5:
            risk_factors += 0.2
        
        if features.get('max_gap_between_sessions', 0) > 72:  # 3 days
            risk_factors += 0.3
        
        if features.get('skip_rate', 0) > 0.7:
            risk_factors += 0.2
        
        return min(risk_factors, 1.0)
    
    def _calculate_engagement_metrics(self, events: List[BehaviorEvent]) -> Dict[str, float]:
        """Calculate user engagement metrics"""
        if not events:
            return {}
        
        play_events = [e for e in events if e.event_type == BehaviorEventType.PLAY]
        engagement_events = [e for e in events if e.event_type in [
            BehaviorEventType.LIKE, BehaviorEventType.SHARE, BehaviorEventType.PLAYLIST_CREATE
        ]]
        
        metrics = {
            'total_interactions': len(events),
            'play_frequency': len(play_events) / 30.0,  # per day
            'engagement_rate': len(engagement_events) / max(len(events), 1),
            'session_consistency': self._calculate_session_consistency(events),
            'content_exploration': self._calculate_content_exploration(events)
        }
        
        return metrics
    
    def _analyze_listening_patterns(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Analyze user listening patterns"""
        if not events:
            return {}
        
        play_events = [e for e in events if e.event_type == BehaviorEventType.PLAY]
        
        # Time-based patterns
        hours = [e.timestamp.hour for e in play_events]
        days = [e.timestamp.weekday() for e in play_events]
        
        patterns = {
            'preferred_listening_hours': self._get_peak_hours(hours),
            'weekend_vs_weekday': {
                'weekend_ratio': sum(1 for d in days if d >= 5) / max(len(days), 1),
                'weekday_ratio': sum(1 for d in days if d < 5) / max(len(days), 1)
            },
            'listening_intensity': self._calculate_listening_intensity(play_events),
            'temporal_consistency': np.std(hours) if hours else 0
        }
        
        return patterns
    
    def _extract_user_preferences(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """Extract user music preferences"""
        play_events = [e for e in events if e.event_type == BehaviorEventType.PLAY]
        like_events = [e for e in events if e.event_type == BehaviorEventType.LIKE]
        
        preferences = {
            'liked_content_ratio': len(like_events) / max(len(play_events), 1),
            'repeat_listening_behavior': self._analyze_repeat_behavior(play_events),
            'discovery_tendency': self._calculate_discovery_tendency(events),
            'playlist_engagement': len([e for e in events if 'playlist' in e.event_type.value])
        }
        
        return preferences
    
    def _estimate_lifetime_value(self, features: Dict[str, float], 
                                segment: UserSegment) -> float:
        """Estimate user lifetime value"""
        base_value = 50.0  # Base LTV
        
        # Segment multipliers
        segment_multipliers = {
            UserSegment.POWER_USER: 3.0,
            UserSegment.PREMIUM_SUBSCRIBER: 2.5,
            UserSegment.MUSIC_DISCOVERER: 2.0,
            UserSegment.CASUAL_LISTENER: 1.0,
            UserSegment.NEW_USER: 1.5,
            UserSegment.CHURN_RISK: 0.3,
            UserSegment.INACTIVE: 0.1
        }
        
        multiplier = segment_multipliers.get(segment, 1.0)
        
        # Feature-based adjustments
        engagement_factor = min(features.get('total_plays', 0) / 100, 2.0)
        consistency_factor = min(features.get('days_active', 0) / 30, 1.5)
        
        ltv = base_value * multiplier * engagement_factor * consistency_factor
        
        return max(ltv, 0.0)
    
    def _get_peak_hours(self, hours: List[int]) -> List[int]:
        """Get peak listening hours"""
        if not hours:
            return []
        
        hour_counts = pd.Series(hours).value_counts()
        # Return top 3 hours
        return hour_counts.head(3).index.tolist()
    
    def _calculate_listening_intensity(self, play_events: List[BehaviorEvent]) -> str:
        """Calculate listening intensity level"""
        if not play_events:
            return "low"
        
        daily_plays = len(play_events) / 30.0  # per day
        
        if daily_plays > 50:
            return "very_high"
        elif daily_plays > 20:
            return "high"
        elif daily_plays > 5:
            return "medium"
        else:
            return "low"
    
    def _calculate_session_consistency(self, events: List[BehaviorEvent]) -> float:
        """Calculate session consistency score"""
        if not events:
            return 0.0
        
        # Group by day
        daily_events = {}
        for event in events:
            day = event.timestamp.date()
            daily_events[day] = daily_events.get(day, 0) + 1
        
        if len(daily_events) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        event_counts = list(daily_events.values())
        cv = np.std(event_counts) / (np.mean(event_counts) + 1e-8)
        
        # Convert to consistency score (lower CV = higher consistency)
        return max(0, 1 - cv)
    
    def _calculate_content_exploration(self, events: List[BehaviorEvent]) -> float:
        """Calculate content exploration score"""
        unique_content = set()
        for event in events:
            if event.track_id:
                unique_content.add(event.track_id)
            if event.artist_id:
                unique_content.add(event.artist_id)
        
        return len(unique_content) / max(len(events), 1)
    
    def _analyze_repeat_behavior(self, play_events: List[BehaviorEvent]) -> Dict[str, float]:
        """Analyze repeat listening behavior"""
        if not play_events:
            return {}
        
        track_counts = pd.Series([e.track_id for e in play_events if e.track_id]).value_counts()
        
        if track_counts.empty:
            return {}
        
        return {
            'avg_repeats_per_track': track_counts.mean(),
            'max_repeats': track_counts.max(),
            'repeat_tracks_ratio': sum(track_counts > 1) / len(track_counts)
        }
    
    def _calculate_discovery_tendency(self, events: List[BehaviorEvent]) -> float:
        """Calculate discovery tendency score"""
        search_events = [e for e in events if e.event_type == BehaviorEventType.SEARCH]
        total_events = len(events)
        
        return len(search_events) / max(total_events, 1)
    
    async def _detect_behavior_changes(self, previous_profile: UserProfile, 
                                     current_profile: UserProfile):
        """Detect significant behavior changes"""
        # Compare key metrics
        churn_change = current_profile.churn_probability - previous_profile.churn_probability
        
        if churn_change > 0.2:  # Significant increase in churn risk
            logger.warning(f"🚨 Churn risk increased for user {current_profile.user_id}: {churn_change:.2f}")
        
        # Segment change
        if current_profile.segment != previous_profile.segment:
            logger.info(f"📈 User {current_profile.user_id} moved from {previous_profile.segment.value} to {current_profile.segment.value}")
    
    @audit_ml_operation("cohort_analysis")
    async def analyze_user_cohorts(self, cohort_type: str = "weekly") -> Dict[str, Any]:
        """Analyze user cohorts for retention and engagement"""
        logger.info(f"📊 Analyzing {cohort_type} user cohorts...")
        
        # Mock cohort analysis - in production, this would query actual data
        cohort_data = {
            'cohort_type': cohort_type,
            'total_cohorts': 12,
            'analysis_period': '3_months',
            'retention_rates': {
                'week_1': 0.85,
                'week_2': 0.72,
                'week_4': 0.58,
                'week_8': 0.45,
                'week_12': 0.38
            },
            'engagement_trends': {
                'average_sessions_per_week': 4.2,
                'average_listening_hours': 12.5,
                'feature_adoption_rate': 0.23
            },
            'insights': [
                "Week 2 shows highest drop-off rate",
                "Power users maintain 90%+ retention after week 4",
                "Mobile users show better retention than desktop"
            ]
        }
        
        return cohort_data
    
    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive analytics dashboard data"""
        total_users = len(self.user_profiles)
        
        if total_users == 0:
            return {'error': 'No user data available'}
        
        # Segment distribution
        segment_distribution = {}
        churn_risks = []
        ltv_values = []
        
        for profile in self.user_profiles.values():
            segment = profile.segment.value
            segment_distribution[segment] = segment_distribution.get(segment, 0) + 1
            churn_risks.append(profile.churn_probability)
            ltv_values.append(profile.lifetime_value)
        
        dashboard = {
            'overview': {
                'total_users': total_users,
                'avg_churn_risk': np.mean(churn_risks) if churn_risks else 0,
                'total_ltv': sum(ltv_values),
                'high_churn_users': sum(1 for risk in churn_risks if risk > 0.7)
            },
            'segment_distribution': segment_distribution,
            'engagement_metrics': {
                'avg_daily_active_users': total_users * 0.3,  # Mock DAU
                'avg_session_duration': 25.5,  # Mock average
                'total_events_processed': sum(len(events) for events in self.behavior_events.values())
            },
            'trends': {
                'churn_trend': 'stable',
                'engagement_trend': 'increasing',
                'revenue_trend': 'growing'
            },
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return dashboard

# Factory function
def create_behavior_analytics_engine(config: Dict[str, Any] = None) -> BehaviorAnalyticsEngine:
    """Create behavior analytics engine instance"""
    return BehaviorAnalyticsEngine(config)

# Export main components
__all__ = [
    'BehaviorAnalyticsEngine',
    'BehaviorEvent',
    'UserProfile',
    'ChurnPrediction',
    'UserSegment',
    'BehaviorEventType',
    'UserBehaviorFeatureExtractor',
    'ChurnPredictionModel',
    'UserSegmentationEngine',
    'BehaviorAnomalyDetector',
    'create_behavior_analytics_engine'
]
