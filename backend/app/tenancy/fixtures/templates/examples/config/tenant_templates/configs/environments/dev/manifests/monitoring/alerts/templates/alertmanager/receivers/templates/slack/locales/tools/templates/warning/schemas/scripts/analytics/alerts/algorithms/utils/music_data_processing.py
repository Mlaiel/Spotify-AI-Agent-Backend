"""
🎵 Spotify AI Agent - Music Streaming Data Processing Utilities
=============================================================

Advanced Data Processing Utilities for Music Streaming Platform

This module provides specialized data processing utilities specifically designed
for music streaming platforms. It includes audio quality analysis, user behavior
processing, content analytics, and performance optimization tools.

🚀 MUSIC STREAMING DATA PROCESSING FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎧 Audio Quality Analysis:
  • Audio bitrate and quality metric calculation
  • Buffering ratio and latency analysis
  • Audio streaming performance indicators
  • Real-time quality degradation detection
  • Codec performance and efficiency metrics

👥 User Behavior Processing:
  • Listening pattern analysis and segmentation
  • Session duration and engagement metrics
  • Skip rate and completion rate calculation
  • User journey mapping and funnel analysis
  • Churn prediction feature engineering

📊 Content Analytics:
  • Track popularity and trending analysis
  • Genre and artist performance metrics
  • Playlist engagement and success rates
  • Content discovery and recommendation effectiveness
  • Seasonal and temporal pattern analysis

🌍 Geographic Performance:
  • Regional performance metric aggregation
  • CDN and latency analysis by geography
  • Market penetration and user distribution
  • Cross-regional performance comparison
  • Geographic anomaly detection

💰 Revenue and Business Metrics:
  • Revenue per user calculations
  • Subscription conversion and retention metrics
  • Ad performance and completion rates
  • Content licensing cost optimization
  • Business impact and ROI analysis

@Author: Music Data Processing by Fahed Mlaiel
@Version: 2.0.0 (Enterprise Edition)
@Last Updated: 2025-07-19
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math
import statistics
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

# Try to import audio processing libraries
try:
    import librosa
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False
    warnings.warn("Audio processing libraries not available. Some features will be limited.")

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AudioQualityMetrics:
    """
    Comprehensive audio quality metrics for streaming analysis.
    """
    # Core Audio Metrics
    bitrate_kbps: float = 0.0              # Audio bitrate in kbps
    sample_rate_hz: int = 44100            # Sample rate in Hz
    bit_depth: int = 16                    # Bit depth
    codec: str = "unknown"                 # Audio codec (MP3, AAC, OGG, FLAC)
    
    # Quality Indicators
    snr_db: float = 0.0                    # Signal-to-noise ratio in dB
    thd_percent: float = 0.0               # Total harmonic distortion
    dynamic_range_db: float = 0.0          # Dynamic range in dB
    frequency_response_score: float = 0.0   # Frequency response quality score
    
    # Streaming Performance
    buffering_events: int = 0               # Number of buffering events
    buffering_duration_ms: float = 0.0      # Total buffering duration
    startup_latency_ms: float = 0.0        # Time to first audio byte
    rebuffering_ratio: float = 0.0         # Rebuffering time / total time
    
    # Network and Delivery
    network_jitter_ms: float = 0.0         # Network jitter
    packet_loss_percent: float = 0.0       # Packet loss percentage
    cdn_response_time_ms: float = 0.0      # CDN response time
    geographic_latency_ms: float = 0.0     # Geographic delivery latency
    
    # User Experience
    perceived_quality_score: float = 0.0   # Subjective quality score (1-10)
    user_satisfaction_rating: float = 0.0  # User satisfaction (1-5)
    skip_likelihood: float = 0.0           # Probability of track skip
    engagement_score: float = 0.0          # Overall engagement metric
    
    def calculate_overall_quality(self) -> float:
        """Calculate overall audio quality score (0-1)."""
        # Weighted combination of quality factors
        weights = {
            'bitrate': 0.25,
            'snr': 0.20,
            'buffering': 0.20,
            'latency': 0.15,
            'network': 0.10,
            'user_experience': 0.10
        }
        
        # Normalize individual metrics to 0-1 scale
        bitrate_score = min(self.bitrate_kbps / 320.0, 1.0)  # 320kbps as reference
        snr_score = min(self.snr_db / 96.0, 1.0)  # 96dB as excellent SNR
        buffering_score = max(1.0 - (self.rebuffering_ratio * 10), 0.0)
        latency_score = max(1.0 - (self.startup_latency_ms / 1000.0), 0.0)
        network_score = max(1.0 - (self.packet_loss_percent / 5.0), 0.0)
        ux_score = self.perceived_quality_score / 10.0
        
        overall_score = (
            weights['bitrate'] * bitrate_score +
            weights['snr'] * snr_score +
            weights['buffering'] * buffering_score +
            weights['latency'] * latency_score +
            weights['network'] * network_score +
            weights['user_experience'] * ux_score
        )
        
        return np.clip(overall_score, 0.0, 1.0)

@dataclass 
class UserBehaviorMetrics:
    """
    Comprehensive user behavior metrics for engagement analysis.
    """
    # Session Metrics
    session_duration_minutes: float = 0.0   # Total session duration
    tracks_played: int = 0                   # Number of tracks played
    tracks_completed: int = 0                # Number of tracks played to completion
    tracks_skipped: int = 0                  # Number of tracks skipped
    tracks_liked: int = 0                    # Number of tracks liked/favorited
    tracks_shared: int = 0                   # Number of tracks shared
    
    # Engagement Patterns
    skip_rate: float = 0.0                   # Skip rate (skips/total tracks)
    completion_rate: float = 0.0             # Completion rate (completed/total)
    like_rate: float = 0.0                   # Like rate (likes/total tracks)
    share_rate: float = 0.0                  # Share rate (shares/total tracks)
    repeat_play_rate: float = 0.0            # Rate of repeat plays
    
    # Discovery and Exploration
    new_artists_discovered: int = 0          # New artists listened to
    new_genres_explored: int = 0             # New genres explored
    playlist_interactions: int = 0           # Playlist creations/modifications
    search_queries: int = 0                  # Number of search queries
    recommendation_clicks: int = 0           # Clicks on recommendations
    
    # Time-based Patterns
    peak_listening_hour: int = 12            # Peak listening hour (0-23)
    weekend_vs_weekday_ratio: float = 1.0    # Weekend/weekday listening ratio
    morning_listening_percent: float = 0.0   # Morning listening percentage
    evening_listening_percent: float = 0.0   # Evening listening percentage
    
    # Device and Context
    mobile_listening_percent: float = 0.0    # Mobile device usage percentage
    offline_listening_percent: float = 0.0   # Offline listening percentage
    social_sharing_frequency: float = 0.0    # Social sharing frequency
    voice_command_usage: int = 0             # Voice command interactions
    
    def calculate_engagement_score(self) -> float:
        """Calculate overall user engagement score (0-1)."""
        # Weighted engagement factors
        weights = {
            'completion': 0.30,
            'interaction': 0.25,
            'discovery': 0.20,
            'session_quality': 0.15,
            'social': 0.10
        }
        
        # Calculate component scores
        completion_score = self.completion_rate
        interaction_score = min((self.like_rate + self.share_rate) / 2.0, 1.0)
        discovery_score = min((self.new_artists_discovered + self.new_genres_explored) / 20.0, 1.0)
        session_score = min(self.session_duration_minutes / 60.0, 1.0)  # 1 hour as good session
        social_score = min(self.social_sharing_frequency / 5.0, 1.0)  # 5 shares as highly social
        
        engagement_score = (
            weights['completion'] * completion_score +
            weights['interaction'] * interaction_score +
            weights['discovery'] * discovery_score +
            weights['session_quality'] * session_score +
            weights['social'] * social_score
        )
        
        return np.clip(engagement_score, 0.0, 1.0)

class MusicDataProcessor:
    """
    Advanced data processor for music streaming platform analytics.
    
    Provides comprehensive data processing capabilities including feature
    engineering, aggregation, normalization, and specialized music analytics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the music data processor.
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'processing_time_ms': [],
            'error_count': 0,
            'last_update': None
        }
        
        logger.info("MusicDataProcessor initialized")
    
    def process_audio_quality_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and enhance audio quality data with advanced metrics.
        
        Args:
            data: DataFrame containing raw audio quality data
            
        Returns:
            Enhanced DataFrame with calculated quality metrics
        """
        start_time = datetime.now()
        
        try:
            processed_data = data.copy()
            
            # Calculate derived audio quality metrics
            if 'bitrate_kbps' in data.columns and 'total_duration_s' in data.columns:
                processed_data['data_consumption_mb'] = (
                    data['bitrate_kbps'] * data['total_duration_s'] / 8000.0
                )
            
            # Calculate buffering metrics
            if 'buffering_events' in data.columns and 'total_duration_s' in data.columns:
                processed_data['buffering_frequency'] = (
                    data['buffering_events'] / data['total_duration_s'] * 60  # per minute
                )
            
            # Calculate quality degradation indicators
            if 'startup_latency_ms' in data.columns:
                processed_data['latency_category'] = pd.cut(
                    data['startup_latency_ms'],
                    bins=[0, 100, 300, 1000, float('inf')],
                    labels=['excellent', 'good', 'fair', 'poor']
                )
            
            # Calculate perceived quality score
            quality_features = ['bitrate_kbps', 'snr_db', 'startup_latency_ms', 
                              'buffering_events', 'packet_loss_percent']
            
            available_features = [f for f in quality_features if f in data.columns]
            
            if available_features:
                processed_data['perceived_quality_score'] = self._calculate_perceived_quality(
                    data[available_features]
                )
            
            # Add time-based features
            if 'timestamp' in data.columns:
                processed_data = self._add_temporal_features(processed_data)
            
            # Update processing statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.processing_stats['total_processed'] += len(data)
            self.processing_stats['processing_time_ms'].append(processing_time)
            self.processing_stats['last_update'] = datetime.now()
            
            logger.info(f"Processed {len(data)} audio quality records in {processing_time:.2f}ms")
            
            return processed_data
            
        except Exception as e:
            self.processing_stats['error_count'] += 1
            logger.error(f"Audio quality processing failed: {str(e)}")
            raise
    
    def process_user_behavior_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and enhance user behavior data with engagement metrics.
        
        Args:
            data: DataFrame containing raw user behavior data
            
        Returns:
            Enhanced DataFrame with calculated behavior metrics
        """
        start_time = datetime.now()
        
        try:
            processed_data = data.copy()
            
            # Calculate engagement rates
            if 'tracks_played' in data.columns and 'tracks_skipped' in data.columns:
                processed_data['skip_rate'] = (
                    data['tracks_skipped'] / np.maximum(data['tracks_played'], 1)
                )
                processed_data['completion_rate'] = 1.0 - processed_data['skip_rate']
            
            # Calculate discovery metrics
            if 'new_artists_discovered' in data.columns and 'session_duration_minutes' in data.columns:
                processed_data['discovery_rate'] = (
                    data['new_artists_discovered'] / np.maximum(data['session_duration_minutes'], 1) * 60
                )
            
            # Calculate loyalty indicators
            if 'repeat_play_rate' in data.columns and 'like_rate' in data.columns:
                processed_data['loyalty_score'] = (
                    (data['repeat_play_rate'] * 0.6) + (data['like_rate'] * 0.4)
                )
            
            # User segmentation based on behavior
            if 'session_duration_minutes' in data.columns and 'tracks_played' in data.columns:
                processed_data['user_segment'] = self._segment_users_by_behavior(
                    data[['session_duration_minutes', 'tracks_played', 'skip_rate']]
                )
            
            # Calculate engagement score
            engagement_features = ['completion_rate', 'like_rate', 'share_rate', 
                                 'discovery_rate', 'session_duration_minutes']
            
            available_features = [f for f in engagement_features if f in processed_data.columns]
            
            if available_features:
                processed_data['engagement_score'] = self._calculate_engagement_score(
                    processed_data[available_features]
                )
            
            # Add behavioral pattern features
            processed_data = self._add_behavioral_patterns(processed_data)
            
            # Update processing statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.processing_stats['total_processed'] += len(data)
            self.processing_stats['processing_time_ms'].append(processing_time)
            
            logger.info(f"Processed {len(data)} user behavior records in {processing_time:.2f}ms")
            
            return processed_data
            
        except Exception as e:
            self.processing_stats['error_count'] += 1
            logger.error(f"User behavior processing failed: {str(e)}")
            raise
    
    def aggregate_geographic_performance(self, data: pd.DataFrame, 
                                       region_column: str = 'region') -> pd.DataFrame:
        """
        Aggregate performance metrics by geographic region.
        
        Args:
            data: DataFrame containing performance data with region information
            region_column: Column name containing region information
            
        Returns:
            DataFrame with aggregated regional performance metrics
        """
        try:
            # Define aggregation functions for different metric types
            agg_functions = {
                'latency_ms': ['mean', 'median', 'p95', 'p99'],
                'throughput': ['mean', 'sum'],
                'error_rate': ['mean', 'max'],
                'user_count': ['sum', 'nunique'],
                'revenue_usd': ['sum', 'mean'],
                'audio_quality_score': ['mean', 'min'],
                'engagement_score': ['mean', 'std']
            }
            
            # Perform aggregation
            regional_stats = []
            
            for region in data[region_column].unique():
                region_data = data[data[region_column] == region]
                
                stats = {'region': region, 'sample_count': len(region_data)}
                
                for metric, functions in agg_functions.items():
                    if metric in region_data.columns:
                        for func in functions:
                            if func == 'p95':
                                stats[f'{metric}_{func}'] = np.percentile(region_data[metric], 95)
                            elif func == 'p99':
                                stats[f'{metric}_{func}'] = np.percentile(region_data[metric], 99)
                            elif func == 'nunique':
                                stats[f'{metric}_{func}'] = region_data[metric].nunique()
                            elif func == 'std':
                                stats[f'{metric}_{func}'] = region_data[metric].std()
                            else:
                                stats[f'{metric}_{func}'] = getattr(region_data[metric], func)()
                
                regional_stats.append(stats)
            
            result_df = pd.DataFrame(regional_stats)
            
            # Calculate relative performance compared to global averages
            for metric in ['latency_ms', 'error_rate', 'audio_quality_score', 'engagement_score']:
                mean_col = f'{metric}_mean'
                if mean_col in result_df.columns:
                    global_mean = result_df[mean_col].mean()
                    result_df[f'{metric}_relative_performance'] = (
                        result_df[mean_col] / global_mean
                    )
            
            logger.info(f"Aggregated geographic performance for {len(result_df)} regions")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Geographic aggregation failed: {str(e)}")
            raise
    
    def calculate_revenue_impact_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive revenue impact metrics.
        
        Args:
            data: DataFrame containing user activity and revenue data
            
        Returns:
            DataFrame with calculated revenue impact metrics
        """
        try:
            processed_data = data.copy()
            
            # Revenue per user calculations
            if 'total_revenue_usd' in data.columns and 'unique_users' in data.columns:
                processed_data['revenue_per_user'] = (
                    data['total_revenue_usd'] / np.maximum(data['unique_users'], 1)
                )
            
            # Subscription conversion metrics
            if 'premium_conversions' in data.columns and 'free_users' in data.columns:
                processed_data['conversion_rate'] = (
                    data['premium_conversions'] / np.maximum(data['free_users'], 1)
                )
            
            # Ad revenue efficiency
            if 'ad_revenue_usd' in data.columns and 'ad_impressions' in data.columns:
                processed_data['revenue_per_impression'] = (
                    data['ad_revenue_usd'] / np.maximum(data['ad_impressions'], 1)
                )
            
            # Content licensing ROI
            if 'content_licensing_cost' in data.columns and 'streaming_hours' in data.columns:
                processed_data['licensing_cost_per_hour'] = (
                    data['content_licensing_cost'] / np.maximum(data['streaming_hours'], 1)
                )
            
            # Churn impact calculation
            if 'churned_users' in data.columns and 'user_lifetime_value' in data.columns:
                processed_data['churn_revenue_impact'] = (
                    data['churned_users'] * data['user_lifetime_value']
                )
            
            # Calculate total business impact score
            impact_features = ['revenue_per_user', 'conversion_rate', 'revenue_per_impression']
            available_features = [f for f in impact_features if f in processed_data.columns]
            
            if available_features:
                processed_data['business_impact_score'] = self._calculate_business_impact_score(
                    processed_data[available_features]
                )
            
            logger.info(f"Calculated revenue impact metrics for {len(data)} records")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Revenue impact calculation failed: {str(e)}")
            raise
    
    def _calculate_perceived_quality(self, features: pd.DataFrame) -> pd.Series:
        """Calculate perceived audio quality score based on technical metrics."""
        # Normalize features to 0-1 scale
        normalized_features = pd.DataFrame()
        
        if 'bitrate_kbps' in features.columns:
            normalized_features['bitrate_score'] = np.clip(features['bitrate_kbps'] / 320.0, 0, 1)
        
        if 'snr_db' in features.columns:
            normalized_features['snr_score'] = np.clip(features['snr_db'] / 96.0, 0, 1)
        
        if 'startup_latency_ms' in features.columns:
            normalized_features['latency_score'] = np.clip(1.0 - (features['startup_latency_ms'] / 1000.0), 0, 1)
        
        if 'buffering_events' in features.columns:
            normalized_features['buffering_score'] = np.clip(1.0 - (features['buffering_events'] / 10.0), 0, 1)
        
        if 'packet_loss_percent' in features.columns:
            normalized_features['network_score'] = np.clip(1.0 - (features['packet_loss_percent'] / 5.0), 0, 1)
        
        # Calculate weighted average
        weights = {'bitrate_score': 0.3, 'snr_score': 0.25, 'latency_score': 0.2, 
                  'buffering_score': 0.15, 'network_score': 0.1}
        
        quality_score = pd.Series(0.0, index=features.index)
        
        for feature, weight in weights.items():
            if feature in normalized_features.columns:
                quality_score += normalized_features[feature] * weight
        
        return quality_score * 10  # Scale to 0-10
    
    def _calculate_engagement_score(self, features: pd.DataFrame) -> pd.Series:
        """Calculate user engagement score based on behavior metrics."""
        # Normalize features to 0-1 scale
        normalized_features = pd.DataFrame()
        
        if 'completion_rate' in features.columns:
            normalized_features['completion'] = np.clip(features['completion_rate'], 0, 1)
        
        if 'like_rate' in features.columns:
            normalized_features['likes'] = np.clip(features['like_rate'], 0, 1)
        
        if 'share_rate' in features.columns:
            normalized_features['shares'] = np.clip(features['share_rate'], 0, 1)
        
        if 'discovery_rate' in features.columns:
            normalized_features['discovery'] = np.clip(features['discovery_rate'] / 2.0, 0, 1)
        
        if 'session_duration_minutes' in features.columns:
            normalized_features['session'] = np.clip(features['session_duration_minutes'] / 60.0, 0, 1)
        
        # Calculate weighted average
        weights = {'completion': 0.3, 'likes': 0.2, 'shares': 0.15, 
                  'discovery': 0.2, 'session': 0.15}
        
        engagement_score = pd.Series(0.0, index=features.index)
        
        for feature, weight in weights.items():
            if feature in normalized_features.columns:
                engagement_score += normalized_features[feature] * weight
        
        return engagement_score
    
    def _calculate_business_impact_score(self, features: pd.DataFrame) -> pd.Series:
        """Calculate business impact score based on revenue metrics."""
        # Normalize features to 0-1 scale
        normalized_features = pd.DataFrame()
        
        if 'revenue_per_user' in features.columns:
            max_rpu = features['revenue_per_user'].quantile(0.95)  # Use 95th percentile as max
            normalized_features['revenue'] = np.clip(features['revenue_per_user'] / max_rpu, 0, 1)
        
        if 'conversion_rate' in features.columns:
            normalized_features['conversion'] = np.clip(features['conversion_rate'] * 10, 0, 1)  # Assuming 10% is excellent
        
        if 'revenue_per_impression' in features.columns:
            max_rpi = features['revenue_per_impression'].quantile(0.95)
            normalized_features['ad_efficiency'] = np.clip(features['revenue_per_impression'] / max_rpi, 0, 1)
        
        # Calculate weighted average
        weights = {'revenue': 0.5, 'conversion': 0.3, 'ad_efficiency': 0.2}
        
        impact_score = pd.Series(0.0, index=features.index)
        
        for feature, weight in weights.items():
            if feature in normalized_features.columns:
                impact_score += normalized_features[feature] * weight
        
        return impact_score
    
    def _segment_users_by_behavior(self, features: pd.DataFrame) -> pd.Series:
        """Segment users based on behavior patterns using clustering."""
        try:
            # Fill missing values
            features_clean = features.fillna(features.median())
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_clean)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Map cluster numbers to meaningful segment names
            cluster_names = {
                0: 'casual_listener',
                1: 'engaged_explorer', 
                2: 'heavy_user',
                3: 'selective_listener'
            }
            
            return pd.Series([cluster_names.get(c, f'cluster_{c}') for c in clusters], 
                           index=features.index)
            
        except Exception as e:
            logger.warning(f"User segmentation failed: {e}. Using default segment.")
            return pd.Series('unknown', index=features.index)
    
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataset."""
        if 'timestamp' not in data.columns:
            return data
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Extract temporal features
        data['hour_of_day'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['is_peak_hours'] = data['hour_of_day'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Cyclical encoding for temporal features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data
    
    def _add_behavioral_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral pattern features."""
        # Listening intensity patterns
        if 'session_duration_minutes' in data.columns and 'tracks_played' in data.columns:
            data['tracks_per_minute'] = (
                data['tracks_played'] / np.maximum(data['session_duration_minutes'], 1)
            )
        
        # Exploration vs familiar content ratio
        if 'new_artists_discovered' in data.columns and 'tracks_played' in data.columns:
            data['exploration_ratio'] = (
                data['new_artists_discovered'] / np.maximum(data['tracks_played'], 1)
            )
        
        # Social engagement indicator
        if 'tracks_shared' in data.columns and 'tracks_liked' in data.columns:
            data['social_engagement'] = data['tracks_shared'] + data['tracks_liked']
        
        return data
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        avg_processing_time = (
            np.mean(self.processing_stats['processing_time_ms']) 
            if self.processing_stats['processing_time_ms'] else 0
        )
        
        return {
            'total_records_processed': self.processing_stats['total_processed'],
            'average_processing_time_ms': avg_processing_time,
            'error_count': self.processing_stats['error_count'],
            'last_update': self.processing_stats['last_update'].isoformat() if self.processing_stats['last_update'] else None,
            'error_rate': (
                self.processing_stats['error_count'] / self.processing_stats['total_processed'] 
                if self.processing_stats['total_processed'] > 0 else 0
            )
        }

# Export classes and functions
__all__ = [
    "AudioQualityMetrics",
    "UserBehaviorMetrics", 
    "MusicDataProcessor"
]
