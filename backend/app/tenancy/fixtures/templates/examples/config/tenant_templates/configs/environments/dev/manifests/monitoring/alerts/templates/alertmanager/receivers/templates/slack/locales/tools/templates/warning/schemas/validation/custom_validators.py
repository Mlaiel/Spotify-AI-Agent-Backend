"""
Validateurs personnalisés spécialisés - Spotify AI Agent
Règles de validation métier pour domaines spécifiques
"""

import re
import json
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse, parse_qs

from pydantic import validator, ValidationError
from pydantic.types import EmailStr, PositiveFloat, PositiveInt

from ..base.enums import AlertLevel, NotificationChannel, MLModelType
from . import ValidationRules


class SpotifyDomainValidators:
    """Validateurs spécifiques au domaine Spotify/Musique"""
    
    # Patterns pour identifiants Spotify
    SPOTIFY_TRACK_ID_PATTERN = re.compile(r'^[A-Za-z0-9]{22}$')
    SPOTIFY_ARTIST_ID_PATTERN = re.compile(r'^[A-Za-z0-9]{22}$')
    SPOTIFY_ALBUM_ID_PATTERN = re.compile(r'^[A-Za-z0-9]{22}$')
    SPOTIFY_PLAYLIST_ID_PATTERN = re.compile(r'^[A-Za-z0-9]{22}$')
    SPOTIFY_USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]+$')
    
    # Patterns pour URIs Spotify
    SPOTIFY_URI_PATTERN = re.compile(
        r'^spotify:(track|artist|album|playlist|user|show|episode):([A-Za-z0-9]{22}|[a-zA-Z0-9_.-]+)$'
    )
    
    @classmethod
    def validate_spotify_id(cls, value: str, entity_type: str) -> str:
        """Valide un ID Spotify selon le type d'entité"""
        if not value or not value.strip():
            raise ValueError(f"Spotify {entity_type} ID cannot be empty")
        
        value = value.strip()
        
        patterns = {
            'track': cls.SPOTIFY_TRACK_ID_PATTERN,
            'artist': cls.SPOTIFY_ARTIST_ID_PATTERN,
            'album': cls.SPOTIFY_ALBUM_ID_PATTERN,
            'playlist': cls.SPOTIFY_PLAYLIST_ID_PATTERN,
            'user': cls.SPOTIFY_USER_ID_PATTERN
        }
        
        pattern = patterns.get(entity_type.lower())
        if not pattern:
            raise ValueError(f"Unknown Spotify entity type: {entity_type}")
        
        if not pattern.match(value):
            raise ValueError(f"Invalid Spotify {entity_type} ID format: {value}")
        
        return value
    
    @classmethod
    def validate_spotify_uri(cls, value: str) -> str:
        """Valide un URI Spotify complet"""
        if not value or not value.strip():
            raise ValueError("Spotify URI cannot be empty")
        
        value = value.strip()
        
        if not cls.SPOTIFY_URI_PATTERN.match(value):
            raise ValueError(f"Invalid Spotify URI format: {value}")
        
        return value
    
    @classmethod
    def validate_audio_features(cls, features: Dict[str, float]) -> Dict[str, float]:
        """Valide les caractéristiques audio Spotify"""
        if not features:
            return features
        
        # Caractéristiques avec leurs plages valides
        valid_ranges = {
            'acousticness': (0.0, 1.0),
            'danceability': (0.0, 1.0),
            'energy': (0.0, 1.0),
            'instrumentalness': (0.0, 1.0),
            'liveness': (0.0, 1.0),
            'speechiness': (0.0, 1.0),
            'valence': (0.0, 1.0),
            'tempo': (0.0, 300.0),
            'loudness': (-60.0, 5.0),
            'mode': (0, 1),
            'key': (0, 11),
            'time_signature': (3, 7),
            'duration_ms': (1000, 3600000)  # 1 seconde à 1 heure
        }
        
        validated_features = {}
        
        for feature, value in features.items():
            if feature not in valid_ranges:
                raise ValueError(f"Unknown audio feature: {feature}")
            
            min_val, max_val = valid_ranges[feature]
            
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Audio feature '{feature}' must be numeric")
            
            if not min_val <= numeric_value <= max_val:
                raise ValueError(
                    f"Audio feature '{feature}' must be between {min_val} and {max_val}"
                )
            
            # Arrondi pour éviter les problèmes de précision
            if feature in ['mode', 'key', 'time_signature', 'duration_ms']:
                validated_features[feature] = int(numeric_value)
            else:
                validated_features[feature] = round(numeric_value, 4)
        
        return validated_features
    
    @classmethod
    def validate_genre(cls, value: str) -> str:
        """Valide un genre musical"""
        if not value or not value.strip():
            raise ValueError("Genre cannot be empty")
        
        value = value.strip().lower()
        
        # Validation basique du format
        if len(value) > 50:
            raise ValueError("Genre name cannot exceed 50 characters")
        
        if not re.match(r'^[a-z0-9\s\-&]+$', value):
            raise ValueError("Genre contains invalid characters")
        
        # Genres interdits ou inappropriés
        forbidden_genres = {'explicit', 'nsfw', 'adult'}
        if value in forbidden_genres:
            raise ValueError(f"Genre '{value}' is not allowed")
        
        return value


class AudioProcessingValidators:
    """Validateurs pour le traitement audio et ML"""
    
    @classmethod
    def validate_audio_format(cls, format_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Valide les spécifications de format audio"""
        if not format_spec:
            raise ValueError("Audio format specification cannot be empty")
        
        validated_spec = {}
        
        # Sample rate
        sample_rate = format_spec.get('sample_rate')
        if sample_rate:
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise ValueError("Sample rate must be a positive integer")
            
            # Rates standards supportés
            valid_rates = {8000, 16000, 22050, 44100, 48000, 96000, 192000}
            if sample_rate not in valid_rates:
                raise ValueError(f"Unsupported sample rate: {sample_rate}")
            
            validated_spec['sample_rate'] = sample_rate
        
        # Channels
        channels = format_spec.get('channels')
        if channels:
            if not isinstance(channels, int) or not 1 <= channels <= 8:
                raise ValueError("Channels must be between 1 and 8")
            
            validated_spec['channels'] = channels
        
        # Bit depth
        bit_depth = format_spec.get('bit_depth')
        if bit_depth:
            valid_depths = {8, 16, 24, 32}
            if bit_depth not in valid_depths:
                raise ValueError(f"Unsupported bit depth: {bit_depth}")
            
            validated_spec['bit_depth'] = bit_depth
        
        # Format
        audio_format = format_spec.get('format')
        if audio_format:
            valid_formats = {'wav', 'mp3', 'flac', 'ogg', 'aac', 'm4a'}
            if audio_format.lower() not in valid_formats:
                raise ValueError(f"Unsupported audio format: {audio_format}")
            
            validated_spec['format'] = audio_format.lower()
        
        return validated_spec
    
    @classmethod
    def validate_ml_model_config(cls, config: Dict[str, Any], 
                                model_type: MLModelType) -> Dict[str, Any]:
        """Valide une configuration de modèle ML"""
        if not config:
            raise ValueError("ML model configuration cannot be empty")
        
        validated_config = {}
        
        # Validation selon le type de modèle
        if model_type == MLModelType.AUDIO_CLASSIFICATION:
            cls._validate_audio_classification_config(config, validated_config)
        elif model_type == MLModelType.RECOMMENDATION:
            cls._validate_recommendation_config(config, validated_config)
        elif model_type == MLModelType.ANOMALY_DETECTION:
            cls._validate_anomaly_detection_config(config, validated_config)
        elif model_type == MLModelType.SENTIMENT_ANALYSIS:
            cls._validate_sentiment_analysis_config(config, validated_config)
        
        # Validation commune
        cls._validate_common_ml_config(config, validated_config)
        
        return validated_config
    
    @classmethod
    def _validate_audio_classification_config(cls, config: Dict[str, Any], 
                                            validated: Dict[str, Any]) -> None:
        """Valide la config spécifique à la classification audio"""
        # Nombre de classes
        num_classes = config.get('num_classes')
        if num_classes:
            if not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError("Number of classes must be at least 2")
            if num_classes > 10000:
                raise ValueError("Number of classes cannot exceed 10000")
            validated['num_classes'] = num_classes
        
        # Architecture du modèle
        architecture = config.get('architecture')
        if architecture:
            valid_archs = {'cnn', 'rnn', 'transformer', 'resnet', 'efficientnet'}
            if architecture.lower() not in valid_archs:
                raise ValueError(f"Unsupported architecture: {architecture}")
            validated['architecture'] = architecture.lower()
    
    @classmethod
    def _validate_recommendation_config(cls, config: Dict[str, Any], 
                                      validated: Dict[str, Any]) -> None:
        """Valide la config spécifique aux recommandations"""
        # Algorithme de recommandation
        algorithm = config.get('algorithm')
        if algorithm:
            valid_algorithms = {
                'collaborative_filtering', 'content_based', 'hybrid',
                'matrix_factorization', 'deep_learning', 'knn'
            }
            if algorithm.lower() not in valid_algorithms:
                raise ValueError(f"Unsupported recommendation algorithm: {algorithm}")
            validated['algorithm'] = algorithm.lower()
        
        # Nombre de recommandations
        num_recommendations = config.get('num_recommendations')
        if num_recommendations:
            if not isinstance(num_recommendations, int) or not 1 <= num_recommendations <= 1000:
                raise ValueError("Number of recommendations must be between 1 and 1000")
            validated['num_recommendations'] = num_recommendations
    
    @classmethod
    def _validate_anomaly_detection_config(cls, config: Dict[str, Any], 
                                         validated: Dict[str, Any]) -> None:
        """Valide la config spécifique à la détection d'anomalies"""
        # Seuil d'anomalie
        threshold = config.get('anomaly_threshold')
        if threshold is not None:
            try:
                threshold_float = float(threshold)
                if not 0.0 <= threshold_float <= 1.0:
                    raise ValueError("Anomaly threshold must be between 0.0 and 1.0")
                validated['anomaly_threshold'] = threshold_float
            except (ValueError, TypeError):
                raise ValueError("Anomaly threshold must be numeric")
        
        # Méthode de détection
        method = config.get('detection_method')
        if method:
            valid_methods = {
                'isolation_forest', 'one_class_svm', 'autoencoder',
                'statistical', 'clustering', 'ensemble'
            }
            if method.lower() not in valid_methods:
                raise ValueError(f"Unsupported detection method: {method}")
            validated['detection_method'] = method.lower()
    
    @classmethod
    def _validate_sentiment_analysis_config(cls, config: Dict[str, Any], 
                                          validated: Dict[str, Any]) -> None:
        """Valide la config spécifique à l'analyse de sentiment"""
        # Langues supportées
        languages = config.get('supported_languages')
        if languages:
            if not isinstance(languages, list):
                raise ValueError("Supported languages must be a list")
            
            valid_languages = {
                'en', 'fr', 'de', 'es', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'
            }
            
            for lang in languages:
                if lang not in valid_languages:
                    raise ValueError(f"Unsupported language: {lang}")
            
            validated['supported_languages'] = languages
        
        # Granularité du sentiment
        granularity = config.get('sentiment_granularity')
        if granularity:
            valid_granularities = {'binary', 'ternary', 'fine_grained'}
            if granularity not in valid_granularities:
                raise ValueError(f"Unsupported sentiment granularity: {granularity}")
            validated['sentiment_granularity'] = granularity
    
    @classmethod
    def _validate_common_ml_config(cls, config: Dict[str, Any], 
                                 validated: Dict[str, Any]) -> None:
        """Valide les paramètres ML communs"""
        # Batch size
        batch_size = config.get('batch_size')
        if batch_size:
            if not isinstance(batch_size, int) or not 1 <= batch_size <= 10000:
                raise ValueError("Batch size must be between 1 and 10000")
            validated['batch_size'] = batch_size
        
        # Learning rate
        learning_rate = config.get('learning_rate')
        if learning_rate is not None:
            try:
                lr_float = float(learning_rate)
                if not 0.0 < lr_float <= 1.0:
                    raise ValueError("Learning rate must be between 0.0 and 1.0")
                validated['learning_rate'] = lr_float
            except (ValueError, TypeError):
                raise ValueError("Learning rate must be numeric")
        
        # Epochs
        epochs = config.get('epochs')
        if epochs:
            if not isinstance(epochs, int) or not 1 <= epochs <= 10000:
                raise ValueError("Epochs must be between 1 and 10000")
            validated['epochs'] = epochs


class PerformanceValidators:
    """Validateurs pour métriques de performance"""
    
    @classmethod
    def validate_latency_metrics(cls, metrics: Dict[str, float]) -> Dict[str, float]:
        """Valide les métriques de latence"""
        if not metrics:
            return metrics
        
        validated_metrics = {}
        
        # Métriques de latence valides (en millisecondes)
        valid_metrics = {
            'p50', 'p90', 'p95', 'p99', 'p99.9',
            'mean', 'median', 'min', 'max', 'stddev'
        }
        
        for metric, value in metrics.items():
            if metric not in valid_metrics:
                raise ValueError(f"Unknown latency metric: {metric}")
            
            try:
                value_float = float(value)
                if value_float < 0:
                    raise ValueError(f"Latency metric '{metric}' cannot be negative")
                if value_float > 3600000:  # 1 heure max
                    raise ValueError(f"Latency metric '{metric}' exceeds maximum (1 hour)")
                
                validated_metrics[metric] = round(value_float, 2)
            except (ValueError, TypeError):
                raise ValueError(f"Latency metric '{metric}' must be numeric")
        
        # Validation de cohérence (p50 <= p90 <= p95 <= p99)
        percentiles = ['p50', 'p90', 'p95', 'p99', 'p99.9']
        prev_value = 0
        
        for percentile in percentiles:
            if percentile in validated_metrics:
                current_value = validated_metrics[percentile]
                if current_value < prev_value:
                    raise ValueError(
                        f"Percentile {percentile} ({current_value}) cannot be less than "
                        f"previous percentile ({prev_value})"
                    )
                prev_value = current_value
        
        return validated_metrics
    
    @classmethod
    def validate_throughput_metrics(cls, metrics: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
        """Valide les métriques de débit"""
        if not metrics:
            return metrics
        
        validated_metrics = {}
        
        # Métriques de débit valides
        valid_metrics = {
            'requests_per_second': float,
            'requests_per_minute': float,
            'requests_per_hour': float,
            'concurrent_users': int,
            'active_connections': int,
            'queue_length': int,
            'processing_rate': float
        }
        
        for metric, value in metrics.items():
            if metric not in valid_metrics:
                raise ValueError(f"Unknown throughput metric: {metric}")
            
            expected_type = valid_metrics[metric]
            
            try:
                if expected_type == int:
                    validated_value = int(value)
                    if validated_value < 0:
                        raise ValueError(f"Throughput metric '{metric}' cannot be negative")
                else:
                    validated_value = float(value)
                    if validated_value < 0.0:
                        raise ValueError(f"Throughput metric '{metric}' cannot be negative")
                    validated_value = round(validated_value, 2)
                
                # Limites raisonnables
                if metric in ['requests_per_second', 'processing_rate'] and validated_value > 1000000:
                    raise ValueError(f"Throughput metric '{metric}' exceeds reasonable limit")
                if metric in ['concurrent_users', 'active_connections'] and validated_value > 100000:
                    raise ValueError(f"Throughput metric '{metric}' exceeds reasonable limit")
                
                validated_metrics[metric] = validated_value
                
            except (ValueError, TypeError):
                raise ValueError(f"Throughput metric '{metric}' must be {expected_type.__name__}")
        
        return validated_metrics
    
    @classmethod
    def validate_resource_usage(cls, usage: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
        """Valide l'utilisation des ressources"""
        if not usage:
            return usage
        
        validated_usage = {}
        
        # Métriques d'utilisation avec leurs limites
        usage_limits = {
            'cpu_percent': (0.0, 100.0, float),
            'memory_percent': (0.0, 100.0, float),
            'memory_bytes': (0, 1099511627776, int),  # 1TB max
            'disk_percent': (0.0, 100.0, float),
            'disk_bytes': (0, 10995116277760, int),  # 10TB max
            'network_in_bytes': (0, 1073741824000, int),  # 1TB/s max
            'network_out_bytes': (0, 1073741824000, int),
            'open_files': (0, 65536, int),
            'threads': (0, 10000, int)
        }
        
        for metric, value in usage.items():
            if metric not in usage_limits:
                raise ValueError(f"Unknown resource usage metric: {metric}")
            
            min_val, max_val, expected_type = usage_limits[metric]
            
            try:
                if expected_type == int:
                    validated_value = int(value)
                else:
                    validated_value = float(value)
                    validated_value = round(validated_value, 2)
                
                if not min_val <= validated_value <= max_val:
                    raise ValueError(
                        f"Resource usage '{metric}' must be between {min_val} and {max_val}"
                    )
                
                validated_usage[metric] = validated_value
                
            except (ValueError, TypeError):
                raise ValueError(f"Resource usage '{metric}' must be {expected_type.__name__}")
        
        return validated_usage


# Décorateurs pour validation spécialisée
def validate_spotify_track_id():
    """Décorateur pour valider un ID de track Spotify"""
    return validator('track_id', allow_reuse=True)(
        lambda v: SpotifyDomainValidators.validate_spotify_id(v, 'track')
    )

def validate_spotify_artist_id():
    """Décorateur pour valider un ID d'artiste Spotify"""
    return validator('artist_id', allow_reuse=True)(
        lambda v: SpotifyDomainValidators.validate_spotify_id(v, 'artist')
    )

def validate_audio_features_field():
    """Décorateur pour valider les caractéristiques audio"""
    return validator('audio_features', allow_reuse=True)(
        SpotifyDomainValidators.validate_audio_features
    )

def validate_latency_metrics_field():
    """Décorateur pour valider les métriques de latence"""
    return validator('latency_metrics', allow_reuse=True)(
        PerformanceValidators.validate_latency_metrics
    )

def validate_throughput_metrics_field():
    """Décorateur pour valider les métriques de débit"""
    return validator('throughput_metrics', allow_reuse=True)(
        PerformanceValidators.validate_throughput_metrics
    )
