"""
ML Collectors - Collecteurs de Machine Learning et IA
===================================================

Collecteurs spécialisés pour surveiller les performances des modèles ML
et les systèmes d'IA du Spotify AI Agent.

Features:
    - Monitoring des modèles ML en temps réel
    - Métriques de performance des agents IA
    - Surveillance de la dérive des données (data drift)
    - Analyse des systèmes de recommandation
    - Monitoring de la génération audio et musique

Author: Ingénieur Machine Learning + Spécialiste Audio IA Team
"""

import asyncio
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
from collections import defaultdict, deque
import pickle
import base64

from . import BaseCollector, CollectorConfig

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types de modèles ML supportés."""
    RECOMMENDATION = "recommendation"
    AUDIO_GENERATION = "audio_generation"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class ModelStatus(Enum):
    """Statuts des modèles."""
    TRAINING = "training"
    DEPLOYED = "deployed"
    SERVING = "serving"
    DEGRADED = "degraded"
    FAILED = "failed"
    RETIRED = "retired"


@dataclass
class ModelMetrics:
    """Métriques de performance d'un modèle."""
    model_id: str
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    mae: Optional[float] = None  # Mean Absolute Error
    rmse: Optional[float] = None  # Root Mean Square Error
    inference_time_ms: float = 0.0
    throughput: float = 0.0  # Requêtes par seconde
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class AudioGenerationMetrics:
    """Métriques spécifiques à la génération audio."""
    model_id: str
    generation_time_seconds: float
    audio_quality_score: float
    musical_coherence_score: float
    creativity_index: float
    genre_accuracy: float
    harmonic_complexity: float
    rhythm_consistency: float
    user_satisfaction_rating: float


class MLModelPerformanceCollector(BaseCollector):
    """Collecteur principal de performance des modèles ML."""
    
    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self.model_registry = ModelRegistry()
        self.drift_detector = DataDriftDetector()
        self.performance_tracker = ModelPerformanceTracker()
        self.anomaly_detector = MLAnomalyDetector()
        
    async def collect(self) -> Dict[str, Any]:
        """Collecte complète des métriques ML."""
        tenant_id = self.config.tags.get('tenant_id', 'default')
        
        try:
            # Récupération des modèles actifs
            active_models = await self.model_registry.get_active_models(tenant_id)
            
            # Collecte des métriques par modèle
            model_metrics = []
            for model in active_models:
                metrics = await self._collect_model_metrics(model)
                model_metrics.append(metrics)
            
            # Détection de dérive des données
            drift_analysis = await self.drift_detector.detect_drift(tenant_id, active_models)
            
            # Analyse des performances globales
            performance_summary = await self.performance_tracker.analyze_performance(model_metrics)
            
            # Détection d'anomalies ML
            ml_anomalies = await self.anomaly_detector.detect_anomalies(model_metrics)
            
            # Score de santé ML global
            ml_health_score = self._calculate_ml_health_score(
                model_metrics, drift_analysis, ml_anomalies
            )
            
            return {
                'ml_model_performance': {
                    'tenant_id': tenant_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'active_models_count': len(active_models),
                    'model_metrics': model_metrics,
                    'drift_analysis': drift_analysis,
                    'performance_summary': performance_summary,
                    'anomalies': ml_anomalies,
                    'ml_health_score': ml_health_score,
                    'recommendations': await self._generate_ml_recommendations(
                        model_metrics, drift_analysis, ml_anomalies
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques ML: {str(e)}")
            raise
    
    async def _collect_model_metrics(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Collecte les métriques pour un modèle spécifique."""
        model_id = model['id']
        model_type = ModelType(model['type'])
        
        # Métriques de base (simulées - en production, requête API/DB)
        base_metrics = ModelMetrics(
            model_id=model_id,
            model_type=model_type,
            accuracy=0.87 + np.random.normal(0, 0.02),
            precision=0.85 + np.random.normal(0, 0.02),
            recall=0.83 + np.random.normal(0, 0.02),
            f1_score=0.84 + np.random.normal(0, 0.02),
            auc_roc=0.91 + np.random.normal(0, 0.01),
            inference_time_ms=45.5 + np.random.normal(0, 5),
            throughput=150.0 + np.random.normal(0, 10),
            memory_usage_mb=512.0 + np.random.normal(0, 50),
            cpu_usage_percent=35.0 + np.random.normal(0, 5)
        )
        
        # Métriques spécifiques par type de modèle
        specific_metrics = await self._get_model_specific_metrics(model_id, model_type)
        
        # Historique de performance
        performance_history = await self._get_performance_history(model_id)
        
        # Métriques d'utilisation
        usage_metrics = await self._get_usage_metrics(model_id)
        
        return {
            'model_id': model_id,
            'model_type': model_type.value,
            'status': model.get('status', ModelStatus.SERVING.value),
            'version': model.get('version', '1.0.0'),
            'deployment_date': model.get('deployment_date'),
            'base_metrics': {
                'accuracy': base_metrics.accuracy,
                'precision': base_metrics.precision,
                'recall': base_metrics.recall,
                'f1_score': base_metrics.f1_score,
                'auc_roc': base_metrics.auc_roc,
                'inference_time_ms': base_metrics.inference_time_ms,
                'throughput': base_metrics.throughput,
                'memory_usage_mb': base_metrics.memory_usage_mb,
                'cpu_usage_percent': base_metrics.cpu_usage_percent
            },
            'specific_metrics': specific_metrics,
            'performance_history': performance_history,
            'usage_metrics': usage_metrics,
            'data_quality': await self._assess_data_quality(model_id),
            'feature_importance': await self._get_feature_importance(model_id)
        }
    
    async def _get_model_specific_metrics(self, model_id: str, 
                                        model_type: ModelType) -> Dict[str, Any]:
        """Récupère les métriques spécifiques au type de modèle."""
        if model_type == ModelType.RECOMMENDATION:
            return await self._get_recommendation_metrics(model_id)
        elif model_type == ModelType.AUDIO_GENERATION:
            return await self._get_audio_generation_metrics(model_id)
        elif model_type == ModelType.NLP:
            return await self._get_nlp_metrics(model_id)
        else:
            return {}
    
    async def _get_recommendation_metrics(self, model_id: str) -> Dict[str, Any]:
        """Métriques spécifiques aux modèles de recommandation."""
        return {
            'click_through_rate': 0.23 + np.random.normal(0, 0.02),
            'conversion_rate': 0.12 + np.random.normal(0, 0.01),
            'diversity_score': 0.68 + np.random.normal(0, 0.05),
            'novelty_score': 0.45 + np.random.normal(0, 0.03),
            'coverage': 0.78 + np.random.normal(0, 0.02),
            'popularity_bias': 0.32 + np.random.normal(0, 0.05),
            'cold_start_performance': 0.56 + np.random.normal(0, 0.04),
            'collaborative_filtering_score': 0.82,
            'content_based_score': 0.75,
            'hybrid_score': 0.88
        }
    
    async def _get_audio_generation_metrics(self, model_id: str) -> Dict[str, Any]:
        """Métriques spécifiques à la génération audio."""
        return {
            'generation_time_seconds': 15.5 + np.random.normal(0, 2),
            'audio_quality_score': 0.89 + np.random.normal(0, 0.03),
            'musical_coherence_score': 0.76 + np.random.normal(0, 0.05),
            'creativity_index': 0.68 + np.random.normal(0, 0.04),
            'genre_accuracy': 0.91 + np.random.normal(0, 0.02),
            'harmonic_complexity': 0.73 + np.random.normal(0, 0.04),
            'rhythm_consistency': 0.85 + np.random.normal(0, 0.03),
            'user_satisfaction_rating': 4.2 + np.random.normal(0, 0.2),
            'copyright_compliance_score': 0.98,
            'originality_score': 0.82
        }
    
    async def _get_nlp_metrics(self, model_id: str) -> Dict[str, Any]:
        """Métriques spécifiques aux modèles NLP."""
        return {
            'bleu_score': 0.78 + np.random.normal(0, 0.03),
            'rouge_l': 0.72 + np.random.normal(0, 0.04),
            'perplexity': 35.2 + np.random.normal(0, 5),
            'semantic_similarity': 0.83 + np.random.normal(0, 0.02),
            'sentiment_accuracy': 0.89 + np.random.normal(0, 0.02),
            'entity_recognition_f1': 0.91 + np.random.normal(0, 0.01),
            'language_detection_accuracy': 0.97
        }
    
    async def _get_performance_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Récupère l'historique de performance."""
        history = []
        for i in range(7):  # 7 derniers jours
            timestamp = datetime.utcnow() - timedelta(days=i)
            history.append({
                'timestamp': timestamp.isoformat(),
                'accuracy': 0.87 + np.random.normal(0, 0.02),
                'inference_time_ms': 45.0 + np.random.normal(0, 3),
                'requests_count': 1000 + int(np.random.normal(0, 100))
            })
        return history
    
    async def _get_usage_metrics(self, model_id: str) -> Dict[str, Any]:
        """Récupère les métriques d'utilisation."""
        return {
            'requests_per_hour': 150 + int(np.random.normal(0, 20)),
            'unique_users': 89 + int(np.random.normal(0, 10)),
            'peak_usage_hour': 14,  # 14h
            'geographic_distribution': {
                'US': 0.45,
                'EU': 0.35,
                'Asia': 0.20
            },
            'error_rate': 0.02 + np.random.normal(0, 0.005),
            'timeout_rate': 0.01 + np.random.normal(0, 0.002)
        }
    
    async def _assess_data_quality(self, model_id: str) -> Dict[str, Any]:
        """Évalue la qualité des données."""
        return {
            'completeness': 0.95 + np.random.normal(0, 0.02),
            'consistency': 0.92 + np.random.normal(0, 0.03),
            'accuracy': 0.89 + np.random.normal(0, 0.04),
            'timeliness': 0.97 + np.random.normal(0, 0.01),
            'validity': 0.94 + np.random.normal(0, 0.02),
            'duplicate_rate': 0.03 + np.random.normal(0, 0.01),
            'missing_values_rate': 0.05 + np.random.normal(0, 0.01)
        }
    
    async def _get_feature_importance(self, model_id: str) -> Dict[str, float]:
        """Récupère l'importance des features."""
        features = [
            'user_listening_history', 'genre_preference', 'time_of_day',
            'playlist_context', 'artist_similarity', 'audio_features',
            'collaborative_signal', 'popularity', 'release_date', 'duration'
        ]
        
        # Simulation d'importance des features
        importance_values = np.random.dirichlet(np.ones(len(features)))
        return dict(zip(features, importance_values.tolist()))
    
    def _calculate_ml_health_score(self, model_metrics: List[Dict], 
                                 drift_analysis: Dict, anomalies: List) -> float:
        """Calcule le score de santé ML global."""
        if not model_metrics:
            return 0.0
        
        # Score moyen de performance des modèles
        accuracy_scores = [
            m['base_metrics']['accuracy'] for m in model_metrics 
            if 'base_metrics' in m and 'accuracy' in m['base_metrics']
        ]
        avg_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0.5
        
        # Pénalité pour la dérive des données
        drift_penalty = drift_analysis.get('drift_score', 0) * 0.2
        
        # Pénalité pour les anomalies
        anomaly_penalty = len(anomalies) * 0.1
        
        # Score final (0-100)
        health_score = (avg_accuracy * 100) - (drift_penalty * 100) - (anomaly_penalty * 10)
        return max(0, min(100, health_score))
    
    async def _generate_ml_recommendations(self, model_metrics: List[Dict],
                                         drift_analysis: Dict, 
                                         anomalies: List) -> List[Dict[str, Any]]:
        """Génère des recommandations ML."""
        recommendations = []
        
        # Recommandations basées sur la performance
        low_performing_models = [
            m for m in model_metrics 
            if m.get('base_metrics', {}).get('accuracy', 1.0) < 0.8
        ]
        
        for model in low_performing_models:
            recommendations.append({
                'type': 'performance_improvement',
                'model_id': model['model_id'],
                'priority': 'high',
                'action': 'Retrain model with recent data',
                'expected_improvement': '5-10% accuracy increase',
                'effort_estimate': 'medium'
            })
        
        # Recommandations basées sur la dérive
        if drift_analysis.get('drift_detected', False):
            recommendations.append({
                'type': 'data_drift',
                'priority': 'high',
                'action': 'Investigate data drift and retrain affected models',
                'affected_models': drift_analysis.get('affected_models', []),
                'effort_estimate': 'high'
            })
        
        # Recommandations basées sur les anomalies
        if anomalies:
            recommendations.append({
                'type': 'anomaly_investigation',
                'priority': 'medium',
                'action': 'Investigate performance anomalies',
                'anomalies_count': len(anomalies),
                'effort_estimate': 'low'
            })
        
        return recommendations
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Valide les données ML collectées."""
        try:
            ml_data = data.get('ml_model_performance', {})
            
            # Vérification des champs obligatoires
            required_fields = ['tenant_id', 'active_models_count', 'ml_health_score']
            for field in required_fields:
                if field not in ml_data:
                    return False
            
            # Validation du score de santé
            health_score = ml_data.get('ml_health_score', -1)
            if not (0 <= health_score <= 100):
                return False
            
            # Validation des métriques de modèle
            model_metrics = ml_data.get('model_metrics', [])
            if not isinstance(model_metrics, list):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation données ML: {str(e)}")
            return False


class ModelRegistry:
    """Registre des modèles ML."""
    
    async def get_active_models(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Récupère la liste des modèles actifs."""
        # Simulation - en production, requête DB
        return [
            {
                'id': 'recommendation_model_v2',
                'type': 'recommendation',
                'status': 'serving',
                'version': '2.1.0',
                'deployment_date': '2024-01-15T10:00:00Z'
            },
            {
                'id': 'audio_generator_v1',
                'type': 'audio_generation',
                'status': 'serving',
                'version': '1.3.2',
                'deployment_date': '2024-01-10T08:30:00Z'
            },
            {
                'id': 'genre_classifier_v3',
                'type': 'classification',
                'status': 'serving',
                'version': '3.0.1',
                'deployment_date': '2024-01-20T14:15:00Z'
            },
            {
                'id': 'lyrics_generator_v1',
                'type': 'nlp',
                'status': 'serving',
                'version': '1.1.0',
                'deployment_date': '2024-01-18T11:45:00Z'
            }
        ]


class DataDriftDetector:
    """Détecteur de dérive des données."""
    
    async def detect_drift(self, tenant_id: str, models: List[Dict]) -> Dict[str, Any]:
        """Détecte la dérive des données pour les modèles."""
        try:
            # Analyse de dérive par modèle
            drift_results = {}
            overall_drift_score = 0.0
            affected_models = []
            
            for model in models:
                model_id = model['id']
                drift_score = await self._calculate_drift_score(model_id)
                
                drift_results[model_id] = {
                    'drift_score': drift_score,
                    'drift_detected': drift_score > 0.1,
                    'severity': self._classify_drift_severity(drift_score),
                    'affected_features': await self._identify_drifted_features(model_id),
                    'recommendations': await self._get_drift_recommendations(model_id, drift_score)
                }
                
                overall_drift_score += drift_score
                if drift_score > 0.1:
                    affected_models.append(model_id)
            
            overall_drift_score /= len(models) if models else 1
            
            return {
                'drift_detected': overall_drift_score > 0.1,
                'overall_drift_score': overall_drift_score,
                'affected_models': affected_models,
                'model_results': drift_results,
                'detection_timestamp': datetime.utcnow().isoformat(),
                'next_check': (datetime.utcnow() + timedelta(hours=6)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur détection drift: {str(e)}")
            return {}
    
    async def _calculate_drift_score(self, model_id: str) -> float:
        """Calcule le score de dérive pour un modèle."""
        # Simulation - en production, utiliserait des techniques comme KS-test, PSI, etc.
        base_drift = np.random.beta(2, 10)  # Distribution biaisée vers les faibles valeurs
        return min(1.0, base_drift)
    
    def _classify_drift_severity(self, drift_score: float) -> str:
        """Classifie la sévérité de la dérive."""
        if drift_score < 0.05:
            return 'low'
        elif drift_score < 0.15:
            return 'medium'
        else:
            return 'high'
    
    async def _identify_drifted_features(self, model_id: str) -> List[str]:
        """Identifie les features qui ont drifté."""
        all_features = [
            'user_age', 'listening_time', 'genre_preference', 
            'device_type', 'location', 'time_of_day'
        ]
        
        # Simulation - sélection aléatoire de features driftées
        num_drifted = np.random.poisson(1)  # En moyenne 1 feature driftée
        return np.random.choice(all_features, size=min(num_drifted, len(all_features)), replace=False).tolist()
    
    async def _get_drift_recommendations(self, model_id: str, drift_score: float) -> List[str]:
        """Recommandations pour traiter la dérive."""
        recommendations = []
        
        if drift_score > 0.15:
            recommendations.append("Immediate model retraining required")
            recommendations.append("Investigate data source changes")
        elif drift_score > 0.05:
            recommendations.append("Schedule model retraining within 7 days")
            recommendations.append("Monitor feature distributions closely")
        else:
            recommendations.append("Continue normal monitoring")
        
        return recommendations


class ModelPerformanceTracker:
    """Tracker de performance des modèles."""
    
    async def analyze_performance(self, model_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyse la performance globale des modèles."""
        if not model_metrics:
            return {}
        
        # Agrégation des métriques
        accuracy_values = [m['base_metrics']['accuracy'] for m in model_metrics]
        inference_times = [m['base_metrics']['inference_time_ms'] for m in model_metrics]
        throughput_values = [m['base_metrics']['throughput'] for m in model_metrics]
        
        # Calculs statistiques
        performance_summary = {
            'total_models': len(model_metrics),
            'average_accuracy': statistics.mean(accuracy_values),
            'median_accuracy': statistics.median(accuracy_values),
            'accuracy_std': statistics.stdev(accuracy_values) if len(accuracy_values) > 1 else 0,
            'average_inference_time_ms': statistics.mean(inference_times),
            'median_inference_time_ms': statistics.median(inference_times),
            'total_throughput': sum(throughput_values),
            'models_by_type': self._count_models_by_type(model_metrics),
            'performance_distribution': self._analyze_performance_distribution(accuracy_values),
            'sla_compliance': await self._check_sla_compliance(model_metrics),
            'trends': await self._analyze_performance_trends(model_metrics)
        }
        
        return performance_summary
    
    def _count_models_by_type(self, model_metrics: List[Dict]) -> Dict[str, int]:
        """Compte les modèles par type."""
        type_counts = defaultdict(int)
        for model in model_metrics:
            model_type = model.get('model_type', 'unknown')
            type_counts[model_type] += 1
        return dict(type_counts)
    
    def _analyze_performance_distribution(self, accuracy_values: List[float]) -> Dict[str, int]:
        """Analyse la distribution des performances."""
        distribution = {
            'excellent': 0,    # > 0.9
            'good': 0,         # 0.8 - 0.9
            'acceptable': 0,   # 0.7 - 0.8
            'poor': 0          # < 0.7
        }
        
        for acc in accuracy_values:
            if acc > 0.9:
                distribution['excellent'] += 1
            elif acc > 0.8:
                distribution['good'] += 1
            elif acc > 0.7:
                distribution['acceptable'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    async def _check_sla_compliance(self, model_metrics: List[Dict]) -> Dict[str, Any]:
        """Vérifie la conformité aux SLA."""
        sla_thresholds = {
            'min_accuracy': 0.8,
            'max_inference_time_ms': 100,
            'min_throughput': 50
        }
        
        compliant_models = 0
        total_models = len(model_metrics)
        
        violations = []
        
        for model in model_metrics:
            base_metrics = model.get('base_metrics', {})
            model_compliant = True
            
            # Vérification accuracy
            if base_metrics.get('accuracy', 0) < sla_thresholds['min_accuracy']:
                violations.append({
                    'model_id': model['model_id'],
                    'violation_type': 'low_accuracy',
                    'current_value': base_metrics.get('accuracy', 0),
                    'threshold': sla_thresholds['min_accuracy']
                })
                model_compliant = False
            
            # Vérification temps d'inférence
            if base_metrics.get('inference_time_ms', 0) > sla_thresholds['max_inference_time_ms']:
                violations.append({
                    'model_id': model['model_id'],
                    'violation_type': 'high_latency',
                    'current_value': base_metrics.get('inference_time_ms', 0),
                    'threshold': sla_thresholds['max_inference_time_ms']
                })
                model_compliant = False
            
            # Vérification throughput
            if base_metrics.get('throughput', 0) < sla_thresholds['min_throughput']:
                violations.append({
                    'model_id': model['model_id'],
                    'violation_type': 'low_throughput',
                    'current_value': base_metrics.get('throughput', 0),
                    'threshold': sla_thresholds['min_throughput']
                })
                model_compliant = False
            
            if model_compliant:
                compliant_models += 1
        
        compliance_rate = compliant_models / total_models if total_models > 0 else 0
        
        return {
            'compliance_rate': compliance_rate,
            'compliant_models': compliant_models,
            'total_models': total_models,
            'violations': violations,
            'sla_thresholds': sla_thresholds
        }
    
    async def _analyze_performance_trends(self, model_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyse les tendances de performance."""
        # En production, analyserait l'historique des performances
        return {
            'accuracy_trend': 'stable',      # 'improving', 'stable', 'degrading'
            'latency_trend': 'improving',
            'throughput_trend': 'stable',
            'overall_trend': 'stable',
            'trend_confidence': 0.85
        }


class MLAnomalyDetector:
    """Détecteur d'anomalies ML."""
    
    async def detect_anomalies(self, model_metrics: List[Dict]) -> List[Dict[str, Any]]:
        """Détecte les anomalies dans les métriques ML."""
        anomalies = []
        
        if not model_metrics:
            return anomalies
        
        # Détection d'anomalies d'accuracy
        accuracy_values = [m['base_metrics']['accuracy'] for m in model_metrics]
        accuracy_anomalies = self._detect_statistical_anomalies(accuracy_values, 'accuracy')
        
        for i, anomaly in enumerate(accuracy_anomalies):
            if anomaly:
                anomalies.append({
                    'model_id': model_metrics[i]['model_id'],
                    'anomaly_type': 'accuracy_anomaly',
                    'severity': 'high' if accuracy_values[i] < 0.7 else 'medium',
                    'current_value': accuracy_values[i],
                    'expected_range': [statistics.mean(accuracy_values) - 2*statistics.stdev(accuracy_values),
                                     statistics.mean(accuracy_values) + 2*statistics.stdev(accuracy_values)],
                    'description': f'Accuracy significantly lower than expected: {accuracy_values[i]:.3f}'
                })
        
        # Détection d'anomalies de latence
        latency_values = [m['base_metrics']['inference_time_ms'] for m in model_metrics]
        latency_anomalies = self._detect_statistical_anomalies(latency_values, 'latency')
        
        for i, anomaly in enumerate(latency_anomalies):
            if anomaly:
                anomalies.append({
                    'model_id': model_metrics[i]['model_id'],
                    'anomaly_type': 'latency_anomaly',
                    'severity': 'high' if latency_values[i] > 200 else 'medium',
                    'current_value': latency_values[i],
                    'expected_range': [0, statistics.mean(latency_values) + 2*statistics.stdev(latency_values)],
                    'description': f'Inference time significantly higher than expected: {latency_values[i]:.1f}ms'
                })
        
        # Détection d'anomalies de throughput
        throughput_values = [m['base_metrics']['throughput'] for m in model_metrics]
        throughput_anomalies = self._detect_statistical_anomalies(throughput_values, 'throughput')
        
        for i, anomaly in enumerate(throughput_anomalies):
            if anomaly:
                anomalies.append({
                    'model_id': model_metrics[i]['model_id'],
                    'anomaly_type': 'throughput_anomaly',
                    'severity': 'medium',
                    'current_value': throughput_values[i],
                    'expected_range': [statistics.mean(throughput_values) - 2*statistics.stdev(throughput_values),
                                     statistics.mean(throughput_values) + 2*statistics.stdev(throughput_values)],
                    'description': f'Throughput outside expected range: {throughput_values[i]:.1f} req/s'
                })
        
        return anomalies
    
    def _detect_statistical_anomalies(self, values: List[float], 
                                    metric_name: str, z_threshold: float = 2.0) -> List[bool]:
        """Détecte les anomalies statistiques using Z-score."""
        if len(values) < 2:
            return [False] * len(values)
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:
            return [False] * len(values)
        
        anomalies = []
        for value in values:
            z_score = abs((value - mean_val) / std_val)
            anomalies.append(z_score > z_threshold)
        
        return anomalies


class AIAgentMetricsCollector(BaseCollector):
    """Collecteur de métriques pour les agents IA."""
    
    async def collect(self) -> Dict[str, Any]:
        """Collecte les métriques des agents IA."""
        tenant_id = self.config.tags.get('tenant_id', 'default')
        
        try:
            # Métriques des agents IA
            agent_performance = await self._collect_agent_performance(tenant_id)
            
            # Métriques de dialogue et interaction
            dialogue_metrics = await self._collect_dialogue_metrics(tenant_id)
            
            # Métriques de satisfaction utilisateur
            satisfaction_metrics = await self._collect_satisfaction_metrics(tenant_id)
            
            # Métriques d'apprentissage et adaptation
            learning_metrics = await self._collect_learning_metrics(tenant_id)
            
            return {
                'ai_agent_metrics': {
                    'tenant_id': tenant_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'agent_performance': agent_performance,
                    'dialogue_metrics': dialogue_metrics,
                    'satisfaction_metrics': satisfaction_metrics,
                    'learning_metrics': learning_metrics,
                    'overall_agent_score': self._calculate_agent_score(
                        agent_performance, dialogue_metrics, satisfaction_metrics
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques agent IA: {str(e)}")
            raise
    
    async def _collect_agent_performance(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques de performance des agents."""
        return {
            'response_time_ms': 847.5 + np.random.normal(0, 100),
            'task_completion_rate': 0.89 + np.random.normal(0, 0.05),
            'accuracy_rate': 0.92 + np.random.normal(0, 0.03),
            'error_rate': 0.03 + np.random.normal(0, 0.01),
            'context_understanding': 0.87 + np.random.normal(0, 0.04),
            'goal_achievement_rate': 0.84 + np.random.normal(0, 0.05),
            'multi_turn_coherence': 0.91 + np.random.normal(0, 0.03),
            'creative_output_quality': 0.78 + np.random.normal(0, 0.06)
        }
    
    async def _collect_dialogue_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques de dialogue."""
        return {
            'average_conversation_length': 5.7 + np.random.normal(0, 1),
            'dialogue_success_rate': 0.86 + np.random.normal(0, 0.04),
            'intent_recognition_accuracy': 0.94 + np.random.normal(0, 0.02),
            'entity_extraction_accuracy': 0.91 + np.random.normal(0, 0.03),
            'response_relevance_score': 0.88 + np.random.normal(0, 0.04),
            'conversation_flow_quality': 0.82 + np.random.normal(0, 0.05),
            'fallback_rate': 0.07 + np.random.normal(0, 0.02),
            'clarification_requests': 0.12 + np.random.normal(0, 0.03)
        }
    
    async def _collect_satisfaction_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques de satisfaction."""
        return {
            'user_satisfaction_score': 4.2 + np.random.normal(0, 0.3),
            'helpfulness_rating': 4.1 + np.random.normal(0, 0.4),
            'ease_of_use_rating': 4.3 + np.random.normal(0, 0.2),
            'response_quality_rating': 4.0 + np.random.normal(0, 0.4),
            'recommendation_rate': 0.78 + np.random.normal(0, 0.05),
            'return_user_rate': 0.67 + np.random.normal(0, 0.06),
            'negative_feedback_rate': 0.08 + np.random.normal(0, 0.02),
            'feature_adoption_rate': 0.72 + np.random.normal(0, 0.05)
        }
    
    async def _collect_learning_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques d'apprentissage."""
        return {
            'adaptation_speed': 0.73 + np.random.normal(0, 0.05),
            'knowledge_retention': 0.89 + np.random.normal(0, 0.03),
            'learning_from_feedback': 0.81 + np.random.normal(0, 0.04),
            'personalization_effectiveness': 0.76 + np.random.normal(0, 0.05),
            'pattern_recognition_improvement': 0.68 + np.random.normal(0, 0.06),
            'new_skill_acquisition_rate': 0.45 + np.random.normal(0, 0.08),
            'knowledge_graph_expansion': 0.34 + np.random.normal(0, 0.05),
            'transfer_learning_success': 0.67 + np.random.normal(0, 0.04)
        }
    
    def _calculate_agent_score(self, performance: Dict, dialogue: Dict, 
                             satisfaction: Dict) -> float:
        """Calcule un score global de l'agent IA."""
        # Pondération des différentes métriques
        performance_score = (
            performance['task_completion_rate'] * 0.3 +
            performance['accuracy_rate'] * 0.25 +
            performance['context_understanding'] * 0.25 +
            performance['goal_achievement_rate'] * 0.2
        )
        
        dialogue_score = (
            dialogue['dialogue_success_rate'] * 0.4 +
            dialogue['intent_recognition_accuracy'] * 0.3 +
            dialogue['response_relevance_score'] * 0.3
        )
        
        satisfaction_score = satisfaction['user_satisfaction_score'] / 5.0
        
        # Score composite
        overall_score = (
            performance_score * 0.4 +
            dialogue_score * 0.3 +
            satisfaction_score * 0.3
        ) * 100
        
        return round(overall_score, 2)
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Valide les données d'agent IA."""
        try:
            agent_data = data.get('ai_agent_metrics', {})
            
            required_sections = ['agent_performance', 'dialogue_metrics', 'satisfaction_metrics']
            for section in required_sections:
                if section not in agent_data:
                    return False
            
            # Validation du score global
            overall_score = agent_data.get('overall_agent_score', -1)
            if not (0 <= overall_score <= 100):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation données agent IA: {str(e)}")
            return False


class RecommendationSystemCollector(BaseCollector):
    """Collecteur spécialisé pour les systèmes de recommandation."""
    
    async def collect(self) -> Dict[str, Any]:
        """Collecte les métriques des systèmes de recommandation."""
        tenant_id = self.config.tags.get('tenant_id', 'default')
        
        try:
            # Métriques de performance des recommandations
            recommendation_performance = await self._collect_recommendation_performance(tenant_id)
            
            # Métriques de diversité et nouveauté
            diversity_metrics = await self._collect_diversity_metrics(tenant_id)
            
            # Métriques d'engagement utilisateur
            engagement_metrics = await self._collect_engagement_metrics(tenant_id)
            
            # Métriques de business impact
            business_impact = await self._collect_business_impact(tenant_id)
            
            return {
                'recommendation_system_metrics': {
                    'tenant_id': tenant_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'performance': recommendation_performance,
                    'diversity': diversity_metrics,
                    'engagement': engagement_metrics,
                    'business_impact': business_impact,
                    'overall_recommendation_score': self._calculate_recommendation_score(
                        recommendation_performance, diversity_metrics, engagement_metrics
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques recommandation: {str(e)}")
            raise
    
    async def _collect_recommendation_performance(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques de performance."""
        return {
            'precision_at_k': {
                'k_5': 0.34 + np.random.normal(0, 0.03),
                'k_10': 0.28 + np.random.normal(0, 0.03),
                'k_20': 0.22 + np.random.normal(0, 0.02)
            },
            'recall_at_k': {
                'k_5': 0.18 + np.random.normal(0, 0.02),
                'k_10': 0.29 + np.random.normal(0, 0.03),
                'k_20': 0.41 + np.random.normal(0, 0.04)
            },
            'ndcg_at_k': {
                'k_5': 0.42 + np.random.normal(0, 0.03),
                'k_10': 0.47 + np.random.normal(0, 0.03),
                'k_20': 0.52 + np.random.normal(0, 0.02)
            },
            'map_score': 0.38 + np.random.normal(0, 0.03),
            'coverage': 0.76 + np.random.normal(0, 0.04),
            'catalog_coverage': 0.68 + np.random.normal(0, 0.05)
        }
    
    async def _collect_diversity_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques de diversité."""
        return {
            'intra_list_diversity': 0.72 + np.random.normal(0, 0.05),
            'genre_diversity': 0.68 + np.random.normal(0, 0.04),
            'artist_diversity': 0.81 + np.random.normal(0, 0.03),
            'temporal_diversity': 0.59 + np.random.normal(0, 0.06),
            'novelty_score': 0.45 + np.random.normal(0, 0.04),
            'serendipity_score': 0.23 + np.random.normal(0, 0.03),
            'popularity_bias': 0.34 + np.random.normal(0, 0.05),
            'long_tail_coverage': 0.42 + np.random.normal(0, 0.04)
        }
    
    async def _collect_engagement_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques d'engagement."""
        return {
            'click_through_rate': 0.23 + np.random.normal(0, 0.02),
            'play_completion_rate': 0.67 + np.random.normal(0, 0.05),
            'like_rate': 0.15 + np.random.normal(0, 0.02),
            'share_rate': 0.04 + np.random.normal(0, 0.01),
            'skip_rate': 0.28 + np.random.normal(0, 0.03),
            'session_length_increase': 1.23 + np.random.normal(0, 0.1),
            'return_rate': 0.78 + np.random.normal(0, 0.04),
            'playlist_addition_rate': 0.12 + np.random.normal(0, 0.02)
        }
    
    async def _collect_business_impact(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques d'impact business."""
        return {
            'conversion_rate': 0.089 + np.random.normal(0, 0.01),
            'revenue_per_recommendation': 0.34 + np.random.normal(0, 0.05),
            'subscription_uplift': 0.12 + np.random.normal(0, 0.02),
            'churn_reduction': 0.08 + np.random.normal(0, 0.01),
            'user_lifetime_value_increase': 1.45 + np.random.normal(0, 0.2),
            'recommendation_attribution': 0.67 + np.random.normal(0, 0.04)
        }
    
    def _calculate_recommendation_score(self, performance: Dict, 
                                      diversity: Dict, engagement: Dict) -> float:
        """Calcule un score global du système de recommandation."""
        # Score de performance
        perf_score = (
            performance['precision_at_k']['k_10'] * 0.3 +
            performance['recall_at_k']['k_10'] * 0.3 +
            performance['ndcg_at_k']['k_10'] * 0.4
        )
        
        # Score de diversité
        div_score = (
            diversity['intra_list_diversity'] * 0.4 +
            diversity['novelty_score'] * 0.3 +
            diversity['serendipity_score'] * 0.3
        )
        
        # Score d'engagement
        eng_score = (
            engagement['click_through_rate'] * 0.3 +
            engagement['play_completion_rate'] * 0.3 +
            engagement['like_rate'] * 0.2 +
            engagement['return_rate'] * 0.2
        )
        
        # Score composite
        overall_score = (perf_score * 0.4 + div_score * 0.3 + eng_score * 0.3) * 100
        return round(overall_score, 2)
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Valide les données de recommandation."""
        try:
            rec_data = data.get('recommendation_system_metrics', {})
            
            required_sections = ['performance', 'diversity', 'engagement', 'business_impact']
            for section in required_sections:
                if section not in rec_data:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation données recommandation: {str(e)}")
            return False


class AudioAnalysisCollector(BaseCollector):
    """Collecteur d'analyse audio avancée."""
    
    async def collect(self) -> Dict[str, Any]:
        """Collecte les métriques d'analyse audio."""
        tenant_id = self.config.tags.get('tenant_id', 'default')
        
        try:
            # Métriques de qualité audio
            audio_quality = await self._collect_audio_quality_metrics(tenant_id)
            
            # Métriques de génération musicale
            generation_metrics = await self._collect_music_generation_metrics(tenant_id)
            
            # Métriques d'analyse de contenu
            content_analysis = await self._collect_content_analysis_metrics(tenant_id)
            
            # Métriques de similarité et classification
            similarity_metrics = await self._collect_similarity_metrics(tenant_id)
            
            return {
                'audio_analysis_metrics': {
                    'tenant_id': tenant_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'audio_quality': audio_quality,
                    'generation_metrics': generation_metrics,
                    'content_analysis': content_analysis,
                    'similarity_metrics': similarity_metrics,
                    'overall_audio_score': self._calculate_audio_score(
                        audio_quality, generation_metrics, content_analysis
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques audio: {str(e)}")
            raise
    
    async def _collect_audio_quality_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques de qualité audio."""
        return {
            'signal_to_noise_ratio': 42.5 + np.random.normal(0, 3),
            'total_harmonic_distortion': 0.02 + np.random.normal(0, 0.005),
            'dynamic_range': 18.7 + np.random.normal(0, 2),
            'frequency_response_flatness': 0.89 + np.random.normal(0, 0.03),
            'stereo_imaging': 0.76 + np.random.normal(0, 0.05),
            'loudness_consistency': 0.92 + np.random.normal(0, 0.02),
            'artifacts_detection': 0.03 + np.random.normal(0, 0.01),
            'mastering_quality': 0.84 + np.random.normal(0, 0.04)
        }
    
    async def _collect_music_generation_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques de génération musicale."""
        return {
            'melodic_coherence': 0.78 + np.random.normal(0, 0.05),
            'harmonic_progression_quality': 0.82 + np.random.normal(0, 0.04),
            'rhythmic_consistency': 0.91 + np.random.normal(0, 0.03),
            'structural_organization': 0.75 + np.random.normal(0, 0.06),
            'genre_authenticity': 0.87 + np.random.normal(0, 0.03),
            'creativity_index': 0.64 + np.random.normal(0, 0.08),
            'musical_complexity': 0.59 + np.random.normal(0, 0.07),
            'emotional_expression': 0.71 + np.random.normal(0, 0.05),
            'instrument_separation': 0.86 + np.random.normal(0, 0.03),
            'tempo_stability': 0.94 + np.random.normal(0, 0.02)
        }
    
    async def _collect_content_analysis_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques d'analyse de contenu."""
        return {
            'genre_classification_accuracy': 0.93 + np.random.normal(0, 0.02),
            'mood_detection_accuracy': 0.87 + np.random.normal(0, 0.03),
            'instrument_recognition_accuracy': 0.91 + np.random.normal(0, 0, 0.02),
            'key_detection_accuracy': 0.89 + np.random.normal(0, 0.03),
            'tempo_detection_accuracy': 0.95 + np.random.normal(0, 0.01),
            'vocal_detection_accuracy': 0.92 + np.random.normal(0, 0.02),
            'energy_level_accuracy': 0.85 + np.random.normal(0, 0.04),
            'danceability_prediction_accuracy': 0.79 + np.random.normal(0, 0.05)
        }
    
    async def _collect_similarity_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Collecte les métriques de similarité."""
        return {
            'audio_fingerprint_accuracy': 0.97 + np.random.normal(0, 0.01),
            'semantic_similarity_accuracy': 0.84 + np.random.normal(0, 0.03),
            'acoustic_similarity_accuracy': 0.89 + np.random.normal(0, 0.02),
            'musical_similarity_accuracy': 0.81 + np.random.normal(0, 0.04),
            'cover_song_detection_accuracy': 0.76 + np.random.normal(0, 0.05),
            'remix_detection_accuracy': 0.72 + np.random.normal(0, 0.06),
            'plagiarism_detection_accuracy': 0.94 + np.random.normal(0, 0.02)
        }
    
    def _calculate_audio_score(self, quality: Dict, generation: Dict, 
                             content: Dict) -> float:
        """Calcule un score global d'analyse audio."""
        # Score de qualité
        quality_score = (
            quality['signal_to_noise_ratio'] / 50.0 * 0.3 +
            (1 - quality['total_harmonic_distortion']) * 0.2 +
            quality['dynamic_range'] / 25.0 * 0.2 +
            quality['frequency_response_flatness'] * 0.3
        )
        
        # Score de génération
        generation_score = (
            generation['melodic_coherence'] * 0.25 +
            generation['harmonic_progression_quality'] * 0.25 +
            generation['rhythmic_consistency'] * 0.25 +
            generation['genre_authenticity'] * 0.25
        )
        
        # Score d'analyse de contenu
        content_score = (
            content['genre_classification_accuracy'] * 0.3 +
            content['mood_detection_accuracy'] * 0.25 +
            content['instrument_recognition_accuracy'] * 0.25 +
            content['tempo_detection_accuracy'] * 0.2
        )
        
        # Score composite
        overall_score = (quality_score * 0.4 + generation_score * 0.35 + content_score * 0.25) * 100
        return round(min(100, max(0, overall_score)), 2)
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Valide les données d'analyse audio."""
        try:
            audio_data = data.get('audio_analysis_metrics', {})
            
            required_sections = ['audio_quality', 'generation_metrics', 'content_analysis']
            for section in required_sections:
                if section not in audio_data:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation données audio: {str(e)}")
            return False


__all__ = [
    'MLModelPerformanceCollector',
    'ModelRegistry',
    'DataDriftDetector',
    'ModelPerformanceTracker',
    'MLAnomalyDetector',
    'AIAgentMetricsCollector',
    'RecommendationSystemCollector',
    'AudioAnalysisCollector',
    'ModelMetrics',
    'AudioGenerationMetrics',
    'ModelType',
    'ModelStatus'
]
