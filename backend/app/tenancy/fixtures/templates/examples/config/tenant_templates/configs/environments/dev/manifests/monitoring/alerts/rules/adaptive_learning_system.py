"""
Système de Configuration Dynamique et Auto-Apprentissage

Ce module fournit un système avancé de configuration automatique qui :
- Apprend des patterns de comportement des applications
- S'adapte automatiquement aux changements d'environnement
- Optimise les seuils d'alerte avec l'intelligence artificielle
- Fournit des recommandations contextuelles intelligentes
- Intègre la détection de drift et l'adaptation continue

Équipe Engineering:
✅ Lead Dev + Architecte IA : Fahed Mlaiel
✅ Data Scientist (MLOps/AutoML)
✅ Platform Engineer (SRE/Observability)
✅ Product Manager (UX/Analytics)

Copyright: © 2025 Spotify Technology S.A.
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import yaml
import pickle
import hashlib

# Machine Learning et AutoML
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
import optuna
import mlflow
import mlflow.sklearn

# Monitoring et métriques
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Configuration et persistance
import redis
import asyncpg
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, DateTime, JSON
from sqlalchemy.orm import sessionmaker

logger = structlog.get_logger(__name__)

# Métriques spécialisées
CONFIG_ADAPTATIONS = Counter('config_adaptations_total', 'Configuration adaptations', ['tenant_id', 'adaptation_type'])
LEARNING_ACCURACY = Gauge('learning_accuracy', 'Learning model accuracy', ['tenant_id', 'model_type'])
DRIFT_DETECTION_ALERTS = Counter('drift_detection_alerts_total', 'Drift detection alerts', ['tenant_id', 'drift_type'])
AUTO_TUNING_IMPROVEMENTS = Gauge('auto_tuning_improvements', 'Auto-tuning improvements', ['tenant_id', 'metric'])

@dataclass
class AdaptiveLearningConfig:
    """Configuration pour l'apprentissage adaptatif"""
    learning_rate: float = 0.01
    adaptation_threshold: float = 0.8
    min_samples_for_adaptation: int = 100
    drift_detection_window: int = 50
    confidence_threshold: float = 0.9
    auto_tune_enabled: bool = True
    feedback_weight: float = 0.3
    exploration_rate: float = 0.1

@dataclass
class BehaviorPattern:
    """Pattern de comportement détecté"""
    pattern_id: str
    pattern_type: str
    frequency: float
    confidence: float
    metrics_involved: List[str]
    conditions: Dict[str, Any]
    temporal_context: Dict[str, Any]
    business_impact: float
    recommendation: str

@dataclass
class ConfigurationRecommendation:
    """Recommandation de configuration"""
    config_path: str
    current_value: Any
    recommended_value: Any
    confidence: float
    reasoning: str
    expected_improvement: float
    risk_level: str
    validation_tests: List[str]

class BehaviorLearningEngine:
    """Moteur d'apprentissage des comportements"""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.behavior_models: Dict[str, Any] = {}
        self.pattern_database: Dict[str, List[BehaviorPattern]] = {}
        self.learning_history: List[Dict] = []
        self.feature_importance: Dict[str, float] = {}
        
    async def learn_behavior_patterns(self, tenant_id: str, 
                                    historical_data: pd.DataFrame) -> List[BehaviorPattern]:
        """Apprend les patterns de comportement à partir des données historiques"""
        try:
            start_time = time.time()
            
            if len(historical_data) < self.config.min_samples_for_adaptation:
                logger.warning("Insufficient data for behavior learning", 
                             tenant_id=tenant_id, samples=len(historical_data))
                return []
            
            # Préparation des features
            features = self._extract_features(historical_data)
            
            # Détection de patterns temporels
            temporal_patterns = await self._detect_temporal_patterns(features)
            
            # Clustering pour identifier des comportements similaires
            behavioral_clusters = await self._perform_behavioral_clustering(features)
            
            # Analyse des corrélations
            correlation_patterns = await self._analyze_correlations(features)
            
            # Détection d'anomalies récurrentes
            anomaly_patterns = await self._detect_anomaly_patterns(features)
            
            # Combinaison des patterns
            all_patterns = []
            all_patterns.extend(temporal_patterns)
            all_patterns.extend(behavioral_clusters)
            all_patterns.extend(correlation_patterns)
            all_patterns.extend(anomaly_patterns)
            
            # Scoring et filtrage des patterns
            validated_patterns = await self._validate_and_score_patterns(all_patterns, features)
            
            # Mise à jour de la base de patterns
            self.pattern_database[tenant_id] = validated_patterns
            
            # Entraînement du modèle prédictif
            await self._train_prediction_model(tenant_id, features, validated_patterns)
            
            learning_time = time.time() - start_time
            
            logger.info(
                "Behavior patterns learned successfully",
                tenant_id=tenant_id,
                patterns_discovered=len(validated_patterns),
                learning_time=learning_time
            )
            
            return validated_patterns
            
        except Exception as e:
            logger.error("Behavior learning failed", error=str(e), tenant_id=tenant_id)
            raise
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extrait les features pour l'apprentissage"""
        features = data.copy()
        
        # Features temporelles
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['month'] = features['timestamp'].dt.month
            features['is_weekend'] = features['day_of_week'].isin([5, 6])
            features['is_business_hours'] = features['hour'].between(9, 17)
        
        # Features de tendance
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['hour', 'day_of_week', 'month']:
                # Moyennes mobiles
                features[f'{col}_ma_5'] = features[col].rolling(window=5).mean()
                features[f'{col}_ma_15'] = features[col].rolling(window=15).mean()
                
                # Tendances
                features[f'{col}_trend'] = features[col].diff()
                features[f'{col}_trend_pct'] = features[col].pct_change()
                
                # Volatilité
                features[f'{col}_volatility'] = features[col].rolling(window=10).std()
        
        # Features d'interaction
        numeric_cols = [col for col in numeric_columns if not col.endswith(('_ma_5', '_ma_15', '_trend', '_trend_pct', '_volatility'))]
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                features[f'{col1}_x_{col2}'] = features[col1] * features[col2]
                if features[col2].std() > 0:
                    features[f'{col1}_ratio_{col2}'] = features[col1] / features[col2]
        
        return features.dropna()
    
    async def _detect_temporal_patterns(self, features: pd.DataFrame) -> List[BehaviorPattern]:
        """Détecte les patterns temporels"""
        patterns = []
        
        try:
            if 'hour' in features.columns:
                # Analyse des patterns horaires
                hourly_stats = features.groupby('hour').agg({
                    col: ['mean', 'std'] for col in features.select_dtypes(include=[np.number]).columns
                    if col not in ['hour', 'day_of_week', 'month']
                })
                
                # Détection de pics horaires
                for metric in hourly_stats.columns.levels[0]:
                    if metric in ['hour', 'day_of_week', 'month']:
                        continue
                        
                    hourly_means = hourly_stats[(metric, 'mean')]
                    peak_hours = hourly_means.nlargest(3).index.tolist()
                    
                    if len(peak_hours) > 0:
                        pattern = BehaviorPattern(
                            pattern_id=f"hourly_peak_{metric}_{hash(str(peak_hours)) % 10000}",
                            pattern_type="temporal_hourly",
                            frequency=len(peak_hours) / 24.0,
                            confidence=0.8,
                            metrics_involved=[metric],
                            conditions={'peak_hours': peak_hours},
                            temporal_context={'type': 'hourly', 'hours': peak_hours},
                            business_impact=0.5,
                            recommendation=f"Ajuster les seuils de {metric} pendant les heures de pic: {peak_hours}"
                        )
                        patterns.append(pattern)
            
            if 'day_of_week' in features.columns:
                # Analyse des patterns hebdomadaires
                weekly_stats = features.groupby('day_of_week').agg({
                    col: ['mean', 'std'] for col in features.select_dtypes(include=[np.number]).columns
                    if col not in ['hour', 'day_of_week', 'month']
                })
                
                for metric in weekly_stats.columns.levels[0]:
                    if metric in ['hour', 'day_of_week', 'month']:
                        continue
                        
                    daily_means = weekly_stats[(metric, 'mean')]
                    weekend_avg = daily_means[[5, 6]].mean()
                    weekday_avg = daily_means[[0, 1, 2, 3, 4]].mean()
                    
                    if abs(weekend_avg - weekday_avg) > weekday_avg * 0.2:
                        pattern = BehaviorPattern(
                            pattern_id=f"weekly_pattern_{metric}_{hash(str([weekend_avg, weekday_avg])) % 10000}",
                            pattern_type="temporal_weekly",
                            frequency=1.0,  # Pattern récurrent chaque semaine
                            confidence=0.85,
                            metrics_involved=[metric],
                            conditions={'weekend_avg': weekend_avg, 'weekday_avg': weekday_avg},
                            temporal_context={'type': 'weekly', 'weekend_factor': weekend_avg / weekday_avg},
                            business_impact=0.7,
                            recommendation=f"Configurer des seuils différentiés pour {metric} entre semaine et weekend"
                        )
                        patterns.append(pattern)
            
        except Exception as e:
            logger.error("Temporal pattern detection failed", error=str(e))
        
        return patterns
    
    async def _perform_behavioral_clustering(self, features: pd.DataFrame) -> List[BehaviorPattern]:
        """Effectue un clustering comportemental"""
        patterns = []
        
        try:
            # Sélection des features numériques pour le clustering
            numeric_features = features.select_dtypes(include=[np.number])
            
            if len(numeric_features.columns) < 2:
                return patterns
            
            # Normalisation
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(numeric_features.fillna(0))
            
            # Détermination du nombre optimal de clusters
            silhouette_scores = []
            k_range = range(2, min(10, len(scaled_features) // 10))
            
            for k in k_range:
                if len(scaled_features) > k:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    cluster_labels = kmeans.fit_predict(scaled_features)
                    score = silhouette_score(scaled_features, cluster_labels)
                    silhouette_scores.append((k, score))
            
            if silhouette_scores:
                optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
                
                # Clustering final
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                # Analyse des clusters
                for cluster_id in range(optimal_k):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_data = numeric_features[cluster_mask]
                    
                    if len(cluster_data) > 5:  # Cluster significatif
                        # Caractéristiques du cluster
                        cluster_profile = cluster_data.mean()
                        cluster_size = len(cluster_data)
                        
                        # Identification des métriques distinctives
                        distinctive_metrics = []
                        for metric in cluster_profile.index:
                            global_mean = numeric_features[metric].mean()
                            cluster_mean = cluster_profile[metric]
                            if abs(cluster_mean - global_mean) > global_mean * 0.3:
                                distinctive_metrics.append(metric)
                        
                        if distinctive_metrics:
                            pattern = BehaviorPattern(
                                pattern_id=f"behavioral_cluster_{cluster_id}_{hash(str(distinctive_metrics)) % 10000}",
                                pattern_type="behavioral_cluster",
                                frequency=cluster_size / len(numeric_features),
                                confidence=0.75,
                                metrics_involved=distinctive_metrics,
                                conditions={'cluster_profile': cluster_profile.to_dict()},
                                temporal_context={'type': 'cluster', 'cluster_id': cluster_id},
                                business_impact=0.6,
                                recommendation=f"Optimiser les règles pour le comportement cluster {cluster_id} avec métriques: {distinctive_metrics}"
                            )
                            patterns.append(pattern)
            
        except Exception as e:
            logger.error("Behavioral clustering failed", error=str(e))
        
        return patterns
    
    async def _analyze_correlations(self, features: pd.DataFrame) -> List[BehaviorPattern]:
        """Analyse les corrélations entre métriques"""
        patterns = []
        
        try:
            numeric_features = features.select_dtypes(include=[np.number])
            correlation_matrix = numeric_features.corr()
            
            # Recherche de corrélations fortes
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Corrélation forte
                        metric1 = correlation_matrix.columns[i]
                        metric2 = correlation_matrix.columns[j]
                        strong_correlations.append((metric1, metric2, corr_value))
            
            # Création de patterns de corrélation
            for metric1, metric2, corr_value in strong_correlations:
                pattern = BehaviorPattern(
                    pattern_id=f"correlation_{metric1}_{metric2}_{hash(f'{metric1}{metric2}') % 10000}",
                    pattern_type="correlation",
                    frequency=1.0,  # Corrélation constante
                    confidence=min(0.95, abs(corr_value)),
                    metrics_involved=[metric1, metric2],
                    conditions={'correlation': corr_value, 'type': 'positive' if corr_value > 0 else 'negative'},
                    temporal_context={'type': 'correlation', 'strength': abs(corr_value)},
                    business_impact=0.8 if abs(corr_value) > 0.9 else 0.6,
                    recommendation=f"Configurer des règles composites pour {metric1} et {metric2} (corrélation: {corr_value:.2f})"
                )
                patterns.append(pattern)
            
        except Exception as e:
            logger.error("Correlation analysis failed", error=str(e))
        
        return patterns
    
    async def _detect_anomaly_patterns(self, features: pd.DataFrame) -> List[BehaviorPattern]:
        """Détecte les patterns d'anomalies récurrentes"""
        patterns = []
        
        try:
            numeric_features = features.select_dtypes(include=[np.number])
            
            for column in numeric_features.columns:
                if column in ['hour', 'day_of_week', 'month']:
                    continue
                    
                data = numeric_features[column].dropna()
                if len(data) < 10:
                    continue
                
                # Détection d'anomalies avec IQR
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies = data[(data < lower_bound) | (data > upper_bound)]
                
                if len(anomalies) > len(data) * 0.05:  # Plus de 5% d'anomalies
                    # Analyse temporelle des anomalies
                    if 'hour' in features.columns:
                        anomaly_indices = anomalies.index
                        anomaly_hours = features.loc[anomaly_indices, 'hour'].value_counts()
                        
                        if len(anomaly_hours) > 0:
                            most_common_hour = anomaly_hours.index[0]
                            pattern = BehaviorPattern(
                                pattern_id=f"anomaly_pattern_{column}_{hash(column) % 10000}",
                                pattern_type="anomaly_temporal",
                                frequency=len(anomalies) / len(data),
                                confidence=0.7,
                                metrics_involved=[column],
                                conditions={
                                    'anomaly_threshold_upper': upper_bound,
                                    'anomaly_threshold_lower': lower_bound,
                                    'common_anomaly_hour': most_common_hour
                                },
                                temporal_context={'type': 'anomaly', 'peak_hour': most_common_hour},
                                business_impact=0.9,
                                recommendation=f"Ajuster les seuils de {column} et surveiller particulièrement à {most_common_hour}h"
                            )
                            patterns.append(pattern)
            
        except Exception as e:
            logger.error("Anomaly pattern detection failed", error=str(e))
        
        return patterns
    
    async def _validate_and_score_patterns(self, patterns: List[BehaviorPattern], 
                                         features: pd.DataFrame) -> List[BehaviorPattern]:
        """Valide et score les patterns détectés"""
        validated_patterns = []
        
        for pattern in patterns:
            try:
                # Validation statistique
                if pattern.confidence > 0.6 and pattern.frequency > 0.05:
                    
                    # Score composite basé sur plusieurs facteurs
                    composite_score = (
                        pattern.confidence * 0.4 +
                        pattern.business_impact * 0.3 +
                        min(pattern.frequency * 2, 1.0) * 0.2 +
                        (len(pattern.metrics_involved) / 10.0) * 0.1
                    )
                    
                    pattern.confidence = composite_score
                    
                    if composite_score > 0.5:
                        validated_patterns.append(pattern)
            
            except Exception as e:
                logger.warning("Pattern validation failed", pattern=pattern.pattern_id, error=str(e))
        
        # Tri par score de confiance
        validated_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return validated_patterns[:20]  # Limite aux 20 meilleurs patterns
    
    async def _train_prediction_model(self, tenant_id: str, features: pd.DataFrame, 
                                    patterns: List[BehaviorPattern]):
        """Entraîne un modèle prédictif basé sur les patterns"""
        try:
            if len(features) < 50:
                return
            
            # Préparation des données d'entraînement
            numeric_features = features.select_dtypes(include=[np.number])
            
            # Création de labels basés sur les patterns
            labels = np.zeros(len(numeric_features))
            
            for i, pattern in enumerate(patterns[:5]):  # Top 5 patterns
                # Création d'un label binaire pour chaque pattern
                pattern_label = np.zeros(len(numeric_features))
                
                if pattern.pattern_type == "temporal_hourly" and 'hour' in features.columns:
                    peak_hours = pattern.conditions.get('peak_hours', [])
                    pattern_label[features['hour'].isin(peak_hours)] = 1
                elif pattern.pattern_type == "correlation":
                    metrics = pattern.metrics_involved
                    if len(metrics) >= 2 and all(m in numeric_features.columns for m in metrics[:2]):
                        corr_condition = np.corrcoef(
                            numeric_features[metrics[0]], 
                            numeric_features[metrics[1]]
                        )[0, 1]
                        if abs(corr_condition) > 0.7:
                            pattern_label[:] = 1
                
                labels += pattern_label * pattern.confidence
            
            # Entraînement du modèle
            if len(np.unique(labels)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    numeric_features.fillna(0), labels, test_size=0.2, random_state=42
                )
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Évaluation
                score = model.score(X_test, y_test)
                self.behavior_models[tenant_id] = model
                
                # Feature importance
                feature_importance = dict(zip(numeric_features.columns, model.feature_importances_))
                self.feature_importance[tenant_id] = feature_importance
                
                LEARNING_ACCURACY.labels(model_type='behavior_prediction', tenant_id=tenant_id).set(score)
                
                logger.info("Prediction model trained", tenant_id=tenant_id, accuracy=score)
        
        except Exception as e:
            logger.error("Prediction model training failed", error=str(e))

class DynamicConfigurationEngine:
    """Moteur de configuration dynamique"""
    
    def __init__(self, learning_engine: BehaviorLearningEngine):
        self.learning_engine = learning_engine
        self.configuration_templates: Dict[str, Dict] = {}
        self.adaptation_history: List[Dict] = []
        self.active_configurations: Dict[str, Dict] = {}
        
    async def generate_adaptive_configuration(self, tenant_id: str, 
                                            current_metrics: Dict[str, Any],
                                            business_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Génère une configuration adaptative basée sur l'apprentissage"""
        try:
            start_time = time.time()
            
            # Récupération des patterns appris
            patterns = self.learning_engine.pattern_database.get(tenant_id, [])
            
            if not patterns:
                logger.warning("No patterns available for adaptive configuration", tenant_id=tenant_id)
                return await self._generate_default_configuration(tenant_id, current_metrics)
            
            # Génération des recommandations
            recommendations = await self._generate_configuration_recommendations(
                tenant_id, patterns, current_metrics, business_context
            )
            
            # Application des optimisations AutoML
            optimized_config = await self._apply_automl_optimization(
                tenant_id, recommendations, current_metrics
            )
            
            # Validation et tests de sécurité
            validated_config = await self._validate_configuration(optimized_config)
            
            # Mise à jour de l'historique
            adaptation_record = {
                'tenant_id': tenant_id,
                'timestamp': datetime.utcnow().isoformat(),
                'patterns_used': [p.pattern_id for p in patterns],
                'recommendations_count': len(recommendations),
                'validation_passed': validated_config.get('validation_status') == 'passed',
                'generation_time': time.time() - start_time
            }
            
            self.adaptation_history.append(adaptation_record)
            self.active_configurations[tenant_id] = validated_config
            
            CONFIG_ADAPTATIONS.labels(
                tenant_id=tenant_id, 
                adaptation_type='full_generation'
            ).inc()
            
            logger.info(
                "Adaptive configuration generated successfully",
                tenant_id=tenant_id,
                patterns_used=len(patterns),
                recommendations=len(recommendations),
                generation_time=adaptation_record['generation_time']
            )
            
            return validated_config
            
        except Exception as e:
            logger.error("Adaptive configuration generation failed", error=str(e), tenant_id=tenant_id)
            return await self._generate_fallback_configuration(tenant_id)
    
    async def _generate_configuration_recommendations(self, tenant_id: str, 
                                                    patterns: List[BehaviorPattern],
                                                    current_metrics: Dict[str, Any],
                                                    business_context: Optional[Dict]) -> List[ConfigurationRecommendation]:
        """Génère des recommandations de configuration basées sur les patterns"""
        recommendations = []
        
        for pattern in patterns:
            try:
                if pattern.pattern_type == "temporal_hourly":
                    # Recommandations pour les patterns horaires
                    peak_hours = pattern.conditions.get('peak_hours', [])
                    current_hour = datetime.utcnow().hour
                    
                    if current_hour in peak_hours:
                        for metric in pattern.metrics_involved:
                            current_value = current_metrics.get(metric, 0)
                            recommended_threshold = current_value * 1.2  # 20% plus élevé pendant les pics
                            
                            recommendation = ConfigurationRecommendation(
                                config_path=f"alerts.{metric}.threshold",
                                current_value=current_value,
                                recommended_value=recommended_threshold,
                                confidence=pattern.confidence,
                                reasoning=f"Ajustement pour pic horaire détecté à {current_hour}h",
                                expected_improvement=0.3,
                                risk_level="low",
                                validation_tests=[f"validate_threshold_{metric}"]
                            )
                            recommendations.append(recommendation)
                
                elif pattern.pattern_type == "correlation":
                    # Recommandations pour les patterns de corrélation
                    if len(pattern.metrics_involved) >= 2:
                        metric1, metric2 = pattern.metrics_involved[:2]
                        correlation = pattern.conditions.get('correlation', 0)
                        
                        recommendation = ConfigurationRecommendation(
                            config_path=f"alerts.composite_rules.{metric1}_{metric2}",
                            current_value=None,
                            recommended_value={
                                'type': 'composite',
                                'metrics': [metric1, metric2],
                                'correlation_threshold': abs(correlation) * 0.8,
                                'operator': 'and' if correlation > 0 else 'or'
                            },
                            confidence=pattern.confidence,
                            reasoning=f"Règle composite basée sur corrélation {correlation:.2f}",
                            expected_improvement=0.4,
                            risk_level="medium",
                            validation_tests=[f"validate_composite_{metric1}_{metric2}"]
                        )
                        recommendations.append(recommendation)
                
                elif pattern.pattern_type == "behavioral_cluster":
                    # Recommandations pour les clusters comportementaux
                    cluster_profile = pattern.conditions.get('cluster_profile', {})
                    
                    for metric, cluster_value in cluster_profile.items():
                        if metric in current_metrics:
                            current_value = current_metrics[metric]
                            # Ajustement basé sur le profil du cluster
                            adjustment_factor = cluster_value / current_value if current_value != 0 else 1.0
                            recommended_value = current_value * max(0.5, min(2.0, adjustment_factor))
                            
                            recommendation = ConfigurationRecommendation(
                                config_path=f"alerts.{metric}.cluster_threshold",
                                current_value=current_value,
                                recommended_value=recommended_value,
                                confidence=pattern.confidence,
                                reasoning=f"Ajustement basé sur cluster comportemental",
                                expected_improvement=0.25,
                                risk_level="low",
                                validation_tests=[f"validate_cluster_threshold_{metric}"]
                            )
                            recommendations.append(recommendation)
                
                elif pattern.pattern_type == "anomaly_temporal":
                    # Recommandations pour les anomalies temporelles
                    anomaly_hour = pattern.conditions.get('common_anomaly_hour')
                    current_hour = datetime.utcnow().hour
                    
                    if abs(current_hour - anomaly_hour) <= 1:  # Une heure avant/après
                        for metric in pattern.metrics_involved:
                            upper_bound = pattern.conditions.get('anomaly_threshold_upper')
                            lower_bound = pattern.conditions.get('anomaly_threshold_lower')
                            
                            if upper_bound and lower_bound:
                                recommendation = ConfigurationRecommendation(
                                    config_path=f"alerts.{metric}.anomaly_bounds",
                                    current_value=current_metrics.get(metric),
                                    recommended_value={
                                        'upper_threshold': upper_bound,
                                        'lower_threshold': lower_bound,
                                        'sensitivity': 'high'
                                    },
                                    confidence=pattern.confidence,
                                    reasoning=f"Surveillance renforcée pendant période d'anomalies ({anomaly_hour}h)",
                                    expected_improvement=0.5,
                                    risk_level="high",
                                    validation_tests=[f"validate_anomaly_bounds_{metric}"]
                                )
                                recommendations.append(recommendation)
            
            except Exception as e:
                logger.warning("Failed to generate recommendation for pattern", 
                             pattern_id=pattern.pattern_id, error=str(e))
        
        return recommendations
    
    async def _apply_automl_optimization(self, tenant_id: str, 
                                       recommendations: List[ConfigurationRecommendation],
                                       current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Applique l'optimisation AutoML aux recommandations"""
        
        def objective(trial):
            """Fonction objectif pour l'optimisation Optuna"""
            score = 0.0
            
            for i, rec in enumerate(recommendations):
                if isinstance(rec.recommended_value, (int, float)):
                    # Optimisation des valeurs numériques
                    optimized_value = trial.suggest_float(
                        f"param_{i}",
                        rec.recommended_value * 0.5,
                        rec.recommended_value * 1.5
                    )
                    
                    # Score basé sur la distance à la valeur recommandée et la confiance
                    distance = abs(optimized_value - rec.recommended_value) / rec.recommended_value
                    score += rec.confidence * (1.0 - distance) * rec.expected_improvement
            
            return score
        
        try:
            # Optimisation avec Optuna
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50, timeout=30)
            
            # Application des valeurs optimisées
            optimized_config = {
                'tenant_id': tenant_id,
                'optimization_method': 'automl_optuna',
                'best_score': study.best_value,
                'configuration': {}
            }
            
            for i, rec in enumerate(recommendations):
                param_name = f"param_{i}"
                if param_name in study.best_params:
                    optimized_value = study.best_params[param_name]
                    optimized_config['configuration'][rec.config_path] = optimized_value
                else:
                    optimized_config['configuration'][rec.config_path] = rec.recommended_value
            
            logger.info("AutoML optimization completed", 
                       tenant_id=tenant_id, best_score=study.best_value)
            
            return optimized_config
            
        except Exception as e:
            logger.error("AutoML optimization failed", error=str(e))
            # Fallback: utiliser les recommandations originales
            return {
                'tenant_id': tenant_id,
                'optimization_method': 'fallback',
                'configuration': {rec.config_path: rec.recommended_value for rec in recommendations}
            }
    
    async def _validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Valide la configuration générée"""
        validation_results = {
            'validation_status': 'passed',
            'validation_checks': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            configuration = config.get('configuration', {})
            
            # Validation des valeurs numériques
            for path, value in configuration.items():
                if isinstance(value, (int, float)):
                    if value < 0:
                        validation_results['errors'].append(f"Negative value not allowed for {path}: {value}")
                    elif value > 1000000:  # Limite arbitraire
                        validation_results['warnings'].append(f"Very high value for {path}: {value}")
                
                validation_results['validation_checks'].append({
                    'path': path,
                    'value': value,
                    'status': 'passed'
                })
            
            # Validation de la cohérence
            threshold_configs = {k: v for k, v in configuration.items() if 'threshold' in k}
            if len(threshold_configs) > 1:
                values = [v for v in threshold_configs.values() if isinstance(v, (int, float))]
                if values and max(values) / min(values) > 10:
                    validation_results['warnings'].append("Large variance in threshold values detected")
            
            # Statut final
            if validation_results['errors']:
                validation_results['validation_status'] = 'failed'
            elif validation_results['warnings']:
                validation_results['validation_status'] = 'passed_with_warnings'
            
            config['validation'] = validation_results
            
        except Exception as e:
            logger.error("Configuration validation failed", error=str(e))
            config['validation'] = {
                'validation_status': 'error',
                'error': str(e)
            }
        
        return config
    
    async def _generate_default_configuration(self, tenant_id: str, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une configuration par défaut"""
        return {
            'tenant_id': tenant_id,
            'configuration_type': 'default',
            'configuration': {
                'alerts.cpu.threshold': 80.0,
                'alerts.memory.threshold': 85.0,
                'alerts.disk.threshold': 90.0,
                'alerts.error_rate.threshold': 0.05,
                'alerts.response_time.threshold': 1000
            },
            'validation': {'validation_status': 'passed'},
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'method': 'default_fallback'
            }
        }
    
    async def _generate_fallback_configuration(self, tenant_id: str) -> Dict[str, Any]:
        """Génère une configuration de fallback en cas d'erreur"""
        return {
            'tenant_id': tenant_id,
            'configuration_type': 'fallback',
            'configuration': {
                'alerts.enabled': True,
                'alerts.default_threshold': 75.0,
                'alerts.notification_enabled': True
            },
            'validation': {'validation_status': 'fallback'},
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'method': 'error_fallback'
            }
        }

class DriftDetectionSystem:
    """Système de détection de drift"""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.baseline_distributions: Dict[str, Any] = {}
        self.drift_history: List[Dict] = []
        
    async def detect_data_drift(self, tenant_id: str, 
                              current_data: pd.DataFrame,
                              baseline_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Détecte le drift dans les données"""
        try:
            drift_results = {
                'tenant_id': tenant_id,
                'drift_detected': False,
                'drift_metrics': {},
                'affected_features': [],
                'drift_severity': 'none',
                'recommendations': []
            }
            
            if baseline_data is None:
                # Utiliser les données de référence stockées
                if tenant_id not in self.baseline_distributions:
                    # Première fois - stocker comme référence
                    self.baseline_distributions[tenant_id] = self._compute_statistical_baseline(current_data)
                    return drift_results
                
                baseline_stats = self.baseline_distributions[tenant_id]
            else:
                baseline_stats = self._compute_statistical_baseline(baseline_data)
            
            current_stats = self._compute_statistical_baseline(current_data)
            
            # Détection de drift pour chaque feature
            numeric_features = current_data.select_dtypes(include=[np.number]).columns
            
            for feature in numeric_features:
                if feature in baseline_stats and feature in current_stats:
                    drift_score = self._calculate_drift_score(
                        baseline_stats[feature], 
                        current_stats[feature]
                    )
                    
                    drift_results['drift_metrics'][feature] = drift_score
                    
                    if drift_score > 0.3:  # Seuil de drift
                        drift_results['affected_features'].append(feature)
                        drift_results['drift_detected'] = True
            
            # Évaluation de la sévérité
            if drift_results['drift_detected']:
                max_drift = max(drift_results['drift_metrics'].values())
                if max_drift > 0.7:
                    drift_results['drift_severity'] = 'high'
                elif max_drift > 0.5:
                    drift_results['drift_severity'] = 'medium'
                else:
                    drift_results['drift_severity'] = 'low'
                
                # Génération de recommandations
                drift_results['recommendations'] = self._generate_drift_recommendations(
                    drift_results['affected_features'], drift_results['drift_severity']
                )
                
                DRIFT_DETECTION_ALERTS.labels(
                    tenant_id=tenant_id,
                    drift_type=drift_results['drift_severity']
                ).inc()
                
                logger.warning("Data drift detected", 
                             tenant_id=tenant_id, 
                             severity=drift_results['drift_severity'],
                             affected_features=drift_results['affected_features'])
            
            # Mise à jour de l'historique
            self.drift_history.append({
                'tenant_id': tenant_id,
                'timestamp': datetime.utcnow().isoformat(),
                'drift_detected': drift_results['drift_detected'],
                'severity': drift_results['drift_severity'],
                'affected_features_count': len(drift_results['affected_features'])
            })
            
            return drift_results
            
        except Exception as e:
            logger.error("Drift detection failed", error=str(e), tenant_id=tenant_id)
            return {'error': str(e), 'drift_detected': False}
    
    def _compute_statistical_baseline(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Calcule les statistiques de référence"""
        baseline = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            series = data[column].dropna()
            if len(series) > 0:
                baseline[column] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'median': float(series.median()),
                    'q25': float(series.quantile(0.25)),
                    'q75': float(series.quantile(0.75)),
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis())
                }
        
        return baseline
    
    def _calculate_drift_score(self, baseline_stats: Dict, current_stats: Dict) -> float:
        """Calcule le score de drift entre deux distributions"""
        try:
            # Comparaison des moments statistiques
            mean_diff = abs(baseline_stats['mean'] - current_stats['mean'])
            std_diff = abs(baseline_stats['std'] - current_stats['std'])
            
            # Normalisation par les valeurs de référence
            baseline_mean = abs(baseline_stats['mean']) + 1e-6
            baseline_std = abs(baseline_stats['std']) + 1e-6
            
            normalized_mean_diff = mean_diff / baseline_mean
            normalized_std_diff = std_diff / baseline_std
            
            # Score composite
            drift_score = (normalized_mean_diff + normalized_std_diff) / 2
            
            # Bonus pour les changements de forme de distribution
            skew_diff = abs(baseline_stats.get('skewness', 0) - current_stats.get('skewness', 0))
            kurtosis_diff = abs(baseline_stats.get('kurtosis', 0) - current_stats.get('kurtosis', 0))
            
            shape_penalty = (skew_diff + kurtosis_diff) / 10.0
            
            return min(1.0, drift_score + shape_penalty)
            
        except Exception as e:
            logger.error("Drift score calculation failed", error=str(e))
            return 0.0
    
    def _generate_drift_recommendations(self, affected_features: List[str], severity: str) -> List[str]:
        """Génère des recommandations basées sur le drift détecté"""
        recommendations = []
        
        if severity == 'high':
            recommendations.append("🚨 Drift critique détecté - Recalibrage immédiat recommandé")
            recommendations.append("📊 Réentraîner tous les modèles ML avec les nouvelles données")
            recommendations.append("⚙️ Réviser les seuils d'alerte pour toutes les métriques affectées")
        elif severity == 'medium':
            recommendations.append("⚠️ Drift modéré détecté - Surveillance renforcée recommandée")
            recommendations.append("🔄 Planifier le réentraînement des modèles dans les 24h")
            recommendations.append("🎯 Ajuster les seuils pour les métriques les plus affectées")
        else:
            recommendations.append("ℹ️ Drift léger détecté - Surveillance continue")
            recommendations.append("📈 Vérifier les tendances à long terme")
        
        for feature in affected_features:
            recommendations.append(f"🔍 Analyser en détail la métrique: {feature}")
        
        recommendations.append("📝 Documenter les changements dans le contexte business")
        
        return recommendations

# Fonction principale d'initialisation
async def create_adaptive_learning_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """Crée un système d'apprentissage adaptatif complet"""
    
    # Configuration d'apprentissage
    learning_config = AdaptiveLearningConfig(
        learning_rate=config.get('learning_rate', 0.01),
        adaptation_threshold=config.get('adaptation_threshold', 0.8),
        min_samples_for_adaptation=config.get('min_samples', 100),
        drift_detection_window=config.get('drift_window', 50),
        auto_tune_enabled=config.get('auto_tune', True)
    )
    
    # Initialisation des composants
    behavior_learning_engine = BehaviorLearningEngine(learning_config)
    dynamic_config_engine = DynamicConfigurationEngine(behavior_learning_engine)
    drift_detection_system = DriftDetectionSystem(learning_config)
    
    logger.info("Adaptive learning system initialized successfully")
    
    return {
        'behavior_learning_engine': behavior_learning_engine,
        'dynamic_config_engine': dynamic_config_engine,
        'drift_detection_system': drift_detection_system,
        'config': learning_config,
        'capabilities': [
            'behavior_pattern_learning',
            'dynamic_configuration_generation',
            'automl_optimization',
            'data_drift_detection',
            'adaptive_threshold_tuning',
            'context_aware_recommendations'
        ]
    }

# Exemple d'utilisation complète
async def demonstrate_adaptive_learning():
    """Démontre le système d'apprentissage adaptatif"""
    
    # Configuration
    config = {
        'learning_rate': 0.01,
        'adaptation_threshold': 0.8,
        'min_samples': 50,  # Réduit pour la démo
        'drift_window': 30,
        'auto_tune': True
    }
    
    # Initialisation du système
    system = await create_adaptive_learning_system(config)
    
    # 1. Génération de données d'exemple
    print("📊 Génération de données d'entraînement...")
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    
    # Simulation de métriques avec patterns réalistes
    np.random.seed(42)
    data = []
    
    for i, timestamp in enumerate(dates):
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        
        # Patterns horaires (pic aux heures de bureau)
        cpu_base = 30 + 40 * (1 if 9 <= hour <= 17 else 0.3)
        cpu_noise = np.random.normal(0, 10)
        cpu_usage = max(0, min(100, cpu_base + cpu_noise))
        
        # Pattern hebdomadaire (moins d'activité le weekend)
        weekend_factor = 0.5 if day_of_week in [5, 6] else 1.0
        memory_usage = max(0, min(100, cpu_usage * 0.8 * weekend_factor + np.random.normal(0, 5)))
        
        # Corrélation avec erreurs
        error_rate = max(0, (cpu_usage - 50) / 1000 + np.random.exponential(0.001))
        
        data.append({
            'timestamp': timestamp,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'error_rate': error_rate,
            'response_time': max(50, 200 + (cpu_usage - 50) * 5 + np.random.normal(0, 20)),
            'disk_usage': max(0, min(100, 40 + np.random.normal(0, 5)))
        })
    
    historical_data = pd.DataFrame(data)
    print(f"Données générées: {len(historical_data)} échantillons")
    
    # 2. Apprentissage des patterns comportementaux
    print("\n🧠 Apprentissage des patterns comportementaux...")
    patterns = await system['behavior_learning_engine'].learn_behavior_patterns(
        'demo-tenant', historical_data
    )
    
    print(f"Patterns découverts: {len(patterns)}")
    for i, pattern in enumerate(patterns[:3]):
        print(f"  {i+1}. {pattern.pattern_type} - Confiance: {pattern.confidence:.2f}")
        print(f"     Métriques: {pattern.metrics_involved}")
        print(f"     Recommandation: {pattern.recommendation}")
    
    # 3. Génération de configuration adaptative
    print("\n⚙️ Génération de configuration adaptative...")
    current_metrics = {
        'cpu_usage': 75.0,
        'memory_usage': 68.0,
        'error_rate': 0.002,
        'response_time': 350,
        'disk_usage': 45.0
    }
    
    adaptive_config = await system['dynamic_config_engine'].generate_adaptive_configuration(
        'demo-tenant', current_metrics, {'environment': 'production', 'service': 'api'}
    )
    
    print("Configuration générée:")
    print(json.dumps(adaptive_config['configuration'], indent=2))
    print(f"Statut de validation: {adaptive_config['validation']['validation_status']}")
    
    # 4. Détection de drift
    print("\n🔍 Détection de drift...")
    
    # Simulation de nouvelles données avec drift
    new_dates = pd.date_range(start='2024-02-01', end='2024-02-07', freq='H')
    new_data = []
    
    for timestamp in new_dates:
        hour = timestamp.hour
        # Simulation d'un drift: augmentation générale de la charge
        cpu_base = 50 + 30 * (1 if 9 <= hour <= 17 else 0.3)  # +20 de base
        cpu_usage = max(0, min(100, cpu_base + np.random.normal(0, 15)))  # Plus de variance
        
        new_data.append({
            'timestamp': timestamp,
            'cpu_usage': cpu_usage,
            'memory_usage': max(0, min(100, cpu_usage * 0.9 + np.random.normal(0, 8))),
            'error_rate': max(0, (cpu_usage - 40) / 800 + np.random.exponential(0.002)),
            'response_time': max(50, 250 + (cpu_usage - 50) * 6 + np.random.normal(0, 30)),
            'disk_usage': max(0, min(100, 50 + np.random.normal(0, 8)))
        })
    
    new_data_df = pd.DataFrame(new_data)
    
    drift_results = await system['drift_detection_system'].detect_data_drift(
        'demo-tenant', new_data_df, historical_data
    )
    
    print(f"Drift détecté: {drift_results['drift_detected']}")
    if drift_results['drift_detected']:
        print(f"Sévérité: {drift_results['drift_severity']}")
        print(f"Métriques affectées: {drift_results['affected_features']}")
        print("Recommandations:")
        for rec in drift_results['recommendations']:
            print(f"  - {rec}")
    
    print("\n✅ Démonstration du système d'apprentissage adaptatif terminée!")

if __name__ == "__main__":
    asyncio.run(demonstrate_adaptive_learning())
