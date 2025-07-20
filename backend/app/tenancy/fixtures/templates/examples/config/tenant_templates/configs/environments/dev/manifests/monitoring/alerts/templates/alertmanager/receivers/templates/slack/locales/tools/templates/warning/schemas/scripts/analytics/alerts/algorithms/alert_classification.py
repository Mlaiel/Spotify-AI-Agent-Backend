"""
Module de classification intelligente des alertes avec analyse d'impact business.

Ce module implémente des algorithmes ML pour :
- Classification automatique des alertes par catégorie
- Prédiction de sévérité basée sur l'historique
- Analyse d'impact business en temps réel
- Scoring de priorité intelligent
- Apprentissage adaptatif des patterns d'alertes

Optimisé pour la production avec cache Redis et métriques Prometheus.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import pickle
import json
import asyncio
from enum import Enum
import redis
import hashlib

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Deep Learning for advanced classification
import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import pipeline, AutoTokenizer, AutoModel

# Monitoring et métriques
from prometheus_client import Counter, Histogram, Gauge, Summary

logger = logging.getLogger(__name__)

# Métriques Prometheus
ALERT_CLASSIFICATION_COUNTER = Counter('alert_classifications_total', 'Total alerts classified', ['category', 'severity'])
CLASSIFICATION_LATENCY = Histogram('alert_classification_duration_seconds', 'Time spent classifying alerts')
BUSINESS_IMPACT_SCORE = Summary('business_impact_scores', 'Business impact scores calculated')
CLASSIFICATION_ACCURACY = Gauge('classification_accuracy', 'Current classification accuracy')

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertCategory(Enum):
    """Catégories d'alertes."""
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    SECURITY = "security"
    CAPACITY = "capacity"
    NETWORK = "network"
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    DATA_QUALITY = "data_quality"
    BUSINESS_LOGIC = "business_logic"
    USER_EXPERIENCE = "user_experience"

class BusinessImpact(Enum):
    """Niveaux d'impact business."""
    CATASTROPHIC = "catastrophic"  # Perte de revenus massive
    SEVERE = "severe"             # Impact significatif sur les utilisateurs
    MODERATE = "moderate"         # Dégradation de service
    MINOR = "minor"              # Impact minimal
    NEGLIGIBLE = "negligible"    # Aucun impact perceptible

@dataclass
class AlertContext:
    """Contexte enrichi d'une alerte."""
    alert_id: str
    timestamp: datetime
    source_system: str
    metric_name: str
    current_value: float
    threshold_value: float
    description: str
    tags: Dict[str, str] = field(default_factory=dict)
    historical_occurrences: int = 0
    related_alerts: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    user_impact_count: int = 0

@dataclass
class ClassificationResult:
    """Résultat de classification d'alerte."""
    alert_id: str
    category: AlertCategory
    severity: AlertSeverity
    business_impact: BusinessImpact
    priority_score: float
    confidence: float
    reasoning: str
    predicted_resolution_time: Optional[timedelta] = None
    recommended_actions: List[str] = field(default_factory=list)
    escalation_path: List[str] = field(default_factory=list)
    
class BaseAlertClassifier(ABC):
    """Classe de base pour tous les classificateurs d'alertes."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.is_trained = False
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.feature_cache = {}
        
    @abstractmethod
    def train(self, training_data: List[Tuple[AlertContext, ClassificationResult]]) -> None:
        """Entraîne le classificateur."""
        pass
    
    @abstractmethod
    def predict(self, alert_context: AlertContext) -> ClassificationResult:
        """Prédit la classification d'une alerte."""
        pass
    
    def extract_features(self, alert_context: AlertContext) -> Dict[str, Any]:
        """Extrait les features pour la classification."""
        cache_key = f"features:{hashlib.md5(str(alert_context).encode()).hexdigest()}"
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        features = {
            # Features temporelles
            'hour_of_day': alert_context.timestamp.hour,
            'day_of_week': alert_context.timestamp.weekday(),
            'is_weekend': alert_context.timestamp.weekday() >= 5,
            'is_business_hours': 9 <= alert_context.timestamp.hour <= 17,
            
            # Features métriques
            'threshold_deviation': abs(alert_context.current_value - alert_context.threshold_value) / alert_context.threshold_value if alert_context.threshold_value != 0 else 0,
            'value_magnitude': abs(alert_context.current_value),
            
            # Features historiques
            'historical_frequency': alert_context.historical_occurrences,
            'has_related_alerts': len(alert_context.related_alerts) > 0,
            'related_alerts_count': len(alert_context.related_alerts),
            
            # Features d'impact
            'affected_services_count': len(alert_context.affected_services),
            'user_impact_count': alert_context.user_impact_count,
            
            # Features textuelles (TF-IDF sera appliqué séparément)
            'description_text': alert_context.description,
            'metric_name_text': alert_context.metric_name,
            'source_system': alert_context.source_system,
            
            # Features des tags
            'tag_count': len(alert_context.tags),
        }
        
        # Ajout des features des tags spécifiques
        important_tags = ['environment', 'team', 'criticality', 'region']
        for tag in important_tags:
            features[f'tag_{tag}'] = alert_context.tags.get(tag, 'unknown')
        
        self.feature_cache[cache_key] = features
        return features
    
    def save_model(self, path: str) -> None:
        """Sauvegarde le modèle."""
        model_data = {
            'model': self.model,
            'config': self.config,
            'model_name': self.model_name,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, path)
        
    def load_model(self, path: str) -> None:
        """Charge un modèle sauvegardé."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.config = model_data['config']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']

class AlertClassifier(BaseAlertClassifier):
    """Classificateur principal utilisant Random Forest pour la catégorisation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        config = config or default_config
        super().__init__("random_forest_classifier", config)
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def train(self, training_data: List[Tuple[AlertContext, ClassificationResult]]) -> None:
        """Entraîne le classificateur Random Forest."""
        
        logger.info(f"Training classifier with {len(training_data)} samples")
        
        try:
            # Extraction des features et labels
            features_list = []
            categories = []
            severities = []
            business_impacts = []
            
            text_features = []
            
            for alert_context, classification_result in training_data:
                features = self.extract_features(alert_context)
                features_list.append(features)
                categories.append(classification_result.category.value)
                severities.append(classification_result.severity.value)
                business_impacts.append(classification_result.business_impact.value)
                
                # Combinaison des features textuelles
                text_features.append(f"{features['description_text']} {features['metric_name_text']}")
            
            # Traitement des features textuelles
            tfidf_features = self.tfidf_vectorizer.fit_transform(text_features).toarray()
            
            # Préparation des features numériques
            numeric_features = []
            categorical_features = []
            
            for features in features_list:
                numeric_row = [
                    features['hour_of_day'], features['day_of_week'],
                    features['threshold_deviation'], features['value_magnitude'],
                    features['historical_frequency'], features['related_alerts_count'],
                    features['affected_services_count'], features['user_impact_count'],
                    features['tag_count'],
                    int(features['is_weekend']), int(features['is_business_hours']),
                    int(features['has_related_alerts'])
                ]
                numeric_features.append(numeric_row)
                
                categorical_row = [
                    features['source_system'],
                    features.get('tag_environment', 'unknown'),
                    features.get('tag_team', 'unknown'),
                    features.get('tag_criticality', 'unknown'),
                    features.get('tag_region', 'unknown')
                ]
                categorical_features.append(categorical_row)
            
            # Encodage des features catégorielles
            categorical_encoded = []
            categorical_names = ['source_system', 'environment', 'team', 'criticality', 'region']
            
            for i, name in enumerate(categorical_names):
                if name not in self.label_encoders:
                    self.label_encoders[name] = LabelEncoder()
                
                column_data = [row[i] for row in categorical_features]
                encoded_column = self.label_encoders[name].fit_transform(column_data)
                categorical_encoded.append(encoded_column)
            
            categorical_encoded = np.array(categorical_encoded).T
            
            # Combinaison de toutes les features
            numeric_features = np.array(numeric_features)
            all_features = np.hstack([numeric_features, categorical_encoded, tfidf_features])
            
            # Normalisation
            all_features = self.scaler.fit_transform(all_features)
            
            # Entraînement des modèles pour chaque tâche
            self.model = {
                'category': RandomForestClassifier(**self.config),
                'severity': RandomForestClassifier(**self.config),
                'business_impact': RandomForestClassifier(**self.config)
            }
            
            # Entraînement
            for task, labels in [('category', categories), ('severity', severities), ('business_impact', business_impacts)]:
                self.model[task].fit(all_features, labels)
                
                # Validation croisée
                cv_scores = cross_val_score(self.model[task], all_features, labels, cv=5)
                logger.info(f"{task} classifier CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            self.is_trained = True
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, alert_context: AlertContext) -> ClassificationResult:
        """Prédit la classification d'une alerte."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
        
        with CLASSIFICATION_LATENCY.time():
            try:
                # Extraction des features
                features = self.extract_features(alert_context)
                
                # Préparation des features pour prédiction
                text_feature = f"{features['description_text']} {features['metric_name_text']}"
                tfidf_feature = self.tfidf_vectorizer.transform([text_feature]).toarray()
                
                numeric_feature = np.array([[
                    features['hour_of_day'], features['day_of_week'],
                    features['threshold_deviation'], features['value_magnitude'],
                    features['historical_frequency'], features['related_alerts_count'],
                    features['affected_services_count'], features['user_impact_count'],
                    features['tag_count'],
                    int(features['is_weekend']), int(features['is_business_hours']),
                    int(features['has_related_alerts'])
                ]])
                
                # Encodage des features catégorielles
                categorical_values = [
                    features['source_system'],
                    features.get('tag_environment', 'unknown'),
                    features.get('tag_team', 'unknown'),
                    features.get('tag_criticality', 'unknown'),
                    features.get('tag_region', 'unknown')
                ]
                
                categorical_encoded = []
                categorical_names = ['source_system', 'environment', 'team', 'criticality', 'region']
                
                for i, name in enumerate(categorical_names):
                    try:
                        encoded_value = self.label_encoders[name].transform([categorical_values[i]])[0]
                    except ValueError:
                        # Valeur inconnue, utiliser l'encodage par défaut
                        encoded_value = 0
                    categorical_encoded.append(encoded_value)
                
                categorical_encoded = np.array(categorical_encoded).reshape(1, -1)
                
                # Combinaison de toutes les features
                all_features = np.hstack([numeric_feature, categorical_encoded, tfidf_feature])
                all_features = self.scaler.transform(all_features)
                
                # Prédictions
                category_pred = self.model['category'].predict(all_features)[0]
                category_proba = self.model['category'].predict_proba(all_features)[0].max()
                
                severity_pred = self.model['severity'].predict(all_features)[0]
                severity_proba = self.model['severity'].predict_proba(all_features)[0].max()
                
                business_impact_pred = self.model['business_impact'].predict(all_features)[0]
                business_impact_proba = self.model['business_impact'].predict_proba(all_features)[0].max()
                
                # Calcul du score de priorité
                priority_score = self._calculate_priority_score(
                    AlertCategory(category_pred),
                    AlertSeverity(severity_pred),
                    BusinessImpact(business_impact_pred),
                    features
                )
                
                # Confiance globale
                overall_confidence = (category_proba + severity_proba + business_impact_proba) / 3
                
                # Génération du raisonnement
                reasoning = self._generate_reasoning(features, category_pred, severity_pred, business_impact_pred)
                
                # Actions recommandées
                recommended_actions = self._get_recommended_actions(
                    AlertCategory(category_pred),
                    AlertSeverity(severity_pred),
                    BusinessImpact(business_impact_pred)
                )
                
                # Chemin d'escalade
                escalation_path = self._get_escalation_path(
                    AlertSeverity(severity_pred),
                    BusinessImpact(business_impact_pred),
                    features.get('tag_team', 'unknown')
                )
                
                result = ClassificationResult(
                    alert_id=alert_context.alert_id,
                    category=AlertCategory(category_pred),
                    severity=AlertSeverity(severity_pred),
                    business_impact=BusinessImpact(business_impact_pred),
                    priority_score=priority_score,
                    confidence=overall_confidence,
                    reasoning=reasoning,
                    recommended_actions=recommended_actions,
                    escalation_path=escalation_path
                )
                
                # Métriques
                ALERT_CLASSIFICATION_COUNTER.labels(
                    category=category_pred,
                    severity=severity_pred
                ).inc()
                
                BUSINESS_IMPACT_SCORE.observe(priority_score)
                
                return result
                
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                raise
    
    def _calculate_priority_score(self, category: AlertCategory, severity: AlertSeverity, 
                                 business_impact: BusinessImpact, features: Dict[str, Any]) -> float:
        """Calcule un score de priorité composite."""
        
        # Poids de base par sévérité
        severity_weights = {
            AlertSeverity.CRITICAL: 1.0,
            AlertSeverity.HIGH: 0.8,
            AlertSeverity.MEDIUM: 0.6,
            AlertSeverity.LOW: 0.4,
            AlertSeverity.INFO: 0.2
        }
        
        # Poids d'impact business
        impact_weights = {
            BusinessImpact.CATASTROPHIC: 1.0,
            BusinessImpact.SEVERE: 0.8,
            BusinessImpact.MODERATE: 0.6,
            BusinessImpact.MINOR: 0.4,
            BusinessImpact.NEGLIGIBLE: 0.2
        }
        
        # Poids par catégorie
        category_weights = {
            AlertCategory.SECURITY: 0.95,
            AlertCategory.AVAILABILITY: 0.9,
            AlertCategory.PERFORMANCE: 0.8,
            AlertCategory.CAPACITY: 0.7,
            AlertCategory.NETWORK: 0.75,
            AlertCategory.APPLICATION: 0.85,
            AlertCategory.INFRASTRUCTURE: 0.8,
            AlertCategory.DATA_QUALITY: 0.6,
            AlertCategory.BUSINESS_LOGIC: 0.9,
            AlertCategory.USER_EXPERIENCE: 0.85
        }
        
        base_score = (
            severity_weights[severity] * 0.4 +
            impact_weights[business_impact] * 0.4 +
            category_weights[category] * 0.2
        )
        
        # Facteurs d'ajustement
        adjustments = 0.0
        
        # Ajustement temporel
        if features['is_business_hours']:
            adjustments += 0.1
        if features['is_weekend']:
            adjustments -= 0.1
        
        # Ajustement basé sur l'historique
        if features['historical_frequency'] > 10:
            adjustments -= 0.1  # Alerte fréquente, moins prioritaire
        elif features['historical_frequency'] == 0:
            adjustments += 0.1  # Nouvelle alerte, plus prioritaire
        
        # Ajustement basé sur l'impact utilisateur
        if features['user_impact_count'] > 100:
            adjustments += 0.2
        elif features['user_impact_count'] > 10:
            adjustments += 0.1
        
        # Ajustement basé sur les services affectés
        if features['affected_services_count'] > 5:
            adjustments += 0.15
        elif features['affected_services_count'] > 2:
            adjustments += 0.05
        
        # Ajustement basé sur la déviation du seuil
        if features['threshold_deviation'] > 2.0:
            adjustments += 0.1
        elif features['threshold_deviation'] > 1.5:
            adjustments += 0.05
        
        final_score = max(0.0, min(1.0, base_score + adjustments))
        return final_score
    
    def _generate_reasoning(self, features: Dict[str, Any], category: str, 
                          severity: str, business_impact: str) -> str:
        """Génère un raisonnement explicatif pour la classification."""
        
        reasoning_parts = [
            f"Classified as {category} alert with {severity} severity and {business_impact} business impact."
        ]
        
        # Facteurs temporels
        if features['is_business_hours']:
            reasoning_parts.append("Occurred during business hours, increasing priority.")
        if features['is_weekend']:
            reasoning_parts.append("Occurred during weekend, potentially reducing immediate impact.")
        
        # Facteurs historiques
        if features['historical_frequency'] > 10:
            reasoning_parts.append(f"This is a recurring alert (occurred {features['historical_frequency']} times before).")
        elif features['historical_frequency'] == 0:
            reasoning_parts.append("This is a new type of alert, requiring immediate attention.")
        
        # Impact utilisateur
        if features['user_impact_count'] > 0:
            reasoning_parts.append(f"Potentially affects {features['user_impact_count']} users.")
        
        # Services affectés
        if features['affected_services_count'] > 1:
            reasoning_parts.append(f"Affects {features['affected_services_count']} services.")
        
        # Déviation métrique
        if features['threshold_deviation'] > 1.5:
            reasoning_parts.append(f"Metric significantly exceeds threshold (deviation: {features['threshold_deviation']:.2f}).")
        
        return " ".join(reasoning_parts)
    
    def _get_recommended_actions(self, category: AlertCategory, severity: AlertSeverity, 
                               business_impact: BusinessImpact) -> List[str]:
        """Retourne les actions recommandées basées sur la classification."""
        
        actions = []
        
        # Actions basées sur la sévérité
        if severity == AlertSeverity.CRITICAL:
            actions.extend([
                "Immediately escalate to on-call engineer",
                "Activate incident response team",
                "Notify stakeholders",
                "Prepare rollback plan"
            ])
        elif severity == AlertSeverity.HIGH:
            actions.extend([
                "Assign to primary team",
                "Investigate within 15 minutes",
                "Prepare mitigation strategy"
            ])
        elif severity == AlertSeverity.MEDIUM:
            actions.extend([
                "Add to team queue",
                "Investigate within 1 hour",
                "Monitor for escalation"
            ])
        else:
            actions.extend([
                "Log for future analysis",
                "Monitor trends"
            ])
        
        # Actions basées sur la catégorie
        category_actions = {
            AlertCategory.SECURITY: [
                "Activate security team",
                "Check for data breach indicators",
                "Review access logs"
            ],
            AlertCategory.PERFORMANCE: [
                "Check system resources",
                "Analyze performance metrics",
                "Consider scaling resources"
            ],
            AlertCategory.AVAILABILITY: [
                "Verify service health",
                "Check dependencies",
                "Test failover mechanisms"
            ],
            AlertCategory.CAPACITY: [
                "Review resource utilization",
                "Plan capacity expansion",
                "Optimize resource allocation"
            ]
        }
        
        if category in category_actions:
            actions.extend(category_actions[category])
        
        return actions
    
    def _get_escalation_path(self, severity: AlertSeverity, business_impact: BusinessImpact, 
                           team: str) -> List[str]:
        """Définit le chemin d'escalade basé sur la classification."""
        
        escalation = []
        
        # Escalade basée sur la sévérité et l'impact
        if severity == AlertSeverity.CRITICAL or business_impact == BusinessImpact.CATASTROPHIC:
            escalation = [
                f"{team}-on-call",
                f"{team}-lead",
                "incident-commander",
                "engineering-director",
                "cto"
            ]
        elif severity == AlertSeverity.HIGH or business_impact == BusinessImpact.SEVERE:
            escalation = [
                f"{team}-on-call",
                f"{team}-lead",
                "engineering-manager"
            ]
        else:
            escalation = [
                f"{team}-on-call",
                f"{team}-lead"
            ]
        
        return escalation

class SeverityPredictor(BaseAlertClassifier):
    """Prédicteur spécialisé pour la sévérité des alertes utilisant XGBoost."""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        config = config or default_config
        super().__init__("gradient_boosting_severity", config)
    
    def train(self, training_data: List[Tuple[AlertContext, ClassificationResult]]) -> None:
        """Entraîne le prédicteur de sévérité."""
        logger.info("Training severity predictor...")
        
        features_list = []
        severities = []
        
        for alert_context, classification_result in training_data:
            features = self.extract_features(alert_context)
            
            # Features spécifiques pour la prédiction de sévérité
            severity_features = [
                features['threshold_deviation'],
                features['user_impact_count'],
                features['affected_services_count'],
                features['historical_frequency'],
                features['hour_of_day'],
                int(features['is_business_hours']),
                int(features['has_related_alerts']),
                features['value_magnitude']
            ]
            
            features_list.append(severity_features)
            severities.append(classification_result.severity.value)
        
        X = np.array(features_list)
        y = severities
        
        self.model = GradientBoostingClassifier(**self.config)
        self.model.fit(X, y)
        
        # Évaluation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        logger.info(f"Severity predictor CV accuracy: {cv_scores.mean():.3f}")
        
        self.is_trained = True
    
    def predict(self, alert_context: AlertContext) -> ClassificationResult:
        """Prédit la sévérité d'une alerte."""
        if not self.is_trained:
            raise ValueError("Severity predictor must be trained")
        
        features = self.extract_features(alert_context)
        
        severity_features = np.array([[
            features['threshold_deviation'],
            features['user_impact_count'],
            features['affected_services_count'],
            features['historical_frequency'],
            features['hour_of_day'],
            int(features['is_business_hours']),
            int(features['has_related_alerts']),
            features['value_magnitude']
        ]])
        
        severity_pred = self.model.predict(severity_features)[0]
        severity_proba = self.model.predict_proba(severity_features)[0].max()
        
        return ClassificationResult(
            alert_id=alert_context.alert_id,
            category=AlertCategory.APPLICATION,  # Valeur par défaut
            severity=AlertSeverity(severity_pred),
            business_impact=BusinessImpact.MODERATE,  # Valeur par défaut
            priority_score=severity_proba,
            confidence=severity_proba,
            reasoning=f"Severity predicted based on threshold deviation and impact metrics."
        )

class BusinessImpactAnalyzer:
    """Analyseur d'impact business utilisant des règles métier avancées."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.impact_rules = self._load_impact_rules()
        self.service_criticality = self._load_service_criticality()
        
    def _load_impact_rules(self) -> Dict[str, Any]:
        """Charge les règles d'impact business."""
        return {
            'revenue_impact_thresholds': {
                'catastrophic': 100000,  # > $100k/hour
                'severe': 10000,        # > $10k/hour
                'moderate': 1000,       # > $1k/hour
                'minor': 100,          # > $100/hour
                'negligible': 0        # <= $100/hour
            },
            'user_impact_thresholds': {
                'catastrophic': 10000,  # > 10k users
                'severe': 1000,        # > 1k users
                'moderate': 100,       # > 100 users
                'minor': 10,          # > 10 users
                'negligible': 0       # <= 10 users
            },
            'sla_breach_multipliers': {
                'critical_sla': 2.0,
                'important_sla': 1.5,
                'standard_sla': 1.0
            }
        }
    
    def _load_service_criticality(self) -> Dict[str, str]:
        """Charge la criticité des services."""
        return {
            'payment-service': 'critical',
            'user-auth': 'critical',
            'streaming-core': 'critical',
            'recommendation-engine': 'important',
            'analytics': 'standard',
            'logging': 'standard'
        }
    
    def analyze_impact(self, alert_context: AlertContext) -> BusinessImpact:
        """Analyse l'impact business d'une alerte."""
        
        impact_score = 0.0
        
        # Impact basé sur le nombre d'utilisateurs affectés
        user_impact = self._calculate_user_impact(alert_context.user_impact_count)
        impact_score += user_impact
        
        # Impact basé sur les services affectés
        service_impact = self._calculate_service_impact(alert_context.affected_services)
        impact_score += service_impact
        
        # Impact basé sur l'historique (SLA breaches)
        historical_impact = self._calculate_historical_impact(alert_context)
        impact_score += historical_impact
        
        # Impact basé sur le moment (business hours)
        temporal_impact = self._calculate_temporal_impact(alert_context.timestamp)
        impact_score += temporal_impact
        
        # Conversion du score en niveau d'impact
        if impact_score >= 0.8:
            return BusinessImpact.CATASTROPHIC
        elif impact_score >= 0.6:
            return BusinessImpact.SEVERE
        elif impact_score >= 0.4:
            return BusinessImpact.MODERATE
        elif impact_score >= 0.2:
            return BusinessImpact.MINOR
        else:
            return BusinessImpact.NEGLIGIBLE
    
    def _calculate_user_impact(self, user_count: int) -> float:
        """Calcule l'impact basé sur le nombre d'utilisateurs."""
        thresholds = self.impact_rules['user_impact_thresholds']
        
        if user_count >= thresholds['catastrophic']:
            return 1.0
        elif user_count >= thresholds['severe']:
            return 0.8
        elif user_count >= thresholds['moderate']:
            return 0.6
        elif user_count >= thresholds['minor']:
            return 0.4
        else:
            return 0.2
    
    def _calculate_service_impact(self, affected_services: List[str]) -> float:
        """Calcule l'impact basé sur les services affectés."""
        if not affected_services:
            return 0.0
        
        max_criticality = 0.0
        
        for service in affected_services:
            criticality = self.service_criticality.get(service, 'standard')
            
            if criticality == 'critical':
                max_criticality = max(max_criticality, 1.0)
            elif criticality == 'important':
                max_criticality = max(max_criticality, 0.7)
            elif criticality == 'standard':
                max_criticality = max(max_criticality, 0.4)
        
        # Bonus pour multiple services affectés
        service_count_multiplier = min(1.2, 1.0 + (len(affected_services) - 1) * 0.1)
        
        return max_criticality * service_count_multiplier
    
    def _calculate_historical_impact(self, alert_context: AlertContext) -> float:
        """Calcule l'impact basé sur l'historique des alertes."""
        # Les alertes récurrentes ont généralement moins d'impact
        # sauf si elles indiquent une dégradation systémique
        
        if alert_context.historical_occurrences == 0:
            return 0.1  # Nouvelle alerte = attention
        elif alert_context.historical_occurrences < 5:
            return 0.0  # Normale
        else:
            return -0.1  # Récurrente = moins prioritaire
    
    def _calculate_temporal_impact(self, timestamp: datetime) -> float:
        """Calcule l'impact basé sur le moment de l'alerte."""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        # Impact plus élevé pendant les heures de bureau
        if 9 <= hour <= 17 and not is_weekend:
            return 0.2
        elif 8 <= hour <= 20 and not is_weekend:
            return 0.1
        else:
            return 0.0

# Factory pour créer des classificateurs
class ClassifierFactory:
    """Factory pour créer des classificateurs configurés."""
    
    @staticmethod
    def create_classifier(classifier_type: str, config: Dict[str, Any] = None) -> BaseAlertClassifier:
        """Crée un classificateur du type spécifié."""
        
        if classifier_type == "random_forest":
            return AlertClassifier(config)
        elif classifier_type == "severity_predictor":
            return SeverityPredictor(config)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    @staticmethod
    def create_ensemble_classifier(config: Dict[str, Any] = None) -> 'EnsembleClassifier':
        """Crée un classificateur d'ensemble."""
        return EnsembleClassifier(config)

class EnsembleClassifier:
    """Classificateur d'ensemble combinant plusieurs approches."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self.classifiers = {
            'random_forest': AlertClassifier(),
            'severity_predictor': SeverityPredictor()
        }
        
        self.impact_analyzer = BusinessImpactAnalyzer()
        self.weights = config.get('weights', {
            'random_forest': 0.6,
            'severity_predictor': 0.4
        })
        
    def train(self, training_data: List[Tuple[AlertContext, ClassificationResult]]) -> None:
        """Entraîne tous les classificateurs de l'ensemble."""
        
        for name, classifier in self.classifiers.items():
            logger.info(f"Training {name}...")
            classifier.train(training_data)
    
    def predict(self, alert_context: AlertContext) -> ClassificationResult:
        """Prédit en combinant les résultats de tous les classificateurs."""
        
        predictions = {}
        
        for name, classifier in self.classifiers.items():
            try:
                pred = classifier.predict(alert_context)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Classifier {name} failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("All classifiers failed")
        
        # Combinaison des prédictions
        combined_result = self._combine_predictions(predictions, alert_context)
        
        return combined_result
    
    def _combine_predictions(self, predictions: Dict[str, ClassificationResult], 
                           alert_context: AlertContext) -> ClassificationResult:
        """Combine les prédictions des différents classificateurs."""
        
        # Vote pondéré pour la catégorie et la sévérité
        category_votes = {}
        severity_votes = {}
        
        total_weight = 0
        weighted_confidence = 0
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0)
            total_weight += weight
            weighted_confidence += pred.confidence * weight
            
            # Votes pour catégorie
            if pred.category not in category_votes:
                category_votes[pred.category] = 0
            category_votes[pred.category] += weight
            
            # Votes pour sévérité
            if pred.severity not in severity_votes:
                severity_votes[pred.severity] = 0
            severity_votes[pred.severity] += weight
        
        # Sélection des prédictions majoritaires
        final_category = max(category_votes, key=category_votes.get)
        final_severity = max(severity_votes, key=severity_votes.get)
        
        # Analyse d'impact business
        business_impact = self.impact_analyzer.analyze_impact(alert_context)
        
        # Calcul du score de priorité final
        priority_score = self._calculate_ensemble_priority(
            final_category, final_severity, business_impact, alert_context
        )
        
        final_confidence = weighted_confidence / total_weight
        
        return ClassificationResult(
            alert_id=alert_context.alert_id,
            category=final_category,
            severity=final_severity,
            business_impact=business_impact,
            priority_score=priority_score,
            confidence=final_confidence,
            reasoning=f"Ensemble prediction combining {len(predictions)} classifiers"
        )
    
    def _calculate_ensemble_priority(self, category: AlertCategory, severity: AlertSeverity,
                                   business_impact: BusinessImpact, alert_context: AlertContext) -> float:
        """Calcule le score de priorité final pour l'ensemble."""
        
        # Utilise la logique du classificateur principal
        classifier = self.classifiers['random_forest']
        features = classifier.extract_features(alert_context)
        
        return classifier._calculate_priority_score(category, severity, business_impact, features)
