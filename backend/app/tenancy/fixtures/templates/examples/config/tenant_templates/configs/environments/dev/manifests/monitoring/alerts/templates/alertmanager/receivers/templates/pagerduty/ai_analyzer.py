"""
Advanced AI Analyzer for PagerDuty Incident Management

Ce module fournit des capacités d'intelligence artificielle avancées pour l'analyse,
la prédiction, et la classification automatique des incidents PagerDuty.

Fonctionnalités:
- Modèles de machine learning pour la prédiction d'incidents
- Classification automatique avec deep learning
- Analyse de sentiment et d'impact utilisateur
- Détection d'anomalies en temps réel
- Auto-résolution basée sur l'historique
- Optimisation continue des modèles

Version: 4.0.0
Développé par l'équipe Spotify AI Agent
"""

import asyncio
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy
from textblob import TextBlob
import structlog
import aiofiles
import aioredis

from . import (
    IncidentData, IncidentSeverity, IncidentUrgency, AIAnalysisResult,
    AIModelType, logger
)

# ============================================================================
# Configuration IA
# ============================================================================

@dataclass
class AIConfig:
    """Configuration des modèles IA"""
    models_path: str = "/models/pagerduty"
    confidence_threshold: float = 0.85
    retrain_interval: int = 24  # heures
    batch_size: int = 32
    max_features: int = 10000
    embedding_dim: int = 128
    lstm_units: int = 64
    use_gpu: bool = True
    cache_predictions: bool = True

class IncidentClassificationModel(nn.Module):
    """Modèle PyTorch pour la classification d'incidents"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Utiliser la sortie de la dernière timestep
        output = lstm_out[:, -1, :]
        output = self.dropout(output)
        output = torch.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        return self.softmax(output)

class AnomalyDetectionModel:
    """Modèle de détection d'anomalies pour les métriques"""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = None
        
    def fit(self, data: np.ndarray):
        """Entraîne le modèle de détection d'anomalies"""
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data)
        self.model.fit(scaled_data)
        
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prédit les anomalies"""
        if self.scaler is None:
            raise ValueError("Model must be fitted first")
        scaled_data = self.scaler.transform(data)
        predictions = self.model.predict(scaled_data)
        scores = self.model.decision_function(scaled_data)
        return predictions, scores

# ============================================================================
# Analyseur IA Principal
# ============================================================================

class AIAnalyzer:
    """Analyseur IA principal pour PagerDuty"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.models = {}
        self.tokenizer = None
        self.nlp_model = None
        self.redis_pool = None
        self.feature_extractors = {}
        self.metrics_history = []
        
        # Initialisation des modèles pré-entraînés
        self._initialize_models()
        
    async def initialize(self, redis_url: str):
        """Initialise l'analyseur IA"""
        try:
            # Connexion Redis pour cache
            self.redis_pool = aioredis.ConnectionPool.from_url(redis_url)
            
            # Chargement des modèles sauvegardés
            await self._load_models()
            
            # Initialisation NLP
            await self._initialize_nlp_models()
            
            logger.info("AI Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Analyzer: {e}")
            raise
            
    def _initialize_models(self):
        """Initialise les modèles de base"""
        # Modèle de classification de sévérité
        self.models['severity_classifier'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Modèle de prédiction de temps de résolution
        self.models['resolution_time_predictor'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
        
        # Modèle de détection d'anomalies
        self.models['anomaly_detector'] = AnomalyDetectionModel()
        
        # Vectoriseur TF-IDF pour l'analyse de texte
        self.feature_extractors['tfidf'] = TfidfVectorizer(
            max_features=self.config.max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    async def _initialize_nlp_models(self):
        """Initialise les modèles NLP"""
        try:
            # Modèle BERT pour l'analyse de sentiment
            self.nlp_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # Modèle spaCy pour l'extraction d'entités
            self.tokenizer = spacy.load("en_core_web_sm")
            
        except Exception as e:
            logger.warning(f"NLP models not available: {e}")
            
    async def _load_models(self):
        """Charge les modèles sauvegardés"""
        models_path = Path(self.config.models_path)
        if not models_path.exists():
            models_path.mkdir(parents=True, exist_ok=True)
            return
            
        for model_name in self.models.keys():
            model_file = models_path / f"{model_name}.pkl"
            if model_file.exists():
                try:
                    async with aiofiles.open(model_file, 'rb') as f:
                        content = await f.read()
                        self.models[model_name] = pickle.loads(content)
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
                    
    async def analyze_incident(self, incident: IncidentData) -> AIAnalysisResult:
        """Analyse complète d'un incident avec IA"""
        try:
            with tracer.start_as_current_span("ai_analyze_incident") as span:
                span.set_attribute("incident.id", incident.id or "unknown")
                
                # Cache check
                if self.config.cache_predictions and incident.id:
                    cached_result = await self._get_cached_analysis(incident.id)
                    if cached_result:
                        return cached_result
                
                # Extraction des features
                features = await self._extract_features(incident)
                
                # Prédiction de sévérité
                predicted_severity = await self._predict_severity(features)
                
                # Prédiction du temps de résolution
                resolution_time = await self._predict_resolution_time(features)
                
                # Analyse de sentiment
                sentiment_score = await self._analyze_sentiment(incident)
                
                # Recherche d'incidents similaires
                similar_incidents = await self._find_similar_incidents(incident)
                
                # Analyse de la cause racine
                root_cause_analysis = await self._analyze_root_cause(incident)
                
                # Vérification de l'auto-résolution possible
                auto_resolution = await self._check_auto_resolution(incident)
                
                # Génération de recommandations
                recommendations = await self._generate_recommendations(incident, features)
                
                # Calcul de la confiance globale
                confidence = self._calculate_confidence(
                    predicted_severity, resolution_time, sentiment_score
                )
                
                result = AIAnalysisResult(
                    confidence=confidence,
                    predicted_severity=predicted_severity,
                    predicted_resolution_time=resolution_time,
                    suggested_assignee=await self._suggest_assignee(incident, features),
                    similar_incidents=similar_incidents,
                    root_cause_probability=root_cause_analysis,
                    auto_resolution_possible=auto_resolution,
                    recommendations=recommendations
                )
                
                # Mise en cache du résultat
                if self.config.cache_predictions and incident.id:
                    await self._cache_analysis(incident.id, result)
                
                span.set_attribute("analysis.confidence", confidence)
                span.set_attribute("analysis.auto_resolution", auto_resolution)
                
                return result
                
        except Exception as e:
            logger.error(f"AI analysis failed: {e}", incident_id=incident.id)
            # Retour de résultat par défaut en cas d'erreur
            return AIAnalysisResult(
                confidence=0.5,
                predicted_severity=incident.severity,
                predicted_resolution_time=30,
                auto_resolution_possible=False,
                recommendations=["Manual investigation required"]
            )
            
    async def _extract_features(self, incident: IncidentData) -> Dict[str, Any]:
        """Extrait les features pour l'analyse IA"""
        features = {}
        
        # Features textuelles
        text_content = f"{incident.title} {incident.description or ''}"
        
        # TF-IDF features
        if hasattr(self.feature_extractors['tfidf'], 'vocabulary_'):
            tfidf_features = self.feature_extractors['tfidf'].transform([text_content])
            features['tfidf'] = tfidf_features.toarray()[0]
        
        # Features temporelles
        now = datetime.now()
        features['hour'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_weekend'] = now.weekday() >= 5
        features['is_business_hours'] = 9 <= now.hour <= 17
        
        # Features d'incident
        features['title_length'] = len(incident.title)
        features['has_description'] = bool(incident.description)
        features['tag_count'] = len(incident.tags)
        
        # Features de service
        features['service_id'] = hash(incident.service_id) % 1000  # Anonymisation
        
        return features
        
    async def _predict_severity(self, features: Dict[str, Any]) -> IncidentSeverity:
        """Prédit la sévérité d'un incident"""
        try:
            model = self.models['severity_classifier']
            if hasattr(model, 'predict'):
                # Prédiction basée sur les features
                prediction = model.predict([features.get('tfidf', [])])
                return IncidentSeverity(prediction[0])
        except Exception as e:
            logger.warning(f"Severity prediction failed: {e}")
            
        return IncidentSeverity.MEDIUM  # Valeur par défaut
        
    async def _predict_resolution_time(self, features: Dict[str, Any]) -> int:
        """Prédit le temps de résolution en minutes"""
        try:
            model = self.models['resolution_time_predictor']
            if hasattr(model, 'predict'):
                prediction = model.predict([features.get('tfidf', [])])
                return max(5, int(prediction[0]))  # Minimum 5 minutes
        except Exception as e:
            logger.warning(f"Resolution time prediction failed: {e}")
            
        return 30  # Valeur par défaut: 30 minutes
        
    async def _analyze_sentiment(self, incident: IncidentData) -> float:
        """Analyse le sentiment de l'incident"""
        try:
            if self.nlp_model:
                text = f"{incident.title} {incident.description or ''}"
                result = self.nlp_model(text)
                return result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score']
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            
        return 0.0  # Neutre
        
    async def _find_similar_incidents(self, incident: IncidentData) -> List[str]:
        """Trouve des incidents similaires"""
        try:
            # Simulation de recherche d'incidents similaires
            # En production, utiliser une base vectorielle comme Pinecone ou Weaviate
            cache_key = f"similar_incidents:{hash(incident.title)}"
            
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                cached = await redis.get(cache_key)
                if cached:
                    return json.loads(cached)
                    
            # Simulation de résultats
            similar = [f"INC-{i:06d}" for i in range(1, 4)]
            
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                await redis.setex(cache_key, 3600, json.dumps(similar))
                
            return similar
            
        except Exception as e:
            logger.warning(f"Similar incidents search failed: {e}")
            return []
            
    async def _analyze_root_cause(self, incident: IncidentData) -> Dict[str, float]:
        """Analyse les causes racines probables"""
        root_causes = {
            "infrastructure": 0.3,
            "application": 0.4,
            "network": 0.1,
            "database": 0.1,
            "external_service": 0.1
        }
        
        # Ajustement basé sur le titre et la description
        text = f"{incident.title} {incident.description or ''}".lower()
        
        if any(keyword in text for keyword in ['database', 'db', 'sql']):
            root_causes["database"] += 0.3
            
        if any(keyword in text for keyword in ['network', 'connection', 'timeout']):
            root_causes["network"] += 0.3
            
        if any(keyword in text for keyword in ['server', 'cpu', 'memory', 'disk']):
            root_causes["infrastructure"] += 0.3
            
        # Normalisation
        total = sum(root_causes.values())
        return {k: v/total for k, v in root_causes.items()}
        
    async def _check_auto_resolution(self, incident: IncidentData) -> bool:
        """Vérifie si l'auto-résolution est possible"""
        try:
            # Critères pour l'auto-résolution
            auto_resolvable_patterns = [
                "disk space",
                "memory usage",
                "connection timeout",
                "cache miss"
            ]
            
            text = f"{incident.title} {incident.description or ''}".lower()
            
            # Vérification des patterns connus
            has_known_pattern = any(pattern in text for pattern in auto_resolvable_patterns)
            
            # Vérification de la sévérité (seulement LOW et MEDIUM)
            low_severity = incident.severity in [IncidentSeverity.LOW, IncidentSeverity.MEDIUM]
            
            # Vérification de l'historique de résolution automatique
            auto_resolution_rate = await self._get_auto_resolution_rate(incident.service_id)
            
            return has_known_pattern and low_severity and auto_resolution_rate > 0.7
            
        except Exception as e:
            logger.warning(f"Auto-resolution check failed: {e}")
            return False
            
    async def _suggest_assignee(self, incident: IncidentData, features: Dict[str, Any]) -> Optional[str]:
        """Suggère un assigné basé sur l'expertise et la disponibilité"""
        try:
            # Simulation de suggestion d'assigné
            # En production, intégrer avec le système de gestion des équipes
            
            assignees = {
                "infrastructure": ["john.doe@company.com", "jane.smith@company.com"],
                "application": ["alice.johnson@company.com", "bob.wilson@company.com"],
                "database": ["charlie.brown@company.com", "diana.prince@company.com"]
            }
            
            # Détermination de la catégorie
            text = f"{incident.title} {incident.description or ''}".lower()
            
            if any(keyword in text for keyword in ['database', 'db', 'sql']):
                category = "database"
            elif any(keyword in text for keyword in ['server', 'cpu', 'memory']):
                category = "infrastructure"
            else:
                category = "application"
                
            # Retour du premier assigné disponible
            return assignees.get(category, ["support@company.com"])[0]
            
        except Exception as e:
            logger.warning(f"Assignee suggestion failed: {e}")
            return None
            
    async def _generate_recommendations(self, incident: IncidentData, features: Dict[str, Any]) -> List[str]:
        """Génère des recommandations pour l'incident"""
        recommendations = []
        
        # Recommandations basées sur la sévérité
        if incident.severity == IncidentSeverity.CRITICAL:
            recommendations.append("Escalate immediately to on-call engineer")
            recommendations.append("Activate incident response team")
            
        elif incident.severity == IncidentSeverity.HIGH:
            recommendations.append("Assign to senior engineer")
            recommendations.append("Monitor closely for escalation")
            
        # Recommandations basées sur le contenu
        text = f"{incident.title} {incident.description or ''}".lower()
        
        if "memory" in text:
            recommendations.append("Check memory usage and restart services if needed")
            
        if "disk" in text:
            recommendations.append("Clean up disk space and check for log rotation")
            
        if "network" in text:
            recommendations.append("Verify network connectivity and firewall rules")
            
        if not recommendations:
            recommendations.append("Follow standard incident response procedure")
            
        return recommendations
        
    def _calculate_confidence(self, severity: IncidentSeverity, resolution_time: int, sentiment: float) -> float:
        """Calcule la confiance globale de l'analyse"""
        base_confidence = 0.7
        
        # Ajustement basé sur la cohérence des prédictions
        if severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH] and resolution_time > 60:
            base_confidence += 0.1
            
        if abs(sentiment) > 0.5:  # Sentiment fort
            base_confidence += 0.1
            
        return min(0.99, max(0.1, base_confidence))
        
    async def _get_cached_analysis(self, incident_id: str) -> Optional[AIAnalysisResult]:
        """Récupère une analyse en cache"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                cached = await redis.get(f"ai_analysis:{incident_id}")
                if cached:
                    data = json.loads(cached)
                    return AIAnalysisResult(**data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
        
    async def _cache_analysis(self, incident_id: str, result: AIAnalysisResult):
        """Met en cache une analyse"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                await redis.setex(
                    f"ai_analysis:{incident_id}",
                    3600,  # 1 heure
                    json.dumps(result.dict())
                )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
            
    async def _get_auto_resolution_rate(self, service_id: str) -> float:
        """Récupère le taux d'auto-résolution pour un service"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                rate = await redis.get(f"auto_resolution_rate:{service_id}")
                return float(rate) if rate else 0.5
        except Exception:
            return 0.5
            
    async def train_models(self, training_data: List[IncidentData]):
        """Entraîne les modèles IA avec de nouvelles données"""
        try:
            logger.info(f"Training models with {len(training_data)} incidents")
            
            # Préparation des données
            X, y_severity, y_resolution = await self._prepare_training_data(training_data)
            
            # Entraînement du modèle de sévérité
            X_train, X_test, y_train, y_test = train_test_split(X, y_severity, test_size=0.2)
            self.models['severity_classifier'].fit(X_train, y_train)
            
            # Évaluation
            predictions = self.models['severity_classifier'].predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            logger.info(f"Severity model accuracy: {accuracy:.3f}")
            
            # Sauvegarde des modèles
            await self._save_models()
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            
    async def _prepare_training_data(self, incidents: List[IncidentData]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prépare les données pour l'entraînement"""
        features = []
        severities = []
        resolution_times = []
        
        for incident in incidents:
            incident_features = await self._extract_features(incident)
            features.append(incident_features.get('tfidf', []))
            severities.append(incident.severity.value)
            resolution_times.append(incident.predicted_resolution_time or 30)
            
        return np.array(features), np.array(severities), np.array(resolution_times)
        
    async def _save_models(self):
        """Sauvegarde les modèles entraînés"""
        models_path = Path(self.config.models_path)
        models_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            if hasattr(model, 'fit'):  # Modèle entraînable
                model_file = models_path / f"{model_name}.pkl"
                try:
                    async with aiofiles.open(model_file, 'wb') as f:
                        await f.write(pickle.dumps(model))
                    logger.info(f"Saved model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to save model {model_name}: {e}")

# ============================================================================
# Interface Publique
# ============================================================================

__all__ = [
    'AIAnalyzer',
    'AIConfig',
    'IncidentClassificationModel',
    'AnomalyDetectionModel'
]
