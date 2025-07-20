"""
Advanced Incident Management System for PagerDuty

Ce module fournit un système de gestion d'incidents ultra-avancé avec intelligence artificielle,
escalade prédictive, résolution automatique, et analytics complets pour PagerDuty.

Fonctionnalités principales:
- Gestion intelligente du cycle de vie des incidents
- Classification automatique par IA
- Escalade prédictive basée sur l'apprentissage automatique
- Résolution automatique avec confiance adaptative
- Analytics et reporting en temps réel
- Intégration multi-services et multi-équipes
- Gestion des SLA et métriques de performance

Version: 4.0.0
Développé par Spotify AI Agent Team
"""

import asyncio
import json
import hashlib
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import structlog
import aiofiles
import aiohttp
import pickle
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import tensorflow as tf
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

logger = structlog.get_logger(__name__)

# ============================================================================
# Enhanced Enums and Data Structures
# ============================================================================

class IncidentPriority(Enum):
    """Priorités d'incident avec scoring numérique"""
    P1_CRITICAL = (1, "Critical - Service Down")
    P2_HIGH = (2, "High - Major Feature Impact")
    P3_MEDIUM = (3, "Medium - Minor Feature Impact")
    P4_LOW = (4, "Low - Minimal Impact")
    P5_INFO = (5, "Info - No Impact")
    
    def __init__(self, level: int, description: str):
        self.level = level
        self.description = description

class IncidentCategory(Enum):
    """Catégories d'incidents avec classification intelligente"""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    DATA_QUALITY = "data_quality"
    USER_EXPERIENCE = "user_experience"
    THIRD_PARTY = "third_party"

class ResolutionMethod(Enum):
    """Méthodes de résolution d'incidents"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    ESCALATED = "escalated"
    AI_ASSISTED = "ai_assisted"
    RUNBOOK = "runbook"
    ROLLBACK = "rollback"
    CONFIGURATION_CHANGE = "configuration_change"
    RESTART = "restart"
    SCALING = "scaling"

class IncidentImpact(Enum):
    """Impact des incidents"""
    WIDESPREAD = "widespread"
    SIGNIFICANT = "significant"
    MODERATE = "moderate"
    MINIMAL = "minimal"
    NONE = "none"

class EscalationReason(Enum):
    """Raisons d'escalade"""
    SLA_BREACH = "sla_breach"
    COMPLEXITY = "complexity"
    REPEATED_FAILURE = "repeated_failure"
    CUSTOMER_IMPACT = "customer_impact"
    SECURITY_CONCERN = "security_concern"
    EXPERTISE_REQUIRED = "expertise_required"
    AUTOMATED_TRIGGER = "automated_trigger"

@dataclass
class IncidentMetrics:
    """Métriques d'incident"""
    mean_time_to_detect: float = 0.0
    mean_time_to_acknowledge: float = 0.0
    mean_time_to_resolve: float = 0.0
    mean_time_to_recovery: float = 0.0
    first_response_time: float = 0.0
    escalation_rate: float = 0.0
    auto_resolution_rate: float = 0.0
    false_positive_rate: float = 0.0
    customer_impact_score: float = 0.0
    business_impact_score: float = 0.0

@dataclass
class SLATarget:
    """Objectifs SLA pour incidents"""
    priority: IncidentPriority
    acknowledgment_time_minutes: int
    resolution_time_minutes: int
    escalation_threshold_minutes: int
    customer_communication_time_minutes: int

@dataclass
class IncidentContext:
    """Contexte enrichi d'incident"""
    incident_id: str
    title: str
    description: str
    priority: IncidentPriority
    category: IncidentCategory
    impact: IncidentImpact
    affected_services: List[str]
    affected_components: List[str]
    affected_users_count: int = 0
    business_impact_estimate: float = 0.0
    related_incidents: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    environment: str = "production"
    region: Optional[str] = None
    customer_facing: bool = True
    compliance_relevant: bool = False
    security_incident: bool = False
    
@dataclass
class ResolutionPrediction:
    """Prédiction de résolution d'incident"""
    incident_id: str
    predicted_resolution_time: float
    confidence_score: float
    suggested_actions: List[str]
    escalation_probability: float
    auto_resolution_probability: float
    similar_incidents: List[str]
    recommended_assignee: Optional[str] = None
    predicted_category: Optional[IncidentCategory] = None

@dataclass
class EscalationEvent:
    """Événement d'escalade"""
    incident_id: str
    escalation_level: int
    reason: EscalationReason
    triggered_by: str
    triggered_at: datetime
    target_team: str
    target_individual: Optional[str] = None
    automated: bool = False
    sla_breach: bool = False
    customer_impact: bool = False

@dataclass
class IncidentTimeline:
    """Timeline d'incident"""
    incident_id: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_event(self, event_type: str, description: str, actor: str, metadata: Optional[Dict] = None):
        """Ajoute un événement à la timeline"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "description": description,
            "actor": actor,
            "metadata": metadata or {}
        }
        self.events.append(event)

# ============================================================================
# Advanced AI-Powered Incident Manager
# ============================================================================

class AIIncidentManager:
    """Gestionnaire d'incidents alimenté par IA"""
    
    def __init__(self, 
                 cache_dir: str,
                 enable_ml_predictions: bool = True,
                 enable_nlp_analysis: bool = True,
                 enable_anomaly_detection: bool = True):
        
        self.cache_dir = Path(cache_dir)
        self.enable_ml_predictions = enable_ml_predictions
        self.enable_nlp_analysis = enable_nlp_analysis
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Cache des incidents
        self.incidents: Dict[str, IncidentContext] = {}
        self.incident_history: List[IncidentContext] = []
        self.timelines: Dict[str, IncidentTimeline] = {}
        self.escalations: Dict[str, List[EscalationEvent]] = {}
        
        # Modèles ML
        self.classification_model: Optional[RandomForestClassifier] = None
        self.resolution_time_model: Optional[RandomForestClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.clustering_model: Optional[DBSCAN] = None
        
        # Modèles NLP
        self.sentiment_analyzer = None
        self.text_classifier = None
        self.text_embedder = None
        
        # Preprocessing
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # SLA Configuration
        self.sla_targets = {
            IncidentPriority.P1_CRITICAL: SLATarget(
                priority=IncidentPriority.P1_CRITICAL,
                acknowledgment_time_minutes=5,
                resolution_time_minutes=60,
                escalation_threshold_minutes=15,
                customer_communication_time_minutes=10
            ),
            IncidentPriority.P2_HIGH: SLATarget(
                priority=IncidentPriority.P2_HIGH,
                acknowledgment_time_minutes=15,
                resolution_time_minutes=240,
                escalation_threshold_minutes=60,
                customer_communication_time_minutes=30
            ),
            IncidentPriority.P3_MEDIUM: SLATarget(
                priority=IncidentPriority.P3_MEDIUM,
                acknowledgment_time_minutes=60,
                resolution_time_minutes=1440,
                escalation_threshold_minutes=240,
                customer_communication_time_minutes=120
            ),
            IncidentPriority.P4_LOW: SLATarget(
                priority=IncidentPriority.P4_LOW,
                acknowledgment_time_minutes=240,
                resolution_time_minutes=4320,
                escalation_threshold_minutes=1440,
                customer_communication_time_minutes=480
            ),
            IncidentPriority.P5_INFO: SLATarget(
                priority=IncidentPriority.P5_INFO,
                acknowledgment_time_minutes=1440,
                resolution_time_minutes=10080,
                escalation_threshold_minutes=4320,
                customer_communication_time_minutes=1440
            )
        }
        
        # Métriques et statistiques
        self.metrics_history: deque = deque(maxlen=10000)
        self.prediction_accuracy: Dict[str, float] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Patterns et règles
        self.incident_patterns: Dict[str, Any] = {}
        self.auto_resolution_rules: List[Dict[str, Any]] = []
        self.escalation_rules: List[Dict[str, Any]] = []
        
        # Initialisation
        asyncio.create_task(self._initialize())
        
        logger.info("AI Incident Manager initialized")
    
    async def _initialize(self):
        """Initialisation des composants IA"""
        
        # Création des répertoires
        await self._ensure_directories()
        
        # Chargement des modèles ML
        if self.enable_ml_predictions:
            await self._load_ml_models()
        
        # Initialisation des modèles NLP
        if self.enable_nlp_analysis:
            await self._initialize_nlp_models()
        
        # Chargement de l'historique
        await self._load_incident_history()
        
        # Entraînement initial si pas de modèles
        if not self.classification_model and len(self.incident_history) > 50:
            await self._train_models()
        
        # Chargement des règles
        await self._load_rules()
        
        # Démarrage des tâches périodiques
        asyncio.create_task(self._periodic_model_retraining())
        asyncio.create_task(self._periodic_performance_evaluation())
        asyncio.create_task(self._periodic_pattern_analysis())
        
        logger.info("AI components initialization completed")
    
    async def _ensure_directories(self):
        """Assure que les répertoires nécessaires existent"""
        
        directories = [
            self.cache_dir / "models",
            self.cache_dir / "incidents",
            self.cache_dir / "timelines",
            self.cache_dir / "escalations",
            self.cache_dir / "predictions",
            self.cache_dir / "patterns",
            self.cache_dir / "rules",
            self.cache_dir / "metrics",
            self.cache_dir / "nlp_cache"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def _load_ml_models(self):
        """Charge les modèles ML depuis le disque"""
        
        models_dir = self.cache_dir / "models"
        
        try:
            # Modèle de classification
            classification_path = models_dir / "incident_classifier.joblib"
            if classification_path.exists():
                self.classification_model = joblib.load(classification_path)
                logger.info("Classification model loaded")
            
            # Modèle de prédiction temps de résolution
            resolution_path = models_dir / "resolution_time_predictor.joblib"
            if resolution_path.exists():
                self.resolution_time_model = joblib.load(resolution_path)
                logger.info("Resolution time model loaded")
            
            # Détecteur d'anomalies
            anomaly_path = models_dir / "anomaly_detector.joblib"
            if anomaly_path.exists():
                self.anomaly_detector = joblib.load(anomaly_path)
                logger.info("Anomaly detector loaded")
            
            # Chargement des encoders et scalers
            encoders_path = models_dir / "label_encoders.joblib"
            if encoders_path.exists():
                self.label_encoders = joblib.load(encoders_path)
            
            scalers_path = models_dir / "scalers.joblib"
            if scalers_path.exists():
                self.scalers = joblib.load(scalers_path)
                
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
    
    async def _initialize_nlp_models(self):
        """Initialise les modèles NLP"""
        
        try:
            # Analyseur de sentiment
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Classificateur de texte pour incidents
            self.text_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Modèle d'embedding pour similarité
            self.text_embedder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            logger.info("NLP models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
    
    async def analyze_incident(self, incident: IncidentContext) -> ResolutionPrediction:
        """Analyse complète d'un incident avec IA"""
        
        logger.info(f"Starting AI analysis for incident: {incident.incident_id}")
        
        # Analyse NLP du titre et description
        nlp_insights = await self._analyze_incident_text(incident)
        
        # Classification automatique
        predicted_category = await self._predict_incident_category(incident)
        
        # Prédiction temps de résolution
        resolution_time = await self._predict_resolution_time(incident)
        
        # Calcul de la probabilité d'escalade
        escalation_prob = await self._calculate_escalation_probability(incident)
        
        # Probabilité de résolution automatique
        auto_resolution_prob = await self._calculate_auto_resolution_probability(incident)
        
        # Recherche d'incidents similaires
        similar_incidents = await self._find_similar_incidents(incident)
        
        # Recommandation d'assignation
        recommended_assignee = await self._recommend_assignee(incident)
        
        # Actions suggérées
        suggested_actions = await self._generate_suggested_actions(incident, nlp_insights)
        
        # Calcul du score de confiance
        confidence_score = await self._calculate_confidence_score(incident, nlp_insights)
        
        prediction = ResolutionPrediction(
            incident_id=incident.incident_id,
            predicted_resolution_time=resolution_time,
            confidence_score=confidence_score,
            suggested_actions=suggested_actions,
            escalation_probability=escalation_prob,
            auto_resolution_probability=auto_resolution_prob,
            similar_incidents=similar_incidents,
            recommended_assignee=recommended_assignee,
            predicted_category=predicted_category
        )
        
        # Sauvegarde de la prédiction
        await self._save_prediction(prediction)
        
        logger.info(f"AI analysis completed for incident: {incident.incident_id}")
        return prediction
    
    async def _analyze_incident_text(self, incident: IncidentContext) -> Dict[str, Any]:
        """Analyse NLP du texte de l'incident"""
        
        if not self.enable_nlp_analysis:
            return {}
        
        text = f"{incident.title} {incident.description}"
        insights = {}
        
        try:
            # Analyse de sentiment
            if self.sentiment_analyzer:
                sentiment = self.sentiment_analyzer(text[:512])  # Limite pour le modèle
                insights["sentiment"] = sentiment[0] if sentiment else None
            
            # Classification zero-shot pour urgence
            if self.text_classifier:
                urgency_labels = ["critical", "high", "medium", "low", "informational"]
                urgency_result = self.text_classifier(text[:512], urgency_labels)
                insights["urgency_classification"] = urgency_result
            
            # Classification par composant
            component_labels = [
                "database", "network", "api", "frontend", "backend", 
                "infrastructure", "security", "performance", "monitoring"
            ]
            component_result = self.text_classifier(text[:512], component_labels)
            insights["component_classification"] = component_result
            
            # Extraction de mots-clés
            keywords = await self._extract_keywords(text)
            insights["keywords"] = keywords
            
            # Détection d'entités
            entities = await self._extract_entities(text)
            insights["entities"] = entities
            
        except Exception as e:
            logger.error(f"NLP analysis failed: {e}")
        
        return insights
    
    async def _predict_incident_category(self, incident: IncidentContext) -> Optional[IncidentCategory]:
        """Prédit la catégorie d'un incident"""
        
        if not self.classification_model:
            return None
        
        try:
            # Préparation des features
            features = await self._extract_features(incident)
            
            # Prédiction
            category_encoded = self.classification_model.predict([features])[0]
            
            # Décodage
            if "category" in self.label_encoders:
                category_name = self.label_encoders["category"].inverse_transform([category_encoded])[0]
                return IncidentCategory(category_name)
                
        except Exception as e:
            logger.error(f"Category prediction failed: {e}")
        
        return None
    
    async def _predict_resolution_time(self, incident: IncidentContext) -> float:
        """Prédit le temps de résolution d'un incident"""
        
        if not self.resolution_time_model:
            # Estimation basée sur la priorité si pas de modèle
            sla = self.sla_targets.get(incident.priority)
            return sla.resolution_time_minutes * 60 if sla else 3600  # en secondes
        
        try:
            # Préparation des features
            features = await self._extract_features(incident)
            
            # Prédiction
            resolution_time = self.resolution_time_model.predict([features])[0]
            
            # Validation et ajustement
            min_time = 300  # 5 minutes minimum
            max_time = 86400  # 24 heures maximum
            
            return max(min_time, min(resolution_time, max_time))
            
        except Exception as e:
            logger.error(f"Resolution time prediction failed: {e}")
            
            # Fallback basé sur SLA
            sla = self.sla_targets.get(incident.priority)
            return sla.resolution_time_minutes * 60 if sla else 3600
    
    async def _calculate_escalation_probability(self, incident: IncidentContext) -> float:
        """Calcule la probabilité d'escalade"""
        
        # Facteurs d'escalade
        escalation_score = 0.0
        
        # Priorité
        priority_scores = {
            IncidentPriority.P1_CRITICAL: 0.8,
            IncidentPriority.P2_HIGH: 0.6,
            IncidentPriority.P3_MEDIUM: 0.3,
            IncidentPriority.P4_LOW: 0.1,
            IncidentPriority.P5_INFO: 0.05
        }
        escalation_score += priority_scores.get(incident.priority, 0.3)
        
        # Impact client
        if incident.customer_facing:
            escalation_score += 0.2
        
        if incident.affected_users_count > 1000:
            escalation_score += 0.3
        elif incident.affected_users_count > 100:
            escalation_score += 0.1
        
        # Composants critiques
        critical_components = ["database", "payment", "authentication", "core-api"]
        if any(comp in critical_components for comp in incident.affected_components):
            escalation_score += 0.2
        
        # Historique d'incidents similaires
        similar_escalated = await self._count_similar_escalated_incidents(incident)
        if similar_escalated > 2:
            escalation_score += 0.3
        
        # Impact business
        if incident.business_impact_estimate > 10000:
            escalation_score += 0.2
        
        # Conformité et sécurité
        if incident.compliance_relevant or incident.security_incident:
            escalation_score += 0.3
        
        return min(1.0, escalation_score)
    
    async def _calculate_auto_resolution_probability(self, incident: IncidentContext) -> float:
        """Calcule la probabilité de résolution automatique"""
        
        auto_resolution_score = 0.0
        
        # Catégories auto-résolvables
        auto_resolvable_categories = [
            IncidentCategory.PERFORMANCE,
            IncidentCategory.AVAILABILITY,
            IncidentCategory.INFRASTRUCTURE
        ]
        if incident.category in auto_resolvable_categories:
            auto_resolution_score += 0.4
        
        # Historique de résolution automatique
        similar_auto_resolved = await self._count_similar_auto_resolved_incidents(incident)
        if similar_auto_resolved > 3:
            auto_resolution_score += 0.5
        
        # Services avec auto-scaling
        auto_scaling_services = ["web-app", "api-gateway", "microservice"]
        if any(service in auto_scaling_services for service in incident.affected_services):
            auto_resolution_score += 0.3
        
        # Environnement (non-prod plus facilement auto-résolvable)
        if incident.environment != "production":
            auto_resolution_score += 0.2
        
        # Priorité faible
        if incident.priority in [IncidentPriority.P4_LOW, IncidentPriority.P5_INFO]:
            auto_resolution_score += 0.2
        
        return min(1.0, auto_resolution_score)
    
    async def _find_similar_incidents(self, incident: IncidentContext) -> List[str]:
        """Trouve des incidents similaires"""
        
        similar_incidents = []
        
        try:
            # Recherche par mots-clés et composants
            for past_incident in self.incident_history[-1000:]:  # Derniers 1000 incidents
                similarity_score = 0.0
                
                # Similarité de titre
                title_similarity = await self._calculate_text_similarity(
                    incident.title, past_incident.title
                )
                similarity_score += title_similarity * 0.4
                
                # Composants affectés
                common_components = set(incident.affected_components) & set(past_incident.affected_components)
                if common_components:
                    similarity_score += len(common_components) / max(
                        len(incident.affected_components), len(past_incident.affected_components)
                    ) * 0.3
                
                # Services affectés
                common_services = set(incident.affected_services) & set(past_incident.affected_services)
                if common_services:
                    similarity_score += len(common_services) / max(
                        len(incident.affected_services), len(past_incident.affected_services)
                    ) * 0.2
                
                # Catégorie et priorité
                if incident.category == past_incident.category:
                    similarity_score += 0.1
                
                if similarity_score > 0.6:
                    similar_incidents.append(past_incident.incident_id)
                    
        except Exception as e:
            logger.error(f"Similar incidents search failed: {e}")
        
        return similar_incidents[:5]  # Top 5
    
    async def _recommend_assignee(self, incident: IncidentContext) -> Optional[str]:
        """Recommande un assignataire pour l'incident"""
        
        # Logique de recommandation basée sur l'expertise
        assignee_expertise = {
            IncidentCategory.DATABASE: ["dba_team", "backend_team"],
            IncidentCategory.NETWORK: ["devops_team", "infrastructure_team"],
            IncidentCategory.SECURITY: ["security_team", "devops_team"],
            IncidentCategory.APPLICATION: ["backend_team", "frontend_team"],
            IncidentCategory.PERFORMANCE: ["devops_team", "backend_team"]
        }
        
        # Historique d'assignation réussie
        successful_assignees = await self._get_successful_assignees(incident.category)
        
        # Disponibilité actuelle (simulée)
        available_teams = ["backend_team", "devops_team", "frontend_team"]
        
        # Sélection intelligente
        recommended_teams = assignee_expertise.get(incident.category, ["backend_team"])
        
        for team in recommended_teams:
            if team in available_teams and team in successful_assignees:
                return team
        
        return recommended_teams[0] if recommended_teams else "backend_team"
    
    async def _generate_suggested_actions(self, 
                                        incident: IncidentContext, 
                                        nlp_insights: Dict[str, Any]) -> List[str]:
        """Génère des actions suggérées pour l'incident"""
        
        actions = []
        
        # Actions basées sur la catégorie
        category_actions = {
            IncidentCategory.DATABASE: [
                "Check database connections and pool status",
                "Review recent database migrations",
                "Analyze slow query logs",
                "Check disk space on database servers"
            ],
            IncidentCategory.NETWORK: [
                "Check network connectivity between services",
                "Review firewall rules and security groups",
                "Analyze network latency metrics",
                "Verify DNS resolution"
            ],
            IncidentCategory.APPLICATION: [
                "Check application logs for errors",
                "Review recent deployments",
                "Analyze application performance metrics",
                "Verify configuration settings"
            ],
            IncidentCategory.PERFORMANCE: [
                "Check CPU and memory utilization",
                "Analyze response time metrics",
                "Review caching effectiveness",
                "Check for resource bottlenecks"
            ]
        }
        
        actions.extend(category_actions.get(incident.category, []))
        
        # Actions basées sur la priorité
        if incident.priority == IncidentPriority.P1_CRITICAL:
            actions.insert(0, "Activate incident response team immediately")
            actions.insert(1, "Notify stakeholders and customers")
        
        # Actions basées sur les composants affectés
        if "payment" in incident.affected_components:
            actions.append("Check payment gateway status and transactions")
        
        if "authentication" in incident.affected_components:
            actions.append("Verify authentication service and user sessions")
        
        # Actions basées sur l'analyse NLP
        keywords = nlp_insights.get("keywords", [])
        if "timeout" in keywords:
            actions.append("Increase timeout values and check connection pools")
        
        if "memory" in keywords or "oom" in keywords:
            actions.append("Check memory usage and potential memory leaks")
        
        if "ssl" in keywords or "certificate" in keywords:
            actions.append("Verify SSL certificates and expiration dates")
        
        # Actions automatiques suggérées
        if incident.priority in [IncidentPriority.P4_LOW, IncidentPriority.P5_INFO]:
            actions.append("Consider auto-resolution if pattern matches known issues")
        
        return actions[:8]  # Limite à 8 actions
    
    async def _calculate_confidence_score(self, 
                                        incident: IncidentContext, 
                                        nlp_insights: Dict[str, Any]) -> float:
        """Calcule le score de confiance de l'analyse"""
        
        confidence = 0.5  # Base confidence
        
        # Disponibilité des données
        if incident.description and len(incident.description) > 50:
            confidence += 0.1
        
        if incident.affected_components:
            confidence += 0.1
        
        if incident.affected_services:
            confidence += 0.1
        
        # Qualité de l'analyse NLP
        if nlp_insights.get("sentiment"):
            confidence += 0.05
        
        if nlp_insights.get("keywords"):
            confidence += 0.05
        
        # Historique d'incidents similaires
        if len(self.incident_history) > 100:
            confidence += 0.1
        
        # Performance du modèle
        model_accuracy = self.prediction_accuracy.get("classification", 0.7)
        confidence += (model_accuracy - 0.5) * 0.2
        
        return min(1.0, confidence)
    
    async def track_incident_timeline(self, incident_id: str, event_type: str, 
                                   description: str, actor: str, metadata: Optional[Dict] = None):
        """Suit la timeline d'un incident"""
        
        if incident_id not in self.timelines:
            self.timelines[incident_id] = IncidentTimeline(incident_id)
        
        self.timelines[incident_id].add_event(event_type, description, actor, metadata)
        
        # Sauvegarde
        await self._save_timeline(self.timelines[incident_id])
        
        # Vérification SLA
        await self._check_sla_compliance(incident_id, event_type)
        
        logger.info(f"Timeline updated for incident {incident_id}: {event_type}")
    
    async def trigger_escalation(self, incident_id: str, reason: EscalationReason, 
                               triggered_by: str, target_team: str) -> EscalationEvent:
        """Déclenche une escalade d'incident"""
        
        # Calcul du niveau d'escalade
        current_escalations = self.escalations.get(incident_id, [])
        escalation_level = len(current_escalations) + 1
        
        # Création de l'événement d'escalade
        escalation = EscalationEvent(
            incident_id=incident_id,
            escalation_level=escalation_level,
            reason=reason,
            triggered_by=triggered_by,
            triggered_at=datetime.now(),
            target_team=target_team,
            automated=(triggered_by == "AI_SYSTEM"),
            sla_breach=(reason == EscalationReason.SLA_BREACH)
        )
        
        # Stockage
        if incident_id not in self.escalations:
            self.escalations[incident_id] = []
        self.escalations[incident_id].append(escalation)
        
        # Mise à jour de la timeline
        await self.track_incident_timeline(
            incident_id, 
            "escalation",
            f"Incident escalated to level {escalation_level} - {reason.value}",
            triggered_by,
            {"escalation_level": escalation_level, "target_team": target_team}
        )
        
        # Sauvegarde
        await self._save_escalation(escalation)
        
        logger.warning(f"Incident {incident_id} escalated to level {escalation_level}")
        return escalation
    
    async def evaluate_auto_resolution(self, incident_id: str) -> Dict[str, Any]:
        """Évalue la possibilité de résolution automatique"""
        
        if incident_id not in self.incidents:
            return {"can_auto_resolve": False, "reason": "Incident not found"}
        
        incident = self.incidents[incident_id]
        
        # Règles de résolution automatique
        auto_resolution_checks = []
        
        # Vérification des symptômes résolus
        if await self._check_symptoms_resolved(incident):
            auto_resolution_checks.append("symptoms_resolved")
        
        # Vérification de la stabilité
        if await self._check_system_stability(incident):
            auto_resolution_checks.append("system_stable")
        
        # Vérification du temps écoulé
        if await self._check_minimum_observation_time(incident):
            auto_resolution_checks.append("observation_time_met")
        
        # Vérification de l'absence de nouveaux symptômes
        if await self._check_no_new_symptoms(incident):
            auto_resolution_checks.append("no_new_symptoms")
        
        # Décision d'auto-résolution
        can_auto_resolve = len(auto_resolution_checks) >= 3
        
        result = {
            "can_auto_resolve": can_auto_resolve,
            "checks_passed": auto_resolution_checks,
            "confidence": len(auto_resolution_checks) / 4,
            "recommendation": "auto_resolve" if can_auto_resolve else "manual_review"
        }
        
        logger.info(f"Auto-resolution evaluation for {incident_id}: {result}")
        return result
    
    async def generate_incident_report(self, incident_id: str) -> Dict[str, Any]:
        """Génère un rapport d'incident complet"""
        
        if incident_id not in self.incidents:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident = self.incidents[incident_id]
        timeline = self.timelines.get(incident_id, IncidentTimeline(incident_id))
        escalations = self.escalations.get(incident_id, [])
        
        # Calcul des métriques
        metrics = await self._calculate_incident_metrics(incident, timeline)
        
        # Analyse post-mortem
        postmortem = await self._generate_postmortem_analysis(incident, timeline)
        
        # Leçons apprises
        lessons_learned = await self._extract_lessons_learned(incident, timeline)
        
        # Recommandations
        recommendations = await self._generate_recommendations(incident, timeline)
        
        report = {
            "incident": asdict(incident),
            "timeline": {
                "events": timeline.events,
                "duration": len(timeline.events)
            },
            "escalations": [asdict(e) for e in escalations],
            "metrics": asdict(metrics),
            "postmortem": postmortem,
            "lessons_learned": lessons_learned,
            "recommendations": recommendations,
            "report_generated_at": datetime.now().isoformat(),
            "report_generated_by": "AI_Incident_Manager"
        }
        
        # Sauvegarde du rapport
        await self._save_incident_report(incident_id, report)
        
        return report
    
    async def _extract_features(self, incident: IncidentContext) -> List[float]:
        """Extrait les features pour les modèles ML"""
        
        features = []
        
        # Features numériques
        features.append(incident.priority.level)
        features.append(len(incident.affected_services))
        features.append(len(incident.affected_components))
        features.append(incident.affected_users_count)
        features.append(incident.business_impact_estimate)
        
        # Features booléennes
        features.append(1.0 if incident.customer_facing else 0.0)
        features.append(1.0 if incident.compliance_relevant else 0.0)
        features.append(1.0 if incident.security_incident else 0.0)
        
        # Features catégorielles encodées
        if "category" in self.label_encoders:
            category_encoded = self.label_encoders["category"].transform([incident.category.value])[0]
            features.append(float(category_encoded))
        else:
            features.append(0.0)
        
        if "environment" in self.label_encoders:
            env_encoded = self.label_encoders["environment"].transform([incident.environment])[0]
            features.append(float(env_encoded))
        else:
            features.append(0.0)
        
        # Features temporelles
        current_hour = datetime.now().hour
        features.append(float(current_hour))
        
        current_day = datetime.now().weekday()
        features.append(float(current_day))
        
        return features
    
    async def _train_models(self):
        """Entraîne les modèles ML"""
        
        if len(self.incident_history) < 50:
            logger.warning("Insufficient data for model training")
            return
        
        logger.info("Starting model training...")
        
        try:
            # Préparation des données
            X = []
            y_category = []
            y_resolution_time = []
            
            for incident in self.incident_history:
                features = await self._extract_features(incident)
                X.append(features)
                y_category.append(incident.category.value)
                
                # Simulation du temps de résolution (à remplacer par les vraies données)
                resolution_time = self.sla_targets[incident.priority].resolution_time_minutes * 60
                y_resolution_time.append(resolution_time)
            
            X = np.array(X)
            
            # Préparation des encoders
            if "category" not in self.label_encoders:
                self.label_encoders["category"] = LabelEncoder()
                y_category_encoded = self.label_encoders["category"].fit_transform(y_category)
            else:
                y_category_encoded = self.label_encoders["category"].transform(y_category)
            
            # Division train/test
            X_train, X_test, y_cat_train, y_cat_test, y_res_train, y_res_test = train_test_split(
                X, y_category_encoded, y_resolution_time, test_size=0.2, random_state=42
            )
            
            # Entraînement du classificateur de catégorie
            self.classification_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            )
            self.classification_model.fit(X_train, y_cat_train)
            
            # Évaluation
            y_pred = self.classification_model.predict(X_test)
            accuracy = accuracy_score(y_cat_test, y_pred)
            self.prediction_accuracy["classification"] = accuracy
            
            # Entraînement du prédicteur de temps de résolution
            self.resolution_time_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
            
            # Discrétisation du temps de résolution en classes
            y_res_classes = pd.cut(y_res_train, bins=5, labels=False)
            self.resolution_time_model.fit(X_train, y_res_classes)
            
            # Entraînement du détecteur d'anomalies
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.anomaly_detector.fit(X_train)
            
            # Sauvegarde des modèles
            await self._save_models()
            
            logger.info(f"Model training completed. Classification accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    async def _save_models(self):
        """Sauvegarde les modèles ML"""
        
        models_dir = self.cache_dir / "models"
        
        try:
            if self.classification_model:
                joblib.dump(self.classification_model, models_dir / "incident_classifier.joblib")
            
            if self.resolution_time_model:
                joblib.dump(self.resolution_time_model, models_dir / "resolution_time_predictor.joblib")
            
            if self.anomaly_detector:
                joblib.dump(self.anomaly_detector, models_dir / "anomaly_detector.joblib")
            
            if self.label_encoders:
                joblib.dump(self.label_encoders, models_dir / "label_encoders.joblib")
            
            if self.scalers:
                joblib.dump(self.scalers, models_dir / "scalers.joblib")
                
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    async def _periodic_model_retraining(self):
        """Réentraînement périodique des modèles"""
        
        while True:
            try:
                await asyncio.sleep(86400)  # 24 heures
                
                # Réentraînement si suffisamment de nouvelles données
                new_incidents_count = len(self.incident_history) - len(self.incidents)
                
                if new_incidents_count > 50:
                    await self._train_models()
                    logger.info("Periodic model retraining completed")
                
            except Exception as e:
                logger.error(f"Periodic retraining error: {e}")
    
    async def _periodic_performance_evaluation(self):
        """Évaluation périodique des performances"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # 1 heure
                
                # Évaluation de la précision des prédictions
                await self._evaluate_prediction_accuracy()
                
                # Analyse des patterns d'escalade
                await self._analyze_escalation_patterns()
                
                # Mise à jour des métriques
                await self._update_performance_metrics()
                
            except Exception as e:
                logger.error(f"Performance evaluation error: {e}")
    
    async def _periodic_pattern_analysis(self):
        """Analyse périodique des patterns"""
        
        while True:
            try:
                await asyncio.sleep(7200)  # 2 heures
                
                # Détection de nouveaux patterns
                await self._detect_incident_patterns()
                
                # Mise à jour des règles d'auto-résolution
                await self._update_auto_resolution_rules()
                
                # Optimisation des règles d'escalade
                await self._optimize_escalation_rules()
                
            except Exception as e:
                logger.error(f"Pattern analysis error: {e}")
    
    # Méthodes utilitaires (implémentations simplifiées)
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extraction de mots-clés du texte"""
        # Implémentation simplifiée
        words = text.lower().split()
        tech_keywords = [
            "database", "api", "network", "server", "timeout", "error",
            "memory", "cpu", "disk", "ssl", "authentication", "payment"
        ]
        return [word for word in words if word in tech_keywords]
    
    async def _extract_entities(self, text: str) -> List[str]:
        """Extraction d'entités du texte"""
        # Implémentation simplifiée
        return []
    
    async def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité entre deux textes"""
        # Implémentation simplifiée basée sur les mots communs
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        return len(words1.intersection(words2)) / len(words1.union(words2))
    
    async def _count_similar_escalated_incidents(self, incident: IncidentContext) -> int:
        """Compte les incidents similaires qui ont été escaladés"""
        # Implémentation simplifiée
        return 1  # Placeholder
    
    async def _count_similar_auto_resolved_incidents(self, incident: IncidentContext) -> int:
        """Compte les incidents similaires auto-résolus"""
        # Implémentation simplifiée
        return 2  # Placeholder
    
    async def _get_successful_assignees(self, category: IncidentCategory) -> List[str]:
        """Obtient les assignataires ayant réussi pour cette catégorie"""
        return ["backend_team", "devops_team"]  # Placeholder
    
    async def _check_symptoms_resolved(self, incident: IncidentContext) -> bool:
        """Vérifie si les symptômes sont résolus"""
        return True  # Placeholder
    
    async def _check_system_stability(self, incident: IncidentContext) -> bool:
        """Vérifie la stabilité du système"""
        return True  # Placeholder
    
    async def _check_minimum_observation_time(self, incident: IncidentContext) -> bool:
        """Vérifie le temps d'observation minimum"""
        return True  # Placeholder
    
    async def _check_no_new_symptoms(self, incident: IncidentContext) -> bool:
        """Vérifie l'absence de nouveaux symptômes"""
        return True  # Placeholder
    
    async def _check_sla_compliance(self, incident_id: str, event_type: str):
        """Vérifie la conformité SLA"""
        # Implémentation de vérification SLA
        pass
    
    async def _calculate_incident_metrics(self, incident: IncidentContext, timeline: IncidentTimeline) -> IncidentMetrics:
        """Calcule les métriques d'incident"""
        return IncidentMetrics()  # Placeholder
    
    async def _generate_postmortem_analysis(self, incident: IncidentContext, timeline: IncidentTimeline) -> Dict[str, Any]:
        """Génère l'analyse post-mortem"""
        return {"summary": "Post-mortem analysis"}  # Placeholder
    
    async def _extract_lessons_learned(self, incident: IncidentContext, timeline: IncidentTimeline) -> List[str]:
        """Extrait les leçons apprises"""
        return ["Lesson 1", "Lesson 2"]  # Placeholder
    
    async def _generate_recommendations(self, incident: IncidentContext, timeline: IncidentTimeline) -> List[str]:
        """Génère des recommandations"""
        return ["Recommendation 1", "Recommendation 2"]  # Placeholder
    
    # Méthodes de persistance
    
    async def _load_incident_history(self):
        """Charge l'historique des incidents"""
        history_file = self.cache_dir / "incidents" / "history.json"
        if history_file.exists():
            try:
                async with aiofiles.open(history_file, 'r') as f:
                    data = json.loads(await f.read())
                # Reconstruction des objets
                # Implémentation simplifiée
            except Exception as e:
                logger.error(f"Failed to load incident history: {e}")
    
    async def _save_prediction(self, prediction: ResolutionPrediction):
        """Sauvegarde une prédiction"""
        prediction_file = self.cache_dir / "predictions" / f"{prediction.incident_id}.json"
        try:
            async with aiofiles.open(prediction_file, 'w') as f:
                await f.write(json.dumps(asdict(prediction), indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
    
    async def _save_timeline(self, timeline: IncidentTimeline):
        """Sauvegarde une timeline"""
        timeline_file = self.cache_dir / "timelines" / f"{timeline.incident_id}.json"
        try:
            async with aiofiles.open(timeline_file, 'w') as f:
                await f.write(json.dumps(asdict(timeline), indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save timeline: {e}")
    
    async def _save_escalation(self, escalation: EscalationEvent):
        """Sauvegarde un événement d'escalade"""
        escalation_file = self.cache_dir / "escalations" / f"{escalation.incident_id}_{escalation.escalation_level}.json"
        try:
            async with aiofiles.open(escalation_file, 'w') as f:
                await f.write(json.dumps(asdict(escalation), indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save escalation: {e}")
    
    async def _save_incident_report(self, incident_id: str, report: Dict[str, Any]):
        """Sauvegarde un rapport d'incident"""
        report_file = self.cache_dir / "reports" / f"{incident_id}_report.json"
        try:
            async with aiofiles.open(report_file, 'w') as f:
                await f.write(json.dumps(report, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save incident report: {e}")
    
    async def _load_rules(self):
        """Charge les règles d'auto-résolution et d'escalade"""
        # Implémentation du chargement des règles
        pass
    
    async def _evaluate_prediction_accuracy(self):
        """Évalue la précision des prédictions"""
        # Implémentation de l'évaluation
        pass
    
    async def _analyze_escalation_patterns(self):
        """Analyse les patterns d'escalade"""
        # Implémentation de l'analyse
        pass
    
    async def _update_performance_metrics(self):
        """Met à jour les métriques de performance"""
        # Implémentation de mise à jour
        pass
    
    async def _detect_incident_patterns(self):
        """Détecte de nouveaux patterns d'incidents"""
        # Implémentation de détection
        pass
    
    async def _update_auto_resolution_rules(self):
        """Met à jour les règles d'auto-résolution"""
        # Implémentation de mise à jour
        pass
    
    async def _optimize_escalation_rules(self):
        """Optimise les règles d'escalade"""
        # Implémentation d'optimisation
        pass
    
    async def get_incident_analytics(self) -> Dict[str, Any]:
        """Obtient les analytics des incidents"""
        
        analytics = {
            "total_incidents": len(self.incidents),
            "incidents_by_priority": {},
            "incidents_by_category": {},
            "resolution_time_trends": {},
            "escalation_rates": {},
            "auto_resolution_rates": {},
            "model_performance": self.model_performance,
            "prediction_accuracy": self.prediction_accuracy
        }
        
        return analytics
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        
        # Sauvegarde finale
        await self._save_models()
        
        # Nettoyage des caches
        self.incidents.clear()
        self.timelines.clear()
        self.escalations.clear()
        
        logger.info("AI Incident Manager cleaned up")

# Export des classes principales
__all__ = [
    "AIIncidentManager",
    "IncidentContext",
    "ResolutionPrediction",
    "EscalationEvent",
    "IncidentTimeline",
    "IncidentPriority",
    "IncidentCategory",
    "ResolutionMethod",
    "IncidentImpact",
    "EscalationReason",
    "IncidentMetrics",
    "SLATarget"
]
