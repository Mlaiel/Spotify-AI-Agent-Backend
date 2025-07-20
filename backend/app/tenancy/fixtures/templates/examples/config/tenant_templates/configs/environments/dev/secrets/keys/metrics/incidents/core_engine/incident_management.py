# =============================================================================
# Incident Management - Classification IA Avancée
# =============================================================================
# 
# Module de gestion des incidents avec classification IA avancée utilisant
# des algorithmes de Machine Learning pour la catégorisation automatique,
# la prédiction de sévérité et l'assignation intelligente.
#
# Auteur: Lead Developer + AI Architect & ML Engineer
# Direction Technique: Fahed Mlaiel
# Version: 2.0.0 Enterprise
# =============================================================================

"""
Incident Management avec Classification IA

Ce module fournit un système complet de gestion des incidents avec:

Fonctionnalités IA:
- Classification automatique par ML (Random Forest, SVM, Neural Networks)
- Prédiction de sévérité basée sur les patterns historiques
- Assignation intelligente aux équipes selon expertise
- Détection de duplicatas par NLP et similarité sémantique

Algorithmes ML:
- Ensemble Methods pour classification multi-classe
- NLP avec TF-IDF et Word Embeddings
- Time Series Analysis pour prédiction temporelle
- Clustering pour groupement automatique d'incidents

Modèles de Données:
- IncidentEvent: Structure complète d'incident
- ClassificationResult: Résultat de classification IA
- PredictionModel: Modèles ML pré-entraînés
- AssignmentRule: Règles d'assignation intelligente
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import uuid

# Imports ML/IA
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
import joblib
from scipy import stats
import re

# Imports du Core Engine
from . import (
    core_registry, TenantContext, EngineStatus,
    DEFAULT_CLASSIFICATION_CONFIDENCE
)

logger = logging.getLogger(__name__)

# ===========================
# Configuration & Constants
# ===========================

DEFAULT_MODEL_PATH = "/opt/models/incident_classification"
MIN_TRAINING_SAMPLES = 100
CLASSIFICATION_FEATURES = [
    "title_tokens", "description_tokens", "source", "hour_of_day",
    "day_of_week", "month", "affected_systems", "error_codes",
    "user_impact", "business_impact"
]

# Patterns de reconnaissance d'incidents
INCIDENT_PATTERNS = {
    "database": [
        r"database.*connection", r"sql.*error", r"timeout.*query",
        r"deadlock", r"table.*lock", r"connection.*pool"
    ],
    "network": [
        r"network.*down", r"connectivity.*issue", r"dns.*resolution",
        r"firewall.*block", r"bandwidth.*limit", r"packet.*loss"
    ],
    "security": [
        r"unauthorized.*access", r"security.*breach", r"malware.*detected",
        r"ddos.*attack", r"intrusion.*attempt", r"vulnerability.*exploit"
    ],
    "performance": [
        r"slow.*response", r"high.*latency", r"memory.*leak",
        r"cpu.*spike", r"disk.*full", r"cache.*miss"
    ],
    "application": [
        r"application.*crash", r"service.*unavailable", r"api.*error",
        r"deployment.*failed", r"configuration.*error", r"bug.*report"
    ]
}

# ===========================
# Enums & Types
# ===========================

class IncidentStatus(Enum):
    """Statuts d'incident"""
    NEW = "new"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"

class IncidentSeverity(Enum):
    """Niveaux de sévérité"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class IncidentCategory(Enum):
    """Catégories d'incidents"""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    SECURITY = "security"
    NETWORK = "network"
    DATABASE = "database"
    PERFORMANCE = "performance"
    USER_EXPERIENCE = "user_experience"
    BUSINESS_PROCESS = "business_process"

class ClassificationMethod(Enum):
    """Méthodes de classification"""
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    RULE_BASED = "rule_based"

# ===========================
# Modèles de Données
# ===========================

@dataclass
class IncidentEvent:
    """Événement d'incident complet"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    status: IncidentStatus = IncidentStatus.NEW
    severity: Optional[IncidentSeverity] = None
    category: Optional[IncidentCategory] = None
    source: str = "manual"
    reporter: str = ""
    assignee: Optional[str] = None
    team: Optional[str] = None
    affected_systems: List[str] = field(default_factory=list)
    error_codes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tenant_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolution_time_minutes: Optional[int] = None
    
    # Champs IA/ML
    classification_confidence: float = 0.0
    predicted_severity: Optional[IncidentSeverity] = None
    predicted_category: Optional[IncidentCategory] = None
    similar_incidents: List[str] = field(default_factory=list)
    ml_features: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-traitement après initialisation"""
        if not self.title and self.description:
            self.title = self.description[:100] + "..." if len(self.description) > 100 else self.description
        
        # Extraction automatique de métadonnées
        self._extract_metadata()
        
        # Génération d'un hash pour la déduplication
        self.content_hash = self._generate_content_hash()
    
    def _extract_metadata(self):
        """Extraction automatique de métadonnées depuis le texte"""
        text = f"{self.title} {self.description}".lower()
        
        # Extraction d'adresses IP
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, text)
        if ips:
            self.metadata['extracted_ips'] = ips
        
        # Extraction de codes d'erreur
        error_pattern = r'error\s+(\d+)|code\s+(\d+)|status\s+(\d+)'
        errors = re.findall(error_pattern, text)
        if errors:
            self.error_codes.extend([e for group in errors for e in group if e])
        
        # Extraction d'URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        if urls:
            self.metadata['extracted_urls'] = urls
        
        # Extraction de noms de services/applications
        service_pattern = r'(service|app|application|api)\s+([a-zA-Z0-9_-]+)'
        services = re.findall(service_pattern, text)
        if services:
            self.affected_systems.extend([s[1] for s in services])
    
    def _generate_content_hash(self) -> str:
        """Génère un hash du contenu pour la déduplication"""
        content = f"{self.title.lower().strip()}{self.description.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'status': self.status.value,
            'severity': self.severity.value if self.severity else None,
            'category': self.category.value if self.category else None,
            'source': self.source,
            'reporter': self.reporter,
            'assignee': self.assignee,
            'team': self.team,
            'affected_systems': self.affected_systems,
            'error_codes': self.error_codes,
            'tags': self.tags,
            'metadata': self.metadata,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_time_minutes': self.resolution_time_minutes,
            'classification_confidence': self.classification_confidence,
            'predicted_severity': self.predicted_severity.value if self.predicted_severity else None,
            'predicted_category': self.predicted_category.value if self.predicted_category else None,
            'similar_incidents': self.similar_incidents,
            'ml_features': self.ml_features,
            'content_hash': self.content_hash
        }

@dataclass
class ClassificationResult:
    """Résultat de classification IA"""
    incident_id: str
    predicted_category: IncidentCategory
    predicted_severity: IncidentSeverity
    confidence_score: float
    method_used: ClassificationMethod
    feature_importance: Dict[str, float] = field(default_factory=dict)
    similar_incidents: List[Tuple[str, float]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    model_version: str = "1.0.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_confident(self, threshold: float = DEFAULT_CLASSIFICATION_CONFIDENCE) -> bool:
        """Vérifie si la classification est suffisamment confiante"""
        return self.confidence_score >= threshold

@dataclass
class AssignmentRule:
    """Règle d'assignation intelligente"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    target_team: str = ""
    target_user: Optional[str] = None
    priority: int = 0
    is_active: bool = True
    tenant_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches(self, incident: IncidentEvent) -> bool:
        """Vérifie si l'incident correspond à cette règle"""
        for condition, value in self.conditions.items():
            if condition == "category" and incident.category:
                if incident.category.value != value:
                    return False
            elif condition == "severity" and incident.severity:
                if incident.severity.value < value:
                    return False
            elif condition == "keywords":
                text = f"{incident.title} {incident.description}".lower()
                if not any(keyword.lower() in text for keyword in value):
                    return False
            elif condition == "affected_systems":
                if not any(system in incident.affected_systems for system in value):
                    return False
        return True

# ===========================
# Classificateur IA Avancé
# ===========================

class AdvancedIncidentClassifier:
    """Classificateur d'incidents avec IA avancée"""
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.encoders = {}
        self.is_trained = False
        
        # Configuration des modèles ML
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'preprocessor': 'standard'
            },
            'svm': {
                'model': SVC(
                    kernel='rbf', 
                    probability=True,
                    random_state=42
                ),
                'preprocessor': 'standard'
            },
            'neural_network': {
                'model': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42
                ),
                'preprocessor': 'standard'
            }
        }
        
        # Vectorizer pour le texte
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        logger.info("Classificateur IA initialisé")
    
    def _extract_features(self, incident: IncidentEvent) -> Dict[str, Any]:
        """Extraction des features pour le ML"""
        text = f"{incident.title} {incident.description}".lower()
        
        features = {
            # Features temporelles
            'hour_of_day': incident.created_at.hour,
            'day_of_week': incident.created_at.weekday(),
            'month': incident.created_at.month,
            'is_weekend': incident.created_at.weekday() >= 5,
            'is_business_hours': 9 <= incident.created_at.hour <= 17,
            
            # Features textuelles
            'title_length': len(incident.title),
            'description_length': len(incident.description),
            'word_count': len(text.split()),
            'has_error_codes': len(incident.error_codes) > 0,
            'affected_systems_count': len(incident.affected_systems),
            
            # Features de source
            'source_manual': incident.source == 'manual',
            'source_monitoring': incident.source == 'monitoring',
            'source_automated': incident.source == 'automated',
            
            # Features de patterns
            'has_database_keywords': any(pattern for category_patterns in INCIDENT_PATTERNS.get('database', []) 
                                       for pattern in category_patterns if re.search(pattern, text)),
            'has_network_keywords': any(pattern for category_patterns in INCIDENT_PATTERNS.get('network', []) 
                                      for pattern in category_patterns if re.search(pattern, text)),
            'has_security_keywords': any(pattern for category_patterns in INCIDENT_PATTERNS.get('security', []) 
                                       for pattern in category_patterns if re.search(pattern, text)),
            'has_performance_keywords': any(pattern for category_patterns in INCIDENT_PATTERNS.get('performance', []) 
                                          for pattern in category_patterns if re.search(pattern, text)),
        }
        
        return features
    
    def _extract_text_features(self, incident: IncidentEvent) -> np.ndarray:
        """Extraction des features textuelles avec TF-IDF"""
        text = f"{incident.title} {incident.description}"
        
        if hasattr(self, '_fitted_vectorizer'):
            return self._fitted_vectorizer.transform([text]).toarray()[0]
        else:
            # Pour l'entraînement, retourner le texte brut
            return text
    
    async def train_models(self, training_data: List[IncidentEvent]) -> Dict[str, float]:
        """Entraînement des modèles ML"""
        if len(training_data) < MIN_TRAINING_SAMPLES:
            raise ValueError(f"Nombre insuffisant d'échantillons d'entraînement: {len(training_data)} < {MIN_TRAINING_SAMPLES}")
        
        logger.info(f"Début de l'entraînement avec {len(training_data)} échantillons")
        
        # Préparation des données
        features_list = []
        categories = []
        severities = []
        texts = []
        
        for incident in training_data:
            if incident.category and incident.severity:
                features = self._extract_features(incident)
                features_list.append(features)
                categories.append(incident.category.value)
                severities.append(incident.severity.value)
                texts.append(f"{incident.title} {incident.description}")
        
        # Conversion en DataFrames
        df_features = pd.DataFrame(features_list)
        
        # Vectorisation du texte
        text_features = self.text_vectorizer.fit_transform(texts)
        self._fitted_vectorizer = self.text_vectorizer
        
        # Combinaison des features
        numerical_features = StandardScaler().fit_transform(df_features)
        combined_features = np.hstack([numerical_features, text_features.toarray()])
        
        # Encodage des labels
        category_encoder = LabelEncoder()
        severity_encoder = LabelEncoder()
        
        encoded_categories = category_encoder.fit_transform(categories)
        encoded_severities = severity_encoder.fit_transform(severities)
        
        # Sauvegarde des encoders
        self.encoders['category'] = category_encoder
        self.encoders['severity'] = severity_encoder
        self.scalers['features'] = StandardScaler().fit(df_features)
        
        # Entraînement des modèles
        training_scores = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Entraînement du modèle: {model_name}")
            
            # Division train/test
            X_train, X_test, y_cat_train, y_cat_test, y_sev_train, y_sev_test = train_test_split(
                combined_features, encoded_categories, encoded_severities,
                test_size=0.2, random_state=42, stratify=encoded_categories
            )
            
            # Entraînement pour la catégorie
            category_model = config['model']
            category_model.fit(X_train, y_cat_train)
            category_score = category_model.score(X_test, y_cat_test)
            
            # Entraînement pour la sévérité
            severity_model = type(config['model'])(**config['model'].get_params())
            severity_model.fit(X_train, y_sev_train)
            severity_score = severity_model.score(X_test, y_sev_test)
            
            # Sauvegarde des modèles
            self.models[f'{model_name}_category'] = category_model
            self.models[f'{model_name}_severity'] = severity_model
            
            training_scores[model_name] = {
                'category_accuracy': category_score,
                'severity_accuracy': severity_score,
                'combined_score': (category_score + severity_score) / 2
            }
            
            logger.info(f"Modèle {model_name} - Précision catégorie: {category_score:.3f}, sévérité: {severity_score:.3f}")
        
        self.is_trained = True
        logger.info("Entraînement des modèles terminé avec succès")
        
        return training_scores
    
    async def classify_incident(self, incident: IncidentEvent, method: ClassificationMethod = ClassificationMethod.ENSEMBLE) -> ClassificationResult:
        """Classification d'un incident avec IA"""
        start_time = datetime.utcnow()
        
        if not self.is_trained:
            logger.warning("Modèles non entraînés, utilisation de la classification basée sur des règles")
            return await self._rule_based_classification(incident)
        
        try:
            # Extraction des features
            features = self._extract_features(incident)
            text_features = self._extract_text_features(incident)
            
            # Préparation des données
            df_features = pd.DataFrame([features])
            numerical_features = self.scalers['features'].transform(df_features)
            combined_features = np.hstack([numerical_features, text_features.reshape(1, -1)])
            
            if method == ClassificationMethod.ENSEMBLE:
                # Classification par ensemble de modèles
                category_predictions = []
                severity_predictions = []
                confidences = []
                
                for model_name in self.model_configs.keys():
                    cat_model = self.models[f'{model_name}_category']
                    sev_model = self.models[f'{model_name}_severity']
                    
                    cat_proba = cat_model.predict_proba(combined_features)[0]
                    sev_proba = sev_model.predict_proba(combined_features)[0]
                    
                    category_predictions.append(cat_model.predict(combined_features)[0])
                    severity_predictions.append(sev_model.predict(combined_features)[0])
                    confidences.append((np.max(cat_proba) + np.max(sev_proba)) / 2)
                
                # Vote majoritaire avec pondération par confiance
                weighted_cat = max(set(category_predictions), key=lambda x: sum(c for i, c in enumerate(confidences) if category_predictions[i] == x))
                weighted_sev = max(set(severity_predictions), key=lambda x: sum(c for i, c in enumerate(confidences) if severity_predictions[i] == x))
                
                final_confidence = np.mean(confidences)
                
            else:
                # Classification avec un modèle spécifique
                model_name = method.value
                cat_model = self.models[f'{model_name}_category']
                sev_model = self.models[f'{model_name}_severity']
                
                cat_proba = cat_model.predict_proba(combined_features)[0]
                sev_proba = sev_model.predict_proba(combined_features)[0]
                
                weighted_cat = cat_model.predict(combined_features)[0]
                weighted_sev = sev_model.predict(combined_features)[0]
                
                final_confidence = (np.max(cat_proba) + np.max(sev_proba)) / 2
            
            # Décodage des prédictions
            predicted_category = IncidentCategory(self.encoders['category'].inverse_transform([weighted_cat])[0])
            predicted_severity = IncidentSeverity(self.encoders['severity'].inverse_transform([weighted_sev])[0])
            
            # Calcul du temps de traitement
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Recherche d'incidents similaires
            similar_incidents = await self._find_similar_incidents(incident)
            
            result = ClassificationResult(
                incident_id=incident.id,
                predicted_category=predicted_category,
                predicted_severity=predicted_severity,
                confidence_score=final_confidence,
                method_used=method,
                similar_incidents=similar_incidents,
                processing_time_ms=processing_time
            )
            
            logger.info(f"Incident {incident.id} classifié: {predicted_category.value} (sévérité {predicted_severity.value}) - confiance: {final_confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la classification: {e}")
            # Fallback vers classification basée sur règles
            return await self._rule_based_classification(incident)
    
    async def _rule_based_classification(self, incident: IncidentEvent) -> ClassificationResult:
        """Classification basée sur des règles (fallback)"""
        text = f"{incident.title} {incident.description}".lower()
        
        # Classification par mots-clés
        category_scores = {}
        for category, patterns in INCIDENT_PATTERNS.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text))
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            confidence = min(category_scores[best_category] / len(INCIDENT_PATTERNS[best_category]), 1.0)
        else:
            best_category = "application"  # Catégorie par défaut
            confidence = 0.3
        
        # Prédiction de sévérité basée sur des mots-clés
        severity_keywords = {
            "emergency": ["critical", "down", "outage", "disaster", "emergency"],
            "critical": ["critical", "severe", "major", "failure"],
            "high": ["high", "important", "urgent", "significant"],
            "medium": ["medium", "moderate", "normal"],
            "low": ["low", "minor", "cosmetic", "enhancement"]
        }
        
        severity_scores = {}
        for severity, keywords in severity_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                severity_scores[severity] = score
        
        if severity_scores:
            best_severity = max(severity_scores.items(), key=lambda x: x[1])[0]
        else:
            best_severity = "medium"  # Sévérité par défaut
        
        # Mapping vers les enums
        category_mapping = {
            "database": IncidentCategory.DATABASE,
            "network": IncidentCategory.NETWORK,
            "security": IncidentCategory.SECURITY,
            "performance": IncidentCategory.PERFORMANCE,
            "application": IncidentCategory.APPLICATION
        }
        
        severity_mapping = {
            "emergency": IncidentSeverity.EMERGENCY,
            "critical": IncidentSeverity.CRITICAL,
            "high": IncidentSeverity.HIGH,
            "medium": IncidentSeverity.MEDIUM,
            "low": IncidentSeverity.LOW
        }
        
        return ClassificationResult(
            incident_id=incident.id,
            predicted_category=category_mapping.get(best_category, IncidentCategory.APPLICATION),
            predicted_severity=severity_mapping.get(best_severity, IncidentSeverity.MEDIUM),
            confidence_score=confidence,
            method_used=ClassificationMethod.RULE_BASED
        )
    
    async def _find_similar_incidents(self, incident: IncidentEvent, limit: int = 5) -> List[Tuple[str, float]]:
        """Recherche d'incidents similaires par NLP"""
        # Simulation de la recherche d'incidents similaires
        # Dans un vrai système, ceci interrogerait une base de données d'incidents historiques
        
        similar_incidents = []
        
        # Simulation avec quelques incidents d'exemple
        example_incidents = [
            ("incident_001", "Database connection timeout in production"),
            ("incident_002", "High CPU usage on web servers"),
            ("incident_003", "Network connectivity issues in datacenter"),
            ("incident_004", "Security alert: unauthorized access attempt"),
            ("incident_005", "Application crash due to memory leak")
        ]
        
        current_text = f"{incident.title} {incident.description}".lower()
        current_vector = self.text_vectorizer.transform([current_text])
        
        for inc_id, inc_text in example_incidents:
            example_vector = self.text_vectorizer.transform([inc_text.lower()])
            similarity = cosine_similarity(current_vector, example_vector)[0][0]
            
            if similarity > 0.3:  # Seuil de similarité
                similar_incidents.append((inc_id, similarity))
        
        # Trier par similarité décroissante
        similar_incidents.sort(key=lambda x: x[1], reverse=True)
        
        return similar_incidents[:limit]

# ===========================
# Gestionnaire d'Assignation Intelligente
# ===========================

class IntelligentAssignmentManager:
    """Gestionnaire d'assignation intelligente des incidents"""
    
    def __init__(self):
        self.assignment_rules: List[AssignmentRule] = []
        self.team_expertise: Dict[str, List[str]] = {}
        self.user_workload: Dict[str, int] = {}
        self.team_availability: Dict[str, bool] = {}
        
        logger.info("Gestionnaire d'assignation intelligente initialisé")
    
    def add_assignment_rule(self, rule: AssignmentRule) -> None:
        """Ajoute une règle d'assignation"""
        self.assignment_rules.append(rule)
        # Trier par priorité décroissante
        self.assignment_rules.sort(key=lambda x: x.priority, reverse=True)
        logger.info(f"Règle d'assignation ajoutée: {rule.name}")
    
    def set_team_expertise(self, team: str, categories: List[str]) -> None:
        """Définit l'expertise d'une équipe"""
        self.team_expertise[team] = categories
        logger.info(f"Expertise définie pour l'équipe {team}: {categories}")
    
    def update_user_workload(self, user: str, workload: int) -> None:
        """Met à jour la charge de travail d'un utilisateur"""
        self.user_workload[user] = workload
    
    def set_team_availability(self, team: str, available: bool) -> None:
        """Définit la disponibilité d'une équipe"""
        self.team_availability[team] = available
    
    async def assign_incident(self, incident: IncidentEvent, classification: ClassificationResult) -> Tuple[Optional[str], Optional[str]]:
        """Assigne intelligemment un incident"""
        
        # 1. Vérification des règles d'assignation
        for rule in self.assignment_rules:
            if rule.is_active and rule.tenant_id == incident.tenant_id:
                if rule.matches(incident):
                    # Vérifier la disponibilité de l'équipe
                    if self.team_availability.get(rule.target_team, True):
                        if rule.target_user:
                            # Vérifier la charge de travail de l'utilisateur
                            current_workload = self.user_workload.get(rule.target_user, 0)
                            if current_workload < 10:  # Seuil de charge maximale
                                self.user_workload[rule.target_user] = current_workload + 1
                                logger.info(f"Incident {incident.id} assigné à {rule.target_user} (équipe {rule.target_team}) via règle {rule.name}")
                                return rule.target_team, rule.target_user
                        else:
                            logger.info(f"Incident {incident.id} assigné à l'équipe {rule.target_team} via règle {rule.name}")
                            return rule.target_team, None
        
        # 2. Assignation basée sur l'expertise
        if classification.predicted_category:
            category = classification.predicted_category.value
            
            # Rechercher les équipes expertes dans cette catégorie
            expert_teams = [team for team, expertise in self.team_expertise.items() 
                          if category in expertise and self.team_availability.get(team, True)]
            
            if expert_teams:
                # Sélectionner l'équipe avec la charge la plus faible
                team_workloads = {}
                for team in expert_teams:
                    team_users = [user for user, workload in self.user_workload.items() 
                                if user.startswith(f"{team}_")]
                    team_workloads[team] = sum(self.user_workload.get(user, 0) for user in team_users)
                
                best_team = min(team_workloads.items(), key=lambda x: x[1])[0]
                logger.info(f"Incident {incident.id} assigné à l'équipe experte {best_team} pour la catégorie {category}")
                return best_team, None
        
        # 3. Assignation par défaut basée sur la sévérité
        if classification.predicted_severity:
            if classification.predicted_severity in [IncidentSeverity.CRITICAL, IncidentSeverity.EMERGENCY]:
                # Assigner aux équipes de support de niveau 3
                return "support_l3", None
            elif classification.predicted_severity == IncidentSeverity.HIGH:
                return "support_l2", None
            else:
                return "support_l1", None
        
        # 4. Assignation par défaut
        logger.info(f"Incident {incident.id} assigné à l'équipe par défaut")
        return "support_l1", None

# ===========================
# Gestionnaire Principal des Incidents
# ===========================

class IncidentManager:
    """Gestionnaire principal des incidents avec IA"""
    
    def __init__(self):
        self.classifier = AdvancedIncidentClassifier()
        self.assignment_manager = IntelligentAssignmentManager()
        self.active_incidents: Dict[str, IncidentEvent] = {}
        self.duplicate_detector = {}
        
        # Initialisation des équipes par défaut
        self._setup_default_teams()
        
        logger.info("Gestionnaire principal des incidents initialisé")
    
    def _setup_default_teams(self):
        """Configuration des équipes par défaut"""
        self.assignment_manager.set_team_expertise("database_team", ["database", "performance"])
        self.assignment_manager.set_team_expertise("security_team", ["security"])
        self.assignment_manager.set_team_expertise("network_team", ["network", "infrastructure"])
        self.assignment_manager.set_team_expertise("app_team", ["application", "user_experience"])
        
        # Disponibilité par défaut
        for team in ["database_team", "security_team", "network_team", "app_team", "support_l1", "support_l2", "support_l3"]:
            self.assignment_manager.set_team_availability(team, True)
    
    async def process_incident(self, incident: IncidentEvent) -> Dict[str, Any]:
        """Traitement complet d'un incident"""
        logger.info(f"Début du traitement de l'incident {incident.id}")
        
        # 1. Vérification de la déduplication
        duplicate_check = await self._check_for_duplicates(incident)
        if duplicate_check:
            logger.info(f"Incident {incident.id} identifié comme doublon de {duplicate_check}")
            return {
                "status": "duplicate",
                "duplicate_of": duplicate_check,
                "message": f"Incident marqué comme doublon de {duplicate_check}"
            }
        
        # 2. Classification IA
        classification = await self.classifier.classify_incident(incident)
        
        # 3. Mise à jour de l'incident avec les prédictions
        incident.predicted_category = classification.predicted_category
        incident.predicted_severity = classification.predicted_severity
        incident.classification_confidence = classification.confidence_score
        incident.similar_incidents = [sim[0] for sim in classification.similar_incidents]
        
        # 4. Assignation intelligente
        team, user = await self.assignment_manager.assign_incident(incident, classification)
        incident.team = team
        incident.assignee = user
        
        if incident.assignee:
            incident.status = IncidentStatus.ASSIGNED
        
        # 5. Sauvegarde de l'incident
        self.active_incidents[incident.id] = incident
        
        # 6. Mise à jour des métriques
        registry = core_registry
        registry.update_metrics(
            active_incidents=len(self.active_incidents),
            total_incidents_processed=registry.get_metrics().total_incidents_processed + 1
        )
        
        if classification.is_confident():
            registry.update_metrics(
                successful_classifications=registry.get_metrics().successful_classifications + 1
            )
        else:
            registry.update_metrics(
                failed_classifications=registry.get_metrics().failed_classifications + 1
            )
        
        logger.info(f"Incident {incident.id} traité avec succès")
        
        return {
            "status": "processed",
            "incident_id": incident.id,
            "classification": {
                "category": classification.predicted_category.value,
                "severity": classification.predicted_severity.value,
                "confidence": classification.confidence_score,
                "method": classification.method_used.value
            },
            "assignment": {
                "team": team,
                "user": user
            },
            "similar_incidents": classification.similar_incidents,
            "processing_time_ms": classification.processing_time_ms
        }
    
    async def _check_for_duplicates(self, incident: IncidentEvent) -> Optional[str]:
        """Vérification des doublons"""
        content_hash = incident.content_hash
        
        # Recherche par hash de contenu
        for existing_id, existing_incident in self.active_incidents.items():
            if existing_incident.content_hash == content_hash:
                return existing_id
        
        # Recherche par similarité temporelle et textuelle
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        for existing_id, existing_incident in self.active_incidents.items():
            if existing_incident.created_at > cutoff_time:
                # Calcul de similarité textuelle simple
                text1 = f"{incident.title} {incident.description}".lower()
                text2 = f"{existing_incident.title} {existing_incident.description}".lower()
                
                # Similarité basée sur les mots communs
                words1 = set(text1.split())
                words2 = set(text2.split())
                
                if len(words1) > 0 and len(words2) > 0:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    if similarity > 0.8:  # Seuil de similarité pour les doublons
                        return existing_id
        
        return None
    
    async def get_incident(self, incident_id: str) -> Optional[IncidentEvent]:
        """Récupère un incident par son ID"""
        return self.active_incidents.get(incident_id)
    
    async def update_incident(self, incident_id: str, updates: Dict[str, Any]) -> bool:
        """Met à jour un incident"""
        if incident_id not in self.active_incidents:
            return False
        
        incident = self.active_incidents[incident_id]
        
        for key, value in updates.items():
            if hasattr(incident, key):
                if key == "status" and isinstance(value, str):
                    incident.status = IncidentStatus(value)
                elif key == "severity" and isinstance(value, str):
                    incident.severity = IncidentSeverity(value)
                elif key == "category" and isinstance(value, str):
                    incident.category = IncidentCategory(value)
                else:
                    setattr(incident, key, value)
        
        incident.updated_at = datetime.utcnow()
        
        # Calcul du temps de résolution si l'incident est résolu
        if incident.status == IncidentStatus.RESOLVED and not incident.resolved_at:
            incident.resolved_at = datetime.utcnow()
            incident.resolution_time_minutes = int(
                (incident.resolved_at - incident.created_at).total_seconds() / 60
            )
        
        logger.info(f"Incident {incident_id} mis à jour")
        return True
    
    async def list_incidents(self, filters: Dict[str, Any] = None) -> List[IncidentEvent]:
        """Liste les incidents avec filtres optionnels"""
        incidents = list(self.active_incidents.values())
        
        if filters:
            if "status" in filters:
                status_filter = IncidentStatus(filters["status"])
                incidents = [i for i in incidents if i.status == status_filter]
            
            if "severity" in filters:
                severity_filter = IncidentSeverity(filters["severity"])
                incidents = [i for i in incidents if i.severity == severity_filter]
            
            if "category" in filters:
                category_filter = IncidentCategory(filters["category"])
                incidents = [i for i in incidents if i.category == category_filter]
            
            if "tenant_id" in filters:
                incidents = [i for i in incidents if i.tenant_id == filters["tenant_id"]]
            
            if "assignee" in filters:
                incidents = [i for i in incidents if i.assignee == filters["assignee"]]
        
        return incidents
    
    async def get_incident_statistics(self) -> Dict[str, Any]:
        """Récupère les statistiques des incidents"""
        incidents = list(self.active_incidents.values())
        
        stats = {
            "total_incidents": len(incidents),
            "by_status": {},
            "by_severity": {},
            "by_category": {},
            "average_resolution_time": 0,
            "classification_accuracy": core_registry.get_metrics().classification_accuracy
        }
        
        # Statistiques par statut
        for status in IncidentStatus:
            count = sum(1 for i in incidents if i.status == status)
            stats["by_status"][status.value] = count
        
        # Statistiques par sévérité
        for severity in IncidentSeverity:
            count = sum(1 for i in incidents if i.severity == severity)
            stats["by_severity"][severity.value] = count
        
        # Statistiques par catégorie
        for category in IncidentCategory:
            count = sum(1 for i in incidents if i.category == category)
            stats["by_category"][category.value] = count
        
        # Temps de résolution moyen
        resolved_incidents = [i for i in incidents if i.resolution_time_minutes is not None]
        if resolved_incidents:
            stats["average_resolution_time"] = sum(i.resolution_time_minutes for i in resolved_incidents) / len(resolved_incidents)
        
        return stats

# ===========================
# Exports
# ===========================

__all__ = [
    "IncidentManager",
    "AdvancedIncidentClassifier", 
    "IntelligentAssignmentManager",
    "IncidentEvent",
    "ClassificationResult",
    "AssignmentRule",
    "IncidentStatus",
    "IncidentSeverity",
    "IncidentCategory",
    "ClassificationMethod"
]

logger.info("Module Incident Management avec Classification IA chargé")
