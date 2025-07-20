"""
🔮 Advanced Personalization Engine - Production-Ready System
============================================================

Moteur de personnalisation ultra-avancé avec ML, adaptation comportementale,
recommandations intelligentes et optimisation d'engagement.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from transformers import pipeline

logger = logging.getLogger(__name__)


class PersonalizationType(Enum):
    """Types de personnalisation"""
    CONTENT_PREFERENCE = "content_preference"
    COMMUNICATION_STYLE = "communication_style"
    FREQUENCY_PREFERENCE = "frequency_preference"
    CHANNEL_PREFERENCE = "channel_preference"
    TIMING_OPTIMIZATION = "timing_optimization"
    CULTURAL_ADAPTATION = "cultural_adaptation"
    EMOTIONAL_TONE = "emotional_tone"
    COMPLEXITY_LEVEL = "complexity_level"


class UserSegment(Enum):
    """Segments d'utilisateurs"""
    POWER_USER = "power_user"
    CASUAL_USER = "casual_user"
    NEW_USER = "new_user"
    ENTERPRISE_USER = "enterprise_user"
    DEVELOPER = "developer"
    CONTENT_CREATOR = "content_creator"
    ANALYST = "analyst"
    ADMINISTRATOR = "administrator"


class PersonalizationGoal(Enum):
    """Objectifs de personnalisation"""
    INCREASE_ENGAGEMENT = "increase_engagement"
    IMPROVE_SATISFACTION = "improve_satisfaction"
    REDUCE_CHURN = "reduce_churn"
    ENHANCE_PRODUCTIVITY = "enhance_productivity"
    BOOST_ADOPTION = "boost_adoption"
    OPTIMIZE_CONVERSION = "optimize_conversion"


@dataclass
class UserProfile:
    """Profil utilisateur pour personnalisation"""
    user_id: str
    tenant_id: str
    
    # Données démographiques
    language: str = "en"
    country: str = "US"
    timezone: str = "UTC"
    locale: str = "en_US"
    
    # Préférences comportementales
    communication_style: str = "professional"  # casual, professional, technical
    preferred_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    activity_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Contexte métier
    role: str = "user"
    department: str = "unknown"
    experience_level: str = "intermediate"  # beginner, intermediate, advanced, expert
    
    # Préférences de contenu
    content_preferences: Dict[str, float] = field(default_factory=dict)
    topic_interests: List[str] = field(default_factory=list)
    complexity_preference: float = 0.5  # 0.0 = simple, 1.0 = complexe
    
    # Historique d'engagement
    engagement_history: List[Dict[str, Any]] = field(default_factory=list)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Métadonnées
    segment: UserSegment = UserSegment.CASUAL_USER
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Scores ML
    engagement_score: float = 0.5
    churn_risk: float = 0.0
    satisfaction_score: float = 0.7
    activity_score: float = 0.5


@dataclass
class PersonalizationContext:
    """Contexte pour la personnalisation"""
    user_profile: UserProfile
    content_type: str = "notification"
    channel: str = "slack"
    priority: str = "medium"
    
    # Contexte temporel
    current_time: datetime = field(default_factory=datetime.utcnow)
    user_local_time: Optional[datetime] = None
    
    # Contexte situationnel
    user_activity_status: str = "active"  # active, busy, away, offline
    current_location: Optional[str] = None
    device_type: str = "desktop"  # desktop, mobile, tablet
    
    # Objectifs
    personalization_goals: List[PersonalizationGoal] = field(default_factory=list)
    
    # Contraintes
    max_content_length: Optional[int] = None
    required_elements: List[str] = field(default_factory=list)
    forbidden_elements: List[str] = field(default_factory=list)
    
    # Configuration
    personalization_strength: float = 0.8  # 0.0 = minimal, 1.0 = maximum
    fallback_strategy: str = "default"


@dataclass
class PersonalizationResult:
    """Résultat de personnalisation"""
    id: str = field(default_factory=lambda: f"pers_{int(datetime.utcnow().timestamp())}")
    original_content: str = ""
    personalized_content: str = ""
    
    # Métadonnées de personnalisation
    applied_personalizations: List[str] = field(default_factory=list)
    personalization_score: float = 0.0
    confidence_score: float = 0.0
    
    # Prédictions
    predicted_engagement: float = 0.0
    predicted_satisfaction: float = 0.0
    expected_response_rate: float = 0.0
    
    # Informations de traitement
    processing_time_ms: float = 0.0
    ml_models_used: List[str] = field(default_factory=list)
    
    # Recommandations
    timing_recommendation: Optional[datetime] = None
    channel_recommendation: Optional[str] = None
    follow_up_suggestions: List[str] = field(default_factory=list)
    
    # Métriques
    created_at: datetime = field(default_factory=datetime.utcnow)


class BehavioralAnalyzer:
    """Analyseur comportemental avancé"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Modèles ML
        self.engagement_model = None
        self.churn_model = None
        self.satisfaction_model = None
        
        # Cache d'analyse
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # Patterns comportementaux
        self.behavioral_patterns = self._load_behavioral_patterns()
        
        self._init_models()
    
    def _init_models(self):
        """Initialisation des modèles ML"""
        
        try:
            # Modèle d'engagement (Random Forest)
            self.engagement_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            # Modèle de churn (Gradient Boosting)
            self.churn_model = RandomForestClassifier(
                n_estimators=50,
                random_state=42
            )
            
            # Modèle de satisfaction (régression)
            self.satisfaction_model = RandomForestClassifier(
                n_estimators=75,
                random_state=42
            )
            
            # Clustering pour segmentation
            self.clustering_model = KMeans(n_clusters=8, random_state=42)
            self.scaler = StandardScaler()
            
            self.logger.info("Behavioral analysis models initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {str(e)}")
    
    def _load_behavioral_patterns(self) -> Dict[str, Any]:
        """Chargement des patterns comportementaux"""
        
        return {
            'engagement_indicators': {
                'high_engagement': [
                    'frequent_logins',
                    'long_sessions',
                    'feature_exploration',
                    'content_sharing',
                    'feedback_providing'
                ],
                'low_engagement': [
                    'infrequent_logins',
                    'short_sessions',
                    'limited_features',
                    'no_interactions',
                    'passive_consumption'
                ]
            },
            'communication_preferences': {
                'immediate': ['push_notifications', 'slack_dm', 'sms'],
                'scheduled': ['email_digest', 'weekly_report'],
                'on_demand': ['dashboard_check', 'manual_sync']
            },
            'content_consumption': {
                'scanner': 'prefers_summaries_and_highlights',
                'deep_reader': 'prefers_detailed_content',
                'visual_learner': 'prefers_charts_and_images',
                'interactive': 'prefers_clickable_elements'
            }
        }
    
    async def analyze_user_behavior(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyse comportementale complète"""
        
        user_id = user_profile.user_id
        
        # Vérification du cache
        cache_key = f"{user_id}_{user_profile.updated_at.isoformat()}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        try:
            # Analyse des patterns d'activité
            activity_analysis = await self._analyze_activity_patterns(user_profile)
            
            # Analyse d'engagement
            engagement_analysis = await self._analyze_engagement(user_profile)
            
            # Analyse de risque de churn
            churn_analysis = await self._analyze_churn_risk(user_profile)
            
            # Segmentation utilisateur
            segment_analysis = await self._determine_user_segment(user_profile)
            
            # Préférences déduites
            preferences_analysis = await self._infer_preferences(user_profile)
            
            # Compilation des résultats
            analysis_result = {
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat(),
                'activity_patterns': activity_analysis,
                'engagement': engagement_analysis,
                'churn_risk': churn_analysis,
                'segment': segment_analysis,
                'inferred_preferences': preferences_analysis,
                'behavioral_insights': await self._generate_behavioral_insights(user_profile)
            }
            
            # Mise en cache
            self.analysis_cache[cache_key] = analysis_result
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Behavioral analysis failed for user {user_id}: {str(e)}")
            return self._get_default_analysis(user_id)
    
    async def _analyze_activity_patterns(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyse des patterns d'activité"""
        
        if not user_profile.engagement_history:
            return {
                'session_frequency': 'unknown',
                'peak_hours': [],
                'preferred_days': [],
                'session_duration_avg': 0,
                'activity_consistency': 0.0
            }
        
        # Conversion en DataFrame pour analyse
        df = pd.DataFrame(user_profile.engagement_history)
        
        if 'timestamp' not in df.columns:
            return self._get_default_activity_patterns()
        
        # Conversion des timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Calcul des métriques
        session_frequency = self._calculate_session_frequency(df)
        peak_hours = self._identify_peak_hours(df)
        preferred_days = self._identify_preferred_days(df)
        avg_duration = self._calculate_avg_session_duration(df)
        consistency = self._calculate_activity_consistency(df)
        
        return {
            'session_frequency': session_frequency,
            'peak_hours': peak_hours,
            'preferred_days': preferred_days,
            'session_duration_avg': avg_duration,
            'activity_consistency': consistency,
            'total_sessions': len(df),
            'data_quality': 'good' if len(df) > 10 else 'limited'
        }
    
    async def _analyze_engagement(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyse d'engagement"""
        
        # Features pour le modèle d'engagement
        features = self._extract_engagement_features(user_profile)
        
        if not features:
            return {
                'score': 0.5,
                'level': 'medium',
                'trends': 'stable',
                'confidence': 0.0
            }
        
        # Prédiction avec le modèle (simulation)
        engagement_score = self._predict_engagement_score(features)
        engagement_level = self._categorize_engagement(engagement_score)
        trends = self._analyze_engagement_trends(user_profile)
        
        return {
            'score': engagement_score,
            'level': engagement_level,
            'trends': trends,
            'confidence': 0.8,
            'key_factors': self._identify_engagement_factors(features),
            'improvement_opportunities': self._suggest_engagement_improvements(features)
        }
    
    async def _analyze_churn_risk(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyse du risque de churn"""
        
        # Features pour le modèle de churn
        features = self._extract_churn_features(user_profile)
        
        if not features:
            return {
                'risk_score': 0.3,
                'risk_level': 'low',
                'confidence': 0.0
            }
        
        # Prédiction du risque de churn
        churn_score = self._predict_churn_score(features)
        risk_level = self._categorize_churn_risk(churn_score)
        
        return {
            'risk_score': churn_score,
            'risk_level': risk_level,
            'confidence': 0.75,
            'risk_factors': self._identify_churn_factors(features),
            'retention_recommendations': self._suggest_retention_strategies(features)
        }
    
    async def _determine_user_segment(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Détermination du segment utilisateur"""
        
        # Features pour la segmentation
        features = self._extract_segmentation_features(user_profile)
        
        if not features:
            return {
                'primary_segment': UserSegment.CASUAL_USER.value,
                'confidence': 0.0,
                'characteristics': []
            }
        
        # Clustering et segmentation
        segment = self._predict_user_segment(features)
        characteristics = self._describe_segment_characteristics(segment, features)
        
        return {
            'primary_segment': segment.value,
            'confidence': 0.8,
            'characteristics': characteristics,
            'segment_migration_probability': self._calculate_segment_migration(features)
        }
    
    async def _infer_preferences(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Inférence des préférences"""
        
        # Analyse des interactions passées
        interaction_preferences = self._analyze_interaction_preferences(user_profile)
        
        # Analyse du contenu consommé
        content_preferences = self._analyze_content_preferences(user_profile)
        
        # Analyse temporelle
        timing_preferences = self._analyze_timing_preferences(user_profile)
        
        # Analyse des canaux
        channel_preferences = self._analyze_channel_preferences(user_profile)
        
        return {
            'interaction_style': interaction_preferences,
            'content_type': content_preferences,
            'timing': timing_preferences,
            'channels': channel_preferences,
            'communication_formality': self._infer_communication_style(user_profile),
            'complexity_tolerance': self._infer_complexity_preference(user_profile)
        }
    
    async def _generate_behavioral_insights(self, user_profile: UserProfile) -> List[str]:
        """Génération d'insights comportementaux"""
        
        insights = []
        
        # Insights basés sur l'activité
        if user_profile.activity_score > 0.8:
            insights.append("Utilisateur très actif - opportunité pour des fonctionnalités avancées")
        elif user_profile.activity_score < 0.3:
            insights.append("Utilisateur peu actif - nécessite engagement et onboarding")
        
        # Insights basés sur l'engagement
        if user_profile.engagement_score > 0.7:
            insights.append("Fort engagement - candidat pour programme ambassadeur")
        
        # Insights basés sur le risque de churn
        if user_profile.churn_risk > 0.6:
            insights.append("Risque de churn élevé - intervention immédiate recommandée")
        
        # Insights temporels
        recent_activity = any(
            datetime.fromisoformat(event.get('timestamp', '1970-01-01')) > datetime.utcnow() - timedelta(days=7)
            for event in user_profile.engagement_history[-10:]
        )
        
        if not recent_activity:
            insights.append("Absence d'activité récente - campagne de réactivation recommandée")
        
        return insights
    
    def _extract_engagement_features(self, user_profile: UserProfile) -> List[float]:
        """Extraction des features d'engagement"""
        
        features = []
        
        # Features temporelles
        features.append(len(user_profile.engagement_history))  # Nombre d'interactions
        features.append(user_profile.activity_score)
        features.append(user_profile.engagement_score)
        
        # Features comportementales
        features.append(len(user_profile.preferred_channels))
        features.append(len(user_profile.topic_interests))
        features.append(user_profile.complexity_preference)
        
        # Features contextuelles
        days_since_creation = (datetime.utcnow() - user_profile.created_at).days
        features.append(min(days_since_creation / 365, 2.0))  # Ancienneté (max 2 ans)
        
        return features[:10]  # Limitation à 10 features
    
    def _extract_churn_features(self, user_profile: UserProfile) -> List[float]:
        """Extraction des features de churn"""
        
        features = []
        
        # Activité récente
        recent_activity = len([
            event for event in user_profile.engagement_history[-30:]
            if datetime.fromisoformat(event.get('timestamp', '1970-01-01')) > datetime.utcnow() - timedelta(days=30)
        ])
        features.append(recent_activity / 30.0)  # Activité quotidienne moyenne
        
        # Tendance d'engagement
        features.append(1.0 - user_profile.churn_risk)  # Score de rétention
        features.append(user_profile.satisfaction_score)
        features.append(user_profile.engagement_score)
        
        # Diversité d'utilisation
        features.append(len(set(event.get('type', 'unknown') for event in user_profile.engagement_history)))
        
        return features[:8]
    
    def _extract_segmentation_features(self, user_profile: UserProfile) -> List[float]:
        """Extraction des features de segmentation"""
        
        features = []
        
        # Profil d'activité
        features.append(user_profile.activity_score)
        features.append(user_profile.engagement_score)
        features.append(len(user_profile.engagement_history))
        
        # Sophistication
        features.append(user_profile.complexity_preference)
        features.append(len(user_profile.topic_interests))
        
        # Expérience
        experience_map = {'beginner': 0.2, 'intermediate': 0.5, 'advanced': 0.8, 'expert': 1.0}
        features.append(experience_map.get(user_profile.experience_level, 0.5))
        
        return features[:8]
    
    def _predict_engagement_score(self, features: List[float]) -> float:
        """Prédiction du score d'engagement"""
        
        if not features:
            return 0.5
        
        # Simulation d'un modèle ML
        base_score = np.mean(features[:3]) if len(features) >= 3 else 0.5
        
        # Ajustements basés sur les features
        if len(features) > 3:
            diversity_bonus = min(features[3] * 0.1, 0.2)
            complexity_factor = features[4] if len(features) > 4 else 0.5
            
            score = base_score + diversity_bonus + (complexity_factor - 0.5) * 0.1
        else:
            score = base_score
        
        return max(0.0, min(1.0, score))
    
    def _predict_churn_score(self, features: List[float]) -> float:
        """Prédiction du score de churn"""
        
        if not features:
            return 0.3
        
        # Simulation d'un modèle de churn
        recent_activity = features[0] if features else 0.5
        satisfaction = features[2] if len(features) > 2 else 0.7
        engagement = features[3] if len(features) > 3 else 0.5
        
        # Score de churn inversement proportionnel à l'engagement
        churn_score = 1.0 - (recent_activity * 0.4 + satisfaction * 0.3 + engagement * 0.3)
        
        return max(0.0, min(1.0, churn_score))
    
    def _predict_user_segment(self, features: List[float]) -> UserSegment:
        """Prédiction du segment utilisateur"""
        
        if not features:
            return UserSegment.CASUAL_USER
        
        activity = features[0] if features else 0.5
        engagement = features[1] if len(features) > 1 else 0.5
        complexity = features[4] if len(features) > 4 else 0.5
        experience = features[5] if len(features) > 5 else 0.5
        
        # Logique de segmentation
        if activity > 0.8 and engagement > 0.8:
            return UserSegment.POWER_USER
        elif complexity > 0.7 and experience > 0.7:
            return UserSegment.DEVELOPER
        elif activity < 0.3:
            return UserSegment.NEW_USER
        elif engagement > 0.6:
            return UserSegment.CONTENT_CREATOR
        else:
            return UserSegment.CASUAL_USER
    
    def _get_default_analysis(self, user_id: str) -> Dict[str, Any]:
        """Analyse par défaut en cas d'erreur"""
        
        return {
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat(),
            'activity_patterns': self._get_default_activity_patterns(),
            'engagement': {'score': 0.5, 'level': 'medium'},
            'churn_risk': {'risk_score': 0.3, 'risk_level': 'low'},
            'segment': {'primary_segment': 'casual_user'},
            'inferred_preferences': {},
            'behavioral_insights': ['Données insuffisantes pour analyse complète']
        }
    
    def _get_default_activity_patterns(self) -> Dict[str, Any]:
        """Patterns d'activité par défaut"""
        
        return {
            'session_frequency': 'unknown',
            'peak_hours': [9, 14, 16],  # Heures de bureau par défaut
            'preferred_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'session_duration_avg': 15,
            'activity_consistency': 0.5
        }


class ContentPersonalizer:
    """Personnaliseur de contenu avancé"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Modèles de personnalisation
        self.sentiment_analyzer = None
        self.style_adapter = None
        
        # Templates et patterns
        self.personalization_templates = self._load_personalization_templates()
        self.style_guides = self._load_style_guides()
        
        self._init_nlp_models()
    
    def _init_nlp_models(self):
        """Initialisation des modèles NLP"""
        
        try:
            # Analyseur de sentiment
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            self.logger.info("NLP models initialized for content personalization")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP models: {str(e)}")
    
    def _load_personalization_templates(self) -> Dict[str, Dict[str, str]]:
        """Chargement des templates de personnalisation"""
        
        return {
            'greeting': {
                'formal': "Dear {name},",
                'casual': "Hi {name}!",
                'professional': "Hello {name},",
                'friendly': "Hey {name}! 👋"
            },
            'closing': {
                'formal': "Best regards,",
                'casual': "Cheers!",
                'professional': "Thank you,",
                'friendly': "Talk soon! 😊"
            },
            'urgency': {
                'high': "⚡ Urgent: ",
                'medium': "📋 Important: ",
                'low': "💡 FYI: ",
                'none': ""
            },
            'enthusiasm': {
                'high': "🎉 Exciting news! ",
                'medium': "Great news! ",
                'low': "We wanted to let you know ",
                'none': ""
            }
        }
    
    def _load_style_guides(self) -> Dict[str, Dict[str, Any]]:
        """Chargement des guides de style"""
        
        return {
            'casual': {
                'contractions': True,
                'emojis': True,
                'exclamations': True,
                'informal_words': True,
                'sentence_length': 'short'
            },
            'professional': {
                'contractions': False,
                'emojis': False,
                'exclamations': False,
                'formal_words': True,
                'sentence_length': 'medium'
            },
            'technical': {
                'contractions': False,
                'technical_terms': True,
                'precise_language': True,
                'detailed_explanations': True,
                'sentence_length': 'long'
            },
            'friendly': {
                'contractions': True,
                'emojis': True,
                'warm_language': True,
                'personal_touches': True,
                'sentence_length': 'short'
            }
        }
    
    async def personalize_content(
        self, 
        content: str, 
        context: PersonalizationContext
    ) -> PersonalizationResult:
        """Personnalisation principale du contenu"""
        
        start_time = datetime.utcnow()
        
        try:
            user_profile = context.user_profile
            original_content = content
            
            # Analyse du contenu original
            content_analysis = await self._analyze_content(content)
            
            # Application des personnalisations par couches
            personalized_content = content
            applied_personalizations = []
            
            # 1. Personnalisation du style de communication
            personalized_content, style_changes = await self._personalize_communication_style(
                personalized_content, context, content_analysis
            )
            applied_personalizations.extend(style_changes)
            
            # 2. Adaptation du niveau de complexité
            personalized_content, complexity_changes = await self._adapt_complexity_level(
                personalized_content, context, content_analysis
            )
            applied_personalizations.extend(complexity_changes)
            
            # 3. Personnalisation émotionnelle
            personalized_content, emotional_changes = await self._personalize_emotional_tone(
                personalized_content, context, content_analysis
            )
            applied_personalizations.extend(emotional_changes)
            
            # 4. Adaptation culturelle/linguistique
            personalized_content, cultural_changes = await self._apply_cultural_personalization(
                personalized_content, context
            )
            applied_personalizations.extend(cultural_changes)
            
            # 5. Optimisation de l'engagement
            personalized_content, engagement_changes = await self._optimize_for_engagement(
                personalized_content, context, content_analysis
            )
            applied_personalizations.extend(engagement_changes)
            
            # Calcul des scores et prédictions
            scores = await self._calculate_personalization_scores(
                original_content, personalized_content, context
            )
            
            # Génération des recommandations
            recommendations = await self._generate_recommendations(context, content_analysis)
            
            # Temps de traitement
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return PersonalizationResult(
                original_content=original_content,
                personalized_content=personalized_content,
                applied_personalizations=applied_personalizations,
                personalization_score=scores['personalization_score'],
                confidence_score=scores['confidence_score'],
                predicted_engagement=scores['predicted_engagement'],
                predicted_satisfaction=scores['predicted_satisfaction'],
                expected_response_rate=scores['expected_response_rate'],
                processing_time_ms=processing_time,
                timing_recommendation=recommendations.get('timing'),
                channel_recommendation=recommendations.get('channel'),
                follow_up_suggestions=recommendations.get('follow_ups', [])
            )
            
        except Exception as e:
            self.logger.error(f"Content personalization failed: {str(e)}")
            
            return PersonalizationResult(
                original_content=content,
                personalized_content=content,
                applied_personalizations=[f"Error: {str(e)}"],
                confidence_score=0.0
            )
    
    async def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyse du contenu original"""
        
        analysis = {
            'length': len(content),
            'word_count': len(content.split()),
            'sentence_count': len([s for s in content.split('.') if s.strip()]),
            'sentiment': 'neutral',
            'complexity': 0.5,
            'tone': 'neutral',
            'urgency': 'medium',
            'topic_detected': 'general'
        }
        
        try:
            # Analyse de sentiment
            if self.sentiment_analyzer:
                sentiment_result = self.sentiment_analyzer(content[:512])  # Limite pour le modèle
                if sentiment_result:
                    analysis['sentiment'] = sentiment_result[0]['label'].lower()
                    analysis['sentiment_confidence'] = sentiment_result[0]['score']
            
            # Analyse de complexité (simulation)
            avg_word_length = np.mean([len(word) for word in content.split()])
            avg_sentence_length = analysis['word_count'] / max(analysis['sentence_count'], 1)
            
            complexity_score = (avg_word_length / 10 + avg_sentence_length / 20) / 2
            analysis['complexity'] = min(1.0, complexity_score)
            
            # Détection d'urgence
            urgency_keywords = ['urgent', 'immediate', 'asap', 'critical', 'emergency']
            if any(keyword in content.lower() for keyword in urgency_keywords):
                analysis['urgency'] = 'high'
            elif any(keyword in content.lower() for keyword in ['soon', 'quick', 'fast']):
                analysis['urgency'] = 'medium'
            else:
                analysis['urgency'] = 'low'
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {str(e)}")
        
        return analysis
    
    async def _personalize_communication_style(
        self, 
        content: str, 
        context: PersonalizationContext,
        content_analysis: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Personnalisation du style de communication"""
        
        user_profile = context.user_profile
        target_style = user_profile.communication_style
        
        if target_style not in self.style_guides:
            return content, []
        
        style_guide = self.style_guides[target_style]
        personalized_content = content
        changes = []
        
        # Application des transformations de style
        if style_guide.get('contractions', False):
            # Ajout de contractions
            contractions = {
                'do not': "don't",
                'cannot': "can't",
                'will not': "won't",
                'you are': "you're",
                'we are': "we're"
            }
            for full, contracted in contractions.items():
                if full in personalized_content.lower():
                    personalized_content = personalized_content.replace(full, contracted)
                    changes.append(f"Applied contraction: {full} -> {contracted}")
        
        # Ajout d'emojis si approprié
        if style_guide.get('emojis', False) and context.channel in ['slack', 'teams']:
            if 'success' in content.lower() or 'completed' in content.lower():
                personalized_content += " ✅"
                changes.append("Added success emoji")
            elif 'warning' in content.lower() or 'attention' in content.lower():
                personalized_content = "⚠️ " + personalized_content
                changes.append("Added warning emoji")
        
        # Ajustement des salutations et clôtures
        greeting_style = 'casual' if target_style == 'casual' else 'professional'
        if user_profile.user_id:  # Si on a le nom d'utilisateur
            user_name = user_profile.user_id.split('@')[0] if '@' in user_profile.user_id else user_profile.user_id
            greeting_template = self.personalization_templates['greeting'].get(greeting_style, "Hi {name},")
            personalized_greeting = greeting_template.format(name=user_name)
            
            if not personalized_content.startswith(('Hi', 'Hello', 'Dear', 'Hey')):
                personalized_content = f"{personalized_greeting}\n\n{personalized_content}"
                changes.append(f"Added personalized greeting ({greeting_style})")
        
        return personalized_content, changes
    
    async def _adapt_complexity_level(
        self, 
        content: str, 
        context: PersonalizationContext,
        content_analysis: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Adaptation du niveau de complexité"""
        
        user_complexity_pref = context.user_profile.complexity_preference
        content_complexity = content_analysis.get('complexity', 0.5)
        
        changes = []
        adapted_content = content
        
        # Si l'utilisateur préfère la simplicité et le contenu est complexe
        if user_complexity_pref < 0.4 and content_complexity > 0.6:
            # Simplification
            simplifications = {
                'utilize': 'use',
                'demonstrate': 'show',
                'facilitate': 'help',
                'subsequently': 'then',
                'consequently': 'so',
                'nevertheless': 'but',
                'furthermore': 'also'
            }
            
            for complex_word, simple_word in simplifications.items():
                if complex_word in adapted_content.lower():
                    adapted_content = adapted_content.replace(complex_word, simple_word)
                    changes.append(f"Simplified: {complex_word} -> {simple_word}")
        
        # Si l'utilisateur préfère la complexité et le contenu est simple
        elif user_complexity_pref > 0.7 and content_complexity < 0.4:
            # Ajout de détails techniques
            if context.user_profile.role in ['developer', 'analyst'] and content_complexity < 0.5:
                # Ajout de détails techniques contextuels
                if 'error' in content.lower():
                    adapted_content += "\n\nTechnical details: This may be related to configuration or system state changes."
                    changes.append("Added technical context for advanced user")
        
        return adapted_content, changes
    
    async def _personalize_emotional_tone(
        self, 
        content: str, 
        context: PersonalizationContext,
        content_analysis: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Personnalisation du ton émotionnel"""
        
        user_profile = context.user_profile
        content_sentiment = content_analysis.get('sentiment', 'neutral')
        
        changes = []
        personalized_content = content
        
        # Adaptation basée sur l'historique d'engagement
        if user_profile.engagement_score > 0.8:
            # Utilisateur très engagé - ton plus enthousiaste
            enthusiasm_level = 'medium'
            if content_sentiment == 'positive':
                enthusiasm_prefix = self.personalization_templates['enthusiasm'][enthusiasm_level]
                if not personalized_content.startswith(('🎉', 'Great', 'Exciting')):
                    personalized_content = f"{enthusiasm_prefix}{personalized_content}"
                    changes.append("Added enthusiasm for highly engaged user")
        
        # Adaptation pour utilisateurs à risque de churn
        elif user_profile.churn_risk > 0.6:
            # Ton plus personnel et attentionné
            if 'we' in personalized_content.lower():
                personalized_content = personalized_content.replace('we', 'I')
                changes.append("Personalized pronouns for at-risk user")
        
        # Adaptation basée sur le contexte temporel
        if context.current_time.hour >= 18:  # Soirée
            if user_profile.communication_style == 'casual':
                personalized_content = personalized_content.replace('Good day', 'Good evening')
                changes.append("Adapted greeting for time of day")
        
        return personalized_content, changes
    
    async def _apply_cultural_personalization(
        self, 
        content: str, 
        context: PersonalizationContext
    ) -> Tuple[str, List[str]]:
        """Application de la personnalisation culturelle"""
        
        user_profile = context.user_profile
        changes = []
        
        # Adaptation basée sur la langue/pays
        if user_profile.language != 'en':
            # Ajout de contexte multilingue si approprié
            if context.channel == 'email' and not content.startswith('English follows'):
                # Note: En production, on utiliserait un service de traduction
                changes.append(f"Considered localization for {user_profile.language}")
        
        # Adaptation des formats culturels
        if user_profile.country == 'DE':
            # Adaptation pour la culture allemande (plus directe)
            if 'maybe' in content or 'perhaps' in content:
                changes.append("Cultural adaptation: German directness preference noted")
        
        return content, changes
    
    async def _optimize_for_engagement(
        self, 
        content: str, 
        context: PersonalizationContext,
        content_analysis: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Optimisation pour l'engagement"""
        
        user_profile = context.user_profile
        changes = []
        optimized_content = content
        
        # Optimisation basée sur les préférences d'engagement
        if user_profile.engagement_score < 0.5:
            # Utilisateur peu engagé - contenu plus attractif
            if not any(char in content for char in ['?', '!']):
                optimized_content += "?"
                changes.append("Added question mark to encourage engagement")
        
        # Optimisation de la longueur basée sur le canal
        if context.channel == 'slack' and len(content) > 200:
            # Slack préfère les messages courts
            if '.' in content:
                sentences = content.split('.')
                optimized_content = sentences[0] + '.'
                if len(sentences) > 1:
                    optimized_content += f"\n\n[Full details: {len(sentences)-1} additional points]"
                changes.append("Condensed content for Slack channel")
        
        # Ajout d'appels à l'action personnalisés
        if PersonalizationGoal.INCREASE_ENGAGEMENT in context.personalization_goals:
            if not any(action_word in content.lower() for action_word in ['click', 'check', 'view', 'try']):
                cta_suggestions = {
                    'developer': "Check out the technical details",
                    'analyst': "Review the data insights", 
                    'user': "Take a look"
                }
                cta = cta_suggestions.get(user_profile.role, "Learn more")
                optimized_content += f"\n\n👉 {cta}"
                changes.append("Added personalized call-to-action")
        
        return optimized_content, changes
    
    async def _calculate_personalization_scores(
        self, 
        original: str, 
        personalized: str, 
        context: PersonalizationContext
    ) -> Dict[str, float]:
        """Calcul des scores de personnalisation"""
        
        # Score de personnalisation basé sur les changements
        personalization_ratio = 1.0 - (len(original) / max(len(personalized), 1))
        personalization_score = min(1.0, abs(personalization_ratio) * 2)
        
        # Score de confiance basé sur la qualité du profil utilisateur
        profile_completeness = self._calculate_profile_completeness(context.user_profile)
        confidence_score = profile_completeness * 0.8  # Ajustement conservateur
        
        # Prédiction d'engagement basée sur l'historique
        user_engagement_history = context.user_profile.engagement_score
        content_engagement_potential = self._estimate_content_engagement(personalized, context)
        predicted_engagement = (user_engagement_history + content_engagement_potential) / 2
        
        # Prédiction de satisfaction
        predicted_satisfaction = min(1.0, predicted_engagement * 1.1)  # Légèrement plus optimiste
        
        # Prédiction de taux de réponse
        base_response_rate = 0.1  # Taux de base de 10%
        personalization_boost = personalization_score * 0.2  # Boost jusqu'à 20%
        expected_response_rate = min(0.8, base_response_rate + personalization_boost)
        
        return {
            'personalization_score': personalization_score,
            'confidence_score': confidence_score,
            'predicted_engagement': predicted_engagement,
            'predicted_satisfaction': predicted_satisfaction,
            'expected_response_rate': expected_response_rate
        }
    
    def _calculate_profile_completeness(self, user_profile: UserProfile) -> float:
        """Calcul de la complétude du profil"""
        
        completeness_factors = [
            1.0 if user_profile.language != "en" else 0.0,  # Langue spécifiée
            1.0 if user_profile.communication_style != "professional" else 0.0,  # Style personnalisé
            1.0 if len(user_profile.preferred_channels) > 1 else 0.0,  # Canaux multiples
            1.0 if len(user_profile.engagement_history) > 10 else 0.0,  # Historique riche
            1.0 if user_profile.complexity_preference != 0.5 else 0.0,  # Préférence de complexité
            1.0 if len(user_profile.topic_interests) > 0 else 0.0,  # Intérêts définis
            1.0 if user_profile.role != "user" else 0.0,  # Rôle spécialisé
            1.0 if user_profile.experience_level != "intermediate" else 0.0  # Niveau d'expérience
        ]
        
        return sum(completeness_factors) / len(completeness_factors)
    
    def _estimate_content_engagement(self, content: str, context: PersonalizationContext) -> float:
        """Estimation du potentiel d'engagement du contenu"""
        
        base_score = 0.5
        
        # Facteurs positifs
        if any(emoji in content for emoji in ['✅', '🎉', '👉', '⚡']):
            base_score += 0.1  # Émojis engageants
        
        if '?' in content:
            base_score += 0.05  # Questions engagent
        
        if len(content.split()) < 50:  # Contenu concis
            base_score += 0.1
        
        # Facteurs négatifs
        if len(content) > 500:  # Trop long
            base_score -= 0.1
        
        # Ajustement par canal
        channel_factors = {
            'slack': 1.2,  # Slack favorise l'engagement
            'email': 0.8,  # Email moins interactif
            'teams': 1.1,
            'discord': 1.3
        }
        
        channel_factor = channel_factors.get(context.channel, 1.0)
        
        return min(1.0, max(0.0, base_score * channel_factor))
    
    async def _generate_recommendations(
        self, 
        context: PersonalizationContext,
        content_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Génération de recommandations"""
        
        recommendations = {}
        user_profile = context.user_profile
        
        # Recommandation de timing
        if user_profile.activity_patterns:
            peak_hours = user_profile.activity_patterns.get('peak_hours', [9, 14])
            if peak_hours:
                next_peak = min([h for h in peak_hours if h > context.current_time.hour], default=peak_hours[0])
                recommendations['timing'] = context.current_time.replace(
                    hour=next_peak, minute=0, second=0, microsecond=0
                )
        
        # Recommandation de canal
        if user_profile.preferred_channels:
            # Logique de sélection du meilleur canal
            urgency = content_analysis.get('urgency', 'medium')
            
            if urgency == 'high' and 'slack' in user_profile.preferred_channels:
                recommendations['channel'] = 'slack'
            elif urgency == 'low' and 'email' in user_profile.preferred_channels:
                recommendations['channel'] = 'email'
            else:
                recommendations['channel'] = user_profile.preferred_channels[0]
        
        # Suggestions de suivi
        follow_ups = []
        
        if user_profile.churn_risk > 0.5:
            follow_ups.append("Schedule follow-up check-in within 3 days")
        
        if user_profile.engagement_score > 0.8:
            follow_ups.append("Consider inviting to beta features")
        
        recommendations['follow_ups'] = follow_ups
        
        return recommendations


class PersonalizationEngine:
    """Moteur principal de personnalisation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Composants principaux
        self.behavioral_analyzer = BehavioralAnalyzer(config)
        self.content_personalizer = ContentPersonalizer(config)
        
        # Cache et métriques
        self.personalization_cache: Dict[str, PersonalizationResult] = {}
        self.performance_metrics = {
            'total_personalizations': 0,
            'avg_processing_time': 0.0,
            'success_rate': 1.0,
            'cache_hit_rate': 0.0
        }
    
    async def personalize(
        self, 
        content: str, 
        user_profile: UserProfile,
        context_overrides: Optional[Dict[str, Any]] = None
    ) -> PersonalizationResult:
        """Point d'entrée principal pour la personnalisation"""
        
        try:
            # Construction du contexte
            context = PersonalizationContext(
                user_profile=user_profile,
                **(context_overrides or {})
            )
            
            # Vérification du cache
            cache_key = self._generate_cache_key(content, context)
            if cache_key in self.personalization_cache:
                self.performance_metrics['cache_hit_rate'] += 1
                return self.personalization_cache[cache_key]
            
            # Analyse comportementale
            behavioral_analysis = await self.behavioral_analyzer.analyze_user_behavior(user_profile)
            
            # Mise à jour du profil avec les insights comportementaux
            updated_profile = await self._enrich_profile_with_insights(user_profile, behavioral_analysis)
            context.user_profile = updated_profile
            
            # Personnalisation du contenu
            result = await self.content_personalizer.personalize_content(content, context)
            
            # Mise en cache
            self.personalization_cache[cache_key] = result
            
            # Mise à jour des métriques
            self._update_performance_metrics(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Personalization engine failed: {str(e)}")
            
            # Retour du contenu original en cas d'erreur
            return PersonalizationResult(
                original_content=content,
                personalized_content=content,
                applied_personalizations=[f"Engine error: {str(e)}"],
                confidence_score=0.0
            )
    
    async def _enrich_profile_with_insights(
        self, 
        profile: UserProfile, 
        behavioral_analysis: Dict[str, Any]
    ) -> UserProfile:
        """Enrichissement du profil avec les insights comportementaux"""
        
        enriched_profile = profile
        
        # Mise à jour du segment
        segment_info = behavioral_analysis.get('segment', {})
        if 'primary_segment' in segment_info:
            try:
                enriched_profile.segment = UserSegment(segment_info['primary_segment'])
            except ValueError:
                pass  # Garde le segment existant si invalide
        
        # Mise à jour des scores
        engagement_info = behavioral_analysis.get('engagement', {})
        if 'score' in engagement_info:
            enriched_profile.engagement_score = engagement_info['score']
        
        churn_info = behavioral_analysis.get('churn_risk', {})
        if 'risk_score' in churn_info:
            enriched_profile.churn_risk = churn_info['risk_score']
        
        # Mise à jour des patterns d'activité
        activity_info = behavioral_analysis.get('activity_patterns', {})
        if activity_info:
            enriched_profile.activity_patterns = activity_info
        
        # Mise à jour des préférences inférées
        preferences_info = behavioral_analysis.get('inferred_preferences', {})
        if preferences_info:
            if 'communication_formality' in preferences_info:
                enriched_profile.communication_style = preferences_info['communication_formality']
            
            if 'complexity_tolerance' in preferences_info:
                enriched_profile.complexity_preference = preferences_info['complexity_tolerance']
        
        enriched_profile.updated_at = datetime.utcnow()
        
        return enriched_profile
    
    def _generate_cache_key(self, content: str, context: PersonalizationContext) -> str:
        """Génération de clé de cache"""
        
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        context_factors = [
            context.user_profile.user_id,
            context.user_profile.communication_style,
            context.channel,
            str(context.personalization_strength),
            str(context.user_profile.updated_at.date())
        ]
        
        context_hash = hashlib.md5('_'.join(context_factors).encode()).hexdigest()[:8]
        
        return f"{content_hash}_{context_hash}"
    
    def _update_performance_metrics(self, result: PersonalizationResult):
        """Mise à jour des métriques de performance"""
        
        self.performance_metrics['total_personalizations'] += 1
        
        # Moyenne mobile du temps de traitement
        current_avg = self.performance_metrics['avg_processing_time']
        total = self.performance_metrics['total_personalizations']
        new_avg = (current_avg * (total - 1) + result.processing_time_ms) / total
        self.performance_metrics['avg_processing_time'] = new_avg
        
        # Calcul du taux de succès
        if result.confidence_score > 0:
            success_count = self.performance_metrics['success_rate'] * (total - 1) + 1
            self.performance_metrics['success_rate'] = success_count / total
    
    async def get_personalization_insights(self, user_id: str) -> Dict[str, Any]:
        """Obtention d'insights de personnalisation pour un utilisateur"""
        
        # Recherche des personnalisations récentes pour cet utilisateur
        user_personalizations = [
            result for result in self.personalization_cache.values()
            if hasattr(result, 'user_id') and result.user_id == user_id
        ]
        
        if not user_personalizations:
            return {'message': 'No personalization data available for this user'}
        
        # Calcul des statistiques
        avg_personalization_score = np.mean([r.personalization_score for r in user_personalizations])
        avg_confidence = np.mean([r.confidence_score for r in user_personalizations])
        
        most_common_personalizations = {}
        for result in user_personalizations:
            for personalization in result.applied_personalizations:
                most_common_personalizations[personalization] = most_common_personalizations.get(personalization, 0) + 1
        
        top_personalizations = sorted(
            most_common_personalizations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            'user_id': user_id,
            'total_personalizations': len(user_personalizations),
            'avg_personalization_score': avg_personalization_score,
            'avg_confidence_score': avg_confidence,
            'top_applied_personalizations': [p[0] for p in top_personalizations],
            'most_recent_personalization': user_personalizations[-1].created_at.isoformat(),
            'performance_trend': 'improving' if avg_confidence > 0.7 else 'stable'
        }
    
    async def get_system_performance(self) -> Dict[str, Any]:
        """Obtention des performances système"""
        
        cache_size = len(self.personalization_cache)
        cache_hit_rate = (
            self.performance_metrics['cache_hit_rate'] / 
            max(self.performance_metrics['total_personalizations'], 1)
        )
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_personalizations': self.performance_metrics['total_personalizations'],
            'avg_processing_time_ms': self.performance_metrics['avg_processing_time'],
            'success_rate': self.performance_metrics['success_rate'],
            'cache_size': cache_size,
            'cache_hit_rate': cache_hit_rate,
            'system_status': 'healthy' if self.performance_metrics['success_rate'] > 0.9 else 'degraded'
        }
