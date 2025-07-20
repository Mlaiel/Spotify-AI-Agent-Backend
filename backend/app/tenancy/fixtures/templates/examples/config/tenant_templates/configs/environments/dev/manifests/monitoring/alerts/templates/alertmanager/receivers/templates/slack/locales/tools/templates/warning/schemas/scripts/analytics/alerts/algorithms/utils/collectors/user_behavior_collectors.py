"""
User Behavior Collectors - Collecteurs de Comportement Utilisateur
================================================================

Collecteurs spécialisés pour analyser le comportement des utilisateurs
et les patterns d'usage du système Spotify AI Agent.

Features:
    - Analyse du parcours utilisateur (user journey)
    - Détection de patterns d'interaction
    - Évolution des préférences utilisateur
    - Prédiction de churn comportemental
    - Segmentation comportementale avancée

Author: Expert Data Visualisation + Analyste Comportement Team
"""

import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
from collections import defaultdict, Counter, deque
import hashlib
import uuid

from . import BaseCollector, CollectorConfig

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types d'interactions utilisateur."""
    PAGE_VIEW = "page_view"
    CLICK = "click"
    SEARCH = "search"
    PLAY_MUSIC = "play_music"
    PAUSE_MUSIC = "pause_music"
    SKIP_TRACK = "skip_track"
    LIKE_TRACK = "like_track"
    CREATE_PLAYLIST = "create_playlist"
    SHARE_CONTENT = "share_content"
    COMMENT = "comment"
    RATE_CONTENT = "rate_content"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    FEATURE_USE = "feature_use"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


class UserSegment(Enum):
    """Segments d'utilisateurs."""
    NEW_USER = "new_user"
    CASUAL_USER = "casual_user"
    REGULAR_USER = "regular_user"
    POWER_USER = "power_user"
    CHURNED_USER = "churned_user"
    RETURNING_USER = "returning_user"


@dataclass
class UserInteraction:
    """Structure d'une interaction utilisateur."""
    interaction_id: str
    user_id: str
    tenant_id: str
    timestamp: datetime
    interaction_type: InteractionType
    page_url: Optional[str] = None
    element_id: Optional[str] = None
    session_id: Optional[str] = None
    device_type: Optional[str] = None
    browser: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    referrer: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserSession:
    """Structure d'une session utilisateur."""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: int = 0
    page_views: int = 0
    interactions_count: int = 0
    bounce: bool = False
    conversion: bool = False
    device_info: Dict[str, str] = field(default_factory=dict)
    exit_page: Optional[str] = None
    goal_completions: List[str] = field(default_factory=list)


class UserJourneyCollector(BaseCollector):
    """Collecteur principal pour l'analyse du parcours utilisateur."""
    
    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self.journey_analyzer = JourneyAnalyzer()
        self.funnel_analyzer = FunnelAnalyzer()
        self.cohort_analyzer = CohortAnalyzer()
        self.behavior_predictor = BehaviorPredictor()
        
    async def collect(self) -> Dict[str, Any]:
        """Collecte complète des données de parcours utilisateur."""
        tenant_id = self.config.tags.get('tenant_id', 'default')
        
        try:
            # Collecte des interactions récentes
            recent_interactions = await self._collect_recent_interactions(tenant_id)
            
            # Analyse des parcours
            journey_analysis = await self.journey_analyzer.analyze_journeys(
                tenant_id, recent_interactions
            )
            
            # Analyse des funnels
            funnel_analysis = await self.funnel_analyzer.analyze_funnels(
                tenant_id, recent_interactions
            )
            
            # Analyse des cohortes
            cohort_analysis = await self.cohort_analyzer.analyze_cohorts(tenant_id)
            
            # Prédictions comportementales
            behavior_predictions = await self.behavior_predictor.predict_behaviors(
                tenant_id, recent_interactions
            )
            
            # Score d'expérience utilisateur
            ux_score = self._calculate_ux_score(
                journey_analysis, funnel_analysis, cohort_analysis
            )
            
            return {
                'user_journey_metrics': {
                    'tenant_id': tenant_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'interactions_analyzed': len(recent_interactions),
                    'journey_analysis': journey_analysis,
                    'funnel_analysis': funnel_analysis,
                    'cohort_analysis': cohort_analysis,
                    'behavior_predictions': behavior_predictions,
                    'ux_score': ux_score,
                    'recommendations': await self._generate_ux_recommendations(
                        journey_analysis, funnel_analysis
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur collecte parcours utilisateur: {str(e)}")
            raise
    
    async def _collect_recent_interactions(self, tenant_id: str) -> List[UserInteraction]:
        """Collecte les interactions utilisateur récentes."""
        # Simulation d'interactions - en production, requête DB
        interactions = []
        
        # Générer des interactions simulées pour les dernières 24h
        users = [f"user_{i}" for i in range(1, 101)]  # 100 utilisateurs
        interaction_types = list(InteractionType)
        
        for i in range(2000):  # 2000 interactions
            user_id = np.random.choice(users)
            interaction_type = np.random.choice(interaction_types)
            timestamp = datetime.utcnow() - timedelta(
                minutes=np.random.randint(0, 24*60)
            )
            
            interaction = UserInteraction(
                interaction_id=str(uuid.uuid4()),
                user_id=user_id,
                tenant_id=tenant_id,
                timestamp=timestamp,
                interaction_type=interaction_type,
                page_url=f"/page/{np.random.randint(1, 20)}",
                session_id=f"session_{user_id}_{timestamp.date()}",
                device_type=np.random.choice(['desktop', 'mobile', 'tablet']),
                browser=np.random.choice(['chrome', 'firefox', 'safari', 'edge']),
                context={
                    'feature_used': np.random.choice(['music_gen', 'collab', 'analytics']),
                    'duration_seconds': np.random.randint(1, 300)
                }
            )
            interactions.append(interaction)
        
        return sorted(interactions, key=lambda x: x.timestamp)
    
    def _calculate_ux_score(self, journey: Dict, funnel: Dict, cohort: Dict) -> float:
        """Calcule un score d'expérience utilisateur."""
        # Score de parcours (40%)
        journey_score = journey.get('completion_rate', 0) * 40
        
        # Score de conversion (35%)
        conversion_rate = funnel.get('overall_conversion_rate', 0)
        conversion_score = conversion_rate * 35
        
        # Score de rétention (25%)
        retention_rate = cohort.get('retention_30_days', 0)
        retention_score = retention_rate * 25
        
        total_score = journey_score + conversion_score + retention_score
        return round(total_score, 2)
    
    async def _generate_ux_recommendations(self, journey: Dict, funnel: Dict) -> List[Dict[str, Any]]:
        """Génère des recommandations UX."""
        recommendations = []
        
        # Recommandations basées sur les points d'abandon
        drop_off_points = funnel.get('drop_off_points', [])
        for point in drop_off_points:
            if point.get('drop_rate', 0) > 0.3:
                recommendations.append({
                    'type': 'reduce_friction',
                    'priority': 'high',
                    'location': point.get('step_name'),
                    'issue': f"High drop-off rate: {point.get('drop_rate', 0):.1%}",
                    'suggestion': 'Simplify the user flow and reduce form fields',
                    'expected_impact': '15-25% improvement in conversion'
                })
        
        # Recommandations basées sur les parcours les plus longs
        avg_journey_length = journey.get('average_journey_length', 0)
        if avg_journey_length > 10:
            recommendations.append({
                'type': 'optimize_navigation',
                'priority': 'medium',
                'issue': f'Long user journeys (avg: {avg_journey_length} steps)',
                'suggestion': 'Improve navigation and add shortcuts to key features',
                'expected_impact': '10-20% reduction in journey length'
            })
        
        # Recommandations basées sur les erreurs
        error_rate = journey.get('error_rate', 0)
        if error_rate > 0.05:
            recommendations.append({
                'type': 'fix_user_errors',
                'priority': 'high',
                'issue': f'High user error rate: {error_rate:.1%}',
                'suggestion': 'Improve error handling and user guidance',
                'expected_impact': '30-40% reduction in user frustration'
            })
        
        return recommendations
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Valide les données de parcours utilisateur."""
        try:
            journey_data = data.get('user_journey_metrics', {})
            
            required_fields = ['tenant_id', 'interactions_analyzed', 'ux_score']
            for field in required_fields:
                if field not in journey_data:
                    return False
            
            # Validation du score UX
            ux_score = journey_data.get('ux_score', -1)
            if not (0 <= ux_score <= 100):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation données parcours: {str(e)}")
            return False


class JourneyAnalyzer:
    """Analyseur de parcours utilisateur."""
    
    async def analyze_journeys(self, tenant_id: str, 
                             interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Analyse les parcours utilisateur."""
        try:
            # Groupement par session
            sessions = self._group_by_session(interactions)
            
            # Analyse des patterns de parcours
            journey_patterns = await self._identify_journey_patterns(sessions)
            
            # Points de friction
            friction_points = await self._identify_friction_points(sessions)
            
            # Parcours optimaux vs réels
            optimal_vs_actual = await self._compare_optimal_journeys(sessions)
            
            # Métriques de parcours
            journey_metrics = self._calculate_journey_metrics(sessions)
            
            return {
                'total_sessions': len(sessions),
                'journey_patterns': journey_patterns,
                'friction_points': friction_points,
                'optimal_vs_actual': optimal_vs_actual,
                'metrics': journey_metrics,
                'completion_rate': journey_metrics.get('completion_rate', 0),
                'average_journey_length': journey_metrics.get('avg_steps', 0),
                'error_rate': journey_metrics.get('error_rate', 0)
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse parcours: {str(e)}")
            return {}
    
    def _group_by_session(self, interactions: List[UserInteraction]) -> Dict[str, List[UserInteraction]]:
        """Groupe les interactions par session."""
        sessions = defaultdict(list)
        for interaction in interactions:
            session_key = f"{interaction.user_id}_{interaction.session_id}"
            sessions[session_key].append(interaction)
        
        # Tri par timestamp dans chaque session
        for session_interactions in sessions.values():
            session_interactions.sort(key=lambda x: x.timestamp)
        
        return dict(sessions)
    
    async def _identify_journey_patterns(self, sessions: Dict) -> List[Dict[str, Any]]:
        """Identifie les patterns de parcours communs."""
        # Extraction des séquences de pages
        page_sequences = []
        for session_interactions in sessions.values():
            sequence = [interaction.page_url for interaction in session_interactions 
                       if interaction.page_url]
            if len(sequence) > 1:
                page_sequences.append(sequence)
        
        # Comptage des séquences communes
        sequence_counts = Counter()
        for sequence in page_sequences:
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                sequence_counts[pair] += 1
        
        # Top patterns
        top_patterns = []
        for (from_page, to_page), count in sequence_counts.most_common(10):
            top_patterns.append({
                'from_page': from_page,
                'to_page': to_page,
                'frequency': count,
                'percentage': count / len(page_sequences) if page_sequences else 0
            })
        
        return top_patterns
    
    async def _identify_friction_points(self, sessions: Dict) -> List[Dict[str, Any]]:
        """Identifie les points de friction dans le parcours."""
        friction_points = []
        
        # Analyse des abandons par page
        page_exits = defaultdict(int)
        page_visits = defaultdict(int)
        
        for session_interactions in sessions.values():
            for i, interaction in enumerate(session_interactions):
                if interaction.page_url:
                    page_visits[interaction.page_url] += 1
                    
                    # Si c'est la dernière interaction de la session
                    if i == len(session_interactions) - 1:
                        page_exits[interaction.page_url] += 1
        
        # Calcul des taux d'abandon
        for page, exits in page_exits.items():
            visits = page_visits[page]
            if visits > 0:
                exit_rate = exits / visits
                if exit_rate > 0.3:  # Seuil de friction
                    friction_points.append({
                        'page': page,
                        'exit_rate': exit_rate,
                        'visits': visits,
                        'exits': exits,
                        'severity': 'high' if exit_rate > 0.5 else 'medium'
                    })
        
        return sorted(friction_points, key=lambda x: x['exit_rate'], reverse=True)
    
    async def _compare_optimal_journeys(self, sessions: Dict) -> Dict[str, Any]:
        """Compare les parcours optimaux avec les parcours réels."""
        # Définition d'un parcours optimal (simulé)
        optimal_journey = ['/home', '/features', '/music-gen', '/create', '/share']
        
        # Analyse des parcours réels
        actual_journeys = []
        for session_interactions in sessions.values():
            journey = [interaction.page_url for interaction in session_interactions 
                      if interaction.page_url]
            if journey:
                actual_journeys.append(journey)
        
        # Calculs de comparaison
        optimal_length = len(optimal_journey)
        actual_lengths = [len(journey) for journey in actual_journeys]
        avg_actual_length = statistics.mean(actual_lengths) if actual_lengths else 0
        
        # Écart par rapport à l'optimal
        length_efficiency = optimal_length / avg_actual_length if avg_actual_length > 0 else 0
        
        # Pages manquées dans les parcours optimaux
        optimal_pages = set(optimal_journey)
        pages_coverage = []
        
        for journey in actual_journeys:
            journey_pages = set(journey)
            coverage = len(optimal_pages.intersection(journey_pages)) / len(optimal_pages)
            pages_coverage.append(coverage)
        
        avg_coverage = statistics.mean(pages_coverage) if pages_coverage else 0
        
        return {
            'optimal_journey': optimal_journey,
            'optimal_length': optimal_length,
            'average_actual_length': avg_actual_length,
            'length_efficiency': length_efficiency,
            'optimal_pages_coverage': avg_coverage,
            'improvement_potential': max(0, 1 - length_efficiency)
        }
    
    def _calculate_journey_metrics(self, sessions: Dict) -> Dict[str, Any]:
        """Calcule les métriques de parcours."""
        if not sessions:
            return {}
        
        session_lengths = []
        completion_count = 0
        error_count = 0
        total_interactions = 0
        
        for session_interactions in sessions.values():
            session_length = len(session_interactions)
            session_lengths.append(session_length)
            total_interactions += session_length
            
            # Critère de complétion (session avec plus de 5 interactions)
            if session_length >= 5:
                completion_count += 1
            
            # Compte des erreurs (interactions avec contexte d'erreur)
            for interaction in session_interactions:
                if interaction.context.get('error', False):
                    error_count += 1
        
        total_sessions = len(sessions)
        
        return {
            'avg_steps': statistics.mean(session_lengths) if session_lengths else 0,
            'median_steps': statistics.median(session_lengths) if session_lengths else 0,
            'completion_rate': completion_count / total_sessions if total_sessions > 0 else 0,
            'error_rate': error_count / total_interactions if total_interactions > 0 else 0,
            'bounce_rate': sum(1 for length in session_lengths if length == 1) / total_sessions if total_sessions > 0 else 0
        }


class InteractionPatternsCollector(BaseCollector):
    """Collecteur de patterns d'interaction."""
    
    async def collect(self) -> Dict[str, Any]:
        """Collecte les patterns d'interaction."""
        tenant_id = self.config.tags.get('tenant_id', 'default')
        
        try:
            # Patterns temporels
            temporal_patterns = await self._analyze_temporal_patterns(tenant_id)
            
            # Patterns de fonctionnalités
            feature_patterns = await self._analyze_feature_usage_patterns(tenant_id)
            
            # Patterns de navigation
            navigation_patterns = await self._analyze_navigation_patterns(tenant_id)
            
            # Patterns d'engagement
            engagement_patterns = await self._analyze_engagement_patterns(tenant_id)
            
            # Patterns de conversion
            conversion_patterns = await self._analyze_conversion_patterns(tenant_id)
            
            return {
                'interaction_patterns': {
                    'tenant_id': tenant_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'temporal': temporal_patterns,
                    'features': feature_patterns,
                    'navigation': navigation_patterns,
                    'engagement': engagement_patterns,
                    'conversion': conversion_patterns,
                    'pattern_score': self._calculate_pattern_score(
                        temporal_patterns, feature_patterns, engagement_patterns
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur collecte patterns interaction: {str(e)}")
            raise
    
    async def _analyze_temporal_patterns(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse les patterns temporels d'utilisation."""
        # Simulation de données temporelles
        hourly_usage = {str(i): np.random.poisson(50) for i in range(24)}
        daily_usage = {
            'monday': 1200,
            'tuesday': 1350,
            'wednesday': 1400,
            'thursday': 1380,
            'friday': 1500,
            'saturday': 800,
            'sunday': 600
        }
        
        # Identification des pics d'activité
        peak_hours = sorted(hourly_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_days = sorted(daily_usage.items(), key=lambda x: x[1], reverse=True)[:2]
        
        return {
            'hourly_distribution': hourly_usage,
            'daily_distribution': daily_usage,
            'peak_hours': [hour for hour, _ in peak_hours],
            'peak_days': [day for day, _ in peak_days],
            'weekly_pattern': 'weekday_heavy',  # Pattern détecté
            'seasonal_trend': 'stable',
            'activity_index': 0.73  # Score d'activité globale
        }
    
    async def _analyze_feature_usage_patterns(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse les patterns d'usage des fonctionnalités."""
        features_usage = {
            'music_generation': {
                'usage_count': 1456,
                'unique_users': 423,
                'avg_session_duration': 284.5,
                'satisfaction_score': 4.2,
                'completion_rate': 0.87,
                'adoption_rate': 0.34
            },
            'collaboration_tools': {
                'usage_count': 892,
                'unique_users': 267,
                'avg_session_duration': 512.3,
                'satisfaction_score': 4.1,
                'completion_rate': 0.79,
                'adoption_rate': 0.21
            },
            'analytics_dashboard': {
                'usage_count': 634,
                'unique_users': 198,
                'avg_session_duration': 156.7,
                'satisfaction_score': 3.8,
                'completion_rate': 0.92,
                'adoption_rate': 0.16
            },
            'spotify_integration': {
                'usage_count': 2134,
                'unique_users': 567,
                'avg_session_duration': 89.2,
                'satisfaction_score': 4.5,
                'completion_rate': 0.94,
                'adoption_rate': 0.45
            }
        }
        
        # Analyse des sequences d'usage
        feature_sequences = [
            ['spotify_integration', 'music_generation', 'collaboration_tools'],
            ['analytics_dashboard', 'music_generation'],
            ['music_generation', 'spotify_integration']
        ]
        
        # Features les plus populaires
        popular_features = sorted(
            features_usage.items(),
            key=lambda x: x[1]['usage_count'],
            reverse=True
        )
        
        return {
            'features_usage': features_usage,
            'popular_features': [name for name, _ in popular_features[:3]],
            'feature_sequences': feature_sequences,
            'cross_feature_usage': 0.67,  # Taux d'usage multi-fonctionnalités
            'feature_stickiness': 0.78,   # Taux de réutilisation
            'discovery_rate': 0.23        # Taux de découverte de nouvelles features
        }
    
    async def _analyze_navigation_patterns(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse les patterns de navigation."""
        navigation_flows = {
            'common_paths': [
                {'path': ['/home', '/features', '/music-gen'], 'frequency': 234},
                {'path': ['/home', '/spotify', '/sync'], 'frequency': 189},
                {'path': ['/features', '/collab', '/invite'], 'frequency': 156},
                {'path': ['/analytics', '/dashboard', '/export'], 'frequency': 89}
            ],
            'entry_pages': {
                '/home': 0.45,
                '/features': 0.23,
                '/spotify': 0.18,
                '/analytics': 0.14
            },
            'exit_pages': {
                '/music-gen': 0.34,
                '/dashboard': 0.28,
                '/collab': 0.21,
                '/sync': 0.17
            }
        }
        
        # Métriques de navigation
        navigation_metrics = {
            'avg_pages_per_session': 4.7,
            'navigation_efficiency': 0.73,  # Ratio chemin optimal/réel
            'backtrack_rate': 0.12,         # Taux de retour en arrière
            'deep_page_reach': 0.34,        # Taux d'accès pages profondes
            'navigation_satisfaction': 4.1   # Score de satisfaction navigation
        }
        
        return {
            'flows': navigation_flows,
            'metrics': navigation_metrics,
            'optimization_opportunities': [
                {
                    'area': 'reduce_backtracking',
                    'current_rate': 0.12,
                    'target_rate': 0.08,
                    'impact': 'medium'
                },
                {
                    'area': 'improve_deep_navigation',
                    'current_rate': 0.34,
                    'target_rate': 0.45,
                    'impact': 'high'
                }
            ]
        }
    
    async def _analyze_engagement_patterns(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse les patterns d'engagement."""
        engagement_segments = {
            'highly_engaged': {
                'percentage': 0.18,
                'avg_session_duration': 847.5,
                'sessions_per_week': 12.3,
                'features_used': 6.7,
                'conversion_rate': 0.87
            },
            'moderately_engaged': {
                'percentage': 0.45,
                'avg_session_duration': 324.2,
                'sessions_per_week': 4.8,
                'features_used': 3.2,
                'conversion_rate': 0.34
            },
            'low_engaged': {
                'percentage': 0.37,
                'avg_session_duration': 89.7,
                'sessions_per_week': 1.2,
                'features_used': 1.1,
                'conversion_rate': 0.05
            }
        }
        
        # Triggers d'engagement
        engagement_triggers = {
            'social_sharing': 0.23,       # Impact sur l'engagement
            'tutorial_completion': 0.34,
            'first_creation': 0.67,
            'collaboration_invite': 0.45,
            'achievement_unlock': 0.28
        }
        
        # Evolution de l'engagement
        engagement_evolution = {
            'week_1': 0.67,
            'week_2': 0.58,
            'week_3': 0.52,
            'week_4': 0.48,
            'month_2': 0.41,
            'month_3': 0.38
        }
        
        return {
            'segments': engagement_segments,
            'triggers': engagement_triggers,
            'evolution': engagement_evolution,
            'engagement_score': 0.64,
            'retention_correlation': 0.82,  # Corrélation engagement-rétention
            'predictive_indicators': [
                'session_frequency',
                'feature_diversity',
                'social_interactions',
                'content_creation'
            ]
        }
    
    async def _analyze_conversion_patterns(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse les patterns de conversion."""
        conversion_funnels = {
            'signup_to_trial': {
                'conversion_rate': 0.78,
                'avg_time_hours': 2.3,
                'drop_off_points': ['email_verification', 'profile_setup']
            },
            'trial_to_paid': {
                'conversion_rate': 0.23,
                'avg_time_days': 12.5,
                'drop_off_points': ['feature_discovery', 'pricing_page']
            },
            'free_to_premium': {
                'conversion_rate': 0.15,
                'avg_time_days': 28.7,
                'drop_off_points': ['usage_limits', 'payment_form']
            }
        }
        
        # Facteurs de conversion
        conversion_factors = {
            'tutorial_completion': 1.45,    # Multiplicateur de conversion
            'social_proof_exposure': 1.23,
            'feature_usage_breadth': 1.67,
            'support_interaction': 0.89,
            'pricing_page_views': 0.76
        }
        
        # Segments de conversion
        conversion_segments = {
            'fast_converters': {
                'percentage': 0.12,
                'avg_conversion_time_days': 3.2,
                'characteristics': ['high_engagement', 'feature_explorer']
            },
            'slow_converters': {
                'percentage': 0.34,
                'avg_conversion_time_days': 45.6,
                'characteristics': ['cautious', 'price_sensitive']
            },
            'non_converters': {
                'percentage': 0.54,
                'avg_trial_length_days': 14.0,
                'characteristics': ['low_engagement', 'feature_limited']
            }
        }
        
        return {
            'funnels': conversion_funnels,
            'factors': conversion_factors,
            'segments': conversion_segments,
            'overall_conversion_rate': 0.19,
            'optimization_potential': 0.31,
            'revenue_impact': {
                'current_monthly': 45678.90,
                'potential_monthly': 67890.12,
                'uplift_percentage': 48.6
            }
        }
    
    def _calculate_pattern_score(self, temporal: Dict, features: Dict, 
                               engagement: Dict) -> float:
        """Calcule un score global des patterns."""
        # Score d'activité temporelle
        activity_score = temporal.get('activity_index', 0) * 25
        
        # Score d'usage des fonctionnalités
        feature_score = features.get('cross_feature_usage', 0) * 30
        
        # Score d'engagement
        engagement_score = engagement.get('engagement_score', 0) * 45
        
        total_score = activity_score + feature_score + engagement_score
        return round(total_score, 2)
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Valide les données de patterns d'interaction."""
        try:
            patterns_data = data.get('interaction_patterns', {})
            
            required_sections = ['temporal', 'features', 'engagement']
            for section in required_sections:
                if section not in patterns_data:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation patterns interaction: {str(e)}")
            return False


class PreferenceEvolutionCollector(BaseCollector):
    """Collecteur d'évolution des préférences utilisateur."""
    
    async def collect(self) -> Dict[str, Any]:
        """Collecte l'évolution des préférences."""
        tenant_id = self.config.tags.get('tenant_id', 'default')
        
        try:
            # Evolution des préférences musicales
            music_preferences = await self._analyze_music_preference_evolution(tenant_id)
            
            # Evolution des préférences de fonctionnalités
            feature_preferences = await self._analyze_feature_preference_evolution(tenant_id)
            
            # Evolution des préférences de contenu
            content_preferences = await self._analyze_content_preference_evolution(tenant_id)
            
            # Prédiction d'évolution
            preference_predictions = await self._predict_preference_evolution(tenant_id)
            
            return {
                'preference_evolution': {
                    'tenant_id': tenant_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'music_preferences': music_preferences,
                    'feature_preferences': feature_preferences,
                    'content_preferences': content_preferences,
                    'predictions': preference_predictions,
                    'stability_score': self._calculate_preference_stability(
                        music_preferences, feature_preferences, content_preferences
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur collecte évolution préférences: {str(e)}")
            raise
    
    async def _analyze_music_preference_evolution(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse l'évolution des préférences musicales."""
        # Simulation d'évolution des genres sur 6 mois
        monthly_preferences = {
            'month_1': {'electronic': 0.35, 'pop': 0.25, 'rock': 0.20, 'jazz': 0.15, 'classical': 0.05},
            'month_2': {'electronic': 0.38, 'pop': 0.23, 'rock': 0.18, 'jazz': 0.16, 'classical': 0.05},
            'month_3': {'electronic': 0.42, 'pop': 0.21, 'rock': 0.16, 'jazz': 0.18, 'classical': 0.03},
            'month_4': {'electronic': 0.40, 'pop': 0.22, 'rock': 0.17, 'jazz': 0.17, 'classical': 0.04},
            'month_5': {'electronic': 0.44, 'pop': 0.20, 'rock': 0.15, 'jazz': 0.19, 'classical': 0.02},
            'month_6': {'electronic': 0.46, 'pop': 0.19, 'rock': 0.14, 'jazz': 0.20, 'classical': 0.01}
        }
        
        # Calcul des tendances
        trends = {}
        for genre in ['electronic', 'pop', 'rock', 'jazz', 'classical']:
            month1_pref = monthly_preferences['month_1'][genre]
            month6_pref = monthly_preferences['month_6'][genre]
            change = month6_pref - month1_pref
            trends[genre] = {
                'change': change,
                'trend': 'increasing' if change > 0.02 else 'decreasing' if change < -0.02 else 'stable'
            }
        
        # Nouvelles découvertes
        discovery_metrics = {
            'new_genres_explored': 2.3,    # Moyenne par utilisateur
            'genre_diversity_score': 0.73,
            'exploration_rate': 0.15,      # Pourcentage de nouvelles écoutes
            'recommendation_influence': 0.45  # Impact des recommandations
        }
        
        return {
            'monthly_evolution': monthly_preferences,
            'trends': trends,
            'discovery': discovery_metrics,
            'preference_volatility': 0.12,  # Stabilité des préférences
            'emerging_interests': ['ambient', 'lo-fi', 'synthwave'],
            'declining_interests': ['classical', 'folk']
        }
    
    async def _analyze_feature_preference_evolution(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse l'évolution des préférences de fonctionnalités."""
        feature_evolution = {
            'music_generation': {
                'initial_usage': 0.23,
                'current_usage': 0.67,
                'trend': 'strong_growth',
                'satisfaction_evolution': [3.8, 4.0, 4.2, 4.3, 4.2, 4.4]
            },
            'collaboration': {
                'initial_usage': 0.45,
                'current_usage': 0.52,
                'trend': 'moderate_growth',
                'satisfaction_evolution': [4.1, 4.2, 4.0, 4.1, 4.2, 4.1]
            },
            'analytics': {
                'initial_usage': 0.12,
                'current_usage': 0.34,
                'trend': 'rapid_growth',
                'satisfaction_evolution': [3.5, 3.7, 3.9, 4.0, 3.8, 4.1]
            },
            'spotify_sync': {
                'initial_usage': 0.78,
                'current_usage': 0.89,
                'trend': 'steady_growth',
                'satisfaction_evolution': [4.3, 4.4, 4.5, 4.6, 4.5, 4.6]
            }
        }
        
        # Adoption patterns
        adoption_patterns = {
            'early_adopters': 0.18,      # Pourcentage d'early adopters
            'feature_champions': 0.12,   # Utilisateurs influents
            'laggards': 0.25,           # Adopteurs tardifs
            'feature_drop_rate': 0.08    # Taux d'abandon de features
        }
        
        return {
            'evolution': feature_evolution,
            'adoption_patterns': adoption_patterns,
            'feature_lifecycle': {
                'emerging': ['voice_commands', 'ai_mastering'],
                'growing': ['analytics', 'music_generation'],
                'mature': ['spotify_sync', 'collaboration'],
                'declining': ['basic_editor']
            },
            'cross_feature_correlation': 0.67  # Corrélation d'usage entre features
        }
    
    async def _analyze_content_preference_evolution(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse l'évolution des préférences de contenu."""
        content_evolution = {
            'content_types': {
                'original_compositions': {
                    'initial_preference': 0.34,
                    'current_preference': 0.56,
                    'growth_rate': 0.64
                },
                'remixes': {
                    'initial_preference': 0.45,
                    'current_preference': 0.38,
                    'growth_rate': -0.16
                },
                'collaborations': {
                    'initial_preference': 0.21,
                    'current_preference': 0.41,
                    'growth_rate': 0.95
                },
                'samples': {
                    'initial_preference': 0.56,
                    'current_preference': 0.48,
                    'growth_rate': -0.14
                }
            },
            'complexity_preferences': {
                'simple_structures': 0.32,
                'moderate_complexity': 0.51,
                'complex_arrangements': 0.17
            },
            'duration_preferences': {
                'short_clips': 0.28,      # < 1 min
                'standard_length': 0.54,  # 1-4 min
                'extended_pieces': 0.18   # > 4 min
            }
        }
        
        # Influences sur les préférences
        preference_influences = {
            'social_trends': 0.34,
            'recommendation_system': 0.28,
            'peer_influence': 0.23,
            'seasonal_patterns': 0.15
        }
        
        # Segmentation par préférences
        preference_segments = {
            'experimental': {
                'percentage': 0.22,
                'characteristics': ['high_novelty_seeking', 'diverse_tastes']
            },
            'mainstream': {
                'percentage': 0.51,
                'characteristics': ['popular_genres', 'trend_following']
            },
            'traditional': {
                'percentage': 0.27,
                'characteristics': ['stable_preferences', 'genre_loyal']
            }
        }
        
        return {
            'content_evolution': content_evolution,
            'influences': preference_influences,
            'segments': preference_segments,
            'personalization_effectiveness': 0.73,
            'preference_maturity': 0.68  # Stabilité des préférences mûres
        }
    
    async def _predict_preference_evolution(self, tenant_id: str) -> Dict[str, Any]:
        """Prédit l'évolution future des préférences."""
        # Prédictions basées sur les tendances actuelles
        predictions = {
            'next_month': {
                'electronic_music': 0.48,    # Augmentation prédite
                'ai_generated_content': 0.73,
                'collaborative_features': 0.58,
                'mobile_usage': 0.82
            },
            'next_quarter': {
                'voice_interfaces': 0.34,
                'real_time_collaboration': 0.67,
                'advanced_analytics': 0.45,
                'social_features': 0.71
            },
            'confidence_scores': {
                'music_preferences': 0.78,
                'feature_adoption': 0.85,
                'content_types': 0.72,
                'usage_patterns': 0.81
            }
        }
        
        # Facteurs d'influence prédits
        influence_factors = {
            'technological_advancement': 0.45,
            'market_trends': 0.32,
            'user_feedback': 0.28,
            'competitive_landscape': 0.23
        }
        
        return {
            'predictions': predictions,
            'influence_factors': influence_factors,
            'prediction_horizon_days': 90,
            'model_accuracy': 0.76,
            'risk_factors': [
                'market_disruption',
                'technology_shift',
                'user_behavior_change'
            ]
        }
    
    def _calculate_preference_stability(self, music: Dict, features: Dict, 
                                      content: Dict) -> float:
        """Calcule un score de stabilité des préférences."""
        # Score basé sur la volatilité des préférences
        music_stability = 1 - music.get('preference_volatility', 0)
        feature_stability = features.get('cross_feature_correlation', 0)
        content_stability = content.get('preference_maturity', 0)
        
        # Score composite
        overall_stability = (
            music_stability * 0.4 +
            feature_stability * 0.35 +
            content_stability * 0.25
        )
        
        return round(overall_stability, 3)
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Valide les données d'évolution des préférences."""
        try:
            pref_data = data.get('preference_evolution', {})
            
            required_sections = ['music_preferences', 'feature_preferences', 'content_preferences']
            for section in required_sections:
                if section not in pref_data:
                    return False
            
            # Validation du score de stabilité
            stability_score = pref_data.get('stability_score', -1)
            if not (0 <= stability_score <= 1):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation évolution préférences: {str(e)}")
            return False


class FunnelAnalyzer:
    """Analyseur de funnel de conversion."""
    
    async def analyze_funnels(self, tenant_id: str, 
                            interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Analyse les funnels de conversion."""
        # Définition des étapes du funnel principal
        main_funnel_steps = [
            'landing',
            'signup',
            'onboarding',
            'first_feature_use',
            'trial_start',
            'paid_conversion'
        ]
        
        # Simulation d'analyse de funnel
        funnel_data = {
            'landing': 10000,
            'signup': 2300,
            'onboarding': 1950,
            'first_feature_use': 1456,
            'trial_start': 892,
            'paid_conversion': 156
        }
        
        # Calcul des taux de conversion
        conversion_rates = {}
        drop_off_points = []
        
        for i in range(1, len(main_funnel_steps)):
            current_step = main_funnel_steps[i]
            previous_step = main_funnel_steps[i-1]
            
            current_count = funnel_data[current_step]
            previous_count = funnel_data[previous_step]
            
            conversion_rate = current_count / previous_count if previous_count > 0 else 0
            conversion_rates[f"{previous_step}_to_{current_step}"] = conversion_rate
            
            # Détection des points d'abandon
            drop_rate = 1 - conversion_rate
            if drop_rate > 0.3:  # Seuil de drop-off important
                drop_off_points.append({
                    'step_name': current_step,
                    'drop_rate': drop_rate,
                    'users_lost': previous_count - current_count,
                    'severity': 'high' if drop_rate > 0.5 else 'medium'
                })
        
        # Conversion globale
        overall_conversion = funnel_data['paid_conversion'] / funnel_data['landing']
        
        return {
            'funnel_steps': main_funnel_steps,
            'funnel_data': funnel_data,
            'conversion_rates': conversion_rates,
            'overall_conversion_rate': overall_conversion,
            'drop_off_points': drop_off_points,
            'optimization_potential': max(0, 0.15 - overall_conversion)  # Benchmark 15%
        }


class CohortAnalyzer:
    """Analyseur de cohortes utilisateur."""
    
    async def analyze_cohorts(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse les cohortes d'utilisateurs."""
        # Simulation d'analyse de cohortes
        cohort_retention = {
            '2024-01': {
                'size': 156,
                'day_1': 0.89,
                'day_7': 0.67,
                'day_30': 0.34,
                'day_90': 0.23
            },
            '2024-02': {
                'size': 189,
                'day_1': 0.91,
                'day_7': 0.72,
                'day_30': 0.41,
                'day_90': 0.28
            },
            '2024-03': {
                'size': 234,
                'day_1': 0.93,
                'day_7': 0.76,
                'day_30': 0.45,
                'day_90': None  # Pas encore disponible
            }
        }
        
        # Métriques de cohorte agrégées
        avg_retention_7_days = statistics.mean([
            cohort['day_7'] for cohort in cohort_retention.values()
        ])
        
        avg_retention_30_days = statistics.mean([
            cohort['day_30'] for cohort in cohort_retention.values()
        ])
        
        return {
            'cohort_data': cohort_retention,
            'retention_7_days': avg_retention_7_days,
            'retention_30_days': avg_retention_30_days,
            'cohort_trends': 'improving',  # Tendance détectée
            'retention_benchmark': 0.35,   # Benchmark industrie
            'performance_vs_benchmark': avg_retention_30_days - 0.35
        }


class BehaviorPredictor:
    """Prédicteur de comportement utilisateur."""
    
    async def predict_behaviors(self, tenant_id: str, 
                              interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Prédit les comportements futurs."""
        # Simulation de prédictions ML
        predictions = {
            'churn_risk': {
                'high_risk_users': 67,
                'medium_risk_users': 134,
                'low_risk_users': 1045,
                'model_accuracy': 0.87
            },
            'conversion_probability': {
                'likely_converters': 89,
                'possible_converters': 156,
                'unlikely_converters': 1001,
                'model_accuracy': 0.82
            },
            'feature_adoption': {
                'early_adopters': 45,
                'mainstream_adopters': 234,
                'laggards': 967,
                'predicted_adoption_rate': 0.67
            },
            'engagement_forecast': {
                'increasing_engagement': 123,
                'stable_engagement': 890,
                'decreasing_engagement': 233,
                'forecast_accuracy': 0.78
            }
        }
        
        # Actions recommandées
        recommended_actions = [
            {
                'target_group': 'high_churn_risk',
                'action': 'personalized_retention_campaign',
                'expected_impact': 0.25,
                'priority': 'high'
            },
            {
                'target_group': 'likely_converters',
                'action': 'targeted_conversion_offers',
                'expected_impact': 0.35,
                'priority': 'high'
            },
            {
                'target_group': 'feature_laggards',
                'action': 'guided_feature_introduction',
                'expected_impact': 0.15,
                'priority': 'medium'
            }
        ]
        
        return {
            'predictions': predictions,
            'recommended_actions': recommended_actions,
            'prediction_confidence': 0.81,
            'model_last_updated': datetime.utcnow().isoformat(),
            'next_model_update': (datetime.utcnow() + timedelta(days=7)).isoformat()
        }


class ChurnPredictionCollector(BaseCollector):
    """Collecteur spécialisé pour la prédiction de churn."""
    
    async def collect(self) -> Dict[str, Any]:
        """Collecte les données de prédiction de churn."""
        tenant_id = self.config.tags.get('tenant_id', 'default')
        
        try:
            # Modèle de prédiction de churn
            churn_model_results = await self._run_churn_prediction_model(tenant_id)
            
            # Facteurs de risque
            risk_factors = await self._analyze_churn_risk_factors(tenant_id)
            
            # Segments de risque
            risk_segmentation = await self._segment_users_by_churn_risk(tenant_id)
            
            # Actions de rétention
            retention_actions = await self._recommend_retention_actions(tenant_id)
            
            return {
                'churn_prediction': {
                    'tenant_id': tenant_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'model_results': churn_model_results,
                    'risk_factors': risk_factors,
                    'risk_segmentation': risk_segmentation,
                    'retention_actions': retention_actions,
                    'overall_churn_risk': churn_model_results.get('predicted_churn_rate', 0),
                    'model_confidence': churn_model_results.get('confidence', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur collecte prédiction churn: {str(e)}")
            raise
    
    async def _run_churn_prediction_model(self, tenant_id: str) -> Dict[str, Any]:
        """Exécute le modèle de prédiction de churn."""
        # Simulation d'un modèle ML de churn
        return {
            'predicted_churn_rate': 0.087,  # 8.7% churn prédit
            'confidence': 0.89,
            'model_version': '2.1.0',
            'features_importance': {
                'days_since_last_login': 0.34,
                'engagement_score_trend': 0.28,
                'feature_usage_diversity': 0.22,
                'support_tickets_count': 0.16
            },
            'prediction_horizon_days': 30,
            'accuracy_metrics': {
                'precision': 0.87,
                'recall': 0.82,
                'f1_score': 0.84,
                'auc_roc': 0.91
            }
        }
    
    async def _analyze_churn_risk_factors(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse les facteurs de risque de churn."""
        return {
            'behavioral_factors': {
                'decreasing_login_frequency': 0.67,
                'reduced_feature_usage': 0.54,
                'shorter_session_duration': 0.43,
                'less_content_creation': 0.38
            },
            'engagement_factors': {
                'low_satisfaction_scores': 0.72,
                'negative_feedback': 0.58,
                'support_complaints': 0.45,
                'feature_abandonment': 0.41
            },
            'technical_factors': {
                'frequent_errors': 0.34,
                'slow_performance': 0.28,
                'mobile_app_issues': 0.23
            },
            'business_factors': {
                'pricing_concerns': 0.56,
                'competitor_activity': 0.34,
                'value_perception': 0.67
            }
        }
    
    async def _segment_users_by_churn_risk(self, tenant_id: str) -> Dict[str, Any]:
        """Segmente les utilisateurs par risque de churn."""
        return {
            'high_risk': {
                'count': 89,
                'percentage': 7.1,
                'characteristics': [
                    'inactive_for_7_days',
                    'low_engagement_score',
                    'support_complaints'
                ],
                'predicted_churn_rate': 0.78,
                'retention_priority': 'critical'
            },
            'medium_risk': {
                'count': 234,
                'percentage': 18.7,
                'characteristics': [
                    'declining_usage',
                    'feature_underutilization',
                    'price_sensitivity'
                ],
                'predicted_churn_rate': 0.34,
                'retention_priority': 'high'
            },
            'low_risk': {
                'count': 923,
                'percentage': 74.2,
                'characteristics': [
                    'regular_usage',
                    'high_satisfaction',
                    'feature_adoption'
                ],
                'predicted_churn_rate': 0.05,
                'retention_priority': 'maintenance'
            }
        }
    
    async def _recommend_retention_actions(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Recommande des actions de rétention."""
        return [
            {
                'target_segment': 'high_risk',
                'action_type': 'personal_outreach',
                'description': 'Personal call from customer success manager',
                'expected_retention_improvement': 0.45,
                'cost_per_user': 50.0,
                'urgency': 'immediate'
            },
            {
                'target_segment': 'medium_risk',
                'action_type': 'automated_campaign',
                'description': 'Email campaign with feature highlights and tips',
                'expected_retention_improvement': 0.25,
                'cost_per_user': 2.5,
                'urgency': 'within_week'
            },
            {
                'target_segment': 'high_risk',
                'action_type': 'discount_offer',
                'description': 'Targeted discount for subscription renewal',
                'expected_retention_improvement': 0.35,
                'cost_per_user': 25.0,
                'urgency': 'immediate'
            },
            {
                'target_segment': 'medium_risk',
                'action_type': 'feature_education',
                'description': 'Personalized tutorial for underused features',
                'expected_retention_improvement': 0.20,
                'cost_per_user': 5.0,
                'urgency': 'within_month'
            }
        ]
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Valide les données de prédiction de churn."""
        try:
            churn_data = data.get('churn_prediction', {})
            
            required_fields = ['model_results', 'risk_segmentation', 'overall_churn_risk']
            for field in required_fields:
                if field not in churn_data:
                    return False
            
            # Validation du taux de churn
            churn_rate = churn_data.get('overall_churn_risk', -1)
            if not (0 <= churn_rate <= 1):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation prédiction churn: {str(e)}")
            return False


__all__ = [
    'UserJourneyCollector',
    'InteractionPatternsCollector',
    'PreferenceEvolutionCollector',
    'ChurnPredictionCollector',
    'JourneyAnalyzer',
    'FunnelAnalyzer',
    'CohortAnalyzer',
    'BehaviorPredictor',
    'UserInteraction',
    'UserSession',
    'InteractionType',
    'UserSegment'
]
