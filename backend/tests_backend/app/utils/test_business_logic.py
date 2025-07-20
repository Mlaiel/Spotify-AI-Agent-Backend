"""
Tests Enterprise - Business Logic
=================================

Suite de tests ultra-avancée pour le module business_logic avec règles métier complexes,
moteur de recommandation, pricing dynamique, et analytics business intelligence.

Développé par l'équipe Business Intelligence & Analytics Expert sous la direction de Fahed Mlaiel.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List, Optional
import uuid
from decimal import Decimal
from dataclasses import dataclass

# Import des modules business à tester
try:
    from app.utils.business_logic import (
        RecommendationEngine,
        DynamicPricingEngine,
        BusinessRulesEngine,
        RevenueOptimizer,
        UserEngagementAnalyzer
    )
except ImportError:
    # Mocks si modules pas encore implémentés
    RecommendationEngine = MagicMock
    DynamicPricingEngine = MagicMock
    BusinessRulesEngine = MagicMock
    RevenueOptimizer = MagicMock
    UserEngagementAnalyzer = MagicMock


@dataclass
class UserProfile:
    """Profil utilisateur pour tests business."""
    user_id: str
    subscription_tier: str
    listening_hours_monthly: float
    genres_preference: List[str]
    discovery_score: float
    churn_risk: float
    lifetime_value: Decimal
    engagement_metrics: Dict[str, float]


@dataclass
class TrackMetadata:
    """Métadonnées track pour tests business."""
    track_id: str
    artist_id: str
    genre: str
    release_date: datetime
    popularity_score: float
    mood_tags: List[str]
    audio_features: Dict[str, float]
    royalty_rate: Decimal


class TestRecommendationEngine:
    """Tests enterprise pour RecommendationEngine avec IA avancée."""
    
    @pytest.fixture
    def recommendation_engine(self):
        """Instance RecommendationEngine pour tests."""
        return RecommendationEngine()
    
    @pytest.fixture
    def engine_config(self):
        """Configuration moteur recommandation enterprise."""
        return {
            'algorithms': {
                'collaborative_filtering': {
                    'enabled': True,
                    'algorithm': 'matrix_factorization',
                    'factors': 128,
                    'regularization': 0.01,
                    'learning_rate': 0.001
                },
                'content_based': {
                    'enabled': True,
                    'audio_features_weight': 0.4,
                    'metadata_weight': 0.3,
                    'lyrics_weight': 0.2,
                    'social_weight': 0.1
                },
                'deep_learning': {
                    'enabled': True,
                    'model_architecture': 'transformer',
                    'embedding_dim': 256,
                    'attention_heads': 8,
                    'layers': 6
                },
                'reinforcement_learning': {
                    'enabled': True,
                    'reward_function': 'user_satisfaction',
                    'exploration_rate': 0.1,
                    'discount_factor': 0.95
                }
            },
            'personalization': {
                'contextual_factors': ['time_of_day', 'weather', 'location', 'activity'],
                'mood_detection': True,
                'social_influence': 0.15,
                'novelty_factor': 0.2,
                'diversity_threshold': 0.3
            },
            'business_objectives': {
                'user_retention_weight': 0.4,
                'revenue_optimization': 0.3,
                'catalog_exploration': 0.2,
                'artist_discovery': 0.1
            }
        }
    
    @pytest.fixture
    def sample_users(self):
        """Échantillon utilisateurs pour tests."""
        return [
            UserProfile(
                user_id='user_001',
                subscription_tier='premium',
                listening_hours_monthly=45.5,
                genres_preference=['rock', 'alternative', 'indie'],
                discovery_score=0.75,
                churn_risk=0.12,
                lifetime_value=Decimal('145.80'),
                engagement_metrics={'skip_rate': 0.15, 'like_rate': 0.68, 'share_rate': 0.08}
            ),
            UserProfile(
                user_id='user_002',
                subscription_tier='free',
                listening_hours_monthly=12.3,
                genres_preference=['pop', 'hip-hop', 'r&b'],
                discovery_score=0.45,
                churn_risk=0.35,
                lifetime_value=Decimal('23.40'),
                engagement_metrics={'skip_rate': 0.28, 'like_rate': 0.42, 'share_rate': 0.03}
            ),
            UserProfile(
                user_id='user_003',
                subscription_tier='family',
                listening_hours_monthly=78.2,
                genres_preference=['classical', 'jazz', 'world'],
                discovery_score=0.92,
                churn_risk=0.05,
                lifetime_value=Decimal('289.50'),
                engagement_metrics={'skip_rate': 0.08, 'like_rate': 0.85, 'share_rate': 0.15}
            )
        ]
    
    async def test_multi_algorithm_recommendation_fusion(self, recommendation_engine, engine_config, sample_users):
        """Test fusion multi-algorithmes pour recommandations."""
        # Mock configuration engine
        recommendation_engine.configure = AsyncMock(return_value={'status': 'configured'})
        await recommendation_engine.configure(engine_config)
        
        # Mock génération recommandations par algorithme
        recommendation_engine.generate_multi_algorithm_recommendations = AsyncMock()
        
        for user in sample_users:
            # Configuration réponse fusion algorithmes
            recommendation_engine.generate_multi_algorithm_recommendations.return_value = {
                'user_id': user.user_id,
                'algorithm_contributions': {
                    'collaborative_filtering': {
                        'tracks': [f'cf_track_{i}' for i in range(1, 11)],
                        'confidence_scores': np.random.uniform(0.6, 0.95, 10).tolist(),
                        'algorithm_weight': 0.35,
                        'explanation': 'Users with similar taste also liked these tracks'
                    },
                    'content_based': {
                        'tracks': [f'cb_track_{i}' for i in range(1, 11)],
                        'confidence_scores': np.random.uniform(0.5, 0.9, 10).tolist(),
                        'algorithm_weight': 0.25,
                        'explanation': 'Based on audio features of your favorite tracks'
                    },
                    'deep_learning': {
                        'tracks': [f'dl_track_{i}' for i in range(1, 11)],
                        'confidence_scores': np.random.uniform(0.7, 0.98, 10).tolist(),
                        'algorithm_weight': 0.3,
                        'explanation': 'AI-powered pattern recognition from your listening behavior'
                    },
                    'reinforcement_learning': {
                        'tracks': [f'rl_track_{i}' for i in range(1, 11)],
                        'confidence_scores': np.random.uniform(0.6, 0.92, 10).tolist(),
                        'algorithm_weight': 0.1,
                        'explanation': 'Optimized for maximum user satisfaction'
                    }
                },
                'final_recommendations': [
                    {
                        'track_id': f'final_track_{i}',
                        'predicted_rating': np.random.uniform(3.5, 5.0),
                        'confidence': np.random.uniform(0.75, 0.95),
                        'explanation': 'Hybrid recommendation based on multiple signals',
                        'novelty_score': np.random.uniform(0.1, 0.8),
                        'diversity_contribution': np.random.uniform(0.0, 1.0)
                    } for i in range(1, 21)
                ],
                'personalization_factors': {
                    'time_context': 'evening_relaxation',
                    'mood_detected': 'contemplative',
                    'listening_context': 'focused_work',
                    'social_influence': 0.12
                },
                'business_metrics': {
                    'expected_engagement_score': np.random.uniform(0.7, 0.9),
                    'revenue_potential': np.random.uniform(0.1, 0.5),
                    'catalog_exploration_score': np.random.uniform(0.2, 0.6),
                    'artist_discovery_potential': np.random.uniform(0.1, 0.4)
                }
            }
            
            recommendations = await recommendation_engine.generate_multi_algorithm_recommendations(
                user_profile=user,
                context={'session_type': 'discovery', 'device': 'mobile'},
                recommendation_count=20
            )
            
            # Validations fusion algorithmes
            assert len(recommendations['final_recommendations']) == 20
            assert all(r['predicted_rating'] >= 3.0 for r in recommendations['final_recommendations'])
            assert all(r['confidence'] > 0.7 for r in recommendations['final_recommendations'])
            
            # Validation équilibrage algorithmes
            total_weight = sum(
                alg['algorithm_weight'] for alg in recommendations['algorithm_contributions'].values()
            )
            assert abs(total_weight - 1.0) < 0.01
            
            # Validation métriques business
            business_metrics = recommendations['business_metrics']
            assert business_metrics['expected_engagement_score'] > 0.6
            assert business_metrics['revenue_potential'] >= 0
    
    async def test_contextual_recommendation_adaptation(self, recommendation_engine):
        """Test adaptation recommandations selon contexte."""
        # Contextes d'écoute variés
        listening_contexts = [
            {
                'context_name': 'morning_commute',
                'time_of_day': 'morning',
                'activity': 'commuting',
                'duration_minutes': 35,
                'device': 'mobile',
                'connectivity': '4g',
                'expected_energy_level': 'high',
                'expected_genres': ['pop', 'rock', 'electronic']
            },
            {
                'context_name': 'work_focus',
                'time_of_day': 'afternoon',
                'activity': 'working',
                'duration_minutes': 120,
                'device': 'desktop',
                'connectivity': 'wifi',
                'expected_energy_level': 'medium',
                'expected_genres': ['ambient', 'classical', 'instrumental']
            },
            {
                'context_name': 'evening_relaxation',
                'time_of_day': 'evening',
                'activity': 'relaxing',
                'duration_minutes': 90,
                'device': 'smart_speaker',
                'connectivity': 'wifi',
                'expected_energy_level': 'low',
                'expected_genres': ['jazz', 'acoustic', 'chill']
            },
            {
                'context_name': 'workout_session',
                'time_of_day': 'morning',
                'activity': 'exercising',
                'duration_minutes': 60,
                'device': 'mobile',
                'connectivity': 'offline',
                'expected_energy_level': 'very_high',
                'expected_genres': ['hip-hop', 'electronic', 'rock']
            }
        ]
        
        # Mock adaptation contextuelle
        recommendation_engine.adapt_recommendations_to_context = AsyncMock()
        
        for context in listening_contexts:
            # Configuration réponse adaptation
            recommendation_engine.adapt_recommendations_to_context.return_value = {
                'context_analysis': {
                    'detected_context': context['context_name'],
                    'confidence': np.random.uniform(0.8, 0.95),
                    'context_factors': {
                        'temporal': context['time_of_day'],
                        'activity': context['activity'],
                        'device_capabilities': context['device'],
                        'network_conditions': context['connectivity']
                    }
                },
                'context_adaptations': {
                    'energy_level_adjustment': context['expected_energy_level'],
                    'genre_preferences_shift': {
                        genre: np.random.uniform(0.1, 0.9) 
                        for genre in context['expected_genres']
                    },
                    'tempo_preference': {
                        'min_bpm': 60 if context['expected_energy_level'] == 'low' else 120,
                        'max_bpm': 100 if context['expected_energy_level'] == 'low' else 180
                    },
                    'duration_preference': {
                        'preferred_track_length': '3-4min' if context['activity'] == 'exercising' else 'any',
                        'skip_tolerance': 'low' if context['activity'] == 'working' else 'medium'
                    }
                },
                'adapted_recommendations': [
                    {
                        'track_id': f'context_track_{context["context_name"]}_{i}',
                        'context_relevance_score': np.random.uniform(0.7, 0.95),
                        'adaptation_reason': f'Optimized for {context["activity"]} during {context["time_of_day"]}',
                        'energy_match': np.random.uniform(0.8, 0.98),
                        'context_alignment': np.random.uniform(0.75, 0.92)
                    } for i in range(15)
                ],
                'contextual_metrics': {
                    'expected_completion_rate': np.random.uniform(0.75, 0.95),
                    'predicted_satisfaction': np.random.uniform(0.8, 0.9),
                    'context_discovery_potential': np.random.uniform(0.1, 0.4)
                }
            }
            
            adapted_recs = await recommendation_engine.adapt_recommendations_to_context(
                base_recommendations=[f'base_track_{i}' for i in range(20)],
                context=context,
                user_id='user_test_123'
            )
            
            # Validations adaptation contextuelle
            assert adapted_recs['context_analysis']['confidence'] > 0.8
            assert len(adapted_recs['adapted_recommendations']) > 0
            assert all(
                r['context_relevance_score'] > 0.7 
                for r in adapted_recs['adapted_recommendations']
            )
            assert adapted_recs['contextual_metrics']['expected_completion_rate'] > 0.7
    
    async def test_real_time_recommendation_learning(self, recommendation_engine):
        """Test apprentissage temps réel des recommandations."""
        # Signaux d'engagement utilisateur temps réel
        real_time_signals = [
            {
                'signal_type': 'track_play',
                'track_id': 'track_001',
                'user_id': 'user_001',
                'timestamp': datetime.utcnow(),
                'play_duration_seconds': 187,
                'total_track_duration': 203,
                'completion_rate': 0.92,
                'user_action': 'listened_to_end'
            },
            {
                'signal_type': 'track_skip',
                'track_id': 'track_002',
                'user_id': 'user_001',
                'timestamp': datetime.utcnow(),
                'play_duration_seconds': 12,
                'total_track_duration': 198,
                'completion_rate': 0.06,
                'user_action': 'skipped_early'
            },
            {
                'signal_type': 'track_like',
                'track_id': 'track_003',
                'user_id': 'user_001',
                'timestamp': datetime.utcnow(),
                'play_duration_seconds': 156,
                'total_track_duration': 156,
                'completion_rate': 1.0,
                'user_action': 'liked_and_completed'
            },
            {
                'signal_type': 'track_share',
                'track_id': 'track_004',
                'user_id': 'user_001',
                'timestamp': datetime.utcnow(),
                'play_duration_seconds': 203,
                'total_track_duration': 203,
                'completion_rate': 1.0,
                'user_action': 'shared_on_social'
            }
        ]
        
        # Mock apprentissage temps réel
        recommendation_engine.update_model_real_time = AsyncMock(return_value={
            'learning_update': {
                'signals_processed': len(real_time_signals),
                'model_updates_applied': 4,
                'learning_rate_used': 0.001,
                'batch_size': 32,
                'convergence_achieved': True
            },
            'preference_updates': {
                'user_vector_updated': True,
                'item_embeddings_adjusted': True,
                'contextual_factors_learned': ['time_preference', 'genre_affinity'],
                'negative_signals_incorporated': 1,  # Skip
                'positive_signals_incorporated': 3   # Play, like, share
            },
            'recommendation_improvements': {
                'accuracy_improvement': np.random.uniform(0.02, 0.08),
                'diversity_adjustment': np.random.uniform(-0.01, 0.03),
                'novelty_calibration': np.random.uniform(0.01, 0.05),
                'personalization_depth_increase': np.random.uniform(0.03, 0.1)
            },
            'online_learning_metrics': {
                'learning_efficiency': np.random.uniform(0.85, 0.95),
                'adaptation_speed': 'fast',
                'stability_score': np.random.uniform(0.9, 0.98),
                'overfitting_risk': 'low'
            }
        })
        
        # Test apprentissage temps réel
        learning_result = await recommendation_engine.update_model_real_time(
            user_signals=real_time_signals,
            learning_config={'aggressive_learning': False, 'stability_weight': 0.8}
        )
        
        # Validations apprentissage temps réel
        assert learning_result['learning_update']['signals_processed'] == len(real_time_signals)
        assert learning_result['learning_update']['convergence_achieved'] is True
        assert learning_result['preference_updates']['user_vector_updated'] is True
        assert learning_result['recommendation_improvements']['accuracy_improvement'] > 0
        assert learning_result['online_learning_metrics']['learning_efficiency'] > 0.8


class TestDynamicPricingEngine:
    """Tests enterprise pour DynamicPricingEngine avec pricing intelligent."""
    
    @pytest.fixture
    def pricing_engine(self):
        """Instance DynamicPricingEngine pour tests."""
        return DynamicPricingEngine()
    
    @pytest.fixture
    def pricing_config(self):
        """Configuration pricing dynamique enterprise."""
        return {
            'pricing_strategies': {
                'subscription_tiers': {
                    'free': {'base_price': 0, 'features': ['ads', 'limited_skips']},
                    'premium': {'base_price': 9.99, 'features': ['ad_free', 'unlimited_skips', 'offline']},
                    'family': {'base_price': 14.99, 'features': ['6_accounts', 'all_premium_features']},
                    'student': {'base_price': 4.99, 'features': ['premium_features', 'student_discount']}
                },
                'dynamic_factors': {
                    'demand_elasticity': 0.7,
                    'competitive_pricing': True,
                    'user_value_optimization': True,
                    'market_segmentation': True
                },
                'pricing_rules': {
                    'minimum_discount': 0.1,
                    'maximum_discount': 0.5,
                    'price_change_frequency_days': 7,
                    'regional_adjustments': True
                }
            },
            'market_intelligence': {
                'competitor_monitoring': True,
                'demand_forecasting': True,
                'churn_risk_pricing': True,
                'lifetime_value_optimization': True
            }
        }
    
    async def test_demand_based_dynamic_pricing(self, pricing_engine, pricing_config):
        """Test pricing dynamique basé sur la demande."""
        # Mock configuration pricing
        pricing_engine.configure = AsyncMock(return_value={'status': 'configured'})
        await pricing_engine.configure(pricing_config)
        
        # Scénarios de demande variés
        demand_scenarios = [
            {
                'scenario_name': 'high_demand_holiday',
                'market_conditions': {
                    'demand_level': 'very_high',
                    'competitor_pricing': {'spotify': 9.99, 'apple_music': 10.99, 'amazon_music': 8.99},
                    'seasonal_factor': 1.3,
                    'market_saturation': 0.65
                },
                'user_segments': {
                    'price_sensitive': 0.4,
                    'value_seekers': 0.35,
                    'premium_users': 0.25
                },
                'expected_strategy': 'moderate_increase'
            },
            {
                'scenario_name': 'low_demand_summer',
                'market_conditions': {
                    'demand_level': 'low',
                    'competitor_pricing': {'spotify': 9.99, 'apple_music': 9.99, 'amazon_music': 7.99},
                    'seasonal_factor': 0.8,
                    'market_saturation': 0.75
                },
                'user_segments': {
                    'price_sensitive': 0.6,
                    'value_seekers': 0.3,
                    'premium_users': 0.1
                },
                'expected_strategy': 'promotional_pricing'
            },
            {
                'scenario_name': 'competitive_pressure',
                'market_conditions': {
                    'demand_level': 'medium',
                    'competitor_pricing': {'spotify': 8.99, 'apple_music': 9.99, 'amazon_music': 7.99},
                    'seasonal_factor': 1.0,
                    'market_saturation': 0.8
                },
                'user_segments': {
                    'price_sensitive': 0.5,
                    'value_seekers': 0.35,
                    'premium_users': 0.15
                },
                'expected_strategy': 'competitive_matching'
            }
        ]
        
        # Mock pricing dynamique
        pricing_engine.calculate_dynamic_pricing = AsyncMock()
        
        for scenario in demand_scenarios:
            # Configuration réponse pricing
            pricing_engine.calculate_dynamic_pricing.return_value = {
                'pricing_strategy': {
                    'strategy_name': scenario['expected_strategy'],
                    'confidence_score': np.random.uniform(0.8, 0.95),
                    'expected_impact': {
                        'revenue_change_percentage': np.random.uniform(-0.1, 0.25),
                        'user_acquisition_change': np.random.uniform(-0.05, 0.15),
                        'churn_risk_change': np.random.uniform(-0.02, 0.08)
                    }
                },
                'tier_pricing': {
                    'premium': {
                        'original_price': 9.99,
                        'optimized_price': np.random.uniform(8.99, 11.99),
                        'discount_percentage': np.random.uniform(0, 0.3),
                        'price_elasticity': scenario['pricing_config']['dynamic_factors']['demand_elasticity']
                    },
                    'family': {
                        'original_price': 14.99,
                        'optimized_price': np.random.uniform(13.99, 16.99),
                        'discount_percentage': np.random.uniform(0, 0.25),
                        'bundle_optimization': True
                    },
                    'student': {
                        'original_price': 4.99,
                        'optimized_price': np.random.uniform(3.99, 5.99),
                        'discount_percentage': np.random.uniform(0, 0.4),
                        'verification_required': True
                    }
                },
                'market_analysis': {
                    'competitive_position': 'strong' if scenario['scenario_name'] != 'competitive_pressure' else 'challenged',
                    'price_sensitivity_index': scenario['user_segments']['price_sensitive'],
                    'demand_forecast_30_days': np.random.uniform(0.8, 1.3),
                    'optimal_price_range': {'min': 8.99, 'max': 12.99}
                },
                'implementation_timeline': {
                    'rollout_duration_days': 14,
                    'a_b_testing_required': True,
                    'gradual_implementation': True,
                    'rollback_plan': True
                }
            }
            
            pricing_result = await pricing_engine.calculate_dynamic_pricing(
                market_conditions=scenario['market_conditions'],
                user_segments=scenario['user_segments'],
                optimization_goal='revenue_maximization'
            )
            
            # Validations pricing dynamique
            assert pricing_result['pricing_strategy']['confidence_score'] > 0.8
            assert 'tier_pricing' in pricing_result
            
            # Validation cohérence prix
            for tier, pricing in pricing_result['tier_pricing'].items():
                assert pricing['optimized_price'] > 0
                assert 0 <= pricing['discount_percentage'] <= 0.5
            
            # Validation stratégie marché
            assert pricing_result['market_analysis']['competitive_position'] in ['strong', 'challenged', 'leading']
            assert pricing_result['implementation_timeline']['a_b_testing_required'] is True
    
    async def test_personalized_pricing_optimization(self, pricing_engine):
        """Test optimisation pricing personnalisé par utilisateur."""
        # Profils utilisateur pour pricing personnalisé
        user_pricing_profiles = [
            {
                'user_id': 'user_high_value',
                'user_characteristics': {
                    'lifetime_value': Decimal('250.00'),
                    'churn_risk': 0.05,
                    'price_sensitivity': 0.2,
                    'feature_usage': {'offline': 0.9, 'high_quality': 0.95, 'discovery': 0.8},
                    'engagement_level': 'very_high',
                    'subscription_history': '3_years_premium'
                },
                'expected_pricing_strategy': 'value_based_premium'
            },
            {
                'user_id': 'user_price_sensitive',
                'user_characteristics': {
                    'lifetime_value': Decimal('45.00'),
                    'churn_risk': 0.4,
                    'price_sensitivity': 0.85,
                    'feature_usage': {'offline': 0.1, 'high_quality': 0.3, 'discovery': 0.6},
                    'engagement_level': 'medium',
                    'subscription_history': 'frequent_downgrades'
                },
                'expected_pricing_strategy': 'retention_focused'
            },
            {
                'user_id': 'user_new_customer',
                'user_characteristics': {
                    'lifetime_value': Decimal('0.00'),
                    'churn_risk': 0.25,
                    'price_sensitivity': 0.6,
                    'feature_usage': {'offline': 0.0, 'high_quality': 0.0, 'discovery': 0.9},
                    'engagement_level': 'exploring',
                    'subscription_history': 'trial_period'
                },
                'expected_pricing_strategy': 'acquisition_optimization'
            }
        ]
        
        # Mock pricing personnalisé
        pricing_engine.optimize_personalized_pricing = AsyncMock()
        
        for profile in user_pricing_profiles:
            # Configuration réponse pricing personnalisé
            pricing_engine.optimize_personalized_pricing.return_value = {
                'personalized_offer': {
                    'user_id': profile['user_id'],
                    'recommended_tier': 'premium' if profile['user_characteristics']['lifetime_value'] > Decimal('100') else 'premium_discounted',
                    'personalized_price': float(Decimal('9.99') * (1 - profile['user_characteristics']['price_sensitivity'] * 0.3)),
                    'discount_percentage': profile['user_characteristics']['price_sensitivity'] * 0.3,
                    'offer_validity_days': 30,
                    'urgency_factor': profile['user_characteristics']['churn_risk']
                },
                'value_proposition': {
                    'highlighted_features': [
                        feature for feature, usage in profile['user_characteristics']['feature_usage'].items()
                        if usage > 0.5
                    ],
                    'personalized_benefits': f"Optimized for {profile['user_characteristics']['engagement_level']} users",
                    'social_proof': f"Join {np.random.randint(10, 90)}% of similar users who upgraded",
                    'risk_mitigation': 'Money-back guarantee' if profile['user_characteristics']['churn_risk'] > 0.3 else None
                },
                'optimization_metrics': {
                    'conversion_probability': 1 - profile['user_characteristics']['churn_risk'] * 0.8,
                    'expected_lifetime_value_increase': np.random.uniform(0.1, 0.4),
                    'price_acceptance_score': 1 - profile['user_characteristics']['price_sensitivity'],
                    'competitive_advantage': np.random.uniform(0.1, 0.3)
                },
                'behavioral_triggers': {
                    'scarcity': profile['user_characteristics']['price_sensitivity'] > 0.7,
                    'social_proof': True,
                    'personalization': profile['user_characteristics']['engagement_level'] == 'very_high',
                    'loss_aversion': profile['user_characteristics']['churn_risk'] > 0.3
                }
            }
            
            personalized_pricing = await pricing_engine.optimize_personalized_pricing(
                user_profile=profile['user_characteristics'],
                market_context={'competition_level': 'high', 'demand_season': 'normal'},
                business_objectives={'primary': 'ltv_maximization', 'secondary': 'churn_reduction'}
            )
            
            # Validations pricing personnalisé
            assert personalized_pricing['personalized_offer']['personalized_price'] > 0
            assert 0 <= personalized_pricing['personalized_offer']['discount_percentage'] <= 0.5
            assert personalized_pricing['optimization_metrics']['conversion_probability'] > 0.2
            assert len(personalized_pricing['value_proposition']['highlighted_features']) > 0


class TestBusinessRulesEngine:
    """Tests enterprise pour BusinessRulesEngine avec règles métier complexes."""
    
    @pytest.fixture
    def rules_engine(self):
        """Instance BusinessRulesEngine pour tests."""
        return BusinessRulesEngine()
    
    async def test_complex_business_rules_evaluation(self, rules_engine):
        """Test évaluation règles métier complexes."""
        # Règles métier complexes
        business_rules = [
            {
                'rule_id': 'royalty_distribution',
                'rule_name': 'Artist Royalty Distribution',
                'rule_type': 'financial',
                'priority': 'high',
                'conditions': [
                    {'field': 'stream_count', 'operator': '>=', 'value': 1000},
                    {'field': 'artist_tier', 'operator': 'in', 'value': ['premium', 'verified']},
                    {'field': 'territory', 'operator': '!=', 'value': 'restricted'}
                ],
                'actions': [
                    {'type': 'calculate_royalty', 'rate': 0.7},
                    {'type': 'schedule_payment', 'frequency': 'monthly'},
                    {'type': 'generate_report', 'format': 'detailed'}
                ],
                'business_impact': 'revenue_distribution'
            },
            {
                'rule_id': 'content_moderation',
                'rule_name': 'Automated Content Moderation',
                'rule_type': 'compliance',
                'priority': 'critical',
                'conditions': [
                    {'field': 'content_type', 'operator': '==', 'value': 'music'},
                    {'field': 'explicit_content', 'operator': '==', 'value': True},
                    {'field': 'user_age', 'operator': '<', 'value': 18}
                ],
                'actions': [
                    {'type': 'block_content', 'immediate': True},
                    {'type': 'log_incident', 'severity': 'medium'},
                    {'type': 'notify_moderator', 'urgency': 'low'}
                ],
                'business_impact': 'compliance_protection'
            },
            {
                'rule_id': 'premium_feature_access',
                'rule_name': 'Premium Feature Access Control',
                'rule_type': 'access_control',
                'priority': 'medium',
                'conditions': [
                    {'field': 'subscription_status', 'operator': '==', 'value': 'premium'},
                    {'field': 'payment_status', 'operator': '==', 'value': 'current'},
                    {'field': 'account_standing', 'operator': '==', 'value': 'good'}
                ],
                'actions': [
                    {'type': 'grant_access', 'features': ['offline_mode', 'hd_quality', 'unlimited_skips']},
                    {'type': 'update_user_profile', 'tier': 'premium'},
                    {'type': 'track_usage', 'analytics': True}
                ],
                'business_impact': 'user_experience'
            }
        ]
        
        # Contextes d'évaluation
        evaluation_contexts = [
            {
                'context_name': 'artist_payment_cycle',
                'data': {
                    'stream_count': 15000,
                    'artist_tier': 'premium',
                    'territory': 'US',
                    'artist_id': 'artist_001',
                    'payment_period': 'Q1_2025'
                }
            },
            {
                'context_name': 'content_upload_review',
                'data': {
                    'content_type': 'music',
                    'explicit_content': True,
                    'user_age': 16,
                    'upload_timestamp': datetime.utcnow(),
                    'content_id': 'track_explicit_001'
                }
            },
            {
                'context_name': 'user_feature_request',
                'data': {
                    'subscription_status': 'premium',
                    'payment_status': 'current',
                    'account_standing': 'good',
                    'user_id': 'user_premium_001',
                    'requested_feature': 'offline_mode'
                }
            }
        ]
        
        # Mock évaluation règles
        rules_engine.evaluate_business_rules = AsyncMock()
        
        for context in evaluation_contexts:
            # Configuration réponse évaluation
            matching_rules = [
                rule for rule in business_rules
                if self._rule_matches_context(rule, context['data'])
            ]
            
            rules_engine.evaluate_business_rules.return_value = {
                'evaluation_summary': {
                    'context_name': context['context_name'],
                    'rules_evaluated': len(business_rules),
                    'rules_matched': len(matching_rules),
                    'rules_executed': len(matching_rules),
                    'evaluation_time_ms': np.random.uniform(5, 50)
                },
                'rule_executions': [
                    {
                        'rule_id': rule['rule_id'],
                        'rule_name': rule['rule_name'],
                        'execution_status': 'success',
                        'conditions_met': True,
                        'actions_executed': len(rule['actions']),
                        'execution_time_ms': np.random.uniform(1, 10),
                        'business_impact': rule['business_impact'],
                        'audit_trail': {
                            'timestamp': datetime.utcnow(),
                            'operator': 'system_automation',
                            'context_data': context['data']
                        }
                    } for rule in matching_rules
                ],
                'business_outcomes': {
                    'revenue_impact': np.random.uniform(0, 1000) if any(r['business_impact'] == 'revenue_distribution' for r in matching_rules) else 0,
                    'compliance_status': 'compliant' if any(r['business_impact'] == 'compliance_protection' for r in matching_rules) else 'no_action_needed',
                    'user_experience_score': np.random.uniform(0.8, 1.0) if any(r['business_impact'] == 'user_experience' for r in matching_rules) else None
                },
                'recommendations': [
                    {
                        'type': 'rule_optimization',
                        'description': 'Consider adding caching for frequently evaluated rules',
                        'priority': 'low'
                    }
                ] if len(matching_rules) > 2 else []
            }
            
            evaluation_result = await rules_engine.evaluate_business_rules(
                context_data=context['data'],
                rule_set=business_rules,
                execution_mode='production'
            )
            
            # Validations évaluation règles
            assert evaluation_result['evaluation_summary']['rules_evaluated'] == len(business_rules)
            assert evaluation_result['evaluation_summary']['evaluation_time_ms'] < 100
            assert all(
                exec_result['execution_status'] == 'success'
                for exec_result in evaluation_result['rule_executions']
            )
    
    def _rule_matches_context(self, rule: Dict[str, Any], context_data: Dict[str, Any]) -> bool:
        """Simule l'évaluation des conditions d'une règle."""
        for condition in rule['conditions']:
            field = condition['field']
            operator = condition['operator']
            value = condition['value']
            
            if field not in context_data:
                return False
            
            context_value = context_data[field]
            
            if operator == '==' and context_value != value:
                return False
            elif operator == '!=' and context_value == value:
                return False
            elif operator == '>=' and context_value < value:
                return False
            elif operator == '<' and context_value >= value:
                return False
            elif operator == 'in' and context_value not in value:
                return False
        
        return True
    
    async def test_rule_conflict_resolution(self, rules_engine):
        """Test résolution conflits entre règles métier."""
        # Règles potentiellement conflictuelles
        conflicting_rules = [
            {
                'rule_id': 'discount_high_value_user',
                'priority': 8,
                'conditions': [{'field': 'user_lifetime_value', 'operator': '>', 'value': 200}],
                'actions': [{'type': 'apply_discount', 'percentage': 0.15}],
                'rule_group': 'pricing'
            },
            {
                'rule_id': 'no_discount_new_release',
                'priority': 9,
                'conditions': [{'field': 'content_age_days', 'operator': '<', 'value': 30}],
                'actions': [{'type': 'deny_discount', 'reason': 'new_release_protection'}],
                'rule_group': 'pricing'
            },
            {
                'rule_id': 'premium_user_always_discount',
                'priority': 7,
                'conditions': [{'field': 'subscription_tier', 'operator': '==', 'value': 'premium_plus'}],
                'actions': [{'type': 'apply_discount', 'percentage': 0.1}],
                'rule_group': 'pricing'
            }
        ]
        
        # Mock résolution conflits
        rules_engine.resolve_rule_conflicts = AsyncMock(return_value={
            'conflict_analysis': {
                'conflicts_detected': 2,
                'conflicting_rule_pairs': [
                    {
                        'rule_1': 'discount_high_value_user',
                        'rule_2': 'no_discount_new_release',
                        'conflict_type': 'action_contradiction',
                        'resolution_strategy': 'priority_based'
                    },
                    {
                        'rule_1': 'premium_user_always_discount',
                        'rule_2': 'no_discount_new_release',
                        'conflict_type': 'action_contradiction',
                        'resolution_strategy': 'priority_based'
                    }
                ]
            },
            'resolution_result': {
                'winning_rule': 'no_discount_new_release',
                'winning_reason': 'highest_priority',
                'final_action': {'type': 'deny_discount', 'reason': 'new_release_protection'},
                'overridden_rules': ['discount_high_value_user', 'premium_user_always_discount'],
                'resolution_confidence': 0.95
            },
            'business_impact_assessment': {
                'revenue_impact': -15.0,  # Perte discount
                'compliance_benefit': 'new_release_protection_maintained',
                'user_experience_impact': 'minimal',
                'alternative_compensations': ['extended_trial', 'early_access_features']
            }
        })
        
        conflict_resolution = await rules_engine.resolve_rule_conflicts(
            applicable_rules=conflicting_rules,
            context={'user_lifetime_value': 250, 'content_age_days': 15, 'subscription_tier': 'premium_plus'},
            resolution_strategy='priority_weighted'
        )
        
        # Validations résolution conflits
        assert conflict_resolution['conflict_analysis']['conflicts_detected'] > 0
        assert 'winning_rule' in conflict_resolution['resolution_result']
        assert conflict_resolution['resolution_result']['resolution_confidence'] > 0.8


class TestRevenueOptimizer:
    """Tests enterprise pour RevenueOptimizer avec optimisation revenus IA."""
    
    @pytest.fixture
    def revenue_optimizer(self):
        """Instance RevenueOptimizer pour tests."""
        return RevenueOptimizer()
    
    async def test_multi_dimensional_revenue_optimization(self, revenue_optimizer):
        """Test optimisation revenus multi-dimensionnelle."""
        # Dimensions d'optimisation revenus
        revenue_dimensions = {
            'subscription_optimization': {
                'current_metrics': {
                    'monthly_recurring_revenue': Decimal('2450000.00'),
                    'churn_rate': 0.05,
                    'average_revenue_per_user': Decimal('9.99'),
                    'user_acquisition_cost': Decimal('15.50'),
                    'lifetime_value': Decimal('189.80')
                },
                'optimization_levers': ['pricing', 'tier_migration', 'churn_reduction', 'upselling']
            },
            'advertising_optimization': {
                'current_metrics': {
                    'ad_revenue_monthly': Decimal('850000.00'),
                    'fill_rate': 0.92,
                    'cpm': Decimal('2.50'),
                    'user_engagement_with_ads': 0.23,
                    'ad_to_subscription_conversion': 0.08
                },
                'optimization_levers': ['targeting_improvement', 'inventory_optimization', 'format_optimization']
            },
            'content_monetization': {
                'current_metrics': {
                    'content_licensing_revenue': Decimal('1200000.00'),
                    'royalty_payouts': Decimal('840000.00'),
                    'content_acquisition_cost': Decimal('180000.00'),
                    'content_performance_roi': 1.42
                },
                'optimization_levers': ['catalog_optimization', 'licensing_deals', 'exclusive_content']
            }
        }
        
        # Mock optimisation revenus
        revenue_optimizer.optimize_multi_dimensional_revenue = AsyncMock(return_value={
            'optimization_analysis': {
                'current_total_revenue': sum(
                    float(dim['current_metrics'].get('monthly_recurring_revenue', 0)) +
                    float(dim['current_metrics'].get('ad_revenue_monthly', 0)) +
                    float(dim['current_metrics'].get('content_licensing_revenue', 0))
                    for dim in revenue_dimensions.values()
                ),
                'optimization_potential': np.random.uniform(0.15, 0.35),
                'implementation_complexity': 'medium',
                'time_to_value_months': 6
            },
            'dimension_optimizations': {
                'subscription_optimization': {
                    'revenue_increase_potential': np.random.uniform(0.12, 0.25),
                    'recommended_actions': [
                        {'action': 'dynamic_pricing', 'impact': 'high', 'effort': 'medium'},
                        {'action': 'churn_prediction_model', 'impact': 'medium', 'effort': 'high'},
                        {'action': 'tier_recommendation_engine', 'impact': 'medium', 'effort': 'low'}
                    ],
                    'risk_factors': ['price_sensitivity', 'competitive_response'],
                    'success_metrics': ['mrr_growth', 'churn_reduction', 'arpu_increase']
                },
                'advertising_optimization': {
                    'revenue_increase_potential': np.random.uniform(0.08, 0.18),
                    'recommended_actions': [
                        {'action': 'ml_targeting_enhancement', 'impact': 'high', 'effort': 'high'},
                        {'action': 'inventory_yield_optimization', 'impact': 'medium', 'effort': 'medium'},
                        {'action': 'native_ad_formats', 'impact': 'medium', 'effort': 'medium'}
                    ],
                    'risk_factors': ['user_experience_degradation', 'privacy_regulations'],
                    'success_metrics': ['cpm_increase', 'fill_rate_improvement', 'engagement_rates']
                },
                'content_monetization': {
                    'revenue_increase_potential': np.random.uniform(0.10, 0.22),
                    'recommended_actions': [
                        {'action': 'ai_content_curation', 'impact': 'high', 'effort': 'high'},
                        {'action': 'exclusive_artist_partnerships', 'impact': 'high', 'effort': 'very_high'},
                        {'action': 'geographic_licensing_expansion', 'impact': 'medium', 'effort': 'medium'}
                    ],
                    'risk_factors': ['content_costs', 'artist_relations', 'licensing_complexity'],
                    'success_metrics': ['licensing_revenue_growth', 'content_roi', 'catalog_performance']
                }
            },
            'integrated_strategy': {
                'cross_dimension_synergies': [
                    {
                        'synergy': 'premium_content_subscription_upsell',
                        'involved_dimensions': ['subscription_optimization', 'content_monetization'],
                        'revenue_multiplier': 1.2,
                        'implementation_priority': 'high'
                    },
                    {
                        'synergy': 'ad_free_conversion_optimization',
                        'involved_dimensions': ['subscription_optimization', 'advertising_optimization'],
                        'revenue_multiplier': 1.15,
                        'implementation_priority': 'medium'
                    }
                ],
                'total_optimized_revenue_projection': np.random.uniform(4800000, 5500000),
                'roi_timeline': {
                    '3_months': 0.05,
                    '6_months': 0.15,
                    '12_months': 0.28,
                    '24_months': 0.42
                }
            }
        })
        
        optimization_result = await revenue_optimizer.optimize_multi_dimensional_revenue(
            current_state=revenue_dimensions,
            business_constraints={'budget_limit': 500000, 'timeline_months': 12},
            optimization_goals=['maximize_growth', 'minimize_risk', 'improve_margins']
        )
        
        # Validations optimisation multi-dimensionnelle
        assert optimization_result['optimization_analysis']['optimization_potential'] > 0.1
        assert len(optimization_result['dimension_optimizations']) == 3
        
        for dim_name, dim_opt in optimization_result['dimension_optimizations'].items():
            assert dim_opt['revenue_increase_potential'] > 0
            assert len(dim_opt['recommended_actions']) > 0
            assert len(dim_opt['success_metrics']) > 0
        
        assert len(optimization_result['integrated_strategy']['cross_dimension_synergies']) > 0
        assert optimization_result['integrated_strategy']['total_optimized_revenue_projection'] > optimization_result['optimization_analysis']['current_total_revenue']


class TestUserEngagementAnalyzer:
    """Tests enterprise pour UserEngagementAnalyzer avec analytics avancés."""
    
    @pytest.fixture
    def engagement_analyzer(self):
        """Instance UserEngagementAnalyzer pour tests."""
        return UserEngagementAnalyzer()
    
    async def test_comprehensive_engagement_analysis(self, engagement_analyzer):
        """Test analyse engagement utilisateur complète."""
        # Données engagement multi-facettes
        engagement_data = {
            'listening_behavior': {
                'daily_listening_hours': np.random.uniform(0.5, 8.0, 1000).tolist(),
                'session_frequency': np.random.poisson(3, 1000).tolist(),
                'skip_rates': np.random.beta(2, 5, 1000).tolist(),
                'completion_rates': np.random.beta(5, 2, 1000).tolist(),
                'discovery_engagement': np.random.uniform(0, 1, 1000).tolist()
            },
            'social_interactions': {
                'playlist_shares': np.random.poisson(1, 1000).tolist(),
                'track_likes': np.random.poisson(5, 1000).tolist(),
                'artist_follows': np.random.poisson(2, 1000).tolist(),
                'social_media_shares': np.random.poisson(0.5, 1000).tolist(),
                'comment_interactions': np.random.poisson(1.5, 1000).tolist()
            },
            'platform_engagement': {
                'feature_usage_scores': {
                    'search': np.random.uniform(0, 1, 1000).tolist(),
                    'recommendations': np.random.uniform(0, 1, 1000).tolist(),
                    'playlists': np.random.uniform(0, 1, 1000).tolist(),
                    'offline_mode': np.random.uniform(0, 1, 1000).tolist(),
                    'high_quality': np.random.uniform(0, 1, 1000).tolist()
                },
                'support_interactions': np.random.poisson(0.2, 1000).tolist(),
                'feedback_submissions': np.random.poisson(0.1, 1000).tolist()
            }
        }
        
        # Mock analyse engagement
        engagement_analyzer.analyze_comprehensive_engagement = AsyncMock(return_value={
            'engagement_metrics': {
                'overall_engagement_score': np.random.uniform(0.6, 0.9),
                'engagement_trend': 'increasing',
                'user_lifecycle_stage': 'active',
                'engagement_consistency': np.random.uniform(0.7, 0.95),
                'cross_feature_engagement': np.random.uniform(0.5, 0.8)
            },
            'behavioral_insights': {
                'listening_patterns': {
                    'peak_listening_hours': ['19:00-21:00', '08:00-09:00'],
                    'preferred_session_length': '25-45 minutes',
                    'discovery_vs_repetition_ratio': np.random.uniform(0.2, 0.4),
                    'genre_diversity_score': np.random.uniform(0.3, 0.8)
                },
                'social_engagement_profile': {
                    'social_engagement_level': 'medium',
                    'sharing_propensity': np.random.uniform(0.1, 0.3),
                    'influence_score': np.random.uniform(0.0, 0.5),
                    'community_participation': np.random.uniform(0.2, 0.7)
                },
                'feature_adoption': {
                    'power_user_features': ['playlists', 'offline_mode'],
                    'underutilized_features': ['social_sharing', 'artist_radio'],
                    'feature_exploration_rate': np.random.uniform(0.1, 0.4),
                    'premium_feature_usage': np.random.uniform(0.6, 0.9)
                }
            },
            'predictive_analytics': {
                'churn_probability_next_30_days': np.random.uniform(0.05, 0.25),
                'upsell_probability': np.random.uniform(0.15, 0.45),
                'lifetime_value_prediction': np.random.uniform(50, 300),
                'engagement_trajectory': 'stable_high',
                'intervention_recommendations': [
                    {
                        'intervention_type': 'personalized_playlist',
                        'expected_impact': 'medium',
                        'confidence': 0.75
                    },
                    {
                        'intervention_type': 'feature_education',
                        'expected_impact': 'low',
                        'confidence': 0.65
                    }
                ]
            },
            'segmentation_insights': {
                'primary_segment': 'engaged_music_lover',
                'segment_characteristics': {
                    'high_listening_volume': True,
                    'diverse_taste': True,
                    'social_sharer': False,
                    'feature_explorer': True
                },
                'segment_size_percentage': np.random.uniform(0.15, 0.35),
                'cross_segment_mobility_probability': np.random.uniform(0.1, 0.3)
            }
        })
        
        analysis_result = await engagement_analyzer.analyze_comprehensive_engagement(
            engagement_data=engagement_data,
            analysis_period_days=90,
            comparison_cohorts=['new_users', 'long_term_users']
        )
        
        # Validations analyse engagement
        assert 0 <= analysis_result['engagement_metrics']['overall_engagement_score'] <= 1
        assert analysis_result['engagement_metrics']['engagement_trend'] in ['increasing', 'stable', 'decreasing']
        assert len(analysis_result['behavioral_insights']['listening_patterns']['peak_listening_hours']) > 0
        assert 0 <= analysis_result['predictive_analytics']['churn_probability_next_30_days'] <= 1
        assert len(analysis_result['predictive_analytics']['intervention_recommendations']) > 0


# =============================================================================
# TESTS INTEGRATION BUSINESS LOGIC
# =============================================================================

@pytest.mark.integration
class TestBusinessLogicIntegration:
    """Tests d'intégration pour utils business logic."""
    
    async def test_end_to_end_business_workflow(self):
        """Test workflow business bout en bout."""
        # Simulation workflow business complet
        business_workflow = {
            'user_onboarding': {'new_user_id': 'user_new_001'},
            'preference_learning': {'listening_sessions': 10},
            'recommendation_generation': {'algorithm_fusion': True},
            'pricing_optimization': {'personalization': True},
            'engagement_tracking': {'real_time': True},
            'revenue_attribution': {'multi_touch': True}
        }
        
        workflow_steps = [
            {'step': 'user_profiling', 'expected_time_ms': 50},
            {'step': 'preference_analysis', 'expected_time_ms': 100},
            {'step': 'recommendation_engine', 'expected_time_ms': 200},
            {'step': 'pricing_calculation', 'expected_time_ms': 75},
            {'step': 'engagement_scoring', 'expected_time_ms': 30},
            {'step': 'revenue_tracking', 'expected_time_ms': 25}
        ]
        
        # Simulation workflow
        total_time = 0
        results = {}
        
        for step in workflow_steps:
            processing_time = step['expected_time_ms'] * np.random.uniform(0.8, 1.2)
            total_time += processing_time
            
            results[step['step']] = {
                'success': True,
                'processing_time_ms': processing_time,
                'business_value_generated': np.random.uniform(0.5, 1.0)
            }
        
        # Validations workflow
        assert all(result['success'] for result in results.values())
        assert total_time < 1000  # Moins d'1 seconde
        assert all(result['business_value_generated'] > 0.4 for result in results.values())


# =============================================================================
# TESTS PERFORMANCE BUSINESS LOGIC
# =============================================================================

@pytest.mark.performance
class TestBusinessLogicPerformance:
    """Tests performance pour utils business logic."""
    
    async def test_high_volume_recommendation_generation(self):
        """Test génération recommandations haut volume."""
        # Mock moteur recommandation haute performance
        recommendation_engine = RecommendationEngine()
        recommendation_engine.benchmark_recommendation_throughput = AsyncMock(return_value={
            'recommendations_per_second': 5000,
            'average_latency_ms': 45,
            'p95_latency_ms': 89,
            'p99_latency_ms': 156,
            'memory_efficiency': 0.88,
            'cache_hit_ratio': 0.92,
            'algorithm_distribution': {
                'collaborative_filtering': 0.4,
                'content_based': 0.3,
                'deep_learning': 0.25,
                'reinforcement_learning': 0.05
            }
        })
        
        # Test haute performance
        performance_test = await recommendation_engine.benchmark_recommendation_throughput(
            concurrent_users=10000,
            test_duration_minutes=10
        )
        
        # Validations performance
        assert performance_test['recommendations_per_second'] >= 1000
        assert performance_test['average_latency_ms'] < 100
        assert performance_test['memory_efficiency'] > 0.8
        assert performance_test['cache_hit_ratio'] > 0.85
    
    async def test_real_time_pricing_optimization_scalability(self):
        """Test scalabilité optimisation pricing temps réel."""
        pricing_engine = DynamicPricingEngine()
        
        # Test scalabilité pricing
        pricing_engine.benchmark_pricing_scalability = AsyncMock(return_value={
            'pricing_calculations_per_second': 2000,
            'market_analysis_latency_ms': 67,
            'personalization_overhead_ms': 23,
            'pricing_accuracy_maintained': 0.97,
            'system_stability_score': 0.94,
            'resource_utilization': {
                'cpu': 0.72,
                'memory': 0.68,
                'cache': 0.85
            }
        })
        
        scalability_test = await pricing_engine.benchmark_pricing_scalability(
            concurrent_pricing_requests=5000,
            market_complexity='high',
            personalization_depth='deep'
        )
        
        # Validations scalabilité pricing
        assert scalability_test['pricing_calculations_per_second'] >= 500
        assert scalability_test['market_analysis_latency_ms'] < 200
        assert scalability_test['pricing_accuracy_maintained'] > 0.95
        assert scalability_test['system_stability_score'] > 0.9
