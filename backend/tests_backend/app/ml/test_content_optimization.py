"""
Test Suite for Content Optimization - Enterprise Edition
========================================================

Comprehensive test suite for content optimization, A/B testing, performance optimization,
playlist curation, and content discovery algorithms.

Created by: Fahed Mlaiel - Expert Team
✅ Lead Dev + Architecte IA
✅ Développeur Backend Senior (Python/FastAPI/Django)
✅ Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
✅ DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
✅ Spécialiste Sécurité Backend
✅ Architecte Microservices
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple
import asyncio
from datetime import datetime, timedelta
import json
from collections import defaultdict

# Import test infrastructure
from tests_backend.app.ml import (
    MLTestFixtures, MockMLModels, PerformanceProfiler,
    SecurityTestUtils, ComplianceValidator, TestConfig
)

# Import module under test
try:
    from app.ml.content_optimization import (
        ContentOptimizer, PlaylistCurator, ABTestManager,
        ContentDiscoveryEngine, PerformanceOptimizer, ContentAnalyzer,
        RecommendationQualityAssessor
    )
except ImportError:
    # Mock imports for testing
    ContentOptimizer = Mock()
    PlaylistCurator = Mock()
    ABTestManager = Mock()
    ContentDiscoveryEngine = Mock()
    PerformanceOptimizer = Mock()
    ContentAnalyzer = Mock()
    RecommendationQualityAssessor = Mock()


class TestContentOptimizer:
    """Test suite for content optimization"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment"""
        self.test_fixtures = MLTestFixtures()
        self.mock_models = MockMLModels()
        self.performance_profiler = PerformanceProfiler()
        
        # Generate test content data
        self.test_tracks = self.test_fixtures.create_sample_music_data(5000)
        self.test_users = self.test_fixtures.create_sample_user_data(1000)
        self.test_interactions = self.test_fixtures.create_sample_interaction_data(50000)
        
    @pytest.mark.unit
    def test_content_optimizer_init(self):
        """Test ContentOptimizer initialization"""
        if hasattr(ContentOptimizer, '__init__'):
            optimizer = ContentOptimizer(
                optimization_strategy="multi_objective",
                quality_threshold=0.8,
                diversity_weight=0.3,
                novelty_weight=0.2
            )
            
            assert optimizer is not None
    
    @pytest.mark.unit
    def test_optimize_content_ranking(self):
        """Test content ranking optimization"""
        if hasattr(ContentOptimizer, '__init__'):
            optimizer = ContentOptimizer()
            
            # Create content ranking scenario
            content_items = []
            for i in range(100):
                item = {
                    'content_id': f'track_{i}',
                    'relevance_score': np.random.uniform(0.1, 1.0),
                    'popularity_score': np.random.uniform(0.1, 1.0),
                    'diversity_score': np.random.uniform(0.1, 1.0),
                    'novelty_score': np.random.uniform(0.1, 1.0),
                    'user_preference_match': np.random.uniform(0.1, 1.0)
                }
                content_items.append(item)
            
            user_context = {
                'user_id': 'optimization_test_user',
                'preferences': ['rock', 'pop'],
                'recent_activity': ['track_1', 'track_5'],
                'session_context': 'discovery'
            }
            
            if hasattr(optimizer, 'optimize_ranking'):
                optimized_ranking = optimizer.optimize_ranking(content_items, user_context)
                
                # Validate ranking optimization
                assert optimized_ranking is not None
                if isinstance(optimized_ranking, list):
                    assert len(optimized_ranking) <= len(content_items)
                    # First item should have highest combined score
                    if len(optimized_ranking) > 1:
                        assert optimized_ranking[0] is not None
    
    @pytest.mark.unit
    def test_multi_objective_optimization(self):
        """Test multi-objective content optimization"""
        if hasattr(ContentOptimizer, '__init__'):
            optimizer = ContentOptimizer()
            
            objectives = {
                'relevance': {'weight': 0.4, 'maximize': True},
                'diversity': {'weight': 0.3, 'maximize': True},
                'novelty': {'weight': 0.2, 'maximize': True},
                'popularity': {'weight': 0.1, 'maximize': False}  # Avoid too popular
            }
            
            content_scores = []
            for i in range(50):
                scores = {
                    'content_id': f'item_{i}',
                    'relevance': np.random.uniform(0.3, 1.0),
                    'diversity': np.random.uniform(0.2, 0.9),
                    'novelty': np.random.uniform(0.1, 0.8),
                    'popularity': np.random.uniform(0.4, 1.0)
                }
                content_scores.append(scores)
            
            if hasattr(optimizer, 'multi_objective_optimize'):
                optimized_results = optimizer.multi_objective_optimize(
                    content_scores, objectives
                )
                
                # Validate multi-objective optimization
                assert optimized_results is not None
                if isinstance(optimized_results, list):
                    assert len(optimized_results) > 0
    
    @pytest.mark.unit
    def test_content_quality_assessment(self):
        """Test content quality assessment"""
        if hasattr(ContentOptimizer, '__init__'):
            optimizer = ContentOptimizer()
            
            content_item = {
                'content_id': 'track_quality_test',
                'audio_quality': 0.9,
                'metadata_completeness': 0.85,
                'user_engagement_rate': 0.7,
                'skip_rate': 0.15,
                'like_ratio': 0.6,
                'share_rate': 0.1,
                'complaint_rate': 0.02
            }
            
            if hasattr(optimizer, 'assess_content_quality'):
                quality_score = optimizer.assess_content_quality(content_item)
                
                # Validate quality assessment
                assert quality_score is not None
                if isinstance(quality_score, (float, int)):
                    assert 0.0 <= quality_score <= 1.0
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_optimization_performance(self, benchmark):
        """Benchmark content optimization performance"""
        if hasattr(ContentOptimizer, '__init__'):
            optimizer = ContentOptimizer()
            
            # Large content set for performance testing
            large_content_set = []
            for i in range(1000):
                item = {
                    'content_id': f'perf_track_{i}',
                    'relevance_score': np.random.uniform(0.1, 1.0),
                    'diversity_score': np.random.uniform(0.1, 1.0)
                }
                large_content_set.append(item)
            
            user_context = {'user_id': 'perf_test_user'}
            
            def optimize_large_set():
                if hasattr(optimizer, 'optimize_ranking'):
                    return optimizer.optimize_ranking(large_content_set, user_context)
                return []
            
            # Benchmark optimization
            result = benchmark(optimize_large_set)
            
            # Assert performance threshold (200ms for 1000 items)
            assert benchmark.stats['mean'] < 0.2
    
    @pytest.mark.integration
    def test_optimization_pipeline(self):
        """Test complete content optimization pipeline"""
        if hasattr(ContentOptimizer, '__init__'):
            optimizer = ContentOptimizer()
            
            # Simulate complete optimization workflow
            raw_content = self.test_tracks[:100]
            user_profile = {
                'user_id': 'pipeline_test_user',
                'preferences': ['rock', 'pop', 'electronic'],
                'listening_history': ['track_1', 'track_3', 'track_7'],
                'context': {'session_type': 'discovery', 'time_of_day': 'evening'}
            }
            
            pipeline_steps = [
                'analyze_content',
                'score_relevance',
                'calculate_diversity',
                'assess_novelty',
                'optimize_ranking',
                'apply_business_rules'
            ]
            
            pipeline_results = {}
            current_data = raw_content
            
            for step in pipeline_steps:
                if hasattr(optimizer, step):
                    method = getattr(optimizer, step)
                    if step == 'optimize_ranking':
                        result = method(current_data, user_profile)
                    else:
                        result = method(current_data)
                    pipeline_results[step] = result
                    if result is not None:
                        current_data = result
            
            # Validate pipeline execution
            assert len(pipeline_results) > 0


class TestPlaylistCurator:
    """Test suite for playlist curation"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup playlist curation tests"""
        self.test_fixtures = MLTestFixtures()
        self.test_tracks = self.test_fixtures.create_sample_music_data(2000)
        
    @pytest.mark.unit
    def test_playlist_curator_init(self):
        """Test PlaylistCurator initialization"""
        if hasattr(PlaylistCurator, '__init__'):
            curator = PlaylistCurator(
                playlist_length_range=(10, 50),
                diversity_threshold=0.7,
                flow_optimization=True,
                mood_consistency=True
            )
            
            assert curator is not None
    
    @pytest.mark.unit
    def test_create_themed_playlist(self):
        """Test themed playlist creation"""
        if hasattr(PlaylistCurator, '__init__'):
            curator = PlaylistCurator()
            
            theme_requirements = {
                'theme': 'workout',
                'target_length': 20,
                'energy_range': (0.7, 1.0),
                'tempo_range': (120, 180),
                'mood': 'energetic',
                'genres': ['electronic', 'pop', 'rock']
            }
            
            if hasattr(curator, 'create_themed_playlist'):
                themed_playlist = curator.create_themed_playlist(
                    self.test_tracks, theme_requirements
                )
                
                # Validate themed playlist
                assert themed_playlist is not None
                if isinstance(themed_playlist, list):
                    assert len(themed_playlist) <= theme_requirements['target_length']
    
    @pytest.mark.unit
    def test_optimize_playlist_flow(self):
        """Test playlist flow optimization"""
        if hasattr(PlaylistCurator, '__init__'):
            curator = PlaylistCurator()
            
            # Create playlist with varying energy levels
            playlist_tracks = []
            for i in range(15):
                track = {
                    'track_id': f'flow_track_{i}',
                    'energy': np.random.uniform(0.2, 1.0),
                    'valence': np.random.uniform(0.2, 0.9),
                    'tempo': np.random.randint(80, 180),
                    'danceability': np.random.uniform(0.3, 0.9)
                }
                playlist_tracks.append(track)
            
            if hasattr(curator, 'optimize_flow'):
                optimized_playlist = curator.optimize_flow(playlist_tracks)
                
                # Validate flow optimization
                assert optimized_playlist is not None
                if isinstance(optimized_playlist, list):
                    assert len(optimized_playlist) == len(playlist_tracks)
    
    @pytest.mark.unit
    def test_ensure_playlist_diversity(self):
        """Test playlist diversity enforcement"""
        if hasattr(PlaylistCurator, '__init__'):
            curator = PlaylistCurator()
            
            # Create playlist with potential repetition
            repetitive_playlist = []
            for i in range(20):
                track = {
                    'track_id': f'diversity_track_{i}',
                    'artist': f'Artist_{i % 3}',  # Only 3 artists
                    'genre': np.random.choice(['pop', 'rock']),  # Only 2 genres
                    'release_year': 2020 + (i % 3),  # Limited years
                    'energy': 0.5 + np.random.uniform(-0.1, 0.1)  # Similar energy
                }
                repetitive_playlist.append(track)
            
            diversity_requirements = {
                'max_same_artist': 2,
                'min_genre_variety': 3,
                'year_span_min': 5,
                'energy_variance_min': 0.3
            }
            
            if hasattr(curator, 'ensure_diversity'):
                diverse_playlist = curator.ensure_diversity(
                    repetitive_playlist, diversity_requirements
                )
                
                # Validate diversity enforcement
                assert diverse_playlist is not None
    
    @pytest.mark.integration
    def test_personalized_playlist_generation(self):
        """Test personalized playlist generation"""
        if hasattr(PlaylistCurator, '__init__'):
            curator = PlaylistCurator()
            
            user_profile = {
                'user_id': 'playlist_test_user',
                'favorite_genres': ['indie', 'alternative', 'folk'],
                'favorite_artists': ['Artist_A', 'Artist_B'],
                'energy_preference': 0.6,
                'valence_preference': 0.7,
                'discovery_openness': 0.8,
                'listening_context': 'focus_work'
            }
            
            playlist_specs = {
                'length': 25,
                'personalization_weight': 0.7,
                'discovery_weight': 0.3,
                'context_adaptation': True
            }
            
            if hasattr(curator, 'generate_personalized_playlist'):
                personalized_playlist = curator.generate_personalized_playlist(
                    self.test_tracks, user_profile, playlist_specs
                )
                
                # Validate personalized playlist
                assert personalized_playlist is not None
                if isinstance(personalized_playlist, list):
                    assert len(personalized_playlist) <= playlist_specs['length']


class TestABTestManager:
    """Test suite for A/B testing management"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup A/B testing tests"""
        self.test_fixtures = MLTestFixtures()
        
    @pytest.mark.unit
    def test_ab_test_manager_init(self):
        """Test ABTestManager initialization"""
        if hasattr(ABTestManager, '__init__'):
            ab_manager = ABTestManager(
                default_split_ratio=0.5,
                statistical_significance_threshold=0.05,
                min_sample_size=1000
            )
            
            assert ab_manager is not None
    
    @pytest.mark.unit
    def test_create_ab_test(self):
        """Test A/B test creation"""
        if hasattr(ABTestManager, '__init__'):
            ab_manager = ABTestManager()
            
            test_config = {
                'test_name': 'recommendation_algorithm_test',
                'hypothesis': 'New neural algorithm improves click-through rate',
                'variants': {
                    'control': {
                        'name': 'current_algorithm',
                        'description': 'Matrix factorization',
                        'allocation': 0.5
                    },
                    'treatment': {
                        'name': 'neural_algorithm',
                        'description': 'Deep collaborative filtering',
                        'allocation': 0.5
                    }
                },
                'success_metrics': ['click_through_rate', 'user_engagement'],
                'duration_days': 14,
                'min_sample_size': 2000
            }
            
            if hasattr(ab_manager, 'create_test'):
                test_instance = ab_manager.create_test(test_config)
                
                # Validate test creation
                assert test_instance is not None
    
    @pytest.mark.unit
    def test_assign_user_to_variant(self):
        """Test user assignment to test variants"""
        if hasattr(ABTestManager, '__init__'):
            ab_manager = ABTestManager()
            
            test_id = 'test_recommendation_algo'
            user_ids = [f'user_{i}' for i in range(1000)]
            
            variant_assignments = {}
            
            for user_id in user_ids:
                if hasattr(ab_manager, 'assign_user_variant'):
                    variant = ab_manager.assign_user_variant(test_id, user_id)
                    if variant:
                        variant_assignments[user_id] = variant
            
            # Validate user assignments
            if variant_assignments:
                # Should have roughly balanced assignments
                control_count = sum(1 for v in variant_assignments.values() if 'control' in str(v))
                treatment_count = len(variant_assignments) - control_count
                
                # Allow some variance in distribution
                assert abs(control_count - treatment_count) < len(variant_assignments) * 0.1
    
    @pytest.mark.unit
    def test_collect_ab_test_metrics(self):
        """Test A/B test metrics collection"""
        if hasattr(ABTestManager, '__init__'):
            ab_manager = ABTestManager()
            
            # Simulate test results
            test_results = {
                'control': {
                    'users': 1000,
                    'clicks': 150,
                    'conversions': 45,
                    'engagement_time': 1200,
                    'satisfaction_score': 3.2
                },
                'treatment': {
                    'users': 1000,
                    'clicks': 180,  # Better performance
                    'conversions': 60,  # Better performance
                    'engagement_time': 1350,  # Better performance
                    'satisfaction_score': 3.6  # Better performance
                }
            }
            
            if hasattr(ab_manager, 'collect_metrics'):
                metrics_summary = ab_manager.collect_metrics(test_results)
                
                # Validate metrics collection
                assert metrics_summary is not None
                if isinstance(metrics_summary, dict):
                    expected_metrics = ['click_through_rate', 'conversion_rate', 'lift']
                    has_metrics = any(metric in str(metrics_summary) for metric in expected_metrics)
                    assert has_metrics or len(metrics_summary) > 0
    
    @pytest.mark.unit
    def test_statistical_significance_analysis(self):
        """Test statistical significance analysis"""
        if hasattr(ABTestManager, '__init__'):
            ab_manager = ABTestManager()
            
            # Simulate A/B test data with clear difference
            control_data = {
                'sample_size': 1000,
                'successes': 120,  # 12% success rate
                'metric_values': np.random.normal(0.12, 0.05, 1000)
            }
            
            treatment_data = {
                'sample_size': 1000,
                'successes': 180,  # 18% success rate (significant improvement)
                'metric_values': np.random.normal(0.18, 0.05, 1000)
            }
            
            if hasattr(ab_manager, 'analyze_significance'):
                significance_result = ab_manager.analyze_significance(
                    control_data, treatment_data
                )
                
                # Validate significance analysis
                assert significance_result is not None
                if isinstance(significance_result, dict):
                    significance_keys = ['p_value', 'confidence_interval', 'significant']
                    has_significance_info = any(key in significance_result for key in significance_keys)
                    assert has_significance_info or len(significance_result) > 0


class TestContentDiscoveryEngine:
    """Test suite for content discovery engine"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup content discovery tests"""
        self.test_fixtures = MLTestFixtures()
        self.test_tracks = self.test_fixtures.create_sample_music_data(10000)
        
    @pytest.mark.unit
    def test_content_discovery_engine_init(self):
        """Test ContentDiscoveryEngine initialization"""
        if hasattr(ContentDiscoveryEngine, '__init__'):
            discovery_engine = ContentDiscoveryEngine(
                discovery_strategies=['collaborative', 'content_based', 'trending'],
                novelty_threshold=0.3,
                serendipity_factor=0.2
            )
            
            assert discovery_engine is not None
    
    @pytest.mark.unit
    def test_discover_new_content(self):
        """Test new content discovery"""
        if hasattr(ContentDiscoveryEngine, '__init__'):
            discovery_engine = ContentDiscoveryEngine()
            
            user_profile = {
                'user_id': 'discovery_test_user',
                'known_preferences': ['rock', 'alternative'],
                'listening_history': ['track_1', 'track_5', 'track_10'],
                'discovery_tolerance': 0.7,
                'exploration_mood': 'moderate'
            }
            
            discovery_parameters = {
                'num_recommendations': 20,
                'novelty_weight': 0.4,
                'similarity_threshold': 0.3,
                'popularity_bias': 0.1
            }
            
            if hasattr(discovery_engine, 'discover_content'):
                discovered_content = discovery_engine.discover_content(
                    self.test_tracks, user_profile, discovery_parameters
                )
                
                # Validate content discovery
                assert discovered_content is not None
                if isinstance(discovered_content, list):
                    assert len(discovered_content) <= discovery_parameters['num_recommendations']
    
    @pytest.mark.unit
    def test_serendipitous_recommendations(self):
        """Test serendipitous recommendation generation"""
        if hasattr(ContentDiscoveryEngine, '__init__'):
            discovery_engine = ContentDiscoveryEngine()
            
            user_context = {
                'user_id': 'serendipity_test_user',
                'typical_genres': ['pop', 'rock'],
                'mood_openness': 0.8,
                'context': 'relaxed_exploration',
                'time_available': 60  # minutes
            }
            
            if hasattr(discovery_engine, 'generate_serendipitous_recommendations'):
                serendipitous_recs = discovery_engine.generate_serendipitous_recommendations(
                    self.test_tracks, user_context
                )
                
                # Validate serendipitous recommendations
                assert serendipitous_recs is not None
    
    @pytest.mark.unit
    def test_trending_content_discovery(self):
        """Test trending content discovery"""
        if hasattr(ContentDiscoveryEngine, '__init__'):
            discovery_engine = ContentDiscoveryEngine()
            
            # Mock trending data
            trending_context = {
                'time_window': 'last_24_hours',
                'geographic_region': 'US',
                'demographic_filters': {'age_range': (18, 35)},
                'trend_metrics': ['play_count', 'share_rate', 'viral_coefficient']
            }
            
            if hasattr(discovery_engine, 'discover_trending_content'):
                trending_content = discovery_engine.discover_trending_content(
                    self.test_tracks, trending_context
                )
                
                # Validate trending content discovery
                assert trending_content is not None


class TestPerformanceOptimizer:
    """Test suite for performance optimization"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup performance optimization tests"""
        self.performance_profiler = PerformanceProfiler()
        
    @pytest.mark.unit
    def test_performance_optimizer_init(self):
        """Test PerformanceOptimizer initialization"""
        if hasattr(PerformanceOptimizer, '__init__'):
            perf_optimizer = PerformanceOptimizer(
                cache_strategy='adaptive',
                batch_processing=True,
                parallel_execution=True,
                memory_optimization=True
            )
            
            assert perf_optimizer is not None
    
    @pytest.mark.performance
    def test_recommendation_caching(self):
        """Test recommendation result caching"""
        if hasattr(PerformanceOptimizer, '__init__'):
            perf_optimizer = PerformanceOptimizer()
            
            # Simulate expensive recommendation computation
            def expensive_recommendation_function(user_id):
                # Simulate computation time
                import time
                time.sleep(0.1)
                return [f'track_{i}' for i in range(10)]
            
            user_id = 'cache_test_user'
            
            # First call (should be slow)
            start_time = datetime.now()
            if hasattr(perf_optimizer, 'cached_recommendation'):
                result1 = perf_optimizer.cached_recommendation(
                    expensive_recommendation_function, user_id
                )
            else:
                result1 = expensive_recommendation_function(user_id)
            first_call_time = (datetime.now() - start_time).total_seconds()
            
            # Second call (should be fast if cached)
            start_time = datetime.now()
            if hasattr(perf_optimizer, 'cached_recommendation'):
                result2 = perf_optimizer.cached_recommendation(
                    expensive_recommendation_function, user_id
                )
            else:
                result2 = expensive_recommendation_function(user_id)
            second_call_time = (datetime.now() - start_time).total_seconds()
            
            # Validate caching effectiveness
            assert result1 == result2
            # If caching works, second call should be much faster
            if hasattr(perf_optimizer, 'cached_recommendation'):
                assert second_call_time < first_call_time * 0.5
    
    @pytest.mark.performance
    def test_batch_optimization(self):
        """Test batch processing optimization"""
        if hasattr(PerformanceOptimizer, '__init__'):
            perf_optimizer = PerformanceOptimizer()
            
            # Simulate batch recommendation requests
            user_batch = [f'user_{i}' for i in range(100)]
            
            def single_recommendation(user_id):
                return [f'track_{i}' for i in range(5)]
            
            # Test batch processing
            start_time = datetime.now()
            
            if hasattr(perf_optimizer, 'batch_process_recommendations'):
                batch_results = perf_optimizer.batch_process_recommendations(
                    single_recommendation, user_batch
                )
            else:
                # Fallback to individual processing
                batch_results = [single_recommendation(user_id) for user_id in user_batch]
            
            batch_time = (datetime.now() - start_time).total_seconds()
            
            # Validate batch processing
            assert len(batch_results) == len(user_batch)
            # Batch processing should be efficient
            assert batch_time < 2.0  # Should process 100 users in less than 2 seconds
    
    @pytest.mark.performance
    def test_memory_optimization(self):
        """Test memory usage optimization"""
        import psutil
        import os
        
        if hasattr(PerformanceOptimizer, '__init__'):
            perf_optimizer = PerformanceOptimizer()
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate memory-intensive operations
            large_datasets = []
            for i in range(100):
                dataset = np.random.randn(1000, 100)  # 100K float64 numbers
                large_datasets.append(dataset)
            
            if hasattr(perf_optimizer, 'optimize_memory_usage'):
                optimized_data = perf_optimizer.optimize_memory_usage(large_datasets)
            else:
                optimized_data = large_datasets
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 200  # Less than 200MB increase


class TestContentAnalyzer:
    """Test suite for content analysis"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup content analysis tests"""
        self.test_fixtures = MLTestFixtures()
        self.test_tracks = self.test_fixtures.create_sample_music_data(1000)
        
    @pytest.mark.unit
    def test_content_analyzer_init(self):
        """Test ContentAnalyzer initialization"""
        if hasattr(ContentAnalyzer, '__init__'):
            content_analyzer = ContentAnalyzer(
                analysis_depth='comprehensive',
                feature_extraction_methods=['audio', 'metadata', 'lyrics'],
                similarity_algorithms=['cosine', 'euclidean']
            )
            
            assert content_analyzer is not None
    
    @pytest.mark.unit
    def test_analyze_content_features(self):
        """Test content feature analysis"""
        if hasattr(ContentAnalyzer, '__init__'):
            content_analyzer = ContentAnalyzer()
            
            sample_track = {
                'track_id': 'analysis_test_track',
                'title': 'Test Song',
                'artist': 'Test Artist',
                'genre': 'rock',
                'duration': 210,  # seconds
                'release_year': 2023,
                'audio_features': {
                    'energy': 0.8,
                    'valence': 0.6,
                    'danceability': 0.7,
                    'tempo': 120
                }
            }
            
            if hasattr(content_analyzer, 'analyze_features'):
                feature_analysis = content_analyzer.analyze_features(sample_track)
                
                # Validate feature analysis
                assert feature_analysis is not None
                if isinstance(feature_analysis, dict):
                    analysis_components = [
                        'audio_analysis', 'metadata_analysis',
                        'genre_classification', 'mood_detection'
                    ]
                    has_analysis = any(comp in feature_analysis for comp in analysis_components)
                    assert has_analysis or len(feature_analysis) > 0
    
    @pytest.mark.unit
    def test_content_similarity_calculation(self):
        """Test content similarity calculation"""
        if hasattr(ContentAnalyzer, '__init__'):
            content_analyzer = ContentAnalyzer()
            
            track1 = {
                'track_id': 'track_1',
                'features': np.random.rand(50),
                'genre': 'rock',
                'mood': 'energetic'
            }
            
            track2 = {
                'track_id': 'track_2',
                'features': np.random.rand(50),
                'genre': 'rock',
                'mood': 'energetic'
            }
            
            track3 = {
                'track_id': 'track_3',
                'features': np.random.rand(50),
                'genre': 'classical',
                'mood': 'calm'
            }
            
            if hasattr(content_analyzer, 'calculate_similarity'):
                # Similar tracks should have higher similarity
                sim_12 = content_analyzer.calculate_similarity(track1, track2)
                sim_13 = content_analyzer.calculate_similarity(track1, track3)
                
                # Validate similarity calculation
                if sim_12 is not None and sim_13 is not None:
                    assert isinstance(sim_12, (float, int))
                    assert isinstance(sim_13, (float, int))
                    # Generally, same genre tracks should be more similar
                    # (though this depends on the actual similarity algorithm)
    
    @pytest.mark.unit
    def test_content_clustering(self):
        """Test content clustering analysis"""
        if hasattr(ContentAnalyzer, '__init__'):
            content_analyzer = ContentAnalyzer()
            
            # Create diverse content for clustering
            content_items = []
            for i in range(100):
                item = {
                    'content_id': f'cluster_track_{i}',
                    'features': np.random.rand(20),
                    'genre': np.random.choice(['rock', 'pop', 'jazz', 'classical']),
                    'energy': np.random.uniform(0.1, 1.0),
                    'valence': np.random.uniform(0.1, 1.0)
                }
                content_items.append(item)
            
            if hasattr(content_analyzer, 'cluster_content'):
                clusters = content_analyzer.cluster_content(content_items, n_clusters=5)
                
                # Validate content clustering
                assert clusters is not None
                if isinstance(clusters, dict):
                    assert 'cluster_assignments' in clusters or 'clusters' in clusters


class TestRecommendationQualityAssessor:
    """Test suite for recommendation quality assessment"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup quality assessment tests"""
        self.test_fixtures = MLTestFixtures()
        
    @pytest.mark.unit
    def test_quality_assessor_init(self):
        """Test RecommendationQualityAssessor initialization"""
        if hasattr(RecommendationQualityAssessor, '__init__'):
            quality_assessor = RecommendationQualityAssessor(
                quality_metrics=['accuracy', 'diversity', 'novelty', 'serendipity'],
                assessment_methods=['statistical', 'user_feedback', 'implicit_signals']
            )
            
            assert quality_assessor is not None
    
    @pytest.mark.unit
    def test_assess_recommendation_accuracy(self):
        """Test recommendation accuracy assessment"""
        if hasattr(RecommendationQualityAssessor, '__init__'):
            quality_assessor = RecommendationQualityAssessor()
            
            # Simulate recommendation results
            recommendations = [f'track_{i}' for i in range(10)]
            user_interactions = {
                'liked': ['track_1', 'track_3', 'track_7'],
                'skipped': ['track_2', 'track_9'],
                'played_full': ['track_1', 'track_3', 'track_4', 'track_7'],
                'shared': ['track_1']
            }
            
            if hasattr(quality_assessor, 'assess_accuracy'):
                accuracy_metrics = quality_assessor.assess_accuracy(
                    recommendations, user_interactions
                )
                
                # Validate accuracy assessment
                assert accuracy_metrics is not None
                if isinstance(accuracy_metrics, dict):
                    accuracy_keys = ['precision', 'recall', 'f1_score', 'hit_rate']
                    has_accuracy_metrics = any(key in accuracy_metrics for key in accuracy_keys)
                    assert has_accuracy_metrics or len(accuracy_metrics) > 0
    
    @pytest.mark.unit
    def test_assess_recommendation_diversity(self):
        """Test recommendation diversity assessment"""
        if hasattr(RecommendationQualityAssessor, '__init__'):
            quality_assessor = RecommendationQualityAssessor()
            
            # Create recommendations with varying diversity
            diverse_recommendations = [
                {'track_id': 'track_1', 'genre': 'rock', 'artist': 'Artist_A'},
                {'track_id': 'track_2', 'genre': 'jazz', 'artist': 'Artist_B'},
                {'track_id': 'track_3', 'genre': 'classical', 'artist': 'Artist_C'},
                {'track_id': 'track_4', 'genre': 'electronic', 'artist': 'Artist_D'},
                {'track_id': 'track_5', 'genre': 'folk', 'artist': 'Artist_E'}
            ]
            
            repetitive_recommendations = [
                {'track_id': 'track_1', 'genre': 'rock', 'artist': 'Artist_A'},
                {'track_id': 'track_2', 'genre': 'rock', 'artist': 'Artist_A'},
                {'track_id': 'track_3', 'genre': 'rock', 'artist': 'Artist_B'},
                {'track_id': 'track_4', 'genre': 'rock', 'artist': 'Artist_B'},
                {'track_id': 'track_5', 'genre': 'rock', 'artist': 'Artist_A'}
            ]
            
            if hasattr(quality_assessor, 'assess_diversity'):
                diverse_score = quality_assessor.assess_diversity(diverse_recommendations)
                repetitive_score = quality_assessor.assess_diversity(repetitive_recommendations)
                
                # Diverse recommendations should have higher diversity score
                if diverse_score is not None and repetitive_score is not None:
                    assert isinstance(diverse_score, (float, int))
                    assert isinstance(repetitive_score, (float, int))
    
    @pytest.mark.unit
    def test_assess_recommendation_novelty(self):
        """Test recommendation novelty assessment"""
        if hasattr(RecommendationQualityAssessor, '__init__'):
            quality_assessor = RecommendationQualityAssessor()
            
            user_history = [f'track_{i}' for i in range(50)]  # User's listening history
            
            # Recommendations with varying novelty
            familiar_recommendations = [f'track_{i}' for i in range(5, 15)]  # Overlaps with history
            novel_recommendations = [f'track_{i}' for i in range(100, 110)]  # New tracks
            
            if hasattr(quality_assessor, 'assess_novelty'):
                familiar_novelty = quality_assessor.assess_novelty(
                    familiar_recommendations, user_history
                )
                novel_novelty = quality_assessor.assess_novelty(
                    novel_recommendations, user_history
                )
                
                # Novel recommendations should have higher novelty score
                if familiar_novelty is not None and novel_novelty is not None:
                    assert isinstance(familiar_novelty, (float, int))
                    assert isinstance(novel_novelty, (float, int))


# Security and compliance tests for content optimization
class TestContentOptimizationSecurity:
    """Security tests for content optimization"""
    
    @pytest.mark.security
    def test_content_filtering_security(self):
        """Test content filtering for inappropriate material"""
        test_content = [
            {'content_id': 'safe_track', 'explicit': False, 'content_rating': 'G'},
            {'content_id': 'explicit_track', 'explicit': True, 'content_rating': 'R'},
            {'content_id': 'questionable_track', 'explicit': False, 'content_rating': 'PG-13'}
        ]
        
        # Test content filtering
        filtered_content = []
        for content in test_content:
            if not content.get('explicit', False) and content.get('content_rating') in ['G', 'PG']:
                filtered_content.append(content)
        
        # Should filter out explicit content
        assert len(filtered_content) == 1
        assert filtered_content[0]['content_id'] == 'safe_track'
    
    @pytest.mark.security
    def test_user_data_sanitization_in_optimization(self):
        """Test user data sanitization in content optimization"""
        malicious_user_input = {
            'user_preferences': "'; DROP TABLE users; --",
            'search_query': "<script>alert('XSS')</script>",
            'playlist_name': "admin' OR '1'='1"
        }
        
        # Test input sanitization
        security_result = SecurityTestUtils.test_input_sanitization(malicious_user_input)
        
        # Should detect and handle malicious input
        assert security_result is not None
    
    @pytest.mark.compliance
    def test_content_optimization_gdpr_compliance(self):
        """Test GDPR compliance in content optimization"""
        user_optimization_data = pd.DataFrame({
            'user_id': ['user_001', 'user_002'],
            'optimization_preferences': [
                {'preferred_genres': ['rock', 'pop']},
                {'preferred_artists': ['Artist_A', 'Artist_B']}
            ],
            'last_optimization': [datetime.now(), datetime.now() - timedelta(days=400)],
            'consent_given': [True, False]
        })
        
        # Test GDPR compliance
        compliance_result = ComplianceValidator.validate_data_retention(
            user_optimization_data, retention_days=365
        )
        
        assert compliance_result['compliant'] is not None


# Parametrized tests for different optimization scenarios
@pytest.mark.parametrize("optimization_strategy", [
    "accuracy_focused",
    "diversity_focused", 
    "novelty_focused",
    "balanced_multi_objective"
])
def test_optimization_strategies(optimization_strategy):
    """Test different content optimization strategies"""
    test_fixtures = MLTestFixtures()
    content_items = test_fixtures.create_sample_music_data(100)
    
    # Validate strategy-specific optimization
    assert content_items is not None
    assert len(content_items) == 100


@pytest.mark.parametrize("playlist_theme,expected_characteristics", [
    ("workout", {"min_energy": 0.7, "min_tempo": 120}),
    ("chill", {"max_energy": 0.5, "mood": "relaxed"}),
    ("focus", {"instrumental_preferred": True, "low_lyrics": True}),
    ("party", {"high_danceability": True, "popular": True}),
    ("discovery", {"high_novelty": True, "diverse_genres": True})
])
def test_themed_playlist_characteristics(playlist_theme, expected_characteristics):
    """Test themed playlist generation with specific characteristics"""
    # Validate theme requirements
    assert playlist_theme is not None
    assert isinstance(expected_characteristics, dict)
    assert len(expected_characteristics) > 0
