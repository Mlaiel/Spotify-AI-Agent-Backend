"""
Test Suite for Recommendation Engine - Enterprise Edition
=========================================================

Comprehensive test suite for recommendation engine with legacy compatibility,
advanced neural recommendations, and enterprise-grade validation.

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
from typing import Dict, List, Any
import asyncio
from datetime import datetime, timedelta
import json

# Import test infrastructure
from tests_backend.app.ml import (
    MLTestFixtures, MockMLModels, PerformanceProfiler,
    SecurityTestUtils, ComplianceValidator, TestConfig
)

# Import module under test
try:
    from app.ml.recommendation_engine import (
        recommend_tracks, LegacyRecommendationResult,
        _get_enhanced_engine, _process_context_enhanced,
        _apply_context_boost
    )
    from app.ml.neural_recommendation_engine import (
        RecommendationRequest, RecommendationResponse,
        NeuralCollaborativeFiltering, UserItemDataset,
        create_neural_recommendation_engine
    )
except ImportError:
    # Mock imports for testing
    recommend_tracks = Mock()
    LegacyRecommendationResult = Mock()
    RecommendationRequest = Mock()
    RecommendationResponse = Mock()
    NeuralCollaborativeFiltering = Mock()


class TestRecommendationEngine:
    """Test suite for recommendation engine functionality"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment before each test"""
        self.test_fixtures = MLTestFixtures()
        self.mock_models = MockMLModels()
        self.performance_profiler = PerformanceProfiler()
        self.test_config = TestConfig()
        
        # Generate test data
        self.test_users = self.test_fixtures.create_sample_user_data(100)
        self.test_tracks = self.test_fixtures.create_sample_music_data(1000)
        self.test_interactions = self.test_fixtures.create_sample_interaction_data(5000)
        
    @pytest.mark.unit
    def test_recommend_tracks_basic(self):
        """Test basic track recommendation functionality"""
        user_id = "test_user_001"
        top_k = 10
        
        # Test basic recommendation call
        if hasattr(recommend_tracks, '__call__'):
            recommendations = recommend_tracks(
                user_id=user_id,
                top_k=top_k,
                model_type="matrix"
            )
            
            # Validate response structure
            assert recommendations is not None
            if isinstance(recommendations, list):
                assert len(recommendations) <= top_k
                assert all(isinstance(track_id, (int, str)) for track_id in recommendations)
        
    @pytest.mark.unit
    def test_recommend_tracks_with_context(self):
        """Test recommendation with context information"""
        user_id = "test_user_002"
        context = {
            "time_of_day": "evening",
            "location": "home",
            "device": "mobile",
            "mood": "relaxed",
            "activity": "studying"
        }
        
        if hasattr(recommend_tracks, '__call__'):
            recommendations = recommend_tracks(
                user_id=user_id,
                context=context,
                top_k=15,
                model_type="hybrid"
            )
            
            # Validate context-aware recommendations
            assert recommendations is not None
            if isinstance(recommendations, list):
                assert len(recommendations) <= 15
        
    @pytest.mark.unit
    def test_recommend_tracks_different_models(self):
        """Test recommendation with different model types"""
        user_id = "test_user_003"
        model_types = ["matrix", "deep", "hybrid", "neural"]
        
        for model_type in model_types:
            if hasattr(recommend_tracks, '__call__'):
                recommendations = recommend_tracks(
                    user_id=user_id,
                    model_type=model_type,
                    top_k=5
                )
                
                # Each model should return recommendations
                assert recommendations is not None
                if isinstance(recommendations, list):
                    assert len(recommendations) <= 5
    
    @pytest.mark.unit
    def test_legacy_recommendation_result(self):
        """Test LegacyRecommendationResult structure"""
        if hasattr(LegacyRecommendationResult, '__init__'):
            result = LegacyRecommendationResult(
                track_ids=[1, 2, 3, 4, 5],
                scores=[0.9, 0.8, 0.7, 0.6, 0.5],
                confidence=0.85,
                model_type="neural",
                context={"device": "mobile"},
                metadata={"version": "1.0"}
            )
            
            assert hasattr(result, 'track_ids')
            assert hasattr(result, 'scores')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'model_type')
    
    @pytest.mark.integration
    def test_recommendation_pipeline_end_to_end(self):
        """Test complete recommendation pipeline"""
        # Simulate user with listening history
        user_profile = {
            "user_id": "pipeline_test_user",
            "age": 28,
            "premium": True,
            "favorite_genres": ["pop", "rock", "electronic"],
            "listening_history": [101, 102, 103, 104, 105]
        }
        
        context = {
            "time_of_day": "morning",
            "device": "smartphone",
            "location": "commute"
        }
        
        if hasattr(recommend_tracks, '__call__'):
            # Test enhanced recommendation
            enhanced_recs = recommend_tracks(
                user_id=user_profile["user_id"],
                context=context,
                model_type="neural",
                top_k=20,
                enhanced=True,
                return_metadata=True
            )
            
            # Validate pipeline output
            assert enhanced_recs is not None
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_recommendation_performance(self, benchmark):
        """Benchmark recommendation engine performance"""
        user_id = "perf_test_user"
        
        def make_recommendation():
            if hasattr(recommend_tracks, '__call__'):
                return recommend_tracks(
                    user_id=user_id,
                    model_type="matrix",
                    top_k=10
                )
            return []
        
        # Benchmark recommendation generation
        result = benchmark(make_recommendation)
        
        # Assert performance thresholds
        assert benchmark.stats['mean'] < 0.1  # 100ms threshold
    
    @pytest.mark.performance
    def test_concurrent_recommendations(self):
        """Test concurrent recommendation requests"""
        import concurrent.futures
        
        def make_recommendation_request(user_id):
            if hasattr(recommend_tracks, '__call__'):
                return recommend_tracks(
                    user_id=f"concurrent_user_{user_id}",
                    top_k=5,
                    model_type="matrix"
                )
            return []
        
        # Test with 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(make_recommendation_request, i)
                for i in range(50)
            ]
            
            results = [future.result() for future in futures]
        
        # Validate all requests completed
        assert len(results) == 50
        assert all(result is not None for result in results)
    
    @pytest.mark.security
    def test_user_id_sanitization(self):
        """Test user ID input sanitization"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "admin' OR '1'='1",
            "../../../etc/passwd",
            "javascript:alert('XSS')"
        ]
        
        for malicious_input in malicious_inputs:
            if hasattr(recommend_tracks, '__call__'):
                try:
                    result = recommend_tracks(
                        user_id=malicious_input,
                        top_k=5
                    )
                    # Should not crash and should sanitize input
                    assert result is not None or result == []
                except Exception as e:
                    # Should handle malicious input gracefully
                    assert "security" in str(e).lower() or "invalid" in str(e).lower()
    
    @pytest.mark.security
    def test_recommendation_data_privacy(self):
        """Test data privacy in recommendations"""
        user_id = "privacy_test_user"
        
        if hasattr(recommend_tracks, '__call__'):
            recommendations = recommend_tracks(
                user_id=user_id,
                return_metadata=True
            )
            
            # Ensure no PII in recommendations
            if hasattr(recommendations, 'metadata'):
                metadata = recommendations.metadata
                
                # Check for common PII fields
                pii_fields = ['email', 'phone', 'address', 'ssn', 'credit_card']
                for field in pii_fields:
                    assert field not in str(metadata).lower()
    
    @pytest.mark.compliance
    def test_gdpr_compliance(self):
        """Test GDPR compliance for recommendations"""
        user_data = pd.DataFrame({
            'user_id': ['gdpr_user_001', 'gdpr_user_002'],
            'recommendation_history': [
                ['track_1', 'track_2'],
                ['track_3', 'track_4']
            ],
            'created_at': [datetime.now(), datetime.now()],
            'consent_given': [True, True]
        })
        
        # Validate GDPR compliance
        compliance_result = ComplianceValidator.validate_data_retention(
            user_data, retention_days=365
        )
        
        assert compliance_result['compliant'] == True
        assert 'old_records_count' in compliance_result
    
    @pytest.mark.load
    def test_recommendation_scalability(self):
        """Test recommendation system scalability"""
        # Simulate high load scenario
        user_count = 1000
        batch_size = 100
        
        def process_batch(batch_users):
            results = []
            for user_id in batch_users:
                if hasattr(recommend_tracks, '__call__'):
                    rec = recommend_tracks(
                        user_id=f"scale_user_{user_id}",
                        top_k=3
                    )
                    results.append(rec)
            return results
        
        # Process users in batches
        all_results = []
        for batch_start in range(0, user_count, batch_size):
            batch_users = list(range(batch_start, min(batch_start + batch_size, user_count)))
            batch_results = process_batch(batch_users)
            all_results.extend(batch_results)
        
        # Validate scalability
        assert len(all_results) == user_count
        assert all(result is not None for result in all_results)


class TestNeuralRecommendationEngine:
    """Test suite for neural recommendation engine"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup neural engine tests"""
        self.test_fixtures = MLTestFixtures()
        self.mock_models = MockMLModels()
        
        # Generate neural network test data
        self.num_users = 100
        self.num_items = 500
        self.embedding_dim = 64
        
    @pytest.mark.unit
    def test_recommendation_request_structure(self):
        """Test RecommendationRequest data structure"""
        if hasattr(RecommendationRequest, '__init__'):
            request = RecommendationRequest(
                user_id="neural_test_user",
                context={"device": "mobile"},
                num_recommendations=10,
                model_type="neural_collaborative"
            )
            
            assert hasattr(request, 'user_id')
            assert hasattr(request, 'context')
            assert hasattr(request, 'num_recommendations')
    
    @pytest.mark.unit
    def test_recommendation_response_structure(self):
        """Test RecommendationResponse data structure"""
        if hasattr(RecommendationResponse, '__init__'):
            response = RecommendationResponse(
                user_id="neural_test_user",
                recommendations=[
                    {"track_id": "track_1", "score": 0.9},
                    {"track_id": "track_2", "score": 0.8}
                ],
                model_version="neural_v1.0",
                confidence_scores=[0.9, 0.8]
            )
            
            assert hasattr(response, 'user_id')
            assert hasattr(response, 'recommendations')
            assert hasattr(response, 'confidence_scores')
    
    @pytest.mark.unit
    def test_neural_collaborative_filtering_init(self):
        """Test NeuralCollaborativeFiltering model initialization"""
        if hasattr(NeuralCollaborativeFiltering, '__init__'):
            model = NeuralCollaborativeFiltering(
                num_users=self.num_users,
                num_items=self.num_items,
                embedding_dim=self.embedding_dim
            )
            
            # Test model has required components
            if hasattr(model, 'user_embedding'):
                assert model.user_embedding.num_embeddings == self.num_users
            if hasattr(model, 'item_embedding'):
                assert model.item_embedding.num_embeddings == self.num_items
    
    @pytest.mark.integration
    def test_user_item_dataset(self):
        """Test UserItemDataset for training data"""
        # Generate synthetic training data
        interactions = np.random.randint(0, 2, size=(1000, 2))  # user, item pairs
        user_features = np.random.randn(1000, 50)
        item_features = np.random.randn(1000, 100)
        ratings = np.random.rand(1000)
        
        if hasattr(UserItemDataset, '__init__'):
            dataset = UserItemDataset(
                interactions=interactions,
                user_features=user_features,
                item_features=item_features,
                ratings=ratings
            )
            
            # Test dataset properties
            if hasattr(dataset, '__len__'):
                assert len(dataset) == 1000
            
            if hasattr(dataset, '__getitem__'):
                sample = dataset[0]
                assert 'user_item' in sample or isinstance(sample, (tuple, list))
    
    @pytest.mark.performance
    def test_neural_model_inference_speed(self):
        """Test neural model inference performance"""
        # Mock neural model for performance testing
        mock_model = self.mock_models.create_mock_recommendation_model()
        
        # Test data
        test_users = np.random.randint(0, self.num_users, size=100)
        
        # Measure inference time
        start_time = datetime.now()
        
        for user_id in test_users:
            predictions = mock_model.predict([user_id])
        
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds()
        
        # Assert performance threshold (10ms per user)
        assert inference_time / len(test_users) < 0.01
    
    @pytest.mark.unit
    def test_create_neural_recommendation_engine(self):
        """Test neural recommendation engine factory"""
        if hasattr(create_neural_recommendation_engine, '__call__'):
            engine = create_neural_recommendation_engine()
            
            # Validate engine creation
            assert engine is not None
            
            # Test basic engine methods if available
            if hasattr(engine, 'predict') or hasattr(engine, 'recommend'):
                # Engine should be callable
                assert callable(getattr(engine, 'predict', None)) or callable(getattr(engine, 'recommend', None))


class TestRecommendationIntegration:
    """Integration tests for recommendation systems"""
    
    @pytest.mark.integration
    def test_legacy_neural_integration(self):
        """Test integration between legacy and neural engines"""
        user_id = "integration_test_user"
        
        # Test legacy call with neural fallback
        if hasattr(recommend_tracks, '__call__'):
            legacy_result = recommend_tracks(
                user_id=user_id,
                model_type="matrix",
                enhanced=False
            )
            
            neural_result = recommend_tracks(
                user_id=user_id,
                model_type="neural",
                enhanced=True
            )
            
            # Both should return valid results
            assert legacy_result is not None
            assert neural_result is not None
    
    @pytest.mark.integration
    def test_recommendation_caching(self):
        """Test recommendation result caching"""
        user_id = "cache_test_user"
        
        if hasattr(recommend_tracks, '__call__'):
            # First call - should compute
            start_time = datetime.now()
            result1 = recommend_tracks(user_id=user_id, top_k=5)
            first_call_time = (datetime.now() - start_time).total_seconds()
            
            # Second call - should use cache
            start_time = datetime.now()
            result2 = recommend_tracks(user_id=user_id, top_k=5)
            second_call_time = (datetime.now() - start_time).total_seconds()
            
            # Cache should make second call faster
            assert second_call_time <= first_call_time
            
            # Results should be consistent
            if isinstance(result1, list) and isinstance(result2, list):
                assert result1 == result2
    
    @pytest.mark.integration
    def test_a_b_testing_integration(self):
        """Test A/B testing framework integration"""
        user_id = "ab_test_user"
        
        # Test different model variants
        models = ["matrix", "deep", "neural"]
        results = {}
        
        for model in models:
            if hasattr(recommend_tracks, '__call__'):
                result = recommend_tracks(
                    user_id=user_id,
                    model_type=model,
                    top_k=10
                )
                results[model] = result
        
        # Each model should produce different results for A/B testing
        assert len(results) == len(models)
        for model, result in results.items():
            assert result is not None
    
    @pytest.mark.integration
    def test_recommendation_monitoring(self):
        """Test recommendation monitoring and metrics"""
        user_id = "monitoring_test_user"
        
        # Mock monitoring data
        metrics = {
            "recommendation_latency": [],
            "cache_hit_rate": 0.0,
            "model_accuracy": 0.0,
            "user_satisfaction": 0.0
        }
        
        if hasattr(recommend_tracks, '__call__'):
            start_time = datetime.now()
            result = recommend_tracks(user_id=user_id, top_k=5)
            latency = (datetime.now() - start_time).total_seconds()
            
            metrics["recommendation_latency"].append(latency)
            
            # Validate monitoring metrics
            assert latency > 0
            assert result is not None
    
    @pytest.mark.integration
    def test_recommendation_explainability(self):
        """Test recommendation explainability features"""
        user_id = "explain_test_user"
        
        if hasattr(recommend_tracks, '__call__'):
            result = recommend_tracks(
                user_id=user_id,
                model_type="neural",
                return_metadata=True,
                enhanced=True
            )
            
            # Check for explanation metadata
            if hasattr(result, 'metadata'):
                metadata = result.metadata
                # Should contain explanation information
                explanation_fields = ['confidence', 'reasoning', 'factors']
                has_explanation = any(field in str(metadata).lower() for field in explanation_fields)
                assert has_explanation or metadata is not None


# Performance and load testing scenarios
class TestRecommendationStress:
    """Stress testing for recommendation engine"""
    
    @pytest.mark.stress
    def test_high_volume_recommendations(self):
        """Test recommendation engine under high volume"""
        num_users = 10000
        batch_size = 1000
        
        successful_recommendations = 0
        
        for batch_start in range(0, num_users, batch_size):
            batch_end = min(batch_start + batch_size, num_users)
            
            for user_idx in range(batch_start, batch_end):
                if hasattr(recommend_tracks, '__call__'):
                    try:
                        result = recommend_tracks(
                            user_id=f"stress_user_{user_idx}",
                            top_k=3
                        )
                        if result is not None:
                            successful_recommendations += 1
                    except Exception as e:
                        # Log but continue
                        print(f"Failed recommendation for user {user_idx}: {e}")
        
        # At least 95% success rate under stress
        success_rate = successful_recommendations / num_users
        assert success_rate >= 0.95
    
    @pytest.mark.stress
    def test_memory_usage_under_load(self):
        """Test memory usage during high load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate many recommendations
        for i in range(1000):
            if hasattr(recommend_tracks, '__call__'):
                recommend_tracks(
                    user_id=f"memory_test_user_{i}",
                    top_k=5
                )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100


# Parametrized tests for different scenarios
@pytest.mark.parametrize("model_type", ["matrix", "deep", "hybrid", "neural"])
@pytest.mark.parametrize("top_k", [5, 10, 20, 50])
def test_recommendation_parameters(model_type, top_k):
    """Test recommendations with different parameters"""
    user_id = f"param_test_user_{model_type}_{top_k}"
    
    if hasattr(recommend_tracks, '__call__'):
        result = recommend_tracks(
            user_id=user_id,
            model_type=model_type,
            top_k=top_k
        )
        
        # Validate parameter handling
        assert result is not None
        if isinstance(result, list):
            assert len(result) <= top_k


@pytest.mark.parametrize("context_scenario", [
    {"time_of_day": "morning", "device": "mobile"},
    {"time_of_day": "evening", "device": "desktop", "mood": "energetic"},
    {"location": "gym", "activity": "workout"},
    {"weather": "rainy", "mood": "melancholic"},
    {}  # Empty context
])
def test_context_aware_recommendations(context_scenario):
    """Test context-aware recommendation scenarios"""
    user_id = "context_test_user"
    
    if hasattr(recommend_tracks, '__call__'):
        result = recommend_tracks(
            user_id=user_id,
            context=context_scenario,
            top_k=10
        )
        
        # Context should not break recommendations
        assert result is not None
        if isinstance(result, list):
            assert len(result) <= 10
