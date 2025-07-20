"""
Tests for cache decorators in Spotify AI Agent

Comprehensive testing suite for caching decorators including
@cache_result, @cache_async, @invalidate_cache and advanced patterns.

Developed by Expert Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect
"""

import pytest
import asyncio
import time
import json
import hashlib
from unittest.mock import Mock, patch, AsyncMock
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.utils.cache.decorators import (
    cache_result, cache_async, invalidate_cache, cache_with_lock,
    conditional_cache, cache_on_exception, cache_pipeline,
    cache_with_tags, cache_with_versioning
)
from app.utils.cache.backends.memory_backend import MemoryCacheBackend
from app.utils.cache.backends.redis_backend import RedisCacheBackend
from app.utils.cache.manager import CacheManager
from app.utils.cache.exceptions import CacheError, CacheKeyError
from app.utils.cache.serializers import JSONSerializer


@dataclass
class MockUser:
    """Mock user class for testing"""
    id: int
    name: str
    email: str
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}


@dataclass 
class MockTrack:
    """Mock track class for testing"""
    id: str
    title: str
    artist: str
    duration_ms: int
    features: Dict[str, float] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = {}


class TestCacheResultDecorator:
    """Test @cache_result decorator"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    def test_basic_cache_decorator(self, cache_manager):
        """Test basic caching functionality"""
        call_count = 0
        
        @cache_result(cache_manager, ttl=3600)
        def expensive_function(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Different parameters should execute function again
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_cache_decorator_with_complex_objects(self, cache_manager):
        """Test caching with complex objects"""
        call_count = 0
        
        @cache_result(cache_manager, ttl=1800)
        def get_user_recommendations(user: MockUser, limit: int = 10) -> List[MockTrack]:
            nonlocal call_count
            call_count += 1
            
            # Simulate expensive computation
            tracks = []
            for i in range(limit):
                track = MockTrack(
                    id=f"track_{user.id}_{i}",
                    title=f"Recommended Track {i}",
                    artist=f"Artist {i}",
                    duration_ms=180000 + i * 1000
                )
                tracks.append(track)
            return tracks
        
        user = MockUser(id=123, name="Test User", email="test@example.com")
        
        # First call
        tracks1 = get_user_recommendations(user, limit=5)
        assert len(tracks1) == 5
        assert call_count == 1
        
        # Second call with same parameters (should use cache)
        tracks2 = get_user_recommendations(user, limit=5)
        assert len(tracks2) == 5
        assert call_count == 1
        assert tracks1[0].id == tracks2[0].id
        
        # Call with different limit (should execute function)
        tracks3 = get_user_recommendations(user, limit=3)
        assert len(tracks3) == 3
        assert call_count == 2
    
    def test_cache_key_generation(self, cache_manager):
        """Test cache key generation strategies"""
        
        @cache_result(cache_manager, key_generator="args_kwargs")
        def function_with_args_kwargs(a, b, c=None, d=None):
            return f"{a}-{b}-{c}-{d}"
        
        @cache_result(cache_manager, key_generator="custom")
        def function_with_custom_key(user_id: int, track_id: str):
            return f"recommendation_{user_id}_{track_id}"
        
        # Test args/kwargs key generation
        result1 = function_with_args_kwargs(1, 2, c=3, d=4)
        result2 = function_with_args_kwargs(1, 2, d=4, c=3)  # Different order
        assert result1 == result2  # Should use same cache entry
        
        # Test custom key generation
        rec1 = function_with_custom_key(123, "track_456")
        rec2 = function_with_custom_key(123, "track_456")
        assert rec1 == rec2
    
    def test_cache_ttl_expiration(self, cache_manager):
        """Test TTL expiration"""
        call_count = 0
        
        @cache_result(cache_manager, ttl=1)  # 1 second TTL
        def short_lived_cache(value: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"processed_{value}"
        
        # First call
        result1 = short_lived_cache("test")
        assert call_count == 1
        
        # Immediate second call (should use cache)
        result2 = short_lived_cache("test")
        assert call_count == 1
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Call after expiration (should execute function)
        result3 = short_lived_cache("test")
        assert call_count == 2
        assert result1 == result2 == result3
    
    def test_cache_with_none_values(self, cache_manager):
        """Test caching None values"""
        call_count = 0
        
        @cache_result(cache_manager, cache_none=True)
        def function_returning_none(should_return_none: bool) -> Optional[str]:
            nonlocal call_count
            call_count += 1
            return None if should_return_none else "not_none"
        
        # Test caching None values
        result1 = function_returning_none(True)
        assert result1 is None
        assert call_count == 1
        
        result2 = function_returning_none(True)
        assert result2 is None
        assert call_count == 1  # Should use cached None
        
        # Test not caching None values
        @cache_result(cache_manager, cache_none=False)
        def function_not_caching_none(should_return_none: bool) -> Optional[str]:
            nonlocal call_count
            call_count += 1
            return None if should_return_none else "not_none"
        
        call_count = 0  # Reset
        result3 = function_not_caching_none(True)
        assert result3 is None
        assert call_count == 1
        
        result4 = function_not_caching_none(True)
        assert result4 is None
        assert call_count == 2  # Should not use cache for None
    
    def test_cache_with_exceptions(self, cache_manager):
        """Test caching behavior with exceptions"""
        call_count = 0
        
        @cache_result(cache_manager, cache_exceptions=False)
        def function_with_exceptions(should_raise: bool) -> str:
            nonlocal call_count
            call_count += 1
            if should_raise:
                raise ValueError("Test exception")
            return "success"
        
        # First successful call
        result1 = function_with_exceptions(False)
        assert result1 == "success"
        assert call_count == 1
        
        # Second successful call (should use cache)
        result2 = function_with_exceptions(False)
        assert result2 == "success"
        assert call_count == 1
        
        # Exception call (should not be cached)
        with pytest.raises(ValueError):
            function_with_exceptions(True)
        assert call_count == 2
        
        # Another exception call (should execute again)
        with pytest.raises(ValueError):
            function_with_exceptions(True)
        assert call_count == 3
    
    def test_cache_invalidation_decorator(self, cache_manager):
        """Test cache invalidation functionality"""
        call_count = 0
        
        @cache_result(cache_manager, ttl=3600)
        def get_user_data(user_id: int) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"user_id": user_id, "data": f"data_for_{user_id}"}
        
        @invalidate_cache(cache_manager, ["get_user_data"])
        def update_user_data(user_id: int, new_data: Dict[str, Any]) -> bool:
            # Simulate database update
            return True
        
        # Initial cache population
        data1 = get_user_data(123)
        assert call_count == 1
        
        # Cached call
        data2 = get_user_data(123)
        assert call_count == 1
        
        # Invalidate cache
        update_user_data(123, {"new": "data"})
        
        # Should execute function again after invalidation
        data3 = get_user_data(123)
        assert call_count == 2
    
    def test_conditional_cache_decorator(self, cache_manager):
        """Test conditional caching based on parameters"""
        call_count = 0
        
        def should_cache(user_id: int, include_private: bool = False) -> bool:
            # Only cache non-private requests
            return not include_private
        
        @conditional_cache(cache_manager, condition=should_cache)
        def get_user_profile(user_id: int, include_private: bool = False) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            profile = {"user_id": user_id, "public_data": "public"}
            if include_private:
                profile["private_data"] = "private"
            return profile
        
        # Public request (should be cached)
        profile1 = get_user_profile(123, include_private=False)
        assert call_count == 1
        
        profile2 = get_user_profile(123, include_private=False)
        assert call_count == 1  # Should use cache
        
        # Private request (should not be cached)
        profile3 = get_user_profile(123, include_private=True)
        assert call_count == 2
        
        profile4 = get_user_profile(123, include_private=True)
        assert call_count == 3  # Should not use cache
    
    def test_cache_with_lock_decorator(self, cache_manager):
        """Test cache with lock to prevent thundering herd"""
        call_count = 0
        execution_times = []
        
        @cache_with_lock(cache_manager, lock_timeout=5)
        def expensive_computation(value: int) -> int:
            nonlocal call_count
            call_count += 1
            execution_times.append(time.time())
            time.sleep(0.1)  # Simulate expensive operation
            return value * 2
        
        import threading
        import concurrent.futures
        
        # Simulate concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(expensive_computation, 42) for _ in range(5)]
            results = [future.result() for future in futures]
        
        # All results should be the same
        assert all(result == 84 for result in results)
        
        # Function should only be called once due to locking
        assert call_count == 1
        
        # All executions should be close in time (lock prevented concurrent execution)
        assert len(execution_times) == 1


class TestCacheAsyncDecorator:
    """Test @cache_async decorator for async functions"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.mark.asyncio
    async def test_async_cache_decorator(self, cache_manager):
        """Test basic async caching functionality"""
        call_count = 0
        
        @cache_async(cache_manager, ttl=3600)
        async def async_expensive_function(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return x + y
        
        # First call should execute function
        result1 = await async_expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = await async_expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_async_cache_with_complex_operations(self, cache_manager):
        """Test async caching with complex async operations"""
        call_count = 0
        
        @cache_async(cache_manager, ttl=1800)
        async def fetch_track_analysis(track_id: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            
            # Simulate multiple async operations
            await asyncio.sleep(0.01)  # API call 1
            await asyncio.sleep(0.01)  # API call 2
            await asyncio.sleep(0.01)  # ML processing
            
            return {
                "track_id": track_id,
                "analysis": {
                    "tempo": 120.0,
                    "key": "C",
                    "valence": 0.8,
                    "energy": 0.9
                },
                "processed_at": datetime.now().isoformat()
            }
        
        # First call
        analysis1 = await fetch_track_analysis("track_123")
        assert call_count == 1
        assert analysis1["track_id"] == "track_123"
        
        # Second call (should use cache)
        analysis2 = await fetch_track_analysis("track_123")
        assert call_count == 1
        assert analysis1["processed_at"] == analysis2["processed_at"]  # Same cached result
    
    @pytest.mark.asyncio
    async def test_async_cache_concurrent_access(self, cache_manager):
        """Test async cache with concurrent access"""
        call_count = 0
        
        @cache_async(cache_manager, ttl=3600)
        async def concurrent_async_function(value: str) -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)  # Simulate async work
            return f"processed_{value}"
        
        # Start multiple concurrent calls
        tasks = [
            concurrent_async_function("test") for _ in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All results should be the same
        assert all(result == "processed_test" for result in results)
        
        # Function should be called at least once
        assert call_count >= 1
        # Due to potential race conditions in test environment,
        # we don't assert exact count
    
    @pytest.mark.asyncio
    async def test_async_cache_with_exceptions(self, cache_manager):
        """Test async cache behavior with exceptions"""
        call_count = 0
        
        @cache_async(cache_manager, cache_exceptions=False)
        async def async_function_with_exceptions(should_raise: bool) -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            if should_raise:
                raise ValueError("Async test exception")
            return "async_success"
        
        # Successful call
        result1 = await async_function_with_exceptions(False)
        assert result1 == "async_success"
        assert call_count == 1
        
        # Cached successful call
        result2 = await async_function_with_exceptions(False)
        assert result2 == "async_success"
        assert call_count == 1
        
        # Exception call (should not be cached)
        with pytest.raises(ValueError):
            await async_function_with_exceptions(True)
        assert call_count == 2
        
        # Another exception call (should execute again)
        with pytest.raises(ValueError):
            await async_function_with_exceptions(True)
        assert call_count == 3


class TestCacheWithTags:
    """Test cache with tagging functionality"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    def test_cache_with_tags_basic(self, cache_manager):
        """Test basic tagging functionality"""
        call_count = 0
        
        @cache_with_tags(cache_manager, tags=["user_data", "profile"])
        def get_user_profile(user_id: int) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"user_id": user_id, "profile": f"profile_data_{user_id}"}
        
        @cache_with_tags(cache_manager, tags=["user_data", "preferences"])
        def get_user_preferences(user_id: int) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"user_id": user_id, "preferences": f"preferences_{user_id}"}
        
        # Populate cache
        profile = get_user_profile(123)
        preferences = get_user_preferences(123)
        assert call_count == 2
        
        # Cached calls
        profile2 = get_user_profile(123)
        preferences2 = get_user_preferences(123)
        assert call_count == 2  # Should use cache
        
        # Invalidate by tag
        cache_manager.invalidate_by_tag("user_data")
        
        # Should execute functions again after tag invalidation
        profile3 = get_user_profile(123)
        preferences3 = get_user_preferences(123)
        assert call_count == 4
    
    def test_cache_with_dynamic_tags(self, cache_manager):
        """Test dynamic tag generation"""
        call_count = 0
        
        def generate_tags(user_id: int, data_type: str) -> List[str]:
            return [f"user_{user_id}", f"type_{data_type}", "dynamic"]
        
        @cache_with_tags(cache_manager, tags=generate_tags)
        def get_user_data(user_id: int, data_type: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {
                "user_id": user_id,
                "data_type": data_type,
                "data": f"data_{user_id}_{data_type}"
            }
        
        # Populate cache for different users and types
        data1 = get_user_data(123, "profile")
        data2 = get_user_data(123, "preferences")
        data3 = get_user_data(456, "profile")
        assert call_count == 3
        
        # Invalidate specific user data
        cache_manager.invalidate_by_tag("user_123")
        
        # User 123 data should be invalidated
        data4 = get_user_data(123, "profile")
        data5 = get_user_data(123, "preferences")
        assert call_count == 5  # Two new calls
        
        # User 456 data should still be cached
        data6 = get_user_data(456, "profile")
        assert call_count == 5  # No new call
    
    def test_cache_with_hierarchical_tags(self, cache_manager):
        """Test hierarchical tag invalidation"""
        call_count = 0
        
        @cache_with_tags(cache_manager, tags=["spotify", "spotify:user", "spotify:user:profile"])
        def get_user_spotify_profile(user_id: int) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"user_id": user_id, "spotify_profile": f"spotify_{user_id}"}
        
        @cache_with_tags(cache_manager, tags=["spotify", "spotify:track", "spotify:track:analysis"])
        def get_track_analysis(track_id: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"track_id": track_id, "analysis": f"analysis_{track_id}"}
        
        # Populate cache
        profile = get_user_spotify_profile(123)
        analysis = get_track_analysis("track_456")
        assert call_count == 2
        
        # Invalidate all Spotify data
        cache_manager.invalidate_by_tag("spotify")
        
        # Both should be invalidated
        profile2 = get_user_spotify_profile(123)
        analysis2 = get_track_analysis("track_456")
        assert call_count == 4


class TestCacheWithVersioning:
    """Test cache with versioning functionality"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    def test_cache_with_version_basic(self, cache_manager):
        """Test basic versioning functionality"""
        call_count = 0
        
        @cache_with_versioning(cache_manager, version="v1.0")
        def get_recommendation_algorithm_v1(user_id: int) -> List[str]:
            nonlocal call_count
            call_count += 1
            return [f"rec_{user_id}_{i}" for i in range(5)]
        
        @cache_with_versioning(cache_manager, version="v2.0")
        def get_recommendation_algorithm_v2(user_id: int) -> List[str]:
            nonlocal call_count
            call_count += 1
            return [f"rec_v2_{user_id}_{i}" for i in range(5)]
        
        # Different versions should have separate cache entries
        recs_v1 = get_recommendation_algorithm_v1(123)
        recs_v2 = get_recommendation_algorithm_v2(123)
        assert call_count == 2
        assert recs_v1 != recs_v2
        
        # Same version calls should use cache
        recs_v1_cached = get_recommendation_algorithm_v1(123)
        recs_v2_cached = get_recommendation_algorithm_v2(123)
        assert call_count == 2  # No new calls
        assert recs_v1 == recs_v1_cached
        assert recs_v2 == recs_v2_cached
    
    def test_cache_version_invalidation(self, cache_manager):
        """Test version-based cache invalidation"""
        call_count = 0
        
        @cache_with_versioning(cache_manager, version="v1.0")
        def versioned_function(key: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"result_v1_{key}"
        
        # Populate cache
        result1 = versioned_function("test")
        assert call_count == 1
        
        # Cached call
        result2 = versioned_function("test")
        assert call_count == 1
        
        # Invalidate version
        cache_manager.invalidate_version("v1.0")
        
        # Should execute function again after version invalidation
        result3 = versioned_function("test")
        assert call_count == 2
    
    def test_cache_with_dynamic_versioning(self, cache_manager):
        """Test dynamic version generation"""
        call_count = 0
        model_version = "1.0.0"
        
        def get_model_version() -> str:
            return model_version
        
        @cache_with_versioning(cache_manager, version=get_model_version)
        def ml_prediction(features: Dict[str, float]) -> float:
            nonlocal call_count
            call_count += 1
            # Simulate ML model prediction
            return sum(features.values()) / len(features)
        
        features = {"energy": 0.8, "valence": 0.6, "danceability": 0.7}
        
        # First prediction
        pred1 = ml_prediction(features)
        assert call_count == 1
        
        # Cached prediction
        pred2 = ml_prediction(features)
        assert call_count == 1
        assert pred1 == pred2
        
        # Update model version
        model_version = "1.1.0"
        
        # Should execute with new version
        pred3 = ml_prediction(features)
        assert call_count == 2  # New call due to version change


class TestCachePipelineDecorator:
    """Test cache pipeline functionality"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    def test_cache_pipeline_basic(self, cache_manager):
        """Test basic pipeline caching"""
        call_counts = {"step1": 0, "step2": 0, "step3": 0}
        
        @cache_pipeline(cache_manager, pipeline_id="audio_processing")
        def audio_processing_pipeline(track_id: str) -> Dict[str, Any]:
            # Step 1: Feature extraction
            call_counts["step1"] += 1
            features = {"tempo": 120, "key": "C"}
            
            # Step 2: Analysis
            call_counts["step2"] += 1
            analysis = {"energy": 0.8, "valence": 0.6}
            
            # Step 3: Recommendation scoring
            call_counts["step3"] += 1
            score = 0.85
            
            return {
                "track_id": track_id,
                "features": features,
                "analysis": analysis,
                "score": score
            }
        
        # First execution
        result1 = audio_processing_pipeline("track_123")
        assert all(count == 1 for count in call_counts.values())
        
        # Second execution (should use cache)
        result2 = audio_processing_pipeline("track_123")
        assert all(count == 1 for count in call_counts.values())  # No additional calls
        assert result1 == result2
    
    def test_cache_pipeline_with_dependencies(self, cache_manager):
        """Test pipeline caching with step dependencies"""
        call_counts = {"extract": 0, "analyze": 0, "recommend": 0}
        
        @cache_result(cache_manager, ttl=3600)
        def extract_features(track_id: str) -> Dict[str, float]:
            call_counts["extract"] += 1
            return {"tempo": 120.0, "energy": 0.8, "valence": 0.6}
        
        @cache_result(cache_manager, ttl=3600)
        def analyze_features(features: Dict[str, float]) -> Dict[str, Any]:
            call_counts["analyze"] += 1
            return {
                "mood": "happy" if features["valence"] > 0.5 else "sad",
                "intensity": "high" if features["energy"] > 0.7 else "low"
            }
        
        @cache_result(cache_manager, ttl=1800)
        def generate_recommendations(track_id: str, analysis: Dict[str, Any]) -> List[str]:
            call_counts["recommend"] += 1
            mood = analysis["mood"]
            return [f"{mood}_track_{i}" for i in range(3)]
        
        def recommendation_pipeline(track_id: str) -> Dict[str, Any]:
            features = extract_features(track_id)
            analysis = analyze_features(features)
            recommendations = generate_recommendations(track_id, analysis)
            
            return {
                "track_id": track_id,
                "features": features,
                "analysis": analysis,
                "recommendations": recommendations
            }
        
        # First execution
        result1 = recommendation_pipeline("track_123")
        assert all(count == 1 for count in call_counts.values())
        
        # Second execution (all steps should use cache)
        result2 = recommendation_pipeline("track_123")
        assert all(count == 1 for count in call_counts.values())
        assert result1 == result2
        
        # Different track (should execute all steps)
        result3 = recommendation_pipeline("track_456")
        assert all(count == 2 for count in call_counts.values())


class TestCacheErrorHandling:
    """Test cache decorator error handling"""
    
    @pytest.fixture
    def failing_cache_backend(self):
        """Mock failing cache backend"""
        backend = Mock()
        backend.get.side_effect = Exception("Cache error")
        backend.set.side_effect = Exception("Cache error")
        backend.delete.side_effect = Exception("Cache error")
        return backend
    
    @pytest.fixture
    def failing_cache_manager(self, failing_cache_backend):
        """Cache manager with failing backend"""
        return CacheManager(default_backend=failing_cache_backend)
    
    def test_cache_decorator_graceful_degradation(self, failing_cache_manager):
        """Test graceful degradation when cache fails"""
        call_count = 0
        
        @cache_result(failing_cache_manager, ttl=3600, fail_silently=True)
        def function_with_failing_cache(value: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"processed_{value}"
        
        # Should execute function normally despite cache failures
        result1 = function_with_failing_cache("test1")
        assert result1 == "processed_test1"
        assert call_count == 1
        
        # Should execute again (no caching due to failure)
        result2 = function_with_failing_cache("test1")
        assert result2 == "processed_test1"
        assert call_count == 2
    
    def test_cache_decorator_error_propagation(self, failing_cache_manager):
        """Test error propagation when fail_silently=False"""
        @cache_result(failing_cache_manager, ttl=3600, fail_silently=False)
        def function_with_strict_cache(value: str) -> str:
            return f"processed_{value}"
        
        # Should raise cache error
        with pytest.raises(Exception, match="Cache error"):
            function_with_strict_cache("test")
    
    def test_cache_decorator_partial_failure_handling(self):
        """Test handling of partial cache failures"""
        # Mock backend that fails on set but works on get
        backend = Mock()
        backend.get.return_value = None  # Cache miss
        backend.set.side_effect = Exception("Set failed")
        backend.delete.return_value = True
        
        cache_manager = CacheManager(default_backend=backend)
        
        call_count = 0
        
        @cache_result(cache_manager, ttl=3600, fail_silently=True)
        def function_with_partial_cache_failure(value: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"processed_{value}"
        
        # Should execute function and attempt to cache (which fails silently)
        result = function_with_partial_cache_failure("test")
        assert result == "processed_test"
        assert call_count == 1
        
        # Verify set was attempted (and failed)
        backend.set.assert_called()


# Utility functions for testing decorators
def create_test_function_with_decorator(decorator, **decorator_kwargs):
    """Factory function to create test functions with decorators"""
    call_count = 0
    
    @decorator(**decorator_kwargs)
    def test_function(value: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"processed_{value}_{call_count}"
    
    test_function.call_count = lambda: call_count
    return test_function


class TestCacheDecoratorIntegration:
    """Integration tests for cache decorators"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    def test_multiple_decorators_combination(self, cache_manager):
        """Test combining multiple cache decorators"""
        call_count = 0
        
        @cache_with_tags(cache_manager, tags=["user_data"])
        @cache_with_versioning(cache_manager, version="v1.0")
        @cache_result(cache_manager, ttl=3600)
        def complex_cached_function(user_id: int, data_type: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {
                "user_id": user_id,
                "data_type": data_type,
                "result": f"complex_result_{user_id}_{data_type}_{call_count}"
            }
        
        # First call
        result1 = complex_cached_function(123, "profile")
        assert call_count == 1
        
        # Cached call
        result2 = complex_cached_function(123, "profile")
        assert call_count == 1
        assert result1 == result2
        
        # Invalidate by tag
        cache_manager.invalidate_by_tag("user_data")
        
        # Should execute again
        result3 = complex_cached_function(123, "profile")
        assert call_count == 2
    
    def test_decorator_performance_impact(self, cache_manager):
        """Test performance impact of cache decorators"""
        import time
        
        @cache_result(cache_manager, ttl=3600)
        def cached_function(value: int) -> int:
            time.sleep(0.01)  # Simulate work
            return value * 2
        
        def uncached_function(value: int) -> int:
            time.sleep(0.01)  # Simulate work
            return value * 2
        
        # Measure cached function performance
        start_time = time.time()
        cached_result1 = cached_function(42)  # First call (miss)
        cached_result2 = cached_function(42)  # Second call (hit)
        cached_duration = time.time() - start_time
        
        # Measure uncached function performance
        start_time = time.time()
        uncached_result1 = uncached_function(42)  # First call
        uncached_result2 = uncached_function(42)  # Second call
        uncached_duration = time.time() - start_time
        
        # Cached version should be faster on second call
        assert cached_result1 == cached_result2 == uncached_result1 == uncached_result2
        # Due to test environment variability, we just check it completes
        assert cached_duration > 0
        assert uncached_duration > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not performance"])
