"""
Tests for cache invalidation strategies in Spotify AI Agent

Comprehensive testing suite for cache invalidation including
tag-based, pattern-based, dependency-based and time-based invalidation.

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
import re
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Set, Optional, Pattern
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.utils.cache.invalidation import (
    TagBasedInvalidation, PatternBasedInvalidation,
    DependencyBasedInvalidation, TimeBasedInvalidation,
    ConditionalInvalidation, BulkInvalidation,
    InvalidationManager, InvalidationStrategy
)
from app.utils.cache.backends.memory_backend import MemoryCacheBackend
from app.utils.cache.backends.redis_backend import RedisCacheBackend
from app.utils.cache.manager import CacheManager
from app.utils.cache.events import CacheEvent, InvalidationEvent


@dataclass
class TestCacheEntry:
    """Test cache entry for invalidation testing"""
    key: str
    value: Any
    tags: List[str] = None
    dependencies: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now()


class TestTagBasedInvalidation:
    """Test tag-based cache invalidation"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.fixture
    def tag_invalidator(self, cache_manager):
        """Tag-based invalidator fixture"""
        return TagBasedInvalidation(cache_manager)
    
    def test_tag_based_invalidation_basic(self, tag_invalidator, cache_manager):
        """Test basic tag-based invalidation"""
        # Set up cache entries with tags
        cache_entries = {
            "user:123:profile": TestCacheEntry("user:123:profile", {"name": "John"}, ["user", "profile", "user:123"]),
            "user:123:preferences": TestCacheEntry("user:123:preferences", {"theme": "dark"}, ["user", "preferences", "user:123"]),
            "user:456:profile": TestCacheEntry("user:456:profile", {"name": "Jane"}, ["user", "profile", "user:456"]),
            "track:789:metadata": TestCacheEntry("track:789:metadata", {"title": "Song"}, ["track", "metadata"]),
        }
        
        # Populate cache with tagged entries
        for key, entry in cache_entries.items():
            cache_manager.set(key, entry.value, tags=entry.tags)
        
        # Verify all entries exist
        for key in cache_entries.keys():
            assert cache_manager.exists(key)
        
        # Invalidate by user tag
        invalidated_keys = tag_invalidator.invalidate_by_tag("user")
        
        # Check that user-related entries are invalidated
        user_keys = ["user:123:profile", "user:123:preferences", "user:456:profile"]
        assert all(key in invalidated_keys for key in user_keys)
        assert "track:789:metadata" not in invalidated_keys
        
        # Verify cache state
        for key in user_keys:
            assert not cache_manager.exists(key)
        assert cache_manager.exists("track:789:metadata")
    
    def test_tag_based_invalidation_specific_user(self, tag_invalidator, cache_manager):
        """Test invalidation of specific user data"""
        # Set up cache entries
        cache_entries = {
            "user:123:profile": {"name": "John", "age": 25},
            "user:123:playlists": ["playlist1", "playlist2"],
            "user:456:profile": {"name": "Jane", "age": 30},
            "global:stats": {"total_users": 1000}
        }
        
        # Populate cache with specific tags
        cache_manager.set("user:123:profile", cache_entries["user:123:profile"], tags=["user:123", "profile"])
        cache_manager.set("user:123:playlists", cache_entries["user:123:playlists"], tags=["user:123", "playlists"])
        cache_manager.set("user:456:profile", cache_entries["user:456:profile"], tags=["user:456", "profile"])
        cache_manager.set("global:stats", cache_entries["global:stats"], tags=["global", "stats"])
        
        # Invalidate specific user data
        invalidated_keys = tag_invalidator.invalidate_by_tag("user:123")
        
        # Check results
        assert "user:123:profile" in invalidated_keys
        assert "user:123:playlists" in invalidated_keys
        assert "user:456:profile" not in invalidated_keys
        assert "global:stats" not in invalidated_keys
        
        # Verify cache state
        assert not cache_manager.exists("user:123:profile")
        assert not cache_manager.exists("user:123:playlists")
        assert cache_manager.exists("user:456:profile")
        assert cache_manager.exists("global:stats")
    
    def test_tag_based_invalidation_multiple_tags(self, tag_invalidator, cache_manager):
        """Test invalidation with multiple tags"""
        # Set up cache entries
        cache_manager.set("user:123:recent_tracks", ["track1", "track2"], tags=["user:123", "tracks", "recent"])
        cache_manager.set("user:123:liked_tracks", ["track3", "track4"], tags=["user:123", "tracks", "liked"])
        cache_manager.set("user:123:profile", {"name": "John"}, tags=["user:123", "profile"])
        cache_manager.set("global:popular_tracks", ["track5", "track6"], tags=["global", "tracks", "popular"])
        
        # Invalidate by multiple tags
        invalidated_keys = tag_invalidator.invalidate_by_tags(["user:123", "tracks"])
        
        # Check that entries with ANY of the tags are invalidated
        expected_keys = ["user:123:recent_tracks", "user:123:liked_tracks", "user:123:profile"]
        assert all(key in invalidated_keys for key in expected_keys)
        assert "global:popular_tracks" not in invalidated_keys
    
    def test_tag_based_invalidation_hierarchical(self, tag_invalidator, cache_manager):
        """Test hierarchical tag invalidation"""
        # Set up hierarchical cache structure
        cache_entries = {
            "spotify:user:123:profile": {"name": "John"},
            "spotify:user:123:playlists": ["p1", "p2"],
            "spotify:track:456:metadata": {"title": "Song"},
            "spotify:track:456:analysis": {"tempo": 120},
            "lastfm:user:123:scrobbles": ["s1", "s2"]
        }
        
        # Populate with hierarchical tags
        cache_manager.set("spotify:user:123:profile", cache_entries["spotify:user:123:profile"], 
                         tags=["spotify", "spotify:user", "spotify:user:123"])
        cache_manager.set("spotify:user:123:playlists", cache_entries["spotify:user:123:playlists"],
                         tags=["spotify", "spotify:user", "spotify:user:123"])
        cache_manager.set("spotify:track:456:metadata", cache_entries["spotify:track:456:metadata"],
                         tags=["spotify", "spotify:track", "spotify:track:456"])
        cache_manager.set("spotify:track:456:analysis", cache_entries["spotify:track:456:analysis"],
                         tags=["spotify", "spotify:track", "spotify:track:456"])
        cache_manager.set("lastfm:user:123:scrobbles", cache_entries["lastfm:user:123:scrobbles"],
                         tags=["lastfm", "lastfm:user", "lastfm:user:123"])
        
        # Invalidate all Spotify data
        invalidated_keys = tag_invalidator.invalidate_by_tag("spotify")
        
        # Check that only Spotify entries are invalidated
        spotify_keys = [
            "spotify:user:123:profile", "spotify:user:123:playlists",
            "spotify:track:456:metadata", "spotify:track:456:analysis"
        ]
        assert all(key in invalidated_keys for key in spotify_keys)
        assert "lastfm:user:123:scrobbles" not in invalidated_keys
        
        # Verify Last.fm data still exists
        assert cache_manager.exists("lastfm:user:123:scrobbles")
    
    def test_tag_based_invalidation_with_wildcards(self, tag_invalidator, cache_manager):
        """Test tag invalidation with wildcard patterns"""
        # Set up cache entries
        for user_id in [123, 456, 789]:
            cache_manager.set(f"user:{user_id}:profile", {"id": user_id}, 
                             tags=[f"user:{user_id}", "profile"])
            cache_manager.set(f"user:{user_id}:settings", {"theme": "dark"}, 
                             tags=[f"user:{user_id}", "settings"])
        
        # Invalidate using wildcard pattern
        invalidated_keys = tag_invalidator.invalidate_by_pattern("user:*")
        
        # Check that all user entries are invalidated
        expected_keys = [
            "user:123:profile", "user:123:settings",
            "user:456:profile", "user:456:settings", 
            "user:789:profile", "user:789:settings"
        ]
        assert len(invalidated_keys) == len(expected_keys)
    
    def test_tag_based_invalidation_events(self, tag_invalidator, cache_manager):
        """Test invalidation event generation"""
        events = []
        
        def event_handler(event: InvalidationEvent):
            events.append(event)
        
        # Subscribe to invalidation events
        tag_invalidator.subscribe_to_events(event_handler)
        
        # Set up cache entries
        cache_manager.set("test:key1", "value1", tags=["test"])
        cache_manager.set("test:key2", "value2", tags=["test"])
        
        # Invalidate by tag
        tag_invalidator.invalidate_by_tag("test")
        
        # Check events
        assert len(events) == 1
        event = events[0]
        assert event.invalidation_type == "tag"
        assert event.tag == "test"
        assert len(event.invalidated_keys) == 2
        assert "test:key1" in event.invalidated_keys
        assert "test:key2" in event.invalidated_keys


class TestPatternBasedInvalidation:
    """Test pattern-based cache invalidation"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.fixture
    def pattern_invalidator(self, cache_manager):
        """Pattern-based invalidator fixture"""
        return PatternBasedInvalidation(cache_manager)
    
    def test_pattern_invalidation_wildcard(self, pattern_invalidator, cache_manager):
        """Test pattern invalidation with wildcards"""
        # Set up cache entries
        cache_entries = {
            "user:123:profile": {"name": "John"},
            "user:123:settings": {"theme": "dark"},
            "user:456:profile": {"name": "Jane"},
            "track:789:metadata": {"title": "Song"},
            "global:config": {"version": "1.0"}
        }
        
        # Populate cache
        for key, value in cache_entries.items():
            cache_manager.set(key, value)
        
        # Invalidate user data with wildcard
        invalidated_keys = pattern_invalidator.invalidate_by_pattern("user:*")
        
        # Check results
        user_keys = ["user:123:profile", "user:123:settings", "user:456:profile"]
        assert all(key in invalidated_keys for key in user_keys)
        assert "track:789:metadata" not in invalidated_keys
        assert "global:config" not in invalidated_keys
        
        # Verify cache state
        for key in user_keys:
            assert not cache_manager.exists(key)
        assert cache_manager.exists("track:789:metadata")
        assert cache_manager.exists("global:config")
    
    def test_pattern_invalidation_regex(self, pattern_invalidator, cache_manager):
        """Test pattern invalidation with regex"""
        # Set up cache entries with different formats
        cache_entries = {
            "user_123_profile": {"name": "John"},
            "user-456-profile": {"name": "Jane"},
            "user.789.profile": {"name": "Bob"},
            "track_123_metadata": {"title": "Song1"},
            "playlist_456_tracks": ["t1", "t2"]
        }
        
        # Populate cache
        for key, value in cache_entries.items():
            cache_manager.set(key, value)
        
        # Invalidate user profiles with regex (any separator)
        pattern = re.compile(r"user[._-]\d+[._-]profile")
        invalidated_keys = pattern_invalidator.invalidate_by_regex(pattern)
        
        # Check results
        profile_keys = ["user_123_profile", "user-456-profile", "user.789.profile"]
        assert all(key in invalidated_keys for key in profile_keys)
        assert "track_123_metadata" not in invalidated_keys
        assert "playlist_456_tracks" not in invalidated_keys
    
    def test_pattern_invalidation_prefix(self, pattern_invalidator, cache_manager):
        """Test prefix-based invalidation"""
        # Set up cache entries
        prefixes = ["spotify:", "lastfm:", "youtube:"]
        for prefix in prefixes:
            for i in range(3):
                key = f"{prefix}data_{i}"
                cache_manager.set(key, f"value_{i}")
        
        # Invalidate by prefix
        invalidated_keys = pattern_invalidator.invalidate_by_prefix("spotify:")
        
        # Check results
        expected_keys = ["spotify:data_0", "spotify:data_1", "spotify:data_2"]
        assert len(invalidated_keys) == 3
        assert all(key.startswith("spotify:") for key in invalidated_keys)
        
        # Verify other prefixes still exist
        assert cache_manager.exists("lastfm:data_0")
        assert cache_manager.exists("youtube:data_0")
    
    def test_pattern_invalidation_suffix(self, pattern_invalidator, cache_manager):
        """Test suffix-based invalidation"""
        # Set up cache entries
        suffixes = [":metadata", ":analysis", ":recommendations"]
        for user_id in [123, 456]:
            for suffix in suffixes:
                key = f"user:{user_id}{suffix}"
                cache_manager.set(key, f"data_for_{user_id}")
        
        # Invalidate by suffix
        invalidated_keys = pattern_invalidator.invalidate_by_suffix(":metadata")
        
        # Check results
        expected_keys = ["user:123:metadata", "user:456:metadata"]
        assert len(invalidated_keys) == 2
        assert all(key.endswith(":metadata") for key in invalidated_keys)
        
        # Verify other suffixes still exist
        assert cache_manager.exists("user:123:analysis")
        assert cache_manager.exists("user:456:recommendations")
    
    def test_pattern_invalidation_complex_patterns(self, pattern_invalidator, cache_manager):
        """Test complex pattern invalidation"""
        # Set up cache entries with timestamp patterns
        timestamps = ["2024-01-01", "2024-01-02", "2024-02-01", "2023-12-31"]
        for ts in timestamps:
            for data_type in ["logs", "metrics", "events"]:
                key = f"analytics:{ts}:{data_type}"
                cache_manager.set(key, f"data_for_{ts}_{data_type}")
        
        # Invalidate January 2024 data
        pattern = "analytics:2024-01-*"
        invalidated_keys = pattern_invalidator.invalidate_by_pattern(pattern)
        
        # Check results
        january_keys = [
            "analytics:2024-01-01:logs", "analytics:2024-01-01:metrics", "analytics:2024-01-01:events",
            "analytics:2024-01-02:logs", "analytics:2024-01-02:metrics", "analytics:2024-01-02:events"
        ]
        assert len(invalidated_keys) == 6
        assert all(key in invalidated_keys for key in january_keys)
        
        # Verify other dates still exist
        assert cache_manager.exists("analytics:2024-02-01:logs")
        assert cache_manager.exists("analytics:2023-12-31:logs")
    
    def test_pattern_invalidation_performance(self, pattern_invalidator, cache_manager):
        """Test pattern invalidation performance with large datasets"""
        # Set up large number of cache entries
        num_entries = 1000
        for i in range(num_entries):
            user_id = i % 100  # 100 different users
            data_type = ["profile", "settings", "history"][i % 3]
            key = f"user:{user_id}:{data_type}:{i}"
            cache_manager.set(key, f"data_{i}")
        
        # Measure invalidation performance
        start_time = time.time()
        invalidated_keys = pattern_invalidator.invalidate_by_pattern("user:1:*")
        duration = time.time() - start_time
        
        # Check performance (should complete quickly)
        assert duration < 1.0, f"Pattern invalidation too slow: {duration}s"
        
        # Check correctness
        assert len(invalidated_keys) > 0
        assert all(key.startswith("user:1:") for key in invalidated_keys)


class TestDependencyBasedInvalidation:
    """Test dependency-based cache invalidation"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.fixture
    def dependency_invalidator(self, cache_manager):
        """Dependency-based invalidator fixture"""
        return DependencyBasedInvalidation(cache_manager)
    
    def test_dependency_invalidation_basic(self, dependency_invalidator, cache_manager):
        """Test basic dependency-based invalidation"""
        # Set up dependency chain: user_profile -> user_recommendations -> playlist_suggestions
        
        # Base data
        cache_manager.set("user:123:profile", {"name": "John", "age": 25}, dependencies=[])
        
        # Dependent on user profile
        cache_manager.set("user:123:recommendations", ["track1", "track2"], 
                         dependencies=["user:123:profile"])
        
        # Dependent on recommendations
        cache_manager.set("user:123:playlist_suggestions", ["playlist1", "playlist2"],
                         dependencies=["user:123:recommendations"])
        
        # Dependent on both profile and recommendations
        cache_manager.set("user:123:dashboard", {"recs": [], "suggestions": []},
                         dependencies=["user:123:profile", "user:123:recommendations"])
        
        # Verify all entries exist
        keys = ["user:123:profile", "user:123:recommendations", 
                "user:123:playlist_suggestions", "user:123:dashboard"]
        for key in keys:
            assert cache_manager.exists(key)
        
        # Invalidate base profile
        invalidated_keys = dependency_invalidator.invalidate_dependencies("user:123:profile")
        
        # Check cascade invalidation
        expected_invalidated = [
            "user:123:recommendations",  # Direct dependent
            "user:123:playlist_suggestions",  # Indirect dependent (through recommendations)
            "user:123:dashboard"  # Direct dependent on profile
        ]
        
        assert all(key in invalidated_keys for key in expected_invalidated)
        assert "user:123:profile" not in invalidated_keys  # Source key not invalidated
        
        # Verify cache state
        assert cache_manager.exists("user:123:profile")  # Source remains
        for key in expected_invalidated:
            assert not cache_manager.exists(key)
    
    def test_dependency_invalidation_circular(self, dependency_invalidator, cache_manager):
        """Test handling of circular dependencies"""
        # Set up circular dependency (should be handled gracefully)
        cache_manager.set("key_a", "value_a", dependencies=["key_b"])
        cache_manager.set("key_b", "value_b", dependencies=["key_c"])
        cache_manager.set("key_c", "value_c", dependencies=["key_a"])  # Circular
        cache_manager.set("key_d", "value_d", dependencies=["key_b"])
        
        # Invalidate one key in the circular chain
        invalidated_keys = dependency_invalidator.invalidate_dependencies("key_a")
        
        # Should handle circular dependency without infinite loop
        assert len(invalidated_keys) > 0
        assert "key_b" in invalidated_keys or "key_c" in invalidated_keys
        
        # Should complete in reasonable time (no infinite loop)
        # The exact behavior may vary based on implementation
    
    def test_dependency_invalidation_complex_graph(self, dependency_invalidator, cache_manager):
        """Test dependency invalidation with complex dependency graph"""
        # Set up complex dependency graph
        dependencies = {
            "user:123:raw_data": [],
            "user:123:processed_data": ["user:123:raw_data"],
            "user:123:features": ["user:123:processed_data"],
            "user:123:ml_model": ["user:123:features"],
            "user:123:predictions": ["user:123:ml_model", "user:123:features"],
            "user:123:recommendations": ["user:123:predictions"],
            "user:123:ui_data": ["user:123:recommendations", "user:123:processed_data"],
            "global:stats": ["user:123:processed_data"]  # Global stat depends on user data
        }
        
        # Populate cache with dependencies
        for key, deps in dependencies.items():
            cache_manager.set(key, f"value_for_{key}", dependencies=deps)
        
        # Invalidate raw data (should cascade through the entire chain)
        invalidated_keys = dependency_invalidator.invalidate_dependencies("user:123:raw_data")
        
        # Check that all dependent keys are invalidated
        expected_invalidated = [
            "user:123:processed_data", "user:123:features", "user:123:ml_model",
            "user:123:predictions", "user:123:recommendations", "user:123:ui_data",
            "global:stats"
        ]
        
        assert all(key in invalidated_keys for key in expected_invalidated)
        assert "user:123:raw_data" not in invalidated_keys  # Source not invalidated
        
        # Verify cache state
        assert cache_manager.exists("user:123:raw_data")
        for key in expected_invalidated:
            assert not cache_manager.exists(key)
    
    def test_dependency_invalidation_partial_chain(self, dependency_invalidator, cache_manager):
        """Test invalidation of partial dependency chain"""
        # Set up dependency chain
        cache_manager.set("step1", "value1", dependencies=[])
        cache_manager.set("step2", "value2", dependencies=["step1"])
        cache_manager.set("step3", "value3", dependencies=["step2"])
        cache_manager.set("step4", "value4", dependencies=["step3"])
        cache_manager.set("independent", "value_indep", dependencies=[])
        
        # Invalidate middle of chain
        invalidated_keys = dependency_invalidator.invalidate_dependencies("step2")
        
        # Check that only downstream dependencies are invalidated
        assert "step3" in invalidated_keys
        assert "step4" in invalidated_keys
        assert "step1" not in invalidated_keys  # Upstream not affected
        assert "step2" not in invalidated_keys  # Source not invalidated
        assert "independent" not in invalidated_keys  # Independent key not affected
        
        # Verify cache state
        assert cache_manager.exists("step1")
        assert cache_manager.exists("step2")
        assert cache_manager.exists("independent")
        assert not cache_manager.exists("step3")
        assert not cache_manager.exists("step4")
    
    def test_dependency_invalidation_multiple_roots(self, dependency_invalidator, cache_manager):
        """Test invalidation with multiple dependency roots"""
        # Set up multiple dependency trees
        cache_manager.set("user_data", "user_value", dependencies=[])
        cache_manager.set("track_data", "track_value", dependencies=[])
        cache_manager.set("combined_analysis", "combined_value", 
                         dependencies=["user_data", "track_data"])
        cache_manager.set("recommendations", "recs_value", 
                         dependencies=["combined_analysis"])
        
        # Invalidate one root
        invalidated_keys = dependency_invalidator.invalidate_dependencies("user_data")
        
        # Check that combined analysis and its dependents are invalidated
        assert "combined_analysis" in invalidated_keys
        assert "recommendations" in invalidated_keys
        assert "track_data" not in invalidated_keys
        
        # Verify cache state
        assert cache_manager.exists("user_data")
        assert cache_manager.exists("track_data")
        assert not cache_manager.exists("combined_analysis")
        assert not cache_manager.exists("recommendations")


class TestTimeBasedInvalidation:
    """Test time-based cache invalidation"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.fixture
    def time_invalidator(self, cache_manager):
        """Time-based invalidator fixture"""
        return TimeBasedInvalidation(cache_manager)
    
    def test_time_based_invalidation_expired(self, time_invalidator, cache_manager):
        """Test invalidation of expired entries"""
        # Set entries with different TTLs
        cache_manager.set("short_lived", "value1", ttl=1)
        cache_manager.set("medium_lived", "value2", ttl=5)
        cache_manager.set("long_lived", "value3", ttl=3600)
        
        # Wait for short-lived entry to expire
        time.sleep(1.1)
        
        # Invalidate expired entries
        invalidated_keys = time_invalidator.invalidate_expired()
        
        # Check results
        assert "short_lived" in invalidated_keys
        assert "medium_lived" not in invalidated_keys
        assert "long_lived" not in invalidated_keys
        
        # Verify cache state
        assert not cache_manager.exists("short_lived")
        assert cache_manager.exists("medium_lived")
        assert cache_manager.exists("long_lived")
    
    def test_time_based_invalidation_by_age(self, time_invalidator, cache_manager):
        """Test invalidation by entry age"""
        # Set entries at different times
        cache_manager.set("old_entry", "value1")
        time.sleep(0.1)
        cache_manager.set("newer_entry", "value2")
        time.sleep(0.1)
        cache_manager.set("newest_entry", "value3")
        
        # Invalidate entries older than 0.15 seconds
        max_age = timedelta(seconds=0.15)
        invalidated_keys = time_invalidator.invalidate_by_age(max_age)
        
        # Check results (exact timing may vary in test environment)
        assert len(invalidated_keys) >= 1
        if "old_entry" in invalidated_keys:
            assert not cache_manager.exists("old_entry")
    
    def test_time_based_invalidation_scheduled(self, time_invalidator, cache_manager):
        """Test scheduled invalidation"""
        # Set up entries with different creation times
        now = datetime.now()
        
        # Mock creation times
        cache_manager.set("entry1", "value1")
        cache_manager.set("entry2", "value2")
        
        # Schedule invalidation for specific time
        future_time = now + timedelta(seconds=1)
        time_invalidator.schedule_invalidation("entry1", future_time)
        
        # Check that entry still exists
        assert cache_manager.exists("entry1")
        
        # Wait for scheduled time
        time.sleep(1.1)
        
        # Process scheduled invalidations
        invalidated_keys = time_invalidator.process_scheduled_invalidations()
        
        # Check results
        assert "entry1" in invalidated_keys
        assert not cache_manager.exists("entry1")
        assert cache_manager.exists("entry2")
    
    def test_time_based_invalidation_cleanup(self, time_invalidator, cache_manager):
        """Test cleanup of old entries"""
        # Set up many entries with short TTLs
        for i in range(10):
            cache_manager.set(f"temp_entry_{i}", f"value_{i}", ttl=1)
        
        # Add some long-lived entries
        for i in range(5):
            cache_manager.set(f"persistent_entry_{i}", f"value_{i}", ttl=3600)
        
        # Wait for short-lived entries to expire
        time.sleep(1.1)
        
        # Run cleanup
        cleaned_count = time_invalidator.cleanup_expired_entries()
        
        # Check results
        assert cleaned_count >= 10  # At least the temp entries
        
        # Verify persistent entries still exist
        for i in range(5):
            assert cache_manager.exists(f"persistent_entry_{i}")


class TestConditionalInvalidation:
    """Test conditional cache invalidation"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.fixture
    def conditional_invalidator(self, cache_manager):
        """Conditional invalidator fixture"""
        return ConditionalInvalidation(cache_manager)
    
    def test_conditional_invalidation_value_based(self, conditional_invalidator, cache_manager):
        """Test conditional invalidation based on cached values"""
        # Set up cache entries with different values
        cache_manager.set("user:123:status", "active")
        cache_manager.set("user:456:status", "inactive")
        cache_manager.set("user:789:status", "suspended")
        cache_manager.set("user:101:data", {"status": "active", "premium": True})
        
        # Define condition: invalidate entries with "active" status
        def is_active_status(key: str, value: Any) -> bool:
            if isinstance(value, str):
                return value == "active"
            elif isinstance(value, dict):
                return value.get("status") == "active"
            return False
        
        # Invalidate based on condition
        invalidated_keys = conditional_invalidator.invalidate_by_condition(is_active_status)
        
        # Check results
        assert "user:123:status" in invalidated_keys
        assert "user:101:data" in invalidated_keys
        assert "user:456:status" not in invalidated_keys
        assert "user:789:status" not in invalidated_keys
        
        # Verify cache state
        assert not cache_manager.exists("user:123:status")
        assert cache_manager.exists("user:456:status")
        assert cache_manager.exists("user:789:status")
    
    def test_conditional_invalidation_key_based(self, conditional_invalidator, cache_manager):
        """Test conditional invalidation based on key patterns"""
        # Set up cache entries
        cache_entries = [
            "user:123:temp_data", "user:456:persistent_data",
            "session:abc:temp_data", "session:def:persistent_data",
            "global:temp_config", "global:persistent_config"
        ]
        
        for key in cache_entries:
            cache_manager.set(key, f"value_for_{key}")
        
        # Define condition: invalidate temporary data
        def is_temp_data(key: str, value: Any) -> bool:
            return "temp" in key
        
        # Invalidate based on condition
        invalidated_keys = conditional_invalidator.invalidate_by_condition(is_temp_data)
        
        # Check results
        temp_keys = ["user:123:temp_data", "session:abc:temp_data", "global:temp_config"]
        assert all(key in invalidated_keys for key in temp_keys)
        
        persistent_keys = ["user:456:persistent_data", "session:def:persistent_data", "global:persistent_config"]
        assert all(key not in invalidated_keys for key in persistent_keys)
        
        # Verify cache state
        for key in temp_keys:
            assert not cache_manager.exists(key)
        for key in persistent_keys:
            assert cache_manager.exists(key)
    
    def test_conditional_invalidation_complex_condition(self, conditional_invalidator, cache_manager):
        """Test conditional invalidation with complex conditions"""
        # Set up cache entries with complex data
        users_data = [
            ("user:123", {"name": "John", "age": 25, "premium": True, "last_login": "2024-01-01"}),
            ("user:456", {"name": "Jane", "age": 17, "premium": False, "last_login": "2024-01-15"}),
            ("user:789", {"name": "Bob", "age": 30, "premium": True, "last_login": "2023-12-01"}),
            ("user:101", {"name": "Alice", "age": 22, "premium": False, "last_login": "2024-01-10"})
        ]
        
        for key, data in users_data:
            cache_manager.set(key, data)
        
        # Define complex condition: invalidate non-premium users or inactive users
        def should_invalidate(key: str, value: Any) -> bool:
            if not isinstance(value, dict):
                return False
            
            # Non-premium users
            if not value.get("premium", False):
                return True
            
            # Inactive users (no login in 2024)
            last_login = value.get("last_login", "")
            if not last_login.startswith("2024"):
                return True
            
            return False
        
        # Invalidate based on complex condition
        invalidated_keys = conditional_invalidator.invalidate_by_condition(should_invalidate)
        
        # Check results
        # user:456 (non-premium), user:789 (inactive), user:101 (non-premium) should be invalidated
        # user:123 (premium and active) should remain
        expected_invalidated = ["user:456", "user:789", "user:101"]
        assert all(key in invalidated_keys for key in expected_invalidated)
        assert "user:123" not in invalidated_keys
        
        # Verify cache state
        assert cache_manager.exists("user:123")
        for key in expected_invalidated:
            assert not cache_manager.exists(key)


class TestBulkInvalidation:
    """Test bulk cache invalidation operations"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.fixture
    def bulk_invalidator(self, cache_manager):
        """Bulk invalidator fixture"""
        return BulkInvalidation(cache_manager)
    
    def test_bulk_invalidation_by_keys(self, bulk_invalidator, cache_manager):
        """Test bulk invalidation by specific keys"""
        # Set up cache entries
        keys = [f"key_{i}" for i in range(20)]
        for key in keys:
            cache_manager.set(key, f"value_{key}")
        
        # Bulk invalidate specific keys
        keys_to_invalidate = keys[:10]  # First 10 keys
        invalidated_count = bulk_invalidator.invalidate_keys(keys_to_invalidate)
        
        # Check results
        assert invalidated_count == 10
        
        # Verify cache state
        for i in range(10):
            assert not cache_manager.exists(f"key_{i}")
        for i in range(10, 20):
            assert cache_manager.exists(f"key_{i}")
    
    def test_bulk_invalidation_by_patterns(self, bulk_invalidator, cache_manager):
        """Test bulk invalidation by multiple patterns"""
        # Set up cache entries with different patterns
        patterns = ["user:", "track:", "playlist:", "album:"]
        for pattern in patterns:
            for i in range(5):
                key = f"{pattern}{i}"
                cache_manager.set(key, f"value_{key}")
        
        # Additional entries that shouldn't match
        cache_manager.set("global:config", "global_value")
        cache_manager.set("system:status", "system_value")
        
        # Bulk invalidate by patterns
        patterns_to_invalidate = ["user:*", "track:*"]
        invalidated_count = bulk_invalidator.invalidate_by_patterns(patterns_to_invalidate)
        
        # Check results
        assert invalidated_count == 10  # 5 user + 5 track entries
        
        # Verify cache state
        for i in range(5):
            assert not cache_manager.exists(f"user:{i}")
            assert not cache_manager.exists(f"track:{i}")
            assert cache_manager.exists(f"playlist:{i}")
            assert cache_manager.exists(f"album:{i}")
        
        assert cache_manager.exists("global:config")
        assert cache_manager.exists("system:status")
    
    def test_bulk_invalidation_performance(self, bulk_invalidator, cache_manager):
        """Test bulk invalidation performance"""
        # Set up large number of cache entries
        num_entries = 1000
        keys = []
        
        for i in range(num_entries):
            key = f"perf_test_key_{i}"
            keys.append(key)
            cache_manager.set(key, f"value_{i}")
        
        # Measure bulk invalidation performance
        start_time = time.time()
        invalidated_count = bulk_invalidator.invalidate_keys(keys)
        duration = time.time() - start_time
        
        # Check performance
        assert duration < 5.0, f"Bulk invalidation too slow: {duration}s"
        assert invalidated_count == num_entries
        
        # Verify all entries are invalidated
        for key in keys:
            assert not cache_manager.exists(key)
    
    def test_bulk_invalidation_batch_processing(self, bulk_invalidator, cache_manager):
        """Test bulk invalidation with batch processing"""
        # Set up cache entries
        num_entries = 100
        keys = [f"batch_key_{i}" for i in range(num_entries)]
        
        for key in keys:
            cache_manager.set(key, f"value_{key}")
        
        # Bulk invalidate with small batch size
        batch_size = 10
        invalidated_count = bulk_invalidator.invalidate_keys_batched(keys, batch_size=batch_size)
        
        # Check results
        assert invalidated_count == num_entries
        
        # Verify all entries are invalidated
        for key in keys:
            assert not cache_manager.exists(key)


class TestInvalidationManager:
    """Test invalidation manager that coordinates different strategies"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    @pytest.fixture
    def invalidation_manager(self, cache_manager):
        """Invalidation manager fixture"""
        return InvalidationManager(cache_manager)
    
    def test_invalidation_manager_registration(self, invalidation_manager, cache_manager):
        """Test registration of invalidation strategies"""
        # Register different strategies
        tag_strategy = TagBasedInvalidation(cache_manager)
        pattern_strategy = PatternBasedInvalidation(cache_manager)
        
        invalidation_manager.register_strategy("tag", tag_strategy)
        invalidation_manager.register_strategy("pattern", pattern_strategy)
        
        # Check strategies are registered
        assert "tag" in invalidation_manager.strategies
        assert "pattern" in invalidation_manager.strategies
        assert invalidation_manager.strategies["tag"] == tag_strategy
        assert invalidation_manager.strategies["pattern"] == pattern_strategy
    
    def test_invalidation_manager_unified_interface(self, invalidation_manager, cache_manager):
        """Test unified invalidation interface"""
        # Set up cache entries
        cache_manager.set("user:123:profile", {"name": "John"}, tags=["user", "profile"])
        cache_manager.set("user:123:settings", {"theme": "dark"}, tags=["user", "settings"])
        cache_manager.set("track:456:metadata", {"title": "Song"})
        
        # Register strategies
        tag_strategy = TagBasedInvalidation(cache_manager)
        pattern_strategy = PatternBasedInvalidation(cache_manager)
        
        invalidation_manager.register_strategy("tag", tag_strategy)
        invalidation_manager.register_strategy("pattern", pattern_strategy)
        
        # Invalidate using unified interface
        invalidated_keys = invalidation_manager.invalidate("tag", tag="user")
        
        # Check results
        assert "user:123:profile" in invalidated_keys
        assert "user:123:settings" in invalidated_keys
        assert "track:456:metadata" not in invalidated_keys
    
    def test_invalidation_manager_chained_operations(self, invalidation_manager, cache_manager):
        """Test chained invalidation operations"""
        # Set up complex cache structure
        for user_id in [123, 456]:
            cache_manager.set(f"user:{user_id}:profile", {"name": f"User{user_id}"}, 
                             tags=[f"user:{user_id}", "profile"])
            cache_manager.set(f"user:{user_id}:temp_data", {"session": "abc"}, 
                             tags=[f"user:{user_id}", "temp"])
        
        # Register strategies
        tag_strategy = TagBasedInvalidation(cache_manager)
        pattern_strategy = PatternBasedInvalidation(cache_manager)
        
        invalidation_manager.register_strategy("tag", tag_strategy)
        invalidation_manager.register_strategy("pattern", pattern_strategy)
        
        # Chain invalidation operations
        operations = [
            {"strategy": "tag", "tag": "temp"},  # Remove temp data
            {"strategy": "pattern", "pattern": "user:123:*"}  # Remove all user 123 data
        ]
        
        total_invalidated = invalidation_manager.invalidate_chain(operations)
        
        # Check results
        assert not cache_manager.exists("user:123:profile")
        assert not cache_manager.exists("user:123:temp_data")
        assert not cache_manager.exists("user:456:temp_data")
        assert cache_manager.exists("user:456:profile")  # Should remain
        
        assert len(total_invalidated) >= 3


# Integration test for all invalidation strategies
class TestInvalidationIntegration:
    """Integration tests for cache invalidation system"""
    
    @pytest.fixture
    def cache_backend(self):
        """Cache backend fixture"""
        return MemoryCacheBackend(max_size=1000, default_ttl=3600)
    
    @pytest.fixture
    def cache_manager(self, cache_backend):
        """Cache manager fixture"""
        return CacheManager(default_backend=cache_backend)
    
    def test_real_world_invalidation_scenario(self, cache_manager):
        """Test real-world cache invalidation scenario"""
        # Simulate Spotify-like caching scenario
        
        # User data
        cache_manager.set("user:123:profile", {"name": "John", "country": "US"}, 
                         tags=["user:123", "profile"])
        cache_manager.set("user:123:playlists", ["p1", "p2", "p3"], 
                         tags=["user:123", "playlists"])
        cache_manager.set("user:123:recommendations", ["t1", "t2", "t3"], 
                         tags=["user:123", "recommendations"],
                         dependencies=["user:123:profile", "user:123:playlists"])
        
        # Track data
        cache_manager.set("track:t1:metadata", {"title": "Song 1", "artist": "Artist A"}, 
                         tags=["track:t1", "metadata"])
        cache_manager.set("track:t1:analysis", {"tempo": 120, "energy": 0.8}, 
                         tags=["track:t1", "analysis"],
                         dependencies=["track:t1:metadata"])
        
        # Global data
        cache_manager.set("global:trending_tracks", ["t1", "t4", "t5"], 
                         tags=["global", "trending"])
        
        # Initialize invalidation strategies
        tag_invalidator = TagBasedInvalidation(cache_manager)
        dependency_invalidator = DependencyBasedInvalidation(cache_manager)
        pattern_invalidator = PatternBasedInvalidation(cache_manager)
        
        # Scenario 1: User updates profile (should invalidate recommendations)
        dependency_invalidator.invalidate_dependencies("user:123:profile")
        
        assert cache_manager.exists("user:123:profile")
        assert cache_manager.exists("user:123:playlists")
        assert not cache_manager.exists("user:123:recommendations")  # Invalidated
        
        # Scenario 2: User deletes account (should invalidate all user data)
        tag_invalidator.invalidate_by_tag("user:123")
        
        assert not cache_manager.exists("user:123:profile")
        assert not cache_manager.exists("user:123:playlists")
        
        # Track and global data should remain
        assert cache_manager.exists("track:t1:metadata")
        assert cache_manager.exists("track:t1:analysis")
        assert cache_manager.exists("global:trending_tracks")
        
        # Scenario 3: Clear all temporary/session data
        pattern_invalidator.invalidate_by_pattern("session:*")  # No session data in this test
        
        # All remaining data should still exist
        assert cache_manager.exists("track:t1:metadata")
        assert cache_manager.exists("global:trending_tracks")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
