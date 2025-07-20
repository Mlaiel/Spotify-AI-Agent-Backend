"""
Tests for cache keys management in Spotify AI Agent

Comprehensive testing suite for cache key generation, normalization,
hashing, collision detection and namespace management.

Developed by Expert Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect
"""

import pytest
import hashlib
import uuid
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from app.utils.cache.keys import (
    CacheKeyGenerator, CacheKeyNormalizer, CacheKeyHasher,
    NamespaceManager, KeyCollisionDetector, KeyVersionManager,
    KeyPatternMatcher, CacheKeyValidator
)
from app.utils.cache.exceptions import InvalidCacheKeyError, KeyCollisionError


@dataclass
class MockCacheObject:
    """Mock cache object for testing"""
    id: int
    name: str
    type: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestCacheKeyGenerator:
    """Test cache key generation"""
    
    @pytest.fixture
    def key_generator(self):
        """Key generator fixture"""
        return CacheKeyGenerator(
            separator=":",
            max_length=250,
            enable_hashing=True,
            hash_algorithm="sha256"
        )
    
    def test_simple_key_generation(self, key_generator):
        """Test simple key generation"""
        generator = key_generator
        
        # Test with simple strings
        key = generator.generate("user", "123", "profile")
        assert key == "user:123:profile"
        
        # Test with mixed types
        key = generator.generate("track", 456, "metadata")
        assert key == "track:456:metadata"
    
    def test_key_generation_with_objects(self, key_generator):
        """Test key generation with complex objects"""
        generator = key_generator
        
        user = MockCacheObject(id=123, name="John Doe", type="user")
        
        # Test with object
        key = generator.generate_from_object("user_profile", user)
        expected_parts = ["user_profile", "123", "John Doe", "user"]
        expected_key = ":".join(str(part) for part in expected_parts)
        assert key == expected_key
    
    def test_key_generation_with_dict(self, key_generator):
        """Test key generation with dictionary"""
        generator = key_generator
        
        params = {
            "user_id": 123,
            "track_id": "track_456",
            "limit": 10,
            "offset": 0
        }
        
        key = generator.generate_from_dict("recommendations", params)
        
        # Should include all parameters in deterministic order
        assert "recommendations" in key
        assert "123" in key
        assert "track_456" in key
        assert "10" in key
        assert "0" in key
    
    def test_key_generation_with_hashing(self, key_generator):
        """Test key generation with automatic hashing"""
        generator = key_generator
        
        # Generate very long key that should be hashed
        long_parts = ["very_long_prefix"] + [f"part_{i}" for i in range(50)]
        
        key = generator.generate(*long_parts)
        
        # Should be shorter than max_length due to hashing
        assert len(key) <= generator.max_length
        
        # Should contain hash
        assert len(key.split(":")[-1]) == 64  # SHA256 hex length
    
    def test_key_generation_with_namespace(self, key_generator):
        """Test key generation with namespace"""
        generator = key_generator
        
        # Set namespace
        generator.set_namespace("spotify:cache:v1")
        
        key = generator.generate("user", "123", "profile")
        
        assert key.startswith("spotify:cache:v1:")
        assert key == "spotify:cache:v1:user:123:profile"
    
    def test_key_generation_normalization(self, key_generator):
        """Test key normalization during generation"""
        generator = key_generator
        
        # Test with special characters and spaces
        key = generator.generate("User Profile", "ID-123", "meta data")
        
        # Should normalize spaces and special characters
        assert " " not in key
        assert "-" not in key or key.count("-") == 0  # Depends on normalization rules
    
    def test_key_generation_collision_detection(self, key_generator):
        """Test collision detection in key generation"""
        generator = key_generator
        
        # Generate keys that might collide
        key1 = generator.generate("user", "123", "profile")
        key2 = generator.generate("user:123", "profile")
        
        # Should generate different keys even if input is similar
        assert key1 != key2
    
    def test_key_generation_with_versions(self, key_generator):
        """Test key generation with versioning"""
        generator = key_generator
        
        # Generate versioned keys
        key_v1 = generator.generate_versioned("user", "123", "profile", version="v1")
        key_v2 = generator.generate_versioned("user", "123", "profile", version="v2")
        
        assert key_v1 != key_v2
        assert "v1" in key_v1
        assert "v2" in key_v2
    
    def test_key_generation_with_ttl_hint(self, key_generator):
        """Test key generation with TTL hints"""
        generator = key_generator
        
        # Generate keys with TTL hints
        short_key = generator.generate_with_ttl_hint("session", "abc123", ttl_seconds=300)
        long_key = generator.generate_with_ttl_hint("user_cache", "456", ttl_seconds=3600)
        
        # Keys should include TTL information
        assert "300" in short_key or "short" in short_key.lower()
        assert "3600" in long_key or "long" in long_key.lower()


class TestCacheKeyNormalizer:
    """Test cache key normalization"""
    
    @pytest.fixture
    def normalizer(self):
        """Key normalizer fixture"""
        return CacheKeyNormalizer(
            lowercase=True,
            replace_spaces=True,
            remove_special_chars=True,
            max_segment_length=50
        )
    
    def test_basic_normalization(self, normalizer):
        """Test basic key normalization"""
        norm = normalizer
        
        # Test case conversion
        assert norm.normalize("USER_PROFILE") == "user_profile"
        
        # Test space replacement
        assert norm.normalize("user profile data") == "user_profile_data"
        
        # Test special character removal
        assert norm.normalize("user@123#profile") == "user123profile"
    
    def test_unicode_normalization(self, normalizer):
        """Test unicode character normalization"""
        norm = normalizer
        
        # Test unicode characters
        unicode_key = "utilisateur_profil_café"
        normalized = norm.normalize(unicode_key)
        
        # Should handle unicode gracefully
        assert normalized == "utilisateur_profil_café"  # Or appropriate normalization
    
    def test_segment_length_limiting(self, normalizer):
        """Test segment length limiting"""
        norm = normalizer
        
        # Long segment
        long_segment = "a" * 100
        normalized = norm.normalize(long_segment)
        
        assert len(normalized) <= norm.max_segment_length
    
    def test_custom_replacements(self, normalizer):
        """Test custom character replacements"""
        norm = normalizer
        
        # Add custom replacements
        norm.add_replacement("-", "_")
        norm.add_replacement("@", "_at_")
        
        # Test replacements
        assert norm.normalize("user-123@domain") == "user_123_at_domain"
    
    def test_normalization_consistency(self, normalizer):
        """Test normalization consistency"""
        norm = normalizer
        
        # Same input should always produce same output
        input_key = "Complex Key With $pecial Ch@rs!"
        
        result1 = norm.normalize(input_key)
        result2 = norm.normalize(input_key)
        
        assert result1 == result2
    
    def test_batch_normalization(self, normalizer):
        """Test batch normalization"""
        norm = normalizer
        
        keys = [
            "User Profile",
            "Track Metadata",
            "Playlist@Info",
            "Session#Data"
        ]
        
        normalized_keys = norm.normalize_batch(keys)
        
        assert len(normalized_keys) == len(keys)
        assert all(isinstance(key, str) for key in normalized_keys)
        
        # Check specific normalizations
        assert "user_profile" in normalized_keys
        assert "track_metadata" in normalized_keys


class TestCacheKeyHasher:
    """Test cache key hashing"""
    
    @pytest.fixture
    def hasher(self):
        """Key hasher fixture"""
        return CacheKeyHasher(
            algorithm="sha256",
            encoding="utf-8",
            hash_threshold=100,  # Hash keys longer than 100 chars
            preserve_prefix=True
        )
    
    def test_basic_hashing(self, hasher):
        """Test basic key hashing"""
        hash_gen = hasher
        
        # Short key (shouldn't be hashed)
        short_key = "user:123:profile"
        result = hash_gen.hash_if_needed(short_key)
        assert result == short_key  # Unchanged
        
        # Long key (should be hashed)
        long_key = "user:123:profile:" + "x" * 200
        result = hash_gen.hash_if_needed(long_key)
        assert len(result) < len(long_key)
        assert result != long_key
    
    def test_hash_algorithms(self):
        """Test different hash algorithms"""
        algorithms = ["md5", "sha1", "sha256", "sha512"]
        
        for algo in algorithms:
            hasher = CacheKeyHasher(algorithm=algo)
            
            key = "test_key_for_hashing_" + "x" * 200
            hashed = hasher.hash_key(key)
            
            # Check hash length based on algorithm
            expected_lengths = {
                "md5": 32,
                "sha1": 40,
                "sha256": 64,
                "sha512": 128
            }
            
            assert len(hashed) == expected_lengths[algo]
    
    def test_prefix_preservation(self, hasher):
        """Test prefix preservation during hashing"""
        hash_gen = hasher
        
        # Long key with meaningful prefix
        long_key = "user:123:recommendations:detailed_analysis:" + "x" * 200
        result = hash_gen.hash_if_needed(long_key)
        
        # Should preserve meaningful prefix
        assert result.startswith("user:123:")
        
        # But overall length should be reduced
        assert len(result) < len(long_key)
    
    def test_hash_consistency(self, hasher):
        """Test hash consistency"""
        hash_gen = hasher
        
        key = "consistency_test_key_" + "x" * 200
        
        hash1 = hash_gen.hash_key(key)
        hash2 = hash_gen.hash_key(key)
        
        assert hash1 == hash2
    
    def test_hash_collision_resistance(self, hasher):
        """Test hash collision resistance"""
        hash_gen = hasher
        
        # Generate many similar keys
        keys = [f"similar_key_{i}:" + "x" * 200 for i in range(1000)]
        hashes = [hash_gen.hash_key(key) for key in keys]
        
        # Should have no collisions
        assert len(set(hashes)) == len(hashes)
    
    def test_hash_with_salt(self):
        """Test hashing with salt"""
        hasher_with_salt = CacheKeyHasher(
            algorithm="sha256",
            salt="spotify_cache_salt_2024"
        )
        
        key = "test_key"
        
        # Same key with different salts should produce different hashes
        hasher_no_salt = CacheKeyHasher(algorithm="sha256")
        
        hash_with_salt = hasher_with_salt.hash_key(key)
        hash_no_salt = hasher_no_salt.hash_key(key)
        
        assert hash_with_salt != hash_no_salt


class TestNamespaceManager:
    """Test namespace management"""
    
    @pytest.fixture
    def namespace_manager(self):
        """Namespace manager fixture"""
        return NamespaceManager(
            default_namespace="spotify",
            separator=":",
            enable_versioning=True,
            enable_environment_isolation=True
        )
    
    def test_namespace_creation(self, namespace_manager):
        """Test namespace creation and management"""
        manager = namespace_manager
        
        # Create namespaces
        manager.create_namespace("users", "User-related data")
        manager.create_namespace("tracks", "Track metadata and analysis")
        manager.create_namespace("playlists", "Playlist data")
        
        # Check namespaces exist
        namespaces = manager.list_namespaces()
        assert "users" in namespaces
        assert "tracks" in namespaces
        assert "playlists" in namespaces
    
    def test_namespace_hierarchy(self, namespace_manager):
        """Test hierarchical namespaces"""
        manager = namespace_manager
        
        # Create hierarchical namespaces
        manager.create_namespace("spotify")
        manager.create_namespace("spotify:api", parent="spotify")
        manager.create_namespace("spotify:api:v1", parent="spotify:api")
        manager.create_namespace("spotify:cache", parent="spotify")
        
        # Test hierarchy
        children = manager.get_child_namespaces("spotify")
        assert "spotify:api" in children
        assert "spotify:cache" in children
        
        parent = manager.get_parent_namespace("spotify:api:v1")
        assert parent == "spotify:api"
    
    def test_namespace_key_generation(self, namespace_manager):
        """Test key generation with namespaces"""
        manager = namespace_manager
        
        # Set active namespace
        manager.set_active_namespace("users")
        
        # Generate namespaced key
        key = manager.generate_namespaced_key("123", "profile")
        assert key.startswith("spotify:users:")
        assert "123" in key
        assert "profile" in key
    
    def test_namespace_isolation(self, namespace_manager):
        """Test namespace isolation"""
        manager = namespace_manager
        
        # Set environment
        manager.set_environment("production")
        
        # Generate keys in different namespaces
        manager.set_active_namespace("users")
        user_key = manager.generate_namespaced_key("123", "profile")
        
        manager.set_active_namespace("tracks")
        track_key = manager.generate_namespaced_key("456", "metadata")
        
        # Keys should be isolated
        assert not user_key.startswith(track_key.split(":")[0:3])
        
        # Both should include environment
        assert "production" in user_key
        assert "production" in track_key
    
    def test_namespace_permissions(self, namespace_manager):
        """Test namespace permissions"""
        manager = namespace_manager
        
        # Create namespace with permissions
        manager.create_namespace("sensitive", "Sensitive data", 
                                restricted=True, 
                                allowed_operations=["read"])
        
        # Check permissions
        permissions = manager.get_namespace_permissions("sensitive")
        assert permissions["restricted"] is True
        assert "read" in permissions["allowed_operations"]
        assert "write" not in permissions.get("allowed_operations", [])
    
    def test_namespace_versioning(self, namespace_manager):
        """Test namespace versioning"""
        manager = namespace_manager
        
        # Create versioned namespace
        manager.create_versioned_namespace("api", version="v1")
        manager.create_versioned_namespace("api", version="v2")
        
        # Generate keys for different versions
        manager.set_active_namespace("api", version="v1")
        key_v1 = manager.generate_namespaced_key("endpoint", "data")
        
        manager.set_active_namespace("api", version="v2")
        key_v2 = manager.generate_namespaced_key("endpoint", "data")
        
        assert key_v1 != key_v2
        assert "v1" in key_v1
        assert "v2" in key_v2


class TestKeyCollisionDetector:
    """Test key collision detection"""
    
    @pytest.fixture
    def collision_detector(self):
        """Collision detector fixture"""
        return KeyCollisionDetector(
            enable_logging=True,
            max_collision_history=1000,
            collision_threshold=0.01  # 1% collision rate triggers alert
        )
    
    def test_collision_detection_basic(self, collision_detector):
        """Test basic collision detection"""
        detector = collision_detector
        
        # Register keys
        key1 = "user:123:profile"
        key2 = "user:123:profile"  # Same key
        key3 = "user:456:profile"  # Different key
        
        # Check for collisions
        collision1 = detector.check_collision(key1, "data1")
        assert collision1 is False  # First occurrence
        
        collision2 = detector.check_collision(key2, "data2")
        assert collision2 is True  # Collision detected
        
        collision3 = detector.check_collision(key3, "data3")
        assert collision3 is False  # Different key
    
    def test_collision_detection_with_hashing(self, collision_detector):
        """Test collision detection with hash-based keys"""
        detector = collision_detector
        
        # Create hash-based keys that might collide
        hasher = CacheKeyHasher(algorithm="md5")  # Higher collision probability
        
        base_keys = [f"key_{i}" for i in range(1000)]
        hashed_keys = [hasher.hash_key(key) for key in base_keys]
        
        collisions = []
        for i, hashed_key in enumerate(hashed_keys):
            if detector.check_collision(hashed_key, f"data_{i}"):
                collisions.append(hashed_key)
        
        # Track collision statistics
        collision_stats = detector.get_collision_statistics()
        assert "total_keys_checked" in collision_stats
        assert "collision_count" in collision_stats
        assert "collision_rate" in collision_stats
    
    def test_collision_prevention_strategies(self, collision_detector):
        """Test collision prevention strategies"""
        detector = collision_detector
        
        # Enable collision prevention
        detector.enable_collision_prevention(True)
        
        original_key = "user:123:data"
        
        # First occurrence
        final_key1 = detector.ensure_unique_key(original_key, "data1")
        assert final_key1 == original_key
        
        # Collision scenario
        final_key2 = detector.ensure_unique_key(original_key, "data2")
        assert final_key2 != original_key  # Should be modified
        assert original_key in final_key2  # Should contain original
    
    def test_collision_resolution(self, collision_detector):
        """Test collision resolution strategies"""
        detector = collision_detector
        
        # Set collision resolution strategy
        detector.set_collision_resolution_strategy("timestamp_suffix")
        
        key = "collision_test_key"
        
        # Generate multiple keys with same base
        resolved_keys = []
        for i in range(10):
            resolved_key = detector.resolve_collision(key, f"data_{i}")
            resolved_keys.append(resolved_key)
        
        # All resolved keys should be unique
        assert len(set(resolved_keys)) == len(resolved_keys)
        
        # All should contain original key
        assert all(key in resolved for resolved in resolved_keys)
    
    def test_collision_monitoring(self, collision_detector):
        """Test collision monitoring and alerting"""
        detector = collision_detector
        
        alerts = []
        
        def alert_handler(alert):
            alerts.append(alert)
        
        detector.set_alert_handler(alert_handler)
        
        # Generate many collisions to trigger alert
        base_key = "high_collision_key"
        for i in range(20):  # Exceed threshold
            detector.check_collision(base_key, f"data_{i}")
        
        # Should trigger alert
        assert len(alerts) >= 1
        alert = alerts[0]
        assert alert["type"] == "high_collision_rate"
        assert alert["key_pattern"] == base_key


class TestKeyVersionManager:
    """Test key version management"""
    
    @pytest.fixture
    def version_manager(self):
        """Version manager fixture"""
        return KeyVersionManager(
            versioning_strategy="semantic",
            default_version="1.0.0",
            enable_backward_compatibility=True
        )
    
    def test_version_generation(self, version_manager):
        """Test version generation for keys"""
        manager = version_manager
        
        # Generate versioned keys
        base_key = "user:123:profile"
        
        v1_key = manager.generate_versioned_key(base_key, "1.0.0")
        v2_key = manager.generate_versioned_key(base_key, "2.0.0")
        
        assert v1_key != v2_key
        assert "1.0.0" in v1_key
        assert "2.0.0" in v2_key
    
    def test_version_compatibility(self, version_manager):
        """Test version compatibility checking"""
        manager = version_manager
        
        # Check compatibility between versions
        assert manager.is_compatible("1.0.0", "1.0.1") is True  # Patch compatible
        assert manager.is_compatible("1.0.0", "1.1.0") is True  # Minor compatible
        assert manager.is_compatible("1.0.0", "2.0.0") is False  # Major incompatible
    
    def test_version_migration(self, version_manager):
        """Test version migration"""
        manager = version_manager
        
        # Register migration
        def migrate_v1_to_v2(key, data):
            # Example migration logic
            return key.replace("v1", "v2"), {"migrated": True, **data}
        
        manager.register_migration("1.0.0", "2.0.0", migrate_v1_to_v2)
        
        # Test migration
        old_key = "user:123:profile:v1"
        old_data = {"name": "John", "age": 30}
        
        new_key, new_data = manager.migrate_key(old_key, old_data, "1.0.0", "2.0.0")
        
        assert "v2" in new_key
        assert new_data["migrated"] is True
        assert new_data["name"] == "John"  # Original data preserved
    
    def test_version_cleanup(self, version_manager):
        """Test cleanup of old versions"""
        manager = version_manager
        
        # Track versions for cleanup
        keys_by_version = {
            "1.0.0": ["key1:v1", "key2:v1", "key3:v1"],
            "1.1.0": ["key1:v1.1", "key2:v1.1"],
            "2.0.0": ["key1:v2", "key2:v2", "key3:v2", "key4:v2"]
        }
        
        for version, keys in keys_by_version.items():
            for key in keys:
                manager.track_version(key, version)
        
        # Cleanup old versions
        cleanup_plan = manager.plan_version_cleanup(
            keep_latest=2,  # Keep latest 2 versions
            min_age_days=0  # No age requirement for testing
        )
        
        assert "1.0.0" in cleanup_plan["versions_to_remove"]
        assert "2.0.0" not in cleanup_plan["versions_to_remove"]
        assert len(cleanup_plan["keys_to_remove"]) >= 3


class TestKeyPatternMatcher:
    """Test key pattern matching"""
    
    @pytest.fixture
    def pattern_matcher(self):
        """Pattern matcher fixture"""
        return KeyPatternMatcher(
            enable_regex=True,
            enable_wildcards=True,
            case_sensitive=False
        )
    
    def test_wildcard_matching(self, pattern_matcher):
        """Test wildcard pattern matching"""
        matcher = pattern_matcher
        
        # Define test keys
        keys = [
            "user:123:profile",
            "user:456:profile", 
            "user:123:settings",
            "track:789:metadata",
            "playlist:abc:tracks"
        ]
        
        # Test wildcard patterns
        user_keys = matcher.match_pattern("user:*", keys)
        assert len(user_keys) == 3
        assert all(key.startswith("user:") for key in user_keys)
        
        profile_keys = matcher.match_pattern("*:profile", keys)
        assert len(profile_keys) == 2
        assert all(key.endswith(":profile") for key in profile_keys)
        
        user_123_keys = matcher.match_pattern("user:123:*", keys)
        assert len(user_123_keys) == 2
        assert all("user:123:" in key for key in user_123_keys)
    
    def test_regex_matching(self, pattern_matcher):
        """Test regex pattern matching"""
        matcher = pattern_matcher
        
        keys = [
            "user_123_profile",
            "user_456_settings",
            "track_789_metadata",
            "playlist_abc_tracks",
            "session_xyz_data"
        ]
        
        # Test regex patterns
        import re
        
        # Match user keys with numeric IDs
        user_pattern = re.compile(r"user_\d+_.*")
        user_keys = matcher.match_regex(user_pattern, keys)
        assert len(user_keys) == 2
        
        # Match keys with 3-character IDs
        three_char_pattern = re.compile(r".*_[a-z]{3}_.*")
        three_char_keys = matcher.match_regex(three_char_pattern, keys)
        assert len(three_char_keys) == 2  # playlist_abc and session_xyz
    
    def test_pattern_compilation_caching(self, pattern_matcher):
        """Test pattern compilation caching"""
        matcher = pattern_matcher
        
        pattern = "user:*:profile"
        keys = [f"user:{i}:profile" for i in range(100)]
        
        # First match (should compile pattern)
        import time
        start_time = time.time()
        result1 = matcher.match_pattern(pattern, keys)
        first_duration = time.time() - start_time
        
        # Second match (should use cached pattern)
        start_time = time.time()
        result2 = matcher.match_pattern(pattern, keys)
        second_duration = time.time() - start_time
        
        # Results should be identical
        assert result1 == result2
        
        # Second match should be faster (cached)
        assert second_duration <= first_duration
    
    def test_complex_pattern_matching(self, pattern_matcher):
        """Test complex pattern matching"""
        matcher = pattern_matcher
        
        keys = [
            "spotify:user:123:profile:v1",
            "spotify:user:456:profile:v2", 
            "spotify:track:789:analysis:v1",
            "lastfm:user:123:scrobbles:v1",
            "apple:track:456:metadata:v2"
        ]
        
        # Complex pattern: Spotify user profiles any version
        spotify_user_pattern = "spotify:user:*:profile:*"
        matches = matcher.match_pattern(spotify_user_pattern, keys)
        assert len(matches) == 2
        assert all("spotify:user:" in key and ":profile:" in key for key in matches)
        
        # Complex pattern: Any service, any track, version 2
        v2_track_pattern = "*:track:*:*:v2"
        v2_matches = matcher.match_pattern(v2_track_pattern, keys)
        assert len(v2_matches) == 1
        assert "apple:track:456:metadata:v2" in v2_matches


class TestCacheKeyValidator:
    """Test cache key validation"""
    
    @pytest.fixture
    def validator(self):
        """Key validator fixture"""
        return CacheKeyValidator(
            max_length=250,
            min_length=3,
            allowed_chars="alphanumeric_underscore_colon",
            forbidden_patterns=["__", "::"],
            enable_namespace_validation=True
        )
    
    def test_basic_validation(self, validator):
        """Test basic key validation"""
        valid = validator
        
        # Valid keys
        assert valid.validate("user:123:profile") is True
        assert valid.validate("track_metadata_456") is True
        assert valid.validate("session:abc123:data") is True
        
        # Invalid keys
        assert valid.validate("") is False  # Too short
        assert valid.validate("a" * 300) is False  # Too long
        assert valid.validate("user@123#profile") is False  # Invalid chars
        assert valid.validate("user::profile") is False  # Forbidden pattern
    
    def test_character_validation(self, validator):
        """Test character validation"""
        valid = validator
        
        # Test different character sets
        valid.set_allowed_chars("alphanumeric")
        assert valid.validate("user123profile") is True
        assert valid.validate("user:123:profile") is False  # Colon not allowed
        
        valid.set_allowed_chars("alphanumeric_underscore_colon")
        assert valid.validate("user:123:profile") is True
        assert valid.validate("user-123-profile") is False  # Dash not allowed
    
    def test_length_validation(self, validator):
        """Test length validation"""
        valid = validator
        
        # Test length boundaries
        assert valid.validate("ab") is False  # Too short
        assert valid.validate("abc") is True  # Minimum length
        assert valid.validate("a" * 250) is True  # Maximum length
        assert valid.validate("a" * 251) is False  # Too long
    
    def test_pattern_validation(self, validator):
        """Test forbidden pattern validation"""
        valid = validator
        
        # Add custom forbidden patterns
        valid.add_forbidden_pattern("admin")
        valid.add_forbidden_pattern("root")
        valid.add_forbidden_pattern("test.*debug")  # Regex pattern
        
        # Test forbidden patterns
        assert valid.validate("user:admin:profile") is False
        assert valid.validate("root:config") is False
        assert valid.validate("test_user_debug") is False
        assert valid.validate("user:normal:profile") is True
    
    def test_namespace_validation(self, validator):
        """Test namespace validation"""
        valid = validator
        
        # Register valid namespaces
        valid.register_namespace("user")
        valid.register_namespace("track")
        valid.register_namespace("playlist")
        
        # Test namespace validation
        assert valid.validate("user:123:profile") is True
        assert valid.validate("track:456:metadata") is True
        assert valid.validate("invalid:789:data") is False  # Unknown namespace
    
    def test_custom_validators(self, validator):
        """Test custom validation functions"""
        valid = validator
        
        # Add custom validator
        def no_consecutive_numbers(key: str) -> bool:
            import re
            return not re.search(r'\d{3,}', key)  # No 3+ consecutive digits
        
        valid.add_custom_validator("no_consecutive_numbers", no_consecutive_numbers)
        
        # Test custom validation
        assert valid.validate("user:12:profile") is True
        assert valid.validate("user:123:profile") is False  # 3 consecutive digits
        assert valid.validate("user:1234:profile") is False  # 4 consecutive digits
    
    def test_validation_error_reporting(self, validator):
        """Test validation error reporting"""
        valid = validator
        
        # Test with detailed error reporting
        valid.enable_detailed_errors(True)
        
        # Invalid key
        key = "user@123#profile::data"
        
        is_valid, errors = valid.validate_with_errors(key)
        
        assert is_valid is False
        assert len(errors) > 0
        
        error_types = [error["type"] for error in errors]
        assert "invalid_characters" in error_types
        assert "forbidden_pattern" in error_types


class TestCacheKeyIntegration:
    """Integration tests for cache key management"""
    
    def test_complete_key_lifecycle(self):
        """Test complete key lifecycle"""
        # Initialize components
        generator = CacheKeyGenerator()
        normalizer = CacheKeyNormalizer()
        hasher = CacheKeyHasher()
        validator = CacheKeyValidator()
        namespace_manager = NamespaceManager()
        
        # Setup namespace
        namespace_manager.create_namespace("users")
        namespace_manager.set_active_namespace("users")
        
        # Generate raw key
        raw_key = generator.generate("User Profile", "ID-123", "Meta Data")
        
        # Normalize key
        normalized_key = normalizer.normalize(raw_key)
        
        # Add namespace
        namespaced_key = namespace_manager.generate_namespaced_key(normalized_key)
        
        # Hash if needed
        final_key = hasher.hash_if_needed(namespaced_key)
        
        # Validate final key
        is_valid = validator.validate(final_key)
        
        assert is_valid is True
        assert len(final_key) <= 250
        assert "users" in final_key
    
    def test_key_collision_handling_workflow(self):
        """Test complete collision handling workflow"""
        # Setup components
        generator = CacheKeyGenerator()
        collision_detector = KeyCollisionDetector()
        
        # Enable collision prevention
        collision_detector.enable_collision_prevention(True)
        
        # Generate keys that might collide
        base_data = {"type": "profile", "user_id": 123}
        
        keys_generated = []
        for i in range(10):
            # Generate key
            raw_key = generator.generate_from_dict("user_profile", base_data)
            
            # Ensure uniqueness
            unique_key = collision_detector.ensure_unique_key(raw_key, f"data_{i}")
            keys_generated.append(unique_key)
        
        # All keys should be unique
        assert len(set(keys_generated)) == len(keys_generated)
    
    def test_key_versioning_workflow(self):
        """Test key versioning workflow"""
        # Setup components
        generator = CacheKeyGenerator()
        version_manager = KeyVersionManager()
        
        # Generate base key
        base_key = generator.generate("user", "123", "profile")
        
        # Create versioned keys
        v1_key = version_manager.generate_versioned_key(base_key, "1.0.0")
        v2_key = version_manager.generate_versioned_key(base_key, "2.0.0")
        
        # Test migration
        def migrate_profile(key, data):
            return key.replace("1.0.0", "2.0.0"), {**data, "migrated": True}
        
        version_manager.register_migration("1.0.0", "2.0.0", migrate_profile)
        
        # Migrate data
        old_data = {"name": "John", "age": 30}
        new_key, new_data = version_manager.migrate_key(v1_key, old_data, "1.0.0", "2.0.0")
        
        assert new_data["migrated"] is True
        assert new_data["name"] == "John"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
