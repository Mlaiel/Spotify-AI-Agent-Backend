"""
Tests pour le système de cache avancé Spotify AI Agent

Module de tests complet pour valider les fonctionnalités de mise en cache,
incluant Redis, cache distribué, stratégies d'invalidation et monitoring.

Architecture développée par :
- Lead Dev + Architecte IA : Fahed Mlaiel
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices
"""

__version__ = "1.0.0"
__author__ = "Fahed Mlaiel & Spotify AI Agent Team"

# Import all test modules for cache functionality
from .test_cache_backends import *
from .test_cache_decorators import *
from .test_cache_invalidation import *
from .test_cache_keys import *
from .test_cache_layers import *
from .test_cache_manager import *
from .test_cache_metrics import *
from .test_cache_patterns import *
from .test_cache_serializers import *
from .test_cache_strategies import *
from .test_distributed_cache import *
from .test_memory_cache import *
from .test_redis_cache import *

__all__ = [
    # Test classes
    "TestCacheBackends",
    "TestCacheDecorators", 
    "TestCacheInvalidation",
    "TestCacheKeys",
    "TestCacheLayers",
    "TestCacheManager",
    "TestCacheMetrics",
    "TestCachePatterns",
    "TestCacheSerializers",
    "TestCacheStrategies",
    "TestDistributedCache",
    "TestMemoryCache",
    "TestRedisCache",
    
    # Test fixtures
    "cache_manager_fixture",
    "redis_client_fixture", 
    "memory_cache_fixture",
    "cache_config_fixture",
    "cache_metrics_fixture",
    
    # Test utilities
    "CacheTestHelper",
    "MockCacheBackend",
    "CachePerformanceBenchmark",
]
