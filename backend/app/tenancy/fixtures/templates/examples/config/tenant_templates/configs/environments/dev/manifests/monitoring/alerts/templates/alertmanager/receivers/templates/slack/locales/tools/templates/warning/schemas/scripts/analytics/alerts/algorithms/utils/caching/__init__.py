"""
Module de Cache Ultra-Avancé pour Spotify AI Agent
=================================================

Module de cache industrialisé avec système multi-niveaux, monitoring intelligent,
analytics en temps réel et intégration complète avec Alertmanager.

Architecture:
- Cache L1: En mémoire (Redis/Memcached)
- Cache L2: Disque SSD avec compression
- Cache L3: Stockage distribué (optionnel)

Fonctionnalités:
- Cache intelligent avec ML pour prédiction des accès
- Monitoring temps réel avec métriques Prometheus
- Alertes automatiques via Alertmanager
- Analytics détaillées pour optimisation
- Support multi-tenant avec isolation des données
- Invalidation intelligente et TTL adaptatif
- Compression et sérialisation optimisées
- Fallback automatique entre niveaux
- Circuit breaker intégré
- Audit complet et logging structuré

Author: Développé sous l'expertise de Fahed Mlaiel
Team: Lead Dev + Architecte IA, Backend Senior, DBA & Data Engineer,
      Spécialiste Sécurité, Architecte Microservices
"""

from .core import (
    CacheManager,
    MultiLevelCache,
    TenantAwareCacheManager,
    CacheMetrics,
    CacheHealthChecker
)

from .strategies import (
    LRUStrategy,
    LFUStrategy,
    TimeBasedStrategy,
    MLPredictiveStrategy,
    AdaptiveStrategy
)

from .backends import (
    RedisBackend,
    MemcachedBackend,
    DiskBackend,
    HybridBackend,
    DistributedBackend
)

from .monitoring import (
    CacheMonitor,
    PrometheusMetricsExporter,
    AlertManager,
    PerformanceAnalyzer,
    HealthMetrics
)

from .analytics import (
    CacheAnalytics,
    UsageAnalyzer,
    TrendAnalyzer,
    PredictiveAnalyzer,
    OptimizationRecommender
)

from .serializers import (
    JSONSerializer,
    PickleSerializer,
    MsgPackSerializer,
    CompressionSerializer,
    EncryptedSerializer
)

from .exceptions import (
    CacheException,
    CacheBackendError,
    CacheMissError,
    CacheTimeoutError,
    CacheSecurityError,
    CacheQuotaExceededError
)

from .utils import (
    CacheKeyGenerator,
    TTLCalculator,
    CompressionUtils,
    SecurityUtils,
    ValidationUtils
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerManager
)

from .tenant_isolation import (
    TenantIsolator,
    TenantQuotaManager,
    TenantSecurityManager
)

from .warming import (
    CacheWarmer,
    PreloadManager,
    WarmupScheduler
)

from .invalidation import (
    InvalidationManager,
    TagBasedInvalidation,
    TimeBasedInvalidation,
    EventBasedInvalidation
)

__version__ = "2.0.0"
__author__ = "Spotify AI Agent Team - Fahed Mlaiel"
__email__ = "dev@spotify-ai-agent.com"

# Configuration par défaut
DEFAULT_CONFIG = {
    "cache_levels": 3,
    "default_ttl": 3600,
    "max_memory_usage": "512MB",
    "compression_enabled": True,
    "monitoring_enabled": True,
    "metrics_export_interval": 30,
    "health_check_interval": 60,
    "circuit_breaker_enabled": True,
    "security_enabled": True,
    "tenant_isolation": True,
    "analytics_enabled": True,
    "ml_predictions": True,
    "auto_optimization": True
}

# Exports principaux
__all__ = [
    # Core
    "CacheManager",
    "MultiLevelCache", 
    "TenantAwareCacheManager",
    "CacheMetrics",
    "CacheHealthChecker",
    
    # Strategies
    "LRUStrategy",
    "LFUStrategy", 
    "TimeBasedStrategy",
    "MLPredictiveStrategy",
    "AdaptiveStrategy",
    
    # Backends
    "RedisBackend",
    "MemcachedBackend",
    "DiskBackend", 
    "HybridBackend",
    "DistributedBackend",
    
    # Monitoring
    "CacheMonitor",
    "PrometheusMetricsExporter",
    "AlertManager",
    "PerformanceAnalyzer",
    "HealthMetrics",
    
    # Analytics
    "CacheAnalytics",
    "UsageAnalyzer",
    "TrendAnalyzer",
    "PredictiveAnalyzer", 
    "OptimizationRecommender",
    
    # Serializers
    "JSONSerializer",
    "PickleSerializer",
    "MsgPackSerializer",
    "CompressionSerializer",
    "EncryptedSerializer",
    
    # Exceptions
    "CacheException",
    "CacheBackendError",
    "CacheMissError", 
    "CacheTimeoutError",
    "CacheSecurityError",
    "CacheQuotaExceededError",
    
    # Utils
    "CacheKeyGenerator",
    "TTLCalculator",
    "CompressionUtils",
    "SecurityUtils",
    "ValidationUtils",
    
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerManager",
    
    # Tenant
    "TenantIsolator", 
    "TenantQuotaManager",
    "TenantSecurityManager",
    
    # Warming
    "CacheWarmer",
    "PreloadManager", 
    "WarmupScheduler",
    
    # Invalidation
    "InvalidationManager",
    "TagBasedInvalidation",
    "TimeBasedInvalidation",
    "EventBasedInvalidation",
    
    # Config
    "DEFAULT_CONFIG"
]
