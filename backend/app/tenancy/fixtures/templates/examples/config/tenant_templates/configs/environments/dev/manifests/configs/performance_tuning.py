"""
Performance Tuning and Optimization Configuration
===============================================

Configuration avancée pour l'optimisation des performances de l'application
Spotify AI Agent. Inclut les paramètres de cache, base de données, réseau,
et optimisations spécifiques aux charges de travail ML/AI.

Author: Fahed Mlaiel
Team: Spotify AI Agent Development Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import timedelta

class PerformanceProfile(Enum):
    """Profils de performance."""
    LOW_LATENCY = "low_latency"
    HIGH_THROUGHPUT = "high_throughput"
    BALANCED = "balanced"
    MEMORY_OPTIMIZED = "memory_optimized"
    CPU_OPTIMIZED = "cpu_optimized"
    COST_OPTIMIZED = "cost_optimized"

class CacheStrategy(Enum):
    """Stratégies de cache."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Stratégie adaptative

class LoadBalancingStrategy(Enum):
    """Stratégies de répartition de charge."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"

@dataclass
class CacheConfiguration:
    """Configuration du cache."""
    enabled: bool = True
    strategy: CacheStrategy = CacheStrategy.LRU
    default_ttl: int = 3600  # secondes
    max_size: int = 10000  # nombre d'entrées
    max_memory: str = "512MB"
    eviction_policy: str = "allkeys-lru"
    compression_enabled: bool = True
    compression_threshold: int = 1024  # bytes
    persistence_enabled: bool = False
    persistence_interval: int = 300  # secondes
    cluster_mode: bool = False
    replication_factor: int = 1

@dataclass
class DatabaseOptimization:
    """Configuration d'optimisation de base de données."""
    connection_pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    query_cache_enabled: bool = True
    query_cache_size: int = 1000
    slow_query_threshold: float = 1.0  # secondes
    index_optimization: bool = True
    vacuum_schedule: str = "daily"
    analyze_schedule: str = "hourly"
    read_replica_enabled: bool = True
    read_write_split: bool = True
    transaction_isolation: str = "READ_COMMITTED"

@dataclass
class WebServerOptimization:
    """Configuration d'optimisation du serveur web."""
    worker_count: int = 4
    worker_class: str = "uvicorn.workers.UvicornWorker"
    worker_timeout: int = 30
    keepalive: int = 2
    max_requests: int = 1000
    max_requests_jitter: int = 50
    preload_app: bool = True
    thread_pool_size: int = 10
    max_header_size: int = 8192
    h11_max_incomplete_event_size: int = 16777216
    enable_http2: bool = True
    compression_enabled: bool = True
    compression_level: int = 6

@dataclass
class MLOptimization:
    """Configuration d'optimisation ML."""
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    model_cache_size: int = 5
    inference_timeout: int = 30
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = True
    model_quantization: bool = False
    tensorrt_optimization: bool = False
    onnx_optimization: bool = False
    batch_inference: bool = True
    async_inference: bool = True

@dataclass
class NetworkOptimization:
    """Configuration d'optimisation réseau."""
    tcp_keepalive: bool = True
    tcp_nodelay: bool = True
    socket_timeout: int = 30
    connect_timeout: int = 10
    max_connections_per_host: int = 100
    max_total_connections: int = 1000
    dns_cache_ttl: int = 300
    http_pool_connections: int = 50
    http_pool_maxsize: int = 50
    retry_attempts: int = 3
    retry_backoff_factor: float = 0.5
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5

@dataclass
class ResourceLimits:
    """Limites de ressources."""
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    cpu_request: str = "1"
    memory_request: str = "2Gi"
    storage_limit: str = "10Gi"
    max_file_descriptors: int = 65536
    max_processes: int = 32768
    max_threads: int = 1024
    gc_threshold: float = 0.8
    swap_limit: str = "1Gi"

class PerformanceTuningManager:
    """Gestionnaire de l'optimisation des performances."""
    
    def __init__(self, 
                 profile: PerformanceProfile = PerformanceProfile.BALANCED,
                 environment: str = "development"):
        self.profile = profile
        self.environment = environment
        self.configurations = self._initialize_configurations()
    
    def _initialize_configurations(self) -> Dict[str, Any]:
        """Initialise les configurations selon le profil."""
        if self.profile == PerformanceProfile.LOW_LATENCY:
            return self._get_low_latency_config()
        elif self.profile == PerformanceProfile.HIGH_THROUGHPUT:
            return self._get_high_throughput_config()
        elif self.profile == PerformanceProfile.MEMORY_OPTIMIZED:
            return self._get_memory_optimized_config()
        elif self.profile == PerformanceProfile.CPU_OPTIMIZED:
            return self._get_cpu_optimized_config()
        elif self.profile == PerformanceProfile.COST_OPTIMIZED:
            return self._get_cost_optimized_config()
        else:
            return self._get_balanced_config()
    
    def _get_low_latency_config(self) -> Dict[str, Any]:
        """Configuration optimisée pour la faible latence."""
        return {
            "cache": CacheConfiguration(
                strategy=CacheStrategy.LRU,
                default_ttl=300,  # TTL plus court
                max_size=50000,  # Cache plus grand
                compression_enabled=False,  # Pas de compression pour éviter la latence
                persistence_enabled=False
            ),
            "database": DatabaseOptimization(
                connection_pool_size=50,  # Pool plus grand
                max_overflow=100,
                pool_timeout=5,  # Timeout plus court
                query_cache_enabled=True,
                query_cache_size=5000,
                read_replica_enabled=True,
                read_write_split=True
            ),
            "webserver": WebServerOptimization(
                worker_count=8,  # Plus de workers
                worker_timeout=15,  # Timeout plus court
                keepalive=5,
                max_requests=500,  # Moins de requêtes par worker
                thread_pool_size=20,
                enable_http2=True,
                compression_enabled=False  # Pas de compression
            ),
            "ml": MLOptimization(
                batch_size=16,  # Batch plus petit
                num_workers=8,
                prefetch_factor=4,
                pin_memory=True,
                model_cache_size=10,
                inference_timeout=10,  # Timeout plus court
                mixed_precision=True,
                batch_inference=False,  # Inference individuelle
                async_inference=True
            ),
            "network": NetworkOptimization(
                tcp_nodelay=True,
                socket_timeout=5,
                connect_timeout=3,
                max_connections_per_host=200,
                dns_cache_ttl=600,
                circuit_breaker_enabled=True,
                circuit_breaker_threshold=3
            ),
            "resources": ResourceLimits(
                cpu_limit="4",
                memory_limit="8Gi",
                cpu_request="2",
                memory_request="4Gi"
            )
        }
    
    def _get_high_throughput_config(self) -> Dict[str, Any]:
        """Configuration optimisée pour le haut débit."""
        return {
            "cache": CacheConfiguration(
                strategy=CacheStrategy.LFU,
                default_ttl=7200,  # TTL plus long
                max_size=100000,  # Cache très grand
                compression_enabled=True,
                compression_threshold=512,
                persistence_enabled=True,
                cluster_mode=True,
                replication_factor=2
            ),
            "database": DatabaseOptimization(
                connection_pool_size=100,  # Pool très grand
                max_overflow=200,
                pool_timeout=60,
                query_cache_enabled=True,
                query_cache_size=10000,
                read_replica_enabled=True,
                read_write_split=True,
                vacuum_schedule="weekly"  # Moins fréquent
            ),
            "webserver": WebServerOptimization(
                worker_count=16,  # Beaucoup de workers
                worker_timeout=60,
                keepalive=10,
                max_requests=2000,  # Plus de requêtes par worker
                preload_app=True,
                thread_pool_size=50,
                compression_enabled=True,
                compression_level=3  # Compression plus rapide
            ),
            "ml": MLOptimization(
                batch_size=128,  # Batch très grand
                num_workers=16,
                prefetch_factor=8,
                model_cache_size=20,
                inference_timeout=60,
                mixed_precision=True,
                batch_inference=True,  # Inference par batch
                async_inference=True
            ),
            "network": NetworkOptimization(
                max_connections_per_host=500,
                max_total_connections=5000,
                http_pool_connections=200,
                http_pool_maxsize=200,
                circuit_breaker_threshold=10
            ),
            "resources": ResourceLimits(
                cpu_limit="8",
                memory_limit="16Gi",
                cpu_request="4",
                memory_request="8Gi",
                max_file_descriptors=131072
            )
        }
    
    def _get_memory_optimized_config(self) -> Dict[str, Any]:
        """Configuration optimisée pour l'utilisation mémoire."""
        return {
            "cache": CacheConfiguration(
                strategy=CacheStrategy.LRU,
                default_ttl=1800,
                max_size=5000,  # Cache plus petit
                max_memory="256MB",  # Limite mémoire stricte
                compression_enabled=True,
                compression_threshold=256,
                eviction_policy="allkeys-lru"
            ),
            "database": DatabaseOptimization(
                connection_pool_size=10,  # Pool plus petit
                max_overflow=15,
                query_cache_size=500,
                vacuum_schedule="daily"
            ),
            "webserver": WebServerOptimization(
                worker_count=2,  # Moins de workers
                thread_pool_size=5,
                max_requests=100,
                compression_enabled=True,
                compression_level=9  # Compression maximale
            ),
            "ml": MLOptimization(
                batch_size=8,  # Batch très petit
                num_workers=2,
                prefetch_factor=1,
                pin_memory=False,  # Pas de pin memory
                model_cache_size=2,
                gpu_memory_fraction=0.6,
                model_quantization=True  # Quantification pour réduire la mémoire
            ),
            "network": NetworkOptimization(
                max_connections_per_host=20,
                max_total_connections=100,
                http_pool_connections=10,
                http_pool_maxsize=10
            ),
            "resources": ResourceLimits(
                cpu_limit="1",
                memory_limit="2Gi",
                cpu_request="0.5",
                memory_request="1Gi",
                gc_threshold=0.6  # GC plus agressif
            )
        }
    
    def _get_cpu_optimized_config(self) -> Dict[str, Any]:
        """Configuration optimisée pour l'utilisation CPU."""
        return {
            "cache": CacheConfiguration(
                strategy=CacheStrategy.ADAPTIVE,
                compression_enabled=False,  # Évite la charge CPU
                persistence_enabled=False
            ),
            "database": DatabaseOptimization(
                connection_pool_size=30,
                pool_pre_ping=False,  # Évite les vérifications CPU
                index_optimization=True
            ),
            "webserver": WebServerOptimization(
                worker_count=1,  # Un seul worker pour éviter la contention
                worker_class="uvicorn.workers.UvicornH11Worker",
                thread_pool_size=1,
                compression_enabled=False
            ),
            "ml": MLOptimization(
                batch_size=64,
                num_workers=1,  # Un seul worker ML
                mixed_precision=False,  # Évite la charge CPU
                tensorrt_optimization=True,  # Optimisation hardware
                onnx_optimization=True
            ),
            "resources": ResourceLimits(
                cpu_limit="0.5",  # Limite CPU stricte
                memory_limit="4Gi",
                cpu_request="0.25",
                max_threads=256  # Limite les threads
            )
        }
    
    def _get_cost_optimized_config(self) -> Dict[str, Any]:
        """Configuration optimisée pour les coûts."""
        return {
            "cache": CacheConfiguration(
                max_size=1000,  # Cache minimal
                max_memory="128MB",
                compression_enabled=True,
                persistence_enabled=False
            ),
            "database": DatabaseOptimization(
                connection_pool_size=5,  # Pool minimal
                max_overflow=5,
                read_replica_enabled=False,  # Pas de réplique
                vacuum_schedule="weekly"
            ),
            "webserver": WebServerOptimization(
                worker_count=1,  # Worker minimal
                max_requests=50,
                compression_enabled=True
            ),
            "ml": MLOptimization(
                batch_size=4,  # Batch minimal
                num_workers=1,
                model_cache_size=1,
                model_quantization=True,
                async_inference=False  # Pas d'async pour simplifier
            ),
            "resources": ResourceLimits(
                cpu_limit="0.25",
                memory_limit="1Gi",
                cpu_request="0.1",
                memory_request="512Mi",
                storage_limit="5Gi"
            )
        }
    
    def _get_balanced_config(self) -> Dict[str, Any]:
        """Configuration équilibrée par défaut."""
        return {
            "cache": CacheConfiguration(),
            "database": DatabaseOptimization(),
            "webserver": WebServerOptimization(),
            "ml": MLOptimization(),
            "network": NetworkOptimization(),
            "resources": ResourceLimits()
        }
    
    def get_configuration(self, component: str) -> Optional[Any]:
        """Récupère la configuration d'un composant."""
        return self.configurations.get(component)
    
    def export_to_env_vars(self) -> Dict[str, str]:
        """Exporte les configurations en variables d'environnement."""
        config = {}
        
        # Configuration Cache
        cache_config = self.get_configuration("cache")
        if cache_config:
            config.update({
                "CACHE_ENABLED": str(cache_config.enabled).lower(),
                "CACHE_STRATEGY": cache_config.strategy.value,
                "CACHE_DEFAULT_TTL": str(cache_config.default_ttl),
                "CACHE_MAX_SIZE": str(cache_config.max_size),
                "CACHE_MAX_MEMORY": cache_config.max_memory,
                "CACHE_COMPRESSION_ENABLED": str(cache_config.compression_enabled).lower(),
                "CACHE_COMPRESSION_THRESHOLD": str(cache_config.compression_threshold),
                "CACHE_PERSISTENCE_ENABLED": str(cache_config.persistence_enabled).lower(),
                "CACHE_CLUSTER_MODE": str(cache_config.cluster_mode).lower()
            })
        
        # Configuration Base de données
        db_config = self.get_configuration("database")
        if db_config:
            config.update({
                "DB_POOL_SIZE": str(db_config.connection_pool_size),
                "DB_MAX_OVERFLOW": str(db_config.max_overflow),
                "DB_POOL_TIMEOUT": str(db_config.pool_timeout),
                "DB_POOL_RECYCLE": str(db_config.pool_recycle),
                "DB_POOL_PRE_PING": str(db_config.pool_pre_ping).lower(),
                "DB_QUERY_CACHE_ENABLED": str(db_config.query_cache_enabled).lower(),
                "DB_QUERY_CACHE_SIZE": str(db_config.query_cache_size),
                "DB_SLOW_QUERY_THRESHOLD": str(db_config.slow_query_threshold),
                "DB_READ_REPLICA_ENABLED": str(db_config.read_replica_enabled).lower(),
                "DB_READ_WRITE_SPLIT": str(db_config.read_write_split).lower()
            })
        
        # Configuration Serveur Web
        web_config = self.get_configuration("webserver")
        if web_config:
            config.update({
                "WEB_WORKER_COUNT": str(web_config.worker_count),
                "WEB_WORKER_CLASS": web_config.worker_class,
                "WEB_WORKER_TIMEOUT": str(web_config.worker_timeout),
                "WEB_KEEPALIVE": str(web_config.keepalive),
                "WEB_MAX_REQUESTS": str(web_config.max_requests),
                "WEB_MAX_REQUESTS_JITTER": str(web_config.max_requests_jitter),
                "WEB_PRELOAD_APP": str(web_config.preload_app).lower(),
                "WEB_THREAD_POOL_SIZE": str(web_config.thread_pool_size),
                "WEB_ENABLE_HTTP2": str(web_config.enable_http2).lower(),
                "WEB_COMPRESSION_ENABLED": str(web_config.compression_enabled).lower(),
                "WEB_COMPRESSION_LEVEL": str(web_config.compression_level)
            })
        
        # Configuration ML
        ml_config = self.get_configuration("ml")
        if ml_config:
            config.update({
                "ML_BATCH_SIZE": str(ml_config.batch_size),
                "ML_NUM_WORKERS": str(ml_config.num_workers),
                "ML_PREFETCH_FACTOR": str(ml_config.prefetch_factor),
                "ML_PIN_MEMORY": str(ml_config.pin_memory).lower(),
                "ML_MODEL_CACHE_SIZE": str(ml_config.model_cache_size),
                "ML_INFERENCE_TIMEOUT": str(ml_config.inference_timeout),
                "ML_GPU_MEMORY_FRACTION": str(ml_config.gpu_memory_fraction),
                "ML_MIXED_PRECISION": str(ml_config.mixed_precision).lower(),
                "ML_MODEL_QUANTIZATION": str(ml_config.model_quantization).lower(),
                "ML_BATCH_INFERENCE": str(ml_config.batch_inference).lower(),
                "ML_ASYNC_INFERENCE": str(ml_config.async_inference).lower()
            })
        
        # Configuration Réseau
        network_config = self.get_configuration("network")
        if network_config:
            config.update({
                "NETWORK_TCP_KEEPALIVE": str(network_config.tcp_keepalive).lower(),
                "NETWORK_TCP_NODELAY": str(network_config.tcp_nodelay).lower(),
                "NETWORK_SOCKET_TIMEOUT": str(network_config.socket_timeout),
                "NETWORK_CONNECT_TIMEOUT": str(network_config.connect_timeout),
                "NETWORK_MAX_CONNECTIONS_PER_HOST": str(network_config.max_connections_per_host),
                "NETWORK_MAX_TOTAL_CONNECTIONS": str(network_config.max_total_connections),
                "NETWORK_DNS_CACHE_TTL": str(network_config.dns_cache_ttl),
                "NETWORK_RETRY_ATTEMPTS": str(network_config.retry_attempts),
                "NETWORK_CIRCUIT_BREAKER_ENABLED": str(network_config.circuit_breaker_enabled).lower()
            })
        
        # Limites de ressources
        resource_config = self.get_configuration("resources")
        if resource_config:
            config.update({
                "RESOURCE_CPU_LIMIT": resource_config.cpu_limit,
                "RESOURCE_MEMORY_LIMIT": resource_config.memory_limit,
                "RESOURCE_CPU_REQUEST": resource_config.cpu_request,
                "RESOURCE_MEMORY_REQUEST": resource_config.memory_request,
                "RESOURCE_STORAGE_LIMIT": resource_config.storage_limit,
                "RESOURCE_MAX_FILE_DESCRIPTORS": str(resource_config.max_file_descriptors),
                "RESOURCE_MAX_PROCESSES": str(resource_config.max_processes),
                "RESOURCE_GC_THRESHOLD": str(resource_config.gc_threshold)
            })
        
        return config
    
    def get_kubernetes_resources(self) -> Dict[str, Any]:
        """Génère les spécifications de ressources Kubernetes."""
        resource_config = self.get_configuration("resources")
        if not resource_config:
            return {}
        
        return {
            "requests": {
                "cpu": resource_config.cpu_request,
                "memory": resource_config.memory_request
            },
            "limits": {
                "cpu": resource_config.cpu_limit,
                "memory": resource_config.memory_limit,
                "ephemeral-storage": resource_config.storage_limit
            }
        }
    
    def get_performance_recommendations(self) -> List[str]:
        """Retourne des recommandations de performance."""
        recommendations = []
        
        if self.environment == "production":
            recommendations.extend([
                "Activez la mise en cache Redis pour les sessions",
                "Configurez des répliques de lecture pour la base de données",
                "Utilisez un CDN pour les ressources statiques",
                "Activez la compression gzip pour les réponses HTTP",
                "Configurez des health checks appropriés",
                "Surveillez les métriques de performance en temps réel"
            ])
        
        if self.profile == PerformanceProfile.LOW_LATENCY:
            recommendations.extend([
                "Désactivez la compression pour réduire la latence",
                "Utilisez des connexions persistantes",
                "Configurez des pools de connexions optimaux",
                "Activez le cache local pour les données fréquemment accédées"
            ])
        
        if self.profile == PerformanceProfile.HIGH_THROUGHPUT:
            recommendations.extend([
                "Augmentez le nombre de workers",
                "Configurez un load balancer avec round-robin",
                "Utilisez le batching pour les opérations ML",
                "Activez la parallélisation des tâches"
            ])
        
        return recommendations

# Exportation des classes
__all__ = [
    'PerformanceProfile',
    'CacheStrategy',
    'LoadBalancingStrategy',
    'CacheConfiguration',
    'DatabaseOptimization',
    'WebServerOptimization',
    'MLOptimization',
    'NetworkOptimization',
    'ResourceLimits',
    'PerformanceTuningManager'
]
