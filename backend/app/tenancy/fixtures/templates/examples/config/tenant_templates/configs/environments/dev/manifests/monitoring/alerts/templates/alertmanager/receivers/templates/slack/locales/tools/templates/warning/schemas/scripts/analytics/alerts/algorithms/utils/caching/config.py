"""
Configuration de Production pour le Système de Cache
===================================================

Configuration optimisée pour un environnement de production avec
haute disponibilité, sécurité renforcée et monitoring complet.

Cette configuration est adaptée pour:
- Production Spotify AI Agent
- Charge élevée (millions de requêtes/jour)
- Multi-tenant avec isolation stricte
- Monitoring et alerting complets
- Sécurité de niveau entreprise

Auteurs: Équipe Spotify AI Agent - Direction technique Fahed Mlaiel
"""

import os
from typing import Dict, Any, List


def get_production_config() -> Dict[str, Any]:
    """Configuration de production optimisée"""
    
    return {
        # Configuration générale
        "environment": "production",
        "debug": False,
        "log_level": "INFO",
        
        # Configuration des niveaux de cache
        "cache_levels": 3,
        "default_ttl": 3600,  # 1 heure
        "max_memory_usage": "2GB",
        
        # L1 - Mémoire locale
        "l1_enabled": True,
        "l1_max_size": 50000,
        "l1_policy": "adaptive",
        "l1_memory_limit": "512MB",
        
        # L2 - Redis
        "l2_enabled": True,
        "redis_url": os.getenv("REDIS_URL", "redis://redis-cluster:6379"),
        "redis_cluster": True,
        "redis_cluster_nodes": [
            "redis-node-1:6379",
            "redis-node-2:6379", 
            "redis-node-3:6379"
        ],
        "redis_password": os.getenv("REDIS_PASSWORD"),
        "redis_ssl": True,
        "redis_ssl_cert_reqs": "required",
        "redis_pool_size": 20,
        "redis_timeout": 5.0,
        
        # L3 - Cache distribué (optionnel)
        "l3_enabled": False,
        "l3_backend": "distributed_redis",
        
        # Compression
        "compression_enabled": True,
        "compression_algorithm": "zstd",
        "compression_level": 6,
        "compression_threshold": 1024,  # 1KB
        "auto_compression": True,
        
        # Sérialisation
        "serialization_format": "adaptive",
        "enable_msgpack": True,
        "enable_pickle": True,
        "serialization_security": "high",
        
        # Sécurité
        "security_enabled": True,
        "encryption_enabled": True,
        "encryption_key": os.getenv("CACHE_ENCRYPTION_KEY"),
        "encryption_algorithm": "AES-256-GCM",
        "tenant_isolation": True,
        "security_level": "high",
        "enable_audit_logging": True,
        "audit_retention_days": 90,
        
        # Multi-tenant
        "tenant_isolation": True,
        "tenant_quota_enabled": True,
        "default_tenant_quota": "100MB",
        "tenant_quota_enforcement": "strict",
        "tenant_key_namespace": True,
        
        # Monitoring
        "monitoring_enabled": True,
        "metrics_enabled": True,
        "metrics_port": 8001,
        "metrics_path": "/metrics",
        "prometheus_enabled": True,
        "health_check_enabled": True,
        "health_check_interval": 30,
        "health_check_timeout": 5,
        
        # Alerting
        "alerting_enabled": True,
        "alert_manager_url": os.getenv("ALERTMANAGER_URL"),
        "slack_webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
        "pagerduty_integration_key": os.getenv("PAGERDUTY_KEY"),
        "email_alerts_enabled": True,
        "alert_thresholds": {
            "low_hit_ratio": 80,
            "high_memory_usage": 85,
            "high_error_rate": 5,
            "high_latency_p95": 100,  # ms
            "circuit_breaker_open": 1
        },
        
        # Performance
        "performance_monitoring": True,
        "benchmarking_enabled": False,  # Désactivé en prod
        "auto_optimization": True,
        "ml_predictions": True,
        "adaptive_ttl": True,
        "cache_warming": True,
        "preload_enabled": True,
        
        # Circuit Breakers
        "circuit_breaker_enabled": True,
        "circuit_breaker_config": {
            "failure_threshold": 5,
            "recovery_timeout": 30,
            "success_threshold": 3,
            "timeout": 5.0
        },
        
        # Analytics
        "analytics_enabled": True,
        "analytics_retention_days": 30,
        "usage_analytics": True,
        "performance_analytics": True,
        "trend_analysis": True,
        "predictive_analytics": True,
        
        # Backup & Recovery
        "backup_enabled": True,
        "backup_interval": 3600,  # 1 heure
        "backup_retention": 7,    # 7 jours
        "recovery_enabled": True,
        "disaster_recovery": True,
        
        # Scaling
        "auto_scaling": True,
        "horizontal_scaling": True,
        "load_balancing": True,
        "sharding_enabled": True,
        "consistent_hashing": True,
        
        # Development/Debug (désactivés en prod)
        "debug_mode": False,
        "verbose_logging": False,
        "profiling_enabled": False,
        "development_tools": False
    }


def get_development_config() -> Dict[str, Any]:
    """Configuration de développement"""
    
    config = get_production_config()
    
    # Overrides pour développement
    config.update({
        "environment": "development",
        "debug": True,
        "log_level": "DEBUG",
        
        # Configuration allégée pour dev
        "l1_max_size": 1000,
        "redis_cluster": False,
        "redis_url": "redis://localhost:6379",
        "redis_ssl": False,
        
        # Sécurité allégée
        "encryption_enabled": False,
        "security_level": "normal",
        
        # Monitoring simplifié
        "health_check_interval": 60,
        "alerting_enabled": False,
        
        # Outils de développement
        "debug_mode": True,
        "verbose_logging": True,
        "profiling_enabled": True,
        "development_tools": True,
        "benchmarking_enabled": True
    })
    
    return config


def get_testing_config() -> Dict[str, Any]:
    """Configuration pour les tests"""
    
    return {
        "environment": "testing",
        "debug": True,
        "log_level": "WARNING",
        
        # Configuration minimale pour tests
        "cache_levels": 1,
        "l1_enabled": True,
        "l1_max_size": 100,
        "l1_policy": "lru",
        
        "l2_enabled": False,
        "l3_enabled": False,
        
        # Désactivation des fonctionnalités non nécessaires pour tests
        "compression_enabled": False,
        "encryption_enabled": False,
        "monitoring_enabled": False,
        "alerting_enabled": False,
        "analytics_enabled": False,
        "backup_enabled": False,
        "circuit_breaker_enabled": False,
        "ml_predictions": False,
        "auto_optimization": False,
        
        # Durées courtes pour tests
        "default_ttl": 60,
        "health_check_interval": 5,
        
        # Mode test
        "testing_mode": True,
        "mock_external_services": True
    }


def get_config_by_environment(env: str = None) -> Dict[str, Any]:
    """Retourne la configuration selon l'environnement"""
    
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")
    
    env = env.lower()
    
    if env == "production":
        return get_production_config()
    elif env == "testing":
        return get_testing_config()
    else:
        return get_development_config()


# Configuration des alertes par défaut
DEFAULT_ALERT_RULES = [
    {
        "name": "cache_low_hit_ratio",
        "condition": "hit_ratio < 80",
        "severity": "warning",
        "description": "Taux de succès du cache faible",
        "notification_channels": ["slack"],
        "auto_resolve": True,
        "resolve_threshold": "hit_ratio > 85"
    },
    {
        "name": "cache_high_memory_usage",
        "condition": "memory_usage_percent > 90",
        "severity": "critical",
        "description": "Utilisation mémoire critique",
        "notification_channels": ["slack", "pagerduty"],
        "auto_resolve": True,
        "resolve_threshold": "memory_usage_percent < 80"
    },
    {
        "name": "cache_circuit_breaker_open",
        "condition": "circuit_breaker_open_count > 0",
        "severity": "critical",
        "description": "Circuit breaker ouvert",
        "notification_channels": ["slack", "pagerduty", "email"],
        "auto_resolve": True,
        "resolve_threshold": "circuit_breaker_open_count == 0"
    },
    {
        "name": "cache_high_error_rate",
        "condition": "error_rate > 5",
        "severity": "error",
        "description": "Taux d'erreur élevé",
        "notification_channels": ["slack"],
        "auto_resolve": True,
        "resolve_threshold": "error_rate < 2"
    },
    {
        "name": "cache_high_latency",
        "condition": "p95_latency_ms > 100",
        "severity": "warning",
        "description": "Latence élevée",
        "notification_channels": ["slack"],
        "auto_resolve": True,
        "resolve_threshold": "p95_latency_ms < 50"
    }
]

# Quotas par défaut par type de tenant
DEFAULT_TENANT_QUOTAS = {
    "free": {
        "memory_limit": "10MB",
        "operations_per_minute": 1000,
        "max_ttl": 3600,
        "compression_enabled": False,
        "encryption_enabled": False
    },
    "basic": {
        "memory_limit": "100MB", 
        "operations_per_minute": 10000,
        "max_ttl": 7200,
        "compression_enabled": True,
        "encryption_enabled": False
    },
    "premium": {
        "memory_limit": "1GB",
        "operations_per_minute": 100000,
        "max_ttl": 86400,
        "compression_enabled": True,
        "encryption_enabled": True
    },
    "enterprise": {
        "memory_limit": "10GB",
        "operations_per_minute": 1000000,
        "max_ttl": 604800,  # 1 semaine
        "compression_enabled": True,
        "encryption_enabled": True,
        "dedicated_resources": True
    }
}

# Configuration de clustering Redis
REDIS_CLUSTER_CONFIG = {
    "startup_nodes": [
        {"host": "redis-node-1", "port": 6379},
        {"host": "redis-node-2", "port": 6379},
        {"host": "redis-node-3", "port": 6379}
    ],
    "decode_responses": True,
    "skip_full_coverage_check": False,
    "health_check_interval": 30,
    "max_connections": 100,
    "retry_on_timeout": True,
    "socket_connect_timeout": 5,
    "socket_timeout": 5
}

# Configuration de monitoring Prometheus
PROMETHEUS_CONFIG = {
    "job_name": "spotify-ai-cache",
    "scrape_interval": "30s",
    "scrape_timeout": "10s",
    "metrics_path": "/metrics",
    "honor_labels": True,
    "static_configs": [
        {
            "targets": ["cache-service:8001"],
            "labels": {
                "service": "spotify-ai-cache",
                "environment": "production",
                "team": "platform"
            }
        }
    ]
}

# Dashboard Grafana
GRAFANA_DASHBOARD_CONFIG = {
    "dashboard": {
        "title": "Spotify AI Agent - Cache System",
        "tags": ["spotify", "cache", "ai-agent"],
        "time": {
            "from": "now-1h",
            "to": "now"
        },
        "refresh": "10s",
        "panels": [
            {
                "title": "Cache Hit Ratio",
                "type": "stat",
                "targets": [
                    {
                        "expr": "spotify_ai_cache_hit_ratio_percent",
                        "legendFormat": "{{level}} - {{tenant_id}}"
                    }
                ],
                "thresholds": [
                    {"color": "red", "value": 0},
                    {"color": "yellow", "value": 80},
                    {"color": "green", "value": 95}
                ]
            },
            {
                "title": "Memory Usage",
                "type": "graph",
                "targets": [
                    {
                        "expr": "spotify_ai_cache_memory_usage_bytes",
                        "legendFormat": "{{level}} Memory Usage"
                    }
                ]
            },
            {
                "title": "Operations Rate",
                "type": "graph", 
                "targets": [
                    {
                        "expr": "rate(spotify_ai_cache_operations_total[5m])",
                        "legendFormat": "{{operation}} - {{status}}"
                    }
                ]
            },
            {
                "title": "Response Times",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, spotify_ai_cache_operation_duration_seconds)",
                        "legendFormat": "P95 Latency"
                    },
                    {
                        "expr": "histogram_quantile(0.50, spotify_ai_cache_operation_duration_seconds)",
                        "legendFormat": "P50 Latency"
                    }
                ]
            }
        ]
    }
}


def validate_config(config: Dict[str, Any]) -> bool:
    """Valide une configuration"""
    
    required_keys = [
        "environment",
        "cache_levels", 
        "l1_enabled",
        "monitoring_enabled"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validation des valeurs
    if config["cache_levels"] < 1 or config["cache_levels"] > 3:
        raise ValueError("cache_levels must be between 1 and 3")
    
    if config.get("l1_max_size", 0) <= 0:
        raise ValueError("l1_max_size must be positive")
    
    return True


def get_docker_compose_config() -> Dict[str, Any]:
    """Configuration Docker Compose pour déploiement"""
    
    return {
        "version": "3.8",
        "services": {
            "cache-service": {
                "build": ".",
                "ports": ["8000:8000", "8001:8001"],
                "environment": [
                    "ENVIRONMENT=production",
                    "REDIS_URL=redis://redis-cluster:6379",
                    "PROMETHEUS_ENABLED=true"
                ],
                "depends_on": ["redis-cluster", "prometheus"],
                "restart": "unless-stopped",
                "deploy": {
                    "replicas": 3,
                    "resources": {
                        "limits": {"memory": "1G", "cpus": "0.5"},
                        "reservations": {"memory": "512M", "cpus": "0.25"}
                    }
                }
            },
            "redis-cluster": {
                "image": "redis:7-alpine",
                "command": "redis-server --cluster-enabled yes --cluster-config-file nodes.conf",
                "ports": ["6379:6379"],
                "volumes": ["redis-data:/data"],
                "restart": "unless-stopped"
            },
            "prometheus": {
                "image": "prom/prometheus:latest",
                "ports": ["9090:9090"],
                "volumes": ["./prometheus.yml:/etc/prometheus/prometheus.yml"],
                "restart": "unless-stopped"
            },
            "grafana": {
                "image": "grafana/grafana:latest",
                "ports": ["3000:3000"],
                "environment": ["GF_SECURITY_ADMIN_PASSWORD=admin"],
                "volumes": ["grafana-data:/var/lib/grafana"],
                "restart": "unless-stopped"
            }
        },
        "volumes": {
            "redis-data": {},
            "grafana-data": {}
        },
        "networks": {
            "cache-network": {"driver": "bridge"}
        }
    }
