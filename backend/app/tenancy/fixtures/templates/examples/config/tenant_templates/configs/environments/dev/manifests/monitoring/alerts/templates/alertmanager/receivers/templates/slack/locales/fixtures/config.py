"""
Configuration Avancée pour l'Environnement de Développement
==========================================================

Configuration complète pour l'environnement de développement du système
de fixtures d'alertes Slack avec toutes les optimisations et configurations
nécessaires pour un développement efficace.

Auteur: Fahed Mlaiel - Lead Developer Achiri
Version: 2.5.0
"""

import os
from typing import Dict, Any, List
from pathlib import Path

# Configuration de base pour l'environnement de développement
DEV_CONFIG = {
    # === Configuration de la base de données ===
    "database": {
        "host": os.getenv("DEV_DB_HOST", "localhost"),
        "port": int(os.getenv("DEV_DB_PORT", 5432)),
        "user": os.getenv("DEV_DB_USER", "postgres"),
        "password": os.getenv("DEV_DB_PASSWORD", "dev_password"),
        "name": os.getenv("DEV_DB_NAME", "spotify_ai_dev"),
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 3600,
        "echo": True,  # Logs SQL en développement
        "ssl_mode": "disable"
    },
    
    # === Configuration Redis ===
    "redis": {
        "host": os.getenv("DEV_REDIS_HOST", "localhost"),
        "port": int(os.getenv("DEV_REDIS_PORT", 6379)),
        "password": os.getenv("DEV_REDIS_PASSWORD", ""),
        "db": int(os.getenv("DEV_REDIS_DB", 0)),
        "decode_responses": True,
        "socket_timeout": 5,
        "socket_connect_timeout": 5,
        "retry_on_timeout": True,
        "health_check_interval": 30
    },
    
    # === Configuration de sécurité ===
    "security": {
        "encryption_key": os.getenv("DEV_ENCRYPTION_KEY", "dev_key_change_in_production"),
        "jwt_secret": os.getenv("DEV_JWT_SECRET", "dev_jwt_secret"),
        "jwt_algorithm": "HS256",
        "jwt_expiration": 3600,  # 1 heure
        "api_key_header": "X-API-Key",
        "rate_limit": {
            "requests_per_minute": 100,
            "burst_size": 200
        },
        "cors": {
            "allowed_origins": ["http://localhost:3000", "http://localhost:8080"],
            "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allowed_headers": ["*"],
            "allow_credentials": True
        }
    },
    
    # === Configuration Slack ===
    "slack": {
        "bot_token": os.getenv("DEV_SLACK_BOT_TOKEN", "xoxb-dev-token"),
        "app_token": os.getenv("DEV_SLACK_APP_TOKEN", "xapp-dev-token"),
        "signing_secret": os.getenv("DEV_SLACK_SIGNING_SECRET", "dev_signing_secret"),
        "webhook_url": os.getenv("DEV_SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/dev/webhook"),
        "default_channel": "#dev-alerts",
        "timeout": 30,
        "retry_count": 3,
        "retry_delay": 1.0
    },
    
    # === Configuration du cache ===
    "cache": {
        "default_ttl": 1800,  # 30 minutes en dev
        "max_size": 500,  # Plus petit en dev
        "cleanup_interval": 300,  # 5 minutes
        "compression": False,  # Pas de compression en dev
        "serializer": "json"
    },
    
    # === Configuration des templates ===
    "templates": {
        "auto_reload": True,  # Rechargement automatique en dev
        "strict_undefined": True,
        "cache_size": 100,
        "enable_async": True,
        "trim_blocks": True,
        "lstrip_blocks": True
    },
    
    # === Configuration du logging ===
    "logging": {
        "level": "DEBUG",
        "format": "json",
        "handlers": {
            "console": {
                "enabled": True,
                "level": "DEBUG"
            },
            "file": {
                "enabled": True,
                "level": "INFO",
                "filename": "/tmp/slack_fixtures_dev.log",
                "max_size": "10MB",
                "backup_count": 3
            },
            "syslog": {
                "enabled": False
            }
        },
        "loggers": {
            "slack_fixtures": "DEBUG",
            "asyncpg": "INFO",
            "aioredis": "INFO",
            "fastapi": "INFO"
        }
    },
    
    # === Configuration du monitoring ===
    "monitoring": {
        "prometheus": {
            "enabled": True,
            "port": 8001,
            "path": "/metrics"
        },
        "health_check": {
            "enabled": True,
            "interval": 30,
            "timeout": 10
        },
        "tracing": {
            "enabled": True,
            "jaeger": {
                "agent_host": "localhost",
                "agent_port": 6831,
                "service_name": "slack-fixtures-dev"
            }
        }
    },
    
    # === Configuration de l'API ===
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,  # Auto-reload en dev
        "debug": True,
        "workers": 1,  # Un seul worker en dev
        "access_log": True,
        "log_level": "debug",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json"
    },
    
    # === Configuration des fixtures ===
    "fixtures": {
        "auto_load_defaults": True,
        "validate_on_load": True,
        "backup_on_update": True,
        "versioning": True,
        "supported_locales": ["fr", "en", "de", "es", "it"],
        "default_locale": "fr",
        "fallback_locale": "en"
    },
    
    # === Configuration des tests ===
    "testing": {
        "enabled": True,
        "mock_external_services": True,
        "test_database": "spotify_ai_test",
        "fixtures_path": "./test_fixtures",
        "coverage_threshold": 80
    },
    
    # === Configuration du développement ===
    "development": {
        "hot_reload": True,
        "debug_toolbar": True,
        "profiling": True,
        "mock_data": True,
        "seed_database": True,
        "auto_migrate": True
    }
}

# Configuration des tenants de développement
DEV_TENANTS = [
    {
        "id": "dev-tenant-01",
        "name": "Tenant de Développement 1",
        "description": "Tenant principal pour les tests de développement",
        "settings": {
            "slack_channel": "#dev-tenant-01",
            "timezone": "Europe/Paris",
            "locale": "fr",
            "alert_threshold": "warning"
        }
    },
    {
        "id": "dev-tenant-02", 
        "name": "Tenant de Développement 2",
        "description": "Tenant secondaire pour tests multi-tenant",
        "settings": {
            "slack_channel": "#dev-tenant-02",
            "timezone": "America/New_York",
            "locale": "en",
            "alert_threshold": "info"
        }
    },
    {
        "id": "demo-tenant",
        "name": "Tenant de Démonstration",
        "description": "Tenant pour les démonstrations",
        "settings": {
            "slack_channel": "#demo-alerts",
            "timezone": "Europe/London",
            "locale": "en",
            "alert_threshold": "low"
        }
    }
]

# Configuration des alertes de test
DEV_TEST_ALERTS = [
    {
        "type": "system_critical",
        "severity": "critical",
        "test_data": {
            "summary": "Test Critical System Alert",
            "description": "This is a test critical alert for development",
            "metric_name": "cpu_usage",
            "metric_value": "95%",
            "threshold": "90%",
            "hostname": "dev-server-01"
        }
    },
    {
        "type": "application_error",
        "severity": "warning",
        "test_data": {
            "summary": "Test Application Error",
            "description": "Test error for development purposes",
            "application": "spotify-ai-agent",
            "service": "fixtures-api",
            "error_message": "Connection timeout to database",
            "error_code": "DB_TIMEOUT"
        }
    },
    {
        "type": "performance_degradation",
        "severity": "warning",
        "test_data": {
            "summary": "Test Performance Alert",
            "description": "Performance degradation detected in development",
            "service": "slack-fixtures",
            "current_value": "2.5s",
            "baseline_value": "1.2s",
            "degradation_percentage": 108
        }
    }
]

# Configuration des webhooks de test
DEV_WEBHOOKS = {
    "slack": {
        "test_webhook": os.getenv("DEV_SLACK_TEST_WEBHOOK", "https://hooks.slack.com/services/test"),
        "validation_webhook": os.getenv("DEV_SLACK_VALIDATION_WEBHOOK", "https://hooks.slack.com/services/validation")
    },
    "teams": {
        "test_webhook": os.getenv("DEV_TEAMS_TEST_WEBHOOK", "https://outlook.office.com/webhook/test")
    }
}

# Scripts de développement
DEV_SCRIPTS = {
    "setup": {
        "description": "Configuration initiale de l'environnement de développement",
        "command": "./scripts/dev_setup.sh"
    },
    "migrate": {
        "description": "Exécution des migrations de base de données",
        "command": "./scripts/dev_migrate.sh"
    },
    "seed": {
        "description": "Chargement des données de test",
        "command": "./scripts/dev_seed.sh"
    },
    "test": {
        "description": "Exécution de la suite de tests",
        "command": "./scripts/dev_test.sh"
    },
    "reset": {
        "description": "Reset complet de l'environnement",
        "command": "./scripts/dev_reset.sh"
    }
}

# Configuration Docker pour le développement
DOCKER_DEV_CONFIG = {
    "services": {
        "postgres": {
            "image": "postgres:14-alpine",
            "environment": {
                "POSTGRES_DB": "spotify_ai_dev",
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": "dev_password"
            },
            "ports": ["5432:5432"],
            "volumes": ["postgres_dev_data:/var/lib/postgresql/data"]
        },
        "redis": {
            "image": "redis:7-alpine",
            "ports": ["6379:6379"],
            "command": "redis-server --appendonly yes",
            "volumes": ["redis_dev_data:/data"]
        },
        "jaeger": {
            "image": "jaegertracing/all-in-one:latest",
            "ports": [
                "16686:16686",  # UI
                "14268:14268",  # HTTP
                "6831:6831/udp"  # UDP
            ],
            "environment": {
                "COLLECTOR_OTLP_ENABLED": "true"
            }
        },
        "prometheus": {
            "image": "prom/prometheus:latest",
            "ports": ["9090:9090"],
            "volumes": ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
        },
        "grafana": {
            "image": "grafana/grafana:latest",
            "ports": ["3000:3000"],
            "environment": {
                "GF_SECURITY_ADMIN_PASSWORD": "admin"
            },
            "volumes": ["grafana_dev_data:/var/lib/grafana"]
        }
    }
}

# Variables d'environnement requises pour le développement
REQUIRED_ENV_VARS = [
    "DEV_DB_PASSWORD",
    "DEV_SLACK_BOT_TOKEN",
    "DEV_SLACK_WEBHOOK_URL",
    "DEV_ENCRYPTION_KEY"
]

# Variables d'environnement optionnelles avec valeurs par défaut
OPTIONAL_ENV_VARS = {
    "DEV_DB_HOST": "localhost",
    "DEV_DB_PORT": "5432",
    "DEV_REDIS_HOST": "localhost", 
    "DEV_REDIS_PORT": "6379",
    "DEV_API_PORT": "8000",
    "DEV_LOG_LEVEL": "DEBUG"
}

def get_dev_config() -> Dict[str, Any]:
    """
    Retourne la configuration complète pour l'environnement de développement.
    
    Returns:
        Configuration complète
    """
    return DEV_CONFIG

def validate_dev_environment() -> List[str]:
    """
    Valide que l'environnement de développement est correctement configuré.
    
    Returns:
        Liste des erreurs de configuration
    """
    errors = []
    
    # Vérification des variables d'environnement requises
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            errors.append(f"Variable d'environnement manquante: {var}")
    
    # Vérification des chemins
    required_paths = [
        "./templates",
        "./secrets", 
        "./logs"
    ]
    
    for path in required_paths:
        if not Path(path).exists():
            errors.append(f"Répertoire manquant: {path}")
    
    return errors

def setup_dev_environment():
    """
    Configure l'environnement de développement.
    """
    # Création des répertoires nécessaires
    Path("./templates").mkdir(exist_ok=True)
    Path("./secrets").mkdir(exist_ok=True, mode=0o700)
    Path("./logs").mkdir(exist_ok=True)
    Path("./test_fixtures").mkdir(exist_ok=True)
    
    # Génération des clés de développement si nécessaire
    if not os.getenv("DEV_ENCRYPTION_KEY"):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        print(f"Clé de chiffrement générée: {key}")
        print("Ajoutez la variable d'environnement: DEV_ENCRYPTION_KEY=" + key)

# Export des configurations
__all__ = [
    "DEV_CONFIG",
    "DEV_TENANTS", 
    "DEV_TEST_ALERTS",
    "DEV_WEBHOOKS",
    "DEV_SCRIPTS",
    "DOCKER_DEV_CONFIG",
    "get_dev_config",
    "validate_dev_environment",
    "setup_dev_environment"
]
