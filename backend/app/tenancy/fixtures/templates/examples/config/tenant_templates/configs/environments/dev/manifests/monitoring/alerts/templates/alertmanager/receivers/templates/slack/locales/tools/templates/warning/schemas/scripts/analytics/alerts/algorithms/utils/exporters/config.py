"""
Advanced Metrics Exporters Configuration
========================================

Configuration centralisée pour tous les exportateurs de métriques
avec gestion environnementale et validation.
"""

import os
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    """Environnements supportés."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ExporterConfiguration:
    """Configuration principale pour tous les exportateurs."""
    
    # Informations générales
    tenant_id: str
    environment: Environment = Environment.PRODUCTION
    service_name: str = "spotify-ai-agent"
    version: str = "2.1.0"
    
    # Configuration Prometheus
    prometheus: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "url": "http://localhost:9090",
        "encryption_enabled": True,
        "compression_enabled": True,
        "batch_size": 1000,
        "rate_limit_per_second": 2000,
        "push_gateway_url": "http://localhost:9091"
    })
    
    # Configuration Grafana
    grafana: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "url": "http://localhost:3000",
        "api_key": "",
        "org_id": 1,
        "create_dashboards": True,
        "dashboard_types": ["ai_metrics", "business_metrics", "performance_overview"]
    })
    
    # Configuration Elasticsearch
    elasticsearch: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "hosts": ["http://localhost:9200"],
        "username": "",
        "password": "",
        "index_prefix": "spotify-ai-metrics",
        "retention_days": 90,
        "compression_enabled": True
    })
    
    # Configuration InfluxDB
    influxdb: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "url": "http://localhost:8086",
        "token": "",
        "org": "spotify-ai",
        "bucket_prefix": "spotify-ai",
        "retention_period": "90d",
        "enable_downsampling": True
    })
    
    # Configuration des exportateurs tiers
    third_party: Dict[str, Any] = field(default_factory=lambda: {
        "datadog": {
            "enabled": False,
            "api_key": "",
            "app_key": "",
            "site": "datadoghq.com"
        },
        "newrelic": {
            "enabled": False,
            "api_key": "",
            "account_id": ""
        },
        "splunk": {
            "enabled": False,
            "hec_url": "",
            "hec_token": "",
            "index": "spotify_ai_metrics"
        }
    })
    
    # Configuration batch
    batch: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "batch_size": 1000,
        "max_batch_size": 10000,
        "batch_timeout": 30.0,
        "compression_enabled": True,
        "parallel_workers": 4,
        "memory_limit_mb": 512
    })
    
    # Configuration streaming
    streaming: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "protocols": {
            "websocket": {
                "enabled": True,
                "url": "ws://localhost:8080/metrics/stream",
                "max_connections": 100
            },
            "sse": {
                "enabled": True,
                "url": "http://localhost:8080/metrics/sse",
                "enable_batching": True
            },
            "mqtt": {
                "enabled": False,
                "url": "mqtt://localhost:1883",
                "topic_prefix": "spotify/ai/metrics"
            }
        }
    })
    
    # Configuration de sécurité
    security: Dict[str, Any] = field(default_factory=lambda: {
        "encryption_key": "",
        "auth_tokens": {},
        "tls_enabled": True,
        "verify_certificates": True
    })
    
    # Configuration de monitoring
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        "metrics_retention_hours": 24,
        "performance_monitoring": True,
        "health_checks_enabled": True,
        "alerting_enabled": True
    })


class ConfigurationManager:
    """Gestionnaire de configuration pour les exportateurs."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config: Optional[ExporterConfiguration] = None
        
    def _get_default_config_path(self) -> str:
        """Retourne le chemin de configuration par défaut."""
        env = os.getenv('SPOTIFY_AI_ENV', 'production')
        return f"/etc/spotify-ai/exporters_{env}.yaml"
        
    def load_configuration(self, tenant_id: str) -> ExporterConfiguration:
        """Charge la configuration pour un tenant."""
        try:
            # Charger depuis fichier YAML si existe
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                # Merger avec la configuration par défaut
                return self._merge_with_defaults(tenant_id, config_data)
            else:
                # Configuration par défaut
                return self._create_default_configuration(tenant_id)
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._create_default_configuration(tenant_id)
            
    def _create_default_configuration(self, tenant_id: str) -> ExporterConfiguration:
        """Crée une configuration par défaut."""
        env_name = os.getenv('SPOTIFY_AI_ENV', 'production')
        environment = Environment(env_name)
        
        config = ExporterConfiguration(
            tenant_id=tenant_id,
            environment=environment
        )
        
        # Adapter selon l'environnement
        if environment == Environment.DEVELOPMENT:
            config.prometheus["rate_limit_per_second"] = 100
            config.batch["batch_size"] = 100
            config.monitoring["metrics_retention_hours"] = 1
        elif environment == Environment.STAGING:
            config.prometheus["rate_limit_per_second"] = 500
            config.batch["batch_size"] = 500
            config.monitoring["metrics_retention_hours"] = 12
            
        return config
        
    def _merge_with_defaults(self, tenant_id: str, config_data: Dict[str, Any]) -> ExporterConfiguration:
        """Merge la configuration chargée avec les défauts."""
        default_config = self._create_default_configuration(tenant_id)
        
        # Merge récursif
        def deep_merge(default: Dict, override: Dict) -> Dict:
            result = default.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
            
        # Convertir en dict pour le merge
        default_dict = self._config_to_dict(default_config)
        merged_dict = deep_merge(default_dict, config_data)
        
        # Reconvertir en objet configuration
        return self._dict_to_config(merged_dict)
        
    def _config_to_dict(self, config: ExporterConfiguration) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        return {
            'tenant_id': config.tenant_id,
            'environment': config.environment.value,
            'service_name': config.service_name,
            'version': config.version,
            'prometheus': config.prometheus,
            'grafana': config.grafana,
            'elasticsearch': config.elasticsearch,
            'influxdb': config.influxdb,
            'third_party': config.third_party,
            'batch': config.batch,
            'streaming': config.streaming,
            'security': config.security,
            'monitoring': config.monitoring
        }
        
    def _dict_to_config(self, data: Dict[str, Any]) -> ExporterConfiguration:
        """Convertit un dictionnaire en configuration."""
        return ExporterConfiguration(
            tenant_id=data['tenant_id'],
            environment=Environment(data.get('environment', 'production')),
            service_name=data.get('service_name', 'spotify-ai-agent'),
            version=data.get('version', '2.1.0'),
            prometheus=data.get('prometheus', {}),
            grafana=data.get('grafana', {}),
            elasticsearch=data.get('elasticsearch', {}),
            influxdb=data.get('influxdb', {}),
            third_party=data.get('third_party', {}),
            batch=data.get('batch', {}),
            streaming=data.get('streaming', {}),
            security=data.get('security', {}),
            monitoring=data.get('monitoring', {})
        )
        
    def save_configuration(self, config: ExporterConfiguration):
        """Sauvegarde la configuration."""
        try:
            config_dict = self._config_to_dict(config)
            
            # Créer le répertoire si nécessaire
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            
    def validate_configuration(self, config: ExporterConfiguration) -> List[str]:
        """Valide la configuration et retourne les erreurs."""
        errors = []
        
        # Validation tenant_id
        if not config.tenant_id or len(config.tenant_id) < 3:
            errors.append("tenant_id must be at least 3 characters long")
            
        # Validation Prometheus
        if config.prometheus.get("enabled", False):
            if not config.prometheus.get("url"):
                errors.append("Prometheus URL is required when enabled")
                
        # Validation Grafana
        if config.grafana.get("enabled", False):
            if not config.grafana.get("url"):
                errors.append("Grafana URL is required when enabled")
            if not config.grafana.get("api_key"):
                errors.append("Grafana API key is required when enabled")
                
        # Validation Elasticsearch
        if config.elasticsearch.get("enabled", False):
            hosts = config.elasticsearch.get("hosts", [])
            if not hosts:
                errors.append("Elasticsearch hosts are required when enabled")
                
        # Validation InfluxDB
        if config.influxdb.get("enabled", False):
            if not config.influxdb.get("url"):
                errors.append("InfluxDB URL is required when enabled")
            if not config.influxdb.get("token"):
                errors.append("InfluxDB token is required when enabled")
                
        # Validation third-party
        for service, service_config in config.third_party.items():
            if service_config.get("enabled", False):
                if service == "datadog":
                    if not service_config.get("api_key"):
                        errors.append("Datadog API key is required when enabled")
                elif service == "newrelic":
                    if not service_config.get("api_key"):
                        errors.append("New Relic API key is required when enabled")
                elif service == "splunk":
                    if not service_config.get("hec_url") or not service_config.get("hec_token"):
                        errors.append("Splunk HEC URL and token are required when enabled")
                        
        return errors
        
    def get_environment_template(self, environment: Environment) -> Dict[str, Any]:
        """Retourne un template de configuration pour un environnement."""
        templates = {
            Environment.DEVELOPMENT: {
                "prometheus": {
                    "enabled": True,
                    "url": "http://localhost:9090",
                    "rate_limit_per_second": 100
                },
                "grafana": {
                    "enabled": True,
                    "url": "http://localhost:3000"
                },
                "elasticsearch": {
                    "enabled": False
                },
                "influxdb": {
                    "enabled": False
                },
                "batch": {
                    "batch_size": 100,
                    "parallel_workers": 2
                },
                "streaming": {
                    "enabled": False
                }
            },
            Environment.STAGING: {
                "prometheus": {
                    "enabled": True,
                    "url": "http://prometheus-staging:9090",
                    "rate_limit_per_second": 500
                },
                "grafana": {
                    "enabled": True,
                    "url": "http://grafana-staging:3000"
                },
                "elasticsearch": {
                    "enabled": True,
                    "hosts": ["http://elasticsearch-staging:9200"]
                },
                "influxdb": {
                    "enabled": False
                },
                "batch": {
                    "batch_size": 500,
                    "parallel_workers": 4
                }
            },
            Environment.PRODUCTION: {
                "prometheus": {
                    "enabled": True,
                    "url": "http://prometheus-prod:9090",
                    "rate_limit_per_second": 2000,
                    "encryption_enabled": True
                },
                "grafana": {
                    "enabled": True,
                    "url": "http://grafana-prod:3000"
                },
                "elasticsearch": {
                    "enabled": True,
                    "hosts": [
                        "http://elasticsearch-prod-1:9200",
                        "http://elasticsearch-prod-2:9200",
                        "http://elasticsearch-prod-3:9200"
                    ]
                },
                "influxdb": {
                    "enabled": True,
                    "url": "http://influxdb-prod:8086"
                },
                "batch": {
                    "batch_size": 1000,
                    "parallel_workers": 8,
                    "memory_limit_mb": 1024
                },
                "streaming": {
                    "enabled": True
                },
                "third_party": {
                    "datadog": {
                        "enabled": True
                    }
                }
            }
        }
        
        return templates.get(environment, {})


# Configuration de sécurité
class SecurityManager:
    """Gestionnaire de sécurité pour les exportateurs."""
    
    @staticmethod
    def generate_encryption_key() -> str:
        """Génère une clé de chiffrement."""
        from cryptography.fernet import Fernet
        return Fernet.generate_key().decode()
        
    @staticmethod
    def generate_auth_token() -> str:
        """Génère un token d'authentification."""
        import secrets
        return secrets.token_urlsafe(32)
        
    @staticmethod
    def encrypt_sensitive_data(data: str, key: str) -> str:
        """Chiffre des données sensibles."""
        from cryptography.fernet import Fernet
        cipher = Fernet(key.encode())
        return cipher.encrypt(data.encode()).decode()
        
    @staticmethod
    def decrypt_sensitive_data(encrypted_data: str, key: str) -> str:
        """Déchiffre des données sensibles."""
        from cryptography.fernet import Fernet
        cipher = Fernet(key.encode())
        return cipher.decrypt(encrypted_data.encode()).decode()


# Example configuration files
EXAMPLE_DEVELOPMENT_CONFIG = """
# Spotify AI Agent - Development Configuration
tenant_id: "spotify_artist_dev_test"
environment: "development"
service_name: "spotify-ai-agent"
version: "2.1.0"

prometheus:
  enabled: true
  url: "http://localhost:9090"
  encryption_enabled: false
  rate_limit_per_second: 100
  batch_size: 100

grafana:
  enabled: true
  url: "http://localhost:3000"
  api_key: "dev-api-key"
  org_id: 1

elasticsearch:
  enabled: false

influxdb:
  enabled: false

batch:
  enabled: true
  batch_size: 100
  parallel_workers: 2
  memory_limit_mb: 256

streaming:
  enabled: false

security:
  encryption_key: ""
  tls_enabled: false

monitoring:
  metrics_retention_hours: 1
  performance_monitoring: true
"""

EXAMPLE_PRODUCTION_CONFIG = """
# Spotify AI Agent - Production Configuration
tenant_id: "spotify_artist_production"
environment: "production"
service_name: "spotify-ai-agent"
version: "2.1.0"

prometheus:
  enabled: true
  url: "https://prometheus-prod.spotify-ai.com"
  encryption_enabled: true
  compression_enabled: true
  rate_limit_per_second: 2000
  batch_size: 1000

grafana:
  enabled: true
  url: "https://grafana-prod.spotify-ai.com"
  api_key: "${GRAFANA_API_KEY}"
  org_id: 1
  create_dashboards: true

elasticsearch:
  enabled: true
  hosts:
    - "https://elasticsearch-prod-1.spotify-ai.com:9200"
    - "https://elasticsearch-prod-2.spotify-ai.com:9200"
    - "https://elasticsearch-prod-3.spotify-ai.com:9200"
  username: "${ELASTICSEARCH_USERNAME}"
  password: "${ELASTICSEARCH_PASSWORD}"
  retention_days: 90

influxdb:
  enabled: true
  url: "https://influxdb-prod.spotify-ai.com:8086"
  token: "${INFLUXDB_TOKEN}"
  org: "spotify-ai"
  retention_period: "90d"

third_party:
  datadog:
    enabled: true
    api_key: "${DATADOG_API_KEY}"
    app_key: "${DATADOG_APP_KEY}"
  
  newrelic:
    enabled: false
    
  splunk:
    enabled: false

batch:
  enabled: true
  batch_size: 1000
  max_batch_size: 10000
  parallel_workers: 8
  memory_limit_mb: 1024
  compression_enabled: true

streaming:
  enabled: true
  protocols:
    websocket:
      enabled: true
      url: "wss://metrics-stream.spotify-ai.com/ws"
      max_connections: 100
    
    sse:
      enabled: true
      url: "https://metrics-stream.spotify-ai.com/sse"
      
    mqtt:
      enabled: false

security:
  encryption_key: "${SPOTIFY_AI_ENCRYPTION_KEY}"
  tls_enabled: true
  verify_certificates: true

monitoring:
  metrics_retention_hours: 24
  performance_monitoring: true
  health_checks_enabled: true
  alerting_enabled: true
"""


if __name__ == "__main__":
    import structlog
    
    logger = structlog.get_logger(__name__)
    
    # Test de configuration
    config_manager = ConfigurationManager()
    
    # Charger configuration pour un tenant
    config = config_manager.load_configuration("spotify_artist_daft_punk")
    
    # Valider
    errors = config_manager.validate_configuration(config)
    if errors:
        print(f"Configuration errors: {errors}")
    else:
        print("Configuration is valid")
        
    # Sauvegarder un exemple
    config_manager.save_configuration(config)
    
    # Générer des clés de sécurité
    encryption_key = SecurityManager.generate_encryption_key()
    auth_token = SecurityManager.generate_auth_token()
    
    print(f"Generated encryption key: {encryption_key}")
    print(f"Generated auth token: {auth_token}")
