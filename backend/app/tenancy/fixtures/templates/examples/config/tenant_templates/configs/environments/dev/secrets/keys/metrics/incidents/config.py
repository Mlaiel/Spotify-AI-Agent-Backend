"""
Configuration Avancée - Système de Gestion des Incidents et Métriques
=====================================================================

Configuration centralisée enterprise-grade pour:
- Paramètres de monitoring et alerting
- Règles de détection et classification
- Seuils et triggers personnalisables
- Intégrations avec systèmes externes
- Politiques de sécurité et conformité

Structure de configuration:
    - monitoring: Configuration du monitoring
    - alerting: Règles d'alerting
    - thresholds: Seuils et limites
    - integrations: Intégrations externes
    - security: Politiques de sécurité
    - compliance: Règles de conformité
"""

import os
import json
import yaml
from datetime import timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)

class Environment(Enum):
    """Environnements supportés"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

class IntegrationType(Enum):
    """Types d'intégrations supportées"""
    MONITORING = "monitoring"
    ALERTING = "alerting"
    TICKETING = "ticketing"
    COMMUNICATION = "communication"
    SECURITY = "security"
    COMPLIANCE = "compliance"

@dataclass
class MonitoringConfig:
    """Configuration du monitoring"""
    enabled: bool = True
    collection_interval: int = 60  # secondes
    retention_period: int = 30  # jours
    real_time_streaming: bool = True
    batch_size: int = 1000
    compression_enabled: bool = True
    encryption_at_rest: bool = True
    
    # Métriques système
    system_metrics: Dict[str, bool] = field(default_factory=lambda: {
        "cpu_usage": True,
        "memory_usage": True,
        "disk_usage": True,
        "network_io": True,
        "process_count": True,
        "file_descriptors": True
    })
    
    # Métriques application
    application_metrics: Dict[str, bool] = field(default_factory=lambda: {
        "response_time": True,
        "throughput": True,
        "error_rate": True,
        "active_connections": True,
        "queue_depth": True,
        "cache_hit_rate": True
    })
    
    # Métriques métier
    business_metrics: Dict[str, bool] = field(default_factory=lambda: {
        "user_registrations": True,
        "revenue": True,
        "conversion_rate": True,
        "churn_rate": True,
        "feature_usage": True
    })

@dataclass
class AlertingConfig:
    """Configuration de l'alerting"""
    enabled: bool = True
    default_severity: str = "medium"
    escalation_enabled: bool = True
    auto_resolution: bool = True
    notification_throttling: bool = True
    
    # Canaux de notification
    notification_channels: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "email": {
            "enabled": True,
            "smtp_server": "smtp.company.com",
            "port": 587,
            "use_tls": True,
            "from_address": "alerts@company.com",
            "rate_limit": 10  # messages par minute
        },
        "slack": {
            "enabled": True,
            "webhook_url": "${SLACK_WEBHOOK_URL}",
            "channel": "#incidents",
            "rate_limit": 5
        },
        "sms": {
            "enabled": False,
            "provider": "twilio",
            "rate_limit": 2
        },
        "webhook": {
            "enabled": True,
            "endpoints": ["${WEBHOOK_ENDPOINT}"],
            "retry_count": 3
        }
    })
    
    # Templates de notification
    notification_templates: Dict[str, str] = field(default_factory=lambda: {
        "incident_created": """
🚨 **Incident Détecté**
**ID**: {incident_id}
**Criticité**: {severity}
**Catégorie**: {category}
**Titre**: {title}
**Description**: {description}
**Systèmes affectés**: {affected_systems}
**Timestamp**: {timestamp}
        """,
        "incident_escalated": """
⬆️ **Incident Escaladé**
**ID**: {incident_id}
**Niveau**: {escalation_level}
**Raison**: {escalation_reason}
**Assigné à**: {assignee}
        """,
        "incident_resolved": """
✅ **Incident Résolu**
**ID**: {incident_id}
**Durée**: {resolution_time}
**Solution**: {resolution_notes}
        """
    })

@dataclass
class ThresholdConfig:
    """Configuration des seuils"""
    # Seuils système
    system_thresholds: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=lambda: {
        "cpu_usage": {
            "warning": 70.0,
            "critical": 90.0,
            "unit": "percent"
        },
        "memory_usage": {
            "warning": 80.0,
            "critical": 95.0,
            "unit": "percent"
        },
        "disk_usage": {
            "warning": 85.0,
            "critical": 95.0,
            "unit": "percent"
        },
        "response_time": {
            "warning": 1000.0,
            "critical": 5000.0,
            "unit": "milliseconds"
        },
        "error_rate": {
            "warning": 5.0,
            "critical": 10.0,
            "unit": "percent"
        }
    })
    
    # Seuils métier
    business_thresholds: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=lambda: {
        "conversion_rate": {
            "warning": 2.0,  # En dessous de 2%
            "critical": 1.0,  # En dessous de 1%
            "unit": "percent"
        },
        "revenue_drop": {
            "warning": 15.0,  # Baisse de 15%
            "critical": 30.0,  # Baisse de 30%
            "unit": "percent"
        },
        "user_churn": {
            "warning": 5.0,
            "critical": 10.0,
            "unit": "percent"
        }
    })
    
    # Configuration des fenêtres temporelles
    time_windows: Dict[str, int] = field(default_factory=lambda: {
        "short_term": 300,    # 5 minutes
        "medium_term": 1800,  # 30 minutes
        "long_term": 3600     # 1 heure
    })

@dataclass
class IntegrationConfig:
    """Configuration des intégrations"""
    # Intégrations de monitoring
    prometheus: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "endpoint": "http://prometheus:9090",
        "scrape_interval": "30s",
        "retention": "15d",
        "external_labels": {
            "environment": "${ENVIRONMENT}",
            "cluster": "${CLUSTER_NAME}"
        }
    })
    
    grafana: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "url": "http://grafana:3000",
        "api_key": "${GRAFANA_API_KEY}",
        "dashboards_path": "/dashboards",
        "auto_provisioning": True
    })
    
    elasticsearch: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "hosts": ["elasticsearch:9200"],
        "index_prefix": "incidents",
        "retention_days": 90,
        "authentication": {
            "username": "${ES_USERNAME}",
            "password": "${ES_PASSWORD}"
        }
    })
    
    # Intégrations d'alerting
    pagerduty: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "integration_key": "${PAGERDUTY_INTEGRATION_KEY}",
        "severity_mapping": {
            "critical": "critical",
            "high": "error",
            "medium": "warning",
            "low": "info"
        }
    })
    
    opsgenie: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "api_key": "${OPSGENIE_API_KEY}",
        "team": "incident-response",
        "priority_mapping": {
            "critical": "P1",
            "high": "P2",
            "medium": "P3",
            "low": "P4"
        }
    })
    
    # Intégrations de communication
    slack: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "bot_token": "${SLACK_BOT_TOKEN}",
        "channels": {
            "incidents": "#incidents",
            "critical": "#critical-alerts",
            "security": "#security-alerts"
        },
        "user_groups": {
            "oncall": "@oncall",
            "security": "@security-team",
            "management": "@management"
        }
    })
    
    # Intégrations SIEM/Security
    splunk: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "host": "splunk.company.com",
        "port": 8089,
        "index": "security_incidents",
        "authentication": {
            "username": "${SPLUNK_USERNAME}",
            "password": "${SPLUNK_PASSWORD}"
        }
    })

@dataclass
class SecurityConfig:
    """Configuration de sécurité"""
    # Chiffrement
    encryption: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "algorithm": "AES-256-GCM",
        "key_rotation_days": 90,
        "key_management": "vault"
    })
    
    # Authentification
    authentication: Dict[str, Any] = field(default_factory=lambda: {
        "method": "oauth2",
        "provider": "keycloak",
        "token_expiry": 3600,
        "refresh_enabled": True,
        "mfa_required": True
    })
    
    # Autorisation
    authorization: Dict[str, Any] = field(default_factory=lambda: {
        "rbac_enabled": True,
        "default_role": "viewer",
        "admin_approval_required": True,
        "audit_all_actions": True
    })
    
    # Politiques de sécurité
    security_policies: Dict[str, Any] = field(default_factory=lambda: {
        "password_policy": {
            "min_length": 12,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_special": True,
            "history_count": 5
        },
        "session_policy": {
            "max_concurrent_sessions": 3,
            "idle_timeout": 1800,
            "absolute_timeout": 28800
        },
        "api_policy": {
            "rate_limiting": True,
            "max_requests_per_minute": 1000,
            "ip_whitelist_enabled": False,
            "api_key_required": True
        }
    })

@dataclass
class ComplianceConfig:
    """Configuration de conformité"""
    # Réglementations
    regulations: Dict[str, bool] = field(default_factory=lambda: {
        "gdpr": True,
        "sox": True,
        "pci_dss": False,
        "hipaa": False,
        "iso27001": True
    })
    
    # Audit et logging
    audit: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "log_all_actions": True,
        "retention_years": 7,
        "encryption_required": True,
        "immutable_logs": True,
        "real_time_monitoring": True
    })
    
    # Rapports de conformité
    reporting: Dict[str, Any] = field(default_factory=lambda: {
        "automated_reports": True,
        "report_frequency": "monthly",
        "recipients": ["compliance@company.com"],
        "formats": ["pdf", "json"],
        "include_metrics": True
    })
    
    # Politiques de rétention
    data_retention: Dict[str, Any] = field(default_factory=lambda: {
        "incident_data": {
            "retention_days": 2555,  # 7 ans
            "archive_after_days": 365,
            "deletion_policy": "secure_wipe"
        },
        "metrics_data": {
            "raw_data_days": 90,
            "aggregated_data_years": 2,
            "compression_enabled": True
        },
        "audit_logs": {
            "retention_years": 10,
            "immutable": True,
            "backup_required": True
        }
    })

@dataclass
class AdvancedConfiguration:
    """Configuration principale du système"""
    environment: Environment = Environment.DEVELOPMENT
    version: str = "2.0.0"
    debug_mode: bool = True
    
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    integrations: IntegrationConfig = field(default_factory=IntegrationConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    
    # Configuration avancée
    advanced_features: Dict[str, bool] = field(default_factory=lambda: {
        "ml_anomaly_detection": True,
        "predictive_analytics": True,
        "auto_remediation": True,
        "intelligent_escalation": True,
        "cross_metric_correlation": True,
        "real_time_dashboards": True,
        "mobile_notifications": False,
        "ai_root_cause_analysis": True
    })
    
    # Limites et quotas
    limits: Dict[str, int] = field(default_factory=lambda: {
        "max_incidents_per_hour": 1000,
        "max_metrics_per_second": 10000,
        "max_concurrent_investigations": 50,
        "max_escalation_levels": 5,
        "max_notification_rate": 100
    })
    
    # Configuration de performance
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "caching": {
            "enabled": True,
            "ttl_seconds": 300,
            "max_size_mb": 1024
        },
        "database": {
            "connection_pool_size": 20,
            "max_overflow": 30,
            "pool_timeout": 30,
            "pool_recycle": 3600
        },
        "async_processing": {
            "enabled": True,
            "worker_count": 10,
            "queue_size": 10000
        }
    })

class ConfigurationManager:
    """Gestionnaire de configuration avancé"""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.config = AdvancedConfiguration(environment=environment)
        self.config_cache = {}
        self.watchers = []
        
    def load_from_file(self, file_path: str) -> None:
        """Chargement de la configuration depuis un fichier"""
        try:
            with open(file_path, 'r') as file:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    config_data = yaml.safe_load(file)
                elif file_path.endswith('.json'):
                    config_data = json.load(file)
                else:
                    raise ValueError(f"Format de fichier non supporté: {file_path}")
                
                self._update_config_from_dict(config_data)
                logger.info(f"Configuration chargée depuis {file_path}")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise
    
    def save_to_file(self, file_path: str) -> None:
        """Sauvegarde de la configuration vers un fichier"""
        try:
            config_dict = asdict(self.config)
            
            with open(file_path, 'w') as file:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.safe_dump(config_dict, file, default_flow_style=False, indent=2)
                elif file_path.endswith('.json'):
                    json.dump(config_dict, file, indent=2, default=str)
                else:
                    raise ValueError(f"Format de fichier non supporté: {file_path}")
                
                logger.info(f"Configuration sauvegardée vers {file_path}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration: {e}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Récupération d'une valeur de configuration par chemin"""
        keys = key_path.split('.')
        current = asdict(self.config)
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, key_path: str, value: Any) -> None:
        """Définition d'une valeur de configuration par chemin"""
        keys = key_path.split('.')
        config_dict = asdict(self.config)
        current = config_dict
        
        # Navigation jusqu'à l'avant-dernier niveau
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Définition de la valeur finale
        current[keys[-1]] = value
        
        # Mise à jour de la configuration
        self._update_config_from_dict(config_dict)
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Mise à jour de la configuration depuis un dictionnaire"""
        # Reconstruction de l'objet de configuration
        # Implémentation simplifiée - en production, utiliser un système plus robuste
        
        if 'environment' in config_dict:
            self.config.environment = Environment(config_dict['environment'])
        
        if 'monitoring' in config_dict:
            self.config.monitoring = MonitoringConfig(**config_dict['monitoring'])
        
        if 'alerting' in config_dict:
            self.config.alerting = AlertingConfig(**config_dict['alerting'])
        
        # ... (autres sections de configuration)
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validation de la configuration"""
        errors = {}
        warnings = []
        
        # Validation des seuils
        for metric, thresholds in self.config.thresholds.system_thresholds.items():
            if thresholds.get('warning', 0) >= thresholds.get('critical', 100):
                if metric not in errors:
                    errors[metric] = []
                errors[metric].append("Warning threshold must be less than critical threshold")
        
        # Validation des intégrations
        if self.config.integrations.prometheus['enabled'] and not self.config.integrations.prometheus.get('endpoint'):
            errors['prometheus'] = ['Endpoint required when Prometheus is enabled']
        
        # Validation de sécurité
        if self.config.environment == Environment.PRODUCTION:
            if self.config.debug_mode:
                warnings.append("Debug mode should be disabled in production")
            if not self.config.security.encryption['enabled']:
                errors['security'] = ['Encryption must be enabled in production']
        
        return {
            'errors': errors,
            'warnings': warnings,
            'valid': len(errors) == 0
        }
    
    def get_environment_specific_config(self) -> Dict[str, Any]:
        """Récupération de la configuration spécifique à l'environnement"""
        base_config = asdict(self.config)
        
        # Ajustements spécifiques à l'environnement
        if self.environment == Environment.PRODUCTION:
            base_config['debug_mode'] = False
            base_config['monitoring']['collection_interval'] = 30
            base_config['security']['encryption']['enabled'] = True
            
        elif self.environment == Environment.DEVELOPMENT:
            base_config['debug_mode'] = True
            base_config['monitoring']['collection_interval'] = 60
            base_config['alerting']['notification_channels']['email']['enabled'] = False
        
        elif self.environment == Environment.STAGING:
            base_config['debug_mode'] = False
            base_config['monitoring']['collection_interval'] = 45
        
        return base_config
    
    def export_for_deployment(self, format: str = 'yaml') -> str:
        """Export de la configuration pour déploiement"""
        config = self.get_environment_specific_config()
        
        # Résolution des variables d'environnement
        config = self._resolve_environment_variables(config)
        
        if format == 'yaml':
            return yaml.safe_dump(config, default_flow_style=False, indent=2)
        elif format == 'json':
            return json.dumps(config, indent=2, default=str)
        else:
            raise ValueError(f"Format non supporté: {format}")
    
    def _resolve_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Résolution des variables d'environnement dans la configuration"""
        def resolve_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value
        
        return resolve_value(config)

# Instances par défaut
dev_config = ConfigurationManager(Environment.DEVELOPMENT)
staging_config = ConfigurationManager(Environment.STAGING)
prod_config = ConfigurationManager(Environment.PRODUCTION)

# Configuration par défaut pour l'environnement de développement
DEFAULT_DEV_CONFIG = """
environment: development
version: "2.0.0"
debug_mode: true

monitoring:
  enabled: true
  collection_interval: 60
  retention_period: 30
  real_time_streaming: true
  batch_size: 1000

alerting:
  enabled: true
  default_severity: "medium"
  escalation_enabled: true
  auto_resolution: true

thresholds:
  system_thresholds:
    cpu_usage:
      warning: 70.0
      critical: 90.0
      unit: "percent"
    memory_usage:
      warning: 80.0
      critical: 95.0
      unit: "percent"
    response_time:
      warning: 1000.0
      critical: 5000.0
      unit: "milliseconds"

security:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
  authentication:
    method: "oauth2"
    mfa_required: false

advanced_features:
  ml_anomaly_detection: true
  predictive_analytics: true
  auto_remediation: true
  intelligent_escalation: true
"""
