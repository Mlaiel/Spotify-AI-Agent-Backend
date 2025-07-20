"""
Générateur de schémas avancés pour l'architecture multi-tenant.

Ce module fournit des utilitaires pour générer automatiquement des configurations
tenant avancées avec validation Pydantic et templates personnalisables.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SubscriptionTier(str, Enum):
    """Niveaux d'abonnement supportés."""
    FREE = "free"
    PREMIUM = "premium" 
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"

class IsolationType(str, Enum):
    """Types d'isolation des données."""
    SCHEMA = "schema"
    DATABASE = "database"
    CLUSTER = "cluster"

class ComplianceMode(str, Enum):
    """Modes de compliance supportés."""
    NONE = "none"
    GDPR = "gdpr"
    SOX = "sox"
    HIPAA = "hipaa"
    PCI = "pci"

@dataclass
class DatabaseIsolationConfig:
    """Configuration d'isolation de base de données."""
    type: IsolationType = IsolationType.SCHEMA
    connection_pool_size: int = 20
    read_replicas: List[str] = field(default_factory=list)
    encryption_at_rest: bool = True
    backup_retention_days: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "connection_pool_size": self.connection_pool_size,
            "read_replicas": self.read_replicas,
            "encryption_at_rest": self.encryption_at_rest,
            "backup_retention_days": self.backup_retention_days
        }

@dataclass 
class CacheIsolationConfig:
    """Configuration d'isolation du cache Redis."""
    redis_namespace: str
    max_memory_mb: int = 512
    ttl_default_seconds: int = 3600
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "redis_namespace": self.redis_namespace,
            "max_memory_mb": self.max_memory_mb,
            "ttl_default_seconds": self.ttl_default_seconds
        }

@dataclass
class StorageIsolationConfig:
    """Configuration d'isolation du stockage."""
    s3_bucket_prefix: str
    max_storage_gb: int
    cdn_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "s3_bucket_prefix": self.s3_bucket_prefix,
            "max_storage_gb": self.max_storage_gb,
            "cdn_enabled": self.cdn_enabled
        }

@dataclass
class IsolationConfig:
    """Configuration complète d'isolation."""
    database_isolation: DatabaseIsolationConfig
    cache_isolation: CacheIsolationConfig
    storage_isolation: StorageIsolationConfig
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "database_isolation": self.database_isolation.to_dict(),
            "cache_isolation": self.cache_isolation.to_dict(),
            "storage_isolation": self.storage_isolation.to_dict()
        }

@dataclass
class MonitoringConfig:
    """Configuration du monitoring Prometheus/Grafana."""
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    retention_days: int = 90
    scrape_interval_seconds: int = 30
    custom_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prometheus_enabled": self.prometheus_enabled,
            "grafana_enabled": self.grafana_enabled,
            "retention_days": self.retention_days,
            "scrape_interval_seconds": self.scrape_interval_seconds,
            "custom_metrics": self.custom_metrics
        }

@dataclass
class SlackChannel:
    """Configuration d'un canal Slack."""
    channel: str
    severity_levels: List[str]
    webhook_url: str

@dataclass
class AlertConfig:
    """Configuration des alertes."""
    slack_enabled: bool = True
    slack_channels: List[SlackChannel] = field(default_factory=list)
    email_notifications_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    pagerduty_enabled: bool = False
    pagerduty_integration_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "slack_enabled": self.slack_enabled,
            "slack_channels": [
                {
                    "channel": ch.channel,
                    "severity_levels": ch.severity_levels,
                    "webhook_url": ch.webhook_url
                } for ch in self.slack_channels
            ],
            "email_notifications": {
                "enabled": self.email_notifications_enabled,
                "recipients": self.email_recipients
            },
            "pagerduty_integration": {
                "enabled": self.pagerduty_enabled,
                "integration_key": self.pagerduty_integration_key
            }
        }

@dataclass
class SecurityConfig:
    """Configuration de sécurité."""
    encryption_key_rotation_days: int = 90
    audit_logs_enabled: bool = True
    compliance_mode: ComplianceMode = ComplianceMode.GDPR
    session_timeout_minutes: int = 60
    mfa_required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "encryption_key_rotation_days": self.encryption_key_rotation_days,
            "audit_logs_enabled": self.audit_logs_enabled,
            "compliance_mode": self.compliance_mode.value,
            "session_timeout_minutes": self.session_timeout_minutes,
            "mfa_required": self.mfa_required
        }

@dataclass
class FeatureFlags:
    """Gestion des fonctionnalités par tenant."""
    ai_features_enabled: bool = True
    advanced_analytics: bool = False
    real_time_collaboration: bool = False
    api_rate_limiting_enabled: bool = True
    requests_per_minute: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ai_features_enabled": self.ai_features_enabled,
            "advanced_analytics": self.advanced_analytics,
            "real_time_collaboration": self.real_time_collaboration,
            "api_rate_limiting": {
                "enabled": self.api_rate_limiting_enabled,
                "requests_per_minute": self.requests_per_minute
            }
        }

@dataclass
class TenantMetadata:
    """Métadonnées du tenant."""
    created_by: str
    environment: str
    version: str
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_by": self.created_by,
            "environment": self.environment,
            "version": self.version,
            "tags": self.tags
        }

@dataclass
class TenantConfig:
    """Configuration complète d'un tenant."""
    tenant_id: str
    tenant_name: str
    subscription_tier: SubscriptionTier
    isolation_config: IsolationConfig
    monitoring_config: MonitoringConfig
    alert_config: AlertConfig
    security_config: SecurityConfig = field(default_factory=SecurityConfig)
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    metadata: Optional[TenantMetadata] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        config = {
            "tenant_id": self.tenant_id,
            "tenant_name": self.tenant_name,
            "subscription_tier": self.subscription_tier.value,
            "isolation_config": self.isolation_config.to_dict(),
            "monitoring_config": self.monitoring_config.to_dict(),
            "alert_config": self.alert_config.to_dict(),
            "security_config": self.security_config.to_dict(),
            "feature_flags": self.feature_flags.to_dict(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if self.metadata:
            config["metadata"] = self.metadata.to_dict()
            
        return config
    
    def to_json(self, indent: int = 2) -> str:
        """Convertit la configuration en JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_yaml(self) -> str:
        """Convertit la configuration en YAML."""
        return yaml.dump(self.to_dict(), allow_unicode=True, default_flow_style=False)

class TenantConfigGenerator:
    """Générateur de configurations tenant avancées."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_premium_config(self, tenant_id: str, tenant_name: str) -> TenantConfig:
        """Génère une configuration Premium optimisée."""
        cache_config = CacheIsolationConfig(
            redis_namespace=f"{tenant_id}:cache",
            max_memory_mb=1024,
            ttl_default_seconds=3600
        )
        
        storage_config = StorageIsolationConfig(
            s3_bucket_prefix=tenant_id,
            max_storage_gb=100,
            cdn_enabled=True
        )
        
        db_config = DatabaseIsolationConfig(
            type=IsolationType.SCHEMA,
            connection_pool_size=30,
            encryption_at_rest=True,
            backup_retention_days=30
        )
        
        isolation_config = IsolationConfig(
            database_isolation=db_config,
            cache_isolation=cache_config,
            storage_isolation=storage_config
        )
        
        monitoring_config = MonitoringConfig(
            prometheus_enabled=True,
            grafana_enabled=True,
            retention_days=90,
            scrape_interval_seconds=30
        )
        
        slack_channel = SlackChannel(
            channel="#alerts-premium",
            severity_levels=["critical", "warning"],
            webhook_url="https://hooks.slack.com/services/premium"
        )
        
        alert_config = AlertConfig(
            slack_enabled=True,
            slack_channels=[slack_channel],
            email_notifications_enabled=True
        )
        
        security_config = SecurityConfig(
            compliance_mode=ComplianceMode.GDPR,
            mfa_required=True,
            audit_logs_enabled=True
        )
        
        feature_flags = FeatureFlags(
            ai_features_enabled=True,
            advanced_analytics=True,
            real_time_collaboration=True,
            requests_per_minute=5000
        )
        
        metadata = TenantMetadata(
            created_by="system",
            environment="prod",
            version="2.0.0",
            tags=["premium", "production"]
        )
        
        return TenantConfig(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            subscription_tier=SubscriptionTier.PREMIUM,
            isolation_config=isolation_config,
            monitoring_config=monitoring_config,
            alert_config=alert_config,
            security_config=security_config,
            feature_flags=feature_flags,
            metadata=metadata
        )
    
    def generate_enterprise_config(self, tenant_id: str, tenant_name: str) -> TenantConfig:
        """Génère une configuration Enterprise ultra-avancée."""
        cache_config = CacheIsolationConfig(
            redis_namespace=f"{tenant_id}:cache",
            max_memory_mb=4096,
            ttl_default_seconds=7200
        )
        
        storage_config = StorageIsolationConfig(
            s3_bucket_prefix=tenant_id,
            max_storage_gb=1000,
            cdn_enabled=True
        )
        
        db_config = DatabaseIsolationConfig(
            type=IsolationType.DATABASE,
            connection_pool_size=100,
            read_replicas=["replica-1", "replica-2"],
            encryption_at_rest=True,
            backup_retention_days=365
        )
        
        isolation_config = IsolationConfig(
            database_isolation=db_config,
            cache_isolation=cache_config,
            storage_isolation=storage_config
        )
        
        custom_metrics = [
            {
                "name": "tenant_api_requests_total",
                "type": "counter",
                "description": "Total API requests per tenant",
                "labels": ["tenant_id", "endpoint", "method"]
            },
            {
                "name": "tenant_ai_model_latency",
                "type": "histogram",
                "description": "AI model inference latency",
                "labels": ["tenant_id", "model_name", "version"]
            }
        ]
        
        monitoring_config = MonitoringConfig(
            prometheus_enabled=True,
            grafana_enabled=True,
            retention_days=365,
            scrape_interval_seconds=15,
            custom_metrics=custom_metrics
        )
        
        slack_channels = [
            SlackChannel("#alerts-enterprise", ["critical", "warning"], "https://hooks.slack.com/enterprise"),
            SlackChannel("#security-alerts", ["security"], "https://hooks.slack.com/security")
        ]
        
        alert_config = AlertConfig(
            slack_enabled=True,
            slack_channels=slack_channels,
            email_notifications_enabled=True,
            pagerduty_enabled=True,
            pagerduty_integration_key="enterprise-key"
        )
        
        security_config = SecurityConfig(
            compliance_mode=ComplianceMode.SOX,
            mfa_required=True,
            audit_logs_enabled=True,
            encryption_key_rotation_days=30,
            session_timeout_minutes=30
        )
        
        feature_flags = FeatureFlags(
            ai_features_enabled=True,
            advanced_analytics=True,
            real_time_collaboration=True,
            requests_per_minute=10000
        )
        
        metadata = TenantMetadata(
            created_by="enterprise-admin",
            environment="prod",
            version="2.0.0",
            tags=["enterprise", "production", "high-availability"]
        )
        
        return TenantConfig(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            subscription_tier=SubscriptionTier.ENTERPRISE,
            isolation_config=isolation_config,
            monitoring_config=monitoring_config,
            alert_config=alert_config,
            security_config=security_config,
            feature_flags=feature_flags,
            metadata=metadata
        )
    
    def generate_config_by_tier(self, tenant_id: str, tenant_name: str, tier: SubscriptionTier) -> TenantConfig:
        """Génère une configuration selon le tier d'abonnement."""
        if tier == SubscriptionTier.PREMIUM:
            return self.generate_premium_config(tenant_id, tenant_name)
        elif tier in [SubscriptionTier.ENTERPRISE, SubscriptionTier.ENTERPRISE_PLUS]:
            return self.generate_enterprise_config(tenant_id, tenant_name)
        else:
            # Configuration basique pour free tier
            return self._generate_free_config(tenant_id, tenant_name)
    
    def _generate_free_config(self, tenant_id: str, tenant_name: str) -> TenantConfig:
        """Génère une configuration basique pour le tier gratuit."""
        cache_config = CacheIsolationConfig(
            redis_namespace=f"{tenant_id}:cache",
            max_memory_mb=128,
            ttl_default_seconds=1800
        )
        
        storage_config = StorageIsolationConfig(
            s3_bucket_prefix=tenant_id,
            max_storage_gb=5,
            cdn_enabled=False
        )
        
        db_config = DatabaseIsolationConfig(
            type=IsolationType.SCHEMA,
            connection_pool_size=5,
            encryption_at_rest=False,
            backup_retention_days=7
        )
        
        isolation_config = IsolationConfig(
            database_isolation=db_config,
            cache_isolation=cache_config,
            storage_isolation=storage_config
        )
        
        monitoring_config = MonitoringConfig(
            prometheus_enabled=True,
            grafana_enabled=False,
            retention_days=30
        )
        
        alert_config = AlertConfig(
            slack_enabled=False,
            email_notifications_enabled=False
        )
        
        security_config = SecurityConfig(
            compliance_mode=ComplianceMode.NONE,
            mfa_required=False
        )
        
        feature_flags = FeatureFlags(
            ai_features_enabled=True,
            advanced_analytics=False,
            real_time_collaboration=False,
            requests_per_minute=100
        )
        
        return TenantConfig(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            subscription_tier=SubscriptionTier.FREE,
            isolation_config=isolation_config,
            monitoring_config=monitoring_config,
            alert_config=alert_config,
            security_config=security_config,
            feature_flags=feature_flags
        )
    
    def save_config(self, config: TenantConfig, output_path: Path, format: str = "json") -> bool:
        """Sauvegarde la configuration dans un fichier."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(config.to_json())
            elif format.lower() == "yaml":
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(config.to_yaml())
            else:
                raise ValueError(f"Format non supporté: {format}")
            
            self.logger.info(f"Configuration sauvegardée: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False

# Fonction utilitaire pour générer des exemples
def generate_example_configs() -> Dict[str, TenantConfig]:
    """Génère des configurations d'exemple pour tous les tiers."""
    generator = TenantConfigGenerator()
    
    configs = {
        "free": generator.generate_config_by_tier("demo-free", "Demo Free Tier", SubscriptionTier.FREE),
        "premium": generator.generate_config_by_tier("demo-premium", "Demo Premium", SubscriptionTier.PREMIUM),
        "enterprise": generator.generate_config_by_tier("demo-enterprise", "Demo Enterprise", SubscriptionTier.ENTERPRISE)
    }
    
    return configs

if __name__ == "__main__":
    # Génération d'exemples
    configs = generate_example_configs()
    generator = TenantConfigGenerator()
    
    for tier, config in configs.items():
        output_path = Path(f"examples/{tier}_tenant_config.json")
        generator.save_config(config, output_path)
        print(f"Configuration {tier} générée: {output_path}")
