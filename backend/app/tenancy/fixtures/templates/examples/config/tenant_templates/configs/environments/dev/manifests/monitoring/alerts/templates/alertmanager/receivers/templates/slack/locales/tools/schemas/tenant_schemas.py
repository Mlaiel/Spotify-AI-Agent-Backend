"""
Schémas Pydantic avancés pour la configuration des tenants.

Ce module définit tous les schémas pour la configuration multi-tenant,
ressources, quotas, sécurité et isolation des données.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, IPvAnyAddress
from decimal import Decimal
import re


class TenantStatus(str, Enum):
    """États possibles d'un tenant."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    MAINTENANCE = "maintenance"
    ARCHIVED = "archived"


class TenantTier(str, Enum):
    """Niveaux de service tenant."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class EnvironmentType(str, Enum):
    """Types d'environnements."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    SANDBOX = "sandbox"


class ResourceType(str, Enum):
    """Types de ressources."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"


class StorageType(str, Enum):
    """Types de stockage."""
    SSD = "ssd"
    HDD = "hdd"
    NVME = "nvme"
    NETWORK = "network"
    OBJECT = "object"
    BLOCK = "block"


class NetworkProtocol(str, Enum):
    """Protocoles réseau."""
    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"
    WEBSOCKET = "websocket"


class BackupStrategy(str, Enum):
    """Stratégies de sauvegarde."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"
    CONTINUOUS = "continuous"


class TenantResourceSchema(BaseModel):
    """Schéma pour les ressources d'un tenant."""
    type: ResourceType = Field(..., description="Type de ressource")
    name: str = Field(..., description="Nom de la ressource")
    capacity: Union[int, float, str] = Field(..., description="Capacité allouée")
    unit: str = Field(..., description="Unité de mesure")
    used: Union[int, float] = Field(0, description="Ressource utilisée")
    reserved: Union[int, float] = Field(0, description="Ressource réservée")
    available: Union[int, float] = Field(0, description="Ressource disponible")
    cost_per_unit: Optional[Decimal] = Field(None, description="Coût par unité")
    billing_cycle: str = Field("monthly", description="Cycle de facturation")
    auto_scaling: bool = Field(False, description="Auto-scaling activé")
    scaling_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration auto-scaling")
    
    class Config:
        use_enum_values = True


class TenantLimitSchema(BaseModel):
    """Schéma pour les limites d'un tenant."""
    resource_type: ResourceType = Field(..., description="Type de ressource")
    soft_limit: Union[int, float] = Field(..., description="Limite souple")
    hard_limit: Union[int, float] = Field(..., description="Limite dure")
    burst_limit: Optional[Union[int, float]] = Field(None, description="Limite de pic")
    reset_period: str = Field("1h", description="Période de reset")
    warning_threshold: float = Field(0.8, description="Seuil d'alerte (80%)")
    critical_threshold: float = Field(0.95, description="Seuil critique (95%)")
    enforcement: bool = Field(True, description="Application de la limite")
    grace_period: str = Field("5m", description="Période de grâce")
    
    class Config:
        use_enum_values = True
    
    @validator('warning_threshold', 'critical_threshold')
    def validate_thresholds(cls, v):
        """Valide les seuils."""
        if not 0 <= v <= 1:
            raise ValueError('Les seuils doivent être entre 0 et 1')
        return v


class TenantQuotaSchema(BaseModel):
    """Schéma pour les quotas d'un tenant."""
    name: str = Field(..., description="Nom du quota")
    description: Optional[str] = Field(None, description="Description du quota")
    value: Union[int, float] = Field(..., description="Valeur du quota")
    unit: str = Field(..., description="Unité du quota")
    period: str = Field("1d", description="Période du quota")
    consumed: Union[int, float] = Field(0, description="Quota consommé")
    remaining: Union[int, float] = Field(0, description="Quota restant")
    reset_at: Optional[datetime] = Field(None, description="Prochaine remise à zéro")
    last_reset: Optional[datetime] = Field(None, description="Dernière remise à zéro")
    auto_renew: bool = Field(True, description="Renouvellement automatique")
    notifications: List[float] = Field([0.5, 0.8, 0.95], description="Seuils de notification")


class TenantSecuritySchema(BaseModel):
    """Schéma pour la sécurité d'un tenant."""
    encryption_at_rest: bool = Field(True, description="Chiffrement au repos")
    encryption_in_transit: bool = Field(True, description="Chiffrement en transit")
    encryption_algorithm: str = Field("AES-256", description="Algorithme de chiffrement")
    key_management: Dict[str, Any] = Field(default_factory=dict, description="Gestion des clés")
    
    authentication_methods: List[str] = Field(
        default_factory=lambda: ["password", "mfa"],
        description="Méthodes d'authentification"
    )
    password_policy: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_length": 12,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_symbols": True,
            "max_age_days": 90,
            "history_count": 5
        },
        description="Politique de mot de passe"
    )
    
    session_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "timeout_minutes": 60,
            "max_concurrent": 5,
            "require_fresh_login": False,
            "secure_cookies": True
        },
        description="Configuration des sessions"
    )
    
    ip_whitelist: List[str] = Field(default_factory=list, description="Liste blanche IP")
    ip_blacklist: List[str] = Field(default_factory=list, description="Liste noire IP")
    rate_limiting: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "requests_per_minute": 1000,
            "burst_size": 100,
            "ban_duration": "1h"
        },
        description="Limitation de débit"
    )
    
    audit_logging: bool = Field(True, description="Journalisation d'audit")
    compliance_standards: List[str] = Field(
        default_factory=lambda: ["SOC2", "GDPR"],
        description="Standards de conformité"
    )
    data_retention_days: int = Field(365, description="Rétention des données en jours")
    anonymization_enabled: bool = Field(True, description="Anonymisation activée")


class TenantNetworkSchema(BaseModel):
    """Schéma pour la configuration réseau d'un tenant."""
    vpc_id: Optional[str] = Field(None, description="ID du VPC")
    subnet_ids: List[str] = Field(default_factory=list, description="IDs des sous-réseaux")
    security_group_ids: List[str] = Field(default_factory=list, description="IDs des groupes de sécurité")
    
    allowed_protocols: List[NetworkProtocol] = Field(
        default_factory=lambda: [NetworkProtocol.HTTPS, NetworkProtocol.GRPC],
        description="Protocoles autorisés"
    )
    
    ingress_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Règles d'entrée")
    egress_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Règles de sortie")
    
    load_balancer_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration load balancer")
    cdn_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration CDN")
    
    bandwidth_limit: Optional[str] = Field(None, description="Limite de bande passante")
    connection_limit: int = Field(1000, description="Limite de connexions")
    timeout_config: Dict[str, int] = Field(
        default_factory=lambda: {
            "connect_timeout": 10,
            "read_timeout": 30,
            "write_timeout": 30,
            "idle_timeout": 300
        },
        description="Configuration des timeouts"
    )
    
    class Config:
        use_enum_values = True


class TenantStorageSchema(BaseModel):
    """Schéma pour le stockage d'un tenant."""
    type: StorageType = Field(..., description="Type de stockage")
    capacity: str = Field(..., description="Capacité totale")
    used: str = Field("0GB", description="Espace utilisé")
    available: str = Field(..., description="Espace disponible")
    
    encryption_enabled: bool = Field(True, description="Chiffrement activé")
    compression_enabled: bool = Field(True, description="Compression activée")
    deduplication_enabled: bool = Field(True, description="Déduplication activée")
    
    backup_enabled: bool = Field(True, description="Sauvegarde activée")
    backup_strategy: BackupStrategy = Field(BackupStrategy.INCREMENTAL, description="Stratégie de sauvegarde")
    backup_schedule: str = Field("0 2 * * *", description="Planning de sauvegarde (cron)")
    backup_retention_days: int = Field(30, description="Rétention des sauvegardes")
    
    replication_enabled: bool = Field(False, description="Réplication activée")
    replication_factor: int = Field(1, description="Facteur de réplication")
    replication_zones: List[str] = Field(default_factory=list, description="Zones de réplication")
    
    performance_tier: str = Field("standard", description="Niveau de performance")
    iops_limit: Optional[int] = Field(None, description="Limite IOPS")
    throughput_limit: Optional[str] = Field(None, description="Limite de débit")
    
    class Config:
        use_enum_values = True


class TenantComputeSchema(BaseModel):
    """Schéma pour les ressources de calcul d'un tenant."""
    cpu_cores: int = Field(..., description="Nombre de cœurs CPU")
    cpu_frequency: Optional[str] = Field(None, description="Fréquence CPU")
    memory_gb: int = Field(..., description="Mémoire en GB")
    memory_type: str = Field("DDR4", description="Type de mémoire")
    
    container_limits: Dict[str, str] = Field(
        default_factory=lambda: {
            "max_containers": "10",
            "max_cpu_per_container": "2",
            "max_memory_per_container": "4Gi"
        },
        description="Limites des conteneurs"
    )
    
    kubernetes_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "namespace": "",
            "resource_quotas": {},
            "network_policies": [],
            "pod_security_policies": []
        },
        description="Configuration Kubernetes"
    )
    
    auto_scaling: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,
            "min_replicas": 1,
            "max_replicas": 10,
            "target_cpu_utilization": 70,
            "target_memory_utilization": 80
        },
        description="Configuration auto-scaling"
    )
    
    health_checks: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "interval": "30s",
            "timeout": "5s",
            "retries": 3,
            "start_period": "60s"
        },
        description="Configuration health checks"
    )


class TenantMonitoringSchema(BaseModel):
    """Schéma pour le monitoring d'un tenant."""
    metrics_retention_days: int = Field(30, description="Rétention des métriques")
    logs_retention_days: int = Field(7, description="Rétention des logs")
    traces_retention_days: int = Field(3, description="Rétention des traces")
    
    custom_dashboards: List[str] = Field(default_factory=list, description="Dashboards personnalisés")
    alert_rules: List[str] = Field(default_factory=list, description="Règles d'alerte")
    notification_channels: List[str] = Field(default_factory=list, description="Canaux de notification")
    
    sampling_rate: float = Field(0.1, description="Taux d'échantillonnage")
    metrics_interval: str = Field("30s", description="Intervalle des métriques")
    health_check_interval: str = Field("10s", description="Intervalle health check")
    
    sla_targets: Dict[str, float] = Field(
        default_factory=lambda: {
            "availability": 99.9,
            "response_time_p95": 500.0,
            "error_rate": 0.1
        },
        description="Objectifs SLA"
    )


class TenantLoggingSchema(BaseModel):
    """Schéma pour la configuration de logging d'un tenant."""
    log_level: str = Field("INFO", description="Niveau de log")
    log_format: str = Field("json", description="Format des logs")
    
    destinations: List[str] = Field(
        default_factory=lambda: ["stdout", "file"],
        description="Destinations des logs"
    )
    
    structured_logging: bool = Field(True, description="Logs structurés")
    sensitive_data_masking: bool = Field(True, description="Masquage des données sensibles")
    
    retention_policy: Dict[str, str] = Field(
        default_factory=lambda: {
            "debug": "1d",
            "info": "7d",
            "warning": "30d",
            "error": "90d",
            "critical": "365d"
        },
        description="Politique de rétention par niveau"
    )
    
    aggregation_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Règles d'agrégation"
    )


class TenantBackupSchema(BaseModel):
    """Schéma pour la configuration de sauvegarde d'un tenant."""
    enabled: bool = Field(True, description="Sauvegarde activée")
    strategy: BackupStrategy = Field(BackupStrategy.INCREMENTAL, description="Stratégie de sauvegarde")
    
    schedule: Dict[str, str] = Field(
        default_factory=lambda: {
            "full": "0 2 * * 0",  # Dimanche 2h
            "incremental": "0 2 * * 1-6",  # Lundi-Samedi 2h
            "differential": "0 14 * * *"  # Quotidien 14h
        },
        description="Planning de sauvegarde"
    )
    
    retention: Dict[str, str] = Field(
        default_factory=lambda: {
            "daily": "7d",
            "weekly": "4w",
            "monthly": "12m",
            "yearly": "7y"
        },
        description="Rétention des sauvegardes"
    )
    
    encryption: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "algorithm": "AES-256",
            "key_rotation_days": 90
        },
        description="Configuration de chiffrement"
    )
    
    compression: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "algorithm": "gzip",
            "level": 6
        },
        description="Configuration de compression"
    )
    
    verification: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "checksum": "sha256",
            "restore_test_frequency": "monthly"
        },
        description="Vérification des sauvegardes"
    )
    
    destinations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Destinations de sauvegarde"
    )
    
    class Config:
        use_enum_values = True


class TenantEnvironmentSchema(BaseModel):
    """Schéma pour un environnement de tenant."""
    name: str = Field(..., description="Nom de l'environnement")
    type: EnvironmentType = Field(..., description="Type d'environnement")
    description: Optional[str] = Field(None, description="Description")
    
    resources: List[TenantResourceSchema] = Field(..., description="Ressources allouées")
    limits: List[TenantLimitSchema] = Field(..., description="Limites de ressources")
    quotas: List[TenantQuotaSchema] = Field(..., description="Quotas configurés")
    
    security: TenantSecuritySchema = Field(..., description="Configuration sécurité")
    network: TenantNetworkSchema = Field(..., description="Configuration réseau")
    storage: TenantStorageSchema = Field(..., description="Configuration stockage")
    compute: TenantComputeSchema = Field(..., description="Configuration calcul")
    monitoring: TenantMonitoringSchema = Field(..., description="Configuration monitoring")
    logging: TenantLoggingSchema = Field(..., description="Configuration logging")
    backup: TenantBackupSchema = Field(..., description="Configuration sauvegarde")
    
    variables: Dict[str, str] = Field(default_factory=dict, description="Variables d'environnement")
    secrets: List[str] = Field(default_factory=list, description="Secrets configurés")
    
    deployment_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration de déploiement")
    feature_flags: Dict[str, bool] = Field(default_factory=dict, description="Feature flags")
    
    created_at: Optional[datetime] = Field(None, description="Date de création")
    updated_at: Optional[datetime] = Field(None, description="Date de mise à jour")
    
    class Config:
        use_enum_values = True


class TenantConfigSchema(BaseModel):
    """Schéma principal de configuration d'un tenant."""
    id: str = Field(..., description="Identifiant unique du tenant")
    name: str = Field(..., description="Nom du tenant")
    display_name: str = Field(..., description="Nom affiché")
    description: Optional[str] = Field(None, description="Description du tenant")
    
    status: TenantStatus = Field(TenantStatus.ACTIVE, description="État du tenant")
    tier: TenantTier = Field(TenantTier.BASIC, description="Niveau de service")
    
    organization: Dict[str, str] = Field(..., description="Informations organisation")
    contact: Dict[str, str] = Field(..., description="Informations de contact")
    billing: Dict[str, Any] = Field(default_factory=dict, description="Informations de facturation")
    
    environments: List[TenantEnvironmentSchema] = Field(..., description="Environnements configurés")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées personnalisées")
    tags: List[str] = Field(default_factory=list, description="Tags de classification")
    
    created_at: Optional[datetime] = Field(None, description="Date de création")
    updated_at: Optional[datetime] = Field(None, description="Date de mise à jour")
    expires_at: Optional[datetime] = Field(None, description="Date d'expiration")
    
    class Config:
        use_enum_values = True
    
    @validator('id')
    def validate_tenant_id(cls, v):
        """Valide l'ID du tenant."""
        if not re.match(r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?$', v):
            raise ValueError('ID tenant invalide. Utilisez des lettres minuscules, chiffres et tirets.')
        return v
