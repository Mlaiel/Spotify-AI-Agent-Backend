"""
Schémas d'environnement - Module Python.

Ce module fournit les classes de validation pour la configuration
des environnements, infrastructure et déploiements.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class EnvironmentName(str, Enum):
    """Noms d'environnement."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class TierLevel(str, Enum):
    """Niveaux de tier."""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class InstanceType(str, Enum):
    """Types d'instances."""
    MICRO = "t3.micro"
    SMALL = "t3.small"
    MEDIUM = "t3.medium"
    LARGE = "t3.large"
    XLARGE = "t3.xlarge"


class LoadBalancerType(str, Enum):
    """Types de load balancer."""
    APPLICATION = "application"
    NETWORK = "network"
    CLASSIC = "classic"


class SubnetType(str, Enum):
    """Types de sous-réseaux."""
    PUBLIC = "public"
    PRIVATE = "private"
    ISOLATED = "isolated"


class DatabaseEngine(str, Enum):
    """Moteurs de base de données."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MARIADB = "mariadb"
    ORACLE = "oracle"
    MSSQL = "mssql"


class StorageType(str, Enum):
    """Types de stockage."""
    STANDARD = "standard"
    GP2 = "gp2"
    GP3 = "gp3"
    IO1 = "io1"
    IO2 = "io2"


class CacheEngine(str, Enum):
    """Moteurs de cache."""
    REDIS = "redis"
    MEMCACHED = "memcached"


class ObjectStorageProvider(str, Enum):
    """Fournisseurs de stockage objet."""
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    MINIO = "minio"


class BucketPurpose(str, Enum):
    """Objectifs des buckets."""
    UPLOADS = "uploads"
    BACKUPS = "backups"
    LOGS = "logs"
    STATIC = "static"
    MEDIA = "media"


class MessageQueueType(str, Enum):
    """Types de files de messages."""
    SQS = "sqs"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    REDIS = "redis"


class TopicType(str, Enum):
    """Types de topics."""
    SNS = "sns"
    KAFKA = "kafka"
    EVENTBRIDGE = "eventbridge"


class MonitoringProvider(str, Enum):
    """Fournisseurs de monitoring."""
    CLOUDWATCH = "cloudwatch"
    PROMETHEUS = "prometheus"
    DATADOG = "datadog"
    NEWRELIC = "newrelic"


class LogLevel(str, Enum):
    """Niveaux de log."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class NotificationChannelType(str, Enum):
    """Types de canaux de notification."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


class SeverityLevel(str, Enum):
    """Niveaux de sévérité."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FeatureFlagProvider(str, Enum):
    """Fournisseurs de feature flags."""
    LAUNCHDARKLY = "launchdarkly"
    SPLIT = "split"
    FLAGSMITH = "flagsmith"
    INTERNAL = "internal"


class ComplianceFramework(str, Enum):
    """Frameworks de conformité."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI = "pci"
    ISO27001 = "iso27001"


class EnvironmentInfo(BaseModel):
    """Informations sur l'environnement."""
    name: EnvironmentName
    description: Optional[str] = None
    tier: TierLevel
    region: str
    availability_zone: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)


class AutoScalingConfig(BaseModel):
    """Configuration d'auto-scaling."""
    enabled: bool = True
    cpu_threshold: float = Field(70.0, ge=0, le=100)
    memory_threshold: float = Field(80.0, ge=0, le=100)
    scale_up_cooldown: str = "5m"
    scale_down_cooldown: str = "10m"


class ComputeInstance(BaseModel):
    """Instance de calcul."""
    name: str
    type: str
    cpu: Optional[float] = None
    memory: Optional[str] = None
    storage: Optional[str] = None
    min_instances: int = Field(1, ge=1)
    max_instances: int = Field(3, ge=1)
    auto_scaling: AutoScalingConfig = Field(default_factory=AutoScalingConfig)

    @validator('max_instances')
    def validate_max_instances(cls, v, values):
        """Valide que max_instances >= min_instances."""
        if 'min_instances' in values and v < values['min_instances']:
            raise ValueError('max_instances doit être >= min_instances')
        return v


class HealthCheckConfig(BaseModel):
    """Configuration de health check."""
    path: str = "/health"
    interval: str = "30s"
    timeout: str = "5s"
    healthy_threshold: int = 2
    unhealthy_threshold: int = 5


class LoadBalancerConfig(BaseModel):
    """Configuration du load balancer."""
    enabled: bool = True
    type: LoadBalancerType = LoadBalancerType.APPLICATION
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)


class ComputeConfig(BaseModel):
    """Configuration de calcul."""
    instances: List[ComputeInstance] = Field(default_factory=list)
    load_balancer: LoadBalancerConfig = Field(default_factory=LoadBalancerConfig)


class VPCConfig(BaseModel):
    """Configuration VPC."""
    cidr: str = "10.0.0.0/16"
    enable_dns_hostnames: bool = True
    enable_dns_support: bool = True


class SubnetConfig(BaseModel):
    """Configuration de sous-réseau."""
    name: str
    type: SubnetType
    cidr: str
    availability_zone: Optional[str] = None


class SecurityRule(BaseModel):
    """Règle de sécurité."""
    protocol: str
    from_port: int
    to_port: int
    source: Optional[str] = None
    destination: Optional[str] = None


class SecurityGroupConfig(BaseModel):
    """Configuration de groupe de sécurité."""
    name: str
    description: str
    ingress_rules: List[SecurityRule] = Field(default_factory=list)
    egress_rules: List[SecurityRule] = Field(default_factory=list)


class NetworkingConfig(BaseModel):
    """Configuration réseau."""
    vpc: VPCConfig = Field(default_factory=VPCConfig)
    subnets: List[SubnetConfig] = Field(default_factory=list)
    security_groups: List[SecurityGroupConfig] = Field(default_factory=list)


class InfrastructureConfig(BaseModel):
    """Configuration d'infrastructure."""
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    networking: NetworkingConfig = Field(default_factory=NetworkingConfig)


class DatabaseBackupConfig(BaseModel):
    """Configuration de sauvegarde."""
    retention_period: int = Field(7, ge=0, le=35)
    backup_window: str = "03:00-04:00"
    maintenance_window: str = "sun:04:00-sun:05:00"
    point_in_time_recovery: bool = True


class DatabaseMonitoringConfig(BaseModel):
    """Configuration de monitoring base de données."""
    enabled: bool = True
    enhanced_monitoring: bool = False
    performance_insights: bool = False


class PrimaryDatabaseConfig(BaseModel):
    """Configuration de base de données principale."""
    engine: DatabaseEngine
    version: str
    instance_class: str
    allocated_storage: int = 20
    max_allocated_storage: Optional[int] = None
    storage_type: StorageType = StorageType.GP2
    iops: Optional[int] = None
    multi_az: bool = False
    backup: DatabaseBackupConfig = Field(default_factory=DatabaseBackupConfig)
    monitoring: DatabaseMonitoringConfig = Field(default_factory=DatabaseMonitoringConfig)


class ReadReplicaConfig(BaseModel):
    """Configuration de replica en lecture."""
    identifier: str
    instance_class: str
    region: Optional[str] = None
    availability_zone: Optional[str] = None


class ReplicationGroupConfig(BaseModel):
    """Configuration de groupe de réplication."""
    enabled: bool = False
    num_node_groups: int = 1
    replicas_per_node_group: int = 1


class CacheConfig(BaseModel):
    """Configuration de cache."""
    engine: CacheEngine = CacheEngine.REDIS
    version: str = "6.2"
    node_type: str = "cache.t3.micro"
    num_cache_nodes: int = Field(1, ge=1)
    replication_group: ReplicationGroupConfig = Field(default_factory=ReplicationGroupConfig)


class DatabasesConfig(BaseModel):
    """Configuration des bases de données."""
    primary: PrimaryDatabaseConfig
    read_replicas: List[ReadReplicaConfig] = Field(default_factory=list)
    cache: Optional[CacheConfig] = None


class LifecycleTransition(BaseModel):
    """Transition de cycle de vie."""
    days: int
    storage_class: str


class LifecycleExpiration(BaseModel):
    """Expiration de cycle de vie."""
    days: int


class LifecyclePolicyConfig(BaseModel):
    """Configuration de politique de cycle de vie."""
    transitions: List[LifecycleTransition] = Field(default_factory=list)
    expiration: Optional[LifecycleExpiration] = None


class BucketConfig(BaseModel):
    """Configuration de bucket."""
    name: str
    purpose: BucketPurpose
    storage_class: Optional[str] = None
    versioning: bool = False
    lifecycle_policy: Optional[LifecyclePolicyConfig] = None


class ObjectStorageConfig(BaseModel):
    """Configuration de stockage objet."""
    provider: ObjectStorageProvider = ObjectStorageProvider.S3
    buckets: List[BucketConfig] = Field(default_factory=list)


class VolumeConfig(BaseModel):
    """Configuration de volume."""
    name: str
    size: str
    type: StorageType = StorageType.GP2
    encrypted: bool = True
    snapshot_schedule: Optional[str] = None


class BlockStorageConfig(BaseModel):
    """Configuration de stockage bloc."""
    volumes: List[VolumeConfig] = Field(default_factory=list)


class StorageConfig(BaseModel):
    """Configuration de stockage."""
    object_storage: ObjectStorageConfig = Field(default_factory=ObjectStorageConfig)
    block_storage: BlockStorageConfig = Field(default_factory=BlockStorageConfig)


class QueueConfig(BaseModel):
    """Configuration de file de messages."""
    name: str
    type: MessageQueueType
    dlq_enabled: bool = True
    max_receive_count: int = 3
    visibility_timeout: str = "30s"
    message_retention: str = "14d"


class SubscriptionConfig(BaseModel):
    """Configuration d'abonnement."""
    protocol: str
    endpoint: str
    filter_policy: Dict[str, Any] = Field(default_factory=dict)


class TopicConfig(BaseModel):
    """Configuration de topic."""
    name: str
    type: TopicType
    subscriptions: List[SubscriptionConfig] = Field(default_factory=list)


class MessagingConfig(BaseModel):
    """Configuration de messagerie."""
    queues: List[QueueConfig] = Field(default_factory=list)
    topics: List[TopicConfig] = Field(default_factory=list)


class CustomMetricConfig(BaseModel):
    """Configuration de métrique personnalisée."""
    name: str
    namespace: str
    dimensions: Dict[str, str] = Field(default_factory=dict)


class MetricsConfig(BaseModel):
    """Configuration des métriques."""
    provider: MonitoringProvider = MonitoringProvider.CLOUDWATCH
    retention_days: int = 90
    custom_metrics: List[CustomMetricConfig] = Field(default_factory=list)


class LogGroupConfig(BaseModel):
    """Configuration de groupe de logs."""
    name: str
    retention_days: int = 30
    log_level: LogLevel = LogLevel.INFO


class LoggingConfig(BaseModel):
    """Configuration de logging."""
    provider: MonitoringProvider = MonitoringProvider.CLOUDWATCH
    log_groups: List[LogGroupConfig] = Field(default_factory=list)


class NotificationChannelConfig(BaseModel):
    """Configuration de canal de notification."""
    type: NotificationChannelType
    endpoint: str
    severity: List[SeverityLevel] = Field(default_factory=lambda: [SeverityLevel.CRITICAL])


class AlertingConfig(BaseModel):
    """Configuration d'alerting."""
    provider: MonitoringProvider = MonitoringProvider.CLOUDWATCH
    notification_channels: List[NotificationChannelConfig] = Field(default_factory=list)


class MonitoringConfig(BaseModel):
    """Configuration de monitoring."""
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)


class EncryptionConfig(BaseModel):
    """Configuration de chiffrement."""
    at_rest: bool = True
    in_transit: bool = True
    kms_key_id: Optional[str] = None


class IAMRoleConfig(BaseModel):
    """Configuration de rôle IAM."""
    name: str
    policies: List[str] = Field(default_factory=list)


class ServiceAccountConfig(BaseModel):
    """Configuration de compte de service."""
    name: str
    permissions: List[str] = Field(default_factory=list)


class AccessControlConfig(BaseModel):
    """Configuration de contrôle d'accès."""
    iam_roles: List[IAMRoleConfig] = Field(default_factory=list)
    service_accounts: List[ServiceAccountConfig] = Field(default_factory=list)


class NetworkSecurityConfig(BaseModel):
    """Configuration de sécurité réseau."""
    waf_enabled: bool = True
    ddos_protection: bool = True
    ip_whitelist: List[str] = Field(default_factory=list)


class SecurityConfig(BaseModel):
    """Configuration de sécurité."""
    encryption: EncryptionConfig = Field(default_factory=EncryptionConfig)
    access_control: AccessControlConfig = Field(default_factory=AccessControlConfig)
    network_security: NetworkSecurityConfig = Field(default_factory=NetworkSecurityConfig)


class FeatureFlagCondition(BaseModel):
    """Condition de feature flag."""
    pass  # Placeholder pour conditions complexes


class FeatureFlagConfig(BaseModel):
    """Configuration de feature flag."""
    name: str
    enabled: bool = False
    rollout_percentage: float = Field(0.0, ge=0, le=100)
    conditions: List[Dict[str, Any]] = Field(default_factory=list)


class FeatureFlagsConfig(BaseModel):
    """Configuration des feature flags."""
    provider: FeatureFlagProvider = FeatureFlagProvider.INTERNAL
    flags: List[FeatureFlagConfig] = Field(default_factory=list)


class DataResidencyConfig(BaseModel):
    """Configuration de résidence des données."""
    regions: List[str] = Field(default_factory=list)
    cross_border_transfer: bool = False


class AuditLoggingConfig(BaseModel):
    """Configuration de logging d'audit."""
    enabled: bool = True
    retention_years: int = Field(7, ge=1)


class ComplianceConfig(BaseModel):
    """Configuration de conformité."""
    frameworks: List[ComplianceFramework] = Field(default_factory=list)
    data_residency: DataResidencyConfig = Field(default_factory=DataResidencyConfig)
    audit_logging: AuditLoggingConfig = Field(default_factory=AuditLoggingConfig)


class EnvironmentConfigSchema(BaseModel):
    """Schéma complet de configuration d'environnement."""
    environment: EnvironmentInfo
    infrastructure: InfrastructureConfig
    databases: DatabasesConfig
    storage: StorageConfig
    messaging: MessagingConfig = Field(default_factory=MessagingConfig)
    monitoring: MonitoringConfig
    security: SecurityConfig
    feature_flags: FeatureFlagsConfig = Field(default_factory=FeatureFlagsConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)

    @validator('databases')
    def validate_databases(cls, v):
        """Valide la configuration des bases de données."""
        if not v.primary:
            raise ValueError("Une base de données principale doit être configurée")
        return v

    @validator('security')
    def validate_security(cls, v, values):
        """Valide la configuration de sécurité selon l'environnement."""
        if 'environment' in values:
            env_name = values['environment'].name
            if env_name == EnvironmentName.PRODUCTION:
                if not v.encryption.at_rest:
                    raise ValueError("Le chiffrement au repos est requis en production")
                if not v.network_security.waf_enabled:
                    raise ValueError("WAF doit être activé en production")
        return v

    class Config:
        """Configuration Pydantic."""
        use_enum_values = True
        validate_assignment = True
