"""
Enumerations Module for Advanced Scripts System

This module defines all enumerations used across the scripts ecosystem
for consistent type definitions and value constraints.

Version: 3.0.0
Developed by Spotify AI Agent Team
Lead Developer & AI Architect: Fahed Mlaiel
"""

from enum import Enum, IntEnum, Flag, auto
from typing import List, Dict, Any

# ============================================================================
# System and Environment Enums
# ============================================================================

class EnvironmentType(Enum):
    """Types d'environnements supportés"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"
    SANDBOX = "sandbox"

class DeploymentMode(Enum):
    """Modes de déploiement"""
    STANDALONE = "standalone"
    CLUSTER = "cluster"
    MULTI_CLUSTER = "multi_cluster"
    EDGE = "edge"
    HYBRID = "hybrid"

class RuntimePlatform(Enum):
    """Plateformes d'exécution"""
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    OPENSHIFT = "openshift"
    DOCKER_SWARM = "docker_swarm"
    NOMAD = "nomad"
    BARE_METAL = "bare_metal"

# ============================================================================
# Cloud Provider Enums
# ============================================================================

class CloudProvider(Enum):
    """Fournisseurs de services cloud"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ORACLE_CLOUD = "oracle_cloud"
    IBM_CLOUD = "ibm_cloud"
    ALIBABA_CLOUD = "alibaba_cloud"
    DIGITAL_OCEAN = "digital_ocean"
    LINODE = "linode"
    VULTR = "vultr"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"
    MULTI_CLOUD = "multi_cloud"

class CloudRegion(Enum):
    """Régions cloud principales"""
    # AWS Regions
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    
    # Azure Regions
    EAST_US = "eastus"
    WEST_US_2 = "westus2"
    WEST_EUROPE = "westeurope"
    SOUTHEAST_ASIA = "southeastasia"
    
    # GCP Regions
    US_CENTRAL1 = "us-central1"
    EUROPE_WEST1 = "europe-west1"
    ASIA_NORTHEAST1 = "asia-northeast1"

class StorageClass(Enum):
    """Classes de stockage cloud"""
    # AWS S3
    STANDARD = "standard"
    STANDARD_IA = "standard_ia"
    ONE_ZONE_IA = "one_zone_ia"
    GLACIER = "glacier"
    DEEP_ARCHIVE = "deep_archive"
    
    # Azure Blob
    HOT = "hot"
    COOL = "cool"
    ARCHIVE = "archive"
    
    # GCP Storage
    MULTI_REGIONAL = "multi_regional"
    REGIONAL = "regional"
    NEARLINE = "nearline"
    COLDLINE = "coldline"

# ============================================================================
# Deployment and Operations Enums
# ============================================================================

class DeploymentStrategy(Enum):
    """Stratégies de déploiement"""
    BLUE_GREEN = "blue_green"
    ROLLING_UPDATE = "rolling_update"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"
    SHADOW = "shadow"
    FEATURE_TOGGLE = "feature_toggle"

class DeploymentPhase(Enum):
    """Phases de déploiement"""
    PLANNING = "planning"
    VALIDATION = "validation"
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETION = "completion"
    ROLLBACK = "rollback"
    CLEANUP = "cleanup"

class DeploymentStatus(Enum):
    """Statuts de déploiement"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"
    PARTIAL = "partial"

class HealthCheckType(Enum):
    """Types de vérifications de santé"""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"
    CUSTOM = "custom"

class ScalingDirection(Enum):
    """Direction du scaling"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ScalingTrigger(Enum):
    """Déclencheurs de scaling"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_DEPTH = "queue_depth"
    CUSTOM_METRIC = "custom_metric"
    SCHEDULED = "scheduled"
    MANUAL = "manual"

# ============================================================================
# Monitoring and Alerting Enums
# ============================================================================

class MetricType(Enum):
    """Types de métriques"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"
    PERCENTAGE = "percentage"

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertState(Enum):
    """États des alertes"""
    ACTIVE = "active"
    PENDING = "pending"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"

class MonitoringDataSource(Enum):
    """Sources de données de monitoring"""
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    INFLUXDB = "influxdb"
    ELASTICSEARCH = "elasticsearch"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    SPLUNK = "splunk"
    CUSTOM = "custom"

class NotificationChannel(Enum):
    """Canaux de notification"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    VICTOROPS = "victorops"
    TEAMS = "teams"
    DISCORD = "discord"

# ============================================================================
# Security Enums
# ============================================================================

class SecurityEventType(Enum):
    """Types d'événements de sécurité"""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    MALWARE_DETECTION = "malware_detection"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"
    CONFIGURATION_CHANGE = "configuration_change"
    CREDENTIAL_THEFT = "credential_theft"
    DDoS_ATTACK = "ddos_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    CSRF_ATTACK = "csrf_attack"

class ThreatLevel(Enum):
    """Niveaux de menace"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    SEVERE = "severe"

class SecurityFramework(Enum):
    """Frameworks de sécurité"""
    NIST = "nist"
    ISO_27001 = "iso_27001"
    SOC_2 = "soc_2"
    CIS = "cis"
    OWASP = "owasp"
    SANS = "sans"
    MITRE_ATT_CK = "mitre_att_ck"

class ComplianceStandard(Enum):
    """Standards de conformité"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOX = "sox"
    FISMA = "fisma"
    COPPA = "coppa"
    CCPA = "ccpa"

class EncryptionAlgorithm(Enum):
    """Algorithmes de chiffrement"""
    AES_128 = "aes_128"
    AES_256 = "aes_256"
    CHACHA20 = "chacha20"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECDSA = "ecdsa"
    ED25519 = "ed25519"

class HashAlgorithm(Enum):
    """Algorithmes de hachage"""
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    ARGON2 = "argon2"
    BCRYPT = "bcrypt"
    SCRYPT = "scrypt"

# ============================================================================
# Backup and Storage Enums
# ============================================================================

class BackupType(Enum):
    """Types de sauvegarde"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"
    CONTINUOUS = "continuous"
    ARCHIVE = "archive"

class BackupStatus(Enum):
    """Statuts de sauvegarde"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CORRUPTED = "corrupted"
    EXPIRED = "expired"

class CompressionAlgorithm(Enum):
    """Algorithmes de compression"""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    BROTLI = "brotli"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"

class StorageProtocol(Enum):
    """Protocoles de stockage"""
    LOCAL = "local"
    NFS = "nfs"
    CIFS = "cifs"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"
    FTP = "ftp"
    SFTP = "sftp"
    RSYNC = "rsync"

# ============================================================================
# Performance and Optimization Enums
# ============================================================================

class OptimizationTarget(Enum):
    """Objectifs d'optimisation"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    COST = "cost"
    AVAILABILITY = "availability"
    RELIABILITY = "reliability"
    SECURITY = "security"
    BALANCED = "balanced"

class ResourceType(Enum):
    """Types de ressources"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    CACHE = "cache"

class PerformanceMetric(Enum):
    """Métriques de performance"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    RESOURCE_UTILIZATION = "resource_utilization"
    QUEUE_TIME = "queue_time"
    CONNECTION_TIME = "connection_time"

class LoadTestType(Enum):
    """Types de tests de charge"""
    LOAD = "load"
    STRESS = "stress"
    SPIKE = "spike"
    VOLUME = "volume"
    ENDURANCE = "endurance"
    SCALABILITY = "scalability"

# ============================================================================
# Network and Communication Enums
# ============================================================================

class NetworkProtocol(Enum):
    """Protocoles réseau"""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    AMQP = "amqp"

class LoadBalancingAlgorithm(Enum):
    """Algorithmes de load balancing"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    IP_HASH = "ip_hash"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"

class ProxyType(Enum):
    """Types de proxy"""
    REVERSE = "reverse"
    FORWARD = "forward"
    TRANSPARENT = "transparent"
    SOCKS = "socks"
    HTTP = "http"

# ============================================================================
# Data and Database Enums
# ============================================================================

class DatabaseType(Enum):
    """Types de bases de données"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    CASSANDRA = "cassandra"
    INFLUXDB = "influxdb"
    SQLITE = "sqlite"

class DataFormat(Enum):
    """Formats de données"""
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"

class SerializationFormat(Enum):
    """Formats de sérialisation"""
    JSON = "json"
    YAML = "yaml"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"
    AVRO = "avro"

# ============================================================================
# Machine Learning Enums
# ============================================================================

class MLAlgorithm(Enum):
    """Algorithmes de machine learning"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    ISOLATION_FOREST = "isolation_forest"
    DBSCAN = "dbscan"
    K_MEANS = "k_means"
    LSTM = "lstm"
    TRANSFORMER = "transformer"

class MLTaskType(Enum):
    """Types de tâches ML"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    COMPUTER_VISION = "computer_vision"

class ModelStatus(Enum):
    """Statuts des modèles ML"""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"

# ============================================================================
# Logging and Audit Enums
# ============================================================================

class LogLevel(Enum):
    """Niveaux de log"""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"

class AuditEventType(Enum):
    """Types d'événements d'audit"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHANGE = "permission_change"
    CONFIGURATION_CHANGE = "configuration_change"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    BACKUP_OPERATION = "backup_operation"
    RESTORE_OPERATION = "restore_operation"

class LogFormat(Enum):
    """Formats de log"""
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    SYSLOG = "syslog"
    CEF = "cef"
    GELF = "gelf"

# ============================================================================
# Error and Exception Enums
# ============================================================================

class ErrorCategory(Enum):
    """Catégories d'erreurs"""
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"

class ErrorSeverity(Enum):
    """Sévérité des erreurs"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# ============================================================================
# Feature Flags Enums
# ============================================================================

class FeatureState(Enum):
    """États des feature flags"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"
    BETA = "beta"

class FeatureScope(Enum):
    """Portée des features"""
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    ENVIRONMENT = "environment"

# ============================================================================
# Priority and Urgency Enums
# ============================================================================

class Priority(IntEnum):
    """Niveaux de priorité (ordonnés)"""
    LOWEST = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    HIGHEST = 5
    CRITICAL = 6

class Urgency(IntEnum):
    """Niveaux d'urgence (ordonnés)"""
    DEFERRED = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    URGENT = 5
    EMERGENCY = 6

# ============================================================================
# Status Flags (Flag Enum for bitwise operations)
# ============================================================================

class SystemFlags(Flag):
    """Flags système pour opérations bitwise"""
    NONE = 0
    MONITORING_ENABLED = auto()
    BACKUP_ENABLED = auto()
    SECURITY_ENABLED = auto()
    AI_ENABLED = auto()
    DEBUG_MODE = auto()
    MAINTENANCE_MODE = auto()
    DISASTER_RECOVERY_MODE = auto()
    HIGH_AVAILABILITY_MODE = auto()

class CapabilityFlags(Flag):
    """Flags de capacités"""
    NONE = 0
    AUTO_SCALING = auto()
    LOAD_BALANCING = auto()
    HEALTH_CHECKS = auto()
    CIRCUIT_BREAKER = auto()
    RATE_LIMITING = auto()
    CACHING = auto()
    COMPRESSION = auto()
    ENCRYPTION = auto()

# ============================================================================
# Utility Functions
# ============================================================================

def get_enum_values(enum_class) -> List[str]:
    """Retourne toutes les valeurs d'une énumération"""
    return [item.value for item in enum_class]

def get_enum_names(enum_class) -> List[str]:
    """Retourne tous les noms d'une énumération"""
    return [item.name for item in enum_class]

def enum_to_dict(enum_class) -> Dict[str, str]:
    """Convertit une énumération en dictionnaire"""
    return {item.name: item.value for item in enum_class}

def validate_enum_value(value: str, enum_class) -> bool:
    """Valide qu'une valeur appartient à une énumération"""
    return value in get_enum_values(enum_class)

def find_enum_by_value(value: str, enum_class):
    """Trouve un élément d'énumération par sa valeur"""
    for item in enum_class:
        if item.value == value:
            return item
    return None

# Export des énumérations principales
__all__ = [
    # Environment and System
    "EnvironmentType", "DeploymentMode", "RuntimePlatform",
    
    # Cloud
    "CloudProvider", "CloudRegion", "StorageClass",
    
    # Deployment
    "DeploymentStrategy", "DeploymentPhase", "DeploymentStatus",
    "HealthCheckType", "ScalingDirection", "ScalingTrigger",
    
    # Monitoring
    "MetricType", "AlertSeverity", "AlertState", "MonitoringDataSource",
    "NotificationChannel",
    
    # Security
    "SecurityEventType", "ThreatLevel", "SecurityFramework", "ComplianceStandard",
    "EncryptionAlgorithm", "HashAlgorithm",
    
    # Backup
    "BackupType", "BackupStatus", "CompressionAlgorithm", "StorageProtocol",
    
    # Performance
    "OptimizationTarget", "ResourceType", "PerformanceMetric", "LoadTestType",
    
    # Network
    "NetworkProtocol", "LoadBalancingAlgorithm", "ProxyType",
    
    # Data
    "DatabaseType", "DataFormat", "SerializationFormat",
    
    # ML
    "MLAlgorithm", "MLTaskType", "ModelStatus",
    
    # Logging
    "LogLevel", "AuditEventType", "LogFormat",
    
    # Errors
    "ErrorCategory", "ErrorSeverity",
    
    # Features
    "FeatureState", "FeatureScope",
    
    # Priority
    "Priority", "Urgency",
    
    # Flags
    "SystemFlags", "CapabilityFlags",
    
    # Utility functions
    "get_enum_values", "get_enum_names", "enum_to_dict",
    "validate_enum_value", "find_enum_by_value"
]
