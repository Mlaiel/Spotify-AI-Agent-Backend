"""
Constants Module for Advanced Scripts System

This module defines all constants, configuration defaults, and system-wide
parameters used across the scripts ecosystem.

Version: 3.0.0
Developed by Spotify AI Agent Team
Lead Developer & AI Architect: Fahed Mlaiel
"""

import os
from enum import Enum
from typing import Dict, List, Tuple

# ============================================================================
# System Information
# ============================================================================

SYSTEM_NAME = "Alertmanager Advanced Scripts"
SYSTEM_VERSION = "3.0.0"
SYSTEM_CODENAME = "Quantum Guardian"
DEVELOPER = "Fahed Mlaiel"
ORGANIZATION = "Spotify AI Agent Team"
LICENSE = "Enterprise License"
COPYRIGHT = "© 2024 Spotify AI Agent Team. All rights reserved."

# ============================================================================
# API and Service Configuration
# ============================================================================

# Default ports
DEFAULT_PROMETHEUS_PORT = 9090
DEFAULT_ALERTMANAGER_PORT = 9093
DEFAULT_GRAFANA_PORT = 3000
DEFAULT_REDIS_PORT = 6379
DEFAULT_POSTGRES_PORT = 5432
DEFAULT_MONGODB_PORT = 27017

# Default hosts
DEFAULT_PROMETHEUS_HOST = "prometheus"
DEFAULT_ALERTMANAGER_HOST = "alertmanager"
DEFAULT_REDIS_HOST = "redis"
DEFAULT_POSTGRES_HOST = "postgres"
DEFAULT_MONGODB_HOST = "mongodb"

# API endpoints
PROMETHEUS_API_BASE = "/api/v1"
ALERTMANAGER_API_BASE = "/api/v2"
GRAFANA_API_BASE = "/api"

# Service URLs
DEFAULT_PROMETHEUS_URL = f"http://{DEFAULT_PROMETHEUS_HOST}:{DEFAULT_PROMETHEUS_PORT}"
DEFAULT_ALERTMANAGER_URL = f"http://{DEFAULT_ALERTMANAGER_HOST}:{DEFAULT_ALERTMANAGER_PORT}"

# ============================================================================
# Directory and File Paths
# ============================================================================

# Base directories
DEFAULT_CONFIG_DIR = "/etc/alertmanager"
DEFAULT_DATA_DIR = "/var/lib/alertmanager"
DEFAULT_LOG_DIR = "/var/log/alertmanager"
DEFAULT_BACKUP_DIR = "/var/backups/alertmanager"
DEFAULT_TEMP_DIR = "/tmp/alertmanager"
DEFAULT_CACHE_DIR = "/var/cache/alertmanager"

# Configuration files
ALERTMANAGER_CONFIG_FILE = "alertmanager.yml"
SCRIPTS_CONFIG_FILE = "scripts-config.yml"
SECURITY_CONFIG_FILE = "security-config.yml"
MONITORING_CONFIG_FILE = "monitoring-config.yml"

# Log files
MAIN_LOG_FILE = "alertmanager.log"
SCRIPTS_LOG_FILE = "scripts.log"
SECURITY_LOG_FILE = "security.log"
AUDIT_LOG_FILE = "audit.log"
PERFORMANCE_LOG_FILE = "performance.log"

# Backup files
BACKUP_METADATA_FILE = "backup-metadata.json"
BACKUP_INDEX_FILE = "backup-index.json"

# ============================================================================
# Database Configuration
# ============================================================================

# Database names
DEFAULT_POSTGRES_DB = "alertmanager"
DEFAULT_MONGODB_DB = "alertmanager"

# Database users
DEFAULT_POSTGRES_USER = "alertmanager"
DEFAULT_MONGODB_USER = "alertmanager"

# Connection pools
DEFAULT_DB_POOL_SIZE = 10
DEFAULT_DB_POOL_MAX_SIZE = 20
DEFAULT_DB_CONNECTION_TIMEOUT = 30
DEFAULT_DB_COMMAND_TIMEOUT = 300

# ============================================================================
# Security Configuration
# ============================================================================

# Encryption
DEFAULT_ENCRYPTION_ALGORITHM = "aes_256"
DEFAULT_KEY_DERIVATION_ITERATIONS = 100000
ENCRYPTION_KEY_LENGTH = 32  # 256 bits
IV_LENGTH = 16  # 128 bits

# Hashing
DEFAULT_HASH_ALGORITHM = "sha256"
HMAC_KEY_LENGTH = 32

# Session management
DEFAULT_SESSION_TIMEOUT = 3600  # 1 hour
DEFAULT_JWT_EXPIRY = 3600  # 1 hour
DEFAULT_REFRESH_TOKEN_EXPIRY = 86400  # 24 hours

# Rate limiting
DEFAULT_RATE_LIMIT_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW = 60  # seconds
DEFAULT_RATE_LIMIT_BURST = 20

# Password policy
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128
PASSWORD_REQUIRE_UPPERCASE = True
PASSWORD_REQUIRE_LOWERCASE = True
PASSWORD_REQUIRE_DIGITS = True
PASSWORD_REQUIRE_SPECIAL = True

# ============================================================================
# Monitoring and Metrics
# ============================================================================

# Metric collection intervals (seconds)
METRICS_COLLECTION_INTERVAL = 30
PERFORMANCE_METRICS_INTERVAL = 10
SECURITY_METRICS_INTERVAL = 60
SYSTEM_METRICS_INTERVAL = 60

# Metric retention periods (seconds)
SHORT_TERM_RETENTION = 86400  # 1 day
MEDIUM_TERM_RETENTION = 604800  # 1 week
LONG_TERM_RETENTION = 2592000  # 30 days

# Alert thresholds
CPU_WARNING_THRESHOLD = 70.0  # percentage
CPU_CRITICAL_THRESHOLD = 90.0  # percentage
MEMORY_WARNING_THRESHOLD = 80.0  # percentage
MEMORY_CRITICAL_THRESHOLD = 95.0  # percentage
DISK_WARNING_THRESHOLD = 80.0  # percentage
DISK_CRITICAL_THRESHOLD = 95.0  # percentage

# Response time thresholds (milliseconds)
RESPONSE_TIME_WARNING = 1000
RESPONSE_TIME_CRITICAL = 5000

# Error rate thresholds (percentage)
ERROR_RATE_WARNING = 1.0
ERROR_RATE_CRITICAL = 5.0

# ============================================================================
# Backup and Recovery
# ============================================================================

# Backup intervals
BACKUP_FULL_INTERVAL = 86400  # 24 hours
BACKUP_INCREMENTAL_INTERVAL = 3600  # 1 hour
BACKUP_VERIFICATION_INTERVAL = 21600  # 6 hours

# Backup retention
BACKUP_RETENTION_DAILY = 7  # 7 daily backups
BACKUP_RETENTION_WEEKLY = 4  # 4 weekly backups
BACKUP_RETENTION_MONTHLY = 12  # 12 monthly backups

# Compression settings
DEFAULT_COMPRESSION_ALGORITHM = "brotli"
COMPRESSION_LEVEL = 6

# Encryption for backups
BACKUP_ENCRYPTION_ENABLED = True
BACKUP_ENCRYPTION_ALGORITHM = "aes_256"

# Recovery targets
DEFAULT_RTO = 900  # 15 minutes (Recovery Time Objective)
DEFAULT_RPO = 300  # 5 minutes (Recovery Point Objective)

# ============================================================================
# Performance Optimization
# ============================================================================

# Auto-scaling thresholds
AUTOSCALE_CPU_THRESHOLD = 70.0
AUTOSCALE_MEMORY_THRESHOLD = 80.0
AUTOSCALE_MIN_REPLICAS = 2
AUTOSCALE_MAX_REPLICAS = 10
AUTOSCALE_SCALE_UP_COOLDOWN = 300  # 5 minutes
AUTOSCALE_SCALE_DOWN_COOLDOWN = 600  # 10 minutes

# Cache settings
CACHE_DEFAULT_TTL = 300  # 5 minutes
CACHE_MAX_SIZE = 1000  # number of entries
CACHE_CLEANUP_INTERVAL = 60  # seconds

# Connection pooling
HTTP_POOL_SIZE = 20
HTTP_POOL_MAX_SIZE = 50
HTTP_CONNECTION_TIMEOUT = 30
HTTP_READ_TIMEOUT = 60

# Garbage collection
GC_THRESHOLD_0 = 700
GC_THRESHOLD_1 = 10
GC_THRESHOLD_2 = 10

# ============================================================================
# Deployment Configuration
# ============================================================================

# Deployment strategies
DEPLOYMENT_STRATEGIES = ["blue_green", "rolling_update", "canary", "recreate"]
DEFAULT_DEPLOYMENT_STRATEGY = "blue_green"

# Deployment timeouts (seconds)
DEPLOYMENT_TIMEOUT = 1800  # 30 minutes
ROLLBACK_TIMEOUT = 600  # 10 minutes
HEALTH_CHECK_TIMEOUT = 300  # 5 minutes

# Deployment validation
VALIDATION_RETRIES = 3
VALIDATION_INTERVAL = 30  # seconds
POST_DEPLOYMENT_WAIT = 120  # seconds

# Canary deployment
CANARY_INITIAL_PERCENTAGE = 10
CANARY_INCREMENT_PERCENTAGE = 25
CANARY_PROMOTION_INTERVAL = 300  # 5 minutes

# ============================================================================
# Network Configuration
# ============================================================================

# Timeouts
NETWORK_CONNECT_TIMEOUT = 10
NETWORK_READ_TIMEOUT = 30
NETWORK_WRITE_TIMEOUT = 30

# Retry configuration
NETWORK_RETRY_ATTEMPTS = 3
NETWORK_RETRY_BACKOFF = 2.0
NETWORK_RETRY_MAX_DELAY = 60

# TLS configuration
TLS_MIN_VERSION = "TLSv1.2"
TLS_CIPHERS = [
    "ECDHE-RSA-AES256-GCM-SHA384",
    "ECDHE-RSA-AES128-GCM-SHA256",
    "ECDHE-RSA-AES256-SHA384",
    "ECDHE-RSA-AES128-SHA256"
]

# ============================================================================
# Cloud Provider Configuration
# ============================================================================

# AWS settings
AWS_DEFAULT_REGION = "us-west-2"
AWS_S3_STORAGE_CLASS = "STANDARD_IA"
AWS_S3_ENCRYPTION = "AES256"

# Azure settings
AZURE_DEFAULT_LOCATION = "West US 2"
AZURE_STORAGE_TIER = "Cool"
AZURE_REDUNDANCY = "LRS"

# GCP settings
GCP_DEFAULT_REGION = "us-west1"
GCP_STORAGE_CLASS = "NEARLINE"
GCP_DEFAULT_ZONE = "us-west1-a"

# ============================================================================
# Logging Configuration
# ============================================================================

# Log levels
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LEVEL = "INFO"
SECURITY_LOG_LEVEL = "WARNING"
AUDIT_LOG_LEVEL = "INFO"

# Log formats
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log rotation
LOG_MAX_SIZE = 100 * 1024 * 1024  # 100MB
LOG_BACKUP_COUNT = 5
LOG_ROTATION_INTERVAL = "midnight"

# Log retention
LOG_RETENTION_DAYS = 30
AUDIT_LOG_RETENTION_DAYS = 90
SECURITY_LOG_RETENTION_DAYS = 180

# ============================================================================
# Error Codes and Messages
# ============================================================================

class ErrorCodes(Enum):
    """Codes d'erreur standardisés"""
    # System errors (1000-1999)
    SYSTEM_ERROR = 1000
    CONFIGURATION_ERROR = 1001
    DEPENDENCY_ERROR = 1002
    
    # Authentication errors (2000-2999)
    AUTH_FAILED = 2000
    TOKEN_INVALID = 2001
    TOKEN_EXPIRED = 2002
    PERMISSION_DENIED = 2003
    
    # Validation errors (3000-3999)
    VALIDATION_FAILED = 3000
    INVALID_INPUT = 3001
    MISSING_PARAMETER = 3002
    
    # Database errors (4000-4999)
    DATABASE_ERROR = 4000
    CONNECTION_FAILED = 4001
    QUERY_FAILED = 4002
    
    # Network errors (5000-5999)
    NETWORK_ERROR = 5000
    CONNECTION_TIMEOUT = 5001
    SERVICE_UNAVAILABLE = 5002
    
    # Deployment errors (6000-6999)
    DEPLOYMENT_FAILED = 6000
    ROLLBACK_FAILED = 6001
    HEALTH_CHECK_FAILED = 6002
    
    # Backup errors (7000-7999)
    BACKUP_FAILED = 7000
    RESTORE_FAILED = 7001
    VERIFICATION_FAILED = 7002
    
    # Security errors (8000-8999)
    SECURITY_VIOLATION = 8000
    ENCRYPTION_FAILED = 8001
    AUDIT_FAILED = 8002

ERROR_MESSAGES = {
    ErrorCodes.SYSTEM_ERROR: "A system error occurred",
    ErrorCodes.CONFIGURATION_ERROR: "Configuration error detected",
    ErrorCodes.DEPENDENCY_ERROR: "Dependency error or missing component",
    ErrorCodes.AUTH_FAILED: "Authentication failed",
    ErrorCodes.TOKEN_INVALID: "Invalid authentication token",
    ErrorCodes.TOKEN_EXPIRED: "Authentication token has expired",
    ErrorCodes.PERMISSION_DENIED: "Permission denied for this operation",
    ErrorCodes.VALIDATION_FAILED: "Input validation failed",
    ErrorCodes.INVALID_INPUT: "Invalid input provided",
    ErrorCodes.MISSING_PARAMETER: "Required parameter is missing",
    ErrorCodes.DATABASE_ERROR: "Database operation failed",
    ErrorCodes.CONNECTION_FAILED: "Failed to connect to database",
    ErrorCodes.QUERY_FAILED: "Database query execution failed",
    ErrorCodes.NETWORK_ERROR: "Network operation failed",
    ErrorCodes.CONNECTION_TIMEOUT: "Network connection timeout",
    ErrorCodes.SERVICE_UNAVAILABLE: "Service is currently unavailable",
    ErrorCodes.DEPLOYMENT_FAILED: "Deployment operation failed",
    ErrorCodes.ROLLBACK_FAILED: "Rollback operation failed",
    ErrorCodes.HEALTH_CHECK_FAILED: "Health check validation failed",
    ErrorCodes.BACKUP_FAILED: "Backup operation failed",
    ErrorCodes.RESTORE_FAILED: "Restore operation failed",
    ErrorCodes.VERIFICATION_FAILED: "Backup verification failed",
    ErrorCodes.SECURITY_VIOLATION: "Security policy violation detected",
    ErrorCodes.ENCRYPTION_FAILED: "Encryption operation failed",
    ErrorCodes.AUDIT_FAILED: "Audit logging failed"
}

# ============================================================================
# Feature Flags
# ============================================================================

class FeatureFlags(Enum):
    """Feature flags pour activer/désactiver des fonctionnalités"""
    AI_OPTIMIZATION = "ai_optimization"
    PREDICTIVE_SCALING = "predictive_scaling"
    ADVANCED_SECURITY = "advanced_security"
    MULTI_CLOUD_BACKUP = "multi_cloud_backup"
    REAL_TIME_MONITORING = "real_time_monitoring"
    AUTO_REMEDIATION = "auto_remediation"
    FORENSIC_ANALYSIS = "forensic_analysis"
    BEHAVIORAL_ANALYTICS = "behavioral_analytics"

# Feature flags par défaut (activées/désactivées)
DEFAULT_FEATURE_FLAGS = {
    FeatureFlags.AI_OPTIMIZATION: True,
    FeatureFlags.PREDICTIVE_SCALING: True,
    FeatureFlags.ADVANCED_SECURITY: True,
    FeatureFlags.MULTI_CLOUD_BACKUP: True,
    FeatureFlags.REAL_TIME_MONITORING: True,
    FeatureFlags.AUTO_REMEDIATION: False,  # Prudence par défaut
    FeatureFlags.FORENSIC_ANALYSIS: True,
    FeatureFlags.BEHAVIORAL_ANALYTICS: True
}

# ============================================================================
# API Rate Limits
# ============================================================================

API_RATE_LIMITS = {
    "deployment": {
        "requests_per_minute": 10,
        "burst": 5
    },
    "backup": {
        "requests_per_minute": 30,
        "burst": 10
    },
    "monitoring": {
        "requests_per_minute": 100,
        "burst": 20
    },
    "security": {
        "requests_per_minute": 50,
        "burst": 15
    },
    "performance": {
        "requests_per_minute": 60,
        "burst": 20
    }
}

# ============================================================================
# Environment Variables
# ============================================================================

REQUIRED_ENV_VARS = [
    "PROMETHEUS_URL",
    "ALERTMANAGER_URL",
    "POSTGRES_HOST",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD"
]

OPTIONAL_ENV_VARS = {
    "REDIS_HOST": DEFAULT_REDIS_HOST,
    "REDIS_PORT": str(DEFAULT_REDIS_PORT),
    "LOG_LEVEL": DEFAULT_LOG_LEVEL,
    "BACKUP_ENCRYPTION_KEY": None,
    "AWS_REGION": AWS_DEFAULT_REGION,
    "AZURE_LOCATION": AZURE_DEFAULT_LOCATION,
    "GCP_REGION": GCP_DEFAULT_REGION
}

# ============================================================================
# Kubernetes Configuration
# ============================================================================

# Namespaces
K8S_DEFAULT_NAMESPACE = "monitoring"
K8S_SYSTEM_NAMESPACE = "kube-system"

# Labels et annotations
K8S_LABELS = {
    "app.kubernetes.io/name": "alertmanager-scripts",
    "app.kubernetes.io/version": SYSTEM_VERSION,
    "app.kubernetes.io/component": "monitoring",
    "app.kubernetes.io/part-of": "alertmanager",
    "app.kubernetes.io/managed-by": "spotify-ai-agent"
}

K8S_ANNOTATIONS = {
    "prometheus.io/scrape": "true",
    "prometheus.io/port": "8080",
    "prometheus.io/path": "/metrics"
}

# Resource limits
K8S_RESOURCE_LIMITS = {
    "cpu": "1000m",
    "memory": "1Gi"
}

K8S_RESOURCE_REQUESTS = {
    "cpu": "500m",
    "memory": "512Mi"
}

# ============================================================================
# Compliance and Audit
# ============================================================================

COMPLIANCE_FRAMEWORKS = [
    "SOX",
    "GDPR",
    "HIPAA",
    "PCI-DSS",
    "ISO-27001",
    "NIST",
    "CIS"
]

AUDIT_EVENTS = [
    "user_login",
    "user_logout",
    "configuration_change",
    "deployment_start",
    "deployment_complete",
    "backup_start",
    "backup_complete",
    "restore_start",
    "restore_complete",
    "security_alert",
    "performance_alert",
    "system_error"
]

# ============================================================================
# Machine Learning Configuration
# ============================================================================

ML_MODEL_CONFIGS = {
    "anomaly_detection": {
        "algorithm": "isolation_forest",
        "contamination": 0.1,
        "n_estimators": 100
    },
    "performance_prediction": {
        "algorithm": "random_forest",
        "n_estimators": 200,
        "max_depth": 10
    },
    "load_forecasting": {
        "algorithm": "lstm",
        "sequence_length": 24,
        "hidden_units": 50
    },
    "threat_detection": {
        "algorithm": "gradient_boosting",
        "n_estimators": 100,
        "learning_rate": 0.1
    }
}

# Seuils ML
ML_CONFIDENCE_THRESHOLD = 0.8
ML_ANOMALY_THRESHOLD = 0.7
ML_PREDICTION_HORIZON = 3600  # 1 hour in seconds

# ============================================================================
# Health Check Configuration
# ============================================================================

HEALTH_CHECK_ENDPOINTS = {
    "liveness": "/health/live",
    "readiness": "/health/ready",
    "startup": "/health/startup"
}

HEALTH_CHECK_INTERVALS = {
    "liveness": 30,  # seconds
    "readiness": 10,  # seconds
    "startup": 5    # seconds
}

HEALTH_CHECK_TIMEOUTS = {
    "liveness": 5,   # seconds
    "readiness": 3,  # seconds
    "startup": 10   # seconds
}

# ============================================================================
# Notification Configuration
# ============================================================================

NOTIFICATION_CHANNELS = [
    "email",
    "slack",
    "webhooks",
    "sms",
    "pagerduty"
]

NOTIFICATION_PRIORITIES = [
    "low",
    "medium", 
    "high",
    "critical",
    "emergency"
]

# Délais de notification (secondes)
NOTIFICATION_DELAYS = {
    "low": 3600,      # 1 hour
    "medium": 1800,   # 30 minutes
    "high": 300,      # 5 minutes
    "critical": 60,   # 1 minute
    "emergency": 0    # Immediate
}

# ============================================================================
# Utility Functions
# ============================================================================

def get_env_var(name: str, default: str = None, required: bool = False) -> str:
    """Récupère une variable d'environnement avec gestion des erreurs"""
    value = os.getenv(name, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable '{name}' is not set")
    
    return value

def get_config_path(filename: str) -> str:
    """Retourne le chemin complet d'un fichier de configuration"""
    config_dir = get_env_var("CONFIG_DIR", DEFAULT_CONFIG_DIR)
    return os.path.join(config_dir, filename)

def get_log_path(filename: str) -> str:
    """Retourne le chemin complet d'un fichier de log"""
    log_dir = get_env_var("LOG_DIR", DEFAULT_LOG_DIR)
    return os.path.join(log_dir, filename)

def is_feature_enabled(feature: FeatureFlags) -> bool:
    """Vérifie si une fonctionnalité est activée"""
    env_var = f"FEATURE_{feature.value.upper()}"
    env_value = get_env_var(env_var)
    
    if env_value is not None:
        return env_value.lower() in ['true', '1', 'yes', 'on']
    
    return DEFAULT_FEATURE_FLAGS.get(feature, False)

# Export des constantes principales
__all__ = [
    # System info
    "SYSTEM_NAME", "SYSTEM_VERSION", "SYSTEM_CODENAME", "DEVELOPER", "ORGANIZATION",
    
    # Configuration
    "DEFAULT_PROMETHEUS_URL", "DEFAULT_ALERTMANAGER_URL",
    "DEFAULT_CONFIG_DIR", "DEFAULT_DATA_DIR", "DEFAULT_LOG_DIR",
    
    # Thresholds
    "CPU_WARNING_THRESHOLD", "CPU_CRITICAL_THRESHOLD",
    "MEMORY_WARNING_THRESHOLD", "MEMORY_CRITICAL_THRESHOLD",
    "RESPONSE_TIME_WARNING", "RESPONSE_TIME_CRITICAL",
    
    # Intervals
    "METRICS_COLLECTION_INTERVAL", "BACKUP_FULL_INTERVAL",
    
    # Enums
    "ErrorCodes", "FeatureFlags",
    
    # Dictionaries
    "ERROR_MESSAGES", "DEFAULT_FEATURE_FLAGS", "API_RATE_LIMITS",
    "ML_MODEL_CONFIGS", "NOTIFICATION_DELAYS",
    
    # Functions
    "get_env_var", "get_config_path", "get_log_path", "is_feature_enabled"
]
