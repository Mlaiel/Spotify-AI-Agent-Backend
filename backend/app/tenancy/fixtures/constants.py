"""
Spotify AI Agent - Fixture Constants and Configuration
=====================================================

Central configuration constants for the fixture system.
"""

# Performance Configuration
DEFAULT_BATCH_SIZE = 1000
MAX_CONCURRENT_OPERATIONS = 10
FIXTURE_CACHE_TTL = 3600  # seconds
VALIDATION_TIMEOUT = 300  # seconds
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0  # seconds

# Memory Management
MAX_MEMORY_USAGE_MB = 512
MEMORY_THRESHOLD_WARNING = 0.8
MEMORY_CHECK_INTERVAL = 60  # seconds

# Database Configuration
CONNECTION_POOL_SIZE = 20
CONNECTION_TIMEOUT = 30
QUERY_TIMEOUT = 60
TRANSACTION_TIMEOUT = 300

# Security Configuration
MAX_TENANT_COUNT = 1000
MAX_FIXTURE_SIZE_MB = 100
ALLOWED_FIXTURE_TYPES = {
    "json", "yaml", "sql", "csv", "xml"
}
SENSITIVE_FIELDS = {
    "password", "token", "secret", "key", "credential"
}

# Monitoring Configuration
METRICS_COLLECTION_INTERVAL = 30  # seconds
LOG_RETENTION_DAYS = 30
ALERT_THRESHOLD_ERROR_RATE = 0.05
ALERT_THRESHOLD_LATENCY_P95 = 5.0  # seconds

# File System Configuration
FIXTURE_BASE_PATH = "/app/fixtures"
BACKUP_PATH = "/app/backups/fixtures"
TEMP_PATH = "/tmp/fixtures"
MAX_FILE_SIZE_MB = 50

# Template Configuration
TEMPLATE_ENCODING = "utf-8"
TEMPLATE_VALIDATION_STRICT = True
DEFAULT_TEMPLATE_VERSION = "1.0.0"

# Feature Flags
ENABLE_PERFORMANCE_MONITORING = True
ENABLE_DATA_VALIDATION = True
ENABLE_AUDIT_LOGGING = True
ENABLE_CACHE_OPTIMIZATION = True
ENABLE_PARALLEL_PROCESSING = True
ENABLE_AUTO_ROLLBACK = True
ENABLE_SCHEMA_VALIDATION = True
ENABLE_DEPENDENCY_CHECKING = True

# Cache Configuration
REDIS_CACHE_TTL = 3600
MEMORY_CACHE_SIZE = 1000
CACHE_KEY_PREFIX = "fixture:"
CACHE_COMPRESSION = True

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
LOG_MAX_SIZE_MB = 100
LOG_BACKUP_COUNT = 5

# API Rate Limiting
API_RATE_LIMIT_PER_MINUTE = 1000
API_BURST_LIMIT = 100
API_RATE_LIMIT_WINDOW = 60  # seconds

# Tenant Configuration
DEFAULT_TENANT_LIMITS = {
    "max_fixtures": 100,
    "max_data_size_mb": 1000,
    "max_concurrent_operations": 5,
    "api_calls_per_hour": 10000
}

PREMIUM_TENANT_LIMITS = {
    "max_fixtures": 1000,
    "max_data_size_mb": 10000,
    "max_concurrent_operations": 20,
    "api_calls_per_hour": 100000
}

ENTERPRISE_TENANT_LIMITS = {
    "max_fixtures": 10000,
    "max_data_size_mb": 100000,
    "max_concurrent_operations": 50,
    "api_calls_per_hour": 1000000
}

# Data Types Configuration
SUPPORTED_DATA_TYPES = {
    "string", "integer", "float", "boolean", "datetime", 
    "json", "binary", "uuid", "email", "url"
}

FIXTURE_SCHEMAS = {
    "tenant": "tenant_schema.json",
    "config": "config_schema.json", 
    "data": "data_schema.json",
    "analytics": "analytics_schema.json"
}

# Validation Rules
VALIDATION_RULES = {
    "required_fields": ["id", "type", "version"],
    "max_string_length": 1000,
    "max_array_size": 10000,
    "max_object_depth": 10,
    "allowed_characters": r"^[a-zA-Z0-9_\-\.]+$"
}

# Error Messages
ERROR_MESSAGES = {
    "fixture_not_found": "Fixture with ID {fixture_id} not found",
    "validation_failed": "Fixture validation failed: {errors}",
    "timeout_exceeded": "Operation timed out after {timeout} seconds",
    "dependency_missing": "Missing dependency: {dependency_id}",
    "permission_denied": "Insufficient permissions for operation",
    "data_conflict": "Data conflict detected: {details}",
    "rollback_failed": "Rollback operation failed: {error}",
    "schema_invalid": "Invalid schema: {schema_errors}",
    "config_error": "Configuration error: {config_errors}"
}

# Status Codes
STATUS_CODES = {
    "success": 200,
    "created": 201,
    "accepted": 202,
    "bad_request": 400,
    "unauthorized": 401,
    "forbidden": 403,
    "not_found": 404,
    "conflict": 409,
    "timeout": 408,
    "internal_error": 500,
    "service_unavailable": 503
}

# Environment Configuration
ENVIRONMENT_CONFIGS = {
    "development": {
        "log_level": "DEBUG",
        "enable_debug_mode": True,
        "cache_ttl": 300,
        "validation_strict": False
    },
    "staging": {
        "log_level": "INFO", 
        "enable_debug_mode": False,
        "cache_ttl": 1800,
        "validation_strict": True
    },
    "production": {
        "log_level": "WARNING",
        "enable_debug_mode": False,
        "cache_ttl": 3600,
        "validation_strict": True
    }
}

# Metrics Configuration
METRICS_NAMES = {
    "fixture_execution_time": "fixture_execution_duration_seconds",
    "fixture_success_rate": "fixture_success_rate",
    "fixture_error_rate": "fixture_error_rate", 
    "memory_usage": "fixture_memory_usage_bytes",
    "database_connections": "fixture_db_connections_active",
    "cache_hit_rate": "fixture_cache_hit_rate"
}

# Health Check Configuration
HEALTH_CHECK_ENDPOINTS = {
    "database": "/health/database",
    "cache": "/health/cache",
    "storage": "/health/storage",
    "external_apis": "/health/external"
}

HEALTH_CHECK_TIMEOUT = 5  # seconds
HEALTH_CHECK_INTERVAL = 30  # seconds
