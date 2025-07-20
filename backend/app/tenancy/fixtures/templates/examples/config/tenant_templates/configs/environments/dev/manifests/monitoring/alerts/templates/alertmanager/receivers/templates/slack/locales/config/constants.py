"""
Constants and Configuration Values for Slack Alert System.

This module contains all constants, default values, and configuration
parameters used throughout the Slack alert configuration system.

Author: Fahed Mlaiel
Version: 1.0.0
"""

from enum import Enum
from typing import Dict, List

# Default Configuration Values
DEFAULT_LOCALE = "en_US"
DEFAULT_ENVIRONMENT = "production"
DEFAULT_CACHE_TTL = 3600  # 1 hour
TEMPLATE_CACHE_TTL = 7200  # 2 hours
LOCALE_CACHE_TTL = 1800   # 30 minutes
CACHE_TTL_CONFIG = 3600   # 1 hour

# Performance and Limits
MAX_RETRIES = 3
MAX_TEMPLATE_SIZE = 100 * 1024  # 100KB
MAX_MESSAGE_LENGTH = 40000  # Slack limit
MAX_CONCURRENT_ALERTS = 1000
RATE_LIMIT_PER_SECOND = 100
CONNECTION_POOL_SIZE = 50

# Security Configuration
ENCRYPTION_KEY_SIZE = 32
TOKEN_EXPIRY_MINUTES = 60
MAX_LOGIN_ATTEMPTS = 5
AUDIT_LOG_RETENTION_DAYS = 90
SESSION_TIMEOUT_MINUTES = 120

# Supported Locales with Cultural Information
SUPPORTED_LOCALES = [
    "en_US",  # English (United States)
    "en_GB",  # English (United Kingdom)
    "fr_FR",  # French (France)
    "es_ES",  # Spanish (Spain)
    "es_MX",  # Spanish (Mexico)
    "de_DE",  # German (Germany)
    "it_IT",  # Italian (Italy)
    "pt_BR",  # Portuguese (Brazil)
    "pt_PT",  # Portuguese (Portugal)
    "ja_JP",  # Japanese (Japan)
    "ko_KR",  # Korean (South Korea)
    "zh_CN",  # Chinese (Simplified)
    "zh_TW",  # Chinese (Traditional)
    "ru_RU",  # Russian (Russia)
    "ar_SA",  # Arabic (Saudi Arabia)
    "hi_IN",  # Hindi (India)
    "nl_NL",  # Dutch (Netherlands)
    "sv_SE",  # Swedish (Sweden)
    "da_DK",  # Danish (Denmark)
    "no_NO",  # Norwegian (Norway)
    "fi_FI",  # Finnish (Finland)
    "tr_TR",  # Turkish (Turkey)
    "pl_PL",  # Polish (Poland)
    "cs_CZ",  # Czech (Czech Republic)
    "hu_HU",  # Hungarian (Hungary)
    "ro_RO",  # Romanian (Romania)
    "bg_BG",  # Bulgarian (Bulgaria)
    "hr_HR",  # Croatian (Croatia)
    "sk_SK",  # Slovak (Slovakia)
    "sl_SI"   # Slovenian (Slovenia)
]

# Fallback Locale Mapping
FALLBACK_LOCALES = {
    "en_GB": ["en_US"],
    "es_MX": ["es_ES", "en_US"],
    "pt_PT": ["pt_BR", "en_US"],
    "zh_TW": ["zh_CN", "en_US"],
    "ar_SA": ["en_US"],
    "hi_IN": ["en_US"],
    "nl_NL": ["en_US"],
    "sv_SE": ["en_US"],
    "da_DK": ["en_US"],
    "no_NO": ["en_US"],
    "fi_FI": ["en_US"],
    "tr_TR": ["en_US"],
    "pl_PL": ["en_US"],
    "cs_CZ": ["en_US"],
    "hu_HU": ["en_US"],
    "ro_RO": ["en_US"],
    "bg_BG": ["en_US"],
    "hr_HR": ["en_US"],
    "sk_SK": ["en_US"],
    "sl_SI": ["en_US"]
}

# Alert Priority Levels
class AlertPriority(Enum):
    """Alert priority levels with associated metadata."""
    CRITICAL = {
        "name": "critical",
        "level": 1,
        "color": "#FF0000",
        "emoji": "🚨",
        "notification": True,
        "escalation": True
    }
    HIGH = {
        "name": "high", 
        "level": 2,
        "color": "#FF8C00",
        "emoji": "⚠️",
        "notification": True,
        "escalation": True
    }
    MEDIUM = {
        "name": "medium",
        "level": 3,
        "color": "#FFD700",
        "emoji": "⚡",
        "notification": True,
        "escalation": False
    }
    LOW = {
        "name": "low",
        "level": 4,
        "color": "#32CD32",
        "emoji": "ℹ️",
        "notification": False,
        "escalation": False
    }
    INFO = {
        "name": "info",
        "level": 5,
        "color": "#87CEEB",
        "emoji": "📢",
        "notification": False,
        "escalation": False
    }

ALERT_PRIORITIES = [priority.value["name"] for priority in AlertPriority]

# Security Levels
SECURITY_LEVELS = {
    "low": {
        "encryption": "basic",
        "audit_level": "minimal",
        "session_timeout": 240,  # 4 hours
        "token_rotation": False
    },
    "medium": {
        "encryption": "standard",
        "audit_level": "standard",
        "session_timeout": 120,  # 2 hours
        "token_rotation": True
    },
    "high": {
        "encryption": "advanced",
        "audit_level": "detailed",
        "session_timeout": 60,   # 1 hour
        "token_rotation": True
    },
    "critical": {
        "encryption": "maximum",
        "audit_level": "comprehensive",
        "session_timeout": 30,   # 30 minutes
        "token_rotation": True
    }
}

# Slack-Specific Constants
SLACK_MESSAGE_TYPES = [
    "text",
    "attachments",
    "blocks",
    "interactive"
]

SLACK_ATTACHMENT_COLORS = {
    "good": "#36A64F",
    "warning": "#FF9500", 
    "danger": "#FF0000",
    "info": "#2196F3",
    "primary": "#007BFF",
    "secondary": "#6C757D",
    "success": "#28A745",
    "error": "#DC3545"
}

SLACK_EMOJI_MAP = {
    # Status emojis
    "success": "white_check_mark",
    "error": "x",
    "warning": "warning",
    "info": "information_source",
    "question": "question",
    "critical": "rotating_light",
    "alert": "warning",
    
    # System emojis
    "server": "desktop_computer",
    "database": "floppy_disk",
    "network": "satellite",
    "security": "lock",
    "monitoring": "eyes",
    "performance": "chart_with_upwards_trend",
    
    # Action emojis
    "start": "arrow_forward",
    "stop": "stop_sign",
    "restart": "arrows_counterclockwise",
    "pause": "pause_button",
    "resume": "arrow_forward",
    "deploy": "rocket",
    
    # Priority emojis
    "high_priority": "red_circle",
    "medium_priority": "yellow_circle", 
    "low_priority": "green_circle",
    "urgent": "sos",
    
    # Time emojis
    "clock": "clock3",
    "calendar": "calendar",
    "timer": "stopwatch",
    "schedule": "alarm_clock"
}

# Message Template Types
MESSAGE_TEMPLATES = {
    "alert": {
        "critical_system_failure": {
            "title": "🚨 Critical System Failure",
            "color": "danger",
            "priority": "critical",
            "escalation": True
        },
        "high_resource_usage": {
            "title": "⚠️ High Resource Usage Alert",
            "color": "warning", 
            "priority": "high",
            "escalation": True
        },
        "service_degradation": {
            "title": "⚡ Service Performance Degradation",
            "color": "warning",
            "priority": "medium",
            "escalation": False
        },
        "maintenance_notification": {
            "title": "🔧 Maintenance Notification",
            "color": "info",
            "priority": "low",
            "escalation": False
        },
        "deployment_notification": {
            "title": "🚀 Deployment Notification",
            "color": "good",
            "priority": "info",
            "escalation": False
        }
    },
    "status": {
        "service_up": {
            "title": "✅ Service Restored",
            "color": "good",
            "priority": "info",
            "escalation": False
        },
        "service_down": {
            "title": "❌ Service Down",
            "color": "danger",
            "priority": "critical",
            "escalation": True
        },
        "partial_outage": {
            "title": "⚠️ Partial Service Outage",
            "color": "warning",
            "priority": "high",
            "escalation": True
        }
    },
    "metric": {
        "threshold_exceeded": {
            "title": "📊 Metric Threshold Exceeded",
            "color": "warning",
            "priority": "medium",
            "escalation": False
        },
        "anomaly_detected": {
            "title": "🔍 Anomaly Detected",
            "color": "info",
            "priority": "medium",
            "escalation": False
        }
    }
}

# Environment Configuration
ENVIRONMENT_CONFIGS = {
    "development": {
        "debug": True,
        "cache_ttl": 300,     # 5 minutes
        "rate_limit": 1000,   # Higher limit for development
        "encryption": "basic",
        "audit_level": "minimal",
        "monitoring": "basic"
    },
    "staging": {
        "debug": False,
        "cache_ttl": 1800,    # 30 minutes
        "rate_limit": 500,
        "encryption": "standard",
        "audit_level": "standard",
        "monitoring": "standard"
    },
    "production": {
        "debug": False,
        "cache_ttl": 3600,    # 1 hour
        "rate_limit": 100,
        "encryption": "advanced",
        "audit_level": "comprehensive",
        "monitoring": "advanced"
    }
}

# Database Configuration
DATABASE_SETTINGS = {
    "connection_pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "echo": False,
    "pool_pre_ping": True
}

# Redis Configuration
REDIS_SETTINGS = {
    "decode_responses": True,
    "retry_on_timeout": True,
    "socket_keepalive": True,
    "socket_keepalive_options": {},
    "health_check_interval": 30,
    "max_connections": 100
}

# Monitoring and Metrics
METRICS_CONFIG = {
    "enable_prometheus": True,
    "metrics_port": 9090,
    "metrics_path": "/metrics",
    "collect_runtime_metrics": True,
    "collect_gc_metrics": True,
    "histogram_buckets": [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
}

# HTTP Client Configuration
HTTP_CLIENT_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "backoff_factor": 0.3,
    "verify_ssl": True,
    "follow_redirects": True,
    "max_redirects": 5
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG", 
            "formatter": "detailed",
            "filename": "logs/slack_config.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        },
        "json_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": "logs/slack_config.json",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "slack_config": {
            "level": "DEBUG",
            "handlers": ["console", "file", "json_file"],
            "propagate": False
        },
        "security": {
            "level": "INFO",
            "handlers": ["console", "json_file"],
            "propagate": False
        },
        "audit": {
            "level": "INFO",
            "handlers": ["json_file"],
            "propagate": False
        }
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"]
    }
}

# Validation Rules
VALIDATION_RULES = {
    "slack_token": {
        "pattern": r"^xox[bpas]-\d+-\d+-\d+-[a-f0-9]+$",
        "max_length": 200
    },
    "slack_webhook": {
        "pattern": r"^https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[a-zA-Z0-9]+$",
        "max_length": 500
    },
    "channel_id": {
        "pattern": r"^[A-Z0-9]+$",
        "min_length": 9,
        "max_length": 11
    },
    "user_id": {
        "pattern": r"^[UW][A-Z0-9]+$",
        "min_length": 9,
        "max_length": 11
    },
    "tenant_id": {
        "pattern": r"^[a-zA-Z0-9_-]+$",
        "min_length": 3,
        "max_length": 50
    },
    "template_name": {
        "pattern": r"^[a-zA-Z0-9_-]+$",
        "min_length": 1,
        "max_length": 100
    },
    "locale_code": {
        "pattern": r"^[a-z]{2}_[A-Z]{2}$",
        "length": 5
    }
}

# Feature Flags
FEATURE_FLAGS = {
    "enable_multi_tenant": True,
    "enable_template_caching": True,
    "enable_locale_caching": True,
    "enable_performance_monitoring": True,
    "enable_security_monitoring": True,
    "enable_audit_logging": True,
    "enable_rate_limiting": True,
    "enable_encryption": True,
    "enable_token_rotation": True,
    "enable_session_management": True,
    "enable_input_validation": True,
    "enable_output_sanitization": True,
    "enable_compliance_mode": True,
    "enable_advanced_analytics": True,
    "enable_real_time_alerts": True
}

# API Versioning
API_VERSIONS = {
    "v1": {
        "supported": True,
        "deprecated": False,
        "sunset_date": None,
        "features": ["basic_config", "templates", "locales"]
    },
    "v2": {
        "supported": True,
        "deprecated": False,
        "sunset_date": None,
        "features": ["basic_config", "templates", "locales", "security", "monitoring"]
    }
}

# Error Codes
ERROR_CODES = {
    # Configuration Errors (1000-1999)
    "CONFIG_NOT_FOUND": 1001,
    "CONFIG_INVALID": 1002,
    "CONFIG_CREATION_FAILED": 1003,
    "CONFIG_UPDATE_FAILED": 1004,
    "CONFIG_DELETE_FAILED": 1005,
    
    # Template Errors (2000-2999)
    "TEMPLATE_NOT_FOUND": 2001,
    "TEMPLATE_COMPILATION_ERROR": 2002,
    "TEMPLATE_RENDER_ERROR": 2003,
    "TEMPLATE_TOO_LARGE": 2004,
    "TEMPLATE_SYNTAX_ERROR": 2005,
    
    # Locale Errors (3000-3999)
    "LOCALE_NOT_SUPPORTED": 3001,
    "TRANSLATION_NOT_FOUND": 3002,
    "LOCALE_FORMAT_ERROR": 3003,
    "LOCALE_LOADING_ERROR": 3004,
    
    # Security Errors (4000-4999)
    "AUTHENTICATION_FAILED": 4001,
    "AUTHORIZATION_FAILED": 4002,
    "TOKEN_EXPIRED": 4003,
    "TOKEN_INVALID": 4004,
    "ENCRYPTION_FAILED": 4005,
    "DECRYPTION_FAILED": 4006,
    "RATE_LIMIT_EXCEEDED": 4007,
    "ACCOUNT_LOCKED": 4008,
    
    # System Errors (5000-5999)
    "DATABASE_ERROR": 5001,
    "CACHE_ERROR": 5002,
    "NETWORK_ERROR": 5003,
    "INTERNAL_ERROR": 5004,
    "SERVICE_UNAVAILABLE": 5005
}

# Success Messages
SUCCESS_MESSAGES = {
    "CONFIG_CREATED": "Configuration created successfully",
    "CONFIG_UPDATED": "Configuration updated successfully", 
    "CONFIG_DELETED": "Configuration deleted successfully",
    "TEMPLATE_CREATED": "Template created successfully",
    "TEMPLATE_UPDATED": "Template updated successfully",
    "TEMPLATE_DELETED": "Template deleted successfully",
    "USER_AUTHENTICATED": "User authenticated successfully",
    "TOKEN_CREATED": "Access token created successfully",
    "CACHE_CLEARED": "Cache cleared successfully",
    "SYSTEM_HEALTHY": "System health check passed"
}
