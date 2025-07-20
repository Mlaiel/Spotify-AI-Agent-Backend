"""
Spotify AI Agent - Advanced Tenant Monitoring Locales Tools Module
Système avancé de gestion des locales pour le monitoring multi-tenant

Author: Fahed Mlaiel
Lead Developer & AI Architect
Senior Backend Developer (Python/FastAPI/Django)
ML Engineer (TensorFlow/PyTorch/Hugging Face)
DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
Backend Security Specialist
Microservices Architect
"""

from .locale_manager import LocaleManager, TenantLocaleManager
from .locale_loader import LocaleLoader, DynamicLocaleLoader
from .locale_validator import LocaleValidator, ComplianceValidator
from .locale_optimizer import LocaleOptimizer, CacheOptimizer
from .locale_analytics import LocaleAnalytics, UsageTracker
from .locale_security import LocaleSecurity, TenantIsolation
from .locale_processor import LocaleProcessor, MessageFormatter
from .locale_cache import LocaleCache, DistributedCache
from .locale_monitoring import LocaleMonitoring, MetricsCollector
from .locale_migration import LocaleMigration, VersionManager

__version__ = "2.0.0"
__author__ = "Fahed Mlaiel"

__all__ = [
    # Core Managers
    "LocaleManager",
    "TenantLocaleManager",
    
    # Loaders
    "LocaleLoader", 
    "DynamicLocaleLoader",
    
    # Validation
    "LocaleValidator",
    "ComplianceValidator",
    
    # Optimization
    "LocaleOptimizer",
    "CacheOptimizer",
    
    # Analytics
    "LocaleAnalytics",
    "UsageTracker",
    
    # Security
    "LocaleSecurity",
    "TenantIsolation",
    
    # Processing
    "LocaleProcessor",
    "MessageFormatter",
    
    # Caching
    "LocaleCache",
    "DistributedCache",
    
    # Monitoring
    "LocaleMonitoring",
    "MetricsCollector",
    
    # Migration
    "LocaleMigrationManager",
    "MigrationEngine",
    "BackupManager",
    
    # Enhanced Monitoring
    "AlertManager"
]
