"""
Spotify AI Agent - Tenancy Fixtures Module
==========================================

Enterprise-grade tenant data fixtures and initialization system.
Provides comprehensive data seeding, schema initialization, and 
tenant-specific configuration management.

Module Components:
- Base fixtures infrastructure
- Tenant-specific data loaders
- Schema initialization utilities
- Multi-tenant configuration management
- Performance monitoring and validation
- Compliance and audit fixtures

Author: Development Team
Lead: Fahed Mlaiel
Version: 1.0.0
"""

from .base import BaseFixture, FixtureManager
from .tenant_fixtures import TenantFixture, TenantDataLoader
from .schema_fixtures import SchemaFixture, SchemaInitializer
from .config_fixtures import ConfigFixture, ConfigurationManager
from .data_loaders import (
    DataLoader,
    SpotifyDataLoader,
    AIModelLoader,
    AnalyticsLoader,
    CollaborationLoader
)
from .validators import FixtureValidator, DataIntegrityValidator
from .monitoring import FixtureMonitor, PerformanceTracker
from .utils import (
    FixtureUtils,
    TenantUtils,
    ValidationUtils,
    ConfigUtils
)

__all__ = [
    # Core Components
    'BaseFixture',
    'FixtureManager',
    
    # Tenant Management
    'TenantFixture',
    'TenantDataLoader',
    
    # Schema Management
    'SchemaFixture',
    'SchemaInitializer',
    
    # Configuration
    'ConfigFixture',
    'ConfigurationManager',
    
    # Data Loaders
    'DataLoader',
    'SpotifyDataLoader',
    'AIModelLoader',
    'AnalyticsLoader',
    'CollaborationLoader',
    
    # Validation
    'FixtureValidator',
    'DataIntegrityValidator',
    
    # Monitoring
    'FixtureMonitor',
    'PerformanceTracker',
    
    # Utilities
    'FixtureUtils',
    'TenantUtils',
    'ValidationUtils',
    'ConfigUtils'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Spotify AI Agent Development Team"
__license__ = "Proprietary"
__description__ = "Enterprise tenancy fixtures and data management system"

# Configuration constants
DEFAULT_BATCH_SIZE = 1000
MAX_CONCURRENT_OPERATIONS = 10
FIXTURE_CACHE_TTL = 3600
VALIDATION_TIMEOUT = 300

# Feature flags
ENABLE_PERFORMANCE_MONITORING = True
ENABLE_DATA_VALIDATION = True
ENABLE_AUDIT_LOGGING = True
ENABLE_CACHE_OPTIMIZATION = True
