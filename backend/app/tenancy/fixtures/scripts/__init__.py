#!/usr/bin/env python3
"""
Spotify AI Agent - Fixture Scripts Package
==========================================

This package contains executable scripts for fixture management
and tenant operations in the Spotify AI Agent backend.

Scripts provided:
- init_tenant.py: Initialize new tenant with fixtures
- load_fixtures.py: Load fixture data for existing tenants  
- validate_data.py: Validate tenant data and fixtures
- cleanup.py: Clean up tenant data and temporary files
- backup.py: Create backups of tenant data
- migrate.py: Migrate fixtures between versions
- monitor.py: Monitor fixture operations and health

Author: Expert Development Team (Fahed Mlaiel)
Created: 2025-01-02
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Expert Development Team (Fahed Mlaiel)"

# Scripts metadata
SCRIPTS = {
    "init_tenant": {
        "description": "Initialize new tenant with complete fixture setup",
        "usage": "python -m app.tenancy.fixtures.scripts.init_tenant --tenant-id <id> --tier <tier>",
        "requires": ["database", "redis"]
    },
    "load_fixtures": {
        "description": "Load fixture data for existing tenant",
        "usage": "python -m app.tenancy.fixtures.scripts.load_fixtures --tenant-id <id> --data-type <type>",
        "requires": ["database"]
    },
    "validate_data": {
        "description": "Validate tenant data integrity and compliance",
        "usage": "python -m app.tenancy.fixtures.scripts.validate_data --tenant-id <id>",
        "requires": ["database"]
    },
    "cleanup": {
        "description": "Clean up tenant data and temporary files",
        "usage": "python -m app.tenancy.fixtures.scripts.cleanup --tenant-id <id> [--dry-run]",
        "requires": ["database", "filesystem"]
    },
    "backup": {
        "description": "Create comprehensive backup of tenant data",
        "usage": "python -m app.tenancy.fixtures.scripts.backup --tenant-id <id> --output <path>",
        "requires": ["database", "filesystem"]
    },
    "migrate": {
        "description": "Migrate fixtures to new version",
        "usage": "python -m app.tenancy.fixtures.scripts.migrate --from-version <ver> --to-version <ver>",
        "requires": ["database"]
    },
    "monitor": {
        "description": "Monitor fixture operations and system health",
        "usage": "python -m app.tenancy.fixtures.scripts.monitor [--continuous]",
        "requires": ["database", "redis"]
    }
}

# Export script functions for programmatic use
from .init_tenant import init_tenant, TenantInitializer
from .load_fixtures import load_fixtures, FixtureLoader
from .validate_data import validate_data, DataValidator
from .cleanup import cleanup_data, DataCleanup
from .backup import backup_data, restore_data, BackupManager
from .migrate import migrate_fixtures, FixtureMigrator
from .monitor import monitor_fixtures, FixtureMonitoringSystem

__all__ = [
    "SCRIPTS",
    "init_tenant",
    "TenantInitializer", 
    "load_fixtures",
    "FixtureLoader",
    "validate_data",
    "DataValidator",
    "cleanup_data",
    "DataCleanup",
    "backup_data",
    "restore_data",
    "BackupManager",
    "migrate_fixtures",
    "FixtureMigrator",
    "monitor_fixtures",
    "FixtureMonitoringSystem"
]
