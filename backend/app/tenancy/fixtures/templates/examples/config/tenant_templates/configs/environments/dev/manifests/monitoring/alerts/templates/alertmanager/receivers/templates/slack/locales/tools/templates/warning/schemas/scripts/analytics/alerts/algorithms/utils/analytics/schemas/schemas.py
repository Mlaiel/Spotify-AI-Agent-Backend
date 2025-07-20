"""
Data Validation Schemas Package
==============================

Advanced validation schemas for Spotify AI Agent analytics data.
"""

from .schemas import *

__version__ = "1.0.0"
__author__ = "Fahed Mlaiel"

# Export all schemas and utilities
__all__ = [
    'SCHEMA_REGISTRY',
    'get_schema',
    'validate_data',
    'validate_tenant_access',
    'validate_data_retention',
    'sanitize_user_input'
]
