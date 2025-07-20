"""
Spotify AI Agent - Custom Fixture Exceptions
===========================================

Comprehensive exception hierarchy for fixture operations
providing detailed error context and recovery information.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID


class FixtureError(Exception):
    """Base exception for all fixture operations."""
    
    def __init__(
        self,
        message: str,
        fixture_id: Optional[UUID] = None,
        tenant_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.fixture_id = fixture_id
        self.tenant_id = tenant_id
        self.context = context or {}
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.fixture_id:
            parts.append(f"fixture_id={self.fixture_id}")
        if self.tenant_id:
            parts.append(f"tenant_id={self.tenant_id}")
        return " | ".join(parts)


class FixtureValidationError(FixtureError):
    """Raised when fixture validation fails."""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []


class FixtureTimeoutError(FixtureError):
    """Raised when fixture operation times out."""
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration


class FixtureConflictError(FixtureError):
    """Raised when fixture conflicts with existing data."""
    
    def __init__(
        self,
        message: str,
        conflicting_records: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.conflicting_records = conflicting_records or []


class FixtureDependencyError(FixtureError):
    """Raised when fixture dependencies cannot be resolved."""
    
    def __init__(
        self,
        message: str,
        missing_dependencies: Optional[List[UUID]] = None,
        circular_dependencies: Optional[List[UUID]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.missing_dependencies = missing_dependencies or []
        self.circular_dependencies = circular_dependencies or []


class FixtureDataError(FixtureError):
    """Raised when fixture data is invalid or corrupted."""
    
    def __init__(
        self,
        message: str,
        data_errors: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.data_errors = data_errors or []


class FixturePermissionError(FixtureError):
    """Raised when insufficient permissions for fixture operation."""
    
    def __init__(
        self,
        message: str,
        required_permissions: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.required_permissions = required_permissions or []


class FixtureRollbackError(FixtureError):
    """Raised when fixture rollback operation fails."""
    
    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.original_error = original_error


class FixtureSchemaError(FixtureError):
    """Raised when schema-related fixture operations fail."""
    
    def __init__(
        self,
        message: str,
        schema_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.schema_name = schema_name


class FixtureConfigError(FixtureError):
    """Raised when fixture configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        config_errors: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.config_errors = config_errors or []
