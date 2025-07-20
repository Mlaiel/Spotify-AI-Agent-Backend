"""
Custom Exceptions for Slack Alert Configuration System.

This module defines all custom exceptions used throughout the Slack alert
configuration system with detailed error information and proper hierarchy.

Author: Fahed Mlaiel
Version: 1.0.0
"""

from typing import Dict, Any, Optional


class SlackConfigBaseException(Exception):
    """Base exception for all Slack configuration errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base exception.
        
        Args:
            message: Error message
            error_code: Unique error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


# Configuration Exceptions
class ConfigurationError(SlackConfigBaseException):
    """General configuration error."""
    pass


class TenantNotFoundError(ConfigurationError):
    """Tenant configuration not found."""
    
    def __init__(self, tenant_id: str, environment: str = ""):
        message = f"Tenant configuration not found: {tenant_id}"
        if environment:
            message += f" in environment: {environment}"
        super().__init__(message, "TENANT_NOT_FOUND", {"tenant_id": tenant_id, "environment": environment})


class ConfigValidationError(ConfigurationError):
    """Configuration validation failed."""
    
    def __init__(self, validation_errors: Dict[str, Any]):
        message = "Configuration validation failed"
        super().__init__(message, "CONFIG_VALIDATION_FAILED", {"validation_errors": validation_errors})


class ConfigCreationError(ConfigurationError):
    """Configuration creation failed."""
    
    def __init__(self, tenant_id: str, reason: str):
        message = f"Failed to create configuration for tenant {tenant_id}: {reason}"
        super().__init__(message, "CONFIG_CREATION_FAILED", {"tenant_id": tenant_id, "reason": reason})


class ConfigUpdateError(ConfigurationError):
    """Configuration update failed."""
    
    def __init__(self, tenant_id: str, reason: str):
        message = f"Failed to update configuration for tenant {tenant_id}: {reason}"
        super().__init__(message, "CONFIG_UPDATE_FAILED", {"tenant_id": tenant_id, "reason": reason})


class ConfigDeletionError(ConfigurationError):
    """Configuration deletion failed."""
    
    def __init__(self, tenant_id: str, reason: str):
        message = f"Failed to delete configuration for tenant {tenant_id}: {reason}"
        super().__init__(message, "CONFIG_DELETION_FAILED", {"tenant_id": tenant_id, "reason": reason})


# Template Exceptions
class TemplateError(SlackConfigBaseException):
    """General template error."""
    pass


class TemplateNotFoundError(TemplateError):
    """Template not found."""
    
    def __init__(self, template_name: str):
        message = f"Template not found: {template_name}"
        super().__init__(message, "TEMPLATE_NOT_FOUND", {"template_name": template_name})


class TemplateCompilationError(TemplateError):
    """Template compilation failed."""
    
    def __init__(self, template_name: str, compilation_error: str):
        message = f"Template compilation failed for {template_name}: {compilation_error}"
        super().__init__(message, "TEMPLATE_COMPILATION_ERROR", {
            "template_name": template_name,
            "compilation_error": compilation_error
        })


class TemplateRenderError(TemplateError):
    """Template rendering failed."""
    
    def __init__(self, template_name: str, render_error: str, context: Optional[Dict[str, Any]] = None):
        message = f"Template rendering failed for {template_name}: {render_error}"
        super().__init__(message, "TEMPLATE_RENDER_ERROR", {
            "template_name": template_name,
            "render_error": render_error,
            "context": context
        })


class TemplateSizeError(TemplateError):
    """Template exceeds size limits."""
    
    def __init__(self, template_name: str, size: int, max_size: int):
        message = f"Template {template_name} size ({size} bytes) exceeds maximum ({max_size} bytes)"
        super().__init__(message, "TEMPLATE_SIZE_ERROR", {
            "template_name": template_name,
            "size": size,
            "max_size": max_size
        })


class TemplateSyntaxError(TemplateError):
    """Template syntax error."""
    
    def __init__(self, template_name: str, syntax_error: str, line_number: Optional[int] = None):
        message = f"Template syntax error in {template_name}: {syntax_error}"
        if line_number:
            message += f" (line {line_number})"
        super().__init__(message, "TEMPLATE_SYNTAX_ERROR", {
            "template_name": template_name,
            "syntax_error": syntax_error,
            "line_number": line_number
        })


class TemplateVariableError(TemplateError):
    """Template variable error."""
    
    def __init__(self, template_name: str, variable_name: str, error_details: str):
        message = f"Template variable error in {template_name} for variable '{variable_name}': {error_details}"
        super().__init__(message, "TEMPLATE_VARIABLE_ERROR", {
            "template_name": template_name,
            "variable_name": variable_name,
            "error_details": error_details
        })


# Locale Exceptions
class LocaleError(SlackConfigBaseException):
    """General locale error."""
    pass


class InvalidLocaleError(LocaleError):
    """Invalid locale code."""
    
    def __init__(self, locale_code: str, supported_locales: Optional[list] = None):
        message = f"Invalid locale code: {locale_code}"
        if supported_locales:
            message += f". Supported locales: {', '.join(supported_locales[:5])}{'...' if len(supported_locales) > 5 else ''}"
        super().__init__(message, "INVALID_LOCALE", {
            "locale_code": locale_code,
            "supported_locales": supported_locales
        })


class TranslationNotFoundError(LocaleError):
    """Translation not found."""
    
    def __init__(self, key: str, locale: str, fallback_attempted: bool = False):
        message = f"Translation not found for key '{key}' in locale '{locale}'"
        if fallback_attempted:
            message += " (fallback also failed)"
        super().__init__(message, "TRANSLATION_NOT_FOUND", {
            "key": key,
            "locale": locale,
            "fallback_attempted": fallback_attempted
        })


class LocaleLoadingError(LocaleError):
    """Locale loading failed."""
    
    def __init__(self, locale: str, reason: str):
        message = f"Failed to load locale {locale}: {reason}"
        super().__init__(message, "LOCALE_LOADING_ERROR", {"locale": locale, "reason": reason})


class FormatError(LocaleError):
    """Formatting error."""
    
    def __init__(self, format_type: str, value: Any, locale: str, reason: str):
        message = f"Failed to format {format_type} value '{value}' for locale '{locale}': {reason}"
        super().__init__(message, "FORMAT_ERROR", {
            "format_type": format_type,
            "value": str(value),
            "locale": locale,
            "reason": reason
        })


class PluralizationError(LocaleError):
    """Pluralization error."""
    
    def __init__(self, locale: str, count: int, rule_error: str):
        message = f"Pluralization failed for locale '{locale}' with count {count}: {rule_error}"
        super().__init__(message, "PLURALIZATION_ERROR", {
            "locale": locale,
            "count": count,
            "rule_error": rule_error
        })


# Security Exceptions
class SecurityError(SlackConfigBaseException):
    """General security error."""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed."""
    
    def __init__(self, username: str, reason: str, ip_address: Optional[str] = None):
        message = f"Authentication failed for user '{username}': {reason}"
        super().__init__(message, "AUTHENTICATION_FAILED", {
            "username": username,
            "reason": reason,
            "ip_address": ip_address
        })


class AuthorizationError(SecurityError):
    """Authorization failed."""
    
    def __init__(self, user_id: str, required_permission: str, resource: Optional[str] = None):
        message = f"User '{user_id}' lacks required permission '{required_permission}'"
        if resource:
            message += f" for resource '{resource}'"
        super().__init__(message, "AUTHORIZATION_FAILED", {
            "user_id": user_id,
            "required_permission": required_permission,
            "resource": resource
        })


class TokenExpiredError(SecurityError):
    """Token has expired."""
    
    def __init__(self, token_id: str, expired_at: str):
        message = f"Token {token_id} expired at {expired_at}"
        super().__init__(message, "TOKEN_EXPIRED", {
            "token_id": token_id,
            "expired_at": expired_at
        })


class InvalidTokenError(SecurityError):
    """Invalid token."""
    
    def __init__(self, reason: str, token_id: Optional[str] = None):
        message = f"Invalid token: {reason}"
        super().__init__(message, "INVALID_TOKEN", {
            "reason": reason,
            "token_id": token_id
        })


class EncryptionError(SecurityError):
    """Encryption/decryption error."""
    
    def __init__(self, operation: str, reason: str):
        message = f"Encryption {operation} failed: {reason}"
        super().__init__(message, "ENCRYPTION_ERROR", {
            "operation": operation,
            "reason": reason
        })


class RateLimitExceededError(SecurityError):
    """Rate limit exceeded."""
    
    def __init__(self, identifier: str, operation: str, limit: int, window_seconds: int):
        message = f"Rate limit exceeded for {identifier} on {operation}: {limit} requests per {window_seconds} seconds"
        super().__init__(message, "RATE_LIMIT_EXCEEDED", {
            "identifier": identifier,
            "operation": operation,
            "limit": limit,
            "window_seconds": window_seconds
        })


class AccountLockedError(SecurityError):
    """Account is locked."""
    
    def __init__(self, username: str, locked_until: str, reason: str):
        message = f"Account '{username}' is locked until {locked_until}: {reason}"
        super().__init__(message, "ACCOUNT_LOCKED", {
            "username": username,
            "locked_until": locked_until,
            "reason": reason
        })


class PasswordPolicyError(SecurityError):
    """Password does not meet policy requirements."""
    
    def __init__(self, policy_violations: list):
        message = f"Password policy violations: {', '.join(policy_violations)}"
        super().__init__(message, "PASSWORD_POLICY_ERROR", {
            "policy_violations": policy_violations
        })


class SessionExpiredError(SecurityError):
    """Session has expired."""
    
    def __init__(self, session_id: str, expired_at: str):
        message = f"Session {session_id} expired at {expired_at}"
        super().__init__(message, "SESSION_EXPIRED", {
            "session_id": session_id,
            "expired_at": expired_at
        })


# Cache Exceptions
class CacheError(SlackConfigBaseException):
    """General cache error."""
    pass


class CacheConnectionError(CacheError):
    """Cache connection failed."""
    
    def __init__(self, cache_type: str, reason: str):
        message = f"Failed to connect to {cache_type} cache: {reason}"
        super().__init__(message, "CACHE_CONNECTION_ERROR", {
            "cache_type": cache_type,
            "reason": reason
        })


class CacheOperationError(CacheError):
    """Cache operation failed."""
    
    def __init__(self, operation: str, key: str, reason: str):
        message = f"Cache {operation} operation failed for key '{key}': {reason}"
        super().__init__(message, "CACHE_OPERATION_ERROR", {
            "operation": operation,
            "key": key,
            "reason": reason
        })


class CacheSerializationError(CacheError):
    """Cache serialization/deserialization error."""
    
    def __init__(self, operation: str, data_type: str, reason: str):
        message = f"Cache {operation} failed for {data_type}: {reason}"
        super().__init__(message, "CACHE_SERIALIZATION_ERROR", {
            "operation": operation,
            "data_type": data_type,
            "reason": reason
        })


# Database Exceptions
class DatabaseError(SlackConfigBaseException):
    """General database error."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Database connection failed."""
    
    def __init__(self, database_url: str, reason: str):
        message = f"Failed to connect to database: {reason}"
        super().__init__(message, "DATABASE_CONNECTION_ERROR", {
            "database_url": database_url[:50] + "...",  # Mask sensitive info
            "reason": reason
        })


class DatabaseQueryError(DatabaseError):
    """Database query failed."""
    
    def __init__(self, query_type: str, table: str, reason: str):
        message = f"Database {query_type} query failed on table '{table}': {reason}"
        super().__init__(message, "DATABASE_QUERY_ERROR", {
            "query_type": query_type,
            "table": table,
            "reason": reason
        })


class DatabaseTransactionError(DatabaseError):
    """Database transaction failed."""
    
    def __init__(self, operation: str, reason: str):
        message = f"Database transaction {operation} failed: {reason}"
        super().__init__(message, "DATABASE_TRANSACTION_ERROR", {
            "operation": operation,
            "reason": reason
        })


# Validation Exceptions
class ValidationError(SlackConfigBaseException):
    """General validation error."""
    pass


class SchemaValidationError(ValidationError):
    """Schema validation failed."""
    
    def __init__(self, schema_name: str, validation_errors: Dict[str, Any]):
        message = f"Schema validation failed for {schema_name}"
        super().__init__(message, "SCHEMA_VALIDATION_ERROR", {
            "schema_name": schema_name,
            "validation_errors": validation_errors
        })


class InputValidationError(ValidationError):
    """Input validation failed."""
    
    def __init__(self, field_name: str, value: Any, rule: str, expected: str):
        message = f"Input validation failed for field '{field_name}': {rule}. Expected: {expected}"
        super().__init__(message, "INPUT_VALIDATION_ERROR", {
            "field_name": field_name,
            "value": str(value)[:100],  # Limit value length
            "rule": rule,
            "expected": expected
        })


class SlackValidationError(ValidationError):
    """Slack-specific validation error."""
    
    def __init__(self, validation_type: str, value: str, reason: str):
        message = f"Slack {validation_type} validation failed: {reason}"
        super().__init__(message, "SLACK_VALIDATION_ERROR", {
            "validation_type": validation_type,
            "value": value[:50] + "..." if len(value) > 50 else value,
            "reason": reason
        })


# Network and API Exceptions
class NetworkError(SlackConfigBaseException):
    """Network operation failed."""
    pass


class SlackAPIError(NetworkError):
    """Slack API error."""
    
    def __init__(self, endpoint: str, status_code: int, error_message: str):
        message = f"Slack API error at {endpoint}: {status_code} - {error_message}"
        super().__init__(message, "SLACK_API_ERROR", {
            "endpoint": endpoint,
            "status_code": status_code,
            "error_message": error_message
        })


class HTTPTimeoutError(NetworkError):
    """HTTP request timeout."""
    
    def __init__(self, url: str, timeout_seconds: int):
        message = f"HTTP request to {url} timed out after {timeout_seconds} seconds"
        super().__init__(message, "HTTP_TIMEOUT_ERROR", {
            "url": url,
            "timeout_seconds": timeout_seconds
        })


class HTTPConnectionError(NetworkError):
    """HTTP connection error."""
    
    def __init__(self, url: str, reason: str):
        message = f"HTTP connection to {url} failed: {reason}"
        super().__init__(message, "HTTP_CONNECTION_ERROR", {
            "url": url,
            "reason": reason
        })


# Performance Exceptions
class PerformanceError(SlackConfigBaseException):
    """Performance-related error."""
    pass


class PerformanceThresholdExceededError(PerformanceError):
    """Performance threshold exceeded."""
    
    def __init__(self, metric_name: str, value: float, threshold: float, unit: str):
        message = f"Performance threshold exceeded for {metric_name}: {value}{unit} > {threshold}{unit}"
        super().__init__(message, "PERFORMANCE_THRESHOLD_EXCEEDED", {
            "metric_name": metric_name,
            "value": value,
            "threshold": threshold,
            "unit": unit
        })


class ResourceExhaustedError(PerformanceError):
    """System resource exhausted."""
    
    def __init__(self, resource_type: str, current_usage: float, limit: float):
        message = f"Resource exhausted: {resource_type} usage {current_usage:.1f}% exceeds limit {limit:.1f}%"
        super().__init__(message, "RESOURCE_EXHAUSTED", {
            "resource_type": resource_type,
            "current_usage": current_usage,
            "limit": limit
        })


# System Exceptions
class SystemError(SlackConfigBaseException):
    """General system error."""
    pass


class ServiceUnavailableError(SystemError):
    """Service is unavailable."""
    
    def __init__(self, service_name: str, reason: str, retry_after: Optional[int] = None):
        message = f"Service '{service_name}' is unavailable: {reason}"
        if retry_after:
            message += f" (retry after {retry_after} seconds)"
        super().__init__(message, "SERVICE_UNAVAILABLE", {
            "service_name": service_name,
            "reason": reason,
            "retry_after": retry_after
        })


class DependencyError(SystemError):
    """Dependency error."""
    
    def __init__(self, dependency_name: str, reason: str):
        message = f"Dependency '{dependency_name}' error: {reason}"
        super().__init__(message, "DEPENDENCY_ERROR", {
            "dependency_name": dependency_name,
            "reason": reason
        })


class ConfigurationMissingError(SystemError):
    """Required configuration is missing."""
    
    def __init__(self, config_key: str, config_section: Optional[str] = None):
        message = f"Required configuration missing: {config_key}"
        if config_section:
            message += f" in section '{config_section}'"
        super().__init__(message, "CONFIGURATION_MISSING", {
            "config_key": config_key,
            "config_section": config_section
        })


# Utility function for exception handling
def handle_exception(exc: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response.
    
    Args:
        exc: Exception to handle
        
    Returns:
        Standardized error dictionary
    """
    if isinstance(exc, SlackConfigBaseException):
        return exc.to_dict()
    else:
        return {
            "error": "UnhandledException",
            "message": str(exc),
            "error_code": "UNHANDLED_ERROR",
            "details": {"exception_type": exc.__class__.__name__}
        }
