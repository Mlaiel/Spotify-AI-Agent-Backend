#!/usr/bin/env python3
"""
Spotify AI Agent - Template Validators
=====================================

Comprehensive validation system for template integrity, security,
and compliance across all template categories.

Validators:
- Schema validation (structure, types, constraints)
- Security validation (XSS, injection, sensitive data)
- Business logic validation (rules, dependencies)
- Performance validation (size, complexity limits)
- Compliance validation (GDPR, data retention)

Author: Expert Development Team
"""

import json
import re
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import jsonschema
from jsonschema import Draft7Validator, FormatChecker
import bleach
from email_validator import validate_email, EmailNotValidError

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of template validation."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field_path: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    template_id: str
    template_type: str
    is_valid: bool
    total_issues: int
    results: List[ValidationResult]
    validation_timestamp: datetime
    performance_metrics: Dict[str, Any]
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationResult]:
        """Get validation issues by severity level."""
        return [result for result in self.results if result.severity == severity]
    
    def has_critical_issues(self) -> bool:
        """Check if template has critical validation issues."""
        return any(result.severity == ValidationSeverity.CRITICAL for result in self.results)
    
    def has_errors(self) -> bool:
        """Check if template has error-level issues."""
        return any(result.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for result in self.results)


class BaseValidator(ABC):
    """Base class for all template validators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def validate(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate template and return results."""
        pass
    
    def _create_result(
        self,
        is_valid: bool,
        severity: ValidationSeverity,
        message: str,
        field_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        fix_suggestion: Optional[str] = None
    ) -> ValidationResult:
        """Create validation result."""
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            message=message,
            field_path=field_path,
            details=details,
            fix_suggestion=fix_suggestion
        )


class SchemaValidator(BaseValidator):
    """Validates template structure against JSON schemas."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.schemas = self._load_schemas()
        self.format_checker = FormatChecker()
    
    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load template schemas for validation."""
        return {
            "tenant_init": {
                "type": "object",
                "required": ["tenant_id", "tier", "configuration"],
                "properties": {
                    "tenant_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
                    "tenant_name": {"type": "string", "maxLength": 100},
                    "tier": {"type": "string", "enum": ["starter", "professional", "enterprise"]},
                    "configuration": {
                        "type": "object",
                        "required": ["limits", "features", "security"],
                        "properties": {
                            "limits": {
                                "type": "object",
                                "required": ["max_users", "storage_gb"],
                                "properties": {
                                    "max_users": {"type": "integer", "minimum": -1},
                                    "storage_gb": {"type": "integer", "minimum": 1},
                                    "ai_sessions_per_month": {"type": "integer", "minimum": -1},
                                    "api_rate_limit_per_hour": {"type": "integer", "minimum": 1}
                                }
                            },
                            "features": {
                                "type": "object",
                                "required": ["enabled"],
                                "properties": {
                                    "enabled": {"type": "array", "items": {"type": "string"}},
                                    "disabled": {"type": "array", "items": {"type": "string"}}
                                }
                            },
                            "security": {
                                "type": "object",
                                "required": ["password_policy", "session_timeout_minutes"],
                                "properties": {
                                    "password_policy": {
                                        "type": "object",
                                        "required": ["min_length"],
                                        "properties": {
                                            "min_length": {"type": "integer", "minimum": 6, "maximum": 128},
                                            "require_special_chars": {"type": "boolean"},
                                            "require_numbers": {"type": "boolean"},
                                            "require_uppercase": {"type": "boolean"},
                                            "max_age_days": {"type": "integer", "minimum": 1, "maximum": 365}
                                        }
                                    },
                                    "session_timeout_minutes": {"type": "integer", "minimum": 5, "maximum": 1440},
                                    "mfa_required": {"type": "boolean"},
                                    "ip_whitelist": {"type": "array", "items": {"type": "string", "format": "ipv4"}},
                                    "audit_logging": {"type": "boolean"}
                                }
                            }
                        }
                    },
                    "status": {"type": "string", "enum": ["active", "inactive", "suspended"]}
                }
            },
            "user_profile": {
                "type": "object",
                "required": ["user_id", "tenant_id", "email"],
                "properties": {
                    "user_id": {"type": "string", "pattern": "^[a-fA-F0-9-]+$"},
                    "tenant_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
                    "email": {"type": "string", "format": "email"},
                    "username": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$", "maxLength": 50},
                    "display_name": {"type": "string", "maxLength": 100},
                    "role": {"type": "string", "enum": ["admin", "manager", "user", "guest"]},
                    "status": {"type": "string", "enum": ["active", "inactive", "suspended", "pending"]},
                    "profile": {
                        "type": "object",
                        "properties": {
                            "bio": {"type": "string", "maxLength": 500},
                            "location": {"type": "string", "maxLength": 100},
                            "website": {"type": "string", "format": "uri"},
                            "music_preferences": {
                                "type": "object",
                                "properties": {
                                    "favorite_genres": {"type": "array", "items": {"type": "string"}},
                                    "favorite_artists": {"type": "array", "items": {"type": "string"}},
                                    "listening_habits": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            },
            "content_types": {
                "type": "object",
                "required": ["tenant_id", "content_types"],
                "properties": {
                    "tenant_id": {"type": "string"},
                    "content_types": {
                        "type": "object",
                        "patternProperties": {
                            "^[a-z_]+$": {
                                "type": "object",
                                "required": ["name", "description", "fields"],
                                "properties": {
                                    "name": {"type": "string", "maxLength": 100},
                                    "description": {"type": "string", "maxLength": 500},
                                    "fields": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def validate(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate template against schema."""
        results = []
        
        # Determine template type
        template_type = self._determine_template_type(template)
        
        if not template_type:
            results.append(self._create_result(
                False, ValidationSeverity.ERROR,
                "Cannot determine template type",
                fix_suggestion="Ensure template has valid _metadata.template_type field"
            ))
            return results
        
        # Get schema for template type
        schema = self.schemas.get(template_type)
        if not schema:
            results.append(self._create_result(
                False, ValidationSeverity.WARNING,
                f"No schema found for template type: {template_type}",
                details={"template_type": template_type}
            ))
            return results
        
        # Validate against schema
        try:
            validator = Draft7Validator(schema, format_checker=self.format_checker)
            
            # Check if template is valid
            if validator.is_valid(template):
                results.append(self._create_result(
                    True, ValidationSeverity.INFO,
                    "Template structure is valid"
                ))
            else:
                # Collect all validation errors
                for error in validator.iter_errors(template):
                    field_path = ".".join(str(x) for x in error.absolute_path) if error.absolute_path else "root"
                    
                    results.append(self._create_result(
                        False, ValidationSeverity.ERROR,
                        f"Schema validation failed: {error.message}",
                        field_path=field_path,
                        details={"schema_error": error.message, "failed_value": error.instance},
                        fix_suggestion=self._get_schema_fix_suggestion(error)
                    ))
        
        except Exception as e:
            results.append(self._create_result(
                False, ValidationSeverity.CRITICAL,
                f"Schema validation error: {str(e)}",
                details={"exception": str(e)}
            ))
        
        return results
    
    def _determine_template_type(self, template: Dict[str, Any]) -> Optional[str]:
        """Determine template type from metadata or structure."""
        # Check metadata first
        metadata = template.get("_metadata", {})
        template_type = metadata.get("template_type")
        
        if template_type:
            return template_type
        
        # Fallback to structure-based detection
        if "tenant_id" in template and "configuration" in template and "tier" in template:
            return "tenant_init"
        elif "user_id" in template and "email" in template and "profile" in template:
            return "user_profile"
        elif "content_types" in template:
            return "content_types"
        
        return None
    
    def _get_schema_fix_suggestion(self, error) -> Optional[str]:
        """Get fix suggestion for schema validation error."""
        if "required" in error.message.lower():
            return f"Add the required field: {error.message.split()[-1]}"
        elif "pattern" in error.message.lower():
            return "Ensure the value matches the required pattern"
        elif "enum" in error.message.lower():
            return "Use one of the allowed values"
        elif "type" in error.message.lower():
            return "Ensure the field has the correct data type"
        elif "format" in error.message.lower():
            return "Ensure the field follows the required format (e.g., email, URI)"
        return None


class SecurityValidator(BaseValidator):
    """Validates template security and prevents vulnerabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'eval\s*\(',
            r'Function\s*\(',
            r'setTimeout\s*\(',
            r'setInterval\s*\(',
        ]
        self.sensitive_fields = [
            'password', 'secret', 'token', 'key', 'credential',
            'client_secret', 'api_key', 'webhook_secret'
        ]
    
    def validate(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate template security."""
        results = []
        
        # Check for XSS vulnerabilities
        results.extend(self._check_xss_vulnerabilities(template))
        
        # Check for injection attacks
        results.extend(self._check_injection_attacks(template))
        
        # Check for sensitive data exposure
        results.extend(self._check_sensitive_data(template))
        
        # Check for unsafe template expressions
        results.extend(self._check_unsafe_expressions(template))
        
        # Check permissions and access control
        results.extend(self._check_permissions(template))
        
        return results
    
    def _check_xss_vulnerabilities(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check for XSS vulnerabilities in template."""
        results = []
        
        def check_value(value, path=""):
            if isinstance(value, str):
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        results.append(self._create_result(
                            False, ValidationSeverity.CRITICAL,
                            f"Potential XSS vulnerability detected",
                            field_path=path,
                            details={"pattern": pattern, "value": value[:100]},
                            fix_suggestion="Remove or properly escape dangerous HTML/JavaScript content"
                        ))
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_value(v, f"{path}.{k}" if path else k)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_value(item, f"{path}[{i}]" if path else f"[{i}]")
        
        check_value(template)
        return results
    
    def _check_injection_attacks(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check for SQL injection and other injection vulnerabilities."""
        results = []
        
        injection_patterns = [
            r'(\bUNION\b|\bSELECT\b|\bINSERT\b|\bDELETE\b|\bUPDATE\b|\bDROP\b)',
            r'(\bEXEC\b|\bEXECUTE\b)',
            r'(\${.*})',  # Template injection
            r'(\#\{.*\})',  # Template injection
        ]
        
        def check_injection(value, path=""):
            if isinstance(value, str):
                for pattern in injection_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        results.append(self._create_result(
                            False, ValidationSeverity.ERROR,
                            f"Potential injection vulnerability detected",
                            field_path=path,
                            details={"pattern": pattern, "value": value[:100]},
                            fix_suggestion="Sanitize input and use parameterized queries"
                        ))
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_injection(v, f"{path}.{k}" if path else k)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_injection(item, f"{path}[{i}]" if path else f"[{i}]")
        
        check_injection(template)
        return results
    
    def _check_sensitive_data(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check for exposed sensitive data."""
        results = []
        
        def check_sensitive(key, value, path=""):
            key_lower = key.lower()
            
            # Check if field name suggests sensitive data
            is_sensitive_field = any(sensitive in key_lower for sensitive in self.sensitive_fields)
            
            if is_sensitive_field and isinstance(value, str):
                # Check if value appears to be plain text (not encrypted/hashed)
                if not self._appears_encrypted(value):
                    results.append(self._create_result(
                        False, ValidationSeverity.ERROR,
                        f"Potential sensitive data exposure in field '{key}'",
                        field_path=f"{path}.{key}" if path else key,
                        details={"field": key, "value_length": len(value)},
                        fix_suggestion="Encrypt sensitive data or use template variables"
                    ))
            
            # Check for patterns that look like sensitive data
            if isinstance(value, str):
                # API keys, tokens
                if re.match(r'^[A-Za-z0-9_-]{20,}$', value) and len(value) > 20:
                    results.append(self._create_result(
                        False, ValidationSeverity.WARNING,
                        f"Field '{key}' contains what appears to be an API key or token",
                        field_path=f"{path}.{key}" if path else key,
                        fix_suggestion="Use template variables for sensitive data"
                    ))
                
                # Email addresses in unexpected places
                if "@" in value and not key_lower.endswith(("email", "address")):
                    try:
                        validate_email(value)
                        results.append(self._create_result(
                            False, ValidationSeverity.INFO,
                            f"Email address found in field '{key}' - ensure this is intentional",
                            field_path=f"{path}.{key}" if path else key
                        ))
                    except EmailNotValidError:
                        pass
        
        def traverse(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_sensitive(k, v, path)
                    if isinstance(v, (dict, list)):
                        traverse(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        traverse(item, f"{path}[{i}]" if path else f"[{i}]")
        
        traverse(template)
        return results
    
    def _appears_encrypted(self, value: str) -> bool:
        """Check if a value appears to be encrypted or hashed."""
        # Common patterns for encrypted/hashed data
        patterns = [
            r'^{{.*}}$',  # Template variable
            r'^\$[0-9]+\$',  # Bcrypt/scrypt
            r'^[a-fA-F0-9]{32}$',  # MD5
            r'^[a-fA-F0-9]{40}$',  # SHA1
            r'^[a-fA-F0-9]{64}$',  # SHA256
            r'^[A-Za-z0-9+/]{32,}={0,2}$',  # Base64
        ]
        
        return any(re.match(pattern, value) for pattern in patterns)
    
    def _check_unsafe_expressions(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check for unsafe template expressions."""
        results = []
        
        unsafe_functions = [
            'eval', 'exec', 'compile', 'open', 'file', '__import__',
            'getattr', 'setattr', 'delattr', 'globals', 'locals',
            'vars', 'dir', 'hasattr'
        ]
        
        def check_expressions(value, path=""):
            if isinstance(value, str) and "{{" in value:
                # Extract template expressions
                expressions = re.findall(r'\{\{([^}]+)\}\}', value)
                
                for expr in expressions:
                    expr_clean = expr.strip()
                    
                    # Check for unsafe function calls
                    for unsafe_func in unsafe_functions:
                        if unsafe_func in expr_clean:
                            results.append(self._create_result(
                                False, ValidationSeverity.CRITICAL,
                                f"Unsafe function '{unsafe_func}' in template expression",
                                field_path=path,
                                details={"expression": expr, "function": unsafe_func},
                                fix_suggestion=f"Remove or replace unsafe function '{unsafe_func}'"
                            ))
                    
                    # Check for attribute access that might be dangerous
                    if re.search(r'\.__[a-zA-Z_]+__', expr_clean):
                        results.append(self._create_result(
                            False, ValidationSeverity.ERROR,
                            "Direct access to dunder attributes is potentially unsafe",
                            field_path=path,
                            details={"expression": expr},
                            fix_suggestion="Avoid direct access to private/magic attributes"
                        ))
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_expressions(v, f"{path}.{k}" if path else k)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_expressions(item, f"{path}[{i}]" if path else f"[{i}]")
        
        check_expressions(template)
        return results
    
    def _check_permissions(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check permissions and access control configuration."""
        results = []
        
        # Check for overly permissive configurations
        def check_permissions_config(obj, path=""):
            if isinstance(obj, dict):
                # Check for wildcard permissions
                for key, value in obj.items():
                    if "permission" in key.lower() and isinstance(value, (str, list)):
                        if isinstance(value, str) and value == "*":
                            results.append(self._create_result(
                                False, ValidationSeverity.WARNING,
                                "Wildcard permission detected - ensure this is intentional",
                                field_path=f"{path}.{key}" if path else key,
                                fix_suggestion="Use specific permissions instead of wildcards"
                            ))
                        elif isinstance(value, list) and "*" in value:
                            results.append(self._create_result(
                                False, ValidationSeverity.WARNING,
                                "Wildcard permission in list - ensure this is intentional",
                                field_path=f"{path}.{key}" if path else key,
                                fix_suggestion="Use specific permissions instead of wildcards"
                            ))
                
                # Recursively check nested objects
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        check_permissions_config(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        check_permissions_config(item, f"{path}[{i}]" if path else f"[{i}]")
        
        check_permissions_config(template)
        return results


class BusinessLogicValidator(BaseValidator):
    """Validates business logic and data consistency."""
    
    def validate(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate business logic and consistency."""
        results = []
        
        # Check tenant-specific validations
        results.extend(self._validate_tenant_logic(template))
        
        # Check user-specific validations
        results.extend(self._validate_user_logic(template))
        
        # Check content-specific validations
        results.extend(self._validate_content_logic(template))
        
        # Check cross-references and dependencies
        results.extend(self._validate_references(template, context))
        
        return results
    
    def _validate_tenant_logic(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Validate tenant-specific business logic."""
        results = []
        
        if "tier" in template and "configuration" in template:
            tier = template["tier"]
            config = template["configuration"]
            
            # Validate tier-based limits
            if "limits" in config:
                limits = config["limits"]
                
                # Check logical consistency of limits
                if tier == "starter":
                    if limits.get("max_users", 0) > 10:
                        results.append(self._create_result(
                            False, ValidationSeverity.ERROR,
                            "Starter tier cannot have more than 10 users",
                            field_path="configuration.limits.max_users",
                            fix_suggestion="Reduce max_users to 10 or upgrade tier"
                        ))
                
                # Check for negative values where they don't make sense
                for limit_key, limit_value in limits.items():
                    if isinstance(limit_value, int) and limit_value < -1:
                        results.append(self._create_result(
                            False, ValidationSeverity.ERROR,
                            f"Invalid limit value: {limit_key} cannot be less than -1",
                            field_path=f"configuration.limits.{limit_key}",
                            fix_suggestion="Use -1 for unlimited or a positive value"
                        ))
        
        return results
    
    def _validate_user_logic(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Validate user-specific business logic."""
        results = []
        
        # Check email format if present
        if "email" in template:
            try:
                validate_email(template["email"])
            except EmailNotValidError as e:
                results.append(self._create_result(
                    False, ValidationSeverity.ERROR,
                    f"Invalid email format: {str(e)}",
                    field_path="email",
                    fix_suggestion="Provide a valid email address"
                ))
        
        # Check role validity
        if "role" in template:
            valid_roles = ["admin", "manager", "user", "guest"]
            if template["role"] not in valid_roles:
                results.append(self._create_result(
                    False, ValidationSeverity.ERROR,
                    f"Invalid role: {template['role']}",
                    field_path="role",
                    fix_suggestion=f"Use one of: {', '.join(valid_roles)}"
                ))
        
        return results
    
    def _validate_content_logic(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Validate content-specific business logic."""
        results = []
        
        if "content_types" in template:
            content_types = template["content_types"]
            
            for content_type_name, content_type in content_types.items():
                # Validate field definitions
                if "fields" in content_type:
                    for field_name, field_def in content_type["fields"].items():
                        # Check required field definition
                        if not isinstance(field_def, dict):
                            results.append(self._create_result(
                                False, ValidationSeverity.ERROR,
                                f"Field definition must be an object: {field_name}",
                                field_path=f"content_types.{content_type_name}.fields.{field_name}",
                                fix_suggestion="Provide proper field definition with type and constraints"
                            ))
                            continue
                        
                        # Validate field type
                        if "type" not in field_def:
                            results.append(self._create_result(
                                False, ValidationSeverity.ERROR,
                                f"Field type is required: {field_name}",
                                field_path=f"content_types.{content_type_name}.fields.{field_name}",
                                fix_suggestion="Add 'type' property to field definition"
                            ))
        
        return results
    
    def _validate_references(self, template: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate cross-references and dependencies."""
        results = []
        
        # Check tenant_id consistency
        tenant_ids = []
        
        def collect_tenant_ids(obj, path=""):
            if isinstance(obj, dict):
                if "tenant_id" in obj:
                    tenant_ids.append((obj["tenant_id"], path))
                for k, v in obj.items():
                    collect_tenant_ids(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    collect_tenant_ids(item, f"{path}[{i}]" if path else f"[{i}]")
        
        collect_tenant_ids(template)
        
        # Check for inconsistent tenant_ids
        if len(set(tid[0] for tid in tenant_ids)) > 1:
            results.append(self._create_result(
                False, ValidationSeverity.ERROR,
                "Inconsistent tenant_id values found in template",
                details={"tenant_ids": [tid[0] for tid in tenant_ids]},
                fix_suggestion="Ensure all tenant_id fields have the same value"
            ))
        
        return results


class PerformanceValidator(BaseValidator):
    """Validates template performance characteristics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_template_size = config.get("max_template_size", 1024 * 1024)  # 1MB
        self.max_depth = config.get("max_depth", 20)
        self.max_array_length = config.get("max_array_length", 1000)
    
    def validate(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate template performance characteristics."""
        results = []
        
        # Check template size
        results.extend(self._check_template_size(template))
        
        # Check nesting depth
        results.extend(self._check_nesting_depth(template))
        
        # Check array lengths
        results.extend(self._check_array_lengths(template))
        
        # Check template complexity
        results.extend(self._check_template_complexity(template))
        
        return results
    
    def _check_template_size(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check if template size is within limits."""
        results = []
        
        template_json = json.dumps(template)
        size_bytes = len(template_json.encode('utf-8'))
        
        if size_bytes > self.max_template_size:
            results.append(self._create_result(
                False, ValidationSeverity.WARNING,
                f"Template size ({size_bytes} bytes) exceeds recommended limit ({self.max_template_size} bytes)",
                details={"size_bytes": size_bytes, "limit_bytes": self.max_template_size},
                fix_suggestion="Consider breaking template into smaller components"
            ))
        
        return results
    
    def _check_nesting_depth(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check template nesting depth."""
        results = []
        
        def get_depth(obj, current_depth=0):
            if isinstance(obj, dict):
                if not obj:  # Empty dict
                    return current_depth
                return max(get_depth(v, current_depth + 1) for v in obj.values())
            elif isinstance(obj, list):
                if not obj:  # Empty list
                    return current_depth
                return max(get_depth(item, current_depth + 1) for item in obj)
            else:
                return current_depth
        
        max_depth = get_depth(template)
        
        if max_depth > self.max_depth:
            results.append(self._create_result(
                False, ValidationSeverity.WARNING,
                f"Template nesting depth ({max_depth}) exceeds recommended limit ({self.max_depth})",
                details={"depth": max_depth, "limit": self.max_depth},
                fix_suggestion="Reduce nesting depth by flattening structure or using references"
            ))
        
        return results
    
    def _check_array_lengths(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check array lengths in template."""
        results = []
        
        def check_arrays(obj, path=""):
            if isinstance(obj, list):
                if len(obj) > self.max_array_length:
                    results.append(self._create_result(
                        False, ValidationSeverity.WARNING,
                        f"Array length ({len(obj)}) exceeds recommended limit ({self.max_array_length})",
                        field_path=path,
                        details={"length": len(obj), "limit": self.max_array_length},
                        fix_suggestion="Consider pagination or chunking for large arrays"
                    ))
                
                for i, item in enumerate(obj):
                    check_arrays(item, f"{path}[{i}]" if path else f"[{i}]")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_arrays(v, f"{path}.{k}" if path else k)
        
        check_arrays(template)
        return results
    
    def _check_template_complexity(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check template complexity metrics."""
        results = []
        
        # Count template expressions
        expression_count = 0
        
        def count_expressions(obj):
            nonlocal expression_count
            if isinstance(obj, str):
                expression_count += len(re.findall(r'\{\{[^}]+\}\}', obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    count_expressions(v)
            elif isinstance(obj, list):
                for item in obj:
                    count_expressions(item)
        
        count_expressions(template)
        
        if expression_count > 100:
            results.append(self._create_result(
                False, ValidationSeverity.INFO,
                f"High number of template expressions ({expression_count}) may impact performance",
                details={"expression_count": expression_count},
                fix_suggestion="Consider simplifying template or pre-computing values"
            ))
        
        return results


class ComplianceValidator(BaseValidator):
    """Validates template compliance with regulations (GDPR, etc.)."""
    
    def validate(self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate template compliance."""
        results = []
        
        # Check GDPR compliance
        results.extend(self._check_gdpr_compliance(template))
        
        # Check data retention policies
        results.extend(self._check_data_retention(template))
        
        # Check consent mechanisms
        results.extend(self._check_consent_mechanisms(template))
        
        return results
    
    def _check_gdpr_compliance(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check GDPR compliance requirements."""
        results = []
        
        personal_data_fields = [
            'email', 'first_name', 'last_name', 'phone', 'address',
            'ip_address', 'user_agent', 'location', 'bio'
        ]
        
        # Check for personal data handling
        def check_personal_data(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() in personal_data_fields:
                        # Check if there's indication of proper handling
                        if not self._has_privacy_controls(template):
                            results.append(self._create_result(
                                False, ValidationSeverity.WARNING,
                                f"Personal data field '{key}' found without privacy controls",
                                field_path=f"{path}.{key}" if path else key,
                                fix_suggestion="Add privacy controls and consent mechanisms"
                            ))
                    
                    check_personal_data(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_personal_data(item, f"{path}[{i}]" if path else f"[{i}]")
        
        check_personal_data(template)
        return results
    
    def _has_privacy_controls(self, template: Dict[str, Any]) -> bool:
        """Check if template has privacy controls."""
        privacy_indicators = [
            'privacy', 'consent', 'gdpr', 'data_retention',
            'anonymization', 'pseudonymization'
        ]
        
        template_str = json.dumps(template).lower()
        return any(indicator in template_str for indicator in privacy_indicators)
    
    def _check_data_retention(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check data retention policies."""
        results = []
        
        def check_retention(obj, path=""):
            if isinstance(obj, dict):
                # Look for retention-related fields
                retention_fields = [k for k in obj.keys() if 'retention' in k.lower()]
                
                for field in retention_fields:
                    value = obj[field]
                    if isinstance(value, int) and value > 2555:  # > 7 years in days
                        results.append(self._create_result(
                            False, ValidationSeverity.WARNING,
                            f"Data retention period ({value} days) may be excessive",
                            field_path=f"{path}.{field}" if path else field,
                            fix_suggestion="Consider shorter retention periods for compliance"
                        ))
                
                for k, v in obj.items():
                    check_retention(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_retention(item, f"{path}[{i}]" if path else f"[{i}]")
        
        check_retention(template)
        return results
    
    def _check_consent_mechanisms(self, template: Dict[str, Any]) -> List[ValidationResult]:
        """Check for proper consent mechanisms."""
        results = []
        
        # Look for user-related templates without consent mechanisms
        if ("user" in str(template).lower() and 
            "consent" not in str(template).lower() and 
            "privacy" not in str(template).lower()):
            
            results.append(self._create_result(
                False, ValidationSeverity.INFO,
                "User template may need consent mechanisms for GDPR compliance",
                fix_suggestion="Add consent tracking and privacy preference fields"
            ))
        
        return results


class TemplateValidationEngine:
    """Main template validation engine that orchestrates all validators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validators = self._initialize_validators()
    
    def _initialize_validators(self) -> List[BaseValidator]:
        """Initialize all validators."""
        return [
            SchemaValidator(self.config.get("schema", {})),
            SecurityValidator(self.config.get("security", {})),
            BusinessLogicValidator(self.config.get("business_logic", {})),
            PerformanceValidator(self.config.get("performance", {})),
            ComplianceValidator(self.config.get("compliance", {}))
        ]
    
    def validate_template(
        self,
        template: Dict[str, Any],
        template_id: str,
        template_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """Validate template using all validators."""
        start_time = datetime.now()
        all_results = []
        
        # Run all validators
        for validator in self.validators:
            try:
                validator_start = datetime.now()
                results = validator.validate(template, context)
                validator_end = datetime.now()
                
                all_results.extend(results)
                
                logger.debug(f"{validator.name} completed in {(validator_end - validator_start).total_seconds():.3f}s")
            
            except Exception as e:
                logger.error(f"Validator {validator.name} failed: {str(e)}")
                all_results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Validator {validator.name} failed: {str(e)}",
                    details={"exception": str(e)}
                ))
        
        end_time = datetime.now()
        
        # Determine overall validity
        is_valid = not any(result.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for result in all_results if not result.is_valid)
        
        # Calculate performance metrics
        performance_metrics = {
            "total_validation_time_ms": (end_time - start_time).total_seconds() * 1000,
            "validators_run": len(self.validators),
            "total_checks": len(all_results)
        }
        
        return ValidationReport(
            template_id=template_id,
            template_type=template_type,
            is_valid=is_valid,
            total_issues=len([r for r in all_results if not r.is_valid]),
            results=all_results,
            validation_timestamp=datetime.now(timezone.utc),
            performance_metrics=performance_metrics
        )
    
    def add_validator(self, validator: BaseValidator):
        """Add custom validator to the engine."""
        self.validators.append(validator)
    
    def remove_validator(self, validator_class):
        """Remove validator by class."""
        self.validators = [v for v in self.validators if not isinstance(v, validator_class)]


# Utility functions for validation
def validate_template_batch(
    templates: List[Tuple[str, str, Dict[str, Any]]],
    config: Optional[Dict[str, Any]] = None
) -> List[ValidationReport]:
    """Validate multiple templates in batch."""
    engine = TemplateValidationEngine(config)
    reports = []
    
    for template_id, template_type, template in templates:
        report = engine.validate_template(template, template_id, template_type)
        reports.append(report)
    
    return reports


def create_validation_summary(reports: List[ValidationReport]) -> Dict[str, Any]:
    """Create summary of validation results."""
    total_templates = len(reports)
    valid_templates = len([r for r in reports if r.is_valid])
    total_issues = sum(r.total_issues for r in reports)
    
    severity_counts = {
        "critical": 0,
        "error": 0,
        "warning": 0,
        "info": 0
    }
    
    for report in reports:
        for result in report.results:
            if not result.is_valid:
                severity_counts[result.severity.value] += 1
    
    return {
        "total_templates": total_templates,
        "valid_templates": valid_templates,
        "invalid_templates": total_templates - valid_templates,
        "total_issues": total_issues,
        "issues_by_severity": severity_counts,
        "validation_success_rate": (valid_templates / total_templates * 100) if total_templates > 0 else 0
    }
