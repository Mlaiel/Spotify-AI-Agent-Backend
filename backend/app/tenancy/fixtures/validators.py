"""
Spotify AI Agent - Fixture Validators
===================================

Comprehensive validation system for fixture data integrity,
business rules, and compliance requirements.
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.tenancy.fixtures.exceptions import (
    FixtureValidationError,
    FixtureDataError,
    FixtureConflictError
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Validation category types."""
    DATA_TYPE = "data_type"
    BUSINESS_RULE = "business_rule"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    validator_name: str
    level: ValidationLevel
    category: ValidationCategory
    message: str
    field_name: Optional[str] = None
    record_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "validator": self.validator_name,
            "level": self.level.value,
            "category": self.category.value,
            "message": self.message,
            "field": self.field_name,
            "record_id": self.record_id,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    total_checks: int = 0
    passed_checks: int = 0
    info_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    critical_count: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    
    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)
        self.total_checks += 1
        
        if result.level == ValidationLevel.INFO:
            self.info_count += 1
        elif result.level == ValidationLevel.WARNING:
            self.warning_count += 1
        elif result.level == ValidationLevel.ERROR:
            self.error_count += 1
        elif result.level == ValidationLevel.CRITICAL:
            self.critical_count += 1
    
    def is_valid(self) -> bool:
        """Check if validation passed (no errors or critical issues)."""
        return self.error_count == 0 and self.critical_count == 0
    
    def get_errors(self) -> List[ValidationResult]:
        """Get all error and critical results."""
        return [r for r in self.results 
                if r.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "summary": {
                "info": self.info_count,
                "warning": self.warning_count,
                "error": self.error_count,
                "critical": self.critical_count
            },
            "is_valid": self.is_valid(),
            "results": [r.to_dict() for r in self.results]
        }


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, name: str, category: ValidationCategory):
        self.name = name
        self.category = category
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def validate(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Perform validation and return results."""
        pass
    
    def create_result(
        self,
        level: ValidationLevel,
        message: str,
        field_name: Optional[str] = None,
        record_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Create a validation result."""
        return ValidationResult(
            validator_name=self.name,
            level=level,
            category=self.category,
            message=message,
            field_name=field_name,
            record_id=record_id,
            details=details or {}
        )


class DataTypeValidator(BaseValidator):
    """Validator for data types and formats."""
    
    def __init__(self):
        super().__init__("data_type_validator", ValidationCategory.DATA_TYPE)
    
    async def validate(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate data types."""
        results = []
        schema = context.get("schema", {}) if context else {}
        
        for field_name, value in data.items():
            field_schema = schema.get(field_name, {})
            expected_type = field_schema.get("type")
            
            if expected_type:
                type_result = await self._validate_type(field_name, value, expected_type)
                if type_result:
                    results.append(type_result)
            
            # Additional format validations
            format_results = await self._validate_formats(field_name, value, field_schema)
            results.extend(format_results)
        
        return results
    
    async def _validate_type(
        self,
        field_name: str,
        value: Any,
        expected_type: str
    ) -> Optional[ValidationResult]:
        """Validate field type."""
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int),
            "float": lambda v: isinstance(v, (int, float)),
            "boolean": lambda v: isinstance(v, bool),
            "list": lambda v: isinstance(v, list),
            "dict": lambda v: isinstance(v, dict),
            "uuid": lambda v: self._is_valid_uuid(v),
            "email": lambda v: self._is_valid_email(v),
            "url": lambda v: self._is_valid_url(v),
            "datetime": lambda v: self._is_valid_datetime(v)
        }
        
        if expected_type in type_checks:
            if not type_checks[expected_type](value):
                return self.create_result(
                    ValidationLevel.ERROR,
                    f"Invalid type for field {field_name}. Expected {expected_type}, got {type(value).__name__}",
                    field_name=field_name,
                    details={"expected_type": expected_type, "actual_type": type(value).__name__}
                )
        
        return None
    
    async def _validate_formats(
        self,
        field_name: str,
        value: Any,
        field_schema: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate field formats."""
        results = []
        
        # String length validation
        if isinstance(value, str):
            min_length = field_schema.get("min_length")
            max_length = field_schema.get("max_length")
            
            if min_length and len(value) < min_length:
                results.append(self.create_result(
                    ValidationLevel.ERROR,
                    f"Field {field_name} too short. Minimum length: {min_length}",
                    field_name=field_name
                ))
            
            if max_length and len(value) > max_length:
                results.append(self.create_result(
                    ValidationLevel.ERROR,
                    f"Field {field_name} too long. Maximum length: {max_length}",
                    field_name=field_name
                ))
            
            # Pattern validation
            pattern = field_schema.get("pattern")
            if pattern and not re.match(pattern, value):
                results.append(self.create_result(
                    ValidationLevel.ERROR,
                    f"Field {field_name} does not match required pattern",
                    field_name=field_name,
                    details={"pattern": pattern}
                ))
        
        # Numeric range validation
        if isinstance(value, (int, float)):
            min_value = field_schema.get("min_value")
            max_value = field_schema.get("max_value")
            
            if min_value is not None and value < min_value:
                results.append(self.create_result(
                    ValidationLevel.ERROR,
                    f"Field {field_name} below minimum value: {min_value}",
                    field_name=field_name
                ))
            
            if max_value is not None and value > max_value:
                results.append(self.create_result(
                    ValidationLevel.ERROR,
                    f"Field {field_name} above maximum value: {max_value}",
                    field_name=field_name
                ))
        
        return results
    
    def _is_valid_uuid(self, value: Any) -> bool:
        """Check if value is a valid UUID."""
        if isinstance(value, UUID):
            return True
        if isinstance(value, str):
            try:
                UUID(value)
                return True
            except ValueError:
                return False
        return False
    
    def _is_valid_email(self, value: Any) -> bool:
        """Check if value is a valid email."""
        if not isinstance(value, str):
            return False
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, value))
    
    def _is_valid_url(self, value: Any) -> bool:
        """Check if value is a valid URL."""
        if not isinstance(value, str):
            return False
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, value))
    
    def _is_valid_datetime(self, value: Any) -> bool:
        """Check if value is a valid datetime."""
        if isinstance(value, datetime):
            return True
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return True
            except ValueError:
                return False
        return False


class BusinessRuleValidator(BaseValidator):
    """Validator for business rules and logic."""
    
    def __init__(self, tenant_id: str):
        super().__init__("business_rule_validator", ValidationCategory.BUSINESS_RULE)
        self.tenant_id = tenant_id
    
    async def validate(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate business rules."""
        results = []
        data_type = context.get("data_type") if context else None
        
        if data_type == "tenant":
            results.extend(await self._validate_tenant_rules(data))
        elif data_type == "user":
            results.extend(await self._validate_user_rules(data))
        elif data_type == "spotify_connection":
            results.extend(await self._validate_spotify_rules(data))
        elif data_type == "collaboration":
            results.extend(await self._validate_collaboration_rules(data))
        
        return results
    
    async def _validate_tenant_rules(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate tenant-specific business rules."""
        results = []
        
        # Tenant ID format validation
        tenant_id = data.get("tenant_id")
        if tenant_id:
            if not re.match(r'^[a-z0-9_-]+$', tenant_id):
                results.append(self.create_result(
                    ValidationLevel.ERROR,
                    "Tenant ID must contain only lowercase letters, numbers, hyphens, and underscores",
                    field_name="tenant_id"
                ))
            
            if len(tenant_id) < 3 or len(tenant_id) > 50:
                results.append(self.create_result(
                    ValidationLevel.ERROR,
                    "Tenant ID must be between 3 and 50 characters",
                    field_name="tenant_id"
                ))
        
        # Tier validation
        tier = data.get("tier")
        if tier and tier not in ["free", "basic", "premium", "enterprise"]:
            results.append(self.create_result(
                ValidationLevel.ERROR,
                "Invalid tenant tier",
                field_name="tier",
                details={"allowed_tiers": ["free", "basic", "premium", "enterprise"]}
            ))
        
        # Domain validation for premium+ tiers
        domain = data.get("domain")
        if tier in ["premium", "enterprise"] and not domain:
            results.append(self.create_result(
                ValidationLevel.WARNING,
                "Premium and Enterprise tiers should have a custom domain",
                field_name="domain"
            ))
        
        return results
    
    async def _validate_user_rules(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate user-specific business rules."""
        results = []
        
        # Email domain validation for enterprise tenants
        email = data.get("email")
        if email and self.tenant_id.startswith("enterprise_"):
            domain = email.split("@")[1] if "@" in email else ""
            if not domain.endswith(".corp") and not domain.endswith(".enterprise"):
                results.append(self.create_result(
                    ValidationLevel.WARNING,
                    "Enterprise users should use corporate email domains",
                    field_name="email"
                ))
        
        # Role validation
        role = data.get("role")
        valid_roles = ["user", "admin", "moderator", "collaborator"]
        if role and role not in valid_roles:
            results.append(self.create_result(
                ValidationLevel.ERROR,
                "Invalid user role",
                field_name="role",
                details={"allowed_roles": valid_roles}
            ))
        
        return results
    
    async def _validate_spotify_rules(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate Spotify connection rules."""
        results = []
        
        # Token expiration validation
        expires_at = data.get("token_expires_at")
        if expires_at:
            if isinstance(expires_at, str):
                try:
                    expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                except ValueError:
                    results.append(self.create_result(
                        ValidationLevel.ERROR,
                        "Invalid token expiration date format",
                        field_name="token_expires_at"
                    ))
                    return results
            
            if expires_at < datetime.now(timezone.utc):
                results.append(self.create_result(
                    ValidationLevel.WARNING,
                    "Spotify token has already expired",
                    field_name="token_expires_at"
                ))
        
        # Scopes validation
        scopes = data.get("scopes", [])
        required_scopes = ["user-read-private", "user-read-email"]
        missing_scopes = [scope for scope in required_scopes if scope not in scopes]
        
        if missing_scopes:
            results.append(self.create_result(
                ValidationLevel.WARNING,
                "Missing recommended Spotify scopes",
                field_name="scopes",
                details={"missing_scopes": missing_scopes}
            ))
        
        return results
    
    async def _validate_collaboration_rules(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate collaboration rules."""
        results = []
        
        # Collaboration type validation
        collab_type = data.get("collaboration_type")
        valid_types = ["songwriting", "production", "mixing", "mastering", "general"]
        if collab_type and collab_type not in valid_types:
            results.append(self.create_result(
                ValidationLevel.ERROR,
                "Invalid collaboration type",
                field_name="collaboration_type",
                details={"allowed_types": valid_types}
            ))
        
        # Title validation
        title = data.get("title")
        if title and len(title) < 5:
            results.append(self.create_result(
                ValidationLevel.WARNING,
                "Collaboration title should be more descriptive",
                field_name="title"
            ))
        
        return results


class ReferentialIntegrityValidator(BaseValidator):
    """Validator for referential integrity and foreign key constraints."""
    
    def __init__(self, session: AsyncSession):
        super().__init__("referential_integrity_validator", ValidationCategory.REFERENTIAL_INTEGRITY)
        self.session = session
    
    async def validate(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate referential integrity."""
        results = []
        schema = context.get("schema") if context else "public"
        table = context.get("table") if context else None
        
        if not table:
            return results
        
        # Check foreign key references
        foreign_keys = context.get("foreign_keys", []) if context else []
        for fk in foreign_keys:
            result = await self._validate_foreign_key(data, fk, schema)
            if result:
                results.append(result)
        
        # Check unique constraints
        unique_fields = context.get("unique_fields", []) if context else []
        for field in unique_fields:
            result = await self._validate_unique_constraint(data, field, table, schema)
            if result:
                results.append(result)
        
        return results
    
    async def _validate_foreign_key(
        self,
        data: Dict[str, Any],
        foreign_key: Dict[str, str],
        schema: str
    ) -> Optional[ValidationResult]:
        """Validate foreign key reference."""
        field_name = foreign_key["field"]
        ref_table = foreign_key["ref_table"]
        ref_field = foreign_key["ref_field"]
        
        value = data.get(field_name)
        if value is None:
            return None  # NULL values are allowed for foreign keys
        
        # Check if referenced record exists
        query = f"""
        SELECT 1 FROM {schema}.{ref_table} 
        WHERE {ref_field} = :value
        """
        
        try:
            result = await self.session.execute(text(query), {"value": value})
            if not result.first():
                return self.create_result(
                    ValidationLevel.ERROR,
                    f"Referenced record not found in {ref_table}.{ref_field}",
                    field_name=field_name,
                    details={
                        "ref_table": ref_table,
                        "ref_field": ref_field,
                        "value": value
                    }
                )
        except Exception as e:
            return self.create_result(
                ValidationLevel.ERROR,
                f"Error validating foreign key: {str(e)}",
                field_name=field_name
            )
        
        return None
    
    async def _validate_unique_constraint(
        self,
        data: Dict[str, Any],
        field_name: str,
        table: str,
        schema: str
    ) -> Optional[ValidationResult]:
        """Validate unique constraint."""
        value = data.get(field_name)
        if value is None:
            return None
        
        # Check for existing record with same value
        query = f"""
        SELECT 1 FROM {schema}.{table} 
        WHERE {field_name} = :value
        """
        
        try:
            result = await self.session.execute(text(query), {"value": value})
            if result.first():
                return self.create_result(
                    ValidationLevel.ERROR,
                    f"Duplicate value found for unique field {field_name}",
                    field_name=field_name,
                    details={"value": value}
                )
        except Exception as e:
            return self.create_result(
                ValidationLevel.ERROR,
                f"Error validating unique constraint: {str(e)}",
                field_name=field_name
            )
        
        return None


class SecurityValidator(BaseValidator):
    """Validator for security requirements and sensitive data."""
    
    def __init__(self):
        super().__init__("security_validator", ValidationCategory.SECURITY)
        self.sensitive_patterns = [
            r'password', r'secret', r'token', r'key', r'credential',
            r'private', r'confidential', r'restricted'
        ]
    
    async def validate(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate security requirements."""
        results = []
        
        for field_name, value in data.items():
            # Check for sensitive data in plain text
            if self._is_sensitive_field(field_name) and isinstance(value, str):
                if not self._appears_encrypted(value):
                    results.append(self.create_result(
                        ValidationLevel.WARNING,
                        f"Sensitive field {field_name} appears to be stored in plain text",
                        field_name=field_name
                    ))
            
            # Password strength validation
            if "password" in field_name.lower() and isinstance(value, str):
                strength_result = await self._validate_password_strength(field_name, value)
                if strength_result:
                    results.append(strength_result)
            
            # Check for potential injection patterns
            injection_result = await self._check_injection_patterns(field_name, value)
            if injection_result:
                results.append(injection_result)
        
        return results
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if field contains sensitive data."""
        field_lower = field_name.lower()
        return any(pattern in field_lower for pattern in self.sensitive_patterns)
    
    def _appears_encrypted(self, value: str) -> bool:
        """Check if value appears to be encrypted."""
        # Simple heuristics for encrypted data
        if len(value) < 10:
            return False
        
        # Check for base64-like patterns
        if re.match(r'^[A-Za-z0-9+/=]+$', value) and len(value) % 4 == 0:
            return True
        
        # Check for hash-like patterns
        if re.match(r'^[a-f0-9]{32,}$', value):
            return True
        
        # Check for encrypted prefixes
        encrypted_prefixes = ['enc_', 'hash_', 'crypt_', '$2b$', '$argon2']
        return any(value.startswith(prefix) for prefix in encrypted_prefixes)
    
    async def _validate_password_strength(
        self,
        field_name: str,
        password: str
    ) -> Optional[ValidationResult]:
        """Validate password strength."""
        issues = []
        
        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        
        if not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            issues.append("Password must contain at least one number")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one special character")
        
        # Check for common patterns
        common_patterns = [
            r'123456', r'password', r'qwerty', r'admin', r'root'
        ]
        for pattern in common_patterns:
            if re.search(pattern, password, re.IGNORECASE):
                issues.append("Password contains common patterns")
                break
        
        if issues:
            return self.create_result(
                ValidationLevel.WARNING,
                f"Weak password detected: {'; '.join(issues)}",
                field_name=field_name
            )
        
        return None
    
    async def _check_injection_patterns(
        self,
        field_name: str,
        value: Any
    ) -> Optional[ValidationResult]:
        """Check for potential injection attack patterns."""
        if not isinstance(value, str):
            return None
        
        # SQL injection patterns
        sql_patterns = [
            r"';.*--", r"union\s+select", r"drop\s+table",
            r"insert\s+into", r"delete\s+from", r"update\s+.*set"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return self.create_result(
                    ValidationLevel.CRITICAL,
                    f"Potential SQL injection pattern detected in field {field_name}",
                    field_name=field_name,
                    details={"pattern": pattern}
                )
        
        # XSS patterns
        xss_patterns = [
            r"<script.*?>", r"javascript:", r"on\w+\s*=",
            r"<iframe.*?>", r"<object.*?>"
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return self.create_result(
                    ValidationLevel.CRITICAL,
                    f"Potential XSS pattern detected in field {field_name}",
                    field_name=field_name,
                    details={"pattern": pattern}
                )
        
        return None


class FixtureValidator:
    """
    Main fixture validator that orchestrates all validation checks.
    
    Provides:
    - Comprehensive data validation
    - Configurable validation rules
    - Performance monitoring
    - Detailed reporting
    """
    
    def __init__(self, tenant_id: str, session: Optional[AsyncSession] = None):
        self.tenant_id = tenant_id
        self.session = session
        self.validators: List[BaseValidator] = []
        
        # Initialize default validators
        self._initialize_validators()
        
        self.logger = logging.getLogger(f"{__name__}.FixtureValidator")
    
    def _initialize_validators(self) -> None:
        """Initialize default validators."""
        self.validators = [
            DataTypeValidator(),
            BusinessRuleValidator(self.tenant_id),
            SecurityValidator()
        ]
        
        # Add referential integrity validator if session available
        if self.session:
            self.validators.append(ReferentialIntegrityValidator(self.session))
    
    def add_validator(self, validator: BaseValidator) -> None:
        """Add a custom validator."""
        self.validators.append(validator)
    
    async def validate_data(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationSummary:
        """
        Validate data using all configured validators.
        
        Args:
            data: Data to validate (single record or list of records)
            context: Validation context with schema info, etc.
            
        Returns:
            ValidationSummary: Complete validation results
        """
        summary = ValidationSummary()
        
        # Normalize data to list
        records = data if isinstance(data, list) else [data]
        
        for record_idx, record in enumerate(records):
            record_id = record.get("id") or str(record_idx)
            
            # Run all validators on the record
            for validator in self.validators:
                try:
                    results = await validator.validate(record, context)
                    for result in results:
                        result.record_id = record_id
                        summary.add_result(result)
                        
                except Exception as e:
                    # Create error result for validator failure
                    error_result = ValidationResult(
                        validator_name=validator.name,
                        level=ValidationLevel.CRITICAL,
                        category=validator.category,
                        message=f"Validator failed: {str(e)}",
                        record_id=record_id
                    )
                    summary.add_result(error_result)
                    self.logger.error(f"Validator {validator.name} failed: {e}")
        
        # Calculate passed checks
        summary.passed_checks = summary.total_checks - summary.error_count - summary.critical_count
        
        self.logger.info(
            f"Validation completed: {summary.total_checks} checks, "
            f"{summary.error_count} errors, {summary.warning_count} warnings"
        )
        
        return summary
    
    async def validate_batch(
        self,
        records: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        batch_size: int = 100
    ) -> ValidationSummary:
        """Validate large datasets in batches."""
        summary = ValidationSummary()
        
        # Process records in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            batch_summary = await self.validate_data(batch, context)
            
            # Merge results
            summary.total_checks += batch_summary.total_checks
            summary.passed_checks += batch_summary.passed_checks
            summary.info_count += batch_summary.info_count
            summary.warning_count += batch_summary.warning_count
            summary.error_count += batch_summary.error_count
            summary.critical_count += batch_summary.critical_count
            summary.results.extend(batch_summary.results)
            
            self.logger.debug(f"Processed batch {i//batch_size + 1} of {len(records)//batch_size + 1}")
        
        return summary


class DataIntegrityValidator:
    """
    Specialized validator for data integrity across the entire dataset.
    
    Performs cross-record validations and integrity checks.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.DataIntegrityValidator")
    
    async def validate_tenant_integrity(self, tenant_id: str) -> ValidationSummary:
        """Validate overall data integrity for a tenant."""
        summary = ValidationSummary()
        
        # Check orphaned records
        orphan_results = await self._check_orphaned_records(tenant_id)
        for result in orphan_results:
            summary.add_result(result)
        
        # Check data consistency
        consistency_results = await self._check_data_consistency(tenant_id)
        for result in consistency_results:
            summary.add_result(result)
        
        # Check resource limits
        limit_results = await self._check_resource_limits(tenant_id)
        for result in limit_results:
            summary.add_result(result)
        
        return summary
    
    async def _check_orphaned_records(self, tenant_id: str) -> List[ValidationResult]:
        """Check for orphaned records."""
        results = []
        schema = f"tenant_{tenant_id}"
        
        # Check for users without Spotify connections in premium+ tiers
        query = f"""
        SELECT COUNT(*) as count FROM {schema}.users u
        LEFT JOIN {schema}.spotify_connections sc ON u.id = sc.user_id
        WHERE sc.id IS NULL AND u.role != 'admin'
        """
        
        try:
            result = await self.session.execute(text(query))
            count = result.scalar()
            
            if count > 0:
                results.append(ValidationResult(
                    validator_name="data_integrity_validator",
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.REFERENTIAL_INTEGRITY,
                    message=f"Found {count} users without Spotify connections",
                    details={"orphaned_users": count}
                ))
        except Exception as e:
            self.logger.error(f"Error checking orphaned records: {e}")
        
        return results
    
    async def _check_data_consistency(self, tenant_id: str) -> List[ValidationResult]:
        """Check data consistency across tables."""
        results = []
        schema = f"tenant_{tenant_id}"
        
        # Check for inconsistent collaboration participant counts
        query = f"""
        SELECT c.id, c.title, 
               (SELECT COUNT(*) FROM {schema}.collaboration_participants cp 
                WHERE cp.collaboration_id = c.id) as participant_count
        FROM {schema}.collaborations c
        WHERE c.status = 'active'
        """
        
        try:
            result = await self.session.execute(text(query))
            for row in result:
                if row.participant_count == 0:
                    results.append(ValidationResult(
                        validator_name="data_integrity_validator",
                        level=ValidationLevel.WARNING,
                        category=ValidationCategory.BUSINESS_RULE,
                        message=f"Active collaboration '{row.title}' has no participants",
                        record_id=str(row.id)
                    ))
        except Exception as e:
            self.logger.error(f"Error checking data consistency: {e}")
        
        return results
    
    async def _check_resource_limits(self, tenant_id: str) -> List[ValidationResult]:
        """Check resource usage against limits."""
        results = []
        
        # This would check against tenant limits
        # Implementation depends on specific business rules
        
        return results
