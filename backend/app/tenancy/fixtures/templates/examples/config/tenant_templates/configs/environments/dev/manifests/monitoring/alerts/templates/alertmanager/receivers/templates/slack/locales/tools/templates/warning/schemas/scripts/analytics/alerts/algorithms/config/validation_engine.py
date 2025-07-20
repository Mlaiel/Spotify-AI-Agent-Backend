"""
Advanced Validation Engine for Spotify AI Agent Configuration Management

This module provides comprehensive validation capabilities including schema validation,
business rule validation, performance constraint validation, and security compliance
validation for music streaming platform configurations.

Author: Fahed Mlaiel (Expert Backend Developer & ML Engineer)
Version: 2.0.0 (Enterprise Edition)
"""

import re
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable, Type, Set, Tuple
from collections import defaultdict
import ipaddress
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationCategory(Enum):
    """Categories of validation rules"""
    SCHEMA = "schema"
    BUSINESS_LOGIC = "business_logic"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    INFRASTRUCTURE = "infrastructure"
    MUSIC_STREAMING = "music_streaming"


@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    severity: ValidationSeverity
    category: ValidationCategory
    rule_name: str
    message: str
    field_path: Optional[str] = None
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report"""
    config_name: str
    validation_timestamp: datetime
    overall_valid: bool
    results: List[ValidationResult]
    summary: Dict[ValidationSeverity, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate summary statistics"""
        self.summary = defaultdict(int)
        for result in self.results:
            self.summary[result.severity] += 1
        
        # Determine overall validity
        self.overall_valid = (
            self.summary[ValidationSeverity.CRITICAL] == 0 and
            self.summary[ValidationSeverity.ERROR] == 0
        )


class ValidationRule(ABC):
    """Abstract base class for validation rules"""
    
    def __init__(self, name: str, category: ValidationCategory, severity: ValidationSeverity):
        self.name = name
        self.category = category
        self.severity = severity
    
    @abstractmethod
    def validate(self, config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate configuration and return results"""
        pass
    
    def create_result(self, is_valid: bool, message: str, field_path: Optional[str] = None, 
                     suggested_fix: Optional[str] = None, **metadata) -> ValidationResult:
        """Create validation result"""
        return ValidationResult(
            is_valid=is_valid,
            severity=self.severity,
            category=self.category,
            rule_name=self.name,
            message=message,
            field_path=field_path,
            suggested_fix=suggested_fix,
            metadata=metadata
        )


class SchemaValidationRule(ValidationRule):
    """Schema-based validation rule"""
    
    def __init__(self, name: str, schema: Dict[str, Any], severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, ValidationCategory.SCHEMA, severity)
        self.schema = schema
    
    def validate(self, config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate against JSON schema"""
        results = []
        
        try:
            import jsonschema
            jsonschema.validate(config, self.schema)
            results.append(self.create_result(True, f"Schema validation passed for {self.name}"))
        except ImportError:
            results.append(self.create_result(
                False, 
                "jsonschema library not available for schema validation",
                suggested_fix="Install jsonschema: pip install jsonschema"
            ))
        except jsonschema.ValidationError as e:
            results.append(self.create_result(
                False,
                f"Schema validation failed: {e.message}",
                field_path='.'.join(str(p) for p in e.absolute_path),
                suggested_fix=f"Fix schema violation at {e.json_path}"
            ))
        except Exception as e:
            results.append(self.create_result(
                False,
                f"Schema validation error: {str(e)}"
            ))
        
        return results


class RangeValidationRule(ValidationRule):
    """Validates numeric ranges"""
    
    def __init__(self, name: str, field_path: str, min_value: Optional[float] = None, 
                 max_value: Optional[float] = None, severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, ValidationCategory.BUSINESS_LOGIC, severity)
        self.field_path = field_path
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate numeric range"""
        results = []
        value = self._get_nested_value(config, self.field_path)
        
        if value is None:
            if self.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
                results.append(self.create_result(
                    False,
                    f"Required field {self.field_path} is missing",
                    field_path=self.field_path,
                    suggested_fix=f"Add {self.field_path} field with appropriate value"
                ))
            return results
        
        if not isinstance(value, (int, float)):
            results.append(self.create_result(
                False,
                f"Field {self.field_path} must be numeric, got {type(value).__name__}",
                field_path=self.field_path,
                suggested_fix=f"Change {self.field_path} to numeric value"
            ))
            return results
        
        # Check minimum value
        if self.min_value is not None and value < self.min_value:
            results.append(self.create_result(
                False,
                f"Field {self.field_path} value {value} is below minimum {self.min_value}",
                field_path=self.field_path,
                suggested_fix=f"Set {self.field_path} to at least {self.min_value}"
            ))
        
        # Check maximum value
        if self.max_value is not None and value > self.max_value:
            results.append(self.create_result(
                False,
                f"Field {self.field_path} value {value} is above maximum {self.max_value}",
                field_path=self.field_path,
                suggested_fix=f"Set {self.field_path} to at most {self.max_value}"
            ))
        
        if not results:
            results.append(self.create_result(
                True,
                f"Field {self.field_path} value {value} is within valid range"
            ))
        
        return results
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested configuration value by path"""
        parts = path.split('.')
        current = config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current


class MusicStreamingValidationRule(ValidationRule):
    """Music streaming specific validation rules"""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.WARNING):
        super().__init__(name, ValidationCategory.MUSIC_STREAMING, severity)
    
    def validate(self, config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate music streaming specific configurations"""
        results = []
        
        # Validate audio quality settings
        results.extend(self._validate_audio_quality(config))
        
        # Validate user segment configurations
        results.extend(self._validate_user_segments(config))
        
        # Validate recommendation engine settings
        results.extend(self._validate_recommendation_settings(config))
        
        # Validate streaming performance thresholds
        results.extend(self._validate_streaming_performance(config))
        
        return results
    
    def _validate_audio_quality(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate audio quality configurations"""
        results = []
        
        audio_config = config.get('audio_quality', {})
        
        # Validate bitrate settings
        if 'bitrates' in audio_config:
            bitrates = audio_config['bitrates']
            
            # Check for standard bitrates
            standard_bitrates = [96, 128, 160, 192, 256, 320]
            for quality, bitrate in bitrates.items():
                if bitrate not in standard_bitrates:
                    results.append(self.create_result(
                        False,
                        f"Non-standard bitrate {bitrate} for quality {quality}",
                        field_path=f"audio_quality.bitrates.{quality}",
                        suggested_fix=f"Use standard bitrate from {standard_bitrates}"
                    ))
        
        # Validate codec settings
        if 'codecs' in audio_config:
            supported_codecs = ['mp3', 'aac', 'ogg', 'flac']
            codecs = audio_config['codecs']
            
            for codec in codecs:
                if codec.lower() not in supported_codecs:
                    results.append(self.create_result(
                        False,
                        f"Unsupported codec: {codec}",
                        field_path="audio_quality.codecs",
                        suggested_fix=f"Use supported codec from {supported_codecs}"
                    ))
        
        return results
    
    def _validate_user_segments(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate user segment configurations"""
        results = []
        
        user_segments = config.get('user_segments', {})
        
        # Validate segment priority settings
        if 'priority_levels' in user_segments:
            priorities = user_segments['priority_levels']
            
            # Check for valid priority order
            valid_segments = ['premium', 'family', 'student', 'free']
            for segment, priority in priorities.items():
                if segment not in valid_segments:
                    results.append(self.create_result(
                        False,
                        f"Unknown user segment: {segment}",
                        field_path=f"user_segments.priority_levels.{segment}",
                        suggested_fix=f"Use valid segment from {valid_segments}"
                    ))
                
                if not isinstance(priority, int) or priority < 1 or priority > 10:
                    results.append(self.create_result(
                        False,
                        f"Invalid priority {priority} for segment {segment}",
                        field_path=f"user_segments.priority_levels.{segment}",
                        suggested_fix="Set priority between 1-10"
                    ))
        
        return results
    
    def _validate_recommendation_settings(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate recommendation engine settings"""
        results = []
        
        rec_config = config.get('recommendation_engine', {})
        
        # Validate model parameters
        if 'models' in rec_config:
            models = rec_config['models']
            
            for model_name, model_config in models.items():
                # Check learning rate
                if 'learning_rate' in model_config:
                    lr = model_config['learning_rate']
                    if not (0.0001 <= lr <= 1.0):
                        results.append(self.create_result(
                            False,
                            f"Learning rate {lr} outside recommended range [0.0001, 1.0]",
                            field_path=f"recommendation_engine.models.{model_name}.learning_rate",
                            suggested_fix="Set learning rate between 0.0001 and 1.0"
                        ))
                
                # Check embedding dimensions
                if 'embedding_dim' in model_config:
                    dim = model_config['embedding_dim']
                    if not isinstance(dim, int) or dim < 32 or dim > 1024:
                        results.append(self.create_result(
                            False,
                            f"Embedding dimension {dim} outside recommended range [32, 1024]",
                            field_path=f"recommendation_engine.models.{model_name}.embedding_dim",
                            suggested_fix="Set embedding dimension between 32 and 1024"
                        ))
        
        return results
    
    def _validate_streaming_performance(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate streaming performance thresholds"""
        results = []
        
        perf_config = config.get('performance_thresholds', {})
        
        # Validate latency thresholds
        if 'latency' in perf_config:
            latency = perf_config['latency']
            
            # Check API response time
            if 'api_response_ms' in latency:
                api_latency = latency['api_response_ms']
                if api_latency > 500:
                    results.append(self.create_result(
                        False,
                        f"API response time {api_latency}ms too high",
                        field_path="performance_thresholds.latency.api_response_ms",
                        suggested_fix="Set API response time below 500ms"
                    ))
            
            # Check stream start time
            if 'stream_start_ms' in latency:
                stream_latency = latency['stream_start_ms']
                if stream_latency > 2000:
                    results.append(self.create_result(
                        False,
                        f"Stream start time {stream_latency}ms too high",
                        field_path="performance_thresholds.latency.stream_start_ms",
                        suggested_fix="Set stream start time below 2000ms"
                    ))
        
        return results


class SecurityValidationRule(ValidationRule):
    """Security-focused validation rules"""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.CRITICAL):
        super().__init__(name, ValidationCategory.SECURITY, severity)
    
    def validate(self, config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate security configurations"""
        results = []
        
        # Validate encryption settings
        results.extend(self._validate_encryption(config))
        
        # Validate authentication settings
        results.extend(self._validate_authentication(config))
        
        # Validate network security
        results.extend(self._validate_network_security(config))
        
        # Validate sensitive data handling
        results.extend(self._validate_sensitive_data(config))
        
        return results
    
    def _validate_encryption(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate encryption settings"""
        results = []
        
        encryption = config.get('encryption', {})
        
        if not encryption:
            results.append(self.create_result(
                False,
                "No encryption configuration found",
                suggested_fix="Add encryption configuration section"
            ))
            return results
        
        # Check encryption algorithms
        if 'algorithm' in encryption:
            algorithm = encryption['algorithm'].upper()
            weak_algorithms = ['DES', 'RC4', 'MD5']
            
            if algorithm in weak_algorithms:
                results.append(self.create_result(
                    False,
                    f"Weak encryption algorithm: {algorithm}",
                    field_path="encryption.algorithm",
                    suggested_fix="Use strong encryption like AES-256"
                ))
        
        # Check key lengths
        if 'key_length' in encryption:
            key_length = encryption['key_length']
            if key_length < 256:
                results.append(self.create_result(
                    False,
                    f"Encryption key length {key_length} too short",
                    field_path="encryption.key_length",
                    suggested_fix="Use minimum 256-bit encryption keys"
                ))
        
        return results
    
    def _validate_authentication(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate authentication settings"""
        results = []
        
        auth = config.get('authentication', {})
        
        # Check token expiration
        if 'token_expiry_hours' in auth:
            expiry = auth['token_expiry_hours']
            if expiry > 24:
                results.append(self.create_result(
                    False,
                    f"Token expiry {expiry} hours too long",
                    field_path="authentication.token_expiry_hours",
                    suggested_fix="Set token expiry to maximum 24 hours",
                    severity=ValidationSeverity.WARNING
                ))
        
        # Check password policies
        if 'password_policy' in auth:
            policy = auth['password_policy']
            
            if policy.get('min_length', 0) < 12:
                results.append(self.create_result(
                    False,
                    f"Password minimum length too short",
                    field_path="authentication.password_policy.min_length",
                    suggested_fix="Set minimum password length to 12 characters"
                ))
            
            if not policy.get('require_special_chars', False):
                results.append(self.create_result(
                    False,
                    "Password policy should require special characters",
                    field_path="authentication.password_policy.require_special_chars",
                    suggested_fix="Enable special character requirement"
                ))
        
        return results
    
    def _validate_network_security(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate network security settings"""
        results = []
        
        network = config.get('network', {})
        
        # Check for HTTPS enforcement
        if 'enforce_https' in network:
            if not network['enforce_https']:
                results.append(self.create_result(
                    False,
                    "HTTPS is not enforced",
                    field_path="network.enforce_https",
                    suggested_fix="Enable HTTPS enforcement"
                ))
        
        # Validate CORS settings
        if 'cors' in network:
            cors = network['cors']
            
            if cors.get('allow_origins') == ['*']:
                results.append(self.create_result(
                    False,
                    "CORS allows all origins (*)",
                    field_path="network.cors.allow_origins",
                    suggested_fix="Restrict CORS to specific domains"
                ))
        
        return results
    
    def _validate_sensitive_data(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate sensitive data handling"""
        results = []
        
        # Look for potential secrets in configuration
        sensitive_patterns = [
            (r'password', 'password'),
            (r'secret', 'secret'),
            (r'key', 'API key'),
            (r'token', 'token'),
            (r'credential', 'credential')
        ]
        
        def scan_dict(obj, path=""):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                if isinstance(value, dict):
                    scan_dict(value, current_path)
                elif isinstance(value, str):
                    # Check if key suggests sensitive data
                    for pattern, description in sensitive_patterns:
                        if re.search(pattern, key.lower()):
                            # Check if value looks like plaintext (not encrypted/hashed)
                            if len(value) < 50 and not value.startswith('$') and not value.startswith('enc:'):
                                results.append(self.create_result(
                                    False,
                                    f"Potential plaintext {description} found",
                                    field_path=current_path,
                                    suggested_fix=f"Encrypt or use environment variable for {description}"
                                ))
        
        scan_dict(config)
        return results


class PerformanceValidationRule(ValidationRule):
    """Performance-related validation rules"""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.WARNING):
        super().__init__(name, ValidationCategory.PERFORMANCE, severity)
    
    def validate(self, config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate performance configurations"""
        results = []
        
        # Validate cache configurations
        results.extend(self._validate_cache_config(config))
        
        # Validate database connection pools
        results.extend(self._validate_db_pools(config))
        
        # Validate resource limits
        results.extend(self._validate_resource_limits(config))
        
        return results
    
    def _validate_cache_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate cache configurations"""
        results = []
        
        cache = config.get('cache', {})
        
        if 'redis' in cache:
            redis_config = cache['redis']
            
            # Check connection pool size
            if 'max_connections' in redis_config:
                max_conn = redis_config['max_connections']
                if max_conn < 10:
                    results.append(self.create_result(
                        False,
                        f"Redis max connections {max_conn} too low",
                        field_path="cache.redis.max_connections",
                        suggested_fix="Increase Redis max connections to at least 10"
                    ))
                elif max_conn > 1000:
                    results.append(self.create_result(
                        False,
                        f"Redis max connections {max_conn} too high",
                        field_path="cache.redis.max_connections",
                        suggested_fix="Reduce Redis max connections to under 1000"
                    ))
            
            # Check TTL settings
            if 'default_ttl' in redis_config:
                ttl = redis_config['default_ttl']
                if ttl < 60:
                    results.append(self.create_result(
                        False,
                        f"Redis TTL {ttl} seconds too short",
                        field_path="cache.redis.default_ttl",
                        suggested_fix="Set Redis TTL to at least 60 seconds"
                    ))
        
        return results
    
    def _validate_db_pools(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate database connection pool configurations"""
        results = []
        
        database = config.get('database', {})
        
        if 'connection_pool' in database:
            pool = database['connection_pool']
            
            # Check pool sizes
            min_size = pool.get('min_size', 0)
            max_size = pool.get('max_size', 0)
            
            if min_size < 5:
                results.append(self.create_result(
                    False,
                    f"Database pool min_size {min_size} too low",
                    field_path="database.connection_pool.min_size",
                    suggested_fix="Set database pool min_size to at least 5"
                ))
            
            if max_size > 100:
                results.append(self.create_result(
                    False,
                    f"Database pool max_size {max_size} too high",
                    field_path="database.connection_pool.max_size",
                    suggested_fix="Reduce database pool max_size to under 100"
                ))
            
            if min_size >= max_size:
                results.append(self.create_result(
                    False,
                    "Database pool min_size >= max_size",
                    field_path="database.connection_pool",
                    suggested_fix="Set min_size < max_size for database pool"
                ))
        
        return results
    
    def _validate_resource_limits(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate resource limit configurations"""
        results = []
        
        resources = config.get('resources', {})
        
        # Check memory limits
        if 'memory_limit_mb' in resources:
            memory_mb = resources['memory_limit_mb']
            if memory_mb < 512:
                results.append(self.create_result(
                    False,
                    f"Memory limit {memory_mb}MB too low",
                    field_path="resources.memory_limit_mb",
                    suggested_fix="Set memory limit to at least 512MB"
                ))
        
        # Check CPU limits
        if 'cpu_cores' in resources:
            cpu_cores = resources['cpu_cores']
            if cpu_cores < 0.5:
                results.append(self.create_result(
                    False,
                    f"CPU cores {cpu_cores} too low",
                    field_path="resources.cpu_cores",
                    suggested_fix="Allocate at least 0.5 CPU cores"
                ))
        
        return results


class ComplianceValidationRule(ValidationRule):
    """GDPR and compliance validation rules"""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(name, ValidationCategory.COMPLIANCE, severity)
    
    def validate(self, config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate compliance configurations"""
        results = []
        
        # GDPR compliance checks
        results.extend(self._validate_gdpr_compliance(config))
        
        # Data retention policies
        results.extend(self._validate_data_retention(config))
        
        # Audit logging
        results.extend(self._validate_audit_logging(config))
        
        return results
    
    def _validate_gdpr_compliance(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate GDPR compliance settings"""
        results = []
        
        gdpr = config.get('gdpr', {})
        
        # Check data processing consent
        if 'consent_required' not in gdpr:
            results.append(self.create_result(
                False,
                "GDPR consent requirement not specified",
                suggested_fix="Add gdpr.consent_required configuration"
            ))
        
        # Check right to be forgotten
        if 'right_to_be_forgotten' not in gdpr:
            results.append(self.create_result(
                False,
                "GDPR right to be forgotten not configured",
                suggested_fix="Add gdpr.right_to_be_forgotten configuration"
            ))
        
        # Check data portability
        if 'data_portability' not in gdpr:
            results.append(self.create_result(
                False,
                "GDPR data portability not configured",
                suggested_fix="Add gdpr.data_portability configuration"
            ))
        
        return results
    
    def _validate_data_retention(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate data retention policies"""
        results = []
        
        retention = config.get('data_retention', {})
        
        if not retention:
            results.append(self.create_result(
                False,
                "No data retention policy configured",
                suggested_fix="Add data_retention configuration section"
            ))
            return results
        
        # Check retention periods
        if 'user_data_days' in retention:
            days = retention['user_data_days']
            if days > 2555:  # 7 years
                results.append(self.create_result(
                    False,
                    f"User data retention {days} days exceeds GDPR recommendations",
                    field_path="data_retention.user_data_days",
                    suggested_fix="Reduce user data retention to maximum 7 years"
                ))
        
        if 'logs_days' in retention:
            log_days = retention['logs_days']
            if log_days > 365:
                results.append(self.create_result(
                    False,
                    f"Log retention {log_days} days too long",
                    field_path="data_retention.logs_days",
                    suggested_fix="Reduce log retention to maximum 365 days"
                ))
        
        return results
    
    def _validate_audit_logging(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate audit logging configuration"""
        results = []
        
        audit = config.get('audit_logging', {})
        
        if not audit.get('enabled', False):
            results.append(self.create_result(
                False,
                "Audit logging is not enabled",
                field_path="audit_logging.enabled",
                suggested_fix="Enable audit logging for compliance"
            ))
        
        # Check required audit events
        required_events = ['user_login', 'data_access', 'data_modification', 'admin_actions']
        logged_events = audit.get('events', [])
        
        for event in required_events:
            if event not in logged_events:
                results.append(self.create_result(
                    False,
                    f"Required audit event '{event}' not logged",
                    field_path="audit_logging.events",
                    suggested_fix=f"Add '{event}' to audit logging events"
                ))
        
        return results


class ConfigurationValidator:
    """Main configuration validation engine"""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules"""
        # Music streaming specific rules
        self.add_rule(MusicStreamingValidationRule("music_streaming_validation"))
        
        # Security rules
        self.add_rule(SecurityValidationRule("security_validation"))
        
        # Performance rules
        self.add_rule(PerformanceValidationRule("performance_validation"))
        
        # Compliance rules
        self.add_rule(ComplianceValidationRule("compliance_validation"))
        
        # Range validation rules for common configuration values
        self.add_rule(RangeValidationRule(
            "anomaly_detection_contamination",
            "anomaly_detection.models.isolation_forest.contamination",
            min_value=0.001,
            max_value=0.5
        ))
        
        self.add_rule(RangeValidationRule(
            "api_rate_limit",
            "api.rate_limit.requests_per_minute",
            min_value=1,
            max_value=10000
        ))
        
        self.add_rule(RangeValidationRule(
            "cache_ttl",
            "cache.default_ttl_seconds",
            min_value=60,
            max_value=86400
        ))
    
    def add_rule(self, rule: ValidationRule):
        """Add validation rule"""
        self.rules.append(rule)
        logger.debug(f"Added validation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove validation rule by name"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        logger.debug(f"Removed validation rule: {rule_name}")
    
    def validate_config(self, config: Dict[str, Any], config_name: str = "unknown", 
                       context: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """Validate configuration against all rules"""
        all_results = []
        
        for rule in self.rules:
            try:
                results = rule.validate(config, context)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error running validation rule {rule.name}: {e}")
                all_results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    category=rule.category,
                    rule_name=rule.name,
                    message=f"Validation rule failed with error: {str(e)}",
                    metadata={'exception': str(e)}
                ))
        
        return ValidationReport(
            config_name=config_name,
            validation_timestamp=datetime.now(),
            overall_valid=False,  # Will be calculated in __post_init__
            results=all_results
        )
    
    def validate_config_file(self, config_file_path: str, context: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """Validate configuration from file"""
        import yaml
        
        config_path = Path(config_file_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {config_path.suffix}")
            
            return self.validate_config(config, config_path.name, context)
            
        except Exception as e:
            logger.error(f"Error loading configuration file {config_file_path}: {e}")
            return ValidationReport(
                config_name=config_path.name,
                validation_timestamp=datetime.now(),
                overall_valid=False,
                results=[ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.SCHEMA,
                    rule_name="file_loading",
                    message=f"Failed to load configuration file: {str(e)}",
                    metadata={'file_path': config_file_path, 'exception': str(e)}
                )]
            )
    
    def get_validation_summary(self, report: ValidationReport) -> str:
        """Get human-readable validation summary"""
        lines = [
            f"Configuration Validation Report: {report.config_name}",
            f"Validation Time: {report.validation_timestamp}",
            f"Overall Status: {'✅ VALID' if report.overall_valid else '❌ INVALID'}",
            "",
            "Summary:",
            f"  Critical: {report.summary.get(ValidationSeverity.CRITICAL, 0)}",
            f"  Errors: {report.summary.get(ValidationSeverity.ERROR, 0)}",
            f"  Warnings: {report.summary.get(ValidationSeverity.WARNING, 0)}",
            f"  Info: {report.summary.get(ValidationSeverity.INFO, 0)}",
            ""
        ]
        
        if not report.overall_valid:
            lines.append("Issues Found:")
            for result in report.results:
                if result.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
                    icon = "🔴" if result.severity == ValidationSeverity.CRITICAL else "🟠"
                    lines.append(f"  {icon} {result.message}")
                    if result.field_path:
                        lines.append(f"      Field: {result.field_path}")
                    if result.suggested_fix:
                        lines.append(f"      Fix: {result.suggested_fix}")
                    lines.append("")
        
        return "\n".join(lines)


# Global validator instance
_config_validator: Optional[ConfigurationValidator] = None


def get_config_validator() -> ConfigurationValidator:
    """Get global configuration validator instance"""
    global _config_validator
    if _config_validator is None:
        _config_validator = ConfigurationValidator()
    return _config_validator


def validate_configuration(config: Dict[str, Any], config_name: str = "unknown") -> ValidationReport:
    """Convenience function to validate configuration"""
    validator = get_config_validator()
    return validator.validate_config(config, config_name)


# Export all classes and functions
__all__ = [
    'ValidationSeverity',
    'ValidationCategory',
    'ValidationResult',
    'ValidationReport',
    'ValidationRule',
    'SchemaValidationRule',
    'RangeValidationRule',
    'MusicStreamingValidationRule',
    'SecurityValidationRule',
    'PerformanceValidationRule',
    'ComplianceValidationRule',
    'ConfigurationValidator',
    'get_config_validator',
    'validate_configuration'
]
