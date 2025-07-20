"""
Enterprise-grade configuration management schemas for multi-tenant alerting systems.

This module provides comprehensive Pydantic schemas for managing complex configurations
across multiple tenants with advanced validation, security, and compliance features.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import SecretStr, EmailStr


class ConfigPriorityLevel(str, Enum):
    """Configuration priority levels for tenant hierarchy."""
    SYSTEM = "system"
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    OVERRIDE = "override"


class ComplianceStandard(str, Enum):
    """Supported compliance standards."""
    SOC2_TYPE_I = "soc2_type_i"
    SOC2_TYPE_II = "soc2_type_ii"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    ISO_27001 = "iso_27001"
    PCI_DSS = "pci_dss"
    NIST = "nist"


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_4096 = "rsa_4096"
    ECDSA_P256 = "ecdsa_p256"


class ConfigurationSourceSchema(BaseModel):
    """Schema for configuration source tracking."""
    source_type: str = Field(..., description="Type of configuration source")
    source_id: str = Field(..., description="Unique identifier for the source")
    source_url: Optional[str] = Field(None, description="URL or path to source")
    version: str = Field(..., description="Version of the configuration")
    checksum: str = Field(..., description="SHA-256 checksum of configuration")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "source_type": "git_repository",
                "source_id": "spotify-ai-configs",
                "source_url": "https://github.com/spotify/ai-configs.git",
                "version": "v2.1.0",
                "checksum": "sha256:abc123...",
                "created_at": "2025-01-19T10:00:00Z",
                "updated_at": "2025-01-19T10:00:00Z"
            }
        }


class ConfigurationEncryptionSchema(BaseModel):
    """Schema for configuration encryption settings."""
    enabled: bool = Field(True, description="Enable encryption for sensitive data")
    algorithm: EncryptionAlgorithm = Field(
        EncryptionAlgorithm.AES_256_GCM,
        description="Encryption algorithm to use"
    )
    key_rotation_days: int = Field(90, ge=30, le=365, description="Key rotation interval in days")
    at_rest_encryption: bool = Field(True, description="Enable encryption at rest")
    in_transit_encryption: bool = Field(True, description="Enable encryption in transit")
    key_management_service: str = Field("vault", description="Key management service")
    
    class Config:
        schema_extra = {
            "example": {
                "enabled": True,
                "algorithm": "aes_256_gcm",
                "key_rotation_days": 90,
                "at_rest_encryption": True,
                "in_transit_encryption": True,
                "key_management_service": "vault"
            }
        }


class ConfigurationComplianceSchema(BaseModel):
    """Schema for compliance requirements."""
    standards: List[ComplianceStandard] = Field(..., description="Required compliance standards")
    audit_retention_days: int = Field(2555, ge=365, description="Audit log retention period")
    data_classification: str = Field(..., description="Data classification level")
    geographic_restrictions: List[str] = Field([], description="Geographic data restrictions")
    privacy_controls: Dict[str, Any] = Field({}, description="Privacy control settings")
    
    @validator('audit_retention_days')
    def validate_retention(cls, v, values):
        """Validate audit retention meets compliance requirements."""
        standards = values.get('standards', [])
        if ComplianceStandard.SOC2_TYPE_II in standards and v < 365:
            raise ValueError("SOC2 Type II requires minimum 365 days retention")
        if ComplianceStandard.HIPAA in standards and v < 2555:  # 7 years
            raise ValueError("HIPAA requires minimum 7 years retention")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "standards": ["soc2_type_ii", "gdpr"],
                "audit_retention_days": 2555,
                "data_classification": "confidential",
                "geographic_restrictions": ["EU", "US"],
                "privacy_controls": {
                    "right_to_be_forgotten": True,
                    "data_portability": True,
                    "consent_management": True
                }
            }
        }


class ConfigurationBackupSchema(BaseModel):
    """Schema for configuration backup settings."""
    enabled: bool = Field(True, description="Enable automatic backups")
    frequency_hours: int = Field(6, ge=1, le=168, description="Backup frequency in hours")
    retention_days: int = Field(90, ge=7, description="Backup retention period")
    compression_enabled: bool = Field(True, description="Enable backup compression")
    encryption_enabled: bool = Field(True, description="Enable backup encryption")
    remote_storage: Dict[str, Any] = Field({}, description="Remote storage configuration")
    verification_enabled: bool = Field(True, description="Enable backup verification")
    
    class Config:
        schema_extra = {
            "example": {
                "enabled": True,
                "frequency_hours": 6,
                "retention_days": 90,
                "compression_enabled": True,
                "encryption_enabled": True,
                "remote_storage": {
                    "provider": "aws_s3",
                    "bucket": "spotify-ai-backups",
                    "region": "us-west-2"
                },
                "verification_enabled": True
            }
        }


class ConfigurationValidationSchema(BaseModel):
    """Schema for configuration validation settings."""
    strict_mode: bool = Field(True, description="Enable strict validation mode")
    schema_validation: bool = Field(True, description="Enable schema validation")
    business_rule_validation: bool = Field(True, description="Enable business rule validation")
    cross_reference_validation: bool = Field(True, description="Enable cross-reference validation")
    performance_validation: bool = Field(True, description="Enable performance validation")
    security_validation: bool = Field(True, description="Enable security validation")
    custom_validators: List[str] = Field([], description="Custom validator functions")
    
    class Config:
        schema_extra = {
            "example": {
                "strict_mode": True,
                "schema_validation": True,
                "business_rule_validation": True,
                "cross_reference_validation": True,
                "performance_validation": True,
                "security_validation": True,
                "custom_validators": ["tenant_isolation_validator", "resource_limit_validator"]
            }
        }


class ConfigurationDeploymentSchema(BaseModel):
    """Schema for configuration deployment settings."""
    strategy: str = Field("blue_green", description="Deployment strategy")
    rollback_enabled: bool = Field(True, description="Enable automatic rollback")
    canary_percentage: int = Field(10, ge=1, le=50, description="Canary deployment percentage")
    health_check_timeout: int = Field(300, ge=30, description="Health check timeout in seconds")
    deployment_timeout: int = Field(1800, ge=300, description="Deployment timeout in seconds")
    notification_channels: List[str] = Field([], description="Deployment notification channels")
    approval_required: bool = Field(True, description="Require manual approval for production")
    
    class Config:
        schema_extra = {
            "example": {
                "strategy": "blue_green",
                "rollback_enabled": True,
                "canary_percentage": 10,
                "health_check_timeout": 300,
                "deployment_timeout": 1800,
                "notification_channels": ["#ops-deployments", "ops-team@spotify.com"],
                "approval_required": True
            }
        }


class EnterpriseConfigurationSchema(BaseModel):
    """Master schema for enterprise configuration management."""
    config_id: UUID = Field(..., description="Unique configuration identifier")
    name: str = Field(..., min_length=3, max_length=100, description="Configuration name")
    description: str = Field(..., max_length=500, description="Configuration description")
    version: str = Field(..., description="Semantic version of configuration")
    environment: str = Field(..., description="Target environment")
    tenant_id: Optional[str] = Field(None, description="Associated tenant ID")
    priority: ConfigPriorityLevel = Field(
        ConfigPriorityLevel.TENANT,
        description="Configuration priority level"
    )
    
    # Core configuration components
    source: ConfigurationSourceSchema = Field(..., description="Configuration source information")
    encryption: ConfigurationEncryptionSchema = Field(..., description="Encryption settings")
    compliance: ConfigurationComplianceSchema = Field(..., description="Compliance requirements")
    backup: ConfigurationBackupSchema = Field(..., description="Backup configuration")
    validation: ConfigurationValidationSchema = Field(..., description="Validation settings")
    deployment: ConfigurationDeploymentSchema = Field(..., description="Deployment configuration")
    
    # Metadata and tracking
    tags: Dict[str, str] = Field({}, description="Configuration tags for categorization")
    owner: EmailStr = Field(..., description="Configuration owner email")
    reviewers: List[EmailStr] = Field([], description="Configuration reviewers")
    dependencies: List[str] = Field([], description="Configuration dependencies")
    
    # Lifecycle management
    active: bool = Field(True, description="Configuration is active")
    deprecated: bool = Field(False, description="Configuration is deprecated")
    deprecation_date: Optional[datetime] = Field(None, description="Deprecation date")
    end_of_life_date: Optional[datetime] = Field(None, description="End of life date")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_validated_at: Optional[datetime] = Field(None, description="Last validation timestamp")
    last_deployed_at: Optional[datetime] = Field(None, description="Last deployment timestamp")
    
    @root_validator
    def validate_lifecycle(cls, values):
        """Validate configuration lifecycle consistency."""
        deprecated = values.get('deprecated', False)
        deprecation_date = values.get('deprecation_date')
        end_of_life_date = values.get('end_of_life_date')
        
        if deprecated and not deprecation_date:
            raise ValueError("Deprecation date required when configuration is deprecated")
        
        if deprecation_date and end_of_life_date:
            if end_of_life_date <= deprecation_date:
                raise ValueError("End of life date must be after deprecation date")
        
        return values
    
    @validator('version')
    def validate_semantic_version(cls, v):
        """Validate semantic versioning format."""
        import re
        pattern = r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
        if not re.match(pattern, v):
            raise ValueError("Version must follow semantic versioning (e.g., 1.2.3)")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "config_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "production-alerting-config",
                "description": "Production alerting configuration for Spotify AI Agent",
                "version": "2.1.0",
                "environment": "production",
                "tenant_id": "spotify-premium-001",
                "priority": "tenant",
                "source": {
                    "source_type": "git_repository",
                    "source_id": "spotify-ai-configs",
                    "version": "v2.1.0",
                    "checksum": "sha256:abc123..."
                },
                "tags": {
                    "team": "platform",
                    "service": "alerting",
                    "criticality": "high"
                },
                "owner": "platform-team@spotify.com",
                "reviewers": ["ops-team@spotify.com"],
                "dependencies": ["base-monitoring-config"],
                "active": True,
                "deprecated": False
            }
        }


class ConfigurationChangelogSchema(BaseModel):
    """Schema for tracking configuration changes."""
    change_id: UUID = Field(..., description="Unique change identifier")
    config_id: UUID = Field(..., description="Configuration identifier")
    change_type: str = Field(..., description="Type of change (create, update, delete)")
    previous_version: Optional[str] = Field(None, description="Previous configuration version")
    new_version: str = Field(..., description="New configuration version")
    changed_by: EmailStr = Field(..., description="User who made the change")
    change_reason: str = Field(..., description="Reason for the change")
    approved_by: Optional[EmailStr] = Field(None, description="User who approved the change")
    rollback_available: bool = Field(True, description="Rollback is available")
    impact_assessment: Dict[str, Any] = Field({}, description="Change impact assessment")
    
    # Change details
    fields_changed: List[str] = Field([], description="List of changed fields")
    diff: Dict[str, Any] = Field({}, description="Detailed diff of changes")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    applied_at: Optional[datetime] = Field(None, description="When change was applied")
    
    class Config:
        schema_extra = {
            "example": {
                "change_id": "660e8400-e29b-41d4-a716-446655440001",
                "config_id": "550e8400-e29b-41d4-a716-446655440000",
                "change_type": "update",
                "previous_version": "2.0.0",
                "new_version": "2.1.0",
                "changed_by": "admin@spotify.com",
                "change_reason": "Updated alert thresholds for better accuracy",
                "approved_by": "manager@spotify.com",
                "rollback_available": True,
                "impact_assessment": {
                    "risk_level": "medium",
                    "affected_tenants": 15,
                    "downtime_expected": False
                },
                "fields_changed": ["alert_thresholds", "escalation_rules"],
                "diff": {
                    "alert_thresholds.cpu_warning": {"old": 0.8, "new": 0.85},
                    "escalation_rules.timeout": {"old": 300, "new": 600}
                }
            }
        }


# Export all schemas for easy imports
__all__ = [
    "ConfigPriorityLevel",
    "ComplianceStandard", 
    "EncryptionAlgorithm",
    "ConfigurationSourceSchema",
    "ConfigurationEncryptionSchema",
    "ConfigurationComplianceSchema",
    "ConfigurationBackupSchema",
    "ConfigurationValidationSchema",
    "ConfigurationDeploymentSchema",
    "EnterpriseConfigurationSchema",
    "ConfigurationChangelogSchema"
]
