"""
Enterprise Security Schemas for Multi-Tenant SaaS Platform
Advanced security validation and configuration models.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, root_validator
import re


class SecurityLevel(str, Enum):
    """Security classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"
    TOP_SECRET = "top_secret"


class AuthenticationMethod(str, Enum):
    """Authentication methods supported."""
    PASSWORD = "password"
    MFA = "mfa"
    OAUTH2 = "oauth2"
    SAML = "saml"
    LDAP = "ldap"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"


class EncryptionAlgorithm(str, Enum):
    """Encryption algorithms."""
    AES256 = "aes256"
    RSA4096 = "rsa4096"
    CHACHA20 = "chacha20"
    ELLIPTIC_CURVE = "elliptic_curve"


class SecurityPolicySchema(BaseModel):
    """Comprehensive security policy configuration."""
    
    policy_id: str = Field(..., description="Unique policy identifier")
    name: str = Field(..., min_length=3, max_length=100)
    description: str = Field(..., max_length=500)
    security_level: SecurityLevel
    
    # Password policies
    password_min_length: int = Field(12, ge=8, le=128)
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    password_expiry_days: int = Field(90, ge=30, le=365)
    password_history_count: int = Field(12, ge=5, le=24)
    
    # Session management
    session_timeout_minutes: int = Field(30, ge=5, le=480)
    max_concurrent_sessions: int = Field(3, ge=1, le=10)
    idle_timeout_minutes: int = Field(15, ge=5, le=60)
    
    # Access control
    max_login_attempts: int = Field(5, ge=3, le=10)
    lockout_duration_minutes: int = Field(30, ge=15, le=120)
    ip_whitelist: Optional[List[str]] = None
    ip_blacklist: Optional[List[str]] = None
    
    # Multi-factor authentication
    mfa_required: bool = True
    mfa_methods: List[AuthenticationMethod] = [AuthenticationMethod.MFA]
    backup_codes_count: int = Field(10, ge=5, le=20)
    
    # Compliance settings
    gdpr_compliant: bool = True
    hipaa_compliant: bool = False
    sox_compliant: bool = False
    pci_compliant: bool = False
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('ip_whitelist', 'ip_blacklist')
    def validate_ip_addresses(cls, v):
        if v:
            ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}(/\d{1,2})?$')
            for ip in v:
                if not ip_pattern.match(ip):
                    raise ValueError(f"Invalid IP address format: {ip}")
        return v


class EncryptionConfigSchema(BaseModel):
    """Advanced encryption configuration."""
    
    algorithm: EncryptionAlgorithm
    key_size: int = Field(..., description="Key size in bits")
    key_rotation_days: int = Field(90, ge=30, le=365)
    
    # At-rest encryption
    encrypt_database: bool = True
    encrypt_files: bool = True
    encrypt_logs: bool = True
    encrypt_backups: bool = True
    
    # In-transit encryption
    tls_version: str = Field("1.3", regex=r"^1\.[23]$")
    cipher_suites: List[str] = Field(default_factory=lambda: [
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "TLS_AES_128_GCM_SHA256"
    ])
    
    # Key management
    key_escrow_enabled: bool = False
    hardware_security_module: bool = False
    key_derivation_function: str = "PBKDF2"
    salt_rounds: int = Field(100000, ge=50000, le=500000)
    
    @validator('key_size')
    def validate_key_size(cls, v, values):
        algorithm = values.get('algorithm')
        if algorithm == EncryptionAlgorithm.AES256 and v != 256:
            raise ValueError("AES256 requires 256-bit key")
        elif algorithm == EncryptionAlgorithm.RSA4096 and v != 4096:
            raise ValueError("RSA4096 requires 4096-bit key")
        return v


class AuditConfigSchema(BaseModel):
    """Comprehensive audit and logging configuration."""
    
    enabled: bool = True
    retention_days: int = Field(2555, ge=365, le=3650)  # 7 years default
    
    # Event categories to audit
    authentication_events: bool = True
    authorization_events: bool = True
    data_access_events: bool = True
    configuration_changes: bool = True
    system_events: bool = True
    security_events: bool = True
    
    # Storage configuration
    storage_encrypted: bool = True
    storage_compressed: bool = True
    storage_redundancy: int = Field(3, ge=1, le=5)
    
    # Real-time monitoring
    real_time_alerts: bool = True
    anomaly_detection: bool = True
    threat_detection: bool = True
    
    # Export capabilities
    export_formats: List[str] = Field(default_factory=lambda: ["json", "csv", "xml"])
    api_access_enabled: bool = True
    
    # Compliance requirements
    siem_integration: bool = True
    log_forwarding_enabled: bool = True
    integrity_checking: bool = True


class SecurityMonitoringSchema(BaseModel):
    """Advanced security monitoring configuration."""
    
    # Threat detection
    intrusion_detection: bool = True
    vulnerability_scanning: bool = True
    malware_detection: bool = True
    ddos_protection: bool = True
    
    # Behavioral analysis
    user_behavior_analytics: bool = True
    entity_behavior_analytics: bool = True
    anomaly_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "login_frequency": 0.95,
        "data_access": 0.90,
        "privilege_escalation": 0.99,
        "unusual_activity": 0.85
    })
    
    # Network security
    network_segmentation: bool = True
    firewall_rules: List[Dict[str, Any]] = Field(default_factory=list)
    web_application_firewall: bool = True
    
    # Incident response
    automated_response: bool = True
    incident_escalation: bool = True
    forensic_collection: bool = True
    
    # Metrics and reporting
    security_dashboards: bool = True
    threat_intelligence: bool = True
    risk_scoring: bool = True


class ComplianceSchema(BaseModel):
    """Compliance framework configuration."""
    
    # Regulatory frameworks
    gdpr_enabled: bool = True
    ccpa_enabled: bool = False
    hipaa_enabled: bool = False
    sox_enabled: bool = False
    pci_dss_enabled: bool = False
    iso27001_enabled: bool = True
    nist_enabled: bool = True
    
    # Data governance
    data_classification: bool = True
    data_lineage: bool = True
    data_retention_policies: bool = True
    right_to_deletion: bool = True
    
    # Privacy controls
    consent_management: bool = True
    privacy_by_design: bool = True
    data_minimization: bool = True
    purpose_limitation: bool = True
    
    # Reporting and auditing
    compliance_reporting: bool = True
    automated_compliance_checks: bool = True
    third_party_assessments: bool = True
    
    # Breach management
    breach_detection: bool = True
    breach_notification: bool = True
    breach_response_plan: bool = True


class SecurityConfigSchema(BaseModel):
    """Main security configuration schema."""
    
    tenant_id: str = Field(..., description="Tenant identifier")
    environment: str = Field(..., regex=r"^(dev|staging|prod)$")
    
    # Core security components
    security_policy: SecurityPolicySchema
    encryption_config: EncryptionConfigSchema
    audit_config: AuditConfigSchema
    monitoring_config: SecurityMonitoringSchema
    compliance_config: ComplianceSchema
    
    # Additional settings
    security_headers: Dict[str, str] = Field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    })
    
    # Emergency procedures
    emergency_access: bool = False
    break_glass_procedures: bool = True
    incident_response_team: List[str] = Field(default_factory=list)
    
    # Integration settings
    external_security_tools: List[str] = Field(default_factory=list)
    api_security_enabled: bool = True
    rate_limiting_enabled: bool = True
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "tenant_001",
                "environment": "prod",
                "security_policy": {
                    "policy_id": "pol_secure_001",
                    "name": "Enterprise Security Policy",
                    "description": "High-security policy for production environment",
                    "security_level": "high"
                }
            }
        }
