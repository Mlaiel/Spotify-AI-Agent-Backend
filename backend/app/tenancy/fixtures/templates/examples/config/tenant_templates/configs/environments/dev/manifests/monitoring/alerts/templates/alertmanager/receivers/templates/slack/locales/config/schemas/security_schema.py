"""
Schémas de sécurité - Module Python.

Ce module fournit les classes de validation pour les politiques
de sécurité, authentification, autorisation et chiffrement.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class AuthProviderType(str, Enum):
    """Types de fournisseurs d'authentification."""
    OAUTH2 = "oauth2"
    SAML = "saml"
    LDAP = "ldap"
    JWT = "jwt"
    API_KEY = "api_key"


class MFAMethod(str, Enum):
    """Méthodes d'authentification multi-facteurs."""
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    WEBAUTHN = "webauthn"


class SameSitePolicy(str, Enum):
    """Politiques SameSite pour les cookies."""
    STRICT = "strict"
    LAX = "lax"
    NONE = "none"


class PermissionAction(str, Enum):
    """Actions de permissions."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    EXECUTE = "execute"


class PolicyEffect(str, Enum):
    """Effet des politiques ABAC."""
    ALLOW = "allow"
    DENY = "deny"


class EncryptionAlgorithm(str, Enum):
    """Algorithmes de chiffrement."""
    AES_256 = "AES-256"
    CHACHA20_POLY1305 = "ChaCha20-Poly1305"


class KeyProvider(str, Enum):
    """Fournisseurs de gestion de clés."""
    VAULT = "vault"
    AWS_KMS = "aws_kms"
    AZURE_KEY_VAULT = "azure_key_vault"
    GCP_KMS = "gcp_kms"
    LOCAL = "local"


class TLSVersion(str, Enum):
    """Versions TLS supportées."""
    V1_2 = "1.2"
    V1_3 = "1.3"


class NetworkProtocol(str, Enum):
    """Protocoles réseau."""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ANY = "any"


class FirewallAction(str, Enum):
    """Actions de firewall."""
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"


class DDoSAction(str, Enum):
    """Actions de protection DDoS."""
    BLOCK = "block"
    CAPTCHA = "captcha"
    RATE_LIMIT = "rate_limit"
    ALERT = "alert"


class AuditEvent(str, Enum):
    """Types d'événements d'audit."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_MANAGEMENT = "user_management"
    SYSTEM_EVENTS = "system_events"


class ComplianceFramework(str, Enum):
    """Frameworks de conformité."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"


class AuthProviderConfig(BaseModel):
    """Configuration d'un fournisseur d'authentification."""
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    redirect_uri: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    token_endpoint: Optional[str] = None
    userinfo_endpoint: Optional[str] = None


class AuthProvider(BaseModel):
    """Fournisseur d'authentification."""
    name: str
    type: AuthProviderType
    config: AuthProviderConfig = Field(default_factory=AuthProviderConfig)
    enabled: bool = True
    priority: int = Field(1, ge=1)


class SessionConfig(BaseModel):
    """Configuration de session."""
    timeout: int = Field(3600, ge=300, le=86400)
    refresh_threshold: int = Field(300, ge=60)
    secure_cookies: bool = True
    same_site: SameSitePolicy = SameSitePolicy.LAX


class MFAConfig(BaseModel):
    """Configuration MFA."""
    required: bool = False
    methods: List[MFAMethod] = Field(default_factory=lambda: [MFAMethod.TOTP])
    backup_codes: bool = True
    grace_period: int = Field(0, ge=0)


class PasswordPolicy(BaseModel):
    """Politique de mot de passe."""
    min_length: int = Field(12, ge=8)
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_symbols: bool = True
    max_age_days: int = Field(90, ge=30)
    history_count: int = Field(5, ge=3)


class AuthenticationConfig(BaseModel):
    """Configuration d'authentification."""
    providers: List[AuthProvider] = Field(default_factory=list)
    session: SessionConfig = Field(default_factory=SessionConfig)
    mfa: MFAConfig = Field(default_factory=MFAConfig)
    password_policy: PasswordPolicy = Field(default_factory=PasswordPolicy)


class Permission(BaseModel):
    """Permission RBAC."""
    name: str
    resource: str
    action: PermissionAction
    conditions: List[Dict[str, Any]] = Field(default_factory=list)


class Role(BaseModel):
    """Rôle RBAC."""
    name: str
    description: Optional[str] = None
    permissions: List[str]
    inherit_from: List[str] = Field(default_factory=list)


class RBACConfig(BaseModel):
    """Configuration RBAC."""
    enabled: bool = True
    roles: List[Role] = Field(default_factory=list)
    permissions: List[Permission] = Field(default_factory=list)


class ABACPolicy(BaseModel):
    """Politique ABAC."""
    name: str
    effect: PolicyEffect
    subject: Dict[str, Any] = Field(default_factory=dict)
    resource: Dict[str, Any] = Field(default_factory=dict)
    action: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, Any] = Field(default_factory=dict)
    condition: Optional[str] = None


class ABACConfig(BaseModel):
    """Configuration ABAC."""
    enabled: bool = False
    policies: List[ABACPolicy] = Field(default_factory=list)


class AuthorizationConfig(BaseModel):
    """Configuration d'autorisation."""
    rbac: RBACConfig = Field(default_factory=RBACConfig)
    abac: ABACConfig = Field(default_factory=ABACConfig)


class KeyRotationConfig(BaseModel):
    """Configuration de rotation des clés."""
    enabled: bool = True
    interval_days: int = Field(90, ge=30)


class KeyManagementConfig(BaseModel):
    """Configuration de gestion des clés."""
    provider: KeyProvider = KeyProvider.LOCAL
    key_rotation: KeyRotationConfig = Field(default_factory=KeyRotationConfig)


class EncryptionAtRestConfig(BaseModel):
    """Configuration de chiffrement au repos."""
    enabled: bool = True
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256
    key_management: KeyManagementConfig = Field(default_factory=KeyManagementConfig)


class CertificateManagementConfig(BaseModel):
    """Configuration de gestion des certificats."""
    provider: str = "letsencrypt"
    auto_renewal: bool = True


class EncryptionInTransitConfig(BaseModel):
    """Configuration de chiffrement en transit."""
    enabled: bool = True
    tls_version: TLSVersion = TLSVersion.V1_3
    cipher_suites: List[str] = Field(default_factory=list)
    certificate_management: CertificateManagementConfig = Field(
        default_factory=CertificateManagementConfig
    )


class EncryptionConfig(BaseModel):
    """Configuration de chiffrement."""
    at_rest: EncryptionAtRestConfig = Field(default_factory=EncryptionAtRestConfig)
    in_transit: EncryptionInTransitConfig = Field(
        default_factory=EncryptionInTransitConfig
    )


class FirewallRule(BaseModel):
    """Règle de firewall."""
    name: str
    action: FirewallAction
    protocol: NetworkProtocol
    source: str
    destination: str
    port: str


class FirewallConfig(BaseModel):
    """Configuration de firewall."""
    enabled: bool = True
    rules: List[FirewallRule] = Field(default_factory=list)


class RateLimits(BaseModel):
    """Limites de taux."""
    requests_per_second: Optional[int] = None
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None


class RateLimitingConfig(BaseModel):
    """Configuration de limitation de taux."""
    enabled: bool = True
    global_limits: RateLimits = Field(default_factory=RateLimits)
    tenant_limits: RateLimits = Field(default_factory=RateLimits)
    user_limits: RateLimits = Field(default_factory=RateLimits)


class DDoSProtectionConfig(BaseModel):
    """Configuration de protection DDoS."""
    enabled: bool = True
    threshold: int = 1000
    mitigation_actions: List[DDoSAction] = Field(
        default_factory=lambda: [DDoSAction.RATE_LIMIT, DDoSAction.ALERT]
    )


class NetworkConfig(BaseModel):
    """Configuration réseau."""
    firewall: FirewallConfig = Field(default_factory=FirewallConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    ddos_protection: DDoSProtectionConfig = Field(default_factory=DDoSProtectionConfig)


class ArchivalConfig(BaseModel):
    """Configuration d'archivage."""
    enabled: bool = True
    provider: str = "s3"
    schedule: str = "monthly"


class RetentionConfig(BaseModel):
    """Configuration de rétention."""
    days: int = Field(365, ge=30)
    compression: bool = True
    archival: ArchivalConfig = Field(default_factory=ArchivalConfig)


class RealTimeMonitoringConfig(BaseModel):
    """Configuration de monitoring en temps réel."""
    enabled: bool = True
    suspicious_activity_detection: bool = True
    automated_response: bool = False


class AuditConfig(BaseModel):
    """Configuration d'audit."""
    enabled: bool = True
    events: List[AuditEvent] = Field(
        default_factory=lambda: [
            AuditEvent.AUTHENTICATION,
            AuditEvent.AUTHORIZATION,
            AuditEvent.DATA_ACCESS,
        ]
    )
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    real_time_monitoring: RealTimeMonitoringConfig = Field(
        default_factory=RealTimeMonitoringConfig
    )


class DataClassificationLevel(BaseModel):
    """Niveau de classification des données."""
    name: str
    description: str
    handling_requirements: List[str] = Field(default_factory=list)


class DataClassificationConfig(BaseModel):
    """Configuration de classification des données."""
    levels: List[DataClassificationLevel] = Field(default_factory=list)


class PrivacyControlsConfig(BaseModel):
    """Configuration des contrôles de confidentialité."""
    data_minimization: bool = True
    purpose_limitation: bool = True
    consent_management: bool = True
    right_to_be_forgotten: bool = True


class ComplianceConfig(BaseModel):
    """Configuration de conformité."""
    frameworks: List[ComplianceFramework] = Field(default_factory=list)
    data_classification: DataClassificationConfig = Field(
        default_factory=DataClassificationConfig
    )
    privacy_controls: PrivacyControlsConfig = Field(
        default_factory=PrivacyControlsConfig
    )


class SecurityPolicySchema(BaseModel):
    """Schéma complet des politiques de sécurité."""
    authentication: AuthenticationConfig
    authorization: AuthorizationConfig
    encryption: EncryptionConfig
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    audit: AuditConfig
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)

    @validator('authentication')
    def validate_authentication(cls, v):
        """Valide la configuration d'authentification."""
        if not v.providers:
            raise ValueError("Au moins un fournisseur d'authentification doit être configuré")
        return v

    @validator('authorization')
    def validate_authorization(cls, v):
        """Valide la configuration d'autorisation."""
        if not v.rbac.enabled and not v.abac.enabled:
            raise ValueError("Au moins RBAC ou ABAC doit être activé")
        return v

    class Config:
        """Configuration Pydantic."""
        use_enum_values = True
        validate_assignment = True


class EncryptionSchema(BaseModel):
    """Schéma simplifié pour le chiffrement."""
    enabled: bool = True
    algorithms: List[EncryptionAlgorithm] = Field(
        default_factory=lambda: [EncryptionAlgorithm.AES_256]
    )
    key_providers: List[KeyProvider] = Field(
        default_factory=lambda: [KeyProvider.LOCAL]
    )
    rotation_enabled: bool = True

    class Config:
        """Configuration Pydantic."""
        use_enum_values = True
