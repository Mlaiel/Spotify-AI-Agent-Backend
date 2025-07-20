"""
Security Validation Schemas for Multi-Tenant Architecture
========================================================

Ce module définit les schémas de validation pour l'architecture de sécurité
multi-tenant du Spotify AI Agent.

Auteur: Fahed Mlaiel
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
import re
import ipaddress
from .core import SecurityLevel, ThreatType


class SecurityAction(Enum):
    """Actions de sécurité disponibles"""
    ALLOW = "allow"
    DENY = "deny"
    MONITOR = "monitor"
    ALERT = "alert"
    BLOCK = "block"
    ESCALATE = "escalate"


class ComplianceStandard(Enum):
    """Standards de conformité supportés"""
    GDPR = "gdpr"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"


class PermissionType(Enum):
    """Types de permissions"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"
    OWNER = "owner"


# ================================
# Base Security Schema
# ================================

class BaseSecuritySchema(BaseModel):
    """Schéma de base pour tous les objets de sécurité"""
    
    id: Optional[str] = Field(None, description="Identifiant unique")
    tenant_id: str = Field(..., description="Identifiant du tenant")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(..., description="Créateur de l'objet")
    version: int = Field(default=1, description="Version du schéma")
    active: bool = Field(default=True, description="Objet actif")
    
    class Config:
        use_enum_values = True
        validate_assignment = True


# ================================
# Tenant Security Schema
# ================================

class TenantSecuritySchema(BaseSecuritySchema):
    """Schéma de configuration de sécurité pour un tenant"""
    
    # Configuration de base
    tenant_name: str = Field(..., min_length=1, max_length=100)
    security_level: SecurityLevel = Field(default=SecurityLevel.MEDIUM)
    
    # Configuration de chiffrement
    encryption_algorithm: str = Field(default="AES-256-GCM")
    encryption_key_rotation_hours: int = Field(default=24, ge=1, le=8760)
    data_at_rest_encryption: bool = Field(default=True)
    data_in_transit_encryption: bool = Field(default=True)
    
    # Configuration réseau
    allowed_ip_ranges: List[str] = Field(default_factory=list)
    blocked_ip_ranges: List[str] = Field(default_factory=list)
    geo_restrictions: List[str] = Field(default_factory=list)
    vpn_required: bool = Field(default=False)
    
    # Configuration d'authentification
    mfa_required: bool = Field(default=True)
    password_policy: Dict[str, Any] = Field(default_factory=dict)
    session_timeout_minutes: int = Field(default=480, ge=1, le=1440)
    max_concurrent_sessions: int = Field(default=5, ge=1, le=100)
    
    # Configuration de monitoring
    monitoring_enabled: bool = Field(default=True)
    threat_detection_enabled: bool = Field(default=True)
    anomaly_detection_enabled: bool = Field(default=True)
    behavioral_analysis_enabled: bool = Field(default=True)
    
    # Configuration de conformité
    compliance_standards: List[ComplianceStandard] = Field(default_factory=list)
    data_retention_days: int = Field(default=365, ge=1, le=3650)
    right_to_be_forgotten: bool = Field(default=True)
    consent_management: bool = Field(default=True)
    
    # Configuration des alertes
    alert_channels: List[str] = Field(default_factory=list)
    escalation_levels: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    notification_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration des limites
    rate_limits: Dict[str, int] = Field(default_factory=dict)
    resource_quotas: Dict[str, int] = Field(default_factory=dict)
    api_limits: Dict[str, int] = Field(default_factory=dict)
    
    @validator('allowed_ip_ranges', 'blocked_ip_ranges')
    def validate_ip_ranges(cls, v):
        """Valide les plages d'adresses IP"""
        for ip_range in v:
            try:
                ipaddress.ip_network(ip_range, strict=False)
            except ValueError:
                raise ValueError(f"Invalid IP range: {ip_range}")
        return v
    
    @validator('password_policy')
    def validate_password_policy(cls, v):
        """Valide la politique de mots de passe"""
        default_policy = {
            "min_length": 12,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_special_chars": True,
            "history_count": 5,
            "max_age_days": 90
        }
        
        if not v:
            return default_policy
        
        # Merge avec les valeurs par défaut
        for key, default_value in default_policy.items():
            if key not in v:
                v[key] = default_value
        
        return v
    
    @validator('alert_channels')
    def validate_alert_channels(cls, v):
        """Valide les canaux d'alertes"""
        valid_channels = ['slack', 'email', 'sms', 'webhook', 'siem', 'dashboard']
        for channel in v:
            if channel not in valid_channels:
                raise ValueError(f"Invalid alert channel: {channel}")
        return v


# ================================
# Security Rule Schema
# ================================

class SecurityRuleCondition(BaseModel):
    """Condition d'une règle de sécurité"""
    
    field: str = Field(..., description="Champ à évaluer")
    operator: str = Field(..., description="Opérateur de comparaison")
    value: Any = Field(..., description="Valeur de comparaison")
    case_sensitive: bool = Field(default=True)
    
    @validator('operator')
    def validate_operator(cls, v):
        """Valide l'opérateur"""
        valid_operators = [
            'equals', 'not_equals', 'contains', 'not_contains',
            'starts_with', 'ends_with', 'regex', 'greater_than',
            'less_than', 'greater_equal', 'less_equal', 'in', 'not_in'
        ]
        if v not in valid_operators:
            raise ValueError(f"Invalid operator: {v}")
        return v


class SecurityRuleAction(BaseModel):
    """Action d'une règle de sécurité"""
    
    type: SecurityAction = Field(..., description="Type d'action")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    notification: bool = Field(default=True)
    escalation: bool = Field(default=False)
    
    @validator('parameters')
    def validate_parameters(cls, v, values):
        """Valide les paramètres selon le type d'action"""
        action_type = values.get('type')
        
        if action_type == SecurityAction.BLOCK:
            if 'duration_minutes' not in v:
                v['duration_minutes'] = 60
            if 'reason' not in v:
                v['reason'] = "Security rule violation"
        
        elif action_type == SecurityAction.ALERT:
            if 'severity' not in v:
                v['severity'] = SecurityLevel.MEDIUM.value
            if 'channels' not in v:
                v['channels'] = ['email']
        
        return v


class SecurityRuleSchema(BaseSecuritySchema):
    """Schéma pour les règles de sécurité personnalisées"""
    
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=1000)
    category: str = Field(..., description="Catégorie de la règle")
    
    # Configuration de la règle
    conditions: List[SecurityRuleCondition] = Field(..., min_items=1)
    conditions_logic: str = Field(default="AND", description="Logique entre conditions")
    actions: List[SecurityRuleAction] = Field(..., min_items=1)
    
    # Configuration d'activation
    enabled: bool = Field(default=True)
    priority: int = Field(default=50, ge=1, le=100)
    applies_to: List[str] = Field(default_factory=list, description="Ressources concernées")
    
    # Configuration temporelle
    effective_from: Optional[datetime] = Field(None)
    effective_until: Optional[datetime] = Field(None)
    schedule: Optional[Dict[str, Any]] = Field(None, description="Planification d'activation")
    
    # Métriques
    trigger_count: int = Field(default=0, ge=0)
    last_triggered: Optional[datetime] = Field(None)
    false_positive_count: int = Field(default=0, ge=0)
    
    @validator('conditions_logic')
    def validate_conditions_logic(cls, v):
        """Valide la logique des conditions"""
        if v not in ['AND', 'OR']:
            raise ValueError("conditions_logic must be 'AND' or 'OR'")
        return v
    
    @validator('category')
    def validate_category(cls, v):
        """Valide la catégorie de la règle"""
        valid_categories = [
            'access_control', 'data_protection', 'network_security',
            'authentication', 'authorization', 'compliance',
            'threat_detection', 'anomaly_detection', 'audit',
            'privacy', 'encryption', 'monitoring'
        ]
        if v not in valid_categories:
            raise ValueError(f"Invalid category: {v}")
        return v
    
    @root_validator
    def validate_effective_dates(cls, values):
        """Valide les dates d'efficacité"""
        effective_from = values.get('effective_from')
        effective_until = values.get('effective_until')
        
        if effective_from and effective_until:
            if effective_from >= effective_until:
                raise ValueError("effective_from must be before effective_until")
        
        return values


# ================================
# Alert Configuration Schema
# ================================

class AlertChannelConfig(BaseModel):
    """Configuration d'un canal d'alerte"""
    
    channel_type: str = Field(..., description="Type de canal")
    enabled: bool = Field(default=True)
    config: Dict[str, Any] = Field(..., description="Configuration spécifique")
    retry_policy: Dict[str, int] = Field(default_factory=dict)
    rate_limit: Optional[int] = Field(None, description="Limite par heure")
    
    @validator('channel_type')
    def validate_channel_type(cls, v):
        """Valide le type de canal"""
        valid_types = ['slack', 'email', 'sms', 'webhook', 'siem', 'dashboard', 'push']
        if v not in valid_types:
            raise ValueError(f"Invalid channel type: {v}")
        return v
    
    @validator('retry_policy')
    def validate_retry_policy(cls, v):
        """Valide la politique de retry"""
        if not v:
            return {
                "max_retries": 3,
                "initial_delay_seconds": 5,
                "max_delay_seconds": 300,
                "backoff_multiplier": 2
            }
        return v


class AlertTemplate(BaseModel):
    """Template d'alerte"""
    
    name: str = Field(..., min_length=1, max_length=100)
    subject_template: str = Field(..., description="Template du sujet")
    body_template: str = Field(..., description="Template du corps")
    severity: SecurityLevel = Field(..., description="Niveau de sévérité")
    format_type: str = Field(default="markdown", description="Format du message")
    variables: List[str] = Field(default_factory=list, description="Variables disponibles")
    
    @validator('format_type')
    def validate_format_type(cls, v):
        """Valide le type de format"""
        if v not in ['plain', 'markdown', 'html', 'json']:
            raise ValueError(f"Invalid format type: {v}")
        return v


class AlertConfigSchema(BaseSecuritySchema):
    """Schéma de configuration des alertes"""
    
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=500)
    
    # Configuration des canaux
    channels: Dict[str, AlertChannelConfig] = Field(..., min_items=1)
    default_channel: str = Field(..., description="Canal par défaut")
    
    # Configuration des templates
    templates: Dict[SecurityLevel, AlertTemplate] = Field(..., min_items=1)
    
    # Configuration de l'escalade
    escalation_enabled: bool = Field(default=True)
    escalation_rules: List[Dict[str, Any]] = Field(default_factory=list)
    escalation_timeout_minutes: int = Field(default=30, ge=1, le=1440)
    
    # Configuration de la suppression
    suppression_enabled: bool = Field(default=True)
    suppression_window_minutes: int = Field(default=5, ge=1, le=60)
    max_alerts_per_window: int = Field(default=10, ge=1, le=1000)
    
    # Configuration de l'agrégation
    aggregation_enabled: bool = Field(default=True)
    aggregation_window_minutes: int = Field(default=15, ge=1, le=60)
    aggregation_threshold: int = Field(default=5, ge=1, le=100)
    
    @validator('default_channel')
    def validate_default_channel(cls, v, values):
        """Valide que le canal par défaut existe"""
        channels = values.get('channels', {})
        if v not in channels:
            raise ValueError(f"Default channel '{v}' not found in channels")
        return v
    
    @root_validator
    def validate_escalation_rules(cls, values):
        """Valide les règles d'escalade"""
        if values.get('escalation_enabled'):
            rules = values.get('escalation_rules', [])
            if not rules:
                # Règles par défaut
                values['escalation_rules'] = [
                    {
                        "severity": SecurityLevel.HIGH.value,
                        "delay_minutes": 15,
                        "channels": ["email", "slack"]
                    },
                    {
                        "severity": SecurityLevel.CRITICAL.value,
                        "delay_minutes": 5,
                        "channels": ["sms", "slack", "email"]
                    }
                ]
        
        return values


# ================================
# Permission Schema
# ================================

class ResourcePermission(BaseModel):
    """Permission sur une ressource"""
    
    resource_type: str = Field(..., description="Type de ressource")
    resource_id: Optional[str] = Field(None, description="ID spécifique de la ressource")
    permissions: List[PermissionType] = Field(..., min_items=1)
    conditions: List[SecurityRuleCondition] = Field(default_factory=list)
    
    # Métadonnées temporelles
    granted_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None)
    last_used: Optional[datetime] = Field(None)
    
    @validator('resource_type')
    def validate_resource_type(cls, v):
        """Valide le type de ressource"""
        valid_types = [
            'audio_file', 'playlist', 'user_data', 'ml_model',
            'analytics_data', 'system_config', 'tenant_config',
            'security_rule', 'audit_log', 'api_endpoint'
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid resource type: {v}")
        return v


class RoleDefinition(BaseModel):
    """Définition d'un rôle"""
    
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=500)
    permissions: List[ResourcePermission] = Field(..., min_items=1)
    inherits_from: List[str] = Field(default_factory=list, description="Rôles hérités")
    
    # Configuration
    is_system_role: bool = Field(default=False)
    is_assignable: bool = Field(default=True)
    max_assignments: Optional[int] = Field(None, ge=1)
    
    @validator('name')
    def validate_name(cls, v):
        """Valide le nom du rôle"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Role name can only contain alphanumeric characters, underscores, and hyphens")
        return v


class PermissionSchema(BaseSecuritySchema):
    """Schéma de gestion des permissions"""
    
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=500)
    
    # Définition des rôles
    roles: Dict[str, RoleDefinition] = Field(..., min_items=1)
    default_role: str = Field(..., description="Rôle par défaut")
    
    # Configuration RBAC
    rbac_enabled: bool = Field(default=True)
    abac_enabled: bool = Field(default=False, description="Attribute-Based Access Control")
    dynamic_permissions: bool = Field(default=False)
    
    # Configuration d'héritage
    inheritance_enabled: bool = Field(default=True)
    max_inheritance_depth: int = Field(default=5, ge=1, le=10)
    
    # Configuration de validation
    strict_validation: bool = Field(default=True)
    permission_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)
    
    @validator('default_role')
    def validate_default_role(cls, v, values):
        """Valide que le rôle par défaut existe"""
        roles = values.get('roles', {})
        if v not in roles:
            raise ValueError(f"Default role '{v}' not found in roles")
        return v
    
    @root_validator
    def validate_role_inheritance(cls, values):
        """Valide l'héritage des rôles"""
        roles = values.get('roles', {})
        
        for role_name, role_def in roles.items():
            for inherited_role in role_def.inherits_from:
                if inherited_role not in roles:
                    raise ValueError(f"Role '{role_name}' inherits from non-existent role '{inherited_role}'")
                
                # Vérification des cycles d'héritage
                if cls._has_inheritance_cycle(roles, role_name, inherited_role):
                    raise ValueError(f"Inheritance cycle detected involving role '{role_name}'")
        
        return values
    
    @staticmethod
    def _has_inheritance_cycle(roles: Dict[str, RoleDefinition], start_role: str, 
                             current_role: str, visited: set = None) -> bool:
        """Détecte les cycles d'héritage"""
        if visited is None:
            visited = set()
        
        if current_role in visited:
            return True
        
        visited.add(current_role)
        
        role_def = roles.get(current_role)
        if not role_def:
            return False
        
        for inherited_role in role_def.inherits_from:
            if PermissionSchema._has_inheritance_cycle(roles, start_role, inherited_role, visited.copy()):
                return True
        
        return False


# ================================
# Audit Schema
# ================================

class AuditEventType(Enum):
    """Types d'événements d'audit"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIG_CHANGE = "config_change"
    SECURITY_RULE_TRIGGERED = "security_rule_triggered"
    ALERT_SENT = "alert_sent"
    SYSTEM_ERROR = "system_error"


class AuditSchema(BaseSecuritySchema):
    """Schéma pour les événements d'audit"""
    
    event_type: AuditEventType = Field(..., description="Type d'événement")
    user_id: Optional[str] = Field(None, description="Utilisateur concerné")
    session_id: Optional[str] = Field(None, description="Session concernée")
    
    # Détails de l'événement
    resource_type: Optional[str] = Field(None)
    resource_id: Optional[str] = Field(None)
    action: str = Field(..., description="Action effectuée")
    result: str = Field(..., description="Résultat de l'action")
    
    # Contexte technique
    source_ip: Optional[str] = Field(None)
    user_agent: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)
    api_endpoint: Optional[str] = Field(None)
    
    # Métadonnées
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sensitive_data_logged: bool = Field(default=False)
    
    # Géolocalisation
    country: Optional[str] = Field(None)
    region: Optional[str] = Field(None)
    city: Optional[str] = Field(None)
    
    # Classification
    severity: SecurityLevel = Field(default=SecurityLevel.LOW)
    compliance_relevant: bool = Field(default=False)
    retention_period_days: int = Field(default=365, ge=1, le=3650)
    
    @validator('source_ip')
    def validate_source_ip(cls, v):
        """Valide l'adresse IP source"""
        if v:
            try:
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError(f"Invalid IP address: {v}")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v, values):
        """Valide les métadonnées"""
        # Vérification des données sensibles
        if values.get('sensitive_data_logged', False):
            # Masquer les données sensibles
            sensitive_keys = ['password', 'token', 'key', 'secret', 'credit_card']
            for key in list(v.keys()):
                if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                    v[key] = "[REDACTED]"
        
        return v
