"""
👤 User Model - Modèle Utilisateur Multi-Tenant
==============================================

Modèle de données pour les utilisateurs dans l'architecture multi-tenant.
Gère les utilisateurs, rôles, permissions et authentification.

Author: Spécialiste Sécurité + Lead Dev - Fahed Mlaiel
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib

from sqlalchemy import Column, String, DateTime, Boolean, JSON, Integer, Text, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, validator, EmailStr

Base = declarative_base()


class UserStatus(str, Enum):
    """États des utilisateurs"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    LOCKED = "locked"


class AuthenticationMethod(str, Enum):
    """Méthodes d'authentification"""
    PASSWORD = "password"
    SSO = "sso"
    MFA = "mfa"
    API_KEY = "api_key"
    OAUTH = "oauth"


class PermissionScope(str, Enum):
    """Portées des permissions"""
    TENANT = "tenant"            # Accès global au tenant
    RESOURCE = "resource"        # Accès à une ressource spécifique
    FEATURE = "feature"         # Accès à une fonctionnalité
    API = "api"                 # Accès API


class PermissionAction(str, Enum):
    """Actions des permissions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


# Table d'association pour les rôles utilisateur
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', String, ForeignKey('tenant_users.id'), primary_key=True),
    Column('role_id', String, ForeignKey('user_roles_def.id'), primary_key=True)
)


class TenantUser(Base):
    """
    Modèle utilisateur multi-tenant.
    
    Attributes:
        id: Identifiant unique
        tenant_id: ID du tenant propriétaire
        email: Adresse email (unique par tenant)
        username: Nom d'utilisateur (unique par tenant)
        password_hash: Hash du mot de passe
        first_name: Prénom
        last_name: Nom de famille
        status: État du compte
        auth_methods: Méthodes d'authentification autorisées
        preferences: Préférences utilisateur
        metadata: Métadonnées additionnelles
        last_login_at: Dernière connexion
        password_changed_at: Dernière modification du mot de passe
        failed_login_attempts: Tentatives d'échec de connexion
        locked_until: Verrouillage jusqu'à (si applicable)
        created_at: Date de création
        updated_at: Date de dernière modification
    """
    
    __tablename__ = "tenant_users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    
    # Identifiants
    email = Column(String(320), nullable=False)  # RFC 5321 max length
    username = Column(String(150), nullable=False)
    password_hash = Column(String(255), nullable=True)  # Nullable pour SSO
    
    # Informations personnelles
    first_name = Column(String(150))
    last_name = Column(String(150))
    full_name = Column(String(300))
    avatar_url = Column(String(500))
    
    # État et sécurité
    status = Column(String(50), nullable=False, default=UserStatus.PENDING)
    auth_methods = Column(JSON, default=list)
    is_verified = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    
    # Configuration
    preferences = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    # Tracking sécurité
    last_login_at = Column(DateTime)
    last_login_ip = Column(String(45))  # IPv6 max length
    password_changed_at = Column(DateTime)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relations
    tenant = relationship("Tenant", back_populates="users")
    roles = relationship("UserRole", secondary=user_roles, back_populates="users")
    api_keys = relationship("UserApiKey", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<TenantUser(id='{self.id}', email='{self.email}', tenant_id='{self.tenant_id}')>"
    
    @property
    def is_active(self) -> bool:
        """Vérifier si l'utilisateur est actif"""
        return self.status == UserStatus.ACTIVE and self.is_verified
    
    @property
    def is_locked(self) -> bool:
        """Vérifier si l'utilisateur est verrouillé"""
        if self.status == UserStatus.LOCKED:
            return True
        if self.locked_until and datetime.utcnow() < self.locked_until:
            return True
        return False
    
    @property
    def display_name(self) -> str:
        """Nom d'affichage"""
        if self.full_name:
            return self.full_name
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        if self.first_name:
            return self.first_name
        return self.username or self.email
    
    def has_permission(self, permission: str, resource: str = None) -> bool:
        """Vérifier si l'utilisateur a une permission"""
        # Admin a toutes les permissions
        if self.is_admin:
            return True
        
        # Vérification via les rôles
        for role in self.roles:
            if role.has_permission(permission, resource):
                return True
        
        return False
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Récupérer une préférence"""
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value: Any):
        """Définir une préférence"""
        if not self.preferences:
            self.preferences = {}
        self.preferences[key] = value
    
    def increment_failed_login(self, max_attempts: int = 5, lockout_duration: int = 300):
        """Incrémenter les échecs de connexion"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= max_attempts:
            self.status = UserStatus.LOCKED
            self.locked_until = datetime.utcnow() + timedelta(seconds=lockout_duration)
    
    def reset_failed_login(self):
        """Réinitialiser les échecs de connexion"""
        self.failed_login_attempts = 0
        if self.status == UserStatus.LOCKED:
            self.status = UserStatus.ACTIVE
        self.locked_until = None
    
    def record_login(self, ip_address: str = None):
        """Enregistrer une connexion réussie"""
        self.last_login_at = datetime.utcnow()
        if ip_address:
            self.last_login_ip = ip_address
        self.reset_failed_login()


class UserRole(Base):
    """
    Rôles utilisateur avec permissions.
    
    Définit les rôles disponibles et leurs permissions
    associées dans le contexte d'un tenant.
    """
    
    __tablename__ = "user_roles_def"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    name = Column(String(100), nullable=False)
    display_name = Column(String(200))
    description = Column(Text)
    permissions = Column(JSON, default=list)
    is_system_role = Column(Boolean, default=False)  # Rôle système non modifiable
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relations
    users = relationship("TenantUser", secondary=user_roles, back_populates="roles")
    
    def __repr__(self):
        return f"<UserRole(id='{self.id}', name='{self.name}', tenant_id='{self.tenant_id}')>"
    
    def has_permission(self, permission: str, resource: str = None) -> bool:
        """Vérifier si le rôle a une permission"""
        if not self.permissions:
            return False
        
        # Recherche directe de la permission
        if permission in self.permissions:
            return True
        
        # Recherche avec ressource spécifique
        if resource:
            resource_permission = f"{resource}:{permission}"
            if resource_permission in self.permissions:
                return True
        
        # Vérification des permissions wildcard
        for perm in self.permissions:
            if perm.endswith('*'):
                prefix = perm[:-1]
                if permission.startswith(prefix):
                    return True
                if resource and resource_permission.startswith(prefix):
                    return True
        
        return False
    
    def add_permission(self, permission: str):
        """Ajouter une permission"""
        if not self.permissions:
            self.permissions = []
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def remove_permission(self, permission: str):
        """Retirer une permission"""
        if self.permissions and permission in self.permissions:
            self.permissions.remove(permission)


class UserPermission(Base):
    """
    Permissions spécifiques par utilisateur.
    
    Permet d'attribuer des permissions individuelles
    en plus des rôles.
    """
    
    __tablename__ = "user_permissions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("tenant_users.id"), nullable=False)
    permission = Column(String(255), nullable=False)
    resource = Column(String(255))  # Ressource spécifique (optionnel)
    scope = Column(String(50), nullable=False, default=PermissionScope.TENANT)
    granted_by = Column(String, ForeignKey("tenant_users.id"))
    expires_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserPermission(user_id='{self.user_id}', permission='{self.permission}')>"
    
    @property
    def is_active(self) -> bool:
        """Vérifier si la permission est active"""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


class UserApiKey(Base):
    """
    Clés API pour l'authentification programmatique.
    """
    
    __tablename__ = "user_api_keys"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("tenant_users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), nullable=False)
    permissions = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime)
    usage_count = Column(Integer, default=0)
    expires_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relations
    user = relationship("TenantUser", back_populates="api_keys")
    
    def __repr__(self):
        return f"<UserApiKey(id='{self.id}', name='{self.name}', user_id='{self.user_id}')>"
    
    @property
    def is_valid(self) -> bool:
        """Vérifier si la clé API est valide"""
        if not self.is_active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def record_usage(self):
        """Enregistrer une utilisation"""
        self.last_used_at = datetime.utcnow()
        self.usage_count += 1


# Modèles Pydantic pour l'API

class UserCreate(BaseModel):
    """Modèle de création d'utilisateur"""
    email: EmailStr
    username: str
    password: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    auth_methods: List[AuthenticationMethod] = [AuthenticationMethod.PASSWORD]
    roles: List[str] = []
    preferences: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    @validator("username")
    def validate_username(cls, v):
        if not v or len(v) < 3:
            raise ValueError("Le nom d'utilisateur doit contenir au moins 3 caractères")
        if not v.isalnum() and '_' not in v and '-' not in v:
            raise ValueError("Le nom d'utilisateur ne peut contenir que des lettres, chiffres, tirets et underscores")
        return v.lower()


class UserUpdate(BaseModel):
    """Modèle de mise à jour d'utilisateur"""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    status: Optional[UserStatus] = None
    auth_methods: Optional[List[AuthenticationMethod]] = None
    roles: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class UserResponse(BaseModel):
    """Modèle de réponse utilisateur"""
    id: str
    tenant_id: str
    email: str
    username: str
    first_name: Optional[str]
    last_name: Optional[str]
    display_name: str
    status: UserStatus
    auth_methods: List[str]
    is_verified: bool
    is_admin: bool
    preferences: Dict[str, Any]
    metadata: Dict[str, Any]
    last_login_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    is_locked: bool
    roles: List[str] = []
    
    class Config:
        from_attributes = True


class UserRoleCreate(BaseModel):
    """Modèle de création de rôle"""
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    permissions: List[str] = []
    
    @validator("name")
    def validate_name(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Le nom du rôle doit contenir au moins 2 caractères")
        return v.upper().replace(' ', '_')


class UserRoleResponse(BaseModel):
    """Modèle de réponse rôle"""
    id: str
    tenant_id: str
    name: str
    display_name: Optional[str]
    description: Optional[str]
    permissions: List[str]
    is_system_role: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ApiKeyCreate(BaseModel):
    """Modèle de création de clé API"""
    name: str
    permissions: List[str] = []
    expires_at: Optional[datetime] = None


class ApiKeyResponse(BaseModel):
    """Modèle de réponse clé API"""
    id: str
    name: str
    permissions: List[str]
    is_active: bool
    last_used_at: Optional[datetime]
    usage_count: int
    expires_at: Optional[datetime]
    created_at: datetime
    is_valid: bool
    
    class Config:
        from_attributes = True


# Rôles système par défaut
DEFAULT_SYSTEM_ROLES = {
    "SUPER_ADMIN": {
        "display_name": "Super Administrateur",
        "description": "Accès complet à toutes les fonctionnalités",
        "permissions": ["*"]
    },
    "ADMIN": {
        "display_name": "Administrateur",
        "description": "Administration du tenant",
        "permissions": [
            "tenant:admin",
            "users:*",
            "roles:*",
            "settings:*",
            "billing:read",
            "analytics:read"
        ]
    },
    "MANAGER": {
        "display_name": "Gestionnaire",
        "description": "Gestion des utilisateurs et contenu",
        "permissions": [
            "users:read",
            "users:create",
            "users:update",
            "content:*",
            "analytics:read"
        ]
    },
    "USER": {
        "display_name": "Utilisateur",
        "description": "Accès utilisateur standard",
        "permissions": [
            "profile:read",
            "profile:update",
            "content:read",
            "content:create"
        ]
    },
    "VIEWER": {
        "display_name": "Lecteur",
        "description": "Accès en lecture seule",
        "permissions": [
            "profile:read",
            "content:read"
        ]
    }
}

# Permissions disponibles par catégorie
AVAILABLE_PERMISSIONS = {
    "tenant": ["read", "update", "admin"],
    "users": ["create", "read", "update", "delete", "admin"],
    "roles": ["create", "read", "update", "delete"],
    "content": ["create", "read", "update", "delete"],
    "settings": ["read", "update"],
    "billing": ["read", "update"],
    "analytics": ["read"],
    "api": ["read", "write"],
    "profile": ["read", "update"]
}
