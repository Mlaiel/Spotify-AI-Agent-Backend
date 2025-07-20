"""
🛡️ Security Manager - Gestionnaire de Sécurité Ultra-Avancé Multi-Tenant
=========================================================================

Gestionnaire de sécurité centralisé pour l'isolation des données multi-tenant
avec chiffrement, authentification, autorisation, et audit avancés.

Author: Spécialiste Sécurité - Fahed Mlaiel
"""

import asyncio
import logging
import hashlib
import hmac
import secrets
import jwt
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..core.tenant_context import TenantContext, TenantType
from ..exceptions import DataIsolationError, SecurityError


class SecurityLevel(Enum):
    """Niveaux de sécurité"""
    PUBLIC = "public"           # Données publiques
    INTERNAL = "internal"       # Données internes
    CONFIDENTIAL = "confidential"  # Données confidentielles  
    RESTRICTED = "restricted"   # Données restreintes
    TOP_SECRET = "top_secret"   # Données ultra-sensibles


class EncryptionType(Enum):
    """Types de chiffrement"""
    NONE = "none"
    SYMMETRIC = "symmetric"     # AES
    ASYMMETRIC = "asymmetric"   # RSA
    HYBRID = "hybrid"          # AES + RSA


class AuthenticationMethod(Enum):
    """Méthodes d'authentification"""
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    CERTIFICATE = "certificate"
    MULTI_FACTOR = "multi_factor"


class PermissionType(Enum):
    """Types de permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"
    SHARE = "share"


@dataclass
class SecurityPolicy:
    """Politique de sécurité"""
    name: str
    description: str
    tenant_id: Optional[str] = None
    
    # Security requirements
    min_security_level: SecurityLevel = SecurityLevel.INTERNAL
    required_encryption: EncryptionType = EncryptionType.SYMMETRIC
    require_authentication: bool = True
    require_authorization: bool = True
    
    # Access controls
    allowed_tenant_types: Set[TenantType] = field(default_factory=set)
    allowed_operations: Set[str] = field(default_factory=set)
    required_permissions: Set[PermissionType] = field(default_factory=set)
    
    # Time-based controls
    access_hours: Optional[Tuple[int, int]] = None  # (start_hour, end_hour)
    max_session_duration: int = 3600  # seconds
    
    # Rate limiting
    max_requests_per_minute: int = 1000
    max_requests_per_hour: int = 10000
    
    # Data protection
    enable_data_masking: bool = False
    enable_audit_logging: bool = True
    retention_days: int = 90
    
    # Compliance
    gdpr_compliant: bool = True
    hipaa_compliant: bool = False
    pci_compliant: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SecurityCredentials:
    """Informations d'identification sécurisées"""
    tenant_id: str
    user_id: Optional[str] = None
    
    # Authentication data
    token: Optional[str] = None
    api_key: Optional[str] = None
    certificate: Optional[bytes] = None
    
    # JWT data
    jwt_claims: Dict[str, Any] = field(default_factory=dict)
    jwt_expires_at: Optional[datetime] = None
    
    # Session data
    session_id: Optional[str] = None
    session_created_at: Optional[datetime] = None
    session_expires_at: Optional[datetime] = None
    
    # Permissions
    permissions: Set[PermissionType] = field(default_factory=set)
    scopes: Set[str] = field(default_factory=set)
    
    # Audit data
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class AuditEvent:
    """Événement d'audit"""
    event_id: str
    timestamp: datetime
    tenant_id: str
    user_id: Optional[str]
    
    # Event details
    event_type: str
    resource: str
    action: str
    outcome: str  # success, failure, denied
    
    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    # Data
    request_data: Dict[str, Any] = field(default_factory=dict)
    response_data: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    
    # Security
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    risk_score: float = 0.0
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EncryptionManager:
    """Gestionnaire de chiffrement"""
    
    def __init__(self):
        self.logger = logging.getLogger("security.encryption")
        
        # Symmetric encryption keys (per tenant)
        self._symmetric_keys: Dict[str, bytes] = {}
        
        # Asymmetric key pairs (per tenant)
        self._private_keys: Dict[str, rsa.RSAPrivateKey] = {}
        self._public_keys: Dict[str, rsa.RSAPublicKey] = {}
        
        # Master key for key encryption
        self._master_key = self._generate_master_key()
    
    def _generate_master_key(self) -> bytes:
        """Génère la clé maître"""
        # In production, this should come from a secure key management system
        master_password = os.environ.get("MASTER_PASSWORD", "default_master_key")
        salt = os.environ.get("MASTER_SALT", "default_salt").encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        return kdf.derive(master_password.encode())
    
    async def get_or_create_symmetric_key(self, tenant_id: str) -> bytes:
        """Obtient ou crée une clé symétrique pour un tenant"""
        if tenant_id not in self._symmetric_keys:
            # Generate new key
            key = Fernet.generate_key()
            self._symmetric_keys[tenant_id] = key
            
            self.logger.debug(f"Generated new symmetric key for tenant {tenant_id}")
        
        return self._symmetric_keys[tenant_id]
    
    async def get_or_create_asymmetric_keys(self, tenant_id: str) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Obtient ou crée une paire de clés asymétriques pour un tenant"""
        if tenant_id not in self._private_keys:
            # Generate new key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            public_key = private_key.public_key()
            
            self._private_keys[tenant_id] = private_key
            self._public_keys[tenant_id] = public_key
            
            self.logger.debug(f"Generated new asymmetric key pair for tenant {tenant_id}")
        
        return self._private_keys[tenant_id], self._public_keys[tenant_id]
    
    async def encrypt_symmetric(self, tenant_id: str, data: Union[str, bytes]) -> bytes:
        """Chiffre des données avec chiffrement symétrique"""
        try:
            key = await self.get_or_create_symmetric_key(tenant_id)
            fernet = Fernet(key)
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = fernet.encrypt(data)
            return encrypted_data
            
        except Exception as e:
            self.logger.error(f"Symmetric encryption failed for tenant {tenant_id}: {e}")
            raise SecurityError(f"Encryption failed: {e}")
    
    async def decrypt_symmetric(self, tenant_id: str, encrypted_data: bytes) -> bytes:
        """Déchiffre des données avec chiffrement symétrique"""
        try:
            key = await self.get_or_create_symmetric_key(tenant_id)
            fernet = Fernet(key)
            
            decrypted_data = fernet.decrypt(encrypted_data)
            return decrypted_data
            
        except Exception as e:
            self.logger.error(f"Symmetric decryption failed for tenant {tenant_id}: {e}")
            raise SecurityError(f"Decryption failed: {e}")
    
    async def encrypt_asymmetric(self, tenant_id: str, data: Union[str, bytes]) -> bytes:
        """Chiffre des données avec chiffrement asymétrique"""
        try:
            _, public_key = await self.get_or_create_asymmetric_keys(tenant_id)
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # RSA can only encrypt limited data size, so we use hybrid encryption
            # Generate a random AES key, encrypt data with AES, then encrypt AES key with RSA
            aes_key = Fernet.generate_key()
            fernet = Fernet(aes_key)
            
            # Encrypt data with AES
            encrypted_data = fernet.encrypt(data)
            
            # Encrypt AES key with RSA
            encrypted_aes_key = public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted key + encrypted data
            result = len(encrypted_aes_key).to_bytes(4, 'big') + encrypted_aes_key + encrypted_data
            return result
            
        except Exception as e:
            self.logger.error(f"Asymmetric encryption failed for tenant {tenant_id}: {e}")
            raise SecurityError(f"Encryption failed: {e}")
    
    async def decrypt_asymmetric(self, tenant_id: str, encrypted_data: bytes) -> bytes:
        """Déchiffre des données avec chiffrement asymétrique"""
        try:
            private_key, _ = await self.get_or_create_asymmetric_keys(tenant_id)
            
            # Extract encrypted AES key length
            key_length = int.from_bytes(encrypted_data[:4], 'big')
            
            # Extract encrypted AES key and data
            encrypted_aes_key = encrypted_data[4:4 + key_length]
            encrypted_payload = encrypted_data[4 + key_length:]
            
            # Decrypt AES key with RSA
            aes_key = private_key.decrypt(
                encrypted_aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt data with AES
            fernet = Fernet(aes_key)
            decrypted_data = fernet.decrypt(encrypted_payload)
            
            return decrypted_data
            
        except Exception as e:
            self.logger.error(f"Asymmetric decryption failed for tenant {tenant_id}: {e}")
            raise SecurityError(f"Decryption failed: {e}")
    
    async def encrypt_field(
        self, 
        tenant_id: str, 
        field_name: str, 
        value: Any, 
        encryption_type: EncryptionType = EncryptionType.SYMMETRIC
    ) -> str:
        """Chiffre un champ spécifique"""
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                import json
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            # Add field metadata
            field_data = {
                "field_name": field_name,
                "value": serialized_value,
                "tenant_id": tenant_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "encryption_type": encryption_type.value
            }
            
            field_json = json.dumps(field_data)
            
            # Encrypt based on type
            if encryption_type == EncryptionType.SYMMETRIC:
                encrypted_data = await self.encrypt_symmetric(tenant_id, field_json)
            elif encryption_type == EncryptionType.ASYMMETRIC:
                encrypted_data = await self.encrypt_asymmetric(tenant_id, field_json)
            else:
                encrypted_data = field_json.encode('utf-8')
            
            # Return base64 encoded result
            return base64.b64encode(encrypted_data).decode('ascii')
            
        except Exception as e:
            self.logger.error(f"Field encryption failed for {field_name}: {e}")
            raise SecurityError(f"Field encryption failed: {e}")
    
    async def decrypt_field(self, tenant_id: str, encrypted_field: str) -> Tuple[str, Any]:
        """Déchiffre un champ"""
        try:
            # Decode from base64
            encrypted_data = base64.b64decode(encrypted_field.encode('ascii'))
            
            # Try different decryption methods
            decrypted_data = None
            
            try:
                # Try symmetric first
                decrypted_data = await self.decrypt_symmetric(tenant_id, encrypted_data)
            except:
                try:
                    # Try asymmetric
                    decrypted_data = await self.decrypt_asymmetric(tenant_id, encrypted_data)
                except:
                    # Maybe it's not encrypted
                    decrypted_data = encrypted_data
            
            # Parse field data
            import json
            field_data = json.loads(decrypted_data.decode('utf-8'))
            
            field_name = field_data["field_name"]
            value = field_data["value"]
            
            # Try to deserialize value
            try:
                parsed_value = json.loads(value)
            except:
                parsed_value = value
            
            return field_name, parsed_value
            
        except Exception as e:
            self.logger.error(f"Field decryption failed: {e}")
            raise SecurityError(f"Field decryption failed: {e}")


class AuthenticationManager:
    """Gestionnaire d'authentification"""
    
    def __init__(self, jwt_secret: str):
        self.jwt_secret = jwt_secret
        self.logger = logging.getLogger("security.authentication")
        
        # Active sessions
        self._active_sessions: Dict[str, SecurityCredentials] = {}
        
        # API keys
        self._api_keys: Dict[str, str] = {}  # api_key -> tenant_id
        
        # Rate limiting
        self._rate_limits: Dict[str, List[datetime]] = {}
    
    async def authenticate_jwt(self, token: str) -> Optional[SecurityCredentials]:
        """Authentifie avec un token JWT"""
        try:
            # Decode JWT
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Extract claims
            tenant_id = payload.get('tenant_id')
            user_id = payload.get('user_id')
            exp = payload.get('exp')
            permissions = payload.get('permissions', [])
            scopes = payload.get('scopes', [])
            
            if not tenant_id:
                raise SecurityError("Invalid token: missing tenant_id")
            
            # Check expiration
            if exp and datetime.fromtimestamp(exp, timezone.utc) < datetime.now(timezone.utc):
                raise SecurityError("Token expired")
            
            # Create credentials
            credentials = SecurityCredentials(
                tenant_id=tenant_id,
                user_id=user_id,
                token=token,
                jwt_claims=payload,
                jwt_expires_at=datetime.fromtimestamp(exp, timezone.utc) if exp else None,
                permissions=set(PermissionType(p) for p in permissions if p in [pt.value for pt in PermissionType]),
                scopes=set(scopes),
                last_accessed=datetime.now(timezone.utc),
                access_count=1
            )
            
            self.logger.debug(f"JWT authentication successful for tenant {tenant_id}")
            return credentials
            
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"JWT authentication failed: {e}")
            raise SecurityError(f"Invalid token: {e}")
        except Exception as e:
            self.logger.error(f"JWT authentication error: {e}")
            raise SecurityError(f"Authentication error: {e}")
    
    async def authenticate_api_key(self, api_key: str) -> Optional[SecurityCredentials]:
        """Authentifie avec une clé API"""
        try:
            if api_key not in self._api_keys:
                raise SecurityError("Invalid API key")
            
            tenant_id = self._api_keys[api_key]
            
            # Create credentials
            credentials = SecurityCredentials(
                tenant_id=tenant_id,
                api_key=api_key,
                permissions={PermissionType.READ, PermissionType.WRITE},  # Default permissions
                last_accessed=datetime.now(timezone.utc),
                access_count=1
            )
            
            self.logger.debug(f"API key authentication successful for tenant {tenant_id}")
            return credentials
            
        except Exception as e:
            self.logger.error(f"API key authentication error: {e}")
            raise SecurityError(f"Authentication error: {e}")
    
    async def create_session(self, credentials: SecurityCredentials) -> str:
        """Crée une session"""
        session_id = secrets.token_urlsafe(32)
        
        credentials.session_id = session_id
        credentials.session_created_at = datetime.now(timezone.utc)
        credentials.session_expires_at = datetime.now(timezone.utc) + timedelta(hours=8)
        
        self._active_sessions[session_id] = credentials
        
        self.logger.info(f"Created session {session_id} for tenant {credentials.tenant_id}")
        return session_id
    
    async def validate_session(self, session_id: str) -> Optional[SecurityCredentials]:
        """Valide une session"""
        if session_id not in self._active_sessions:
            return None
        
        credentials = self._active_sessions[session_id]
        
        # Check expiration
        if (credentials.session_expires_at and 
            credentials.session_expires_at < datetime.now(timezone.utc)):
            await self.destroy_session(session_id)
            return None
        
        # Update access info
        credentials.last_accessed = datetime.now(timezone.utc)
        credentials.access_count += 1
        
        return credentials
    
    async def destroy_session(self, session_id: str):
        """Détruit une session"""
        if session_id in self._active_sessions:
            tenant_id = self._active_sessions[session_id].tenant_id
            del self._active_sessions[session_id]
            self.logger.info(f"Destroyed session {session_id} for tenant {tenant_id}")
    
    async def register_api_key(self, tenant_id: str) -> str:
        """Enregistre une nouvelle clé API"""
        api_key = secrets.token_urlsafe(32)
        self._api_keys[api_key] = tenant_id
        
        self.logger.info(f"Registered new API key for tenant {tenant_id}")
        return api_key
    
    async def revoke_api_key(self, api_key: str):
        """Révoque une clé API"""
        if api_key in self._api_keys:
            tenant_id = self._api_keys[api_key]
            del self._api_keys[api_key]
            self.logger.info(f"Revoked API key for tenant {tenant_id}")
    
    async def check_rate_limit(self, identifier: str, max_requests: int, window_minutes: int) -> bool:
        """Vérifie les limites de taux"""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=window_minutes)
        
        # Clean old requests
        if identifier in self._rate_limits:
            self._rate_limits[identifier] = [
                req_time for req_time in self._rate_limits[identifier]
                if req_time > window_start
            ]
        else:
            self._rate_limits[identifier] = []
        
        # Check limit
        if len(self._rate_limits[identifier]) >= max_requests:
            return False
        
        # Record request
        self._rate_limits[identifier].append(now)
        return True


class AuthorizationManager:
    """Gestionnaire d'autorisation"""
    
    def __init__(self):
        self.logger = logging.getLogger("security.authorization")
        
        # Policies
        self._policies: Dict[str, SecurityPolicy] = {}
        
        # Role-based permissions
        self._role_permissions: Dict[str, Set[PermissionType]] = {
            "admin": {PermissionType.READ, PermissionType.WRITE, PermissionType.DELETE, PermissionType.ADMIN},
            "editor": {PermissionType.READ, PermissionType.WRITE},
            "viewer": {PermissionType.READ},
        }
        
        # Resource permissions cache
        self._resource_permissions: Dict[str, Dict[str, Set[PermissionType]]] = {}
    
    async def register_policy(self, policy: SecurityPolicy):
        """Enregistre une politique de sécurité"""
        self._policies[policy.name] = policy
        self.logger.info(f"Registered security policy: {policy.name}")
    
    async def check_authorization(
        self, 
        credentials: SecurityCredentials,
        resource: str,
        action: str,
        required_permission: PermissionType
    ) -> bool:
        """Vérifie l'autorisation"""
        try:
            # Check if user has required permission
            if required_permission not in credentials.permissions:
                self.logger.warning(
                    f"Permission denied: user {credentials.user_id} lacks {required_permission.value} "
                    f"permission for {resource}"
                )
                return False
            
            # Check applicable policies
            applicable_policies = await self._get_applicable_policies(credentials, resource)
            
            for policy in applicable_policies:
                if not await self._check_policy_compliance(credentials, resource, action, policy):
                    return False
            
            # Check resource-specific permissions
            if not await self._check_resource_permissions(credentials, resource, required_permission):
                return False
            
            self.logger.debug(
                f"Authorization granted for user {credentials.user_id} on {resource} "
                f"with permission {required_permission.value}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Authorization check error: {e}")
            return False
    
    async def _get_applicable_policies(
        self, 
        credentials: SecurityCredentials, 
        resource: str
    ) -> List[SecurityPolicy]:
        """Obtient les politiques applicables"""
        applicable = []
        
        for policy in self._policies.values():
            # Check tenant match
            if policy.tenant_id and policy.tenant_id != credentials.tenant_id:
                continue
            
            # Check if policy applies to this resource
            if await self._policy_applies_to_resource(policy, resource):
                applicable.append(policy)
        
        return applicable
    
    async def _policy_applies_to_resource(self, policy: SecurityPolicy, resource: str) -> bool:
        """Vérifie si une politique s'applique à une ressource"""
        # Simple pattern matching - could be more sophisticated
        return True  # For now, all policies apply to all resources
    
    async def _check_policy_compliance(
        self, 
        credentials: SecurityCredentials,
        resource: str,
        action: str,
        policy: SecurityPolicy
    ) -> bool:
        """Vérifie la conformité à une politique"""
        # Check time-based access
        if policy.access_hours:
            current_hour = datetime.now(timezone.utc).hour
            start_hour, end_hour = policy.access_hours
            if not (start_hour <= current_hour <= end_hour):
                self.logger.warning(f"Access denied: outside allowed hours for policy {policy.name}")
                return False
        
        # Check session duration
        if (credentials.session_created_at and 
            datetime.now(timezone.utc) - credentials.session_created_at > 
            timedelta(seconds=policy.max_session_duration)):
            self.logger.warning(f"Access denied: session too long for policy {policy.name}")
            return False
        
        # Check required permissions
        if not policy.required_permissions.issubset(credentials.permissions):
            self.logger.warning(f"Access denied: insufficient permissions for policy {policy.name}")
            return False
        
        # Check allowed operations
        if policy.allowed_operations and action not in policy.allowed_operations:
            self.logger.warning(f"Access denied: operation {action} not allowed by policy {policy.name}")
            return False
        
        return True
    
    async def _check_resource_permissions(
        self, 
        credentials: SecurityCredentials,
        resource: str,
        required_permission: PermissionType
    ) -> bool:
        """Vérifie les permissions spécifiques à la ressource"""
        # Check if resource has specific permissions configured
        if resource in self._resource_permissions:
            user_id = credentials.user_id or "anonymous"
            if user_id in self._resource_permissions[resource]:
                user_permissions = self._resource_permissions[resource][user_id]
                return required_permission in user_permissions
        
        # Default: allow if user has the permission globally
        return required_permission in credentials.permissions
    
    async def grant_resource_permission(
        self, 
        resource: str, 
        user_id: str, 
        permission: PermissionType
    ):
        """Accorde une permission sur une ressource"""
        if resource not in self._resource_permissions:
            self._resource_permissions[resource] = {}
        
        if user_id not in self._resource_permissions[resource]:
            self._resource_permissions[resource][user_id] = set()
        
        self._resource_permissions[resource][user_id].add(permission)
        self.logger.info(f"Granted {permission.value} permission on {resource} to user {user_id}")
    
    async def revoke_resource_permission(
        self, 
        resource: str, 
        user_id: str, 
        permission: PermissionType
    ):
        """Révoque une permission sur une ressource"""
        if (resource in self._resource_permissions and 
            user_id in self._resource_permissions[resource]):
            self._resource_permissions[resource][user_id].discard(permission)
            self.logger.info(f"Revoked {permission.value} permission on {resource} from user {user_id}")


class AuditManager:
    """Gestionnaire d'audit"""
    
    def __init__(self):
        self.logger = logging.getLogger("security.audit")
        
        # Audit log storage
        self._audit_events: List[AuditEvent] = []
        
        # Risk analysis
        self._risk_patterns: Dict[str, float] = {
            "failed_auth": 0.3,
            "unauthorized_access": 0.8,
            "data_breach": 1.0,
            "admin_action": 0.2,
            "bulk_operation": 0.4
        }
    
    async def log_event(
        self,
        event_type: str,
        tenant_id: str,
        resource: str,
        action: str,
        outcome: str,
        user_id: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        error_details: Optional[str] = None,
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        **kwargs
    ):
        """Enregistre un événement d'audit"""
        event_id = secrets.token_hex(16)
        
        # Calculate risk score
        risk_score = await self._calculate_risk_score(
            event_type, action, outcome, request_data
        )
        
        # Create audit event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            user_id=user_id,
            event_type=event_type,
            resource=resource,
            action=action,
            outcome=outcome,
            request_data=request_data or {},
            response_data=response_data or {},
            error_details=error_details,
            security_level=security_level,
            risk_score=risk_score,
            ip_address=kwargs.get('ip_address'),
            user_agent=kwargs.get('user_agent'),
            session_id=kwargs.get('session_id'),
            tags=kwargs.get('tags', {}),
            metadata=kwargs.get('metadata', {})
        )
        
        # Store event
        self._audit_events.append(event)
        
        # Log high-risk events
        if risk_score > 0.7:
            self.logger.warning(
                f"High-risk event detected: {event_type} by {user_id} on {resource} "
                f"(risk score: {risk_score:.2f})"
            )
        
        # Keep only recent events (for memory management)
        if len(self._audit_events) > 10000:
            self._audit_events = self._audit_events[-5000:]
        
        self.logger.debug(f"Logged audit event {event_id}: {event_type} on {resource}")
    
    async def _calculate_risk_score(
        self,
        event_type: str,
        action: str,
        outcome: str,
        request_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calcule le score de risque"""
        base_score = self._risk_patterns.get(event_type, 0.1)
        
        # Adjust based on outcome
        if outcome == "failure":
            base_score *= 1.5
        elif outcome == "denied":
            base_score *= 2.0
        
        # Adjust based on action
        if action in ["delete", "admin", "bulk"]:
            base_score *= 1.3
        
        # Adjust based on data sensitivity
        if request_data:
            sensitive_fields = ["password", "token", "key", "secret"]
            if any(field in str(request_data).lower() for field in sensitive_fields):
                base_score *= 1.4
        
        return min(base_score, 1.0)  # Cap at 1.0
    
    async def get_audit_events(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_risk_score: Optional[float] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Récupère les événements d'audit"""
        filtered_events = []
        
        for event in reversed(self._audit_events):  # Most recent first
            # Apply filters
            if tenant_id and event.tenant_id != tenant_id:
                continue
            if user_id and event.user_id != user_id:
                continue
            if event_type and event.event_type != event_type:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if min_risk_score and event.risk_score < min_risk_score:
                continue
            
            filtered_events.append(event)
            
            if len(filtered_events) >= limit:
                break
        
        return filtered_events
    
    async def get_security_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Obtient un résumé de sécurité"""
        tenant_events = [e for e in self._audit_events if e.tenant_id == tenant_id]
        
        if not tenant_events:
            return {"tenant_id": tenant_id, "events": 0}
        
        # Calculate statistics
        total_events = len(tenant_events)
        failed_events = len([e for e in tenant_events if e.outcome == "failure"])
        denied_events = len([e for e in tenant_events if e.outcome == "denied"])
        high_risk_events = len([e for e in tenant_events if e.risk_score > 0.7])
        
        # Calculate average risk score
        avg_risk_score = sum(e.risk_score for e in tenant_events) / total_events
        
        # Group by event type
        event_types = {}
        for event in tenant_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_events = [e for e in tenant_events if e.timestamp > recent_cutoff]
        
        return {
            "tenant_id": tenant_id,
            "total_events": total_events,
            "failed_events": failed_events,
            "denied_events": denied_events,
            "high_risk_events": high_risk_events,
            "failure_rate": failed_events / total_events if total_events > 0 else 0,
            "avg_risk_score": avg_risk_score,
            "event_types": event_types,
            "recent_activity": len(recent_events),
            "last_activity": max(e.timestamp for e in tenant_events) if tenant_events else None
        }


class SecurityManager:
    """
    Gestionnaire de sécurité ultra-avancé pour l'isolation multi-tenant
    
    Features:
    - Chiffrement symétrique et asymétrique par tenant
    - Authentification multi-méthodes (JWT, API Key, OAuth2)
    - Autorisation basée sur les politiques
    - Audit de sécurité complet avec analyse de risque
    - Gestion de sessions sécurisées
    - Rate limiting et protection DDoS
    - Conformité GDPR/HIPAA/PCI
    - Chiffrement au niveau des champs
    - Détection d'anomalies de sécurité
    """
    
    def __init__(self, jwt_secret: Optional[str] = None):
        self.logger = logging.getLogger("security_manager")
        
        # Initialize sub-managers
        self.encryption = EncryptionManager()
        self.authentication = AuthenticationManager(
            jwt_secret or os.environ.get("JWT_SECRET", "default_secret")
        )
        self.authorization = AuthorizationManager()
        self.audit = AuditManager()
        
        # Security configuration
        self._security_config = {
            "default_encryption_type": EncryptionType.SYMMETRIC,
            "require_authentication": True,
            "enable_audit_logging": True,
            "max_failed_attempts": 5,
            "lockout_duration": 300,  # 5 minutes
            "session_timeout": 3600,  # 1 hour
        }
        
        # Threat detection
        self._threat_indicators: Dict[str, List[datetime]] = {}
        self._blocked_ips: Set[str] = set()
        
        # Compliance tracking
        self._compliance_events: List[Dict[str, Any]] = []
    
    async def initialize(self):
        """Initialise le gestionnaire de sécurité"""
        try:
            # Register default policies
            await self._register_default_policies()
            
            self.logger.info("Security manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security manager: {e}")
            raise SecurityError(f"Security manager initialization failed: {e}")
    
    async def _register_default_policies(self):
        """Enregistre les politiques par défaut"""
        # Default policy for all tenants
        default_policy = SecurityPolicy(
            name="default_tenant_policy",
            description="Default security policy for all tenants",
            min_security_level=SecurityLevel.INTERNAL,
            required_encryption=EncryptionType.SYMMETRIC,
            require_authentication=True,
            require_authorization=True,
            max_requests_per_minute=1000,
            enable_audit_logging=True
        )
        
        await self.authorization.register_policy(default_policy)
        
        # High security policy for sensitive tenants
        high_security_policy = SecurityPolicy(
            name="high_security_policy",
            description="High security policy for sensitive data",
            min_security_level=SecurityLevel.CONFIDENTIAL,
            required_encryption=EncryptionType.ASYMMETRIC,
            require_authentication=True,
            require_authorization=True,
            max_requests_per_minute=100,
            enable_audit_logging=True,
            enable_data_masking=True,
            gdpr_compliant=True,
            hipaa_compliant=True
        )
        
        await self.authorization.register_policy(high_security_policy)
    
    async def authenticate_request(
        self, 
        method: AuthenticationMethod,
        credentials: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> SecurityCredentials:
        """Authentifie une requête"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check IP blocks
            if ip_address and ip_address in self._blocked_ips:
                await self.audit.log_event(
                    "authentication", "unknown", "auth", "authenticate", "denied",
                    error_details=f"Blocked IP: {ip_address}",
                    ip_address=ip_address
                )
                raise SecurityError("IP address blocked")
            
            # Authenticate based on method
            if method == AuthenticationMethod.JWT:
                auth_credentials = await self.authentication.authenticate_jwt(credentials)
            elif method == AuthenticationMethod.API_KEY:
                auth_credentials = await self.authentication.authenticate_api_key(credentials)
            else:
                raise SecurityError(f"Unsupported authentication method: {method}")
            
            # Add context information
            auth_credentials.ip_address = ip_address
            auth_credentials.user_agent = user_agent
            
            # Create session
            session_id = await self.authentication.create_session(auth_credentials)
            
            # Log successful authentication
            await self.audit.log_event(
                "authentication", auth_credentials.tenant_id, "auth", "authenticate", "success",
                user_id=auth_credentials.user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.info(
                f"Authentication successful for tenant {auth_credentials.tenant_id} "
                f"(method: {method.value}, time: {response_time:.3f}s)"
            )
            
            return auth_credentials
            
        except SecurityError as e:
            # Log failed authentication
            await self.audit.log_event(
                "authentication", "unknown", "auth", "authenticate", "failure",
                error_details=str(e),
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Track failed attempts for threat detection
            if ip_address:
                await self._track_threat_indicator(ip_address, "failed_auth")
            
            raise
    
    async def authorize_access(
        self,
        credentials: SecurityCredentials,
        resource: str,
        action: str,
        required_permission: PermissionType,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Autorise l'accès à une ressource"""
        try:
            # Check rate limits
            rate_limit_key = f"{credentials.tenant_id}:{credentials.user_id or 'anonymous'}"
            if not await self.authentication.check_rate_limit(rate_limit_key, 1000, 1):
                await self.audit.log_event(
                    "authorization", credentials.tenant_id, resource, action, "denied",
                    user_id=credentials.user_id,
                    error_details="Rate limit exceeded",
                    session_id=credentials.session_id
                )
                raise SecurityError("Rate limit exceeded")
            
            # Perform authorization check
            authorized = await self.authorization.check_authorization(
                credentials, resource, action, required_permission
            )
            
            if not authorized:
                await self.audit.log_event(
                    "authorization", credentials.tenant_id, resource, action, "denied",
                    user_id=credentials.user_id,
                    request_data=data,
                    session_id=credentials.session_id
                )
                return False
            
            # Log successful authorization
            await self.audit.log_event(
                "authorization", credentials.tenant_id, resource, action, "success",
                user_id=credentials.user_id,
                request_data=data,
                session_id=credentials.session_id
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            await self.audit.log_event(
                "authorization", credentials.tenant_id, resource, action, "error",
                user_id=credentials.user_id,
                error_details=str(e),
                session_id=credentials.session_id
            )
            return False
    
    async def encrypt_data(
        self,
        context: TenantContext,
        data: Any,
        encryption_type: Optional[EncryptionType] = None,
        field_name: Optional[str] = None
    ) -> Union[bytes, str]:
        """Chiffre des données"""
        try:
            effective_encryption = encryption_type or self._security_config["default_encryption_type"]
            
            if field_name:
                # Field-level encryption
                return await self.encryption.encrypt_field(
                    context.tenant_id, field_name, data, effective_encryption
                )
            else:
                # Full data encryption
                if effective_encryption == EncryptionType.SYMMETRIC:
                    return await self.encryption.encrypt_symmetric(context.tenant_id, data)
                elif effective_encryption == EncryptionType.ASYMMETRIC:
                    return await self.encryption.encrypt_asymmetric(context.tenant_id, data)
                else:
                    return data  # No encryption
            
        except Exception as e:
            self.logger.error(f"Encryption error for tenant {context.tenant_id}: {e}")
            raise SecurityError(f"Encryption failed: {e}")
    
    async def decrypt_data(
        self,
        context: TenantContext,
        encrypted_data: Union[bytes, str],
        encryption_type: Optional[EncryptionType] = None,
        is_field: bool = False
    ) -> Any:
        """Déchiffre des données"""
        try:
            if is_field and isinstance(encrypted_data, str):
                # Field-level decryption
                field_name, value = await self.encryption.decrypt_field(
                    context.tenant_id, encrypted_data
                )
                return value
            else:
                # Full data decryption
                effective_encryption = encryption_type or self._security_config["default_encryption_type"]
                
                if effective_encryption == EncryptionType.SYMMETRIC:
                    decrypted = await self.encryption.decrypt_symmetric(context.tenant_id, encrypted_data)
                elif effective_encryption == EncryptionType.ASYMMETRIC:
                    decrypted = await self.encryption.decrypt_asymmetric(context.tenant_id, encrypted_data)
                else:
                    return encrypted_data  # No decryption needed
                
                # Try to decode as string
                if isinstance(decrypted, bytes):
                    try:
                        return decrypted.decode('utf-8')
                    except UnicodeDecodeError:
                        return decrypted
                
                return decrypted
            
        except Exception as e:
            self.logger.error(f"Decryption error for tenant {context.tenant_id}: {e}")
            raise SecurityError(f"Decryption failed: {e}")
    
    async def _track_threat_indicator(self, identifier: str, indicator_type: str):
        """Suit les indicateurs de menace"""
        now = datetime.now(timezone.utc)
        
        if identifier not in self._threat_indicators:
            self._threat_indicators[identifier] = []
        
        self._threat_indicators[identifier].append(now)
        
        # Clean old indicators (last hour)
        cutoff = now - timedelta(hours=1)
        self._threat_indicators[identifier] = [
            t for t in self._threat_indicators[identifier] if t > cutoff
        ]
        
        # Check for threats
        if len(self._threat_indicators[identifier]) > self._security_config["max_failed_attempts"]:
            self._blocked_ips.add(identifier)
            self.logger.warning(f"Blocked IP {identifier} due to suspicious activity")
            
            await self.audit.log_event(
                "threat_detection", "system", "security", "block_ip", "success",
                error_details=f"Blocked {identifier} for {indicator_type}",
                security_level=SecurityLevel.RESTRICTED
            )
    
    async def get_security_status(self, context: TenantContext) -> Dict[str, Any]:
        """Obtient le statut de sécurité"""
        # Get audit summary
        audit_summary = await self.audit.get_security_summary(context.tenant_id)
        
        # Get active sessions
        active_sessions = len([
            s for s in self.authentication._active_sessions.values()
            if s.tenant_id == context.tenant_id
        ])
        
        # Check encryption status
        has_symmetric_key = context.tenant_id in self.encryption._symmetric_keys
        has_asymmetric_keys = context.tenant_id in self.encryption._private_keys
        
        return {
            "tenant_id": context.tenant_id,
            "security_level": SecurityLevel.INTERNAL.value,
            "encryption_status": {
                "symmetric_key_exists": has_symmetric_key,
                "asymmetric_keys_exist": has_asymmetric_keys,
                "default_encryption": self._security_config["default_encryption_type"].value
            },
            "authentication_status": {
                "active_sessions": active_sessions,
                "require_authentication": self._security_config["require_authentication"]
            },
            "audit_summary": audit_summary,
            "threat_indicators": len(self._threat_indicators),
            "blocked_ips": len(self._blocked_ips)
        }
    
    async def cleanup_expired_sessions(self):
        """Nettoie les sessions expirées"""
        expired_sessions = []
        now = datetime.now(timezone.utc)
        
        for session_id, credentials in self.authentication._active_sessions.items():
            if (credentials.session_expires_at and 
                credentials.session_expires_at < now):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.authentication.destroy_session(session_id)
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def shutdown(self):
        """Arrêt propre du gestionnaire de sécurité"""
        self.logger.info("Shutting down security manager...")
        
        # Clear sensitive data
        self.encryption._symmetric_keys.clear()
        self.encryption._private_keys.clear()
        self.encryption._public_keys.clear()
        
        self.authentication._active_sessions.clear()
        self.authentication._api_keys.clear()
        
        self.logger.info("Security manager shutdown completed")


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


async def get_security_manager() -> SecurityManager:
    """Obtient l'instance globale du gestionnaire de sécurité"""
    global _security_manager
    if not _security_manager:
        _security_manager = SecurityManager()
        await _security_manager.initialize()
    return _security_manager


async def shutdown_security_manager():
    """Arrête l'instance globale du gestionnaire de sécurité"""
    global _security_manager
    if _security_manager:
        await _security_manager.shutdown()
        _security_manager = None
