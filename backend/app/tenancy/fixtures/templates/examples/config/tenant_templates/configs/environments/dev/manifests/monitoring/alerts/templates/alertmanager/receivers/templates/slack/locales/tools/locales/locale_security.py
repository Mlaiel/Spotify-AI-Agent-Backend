"""
Sécurité de Locales Avancée pour Spotify AI Agent
Système de sécurité et d'isolation multi-tenant pour les locales
"""

import hashlib
import hmac
import secrets
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Configuration de sécurité"""
    encryption_enabled: bool = True
    signature_verification: bool = True
    access_control_enabled: bool = True
    audit_logging: bool = True
    rate_limiting: bool = True
    tenant_isolation: bool = True
    encryption_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    session_timeout: int = 3600  # 1 heure
    max_failed_attempts: int = 5
    lockout_duration: int = 300  # 5 minutes
    
    def __post_init__(self):
        if not self.encryption_key:
            self.encryption_key = Fernet.generate_key().decode()
        if not self.jwt_secret:
            self.jwt_secret = secrets.token_urlsafe(32)


@dataclass
class SecurityPrincipal:
    """Principal de sécurité (utilisateur/service)"""
    id: str
    tenant_id: str
    roles: Set[str]
    permissions: Set[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessAttempt:
    """Tentative d'accès"""
    principal_id: str
    tenant_id: str
    resource: str
    action: str
    timestamp: datetime
    success: bool
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Permission:
    """Constantes de permissions"""
    READ_LOCALE = "locale:read"
    WRITE_LOCALE = "locale:write"
    DELETE_LOCALE = "locale:delete"
    MANAGE_LOCALES = "locale:manage"
    READ_TENANT_LOCALES = "tenant:locales:read"
    WRITE_TENANT_LOCALES = "tenant:locales:write"
    ADMIN_ALL = "admin:all"


class Role:
    """Constantes de rôles"""
    LOCALE_READER = "locale_reader"
    LOCALE_EDITOR = "locale_editor"
    LOCALE_ADMIN = "locale_admin"
    TENANT_USER = "tenant_user"
    TENANT_ADMIN = "tenant_admin"
    SYSTEM_ADMIN = "system_admin"


class SecurityProvider(ABC):
    """Interface pour les fournisseurs de sécurité"""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[SecurityPrincipal]:
        """Authentifie un principal"""
        pass
    
    @abstractmethod
    async def authorize(self, principal: SecurityPrincipal, resource: str, action: str) -> bool:
        """Autorise un accès"""
        pass
    
    @abstractmethod
    async def encrypt_data(self, data: Dict[str, Any], tenant_id: str) -> bytes:
        """Crypte des données"""
        pass
    
    @abstractmethod
    async def decrypt_data(self, encrypted_data: bytes, tenant_id: str) -> Dict[str, Any]:
        """Décrypte des données"""
        pass


class StandardSecurityProvider(SecurityProvider):
    """Fournisseur de sécurité standard"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._cipher = Fernet(config.encryption_key.encode())
        self._principals = {}
        self._failed_attempts = defaultdict(int)
        self._lockouts = {}
        self._audit_log = []
        self._lock = threading.RLock()
        
        # Rôles et permissions par défaut
        self._role_permissions = {
            Role.LOCALE_READER: {Permission.READ_LOCALE},
            Role.LOCALE_EDITOR: {Permission.READ_LOCALE, Permission.WRITE_LOCALE},
            Role.LOCALE_ADMIN: {
                Permission.READ_LOCALE, Permission.WRITE_LOCALE, 
                Permission.DELETE_LOCALE, Permission.MANAGE_LOCALES
            },
            Role.TENANT_USER: {Permission.READ_TENANT_LOCALES},
            Role.TENANT_ADMIN: {
                Permission.READ_TENANT_LOCALES, Permission.WRITE_TENANT_LOCALES
            },
            Role.SYSTEM_ADMIN: {Permission.ADMIN_ALL}
        }
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[SecurityPrincipal]:
        """Authentifie un principal"""
        try:
            principal_id = credentials.get('id')
            if not principal_id:
                return None
            
            # Vérifier les tentatives échouées
            if await self._is_locked_out(principal_id):
                logger.warning(f"Principal {principal_id} is locked out")
                return None
            
            # Vérifier le token JWT si fourni
            if 'token' in credentials:
                try:
                    payload = jwt.decode(
                        credentials['token'],
                        self.config.jwt_secret,
                        algorithms=['HS256']
                    )
                    
                    principal = SecurityPrincipal(
                        id=payload['sub'],
                        tenant_id=payload['tenant_id'],
                        roles=set(payload.get('roles', [])),
                        permissions=set(payload.get('permissions', [])),
                        created_at=datetime.fromtimestamp(payload['iat']),
                        last_login=datetime.now()
                    )
                    
                    # Enrichir avec les permissions des rôles
                    await self._enrich_permissions(principal)
                    
                    # Réinitialiser les tentatives échouées
                    await self._reset_failed_attempts(principal_id)
                    
                    return principal
                    
                except jwt.InvalidTokenError as e:
                    logger.warning(f"Invalid token for {principal_id}: {e}")
                    await self._record_failed_attempt(principal_id)
                    return None
            
            # Authentification par API key
            if 'api_key' in credentials:
                # Implémentation de l'authentification par API key
                # Pour cet exemple, on simule
                pass
            
            # Autres méthodes d'authentification...
            
            await self._record_failed_attempt(principal_id)
            return None
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def authorize(self, principal: SecurityPrincipal, resource: str, action: str) -> bool:
        """Autorise un accès"""
        try:
            # Vérifier si le principal a la permission directe
            required_permission = f"{resource}:{action}"
            
            if Permission.ADMIN_ALL in principal.permissions:
                return True
            
            if required_permission in principal.permissions:
                return True
            
            # Vérifier les permissions génériques
            if action == "read" and Permission.READ_LOCALE in principal.permissions:
                return True
            
            if action == "write" and Permission.WRITE_LOCALE in principal.permissions:
                return True
            
            if action == "delete" and Permission.DELETE_LOCALE in principal.permissions:
                return True
            
            # Vérifications spécifiques au tenant
            if resource.startswith(f"tenant:{principal.tenant_id}"):
                if Permission.READ_TENANT_LOCALES in principal.permissions and action == "read":
                    return True
                if Permission.WRITE_TENANT_LOCALES in principal.permissions and action in ["read", "write"]:
                    return True
            
            logger.warning(f"Access denied for {principal.id}: {resource}:{action}")
            return False
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            return False
    
    async def encrypt_data(self, data: Dict[str, Any], tenant_id: str) -> bytes:
        """Crypte des données pour un tenant"""
        try:
            if not self.config.encryption_enabled:
                import json
                return json.dumps(data).encode()
            
            # Ajouter des métadonnées de sécurité
            secure_data = {
                'data': data,
                'tenant_id': tenant_id,
                'timestamp': datetime.now().isoformat(),
                'checksum': await self._calculate_checksum(data)
            }
            
            import json
            json_data = json.dumps(secure_data).encode()
            
            # Crypter avec la clé du tenant (ou globale)
            tenant_key = await self._get_tenant_key(tenant_id)
            cipher = Fernet(tenant_key)
            
            return cipher.encrypt(json_data)
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: bytes, tenant_id: str) -> Dict[str, Any]:
        """Décrypte des données pour un tenant"""
        try:
            if not self.config.encryption_enabled:
                import json
                return json.loads(encrypted_data.decode())
            
            # Décrypter avec la clé du tenant
            tenant_key = await self._get_tenant_key(tenant_id)
            cipher = Fernet(tenant_key)
            
            json_data = cipher.decrypt(encrypted_data)
            
            import json
            secure_data = json.loads(json_data.decode())
            
            # Vérifier l'intégrité
            if secure_data.get('tenant_id') != tenant_id:
                raise ValueError("Tenant ID mismatch in encrypted data")
            
            expected_checksum = await self._calculate_checksum(secure_data['data'])
            if secure_data.get('checksum') != expected_checksum:
                raise ValueError("Data integrity check failed")
            
            return secure_data['data']
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
    
    async def generate_token(self, principal: SecurityPrincipal, expires_in: int = None) -> str:
        """Génère un token JWT"""
        try:
            if expires_in is None:
                expires_in = self.config.session_timeout
            
            now = datetime.now()
            payload = {
                'sub': principal.id,
                'tenant_id': principal.tenant_id,
                'roles': list(principal.roles),
                'permissions': list(principal.permissions),
                'iat': int(now.timestamp()),
                'exp': int((now + timedelta(seconds=expires_in)).timestamp())
            }
            
            token = jwt.encode(payload, self.config.jwt_secret, algorithm='HS256')
            
            # Enregistrer dans l'audit
            await self._audit_action(
                principal.id,
                principal.tenant_id,
                "token_generated",
                {"expires_in": expires_in}
            )
            
            return token
            
        except Exception as e:
            logger.error(f"Token generation error: {e}")
            raise
    
    async def validate_data_signature(self, data: Dict[str, Any], signature: str, tenant_id: str) -> bool:
        """Valide la signature des données"""
        try:
            if not self.config.signature_verification:
                return True
            
            # Calculer la signature attendue
            expected_signature = await self._sign_data(data, tenant_id)
            
            # Comparaison sécurisée
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Signature validation error: {e}")
            return False
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de sécurité"""
        with self._lock:
            return {
                'active_principals': len(self._principals),
                'failed_attempts_total': sum(self._failed_attempts.values()),
                'active_lockouts': len(self._lockouts),
                'audit_entries': len(self._audit_log),
                'security_config': {
                    'encryption_enabled': self.config.encryption_enabled,
                    'signature_verification': self.config.signature_verification,
                    'access_control_enabled': self.config.access_control_enabled,
                    'audit_logging': self.config.audit_logging,
                    'tenant_isolation': self.config.tenant_isolation
                }
            }
    
    async def _enrich_permissions(self, principal: SecurityPrincipal):
        """Enrichit les permissions basées sur les rôles"""
        for role in principal.roles:
            if role in self._role_permissions:
                principal.permissions.update(self._role_permissions[role])
    
    async def _is_locked_out(self, principal_id: str) -> bool:
        """Vérifie si un principal est verrouillé"""
        with self._lock:
            if principal_id in self._lockouts:
                if datetime.now() < self._lockouts[principal_id]:
                    return True
                else:
                    # Expiration du verrouillage
                    del self._lockouts[principal_id]
                    self._failed_attempts[principal_id] = 0
            
            return False
    
    async def _record_failed_attempt(self, principal_id: str):
        """Enregistre une tentative échouée"""
        with self._lock:
            self._failed_attempts[principal_id] += 1
            
            if self._failed_attempts[principal_id] >= self.config.max_failed_attempts:
                # Verrouiller le compte
                lockout_until = datetime.now() + timedelta(seconds=self.config.lockout_duration)
                self._lockouts[principal_id] = lockout_until
                
                logger.warning(f"Principal {principal_id} locked out until {lockout_until}")
    
    async def _reset_failed_attempts(self, principal_id: str):
        """Réinitialise les tentatives échouées"""
        with self._lock:
            self._failed_attempts[principal_id] = 0
            if principal_id in self._lockouts:
                del self._lockouts[principal_id]
    
    async def _get_tenant_key(self, tenant_id: str) -> bytes:
        """Obtient la clé de cryptage pour un tenant"""
        # Pour cet exemple, on utilise une clé dérivée du tenant_id
        # En production, les clés devraient être stockées sécurisément
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=tenant_id.encode(),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.config.encryption_key.encode()))
        return key
    
    async def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calcule un checksum des données"""
        import json
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    async def _sign_data(self, data: Dict[str, Any], tenant_id: str) -> str:
        """Signe des données"""
        import json
        data_str = json.dumps(data, sort_keys=True)
        tenant_secret = f"{self.config.jwt_secret}:{tenant_id}"
        
        signature = hmac.new(
            tenant_secret.encode(),
            data_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    async def _audit_action(
        self,
        principal_id: str,
        tenant_id: str,
        action: str,
        metadata: Dict[str, Any] = None
    ):
        """Enregistre une action dans l'audit"""
        if not self.config.audit_logging:
            return
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'principal_id': principal_id,
            'tenant_id': tenant_id,
            'action': action,
            'metadata': metadata or {}
        }
        
        with self._lock:
            self._audit_log.append(audit_entry)
            
            # Limiter la taille du log
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-5000:]


class TenantIsolation:
    """Gestionnaire d'isolation des tenants"""
    
    def __init__(self, security_provider: SecurityProvider):
        self.security_provider = security_provider
        self._tenant_boundaries = {}
        self._isolation_rules = {}
        self._lock = threading.RLock()
    
    async def enforce_tenant_isolation(
        self,
        principal: SecurityPrincipal,
        resource_tenant_id: str,
        action: str
    ) -> bool:
        """Applique l'isolation entre tenants"""
        try:
            # Vérifier si le principal peut accéder aux ressources du tenant
            if principal.tenant_id == resource_tenant_id:
                return True
            
            # Vérifier les permissions cross-tenant
            if Permission.ADMIN_ALL in principal.permissions:
                return True
            
            # Vérifier les règles d'isolation spécifiques
            isolation_rule = await self._get_isolation_rule(
                principal.tenant_id,
                resource_tenant_id
            )
            
            if isolation_rule and action in isolation_rule.get('allowed_actions', []):
                return True
            
            logger.warning(
                f"Tenant isolation violation: {principal.tenant_id} "
                f"trying to access {resource_tenant_id} for {action}"
            )
            
            return False
            
        except Exception as e:
            logger.error(f"Tenant isolation error: {e}")
            return False
    
    async def create_tenant_boundary(
        self,
        tenant_id: str,
        boundary_config: Dict[str, Any]
    ):
        """Crée une frontière de tenant"""
        with self._lock:
            self._tenant_boundaries[tenant_id] = {
                'created_at': datetime.now(),
                'config': boundary_config,
                'isolation_level': boundary_config.get('isolation_level', 'strict')
            }
    
    async def set_isolation_rule(
        self,
        source_tenant: str,
        target_tenant: str,
        rule: Dict[str, Any]
    ):
        """Définit une règle d'isolation entre tenants"""
        with self._lock:
            rule_key = f"{source_tenant}:{target_tenant}"
            self._isolation_rules[rule_key] = {
                'created_at': datetime.now(),
                'allowed_actions': rule.get('allowed_actions', []),
                'conditions': rule.get('conditions', {}),
                'expires_at': rule.get('expires_at')
            }
    
    async def get_tenant_isolation_status(self, tenant_id: str) -> Dict[str, Any]:
        """Retourne le statut d'isolation d'un tenant"""
        with self._lock:
            boundary = self._tenant_boundaries.get(tenant_id, {})
            
            # Compter les règles impliquant ce tenant
            outbound_rules = len([
                k for k in self._isolation_rules.keys()
                if k.startswith(f"{tenant_id}:")
            ])
            
            inbound_rules = len([
                k for k in self._isolation_rules.keys()
                if k.endswith(f":{tenant_id}")
            ])
            
            return {
                'tenant_id': tenant_id,
                'has_boundary': bool(boundary),
                'isolation_level': boundary.get('config', {}).get('isolation_level', 'default'),
                'outbound_rules': outbound_rules,
                'inbound_rules': inbound_rules,
                'created_at': boundary.get('created_at')
            }
    
    async def _get_isolation_rule(
        self,
        source_tenant: str,
        target_tenant: str
    ) -> Optional[Dict[str, Any]]:
        """Récupère une règle d'isolation"""
        with self._lock:
            rule_key = f"{source_tenant}:{target_tenant}"
            rule = self._isolation_rules.get(rule_key)
            
            if rule:
                # Vérifier l'expiration
                expires_at = rule.get('expires_at')
                if expires_at and datetime.now() > expires_at:
                    del self._isolation_rules[rule_key]
                    return None
                
                return rule
            
            return None


class LocaleSecurity:
    """Gestionnaire de sécurité principal pour les locales"""
    
    def __init__(
        self,
        security_provider: SecurityProvider,
        tenant_isolation: TenantIsolation
    ):
        self.security_provider = security_provider
        self.tenant_isolation = tenant_isolation
        self._access_attempts = []
        self._security_events = []
        self._lock = threading.RLock()
    
    async def secure_locale_access(
        self,
        principal: SecurityPrincipal,
        locale_code: str,
        tenant_id: str,
        action: str
    ) -> bool:
        """Sécurise l'accès à une locale"""
        try:
            # Vérifier l'isolation tenant
            if not await self.tenant_isolation.enforce_tenant_isolation(
                principal, tenant_id, action
            ):
                await self._record_security_event(
                    "tenant_isolation_violation",
                    principal,
                    {'locale_code': locale_code, 'tenant_id': tenant_id, 'action': action}
                )
                return False
            
            # Vérifier les autorisations
            resource = f"locale:{locale_code}"
            if not await self.security_provider.authorize(principal, resource, action):
                await self._record_security_event(
                    "authorization_denied",
                    principal,
                    {'locale_code': locale_code, 'tenant_id': tenant_id, 'action': action}
                )
                return False
            
            # Enregistrer l'accès réussi
            await self._record_access_attempt(principal, locale_code, tenant_id, action, True)
            
            return True
            
        except Exception as e:
            logger.error(f"Secure locale access error: {e}")
            await self._record_security_event(
                "security_error",
                principal,
                {'error': str(e), 'locale_code': locale_code, 'tenant_id': tenant_id}
            )
            return False
    
    async def encrypt_locale_data(
        self,
        data: Dict[str, Any],
        tenant_id: str
    ) -> bytes:
        """Crypte les données de locale"""
        return await self.security_provider.encrypt_data(data, tenant_id)
    
    async def decrypt_locale_data(
        self,
        encrypted_data: bytes,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Décrypte les données de locale"""
        return await self.security_provider.decrypt_data(encrypted_data, tenant_id)
    
    async def get_security_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Génère un rapport de sécurité"""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(days=7)
            if not end_time:
                end_time = datetime.now()
            
            with self._lock:
                # Filtrer les événements par période
                filtered_attempts = [
                    attempt for attempt in self._access_attempts
                    if start_time <= attempt.timestamp <= end_time
                ]
                
                filtered_events = [
                    event for event in self._security_events
                    if start_time <= event['timestamp'] <= end_time
                ]
            
            # Calculer les métriques
            total_attempts = len(filtered_attempts)
            successful_attempts = len([a for a in filtered_attempts if a.success])
            failed_attempts = total_attempts - successful_attempts
            
            security_violations = len([
                e for e in filtered_events
                if e['type'] in ['tenant_isolation_violation', 'authorization_denied']
            ])
            
            # Analyse des tendances
            tenant_activity = defaultdict(int)
            for attempt in filtered_attempts:
                tenant_activity[attempt.tenant_id] += 1
            
            return {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'access_metrics': {
                    'total_attempts': total_attempts,
                    'successful_attempts': successful_attempts,
                    'failed_attempts': failed_attempts,
                    'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0
                },
                'security_metrics': {
                    'security_violations': security_violations,
                    'security_events': len(filtered_events),
                    'violation_rate': security_violations / total_attempts if total_attempts > 0 else 0
                },
                'tenant_activity': dict(tenant_activity),
                'provider_metrics': await self.security_provider.get_security_metrics(),
                'recommendations': await self._generate_security_recommendations(
                    filtered_attempts, filtered_events
                )
            }
            
        except Exception as e:
            logger.error(f"Security report error: {e}")
            return {}
    
    async def _record_access_attempt(
        self,
        principal: SecurityPrincipal,
        locale_code: str,
        tenant_id: str,
        action: str,
        success: bool
    ):
        """Enregistre une tentative d'accès"""
        attempt = AccessAttempt(
            principal_id=principal.id,
            tenant_id=tenant_id,
            resource=f"locale:{locale_code}",
            action=action,
            timestamp=datetime.now(),
            success=success
        )
        
        with self._lock:
            self._access_attempts.append(attempt)
            
            # Limiter la taille
            if len(self._access_attempts) > 10000:
                self._access_attempts = self._access_attempts[-5000:]
    
    async def _record_security_event(
        self,
        event_type: str,
        principal: SecurityPrincipal,
        metadata: Dict[str, Any]
    ):
        """Enregistre un événement de sécurité"""
        event = {
            'type': event_type,
            'principal_id': principal.id,
            'tenant_id': principal.tenant_id,
            'timestamp': datetime.now(),
            'metadata': metadata
        }
        
        with self._lock:
            self._security_events.append(event)
            
            # Limiter la taille
            if len(self._security_events) > 5000:
                self._security_events = self._security_events[-2500:]
    
    async def _generate_security_recommendations(
        self,
        attempts: List[AccessAttempt],
        events: List[Dict[str, Any]]
    ) -> List[str]:
        """Génère des recommandations de sécurité"""
        recommendations = []
        
        if not attempts:
            return recommendations
        
        # Analyser le taux d'échec
        failed_rate = len([a for a in attempts if not a.success]) / len(attempts)
        if failed_rate > 0.1:  # Plus de 10% d'échecs
            recommendations.append(
                f"Taux d'échec élevé ({failed_rate:.1%}): vérifier les permissions et authentification"
            )
        
        # Analyser les violations de sécurité
        violations = len([e for e in events if 'violation' in e['type']])
        if violations > 0:
            recommendations.append(
                f"{violations} violations de sécurité détectées: renforcer les contrôles d'accès"
            )
        
        # Analyser l'activité par tenant
        tenant_attempts = defaultdict(int)
        for attempt in attempts:
            tenant_attempts[attempt.tenant_id] += 1
        
        if len(tenant_attempts) > 50:  # Beaucoup de tenants actifs
            recommendations.append(
                "Activité multi-tenant élevée: considérer l'optimisation du cache et de l'isolation"
            )
        
        return recommendations
