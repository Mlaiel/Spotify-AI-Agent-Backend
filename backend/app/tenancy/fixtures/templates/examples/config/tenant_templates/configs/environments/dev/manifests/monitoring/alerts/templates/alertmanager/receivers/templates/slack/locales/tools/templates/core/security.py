"""
Advanced Security Manager
=========================

Gestionnaire de sécurité avancé avec authentification multi-facteurs,
autorisation granulaire, audit, et protection contre les menaces.

Auteur: Fahed Mlaiel
"""

import asyncio
import hashlib
import secrets
import logging
import json
import time
import hmac
import base64
from typing import Dict, Any, Optional, List, Tuple, Union, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aiofiles
import aioredis
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import ipaddress
import re
import pyotp
import qrcode
from PIL import Image
import io
import weakref

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Niveaux de sécurité"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthMethod(Enum):
    """Méthodes d'authentification"""
    PASSWORD = "password"
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BIOMETRIC = "biometric"
    CERTIFICATE = "certificate"
    OAUTH = "oauth"


class Permission(Enum):
    """Permissions système"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"
    AUDIT = "audit"


@dataclass
class SecurityEvent:
    """Événement de sécurité"""
    event_id: str
    event_type: str
    user_id: Optional[str]
    tenant_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    severity: SecurityLevel
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0


@dataclass
class AuthSession:
    """Session d'authentification"""
    session_id: str
    user_id: str
    tenant_id: Optional[str]
    auth_methods: List[AuthMethod]
    permissions: Set[Permission]
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Politique de sécurité"""
    policy_id: str
    name: str
    tenant_id: Optional[str]
    security_level: SecurityLevel
    password_policy: Dict[str, Any]
    session_policy: Dict[str, Any]
    mfa_requirements: List[AuthMethod]
    ip_restrictions: List[str]
    allowed_countries: List[str]
    rate_limits: Dict[str, int]
    audit_requirements: List[str]
    enabled: bool = True


@dataclass
class ThreatIndicator:
    """Indicateur de menace"""
    indicator_type: str
    value: str
    severity: SecurityLevel
    source: str
    first_seen: datetime
    last_seen: datetime
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityManager:
    """
    Gestionnaire de sécurité avancé
    
    Fonctionnalités:
    - Authentification multi-facteurs (TOTP, SMS, Email)
    - Autorisation granulaire basée sur les rôles
    - Audit et monitoring de sécurité
    - Protection contre les attaques (rate limiting, IP blocking)
    - Chiffrement des données sensibles
    - Détection d'anomalies et menaces
    - Gestion des sessions sécurisées
    - Politiques de sécurité par tenant
    """
    
    def __init__(self, config: Dict[str, Any], redis_client: Optional[aioredis.Redis] = None):
        """
        Initialise le gestionnaire de sécurité
        
        Args:
            config: Configuration de sécurité
            redis_client: Client Redis pour le cache
        """
        self.config = config
        self.redis_client = redis_client
        self.is_initialized = False
        
        # Clés de chiffrement
        self.encryption_key = None
        self.jwt_secret = config.get("jwt_secret", secrets.token_urlsafe(64))
        self.jwt_algorithm = config.get("jwt_algorithm", "HS256")
        
        # Sessions actives
        self.active_sessions: Dict[str, AuthSession] = {}
        
        # Politiques de sécurité
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.default_policy: Optional[SecurityPolicy] = None
        
        # Events et audit
        self.security_events: List[SecurityEvent] = []
        self.audit_log_path = config.get("audit_log_path", "security_audit.log")
        
        # Rate limiting
        self.rate_limit_cache: Dict[str, Dict[str, int]] = {}
        
        # IP blocking
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, float] = {}
        
        # Threat intelligence
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        
        # Callbacks pour les événements de sécurité
        self.security_event_callbacks: List[Callable] = []
        
        # Métriques
        self.metrics = {
            "auth_attempts": 0,
            "auth_successes": 0,
            "auth_failures": 0,
            "security_events": 0,
            "blocked_requests": 0,
            "active_sessions": 0,
            "mfa_challenges": 0,
            "threat_detections": 0
        }
        
        logger.info("SecurityManager initialisé")
    
    async def initialize(self) -> None:
        """Initialise le gestionnaire de sécurité"""
        if self.is_initialized:
            return
        
        logger.info("Initialisation du SecurityManager...")
        
        try:
            # Configuration du chiffrement
            await self._setup_encryption()
            
            # Chargement des politiques de sécurité
            await self._load_security_policies()
            
            # Configuration de l'audit
            await self._setup_audit_logging()
            
            # Chargement des indicateurs de menaces
            await self._load_threat_indicators()
            
            # Configuration du rate limiting
            await self._setup_rate_limiting()
            
            # Nettoyage périodique
            asyncio.create_task(self._cleanup_task())
            
            self.is_initialized = True
            logger.info("SecurityManager initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du SecurityManager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Arrêt propre du gestionnaire de sécurité"""
        if not self.is_initialized:
            return
        
        logger.info("Arrêt du SecurityManager...")
        
        try:
            # Révocation de toutes les sessions
            await self._revoke_all_sessions()
            
            # Sauvegarde des événements de sécurité
            await self._save_security_events()
            
            # Nettoyage
            self.active_sessions.clear()
            self.security_events.clear()
            self.rate_limit_cache.clear()
            
            self.is_initialized = False
            logger.info("SecurityManager arrêté avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {e}")
    
    async def _setup_encryption(self) -> None:
        """Configure le système de chiffrement"""
        # Génération ou chargement de la clé de chiffrement
        encryption_key_str = self.config.get("encryption_key")
        
        if encryption_key_str:
            self.encryption_key = encryption_key_str.encode()
        else:
            # Génération d'une nouvelle clé
            self.encryption_key = Fernet.generate_key()
            logger.warning("Nouvelle clé de chiffrement générée. Sauvegarder pour la production!")
        
        # Test du chiffrement
        test_fernet = Fernet(self.encryption_key)
        test_data = b"test encryption"
        encrypted = test_fernet.encrypt(test_data)
        decrypted = test_fernet.decrypt(encrypted)
        
        if decrypted != test_data:
            raise ValueError("Erreur de configuration du chiffrement")
        
        logger.info("Système de chiffrement configuré")
    
    async def _load_security_policies(self) -> None:
        """Charge les politiques de sécurité"""
        # Politique par défaut
        default_policy = SecurityPolicy(
            policy_id="default",
            name="Politique par défaut",
            tenant_id=None,
            security_level=SecurityLevel.MEDIUM,
            password_policy={
                "min_length": 8,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_symbols": True,
                "max_age_days": 90,
                "history_count": 5
            },
            session_policy={
                "max_duration_hours": 8,
                "idle_timeout_minutes": 30,
                "max_concurrent_sessions": 3
            },
            mfa_requirements=[AuthMethod.TOTP],
            ip_restrictions=[],
            allowed_countries=[],
            rate_limits={
                "auth_attempts": 5,
                "api_requests": 1000,
                "window_minutes": 15
            },
            audit_requirements=["auth", "admin_actions", "data_access"]
        )
        
        self.default_policy = default_policy
        self.security_policies["default"] = default_policy
        
        # Chargement des politiques personnalisées depuis les fichiers
        policies_dir = self.config.get("policies_dir", "security_policies")
        if Path(policies_dir).exists():
            await self._load_policies_from_directory(Path(policies_dir))
        
        logger.info(f"Chargées {len(self.security_policies)} politiques de sécurité")
    
    async def _load_policies_from_directory(self, policies_dir: Path) -> None:
        """Charge les politiques depuis un répertoire"""
        for policy_file in policies_dir.glob("*.json"):
            try:
                async with aiofiles.open(policy_file, 'r') as f:
                    content = await f.read()
                    policy_data = json.loads(content)
                
                policy = SecurityPolicy(**policy_data)
                self.security_policies[policy.policy_id] = policy
                
                logger.debug(f"Politique chargée: {policy.name}")
                
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la politique {policy_file}: {e}")
    
    async def _setup_audit_logging(self) -> None:
        """Configure l'audit logging"""
        # Configuration du logger d'audit
        audit_logger = logging.getLogger("security_audit")
        audit_logger.setLevel(logging.INFO)
        
        # Handler pour fichier d'audit
        audit_handler = logging.FileHandler(self.audit_log_path)
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger.addHandler(audit_handler)
        
        logger.info("Audit logging configuré")
    
    async def _load_threat_indicators(self) -> None:
        """Charge les indicateurs de menaces"""
        # Chargement depuis fichier local
        threat_file = self.config.get("threat_indicators_file", "threat_indicators.json")
        if Path(threat_file).exists():
            try:
                async with aiofiles.open(threat_file, 'r') as f:
                    content = await f.read()
                    threats_data = json.loads(content)
                
                for threat_data in threats_data:
                    threat = ThreatIndicator(**threat_data)
                    self.threat_indicators[threat.value] = threat
                
                logger.info(f"Chargés {len(self.threat_indicators)} indicateurs de menaces")
                
            except Exception as e:
                logger.error(f"Erreur lors du chargement des indicateurs de menaces: {e}")
        
        # Chargement depuis sources externes (optionnel)
        threat_feeds = self.config.get("threat_feeds", [])
        for feed_url in threat_feeds:
            await self._load_threat_feed(feed_url)
    
    async def _load_threat_feed(self, feed_url: str) -> None:
        """Charge un feed de menaces externe"""
        try:
            # Implémentation du chargement de feeds externes
            # (nécessiterait aiohttp pour les requêtes HTTP)
            logger.debug(f"Chargement du feed de menaces: {feed_url}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du feed {feed_url}: {e}")
    
    async def _setup_rate_limiting(self) -> None:
        """Configure le rate limiting"""
        # Configuration des limites par défaut
        self.default_rate_limits = {
            "auth_attempts": 5,
            "api_requests": 1000,
            "password_reset": 3,
            "mfa_attempts": 3
        }
        
        logger.info("Rate limiting configuré")
    
    async def _cleanup_task(self) -> None:
        """Tâche de nettoyage périodique"""
        while self.is_initialized:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                # Nettoyage des sessions expirées
                await self._cleanup_expired_sessions()
                
                # Nettoyage du cache de rate limiting
                await self._cleanup_rate_limit_cache()
                
                # Nettoyage des IPs suspectes
                await self._cleanup_suspicious_ips()
                
                # Sauvegarde périodique des événements
                await self._save_security_events()
                
            except Exception as e:
                logger.error(f"Erreur lors du nettoyage: {e}")
    
    # API d'authentification
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str,
        tenant_id: Optional[str] = None,
        additional_factors: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, Optional[AuthSession], Optional[str]]:
        """
        Authentifie un utilisateur
        
        Args:
            username: Nom d'utilisateur
            password: Mot de passe
            ip_address: Adresse IP
            user_agent: User agent
            tenant_id: ID du tenant
            additional_factors: Facteurs d'authentification supplémentaires
            
        Returns:
            (success, session, error_message)
        """
        self.metrics["auth_attempts"] += 1
        
        try:
            # Vérification du rate limiting
            if not await self._check_rate_limit("auth_attempts", ip_address):
                await self._log_security_event(
                    "AUTH_RATE_LIMIT_EXCEEDED",
                    None,
                    tenant_id,
                    ip_address,
                    user_agent,
                    SecurityLevel.MEDIUM,
                    f"Trop de tentatives d'authentification depuis {ip_address}"
                )
                return False, None, "Trop de tentatives d'authentification"
            
            # Vérification des IPs bloquées
            if await self._is_ip_blocked(ip_address):
                await self._log_security_event(
                    "AUTH_BLOCKED_IP",
                    None,
                    tenant_id,
                    ip_address,
                    user_agent,
                    SecurityLevel.HIGH,
                    f"Tentative d'authentification depuis IP bloquée: {ip_address}"
                )
                return False, None, "Adresse IP bloquée"
            
            # Récupération de la politique de sécurité
            policy = await self._get_security_policy(tenant_id)
            
            # Vérification des restrictions géographiques
            if not await self._check_geographic_restrictions(ip_address, policy):
                await self._log_security_event(
                    "AUTH_GEO_RESTRICTION",
                    username,
                    tenant_id,
                    ip_address,
                    user_agent,
                    SecurityLevel.MEDIUM,
                    f"Tentative d'authentification depuis pays non autorisé"
                )
                return False, None, "Accès géographique non autorisé"
            
            # Vérification des informations d'identification
            user_id = await self._verify_credentials(username, password, tenant_id)
            if not user_id:
                await self._log_security_event(
                    "AUTH_INVALID_CREDENTIALS",
                    username,
                    tenant_id,
                    ip_address,
                    user_agent,
                    SecurityLevel.MEDIUM,
                    "Informations d'identification invalides"
                )
                self.metrics["auth_failures"] += 1
                return False, None, "Informations d'identification invalides"
            
            # Vérification MFA si requis
            auth_methods = [AuthMethod.PASSWORD]
            if policy.mfa_requirements and additional_factors:
                mfa_success = await self._verify_mfa(
                    user_id, policy.mfa_requirements, additional_factors
                )
                if not mfa_success:
                    await self._log_security_event(
                        "AUTH_MFA_FAILED",
                        user_id,
                        tenant_id,
                        ip_address,
                        user_agent,
                        SecurityLevel.HIGH,
                        "Échec de l'authentification multi-facteurs"
                    )
                    self.metrics["auth_failures"] += 1
                    return False, None, "Authentification multi-facteurs requise"
                
                auth_methods.extend(policy.mfa_requirements)
            elif policy.mfa_requirements:
                # MFA requis mais non fourni
                return False, None, "Authentification multi-facteurs requise"
            
            # Création de la session
            session = await self._create_session(
                user_id, tenant_id, auth_methods, ip_address, user_agent, policy
            )
            
            await self._log_security_event(
                "AUTH_SUCCESS",
                user_id,
                tenant_id,
                ip_address,
                user_agent,
                SecurityLevel.LOW,
                "Authentification réussie"
            )
            
            self.metrics["auth_successes"] += 1
            self.metrics["active_sessions"] = len(self.active_sessions)
            
            return True, session, None
            
        except Exception as e:
            logger.error(f"Erreur lors de l'authentification: {e}")
            await self._log_security_event(
                "AUTH_ERROR",
                username,
                tenant_id,
                ip_address,
                user_agent,
                SecurityLevel.HIGH,
                f"Erreur d'authentification: {str(e)}"
            )
            return False, None, "Erreur interne"
    
    async def _verify_credentials(
        self,
        username: str,
        password: str,
        tenant_id: Optional[str]
    ) -> Optional[str]:
        """Vérifie les informations d'identification"""
        # Dans une implémentation réelle, on ferait appel à la base de données
        # Pour cet exemple, on simule une vérification
        
        # Hash du mot de passe avec bcrypt
        password_bytes = password.encode('utf-8')
        
        # Simulation d'un utilisateur de test
        if username == "admin" and password == "admin123":
            return "user_001"
        
        return None
    
    async def _verify_mfa(
        self,
        user_id: str,
        required_methods: List[AuthMethod],
        provided_factors: Dict[str, str]
    ) -> bool:
        """Vérifie l'authentification multi-facteurs"""
        for method in required_methods:
            if method == AuthMethod.TOTP:
                if "totp" not in provided_factors:
                    return False
                
                # Vérification du code TOTP
                if not await self._verify_totp(user_id, provided_factors["totp"]):
                    return False
            
            elif method == AuthMethod.SMS:
                if "sms_code" not in provided_factors:
                    return False
                
                # Vérification du code SMS
                if not await self._verify_sms_code(user_id, provided_factors["sms_code"]):
                    return False
            
            elif method == AuthMethod.EMAIL:
                if "email_code" not in provided_factors:
                    return False
                
                # Vérification du code email
                if not await self._verify_email_code(user_id, provided_factors["email_code"]):
                    return False
        
        return True
    
    async def _verify_totp(self, user_id: str, totp_code: str) -> bool:
        """Vérifie un code TOTP"""
        # Récupération du secret TOTP de l'utilisateur
        secret = await self._get_user_totp_secret(user_id)
        if not secret:
            return False
        
        # Vérification du code
        totp = pyotp.TOTP(secret)
        return totp.verify(totp_code, valid_window=1)
    
    async def _verify_sms_code(self, user_id: str, sms_code: str) -> bool:
        """Vérifie un code SMS"""
        # Dans une implémentation réelle, on vérifierait contre un code stocké
        # avec une expiration
        return True  # Simulation
    
    async def _verify_email_code(self, user_id: str, email_code: str) -> bool:
        """Vérifie un code email"""
        # Dans une implémentation réelle, on vérifierait contre un code stocké
        # avec une expiration
        return True  # Simulation
    
    async def _get_user_totp_secret(self, user_id: str) -> Optional[str]:
        """Récupère le secret TOTP d'un utilisateur"""
        # Dans une implémentation réelle, on récupérerait depuis la base de données
        # Simulation avec un secret fixe
        return "JBSWY3DPEHPK3PXP"  # Secret d'exemple
    
    async def _create_session(
        self,
        user_id: str,
        tenant_id: Optional[str],
        auth_methods: List[AuthMethod],
        ip_address: str,
        user_agent: str,
        policy: SecurityPolicy
    ) -> AuthSession:
        """Crée une session d'authentification"""
        session_id = secrets.token_urlsafe(32)
        now = datetime.utcnow()
        
        # Permissions par défaut
        permissions = {Permission.READ}
        
        # Ajout des permissions selon le rôle (simulation)
        if "admin" in user_id.lower():
            permissions.update({Permission.WRITE, Permission.DELETE, Permission.ADMIN})
        
        session = AuthSession(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            auth_methods=auth_methods,
            permissions=permissions,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(hours=policy.session_policy["max_duration_hours"])
        )
        
        # Stockage de la session
        self.active_sessions[session_id] = session
        
        # Stockage dans Redis si disponible
        if self.redis_client:
            session_data = {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "ip_address": ip_address,
                "created_at": now.isoformat(),
                "expires_at": session.expires_at.isoformat()
            }
            await self.redis_client.setex(
                f"session:{session_id}",
                int(policy.session_policy["max_duration_hours"] * 3600),
                json.dumps(session_data)
            )
        
        return session
    
    async def validate_session(
        self,
        session_id: str,
        ip_address: str,
        user_agent: str
    ) -> Tuple[bool, Optional[AuthSession]]:
        """
        Valide une session
        
        Args:
            session_id: ID de session
            ip_address: Adresse IP
            user_agent: User agent
            
        Returns:
            (is_valid, session)
        """
        if session_id not in self.active_sessions:
            return False, None
        
        session = self.active_sessions[session_id]
        
        # Vérification de l'expiration
        if datetime.utcnow() > session.expires_at:
            await self._revoke_session(session_id, "EXPIRED")
            return False, None
        
        # Vérification de l'IP (optionnel selon la politique)
        policy = await self._get_security_policy(session.tenant_id)
        if policy.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            if session.ip_address != ip_address:
                await self._log_security_event(
                    "SESSION_IP_MISMATCH",
                    session.user_id,
                    session.tenant_id,
                    ip_address,
                    user_agent,
                    SecurityLevel.HIGH,
                    f"Changement d'IP pour la session {session_id}"
                )
                await self._revoke_session(session_id, "IP_MISMATCH")
                return False, None
        
        # Vérification du timeout d'inactivité
        idle_timeout = timedelta(minutes=policy.session_policy["idle_timeout_minutes"])
        if datetime.utcnow() - session.last_activity > idle_timeout:
            await self._revoke_session(session_id, "IDLE_TIMEOUT")
            return False, None
        
        # Mise à jour de la dernière activité
        session.last_activity = datetime.utcnow()
        
        return True, session
    
    async def _revoke_session(self, session_id: str, reason: str) -> None:
        """Révoque une session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            await self._log_security_event(
                "SESSION_REVOKED",
                session.user_id,
                session.tenant_id,
                session.ip_address,
                session.user_agent,
                SecurityLevel.LOW,
                f"Session révoquée: {reason}"
            )
            
            del self.active_sessions[session_id]
            
            # Suppression dans Redis
            if self.redis_client:
                await self.redis_client.delete(f"session:{session_id}")
        
        self.metrics["active_sessions"] = len(self.active_sessions)
    
    async def _revoke_all_sessions(self) -> None:
        """Révoque toutes les sessions"""
        for session_id in list(self.active_sessions.keys()):
            await self._revoke_session(session_id, "SHUTDOWN")
    
    # API d'autorisation
    
    async def check_permission(
        self,
        session_id: str,
        permission: Permission,
        resource: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Vérifie une permission
        
        Args:
            session_id: ID de session
            permission: Permission à vérifier
            resource: Ressource spécifique (optionnel)
            tenant_id: ID du tenant
            
        Returns:
            True si autorisé
        """
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Vérification du tenant
        if tenant_id and session.tenant_id != tenant_id:
            return False
        
        # Vérification de la permission
        if permission in session.permissions:
            return True
        
        # Vérification des permissions administrateur
        if Permission.ADMIN in session.permissions:
            return True
        
        return False
    
    # API de gestion des menaces
    
    async def _check_rate_limit(self, action: str, identifier: str) -> bool:
        """Vérifie le rate limiting"""
        current_time = int(time.time())
        window_key = f"{action}:{identifier}:{current_time // 900}"  # 15 min window
        
        if window_key not in self.rate_limit_cache:
            self.rate_limit_cache[window_key] = 0
        
        self.rate_limit_cache[window_key] += 1
        
        # Récupération de la limite
        limit = self.default_rate_limits.get(action, 100)
        
        return self.rate_limit_cache[window_key] <= limit
    
    async def _is_ip_blocked(self, ip_address: str) -> bool:
        """Vérifie si une IP est bloquée"""
        return ip_address in self.blocked_ips
    
    async def _check_geographic_restrictions(
        self,
        ip_address: str,
        policy: SecurityPolicy
    ) -> bool:
        """Vérifie les restrictions géographiques"""
        if not policy.allowed_countries:
            return True
        
        # Dans une implémentation réelle, on utiliserait une base de données GeoIP
        # Pour cet exemple, on autorise toutes les IPs
        return True
    
    async def _get_security_policy(self, tenant_id: Optional[str]) -> SecurityPolicy:
        """Récupère la politique de sécurité"""
        if tenant_id and tenant_id in self.security_policies:
            return self.security_policies[tenant_id]
        
        return self.default_policy
    
    # API de chiffrement
    
    def encrypt_data(self, data: str) -> str:
        """
        Chiffre des données
        
        Args:
            data: Données à chiffrer
            
        Returns:
            Données chiffrées (base64)
        """
        if not self.encryption_key:
            raise ValueError("Clé de chiffrement non configurée")
        
        fernet = Fernet(self.encryption_key)
        encrypted_data = fernet.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Déchiffre des données
        
        Args:
            encrypted_data: Données chiffrées (base64)
            
        Returns:
            Données déchiffrées
        """
        if not self.encryption_key:
            raise ValueError("Clé de chiffrement non configurée")
        
        fernet = Fernet(self.encryption_key)
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_data = fernet.decrypt(encrypted_bytes)
        return decrypted_data.decode()
    
    # API de tokens JWT
    
    def generate_jwt_token(
        self,
        payload: Dict[str, Any],
        expires_in: Optional[int] = None
    ) -> str:
        """
        Génère un token JWT
        
        Args:
            payload: Données du token
            expires_in: Expiration en secondes
            
        Returns:
            Token JWT
        """
        if expires_in:
            payload["exp"] = datetime.utcnow() + timedelta(seconds=expires_in)
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Vérifie un token JWT
        
        Args:
            token: Token JWT
            
        Returns:
            Payload du token si valide
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.InvalidTokenError:
            return None
    
    # API MFA
    
    async def generate_totp_secret(self, user_id: str) -> Tuple[str, str]:
        """
        Génère un secret TOTP
        
        Args:
            user_id: ID utilisateur
            
        Returns:
            (secret, qr_code_url)
        """
        secret = pyotp.random_base32()
        
        # Génération de l'URL pour le QR code
        totp = pyotp.TOTP(secret)
        qr_url = totp.provisioning_uri(
            name=user_id,
            issuer_name=self.config.get("app_name", "Spotify AI Agent")
        )
        
        return secret, qr_url
    
    async def generate_qr_code(self, qr_url: str) -> bytes:
        """
        Génère un QR code
        
        Args:
            qr_url: URL du QR code
            
        Returns:
            Image QR code en bytes
        """
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Conversion en bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    # API d'audit et logging
    
    async def _log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        tenant_id: Optional[str],
        ip_address: str,
        user_agent: str,
        severity: SecurityLevel,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log un événement de sécurité"""
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=event_type,
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            severity=severity,
            description=description,
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        self.metrics["security_events"] += 1
        
        # Log dans le fichier d'audit
        audit_logger = logging.getLogger("security_audit")
        audit_logger.info(json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "user_id": event.user_id,
            "tenant_id": event.tenant_id,
            "ip_address": event.ip_address,
            "severity": event.severity.value,
            "description": event.description,
            "timestamp": event.timestamp.isoformat(),
            "metadata": event.metadata
        }))
        
        # Appel des callbacks
        for callback in self.security_event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Erreur dans le callback d'événement de sécurité: {e}")
    
    async def _save_security_events(self) -> None:
        """Sauvegarde les événements de sécurité"""
        # Dans une implémentation réelle, on sauvegarderait en base de données
        # ou dans un système de log centralisé
        pass
    
    # API de nettoyage
    
    async def _cleanup_expired_sessions(self) -> None:
        """Nettoie les sessions expirées"""
        now = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if now > session.expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self._revoke_session(session_id, "EXPIRED")
        
        if expired_sessions:
            logger.info(f"Nettoyées {len(expired_sessions)} sessions expirées")
    
    async def _cleanup_rate_limit_cache(self) -> None:
        """Nettoie le cache de rate limiting"""
        current_time = int(time.time())
        expired_keys = []
        
        for key in self.rate_limit_cache:
            # Extraction du timestamp de la clé
            parts = key.split(":")
            if len(parts) >= 3:
                try:
                    key_time = int(parts[2])
                    if current_time - key_time > 1800:  # 30 minutes
                        expired_keys.append(key)
                except ValueError:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self.rate_limit_cache[key]
        
        if expired_keys:
            logger.debug(f"Nettoyées {len(expired_keys)} entrées de rate limiting")
    
    async def _cleanup_suspicious_ips(self) -> None:
        """Nettoie les IPs suspectes anciennes"""
        current_time = time.time()
        expired_ips = []
        
        for ip, last_seen in self.suspicious_ips.items():
            if current_time - last_seen > 86400:  # 24 heures
                expired_ips.append(ip)
        
        for ip in expired_ips:
            del self.suspicious_ips[ip]
    
    # API publique
    
    def add_security_event_callback(self, callback: Callable) -> None:
        """Ajoute un callback pour les événements de sécurité"""
        self.security_event_callbacks.append(callback)
    
    def remove_security_event_callback(self, callback: Callable) -> None:
        """Supprime un callback d'événement de sécurité"""
        if callback in self.security_event_callbacks:
            self.security_event_callbacks.remove(callback)
    
    async def block_ip(self, ip_address: str, reason: str) -> None:
        """Bloque une adresse IP"""
        self.blocked_ips.add(ip_address)
        
        await self._log_security_event(
            "IP_BLOCKED",
            None,
            None,
            ip_address,
            "",
            SecurityLevel.HIGH,
            f"IP bloquée: {reason}"
        )
    
    async def unblock_ip(self, ip_address: str) -> None:
        """Débloque une adresse IP"""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            
            await self._log_security_event(
                "IP_UNBLOCKED",
                None,
                None,
                ip_address,
                "",
                SecurityLevel.LOW,
                "IP débloquée"
            )
    
    async def get_security_events(
        self,
        limit: Optional[int] = None,
        severity: Optional[SecurityLevel] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[SecurityEvent]:
        """
        Récupère les événements de sécurité
        
        Args:
            limit: Limite du nombre d'événements
            severity: Filtrage par sévérité
            event_type: Filtrage par type d'événement
            user_id: Filtrage par utilisateur
            tenant_id: Filtrage par tenant
            
        Returns:
            Liste des événements
        """
        events = self.security_events.copy()
        
        # Filtrage
        if severity:
            events = [e for e in events if e.severity == severity]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]
        
        # Tri par timestamp décroissant
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Limitation
        if limit:
            events = events[:limit]
        
        return events
    
    async def get_active_sessions(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[AuthSession]:
        """
        Récupère les sessions actives
        
        Args:
            user_id: Filtrage par utilisateur
            tenant_id: Filtrage par tenant
            
        Returns:
            Liste des sessions actives
        """
        sessions = list(self.active_sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        
        if tenant_id:
            sessions = [s for s in sessions if s.tenant_id == tenant_id]
        
        return sessions
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de sécurité
        
        Returns:
            Métriques
        """
        return {
            **self.metrics,
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_ips),
            "security_policies": len(self.security_policies),
            "threat_indicators": len(self.threat_indicators),
            "rate_limit_entries": len(self.rate_limit_cache),
            "security_events_count": len(self.security_events)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérification de l'état de santé du gestionnaire
        
        Returns:
            Rapport d'état
        """
        try:
            return {
                "status": "healthy",
                "is_initialized": self.is_initialized,
                "encryption_configured": self.encryption_key is not None,
                "jwt_configured": bool(self.jwt_secret),
                "redis_available": self.redis_client is not None,
                "active_sessions": len(self.active_sessions),
                "security_policies": len(self.security_policies)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_initialized": self.is_initialized
            }
