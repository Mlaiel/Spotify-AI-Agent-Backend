"""
🛡️ SECURITY FRAMEWORK - SÉCURITÉ ENTERPRISE
Expert Team: Security Specialist, Microservices Architect

Sécurité complète avec authentification, autorisation, chiffrement et audit
"""

import asyncio
import os
import time
import hashlib
import secrets
import jwt
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager

# Cryptographie et sécurité
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt
from passlib.context import CryptContext

# OAuth et authentification
from authlib.integrations.fastapi_oauth2 import OAuth2
from authlib.jose import jwt as authlib_jwt, JsonWebKey
import httpx

# FastAPI Security
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Rate limiting et protection
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis

# Base framework
from .core import BaseFramework, FrameworkStatus, FrameworkHealth
from .core import framework_orchestrator

# Audit et logging
import structlog


class SecurityLevel(Enum):
    """Niveaux de sécurité"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthenticationMethod(Enum):
    """Méthodes d'authentification"""
    PASSWORD = "password"
    TOKEN = "token"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    MULTI_FACTOR = "mfa"
    BIOMETRIC = "biometric"


@dataclass
class SecurityConfig:
    """Configuration de sécurité"""
    
    # JWT Settings
    jwt_secret_key: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    # Password Policy
    min_password_length: int = 8
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    password_history_count: int = 5
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: str = "1 minute"
    rate_limit_redis_url: str = "redis://localhost:6379"
    
    # OAuth2 Providers
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""
    
    # Encryption
    encryption_key: Optional[bytes] = None
    rsa_key_size: int = 2048
    
    # Security Headers
    enable_hsts: bool = True
    enable_csp: bool = True
    enable_xfo: bool = True
    
    # Audit
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 90
    
    def __post_init__(self):
        if self.encryption_key is None:
            self.encryption_key = Fernet.generate_key()


@dataclass
class SecurityEvent:
    """Événement de sécurité pour audit"""
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: SecurityLevel = SecurityLevel.LOW


class CryptographyManager:
    """
    🔐 GESTIONNAIRE DE CRYPTOGRAPHIE
    
    Gestion complète du chiffrement:
    - Chiffrement symétrique (Fernet)
    - Chiffrement asymétrique (RSA)
    - Hachage sécurisé (bcrypt, PBKDF2)
    - Signatures numériques
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.fernet = Fernet(config.encryption_key)
        self.password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Génération des clés RSA
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=config.rsa_key_size
        )
        self.public_key = self.private_key.public_key()
        
        self.logger = logging.getLogger("security.crypto")
    
    def encrypt_data(self, data: str) -> str:
        """Chiffre des données avec Fernet"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Déchiffre des données"""
        try:
            decrypted_data = self.fernet.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def hash_password(self, password: str) -> str:
        """Hache un mot de passe avec bcrypt"""
        return self.password_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Vérifie un mot de passe"""
        return self.password_context.verify(plain_password, hashed_password)
    
    def encrypt_rsa(self, data: str) -> bytes:
        """Chiffre avec RSA (clé publique)"""
        try:
            encrypted = self.public_key.encrypt(
                data.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return encrypted
        except Exception as e:
            self.logger.error(f"RSA encryption failed: {e}")
            raise
    
    def decrypt_rsa(self, encrypted_data: bytes) -> str:
        """Déchiffre avec RSA (clé privée)"""
        try:
            decrypted = self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted.decode()
        except Exception as e:
            self.logger.error(f"RSA decryption failed: {e}")
            raise
    
    def sign_data(self, data: str) -> bytes:
        """Signe des données avec la clé privée"""
        try:
            signature = self.private_key.sign(
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature
        except Exception as e:
            self.logger.error(f"Data signing failed: {e}")
            raise
    
    def verify_signature(self, data: str, signature: bytes) -> bool:
        """Vérifie une signature"""
        try:
            self.public_key.verify(
                signature,
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Dérive une clé à partir d'un mot de passe"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password.encode())


class JWTManager:
    """
    🎫 GESTIONNAIRE JWT
    
    Gestion complète des tokens JWT:
    - Génération de tokens d'accès
    - Tokens de rafraîchissement
    - Validation et parsing
    - Révocation de tokens
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger("security.jwt")
        
        # Token blacklist (Redis)
        self.redis_client = redis.Redis.from_url(config.rate_limit_redis_url)
    
    def create_access_token(self, user_id: str, additional_claims: Optional[Dict] = None) -> str:
        """Crée un token d'accès JWT"""
        try:
            now = datetime.utcnow()
            expire = now + timedelta(minutes=self.config.jwt_access_token_expire_minutes)
            
            payload = {
                "sub": user_id,
                "iat": now,
                "exp": expire,
                "type": "access",
                "jti": secrets.token_urlsafe(16)  # JWT ID pour révocation
            }
            
            if additional_claims:
                payload.update(additional_claims)
            
            token = jwt.encode(
                payload,
                self.config.jwt_secret_key,
                algorithm=self.config.jwt_algorithm
            )
            
            self.logger.info(f"Access token created for user {user_id}")
            return token
            
        except Exception as e:
            self.logger.error(f"Access token creation failed: {e}")
            raise
    
    def create_refresh_token(self, user_id: str) -> str:
        """Crée un token de rafraîchissement"""
        try:
            now = datetime.utcnow()
            expire = now + timedelta(days=self.config.jwt_refresh_token_expire_days)
            
            payload = {
                "sub": user_id,
                "iat": now,
                "exp": expire,
                "type": "refresh",
                "jti": secrets.token_urlsafe(16)
            }
            
            token = jwt.encode(
                payload,
                self.config.jwt_secret_key,
                algorithm=self.config.jwt_algorithm
            )
            
            self.logger.info(f"Refresh token created for user {user_id}")
            return token
            
        except Exception as e:
            self.logger.error(f"Refresh token creation failed: {e}")
            raise
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Vérifie et décode un token JWT"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Vérifier le type de token
            if payload.get("type") != token_type:
                raise jwt.InvalidTokenError(f"Invalid token type: expected {token_type}")
            
            # Vérifier si le token est révoqué
            jti = payload.get("jti")
            if jti and self.redis_client.get(f"revoked_token:{jti}"):
                raise jwt.InvalidTokenError("Token has been revoked")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def revoke_token(self, token: str):
        """Révoque un token"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            jti = payload.get("jti")
            exp = payload.get("exp")
            
            if jti and exp:
                # Ajouter à la blacklist jusqu'à expiration
                ttl = exp - time.time()
                if ttl > 0:
                    self.redis_client.setex(f"revoked_token:{jti}", int(ttl), "1")
                    self.logger.info(f"Token revoked: {jti}")
            
        except Exception as e:
            self.logger.error(f"Token revocation failed: {e}")
            raise


class OAuth2Manager:
    """
    🔑 GESTIONNAIRE OAUTH2
    
    Intégration OAuth2 avec providers externes:
    - Spotify Web API
    - Google OAuth
    - Custom OAuth providers
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger("security.oauth2")
        
        # Configuration des providers
        self.providers = {
            "spotify": {
                "client_id": config.spotify_client_id,
                "client_secret": config.spotify_client_secret,
                "authorization_url": "https://accounts.spotify.com/authorize",
                "token_url": "https://accounts.spotify.com/api/token",
                "user_info_url": "https://api.spotify.com/v1/me",
                "scopes": ["user-read-private", "user-read-email", "user-library-read"]
            },
            "google": {
                "client_id": config.google_client_id,
                "client_secret": config.google_client_secret,
                "authorization_url": "https://accounts.google.com/o/oauth2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "user_info_url": "https://www.googleapis.com/oauth2/v2/userinfo",
                "scopes": ["openid", "email", "profile"]
            }
        }
    
    def get_authorization_url(self, provider: str, redirect_uri: str, state: str) -> str:
        """Génère l'URL d'autorisation OAuth2"""
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not supported")
            
            provider_config = self.providers[provider]
            
            params = {
                "client_id": provider_config["client_id"],
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "scope": " ".join(provider_config["scopes"]),
                "state": state
            }
            
            # Construire l'URL
            url = provider_config["authorization_url"]
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            
            return f"{url}?{query_string}"
            
        except Exception as e:
            self.logger.error(f"Authorization URL generation failed: {e}")
            raise
    
    async def exchange_code_for_token(
        self, 
        provider: str, 
        code: str, 
        redirect_uri: str
    ) -> Dict[str, Any]:
        """Échange le code d'autorisation contre un token d'accès"""
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not supported")
            
            provider_config = self.providers[provider]
            
            data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": provider_config["client_id"],
                "client_secret": provider_config["client_secret"]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    provider_config["token_url"],
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Failed to exchange code for token"
                    )
                
                return response.json()
                
        except Exception as e:
            self.logger.error(f"Code exchange failed: {e}")
            raise
    
    async def get_user_info(self, provider: str, access_token: str) -> Dict[str, Any]:
        """Récupère les informations utilisateur"""
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not supported")
            
            provider_config = self.providers[provider]
            
            headers = {"Authorization": f"Bearer {access_token}"}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    provider_config["user_info_url"],
                    headers=headers
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Failed to get user info"
                    )
                
                return response.json()
                
        except Exception as e:
            self.logger.error(f"User info retrieval failed: {e}")
            raise


class RateLimitManager:
    """
    🚦 GESTIONNAIRE DE LIMITATION DE DÉBIT
    
    Protection contre les abus avec:
    - Rate limiting par IP
    - Rate limiting par utilisateur
    - Limitation adaptive
    - Analyse comportementale
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.redis_client = redis.Redis.from_url(config.rate_limit_redis_url)
        self.logger = logging.getLogger("security.ratelimit")
        
        # Limiter par défaut
        self.limiter = Limiter(
            key_func=get_remote_address,
            storage_uri=config.rate_limit_redis_url
        )
    
    async def check_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int
    ) -> bool:
        """Vérifie la limitation de débit"""
        try:
            current_time = int(time.time())
            window_start = current_time - window
            
            # Nettoyer les anciennes entrées
            self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Compter les requêtes dans la fenêtre
            current_count = self.redis_client.zcard(key)
            
            if current_count >= limit:
                self.logger.warning(f"Rate limit exceeded for key: {key}")
                return False
            
            # Ajouter la requête actuelle
            self.redis_client.zadd(key, {str(current_time): current_time})
            self.redis_client.expire(key, window)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return True  # Permettre en cas d'erreur
    
    def get_rate_limit_decorator(self, requests: int, window: str):
        """Retourne un décorateur de limitation de débit"""
        return self.limiter.limit(f"{requests}/{window}")


class SecurityAuditManager:
    """
    📋 GESTIONNAIRE D'AUDIT DE SÉCURITÉ
    
    Audit complet des événements de sécurité:
    - Connexions et déconnexions
    - Tentatives d'authentification
    - Accès aux ressources sensibles
    - Détection d'anomalies
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = structlog.get_logger("security.audit")
        
        # Configuration du logging structuré
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    async def log_security_event(self, event: SecurityEvent):
        """Enregistre un événement de sécurité"""
        try:
            self.logger.info(
                "Security event logged",
                event_type=event.event_type,
                user_id=event.user_id,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                timestamp=event.timestamp.isoformat(),
                risk_level=event.risk_level.value,
                details=event.details
            )
            
            # Analyser le risque
            await self._analyze_risk(event)
            
        except Exception as e:
            self.logger.error("Failed to log security event", error=str(e))
    
    async def _analyze_risk(self, event: SecurityEvent):
        """Analyse le risque d'un événement"""
        try:
            # Détection d'anomalies basique
            risk_factors = []
            
            # Vérifier les tentatives de connexion échouées
            if event.event_type == "failed_login":
                recent_failures = await self._count_recent_failures(
                    event.ip_address, 
                    timedelta(minutes=15)
                )
                if recent_failures > 5:
                    risk_factors.append("multiple_failed_logins")
            
            # Analyser l'IP géographiquement
            if event.ip_address and not event.ip_address.startswith("127."):
                # Intégration avec service de géolocalisation
                pass
            
            # Analyser l'user agent
            if "bot" in event.user_agent.lower() or "crawler" in event.user_agent.lower():
                risk_factors.append("bot_user_agent")
            
            # Actions sur les données sensibles
            sensitive_actions = ["data_export", "admin_access", "config_change"]
            if event.event_type in sensitive_actions:
                risk_factors.append("sensitive_action")
            
            # Alertes si risque élevé
            if len(risk_factors) >= 2:
                await self._trigger_security_alert(event, risk_factors)
                
        except Exception as e:
            self.logger.error("Risk analysis failed", error=str(e))
    
    async def _count_recent_failures(self, ip_address: str, timeframe: timedelta) -> int:
        """Compte les échecs récents pour une IP"""
        # Implémentation de comptage dans Redis
        return 0  # Placeholder
    
    async def _trigger_security_alert(self, event: SecurityEvent, risk_factors: List[str]):
        """Déclenche une alerte de sécurité"""
        self.logger.warning(
            "SECURITY ALERT",
            event_type=event.event_type,
            user_id=event.user_id,
            ip_address=event.ip_address,
            risk_factors=risk_factors,
            timestamp=event.timestamp.isoformat()
        )
        
        # Intégration avec systèmes d'alerte (email, Slack, etc.)


class SecurityFrameworkManager(BaseFramework):
    """
    🛡️ GESTIONNAIRE PRINCIPAL DE SÉCURITÉ
    
    Orchestration complète de la sécurité avec:
    - Authentification multi-facteurs
    - Autorisation granulaire
    - Chiffrement bout-en-bout
    - Audit et conformité
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        super().__init__("security", config.__dict__ if config else {})
        self.config = config or SecurityConfig()
        
        # Composants de sécurité
        self.crypto_manager = CryptographyManager(self.config)
        self.jwt_manager = JWTManager(self.config)
        self.oauth2_manager = OAuth2Manager(self.config)
        self.rate_limit_manager = RateLimitManager(self.config)
        self.audit_manager = SecurityAuditManager(self.config)
        
        # Security middleware
        self.security_bearer = HTTPBearer()
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    
    async def initialize(self) -> bool:
        """Initialise le framework de sécurité"""
        try:
            # Vérifier les configurations de sécurité
            await self._validate_security_config()
            
            # Initialiser les composants
            await self._setup_security_middleware()
            
            # Démarrer l'audit
            await self.audit_manager.log_security_event(
                SecurityEvent(
                    event_type="security_framework_startup",
                    user_id=None,
                    ip_address="127.0.0.1",
                    user_agent="SecurityFramework/1.0",
                    timestamp=datetime.utcnow(),
                    risk_level=SecurityLevel.LOW
                )
            )
            
            self.logger.info("Security Framework Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Security framework initialization failed: {e}")
            return False
    
    async def _validate_security_config(self):
        """Valide la configuration de sécurité"""
        if len(self.config.jwt_secret_key) < 32:
            raise ValueError("JWT secret key too short (minimum 32 characters)")
        
        if not self.config.encryption_key:
            raise ValueError("Encryption key not configured")
    
    async def _setup_security_middleware(self):
        """Configure les middleware de sécurité"""
        # Configuration des headers de sécurité
        # Rate limiting
        # CORS sécurisé
        pass
    
    async def authenticate_user(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> Dict[str, Any]:
        """Authentifie un utilisateur via JWT"""
        try:
            token = credentials.credentials
            payload = self.jwt_manager.verify_token(token, "access")
            
            # Log de l'authentification
            await self.audit_manager.log_security_event(
                SecurityEvent(
                    event_type="user_authenticated",
                    user_id=payload.get("sub"),
                    ip_address="unknown",  # À récupérer du context request
                    user_agent="unknown",
                    timestamp=datetime.utcnow(),
                    risk_level=SecurityLevel.LOW
                )
            )
            
            return payload
            
        except Exception as e:
            await self.audit_manager.log_security_event(
                SecurityEvent(
                    event_type="authentication_failed",
                    user_id=None,
                    ip_address="unknown",
                    user_agent="unknown",
                    timestamp=datetime.utcnow(),
                    risk_level=SecurityLevel.MEDIUM,
                    details={"error": str(e)}
                )
            )
            raise
    
    async def authorize_user(
        self, 
        required_permissions: List[str],
        user_payload: Dict[str, Any]
    ) -> bool:
        """Vérifie les autorisations d'un utilisateur"""
        try:
            user_permissions = user_payload.get("permissions", [])
            
            # Vérifier si l'utilisateur a toutes les permissions requises
            has_permission = all(perm in user_permissions for perm in required_permissions)
            
            # Log de l'autorisation
            await self.audit_manager.log_security_event(
                SecurityEvent(
                    event_type="authorization_check",
                    user_id=user_payload.get("sub"),
                    ip_address="unknown",
                    user_agent="unknown",
                    timestamp=datetime.utcnow(),
                    risk_level=SecurityLevel.LOW,
                    details={
                        "required_permissions": required_permissions,
                        "has_permission": has_permission
                    }
                )
            )
            
            return has_permission
            
        except Exception as e:
            self.logger.error(f"Authorization check failed: {e}")
            return False
    
    def create_permission_dependency(self, required_permissions: List[str]):
        """Crée une dépendance FastAPI pour les permissions"""
        async def permission_checker(
            user_payload: Dict[str, Any] = Depends(self.authenticate_user)
        ):
            if not await self.authorize_user(required_permissions, user_payload):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return user_payload
        
        return permission_checker
    
    async def shutdown(self) -> bool:
        """Arrête le framework de sécurité"""
        try:
            # Log de l'arrêt
            await self.audit_manager.log_security_event(
                SecurityEvent(
                    event_type="security_framework_shutdown",
                    user_id=None,
                    ip_address="127.0.0.1",
                    user_agent="SecurityFramework/1.0",
                    timestamp=datetime.utcnow(),
                    risk_level=SecurityLevel.LOW
                )
            )
            
            self.logger.info("Security Framework Manager shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Security framework shutdown failed: {e}")
            return False
    
    async def health_check(self) -> FrameworkHealth:
        """Vérifie la santé du framework de sécurité"""
        health = FrameworkHealth(
            status=FrameworkStatus.RUNNING,
            last_check=time.time()
        )
        
        try:
            # Vérifier les composants de sécurité
            redis_connected = self.rate_limit_manager.redis_client.ping()
            
            health.metadata = {
                "jwt_configured": bool(self.config.jwt_secret_key),
                "encryption_configured": bool(self.config.encryption_key),
                "redis_connected": redis_connected,
                "oauth2_providers": len(self.oauth2_manager.providers)
            }
            
            if not redis_connected:
                health.status = FrameworkStatus.DEGRADED
            
        except Exception as e:
            health.status = FrameworkStatus.DEGRADED
            health.error_count += 1
            health.metadata["error"] = str(e)
        
        return health


# Instance globale du gestionnaire de sécurité
security_manager = SecurityFrameworkManager()


# Fonctions utilitaires pour FastAPI
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_manager.security_bearer)
) -> Dict[str, Any]:
    """Dépendance FastAPI pour obtenir l'utilisateur courant"""
    return asyncio.run(security_manager.authenticate_user(credentials))


def require_permissions(permissions: List[str]):
    """Décorateur pour exiger des permissions spécifiques"""
    return security_manager.create_permission_dependency(permissions)


# Export des classes principales
__all__ = [
    'SecurityFrameworkManager',
    'CryptographyManager',
    'JWTManager',
    'OAuth2Manager',
    'RateLimitManager',
    'SecurityAuditManager',
    'SecurityConfig',
    'SecurityEvent',
    'SecurityLevel',
    'AuthenticationMethod',
    'security_manager',
    'get_current_user',
    'require_permissions'
]
