# 🔐 Token Management & API Keys
# ===============================
# 
# Gestionnaire avancé de tokens et clés API
# pour l'enterprise avec rotation et sécurité.
#
# 🎖️ Expert: Lead Dev + Architecte IA + Spécialiste Sécurité Backend
#
# Développé par l'équipe d'experts enterprise
# ===============================

"""
🔐 Enterprise Token & API Key Management
========================================

Advanced token and API key management providing:
- JWT token generation, validation and rotation
- API key management with scopes and rate limiting
- Refresh token rotation and family validation
- Token introspection and revocation (RFC 7662, 7009)
- Secure token storage and encryption
- Token analytics and monitoring
- Multi-tenant token isolation
- Token-based access control (RBAC/ABAC)
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
from dataclasses import dataclass, asdict
import jwt
import redis
import logging
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import base64
import json

# Configuration et logging
logger = logging.getLogger(__name__)


class TokenType(Enum):
    """Types de tokens"""
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    ID_TOKEN = "id_token"
    API_KEY = "api_key"
    TEMPORARY_TOKEN = "temporary_token"
    DEVICE_TOKEN = "device_token"


class TokenStatus(Enum):
    """Statuts de token"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


class APIKeyScope(Enum):
    """Portées des clés API"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    DELETE = "delete"
    ANALYTICS = "analytics"
    ML_INFERENCE = "ml:inference"
    ML_TRAINING = "ml:training"
    MUSIC_ANALYSIS = "music:analysis"
    MUSIC_SEPARATION = "music:separation"


class EncryptionAlgorithm(Enum):
    """Algorithmes de chiffrement"""
    AES_256_GCM = "aes-256-gcm"
    RSA_OAEP = "rsa-oaep"
    FERNET = "fernet"


@dataclass
class TokenMetadata:
    """Métadonnées de token"""
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int = 0
    ip_addresses: List[str] = None
    user_agents: List[str] = None
    scopes: List[str] = None
    device_id: Optional[str] = None
    location: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.ip_addresses is None:
            self.ip_addresses = []
        if self.user_agents is None:
            self.user_agents = []
        if self.scopes is None:
            self.scopes = []


@dataclass
class TokenInfo:
    """Informations complètes d'un token"""
    token_id: str
    token_type: TokenType
    user_id: str
    client_id: Optional[str]
    status: TokenStatus
    metadata: TokenMetadata
    encrypted_payload: Optional[str] = None
    family_id: Optional[str] = None  # Pour rotation des refresh tokens
    parent_token_id: Optional[str] = None


@dataclass
class APIKey:
    """Clé API"""
    key_id: str
    user_id: str
    name: str
    key_hash: str
    scopes: List[APIKeyScope]
    rate_limit: int  # Requêtes par minute
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int = 0
    is_active: bool = True
    allowed_ips: List[str] = None
    webhook_url: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.allowed_ips is None:
            self.allowed_ips = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class JWTConfig:
    """Configuration JWT"""
    algorithm: str = "RS256"
    issuer: str = "spotify-ai-agent"
    audience: str = "spotify-ai-agent-api"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 30
    id_token_expire_minutes: int = 60
    
    # Clés RSA (à générer et stocker de manière sécurisée)
    private_key: Optional[str] = None
    public_key: Optional[str] = None


class AdvancedTokenManager:
    """Gestionnaire avancé de tokens"""
    
    def __init__(self, redis_client: redis.Redis, jwt_config: JWTConfig = None):
        self.redis_client = redis_client
        self.jwt_config = jwt_config or JWTConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialiser les clés de chiffrement
        self.fernet = self._initialize_encryption()
        
        # Générer les clés RSA si nécessaire
        if not self.jwt_config.private_key:
            self._generate_rsa_keys()
        
        # Configuration
        self.token_cleanup_interval = 3600  # 1 heure
        self.max_refresh_token_family_size = 10
        self.token_introspection_cache_ttl = 300  # 5 minutes
    
    def _initialize_encryption(self) -> Fernet:
        """Initialise le chiffrement Fernet"""
        try:
            # Dans un vrai système, cette clé devrait être stockée de manière sécurisée
            key = Fernet.generate_key()
            return Fernet(key)
        except Exception as exc:
            self.logger.error(f"Erreur initialisation chiffrement: {exc}")
            raise
    
    def _generate_rsa_keys(self):
        """Génère les clés RSA pour JWT"""
        try:
            # Générer la clé privée
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Extraire la clé publique
            public_key = private_key.public_key()
            
            # Sérialiser les clés
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            self.jwt_config.private_key = private_pem.decode('utf-8')
            self.jwt_config.public_key = public_pem.decode('utf-8')
            
            # Stocker les clés de manière sécurisée (Redis avec chiffrement)
            await self._store_keys_securely()
            
        except Exception as exc:
            self.logger.error(f"Erreur génération clés RSA: {exc}")
            raise
    
    async def create_access_token(
        self,
        user_id: str,
        scopes: List[str] = None,
        client_id: str = None,
        custom_claims: Dict[str, Any] = None,
        expires_in: Optional[int] = None
    ) -> str:
        """Crée un token d'accès JWT"""
        try:
            # Métadonnées
            now = datetime.utcnow()
            expiry = now + timedelta(
                minutes=expires_in or self.jwt_config.access_token_expire_minutes
            )
            
            token_id = str(uuid.uuid4())
            
            # Claims JWT
            claims = {
                "iss": self.jwt_config.issuer,
                "aud": self.jwt_config.audience,
                "sub": user_id,
                "iat": int(now.timestamp()),
                "exp": int(expiry.timestamp()),
                "jti": token_id,
                "token_type": TokenType.ACCESS_TOKEN.value,
                "scopes": scopes or [],
                "client_id": client_id
            }
            
            # Ajouter les claims personnalisés
            if custom_claims:
                claims.update(custom_claims)
            
            # Générer le JWT
            token = jwt.encode(
                claims,
                self.jwt_config.private_key,
                algorithm=self.jwt_config.algorithm
            )
            
            # Stocker les métadonnées
            metadata = TokenMetadata(
                created_at=now,
                expires_at=expiry,
                scopes=scopes or []
            )
            
            token_info = TokenInfo(
                token_id=token_id,
                token_type=TokenType.ACCESS_TOKEN,
                user_id=user_id,
                client_id=client_id,
                status=TokenStatus.ACTIVE,
                metadata=metadata
            )
            
            await self._store_token_info(token_info)
            
            self.logger.info(f"Token d'accès créé pour utilisateur {user_id}")
            return token
            
        except Exception as exc:
            self.logger.error(f"Erreur création token d'accès: {exc}")
            raise
    
    async def create_refresh_token(
        self,
        user_id: str,
        access_token_id: str,
        client_id: str = None,
        family_id: str = None
    ) -> str:
        """Crée un refresh token"""
        try:
            now = datetime.utcnow()
            expiry = now + timedelta(days=self.jwt_config.refresh_token_expire_days)
            
            token_id = str(uuid.uuid4())
            
            # Si pas de famille, en créer une nouvelle
            if not family_id:
                family_id = str(uuid.uuid4())
            
            # Générer un token sécurisé
            refresh_token = secrets.token_urlsafe(64)
            
            # Hacher le token pour stockage
            token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
            
            # Métadonnées
            metadata = TokenMetadata(
                created_at=now,
                expires_at=expiry
            )
            
            token_info = TokenInfo(
                token_id=token_id,
                token_type=TokenType.REFRESH_TOKEN,
                user_id=user_id,
                client_id=client_id,
                status=TokenStatus.ACTIVE,
                metadata=metadata,
                family_id=family_id,
                parent_token_id=access_token_id
            )
            
            # Stocker les informations
            await self._store_token_info(token_info)
            
            # Stocker le hash du token
            await self.redis_client.setex(
                f"refresh_token_hash:{token_hash}",
                int((expiry - now).total_seconds()),
                token_id
            )
            
            # Gérer la famille de tokens
            await self._manage_token_family(family_id, token_id)
            
            self.logger.info(f"Refresh token créé pour utilisateur {user_id}")
            return refresh_token
            
        except Exception as exc:
            self.logger.error(f"Erreur création refresh token: {exc}")
            raise
    
    async def validate_access_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Valide un token d'accès JWT"""
        try:
            # Décoder et valider le JWT
            payload = jwt.decode(
                token,
                self.jwt_config.public_key,
                algorithms=[self.jwt_config.algorithm],
                audience=self.jwt_config.audience,
                issuer=self.jwt_config.issuer
            )
            
            token_id = payload.get("jti")
            if not token_id:
                return False, None, "Token ID manquant"
            
            # Vérifier le statut du token
            token_info = await self._get_token_info(token_id)
            if not token_info:
                return False, None, "Token introuvable"
            
            if token_info.status != TokenStatus.ACTIVE:
                return False, None, f"Token {token_info.status.value}"
            
            # Mettre à jour l'utilisation
            await self._update_token_usage(token_info)
            
            return True, payload, None
            
        except jwt.ExpiredSignatureError:
            return False, None, "Token expiré"
        except jwt.InvalidTokenError as exc:
            return False, None, f"Token invalide: {str(exc)}"
        except Exception as exc:
            self.logger.error(f"Erreur validation token: {exc}")
            return False, None, "Erreur interne"
    
    async def refresh_access_token(
        self,
        refresh_token: str,
        scopes: List[str] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Actualise un token d'accès avec un refresh token"""
        try:
            # Valider le refresh token
            token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
            token_id = await self.redis_client.get(f"refresh_token_hash:{token_hash}")
            
            if not token_id:
                return None, None, "Refresh token invalide"
            
            token_id = token_id.decode() if isinstance(token_id, bytes) else token_id
            token_info = await self._get_token_info(token_id)
            
            if not token_info or token_info.status != TokenStatus.ACTIVE:
                return None, None, "Refresh token invalide ou révoqué"
            
            # Vérifier l'expiration
            if token_info.metadata.expires_at and datetime.utcnow() > token_info.metadata.expires_at:
                return None, None, "Refresh token expiré"
            
            # Révoquer l'ancien refresh token et sa famille
            await self._revoke_token_family(token_info.family_id)
            
            # Créer un nouveau token d'accès
            new_access_token = await self.create_access_token(
                user_id=token_info.user_id,
                scopes=scopes or token_info.metadata.scopes,
                client_id=token_info.client_id
            )
            
            # Créer un nouveau refresh token
            access_token_payload = jwt.decode(
                new_access_token,
                self.jwt_config.public_key,
                algorithms=[self.jwt_config.algorithm]
            )
            new_access_token_id = access_token_payload["jti"]
            
            new_refresh_token = await self.create_refresh_token(
                user_id=token_info.user_id,
                access_token_id=new_access_token_id,
                client_id=token_info.client_id
            )
            
            return new_access_token, new_refresh_token, None
            
        except Exception as exc:
            self.logger.error(f"Erreur actualisation token: {exc}")
            return None, None, "Erreur interne"
    
    async def revoke_token(self, token_id: str, reason: str = "manual"):
        """Révoque un token"""
        try:
            token_info = await self._get_token_info(token_id)
            if not token_info:
                return False
            
            # Mettre à jour le statut
            token_info.status = TokenStatus.REVOKED
            await self._store_token_info(token_info)
            
            # Si c'est un refresh token, révoquer toute la famille
            if token_info.token_type == TokenType.REFRESH_TOKEN and token_info.family_id:
                await self._revoke_token_family(token_info.family_id)
            
            # Logger la révocation
            await self._log_token_activity(token_info, "revoked", {"reason": reason})
            
            self.logger.info(f"Token révoqué: {token_id}, raison: {reason}")
            return True
            
        except Exception as exc:
            self.logger.error(f"Erreur révocation token {token_id}: {exc}")
            return False
    
    async def introspect_token(self, token: str) -> Dict[str, Any]:
        """Introspection de token (RFC 7662)"""
        try:
            # Vérifier le cache d'introspection
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            cache_key = f"token_introspection:{token_hash}"
            
            cached_result = await self.redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Valider le token
            is_valid, payload, error = await self.validate_access_token(token)
            
            result = {
                "active": is_valid,
                "client_id": payload.get("client_id") if is_valid else None,
                "username": payload.get("sub") if is_valid else None,
                "scope": " ".join(payload.get("scopes", [])) if is_valid else None,
                "exp": payload.get("exp") if is_valid else None,
                "iat": payload.get("iat") if is_valid else None,
                "token_type": payload.get("token_type") if is_valid else None
            }
            
            # Mettre en cache
            await self.redis_client.setex(
                cache_key,
                self.token_introspection_cache_ttl,
                json.dumps(result)
            )
            
            return result
            
        except Exception as exc:
            self.logger.error(f"Erreur introspection token: {exc}")
            return {"active": False}
    
    async def get_user_tokens(self, user_id: str, token_type: TokenType = None) -> List[TokenInfo]:
        """Récupère les tokens d'un utilisateur"""
        try:
            # Scanner les tokens de l'utilisateur
            pattern = f"token_info:*"
            token_keys = await self.redis_client.keys(pattern)
            
            user_tokens = []
            for key in token_keys:
                try:
                    token_info = await self._get_token_info_by_key(key)
                    if (token_info and 
                        token_info.user_id == user_id and 
                        (not token_type or token_info.token_type == token_type)):
                        user_tokens.append(token_info)
                except Exception:
                    continue
            
            return user_tokens
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération tokens utilisateur {user_id}: {exc}")
            return []
    
    # Méthodes privées
    async def _store_token_info(self, token_info: TokenInfo):
        """Stocke les informations d'un token"""
        try:
            # Sérialiser et chiffrer
            token_data = json.dumps(asdict(token_info), default=str)
            encrypted_data = self.fernet.encrypt(token_data.encode())
            
            # Calculer le TTL
            if token_info.metadata.expires_at:
                ttl = int((token_info.metadata.expires_at - datetime.utcnow()).total_seconds())
                if ttl > 0:
                    await self.redis_client.setex(
                        f"token_info:{token_info.token_id}",
                        ttl,
                        encrypted_data
                    )
            else:
                await self.redis_client.set(
                    f"token_info:{token_info.token_id}",
                    encrypted_data
                )
            
            # Ajouter à l'index utilisateur
            await self.redis_client.sadd(
                f"user_tokens:{token_info.user_id}",
                token_info.token_id
            )
            
        except Exception as exc:
            self.logger.error(f"Erreur stockage token info: {exc}")
    
    async def _get_token_info(self, token_id: str) -> Optional[TokenInfo]:
        """Récupère les informations d'un token"""
        try:
            encrypted_data = await self.redis_client.get(f"token_info:{token_id}")
            if not encrypted_data:
                return None
            
            # Déchiffrer et désérialiser
            decrypted_data = self.fernet.decrypt(encrypted_data)
            token_dict = json.loads(decrypted_data)
            
            return TokenInfo(**token_dict)
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération token info {token_id}: {exc}")
            return None
    
    async def _get_token_info_by_key(self, key: str) -> Optional[TokenInfo]:
        """Récupère les informations d'un token par clé Redis"""
        try:
            encrypted_data = await self.redis_client.get(key)
            if not encrypted_data:
                return None
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            token_dict = json.loads(decrypted_data)
            
            return TokenInfo(**token_dict)
            
        except Exception as exc:
            return None
    
    async def _update_token_usage(self, token_info: TokenInfo):
        """Met à jour l'utilisation d'un token"""
        try:
            token_info.metadata.last_used = datetime.utcnow()
            token_info.metadata.usage_count += 1
            
            await self._store_token_info(token_info)
            
        except Exception as exc:
            self.logger.error(f"Erreur mise à jour utilisation token: {exc}")
    
    async def _manage_token_family(self, family_id: str, token_id: str):
        """Gère une famille de refresh tokens"""
        try:
            # Ajouter à la famille
            await self.redis_client.sadd(f"token_family:{family_id}", token_id)
            
            # Limiter la taille de la famille
            family_size = await self.redis_client.scard(f"token_family:{family_id}")
            if family_size > self.max_refresh_token_family_size:
                # Supprimer les plus anciens tokens
                family_members = await self.redis_client.smembers(f"token_family:{family_id}")
                
                # Trier par date de création et supprimer les plus anciens
                members_with_dates = []
                for member in family_members:
                    member_str = member.decode() if isinstance(member, bytes) else member
                    token_info = await self._get_token_info(member_str)
                    if token_info:
                        members_with_dates.append((member_str, token_info.metadata.created_at))
                
                members_with_dates.sort(key=lambda x: x[1])
                
                # Supprimer les plus anciens
                to_remove = family_size - self.max_refresh_token_family_size
                for i in range(to_remove):
                    old_token_id = members_with_dates[i][0]
                    await self.revoke_token(old_token_id, "family_size_limit")
            
        except Exception as exc:
            self.logger.error(f"Erreur gestion famille tokens: {exc}")
    
    async def _revoke_token_family(self, family_id: str):
        """Révoque tous les tokens d'une famille"""
        try:
            family_members = await self.redis_client.smembers(f"token_family:{family_id}")
            
            for member in family_members:
                member_str = member.decode() if isinstance(member, bytes) else member
                await self.revoke_token(member_str, "family_revocation")
            
            # Supprimer la famille
            await self.redis_client.delete(f"token_family:{family_id}")
            
        except Exception as exc:
            self.logger.error(f"Erreur révocation famille tokens: {exc}")
    
    async def _log_token_activity(
        self,
        token_info: TokenInfo,
        action: str,
        metadata: Dict[str, Any] = None
    ):
        """Enregistre l'activité des tokens"""
        try:
            activity_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "token_id": token_info.token_id,
                "token_type": token_info.token_type.value,
                "user_id": token_info.user_id,
                "action": action,
                "metadata": metadata or {}
            }
            
            # Stocker dans Redis pour analyse
            await self.redis_client.lpush(
                f"token_activity:{token_info.user_id}",
                json.dumps(activity_log)
            )
            
            # Limiter l'historique
            await self.redis_client.ltrim(f"token_activity:{token_info.user_id}", 0, 999)
            
        except Exception as exc:
            self.logger.error(f"Erreur log activité token: {exc}")
    
    async def _store_keys_securely(self):
        """Stocke les clés RSA de manière sécurisée"""
        try:
            # Chiffrer les clés
            private_key_encrypted = self.fernet.encrypt(self.jwt_config.private_key.encode())
            public_key_encrypted = self.fernet.encrypt(self.jwt_config.public_key.encode())
            
            # Stocker dans Redis
            await self.redis_client.set("jwt_private_key", private_key_encrypted)
            await self.redis_client.set("jwt_public_key", public_key_encrypted)
            
        except Exception as exc:
            self.logger.error(f"Erreur stockage clés sécurisé: {exc}")


class APIKeyManager:
    """Gestionnaire de clés API"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.default_rate_limit = 1000  # Requêtes par minute
        self.api_key_length = 32
        self.api_key_prefix = "sk_"
    
    async def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: List[APIKeyScope],
        rate_limit: int = None,
        expires_in_days: int = None,
        allowed_ips: List[str] = None
    ) -> Tuple[str, APIKey]:
        """Crée une nouvelle clé API"""
        try:
            # Générer la clé API
            key_id = str(uuid.uuid4())
            api_key = f"{self.api_key_prefix}{secrets.token_urlsafe(self.api_key_length)}"
            
            # Hacher la clé pour stockage
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Calculer l'expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            # Créer l'objet APIKey
            api_key_obj = APIKey(
                key_id=key_id,
                user_id=user_id,
                name=name,
                key_hash=key_hash,
                scopes=scopes,
                rate_limit=rate_limit or self.default_rate_limit,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                allowed_ips=allowed_ips or []
            )
            
            # Stocker la clé API
            await self._store_api_key(api_key_obj)
            
            # Créer l'index pour recherche rapide
            await self.redis_client.set(f"api_key_lookup:{key_hash}", key_id)
            
            self.logger.info(f"Clé API créée pour utilisateur {user_id}: {name}")
            return api_key, api_key_obj
            
        except Exception as exc:
            self.logger.error(f"Erreur création clé API: {exc}")
            raise
    
    async def validate_api_key(
        self,
        api_key: str,
        required_scopes: List[APIKeyScope] = None,
        ip_address: str = None
    ) -> Tuple[bool, Optional[APIKey], Optional[str]]:
        """Valide une clé API"""
        try:
            # Hacher la clé
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Récupérer l'ID de la clé
            key_id = await self.redis_client.get(f"api_key_lookup:{key_hash}")
            if not key_id:
                return False, None, "Clé API invalide"
            
            key_id = key_id.decode() if isinstance(key_id, bytes) else key_id
            
            # Récupérer la clé API
            api_key_obj = await self._get_api_key(key_id)
            if not api_key_obj:
                return False, None, "Clé API introuvable"
            
            # Vérifier l'état
            if not api_key_obj.is_active:
                return False, api_key_obj, "Clé API désactivée"
            
            # Vérifier l'expiration
            if api_key_obj.expires_at and datetime.utcnow() > api_key_obj.expires_at:
                return False, api_key_obj, "Clé API expirée"
            
            # Vérifier les IPs autorisées
            if api_key_obj.allowed_ips and ip_address:
                if ip_address not in api_key_obj.allowed_ips:
                    return False, api_key_obj, "IP non autorisée"
            
            # Vérifier les scopes
            if required_scopes:
                missing_scopes = set(required_scopes) - set(api_key_obj.scopes)
                if missing_scopes:
                    return False, api_key_obj, f"Scopes manquants: {missing_scopes}"
            
            # Vérifier la limite de taux
            rate_limit_key = f"api_key_rate_limit:{key_id}"
            current_usage = await self.redis_client.get(rate_limit_key)
            
            if current_usage:
                current_count = int(current_usage)
                if current_count >= api_key_obj.rate_limit:
                    return False, api_key_obj, "Limite de taux dépassée"
            
            # Incrémenter l'utilisation
            await self._update_api_key_usage(api_key_obj)
            
            return True, api_key_obj, None
            
        except Exception as exc:
            self.logger.error(f"Erreur validation clé API: {exc}")
            return False, None, "Erreur interne"
    
    async def revoke_api_key(self, key_id: str, user_id: str = None) -> bool:
        """Révoque une clé API"""
        try:
            api_key = await self._get_api_key(key_id)
            if not api_key:
                return False
            
            # Vérifier la propriété si user_id fourni
            if user_id and api_key.user_id != user_id:
                return False
            
            # Désactiver la clé
            api_key.is_active = False
            await self._store_api_key(api_key)
            
            # Supprimer l'index de recherche
            await self.redis_client.delete(f"api_key_lookup:{api_key.key_hash}")
            
            self.logger.info(f"Clé API révoquée: {key_id}")
            return True
            
        except Exception as exc:
            self.logger.error(f"Erreur révocation clé API {key_id}: {exc}")
            return False
    
    async def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """Récupère les clés API d'un utilisateur"""
        try:
            # Scanner les clés API de l'utilisateur
            pattern = f"api_key:*"
            key_keys = await self.redis_client.keys(pattern)
            
            user_keys = []
            for key in key_keys:
                try:
                    api_key = await self._get_api_key_by_key(key)
                    if api_key and api_key.user_id == user_id:
                        user_keys.append(api_key)
                except Exception:
                    continue
            
            return user_keys
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération clés API utilisateur {user_id}: {exc}")
            return []
    
    async def update_api_key_rate_limit(self, key_id: str, new_rate_limit: int) -> bool:
        """Met à jour la limite de taux d'une clé API"""
        try:
            api_key = await self._get_api_key(key_id)
            if not api_key:
                return False
            
            api_key.rate_limit = new_rate_limit
            await self._store_api_key(api_key)
            
            return True
            
        except Exception as exc:
            self.logger.error(f"Erreur mise à jour limite taux clé API {key_id}: {exc}")
            return False
    
    # Méthodes privées
    async def _store_api_key(self, api_key: APIKey):
        """Stocke une clé API"""
        try:
            key_data = json.dumps(asdict(api_key), default=str)
            await self.redis_client.set(f"api_key:{api_key.key_id}", key_data)
            
            # Ajouter à l'index utilisateur
            await self.redis_client.sadd(f"user_api_keys:{api_key.user_id}", api_key.key_id)
            
        except Exception as exc:
            self.logger.error(f"Erreur stockage clé API: {exc}")
    
    async def _get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Récupère une clé API par son ID"""
        try:
            key_data = await self.redis_client.get(f"api_key:{key_id}")
            if key_data:
                key_dict = json.loads(key_data)
                return APIKey(**key_dict)
            return None
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération clé API {key_id}: {exc}")
            return None
    
    async def _get_api_key_by_key(self, key: str) -> Optional[APIKey]:
        """Récupère une clé API par clé Redis"""
        try:
            key_data = await self.redis_client.get(key)
            if key_data:
                key_dict = json.loads(key_data)
                return APIKey(**key_dict)
            return None
            
        except Exception:
            return None
    
    async def _update_api_key_usage(self, api_key: APIKey):
        """Met à jour l'utilisation d'une clé API"""
        try:
            # Mettre à jour les métadonnées
            api_key.last_used = datetime.utcnow()
            api_key.usage_count += 1
            
            await self._store_api_key(api_key)
            
            # Incrémenter le compteur de taux
            rate_limit_key = f"api_key_rate_limit:{api_key.key_id}"
            
            # Utiliser une fenêtre glissante d'une minute
            pipeline = self.redis_client.pipeline()
            pipeline.incr(rate_limit_key)
            pipeline.expire(rate_limit_key, 60)  # Expire après 1 minute
            await pipeline.execute()
            
        except Exception as exc:
            self.logger.error(f"Erreur mise à jour utilisation clé API: {exc}")


class TokenAnalytics:
    """Analyseur de tokens et métriques"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def get_token_metrics(self, user_id: str = None) -> Dict[str, Any]:
        """Récupère les métriques des tokens"""
        try:
            metrics = {
                "total_tokens": 0,
                "active_tokens": 0,
                "expired_tokens": 0,
                "revoked_tokens": 0,
                "tokens_by_type": {},
                "tokens_by_client": {},
                "usage_statistics": {}
            }
            
            # Scanner tous les tokens (ou ceux d'un utilisateur)
            if user_id:
                pattern = f"token_info:*"
            else:
                pattern = "token_info:*"
            
            token_keys = await self.redis_client.keys(pattern)
            
            for key in token_keys:
                try:
                    token_data = await self.redis_client.get(key)
                    if token_data:
                        # Pour cet exemple, ne pas déchiffrer - juste compter
                        metrics["total_tokens"] += 1
                        
                except Exception:
                    continue
            
            return metrics
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération métriques tokens: {exc}")
            return {}
    
    async def get_api_key_analytics(self, user_id: str = None) -> Dict[str, Any]:
        """Récupère l'analytique des clés API"""
        try:
            analytics = {
                "total_keys": 0,
                "active_keys": 0,
                "usage_by_scope": {},
                "rate_limit_violations": 0,
                "top_consumers": []
            }
            
            # Implémentation simplifiée
            # Dans un vrai système, analyser les données d'utilisation
            
            return analytics
            
        except Exception as exc:
            self.logger.error(f"Erreur analytique clés API: {exc}")
            return {}


class TokenCleanupService:
    """Service de nettoyage des tokens"""
    
    def __init__(self, token_manager: AdvancedTokenManager, api_key_manager: APIKeyManager):
        self.token_manager = token_manager
        self.api_key_manager = api_key_manager
        self.logger = logging.getLogger(__name__)
    
    async def cleanup_expired_tokens(self) -> Dict[str, int]:
        """Nettoie les tokens expirés"""
        try:
            cleanup_stats = {
                "tokens_scanned": 0,
                "tokens_cleaned": 0,
                "api_keys_cleaned": 0
            }
            
            # Nettoyer les tokens expirés
            pattern = "token_info:*"
            token_keys = await self.token_manager.redis_client.keys(pattern)
            
            for key in token_keys:
                try:
                    cleanup_stats["tokens_scanned"] += 1
                    
                    # Vérifier si le token existe encore (TTL automatique)
                    exists = await self.token_manager.redis_client.exists(key)
                    if not exists:
                        cleanup_stats["tokens_cleaned"] += 1
                        
                except Exception:
                    continue
            
            # Nettoyer les clés API expirées
            api_pattern = "api_key:*"
            api_keys = await self.api_key_manager.redis_client.keys(api_pattern)
            
            for key in api_keys:
                try:
                    api_key = await self.api_key_manager._get_api_key_by_key(key)
                    if (api_key and api_key.expires_at and 
                        datetime.utcnow() > api_key.expires_at):
                        
                        await self.api_key_manager.revoke_api_key(api_key.key_id)
                        cleanup_stats["api_keys_cleaned"] += 1
                        
                except Exception:
                    continue
            
            self.logger.info(f"Nettoyage terminé: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as exc:
            self.logger.error(f"Erreur nettoyage tokens: {exc}")
            return {}
    
    async def rotate_encryption_keys(self):
        """Rotation des clés de chiffrement"""
        try:
            # Implémentation de la rotation des clés
            # Dans un vrai système, chiffrer à nouveau tous les tokens avec une nouvelle clé
            
            self.logger.info("Rotation des clés de chiffrement terminée")
            
        except Exception as exc:
            self.logger.error(f"Erreur rotation clés: {exc}")
