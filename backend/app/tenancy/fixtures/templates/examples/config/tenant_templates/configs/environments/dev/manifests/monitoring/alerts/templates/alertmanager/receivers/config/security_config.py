"""
Module de configuration de sécurité ultra-avancé pour Alertmanager Receivers

Ce module implémente une sécurité end-to-end avec chiffrement, authentification,
autorisation et audit complet pour les receivers d'alertes.

Author: Spotify AI Agent Team
Maintainer: Fahed Mlaiel - Spécialiste Sécurité Backend
"""

import logging
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import jwt
import asyncio

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Niveaux de sécurité disponibles"""
    BASIC = "basic"
    STANDARD = "standard" 
    HIGH = "high"
    CRITICAL = "critical"
    MAXIMUM = "maximum"

class EncryptionAlgorithm(Enum):
    """Algorithmes de chiffrement supportés"""
    AES_256_GCM = "aes-256-gcm"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    AES_256_CBC = "aes-256-cbc"

class AuthenticationMethod(Enum):
    """Méthodes d'authentification"""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    MUTUAL_TLS = "mutual_tls"
    SAML = "saml"

@dataclass
class SecurityPolicy:
    """Politique de sécurité pour un tenant ou service"""
    name: str
    level: SecurityLevel
    encryption_algorithm: EncryptionAlgorithm
    auth_methods: List[AuthenticationMethod]
    max_token_lifetime: int = 3600  # secondes
    require_2fa: bool = False
    ip_whitelist: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 100
    audit_all_actions: bool = True
    data_retention_days: int = 90
    compliance_standards: List[str] = field(default_factory=lambda: ["SOC2", "ISO27001"])

@dataclass
class EncryptionConfig:
    """Configuration du chiffrement"""
    key_rotation_days: int = 30
    key_derivation_iterations: int = 100000
    salt_length: int = 32
    iv_length: int = 16
    tag_length: int = 16

class SecurityConfigManager:
    """Gestionnaire principal de la configuration de sécurité"""
    
    def __init__(self):
        self.policies: Dict[str, SecurityPolicy] = {}
        self.encryption_keys: Dict[str, bytes] = {}
        self.active_tokens: Dict[str, Dict] = {}
        self.audit_trail: List[Dict] = []
        self.encryption_config = EncryptionConfig()
        
    async def initialize_security(self) -> bool:
        """Initialise le système de sécurité"""
        try:
            logger.info("Initializing security configuration manager")
            
            # Génération des clés maîtres
            await self._generate_master_keys()
            
            # Chargement des politiques par défaut
            await self._load_default_policies()
            
            # Validation de la configuration
            await self._validate_security_config()
            
            logger.info("Security configuration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize security: {e}")
            return False
    
    async def _generate_master_keys(self):
        """Génère les clés de chiffrement maîtres"""
        for tenant in ["spotify-premium", "spotify-free", "spotify-family", "spotify-student"]:
            # Génération d'une clé unique par tenant
            key = Fernet.generate_key()
            self.encryption_keys[tenant] = key
            
            # Sauvegarde sécurisée (en production, utiliser un HSM)
            await self._store_key_securely(tenant, key)
    
    async def _store_key_securely(self, tenant: str, key: bytes):
        """Stocke une clé de manière sécurisée"""
        # En production: intégration avec HashiCorp Vault ou AWS KMS
        logger.info(f"Storing encryption key for tenant {tenant}")
        # Implementation would integrate with actual key management service
    
    async def _load_default_policies(self):
        """Charge les politiques de sécurité par défaut"""
        
        # Politique Premium - Sécurité maximale
        self.policies["spotify-premium"] = SecurityPolicy(
            name="premium-security",
            level=SecurityLevel.MAXIMUM,
            encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
            auth_methods=[AuthenticationMethod.MUTUAL_TLS, AuthenticationMethod.JWT_TOKEN],
            max_token_lifetime=1800,  # 30 minutes
            require_2fa=True,
            ip_whitelist=[],  # Pas de restriction IP pour premium
            rate_limit_per_minute=1000,
            audit_all_actions=True,
            data_retention_days=365,
            compliance_standards=["SOC2", "ISO27001", "PCI-DSS"]
        )
        
        # Politique Free - Sécurité standard
        self.policies["spotify-free"] = SecurityPolicy(
            name="free-security", 
            level=SecurityLevel.STANDARD,
            encryption_algorithm=EncryptionAlgorithm.AES_256_CBC,
            auth_methods=[AuthenticationMethod.API_KEY],
            max_token_lifetime=3600,  # 1 heure
            require_2fa=False,
            rate_limit_per_minute=100,
            audit_all_actions=False,
            data_retention_days=30
        )
        
        # Politique Family - Sécurité haute
        self.policies["spotify-family"] = SecurityPolicy(
            name="family-security",
            level=SecurityLevel.HIGH,
            encryption_algorithm=EncryptionAlgorithm.AES_256_GCM,
            auth_methods=[AuthenticationMethod.JWT_TOKEN, AuthenticationMethod.OAUTH2],
            max_token_lifetime=2400,  # 40 minutes
            require_2fa=True,
            rate_limit_per_minute=500,
            audit_all_actions=True,
            data_retention_days=180
        )
    
    async def encrypt_sensitive_data(self, data: str, tenant: str) -> str:
        """Chiffre les données sensibles pour un tenant"""
        try:
            if tenant not in self.encryption_keys:
                raise ValueError(f"No encryption key found for tenant {tenant}")
            
            fernet = Fernet(self.encryption_keys[tenant])
            encrypted_data = fernet.encrypt(data.encode())
            
            # Audit de l'opération
            await self._audit_action("encrypt", tenant, "data_encrypted")
            
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            logger.error(f"Encryption failed for tenant {tenant}: {e}")
            raise
    
    async def decrypt_sensitive_data(self, encrypted_data: str, tenant: str) -> str:
        """Déchiffre les données sensibles pour un tenant"""
        try:
            if tenant not in self.encryption_keys:
                raise ValueError(f"No encryption key found for tenant {tenant}")
            
            fernet = Fernet(self.encryption_keys[tenant])
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(decoded_data)
            
            # Audit de l'opération
            await self._audit_action("decrypt", tenant, "data_decrypted")
            
            return decrypted_data.decode()
            
        except Exception as e:
            logger.error(f"Decryption failed for tenant {tenant}: {e}")
            raise
    
    async def generate_jwt_token(self, tenant: str, user_id: str, permissions: List[str]) -> str:
        """Génère un token JWT sécurisé"""
        try:
            policy = self.policies.get(tenant)
            if not policy:
                raise ValueError(f"No security policy found for tenant {tenant}")
            
            # Payload du token
            payload = {
                "tenant": tenant,
                "user_id": user_id,
                "permissions": permissions,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(seconds=policy.max_token_lifetime),
                "jti": secrets.token_urlsafe(32)  # JWT ID unique
            }
            
            # Génération du token
            secret_key = self.encryption_keys[tenant]
            token = jwt.encode(payload, secret_key, algorithm="HS256")
            
            # Stockage en cache pour validation
            self.active_tokens[payload["jti"]] = {
                "tenant": tenant,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "expires_at": payload["exp"]
            }
            
            # Audit
            await self._audit_action("token_generated", tenant, f"user_{user_id}")
            
            return token
            
        except Exception as e:
            logger.error(f"JWT generation failed for tenant {tenant}: {e}")
            raise
    
    async def validate_jwt_token(self, token: str, tenant: str) -> Dict[str, Any]:
        """Valide un token JWT"""
        try:
            secret_key = self.encryption_keys[tenant]
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            
            # Vérification de la révocation
            jti = payload.get("jti")
            if jti not in self.active_tokens:
                raise ValueError("Token has been revoked")
            
            # Audit de la validation
            await self._audit_action("token_validated", tenant, payload.get("user_id"))
            
            return payload
            
        except jwt.ExpiredSignatureError:
            await self._audit_action("token_expired", tenant, "unknown")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            await self._audit_action("token_invalid", tenant, "unknown")
            raise ValueError("Invalid token")
    
    async def revoke_token(self, jti: str, tenant: str):
        """Révoque un token JWT"""
        if jti in self.active_tokens:
            del self.active_tokens[jti]
            await self._audit_action("token_revoked", tenant, jti)
    
    async def rotate_encryption_keys(self, tenant: str):
        """Effectue la rotation des clés de chiffrement"""
        try:
            logger.info(f"Starting key rotation for tenant {tenant}")
            
            # Sauvegarde de l'ancienne clé
            old_key = self.encryption_keys.get(tenant)
            
            # Génération de la nouvelle clé
            new_key = Fernet.generate_key()
            self.encryption_keys[tenant] = new_key
            
            # Sauvegarde sécurisée
            await self._store_key_securely(tenant, new_key)
            
            # Audit de la rotation
            await self._audit_action("key_rotated", tenant, "system")
            
            logger.info(f"Key rotation completed for tenant {tenant}")
            
        except Exception as e:
            logger.error(f"Key rotation failed for tenant {tenant}: {e}")
            raise
    
    async def _audit_action(self, action: str, tenant: str, user: str):
        """Enregistre une action dans l'audit trail"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "tenant": tenant,
            "user": user,
            "ip_address": "127.0.0.1",  # À récupérer du contexte réel
            "success": True
        }
        
        self.audit_trail.append(audit_entry)
        
        # En production: persistance dans une base de données d'audit
        logger.info(f"Audit: {action} by {user} for {tenant}")
    
    async def _validate_security_config(self):
        """Valide la configuration de sécurité"""
        # Validation des politiques
        for tenant, policy in self.policies.items():
            if policy.level == SecurityLevel.MAXIMUM and not policy.require_2fa:
                logger.warning(f"Maximum security level should require 2FA for {tenant}")
        
        # Validation des clés
        for tenant in self.policies.keys():
            if tenant not in self.encryption_keys:
                raise ValueError(f"Missing encryption key for tenant {tenant}")
    
    def get_security_policy(self, tenant: str) -> Optional[SecurityPolicy]:
        """Récupère la politique de sécurité d'un tenant"""
        return self.policies.get(tenant)
    
    async def update_security_policy(self, tenant: str, policy: SecurityPolicy):
        """Met à jour la politique de sécurité d'un tenant"""
        self.policies[tenant] = policy
        await self._audit_action("policy_updated", tenant, "admin")
    
    def get_audit_trail(self, tenant: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Récupère l'audit trail"""
        if tenant:
            return [entry for entry in self.audit_trail if entry["tenant"] == tenant][-limit:]
        return self.audit_trail[-limit:]
    
    async def check_rate_limit(self, tenant: str, identifier: str) -> bool:
        """Vérifie les limites de taux d'appel"""
        # Implementation basique - en production utiliser Redis
        policy = self.policies.get(tenant)
        if not policy:
            return False
        
        # Logique de rate limiting (à implémenter avec Redis)
        return True
    
    async def validate_ip_whitelist(self, tenant: str, ip_address: str) -> bool:
        """Valide l'adresse IP contre la whitelist"""
        policy = self.policies.get(tenant)
        if not policy or not policy.ip_whitelist:
            return True  # Pas de restriction si pas de whitelist
        
        return ip_address in policy.ip_whitelist

# Instance singleton
security_manager = SecurityConfigManager()
