"""
Enterprise Secrets Manager for Multi-Tenant Configuration

Ce module fournit une gestion sécurisée des secrets pour les configurations
d'environnement avec chiffrement, rotation et audit.

Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import os
import json
import base64
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SecretType(str, Enum):
    """Types de secrets supportés."""
    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    JWT_SECRET = "jwt_secret"
    ENCRYPTION_KEY = "encryption_key"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    OAUTH_CLIENT_SECRET = "oauth_client_secret"
    WEBHOOK_SECRET = "webhook_secret"


class SecretScope(str, Enum):
    """Portée des secrets."""
    GLOBAL = "global"
    ENVIRONMENT = "environment"
    TENANT = "tenant"
    SERVICE = "service"


@dataclass
class SecretMetadata:
    """Métadonnées d'un secret."""
    
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    rotation_interval: Optional[timedelta] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def is_expired(self) -> bool:
        """Vérifie si le secret a expiré."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def needs_rotation(self) -> bool:
        """Vérifie si le secret doit être tourné."""
        if self.rotation_interval is None:
            return False
        return datetime.now() - self.updated_at > self.rotation_interval
    
    def record_access(self) -> None:
        """Enregistre un accès au secret."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class Secret:
    """Représentation d'un secret."""
    
    key: str
    value: str
    secret_type: SecretType
    scope: SecretScope
    environment: str
    metadata: SecretMetadata
    encrypted: bool = True
    
    def __post_init__(self):
        """Post-initialisation."""
        if not self.encrypted:
            logger.warning(f"Secret non chiffré: {self.key}")
    
    def is_valid(self) -> bool:
        """Vérifie la validité du secret."""
        return not self.metadata.is_expired()
    
    def mask_value(self, show_chars: int = 4) -> str:
        """Retourne une version masquée du secret."""
        if len(self.value) <= show_chars:
            return "*" * len(self.value)
        return self.value[:show_chars] + "*" * (len(self.value) - show_chars)


class EncryptionManager:
    """Gestionnaire de chiffrement pour les secrets."""
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialise le gestionnaire de chiffrement.
        
        Args:
            master_key: Clé maître pour le chiffrement (optionnel)
        """
        self._master_key = master_key or self._generate_master_key()
        self._fernet = self._create_fernet()
    
    def _generate_master_key(self) -> str:
        """Génère une clé maître."""
        # En production, cette clé devrait venir d'un HSM ou KMS
        key = Fernet.generate_key()
        return base64.urlsafe_b64encode(key).decode()
    
    def _create_fernet(self) -> Fernet:
        """Crée une instance Fernet."""
        key_bytes = base64.urlsafe_b64decode(self._master_key.encode())
        return Fernet(key_bytes)
    
    def encrypt(self, plaintext: str) -> str:
        """Chiffre un texte en clair."""
        try:
            encrypted_bytes = self._fernet.encrypt(plaintext.encode())
            return base64.urlsafe_b64encode(encrypted_bytes).decode()
        except Exception as e:
            logger.error(f"Erreur de chiffrement: {e}")
            raise
    
    def decrypt(self, ciphertext: str) -> str:
        """Déchiffre un texte chiffré."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(ciphertext.encode())
            decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Erreur de déchiffrement: {e}")
            raise
    
    def rotate_key(self) -> str:
        """Effectue une rotation de la clé maître."""
        old_fernet = self._fernet
        new_master_key = self._generate_master_key()
        new_fernet = Fernet(base64.urlsafe_b64decode(new_master_key.encode()))
        
        # En production, il faudrait re-chiffrer tous les secrets existants
        self._master_key = new_master_key
        self._fernet = new_fernet
        
        logger.info("Rotation de la clé maître effectuée")
        return new_master_key


class SecretGenerator:
    """Générateur de secrets sécurisés."""
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """Génère une clé API sécurisée."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_password(length: int = 16, 
                         include_symbols: bool = True,
                         include_numbers: bool = True,
                         include_uppercase: bool = True,
                         include_lowercase: bool = True) -> str:
        """Génère un mot de passe sécurisé."""
        characters = ""
        
        if include_lowercase:
            characters += "abcdefghijklmnopqrstuvwxyz"
        if include_uppercase:
            characters += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if include_numbers:
            characters += "0123456789"
        if include_symbols:
            characters += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        if not characters:
            raise ValueError("Au moins un type de caractère doit être inclus")
        
        return ''.join(secrets.choice(characters) for _ in range(length))
    
    @staticmethod
    def generate_jwt_secret(length: int = 64) -> str:
        """Génère un secret JWT sécurisé."""
        return secrets.token_hex(length)
    
    @staticmethod
    def generate_encryption_key() -> str:
        """Génère une clé de chiffrement."""
        return base64.urlsafe_b64encode(Fernet.generate_key()).decode()


class SecretsManager:
    """Gestionnaire de secrets enterprise."""
    
    def __init__(self, 
                 environment: str,
                 secrets_dir: Optional[Union[str, Path]] = None,
                 use_keyring: bool = True,
                 master_key: Optional[str] = None):
        """
        Initialise le gestionnaire de secrets.
        
        Args:
            environment: Environnement cible
            secrets_dir: Répertoire des secrets
            use_keyring: Utilise le keyring système
            master_key: Clé maître pour le chiffrement
        """
        self.environment = environment
        self.secrets_dir = Path(secrets_dir) if secrets_dir else Path.cwd() / "secrets" / environment
        self.use_keyring = use_keyring
        
        # Initialisation
        self._secrets: Dict[str, Secret] = {}
        self._encryption_manager = EncryptionManager(master_key)
        self._lock = threading.RLock()
        self._audit_log: List[Dict[str, Any]] = []
        
        # Création du répertoire des secrets
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        
        # Chargement des secrets existants
        self._load_secrets()
    
    def _load_secrets(self) -> None:
        """Charge tous les secrets depuis le stockage."""
        secrets_file = self.secrets_dir / "secrets.json"
        
        if secrets_file.exists():
            try:
                with open(secrets_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for secret_data in data.get("secrets", []):
                    secret = self._deserialize_secret(secret_data)
                    self._secrets[secret.key] = secret
                
                logger.info(f"Chargé {len(self._secrets)} secrets pour {self.environment}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des secrets: {e}")
    
    def _save_secrets(self) -> None:
        """Sauvegarde tous les secrets."""
        secrets_file = self.secrets_dir / "secrets.json"
        
        try:
            data = {
                "environment": self.environment,
                "updated_at": datetime.now().isoformat(),
                "secrets": [self._serialize_secret(secret) for secret in self._secrets.values()]
            }
            
            with open(secrets_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des secrets: {e}")
            raise
    
    def _serialize_secret(self, secret: Secret) -> Dict[str, Any]:
        """Sérialise un secret pour le stockage."""
        return {
            "key": secret.key,
            "value": secret.value,  # Déjà chiffré
            "secret_type": secret.secret_type.value,
            "scope": secret.scope.value,
            "environment": secret.environment,
            "encrypted": secret.encrypted,
            "metadata": {
                "created_at": secret.metadata.created_at.isoformat(),
                "updated_at": secret.metadata.updated_at.isoformat(),
                "expires_at": secret.metadata.expires_at.isoformat() if secret.metadata.expires_at else None,
                "rotation_interval": str(secret.metadata.rotation_interval) if secret.metadata.rotation_interval else None,
                "last_accessed": secret.metadata.last_accessed.isoformat() if secret.metadata.last_accessed else None,
                "access_count": secret.metadata.access_count,
                "tags": secret.metadata.tags,
                "description": secret.metadata.description
            }
        }
    
    def _deserialize_secret(self, data: Dict[str, Any]) -> Secret:
        """Désérialise un secret depuis le stockage."""
        metadata_data = data["metadata"]
        
        metadata = SecretMetadata(
            created_at=datetime.fromisoformat(metadata_data["created_at"]),
            updated_at=datetime.fromisoformat(metadata_data["updated_at"]),
            expires_at=datetime.fromisoformat(metadata_data["expires_at"]) if metadata_data.get("expires_at") else None,
            rotation_interval=timedelta(seconds=int(metadata_data["rotation_interval"].split(':')[2])) if metadata_data.get("rotation_interval") else None,
            last_accessed=datetime.fromisoformat(metadata_data["last_accessed"]) if metadata_data.get("last_accessed") else None,
            access_count=metadata_data.get("access_count", 0),
            tags=metadata_data.get("tags", []),
            description=metadata_data.get("description", "")
        )
        
        return Secret(
            key=data["key"],
            value=data["value"],
            secret_type=SecretType(data["secret_type"]),
            scope=SecretScope(data["scope"]),
            environment=data["environment"],
            metadata=metadata,
            encrypted=data.get("encrypted", True)
        )
    
    def set_secret(self, 
                   key: str,
                   value: str,
                   secret_type: SecretType = SecretType.API_KEY,
                   scope: SecretScope = SecretScope.ENVIRONMENT,
                   expires_in: Optional[timedelta] = None,
                   rotation_interval: Optional[timedelta] = None,
                   tags: List[str] = None,
                   description: str = "") -> None:
        """
        Définit un secret.
        
        Args:
            key: Clé du secret
            value: Valeur du secret
            secret_type: Type de secret
            scope: Portée du secret
            expires_in: Délai d'expiration
            rotation_interval: Intervalle de rotation
            tags: Tags associés
            description: Description du secret
        """
        with self._lock:
            # Chiffrement de la valeur
            encrypted_value = self._encryption_manager.encrypt(value)
            
            # Création des métadonnées
            now = datetime.now()
            expires_at = now + expires_in if expires_in else None
            
            metadata = SecretMetadata(
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                rotation_interval=rotation_interval,
                tags=tags or [],
                description=description
            )
            
            # Création du secret
            secret = Secret(
                key=key,
                value=encrypted_value,
                secret_type=secret_type,
                scope=scope,
                environment=self.environment,
                metadata=metadata,
                encrypted=True
            )
            
            self._secrets[key] = secret
            self._save_secrets()
            
            # Audit
            self._log_audit("set_secret", key, {"secret_type": secret_type.value})
            
            logger.info(f"Secret défini: {key} ({secret_type.value})")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Récupère un secret déchiffré.
        
        Args:
            key: Clé du secret
            default: Valeur par défaut
            
        Returns:
            Valeur déchiffrée du secret
        """
        with self._lock:
            secret = self._secrets.get(key)
            
            if not secret:
                # Tentative de récupération depuis keyring
                if self.use_keyring:
                    value = keyring.get_password(f"spotify-ai-agent-{self.environment}", key)
                    if value:
                        return value
                
                return default
            
            if not secret.is_valid():
                logger.warning(f"Secret expiré: {key}")
                return default
            
            try:
                # Déchiffrement
                decrypted_value = self._encryption_manager.decrypt(secret.value)
                
                # Enregistrement de l'accès
                secret.metadata.record_access()
                
                # Audit
                self._log_audit("get_secret", key, {"access_count": secret.metadata.access_count})
                
                return decrypted_value
                
            except Exception as e:
                logger.error(f"Erreur lors du déchiffrement de {key}: {e}")
                return default
    
    def delete_secret(self, key: str) -> bool:
        """
        Supprime un secret.
        
        Args:
            key: Clé du secret
            
        Returns:
            True si supprimé avec succès
        """
        with self._lock:
            if key in self._secrets:
                del self._secrets[key]
                self._save_secrets()
                
                # Suppression du keyring si utilisé
                if self.use_keyring:
                    try:
                        keyring.delete_password(f"spotify-ai-agent-{self.environment}", key)
                    except Exception:
                        pass  # Le secret n'était peut-être pas dans le keyring
                
                # Audit
                self._log_audit("delete_secret", key)
                
                logger.info(f"Secret supprimé: {key}")
                return True
            
            return False
    
    def list_secrets(self, 
                    secret_type: Optional[SecretType] = None,
                    scope: Optional[SecretScope] = None,
                    include_expired: bool = False) -> List[Dict[str, Any]]:
        """
        Liste les secrets disponibles.
        
        Args:
            secret_type: Filtre par type de secret
            scope: Filtre par portée
            include_expired: Inclut les secrets expirés
            
        Returns:
            Liste des secrets (sans valeurs)
        """
        with self._lock:
            result = []
            
            for secret in self._secrets.values():
                # Filtres
                if secret_type and secret.secret_type != secret_type:
                    continue
                if scope and secret.scope != scope:
                    continue
                if not include_expired and not secret.is_valid():
                    continue
                
                result.append({
                    "key": secret.key,
                    "secret_type": secret.secret_type.value,
                    "scope": secret.scope.value,
                    "environment": secret.environment,
                    "created_at": secret.metadata.created_at.isoformat(),
                    "updated_at": secret.metadata.updated_at.isoformat(),
                    "expires_at": secret.metadata.expires_at.isoformat() if secret.metadata.expires_at else None,
                    "last_accessed": secret.metadata.last_accessed.isoformat() if secret.metadata.last_accessed else None,
                    "access_count": secret.metadata.access_count,
                    "tags": secret.metadata.tags,
                    "description": secret.metadata.description,
                    "is_valid": secret.is_valid(),
                    "needs_rotation": secret.metadata.needs_rotation(),
                    "masked_value": secret.mask_value()
                })
            
            return result
    
    def rotate_secret(self, key: str, new_value: Optional[str] = None) -> bool:
        """
        Effectue la rotation d'un secret.
        
        Args:
            key: Clé du secret
            new_value: Nouvelle valeur (générée automatiquement si None)
            
        Returns:
            True si rotation réussie
        """
        with self._lock:
            secret = self._secrets.get(key)
            if not secret:
                logger.error(f"Secret non trouvé pour rotation: {key}")
                return False
            
            # Génération de la nouvelle valeur si nécessaire
            if new_value is None:
                if secret.secret_type == SecretType.API_KEY:
                    new_value = SecretGenerator.generate_api_key()
                elif secret.secret_type == SecretType.DATABASE_PASSWORD:
                    new_value = SecretGenerator.generate_password()
                elif secret.secret_type == SecretType.JWT_SECRET:
                    new_value = SecretGenerator.generate_jwt_secret()
                elif secret.secret_type == SecretType.ENCRYPTION_KEY:
                    new_value = SecretGenerator.generate_encryption_key()
                else:
                    logger.error(f"Type de secret non supporté pour génération automatique: {secret.secret_type}")
                    return False
            
            # Chiffrement de la nouvelle valeur
            encrypted_value = self._encryption_manager.encrypt(new_value)
            
            # Mise à jour du secret
            secret.value = encrypted_value
            secret.metadata.updated_at = datetime.now()
            
            self._save_secrets()
            
            # Audit
            self._log_audit("rotate_secret", key, {"secret_type": secret.secret_type.value})
            
            logger.info(f"Secret tourné: {key}")
            return True
    
    def rotate_expired_secrets(self) -> List[str]:
        """
        Effectue la rotation de tous les secrets expirés.
        
        Returns:
            Liste des clés des secrets tournés
        """
        rotated_keys = []
        
        for key, secret in self._secrets.items():
            if secret.metadata.needs_rotation():
                if self.rotate_secret(key):
                    rotated_keys.append(key)
        
        return rotated_keys
    
    def backup_secrets(self, backup_path: Optional[Union[str, Path]] = None) -> str:
        """
        Sauvegarde tous les secrets.
        
        Args:
            backup_path: Chemin de sauvegarde
            
        Returns:
            Chemin du fichier de sauvegarde
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.secrets_dir / f"backup_{timestamp}.json"
        else:
            backup_path = Path(backup_path)
        
        backup_data = {
            "environment": self.environment,
            "backup_date": datetime.now().isoformat(),
            "secrets": [self._serialize_secret(secret) for secret in self._secrets.values()]
        }
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        # Audit
        self._log_audit("backup_secrets", backup_path=str(backup_path))
        
        logger.info(f"Sauvegarde créée: {backup_path}")
        return str(backup_path)
    
    def restore_secrets(self, backup_path: Union[str, Path]) -> bool:
        """
        Restaure les secrets depuis une sauvegarde.
        
        Args:
            backup_path: Chemin de la sauvegarde
            
        Returns:
            True si restauration réussie
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            logger.error(f"Fichier de sauvegarde non trouvé: {backup_path}")
            return False
        
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Vérification de l'environnement
            if backup_data.get("environment") != self.environment:
                logger.warning(f"Environnement de sauvegarde différent: {backup_data.get('environment')} vs {self.environment}")
            
            # Restauration des secrets
            with self._lock:
                self._secrets.clear()
                
                for secret_data in backup_data.get("secrets", []):
                    secret = self._deserialize_secret(secret_data)
                    self._secrets[secret.key] = secret
                
                self._save_secrets()
            
            # Audit
            self._log_audit("restore_secrets", backup_path=str(backup_path), count=len(backup_data.get("secrets", [])))
            
            logger.info(f"Secrets restaurés depuis: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la restauration: {e}")
            return False
    
    def _log_audit(self, action: str, key: str = None, **kwargs) -> None:
        """Enregistre un événement d'audit."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "action": action,
            "key": key,
            **kwargs
        }
        
        self._audit_log.append(audit_entry)
        
        # Limitation de la taille du log d'audit en mémoire
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-500:]  # Garde les 500 derniers
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Récupère le log d'audit.
        
        Args:
            limit: Nombre maximum d'entrées
            
        Returns:
            Liste des entrées d'audit
        """
        return self._audit_log[-limit:]
    
    @contextmanager
    def temporary_secret(self, key: str, value: str):
        """
        Crée un secret temporaire.
        
        Args:
            key: Clé du secret temporaire
            value: Valeur du secret temporaire
        """
        # Sauvegarde de l'ancien secret s'il existe
        old_secret = self._secrets.get(key)
        
        # Création du secret temporaire
        encrypted_value = self._encryption_manager.encrypt(value)
        temp_metadata = SecretMetadata(
            created_at=datetime.now(),
            updated_at=datetime.now(),
            description="Temporary secret"
        )
        
        temp_secret = Secret(
            key=key,
            value=encrypted_value,
            secret_type=SecretType.API_KEY,
            scope=SecretScope.ENVIRONMENT,
            environment=self.environment,
            metadata=temp_metadata,
            encrypted=True
        )
        
        self._secrets[key] = temp_secret
        
        try:
            yield
        finally:
            # Restauration de l'ancien secret ou suppression
            if old_secret:
                self._secrets[key] = old_secret
            else:
                del self._secrets[key]


def get_secrets_manager(environment: str, **kwargs) -> SecretsManager:
    """
    Factory function pour obtenir un gestionnaire de secrets.
    
    Args:
        environment: Environnement cible
        **kwargs: Arguments additionnels
        
    Returns:
        Instance de SecretsManager
    """
    return SecretsManager(environment, **kwargs)


__all__ = [
    "SecretType",
    "SecretScope",
    "SecretMetadata",
    "Secret",
    "EncryptionManager",
    "SecretGenerator",
    "SecretsManager",
    "get_secrets_manager"
]
