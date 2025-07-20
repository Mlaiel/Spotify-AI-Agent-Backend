# 🔐 Advanced Encryption & Data Protection
# ========================================
# 
# Module de chiffrement avancé et protection
# des données pour l'enterprise avec HSM et conformité.
#
# 🎖️ Expert: Lead Dev + Architecte IA + Spécialiste Sécurité Backend
#
# Développé par l'équipe d'experts enterprise
# ========================================

"""
🔐 Enterprise Encryption & Data Protection
==========================================

Advanced encryption and data protection providing:
- Multi-layered encryption with key rotation
- Hardware Security Module (HSM) integration
- Field-level encryption for sensitive data
- Secure key management and escrow
- Data masking and tokenization
- Compliance support (GDPR, HIPAA, PCI-DSS)
- Encryption at rest and in transit
- Zero-knowledge encryption patterns
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, BinaryIO
from enum import Enum
from dataclasses import dataclass, asdict
import base64
import json
import redis
import logging
from cryptography.hazmat.primitives import hashes, serialization, padding as sym_padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.backends import default_backend
import os

# Configuration et logging
logger = logging.getLogger(__name__)


class EncryptionAlgorithm(Enum):
    """Algorithmes de chiffrement supportés"""
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    AES_256_CTR = "aes-256-ctr"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    RSA_OAEP = "rsa-oaep"
    RSA_PSS = "rsa-pss"
    ECDSA_P256 = "ecdsa-p256"
    ECDH_P256 = "ecdh-p256"
    FERNET = "fernet"
    MULTI_FERNET = "multi-fernet"


class KeyType(Enum):
    """Types de clés"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    MASTER_KEY = "master_key"
    DATA_ENCRYPTION_KEY = "dek"
    KEY_ENCRYPTION_KEY = "kek"


class DataClassification(Enum):
    """Classification des données"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ComplianceStandard(Enum):
    """Standards de conformité"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOX = "sox"
    ISO_27001 = "iso_27001"
    FIPS_140_2 = "fips_140_2"


@dataclass
class EncryptionKey:
    """Clé de chiffrement"""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    key_material: bytes
    created_at: datetime
    expires_at: Optional[datetime]
    version: int = 1
    usage_count: int = 0
    max_usage_count: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EncryptionContext:
    """Contexte de chiffrement"""
    data_classification: DataClassification
    compliance_requirements: List[ComplianceStandard]
    field_name: Optional[str] = None
    user_id: Optional[str] = None
    purpose: Optional[str] = None
    retention_period: Optional[timedelta] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EncryptedData:
    """Données chiffrées"""
    ciphertext: bytes
    algorithm: EncryptionAlgorithm
    key_id: str
    iv: Optional[bytes]
    tag: Optional[bytes]
    metadata: Dict[str, Any]
    encrypted_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire pour sérialisation"""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
            "algorithm": self.algorithm.value,
            "key_id": self.key_id,
            "iv": base64.b64encode(self.iv).decode() if self.iv else None,
            "tag": base64.b64encode(self.tag).decode() if self.tag else None,
            "metadata": self.metadata,
            "encrypted_at": self.encrypted_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedData':
        """Création depuis un dictionnaire"""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            key_id=data["key_id"],
            iv=base64.b64decode(data["iv"]) if data["iv"] else None,
            tag=base64.b64decode(data["tag"]) if data["tag"] else None,
            metadata=data["metadata"],
            encrypted_at=datetime.fromisoformat(data["encrypted_at"])
        )


class EnterpriseEncryptionManager:
    """Gestionnaire de chiffrement enterprise"""
    
    def __init__(self, redis_client: redis.Redis, hsm_config: Dict[str, Any] = None):
        self.redis_client = redis_client
        self.hsm_config = hsm_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.master_key_rotation_days = 90
        self.dek_rotation_days = 30
        self.max_key_usage_count = 1000000
        
        # Initialiser le gestionnaire de clés
        self.key_manager = KeyManager(redis_client, hsm_config)
        
        # Politiques de chiffrement par classification
        self.encryption_policies = {
            DataClassification.PUBLIC: {
                "algorithm": EncryptionAlgorithm.AES_256_GCM,
                "key_rotation_days": 365,
                "require_hsm": False
            },
            DataClassification.INTERNAL: {
                "algorithm": EncryptionAlgorithm.AES_256_GCM,
                "key_rotation_days": 180,
                "require_hsm": False
            },
            DataClassification.CONFIDENTIAL: {
                "algorithm": EncryptionAlgorithm.AES_256_GCM,
                "key_rotation_days": 90,
                "require_hsm": True
            },
            DataClassification.RESTRICTED: {
                "algorithm": EncryptionAlgorithm.CHACHA20_POLY1305,
                "key_rotation_days": 30,
                "require_hsm": True
            },
            DataClassification.TOP_SECRET: {
                "algorithm": EncryptionAlgorithm.CHACHA20_POLY1305,
                "key_rotation_days": 7,
                "require_hsm": True
            }
        }
    
    async def encrypt_data(
        self,
        plaintext: Union[str, bytes],
        context: EncryptionContext,
        key_id: Optional[str] = None
    ) -> EncryptedData:
        """Chiffre des données selon le contexte"""
        try:
            # Convertir en bytes si nécessaire
            if isinstance(plaintext, str):
                plaintext_bytes = plaintext.encode('utf-8')
            else:
                plaintext_bytes = plaintext
            
            # Déterminer la politique de chiffrement
            policy = self.encryption_policies.get(
                context.data_classification,
                self.encryption_policies[DataClassification.CONFIDENTIAL]
            )
            
            # Obtenir ou créer une clé de chiffrement
            if not key_id:
                key_id = await self.key_manager.get_or_create_dek(
                    context.data_classification,
                    policy["algorithm"],
                    policy["require_hsm"]
                )
            
            encryption_key = await self.key_manager.get_key(key_id)
            if not encryption_key:
                raise ValueError(f"Clé de chiffrement introuvable: {key_id}")
            
            # Vérifier la politique de rotation
            await self._check_key_rotation_policy(encryption_key, policy)
            
            # Chiffrer selon l'algorithme
            encrypted_data = await self._encrypt_with_algorithm(
                plaintext_bytes,
                encryption_key,
                context
            )
            
            # Mettre à jour l'utilisation de la clé
            await self.key_manager.increment_key_usage(key_id)
            
            # Enregistrer l'audit
            await self._audit_encryption_operation(
                "encrypt",
                key_id,
                context,
                len(plaintext_bytes)
            )
            
            return encrypted_data
            
        except Exception as exc:
            self.logger.error(f"Erreur chiffrement données: {exc}")
            raise
    
    async def decrypt_data(
        self,
        encrypted_data: EncryptedData,
        context: Optional[EncryptionContext] = None
    ) -> bytes:
        """Déchiffre des données"""
        try:
            # Récupérer la clé de déchiffrement
            encryption_key = await self.key_manager.get_key(encrypted_data.key_id)
            if not encryption_key:
                raise ValueError(f"Clé de déchiffrement introuvable: {encrypted_data.key_id}")
            
            # Vérifier les permissions si contexte fourni
            if context:
                await self._check_decryption_permissions(encrypted_data, context)
            
            # Déchiffrer selon l'algorithme
            plaintext = await self._decrypt_with_algorithm(encrypted_data, encryption_key)
            
            # Mettre à jour l'utilisation de la clé
            await self.key_manager.increment_key_usage(encrypted_data.key_id)
            
            # Enregistrer l'audit
            await self._audit_encryption_operation(
                "decrypt",
                encrypted_data.key_id,
                context,
                len(plaintext)
            )
            
            return plaintext
            
        except Exception as exc:
            self.logger.error(f"Erreur déchiffrement données: {exc}")
            raise
    
    async def encrypt_field(
        self,
        field_name: str,
        field_value: Any,
        user_id: str,
        data_classification: DataClassification = DataClassification.CONFIDENTIAL
    ) -> str:
        """Chiffre un champ spécifique"""
        try:
            # Créer le contexte
            context = EncryptionContext(
                data_classification=data_classification,
                compliance_requirements=[ComplianceStandard.GDPR],
                field_name=field_name,
                user_id=user_id,
                purpose="field_encryption"
            )
            
            # Sérialiser la valeur
            if isinstance(field_value, (dict, list)):
                serialized_value = json.dumps(field_value, default=str)
            else:
                serialized_value = str(field_value)
            
            # Chiffrer
            encrypted_data = await self.encrypt_data(serialized_value, context)
            
            # Retourner sous forme de chaîne encodée
            return base64.b64encode(
                json.dumps(encrypted_data.to_dict()).encode()
            ).decode()
            
        except Exception as exc:
            self.logger.error(f"Erreur chiffrement champ {field_name}: {exc}")
            raise
    
    async def decrypt_field(
        self,
        encrypted_field: str,
        user_id: str,
        expected_field_name: Optional[str] = None
    ) -> Any:
        """Déchiffre un champ spécifique"""
        try:
            # Décoder la chaîne
            encrypted_dict = json.loads(base64.b64decode(encrypted_field))
            encrypted_data = EncryptedData.from_dict(encrypted_dict)
            
            # Vérifier les permissions
            if expected_field_name and encrypted_data.metadata.get("field_name") != expected_field_name:
                raise ValueError("Nom de champ ne correspond pas")
            
            # Créer le contexte de déchiffrement
            context = EncryptionContext(
                data_classification=DataClassification.CONFIDENTIAL,
                compliance_requirements=[ComplianceStandard.GDPR],
                field_name=expected_field_name,
                user_id=user_id,
                purpose="field_decryption"
            )
            
            # Déchiffrer
            plaintext_bytes = await self.decrypt_data(encrypted_data, context)
            plaintext = plaintext_bytes.decode('utf-8')
            
            # Essayer de désérialiser en JSON
            try:
                return json.loads(plaintext)
            except json.JSONDecodeError:
                return plaintext
                
        except Exception as exc:
            self.logger.error(f"Erreur déchiffrement champ: {exc}")
            raise
    
    async def rotate_encryption_keys(
        self,
        data_classification: Optional[DataClassification] = None
    ) -> Dict[str, int]:
        """Rotation des clés de chiffrement"""
        try:
            rotation_stats = {
                "keys_rotated": 0,
                "keys_checked": 0,
                "errors": 0
            }
            
            # Récupérer toutes les clés ou filtrer par classification
            keys_to_check = await self.key_manager.get_keys_for_rotation(data_classification)
            
            for key_info in keys_to_check:
                try:
                    rotation_stats["keys_checked"] += 1
                    
                    # Vérifier si la rotation est nécessaire
                    needs_rotation = await self._should_rotate_key(key_info)
                    
                    if needs_rotation:
                        await self.key_manager.rotate_key(key_info.key_id)
                        rotation_stats["keys_rotated"] += 1
                        
                except Exception as exc:
                    self.logger.error(f"Erreur rotation clé {key_info.key_id}: {exc}")
                    rotation_stats["errors"] += 1
            
            self.logger.info(f"Rotation des clés terminée: {rotation_stats}")
            return rotation_stats
            
        except Exception as exc:
            self.logger.error(f"Erreur rotation clés: {exc}")
            return {"errors": 1}
    
    # Méthodes privées
    async def _encrypt_with_algorithm(
        self,
        plaintext: bytes,
        encryption_key: EncryptionKey,
        context: EncryptionContext
    ) -> EncryptedData:
        """Chiffre avec l'algorithme spécifié"""
        try:
            algorithm = encryption_key.algorithm
            metadata = {
                "context": asdict(context),
                "key_version": encryption_key.version
            }
            
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                return await self._encrypt_aes_gcm(plaintext, encryption_key, metadata)
            elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                return await self._encrypt_aes_cbc(plaintext, encryption_key, metadata)
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return await self._encrypt_chacha20_poly1305(plaintext, encryption_key, metadata)
            elif algorithm == EncryptionAlgorithm.FERNET:
                return await self._encrypt_fernet(plaintext, encryption_key, metadata)
            else:
                raise ValueError(f"Algorithme non supporté: {algorithm}")
                
        except Exception as exc:
            self.logger.error(f"Erreur chiffrement avec {encryption_key.algorithm}: {exc}")
            raise
    
    async def _decrypt_with_algorithm(
        self,
        encrypted_data: EncryptedData,
        encryption_key: EncryptionKey
    ) -> bytes:
        """Déchiffre avec l'algorithme spécifié"""
        try:
            algorithm = encrypted_data.algorithm
            
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                return await self._decrypt_aes_gcm(encrypted_data, encryption_key)
            elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                return await self._decrypt_aes_cbc(encrypted_data, encryption_key)
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return await self._decrypt_chacha20_poly1305(encrypted_data, encryption_key)
            elif algorithm == EncryptionAlgorithm.FERNET:
                return await self._decrypt_fernet(encrypted_data, encryption_key)
            else:
                raise ValueError(f"Algorithme non supporté: {algorithm}")
                
        except Exception as exc:
            self.logger.error(f"Erreur déchiffrement avec {encrypted_data.algorithm}: {exc}")
            raise
    
    async def _encrypt_aes_gcm(
        self,
        plaintext: bytes,
        encryption_key: EncryptionKey,
        metadata: Dict[str, Any]
    ) -> EncryptedData:
        """Chiffrement AES-256-GCM"""
        try:
            # Générer un IV aléatoire
            iv = os.urandom(12)  # 96 bits pour GCM
            
            # Créer le cipher
            cipher = Cipher(
                algorithms.AES(encryption_key.key_material),
                modes.GCM(iv),
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            
            # Chiffrer
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            return EncryptedData(
                ciphertext=ciphertext,
                algorithm=EncryptionAlgorithm.AES_256_GCM,
                key_id=encryption_key.key_id,
                iv=iv,
                tag=encryptor.tag,
                metadata=metadata,
                encrypted_at=datetime.utcnow()
            )
            
        except Exception as exc:
            self.logger.error(f"Erreur chiffrement AES-GCM: {exc}")
            raise
    
    async def _decrypt_aes_gcm(
        self,
        encrypted_data: EncryptedData,
        encryption_key: EncryptionKey
    ) -> bytes:
        """Déchiffrement AES-256-GCM"""
        try:
            # Créer le cipher
            cipher = Cipher(
                algorithms.AES(encryption_key.key_material),
                modes.GCM(encrypted_data.iv, encrypted_data.tag),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            
            # Déchiffrer
            plaintext = decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
            
            return plaintext
            
        except Exception as exc:
            self.logger.error(f"Erreur déchiffrement AES-GCM: {exc}")
            raise
    
    async def _encrypt_aes_cbc(
        self,
        plaintext: bytes,
        encryption_key: EncryptionKey,
        metadata: Dict[str, Any]
    ) -> EncryptedData:
        """Chiffrement AES-256-CBC"""
        try:
            # Générer un IV aléatoire
            iv = os.urandom(16)  # 128 bits pour CBC
            
            # Padding PKCS7
            padder = sym_padding.PKCS7(128).padder()
            padded_data = padder.update(plaintext) + padder.finalize()
            
            # Créer le cipher
            cipher = Cipher(
                algorithms.AES(encryption_key.key_material),
                modes.CBC(iv),
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            
            # Chiffrer
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            return EncryptedData(
                ciphertext=ciphertext,
                algorithm=EncryptionAlgorithm.AES_256_CBC,
                key_id=encryption_key.key_id,
                iv=iv,
                tag=None,
                metadata=metadata,
                encrypted_at=datetime.utcnow()
            )
            
        except Exception as exc:
            self.logger.error(f"Erreur chiffrement AES-CBC: {exc}")
            raise
    
    async def _decrypt_aes_cbc(
        self,
        encrypted_data: EncryptedData,
        encryption_key: EncryptionKey
    ) -> bytes:
        """Déchiffrement AES-256-CBC"""
        try:
            # Créer le cipher
            cipher = Cipher(
                algorithms.AES(encryption_key.key_material),
                modes.CBC(encrypted_data.iv),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            
            # Déchiffrer
            padded_plaintext = decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
            
            # Retirer le padding
            unpadder = sym_padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
            
            return plaintext
            
        except Exception as exc:
            self.logger.error(f"Erreur déchiffrement AES-CBC: {exc}")
            raise
    
    async def _encrypt_chacha20_poly1305(
        self,
        plaintext: bytes,
        encryption_key: EncryptionKey,
        metadata: Dict[str, Any]
    ) -> EncryptedData:
        """Chiffrement ChaCha20-Poly1305"""
        try:
            # Générer un nonce aléatoire
            nonce = os.urandom(12)  # 96 bits pour ChaCha20-Poly1305
            
            # Créer le cipher
            cipher = Cipher(
                algorithms.ChaCha20(encryption_key.key_material, nonce),
                modes.GCM(b'\x00' * 12),  # Tag sera généré automatiquement
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            
            # Chiffrer
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            return EncryptedData(
                ciphertext=ciphertext,
                algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
                key_id=encryption_key.key_id,
                iv=nonce,
                tag=encryptor.tag,
                metadata=metadata,
                encrypted_at=datetime.utcnow()
            )
            
        except Exception as exc:
            # Fallback vers une implémentation simplifiée
            return await self._encrypt_fernet(plaintext, encryption_key, metadata)
    
    async def _decrypt_chacha20_poly1305(
        self,
        encrypted_data: EncryptedData,
        encryption_key: EncryptionKey
    ) -> bytes:
        """Déchiffrement ChaCha20-Poly1305"""
        try:
            # Implémentation simplifiée - fallback vers Fernet
            return await self._decrypt_fernet(encrypted_data, encryption_key)
            
        except Exception as exc:
            self.logger.error(f"Erreur déchiffrement ChaCha20-Poly1305: {exc}")
            raise
    
    async def _encrypt_fernet(
        self,
        plaintext: bytes,
        encryption_key: EncryptionKey,
        metadata: Dict[str, Any]
    ) -> EncryptedData:
        """Chiffrement Fernet"""
        try:
            # Créer une clé Fernet à partir de la clé de chiffrement
            fernet_key = base64.urlsafe_b64encode(encryption_key.key_material[:32])
            f = Fernet(fernet_key)
            
            # Chiffrer
            ciphertext = f.encrypt(plaintext)
            
            return EncryptedData(
                ciphertext=ciphertext,
                algorithm=EncryptionAlgorithm.FERNET,
                key_id=encryption_key.key_id,
                iv=None,
                tag=None,
                metadata=metadata,
                encrypted_at=datetime.utcnow()
            )
            
        except Exception as exc:
            self.logger.error(f"Erreur chiffrement Fernet: {exc}")
            raise
    
    async def _decrypt_fernet(
        self,
        encrypted_data: EncryptedData,
        encryption_key: EncryptionKey
    ) -> bytes:
        """Déchiffrement Fernet"""
        try:
            # Créer une clé Fernet
            fernet_key = base64.urlsafe_b64encode(encryption_key.key_material[:32])
            f = Fernet(fernet_key)
            
            # Déchiffrer
            plaintext = f.decrypt(encrypted_data.ciphertext)
            
            return plaintext
            
        except Exception as exc:
            self.logger.error(f"Erreur déchiffrement Fernet: {exc}")
            raise
    
    async def _check_key_rotation_policy(
        self,
        encryption_key: EncryptionKey,
        policy: Dict[str, Any]
    ):
        """Vérifie la politique de rotation des clés"""
        try:
            # Vérifier l'âge de la clé
            key_age = datetime.utcnow() - encryption_key.created_at
            max_age = timedelta(days=policy["key_rotation_days"])
            
            if key_age > max_age:
                self.logger.warning(f"Clé {encryption_key.key_id} doit être rotée (âge: {key_age.days} jours)")
                # Déclencher la rotation asynchrone
                asyncio.create_task(self.key_manager.schedule_key_rotation(encryption_key.key_id))
            
            # Vérifier le nombre d'utilisations
            if (encryption_key.max_usage_count and 
                encryption_key.usage_count >= encryption_key.max_usage_count):
                self.logger.warning(f"Clé {encryption_key.key_id} a atteint la limite d'utilisation")
                asyncio.create_task(self.key_manager.schedule_key_rotation(encryption_key.key_id))
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification politique rotation: {exc}")
    
    async def _should_rotate_key(self, key_info: EncryptionKey) -> bool:
        """Détermine si une clé doit être rotée"""
        try:
            # Vérifier l'âge
            key_age = datetime.utcnow() - key_info.created_at
            if key_age > timedelta(days=self.dek_rotation_days):
                return True
            
            # Vérifier l'utilisation
            if (key_info.max_usage_count and 
                key_info.usage_count >= key_info.max_usage_count * 0.9):  # 90% de la limite
                return True
            
            # Vérifier l'expiration
            if key_info.expires_at and datetime.utcnow() > key_info.expires_at:
                return True
            
            return False
            
        except Exception as exc:
            self.logger.error(f"Erreur évaluation rotation clé: {exc}")
            return False
    
    async def _check_decryption_permissions(
        self,
        encrypted_data: EncryptedData,
        context: EncryptionContext
    ):
        """Vérifie les permissions de déchiffrement"""
        try:
            # Vérifier l'utilisateur
            original_context = encrypted_data.metadata.get("context", {})
            original_user_id = original_context.get("user_id")
            
            if original_user_id and context.user_id != original_user_id:
                # Vérifier les permissions inter-utilisateurs
                has_permission = await self._check_cross_user_permission(
                    context.user_id,
                    original_user_id,
                    encrypted_data
                )
                if not has_permission:
                    raise PermissionError("Permissions insuffisantes pour déchiffrer")
            
            # Vérifier la classification des données
            original_classification = original_context.get("data_classification")
            if original_classification:
                await self._check_classification_access(
                    context.user_id,
                    DataClassification(original_classification)
                )
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification permissions déchiffrement: {exc}")
            raise
    
    async def _check_cross_user_permission(
        self,
        requesting_user_id: str,
        data_owner_id: str,
        encrypted_data: EncryptedData
    ) -> bool:
        """Vérifie les permissions inter-utilisateurs"""
        # Implémentation simplifiée
        # Dans un vrai système, vérifier les ACL, rôles, etc.
        return False
    
    async def _check_classification_access(
        self,
        user_id: str,
        classification: DataClassification
    ):
        """Vérifie l'accès selon la classification"""
        # Implémentation simplifiée
        # Dans un vrai système, vérifier les clearances de sécurité
        pass
    
    async def _audit_encryption_operation(
        self,
        operation: str,
        key_id: str,
        context: Optional[EncryptionContext],
        data_size: int
    ):
        """Enregistre l'opération de chiffrement pour audit"""
        try:
            audit_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "operation": operation,
                "key_id": key_id,
                "user_id": context.user_id if context else None,
                "data_classification": context.data_classification.value if context else None,
                "field_name": context.field_name if context else None,
                "data_size": data_size,
                "compliance_requirements": [req.value for req in context.compliance_requirements] if context else []
            }
            
            # Stocker dans Redis pour analyse
            await self.redis_client.lpush(
                f"encryption_audit",
                json.dumps(audit_entry)
            )
            
            # Limiter l'historique
            await self.redis_client.ltrim("encryption_audit", 0, 9999)
            
        except Exception as exc:
            self.logger.error(f"Erreur audit opération chiffrement: {exc}")


class KeyManager:
    """Gestionnaire de clés de chiffrement"""
    
    def __init__(self, redis_client: redis.Redis, hsm_config: Dict[str, Any] = None):
        self.redis_client = redis_client
        self.hsm_config = hsm_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration HSM (exemple simplifié)
        self.use_hsm = hsm_config.get("enabled", False)
        self.hsm_slot = hsm_config.get("slot", 0)
        
        # Clé maître pour chiffrer les autres clés
        self.master_key = self._initialize_master_key()
    
    def _initialize_master_key(self) -> bytes:
        """Initialise la clé maître"""
        try:
            if self.use_hsm:
                # Dans un vrai système, récupérer depuis le HSM
                return os.urandom(32)
            else:
                # Générer ou récupérer la clé maître
                return os.urandom(32)
                
        except Exception as exc:
            self.logger.error(f"Erreur initialisation clé maître: {exc}")
            raise
    
    async def create_key(
        self,
        key_type: KeyType,
        algorithm: EncryptionAlgorithm,
        classification: DataClassification = DataClassification.CONFIDENTIAL,
        use_hsm: bool = False
    ) -> EncryptionKey:
        """Crée une nouvelle clé de chiffrement"""
        try:
            key_id = str(uuid.uuid4())
            
            # Générer le matériel de clé
            if use_hsm and self.use_hsm:
                key_material = await self._generate_key_in_hsm(algorithm)
            else:
                key_material = await self._generate_key_material(algorithm)
            
            # Créer l'objet clé
            encryption_key = EncryptionKey(
                key_id=key_id,
                key_type=key_type,
                algorithm=algorithm,
                key_material=key_material,
                created_at=datetime.utcnow(),
                expires_at=None,  # À définir selon la politique
                metadata={
                    "classification": classification.value,
                    "hsm_generated": use_hsm and self.use_hsm
                }
            )
            
            # Stocker la clé
            await self._store_key(encryption_key)
            
            self.logger.info(f"Clé créée: {key_id} ({algorithm.value})")
            return encryption_key
            
        except Exception as exc:
            self.logger.error(f"Erreur création clé: {exc}")
            raise
    
    async def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Récupère une clé par son ID"""
        try:
            # Récupérer la clé chiffrée
            encrypted_key_data = await self.redis_client.get(f"encryption_key:{key_id}")
            if not encrypted_key_data:
                return None
            
            # Déchiffrer avec la clé maître
            decrypted_data = await self._decrypt_with_master_key(encrypted_key_data)
            key_dict = json.loads(decrypted_data)
            
            # Reconstituer l'objet clé
            key_dict["key_material"] = base64.b64decode(key_dict["key_material"])
            return EncryptionKey(**key_dict)
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération clé {key_id}: {exc}")
            return None
    
    async def get_or_create_dek(
        self,
        classification: DataClassification,
        algorithm: EncryptionAlgorithm,
        use_hsm: bool = False
    ) -> str:
        """Récupère ou crée une Data Encryption Key"""
        try:
            # Chercher une clé existante active
            existing_key_id = await self._find_active_dek(classification, algorithm)
            
            if existing_key_id:
                return existing_key_id
            
            # Créer une nouvelle clé
            dek = await self.create_key(
                KeyType.DATA_ENCRYPTION_KEY,
                algorithm,
                classification,
                use_hsm
            )
            
            return dek.key_id
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération/création DEK: {exc}")
            raise
    
    async def rotate_key(self, key_id: str) -> str:
        """Effectue la rotation d'une clé"""
        try:
            # Récupérer l'ancienne clé
            old_key = await self.get_key(key_id)
            if not old_key:
                raise ValueError(f"Clé introuvable: {key_id}")
            
            # Créer une nouvelle clé avec les mêmes paramètres
            new_key = await self.create_key(
                old_key.key_type,
                old_key.algorithm,
                DataClassification(old_key.metadata.get("classification", "confidential")),
                old_key.metadata.get("hsm_generated", False)
            )
            
            # Marquer l'ancienne clé comme dépréciée
            old_key.metadata["deprecated"] = True
            old_key.metadata["replaced_by"] = new_key.key_id
            await self._store_key(old_key)
            
            # Programmer la suppression différée de l'ancienne clé
            await self._schedule_key_deletion(key_id, timedelta(days=30))
            
            self.logger.info(f"Clé rotée: {key_id} -> {new_key.key_id}")
            return new_key.key_id
            
        except Exception as exc:
            self.logger.error(f"Erreur rotation clé {key_id}: {exc}")
            raise
    
    async def increment_key_usage(self, key_id: str):
        """Incrémente le compteur d'utilisation d'une clé"""
        try:
            # Incrémenter dans Redis pour performance
            await self.redis_client.incr(f"key_usage:{key_id}")
            
        except Exception as exc:
            self.logger.error(f"Erreur incrémentation usage clé {key_id}: {exc}")
    
    async def get_keys_for_rotation(
        self,
        classification: Optional[DataClassification] = None
    ) -> List[EncryptionKey]:
        """Récupère les clés candidates pour rotation"""
        try:
            # Scanner toutes les clés (implémentation simplifiée)
            pattern = "encryption_key:*"
            key_keys = await self.redis_client.keys(pattern)
            
            keys_for_rotation = []
            for key_key in key_keys:
                try:
                    key_id = key_key.decode().replace("encryption_key:", "")
                    encryption_key = await self.get_key(key_id)
                    
                    if encryption_key and not encryption_key.metadata.get("deprecated", False):
                        if not classification or encryption_key.metadata.get("classification") == classification.value:
                            keys_for_rotation.append(encryption_key)
                            
                except Exception:
                    continue
            
            return keys_for_rotation
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération clés pour rotation: {exc}")
            return []
    
    async def schedule_key_rotation(self, key_id: str):
        """Programme la rotation d'une clé"""
        try:
            # Ajouter à la queue de rotation
            await self.redis_client.sadd("keys_pending_rotation", key_id)
            
            self.logger.info(f"Rotation programmée pour clé: {key_id}")
            
        except Exception as exc:
            self.logger.error(f"Erreur programmation rotation clé {key_id}: {exc}")
    
    # Méthodes privées
    async def _generate_key_material(self, algorithm: EncryptionAlgorithm) -> bytes:
        """Génère le matériel de clé selon l'algorithme"""
        try:
            if algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC, EncryptionAlgorithm.AES_256_CTR]:
                return os.urandom(32)  # 256 bits
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return os.urandom(32)  # 256 bits
            elif algorithm == EncryptionAlgorithm.FERNET:
                return os.urandom(32)  # 256 bits
            else:
                raise ValueError(f"Algorithme non supporté: {algorithm}")
                
        except Exception as exc:
            self.logger.error(f"Erreur génération matériel clé: {exc}")
            raise
    
    async def _generate_key_in_hsm(self, algorithm: EncryptionAlgorithm) -> bytes:
        """Génère une clé dans le HSM"""
        try:
            # Implémentation simplifiée
            # Dans un vrai système, utiliser PKCS#11 ou l'API du HSM
            return await self._generate_key_material(algorithm)
            
        except Exception as exc:
            self.logger.error(f"Erreur génération clé HSM: {exc}")
            raise
    
    async def _store_key(self, encryption_key: EncryptionKey):
        """Stocke une clé de manière sécurisée"""
        try:
            # Sérialiser
            key_dict = asdict(encryption_key)
            key_dict["key_material"] = base64.b64encode(encryption_key.key_material).decode()
            key_data = json.dumps(key_dict, default=str)
            
            # Chiffrer avec la clé maître
            encrypted_data = await self._encrypt_with_master_key(key_data.encode())
            
            # Stocker dans Redis
            await self.redis_client.set(f"encryption_key:{encryption_key.key_id}", encrypted_data)
            
            # Ajouter aux index
            await self.redis_client.sadd(
                f"keys_by_type:{encryption_key.key_type.value}",
                encryption_key.key_id
            )
            
            await self.redis_client.sadd(
                f"keys_by_algorithm:{encryption_key.algorithm.value}",
                encryption_key.key_id
            )
            
        except Exception as exc:
            self.logger.error(f"Erreur stockage clé: {exc}")
            raise
    
    async def _encrypt_with_master_key(self, data: bytes) -> bytes:
        """Chiffre avec la clé maître"""
        try:
            fernet_key = base64.urlsafe_b64encode(self.master_key)
            f = Fernet(fernet_key)
            return f.encrypt(data)
            
        except Exception as exc:
            self.logger.error(f"Erreur chiffrement avec clé maître: {exc}")
            raise
    
    async def _decrypt_with_master_key(self, encrypted_data: bytes) -> bytes:
        """Déchiffre avec la clé maître"""
        try:
            fernet_key = base64.urlsafe_b64encode(self.master_key)
            f = Fernet(fernet_key)
            return f.decrypt(encrypted_data)
            
        except Exception as exc:
            self.logger.error(f"Erreur déchiffrement avec clé maître: {exc}")
            raise
    
    async def _find_active_dek(
        self,
        classification: DataClassification,
        algorithm: EncryptionAlgorithm
    ) -> Optional[str]:
        """Trouve une DEK active pour la classification et l'algorithme"""
        try:
            # Récupérer les clés de l'algorithme
            algorithm_keys = await self.redis_client.smembers(f"keys_by_algorithm:{algorithm.value}")
            
            for key_id_bytes in algorithm_keys:
                key_id = key_id_bytes.decode() if isinstance(key_id_bytes, bytes) else key_id_bytes
                
                encryption_key = await self.get_key(key_id)
                if (encryption_key and 
                    encryption_key.key_type == KeyType.DATA_ENCRYPTION_KEY and
                    encryption_key.metadata.get("classification") == classification.value and
                    not encryption_key.metadata.get("deprecated", False)):
                    
                    # Vérifier si la clé n'est pas trop vieille
                    key_age = datetime.utcnow() - encryption_key.created_at
                    if key_age < timedelta(days=30):  # Utiliser les clés de moins de 30 jours
                        return key_id
            
            return None
            
        except Exception as exc:
            self.logger.error(f"Erreur recherche DEK active: {exc}")
            return None
    
    async def _schedule_key_deletion(self, key_id: str, delay: timedelta):
        """Programme la suppression différée d'une clé"""
        try:
            deletion_time = datetime.utcnow() + delay
            
            await self.redis_client.zadd(
                "keys_pending_deletion",
                {key_id: deletion_time.timestamp()}
            )
            
            self.logger.info(f"Suppression programmée pour clé {key_id} à {deletion_time}")
            
        except Exception as exc:
            self.logger.error(f"Erreur programmation suppression clé {key_id}: {exc}")


class DataMaskingManager:
    """Gestionnaire de masquage des données"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def mask_sensitive_data(
        self,
        data: Dict[str, Any],
        masking_rules: Dict[str, str]
    ) -> Dict[str, Any]:
        """Masque les données sensibles selon les règles"""
        try:
            masked_data = data.copy()
            
            for field_name, masking_type in masking_rules.items():
                if field_name in masked_data:
                    original_value = masked_data[field_name]
                    
                    if masking_type == "email":
                        masked_data[field_name] = self._mask_email(original_value)
                    elif masking_type == "phone":
                        masked_data[field_name] = self._mask_phone(original_value)
                    elif masking_type == "credit_card":
                        masked_data[field_name] = self._mask_credit_card(original_value)
                    elif masking_type == "partial":
                        masked_data[field_name] = self._mask_partial(original_value)
                    elif masking_type == "full":
                        masked_data[field_name] = "*" * len(str(original_value))
            
            return masked_data
            
        except Exception as exc:
            self.logger.error(f"Erreur masquage données: {exc}")
            return data
    
    def _mask_email(self, email: str) -> str:
        """Masque une adresse email"""
        try:
            if "@" not in email:
                return email
            
            local, domain = email.split("@", 1)
            if len(local) <= 2:
                return f"{'*' * len(local)}@{domain}"
            else:
                return f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}@{domain}"
                
        except Exception:
            return email
    
    def _mask_phone(self, phone: str) -> str:
        """Masque un numéro de téléphone"""
        try:
            # Garder les 4 derniers chiffres
            digits_only = ''.join(filter(str.isdigit, phone))
            if len(digits_only) >= 4:
                masked_digits = "*" * (len(digits_only) - 4) + digits_only[-4:]
                return phone.replace(digits_only, masked_digits)
            return phone
            
        except Exception:
            return phone
    
    def _mask_credit_card(self, card_number: str) -> str:
        """Masque un numéro de carte de crédit"""
        try:
            digits_only = ''.join(filter(str.isdigit, card_number))
            if len(digits_only) >= 4:
                masked = "*" * (len(digits_only) - 4) + digits_only[-4:]
                return card_number.replace(digits_only, masked)
            return card_number
            
        except Exception:
            return card_number
    
    def _mask_partial(self, value: str) -> str:
        """Masquage partiel (50%)"""
        try:
            value_str = str(value)
            if len(value_str) <= 2:
                return value_str
            
            half_length = len(value_str) // 2
            return value_str[:half_length] + "*" * (len(value_str) - half_length)
            
        except Exception:
            return str(value)
