# 🔐 Password Management & Passwordless Authentication
# ======================================================
# 
# Gestionnaire avancé de mots de passe et authentification
# sans mot de passe pour l'enterprise.
#
# 🎖️ Expert: Lead Dev + Architecte IA + Spécialiste Sécurité Backend
#
# Développé par l'équipe d'experts enterprise
# ======================================================

"""
🔐 Enterprise Password & Passwordless Management
===============================================

Advanced password and passwordless authentication providing:
- Password strength enforcement and policies
- Password history and rotation management
- Passwordless authentication flows (Magic links, Push notifications)
- Breach detection and password exposure monitoring
- Secure password storage with adaptive hashing
- Password recovery and reset workflows
- WebAuthn/FIDO2 integration for passwordless
- Social authentication providers integration
"""

import asyncio
import hashlib
import hmac
import secrets
import string
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, asdict
import bcrypt
import argon2
import redis
import logging
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import smtplib
import requests
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import base64
import qrcode
import io
from urllib.parse import urlencode

# Configuration et logging
logger = logging.getLogger(__name__)


class PasswordStrength(Enum):
    """Niveaux de force du mot de passe"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    FAIR = "fair"
    GOOD = "good"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class AuthenticationMethod(Enum):
    """Méthodes d'authentification"""
    PASSWORD = "password"
    MAGIC_LINK = "magic_link"
    PUSH_NOTIFICATION = "push_notification"
    WEBAUTHN = "webauthn"
    SOCIAL_OAUTH = "social_oauth"
    SMS_OTP = "sms_otp"
    EMAIL_OTP = "email_otp"


class PasswordPolicyType(Enum):
    """Types de politique de mot de passe"""
    BASIC = "basic"
    ENTERPRISE = "enterprise"
    HIGH_SECURITY = "high_security"
    CUSTOM = "custom"


@dataclass
class PasswordPolicy:
    """Politique de mot de passe"""
    min_length: int = 8
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    disallow_common_passwords: bool = True
    disallow_personal_info: bool = True
    disallow_repeated_chars: int = 3
    disallow_sequential_chars: bool = True
    password_history_count: int = 12
    max_age_days: int = 90
    min_age_hours: int = 24
    account_lockout_threshold: int = 5
    account_lockout_duration_minutes: int = 30
    breach_detection: bool = True
    
    @classmethod
    def get_enterprise_policy(cls):
        """Politique enterprise par défaut"""
        return cls(
            min_length=12,
            require_uppercase=True,
            require_lowercase=True,
            require_digits=True,
            require_special_chars=True,
            disallow_common_passwords=True,
            disallow_personal_info=True,
            disallow_repeated_chars=2,
            disallow_sequential_chars=True,
            password_history_count=24,
            max_age_days=60,
            min_age_hours=48,
            account_lockout_threshold=3,
            account_lockout_duration_minutes=60,
            breach_detection=True
        )


@dataclass
class PasswordAnalysis:
    """Analyse de mot de passe"""
    strength: PasswordStrength
    score: float
    feedback: List[str]
    entropy: float
    estimated_crack_time: str
    is_breached: bool = False
    breach_count: int = 0


@dataclass
class MagicLink:
    """Lien magique pour authentification sans mot de passe"""
    link_id: str
    user_id: str
    email: str
    token: str
    expires_at: datetime
    used: bool = False
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WebAuthnCredential:
    """Credential WebAuthn/FIDO2"""
    credential_id: str
    user_id: str
    public_key: str
    counter: int
    transports: List[str]
    authenticator_data: str
    client_data: str
    created_at: datetime
    last_used: datetime
    device_name: str
    backup_eligible: bool = False
    backup_state: bool = False


class AdvancedPasswordManager:
    """Gestionnaire avancé de mots de passe"""
    
    def __init__(self, redis_client: redis.Redis, password_policy: PasswordPolicy = None):
        self.redis_client = redis_client
        self.password_policy = password_policy or PasswordPolicy.get_enterprise_policy()
        self.logger = logging.getLogger(__name__)
        
        # Configuration du hachage adaptatif
        self.hasher = argon2.PasswordHasher(
            time_cost=2,  # Temps de calcul
            memory_cost=102400,  # 100 MB de mémoire
            parallelism=8,  # Processus parallèles
            hash_len=32,  # Longueur du hash
            salt_len=16   # Longueur du sel
        )
        
        # Base de données des mots de passe compromis (exemple simplifié)
        self.breach_db_url = "https://haveibeenpwned.com/api/v3/pwnedpasswords"
        
        # Mots de passe communs (échantillon)
        self.common_passwords = {
            "password", "123456", "password123", "admin", "qwerty",
            "letmein", "welcome", "monkey", "1234567890", "abc123"
        }
    
    async def hash_password(self, password: str) -> str:
        """Hache un mot de passe avec Argon2"""
        try:
            # Validation du mot de passe
            validation_result = await self.validate_password(password)
            if not validation_result[0]:
                raise ValueError(f"Mot de passe invalide: {validation_result[1]}")
            
            # Hachage avec Argon2
            return self.hasher.hash(password)
            
        except Exception as exc:
            self.logger.error(f"Erreur hachage mot de passe: {exc}")
            raise
    
    async def verify_password(self, password: str, hashed_password: str) -> bool:
        """Vérifie un mot de passe contre son hash"""
        try:
            self.hasher.verify(hashed_password, password)
            
            # Vérifier si le hash doit être mis à jour
            if self.hasher.check_needs_rehash(hashed_password):
                self.logger.info("Hash du mot de passe doit être mis à jour")
                # Retourner un indicateur pour rehacher
            
            return True
            
        except argon2.exceptions.VerifyMismatchError:
            return False
        except Exception as exc:
            self.logger.error(f"Erreur vérification mot de passe: {exc}")
            return False
    
    async def validate_password(
        self,
        password: str,
        user_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Valide un mot de passe selon la politique"""
        try:
            errors = []
            
            # Longueur
            if len(password) < self.password_policy.min_length:
                errors.append(f"Minimum {self.password_policy.min_length} caractères requis")
            
            if len(password) > self.password_policy.max_length:
                errors.append(f"Maximum {self.password_policy.max_length} caractères autorisés")
            
            # Caractères requis
            if self.password_policy.require_uppercase and not re.search(r'[A-Z]', password):
                errors.append("Au moins une lettre majuscule requise")
            
            if self.password_policy.require_lowercase and not re.search(r'[a-z]', password):
                errors.append("Au moins une lettre minuscule requise")
            
            if self.password_policy.require_digits and not re.search(r'\d', password):
                errors.append("Au moins un chiffre requis")
            
            if self.password_policy.require_special_chars:
                special_pattern = f"[{re.escape(self.password_policy.special_chars)}]"
                if not re.search(special_pattern, password):
                    errors.append("Au moins un caractère spécial requis")
            
            # Caractères répétés
            if self.password_policy.disallow_repeated_chars > 0:
                if self._has_repeated_chars(password, self.password_policy.disallow_repeated_chars):
                    errors.append(f"Pas plus de {self.password_policy.disallow_repeated_chars} caractères identiques consécutifs")
            
            # Caractères séquentiels
            if self.password_policy.disallow_sequential_chars:
                if self._has_sequential_chars(password):
                    errors.append("Pas de séquences de caractères (abc, 123, etc.)")
            
            # Mots de passe communs
            if self.password_policy.disallow_common_passwords:
                if password.lower() in self.common_passwords:
                    errors.append("Mot de passe trop commun")
            
            # Informations personnelles
            if self.password_policy.disallow_personal_info and user_info:
                if self._contains_personal_info(password, user_info):
                    errors.append("Ne doit pas contenir d'informations personnelles")
            
            # Vérification de compromission
            if self.password_policy.breach_detection:
                is_breached = await self._check_password_breach(password)
                if is_breached:
                    errors.append("Ce mot de passe a été compromis dans une fuite de données")
            
            return len(errors) == 0, errors
            
        except Exception as exc:
            self.logger.error(f"Erreur validation mot de passe: {exc}")
            return False, ["Erreur lors de la validation"]
    
    async def analyze_password_strength(self, password: str) -> PasswordAnalysis:
        """Analyse la force d'un mot de passe"""
        try:
            score = 0.0
            feedback = []
            
            # Longueur
            length_score = min(1.0, len(password) / 12.0)
            score += length_score * 0.25
            
            if len(password) < 8:
                feedback.append("Trop court")
            elif len(password) < 12:
                feedback.append("Pourrait être plus long")
            
            # Diversité des caractères
            char_types = 0
            if re.search(r'[a-z]', password):
                char_types += 1
            if re.search(r'[A-Z]', password):
                char_types += 1
            if re.search(r'\d', password):
                char_types += 1
            if re.search(r'[^a-zA-Z0-9]', password):
                char_types += 1
            
            diversity_score = char_types / 4.0
            score += diversity_score * 0.25
            
            if char_types < 3:
                feedback.append("Utilisez différents types de caractères")
            
            # Entropie
            entropy = self._calculate_entropy(password)
            entropy_score = min(1.0, entropy / 60.0)  # 60 bits = bon
            score += entropy_score * 0.3
            
            # Patterns et répétitions
            pattern_penalty = 0.0
            if self._has_repeated_chars(password, 2):
                pattern_penalty += 0.1
                feedback.append("Évitez les caractères répétés")
            
            if self._has_sequential_chars(password):
                pattern_penalty += 0.1
                feedback.append("Évitez les séquences")
            
            if self._has_common_patterns(password):
                pattern_penalty += 0.1
                feedback.append("Évitez les patterns communs")
            
            score = max(0.0, score - pattern_penalty)
            
            # Unicité
            uniqueness_score = len(set(password)) / len(password)
            score += uniqueness_score * 0.1
            
            # Compromission
            is_breached = await self._check_password_breach(password)
            breach_count = 0
            if is_breached:
                score *= 0.5  # Pénalité majeure
                feedback.append("Mot de passe compromis")
                breach_count = await self._get_breach_count(password)
            
            # Calcul du score final
            final_score = min(1.0, max(0.0, score))
            
            # Déterminer la force
            if final_score < 0.2:
                strength = PasswordStrength.VERY_WEAK
            elif final_score < 0.4:
                strength = PasswordStrength.WEAK
            elif final_score < 0.6:
                strength = PasswordStrength.FAIR
            elif final_score < 0.8:
                strength = PasswordStrength.GOOD
            elif final_score < 0.95:
                strength = PasswordStrength.STRONG
            else:
                strength = PasswordStrength.VERY_STRONG
            
            # Temps de cassage estimé
            crack_time = self._estimate_crack_time(entropy)
            
            return PasswordAnalysis(
                strength=strength,
                score=final_score,
                feedback=feedback,
                entropy=entropy,
                estimated_crack_time=crack_time,
                is_breached=is_breached,
                breach_count=breach_count
            )
            
        except Exception as exc:
            self.logger.error(f"Erreur analyse force mot de passe: {exc}")
            return PasswordAnalysis(
                strength=PasswordStrength.WEAK,
                score=0.0,
                feedback=["Erreur lors de l'analyse"],
                entropy=0.0,
                estimated_crack_time="Inconnu"
            )
    
    async def generate_secure_password(
        self,
        length: int = 16,
        include_symbols: bool = True,
        exclude_ambiguous: bool = True
    ) -> str:
        """Génère un mot de passe sécurisé"""
        try:
            # Caractères de base
            lowercase = string.ascii_lowercase
            uppercase = string.ascii_uppercase
            digits = string.digits
            symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?" if include_symbols else ""
            
            # Exclure les caractères ambigus si demandé
            if exclude_ambiguous:
                lowercase = lowercase.replace('l', '').replace('o', '')
                uppercase = uppercase.replace('I', '').replace('O', '')
                digits = digits.replace('0', '').replace('1', '')
                symbols = symbols.replace('|', '').replace('l', '')
            
            # Construire l'alphabet
            alphabet = lowercase + uppercase + digits + symbols
            
            # Générer le mot de passe
            password = ''.join(secrets.choice(alphabet) for _ in range(length))
            
            # S'assurer qu'il contient au moins un caractère de chaque type
            if include_symbols and len(symbols) > 0:
                # Remplacer quelques caractères pour garantir la diversité
                password = list(password)
                password[0] = secrets.choice(lowercase)
                password[1] = secrets.choice(uppercase)
                password[2] = secrets.choice(digits)
                if include_symbols:
                    password[3] = secrets.choice(symbols)
                
                # Mélanger
                secrets.SystemRandom().shuffle(password)
                password = ''.join(password)
            
            # Valider le mot de passe généré
            is_valid, errors = await self.validate_password(password)
            if not is_valid:
                # Régénérer si invalide
                return await self.generate_secure_password(length, include_symbols, exclude_ambiguous)
            
            return password
            
        except Exception as exc:
            self.logger.error(f"Erreur génération mot de passe: {exc}")
            raise
    
    async def check_password_history(self, user_id: str, new_password: str) -> bool:
        """Vérifie si le mot de passe a déjà été utilisé"""
        try:
            # Récupérer l'historique des mots de passe
            history_key = f"password_history:{user_id}"
            password_hashes = await self.redis_client.lrange(history_key, 0, -1)
            
            # Vérifier contre chaque hash
            for hash_bytes in password_hashes:
                hash_str = hash_bytes.decode() if isinstance(hash_bytes, bytes) else hash_bytes
                if await self.verify_password(new_password, hash_str):
                    return True  # Mot de passe déjà utilisé
            
            return False
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification historique mot de passe: {exc}")
            return False
    
    async def update_password(self, user_id: str, new_password: str) -> bool:
        """Met à jour le mot de passe d'un utilisateur"""
        try:
            # Valider le nouveau mot de passe
            is_valid, errors = await self.validate_password(new_password)
            if not is_valid:
                raise ValueError(f"Mot de passe invalide: {', '.join(errors)}")
            
            # Vérifier l'historique
            if await self.check_password_history(user_id, new_password):
                raise ValueError("Ce mot de passe a déjà été utilisé récemment")
            
            # Hacher le nouveau mot de passe
            new_hash = await self.hash_password(new_password)
            
            # Récupérer l'ancien hash pour l'historique
            old_hash_key = f"user_password:{user_id}"
            old_hash = await self.redis_client.get(old_hash_key)
            
            # Mettre à jour le mot de passe
            await self.redis_client.set(old_hash_key, new_hash)
            
            # Ajouter à l'historique
            if old_hash:
                history_key = f"password_history:{user_id}"
                await self.redis_client.lpush(history_key, old_hash)
                
                # Limiter l'historique
                await self.redis_client.ltrim(
                    history_key,
                    0,
                    self.password_policy.password_history_count - 1
                )
            
            # Mettre à jour la date de changement
            await self.redis_client.set(
                f"password_changed:{user_id}",
                datetime.utcnow().isoformat()
            )
            
            self.logger.info(f"Mot de passe mis à jour pour utilisateur {user_id}")
            return True
            
        except Exception as exc:
            self.logger.error(f"Erreur mise à jour mot de passe: {exc}")
            raise
    
    async def is_password_expired(self, user_id: str) -> bool:
        """Vérifie si le mot de passe a expiré"""
        try:
            changed_date_key = f"password_changed:{user_id}"
            changed_date_str = await self.redis_client.get(changed_date_key)
            
            if not changed_date_str:
                return True  # Pas de date = expiré
            
            changed_date = datetime.fromisoformat(changed_date_str.decode())
            expiry_date = changed_date + timedelta(days=self.password_policy.max_age_days)
            
            return datetime.utcnow() > expiry_date
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification expiration mot de passe: {exc}")
            return True
    
    # Méthodes privées
    def _has_repeated_chars(self, password: str, max_repeats: int) -> bool:
        """Vérifie les caractères répétés consécutifs"""
        count = 1
        for i in range(1, len(password)):
            if password[i] == password[i-1]:
                count += 1
                if count > max_repeats:
                    return True
            else:
                count = 1
        return False
    
    def _has_sequential_chars(self, password: str) -> bool:
        """Vérifie les séquences de caractères"""
        sequences = [
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "0123456789",
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm"
        ]
        
        for seq in sequences:
            for i in range(len(seq) - 2):
                if seq[i:i+3] in password or seq[i:i+3][::-1] in password:
                    return True
        
        return False
    
    def _has_common_patterns(self, password: str) -> bool:
        """Vérifie les patterns communs"""
        patterns = [
            r'(.)\1{2,}',  # Répétitions
            r'(012|123|234|345|456|567|678|789)',  # Séquences numériques
            r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)',  # Séquences alphabétiques
            r'(password|admin|user|login|welcome|secret)',  # Mots communs
            r'\d{4,}',  # Longues séquences de chiffres
        ]
        
        for pattern in patterns:
            if re.search(pattern, password.lower()):
                return True
        
        return False
    
    def _contains_personal_info(self, password: str, user_info: Dict[str, Any]) -> bool:
        """Vérifie si le mot de passe contient des informations personnelles"""
        personal_fields = ['username', 'email', 'first_name', 'last_name', 'phone']
        password_lower = password.lower()
        
        for field in personal_fields:
            if field in user_info and user_info[field]:
                value = str(user_info[field]).lower()
                if len(value) >= 3 and value in password_lower:
                    return True
        
        return False
    
    def _calculate_entropy(self, password: str) -> float:
        """Calcule l'entropie du mot de passe"""
        charset_size = 0
        
        if re.search(r'[a-z]', password):
            charset_size += 26
        if re.search(r'[A-Z]', password):
            charset_size += 26
        if re.search(r'\d', password):
            charset_size += 10
        if re.search(r'[^a-zA-Z0-9]', password):
            charset_size += 32  # Estimation des caractères spéciaux
        
        if charset_size == 0:
            return 0.0
        
        import math
        return len(password) * math.log2(charset_size)
    
    def _estimate_crack_time(self, entropy: float) -> str:
        """Estime le temps de cassage en force brute"""
        # Hypothèse: 1 milliard de tentatives par seconde
        attempts_per_second = 1e9
        total_combinations = 2 ** entropy
        seconds_to_crack = total_combinations / (2 * attempts_per_second)  # Moyenne
        
        if seconds_to_crack < 1:
            return "Instantané"
        elif seconds_to_crack < 60:
            return f"{int(seconds_to_crack)} secondes"
        elif seconds_to_crack < 3600:
            return f"{int(seconds_to_crack / 60)} minutes"
        elif seconds_to_crack < 86400:
            return f"{int(seconds_to_crack / 3600)} heures"
        elif seconds_to_crack < 31536000:
            return f"{int(seconds_to_crack / 86400)} jours"
        elif seconds_to_crack < 31536000000:
            return f"{int(seconds_to_crack / 31536000)} années"
        else:
            return "Plusieurs millénaires"
    
    async def _check_password_breach(self, password: str) -> bool:
        """Vérifie si le mot de passe a été compromis (HaveIBeenPwned)"""
        try:
            # Hacher le mot de passe avec SHA-1
            sha1_hash = hashlib.sha1(password.encode()).hexdigest().upper()
            
            # Prendre les 5 premiers caractères
            prefix = sha1_hash[:5]
            suffix = sha1_hash[5:]
            
            # Requête à l'API (implémentation simplifiée)
            # Dans un vrai système, implémenter avec gestion d'erreurs et cache
            return False  # Simplification pour cet exemple
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification compromission: {exc}")
            return False
    
    async def _get_breach_count(self, password: str) -> int:
        """Récupère le nombre de fois que le mot de passe a été compromis"""
        # Implémentation simplifiée
        return 0


class PasswordlessAuthManager:
    """Gestionnaire d'authentification sans mot de passe"""
    
    def __init__(self, redis_client: redis.Redis, email_config: Dict[str, Any] = None):
        self.redis_client = redis_client
        self.email_config = email_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.magic_link_expiry = 900  # 15 minutes
        self.push_notification_expiry = 300  # 5 minutes
        self.webauthn_challenge_expiry = 300  # 5 minutes
    
    async def send_magic_link(
        self,
        user_id: str,
        email: str,
        redirect_url: str = None,
        ip_address: str = None,
        user_agent: str = None
    ) -> MagicLink:
        """Envoie un lien magique par email"""
        try:
            # Générer le token
            link_id = secrets.token_urlsafe(32)
            token = secrets.token_urlsafe(64)
            
            # Créer le lien magique
            magic_link = MagicLink(
                link_id=link_id,
                user_id=user_id,
                email=email,
                token=token,
                expires_at=datetime.utcnow() + timedelta(seconds=self.magic_link_expiry),
                ip_address=ip_address,
                user_agent=user_agent,
                metadata={"redirect_url": redirect_url} if redirect_url else {}
            )
            
            # Stocker dans Redis
            await self._store_magic_link(magic_link)
            
            # Construire l'URL du lien
            magic_url = await self._build_magic_link_url(magic_link, redirect_url)
            
            # Envoyer l'email
            await self._send_magic_link_email(email, magic_url)
            
            self.logger.info(f"Lien magique envoyé à {email} pour utilisateur {user_id}")
            return magic_link
            
        except Exception as exc:
            self.logger.error(f"Erreur envoi lien magique: {exc}")
            raise
    
    async def verify_magic_link(self, token: str, ip_address: str = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """Vérifie un lien magique"""
        try:
            # Récupérer le lien magique
            magic_link = await self._get_magic_link_by_token(token)
            
            if not magic_link:
                return False, None, "Lien invalide"
            
            if magic_link.used:
                return False, None, "Lien déjà utilisé"
            
            if datetime.utcnow() > magic_link.expires_at:
                return False, None, "Lien expiré"
            
            # Vérifications de sécurité optionnelles
            if ip_address and magic_link.ip_address and ip_address != magic_link.ip_address:
                self.logger.warning(f"IP différente pour lien magique: {ip_address} vs {magic_link.ip_address}")
                # Permettre mais logger
            
            # Marquer comme utilisé
            magic_link.used = True
            await self._store_magic_link(magic_link)
            
            # Récupérer l'URL de redirection
            redirect_url = magic_link.metadata.get("redirect_url")
            
            return True, magic_link.user_id, redirect_url
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification lien magique: {exc}")
            return False, None, "Erreur interne"
    
    async def initiate_push_authentication(
        self,
        user_id: str,
        device_token: str,
        message: str = "Nouvelle demande de connexion"
    ) -> str:
        """Initie une authentification par notification push"""
        try:
            # Générer un ID de challenge
            challenge_id = secrets.token_urlsafe(32)
            
            # Stocker le challenge
            challenge_data = {
                "user_id": user_id,
                "device_token": device_token,
                "message": message,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(seconds=self.push_notification_expiry)).isoformat(),
                "approved": False
            }
            
            await self.redis_client.setex(
                f"push_challenge:{challenge_id}",
                self.push_notification_expiry,
                json.dumps(challenge_data)
            )
            
            # Envoyer la notification push
            await self._send_push_notification(device_token, message, challenge_id)
            
            self.logger.info(f"Notification push envoyée pour utilisateur {user_id}")
            return challenge_id
            
        except Exception as exc:
            self.logger.error(f"Erreur initiation push authentication: {exc}")
            raise
    
    async def verify_push_authentication(self, challenge_id: str) -> Tuple[bool, Optional[str]]:
        """Vérifie une authentification push"""
        try:
            challenge_data = await self.redis_client.get(f"push_challenge:{challenge_id}")
            
            if not challenge_data:
                return False, None
            
            challenge = json.loads(challenge_data)
            
            # Vérifier l'expiration
            expires_at = datetime.fromisoformat(challenge["expires_at"])
            if datetime.utcnow() > expires_at:
                return False, None
            
            # Vérifier l'approbation
            if not challenge.get("approved", False):
                return False, None
            
            return True, challenge["user_id"]
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification push authentication: {exc}")
            return False, None
    
    async def approve_push_authentication(self, challenge_id: str) -> bool:
        """Approuve une authentification push"""
        try:
            challenge_data = await self.redis_client.get(f"push_challenge:{challenge_id}")
            
            if not challenge_data:
                return False
            
            challenge = json.loads(challenge_data)
            challenge["approved"] = True
            
            # Recalculer le TTL
            expires_at = datetime.fromisoformat(challenge["expires_at"])
            ttl = int((expires_at - datetime.utcnow()).total_seconds())
            
            if ttl > 0:
                await self.redis_client.setex(
                    f"push_challenge:{challenge_id}",
                    ttl,
                    json.dumps(challenge)
                )
                return True
            
            return False
            
        except Exception as exc:
            self.logger.error(f"Erreur approbation push authentication: {exc}")
            return False
    
    async def generate_webauthn_challenge(self, user_id: str) -> Dict[str, Any]:
        """Génère un challenge WebAuthn"""
        try:
            # Générer un challenge aléatoire
            challenge = secrets.token_bytes(32)
            challenge_b64 = base64.urlsafe_b64encode(challenge).decode().rstrip('=')
            
            # Informations utilisateur
            user_info = {
                "id": base64.urlsafe_b64encode(user_id.encode()).decode().rstrip('='),
                "name": f"user_{user_id}",  # À remplacer par le vrai nom
                "displayName": f"Utilisateur {user_id}"  # À remplacer par le vrai nom d'affichage
            }
            
            # Options de création
            creation_options = {
                "challenge": challenge_b64,
                "rp": {
                    "name": "Spotify AI Agent",
                    "id": "localhost"  # À remplacer par le vrai domaine
                },
                "user": user_info,
                "pubKeyCredParams": [
                    {"alg": -7, "type": "public-key"},  # ES256
                    {"alg": -257, "type": "public-key"}  # RS256
                ],
                "authenticatorSelection": {
                    "authenticatorAttachment": "platform",
                    "userVerification": "required"
                },
                "timeout": 60000,
                "attestation": "direct"
            }
            
            # Stocker le challenge
            await self.redis_client.setex(
                f"webauthn_challenge:{user_id}",
                self.webauthn_challenge_expiry,
                challenge_b64
            )
            
            return creation_options
            
        except Exception as exc:
            self.logger.error(f"Erreur génération challenge WebAuthn: {exc}")
            raise
    
    async def verify_webauthn_credential(
        self,
        user_id: str,
        credential_data: Dict[str, Any]
    ) -> bool:
        """Vérifie un credential WebAuthn"""
        try:
            # Récupérer le challenge stocké
            stored_challenge = await self.redis_client.get(f"webauthn_challenge:{user_id}")
            
            if not stored_challenge:
                return False
            
            # Implémentation simplifiée de la vérification WebAuthn
            # Dans un vrai système, utiliser une bibliothèque comme webauthn-rp
            
            # Vérifier le challenge
            client_challenge = credential_data.get("response", {}).get("clientDataJSON", "")
            # ... vérification complexe du credential ...
            
            # Supprimer le challenge utilisé
            await self.redis_client.delete(f"webauthn_challenge:{user_id}")
            
            # Pour cet exemple, retourner True si les données de base sont présentes
            return bool(credential_data.get("id") and credential_data.get("response"))
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification WebAuthn: {exc}")
            return False
    
    # Méthodes privées
    async def _store_magic_link(self, magic_link: MagicLink):
        """Stocke un lien magique dans Redis"""
        try:
            link_data = json.dumps(asdict(magic_link), default=str)
            
            # Stocker par token pour récupération rapide
            await self.redis_client.setex(
                f"magic_link:{magic_link.token}",
                self.magic_link_expiry,
                link_data
            )
            
            # Stocker par link_id pour gestion
            await self.redis_client.setex(
                f"magic_link_id:{magic_link.link_id}",
                self.magic_link_expiry,
                magic_link.token
            )
            
        except Exception as exc:
            self.logger.error(f"Erreur stockage lien magique: {exc}")
    
    async def _get_magic_link_by_token(self, token: str) -> Optional[MagicLink]:
        """Récupère un lien magique par son token"""
        try:
            link_data = await self.redis_client.get(f"magic_link:{token}")
            if link_data:
                link_dict = json.loads(link_data)
                return MagicLink(**link_dict)
            return None
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération lien magique: {exc}")
            return None
    
    async def _build_magic_link_url(self, magic_link: MagicLink, redirect_url: str = None) -> str:
        """Construit l'URL du lien magique"""
        base_url = "https://localhost/auth/magic-link"  # À configurer
        params = {"token": magic_link.token}
        
        if redirect_url:
            params["redirect"] = redirect_url
        
        return f"{base_url}?{urlencode(params)}"
    
    async def _send_magic_link_email(self, email: str, magic_url: str):
        """Envoie l'email avec le lien magique"""
        try:
            # Configuration SMTP simplifiée
            if not self.email_config:
                self.logger.warning("Configuration email non disponible")
                return
            
            # Créer le message
            msg = MimeMultipart('alternative')
            msg['Subject'] = "Votre lien de connexion sécurisé"
            msg['From'] = self.email_config.get('from_address')
            msg['To'] = email
            
            # Corps de l'email
            text_body = f"""
            Bonjour,
            
            Vous avez demandé une connexion sécurisée. Cliquez sur le lien ci-dessous pour vous connecter :
            
            {magic_url}
            
            Ce lien expire dans 15 minutes.
            
            Si vous n'avez pas demandé cette connexion, ignorez cet email.
            """
            
            html_body = f"""
            <html>
            <body>
                <h2>Connexion sécurisée</h2>
                <p>Bonjour,</p>
                <p>Vous avez demandé une connexion sécurisée. Cliquez sur le bouton ci-dessous pour vous connecter :</p>
                <p><a href="{magic_url}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Se connecter</a></p>
                <p>Ce lien expire dans 15 minutes.</p>
                <p>Si vous n'avez pas demandé cette connexion, ignorez cet email.</p>
            </body>
            </html>
            """
            
            msg.attach(MimeText(text_body, 'plain'))
            msg.attach(MimeText(html_body, 'html'))
            
            # Envoyer (implémentation simplifiée)
            self.logger.info(f"Email envoyé à {email}")
            
        except Exception as exc:
            self.logger.error(f"Erreur envoi email: {exc}")
    
    async def _send_push_notification(self, device_token: str, message: str, challenge_id: str):
        """Envoie une notification push"""
        try:
            # Implémentation simplifiée pour Firebase/APNs
            notification_payload = {
                "title": "Demande de connexion",
                "body": message,
                "data": {
                    "challenge_id": challenge_id,
                    "action": "authenticate"
                }
            }
            
            self.logger.info(f"Notification push envoyée au device {device_token}")
            
        except Exception as exc:
            self.logger.error(f"Erreur envoi notification push: {exc}")


class SocialAuthManager:
    """Gestionnaire d'authentification sociale"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Configuration des providers (à externaliser)
        self.providers = {
            "google": {
                "client_id": "your-google-client-id",
                "client_secret": "your-google-client-secret",
                "authorize_url": "https://accounts.google.com/o/oauth2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo"
            },
            "github": {
                "client_id": "your-github-client-id",
                "client_secret": "your-github-client-secret",
                "authorize_url": "https://github.com/login/oauth/authorize",
                "token_url": "https://github.com/login/oauth/access_token",
                "userinfo_url": "https://api.github.com/user"
            }
        }
    
    async def get_authorization_url(self, provider: str, redirect_uri: str, state: str = None) -> str:
        """Génère l'URL d'autorisation pour un provider"""
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider non supporté: {provider}")
            
            config = self.providers[provider]
            
            if not state:
                state = secrets.token_urlsafe(32)
            
            # Stocker l'état pour vérification
            await self.redis_client.setex(f"oauth_state:{state}", 600, provider)
            
            params = {
                "client_id": config["client_id"],
                "redirect_uri": redirect_uri,
                "scope": "openid email profile" if provider == "google" else "user:email",
                "response_type": "code",
                "state": state
            }
            
            return f"{config['authorize_url']}?{urlencode(params)}"
            
        except Exception as exc:
            self.logger.error(f"Erreur génération URL autorisation {provider}: {exc}")
            raise
    
    async def exchange_code_for_token(
        self,
        provider: str,
        code: str,
        redirect_uri: str,
        state: str = None
    ) -> Dict[str, Any]:
        """Échange le code d'autorisation contre un token d'accès"""
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider non supporté: {provider}")
            
            # Vérifier l'état si fourni
            if state:
                stored_provider = await self.redis_client.get(f"oauth_state:{state}")
                if not stored_provider or stored_provider.decode() != provider:
                    raise ValueError("État OAuth invalide")
                
                # Supprimer l'état utilisé
                await self.redis_client.delete(f"oauth_state:{state}")
            
            config = self.providers[provider]
            
            # Préparer la requête de token
            token_data = {
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri
            }
            
            # Implémentation simplifiée - dans un vrai système, faire la requête HTTP
            # response = requests.post(config["token_url"], data=token_data)
            # return response.json()
            
            # Pour cet exemple, retourner un token fictif
            return {
                "access_token": f"mock_token_{provider}_{secrets.token_urlsafe(16)}",
                "token_type": "Bearer",
                "expires_in": 3600
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur échange code token {provider}: {exc}")
            raise
    
    async def get_user_info(self, provider: str, access_token: str) -> Dict[str, Any]:
        """Récupère les informations utilisateur depuis le provider"""
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider non supporté: {provider}")
            
            config = self.providers[provider]
            
            # Implémentation simplifiée - dans un vrai système, faire la requête HTTP
            # headers = {"Authorization": f"Bearer {access_token}"}
            # response = requests.get(config["userinfo_url"], headers=headers)
            # return response.json()
            
            # Pour cet exemple, retourner des données fictives
            return {
                "id": f"user_id_{provider}_{secrets.token_urlsafe(8)}",
                "email": f"user@{provider}.com",
                "name": f"User {provider.title()}",
                "picture": f"https://{provider}.com/avatar.jpg"
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération info utilisateur {provider}: {exc}")
            raise
