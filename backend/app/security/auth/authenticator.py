# 🔐 Advanced Authentication Manager
# ==================================
# 
# Gestionnaire d'authentification enterprise avec MFA,
# authentification biométrique et basée sur les risques.
#
# 🎖️ Expert: Lead Dev + Architecte IA + Spécialiste Sécurité Backend
#
# Développé par l'équipe d'experts enterprise
# ==================================

"""
🔐 Enterprise Authentication Manager
===================================

Advanced authentication system providing:
- Multi-factor authentication (TOTP, SMS, Email, Hardware tokens)
- Biometric authentication (fingerprint, face recognition)
- Risk-based authentication with ML threat detection
- Adaptive authentication based on user behavior
- Passwordless authentication flows
- Device trust management
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
import pyotp
import qrcode
import io
import base64
from PIL import Image
import face_recognition
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
import joblib
import redis
import logging
from fastapi import HTTPException
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configuration et logging
logger = logging.getLogger(__name__)


class AuthenticationMethod(Enum):
    """Méthodes d'authentification disponibles"""
    PASSWORD = "password"
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    PUSH_NOTIFICATION = "push"
    BIOMETRIC_FINGERPRINT = "biometric_fingerprint"
    BIOMETRIC_FACE = "biometric_face"
    HARDWARE_TOKEN = "hardware_token"
    VOICE_RECOGNITION = "voice"
    PASSWORDLESS_MAGIC_LINK = "magic_link"
    WEBAUTHN = "webauthn"


class AuthenticationResult(Enum):
    """Résultats d'authentification"""
    SUCCESS = "success"
    FAILED = "failed"
    MFA_REQUIRED = "mfa_required"
    STEP_UP_REQUIRED = "step_up_required"
    ACCOUNT_LOCKED = "account_locked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DEVICE_NOT_TRUSTED = "device_not_trusted"


class RiskLevel(Enum):
    """Niveaux de risque"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuthenticationContext:
    """Contexte d'authentification"""
    user_id: str
    ip_address: str
    user_agent: str
    device_fingerprint: str
    location: Optional[Dict[str, str]] = None
    timestamp: datetime = None
    risk_score: float = 0.0
    authentication_methods: List[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.authentication_methods is None:
            self.authentication_methods = []


@dataclass
class BiometricData:
    """Données biométriques"""
    user_id: str
    biometric_type: str
    template: bytes
    confidence_score: float
    created_at: datetime
    last_used: datetime = None


class AuthenticationManager:
    """Gestionnaire principal d'authentification"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        self.mfa_authenticator = MultiFactorAuthenticator(redis_client)
        self.biometric_auth = BiometricAuthenticator(redis_client)
        self.risk_auth = RiskBasedAuthenticator(redis_client)
        self.fernet = self._initialize_encryption()
        
        # Configuration
        self.max_failed_attempts = 5
        self.lockout_duration = 3600  # 1 heure
        self.session_timeout = 86400  # 24 heures
        
    def _initialize_encryption(self) -> Fernet:
        """Initialise le chiffrement pour les données sensibles"""
        key = Fernet.generate_key()
        return Fernet(key)
    
    async def authenticate(
        self,
        user_id: str,
        password: str,
        context: AuthenticationContext,
        require_mfa: bool = True
    ) -> Tuple[AuthenticationResult, Dict[str, Any]]:
        """Authentification principale avec analyse de risque"""
        try:
            # Vérification du verrouillage de compte
            if await self._is_account_locked(user_id):
                return AuthenticationResult.ACCOUNT_LOCKED, {
                    "message": "Compte verrouillé temporairement"
                }
            
            # Authentification par mot de passe
            password_valid = await self._verify_password(user_id, password)
            if not password_valid:
                await self._record_failed_attempt(user_id, context)
                return AuthenticationResult.FAILED, {
                    "message": "Identifiants invalides"
                }
            
            # Analyse de risque
            risk_analysis = await self.risk_auth.analyze_authentication_risk(
                user_id, context
            )
            context.risk_score = risk_analysis["risk_score"]
            
            # Déterminer les exigences d'authentification
            auth_requirements = await self._determine_auth_requirements(
                user_id, context, require_mfa
            )
            
            if auth_requirements["mfa_required"]:
                # Initier le processus MFA
                challenge = await self.mfa_authenticator.initiate_challenge(
                    user_id, auth_requirements["methods"]
                )
                return AuthenticationResult.MFA_REQUIRED, {
                    "challenge": challenge,
                    "session_id": context.session_id
                }
            
            # Authentification réussie sans MFA
            session_data = await self._create_session(user_id, context)
            await self._clear_failed_attempts(user_id)
            
            return AuthenticationResult.SUCCESS, {
                "session": session_data,
                "risk_level": risk_analysis["risk_level"]
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur lors de l'authentification: {exc}")
            return AuthenticationResult.FAILED, {
                "message": "Erreur interne"
            }
    
    async def complete_mfa_authentication(
        self,
        user_id: str,
        session_id: str,
        mfa_responses: Dict[str, str]
    ) -> Tuple[AuthenticationResult, Dict[str, Any]]:
        """Complète l'authentification MFA"""
        try:
            # Vérifier la session temporaire
            temp_session = await self._get_temp_session(session_id)
            if not temp_session or temp_session["user_id"] != user_id:
                return AuthenticationResult.FAILED, {
                    "message": "Session invalide"
                }
            
            # Vérifier tous les facteurs d'authentification
            verification_results = []
            for method, response in mfa_responses.items():
                result = await self.mfa_authenticator.verify_factor(
                    user_id, method, response
                )
                verification_results.append(result)
            
            # Tous les facteurs doivent être valides
            if not all(verification_results):
                return AuthenticationResult.FAILED, {
                    "message": "Authentification MFA échouée"
                }
            
            # Créer la session complète
            context = AuthenticationContext(**temp_session["context"])
            session_data = await self._create_session(user_id, context)
            await self._cleanup_temp_session(session_id)
            
            return AuthenticationResult.SUCCESS, {
                "session": session_data
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur MFA: {exc}")
            return AuthenticationResult.FAILED, {"message": "Erreur interne"}
    
    async def biometric_authentication(
        self,
        user_id: str,
        biometric_data: bytes,
        biometric_type: str,
        context: AuthenticationContext
    ) -> Tuple[AuthenticationResult, Dict[str, Any]]:
        """Authentification biométrique"""
        try:
            # Vérification biométrique
            verification_result = await self.biometric_auth.verify_biometric(
                user_id, biometric_data, biometric_type
            )
            
            if not verification_result["verified"]:
                return AuthenticationResult.FAILED, {
                    "message": "Biométrie non reconnue"
                }
            
            # Analyse de risque pour biométrie
            risk_analysis = await self.risk_auth.analyze_biometric_risk(
                user_id, context, verification_result
            )
            
            if risk_analysis["risk_level"] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                return AuthenticationResult.STEP_UP_REQUIRED, {
                    "message": "Authentification supplémentaire requise",
                    "risk_level": risk_analysis["risk_level"]
                }
            
            # Créer la session
            context.authentication_methods.append(biometric_type)
            session_data = await self._create_session(user_id, context)
            
            return AuthenticationResult.SUCCESS, {
                "session": session_data,
                "confidence": verification_result["confidence"]
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur authentification biométrique: {exc}")
            return AuthenticationResult.FAILED, {"message": "Erreur interne"}
    
    async def passwordless_authentication(
        self,
        identifier: str,  # email, phone, etc.
        method: AuthenticationMethod,
        context: AuthenticationContext
    ) -> Tuple[AuthenticationResult, Dict[str, Any]]:
        """Authentification sans mot de passe"""
        try:
            if method == AuthenticationMethod.PASSWORDLESS_MAGIC_LINK:
                # Générer un lien magique
                magic_token = await self._generate_magic_link_token(identifier)
                await self._send_magic_link(identifier, magic_token)
                
                return AuthenticationResult.SUCCESS, {
                    "message": "Lien d'authentification envoyé",
                    "method": "magic_link"
                }
            
            elif method == AuthenticationMethod.WEBAUTHN:
                # Initier WebAuthn
                challenge = await self._initiate_webauthn_challenge(identifier)
                
                return AuthenticationResult.MFA_REQUIRED, {
                    "challenge": challenge,
                    "method": "webauthn"
                }
            
            else:
                return AuthenticationResult.FAILED, {
                    "message": "Méthode non supportée"
                }
                
        except Exception as exc:
            self.logger.error(f"Erreur authentification passwordless: {exc}")
            return AuthenticationResult.FAILED, {"message": "Erreur interne"}
    
    async def _verify_password(self, user_id: str, password: str) -> bool:
        """Vérifie le mot de passe utilisateur"""
        try:
            # Récupérer le hash du mot de passe depuis la base
            stored_hash = await self._get_password_hash(user_id)
            if not stored_hash:
                return False
            
            # Vérifier avec bcrypt
            return bcrypt.checkpw(
                password.encode('utf-8'),
                stored_hash.encode('utf-8')
            )
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification mot de passe: {exc}")
            return False
    
    async def _determine_auth_requirements(
        self,
        user_id: str,
        context: AuthenticationContext,
        require_mfa: bool
    ) -> Dict[str, Any]:
        """Détermine les exigences d'authentification"""
        try:
            requirements = {
                "mfa_required": False,
                "methods": [],
                "step_up_required": False
            }
            
            # MFA obligatoire pour comptes privilégiés
            user_profile = await self._get_user_profile(user_id)
            if user_profile.get("privileged", False):
                requirements["mfa_required"] = True
                requirements["methods"].extend(["totp", "sms"])
            
            # MFA basé sur le risque
            if context.risk_score > 0.6:
                requirements["mfa_required"] = True
                requirements["methods"].append("totp")
            
            # MFA basé sur la localisation
            if await self._is_new_location(user_id, context):
                requirements["mfa_required"] = True
                requirements["methods"].append("email")
            
            # MFA basé sur le dispositif
            if await self._is_new_device(user_id, context):
                requirements["mfa_required"] = True
                requirements["methods"].extend(["totp", "sms"])
            
            # MFA forcé par politique
            if require_mfa:
                requirements["mfa_required"] = True
                if not requirements["methods"]:
                    requirements["methods"].append("totp")
            
            return requirements
            
        except Exception as exc:
            self.logger.error(f"Erreur détermination exigences auth: {exc}")
            return {"mfa_required": False, "methods": []}
    
    async def _create_session(
        self,
        user_id: str,
        context: AuthenticationContext
    ) -> Dict[str, Any]:
        """Crée une session utilisateur sécurisée"""
        try:
            session_id = secrets.token_urlsafe(32)
            session_data = {
                "user_id": user_id,
                "session_id": session_id,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(seconds=self.session_timeout)).isoformat(),
                "ip_address": context.ip_address,
                "user_agent": context.user_agent,
                "device_fingerprint": context.device_fingerprint,
                "authentication_methods": context.authentication_methods,
                "risk_score": context.risk_score
            }
            
            # Stocker la session dans Redis
            await self.redis_client.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(session_data)
            )
            
            # Enregistrer l'activité utilisateur
            await self._record_user_activity(user_id, context)
            
            return session_data
            
        except Exception as exc:
            self.logger.error(f"Erreur création session: {exc}")
            raise
    
    async def _is_account_locked(self, user_id: str) -> bool:
        """Vérifie si le compte est verrouillé"""
        try:
            lock_key = f"account_lock:{user_id}"
            return await self.redis_client.exists(lock_key)
        except Exception:
            return False
    
    async def _record_failed_attempt(
        self,
        user_id: str,
        context: AuthenticationContext
    ):
        """Enregistre une tentative d'authentification échouée"""
        try:
            key = f"failed_attempts:{user_id}"
            current_time = int(time.time())
            
            # Ajouter la tentative échouée
            await self.redis_client.zadd(key, {str(current_time): current_time})
            await self.redis_client.expire(key, 3600)  # Expire après 1 heure
            
            # Compter les tentatives dans la dernière heure
            one_hour_ago = current_time - 3600
            failed_count = await self.redis_client.zcount(key, one_hour_ago, current_time)
            
            # Verrouiller le compte si trop de tentatives
            if failed_count >= self.max_failed_attempts:
                await self._lock_account(user_id)
                
        except Exception as exc:
            self.logger.error(f"Erreur enregistrement tentative échouée: {exc}")
    
    async def _lock_account(self, user_id: str):
        """Verrouille temporairement un compte"""
        try:
            lock_key = f"account_lock:{user_id}"
            await self.redis_client.setex(lock_key, self.lockout_duration, "locked")
            
            # Log de sécurité
            self.logger.warning(f"Compte verrouillé: {user_id}")
            
        except Exception as exc:
            self.logger.error(f"Erreur verrouillage compte: {exc}")
    
    async def _clear_failed_attempts(self, user_id: str):
        """Efface les tentatives d'authentification échouées"""
        try:
            key = f"failed_attempts:{user_id}"
            await self.redis_client.delete(key)
        except Exception as exc:
            self.logger.error(f"Erreur nettoyage tentatives: {exc}")
    
    # Méthodes utilitaires (implémentation simplifiée)
    async def _get_password_hash(self, user_id: str) -> Optional[str]:
        """Récupère le hash du mot de passe depuis la base"""
        # Implémentation avec votre ORM/base de données
        return None
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Récupère le profil utilisateur"""
        # Implémentation avec votre ORM/base de données
        return {}
    
    async def _is_new_location(self, user_id: str, context: AuthenticationContext) -> bool:
        """Vérifie si c'est une nouvelle localisation"""
        # Implémentation basée sur l'historique des localisations
        return False
    
    async def _is_new_device(self, user_id: str, context: AuthenticationContext) -> bool:
        """Vérifie si c'est un nouveau dispositif"""
        # Implémentation basée sur l'empreinte du dispositif
        return False
    
    async def _get_temp_session(self, session_id: str) -> Optional[Dict]:
        """Récupère une session temporaire"""
        try:
            data = await self.redis_client.get(f"temp_session:{session_id}")
            return json.loads(data) if data else None
        except Exception:
            return None
    
    async def _cleanup_temp_session(self, session_id: str):
        """Nettoie une session temporaire"""
        try:
            await self.redis_client.delete(f"temp_session:{session_id}")
        except Exception as exc:
            self.logger.error(f"Erreur nettoyage session temporaire: {exc}")
    
    async def _record_user_activity(self, user_id: str, context: AuthenticationContext):
        """Enregistre l'activité utilisateur pour l'analyse comportementale"""
        try:
            activity_data = {
                "timestamp": context.timestamp.isoformat(),
                "ip_address": context.ip_address,
                "user_agent": context.user_agent,
                "device_fingerprint": context.device_fingerprint,
                "risk_score": context.risk_score
            }
            
            # Stocker dans Redis pour analyse
            key = f"user_activity:{user_id}"
            await self.redis_client.lpush(key, json.dumps(activity_data))
            await self.redis_client.ltrim(key, 0, 100)  # Garder les 100 dernières activités
            await self.redis_client.expire(key, 86400 * 30)  # 30 jours
            
        except Exception as exc:
            self.logger.error(f"Erreur enregistrement activité: {exc}")
    
    async def _generate_magic_link_token(self, identifier: str) -> str:
        """Génère un token pour lien magique"""
        try:
            token_data = {
                "identifier": identifier,
                "timestamp": datetime.utcnow().isoformat(),
                "nonce": secrets.token_urlsafe(16)
            }
            
            # Chiffrer le token
            encrypted_token = self.fernet.encrypt(
                json.dumps(token_data).encode()
            ).decode()
            
            # Stocker temporairement
            await self.redis_client.setex(
                f"magic_token:{encrypted_token}",
                900,  # 15 minutes
                json.dumps(token_data)
            )
            
            return encrypted_token
            
        except Exception as exc:
            self.logger.error(f"Erreur génération token magique: {exc}")
            raise
    
    async def _send_magic_link(self, identifier: str, token: str):
        """Envoie le lien magique par email/SMS"""
        # Implémentation de l'envoi (email/SMS service)
        pass
    
    async def _initiate_webauthn_challenge(self, identifier: str) -> Dict[str, Any]:
        """Initie un challenge WebAuthn"""
        # Implémentation WebAuthn
        return {
            "challenge": secrets.token_urlsafe(32),
            "timeout": 60000,
            "userVerification": "required"
        }


class MultiFactorAuthenticator:
    """Authentificateur multi-facteurs avancé"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def initiate_challenge(
        self,
        user_id: str,
        methods: List[str]
    ) -> Dict[str, Any]:
        """Initie un challenge MFA"""
        try:
            challenge_id = secrets.token_urlsafe(16)
            challenges = {}
            
            for method in methods:
                if method == "totp":
                    challenges["totp"] = await self._initiate_totp_challenge(user_id)
                elif method == "sms":
                    challenges["sms"] = await self._initiate_sms_challenge(user_id)
                elif method == "email":
                    challenges["email"] = await self._initiate_email_challenge(user_id)
                elif method == "push":
                    challenges["push"] = await self._initiate_push_challenge(user_id)
            
            # Stocker le challenge temporairement
            challenge_data = {
                "user_id": user_id,
                "methods": methods,
                "challenges": challenges,
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(
                f"mfa_challenge:{challenge_id}",
                300,  # 5 minutes
                json.dumps(challenge_data)
            )
            
            return {
                "challenge_id": challenge_id,
                "methods": methods,
                "challenges": challenges
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur initiation challenge MFA: {exc}")
            raise
    
    async def verify_factor(
        self,
        user_id: str,
        method: str,
        response: str
    ) -> bool:
        """Vérifie un facteur d'authentification"""
        try:
            if method == "totp":
                return await self._verify_totp(user_id, response)
            elif method == "sms":
                return await self._verify_sms_code(user_id, response)
            elif method == "email":
                return await self._verify_email_code(user_id, response)
            elif method == "push":
                return await self._verify_push_response(user_id, response)
            else:
                return False
                
        except Exception as exc:
            self.logger.error(f"Erreur vérification facteur {method}: {exc}")
            return False
    
    async def _initiate_totp_challenge(self, user_id: str) -> Dict[str, Any]:
        """Initie un challenge TOTP"""
        return {
            "type": "totp",
            "message": "Entrez le code de votre application d'authentification"
        }
    
    async def _initiate_sms_challenge(self, user_id: str) -> Dict[str, Any]:
        """Initie un challenge SMS"""
        try:
            # Générer et envoyer code SMS
            code = self._generate_verification_code()
            await self._send_sms_code(user_id, code)
            
            # Stocker le code temporairement
            await self.redis_client.setex(
                f"sms_code:{user_id}",
                300,  # 5 minutes
                code
            )
            
            return {
                "type": "sms",
                "message": "Code de vérification envoyé par SMS"
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur challenge SMS: {exc}")
            raise
    
    async def _initiate_email_challenge(self, user_id: str) -> Dict[str, Any]:
        """Initie un challenge email"""
        try:
            # Générer et envoyer code email
            code = self._generate_verification_code()
            await self._send_email_code(user_id, code)
            
            # Stocker le code temporairement
            await self.redis_client.setex(
                f"email_code:{user_id}",
                300,  # 5 minutes
                code
            )
            
            return {
                "type": "email",
                "message": "Code de vérification envoyé par email"
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur challenge email: {exc}")
            raise
    
    async def _initiate_push_challenge(self, user_id: str) -> Dict[str, Any]:
        """Initie un challenge push notification"""
        try:
            challenge_code = secrets.token_urlsafe(16)
            
            # Envoyer push notification
            await self._send_push_notification(user_id, challenge_code)
            
            # Stocker le challenge temporairement
            await self.redis_client.setex(
                f"push_challenge:{user_id}",
                120,  # 2 minutes
                challenge_code
            )
            
            return {
                "type": "push",
                "message": "Notification push envoyée à votre appareil"
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur challenge push: {exc}")
            raise
    
    async def _verify_totp(self, user_id: str, code: str) -> bool:
        """Vérifie un code TOTP"""
        try:
            # Récupérer le secret TOTP de l'utilisateur
            secret = await self._get_totp_secret(user_id)
            if not secret:
                return False
            
            # Vérifier le code TOTP
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification TOTP: {exc}")
            return False
    
    async def _verify_sms_code(self, user_id: str, code: str) -> bool:
        """Vérifie un code SMS"""
        try:
            stored_code = await self.redis_client.get(f"sms_code:{user_id}")
            if not stored_code:
                return False
            
            return stored_code.decode() == code
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification SMS: {exc}")
            return False
    
    async def _verify_email_code(self, user_id: str, code: str) -> bool:
        """Vérifie un code email"""
        try:
            stored_code = await self.redis_client.get(f"email_code:{user_id}")
            if not stored_code:
                return False
            
            return stored_code.decode() == code
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification email: {exc}")
            return False
    
    async def _verify_push_response(self, user_id: str, response: str) -> bool:
        """Vérifie la réponse push notification"""
        try:
            stored_challenge = await self.redis_client.get(f"push_challenge:{user_id}")
            if not stored_challenge:
                return False
            
            return stored_challenge.decode() == response
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification push: {exc}")
            return False
    
    def _generate_verification_code(self, length: int = 6) -> str:
        """Génère un code de vérification numérique"""
        return ''.join([str(secrets.randbelow(10)) for _ in range(length)])
    
    # Méthodes utilitaires (implémentation simplifiée)
    async def _get_totp_secret(self, user_id: str) -> Optional[str]:
        """Récupère le secret TOTP de l'utilisateur"""
        # Implémentation avec votre base de données
        return None
    
    async def _send_sms_code(self, user_id: str, code: str):
        """Envoie un code par SMS"""
        # Implémentation avec votre service SMS
        pass
    
    async def _send_email_code(self, user_id: str, code: str):
        """Envoie un code par email"""
        # Implémentation avec votre service email
        pass
    
    async def _send_push_notification(self, user_id: str, challenge: str):
        """Envoie une push notification"""
        # Implémentation avec votre service push
        pass


class BiometricAuthenticator:
    """Authentificateur biométrique avancé"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        self.face_model = None
        self._load_models()
    
    def _load_models(self):
        """Charge les modèles de reconnaissance biométrique"""
        try:
            # Charger le modèle de reconnaissance faciale si disponible
            # self.face_model = tf.keras.models.load_model('face_recognition_model.h5')
            pass
        except Exception as exc:
            self.logger.warning(f"Modèles biométriques non disponibles: {exc}")
    
    async def register_biometric(
        self,
        user_id: str,
        biometric_data: bytes,
        biometric_type: str
    ) -> bool:
        """Enregistre des données biométriques pour un utilisateur"""
        try:
            if biometric_type == "fingerprint":
                template = await self._process_fingerprint(biometric_data)
            elif biometric_type == "face":
                template = await self._process_face(biometric_data)
            else:
                raise ValueError(f"Type biométrique non supporté: {biometric_type}")
            
            if not template:
                return False
            
            # Stocker le template biométrique (chiffré)
            biometric_record = BiometricData(
                user_id=user_id,
                biometric_type=biometric_type,
                template=template,
                confidence_score=1.0,  # Score initial
                created_at=datetime.utcnow()
            )
            
            await self._store_biometric_template(biometric_record)
            return True
            
        except Exception as exc:
            self.logger.error(f"Erreur enregistrement biométrie: {exc}")
            return False
    
    async def verify_biometric(
        self,
        user_id: str,
        biometric_data: bytes,
        biometric_type: str
    ) -> Dict[str, Any]:
        """Vérifie des données biométriques"""
        try:
            # Traiter les données biométriques
            if biometric_type == "fingerprint":
                current_template = await self._process_fingerprint(biometric_data)
            elif biometric_type == "face":
                current_template = await self._process_face(biometric_data)
            else:
                return {"verified": False, "confidence": 0.0}
            
            if not current_template:
                return {"verified": False, "confidence": 0.0}
            
            # Récupérer le template stocké
            stored_template = await self._get_biometric_template(user_id, biometric_type)
            if not stored_template:
                return {"verified": False, "confidence": 0.0}
            
            # Comparer les templates
            confidence = await self._compare_biometric_templates(
                stored_template, current_template, biometric_type
            )
            
            verified = confidence >= 0.8  # Seuil de confiance
            
            if verified:
                await self._update_biometric_usage(user_id, biometric_type)
            
            return {
                "verified": verified,
                "confidence": confidence,
                "biometric_type": biometric_type
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification biométrie: {exc}")
            return {"verified": False, "confidence": 0.0}
    
    async def _process_fingerprint(self, fingerprint_data: bytes) -> Optional[bytes]:
        """Traite et extrait les caractéristiques d'une empreinte digitale"""
        try:
            # Implémentation simplifiée - utiliser une vraie bibliothèque d'empreintes
            # comme PyFingerprint ou une solution commerciale
            
            # Simulation d'extraction de minuties
            template = hashlib.sha256(fingerprint_data).digest()
            return template
            
        except Exception as exc:
            self.logger.error(f"Erreur traitement empreinte: {exc}")
            return None
    
    async def _process_face(self, face_data: bytes) -> Optional[bytes]:
        """Traite et extrait les caractéristiques faciales"""
        try:
            # Convertir les bytes en image
            image = Image.open(io.BytesIO(face_data))
            image_array = np.array(image)
            
            # Utiliser face_recognition pour extraire les encodages
            face_encodings = face_recognition.face_encodings(image_array)
            
            if not face_encodings:
                return None
            
            # Prendre le premier visage détecté
            face_encoding = face_encodings[0]
            
            # Convertir en bytes pour stockage
            return face_encoding.tobytes()
            
        except Exception as exc:
            self.logger.error(f"Erreur traitement visage: {exc}")
            return None
    
    async def _compare_biometric_templates(
        self,
        template1: bytes,
        template2: bytes,
        biometric_type: str
    ) -> float:
        """Compare deux templates biométriques"""
        try:
            if biometric_type == "fingerprint":
                # Comparaison d'empreintes (implémentation simplifiée)
                return 1.0 if template1 == template2 else 0.0
            
            elif biometric_type == "face":
                # Comparaison de visages avec face_recognition
                encoding1 = np.frombuffer(template1, dtype=np.float64)
                encoding2 = np.frombuffer(template2, dtype=np.float64)
                
                # Calculer la distance euclidienne
                distance = np.linalg.norm(encoding1 - encoding2)
                
                # Convertir en score de confiance (plus la distance est faible, plus la confiance est élevée)
                confidence = max(0.0, 1.0 - (distance / 0.6))  # Normalisation
                return confidence
            
            return 0.0
            
        except Exception as exc:
            self.logger.error(f"Erreur comparaison templates: {exc}")
            return 0.0
    
    async def _store_biometric_template(self, biometric_record: BiometricData):
        """Stocke un template biométrique de manière sécurisée"""
        try:
            # Chiffrer le template
            encrypted_template = self._encrypt_biometric_data(biometric_record.template)
            
            # Stocker dans Redis avec une clé sécurisée
            key = f"biometric:{biometric_record.user_id}:{biometric_record.biometric_type}"
            data = {
                "template": encrypted_template.decode(),
                "confidence_score": biometric_record.confidence_score,
                "created_at": biometric_record.created_at.isoformat(),
                "last_used": None
            }
            
            await self.redis_client.set(key, json.dumps(data))
            
        except Exception as exc:
            self.logger.error(f"Erreur stockage template biométrique: {exc}")
    
    async def _get_biometric_template(
        self,
        user_id: str,
        biometric_type: str
    ) -> Optional[bytes]:
        """Récupère un template biométrique"""
        try:
            key = f"biometric:{user_id}:{biometric_type}"
            data = await self.redis_client.get(key)
            
            if not data:
                return None
            
            biometric_data = json.loads(data)
            encrypted_template = biometric_data["template"].encode()
            
            # Déchiffrer le template
            return self._decrypt_biometric_data(encrypted_template)
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération template biométrique: {exc}")
            return None
    
    async def _update_biometric_usage(self, user_id: str, biometric_type: str):
        """Met à jour la date de dernière utilisation"""
        try:
            key = f"biometric:{user_id}:{biometric_type}"
            data = await self.redis_client.get(key)
            
            if data:
                biometric_data = json.loads(data)
                biometric_data["last_used"] = datetime.utcnow().isoformat()
                await self.redis_client.set(key, json.dumps(biometric_data))
                
        except Exception as exc:
            self.logger.error(f"Erreur mise à jour usage biométrique: {exc}")
    
    def _encrypt_biometric_data(self, data: bytes) -> bytes:
        """Chiffre les données biométriques"""
        # Utiliser un chiffrement fort pour les données biométriques
        # Implémentation simplifiée
        return base64.b64encode(data)
    
    def _decrypt_biometric_data(self, encrypted_data: bytes) -> bytes:
        """Déchiffre les données biométriques"""
        # Déchiffrement correspondant
        return base64.b64decode(encrypted_data)


class RiskBasedAuthenticator:
    """Authentificateur basé sur l'analyse de risque"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        self.ml_model = None
        self._load_risk_model()
    
    def _load_risk_model(self):
        """Charge le modèle ML d'analyse de risque"""
        try:
            # Charger un modèle pré-entraîné pour l'analyse de risque
            # self.ml_model = joblib.load('risk_analysis_model.pkl')
            
            # Utiliser Isolation Forest comme modèle par défaut
            self.ml_model = IsolationForest(contamination=0.1, random_state=42)
            
        except Exception as exc:
            self.logger.warning(f"Modèle de risque non disponible: {exc}")
    
    async def analyze_authentication_risk(
        self,
        user_id: str,
        context: AuthenticationContext
    ) -> Dict[str, Any]:
        """Analyse le risque d'une tentative d'authentification"""
        try:
            risk_factors = await self._extract_risk_factors(user_id, context)
            risk_score = await self._calculate_risk_score(risk_factors)
            risk_level = self._determine_risk_level(risk_score)
            
            # Enregistrer l'analyse pour l'apprentissage
            await self._record_risk_analysis(user_id, context, risk_score)
            
            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommendations": self._get_risk_recommendations(risk_level)
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur analyse de risque: {exc}")
            return {
                "risk_score": 0.5,  # Risque moyen par défaut
                "risk_level": RiskLevel.MEDIUM,
                "risk_factors": {},
                "recommendations": []
            }
    
    async def analyze_biometric_risk(
        self,
        user_id: str,
        context: AuthenticationContext,
        biometric_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyse le risque spécifique à l'authentification biométrique"""
        try:
            risk_factors = await self._extract_biometric_risk_factors(
                user_id, context, biometric_result
            )
            
            risk_score = await self._calculate_biometric_risk_score(risk_factors)
            risk_level = self._determine_risk_level(risk_score)
            
            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur analyse risque biométrique: {exc}")
            return {
                "risk_score": 0.5,
                "risk_level": RiskLevel.MEDIUM,
                "risk_factors": {}
            }
    
    async def _extract_risk_factors(
        self,
        user_id: str,
        context: AuthenticationContext
    ) -> Dict[str, float]:
        """Extrait les facteurs de risque"""
        try:
            risk_factors = {}
            
            # Facteur temporel (heure inhabituelle)
            risk_factors["time_anomaly"] = await self._analyze_time_pattern(user_id, context)
            
            # Facteur géographique (localisation inhabituelle)
            risk_factors["location_anomaly"] = await self._analyze_location_pattern(user_id, context)
            
            # Facteur dispositif (nouveau dispositif)
            risk_factors["device_anomaly"] = await self._analyze_device_pattern(user_id, context)
            
            # Facteur comportemental (pattern d'usage)
            risk_factors["behavior_anomaly"] = await self._analyze_behavior_pattern(user_id, context)
            
            # Facteur réseau (IP suspecte)
            risk_factors["network_risk"] = await self._analyze_network_risk(context)
            
            # Facteur fréquence (tentatives répétées)
            risk_factors["frequency_risk"] = await self._analyze_frequency_risk(user_id, context)
            
            return risk_factors
            
        except Exception as exc:
            self.logger.error(f"Erreur extraction facteurs de risque: {exc}")
            return {}
    
    async def _calculate_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Calcule le score de risque global"""
        try:
            if not risk_factors:
                return 0.5  # Risque moyen par défaut
            
            # Pondération des facteurs
            weights = {
                "time_anomaly": 0.15,
                "location_anomaly": 0.25,
                "device_anomaly": 0.20,
                "behavior_anomaly": 0.20,
                "network_risk": 0.15,
                "frequency_risk": 0.05
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for factor, score in risk_factors.items():
                if factor in weights:
                    weighted_score += score * weights[factor]
                    total_weight += weights[factor]
            
            if total_weight > 0:
                return min(weighted_score / total_weight, 1.0)
            else:
                return 0.5
                
        except Exception as exc:
            self.logger.error(f"Erreur calcul score de risque: {exc}")
            return 0.5
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Détermine le niveau de risque"""
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _get_risk_recommendations(self, risk_level: RiskLevel) -> List[str]:
        """Fournit des recommandations basées sur le niveau de risque"""
        recommendations = {
            RiskLevel.LOW: [
                "Authentification standard suffisante"
            ],
            RiskLevel.MEDIUM: [
                "Considérer l'authentification multi-facteurs",
                "Surveiller l'activité de session"
            ],
            RiskLevel.HIGH: [
                "Exiger l'authentification multi-facteurs",
                "Limiter les privilèges de session",
                "Notification de sécurité à l'utilisateur"
            ],
            RiskLevel.CRITICAL: [
                "Bloquer l'authentification",
                "Exiger une vérification manuelle",
                "Alerte de sécurité immédiate",
                "Investigation de sécurité"
            ]
        }
        
        return recommendations.get(risk_level, [])
    
    # Méthodes d'analyse des patterns (implémentation simplifiée)
    async def _analyze_time_pattern(self, user_id: str, context: AuthenticationContext) -> float:
        """Analyse les patterns temporels"""
        try:
            # Récupérer l'historique des connexions
            historical_times = await self._get_historical_login_times(user_id)
            
            if not historical_times:
                return 0.0  # Pas d'historique, pas d'anomalie
            
            current_hour = context.timestamp.hour
            
            # Calculer la fréquence de connexion à cette heure
            hour_frequency = sum(1 for t in historical_times if t.hour == current_hour)
            total_logins = len(historical_times)
            
            if total_logins == 0:
                return 0.0
            
            frequency_ratio = hour_frequency / total_logins
            
            # Plus la fréquence est faible, plus le risque est élevé
            return max(0.0, 1.0 - (frequency_ratio * 4))  # Normalisation
            
        except Exception as exc:
            self.logger.error(f"Erreur analyse pattern temporel: {exc}")
            return 0.0
    
    async def _analyze_location_pattern(self, user_id: str, context: AuthenticationContext) -> float:
        """Analyse les patterns de localisation"""
        try:
            # Récupérer l'historique des localisations
            historical_locations = await self._get_historical_locations(user_id)
            
            if not historical_locations:
                return 0.0
            
            # Comparer avec la localisation actuelle
            current_location = context.location
            if not current_location:
                return 0.3  # Risque modéré si pas de géolocalisation
            
            # Vérifier si la localisation est dans les locations habituelles
            for hist_location in historical_locations:
                distance = self._calculate_location_distance(current_location, hist_location)
                if distance < 100:  # Moins de 100 km
                    return 0.0  # Location familière
            
            return 0.8  # Location inconnue = risque élevé
            
        except Exception as exc:
            self.logger.error(f"Erreur analyse pattern géographique: {exc}")
            return 0.0
    
    async def _analyze_device_pattern(self, user_id: str, context: AuthenticationContext) -> float:
        """Analyse les patterns de dispositifs"""
        try:
            # Récupérer l'historique des dispositifs
            known_devices = await self._get_known_devices(user_id)
            
            current_fingerprint = context.device_fingerprint
            
            if current_fingerprint in known_devices:
                return 0.0  # Dispositif connu
            else:
                return 0.6  # Nouveau dispositif = risque modéré-élevé
                
        except Exception as exc:
            self.logger.error(f"Erreur analyse pattern dispositif: {exc}")
            return 0.0
    
    async def _analyze_behavior_pattern(self, user_id: str, context: AuthenticationContext) -> float:
        """Analyse les patterns comportementaux"""
        try:
            # Implémentation simplifiée - analyser les patterns d'usage
            # Dans une vraie implémentation, utiliser du ML pour détecter les anomalies
            
            user_activities = await self._get_user_activities(user_id)
            
            if len(user_activities) < 10:
                return 0.0  # Pas assez de données
            
            # Utiliser le modèle ML si disponible
            if self.ml_model:
                features = self._extract_behavioral_features(user_activities, context)
                anomaly_score = self.ml_model.decision_function([features])[0]
                # Convertir en score de risque (0-1)
                return max(0.0, min(1.0, -anomaly_score))
            
            return 0.0
            
        except Exception as exc:
            self.logger.error(f"Erreur analyse pattern comportemental: {exc}")
            return 0.0
    
    async def _analyze_network_risk(self, context: AuthenticationContext) -> float:
        """Analyse les risques réseau"""
        try:
            ip_address = context.ip_address
            
            # Vérifier les listes noires
            if await self._is_ip_blacklisted(ip_address):
                return 1.0
            
            # Vérifier si c'est un VPN/Proxy
            if await self._is_vpn_or_proxy(ip_address):
                return 0.4
            
            # Vérifier la réputation de l'IP
            reputation_score = await self._get_ip_reputation(ip_address)
            return max(0.0, 1.0 - reputation_score)
            
        except Exception as exc:
            self.logger.error(f"Erreur analyse risque réseau: {exc}")
            return 0.0
    
    async def _analyze_frequency_risk(self, user_id: str, context: AuthenticationContext) -> float:
        """Analyse les risques de fréquence"""
        try:
            # Compter les tentatives récentes
            recent_attempts = await self._count_recent_attempts(user_id, minutes=15)
            
            if recent_attempts > 10:
                return 1.0  # Très suspect
            elif recent_attempts > 5:
                return 0.6  # Modérément suspect
            elif recent_attempts > 3:
                return 0.3  # Légèrement suspect
            else:
                return 0.0  # Normal
                
        except Exception as exc:
            self.logger.error(f"Erreur analyse risque fréquence: {exc}")
            return 0.0
    
    # Méthodes utilitaires (implémentation simplifiée)
    async def _get_historical_login_times(self, user_id: str) -> List[datetime]:
        """Récupère l'historique des heures de connexion"""
        # Implémentation avec votre base de données
        return []
    
    async def _get_historical_locations(self, user_id: str) -> List[Dict[str, str]]:
        """Récupère l'historique des localisations"""
        # Implémentation avec votre base de données
        return []
    
    async def _get_known_devices(self, user_id: str) -> List[str]:
        """Récupère les dispositifs connus"""
        # Implémentation avec votre base de données
        return []
    
    async def _get_user_activities(self, user_id: str) -> List[Dict]:
        """Récupère les activités utilisateur"""
        try:
            key = f"user_activity:{user_id}"
            activities = await self.redis_client.lrange(key, 0, -1)
            return [json.loads(activity) for activity in activities]
        except Exception:
            return []
    
    def _calculate_location_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calcule la distance entre deux localisations"""
        # Implémentation simplifiée - utiliser haversine dans la vraie version
        return 0.0
    
    def _extract_behavioral_features(self, activities: List[Dict], context: AuthenticationContext) -> List[float]:
        """Extrait les caractéristiques comportementales"""
        # Implémentation simplifiée - extraire des features pour le ML
        return [0.0] * 10  # Placeholder
    
    async def _is_ip_blacklisted(self, ip_address: str) -> bool:
        """Vérifie si l'IP est blacklistée"""
        try:
            return await self.redis_client.sismember("blacklisted_ips", ip_address)
        except Exception:
            return False
    
    async def _is_vpn_or_proxy(self, ip_address: str) -> bool:
        """Vérifie si l'IP est un VPN/Proxy"""
        # Implémentation avec un service de détection VPN/Proxy
        return False
    
    async def _get_ip_reputation(self, ip_address: str) -> float:
        """Récupère la réputation d'une IP"""
        # Implémentation avec un service de réputation IP
        return 1.0  # Bonne réputation par défaut
    
    async def _count_recent_attempts(self, user_id: str, minutes: int) -> int:
        """Compte les tentatives récentes"""
        try:
            key = f"auth_attempts:{user_id}"
            current_time = int(time.time())
            cutoff_time = current_time - (minutes * 60)
            
            return await self.redis_client.zcount(key, cutoff_time, current_time)
        except Exception:
            return 0
    
    async def _record_risk_analysis(
        self,
        user_id: str,
        context: AuthenticationContext,
        risk_score: float
    ):
        """Enregistre l'analyse de risque pour l'apprentissage"""
        try:
            analysis_data = {
                "user_id": user_id,
                "timestamp": context.timestamp.isoformat(),
                "risk_score": risk_score,
                "ip_address": context.ip_address,
                "device_fingerprint": context.device_fingerprint
            }
            
            # Stocker pour analyse ultérieure
            key = f"risk_analysis:{user_id}"
            await self.redis_client.lpush(key, json.dumps(analysis_data))
            await self.redis_client.ltrim(key, 0, 999)  # Garder les 1000 dernières
            await self.redis_client.expire(key, 86400 * 90)  # 90 jours
            
        except Exception as exc:
            self.logger.error(f"Erreur enregistrement analyse risque: {exc}")
    
    async def _extract_biometric_risk_factors(
        self,
        user_id: str,
        context: AuthenticationContext,
        biometric_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extrait les facteurs de risque spécifiques à la biométrie"""
        try:
            risk_factors = {}
            
            # Facteur de confiance biométrique
            confidence = biometric_result.get("confidence", 0.0)
            risk_factors["low_confidence"] = max(0.0, 1.0 - confidence)
            
            # Facteur de tentatives biométriques répétées
            recent_biometric_attempts = await self._count_recent_biometric_attempts(user_id)
            risk_factors["repeated_attempts"] = min(1.0, recent_biometric_attempts / 10.0)
            
            # Facteur de nouveau type biométrique
            biometric_type = biometric_result.get("biometric_type")
            if not await self._has_used_biometric_type_before(user_id, biometric_type):
                risk_factors["new_biometric_type"] = 0.3
            else:
                risk_factors["new_biometric_type"] = 0.0
            
            return risk_factors
            
        except Exception as exc:
            self.logger.error(f"Erreur extraction facteurs risque biométrique: {exc}")
            return {}
    
    async def _calculate_biometric_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Calcule le score de risque pour l'authentification biométrique"""
        try:
            if not risk_factors:
                return 0.0
            
            # Pondération spécifique à la biométrie
            weights = {
                "low_confidence": 0.5,
                "repeated_attempts": 0.3,
                "new_biometric_type": 0.2
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for factor, score in risk_factors.items():
                if factor in weights:
                    weighted_score += score * weights[factor]
                    total_weight += weights[factor]
            
            if total_weight > 0:
                return min(weighted_score / total_weight, 1.0)
            else:
                return 0.0
                
        except Exception as exc:
            self.logger.error(f"Erreur calcul score risque biométrique: {exc}")
            return 0.0
    
    async def _count_recent_biometric_attempts(self, user_id: str) -> int:
        """Compte les tentatives biométriques récentes"""
        try:
            key = f"biometric_attempts:{user_id}"
            current_time = int(time.time())
            one_hour_ago = current_time - 3600
            
            return await self.redis_client.zcount(key, one_hour_ago, current_time)
        except Exception:
            return 0
    
    async def _has_used_biometric_type_before(self, user_id: str, biometric_type: str) -> bool:
        """Vérifie si l'utilisateur a déjà utilisé ce type de biométrie"""
        try:
            key = f"biometric:{user_id}:{biometric_type}"
            return await self.redis_client.exists(key)
        except Exception:
            return False
