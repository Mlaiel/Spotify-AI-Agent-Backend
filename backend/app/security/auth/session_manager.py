# 🔐 Session Management & Device Trust
# ===================================
# 
# Gestionnaire de sessions sécurisées et gestion
# de confiance des dispositifs pour l'enterprise.
#
# 🎖️ Expert: Lead Dev + Architecte IA + Spécialiste Sécurité Backend
#
# Développé par l'équipe d'experts enterprise
# ===================================

"""
🔐 Enterprise Session & Device Management
=========================================

Advanced session and device management providing:
- Secure session management with Redis backend
- Device fingerprinting and trust management
- Session hijacking detection and prevention
- Multi-device session synchronization
- Device-based authentication policies
- Session analytics and monitoring
- Automatic session cleanup and expiration
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, asdict
import redis
import logging
from fastapi import Request, HTTPException
import geoip2.database
import geoip2.errors
import user_agents
from cryptography.fernet import Fernet
import base64

# Configuration et logging
logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Statuts de session"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPICIOUS = "suspicious"


class DeviceStatus(Enum):
    """Statuts de dispositif"""
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    BLOCKED = "blocked"
    PENDING = "pending"


class SessionRiskLevel(Enum):
    """Niveaux de risque de session"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DeviceInfo:
    """Informations sur un dispositif"""
    device_id: str
    user_id: str
    device_fingerprint: str
    device_name: str
    platform: str
    browser: str
    os_name: str
    os_version: str
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    status: DeviceStatus = DeviceStatus.UNTRUSTED
    first_seen: datetime = None
    last_seen: datetime = None
    trust_score: float = 0.0
    location_history: List[Dict] = None
    
    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = datetime.utcnow()
        if self.last_seen is None:
            self.last_seen = datetime.utcnow()
        if self.location_history is None:
            self.location_history = []


@dataclass
class SessionInfo:
    """Informations de session"""
    session_id: str
    user_id: str
    device_id: str
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    status: SessionStatus = SessionStatus.ACTIVE
    risk_level: SessionRiskLevel = SessionRiskLevel.LOW
    location: Optional[Dict] = None
    authentication_methods: List[str] = None
    permissions: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.authentication_methods is None:
            self.authentication_methods = []
        if self.permissions is None:
            self.permissions = []
        if self.metadata is None:
            self.metadata = {}


class SecureSessionManager:
    """Gestionnaire de sessions sécurisées"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        self.device_manager = DeviceManager(redis_client)
        self.fernet = self._initialize_encryption()
        
        # Configuration
        self.default_session_timeout = 86400  # 24 heures
        self.max_sessions_per_user = 10
        self.session_extend_threshold = 1800  # 30 minutes
        self.suspicious_activity_threshold = 5
        
        # GeoIP pour la géolocalisation
        self.geoip_reader = None
        try:
            self.geoip_reader = geoip2.database.Reader('/usr/share/GeoIP/GeoLite2-City.mmdb')
        except Exception:
            self.logger.warning("Base de données GeoIP non disponible")
    
    def _initialize_encryption(self) -> Fernet:
        """Initialise le chiffrement pour les données de session"""
        try:
            key = Fernet.generate_key()
            return Fernet(key)
        except Exception as exc:
            self.logger.error(f"Erreur initialisation chiffrement: {exc}")
            raise
    
    async def create_session(
        self,
        user_id: str,
        request: Request,
        authentication_methods: List[str] = None,
        permissions: List[str] = None,
        custom_timeout: Optional[int] = None
    ) -> SessionInfo:
        """Crée une nouvelle session sécurisée"""
        try:
            # Générer un ID de session unique
            session_id = self._generate_session_id()
            
            # Extraire les informations du dispositif
            device_info = await self._extract_device_info(request)
            
            # Enregistrer ou mettre à jour le dispositif
            device = await self.device_manager.register_or_update_device(
                user_id, device_info
            )
            
            # Déterminer la géolocalisation
            location = await self._get_location_from_ip(request.client.host)
            
            # Calculer le niveau de risque initial
            risk_level = await self._assess_session_risk(
                user_id, device, location, request
            )
            
            # Créer la session
            timeout = custom_timeout or self.default_session_timeout
            session = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                device_id=device.device_id,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent", ""),
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=timeout),
                status=SessionStatus.ACTIVE,
                risk_level=risk_level,
                location=location,
                authentication_methods=authentication_methods or [],
                permissions=permissions or []
            )
            
            # Vérifier les limites de sessions
            await self._enforce_session_limits(user_id)
            
            # Stocker la session
            await self._store_session(session)
            
            # Enregistrer l'activité
            await self._log_session_activity(session, "session_created")
            
            self.logger.info(f"Session créée: {session_id} pour utilisateur {user_id}")
            return session
            
        except Exception as exc:
            self.logger.error(f"Erreur création session: {exc}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Récupère une session par son ID"""
        try:
            session_data = await self.redis_client.get(f"session:{session_id}")
            if not session_data:
                return None
            
            # Déchiffrer et désérialiser
            decrypted_data = self.fernet.decrypt(session_data)
            session_dict = json.loads(decrypted_data)
            
            return SessionInfo(**session_dict)
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération session {session_id}: {exc}")
            return None
    
    async def validate_session(
        self,
        session_id: str,
        request: Request,
        required_permissions: List[str] = None
    ) -> Tuple[bool, Optional[SessionInfo], Optional[str]]:
        """Valide une session et détecte les anomalies"""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False, None, "Session introuvable"
            
            # Vérifier l'expiration
            if datetime.utcnow() > session.expires_at:
                await self.revoke_session(session_id, "expired")
                return False, session, "Session expirée"
            
            # Vérifier le statut
            if session.status != SessionStatus.ACTIVE:
                return False, session, f"Session {session.status.value}"
            
            # Détection de hijacking
            hijacking_detected = await self._detect_session_hijacking(session, request)
            if hijacking_detected:
                await self.revoke_session(session_id, "hijacking_detected")
                return False, session, "Activité suspecte détectée"
            
            # Vérifier les permissions
            if required_permissions:
                missing_permissions = set(required_permissions) - set(session.permissions)
                if missing_permissions:
                    return False, session, f"Permissions manquantes: {missing_permissions}"
            
            # Mettre à jour l'activité
            await self._update_session_activity(session)
            
            return True, session, None
            
        except Exception as exc:
            self.logger.error(f"Erreur validation session {session_id}: {exc}")
            return False, None, "Erreur interne"
    
    async def revoke_session(self, session_id: str, reason: str = "manual"):
        """Révoque une session"""
        try:
            session = await self.get_session(session_id)
            if session:
                session.status = SessionStatus.REVOKED
                await self._store_session(session)
                
                # Log de révocation
                await self._log_session_activity(session, "session_revoked", {"reason": reason})
                
                self.logger.info(f"Session révoquée: {session_id}, raison: {reason}")
            
            # Supprimer de Redis
            await self.redis_client.delete(f"session:{session_id}")
            
        except Exception as exc:
            self.logger.error(f"Erreur révocation session {session_id}: {exc}")
    
    async def revoke_user_sessions(
        self,
        user_id: str,
        exclude_session_id: Optional[str] = None,
        reason: str = "manual"
    ):
        """Révoque toutes les sessions d'un utilisateur"""
        try:
            # Récupérer toutes les sessions de l'utilisateur
            session_ids = await self._get_user_session_ids(user_id)
            
            for session_id in session_ids:
                if exclude_session_id and session_id == exclude_session_id:
                    continue
                
                await self.revoke_session(session_id, reason)
            
            self.logger.info(f"Sessions utilisateur révoquées: {user_id}, raison: {reason}")
            
        except Exception as exc:
            self.logger.error(f"Erreur révocation sessions utilisateur {user_id}: {exc}")
    
    async def extend_session(self, session_id: str, additional_time: Optional[int] = None):
        """Étend la durée d'une session"""
        try:
            session = await self.get_session(session_id)
            if not session or session.status != SessionStatus.ACTIVE:
                return False
            
            # Vérifier si l'extension est autorisée
            time_until_expiry = (session.expires_at - datetime.utcnow()).total_seconds()
            if time_until_expiry > self.session_extend_threshold:
                return False  # Trop tôt pour étendre
            
            # Étendre la session
            extension = additional_time or self.default_session_timeout
            session.expires_at = datetime.utcnow() + timedelta(seconds=extension)
            
            await self._store_session(session)
            await self._log_session_activity(session, "session_extended")
            
            return True
            
        except Exception as exc:
            self.logger.error(f"Erreur extension session {session_id}: {exc}")
            return False
    
    async def get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Récupère toutes les sessions actives d'un utilisateur"""
        try:
            session_ids = await self._get_user_session_ids(user_id)
            sessions = []
            
            for session_id in session_ids:
                session = await self.get_session(session_id)
                if session and session.status == SessionStatus.ACTIVE:
                    sessions.append(session)
            
            return sessions
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération sessions utilisateur {user_id}: {exc}")
            return []
    
    async def cleanup_expired_sessions(self):
        """Nettoie les sessions expirées"""
        try:
            # Cette tâche devrait être exécutée périodiquement
            current_time = datetime.utcnow()
            
            # Scanner les sessions pour trouver les expirées
            # Implémentation simplifiée - dans un vrai système, utiliser un index
            pattern = "session:*"
            session_keys = await self.redis_client.keys(pattern)
            
            expired_count = 0
            for key in session_keys:
                try:
                    session_id = key.decode().replace("session:", "")
                    session = await self.get_session(session_id)
                    
                    if session and current_time > session.expires_at:
                        await self.revoke_session(session_id, "expired")
                        expired_count += 1
                        
                except Exception:
                    continue
            
            self.logger.info(f"Sessions expirées nettoyées: {expired_count}")
            return expired_count
            
        except Exception as exc:
            self.logger.error(f"Erreur nettoyage sessions expirées: {exc}")
            return 0
    
    # Méthodes privées
    def _generate_session_id(self) -> str:
        """Génère un ID de session unique et sécurisé"""
        timestamp = str(int(time.time()))
        random_bytes = secrets.token_bytes(16)
        combined = timestamp.encode() + random_bytes
        return hashlib.sha256(combined).hexdigest()
    
    async def _extract_device_info(self, request: Request) -> Dict[str, Any]:
        """Extrait les informations du dispositif depuis la requête"""
        try:
            user_agent_string = request.headers.get("user-agent", "")
            user_agent = user_agents.parse(user_agent_string)
            
            # Créer une empreinte du dispositif
            device_fingerprint = await self._create_device_fingerprint(request)
            
            return {
                "user_agent": user_agent_string,
                "device_fingerprint": device_fingerprint,
                "platform": user_agent.os.family,
                "browser": user_agent.browser.family,
                "os_name": user_agent.os.family,
                "os_version": user_agent.os.version_string,
                "device_name": f"{user_agent.device.family} {user_agent.browser.family}",
                "screen_resolution": request.headers.get("x-screen-resolution"),
                "timezone": request.headers.get("x-timezone"),
                "language": request.headers.get("accept-language", "").split(",")[0]
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur extraction info dispositif: {exc}")
            return {}
    
    async def _create_device_fingerprint(self, request: Request) -> str:
        """Crée une empreinte unique du dispositif"""
        try:
            # Éléments pour l'empreinte
            elements = [
                request.headers.get("user-agent", ""),
                request.headers.get("accept", ""),
                request.headers.get("accept-language", ""),
                request.headers.get("accept-encoding", ""),
                request.headers.get("x-screen-resolution", ""),
                request.headers.get("x-timezone", ""),
                request.headers.get("x-color-depth", ""),
                request.headers.get("x-platform", "")
            ]
            
            # Combiner et hacher
            combined = "|".join(elements)
            return hashlib.sha256(combined.encode()).hexdigest()
            
        except Exception as exc:
            self.logger.error(f"Erreur création empreinte dispositif: {exc}")
            return "unknown"
    
    async def _get_location_from_ip(self, ip_address: str) -> Optional[Dict[str, str]]:
        """Récupère la géolocalisation depuis l'IP"""
        try:
            if not self.geoip_reader:
                return None
            
            response = self.geoip_reader.city(ip_address)
            return {
                "country": response.country.name,
                "country_code": response.country.iso_code,
                "city": response.city.name,
                "latitude": str(response.location.latitude),
                "longitude": str(response.location.longitude)
            }
            
        except (geoip2.errors.AddressNotFoundError, Exception):
            return None
    
    async def _assess_session_risk(
        self,
        user_id: str,
        device: DeviceInfo,
        location: Optional[Dict],
        request: Request
    ) -> SessionRiskLevel:
        """Évalue le niveau de risque d'une session"""
        try:
            risk_score = 0.0
            
            # Risque basé sur le dispositif
            if device.status == DeviceStatus.UNTRUSTED:
                risk_score += 0.3
            elif device.status == DeviceStatus.BLOCKED:
                risk_score += 1.0
            
            # Risque basé sur la géolocalisation
            if location:
                is_new_location = await self._is_new_location_for_user(user_id, location)
                if is_new_location:
                    risk_score += 0.4
            
            # Risque basé sur l'heure
            current_hour = datetime.utcnow().hour
            is_unusual_time = await self._is_unusual_login_time(user_id, current_hour)
            if is_unusual_time:
                risk_score += 0.2
            
            # Risque basé sur la fréquence
            recent_failed_attempts = await self._count_recent_failed_attempts(user_id)
            if recent_failed_attempts > 3:
                risk_score += 0.3
            
            # Convertir en niveau de risque
            if risk_score < 0.3:
                return SessionRiskLevel.LOW
            elif risk_score < 0.6:
                return SessionRiskLevel.MEDIUM
            elif risk_score < 0.8:
                return SessionRiskLevel.HIGH
            else:
                return SessionRiskLevel.CRITICAL
                
        except Exception as exc:
            self.logger.error(f"Erreur évaluation risque session: {exc}")
            return SessionRiskLevel.MEDIUM
    
    async def _detect_session_hijacking(self, session: SessionInfo, request: Request) -> bool:
        """Détecte les tentatives de détournement de session"""
        try:
            # Vérifier l'IP
            if session.ip_address != request.client.host:
                # Permettre quelques changements d'IP (mobile, etc.)
                ip_changes = await self._count_recent_ip_changes(session.session_id)
                if ip_changes > 3:
                    return True
            
            # Vérifier l'User-Agent
            current_ua = request.headers.get("user-agent", "")
            if session.user_agent != current_ua:
                # Changement d'User-Agent suspect
                ua_changes = await self._count_recent_ua_changes(session.session_id)
                if ua_changes > 2:
                    return True
            
            # Vérifier la géolocalisation
            current_location = await self._get_location_from_ip(request.client.host)
            if current_location and session.location:
                distance = self._calculate_distance(session.location, current_location)
                time_diff = (datetime.utcnow() - session.last_activity).total_seconds()
                
                # Impossible de voyager si vite (plus de 1000 km/h)
                if distance > 0 and time_diff > 0:
                    speed = distance / (time_diff / 3600)  # km/h
                    if speed > 1000:
                        return True
            
            return False
            
        except Exception as exc:
            self.logger.error(f"Erreur détection hijacking: {exc}")
            return False
    
    async def _update_session_activity(self, session: SessionInfo):
        """Met à jour l'activité de la session"""
        try:
            session.last_activity = datetime.utcnow()
            await self._store_session(session)
            
            # Enregistrer l'activité pour analyse
            await self._log_session_activity(session, "activity_update")
            
        except Exception as exc:
            self.logger.error(f"Erreur mise à jour activité session: {exc}")
    
    async def _store_session(self, session: SessionInfo):
        """Stocke une session dans Redis"""
        try:
            # Sérialiser et chiffrer
            session_data = json.dumps(asdict(session), default=str)
            encrypted_data = self.fernet.encrypt(session_data.encode())
            
            # Calculer le TTL
            ttl = int((session.expires_at - datetime.utcnow()).total_seconds())
            if ttl > 0:
                await self.redis_client.setex(
                    f"session:{session.session_id}",
                    ttl,
                    encrypted_data
                )
                
                # Ajouter à l'index utilisateur
                await self.redis_client.sadd(
                    f"user_sessions:{session.user_id}",
                    session.session_id
                )
            
        except Exception as exc:
            self.logger.error(f"Erreur stockage session: {exc}")
    
    async def _enforce_session_limits(self, user_id: str):
        """Applique les limites de sessions par utilisateur"""
        try:
            session_ids = await self._get_user_session_ids(user_id)
            
            if len(session_ids) >= self.max_sessions_per_user:
                # Révoquer les sessions les plus anciennes
                sessions_to_revoke = len(session_ids) - self.max_sessions_per_user + 1
                
                for i in range(sessions_to_revoke):
                    if session_ids:
                        oldest_session_id = session_ids.pop(0)
                        await self.revoke_session(oldest_session_id, "session_limit_exceeded")
            
        except Exception as exc:
            self.logger.error(f"Erreur application limites sessions: {exc}")
    
    async def _get_user_session_ids(self, user_id: str) -> List[str]:
        """Récupère les IDs de session d'un utilisateur"""
        try:
            session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")
            return [sid.decode() if isinstance(sid, bytes) else sid for sid in session_ids]
        except Exception:
            return []
    
    async def _log_session_activity(
        self,
        session: SessionInfo,
        action: str,
        metadata: Dict[str, Any] = None
    ):
        """Enregistre l'activité de session pour audit"""
        try:
            activity_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session.session_id,
                "user_id": session.user_id,
                "action": action,
                "ip_address": session.ip_address,
                "device_id": session.device_id,
                "risk_level": session.risk_level.value,
                "metadata": metadata or {}
            }
            
            # Stocker dans Redis pour analyse
            await self.redis_client.lpush(
                f"session_activity:{session.user_id}",
                json.dumps(activity_log)
            )
            
            # Limiter l'historique (garder les 1000 dernières)
            await self.redis_client.ltrim(f"session_activity:{session.user_id}", 0, 999)
            
        except Exception as exc:
            self.logger.error(f"Erreur log activité session: {exc}")
    
    # Méthodes utilitaires (implémentation simplifiée)
    async def _is_new_location_for_user(self, user_id: str, location: Dict) -> bool:
        """Vérifie si c'est une nouvelle localisation pour l'utilisateur"""
        # Implémentation simplifiée
        return False
    
    async def _is_unusual_login_time(self, user_id: str, hour: int) -> bool:
        """Vérifie si l'heure de connexion est inhabituelle"""
        # Implémentation simplifiée
        return False
    
    async def _count_recent_failed_attempts(self, user_id: str) -> int:
        """Compte les tentatives d'authentification échouées récentes"""
        # Implémentation simplifiée
        return 0
    
    async def _count_recent_ip_changes(self, session_id: str) -> int:
        """Compte les changements d'IP récents pour une session"""
        # Implémentation simplifiée
        return 0
    
    async def _count_recent_ua_changes(self, session_id: str) -> int:
        """Compte les changements d'User-Agent récents"""
        # Implémentation simplifiée
        return 0
    
    def _calculate_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calcule la distance entre deux localisations (en km)"""
        # Implémentation simplifiée - utiliser haversine dans la vraie version
        return 0.0


class DeviceManager:
    """Gestionnaire de dispositifs et confiance"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.trust_threshold = 0.7
        self.device_expiry = 86400 * 90  # 90 jours d'inactivité
    
    async def register_or_update_device(
        self,
        user_id: str,
        device_info: Dict[str, Any]
    ) -> DeviceInfo:
        """Enregistre ou met à jour un dispositif"""
        try:
            device_fingerprint = device_info.get("device_fingerprint", "")
            
            # Chercher un dispositif existant
            existing_device = await self._find_device_by_fingerprint(user_id, device_fingerprint)
            
            if existing_device:
                # Mettre à jour le dispositif existant
                existing_device.last_seen = datetime.utcnow()
                existing_device.platform = device_info.get("platform", existing_device.platform)
                existing_device.browser = device_info.get("browser", existing_device.browser)
                existing_device.os_name = device_info.get("os_name", existing_device.os_name)
                existing_device.os_version = device_info.get("os_version", existing_device.os_version)
                
                # Recalculer le score de confiance
                existing_device.trust_score = await self._calculate_trust_score(existing_device)
                
                await self._store_device(existing_device)
                return existing_device
            
            else:
                # Créer un nouveau dispositif
                device_id = self._generate_device_id(user_id, device_fingerprint)
                
                device = DeviceInfo(
                    device_id=device_id,
                    user_id=user_id,
                    device_fingerprint=device_fingerprint,
                    device_name=device_info.get("device_name", "Dispositif inconnu"),
                    platform=device_info.get("platform", ""),
                    browser=device_info.get("browser", ""),
                    os_name=device_info.get("os_name", ""),
                    os_version=device_info.get("os_version", ""),
                    screen_resolution=device_info.get("screen_resolution"),
                    timezone=device_info.get("timezone"),
                    language=device_info.get("language"),
                    status=DeviceStatus.UNTRUSTED
                )
                
                # Calculer le score de confiance initial
                device.trust_score = await self._calculate_trust_score(device)
                
                await self._store_device(device)
                
                self.logger.info(f"Nouveau dispositif enregistré: {device_id} pour utilisateur {user_id}")
                return device
                
        except Exception as exc:
            self.logger.error(f"Erreur enregistrement dispositif: {exc}")
            raise
    
    async def trust_device(self, user_id: str, device_id: str) -> bool:
        """Marque un dispositif comme digne de confiance"""
        try:
            device = await self._get_device(device_id)
            if not device or device.user_id != user_id:
                return False
            
            device.status = DeviceStatus.TRUSTED
            device.trust_score = 1.0
            
            await self._store_device(device)
            
            self.logger.info(f"Dispositif marqué comme fiable: {device_id}")
            return True
            
        except Exception as exc:
            self.logger.error(f"Erreur approbation dispositif {device_id}: {exc}")
            return False
    
    async def block_device(self, user_id: str, device_id: str) -> bool:
        """Bloque un dispositif"""
        try:
            device = await self._get_device(device_id)
            if not device or device.user_id != user_id:
                return False
            
            device.status = DeviceStatus.BLOCKED
            device.trust_score = 0.0
            
            await self._store_device(device)
            
            self.logger.info(f"Dispositif bloqué: {device_id}")
            return True
            
        except Exception as exc:
            self.logger.error(f"Erreur blocage dispositif {device_id}: {exc}")
            return False
    
    async def get_user_devices(self, user_id: str) -> List[DeviceInfo]:
        """Récupère tous les dispositifs d'un utilisateur"""
        try:
            device_ids = await self.redis_client.smembers(f"user_devices:{user_id}")
            devices = []
            
            for device_id in device_ids:
                device_id_str = device_id.decode() if isinstance(device_id, bytes) else device_id
                device = await self._get_device(device_id_str)
                if device:
                    devices.append(device)
            
            return devices
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération dispositifs utilisateur {user_id}: {exc}")
            return []
    
    async def cleanup_stale_devices(self):
        """Nettoie les dispositifs obsolètes"""
        try:
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(seconds=self.device_expiry)
            
            # Scanner tous les dispositifs (implémentation simplifiée)
            pattern = "device:*"
            device_keys = await self.redis_client.keys(pattern)
            
            cleaned_count = 0
            for key in device_keys:
                try:
                    device_id = key.decode().replace("device:", "")
                    device = await self._get_device(device_id)
                    
                    if device and device.last_seen < cutoff_time:
                        await self._delete_device(device)
                        cleaned_count += 1
                        
                except Exception:
                    continue
            
            self.logger.info(f"Dispositifs obsolètes nettoyés: {cleaned_count}")
            return cleaned_count
            
        except Exception as exc:
            self.logger.error(f"Erreur nettoyage dispositifs obsolètes: {exc}")
            return 0
    
    # Méthodes privées
    def _generate_device_id(self, user_id: str, device_fingerprint: str) -> str:
        """Génère un ID unique pour un dispositif"""
        combined = f"{user_id}:{device_fingerprint}:{int(time.time())}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    async def _calculate_trust_score(self, device: DeviceInfo) -> float:
        """Calcule le score de confiance d'un dispositif"""
        try:
            score = 0.0
            
            # Score basé sur l'âge du dispositif
            device_age_days = (datetime.utcnow() - device.first_seen).days
            age_score = min(1.0, device_age_days / 30.0)  # Max après 30 jours
            score += age_score * 0.3
            
            # Score basé sur la fréquence d'utilisation
            usage_count = await self._get_device_usage_count(device.device_id)
            usage_score = min(1.0, usage_count / 10.0)  # Max après 10 utilisations
            score += usage_score * 0.4
            
            # Score basé sur la cohérence des localisations
            location_consistency = await self._get_location_consistency_score(device.device_id)
            score += location_consistency * 0.2
            
            # Score basé sur l'absence d'activité suspecte
            suspicious_activity = await self._get_suspicious_activity_score(device.device_id)
            score += (1.0 - suspicious_activity) * 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as exc:
            self.logger.error(f"Erreur calcul score confiance: {exc}")
            return 0.0
    
    async def _find_device_by_fingerprint(
        self,
        user_id: str,
        device_fingerprint: str
    ) -> Optional[DeviceInfo]:
        """Trouve un dispositif par son empreinte"""
        try:
            devices = await self.get_user_devices(user_id)
            for device in devices:
                if device.device_fingerprint == device_fingerprint:
                    return device
            return None
            
        except Exception as exc:
            self.logger.error(f"Erreur recherche dispositif par empreinte: {exc}")
            return None
    
    async def _store_device(self, device: DeviceInfo):
        """Stocke un dispositif dans Redis"""
        try:
            # Stocker le dispositif
            device_data = json.dumps(asdict(device), default=str)
            await self.redis_client.set(f"device:{device.device_id}", device_data)
            
            # Ajouter à l'index utilisateur
            await self.redis_client.sadd(f"user_devices:{device.user_id}", device.device_id)
            
            # Définir l'expiration
            await self.redis_client.expire(f"device:{device.device_id}", self.device_expiry)
            
        except Exception as exc:
            self.logger.error(f"Erreur stockage dispositif: {exc}")
    
    async def _get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """Récupère un dispositif par son ID"""
        try:
            device_data = await self.redis_client.get(f"device:{device_id}")
            if device_data:
                device_dict = json.loads(device_data)
                return DeviceInfo(**device_dict)
            return None
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération dispositif {device_id}: {exc}")
            return None
    
    async def _delete_device(self, device: DeviceInfo):
        """Supprime un dispositif"""
        try:
            await self.redis_client.delete(f"device:{device.device_id}")
            await self.redis_client.srem(f"user_devices:{device.user_id}", device.device_id)
            
        except Exception as exc:
            self.logger.error(f"Erreur suppression dispositif: {exc}")
    
    # Méthodes utilitaires (implémentation simplifiée)
    async def _get_device_usage_count(self, device_id: str) -> int:
        """Récupère le nombre d'utilisations d'un dispositif"""
        # Implémentation simplifiée
        return 1
    
    async def _get_location_consistency_score(self, device_id: str) -> float:
        """Calcule le score de cohérence géographique"""
        # Implémentation simplifiée
        return 0.8
    
    async def _get_suspicious_activity_score(self, device_id: str) -> float:
        """Calcule le score d'activité suspecte"""
        # Implémentation simplifiée
        return 0.1


class SessionStore:
    """Store de sessions avec différents backends"""
    
    def __init__(self, backend_type: str = "redis", **config):
        self.backend_type = backend_type
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if backend_type == "redis":
            self.redis_client = redis.Redis(**config)
        else:
            raise ValueError(f"Backend non supporté: {backend_type}")
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Stocke une valeur"""
        try:
            serialized_value = json.dumps(value, default=str)
            if ttl:
                await self.redis_client.setex(key, ttl, serialized_value)
            else:
                await self.redis_client.set(key, serialized_value)
        except Exception as exc:
            self.logger.error(f"Erreur stockage {key}: {exc}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur"""
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as exc:
            self.logger.error(f"Erreur récupération {key}: {exc}")
            return None
    
    async def delete(self, key: str):
        """Supprime une valeur"""
        try:
            await self.redis_client.delete(key)
        except Exception as exc:
            self.logger.error(f"Erreur suppression {key}: {exc}")
    
    async def exists(self, key: str) -> bool:
        """Vérifie l'existence d'une clé"""
        try:
            return await self.redis_client.exists(key)
        except Exception as exc:
            self.logger.error(f"Erreur vérification existence {key}: {exc}")
            return False
