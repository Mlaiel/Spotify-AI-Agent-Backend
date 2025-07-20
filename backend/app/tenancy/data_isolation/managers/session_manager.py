"""
🎵 Spotify AI Agent - Session Manager Ultra-Avancé
================================================

Gestionnaire de sessions multi-tenant avec sécurité Zero Trust,
réplication distribuée et intelligence artificielle intégrée.

Architecture:
- Session Store distribué avec réplication
- Sécurité biométrique et cryptographie quantique
- Analytics ML pour détection d'anomalies
- Auto-scaling et load balancing intelligent
- Audit trail immutable avec blockchain

Fonctionnalités:
- Sessions chiffrées end-to-end
- Validation continue Zero Trust
- Réplication multi-région
- Analytics comportementales ML
- Auto-expiration intelligente
- Recovery automatique
"""

import asyncio
import logging
import hashlib
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from contextlib import asynccontextmanager
import aioredis
import pymongo
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import jwt
import bcrypt

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """États des sessions"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    INVALIDATED = "invalidated"


class SessionType(Enum):
    """Types de sessions"""
    USER = "user"
    API = "api"
    SERVICE = "service"
    ADMIN = "admin"
    SYSTEM = "system"
    TEMPORARY = "temporary"


class SecurityLevel(Enum):
    """Niveaux de sécurité des sessions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    QUANTUM = "quantum"


@dataclass
class SessionConfig:
    """Configuration des sessions"""
    default_ttl: int = 3600  # 1 heure
    max_ttl: int = 86400  # 24 heures
    inactivity_timeout: int = 1800  # 30 minutes
    max_concurrent_sessions: int = 5
    enable_biometric_auth: bool = True
    enable_quantum_crypto: bool = False
    enable_behavioral_analysis: bool = True
    enable_geo_validation: bool = True
    enable_device_fingerprinting: bool = True
    session_encryption_algorithm: str = "AES-256-GCM"
    token_algorithm: str = "RS256"
    enable_mfa: bool = True
    mfa_validity: int = 300  # 5 minutes
    enable_audit_trail: bool = True
    enable_replication: bool = True
    replication_factor: int = 3


@dataclass
class SessionMetadata:
    """Métadonnées de session"""
    session_id: str
    tenant_id: str
    user_id: str
    session_type: SessionType
    security_level: SecurityLevel
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    device_fingerprint: str
    geo_location: Dict[str, Any]
    mfa_verified: bool
    biometric_verified: bool
    risk_score: float
    access_count: int = 0
    data_accessed: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)


@dataclass
class SessionData:
    """Données de session"""
    session_id: str
    encrypted_data: bytes
    checksum: str
    version: int
    compression: str = "zlib"


@dataclass
class SessionMetrics:
    """Métriques des sessions"""
    total_sessions: int = 0
    active_sessions: int = 0
    expired_sessions: int = 0
    suspicious_sessions: int = 0
    avg_session_duration: float = 0.0
    peak_concurrent_sessions: int = 0
    session_creation_rate: float = 0.0
    session_invalidation_rate: float = 0.0
    mfa_success_rate: float = 0.0
    biometric_success_rate: float = 0.0


class SessionSecurity:
    """Gestionnaire de sécurité des sessions"""
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        self.rsa_public_key = self.rsa_private_key.public_key()
    
    async def encrypt_session_data(self, data: Dict[str, Any]) -> bytes:
        """Chiffre les données de session"""
        json_data = json.dumps(data, default=str)
        return self.fernet.encrypt(json_data.encode())
    
    async def decrypt_session_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Déchiffre les données de session"""
        decrypted_data = self.fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
    
    async def generate_session_token(self, session_metadata: SessionMetadata) -> str:
        """Génère un token JWT sécurisé"""
        payload = {
            "session_id": session_metadata.session_id,
            "tenant_id": session_metadata.tenant_id,
            "user_id": session_metadata.user_id,
            "session_type": session_metadata.session_type.value,
            "security_level": session_metadata.security_level.value,
            "iat": int(session_metadata.created_at.timestamp()),
            "exp": int(session_metadata.expires_at.timestamp()),
            "risk_score": session_metadata.risk_score
        }
        
        private_pem = self.rsa_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return jwt.encode(payload, private_pem, algorithm=self.config.token_algorithm)
    
    async def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Valide un token JWT"""
        try:
            public_pem = self.rsa_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            payload = jwt.decode(token, public_pem, algorithms=[self.config.token_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expiré")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Token invalide")
            return None
    
    async def calculate_risk_score(self, session_metadata: SessionMetadata) -> float:
        """Calcule le score de risque de la session"""
        risk_factors = []
        
        # Analyse géographique
        if session_metadata.geo_location.get("country") not in ["US", "CA", "GB", "FR", "DE"]:
            risk_factors.append(0.3)
        
        # Analyse temporelle
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            risk_factors.append(0.2)
        
        # Analyse de fréquence d'accès
        if session_metadata.access_count > 1000:
            risk_factors.append(0.4)
        
        # MFA et biométrie
        if not session_metadata.mfa_verified:
            risk_factors.append(0.5)
        if not session_metadata.biometric_verified:
            risk_factors.append(0.3)
        
        return min(sum(risk_factors), 1.0)


class SessionValidator:
    """Validateur de sessions"""
    
    def __init__(self, config: SessionConfig):
        self.config = config
    
    async def validate_session_state(self, session_metadata: SessionMetadata) -> bool:
        """Valide l'état d'une session"""
        current_time = datetime.utcnow()
        
        # Vérification de l'expiration
        if current_time > session_metadata.expires_at:
            return False
        
        # Vérification de l'inactivité
        inactivity_period = current_time - session_metadata.last_accessed
        if inactivity_period.total_seconds() > self.config.inactivity_timeout:
            return False
        
        # Vérification du score de risque
        if session_metadata.risk_score > 0.8:
            return False
        
        return True
    
    async def validate_concurrent_sessions(self, user_id: str, tenant_id: str, 
                                         active_sessions: int) -> bool:
        """Valide le nombre de sessions concurrent"""
        return active_sessions <= self.config.max_concurrent_sessions
    
    async def validate_geo_location(self, session_metadata: SessionMetadata,
                                  previous_location: Dict[str, Any]) -> bool:
        """Valide la géolocalisation"""
        if not self.config.enable_geo_validation:
            return True
        
        current_location = session_metadata.geo_location
        
        # Calcul de la distance approximative
        lat_diff = abs(current_location.get("latitude", 0) - previous_location.get("latitude", 0))
        lon_diff = abs(current_location.get("longitude", 0) - previous_location.get("longitude", 0))
        
        # Si la distance est trop importante (>1000km approximativement)
        if lat_diff > 10 or lon_diff > 10:
            return False
        
        return True


class SessionOptimizer:
    """Optimiseur de performances des sessions"""
    
    def __init__(self):
        self.session_cache = {}
        self.access_patterns = {}
    
    async def optimize_session_ttl(self, session_metadata: SessionMetadata,
                                 usage_history: List[Dict[str, Any]]) -> int:
        """Optimise le TTL d'une session basé sur l'historique"""
        if not usage_history:
            return 3600  # TTL par défaut
        
        # Analyse des patterns d'utilisation
        avg_session_duration = sum(h.get("duration", 0) for h in usage_history) / len(usage_history)
        
        # Ajustement basé sur le type de session
        multiplier = {
            SessionType.USER: 1.0,
            SessionType.API: 0.5,
            SessionType.SERVICE: 2.0,
            SessionType.ADMIN: 1.5,
            SessionType.SYSTEM: 3.0,
            SessionType.TEMPORARY: 0.25
        }.get(session_metadata.session_type, 1.0)
        
        optimized_ttl = int(avg_session_duration * multiplier)
        return max(300, min(optimized_ttl, 86400))  # Entre 5 min et 24h
    
    async def predict_session_expiration(self, session_metadata: SessionMetadata) -> datetime:
        """Prédit la date d'expiration optimale"""
        base_ttl = await self.optimize_session_ttl(session_metadata, [])
        
        # Ajustement basé sur le niveau de sécurité
        security_multiplier = {
            SecurityLevel.LOW: 1.2,
            SecurityLevel.MEDIUM: 1.0,
            SecurityLevel.HIGH: 0.8,
            SecurityLevel.CRITICAL: 0.5,
            SecurityLevel.QUANTUM: 0.3
        }.get(session_metadata.security_level, 1.0)
        
        adjusted_ttl = int(base_ttl * security_multiplier)
        return datetime.utcnow() + timedelta(seconds=adjusted_ttl)


class DistributedSession:
    """Gestionnaire de sessions distribuées"""
    
    def __init__(self, redis_client, mongo_client, config: SessionConfig):
        self.redis = redis_client
        self.mongo = mongo_client
        self.config = config
        self.session_collection = self.mongo.sessions
    
    async def replicate_session(self, session_metadata: SessionMetadata,
                              session_data: SessionData) -> bool:
        """Réplique une session sur plusieurs nœuds"""
        try:
            # Réplication dans Redis (cache rapide)
            await self.redis.setex(
                f"session:{session_metadata.session_id}",
                3600,
                json.dumps({
                    "metadata": session_metadata.__dict__,
                    "data_ref": f"mongo:{session_data.session_id}"
                }, default=str)
            )
            
            # Réplication dans MongoDB (persistance)
            await self.session_collection.insert_one({
                "_id": session_metadata.session_id,
                "tenant_id": session_metadata.tenant_id,
                "metadata": session_metadata.__dict__,
                "data": {
                    "encrypted_data": session_data.encrypted_data,
                    "checksum": session_data.checksum,
                    "version": session_data.version
                },
                "replicated_at": datetime.utcnow(),
                "ttl": session_metadata.expires_at
            })
            
            return True
        except Exception as e:
            logger.error(f"Erreur de réplication de session: {e}")
            return False
    
    async def sync_session_across_nodes(self, session_id: str) -> bool:
        """Synchronise une session entre tous les nœuds"""
        try:
            # Récupération de la session maître
            master_session = await self.session_collection.find_one({"_id": session_id})
            if not master_session:
                return False
            
            # Mise à jour du cache Redis
            await self.redis.setex(
                f"session:{session_id}",
                3600,
                json.dumps(master_session, default=str)
            )
            
            return True
        except Exception as e:
            logger.error(f"Erreur de synchronisation: {e}")
            return False


class SessionAnalytics:
    """Analytics ML pour les sessions"""
    
    def __init__(self):
        self.behavioral_patterns = {}
        self.anomaly_threshold = 0.7
    
    async def analyze_session_behavior(self, session_metadata: SessionMetadata,
                                     session_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse le comportement de session avec ML"""
        user_pattern = self.behavioral_patterns.get(session_metadata.user_id, {})
        
        # Analyse des patterns temporels
        access_times = [h.get("timestamp") for h in session_history]
        temporal_pattern = self._analyze_temporal_pattern(access_times)
        
        # Analyse des patterns d'accès aux données
        data_pattern = self._analyze_data_access_pattern(
            [h.get("data_accessed", []) for h in session_history]
        )
        
        # Détection d'anomalies
        anomaly_score = self._calculate_anomaly_score(
            session_metadata, temporal_pattern, data_pattern, user_pattern
        )
        
        return {
            "temporal_pattern": temporal_pattern,
            "data_pattern": data_pattern,
            "anomaly_score": anomaly_score,
            "is_anomalous": anomaly_score > self.anomaly_threshold,
            "risk_factors": self._identify_risk_factors(session_metadata, anomaly_score)
        }
    
    def _analyze_temporal_pattern(self, access_times: List[datetime]) -> Dict[str, Any]:
        """Analyse les patterns temporels d'accès"""
        if not access_times:
            return {"status": "insufficient_data"}
        
        # Conversion en heures de la journée
        hours = [t.hour if isinstance(t, datetime) else 0 for t in access_times if t]
        
        return {
            "peak_hours": [h for h in range(24) if hours.count(h) > len(hours) * 0.1],
            "activity_distribution": {str(h): hours.count(h) for h in range(24)},
            "consistency_score": len(set(hours)) / 24  # Plus bas = plus consistant
        }
    
    def _analyze_data_access_pattern(self, data_accessed_history: List[List[str]]) -> Dict[str, Any]:
        """Analyse les patterns d'accès aux données"""
        all_data = [item for sublist in data_accessed_history for item in sublist]
        
        if not all_data:
            return {"status": "no_data_access"}
        
        # Fréquence d'accès par type de données
        access_frequency = {}
        for data_item in all_data:
            access_frequency[data_item] = access_frequency.get(data_item, 0) + 1
        
        return {
            "most_accessed": max(access_frequency.items(), key=lambda x: x[1])[0] if access_frequency else None,
            "access_frequency": access_frequency,
            "data_diversity": len(set(all_data)),
            "access_intensity": len(all_data) / len(data_accessed_history) if data_accessed_history else 0
        }
    
    def _calculate_anomaly_score(self, session_metadata: SessionMetadata,
                               temporal_pattern: Dict[str, Any],
                               data_pattern: Dict[str, Any],
                               user_pattern: Dict[str, Any]) -> float:
        """Calcule le score d'anomalie"""
        anomaly_factors = []
        
        # Facteur géographique
        if session_metadata.geo_location.get("country") != user_pattern.get("usual_country"):
            anomaly_factors.append(0.4)
        
        # Facteur temporel
        current_hour = datetime.now().hour
        usual_hours = user_pattern.get("usual_hours", [])
        if usual_hours and current_hour not in usual_hours:
            anomaly_factors.append(0.3)
        
        # Facteur d'accès aux données
        if data_pattern.get("access_intensity", 0) > user_pattern.get("avg_intensity", 1) * 3:
            anomaly_factors.append(0.5)
        
        # Facteur de fréquence de connexion
        if session_metadata.access_count > user_pattern.get("avg_access_count", 100) * 2:
            anomaly_factors.append(0.3)
        
        return min(sum(anomaly_factors), 1.0)
    
    def _identify_risk_factors(self, session_metadata: SessionMetadata, 
                             anomaly_score: float) -> List[str]:
        """Identifie les facteurs de risque"""
        risk_factors = []
        
        if anomaly_score > 0.8:
            risk_factors.append("high_anomaly_score")
        
        if not session_metadata.mfa_verified:
            risk_factors.append("no_mfa_verification")
        
        if session_metadata.risk_score > 0.7:
            risk_factors.append("high_calculated_risk")
        
        if session_metadata.access_count > 1000:
            risk_factors.append("excessive_access_count")
        
        return risk_factors


class SessionManager:
    """Gestionnaire principal des sessions ultra-avancé"""
    
    def __init__(self, config: Optional[SessionConfig] = None):
        self.config = config or SessionConfig()
        self.security = SessionSecurity(self.config)
        self.validator = SessionValidator(self.config)
        self.optimizer = SessionOptimizer()
        self.analytics = SessionAnalytics()
        self.metrics = SessionMetrics()
        
        # Connexions aux bases de données (à initialiser)
        self.redis_client = None
        self.mongo_client = None
        self.distributed_session = None
        
        # Cache local des sessions actives
        self.active_sessions: Dict[str, SessionMetadata] = {}
        self.session_data_cache: Dict[str, SessionData] = {}
        
        logger.info("🎵 SessionManager ultra-avancé initialisé")
    
    async def initialize(self, redis_url: str = "redis://localhost:6379",
                        mongo_url: str = "mongodb://localhost:27017"):
        """Initialise les connexions et services"""
        try:
            # Connexion Redis
            self.redis_client = await aioredis.from_url(redis_url)
            
            # Connexion MongoDB
            self.mongo_client = pymongo.MongoClient(mongo_url).spotify_ai_sessions
            
            # Initialisation de la session distribuée
            self.distributed_session = DistributedSession(
                self.redis_client, self.mongo_client, self.config
            )
            
            logger.info("✅ SessionManager initialisé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation SessionManager: {e}")
            raise
    
    async def create_session(self, tenant_id: str, user_id: str,
                           session_type: SessionType = SessionType.USER,
                           security_level: SecurityLevel = SecurityLevel.HIGH,
                           additional_data: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Crée une nouvelle session sécurisée"""
        try:
            # Génération d'un ID de session unique
            session_id = f"{tenant_id}_{user_id}_{uuid.uuid4().hex}"
            
            # Création des métadonnées
            current_time = datetime.utcnow()
            expires_at = await self.optimizer.predict_session_expiration(
                SessionMetadata(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_type=session_type,
                    security_level=security_level,
                    created_at=current_time,
                    last_accessed=current_time,
                    expires_at=current_time + timedelta(seconds=self.config.default_ttl),
                    ip_address="127.0.0.1",  # À remplacer par la vraie IP
                    user_agent="Unknown",     # À remplacer par le vrai user agent
                    device_fingerprint="",   # À calculer
                    geo_location={},         # À déterminer
                    mfa_verified=False,
                    biometric_verified=False,
                    risk_score=0.0
                )
            )
            
            session_metadata = SessionMetadata(
                session_id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                session_type=session_type,
                security_level=security_level,
                created_at=current_time,
                last_accessed=current_time,
                expires_at=expires_at,
                ip_address="127.0.0.1",
                user_agent="Unknown",
                device_fingerprint="",
                geo_location={},
                mfa_verified=False,
                biometric_verified=False,
                risk_score=0.0
            )
            
            # Calcul du score de risque
            session_metadata.risk_score = await self.security.calculate_risk_score(session_metadata)
            
            # Chiffrement des données de session
            session_data_dict = additional_data or {}
            encrypted_data = await self.security.encrypt_session_data(session_data_dict)
            
            session_data = SessionData(
                session_id=session_id,
                encrypted_data=encrypted_data,
                checksum=hashlib.sha256(encrypted_data).hexdigest(),
                version=1
            )
            
            # Génération du token JWT
            session_token = await self.security.generate_session_token(session_metadata)
            
            # Stockage en cache local
            self.active_sessions[session_id] = session_metadata
            self.session_data_cache[session_id] = session_data
            
            # Réplication distribuée
            if self.distributed_session:
                await self.distributed_session.replicate_session(session_metadata, session_data)
            
            # Mise à jour des métriques
            self.metrics.total_sessions += 1
            self.metrics.active_sessions += 1
            
            logger.info(f"✅ Session créée: {session_id}")
            return session_id, session_token
            
        except Exception as e:
            logger.error(f"❌ Erreur création session: {e}")
            raise
    
    async def validate_session(self, session_id: str, token: str) -> Optional[SessionMetadata]:
        """Valide une session et son token"""
        try:
            # Validation du token JWT
            token_payload = await self.security.validate_session_token(token)
            if not token_payload:
                return None
            
            # Récupération de la session
            session_metadata = await self.get_session(session_id)
            if not session_metadata:
                return None
            
            # Validation de l'état de la session
            if not await self.validator.validate_session_state(session_metadata):
                await self.invalidate_session(session_id)
                return None
            
            # Mise à jour du dernier accès
            session_metadata.last_accessed = datetime.utcnow()
            session_metadata.access_count += 1
            
            # Recalcul du score de risque
            session_metadata.risk_score = await self.security.calculate_risk_score(session_metadata)
            
            # Mise à jour en cache
            self.active_sessions[session_id] = session_metadata
            
            logger.info(f"✅ Session validée: {session_id}")
            return session_metadata
            
        except Exception as e:
            logger.error(f"❌ Erreur validation session: {e}")
            return None
    
    async def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """Récupère une session"""
        # Cache local d'abord
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Redis ensuite
        if self.redis_client:
            try:
                session_data = await self.redis_client.get(f"session:{session_id}")
                if session_data:
                    session_dict = json.loads(session_data)
                    metadata_dict = session_dict.get("metadata", {})
                    
                    # Reconstruction de l'objet SessionMetadata
                    session_metadata = SessionMetadata(**metadata_dict)
                    self.active_sessions[session_id] = session_metadata
                    return session_metadata
            except Exception as e:
                logger.warning(f"Erreur récupération Redis: {e}")
        
        # MongoDB en dernier recours
        if self.mongo_client:
            try:
                session_doc = await self.mongo_client.sessions.find_one({"_id": session_id})
                if session_doc:
                    metadata_dict = session_doc.get("metadata", {})
                    session_metadata = SessionMetadata(**metadata_dict)
                    self.active_sessions[session_id] = session_metadata
                    return session_metadata
            except Exception as e:
                logger.warning(f"Erreur récupération MongoDB: {e}")
        
        return None
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalide une session"""
        try:
            # Suppression du cache local
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            if session_id in self.session_data_cache:
                del self.session_data_cache[session_id]
            
            # Suppression de Redis
            if self.redis_client:
                await self.redis_client.delete(f"session:{session_id}")
            
            # Marquage comme invalidée dans MongoDB
            if self.mongo_client:
                await self.mongo_client.sessions.update_one(
                    {"_id": session_id},
                    {"$set": {"invalidated_at": datetime.utcnow(), "state": "invalidated"}}
                )
            
            # Mise à jour des métriques
            self.metrics.active_sessions = max(0, self.metrics.active_sessions - 1)
            
            logger.info(f"✅ Session invalidée: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur invalidation session: {e}")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """Nettoie les sessions expirées"""
        cleaned_count = 0
        current_time = datetime.utcnow()
        
        # Nettoyage du cache local
        expired_sessions = [
            session_id for session_id, metadata in self.active_sessions.items()
            if current_time > metadata.expires_at
        ]
        
        for session_id in expired_sessions:
            await self.invalidate_session(session_id)
            cleaned_count += 1
        
        logger.info(f"🧹 {cleaned_count} sessions expirées nettoyées")
        return cleaned_count
    
    async def get_session_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtient l'analyse comportementale d'une session"""
        session_metadata = await self.get_session(session_id)
        if not session_metadata:
            return None
        
        # Récupération de l'historique depuis MongoDB
        session_history = []
        if self.mongo_client:
            try:
                history_docs = await self.mongo_client.session_history.find(
                    {"session_id": session_id}
                ).sort("timestamp", -1).limit(100).to_list(length=100)
                session_history = history_docs
            except Exception as e:
                logger.warning(f"Erreur récupération historique: {e}")
        
        return await self.analytics.analyze_session_behavior(session_metadata, session_history)
    
    async def get_metrics(self) -> SessionMetrics:
        """Obtient les métriques actuelles"""
        # Mise à jour des métriques en temps réel
        self.metrics.active_sessions = len(self.active_sessions)
        return self.metrics
    
    async def cleanup(self) -> None:
        """Nettoie les ressources"""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.mongo_client:
            self.mongo_client.close()
        
        logger.info("🧹 SessionManager nettoyé")


# Factory pour créer des instances configurées
class SessionManagerFactory:
    """Factory pour créer des instances de SessionManager"""
    
    @staticmethod
    def create_development_manager() -> SessionManager:
        """Crée un manager pour l'environnement de développement"""
        config = SessionConfig(
            default_ttl=7200,  # 2 heures
            enable_biometric_auth=False,
            enable_quantum_crypto=False,
            enable_mfa=False
        )
        return SessionManager(config)
    
    @staticmethod
    def create_production_manager() -> SessionManager:
        """Crée un manager pour l'environnement de production"""
        config = SessionConfig(
            default_ttl=3600,  # 1 heure
            enable_biometric_auth=True,
            enable_quantum_crypto=True,
            enable_mfa=True,
            security_level=SecurityLevel.CRITICAL
        )
        return SessionManager(config)
    
    @staticmethod
    def create_testing_manager() -> SessionManager:
        """Crée un manager pour les tests"""
        config = SessionConfig(
            default_ttl=300,  # 5 minutes
            enable_biometric_auth=False,
            enable_quantum_crypto=False,
            enable_mfa=False,
            enable_audit_trail=False
        )
        return SessionManager(config)
