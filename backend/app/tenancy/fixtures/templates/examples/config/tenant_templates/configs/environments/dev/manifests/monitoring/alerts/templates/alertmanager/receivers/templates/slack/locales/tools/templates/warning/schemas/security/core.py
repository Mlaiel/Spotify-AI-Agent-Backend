"""
Core Security Components for Multi-Tenant Architecture
====================================================

Ce module contient les composants centraux de l'architecture de sécurité
multi-tenant pour Spotify AI Agent.

Auteur: Fahed Mlaiel
"""

import asyncio
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
import aiofiles
import asyncpg

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Niveaux de sécurité disponibles"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types de menaces détectées"""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    ANOMALY_DETECTED = "anomaly_detected"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"


@dataclass
class SecurityEvent:
    """Structure d'événement de sécurité"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    user_id: str = ""
    event_type: str = ""
    severity: SecurityLevel = SecurityLevel.LOW
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_ip: str = ""
    user_agent: str = ""
    resource: str = ""
    action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    threat_score: float = 0.0
    is_blocked: bool = False
    escalated: bool = False


@dataclass
class TenantSecurityConfig:
    """Configuration de sécurité par tenant"""
    tenant_id: str
    encryption_key: str
    security_level: SecurityLevel
    allowed_ips: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    monitoring_enabled: bool = True
    threat_detection_enabled: bool = True
    anomaly_detection_enabled: bool = True
    compliance_rules: List[str] = field(default_factory=list)
    alert_channels: List[str] = field(default_factory=list)
    escalation_rules: Dict[str, Any] = field(default_factory=dict)


class SecuritySchemaManager:
    """
    Gestionnaire central des schémas de sécurité multi-tenant
    """
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.tenant_configs: Dict[str, TenantSecurityConfig] = {}
        self.encryption_keys: Dict[str, Fernet] = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialise le gestionnaire de sécurité"""
        try:
            await self._load_tenant_configs()
            await self._setup_encryption_keys()
            logger.info("SecuritySchemaManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SecuritySchemaManager: {e}")
            raise
    
    async def _load_tenant_configs(self):
        """Charge les configurations de sécurité des tenants"""
        async with self._lock:
            try:
                # Récupération depuis Redis cache
                cached_configs = await self.redis.hgetall("tenant_security_configs")
                
                if cached_configs:
                    for tenant_id, config_json in cached_configs.items():
                        config_data = json.loads(config_json)
                        self.tenant_configs[tenant_id.decode()] = TenantSecurityConfig(**config_data)
                else:
                    # Récupération depuis base de données
                    await self._load_from_database()
                    
            except Exception as e:
                logger.error(f"Error loading tenant configs: {e}")
                raise
    
    async def _load_from_database(self):
        """Charge les configurations depuis la base de données"""
        # Implémentation de chargement DB
        query = "SELECT * FROM tenant_security_configs"
        # Simulation d'une récupération DB
        pass
    
    async def _setup_encryption_keys(self):
        """Configure les clés de chiffrement par tenant"""
        for tenant_id, config in self.tenant_configs.items():
            if config.encryption_key:
                key = base64.urlsafe_b64decode(config.encryption_key.encode())
                self.encryption_keys[tenant_id] = Fernet(key)
    
    async def get_tenant_config(self, tenant_id: str) -> Optional[TenantSecurityConfig]:
        """Récupère la configuration de sécurité d'un tenant"""
        return self.tenant_configs.get(tenant_id)
    
    async def update_tenant_config(self, tenant_id: str, config: TenantSecurityConfig):
        """Met à jour la configuration de sécurité d'un tenant"""
        async with self._lock:
            self.tenant_configs[tenant_id] = config
            
            # Mise à jour cache Redis
            config_json = json.dumps(config.__dict__, default=str)
            await self.redis.hset("tenant_security_configs", tenant_id, config_json)
            
            # Mise à jour DB
            # await self._update_database(tenant_id, config)
            
    async def encrypt_tenant_data(self, tenant_id: str, data: str) -> str:
        """Chiffre les données pour un tenant spécifique"""
        if tenant_id not in self.encryption_keys:
            raise ValueError(f"No encryption key found for tenant {tenant_id}")
        
        fernet = self.encryption_keys[tenant_id]
        encrypted_data = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    async def decrypt_tenant_data(self, tenant_id: str, encrypted_data: str) -> str:
        """Déchiffre les données pour un tenant spécifique"""
        if tenant_id not in self.encryption_keys:
            raise ValueError(f"No encryption key found for tenant {tenant_id}")
        
        fernet = self.encryption_keys[tenant_id]
        decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = fernet.decrypt(decoded_data)
        return decrypted_data.decode()


class TenantSecurityValidator:
    """
    Validateur de sécurité multi-tenant avec isolation des données
    """
    
    def __init__(self, tenant_id: str, schema_manager: SecuritySchemaManager):
        self.tenant_id = tenant_id
        self.schema_manager = schema_manager
        self.config: Optional[TenantSecurityConfig] = None
        
    async def initialize(self):
        """Initialise le validateur pour le tenant"""
        self.config = await self.schema_manager.get_tenant_config(self.tenant_id)
        if not self.config:
            raise ValueError(f"No security config found for tenant {self.tenant_id}")
    
    async def validate_access(self, user_id: str, resource: str, action: str, 
                            source_ip: str = None, user_agent: str = None) -> bool:
        """Valide l'accès à une ressource pour un utilisateur"""
        try:
            # Vérification IP si configurée
            if source_ip and not await self._validate_ip(source_ip):
                logger.warning(f"IP {source_ip} blocked for tenant {self.tenant_id}")
                return False
            
            # Vérification des limites de taux
            if not await self._check_rate_limits(user_id, action):
                logger.warning(f"Rate limit exceeded for user {user_id}")
                return False
            
            # Vérification des permissions
            if not await self._validate_permissions(user_id, resource, action):
                logger.warning(f"Permission denied for user {user_id} on {resource}")
                return False
            
            # Log de l'accès autorisé
            await self._log_access(user_id, resource, action, True, source_ip, user_agent)
            return True
            
        except Exception as e:
            logger.error(f"Error validating access: {e}")
            await self._log_access(user_id, resource, action, False, source_ip, user_agent)
            return False
    
    async def _validate_ip(self, source_ip: str) -> bool:
        """Valide l'adresse IP source"""
        if source_ip in self.config.blocked_ips:
            return False
        
        if self.config.allowed_ips and source_ip not in self.config.allowed_ips:
            return False
        
        return True
    
    async def _check_rate_limits(self, user_id: str, action: str) -> bool:
        """Vérifie les limites de taux"""
        if action not in self.config.rate_limits:
            return True
        
        limit = self.config.rate_limits[action]
        key = f"rate_limit:{self.tenant_id}:{user_id}:{action}"
        
        current_count = await self.schema_manager.redis.get(key)
        if current_count and int(current_count) >= limit:
            return False
        
        # Incrémente le compteur
        await self.schema_manager.redis.incr(key)
        await self.schema_manager.redis.expire(key, 3600)  # 1 heure
        return True
    
    async def _validate_permissions(self, user_id: str, resource: str, action: str) -> bool:
        """Valide les permissions utilisateur"""
        # Implémentation de validation des permissions RBAC
        # Logique métier complexe ici
        return True
    
    async def _log_access(self, user_id: str, resource: str, action: str, 
                         allowed: bool, source_ip: str = None, user_agent: str = None):
        """Log les tentatives d'accès"""
        event = SecurityEvent(
            tenant_id=self.tenant_id,
            user_id=user_id,
            event_type="access_attempt",
            severity=SecurityLevel.LOW if allowed else SecurityLevel.MEDIUM,
            source_ip=source_ip or "",
            user_agent=user_agent or "",
            resource=resource,
            action=action,
            metadata={"allowed": allowed}
        )
        
        # Enregistrement en cache et base de données
        await self._store_security_event(event)
    
    async def _store_security_event(self, event: SecurityEvent):
        """Stocke un événement de sécurité"""
        event_json = json.dumps(event.__dict__, default=str)
        
        # Stockage Redis pour accès rapide
        await self.schema_manager.redis.lpush(
            f"security_events:{self.tenant_id}", 
            event_json
        )
        
        # Limitation de la taille de la liste
        await self.schema_manager.redis.ltrim(
            f"security_events:{self.tenant_id}", 
            0, 1000
        )


class SecurityEventProcessor:
    """
    Processeur d'événements de sécurité en temps réel
    """
    
    def __init__(self, schema_manager: SecuritySchemaManager):
        self.schema_manager = schema_manager
        self.event_queue = asyncio.Queue()
        self.processors = []
        self.running = False
        
    async def start(self):
        """Démarre le processeur d'événements"""
        self.running = True
        
        # Démarrage des workers de traitement
        for i in range(5):  # 5 workers parallèles
            task = asyncio.create_task(self._event_worker(f"worker-{i}"))
            self.processors.append(task)
        
        logger.info("SecurityEventProcessor started")
    
    async def stop(self):
        """Arrête le processeur d'événements"""
        self.running = False
        
        # Arrêt des workers
        for processor in self.processors:
            processor.cancel()
        
        await asyncio.gather(*self.processors, return_exceptions=True)
        logger.info("SecurityEventProcessor stopped")
    
    async def process_event(self, event: SecurityEvent):
        """Traite un événement de sécurité"""
        await self.event_queue.put(event)
    
    async def _event_worker(self, worker_name: str):
        """Worker de traitement d'événements"""
        while self.running:
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(), 
                    timeout=1.0
                )
                
                await self._handle_event(event)
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in {worker_name}: {e}")
    
    async def _handle_event(self, event: SecurityEvent):
        """Gère un événement spécifique"""
        try:
            # Analyse de l'événement
            await self._analyze_event(event)
            
            # Détection de menaces
            threat_score = await self._calculate_threat_score(event)
            event.threat_score = threat_score
            
            # Escalade si nécessaire
            if threat_score > 0.8:
                await self._escalate_event(event)
            
            # Stockage persistent
            await self._store_event_persistent(event)
            
        except Exception as e:
            logger.error(f"Error handling event {event.event_id}: {e}")
    
    async def _analyze_event(self, event: SecurityEvent):
        """Analyse un événement de sécurité"""
        # Logique d'analyse complexe
        pass
    
    async def _calculate_threat_score(self, event: SecurityEvent) -> float:
        """Calcule le score de menace d'un événement"""
        score = 0.0
        
        # Facteurs de calcul du score
        severity_weights = {
            SecurityLevel.LOW: 0.1,
            SecurityLevel.MEDIUM: 0.3,
            SecurityLevel.HIGH: 0.7,
            SecurityLevel.CRITICAL: 1.0
        }
        
        score += severity_weights.get(event.severity, 0.1)
        
        # Autres facteurs (fréquence, contexte, historique)
        # Implémentation complexe ici
        
        return min(score, 1.0)
    
    async def _escalate_event(self, event: SecurityEvent):
        """Escalade un événement critique"""
        event.escalated = True
        
        # Notification immédiate
        alerting_engine = AlertingEngine(self.schema_manager)
        await alerting_engine.send_critical_alert(event)
    
    async def _store_event_persistent(self, event: SecurityEvent):
        """Stockage persistent d'un événement"""
        # Implémentation de stockage en base de données
        pass


class AlertingEngine:
    """
    Moteur d'alertes configurables et extensibles
    """
    
    def __init__(self, schema_manager: SecuritySchemaManager):
        self.schema_manager = schema_manager
        self.alert_templates = {}
        
    async def initialize(self):
        """Initialise le moteur d'alertes"""
        await self._load_alert_templates()
        logger.info("AlertingEngine initialized")
    
    async def _load_alert_templates(self):
        """Charge les templates d'alertes"""
        # Chargement depuis configuration
        pass
    
    async def send_alert(self, event: SecurityEvent, alert_type: str = "standard"):
        """Envoie une alerte"""
        config = await self.schema_manager.get_tenant_config(event.tenant_id)
        if not config:
            return
        
        for channel in config.alert_channels:
            await self._send_to_channel(event, channel, alert_type)
    
    async def send_critical_alert(self, event: SecurityEvent):
        """Envoie une alerte critique"""
        await self.send_alert(event, "critical")
        
        # Escalade supplémentaire pour les alertes critiques
        await self._critical_escalation(event)
    
    async def _send_to_channel(self, event: SecurityEvent, channel: str, alert_type: str):
        """Envoie une alerte vers un canal spécifique"""
        if channel == "slack":
            await self._send_slack_alert(event, alert_type)
        elif channel == "email":
            await self._send_email_alert(event, alert_type)
        elif channel == "siem":
            await self._send_siem_alert(event, alert_type)
    
    async def _send_slack_alert(self, event: SecurityEvent, alert_type: str):
        """Envoie une alerte Slack"""
        # Implémentation d'envoi Slack
        pass
    
    async def _send_email_alert(self, event: SecurityEvent, alert_type: str):
        """Envoie une alerte email"""
        # Implémentation d'envoi email
        pass
    
    async def _send_siem_alert(self, event: SecurityEvent, alert_type: str):
        """Envoie une alerte vers SIEM"""
        # Implémentation d'envoi SIEM
        pass
    
    async def _critical_escalation(self, event: SecurityEvent):
        """Escalade critique"""
        # Implémentation d'escalade d'urgence
        pass
    
    async def configure_tenant_alerts(self, tenant_id: str, alert_rules: Dict[str, Any]):
        """Configure les règles d'alertes pour un tenant"""
        config = await self.schema_manager.get_tenant_config(tenant_id)
        if config:
            config.escalation_rules.update(alert_rules)
            await self.schema_manager.update_tenant_config(tenant_id, config)


# Utilitaires de sécurité
class SecurityUtils:
    """Utilitaires de sécurité"""
    
    @staticmethod
    def generate_encryption_key() -> str:
        """Génère une clé de chiffrement sécurisée"""
        key = Fernet.generate_key()
        return base64.urlsafe_b64encode(key).decode()
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> tuple:
        """Hash un mot de passe avec salt"""
        if not salt:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return password_hash.hex(), salt
    
    @staticmethod
    def verify_password(password: str, password_hash: str, salt: str) -> bool:
        """Vérifie un mot de passe"""
        computed_hash, _ = SecurityUtils.hash_password(password, salt)
        return computed_hash == password_hash
    
    @staticmethod
    def generate_session_token() -> str:
        """Génère un token de session sécurisé"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def sanitize_input(input_data: str) -> str:
        """Sanitise les entrées utilisateur"""
        # Implémentation de sanitisation
        return input_data.strip()
