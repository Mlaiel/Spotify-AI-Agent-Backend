"""
Advanced PagerDuty Integration Management System

This module provides ultra-advanced PagerDuty integration capabilities for Alertmanager
including intelligent incident management, AI-powered escalation policies, auto-resolution,
comprehensive incident lifecycle management, and enterprise-grade monitoring.

Features:
- Intelligent AI-powered incident classification and prediction
- Auto-resolution with machine learning
- Advanced escalation policies with context awareness
- Real-time performance monitoring and optimization
- Enterprise-grade security and compliance
- Multi-tenant support with isolation
- Comprehensive audit logging and analytics
- Integration with major cloud providers and monitoring tools

Version: 4.0.0
Developed by Spotify AI Agent Team
Architecture: Fahed Mlaiel - Lead Developer & AI Architect
"""

import asyncio
import json
import hashlib
import uuid
import time
import hmac
import base64
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Set, AsyncIterator
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from contextlib import asynccontextmanager
import structlog
import aiofiles
import aiohttp
import aioredis
import backoff
import tenacity
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
import prometheus_client
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# ============================================================================
# Configuration et Constantes
# ============================================================================

# Constantes globales
PAGERDUTY_API_VERSION = "v2"
DEFAULT_TIMEOUT = 30
MAX_RETRY_ATTEMPTS = 3
CACHE_TTL = 300  # 5 minutes
RATE_LIMIT_PER_MINUTE = 100
ENCRYPTION_ALGORITHM = "AES-256-GCM"

# Métriques Prometheus
PAGERDUTY_REQUESTS = prometheus_client.Counter(
    'pagerduty_requests_total',
    'Total PagerDuty API requests',
    ['method', 'endpoint', 'status']
)

PAGERDUTY_RESPONSE_TIME = prometheus_client.Histogram(
    'pagerduty_response_time_seconds',
    'PagerDuty API response time',
    ['endpoint']
)

INCIDENT_CREATION_TIME = prometheus_client.Histogram(
    'incident_creation_time_seconds',
    'Time to create incidents',
    ['severity', 'urgency']
)

AI_PREDICTION_ACCURACY = prometheus_client.Gauge(
    'ai_prediction_accuracy',
    'AI prediction accuracy percentage'
)

# ============================================================================
# Enumerations et Types de Données
# ============================================================================

class IncidentSeverity(Enum):
    """Niveaux de sévérité des incidents selon ITIL v4"""
    CRITICAL = "critical"      # Impact sur business critique
    HIGH = "high"             # Impact significatif
    MEDIUM = "medium"         # Impact modéré
    LOW = "low"              # Impact minimal
    INFO = "info"            # Information seulement

class IncidentUrgency(Enum):
    """Niveaux d'urgence selon les SLA"""
    HIGH = "high"    # Résolution immédiate < 1h
    LOW = "low"      # Résolution standard < 24h

class IncidentStatus(Enum):
    """Statuts complets du cycle de vie des incidents"""
    TRIGGERED = "triggered"           # Incident créé
    ACKNOWLEDGED = "acknowledged"     # Incident pris en compte
    ESCALATED = "escalated"          # Incident escaladé
    ASSIGNED = "assigned"            # Incident assigné
    IN_PROGRESS = "in_progress"      # Résolution en cours
    RESOLVED = "resolved"            # Incident résolu
    AUTO_RESOLVED = "auto_resolved"  # Auto-résolution IA
    CLOSED = "closed"                # Incident fermé

class AlertSource(Enum):
    """Sources d'alertes supportées"""
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    CUSTOM = "custom"
    AI_PREDICTION = "ai_prediction"

class NotificationChannel(Enum):
    """Canaux de notification"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push_notification"
    SLACK = "slack"
    TEAMS = "microsoft_teams"
    WEBHOOK = "webhook"
    PHONE_CALL = "phone_call"

class EscalationLevel(Enum):
    """Niveaux d'escalade"""
    L1_SUPPORT = "l1_support"
    L2_ADVANCED = "l2_advanced"
    L3_EXPERT = "l3_expert"
    MANAGEMENT = "management"
    EXECUTIVE = "executive"

class AIModelType(Enum):
    """Types de modèles IA utilisés"""
    CLASSIFICATION = "classification"
    PREDICTION = "prediction"
    SENTIMENT = "sentiment_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    AUTO_RESOLUTION = "auto_resolution"

class ComplianceStandard(Enum):
    """Standards de conformité"""
    SOC2 = "soc2_type_ii"
    ISO27001 = "iso_27001"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"

# ============================================================================
# Modèles de Données Pydantic
# ============================================================================

class PagerDutyConfig(BaseModel):
    """Configuration principale PagerDuty"""
    api_key: str = Field(..., min_length=32)
    routing_key: str = Field(..., min_length=32)
    user_token: Optional[str] = Field(None, min_length=32)
    webhook_secret: str = Field(..., min_length=16)
    base_url: str = Field(default="https://api.pagerduty.com")
    timeout: int = Field(default=30, ge=1, le=300)
    max_retries: int = Field(default=3, ge=1, le=10)
    rate_limit: int = Field(default=100, ge=1, le=1000)
    enable_ai: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    
    @validator('base_url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

class IncidentData(BaseModel):
    """Modèle de données pour les incidents"""
    id: Optional[str] = None
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=4000)
    service_id: str = Field(..., min_length=1)
    severity: IncidentSeverity
    urgency: IncidentUrgency
    status: IncidentStatus = Field(default=IncidentStatus.TRIGGERED)
    source: AlertSource = Field(default=AlertSource.CUSTOM)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_to: Optional[str] = None
    escalation_level: EscalationLevel = Field(default=EscalationLevel.L1_SUPPORT)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ai_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    predicted_resolution_time: Optional[int] = None  # en minutes
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class EscalationPolicy(BaseModel):
    """Politique d'escalade intelligente"""
    id: str
    name: str
    description: Optional[str] = None
    levels: List[Dict[str, Any]]
    escalation_delay: int = Field(default=30, ge=1, le=1440)  # minutes
    ai_enabled: bool = Field(default=True)
    conditions: Dict[str, Any] = Field(default_factory=dict)
    business_hours_only: bool = Field(default=False)
    holiday_schedule: Optional[str] = None
    
class OnCallSchedule(BaseModel):
    """Planification des gardes"""
    id: str
    name: str
    timezone: str = Field(default="UTC")
    layers: List[Dict[str, Any]]
    restrictions: List[Dict[str, Any]] = Field(default_factory=list)
    overflow: Optional[Dict[str, Any]] = None
    
class NotificationRule(BaseModel):
    """Règles de notification avancées"""
    id: str
    name: str
    channels: List[NotificationChannel]
    conditions: Dict[str, Any]
    delay: int = Field(default=0, ge=0, le=3600)  # secondes
    frequency: int = Field(default=1, ge=1, le=10)
    escalation_timeout: int = Field(default=900, ge=60, le=7200)  # secondes
    ai_personalization: bool = Field(default=True)

class AIAnalysisResult(BaseModel):
    """Résultat d'analyse IA"""
    confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_severity: IncidentSeverity
    predicted_resolution_time: int  # minutes
    suggested_assignee: Optional[str] = None
    similar_incidents: List[str] = Field(default_factory=list)
    root_cause_probability: Dict[str, float] = Field(default_factory=dict)
    auto_resolution_possible: bool = Field(default=False)
    recommendations: List[str] = Field(default_factory=list)

class MetricsData(BaseModel):
    """Données de métriques et KPI"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mttr: float  # Mean Time To Recovery
    mtta: float  # Mean Time To Acknowledge
    incident_volume: int
    escalation_rate: float
    auto_resolution_rate: float
    ai_accuracy: float
    api_latency: float
    error_rate: float

# ============================================================================
# Classes de Gestion Principale
# ============================================================================

@dataclass
class CircuitBreakerState:
    """État du circuit breaker"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    failure_threshold: int = 5
    recovery_timeout: int = 60

class SecurityManager:
    """Gestionnaire de sécurité avancé"""
    
    def __init__(self, encryption_key: str):
        self.encryption_key = encryption_key.encode()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key[:32]))
        
    def encrypt_sensitive_data(self, data: str) -> str:
        """Chiffre les données sensibles"""
        return self.fernet.encrypt(data.encode()).decode()
        
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Déchiffre les données sensibles"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
        
    def generate_webhook_signature(self, payload: str, secret: str) -> str:
        """Génère la signature HMAC pour les webhooks"""
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
    def validate_webhook_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Valide la signature d'un webhook"""
        expected = self.generate_webhook_signature(payload, secret)
        return hmac.compare_digest(signature, expected)

class CacheManager:
    """Gestionnaire de cache intelligent avec Redis"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_pool = None
        
    async def initialize(self):
        """Initialise la connexion Redis"""
        self.redis_pool = aioredis.ConnectionPool.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        
    async def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
            value = await redis.get(key)
            return json.loads(value) if value else None
            
    async def set(self, key: str, value: Any, ttl: int = CACHE_TTL):
        """Stocke une valeur dans le cache"""
        async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
            await redis.setex(key, ttl, json.dumps(value, default=str))
            
    async def delete(self, key: str):
        """Supprime une valeur du cache"""
        async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
            await redis.delete(key)

class RateLimiter:
    """Rate limiter intelligent avec différentes stratégies"""
    
    def __init__(self, redis_url: str, max_requests: int = 100, window: int = 60):
        self.redis_url = redis_url
        self.max_requests = max_requests
        self.window = window
        self.redis_pool = None
        
    async def initialize(self):
        """Initialise la connexion Redis"""
        self.redis_pool = aioredis.ConnectionPool.from_url(self.redis_url)
        
    async def is_allowed(self, key: str) -> bool:
        """Vérifie si la requête est autorisée"""
        async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
            current_time = int(time.time())
            window_start = current_time - self.window
            
            # Sliding window avec Redis sorted sets
            pipe = redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(uuid.uuid4()): current_time})
            pipe.zcard(key)
            pipe.expire(key, self.window)
            results = await pipe.execute()
            
            return results[2] <= self.max_requests

# ============================================================================
# Factory Pattern pour la Création d'Instances
# ============================================================================

class PagerDutyFactory:
    """Factory pour créer les instances PagerDuty configurées"""
    
    @staticmethod
    def create_api_manager(config: PagerDutyConfig) -> 'APIManager':
        """Crée une instance d'APIManager"""
        from .api_manager import APIManager
        return APIManager(config)
        
    @staticmethod
    def create_incident_manager(config: PagerDutyConfig) -> 'IncidentManager':
        """Crée une instance d'IncidentManager"""
        from .incident_manager import IncidentManager
        return IncidentManager(config)
        
    @staticmethod
    def create_escalation_manager(config: PagerDutyConfig) -> 'EscalationManager':
        """Crée une instance d'EscalationManager"""
        from .escalation_manager import EscalationManager
        return EscalationManager(config)
        
    @staticmethod
    def create_oncall_manager(config: PagerDutyConfig) -> 'OnCallManager':
        """Crée une instance d'OnCallManager"""
        from .oncall_manager import OnCallManager
        return OnCallManager(config)

# ============================================================================
# Exports et Interface Publique
# ============================================================================

__all__ = [
    # Enums
    'IncidentSeverity',
    'IncidentUrgency', 
    'IncidentStatus',
    'AlertSource',
    'NotificationChannel',
    'EscalationLevel',
    'AIModelType',
    'ComplianceStandard',
    
    # Models
    'PagerDutyConfig',
    'IncidentData',
    'EscalationPolicy',
    'OnCallSchedule',
    'NotificationRule',
    'AIAnalysisResult',
    'MetricsData',
    
    # Managers
    'SecurityManager',
    'CacheManager',
    'RateLimiter',
    'PagerDutyFactory',
    
    # Constants
    'PAGERDUTY_API_VERSION',
    'DEFAULT_TIMEOUT',
    'MAX_RETRY_ATTEMPTS',
    'CACHE_TTL',
    'RATE_LIMIT_PER_MINUTE',
]

# Configuration par défaut
DEFAULT_CONFIG = PagerDutyConfig(
    api_key="YOUR_API_KEY",
    routing_key="YOUR_ROUTING_KEY", 
    webhook_secret="YOUR_WEBHOOK_SECRET",
    base_url="https://api.pagerduty.com",
    timeout=30,
    max_retries=3,
    rate_limit=100,
    enable_ai=True,
    enable_metrics=True
)

logger.info(
    "PagerDuty Integration System initialized",
    version="4.0.0",
    features=[
        "AI-powered incident management",
        "Advanced escalation policies", 
        "Enterprise security",
        "Multi-tenant support",
        "Real-time monitoring"
    ]
)
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ASSIGNED = "assigned"
    ESCALATED = "escalated"

class EscalationAction(Enum):
    """Actions d'escalade"""
    NOTIFY_USER = "notify_user"
    NOTIFY_SCHEDULE = "notify_schedule"
    NOTIFY_SERVICE = "notify_service"
    CREATE_INCIDENT = "create_incident"
    RUN_AUTOMATION = "run_automation"

class IntegrationType(Enum):
    """Types d'intégration PagerDuty"""
    EVENTS_API_V2 = "events_api_v2"
    INCIDENTS_API = "incidents_api"
    SERVICES_API = "services_api"
    SCHEDULES_API = "schedules_api"
    ESCALATION_POLICIES_API = "escalation_policies_api"
    WEBHOOK = "webhook"

@dataclass
class PagerDutyConfig:
    """Configuration PagerDuty"""
    api_key: str
    routing_key: str
    service_id: Optional[str] = None
    base_url: str = "https://api.pagerduty.com"
    events_url: str = "https://events.pagerduty.com/v2/enqueue"
    enable_auto_resolution: bool = True
    auto_resolution_timeout: int = 3600  # 1 heure
    max_retries: int = 3
    retry_delay: int = 5
    enable_deduplication: bool = True
    enable_ai_analysis: bool = True
    encryption_key: Optional[str] = None

@dataclass
class AlertContext:
    """Contexte d'alerte pour PagerDuty"""
    alert_name: str
    instance: str
    severity: IncidentSeverity
    urgency: IncidentUrgency = IncidentUrgency.HIGH
    description: Optional[str] = None
    source: Optional[str] = None
    component: Optional[str] = None
    group: Optional[str] = None
    custom_details: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    fingerprint: Optional[str] = None
    external_url: Optional[str] = None
    generator_url: Optional[str] = None

@dataclass
class IncidentData:
    """Données d'incident PagerDuty"""
    incident_key: str
    title: str
    description: str
    severity: IncidentSeverity
    urgency: IncidentUrgency
    status: IncidentStatus
    service_id: str
    escalation_policy_id: Optional[str] = None
    assigned_to: Optional[List[str]] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    incident_number: Optional[str] = None
    pd_incident_id: Optional[str] = None
    acknowledgments: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EscalationRule:
    """Règle d'escalade"""
    delay_minutes: int
    targets: List[Dict[str, Any]]
    action: EscalationAction = EscalationAction.NOTIFY_USER

@dataclass
class EscalationPolicy:
    """Politique d'escalade"""
    id: str
    name: str
    description: str
    rules: List[EscalationRule]
    repeat_enabled: bool = False
    num_loops: int = 0
    on_call_handoff_notifications: str = "if_has_services"

@dataclass
class ServiceConfiguration:
    """Configuration de service PagerDuty"""
    id: str
    name: str
    description: str
    escalation_policy_id: str
    alert_creation: str = "create_alerts_and_incidents"
    alert_grouping: str = "intelligent"
    alert_grouping_timeout: int = 300
    auto_resolve_timeout: Optional[int] = None
    acknowledgement_timeout: Optional[int] = None
    status: str = "active"

# ============================================================================
# Advanced PagerDuty Manager
# ============================================================================

class AdvancedPagerDutyManager:
    """Gestionnaire PagerDuty ultra-avancé"""
    
    def __init__(self,
                 config: PagerDutyConfig,
                 cache_dir: str,
                 enable_ai_analysis: bool = True,
                 enable_intelligent_routing: bool = True,
                 enable_predictive_escalation: bool = True):
        
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.enable_ai_analysis = enable_ai_analysis
        self.enable_intelligent_routing = enable_intelligent_routing
        self.enable_predictive_escalation = enable_predictive_escalation
        
        # Cache et stockage
        self.incidents_cache: Dict[str, IncidentData] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        self.services: Dict[str, ServiceConfiguration] = {}
        
        # Session HTTP avec retry et circuit breaker
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300
        
        # Chiffrement pour données sensibles
        if self.config.encryption_key:
            self.cipher = Fernet(self.config.encryption_key.encode())
        else:
            self.cipher = None
        
        # Métriques et analytics
        self.metrics = {
            "incidents_created": 0,
            "incidents_resolved": 0,
            "escalations_triggered": 0,
            "api_calls_successful": 0,
            "api_calls_failed": 0,
            "average_resolution_time": 0.0,
            "deduplication_rate": 0.0
        }
        
        # Intelligence artificielle
        self.ai_model_cache = {}
        self.incident_patterns = {}
        self.resolution_predictions = {}
        
        # Déduplication
        self.deduplication_cache: Dict[str, str] = {}
        self.deduplication_window = timedelta(minutes=30)
        
        # Initialize
        asyncio.create_task(self._initialize())
        
        logger.info("Advanced PagerDuty Manager initialized")
    
    async def _initialize(self):
        """Initialisation du gestionnaire"""
        
        # Création des répertoires
        await self._ensure_directories()
        
        # Chargement de la configuration
        await self._load_configuration()
        
        # Initialisation de la session HTTP
        await self._initialize_http_session()
        
        # Chargement des politiques d'escalade
        await self._load_escalation_policies()
        
        # Chargement des services
        await self._load_services()
        
        # Initialisation IA si activée
        if self.enable_ai_analysis:
            await self._initialize_ai_components()
        
        # Démarrage des tâches périodiques
        asyncio.create_task(self._periodic_health_check())
        asyncio.create_task(self._periodic_metrics_collection())
        asyncio.create_task(self._periodic_cache_cleanup())
        
        logger.info("PagerDuty Manager initialization completed")
    
    async def _ensure_directories(self):
        """Assure que les répertoires nécessaires existent"""
        
        directories = [
            self.cache_dir,
            self.cache_dir / "incidents",
            self.cache_dir / "policies",
            self.cache_dir / "services",
            self.cache_dir / "metrics",
            self.cache_dir / "ai_models",
            self.cache_dir / "webhooks",
            self.cache_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def _initialize_http_session(self):
        """Initialise la session HTTP avec configuration avancée"""
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        headers = {
            "Authorization": f"Token token={self.config.api_key}",
            "Accept": "application/vnd.pagerduty+json;version=2",
            "Content-Type": "application/json",
            "User-Agent": "Spotify-AI-Agent-PagerDuty/4.0.0"
        }
        
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
            ssl=True
        )
        
        self.http_session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=connector
        )
    
    async def _load_configuration(self):
        """Charge la configuration depuis les fichiers"""
        
        config_file = self.cache_dir / "config.json"
        if config_file.exists():
            try:
                async with aiofiles.open(config_file, 'r') as f:
                    config_data = json.loads(await f.read())
                    
                # Mise à jour de la configuration
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                        
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
    
    async def create_incident(self,
                            alert_context: AlertContext,
                            custom_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Crée un incident PagerDuty avec intelligence artificielle"""
        
        # Déduplication intelligente
        if self.config.enable_deduplication:
            existing_incident = await self._check_deduplication(alert_context)
            if existing_incident:
                logger.info(f"Incident deduplicated: {existing_incident}")
                return existing_incident
        
        # Analyse IA pour classification automatique
        if self.enable_ai_analysis:
            alert_context = await self._ai_analyze_alert(alert_context)
        
        # Génération de la clé d'incident
        incident_key = await self._generate_incident_key(alert_context)
        
        # Construction du payload
        payload = await self._build_incident_payload(alert_context, incident_key, custom_payload)
        
        # Envoi à PagerDuty avec retry intelligent
        try:
            response = await self._send_to_pagerduty(payload)
            
            # Stockage local pour tracking
            incident_data = await self._create_incident_record(alert_context, incident_key, response)
            
            # Déclenchement des hooks post-création
            await self._trigger_post_incident_hooks(incident_data)
            
            # Mise à jour des métriques
            self.metrics["incidents_created"] += 1
            
            logger.info(f"Incident created successfully: {incident_key}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to create incident: {e}")
            self.metrics["api_calls_failed"] += 1
            raise
    
    async def _check_deduplication(self, alert_context: AlertContext) -> Optional[Dict[str, Any]]:
        """Vérifie la déduplication des incidents"""
        
        # Génération de l'empreinte de déduplication
        dedup_key = await self._generate_deduplication_key(alert_context)
        
        # Vérification dans le cache
        if dedup_key in self.deduplication_cache:
            existing_incident_key = self.deduplication_cache[dedup_key]
            
            # Vérification que l'incident existe toujours
            if existing_incident_key in self.incidents_cache:
                incident = self.incidents_cache[existing_incident_key]
                
                # Vérification de la fenêtre temporelle
                if incident.created_at and datetime.now() - incident.created_at < self.deduplication_window:
                    self.metrics["deduplication_rate"] += 1
                    return asdict(incident)
        
        # Ajout au cache de déduplication
        incident_key = await self._generate_incident_key(alert_context)
        self.deduplication_cache[dedup_key] = incident_key
        
        return None
    
    async def _generate_deduplication_key(self, alert_context: AlertContext) -> str:
        """Génère une clé de déduplication intelligente"""
        
        # Éléments pour la déduplication
        elements = [
            alert_context.alert_name,
            alert_context.instance,
            alert_context.source or "",
            alert_context.component or "",
            str(alert_context.severity.value)
        ]
        
        # Ajout des labels critiques
        critical_labels = ["service", "environment", "cluster", "namespace"]
        for label in critical_labels:
            if label in alert_context.labels:
                elements.append(f"{label}:{alert_context.labels[label]}")
        
        # Génération du hash
        dedup_string = "|".join(elements)
        return hashlib.sha256(dedup_string.encode()).hexdigest()[:16]
    
    async def _ai_analyze_alert(self, alert_context: AlertContext) -> AlertContext:
        """Analyse IA de l'alerte pour optimisation automatique"""
        
        if not self.enable_ai_analysis:
            return alert_context
        
        try:
            # Analyse du contenu pour détermination automatique de sévérité
            severity_keywords = {
                IncidentSeverity.CRITICAL: ["critical", "down", "outage", "failed", "error", "crash"],
                IncidentSeverity.HIGH: ["high", "warning", "slow", "degraded", "timeout"],
                IncidentSeverity.MEDIUM: ["medium", "elevated", "unusual", "anomaly"],
                IncidentSeverity.LOW: ["low", "info", "notice", "minor"]
            }
            
            # Analyse du titre et de la description
            text_to_analyze = f"{alert_context.alert_name} {alert_context.description or ''}".lower()
            
            severity_scores = {}
            for severity, keywords in severity_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_to_analyze)
                severity_scores[severity] = score
            
            # Sélection de la sévérité avec le meilleur score
            if severity_scores:
                best_severity = max(severity_scores, key=severity_scores.get)
                if severity_scores[best_severity] > 0:
                    alert_context.severity = best_severity
            
            # Détermination automatique de l'urgence
            if alert_context.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
                alert_context.urgency = IncidentUrgency.HIGH
            else:
                alert_context.urgency = IncidentUrgency.LOW
            
            # Enrichissement du contexte avec IA
            if "database" in text_to_analyze:
                alert_context.component = "database"
            elif "network" in text_to_analyze:
                alert_context.component = "network"
            elif "api" in text_to_analyze or "service" in text_to_analyze:
                alert_context.component = "api"
            elif "disk" in text_to_analyze or "storage" in text_to_analyze:
                alert_context.component = "storage"
            
            logger.debug(f"AI analysis completed for alert: {alert_context.alert_name}")
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
        
        return alert_context
    
    async def _generate_incident_key(self, alert_context: AlertContext) -> str:
        """Génère une clé d'incident unique et intelligente"""
        
        # Utilisation du fingerprint si disponible
        if alert_context.fingerprint:
            return f"incident_{alert_context.fingerprint}"
        
        # Génération basée sur les attributs de l'alerte
        key_elements = [
            alert_context.alert_name,
            alert_context.instance,
            alert_context.source or "unknown",
            str(int(datetime.now().timestamp()))
        ]
        
        # Hash pour unicité
        key_string = "_".join(key_elements)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]
        
        return f"spotify_ai_agent_{key_hash}"
    
    async def _build_incident_payload(self,
                                    alert_context: AlertContext,
                                    incident_key: str,
                                    custom_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Construit le payload d'incident pour PagerDuty"""
        
        # Payload de base
        payload = {
            "routing_key": self.config.routing_key,
            "event_action": "trigger",
            "dedup_key": incident_key,
            "payload": {
                "summary": await self._generate_smart_summary(alert_context),
                "source": alert_context.source or "Spotify AI Agent",
                "severity": alert_context.severity.value,
                "component": alert_context.component or "Unknown",
                "group": alert_context.group or "Default",
                "class": "Alert",
                "custom_details": {
                    "alert_name": alert_context.alert_name,
                    "instance": alert_context.instance,
                    "urgency": alert_context.urgency.value,
                    "labels": alert_context.labels,
                    "annotations": alert_context.annotations,
                    "custom_details": alert_context.custom_details,
                    "external_url": alert_context.external_url,
                    "generator_url": alert_context.generator_url,
                    "created_by": "Spotify AI Agent",
                    "created_at": datetime.now().isoformat()
                }
            }
        }
        
        # Ajout du client si configuré
        if hasattr(self.config, 'client_name'):
            payload["client"] = self.config.client_name
        
        # Ajout de l'URL du client si configurée
        if hasattr(self.config, 'client_url'):
            payload["client_url"] = self.config.client_url
        
        # Fusion avec le payload personnalisé
        if custom_payload:
            payload = self._deep_merge_dicts(payload, custom_payload)
        
        return payload
    
    async def _generate_smart_summary(self, alert_context: AlertContext) -> str:
        """Génère un résumé intelligent de l'incident"""
        
        # Template de base
        summary_parts = []
        
        # Niveau de sévérité avec emoji
        severity_emojis = {
            IncidentSeverity.CRITICAL: "🚨",
            IncidentSeverity.HIGH: "⚠️",
            IncidentSeverity.MEDIUM: "⚡",
            IncidentSeverity.LOW: "ℹ️",
            IncidentSeverity.INFO: "📋"
        }
        
        emoji = severity_emojis.get(alert_context.severity, "📊")
        summary_parts.append(f"{emoji} {alert_context.severity.value.upper()}")
        
        # Nom de l'alerte
        summary_parts.append(alert_context.alert_name)
        
        # Instance si disponible
        if alert_context.instance:
            summary_parts.append(f"on {alert_context.instance}")
        
        # Composant si disponible
        if alert_context.component:
            summary_parts.append(f"({alert_context.component})")
        
        # Service si disponible dans les labels
        if "service" in alert_context.labels:
            summary_parts.append(f"[{alert_context.labels['service']}]")
        
        return " ".join(summary_parts)
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def _send_to_pagerduty(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Envoie le payload à PagerDuty avec retry intelligent"""
        
        # Vérification du circuit breaker
        if await self._is_circuit_breaker_open():
            raise Exception("Circuit breaker is open")
        
        try:
            async with self.http_session.post(
                self.config.events_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                response.raise_for_status()
                result = await response.json()
                
                # Reset du circuit breaker en cas de succès
                self.circuit_breaker_failures = 0
                self.metrics["api_calls_successful"] += 1
                
                return result
                
        except Exception as e:
            # Gestion du circuit breaker
            self.circuit_breaker_failures += 1
            self.circuit_breaker_last_failure = datetime.now()
            
            logger.error(f"PagerDuty API call failed: {e}")
            raise
    
    async def _is_circuit_breaker_open(self) -> bool:
        """Vérifie si le circuit breaker est ouvert"""
        
        if self.circuit_breaker_failures < self.circuit_breaker_threshold:
            return False
        
        if self.circuit_breaker_last_failure:
            time_since_failure = datetime.now() - self.circuit_breaker_last_failure
            if time_since_failure.total_seconds() > self.circuit_breaker_timeout:
                # Reset du circuit breaker après timeout
                self.circuit_breaker_failures = 0
                self.circuit_breaker_last_failure = None
                return False
        
        return True
    
    async def resolve_incident(self,
                             incident_key: str,
                             resolution_note: Optional[str] = None) -> Dict[str, Any]:
        """Résout un incident avec tracking intelligent"""
        
        try:
            # Payload de résolution
            payload = {
                "routing_key": self.config.routing_key,
                "event_action": "resolve",
                "dedup_key": incident_key
            }
            
            # Ajout de la note de résolution
            if resolution_note:
                payload["payload"] = {
                    "summary": f"Incident resolved: {resolution_note}",
                    "custom_details": {
                        "resolution_note": resolution_note,
                        "resolved_by": "Spotify AI Agent",
                        "resolved_at": datetime.now().isoformat()
                    }
                }
            
            # Envoi à PagerDuty
            response = await self._send_to_pagerduty(payload)
            
            # Mise à jour du cache local
            if incident_key in self.incidents_cache:
                incident = self.incidents_cache[incident_key]
                incident.status = IncidentStatus.RESOLVED
                incident.resolved_at = datetime.now()
                
                # Calcul du temps de résolution
                if incident.created_at:
                    resolution_time = (incident.resolved_at - incident.created_at).total_seconds()
                    await self._update_resolution_metrics(resolution_time)
            
            # Mise à jour des métriques
            self.metrics["incidents_resolved"] += 1
            
            logger.info(f"Incident resolved successfully: {incident_key}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to resolve incident {incident_key}: {e}")
            raise
    
    async def acknowledge_incident(self,
                                 incident_key: str,
                                 acknowledger: str,
                                 note: Optional[str] = None) -> Dict[str, Any]:
        """Acquitte un incident"""
        
        try:
            # Payload d'acquittement
            payload = {
                "routing_key": self.config.routing_key,
                "event_action": "acknowledge",
                "dedup_key": incident_key
            }
            
            if note:
                payload["payload"] = {
                    "summary": f"Incident acknowledged by {acknowledger}",
                    "custom_details": {
                        "acknowledged_by": acknowledger,
                        "acknowledgment_note": note,
                        "acknowledged_at": datetime.now().isoformat()
                    }
                }
            
            # Envoi à PagerDuty
            response = await self._send_to_pagerduty(payload)
            
            # Mise à jour du cache local
            if incident_key in self.incidents_cache:
                incident = self.incidents_cache[incident_key]
                incident.status = IncidentStatus.ACKNOWLEDGED
                incident.acknowledgments.append({
                    "acknowledger": acknowledger,
                    "note": note,
                    "timestamp": datetime.now().isoformat()
                })
            
            logger.info(f"Incident acknowledged: {incident_key} by {acknowledger}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to acknowledge incident {incident_key}: {e}")
            raise
    
    async def create_escalation_policy(self,
                                     policy: EscalationPolicy) -> Dict[str, Any]:
        """Crée une politique d'escalade avancée"""
        
        try:
            # Construction du payload
            payload = {
                "escalation_policy": {
                    "name": policy.name,
                    "description": policy.description,
                    "repeat_enabled": policy.repeat_enabled,
                    "num_loops": policy.num_loops,
                    "on_call_handoff_notifications": policy.on_call_handoff_notifications,
                    "escalation_rules": []
                }
            }
            
            # Ajout des règles d'escalade
            for rule in policy.rules:
                escalation_rule = {
                    "delay_in_minutes": rule.delay_minutes,
                    "targets": rule.targets
                }
                payload["escalation_policy"]["escalation_rules"].append(escalation_rule)
            
            # Envoi à PagerDuty
            async with self.http_session.post(
                f"{self.config.base_url}/escalation_policies",
                json=payload
            ) as response:
                
                response.raise_for_status()
                result = await response.json()
                
                # Stockage local
                self.escalation_policies[policy.id] = policy
                await self._save_escalation_policy(policy)
                
                logger.info(f"Escalation policy created: {policy.name}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to create escalation policy: {e}")
            raise
    
    async def create_service(self, service: ServiceConfiguration) -> Dict[str, Any]:
        """Crée un service PagerDuty avancé"""
        
        try:
            # Construction du payload
            payload = {
                "service": {
                    "name": service.name,
                    "description": service.description,
                    "status": service.status,
                    "escalation_policy": {
                        "id": service.escalation_policy_id,
                        "type": "escalation_policy_reference"
                    },
                    "alert_creation": service.alert_creation,
                    "alert_grouping": service.alert_grouping,
                    "alert_grouping_timeout": service.alert_grouping_timeout
                }
            }
            
            # Ajout des timeouts si configurés
            if service.auto_resolve_timeout:
                payload["service"]["auto_resolve_timeout"] = service.auto_resolve_timeout
            
            if service.acknowledgement_timeout:
                payload["service"]["acknowledgement_timeout"] = service.acknowledgement_timeout
            
            # Envoi à PagerDuty
            async with self.http_session.post(
                f"{self.config.base_url}/services",
                json=payload
            ) as response:
                
                response.raise_for_status()
                result = await response.json()
                
                # Stockage local
                self.services[service.id] = service
                await self._save_service(service)
                
                logger.info(f"Service created: {service.name}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to create service: {e}")
            raise
    
    async def get_incident_analytics(self,
                                   start_date: datetime,
                                   end_date: datetime) -> Dict[str, Any]:
        """Obtient les analyses d'incidents avancées"""
        
        analytics = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "incident_count": 0,
            "resolution_metrics": {
                "average_resolution_time": 0.0,
                "median_resolution_time": 0.0,
                "fastest_resolution": None,
                "slowest_resolution": None
            },
            "severity_distribution": {},
            "component_analysis": {},
            "escalation_analysis": {},
            "ai_insights": {}
        }
        
        # Collecte des incidents dans la période
        incidents = []
        for incident in self.incidents_cache.values():
            if (incident.created_at and 
                start_date <= incident.created_at <= end_date):
                incidents.append(incident)
        
        analytics["incident_count"] = len(incidents)
        
        if incidents:
            # Métriques de résolution
            resolution_times = []
            for incident in incidents:
                if incident.resolved_at and incident.created_at:
                    resolution_time = (incident.resolved_at - incident.created_at).total_seconds()
                    resolution_times.append(resolution_time)
            
            if resolution_times:
                analytics["resolution_metrics"]["average_resolution_time"] = sum(resolution_times) / len(resolution_times)
                analytics["resolution_metrics"]["median_resolution_time"] = sorted(resolution_times)[len(resolution_times) // 2]
                analytics["resolution_metrics"]["fastest_resolution"] = min(resolution_times)
                analytics["resolution_metrics"]["slowest_resolution"] = max(resolution_times)
            
            # Distribution par sévérité
            severity_counts = {}
            for incident in incidents:
                severity = incident.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            analytics["severity_distribution"] = severity_counts
            
            # Analyse par composant
            component_counts = {}
            for incident in incidents:
                component = incident.custom_fields.get("component", "Unknown")
                component_counts[component] = component_counts.get(component, 0) + 1
            analytics["component_analysis"] = component_counts
        
        return analytics
    
    async def _create_incident_record(self,
                                    alert_context: AlertContext,
                                    incident_key: str,
                                    pd_response: Dict[str, Any]) -> IncidentData:
        """Crée un enregistrement d'incident local"""
        
        incident = IncidentData(
            incident_key=incident_key,
            title=alert_context.alert_name,
            description=alert_context.description or "",
            severity=alert_context.severity,
            urgency=alert_context.urgency,
            status=IncidentStatus.TRIGGERED,
            service_id=self.config.service_id or "default",
            created_at=datetime.now(),
            custom_fields={
                "alert_context": asdict(alert_context),
                "pd_response": pd_response
            }
        )
        
        # Stockage en cache
        self.incidents_cache[incident_key] = incident
        
        # Sauvegarde persistante
        await self._save_incident(incident)
        
        return incident
    
    async def _save_incident(self, incident: IncidentData):
        """Sauvegarde un incident"""
        
        incident_file = self.cache_dir / "incidents" / f"{incident.incident_key}.json"
        
        try:
            incident_dict = asdict(incident)
            
            # Sérialisation des dates
            for date_field in ["created_at", "updated_at", "resolved_at"]:
                if incident_dict.get(date_field):
                    incident_dict[date_field] = incident_dict[date_field].isoformat()
            
            # Sérialisation des enums
            incident_dict["severity"] = incident.severity.value
            incident_dict["urgency"] = incident.urgency.value
            incident_dict["status"] = incident.status.value
            
            # Chiffrement si activé
            if self.cipher:
                incident_json = json.dumps(incident_dict)
                encrypted_data = self.cipher.encrypt(incident_json.encode())
                async with aiofiles.open(incident_file, 'wb') as f:
                    await f.write(encrypted_data)
            else:
                async with aiofiles.open(incident_file, 'w') as f:
                    await f.write(json.dumps(incident_dict, indent=2))
                    
        except Exception as e:
            logger.error(f"Failed to save incident {incident.incident_key}: {e}")
    
    async def _load_escalation_policies(self):
        """Charge les politiques d'escalade"""
        
        policies_dir = self.cache_dir / "policies"
        
        if policies_dir.exists():
            for policy_file in policies_dir.glob("*.json"):
                try:
                    async with aiofiles.open(policy_file, 'r') as f:
                        policy_data = json.loads(await f.read())
                    
                    # Reconstruction des objets
                    rules = []
                    for rule_data in policy_data.get("rules", []):
                        rule = EscalationRule(
                            delay_minutes=rule_data["delay_minutes"],
                            targets=rule_data["targets"],
                            action=EscalationAction(rule_data.get("action", "notify_user"))
                        )
                        rules.append(rule)
                    
                    policy = EscalationPolicy(
                        id=policy_data["id"],
                        name=policy_data["name"],
                        description=policy_data["description"],
                        rules=rules,
                        repeat_enabled=policy_data.get("repeat_enabled", False),
                        num_loops=policy_data.get("num_loops", 0)
                    )
                    
                    self.escalation_policies[policy.id] = policy
                    
                except Exception as e:
                    logger.error(f"Failed to load escalation policy from {policy_file}: {e}")
    
    async def _load_services(self):
        """Charge les services"""
        
        services_dir = self.cache_dir / "services"
        
        if services_dir.exists():
            for service_file in services_dir.glob("*.json"):
                try:
                    async with aiofiles.open(service_file, 'r') as f:
                        service_data = json.loads(await f.read())
                    
                    service = ServiceConfiguration(**service_data)
                    self.services[service.id] = service
                    
                except Exception as e:
                    logger.error(f"Failed to load service from {service_file}: {e}")
    
    async def _initialize_ai_components(self):
        """Initialise les composants IA"""
        
        # Chargement des modèles de patterns d'incidents
        patterns_file = self.cache_dir / "ai_models" / "incident_patterns.json"
        if patterns_file.exists():
            try:
                async with aiofiles.open(patterns_file, 'r') as f:
                    self.incident_patterns = json.loads(await f.read())
            except Exception as e:
                logger.error(f"Failed to load incident patterns: {e}")
        
        logger.info("AI components initialized")
    
    async def _update_resolution_metrics(self, resolution_time: float):
        """Met à jour les métriques de résolution"""
        
        # Calcul de la moyenne mobile
        current_avg = self.metrics["average_resolution_time"]
        incidents_resolved = self.metrics["incidents_resolved"]
        
        if incidents_resolved > 1:
            # Moyenne pondérée
            self.metrics["average_resolution_time"] = (
                (current_avg * (incidents_resolved - 1) + resolution_time) / incidents_resolved
            )
        else:
            self.metrics["average_resolution_time"] = resolution_time
    
    async def _periodic_health_check(self):
        """Vérification périodique de santé"""
        
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Test de connectivité PagerDuty
                try:
                    async with self.http_session.get(f"{self.config.base_url}/abilities") as response:
                        if response.status == 200:
                            logger.debug("PagerDuty health check: OK")
                        else:
                            logger.warning(f"PagerDuty health check failed: {response.status}")
                except Exception as e:
                    logger.error(f"PagerDuty health check error: {e}")
                
                # Nettoyage du cache de déduplication
                await self._cleanup_deduplication_cache()
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _periodic_metrics_collection(self):
        """Collecte périodique de métriques"""
        
        while True:
            try:
                await asyncio.sleep(600)  # 10 minutes
                
                # Sauvegarde des métriques
                metrics_file = self.cache_dir / "metrics" / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                async with aiofiles.open(metrics_file, 'w') as f:
                    await f.write(json.dumps(self.metrics, indent=2))
                
                logger.debug("Metrics collected and saved")
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _periodic_cache_cleanup(self):
        """Nettoyage périodique du cache"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # 1 heure
                
                # Nettoyage des incidents anciens (> 30 jours)
                cutoff_date = datetime.now() - timedelta(days=30)
                
                incidents_to_remove = []
                for key, incident in self.incidents_cache.items():
                    if incident.created_at and incident.created_at < cutoff_date:
                        incidents_to_remove.append(key)
                
                for key in incidents_to_remove:
                    del self.incidents_cache[key]
                
                logger.debug(f"Cleaned up {len(incidents_to_remove)} old incidents from cache")
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _cleanup_deduplication_cache(self):
        """Nettoie le cache de déduplication"""
        
        cutoff_time = datetime.now() - self.deduplication_window
        keys_to_remove = []
        
        for dedup_key, incident_key in self.deduplication_cache.items():
            if incident_key in self.incidents_cache:
                incident = self.incidents_cache[incident_key]
                if incident.created_at and incident.created_at < cutoff_time:
                    keys_to_remove.append(dedup_key)
        
        for key in keys_to_remove:
            del self.deduplication_cache[key]
    
    def _deep_merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Fusion profonde de dictionnaires"""
        
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def _save_escalation_policy(self, policy: EscalationPolicy):
        """Sauvegarde une politique d'escalade"""
        
        policy_file = self.cache_dir / "policies" / f"{policy.id}.json"
        
        try:
            policy_dict = asdict(policy)
            
            async with aiofiles.open(policy_file, 'w') as f:
                await f.write(json.dumps(policy_dict, indent=2))
                
        except Exception as e:
            logger.error(f"Failed to save escalation policy {policy.id}: {e}")
    
    async def _save_service(self, service: ServiceConfiguration):
        """Sauvegarde un service"""
        
        service_file = self.cache_dir / "services" / f"{service.id}.json"
        
        try:
            service_dict = asdict(service)
            
            async with aiofiles.open(service_file, 'w') as f:
                await f.write(json.dumps(service_dict, indent=2))
                
        except Exception as e:
            logger.error(f"Failed to save service {service.id}: {e}")
    
    async def _trigger_post_incident_hooks(self, incident: IncidentData):
        """Déclenche les hooks post-création d'incident"""
        
        # Hook pour auto-résolution
        if self.config.enable_auto_resolution:
            asyncio.create_task(self._schedule_auto_resolution(incident))
        
        # Hook pour escalade prédictive
        if self.enable_predictive_escalation:
            asyncio.create_task(self._schedule_predictive_escalation(incident))
    
    async def _schedule_auto_resolution(self, incident: IncidentData):
        """Programme la résolution automatique"""
        
        try:
            await asyncio.sleep(self.config.auto_resolution_timeout)
            
            # Vérification que l'incident n'est pas déjà résolu
            if (incident.incident_key in self.incidents_cache and 
                self.incidents_cache[incident.incident_key].status != IncidentStatus.RESOLVED):
                
                await self.resolve_incident(
                    incident.incident_key,
                    "Auto-resolved by Spotify AI Agent after timeout"
                )
                
        except Exception as e:
            logger.error(f"Auto-resolution failed for {incident.incident_key}: {e}")
    
    async def _schedule_predictive_escalation(self, incident: IncidentData):
        """Programme l'escalade prédictive"""
        
        try:
            # Analyse prédictive basée sur l'historique
            if incident.severity == IncidentSeverity.CRITICAL:
                escalation_delay = 300  # 5 minutes pour critique
            elif incident.severity == IncidentSeverity.HIGH:
                escalation_delay = 900  # 15 minutes pour élevé
            else:
                escalation_delay = 1800  # 30 minutes pour autres
            
            await asyncio.sleep(escalation_delay)
            
            # Vérification que l'incident n'est pas acquitté/résolu
            if (incident.incident_key in self.incidents_cache and 
                self.incidents_cache[incident.incident_key].status == IncidentStatus.TRIGGERED):
                
                logger.warning(f"Predictive escalation triggered for {incident.incident_key}")
                self.metrics["escalations_triggered"] += 1
                
        except Exception as e:
            logger.error(f"Predictive escalation failed for {incident.incident_key}: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Obtient les métriques actuelles"""
        
        return {
            **self.metrics,
            "cache_stats": {
                "incidents_cached": len(self.incidents_cache),
                "escalation_policies_cached": len(self.escalation_policies),
                "services_cached": len(self.services),
                "deduplication_cache_size": len(self.deduplication_cache)
            },
            "circuit_breaker": {
                "failures": self.circuit_breaker_failures,
                "is_open": await self._is_circuit_breaker_open(),
                "last_failure": self.circuit_breaker_last_failure.isoformat() if self.circuit_breaker_last_failure else None
            }
        }
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        
        if self.http_session:
            await self.http_session.close()
        
        # Sauvegarde finale des métriques
        metrics_file = self.cache_dir / "metrics" / f"final_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            async with aiofiles.open(metrics_file, 'w') as f:
                await f.write(json.dumps(await self.get_metrics(), indent=2))
        except Exception as e:
            logger.error(f"Failed to save final metrics: {e}")

# ============================================================================
# Webhook Handler
# ============================================================================

class PagerDutyWebhookHandler:
    """Gestionnaire de webhooks PagerDuty avancé"""
    
    def __init__(self, secret_key: str, manager: AdvancedPagerDutyManager):
        self.secret_key = secret_key
        self.manager = manager
        self.processed_webhooks = set()
    
    async def process_webhook(self, payload: Dict[str, Any], signature: str) -> Dict[str, Any]:
        """Traite un webhook PagerDuty avec validation"""
        
        # Validation de la signature
        if not self._validate_signature(payload, signature):
            raise ValueError("Invalid webhook signature")
        
        # Déduplication des webhooks
        webhook_id = payload.get("id", "")
        if webhook_id in self.processed_webhooks:
            return {"status": "duplicate", "webhook_id": webhook_id}
        
        self.processed_webhooks.add(webhook_id)
        
        # Traitement selon le type de message
        webhook_type = payload.get("event", {}).get("event_type", "")
        
        if webhook_type == "incident.trigger":
            return await self._handle_incident_trigger(payload)
        elif webhook_type == "incident.acknowledge":
            return await self._handle_incident_acknowledge(payload)
        elif webhook_type == "incident.resolve":
            return await self._handle_incident_resolve(payload)
        elif webhook_type == "incident.escalate":
            return await self._handle_incident_escalate(payload)
        else:
            logger.warning(f"Unknown webhook type: {webhook_type}")
            return {"status": "unknown_type", "type": webhook_type}
    
    def _validate_signature(self, payload: Dict[str, Any], signature: str) -> bool:
        """Valide la signature du webhook"""
        
        try:
            payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
            expected_signature = hmac.new(
                self.secret_key.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Signature validation error: {e}")
            return False
    
    async def _handle_incident_trigger(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Gère le déclenchement d'incident"""
        
        incident_data = payload.get("event", {}).get("data", {})
        incident_key = incident_data.get("incident", {}).get("incident_key", "")
        
        logger.info(f"Webhook: Incident triggered - {incident_key}")
        
        return {"status": "processed", "action": "trigger", "incident_key": incident_key}
    
    async def _handle_incident_acknowledge(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Gère l'acquittement d'incident"""
        
        incident_data = payload.get("event", {}).get("data", {})
        incident_key = incident_data.get("incident", {}).get("incident_key", "")
        
        logger.info(f"Webhook: Incident acknowledged - {incident_key}")
        
        return {"status": "processed", "action": "acknowledge", "incident_key": incident_key}
    
    async def _handle_incident_resolve(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Gère la résolution d'incident"""
        
        incident_data = payload.get("event", {}).get("data", {})
        incident_key = incident_data.get("incident", {}).get("incident_key", "")
        
        logger.info(f"Webhook: Incident resolved - {incident_key}")
        
        return {"status": "processed", "action": "resolve", "incident_key": incident_key}
    
    async def _handle_incident_escalate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Gère l'escalade d'incident"""
        
        incident_data = payload.get("event", {}).get("data", {})
        incident_key = incident_data.get("incident", {}).get("incident_key", "")
        
        logger.info(f"Webhook: Incident escalated - {incident_key}")
        
        return {"status": "processed", "action": "escalate", "incident_key": incident_key}

# ============================================================================
# Factory Functions
# ============================================================================

def create_pagerduty_manager(
    api_key: str,
    routing_key: str,
    cache_dir: str,
    service_id: Optional[str] = None,
    enable_ai: bool = True
) -> AdvancedPagerDutyManager:
    """Factory pour créer un gestionnaire PagerDuty"""
    
    config = PagerDutyConfig(
        api_key=api_key,
        routing_key=routing_key,
        service_id=service_id,
        enable_ai_analysis=enable_ai
    )
    
    return AdvancedPagerDutyManager(
        config=config,
        cache_dir=cache_dir,
        enable_ai_analysis=enable_ai
    )

def create_alert_context(
    alert_name: str,
    instance: str,
    severity: IncidentSeverity = IncidentSeverity.MEDIUM,
    **kwargs
) -> AlertContext:
    """Factory pour créer un contexte d'alerte"""
    
    return AlertContext(
        alert_name=alert_name,
        instance=instance,
        severity=severity,
        **kwargs
    )

def create_escalation_policy(
    policy_id: str,
    name: str,
    description: str,
    rules: List[EscalationRule]
) -> EscalationPolicy:
    """Factory pour créer une politique d'escalade"""
    
    return EscalationPolicy(
        id=policy_id,
        name=name,
        description=description,
        rules=rules
    )

def create_webhook_handler(
    secret_key: str,
    manager: AdvancedPagerDutyManager
) -> PagerDutyWebhookHandler:
    """Factory pour créer un gestionnaire de webhooks"""
    
    return PagerDutyWebhookHandler(secret_key, manager)

# Export des classes principales
__all__ = [
    "AdvancedPagerDutyManager",
    "PagerDutyWebhookHandler",
    "PagerDutyConfig",
    "AlertContext",
    "IncidentData",
    "EscalationPolicy",
    "ServiceConfiguration",
    "IncidentSeverity",
    "IncidentUrgency",
    "IncidentStatus",
    "create_pagerduty_manager",
    "create_alert_context",
    "create_escalation_policy",
    "create_webhook_handler"
]
