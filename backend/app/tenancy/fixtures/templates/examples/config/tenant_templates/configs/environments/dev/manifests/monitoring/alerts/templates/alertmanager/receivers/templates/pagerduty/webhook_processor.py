"""
Advanced Webhook Processor for PagerDuty Integration

Ce module fournit un processeur de webhooks sophistiqué avec validation de sécurité,
traitement asynchrone, transformation de données, et intégration IA.

Fonctionnalités:
- Validation HMAC et signature des webhooks
- Traitement asynchrone avec queue prioritaire
- Transformation et enrichissement des données
- Retry intelligent avec backoff exponentiel
- Rate limiting et protection DDoS
- Audit logging complet
- Intégration avec systèmes externes

Version: 4.0.0
Développé par l'équipe Spotify AI Agent
"""

import asyncio
import json
import hmac
import hashlib
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import structlog
import aiofiles
import aioredis
import aiohttp
from aiohttp import web, ClientTimeout
import backoff
import tenacity
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, validator
import jsonschema
from collections import defaultdict, deque

from . import (
    IncidentData, IncidentStatus, IncidentSeverity,
    SecurityManager, RateLimiter, logger
)

# ============================================================================
# Configuration Webhooks
# ============================================================================

@dataclass
class WebhookConfig:
    """Configuration du processeur de webhooks"""
    port: int = 8080
    host: str = "0.0.0.0"
    max_payload_size: int = 1024 * 1024  # 1MB
    timeout: int = 30
    max_concurrent_processors: int = 50
    queue_max_size: int = 10000
    retry_max_attempts: int = 3
    rate_limit_per_minute: int = 1000
    enable_signature_validation: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True

class WebhookEventType(Enum):
    """Types d'événements webhook supportés"""
    INCIDENT_TRIGGERED = "incident.triggered"
    INCIDENT_ACKNOWLEDGED = "incident.acknowledged"
    INCIDENT_ESCALATED = "incident.escalated"
    INCIDENT_RESOLVED = "incident.resolved"
    INCIDENT_ASSIGNED = "incident.assigned"
    INCIDENT_DELEGATED = "incident.delegated"
    INCIDENT_PRIORITY_UPDATED = "incident.priority_updated"
    INCIDENT_RESPONDER_ADDED = "incident.responder.added"
    INCIDENT_RESPONDER_REPLIED = "incident.responder.replied"
    INCIDENT_STATUS_UPDATE_PUBLISHED = "incident.status_update_published"
    INCIDENT_REOPENED = "incident.reopened"
    SERVICE_CREATED = "service.created"
    SERVICE_UPDATED = "service.updated"
    SERVICE_DELETED = "service.deleted"

class ProcessingPriority(Enum):
    """Priorités de traitement des webhooks"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class ProcessingStatus(Enum):
    """Statuts de traitement des webhooks"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

# ============================================================================
# Modèles de Données
# ============================================================================

class WebhookPayload(BaseModel):
    """Modèle pour les payloads webhook PagerDuty"""
    event_type: str
    created_on: datetime
    id: str = Field(..., min_length=1)
    data: Dict[str, Any]
    
    @validator('created_on', pre=True)
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

class WebhookEvent(BaseModel):
    """Événement webhook enrichi"""
    webhook_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    payload: WebhookPayload
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    priority: ProcessingPriority = ProcessingPriority.MEDIUM
    status: ProcessingStatus = ProcessingStatus.PENDING
    retry_count: int = 0
    error_message: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    signature_valid: bool = False

class WebhookTransformationRule(BaseModel):
    """Règle de transformation des webhooks"""
    event_type: WebhookEventType
    conditions: Dict[str, Any] = Field(default_factory=dict)
    transformations: List[Dict[str, Any]] = Field(default_factory=list)
    target_systems: List[str] = Field(default_factory=list)
    enabled: bool = True

# ============================================================================
# Processeur Principal
# ============================================================================

class WebhookProcessor:
    """Processeur de webhooks avancé"""
    
    def __init__(self, config: WebhookConfig):
        self.config = config
        self.app = web.Application()
        self.security_manager = None
        self.rate_limiter = None
        self.redis_pool = None
        self.processing_queue = asyncio.Queue(maxsize=config.queue_max_size)
        self.active_processors = 0
        self.webhook_handlers = {}
        self.transformation_rules = []
        self.event_stats = defaultdict(int)
        self.performance_metrics = deque(maxlen=1000)
        
        # Configuration des routes
        self._setup_routes()
        
    async def initialize(self, redis_url: str, encryption_key: str, webhook_secret: str):
        """Initialise le processeur de webhooks"""
        try:
            # Sécurité
            self.security_manager = SecurityManager(encryption_key)
            self.webhook_secret = webhook_secret
            
            # Rate limiting
            if self.config.enable_rate_limiting:
                self.rate_limiter = RateLimiter(redis_url, self.config.rate_limit_per_minute)
                await self.rate_limiter.initialize()
                
            # Redis pour cache et persistance
            self.redis_pool = aioredis.ConnectionPool.from_url(redis_url)
            
            # Chargement des règles de transformation
            await self._load_transformation_rules()
            
            # Démarrage des workers de traitement
            for _ in range(self.config.max_concurrent_processors):
                asyncio.create_task(self._processing_worker())
                
            logger.info("Webhook processor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize webhook processor: {e}")
            raise
            
    def _setup_routes(self):
        """Configure les routes HTTP"""
        self.app.router.add_post('/webhook/pagerduty', self._handle_pagerduty_webhook)
        self.app.router.add_post('/webhook/generic', self._handle_generic_webhook)
        self.app.router.add_get('/webhook/health', self._health_check)
        self.app.router.add_get('/webhook/metrics', self._get_metrics)
        
        # Middleware pour logging et sécurité
        self.app.middlewares.append(self._security_middleware)
        self.app.middlewares.append(self._logging_middleware)
        
    async def start_server(self):
        """Démarre le serveur webhook"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()
        
        logger.info(f"Webhook server started on {self.config.host}:{self.config.port}")
        
    async def _handle_pagerduty_webhook(self, request: web.Request) -> web.Response:
        """Traite les webhooks PagerDuty"""
        try:
            start_time = time.time()
            
            # Validation de la taille du payload
            content_length = int(request.headers.get('Content-Length', 0))
            if content_length > self.config.max_payload_size:
                return web.Response(status=413, text="Payload too large")
                
            # Lecture du payload
            payload_data = await request.text()
            
            # Validation de signature si activée
            if self.config.enable_signature_validation:
                signature = request.headers.get('X-PagerDuty-Signature', '')
                if not self._validate_signature(payload_data, signature):
                    logger.warning("Invalid webhook signature", source_ip=request.remote)
                    return web.Response(status=401, text="Invalid signature")
                    
            # Rate limiting
            if self.config.enable_rate_limiting:
                client_id = request.remote or 'unknown'
                if not await self.rate_limiter.is_allowed(f"webhook:{client_id}"):
                    return web.Response(status=429, text="Rate limit exceeded")
                    
            # Parsing du JSON
            try:
                webhook_data = json.loads(payload_data)
            except json.JSONDecodeError:
                return web.Response(status=400, text="Invalid JSON")
                
            # Validation du schéma
            if not self._validate_webhook_schema(webhook_data):
                return web.Response(status=400, text="Invalid webhook schema")
                
            # Création de l'événement webhook
            event = await self._create_webhook_event(
                webhook_data, request.remote, request.headers.get('User-Agent')
            )
            
            # Ajout à la queue de traitement
            try:
                await asyncio.wait_for(
                    self.processing_queue.put(event),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                return web.Response(status=503, text="Processing queue full")
                
            # Métriques de performance
            processing_time = time.time() - start_time
            self.performance_metrics.append({
                'timestamp': datetime.now(timezone.utc),
                'processing_time': processing_time,
                'event_type': webhook_data.get('event_type', 'unknown'),
                'status': 'accepted'
            })
            
            # Audit logging
            if self.config.enable_audit_logging:
                await self._log_webhook_event(event, 'received')
                
            return web.Response(status=200, text="Webhook received")
            
        except Exception as e:
            logger.error(f"Webhook handling failed: {e}", request_path=request.path)
            return web.Response(status=500, text="Internal server error")
            
    async def _handle_generic_webhook(self, request: web.Request) -> web.Response:
        """Traite les webhooks génériques"""
        # Implémentation similaire mais plus flexible pour d'autres sources
        return web.Response(status=200, text="Generic webhook received")
        
    async def _health_check(self, request: web.Request) -> web.Response:
        """Check de santé du processeur"""
        health_data = {
            'status': 'healthy',
            'queue_size': self.processing_queue.qsize(),
            'active_processors': self.active_processors,
            'uptime': time.time() - getattr(self, 'start_time', time.time()),
            'stats': dict(self.event_stats)
        }
        
        return web.json_response(health_data)
        
    async def _get_metrics(self, request: web.Request) -> web.Response:
        """Retourne les métriques du processeur"""
        metrics = {
            'total_events': sum(self.event_stats.values()),
            'event_breakdown': dict(self.event_stats),
            'average_processing_time': self._calculate_avg_processing_time(),
            'queue_utilization': (self.processing_queue.qsize() / self.config.queue_max_size) * 100
        }
        
        return web.json_response(metrics)
        
    async def _security_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """Middleware de sécurité"""
        # Vérification des headers de sécurité
        if 'X-Forwarded-For' in request.headers:
            real_ip = request.headers['X-Forwarded-For'].split(',')[0].strip()
            request['real_ip'] = real_ip
            
        response = await handler(request)
        
        # Ajout des headers de sécurité
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        return response
        
    async def _logging_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """Middleware de logging"""
        start_time = time.time()
        
        response = await handler(request)
        
        processing_time = time.time() - start_time
        
        logger.info(
            "Webhook request processed",
            method=request.method,
            path=request.path,
            status=response.status,
            processing_time=processing_time,
            source_ip=request.remote,
            user_agent=request.headers.get('User-Agent', 'Unknown')
        )
        
        return response
        
    def _validate_signature(self, payload: str, signature: str) -> bool:
        """Valide la signature HMAC du webhook"""
        try:
            expected_signature = self.security_manager.generate_webhook_signature(
                payload, self.webhook_secret
            )
            return self.security_manager.validate_webhook_signature(
                payload, signature, self.webhook_secret
            )
        except Exception as e:
            logger.warning(f"Signature validation failed: {e}")
            return False
            
    def _validate_webhook_schema(self, data: Dict[str, Any]) -> bool:
        """Valide le schéma du webhook"""
        try:
            # Schéma de base pour les webhooks PagerDuty
            schema = {
                "type": "object",
                "required": ["event_type", "created_on", "id", "data"],
                "properties": {
                    "event_type": {"type": "string"},
                    "created_on": {"type": "string"},
                    "id": {"type": "string"},
                    "data": {"type": "object"}
                }
            }
            
            jsonschema.validate(data, schema)
            return True
            
        except jsonschema.ValidationError:
            return False
            
    async def _create_webhook_event(self, webhook_data: Dict[str, Any], source_ip: str, user_agent: str) -> WebhookEvent:
        """Crée un événement webhook enrichi"""
        try:
            payload = WebhookPayload(**webhook_data)
            
            # Détermination de la priorité
            priority = self._determine_priority(payload.event_type)
            
            event = WebhookEvent(
                payload=payload,
                priority=priority,
                source_ip=source_ip,
                user_agent=user_agent,
                signature_valid=True  # Déjà validée
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to create webhook event: {e}")
            raise
            
    def _determine_priority(self, event_type: str) -> ProcessingPriority:
        """Détermine la priorité de traitement"""
        critical_events = ['incident.triggered', 'incident.escalated']
        high_events = ['incident.acknowledged', 'incident.resolved']
        
        if event_type in critical_events:
            return ProcessingPriority.CRITICAL
        elif event_type in high_events:
            return ProcessingPriority.HIGH
        else:
            return ProcessingPriority.MEDIUM
            
    async def _processing_worker(self):
        """Worker de traitement des webhooks"""
        self.active_processors += 1
        
        try:
            while True:
                try:
                    # Récupération d'un événement de la queue
                    event = await self.processing_queue.get()
                    
                    # Traitement de l'événement
                    await self._process_webhook_event(event)
                    
                    # Marquage comme terminé
                    self.processing_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Processing worker error: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            pass
        finally:
            self.active_processors -= 1
            
    async def _process_webhook_event(self, event: WebhookEvent):
        """Traite un événement webhook"""
        try:
            event.status = ProcessingStatus.PROCESSING
            event.processed_at = datetime.now(timezone.utc)
            
            # Audit logging
            if self.config.enable_audit_logging:
                await self._log_webhook_event(event, 'processing_started')
                
            # Application des transformations
            transformed_data = await self._apply_transformations(event)
            
            # Traitement selon le type d'événement
            handler = self.webhook_handlers.get(event.payload.event_type)
            if handler:
                await handler(event, transformed_data)
            else:
                await self._default_event_handler(event, transformed_data)
                
            # Intégration avec systèmes externes
            await self._forward_to_external_systems(event, transformed_data)
            
            # Mise à jour du statut
            event.status = ProcessingStatus.COMPLETED
            
            # Statistiques
            self.event_stats[event.payload.event_type] += 1
            
            # Audit logging
            if self.config.enable_audit_logging:
                await self._log_webhook_event(event, 'completed')
                
        except Exception as e:
            logger.error(f"Webhook processing failed: {e}", webhook_id=event.webhook_id)
            
            event.status = ProcessingStatus.FAILED
            event.error_message = str(e)
            event.retry_count += 1
            
            # Retry si pas encore au maximum
            if event.retry_count <= self.config.retry_max_attempts:
                event.status = ProcessingStatus.RETRYING
                await asyncio.sleep(2 ** event.retry_count)  # Backoff exponentiel
                await self.processing_queue.put(event)
                
            # Audit logging
            if self.config.enable_audit_logging:
                await self._log_webhook_event(event, 'failed')
                
    async def _apply_transformations(self, event: WebhookEvent) -> Dict[str, Any]:
        """Applique les transformations configurées"""
        transformed_data = event.payload.data.copy()
        
        try:
            # Recherche des règles applicables
            applicable_rules = [
                rule for rule in self.transformation_rules
                if rule.event_type.value == event.payload.event_type and rule.enabled
            ]
            
            for rule in applicable_rules:
                # Vérification des conditions
                if self._check_conditions(transformed_data, rule.conditions):
                    # Application des transformations
                    for transformation in rule.transformations:
                        transformed_data = self._apply_transformation(transformed_data, transformation)
                        
        except Exception as e:
            logger.warning(f"Transformation failed: {e}", webhook_id=event.webhook_id)
            
        return transformed_data
        
    def _check_conditions(self, data: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
        """Vérifie si les conditions sont remplies"""
        for key, expected_value in conditions.items():
            if key not in data or data[key] != expected_value:
                return False
        return True
        
    def _apply_transformation(self, data: Dict[str, Any], transformation: Dict[str, Any]) -> Dict[str, Any]:
        """Applique une transformation spécifique"""
        transform_type = transformation.get('type')
        
        if transform_type == 'add_field':
            data[transformation['field']] = transformation['value']
        elif transform_type == 'remove_field':
            data.pop(transformation['field'], None)
        elif transform_type == 'rename_field':
            old_name = transformation['old_name']
            new_name = transformation['new_name']
            if old_name in data:
                data[new_name] = data.pop(old_name)
                
        return data
        
    async def _default_event_handler(self, event: WebhookEvent, data: Dict[str, Any]):
        """Handler par défaut pour les événements"""
        logger.info(
            "Processing webhook event",
            event_type=event.payload.event_type,
            webhook_id=event.webhook_id,
            data_keys=list(data.keys())
        )
        
    async def _forward_to_external_systems(self, event: WebhookEvent, data: Dict[str, Any]):
        """Transmet les événements aux systèmes externes"""
        try:
            # Exemple: envoi vers Slack, Jira, etc.
            for rule in self.transformation_rules:
                if (rule.event_type.value == event.payload.event_type and 
                    rule.target_systems and rule.enabled):
                    
                    for target_system in rule.target_systems:
                        await self._send_to_target_system(target_system, event, data)
                        
        except Exception as e:
            logger.warning(f"External system forwarding failed: {e}")
            
    async def _send_to_target_system(self, target_system: str, event: WebhookEvent, data: Dict[str, Any]):
        """Envoie vers un système cible spécifique"""
        # Implémentation spécifique selon le système cible
        logger.debug(f"Forwarding to {target_system}", webhook_id=event.webhook_id)
        
    async def _load_transformation_rules(self):
        """Charge les règles de transformation depuis Redis"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                rules_data = await redis.get("webhook:transformation_rules")
                if rules_data:
                    rules_list = json.loads(rules_data)
                    self.transformation_rules = [
                        WebhookTransformationRule(**rule) for rule in rules_list
                    ]
                    
        except Exception as e:
            logger.warning(f"Failed to load transformation rules: {e}")
            
    async def _log_webhook_event(self, event: WebhookEvent, action: str):
        """Log l'événement webhook pour audit"""
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                audit_data = {
                    'webhook_id': event.webhook_id,
                    'event_type': event.payload.event_type,
                    'action': action,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source_ip': event.source_ip,
                    'status': event.status.value,
                    'retry_count': event.retry_count
                }
                
                # Stockage avec TTL de 30 jours
                key = f"audit:webhook:{event.webhook_id}:{action}"
                await redis.setex(key, 30 * 86400, json.dumps(audit_data))
                
        except Exception as e:
            logger.warning(f"Audit logging failed: {e}")
            
    def _calculate_avg_processing_time(self) -> float:
        """Calcule le temps de traitement moyen"""
        if not self.performance_metrics:
            return 0.0
            
        processing_times = [m['processing_time'] for m in self.performance_metrics]
        return sum(processing_times) / len(processing_times)
        
    def register_event_handler(self, event_type: str, handler: Callable):
        """Enregistre un handler pour un type d'événement"""
        self.webhook_handlers[event_type] = handler
        
    async def add_transformation_rule(self, rule: WebhookTransformationRule):
        """Ajoute une règle de transformation"""
        self.transformation_rules.append(rule)
        
        # Sauvegarde en Redis
        try:
            async with aioredis.Redis(connection_pool=self.redis_pool) as redis:
                rules_data = [rule.dict() for rule in self.transformation_rules]
                await redis.set("webhook:transformation_rules", json.dumps(rules_data))
                
        except Exception as e:
            logger.error(f"Failed to save transformation rule: {e}")

# ============================================================================
# Interface Publique
# ============================================================================

__all__ = [
    'WebhookProcessor',
    'WebhookConfig',
    'WebhookEvent',
    'WebhookPayload',
    'WebhookTransformationRule',
    'WebhookEventType',
    'ProcessingPriority',
    'ProcessingStatus'
]
