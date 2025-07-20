"""
Gestionnaire d'événements avancé pour le système de tenancy
Auteur: Fahed Mlaiel - Lead Dev & Architecte IA
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Callable, Type, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from abc import ABC, abstractmethod
import weakref
from collections import defaultdict, deque
import structlog

logger = structlog.get_logger(__name__)

class EventType(Enum):
    """Types d'événements système"""
    TENANT_CREATED = "tenant.created"
    TENANT_UPDATED = "tenant.updated"
    TENANT_DELETED = "tenant.deleted"
    TENANT_ACTIVATED = "tenant.activated"
    TENANT_DEACTIVATED = "tenant.deactivated"
    
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    
    API_REQUEST = "api.request"
    API_ERROR = "api.error"
    API_QUOTA_EXCEEDED = "api.quota_exceeded"
    
    SYSTEM_ALERT = "system.alert"
    SYSTEM_ERROR = "system.error"
    SYSTEM_MAINTENANCE = "system.maintenance"
    
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    
    CUSTOM = "custom"

class EventPriority(Enum):
    """Priorités des événements"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class EventStatus(Enum):
    """Statuts des événements"""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Event:
    """Événement système"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.CUSTOM
    source: str = "system"
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    status: EventStatus = EventStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'événement en dictionnaire"""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['priority'] = self.priority.value
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        result['processed_at'] = self.processed_at.isoformat() if self.processed_at else None
        result['tags'] = list(self.tags)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Crée un événement depuis un dictionnaire"""
        event = cls()
        event.event_id = data.get('event_id', str(uuid.uuid4()))
        event.event_type = EventType(data.get('event_type', EventType.CUSTOM.value))
        event.source = data.get('source', 'system')
        event.tenant_id = data.get('tenant_id')
        event.user_id = data.get('user_id')
        event.data = data.get('data', {})
        event.metadata = data.get('metadata', {})
        event.priority = EventPriority(data.get('priority', EventPriority.NORMAL.value))
        event.status = EventStatus(data.get('status', EventStatus.PENDING.value))
        event.created_at = datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.utcnow()
        event.processed_at = datetime.fromisoformat(data['processed_at']) if data.get('processed_at') else None
        event.retry_count = data.get('retry_count', 0)
        event.max_retries = data.get('max_retries', 3)
        event.tags = set(data.get('tags', []))
        return event

class EventHandler(ABC):
    """Handler d'événements abstrait"""
    
    def __init__(self, name: str, event_types: List[EventType] = None):
        self.name = name
        self.event_types = event_types or []
        self.enabled = True
        self.async_execution = True
        self.filter_conditions: List[Callable[[Event], bool]] = []
        
    def add_filter(self, condition: Callable[[Event], bool]) -> None:
        """Ajoute une condition de filtrage"""
        self.filter_conditions.append(condition)
        
    def can_handle(self, event: Event) -> bool:
        """Vérifie si le handler peut traiter l'événement"""
        if not self.enabled:
            return False
            
        if self.event_types and event.event_type not in self.event_types:
            return False
            
        for condition in self.filter_conditions:
            if not condition(event):
                return False
                
        return True
    
    @abstractmethod
    async def handle(self, event: Event) -> bool:
        """Traite l'événement"""
        pass
        
    async def on_error(self, event: Event, error: Exception) -> bool:
        """Gère les erreurs de traitement"""
        logger.error(f"Erreur dans le handler {self.name}", 
                    event_id=event.event_id, error=str(error))
        return False  # Ne pas continuer le traitement

class TenantEventHandler(EventHandler):
    """Handler pour les événements de tenant"""
    
    def __init__(self):
        super().__init__(
            "TenantEventHandler",
            [EventType.TENANT_CREATED, EventType.TENANT_UPDATED, 
             EventType.TENANT_DELETED, EventType.TENANT_ACTIVATED, 
             EventType.TENANT_DEACTIVATED]
        )
        
    async def handle(self, event: Event) -> bool:
        """Traite les événements de tenant"""
        try:
            if event.event_type == EventType.TENANT_CREATED:
                return await self._handle_tenant_created(event)
            elif event.event_type == EventType.TENANT_UPDATED:
                return await self._handle_tenant_updated(event)
            elif event.event_type == EventType.TENANT_DELETED:
                return await self._handle_tenant_deleted(event)
            elif event.event_type == EventType.TENANT_ACTIVATED:
                return await self._handle_tenant_activated(event)
            elif event.event_type == EventType.TENANT_DEACTIVATED:
                return await self._handle_tenant_deactivated(event)
                
            return True
            
        except Exception as e:
            return await self.on_error(event, e)
    
    async def _handle_tenant_created(self, event: Event) -> bool:
        """Gère la création d'un tenant"""
        tenant_data = event.data
        logger.info("Tenant créé", tenant_id=event.tenant_id, tenant_name=tenant_data.get('name'))
        
        # Actions post-création
        await self._setup_default_resources(event.tenant_id, tenant_data)
        await self._send_welcome_notification(event.tenant_id, tenant_data)
        
        return True
        
    async def _handle_tenant_updated(self, event: Event) -> bool:
        """Gère la mise à jour d'un tenant"""
        changes = event.data.get('changes', {})
        logger.info("Tenant mis à jour", tenant_id=event.tenant_id, changes=list(changes.keys()))
        
        # Traitement des changements spécifiques
        if 'quota' in changes:
            await self._update_quota_limits(event.tenant_id, changes['quota'])
        
        if 'features' in changes:
            await self._update_feature_flags(event.tenant_id, changes['features'])
        
        return True
        
    async def _handle_tenant_deleted(self, event: Event) -> bool:
        """Gère la suppression d'un tenant"""
        logger.info("Tenant supprimé", tenant_id=event.tenant_id)
        
        # Nettoyage des ressources
        await self._cleanup_tenant_resources(event.tenant_id)
        await self._archive_tenant_data(event.tenant_id, event.data)
        
        return True
        
    async def _handle_tenant_activated(self, event: Event) -> bool:
        """Gère l'activation d'un tenant"""
        logger.info("Tenant activé", tenant_id=event.tenant_id)
        
        await self._enable_tenant_services(event.tenant_id)
        return True
        
    async def _handle_tenant_deactivated(self, event: Event) -> bool:
        """Gère la désactivation d'un tenant"""
        logger.info("Tenant désactivé", tenant_id=event.tenant_id)
        
        await self._disable_tenant_services(event.tenant_id)
        return True
    
    async def _setup_default_resources(self, tenant_id: str, tenant_data: Dict[str, Any]) -> None:
        """Configure les ressources par défaut"""
        await asyncio.sleep(0.1)  # Simulation
        
    async def _send_welcome_notification(self, tenant_id: str, tenant_data: Dict[str, Any]) -> None:
        """Envoie une notification de bienvenue"""
        await asyncio.sleep(0.1)  # Simulation
        
    async def _update_quota_limits(self, tenant_id: str, quota_changes: Dict[str, Any]) -> None:
        """Met à jour les limites de quota"""
        await asyncio.sleep(0.1)  # Simulation
        
    async def _update_feature_flags(self, tenant_id: str, feature_changes: Dict[str, Any]) -> None:
        """Met à jour les flags de fonctionnalités"""
        await asyncio.sleep(0.1)  # Simulation
        
    async def _cleanup_tenant_resources(self, tenant_id: str) -> None:
        """Nettoie les ressources du tenant"""
        await asyncio.sleep(0.2)  # Simulation
        
    async def _archive_tenant_data(self, tenant_id: str, tenant_data: Dict[str, Any]) -> None:
        """Archive les données du tenant"""
        await asyncio.sleep(0.1)  # Simulation
        
    async def _enable_tenant_services(self, tenant_id: str) -> None:
        """Active les services du tenant"""
        await asyncio.sleep(0.1)  # Simulation
        
    async def _disable_tenant_services(self, tenant_id: str) -> None:
        """Désactive les services du tenant"""
        await asyncio.sleep(0.1)  # Simulation

class APIEventHandler(EventHandler):
    """Handler pour les événements API"""
    
    def __init__(self):
        super().__init__(
            "APIEventHandler",
            [EventType.API_REQUEST, EventType.API_ERROR, EventType.API_QUOTA_EXCEEDED]
        )
        
    async def handle(self, event: Event) -> bool:
        """Traite les événements API"""
        try:
            if event.event_type == EventType.API_REQUEST:
                return await self._handle_api_request(event)
            elif event.event_type == EventType.API_ERROR:
                return await self._handle_api_error(event)
            elif event.event_type == EventType.API_QUOTA_EXCEEDED:
                return await self._handle_quota_exceeded(event)
                
            return True
            
        except Exception as e:
            return await self.on_error(event, e)
    
    async def _handle_api_request(self, event: Event) -> bool:
        """Gère une requête API"""
        request_data = event.data
        
        # Mise à jour des métriques
        await self._update_request_metrics(event.tenant_id, request_data)
        
        # Vérification des patterns suspects
        if await self._detect_suspicious_activity(event.tenant_id, request_data):
            await self._alert_security_team(event.tenant_id, request_data)
        
        return True
        
    async def _handle_api_error(self, event: Event) -> bool:
        """Gère une erreur API"""
        error_data = event.data
        
        # Logging détaillé
        logger.error("Erreur API détectée", 
                    tenant_id=event.tenant_id,
                    endpoint=error_data.get('endpoint'),
                    status_code=error_data.get('status_code'),
                    error_message=error_data.get('error'))
        
        # Alertes si taux d'erreur élevé
        if await self._check_error_rate_threshold(event.tenant_id):
            await self._escalate_error_alert(event.tenant_id, error_data)
        
        return True
        
    async def _handle_quota_exceeded(self, event: Event) -> bool:
        """Gère le dépassement de quota"""
        quota_data = event.data
        
        logger.warning("Quota dépassé", 
                      tenant_id=event.tenant_id,
                      quota_type=quota_data.get('quota_type'),
                      current_usage=quota_data.get('current_usage'),
                      limit=quota_data.get('limit'))
        
        # Notification au tenant
        await self._notify_quota_exceeded(event.tenant_id, quota_data)
        
        # Application de la limitation
        await self._apply_rate_limiting(event.tenant_id, quota_data)
        
        return True
    
    async def _update_request_metrics(self, tenant_id: str, request_data: Dict[str, Any]) -> None:
        """Met à jour les métriques de requête"""
        await asyncio.sleep(0.01)  # Simulation
        
    async def _detect_suspicious_activity(self, tenant_id: str, request_data: Dict[str, Any]) -> bool:
        """Détecte une activité suspecte"""
        await asyncio.sleep(0.01)  # Simulation
        return False
        
    async def _alert_security_team(self, tenant_id: str, request_data: Dict[str, Any]) -> None:
        """Alerte l'équipe de sécurité"""
        await asyncio.sleep(0.01)  # Simulation
        
    async def _check_error_rate_threshold(self, tenant_id: str) -> bool:
        """Vérifie le seuil de taux d'erreur"""
        await asyncio.sleep(0.01)  # Simulation
        return False
        
    async def _escalate_error_alert(self, tenant_id: str, error_data: Dict[str, Any]) -> None:
        """Escalade une alerte d'erreur"""
        await asyncio.sleep(0.01)  # Simulation
        
    async def _notify_quota_exceeded(self, tenant_id: str, quota_data: Dict[str, Any]) -> None:
        """Notifie le dépassement de quota"""
        await asyncio.sleep(0.01)  # Simulation
        
    async def _apply_rate_limiting(self, tenant_id: str, quota_data: Dict[str, Any]) -> None:
        """Applique la limitation de débit"""
        await asyncio.sleep(0.01)  # Simulation

class NotificationHandler(EventHandler):
    """Handler pour les notifications"""
    
    def __init__(self):
        super().__init__("NotificationHandler")
        self.notification_channels = {
            'email': self._send_email,
            'slack': self._send_slack,
            'webhook': self._send_webhook,
            'sms': self._send_sms
        }
        
    async def handle(self, event: Event) -> bool:
        """Traite les événements de notification"""
        try:
            notification_config = event.data.get('notification', {})
            channels = notification_config.get('channels', ['email'])
            
            for channel in channels:
                if channel in self.notification_channels:
                    await self.notification_channels[channel](event, notification_config)
            
            return True
            
        except Exception as e:
            return await self.on_error(event, e)
    
    async def _send_email(self, event: Event, config: Dict[str, Any]) -> None:
        """Envoie une notification par email"""
        await asyncio.sleep(0.1)  # Simulation
        logger.info("Email envoyé", event_id=event.event_id)
        
    async def _send_slack(self, event: Event, config: Dict[str, Any]) -> None:
        """Envoie une notification Slack"""
        await asyncio.sleep(0.05)  # Simulation
        logger.info("Message Slack envoyé", event_id=event.event_id)
        
    async def _send_webhook(self, event: Event, config: Dict[str, Any]) -> None:
        """Envoie une notification via webhook"""
        await asyncio.sleep(0.1)  # Simulation
        logger.info("Webhook appelé", event_id=event.event_id)
        
    async def _send_sms(self, event: Event, config: Dict[str, Any]) -> None:
        """Envoie une notification SMS"""
        await asyncio.sleep(0.2)  # Simulation
        logger.info("SMS envoyé", event_id=event.event_id)

class EventBus:
    """Bus d'événements central"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.global_handlers: List[EventHandler] = []
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.dead_letter_queue: deque = deque(maxlen=1000)
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.event_history: deque = deque(maxlen=10000)
        self.metrics = {
            'events_processed': 0,
            'events_failed': 0,
            'handlers_executed': 0,
            'handlers_failed': 0
        }
        
    def register_handler(self, handler: EventHandler, 
                        event_types: Optional[List[EventType]] = None) -> None:
        """Enregistre un handler d'événements"""
        if event_types is None:
            event_types = handler.event_types
            
        if not event_types:
            self.global_handlers.append(handler)
        else:
            for event_type in event_types:
                self.handlers[event_type].append(handler)
                
        logger.info(f"Handler {handler.name} enregistré", 
                   event_types=[et.value for et in event_types] if event_types else "global")
    
    def unregister_handler(self, handler: EventHandler) -> None:
        """Désenregistre un handler"""
        # Suppression des handlers spécifiques
        for event_type, handlers in self.handlers.items():
            if handler in handlers:
                handlers.remove(handler)
        
        # Suppression des handlers globaux
        if handler in self.global_handlers:
            self.global_handlers.remove(handler)
            
        logger.info(f"Handler {handler.name} désenregistré")
    
    async def publish(self, event: Event) -> None:
        """Publie un événement"""
        try:
            await self.event_queue.put(event)
            logger.debug("Événement publié", event_id=event.event_id, event_type=event.event_type.value)
        except asyncio.QueueFull:
            logger.error("Queue d'événements pleine", event_id=event.event_id)
            self.dead_letter_queue.append(event)
    
    async def start(self, num_workers: int = 3) -> None:
        """Démarre le bus d'événements"""
        if self.running:
            return
            
        self.running = True
        
        # Démarrage des workers
        for i in range(num_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
            
        logger.info(f"Bus d'événements démarré avec {num_workers} workers")
    
    async def stop(self) -> None:
        """Arrête le bus d'événements"""
        self.running = False
        
        # Arrêt des workers
        for task in self.worker_tasks:
            task.cancel()
            
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        logger.info("Bus d'événements arrêté")
    
    async def _worker(self, worker_name: str) -> None:
        """Worker de traitement des événements"""
        logger.info(f"Worker {worker_name} démarré")
        
        while self.running:
            try:
                # Récupération d'un événement
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Traitement de l'événement
                await self._process_event(event)
                
                # Marquer la tâche comme terminée
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Erreur dans le worker {worker_name}", error=str(e))
        
        logger.info(f"Worker {worker_name} arrêté")
    
    async def _process_event(self, event: Event) -> None:
        """Traite un événement"""
        event.status = EventStatus.PROCESSING
        start_time = datetime.utcnow()
        
        try:
            # Ajout à l'historique
            self.event_history.append(event)
            
            # Récupération des handlers
            applicable_handlers = self._get_applicable_handlers(event)
            
            if not applicable_handlers:
                logger.debug("Aucun handler pour l'événement", 
                           event_id=event.event_id, event_type=event.event_type.value)
                event.status = EventStatus.PROCESSED
                return
            
            # Exécution des handlers
            handler_results = []
            
            for handler in applicable_handlers:
                try:
                    if handler.can_handle(event):
                        result = await handler.handle(event)
                        handler_results.append((handler, result))
                        self.metrics['handlers_executed'] += 1
                        
                        if not result:
                            logger.warning(f"Handler {handler.name} a échoué", 
                                         event_id=event.event_id)
                            self.metrics['handlers_failed'] += 1
                            
                except Exception as e:
                    logger.error(f"Erreur dans le handler {handler.name}", 
                               event_id=event.event_id, error=str(e))
                    handler_results.append((handler, False))
                    self.metrics['handlers_failed'] += 1
            
            # Détermination du statut final
            failed_handlers = [h for h, result in handler_results if not result]
            
            if failed_handlers and len(failed_handlers) == len(handler_results):
                event.status = EventStatus.FAILED
                
                # Retry si possible
                if event.retry_count < event.max_retries:
                    event.retry_count += 1
                    event.status = EventStatus.PENDING
                    await self.publish(event)  # Republication
                    return
                else:
                    self.dead_letter_queue.append(event)
                    self.metrics['events_failed'] += 1
            else:
                event.status = EventStatus.PROCESSED
                self.metrics['events_processed'] += 1
            
            event.processed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error("Erreur lors du traitement de l'événement", 
                        event_id=event.event_id, error=str(e))
            event.status = EventStatus.FAILED
            event.processed_at = datetime.utcnow()
            self.dead_letter_queue.append(event)
            self.metrics['events_failed'] += 1
    
    def _get_applicable_handlers(self, event: Event) -> List[EventHandler]:
        """Récupère les handlers applicables pour un événement"""
        handlers = []
        
        # Handlers spécifiques au type d'événement
        if event.event_type in self.handlers:
            handlers.extend(self.handlers[event.event_type])
        
        # Handlers globaux
        handlers.extend(self.global_handlers)
        
        return handlers
    
    def get_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques du bus"""
        return {
            **self.metrics,
            'queue_size': self.event_queue.qsize(),
            'dead_letter_size': len(self.dead_letter_queue),
            'history_size': len(self.event_history),
            'handlers_count': sum(len(handlers) for handlers in self.handlers.values()) + len(self.global_handlers),
            'running': self.running,
            'workers_count': len(self.worker_tasks)
        }

# Instance globale du bus d'événements
event_bus = EventBus()

# Enregistrement des handlers par défaut
event_bus.register_handler(TenantEventHandler())
event_bus.register_handler(APIEventHandler())
event_bus.register_handler(NotificationHandler())

# Fonctions utilitaires pour la publication d'événements
async def publish_tenant_created(tenant_id: str, tenant_data: Dict[str, Any]) -> None:
    """Publie un événement de création de tenant"""
    event = Event(
        event_type=EventType.TENANT_CREATED,
        source="tenant_service",
        tenant_id=tenant_id,
        data=tenant_data,
        priority=EventPriority.HIGH
    )
    await event_bus.publish(event)

async def publish_api_request(tenant_id: str, request_data: Dict[str, Any]) -> None:
    """Publie un événement de requête API"""
    event = Event(
        event_type=EventType.API_REQUEST,
        source="api_gateway",
        tenant_id=tenant_id,
        data=request_data,
        priority=EventPriority.LOW
    )
    await event_bus.publish(event)

async def publish_system_alert(alert_data: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL) -> None:
    """Publie une alerte système"""
    event = Event(
        event_type=EventType.SYSTEM_ALERT,
        source="monitoring",
        data=alert_data,
        priority=priority
    )
    await event_bus.publish(event)
