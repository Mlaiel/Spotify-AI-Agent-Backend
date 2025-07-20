"""
Services avancés pour la gestion des notifications
=================================================

Services ultra-sophistiqués avec patterns enterprise, circuit breakers,
monitoring avancé, et intelligence artificielle intégrée.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Set, Callable, AsyncIterator
from uuid import UUID, uuid4
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from functools import wraps
import statistics

import aioredis
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from prometheus_client import Counter, Histogram, Gauge
import jinja2
from jinja2 import Template, Environment, FileSystemLoader
from circuitbreaker import circuit
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import *
from .schemas import *
from .channels import *
from .templates import *
from .analytics import *


# Métriques Prometheus
NOTIFICATION_COUNTER = Counter(
    'notifications_total',
    'Total notifications sent',
    ['tenant_id', 'channel_type', 'status', 'priority']
)

NOTIFICATION_DURATION = Histogram(
    'notification_delivery_duration_seconds',
    'Time to deliver notification',
    ['tenant_id', 'channel_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

NOTIFICATION_QUEUE_SIZE = Gauge(
    'notification_queue_size',
    'Number of notifications in queue',
    ['tenant_id', 'priority']
)

NOTIFICATION_ERROR_RATE = Gauge(
    'notification_error_rate',
    'Error rate for notifications',
    ['tenant_id', 'channel_type', 'error_type']
)


class NotificationServiceError(Exception):
    """Exception de base pour les services de notification"""
    pass


class NotificationValidationError(NotificationServiceError):
    """Erreur de validation des données"""
    pass


class NotificationDeliveryError(NotificationServiceError):
    """Erreur de livraison"""
    pass


class NotificationRateLimitError(NotificationServiceError):
    """Erreur de limite de taux"""
    pass


class BaseNotificationService(ABC):
    """Service de base pour tous les types de notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.redis_client: Optional[aioredis.Redis] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self._circuit_breaker_failures = 0
        
    async def initialize(self):
        """Initialisation asynchrone du service"""
        self.redis_client = aioredis.from_url(
            self.config.get('redis_url', 'redis://localhost:6379'),
            decode_responses=True
        )
        
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        
        await self._setup_rate_limiters()
        
    async def cleanup(self):
        """Nettoyage des ressources"""
        if self.redis_client:
            await self.redis_client.close()
        if self.http_client:
            await self.http_client.aclose()
    
    @abstractmethod
    async def send_notification(
        self, 
        notification: NotificationCreateSchema,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Envoyer une notification"""
        pass
    
    async def _setup_rate_limiters(self):
        """Configuration des limiteurs de taux"""
        pass
    
    async def _check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Vérifier les limites de taux avec sliding window"""
        if not self.redis_client:
            return True
            
        now = datetime.now(timezone.utc).timestamp()
        pipeline = self.redis_client.pipeline()
        
        # Nettoyer les entrées expirées
        pipeline.zremrangebyscore(key, 0, now - window)
        
        # Ajouter la requête actuelle
        pipeline.zadd(key, {str(uuid4()): now})
        
        # Compter les requêtes dans la fenêtre
        pipeline.zcard(key)
        
        # Expirer la clé
        pipeline.expire(key, window)
        
        results = await pipeline.execute()
        current_count = results[2]
        
        return current_count <= limit


class NotificationManagerService:
    """Service principal de gestion des notifications"""
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        config: Dict[str, Any]
    ):
        self.db = db_session
        self.redis = redis_client
        self.config = config
        self.logger = logging.getLogger("NotificationManager")
        
        # Services de canaux
        self.channel_services: Dict[str, BaseNotificationService] = {}
        
        # Template engine
        self.template_env = Environment(
            loader=FileSystemLoader(config.get('template_dir', 'templates')),
            autoescape=True,
            enable_async=True
        )
        
        # Analytics service
        self.analytics = NotificationAnalyticsService(db_session, redis_client)
        
        # Queue manager
        self.queue_manager = NotificationQueueManager(db_session, redis_client, config)
        
    async def initialize(self):
        """Initialisation des services"""
        await self._initialize_channel_services()
        await self.analytics.initialize()
        await self.queue_manager.initialize()
        
    async def _initialize_channel_services(self):
        """Initialiser tous les services de canaux"""
        channel_configs = self.config.get('channels', {})
        
        for channel_type, channel_config in channel_configs.items():
            if channel_type == 'slack':
                self.channel_services[channel_type] = SlackNotificationService(channel_config)
            elif channel_type == 'email':
                self.channel_services[channel_type] = EmailNotificationService(channel_config)
            elif channel_type == 'sms':
                self.channel_services[channel_type] = SMSNotificationService(channel_config)
            elif channel_type == 'push':
                self.channel_services[channel_type] = PushNotificationService(channel_config)
            elif channel_type == 'webhook':
                self.channel_services[channel_type] = WebhookNotificationService(channel_config)
            
            if channel_type in self.channel_services:
                await self.channel_services[channel_type].initialize()
    
    async def create_notification(
        self,
        notification_data: NotificationCreateSchema,
        tenant_id: str,
        user_id: Optional[str] = None
    ) -> List[NotificationResponseSchema]:
        """Créer et traiter une nouvelle notification"""
        
        # Validation et enrichissement
        await self._validate_notification(notification_data, tenant_id)
        
        # Appliquer les préférences utilisateur
        processed_notifications = await self._apply_user_preferences(
            notification_data, tenant_id
        )
        
        # Appliquer les règles de routage
        routed_notifications = await self._apply_routing_rules(
            processed_notifications, tenant_id
        )
        
        # Créer les enregistrements en base
        created_notifications = []
        for notif_data in routed_notifications:
            notification = await self._create_notification_record(
                notif_data, tenant_id, user_id
            )
            created_notifications.append(notification)
        
        # Ajouter à la queue de traitement
        for notification in created_notifications:
            await self.queue_manager.enqueue_notification(notification)
        
        # Métriques
        for notification in created_notifications:
            NOTIFICATION_COUNTER.labels(
                tenant_id=tenant_id,
                channel_type=notification.channel_type,
                status='created',
                priority=notification.priority
            ).inc()
        
        return [NotificationResponseSchema.from_orm(n) for n in created_notifications]
    
    async def _validate_notification(
        self,
        notification: NotificationCreateSchema,
        tenant_id: str
    ):
        """Validation avancée des notifications"""
        
        # Vérifier les templates si spécifié
        if notification.template_id:
            template = await self.db.get(NotificationTemplate, notification.template_id)
            if not template or template.tenant_id != tenant_id:
                raise NotificationValidationError("Template non trouvé")
            
            if not template.is_active:
                raise NotificationValidationError("Template inactif")
            
            # Valider les variables du template
            await self._validate_template_variables(template, notification.template_data)
        
        # Vérifier les limites de taux par tenant
        tenant_limit_key = f"tenant_rate_limit:{tenant_id}"
        tenant_limit = self.config.get('tenant_rate_limit', 1000)
        
        if not await self._check_rate_limit(tenant_limit_key, tenant_limit, 3600):
            raise NotificationRateLimitError("Limite de taux tenant dépassée")
        
        # Validation des destinataires
        for recipient in notification.recipients:
            await self._validate_recipient(recipient, tenant_id)
        
        # Validation des canaux
        for channel in notification.channels:
            await self._validate_channel_config(channel, tenant_id)
    
    async def _validate_template_variables(
        self,
        template: NotificationTemplate,
        template_data: Dict[str, Any]
    ):
        """Valider les variables du template"""
        if not template.body_template:
            return
        
        # Extraire les variables requises du template
        template_obj = Template(template.body_template)
        required_vars = set(template_obj.environment.parse(template.body_template).find_all_names())
        
        # Vérifier que toutes les variables sont fournies
        missing_vars = required_vars - set(template_data.keys())
        if missing_vars:
            raise NotificationValidationError(
                f"Variables manquantes dans template_data: {missing_vars}"
            )
    
    async def _apply_user_preferences(
        self,
        notification: NotificationCreateSchema,
        tenant_id: str
    ) -> List[NotificationCreateSchema]:
        """Appliquer les préférences utilisateur"""
        
        processed_notifications = []
        
        for recipient in notification.recipients:
            # Récupérer les préférences utilisateur
            preferences = await self._get_user_preferences(recipient.id, tenant_id)
            
            if not preferences:
                # Pas de préférences, utiliser la config par défaut
                processed_notifications.append(notification)
                continue
            
            # Filtrer par priorité minimum
            if notification.priority.value < preferences.min_priority.value:
                self.logger.info(
                    f"Notification ignorée pour {recipient.id} - priorité trop faible"
                )
                continue
            
            # Vérifier les heures de silence
            if await self._is_in_quiet_hours(preferences):
                # Reporter la notification après les heures de silence
                notification.scheduled_at = await self._calculate_next_send_time(preferences)
            
            # Filtrer les canaux selon les préférences
            allowed_channels = []
            for channel in notification.channels:
                if (channel.type in preferences.enabled_channels and 
                    channel.type not in preferences.disabled_channels):
                    allowed_channels.append(channel)
            
            if not allowed_channels:
                self.logger.warning(
                    f"Aucun canal autorisé pour {recipient.id}"
                )
                continue
            
            # Créer une notification personnalisée
            personalized_notification = notification.copy()
            personalized_notification.recipients = [recipient]
            personalized_notification.channels = allowed_channels
            
            processed_notifications.append(personalized_notification)
        
        return processed_notifications
    
    async def _apply_routing_rules(
        self,
        notifications: List[NotificationCreateSchema],
        tenant_id: str
    ) -> List[NotificationCreateSchema]:
        """Appliquer les règles de routage"""
        
        # Récupérer les règles actives pour le tenant
        rules_query = select(NotificationRule).where(
            and_(
                NotificationRule.tenant_id == tenant_id,
                NotificationRule.is_active == True
            )
        ).order_by(NotificationRule.priority_threshold.desc())
        
        rules = (await self.db.execute(rules_query)).scalars().all()
        
        routed_notifications = []
        
        for notification in notifications:
            matching_rules = []
            
            # Trouver les règles qui correspondent
            for rule in rules:
                if await self._rule_matches_notification(rule, notification):
                    matching_rules.append(rule)
            
            if not matching_rules:
                # Aucune règle, utiliser la config par défaut
                routed_notifications.append(notification)
                continue
            
            # Appliquer la règle avec la priorité la plus élevée
            best_rule = matching_rules[0]
            
            # Modifier la notification selon la règle
            modified_notification = await self._apply_rule_to_notification(
                notification, best_rule
            )
            
            routed_notifications.append(modified_notification)
            
            # Mettre à jour les stats de la règle
            best_rule.execution_count += 1
            best_rule.last_executed = datetime.now(timezone.utc)
        
        await self.db.commit()
        
        return routed_notifications
    
    async def process_notification_queue(self):
        """Traiter la queue des notifications"""
        while True:
            try:
                # Récupérer les notifications prioritaires
                notifications = await self.queue_manager.get_pending_notifications(
                    limit=self.config.get('batch_size', 100)
                )
                
                if not notifications:
                    await asyncio.sleep(1)
                    continue
                
                # Traiter en parallèle
                tasks = []
                for notification in notifications:
                    task = asyncio.create_task(
                        self._process_single_notification(notification)
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                self.logger.error(f"Erreur dans le traitement de la queue: {e}")
                await asyncio.sleep(5)
    
    async def _process_single_notification(
        self,
        notification: Notification
    ):
        """Traiter une notification individuelle"""
        
        try:
            # Marquer comme en cours de traitement
            notification.status = NotificationStatus.PROCESSING
            await self.db.commit()
            
            # Récupérer le service de canal approprié
            channel_service = self.channel_services.get(notification.channel_type)
            if not channel_service:
                raise NotificationDeliveryError(
                    f"Service non disponible pour le canal: {notification.channel_type}"
                )
            
            # Préparer les données de notification
            notification_data = await self._prepare_notification_data(notification)
            
            # Envoyer via le service de canal
            start_time = datetime.now(timezone.utc)
            
            result = await channel_service.send_notification(
                notification_data,
                context={'notification_id': notification.id}
            )
            
            end_time = datetime.now(timezone.utc)
            delivery_time = (end_time - start_time).total_seconds()
            
            # Mettre à jour le statut
            notification.status = NotificationStatus.SENT
            notification.sent_at = end_time
            notification.delivery_time_ms = int(delivery_time * 1000)
            notification.external_id = result.get('external_id')
            
            # Créer un enregistrement de tentative réussie
            attempt = NotificationDeliveryAttempt(
                notification_id=notification.id,
                attempt_number=notification.retry_count + 1,
                status='success',
                started_at=start_time,
                completed_at=end_time,
                duration_ms=int(delivery_time * 1000),
                channel_response=result
            )
            
            self.db.add(attempt)
            await self.db.commit()
            
            # Métriques
            NOTIFICATION_COUNTER.labels(
                tenant_id=notification.tenant_id,
                channel_type=notification.channel_type,
                status='sent',
                priority=notification.priority
            ).inc()
            
            NOTIFICATION_DURATION.labels(
                tenant_id=notification.tenant_id,
                channel_type=notification.channel_type
            ).observe(delivery_time)
            
        except Exception as e:
            await self._handle_notification_failure(notification, e)
    
    async def _handle_notification_failure(
        self,
        notification: Notification,
        error: Exception
    ):
        """Gérer les échecs de notification"""
        
        notification.retry_count += 1
        
        # Créer un enregistrement de tentative échouée
        attempt = NotificationDeliveryAttempt(
            notification_id=notification.id,
            attempt_number=notification.retry_count,
            status='failed',
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            error_message=str(error),
            error_details={'exception_type': type(error).__name__}
        )
        
        self.db.add(attempt)
        
        # Vérifier si on peut réessayer
        if notification.retry_count < notification.max_retries:
            # Calculer le prochain essai avec backoff exponentiel
            backoff_seconds = (2 ** notification.retry_count) * notification.retry_backoff
            notification.next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)
            notification.status = NotificationStatus.RETRYING
            
            # Remettre en queue
            await self.queue_manager.enqueue_notification(notification, delay=backoff_seconds)
            
        else:
            # Échec définitif
            notification.status = NotificationStatus.FAILED
            
            # Déclencher les règles d'escalade si configurées
            await self._trigger_escalation(notification)
        
        await self.db.commit()
        
        # Métriques d'erreur
        NOTIFICATION_ERROR_RATE.labels(
            tenant_id=notification.tenant_id,
            channel_type=notification.channel_type,
            error_type=type(error).__name__
        ).inc()
    
    async def get_notification_status(
        self,
        notification_id: UUID,
        tenant_id: str
    ) -> Optional[NotificationResponseSchema]:
        """Récupérer le statut d'une notification"""
        
        query = select(Notification).where(
            and_(
                Notification.id == notification_id,
                Notification.tenant_id == tenant_id
            )
        )
        
        result = await self.db.execute(query)
        notification = result.scalar_one_or_none()
        
        if not notification:
            return None
        
        return NotificationResponseSchema.from_orm(notification)
    
    async def get_notifications(
        self,
        tenant_id: str,
        recipient_id: Optional[str] = None,
        status: Optional[NotificationStatusEnum] = None,
        channel_type: Optional[ChannelTypeEnum] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[NotificationResponseSchema]:
        """Récupérer les notifications avec filtres"""
        
        query = select(Notification).where(Notification.tenant_id == tenant_id)
        
        if recipient_id:
            query = query.where(Notification.recipient_id == recipient_id)
        
        if status:
            query = query.where(Notification.status == status.value)
        
        if channel_type:
            query = query.where(Notification.channel_type == channel_type.value)
        
        query = query.order_by(Notification.created_at.desc()).limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        notifications = result.scalars().all()
        
        return [NotificationResponseSchema.from_orm(n) for n in notifications]
    
    async def update_notification(
        self,
        notification_id: UUID,
        update_data: NotificationUpdateSchema,
        tenant_id: str
    ) -> Optional[NotificationResponseSchema]:
        """Mettre à jour une notification"""
        
        query = select(Notification).where(
            and_(
                Notification.id == notification_id,
                Notification.tenant_id == tenant_id
            )
        )
        
        result = await self.db.execute(query)
        notification = result.scalar_one_or_none()
        
        if not notification:
            return None
        
        # Appliquer les mises à jour
        update_dict = update_data.dict(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(notification, field, value)
        
        notification.updated_at = datetime.now(timezone.utc)
        
        await self.db.commit()
        
        return NotificationResponseSchema.from_orm(notification)
    
    async def cancel_notification(
        self,
        notification_id: UUID,
        tenant_id: str
    ) -> bool:
        """Annuler une notification"""
        
        query = select(Notification).where(
            and_(
                Notification.id == notification_id,
                Notification.tenant_id == tenant_id,
                Notification.status.in_([
                    NotificationStatus.PENDING,
                    NotificationStatus.RETRYING
                ])
            )
        )
        
        result = await self.db.execute(query)
        notification = result.scalar_one_or_none()
        
        if not notification:
            return False
        
        notification.status = NotificationStatus.CANCELLED
        notification.updated_at = datetime.now(timezone.utc)
        
        # Retirer de la queue
        await self.queue_manager.remove_notification(notification_id)
        
        await self.db.commit()
        
        return True


class NotificationQueueManager:
    """Gestionnaire de queue pour les notifications"""
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        config: Dict[str, Any]
    ):
        self.db = db_session
        self.redis = redis_client
        self.config = config
        self.logger = logging.getLogger("NotificationQueueManager")
    
    async def initialize(self):
        """Initialisation du gestionnaire de queue"""
        await self._setup_redis_structures()
    
    async def _setup_redis_structures(self):
        """Configurer les structures Redis pour la queue"""
        # Créer les indexes pour les queues prioritaires
        pass
    
    async def enqueue_notification(
        self,
        notification: Notification,
        delay: float = 0
    ):
        """Ajouter une notification à la queue"""
        
        # Calculer le score de priorité
        priority_score = self._calculate_priority_score(notification)
        
        # Timestamp d'exécution
        execute_at = datetime.now(timezone.utc).timestamp() + delay
        
        # Données pour la queue
        queue_data = {
            'notification_id': str(notification.id),
            'tenant_id': notification.tenant_id,
            'priority': notification.priority,
            'channel_type': notification.channel_type,
            'execute_at': execute_at,
            'enqueued_at': datetime.now(timezone.utc).timestamp()
        }
        
        # Ajouter à la queue Redis prioritaire
        queue_key = f"notification_queue:{notification.tenant_id}"
        await self.redis.zadd(queue_key, {json.dumps(queue_data): priority_score})
        
        # Mettre à jour les métriques de queue
        NOTIFICATION_QUEUE_SIZE.labels(
            tenant_id=notification.tenant_id,
            priority=notification.priority
        ).inc()
    
    def _calculate_priority_score(self, notification: Notification) -> float:
        """Calculer le score de priorité pour le tri"""
        
        # Score de base selon la priorité
        priority_scores = {
            NotificationPriority.EMERGENCY: 1000,
            NotificationPriority.CRITICAL: 800,
            NotificationPriority.HIGH: 600,
            NotificationPriority.NORMAL: 400,
            NotificationPriority.LOW: 200
        }
        
        base_score = priority_scores.get(notification.priority, 400)
        
        # Ajuster selon l'âge de la notification
        age_hours = (datetime.now(timezone.utc) - notification.created_at).total_seconds() / 3600
        age_penalty = min(age_hours * 10, 100)  # Penalité max de 100 points
        
        # Ajuster selon le nombre de tentatives
        retry_penalty = notification.retry_count * 50
        
        # Score final (plus élevé = plus prioritaire)
        final_score = base_score - age_penalty - retry_penalty
        
        return max(final_score, 1)  # Score minimum de 1
    
    async def get_pending_notifications(
        self,
        limit: int = 100,
        tenant_id: Optional[str] = None
    ) -> List[Notification]:
        """Récupérer les notifications en attente"""
        
        current_timestamp = datetime.now(timezone.utc).timestamp()
        
        if tenant_id:
            queue_keys = [f"notification_queue:{tenant_id}"]
        else:
            # Récupérer toutes les queues
            pattern = "notification_queue:*"
            queue_keys = await self.redis.keys(pattern)
        
        all_notifications = []
        
        for queue_key in queue_keys:
            # Récupérer les éléments prêts à être traités
            items = await self.redis.zrangebyscore(
                queue_key,
                min=0,
                max=float('inf'),
                start=0,
                num=limit,
                withscores=True
            )
            
            for item_data, score in items:
                try:
                    queue_item = json.loads(item_data)
                    
                    # Vérifier si c'est le moment d'exécuter
                    if queue_item['execute_at'] <= current_timestamp:
                        # Récupérer la notification complète
                        notification_id = UUID(queue_item['notification_id'])
                        notification = await self.db.get(Notification, notification_id)
                        
                        if notification and notification.status in [
                            NotificationStatus.PENDING,
                            NotificationStatus.RETRYING
                        ]:
                            all_notifications.append(notification)
                            
                            # Retirer de la queue Redis
                            await self.redis.zrem(queue_key, item_data)
                            
                            # Mettre à jour les métriques
                            NOTIFICATION_QUEUE_SIZE.labels(
                                tenant_id=notification.tenant_id,
                                priority=notification.priority
                            ).dec()
                
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    self.logger.error(f"Erreur lors du parsing de l'item de queue: {e}")
                    # Retirer l'item corrompu
                    await self.redis.zrem(queue_key, item_data)
        
        # Trier par score de priorité
        all_notifications.sort(
            key=lambda n: self._calculate_priority_score(n),
            reverse=True
        )
        
        return all_notifications[:limit]
    
    async def remove_notification(self, notification_id: UUID):
        """Retirer une notification de toutes les queues"""
        
        # Pattern pour toutes les queues
        pattern = "notification_queue:*"
        queue_keys = await self.redis.keys(pattern)
        
        for queue_key in queue_keys:
            # Parcourir tous les éléments de la queue
            items = await self.redis.zrange(queue_key, 0, -1)
            
            for item_data in items:
                try:
                    queue_item = json.loads(item_data)
                    if queue_item['notification_id'] == str(notification_id):
                        await self.redis.zrem(queue_key, item_data)
                        break
                
                except json.JSONDecodeError:
                    continue
    
    async def get_queue_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Obtenir les statistiques de la queue"""
        
        queue_key = f"notification_queue:{tenant_id}"
        
        total_pending = await self.redis.zcard(queue_key)
        
        # Compter par priorité
        priority_counts = {}
        items = await self.redis.zrange(queue_key, 0, -1)
        
        for item_data in items:
            try:
                queue_item = json.loads(item_data)
                priority = queue_item.get('priority', 'normal')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            except json.JSONDecodeError:
                continue
        
        return {
            'total_pending': total_pending,
            'priority_breakdown': priority_counts,
            'queue_key': queue_key
        }
