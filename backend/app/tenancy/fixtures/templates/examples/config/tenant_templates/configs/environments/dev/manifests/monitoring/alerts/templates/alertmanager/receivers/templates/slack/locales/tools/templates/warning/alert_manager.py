"""
Alert Manager - Gestionnaire Central d'Alertes pour Spotify AI Agent
Système d'alerting multi-tenant avec support temps réel et escalade intelligente
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import aioredis
from prometheus_client import Counter, Histogram, Gauge
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import BackgroundTasks

from .slack_notifier import SlackNotifier
from .warning_processor import WarningProcessor
from .locale_manager import LocaleManager
from .template_engine import TemplateEngine
from .schemas import AlertSchema, AlertLevel, AlertStatus
from .utils import SecurityUtils, PerformanceMonitor


class AlertLevel(Enum):
    """Niveaux d'alerte supportés"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class AlertStatus(Enum):
    """Statuts d'alerte"""
    PENDING = "PENDING"
    SENT = "SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    FAILED = "FAILED"


@dataclass
class AlertContext:
    """Contexte d'une alerte"""
    tenant_id: str
    service_name: str
    environment: str
    correlation_id: str
    metadata: Dict[str, Any]
    timestamp: datetime
    source_ip: Optional[str] = None
    user_id: Optional[str] = None


class AlertManager:
    """
    Gestionnaire central d'alertes avec fonctionnalités avancées :
    - Multi-tenant avec isolation complète
    - Escalade intelligente basée sur les règles
    - Cache Redis pour performance optimale
    - Métriques Prometheus intégrées
    - Support de templates dynamiques
    - Déduplication automatique
    - Rate limiting intelligent
    """
    
    def __init__(
        self,
        tenant_id: str,
        redis_client: aioredis.Redis,
        db_session: AsyncSession,
        slack_notifier: SlackNotifier,
        locale_manager: LocaleManager,
        template_engine: TemplateEngine,
        config: Dict[str, Any]
    ):
        self.tenant_id = tenant_id
        self.redis_client = redis_client
        self.db_session = db_session
        self.slack_notifier = slack_notifier
        self.locale_manager = locale_manager
        self.template_engine = template_engine
        self.config = config
        
        # Logging avec correlation
        self.logger = logging.getLogger(f"alert_manager.{tenant_id}")
        
        # Métriques Prometheus
        self.alert_counter = Counter(
            'alerts_total',
            'Total number of alerts sent',
            ['tenant_id', 'level', 'service', 'status']
        )
        
        self.alert_duration = Histogram(
            'alert_processing_duration_seconds',
            'Time spent processing alerts',
            ['tenant_id', 'level']
        )
        
        self.active_alerts = Gauge(
            'active_alerts_count',
            'Number of active alerts',
            ['tenant_id', 'level']
        )
        
        # Cache pour déduplication
        self.dedup_cache = {}
        
        # Processeur de warnings
        self.warning_processor = WarningProcessor(
            redis_client=redis_client,
            config=config.get('warning_processor', {})
        )
        
        # Rate limiter
        self.rate_limiter = self._init_rate_limiter()
        
        # Escalation rules
        self.escalation_rules = self._load_escalation_rules()
        
    async def send_warning(
        self,
        level: Union[str, AlertLevel],
        message: str,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        locale: str = "en",
        priority: int = 5,
        escalate_after: Optional[int] = None
    ) -> str:
        """
        Envoie une alerte avec gestion complète du cycle de vie
        
        Args:
            level: Niveau d'alerte (CRITICAL, HIGH, WARNING, INFO, DEBUG)
            message: Message principal de l'alerte
            context: Contexte additionnel
            tags: Tags pour filtrage et routing
            locale: Locale pour la localisation
            priority: Priorité (1=urgent, 10=faible)
            escalate_after: Escalade automatique après N minutes
            
        Returns:
            str: ID unique de l'alerte
        """
        start_time = time.time()
        alert_id = str(uuid.uuid4())
        
        try:
            # Validation des paramètres
            if isinstance(level, str):
                level = AlertLevel(level.upper())
            
            # Création du contexte
            alert_context = AlertContext(
                tenant_id=self.tenant_id,
                service_name=context.get('service', 'unknown') if context else 'unknown',
                environment=self.config.get('environment', 'dev'),
                correlation_id=alert_id,
                metadata=context or {},
                timestamp=datetime.utcnow()
            )
            
            # Vérification du rate limiting
            if not await self._check_rate_limit(level, alert_context.service_name):
                self.logger.warning(f"Rate limit exceeded for {level.value} alerts")
                return alert_id
            
            # Déduplication
            dedup_key = self._generate_dedup_key(level, message, alert_context)
            if await self._is_duplicate(dedup_key):
                self.logger.info(f"Duplicate alert suppressed: {dedup_key}")
                return alert_id
            
            # Localisation du message
            localized_message = await self.locale_manager.localize_message(
                message=message,
                locale=locale,
                context=alert_context.metadata
            )
            
            # Traitement par le warning processor
            processed_alert = await self.warning_processor.process_warning(
                level=level,
                message=localized_message,
                context=alert_context,
                tags=tags or {}
            )
            
            # Persistance en base
            await self._persist_alert(
                alert_id=alert_id,
                level=level,
                message=localized_message,
                context=alert_context,
                tags=tags,
                priority=priority,
                processed_data=processed_alert
            )
            
            # Envoi des notifications
            await self._send_notifications(
                alert_id=alert_id,
                level=level,
                message=localized_message,
                context=alert_context,
                tags=tags,
                processed_data=processed_alert
            )
            
            # Programmation de l'escalade si nécessaire
            if escalate_after and level in [AlertLevel.CRITICAL, AlertLevel.HIGH]:
                await self._schedule_escalation(alert_id, escalate_after)
            
            # Mise à jour des métriques
            self.alert_counter.labels(
                tenant_id=self.tenant_id,
                level=level.value,
                service=alert_context.service_name,
                status='sent'
            ).inc()
            
            self.active_alerts.labels(
                tenant_id=self.tenant_id,
                level=level.value
            ).inc()
            
            # Mise à jour du cache de déduplication
            await self._update_dedup_cache(dedup_key, alert_id)
            
            processing_time = time.time() - start_time
            self.alert_duration.labels(
                tenant_id=self.tenant_id,
                level=level.value
            ).observe(processing_time)
            
            self.logger.info(
                f"Alert sent successfully",
                extra={
                    "alert_id": alert_id,
                    "level": level.value,
                    "service": alert_context.service_name,
                    "processing_time": processing_time,
                    "tenant_id": self.tenant_id
                }
            )
            
            return alert_id
            
        except Exception as e:
            self.logger.error(
                f"Failed to send alert: {str(e)}",
                extra={
                    "alert_id": alert_id,
                    "level": level.value if hasattr(level, 'value') else str(level),
                    "error": str(e),
                    "tenant_id": self.tenant_id
                },
                exc_info=True
            )
            
            self.alert_counter.labels(
                tenant_id=self.tenant_id,
                level=level.value if hasattr(level, 'value') else str(level),
                service='unknown',
                status='failed'
            ).inc()
            
            raise
    
    async def send_custom_alert(
        self,
        template_name: str,
        severity: str,
        data: Dict[str, Any],
        recipient_channels: Optional[List[str]] = None,
        locale: str = "en"
    ) -> str:
        """
        Envoie une alerte basée sur un template personnalisé
        
        Args:
            template_name: Nom du template à utiliser
            severity: Niveau de sévérité
            data: Données pour le template
            recipient_channels: Canaux de notification spécifiques
            locale: Locale pour la localisation
            
        Returns:
            str: ID de l'alerte
        """
        try:
            # Génération du message à partir du template
            rendered_message = await self.template_engine.render_template(
                template_name=template_name,
                data=data,
                locale=locale,
                tenant_id=self.tenant_id
            )
            
            # Enrichissement des données
            enriched_data = {
                **data,
                'template_name': template_name,
                'recipient_channels': recipient_channels,
                'render_timestamp': datetime.utcnow().isoformat()
            }
            
            # Envoi via la méthode standard
            alert_id = await self.send_warning(
                level=severity,
                message=rendered_message,
                context=enriched_data,
                locale=locale
            )
            
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Failed to send custom alert: {str(e)}")
            raise
    
    async def track_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Suivi de métriques avec alerting automatique basé sur les seuils
        
        Args:
            metric_name: Nom de la métrique
            value: Valeur de la métrique
            tags: Tags additionnels
            timestamp: Timestamp (par défaut: maintenant)
        """
        try:
            metric_key = f"metrics:{self.tenant_id}:{metric_name}"
            metric_data = {
                'value': value,
                'tags': tags or {},
                'timestamp': (timestamp or datetime.utcnow()).isoformat(),
                'tenant_id': self.tenant_id
            }
            
            # Stockage dans Redis
            await self.redis_client.setex(
                metric_key,
                3600,  # TTL: 1 heure
                json.dumps(metric_data)
            )
            
            # Vérification des seuils d'alerte
            await self._check_metric_thresholds(metric_name, value, tags)
            
        except Exception as e:
            self.logger.error(f"Failed to track metric {metric_name}: {str(e)}")
    
    async def get_alert_history(
        self,
        limit: int = 100,
        level_filter: Optional[AlertLevel] = None,
        service_filter: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des alertes avec filtrage avancé
        
        Args:
            limit: Nombre maximum de résultats
            level_filter: Filtrer par niveau d'alerte
            service_filter: Filtrer par service
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            List[Dict]: Liste des alertes
        """
        try:
            # Construction de la requête
            query_filters = {
                'tenant_id': self.tenant_id
            }
            
            if level_filter:
                query_filters['level'] = level_filter.value
            if service_filter:
                query_filters['service_name'] = service_filter
            if start_date:
                query_filters['created_at__gte'] = start_date
            if end_date:
                query_filters['created_at__lte'] = end_date
            
            # Exécution de la requête (implémentation dépendante du modèle)
            alerts = await self._query_alerts(query_filters, limit)
            
            return [alert.to_dict() for alert in alerts]
            
        except Exception as e:
            self.logger.error(f"Failed to get alert history: {str(e)}")
            return []
    
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """
        Marque une alerte comme acquittée
        
        Args:
            alert_id: ID de l'alerte
            user_id: ID de l'utilisateur qui acquitte
            
        Returns:
            bool: True si succès
        """
        try:
            # Mise à jour du statut
            await self._update_alert_status(
                alert_id=alert_id,
                status=AlertStatus.ACKNOWLEDGED,
                user_id=user_id,
                timestamp=datetime.utcnow()
            )
            
            # Annulation de l'escalade programmée
            await self._cancel_escalation(alert_id)
            
            # Notification Slack d'acquittement
            await self.slack_notifier.send_acknowledgment_notification(
                alert_id=alert_id,
                user_id=user_id,
                tenant_id=self.tenant_id
            )
            
            self.logger.info(f"Alert {alert_id} acknowledged by {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert {alert_id}: {str(e)}")
            return False
    
    async def resolve_alert(self, alert_id: str, user_id: str, resolution_note: str = "") -> bool:
        """
        Marque une alerte comme résolue
        
        Args:
            alert_id: ID de l'alerte
            user_id: ID de l'utilisateur qui résout
            resolution_note: Note de résolution
            
        Returns:
            bool: True si succès
        """
        try:
            # Mise à jour du statut
            await self._update_alert_status(
                alert_id=alert_id,
                status=AlertStatus.RESOLVED,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                note=resolution_note
            )
            
            # Mise à jour des métriques
            alert_data = await self._get_alert_data(alert_id)
            if alert_data:
                self.active_alerts.labels(
                    tenant_id=self.tenant_id,
                    level=alert_data['level']
                ).dec()
            
            # Notification de résolution
            await self.slack_notifier.send_resolution_notification(
                alert_id=alert_id,
                user_id=user_id,
                resolution_note=resolution_note,
                tenant_id=self.tenant_id
            )
            
            self.logger.info(f"Alert {alert_id} resolved by {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resolve alert {alert_id}: {str(e)}")
            return False
    
    # Méthodes privées
    
    def _init_rate_limiter(self) -> Dict[str, Any]:
        """Initialise le rate limiter basé sur la configuration"""
        return {
            AlertLevel.CRITICAL: {'max_per_minute': 100, 'max_per_hour': 1000},
            AlertLevel.HIGH: {'max_per_minute': 50, 'max_per_hour': 500},
            AlertLevel.WARNING: {'max_per_minute': 20, 'max_per_hour': 200},
            AlertLevel.INFO: {'max_per_minute': 10, 'max_per_hour': 100},
            AlertLevel.DEBUG: {'max_per_minute': 5, 'max_per_hour': 50}
        }
    
    def _load_escalation_rules(self) -> Dict[str, Any]:
        """Charge les règles d'escalade depuis la configuration"""
        return self.config.get('escalation_rules', {
            AlertLevel.CRITICAL: {'escalate_after_minutes': 5, 'escalate_to': 'on_call_manager'},
            AlertLevel.HIGH: {'escalate_after_minutes': 15, 'escalate_to': 'team_lead'},
            AlertLevel.WARNING: {'escalate_after_minutes': 60, 'escalate_to': 'team_channel'}
        })
    
    async def _check_rate_limit(self, level: AlertLevel, service: str) -> bool:
        """Vérifie si l'alerte respecte les limites de taux"""
        rate_key = f"rate_limit:{self.tenant_id}:{level.value}:{service}"
        current_count = await self.redis_client.get(rate_key)
        
        if current_count is None:
            await self.redis_client.setex(rate_key, 60, 1)
            return True
        
        limits = self.rate_limiter.get(level, {})
        max_per_minute = limits.get('max_per_minute', 10)
        
        if int(current_count) >= max_per_minute:
            return False
        
        await self.redis_client.incr(rate_key)
        return True
    
    def _generate_dedup_key(self, level: AlertLevel, message: str, context: AlertContext) -> str:
        """Génère une clé de déduplication pour l'alerte"""
        key_components = [
            self.tenant_id,
            level.value,
            context.service_name,
            message[:100]  # Premiers 100 caractères du message
        ]
        return ":".join(key_components)
    
    async def _is_duplicate(self, dedup_key: str) -> bool:
        """Vérifie si l'alerte est un doublon"""
        existing = await self.redis_client.get(f"dedup:{dedup_key}")
        return existing is not None
    
    async def _update_dedup_cache(self, dedup_key: str, alert_id: str) -> None:
        """Met à jour le cache de déduplication"""
        await self.redis_client.setex(
            f"dedup:{dedup_key}",
            300,  # 5 minutes
            alert_id
        )
    
    async def _persist_alert(
        self,
        alert_id: str,
        level: AlertLevel,
        message: str,
        context: AlertContext,
        tags: Optional[Dict[str, str]],
        priority: int,
        processed_data: Dict[str, Any]
    ) -> None:
        """Persiste l'alerte en base de données"""
        # Implémentation spécifique au modèle de données
        pass
    
    async def _send_notifications(
        self,
        alert_id: str,
        level: AlertLevel,
        message: str,
        context: AlertContext,
        tags: Optional[Dict[str, str]],
        processed_data: Dict[str, Any]
    ) -> None:
        """Envoie les notifications via les différents canaux"""
        # Notification Slack
        await self.slack_notifier.send_alert_notification(
            alert_id=alert_id,
            level=level,
            message=message,
            context=context,
            tags=tags,
            processed_data=processed_data
        )
        
        # Autres canaux (email, webhook, etc.) selon la configuration
        if self.config.get('email_notifications_enabled', False):
            await self._send_email_notification(alert_id, level, message, context)
        
        if self.config.get('webhook_notifications_enabled', False):
            await self._send_webhook_notification(alert_id, level, message, context)
    
    async def _schedule_escalation(self, alert_id: str, escalate_after: int) -> None:
        """Programme une escalade automatique"""
        escalation_data = {
            'alert_id': alert_id,
            'tenant_id': self.tenant_id,
            'scheduled_at': (datetime.utcnow() + timedelta(minutes=escalate_after)).isoformat()
        }
        
        await self.redis_client.setex(
            f"escalation:{alert_id}",
            escalate_after * 60,
            json.dumps(escalation_data)
        )
    
    async def _check_metric_thresholds(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]]
    ) -> None:
        """Vérifie les seuils de métriques et déclenche des alertes si nécessaire"""
        # Récupération des seuils configurés
        thresholds = self.config.get('metric_thresholds', {}).get(metric_name, {})
        
        for threshold_level, threshold_value in thresholds.items():
            if value >= threshold_value:
                await self.send_warning(
                    level=threshold_level,
                    message=f"Metric {metric_name} exceeded threshold: {value} >= {threshold_value}",
                    context={
                        'metric_name': metric_name,
                        'current_value': value,
                        'threshold_value': threshold_value,
                        'tags': tags or {}
                    }
                )
