"""
Middleware avancé pour le système de notifications
=================================================

Middleware ultra-sophistiqué avec circuit breakers, rate limiting,
observabilité, et sécurité enterprise.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from abc import ABC, abstractmethod
from functools import wraps
from contextvars import ContextVar
import uuid
from dataclasses import dataclass

import aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from prometheus_client import Counter, Histogram, Gauge
import opentelemetry.trace as trace
from opentelemetry import baggage
from opentelemetry.trace import Status, StatusCode
import structlog

from .schemas import *
from .config import NotificationSettings
from .validators import NotificationValidator, ValidationLevel


# Context variables pour le tracking
request_id: ContextVar[str] = ContextVar('request_id', default='')
tenant_id: ContextVar[str] = ContextVar('tenant_id', default='')
user_id: ContextVar[str] = ContextVar('user_id', default='')


@dataclass
class MiddlewareContext:
    """Contexte partagé entre middlewares"""
    request_id: str
    tenant_id: str
    user_id: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class BaseMiddleware(ABC):
    """Middleware de base"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.enabled = self.config.get('enabled', True)
        self.order = self.config.get('order', 100)
    
    @abstractmethod
    async def process_request(
        self,
        notification: NotificationCreateSchema,
        context: MiddlewareContext,
        call_next: Callable
    ) -> Any:
        """Traiter la requête"""
        pass
    
    async def process_response(
        self,
        response: Any,
        context: MiddlewareContext
    ) -> Any:
        """Traiter la réponse"""
        return response
    
    async def process_error(
        self,
        error: Exception,
        context: MiddlewareContext
    ) -> Optional[Exception]:
        """Traiter les erreurs"""
        return error


class TracingMiddleware(BaseMiddleware):
    """Middleware de tracing distribué avec OpenTelemetry"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.tracer = trace.get_tracer(__name__)
    
    async def process_request(
        self,
        notification: NotificationCreateSchema,
        context: MiddlewareContext,
        call_next: Callable
    ) -> Any:
        """Tracer la requête de notification"""
        
        with self.tracer.start_as_current_span(
            "notification.process",
            attributes={
                "notification.request_id": context.request_id,
                "notification.tenant_id": context.tenant_id,
                "notification.user_id": context.user_id or "anonymous",
                "notification.priority": notification.priority.value,
                "notification.channels": [c.type.value for c in notification.channels],
                "notification.recipients_count": len(notification.recipients),
            }
        ) as span:
            
            # Ajouter des tags de baggage
            baggage.set_baggage("tenant.id", context.tenant_id)
            baggage.set_baggage("request.id", context.request_id)
            
            try:
                # Continuer le traitement
                response = await call_next(notification, context)
                
                # Marquer comme succès
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("notification.success", True)
                
                if hasattr(response, '__len__'):
                    span.set_attribute("notification.created_count", len(response))
                
                return response
            
            except Exception as e:
                # Marquer comme erreur
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("notification.success", False)
                span.set_attribute("notification.error.type", type(e).__name__)
                span.set_attribute("notification.error.message", str(e))
                
                raise


class RateLimitingMiddleware(BaseMiddleware):
    """Middleware de limitation de taux avancé"""
    
    def __init__(self, config: Dict[str, Any] = None, redis_client: aioredis.Redis = None):
        super().__init__(config)
        self.redis = redis_client
        
        # Limites par défaut
        self.limits = {
            'global_per_minute': config.get('global_per_minute', 10000),
            'tenant_per_minute': config.get('tenant_per_minute', 1000),
            'user_per_minute': config.get('user_per_minute', 100),
            'burst_allowance': config.get('burst_allowance', 10)
        }
        
        # Métriques
        self.rate_limit_counter = Counter(
            'notification_rate_limit_total',
            'Total rate limit hits',
            ['type', 'tenant_id']
        )
    
    async def process_request(
        self,
        notification: NotificationCreateSchema,
        context: MiddlewareContext,
        call_next: Callable
    ) -> Any:
        """Appliquer les limites de taux"""
        
        if not self.redis:
            self.logger.warning("Redis non disponible, rate limiting désactivé")
            return await call_next(notification, context)
        
        # Vérifier les limites dans l'ordre de priorité
        limits_to_check = [
            ('global', 'global', self.limits['global_per_minute']),
            ('tenant', context.tenant_id, self.limits['tenant_per_minute']),
        ]
        
        if context.user_id:
            limits_to_check.append(
                ('user', f"{context.tenant_id}:{context.user_id}", self.limits['user_per_minute'])
            )
        
        for limit_type, key, limit_value in limits_to_check:
            if not await self._check_rate_limit(limit_type, key, limit_value):
                self.rate_limit_counter.labels(
                    type=limit_type,
                    tenant_id=context.tenant_id
                ).inc()
                
                raise NotificationRateLimitError(
                    f"Limite de taux {limit_type} dépassée: {limit_value}/minute"
                )
        
        # Incrémenter les compteurs
        await self._increment_counters(context)
        
        return await call_next(notification, context)
    
    async def _check_rate_limit(self, limit_type: str, key: str, limit: int) -> bool:
        """Vérifier une limite de taux spécifique"""
        
        rate_key = f"rate_limit:{limit_type}:{key}"
        window_seconds = 60  # 1 minute
        
        now = time.time()
        
        # Utiliser une fenêtre glissante avec Redis
        pipeline = self.redis.pipeline()
        
        # Nettoyer les entrées expirées
        pipeline.zremrangebyscore(rate_key, 0, now - window_seconds)
        
        # Compter les requêtes actuelles
        pipeline.zcard(rate_key)
        
        results = await pipeline.execute()
        current_count = results[1]
        
        # Vérifier si on peut ajouter une nouvelle requête
        if current_count >= limit:
            return False
        
        return True
    
    async def _increment_counters(self, context: MiddlewareContext):
        """Incrémenter tous les compteurs"""
        
        now = time.time()
        request_id = str(uuid.uuid4())
        
        # Incrémenter pour tous les types de limites
        keys_to_increment = [
            f"rate_limit:global:global",
            f"rate_limit:tenant:{context.tenant_id}",
        ]
        
        if context.user_id:
            keys_to_increment.append(
                f"rate_limit:user:{context.tenant_id}:{context.user_id}"
            )
        
        pipeline = self.redis.pipeline()
        
        for key in keys_to_increment:
            # Ajouter l'entrée avec timestamp
            pipeline.zadd(key, {request_id: now})
            # Expirer la clé
            pipeline.expire(key, 120)  # 2 minutes de sécurité
        
        await pipeline.execute()


class ValidationMiddleware(BaseMiddleware):
    """Middleware de validation avancée"""
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        validator: NotificationValidator = None
    ):
        super().__init__(config)
        self.validator = validator
        self.validation_level = ValidationLevel(
            config.get('validation_level', ValidationLevel.STANDARD.value)
        )
        
        # Métriques
        self.validation_counter = Counter(
            'notification_validation_total',
            'Total validation attempts',
            ['result', 'tenant_id']
        )
        
        self.validation_duration = Histogram(
            'notification_validation_duration_seconds',
            'Validation duration',
            ['tenant_id']
        )
    
    async def process_request(
        self,
        notification: NotificationCreateSchema,
        context: MiddlewareContext,
        call_next: Callable
    ) -> Any:
        """Valider la notification"""
        
        if not self.validator:
            self.logger.warning("Validateur non configuré")
            return await call_next(notification, context)
        
        start_time = time.time()
        
        try:
            # Effectuer la validation
            validation_result = await self.validator.validate_notification(
                notification,
                context.tenant_id,
                context.user_id,
                self.validation_level
            )
            
            duration = time.time() - start_time
            self.validation_duration.labels(tenant_id=context.tenant_id).observe(duration)
            
            # Ajouter les résultats au contexte
            context.metadata['validation'] = {
                'is_valid': validation_result.is_valid,
                'errors': [error.message for error in validation_result.errors],
                'warnings': validation_result.warnings,
                'metadata': validation_result.metadata,
                'duration_seconds': duration
            }
            
            # Loguer les avertissements
            if validation_result.warnings:
                self.logger.warning(
                    "Avertissements de validation",
                    warnings=validation_result.warnings,
                    request_id=context.request_id
                )
            
            # Rejeter si validation échoue
            if not validation_result.is_valid:
                error_messages = [error.message for error in validation_result.errors]
                
                self.validation_counter.labels(
                    result='failed',
                    tenant_id=context.tenant_id
                ).inc()
                
                raise NotificationValidationError(
                    f"Validation échouée: {'; '.join(error_messages)}"
                )
            
            self.validation_counter.labels(
                result='success',
                tenant_id=context.tenant_id
            ).inc()
            
            return await call_next(notification, context)
        
        except Exception as e:
            duration = time.time() - start_time
            self.validation_duration.labels(tenant_id=context.tenant_id).observe(duration)
            
            if not isinstance(e, NotificationValidationError):
                self.validation_counter.labels(
                    result='error',
                    tenant_id=context.tenant_id
                ).inc()
            
            raise


class CircuitBreakerMiddleware(BaseMiddleware):
    """Middleware circuit breaker pour résilience"""
    
    def __init__(self, config: Dict[str, Any] = None, redis_client: aioredis.Redis = None):
        super().__init__(config)
        self.redis = redis_client
        
        # Configuration du circuit breaker
        self.failure_threshold = config.get('failure_threshold', 5)
        self.recovery_timeout = config.get('recovery_timeout', 60)  # secondes
        self.success_threshold = config.get('success_threshold', 3)
        
        # États du circuit
        self.CLOSED = 'closed'
        self.OPEN = 'open'
        self.HALF_OPEN = 'half_open'
        
        # Métriques
        self.circuit_breaker_counter = Counter(
            'notification_circuit_breaker_total',
            'Circuit breaker state changes',
            ['state', 'tenant_id']
        )
    
    async def process_request(
        self,
        notification: NotificationCreateSchema,
        context: MiddlewareContext,
        call_next: Callable
    ) -> Any:
        """Appliquer le circuit breaker"""
        
        circuit_key = f"circuit_breaker:tenant:{context.tenant_id}"
        
        # Vérifier l'état du circuit
        circuit_state = await self._get_circuit_state(circuit_key)
        
        if circuit_state == self.OPEN:
            # Circuit ouvert, vérifier si on peut passer en half-open
            if await self._should_attempt_reset(circuit_key):
                await self._set_circuit_state(circuit_key, self.HALF_OPEN)
                circuit_state = self.HALF_OPEN
            else:
                # Rejeter la requête
                self.circuit_breaker_counter.labels(
                    state='rejected',
                    tenant_id=context.tenant_id
                ).inc()
                
                raise NotificationServiceError(
                    "Service temporairement indisponible (circuit breaker ouvert)"
                )
        
        try:
            # Traiter la requête
            response = await call_next(notification, context)
            
            # Succès
            await self._record_success(circuit_key, circuit_state)
            
            return response
        
        except Exception as e:
            # Échec
            await self._record_failure(circuit_key, circuit_state)
            raise
    
    async def _get_circuit_state(self, circuit_key: str) -> str:
        """Obtenir l'état actuel du circuit"""
        if not self.redis:
            return self.CLOSED
        
        state = await self.redis.hget(circuit_key, 'state')
        return state or self.CLOSED
    
    async def _set_circuit_state(self, circuit_key: str, state: str):
        """Définir l'état du circuit"""
        if not self.redis:
            return
        
        await self.redis.hset(circuit_key, mapping={
            'state': state,
            'timestamp': time.time()
        })
        
        # Expirer après 24h
        await self.redis.expire(circuit_key, 86400)
    
    async def _should_attempt_reset(self, circuit_key: str) -> bool:
        """Déterminer si on peut tenter de réinitialiser le circuit"""
        if not self.redis:
            return True
        
        timestamp = await self.redis.hget(circuit_key, 'timestamp')
        if not timestamp:
            return True
        
        elapsed = time.time() - float(timestamp)
        return elapsed >= self.recovery_timeout
    
    async def _record_success(self, circuit_key: str, current_state: str):
        """Enregistrer un succès"""
        if not self.redis:
            return
        
        if current_state == self.HALF_OPEN:
            # Compter les succès consécutifs
            success_count = await self.redis.hincrby(circuit_key, 'success_count', 1)
            
            if success_count >= self.success_threshold:
                # Fermer le circuit
                await self._set_circuit_state(circuit_key, self.CLOSED)
                await self.redis.hdel(circuit_key, 'failure_count', 'success_count')
                
                self.circuit_breaker_counter.labels(
                    state='closed',
                    tenant_id=circuit_key.split(':')[-1]
                ).inc()
        
        elif current_state == self.CLOSED:
            # Réinitialiser le compteur d'échecs
            await self.redis.hdel(circuit_key, 'failure_count')
    
    async def _record_failure(self, circuit_key: str, current_state: str):
        """Enregistrer un échec"""
        if not self.redis:
            return
        
        failure_count = await self.redis.hincrby(circuit_key, 'failure_count', 1)
        
        if current_state == self.CLOSED and failure_count >= self.failure_threshold:
            # Ouvrir le circuit
            await self._set_circuit_state(circuit_key, self.OPEN)
            
            self.circuit_breaker_counter.labels(
                state='opened',
                tenant_id=circuit_key.split(':')[-1]
            ).inc()
        
        elif current_state == self.HALF_OPEN:
            # Retourner à l'état ouvert
            await self._set_circuit_state(circuit_key, self.OPEN)
            await self.redis.hdel(circuit_key, 'success_count')


class AuditMiddleware(BaseMiddleware):
    """Middleware d'audit et logging"""
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        db_session: AsyncSession = None
    ):
        super().__init__(config)
        self.db = db_session
        self.log_level = config.get('log_level', 'INFO')
        self.store_in_db = config.get('store_in_db', True)
        
        # Configurer le logger structuré
        self.audit_logger = structlog.get_logger("audit")
    
    async def process_request(
        self,
        notification: NotificationCreateSchema,
        context: MiddlewareContext,
        call_next: Callable
    ) -> Any:
        """Auditer la requête"""
        
        # Log de début
        self.audit_logger.info(
            "Notification request started",
            request_id=context.request_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            notification_title=notification.title,
            recipients_count=len(notification.recipients),
            channels=[c.type.value for c in notification.channels],
            priority=notification.priority.value
        )
        
        start_time = time.time()
        
        try:
            response = await call_next(notification, context)
            
            duration = time.time() - start_time
            
            # Log de succès
            self.audit_logger.info(
                "Notification request completed",
                request_id=context.request_id,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                duration_seconds=duration,
                result="success",
                notifications_created=len(response) if hasattr(response, '__len__') else 1
            )
            
            # Stocker en base si configuré
            if self.store_in_db and self.db:
                await self._store_audit_record(context, notification, duration, True, None)
            
            return response
        
        except Exception as e:
            duration = time.time() - start_time
            
            # Log d'erreur
            self.audit_logger.error(
                "Notification request failed",
                request_id=context.request_id,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                duration_seconds=duration,
                result="error",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            # Stocker en base si configuré
            if self.store_in_db and self.db:
                await self._store_audit_record(context, notification, duration, False, str(e))
            
            raise
    
    async def _store_audit_record(
        self,
        context: MiddlewareContext,
        notification: NotificationCreateSchema,
        duration: float,
        success: bool,
        error_message: Optional[str]
    ):
        """Stocker l'enregistrement d'audit en base"""
        # Ici, on pourrait créer une table d'audit dédiée
        # Pour l'instant, on log seulement
        pass


class MiddlewarePipeline:
    """Pipeline de middleware pour traiter les notifications"""
    
    def __init__(self, settings: NotificationSettings):
        self.settings = settings
        self.middlewares: List[BaseMiddleware] = []
        self.logger = structlog.get_logger("MiddlewarePipeline")
    
    def add_middleware(self, middleware: BaseMiddleware):
        """Ajouter un middleware au pipeline"""
        self.middlewares.append(middleware)
        # Trier par ordre de priorité
        self.middlewares.sort(key=lambda m: m.order)
    
    def remove_middleware(self, middleware_type: type):
        """Supprimer un type de middleware"""
        self.middlewares = [m for m in self.middlewares if not isinstance(m, middleware_type)]
    
    async def process(
        self,
        notification: NotificationCreateSchema,
        tenant_id: str,
        user_id: Optional[str] = None,
        processor_func: Optional[Callable] = None
    ) -> Any:
        """Traiter une notification à travers le pipeline"""
        
        # Créer le contexte
        context = MiddlewareContext(
            request_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        
        # Définir les context vars
        request_id.set(context.request_id)
        tenant_id_var.set(context.tenant_id)
        if context.user_id:
            user_id_var.set(context.user_id)
        
        # Créer la chaîne de traitement
        async def process_chain(notif: NotificationCreateSchema, ctx: MiddlewareContext):
            if processor_func:
                return await processor_func(notif, ctx)
            else:
                # Traitement par défaut
                return {"status": "processed", "context": ctx.to_dict()}
        
        # Construire la chaîne de middleware
        handler = process_chain
        
        for middleware in reversed([m for m in self.middlewares if m.enabled]):
            current_middleware = middleware
            next_handler = handler
            
            async def middleware_handler(
                notif: NotificationCreateSchema,
                ctx: MiddlewareContext,
                mw=current_middleware,
                next_h=next_handler
            ):
                return await mw.process_request(notif, ctx, lambda n, c: next_h(n, c))
            
            handler = middleware_handler
        
        try:
            # Traiter à travers tous les middlewares
            result = await handler(notification, context)
            
            # Post-traitement des réponses
            for middleware in self.middlewares:
                if middleware.enabled:
                    result = await middleware.process_response(result, context)
            
            return result
        
        except Exception as e:
            # Traitement d'erreur
            processed_error = e
            
            for middleware in self.middlewares:
                if middleware.enabled:
                    middleware_error = await middleware.process_error(processed_error, context)
                    if middleware_error:
                        processed_error = middleware_error
            
            raise processed_error


def create_default_pipeline(
    settings: NotificationSettings,
    db_session: AsyncSession,
    redis_client: aioredis.Redis,
    validator: NotificationValidator
) -> MiddlewarePipeline:
    """Créer un pipeline par défaut avec tous les middlewares"""
    
    pipeline = MiddlewarePipeline(settings)
    
    # Ajouter les middlewares dans l'ordre
    
    # 1. Tracing (ordre 10)
    if settings.is_feature_enabled('distributed_tracing'):
        pipeline.add_middleware(TracingMiddleware({'order': 10}))
    
    # 2. Rate limiting (ordre 20)
    pipeline.add_middleware(RateLimitingMiddleware(
        {
            'order': 20,
            'tenant_per_minute': settings.get_rate_limit('per_tenant_per_minute'),
            'user_per_minute': settings.get_rate_limit('per_user_per_minute')
        },
        redis_client
    ))
    
    # 3. Validation (ordre 30)
    pipeline.add_middleware(ValidationMiddleware(
        {'order': 30, 'validation_level': 'standard'},
        validator
    ))
    
    # 4. Circuit breaker (ordre 40)
    if settings.is_feature_enabled('circuit_breaker'):
        pipeline.add_middleware(CircuitBreakerMiddleware(
            {'order': 40, 'failure_threshold': 5, 'recovery_timeout': 60},
            redis_client
        ))
    
    # 5. Audit (ordre 50)
    pipeline.add_middleware(AuditMiddleware(
        {'order': 50, 'store_in_db': True},
        db_session
    ))
    
    return pipeline
