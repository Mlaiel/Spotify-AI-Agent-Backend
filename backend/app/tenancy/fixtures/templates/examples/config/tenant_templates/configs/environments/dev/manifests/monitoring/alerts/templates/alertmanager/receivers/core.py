"""
Core Alertmanager Receivers Management Module

Gestionnaire principal pour la gestion des receivers d'alertes multi-tenant
avec escalade intelligente et intégration multi-canaux.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
from pathlib import Path
import jinja2
from prometheus_client import Counter, Histogram, Gauge
import aiohttp
import backoff
from circuit_breaker import CircuitBreaker

from .models import (
    ReceiverConfig,
    NotificationChannel,
    EscalationPolicy,
    AlertContext,
    NotificationResult,
    ReceiverHealth
)
from .exceptions import (
    ReceiverConfigError,
    NotificationError,
    TemplateRenderError,
    EscalationError
)
from .utils import (
    ReceiverValidator,
    TemplateRenderer,
    NotificationThrottler,
    MetricsCollector,
    SecretManager,
    AuditLogger
)

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class NotificationStatus(Enum):
    """Status des notifications"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"
    ESCALATED = "escalated"

@dataclass
class ReceiverMetrics:
    """Métriques de performance des receivers"""
    notifications_sent: int = 0
    notifications_failed: int = 0
    avg_response_time: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    circuit_breaker_state: str = "closed"
    
class AlertReceiverManager:
    """
    Gestionnaire principal des receivers d'alertes multi-tenant.
    
    Fonctionnalités:
    - Gestion multi-tenant des configurations
    - Escalade intelligente des alertes
    - Retry logic avec backoff exponentiel
    - Circuit breaker pour la résilience
    - Métriques et monitoring avancés
    - Templates dynamiques
    """
    
    def __init__(
        self,
        tenant_id: str,
        config_path: Optional[str] = None,
        enable_metrics: bool = True,
        enable_audit: bool = True,
        enable_circuit_breaker: bool = True,
        max_retry_attempts: int = 3,
        default_timeout: int = 30
    ):
        self.tenant_id = tenant_id
        self.config_path = config_path
        self.enable_metrics = enable_metrics
        self.enable_audit = enable_audit
        self.enable_circuit_breaker = enable_circuit_breaker
        self.max_retry_attempts = max_retry_attempts
        self.default_timeout = default_timeout
        
        # Composants internes
        self.receivers: Dict[str, ReceiverConfig] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.receiver_metrics: Dict[str, ReceiverMetrics] = {}
        
        # Utilitaires
        self.validator = ReceiverValidator()
        self.template_renderer = TemplateRenderer()
        self.throttler = NotificationThrottler()
        self.secret_manager = SecretManager()
        
        if enable_metrics:
            self.metrics_collector = MetricsCollector()
            self._init_prometheus_metrics()
            
        if enable_audit:
            self.audit_logger = AuditLogger(tenant_id)
            
        # Session HTTP réutilisable
        self.http_session: Optional[aiohttp.ClientSession] = None
        
    def _init_prometheus_metrics(self):
        """Initialise les métriques Prometheus"""
        self.notifications_total = Counter(
            'alertmanager_notifications_total',
            'Total number of notifications sent',
            ['tenant_id', 'receiver_name', 'status', 'channel_type']
        )
        
        self.notification_duration = Histogram(
            'alertmanager_notification_duration_seconds',
            'Time spent sending notifications',
            ['tenant_id', 'receiver_name', 'channel_type']
        )
        
        self.escalation_events = Counter(
            'alertmanager_escalation_events_total',
            'Total number of escalation events',
            ['tenant_id', 'receiver_name', 'escalation_level']
        )
        
        self.receiver_health = Gauge(
            'alertmanager_receiver_health',
            'Health status of receivers (1=healthy, 0=unhealthy)',
            ['tenant_id', 'receiver_name']
        )
        
    async def __aenter__(self):
        """Context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.cleanup()
        
    async def initialize(self):
        """Initialise le gestionnaire"""
        try:
            # Créer la session HTTP
            timeout = aiohttp.ClientTimeout(total=self.default_timeout)
            self.http_session = aiohttp.ClientSession(timeout=timeout)
            
            # Charger la configuration
            if self.config_path:
                await self.load_config(self.config_path)
                
            # Initialiser les circuit breakers
            if self.enable_circuit_breaker:
                await self._init_circuit_breakers()
                
            logger.info(f"AlertReceiverManager initialized for tenant {self.tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AlertReceiverManager: {e}")
            raise ReceiverConfigError(f"Initialization failed: {e}")
            
    async def cleanup(self):
        """Nettoie les ressources"""
        try:
            if self.http_session:
                await self.http_session.close()
                
            logger.info(f"AlertReceiverManager cleaned up for tenant {self.tenant_id}")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            
    async def load_config(self, config_path: str):
        """Charge la configuration depuis un fichier YAML"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise ReceiverConfigError(f"Config file not found: {config_path}")
                
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            # Valider la configuration
            await self.validator.validate_config(config_data, self.tenant_id)
            
            # Charger les receivers pour ce tenant
            tenant_config = config_data.get('tenants', {}).get(self.tenant_id, {})
            
            if not tenant_config:
                raise ReceiverConfigError(f"No configuration found for tenant {self.tenant_id}")
                
            # Charger les receivers
            for receiver_data in tenant_config.get('receivers', []):
                receiver = ReceiverConfig.from_dict(receiver_data)
                self.receivers[receiver.name] = receiver
                self.receiver_metrics[receiver.name] = ReceiverMetrics()
                
            # Charger les politiques d'escalade
            for policy_data in tenant_config.get('escalation_policies', []):
                policy = EscalationPolicy.from_dict(policy_data)
                self.escalation_policies[policy.name] = policy
                
            logger.info(f"Loaded {len(self.receivers)} receivers for tenant {self.tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise ReceiverConfigError(f"Config loading failed: {e}")
            
    async def _init_circuit_breakers(self):
        """Initialise les circuit breakers pour chaque receiver"""
        for receiver_name in self.receivers.keys():
            self.circuit_breakers[receiver_name] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                expected_exception=NotificationError
            )
            
    async def send_alert(
        self,
        alert_context: AlertContext,
        receiver_names: Optional[List[str]] = None
    ) -> Dict[str, NotificationResult]:
        """
        Envoie une alerte aux receivers spécifiés ou tous les receivers.
        
        Args:
            alert_context: Contexte de l'alerte
            receiver_names: Liste des receivers à utiliser (optionnel)
            
        Returns:
            Dictionnaire des résultats par receiver
        """
        results = {}
        
        # Déterminer les receivers à utiliser
        target_receivers = receiver_names or list(self.receivers.keys())
        
        # Filtrer les receivers selon la sévérité
        filtered_receivers = await self._filter_receivers_by_severity(
            target_receivers, alert_context.severity
        )
        
        # Vérifier le throttling
        if await self.throttler.is_throttled(alert_context):
            logger.warning(f"Alert throttled: {alert_context.alert_name}")
            return {}
            
        # Envoyer en parallèle aux receivers appropriés
        tasks = []
        for receiver_name in filtered_receivers:
            if receiver_name in self.receivers:
                task = self._send_to_receiver(receiver_name, alert_context)
                tasks.append((receiver_name, task))
                
        if tasks:
            task_results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            for (receiver_name, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    results[receiver_name] = NotificationResult(
                        success=False,
                        error_message=str(result),
                        timestamp=datetime.utcnow()
                    )
                else:
                    results[receiver_name] = result
                    
        # Gérer l'escalade si nécessaire
        if alert_context.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            await self._handle_escalation(alert_context, results)
            
        # Audit
        if self.enable_audit:
            await self.audit_logger.log_notification(alert_context, results)
            
        return results
        
    async def _send_to_receiver(
        self,
        receiver_name: str,
        alert_context: AlertContext
    ) -> NotificationResult:
        """Envoie une alerte à un receiver spécifique"""
        receiver = self.receivers[receiver_name]
        start_time = time.time()
        
        try:
            # Vérifier le circuit breaker
            if (self.enable_circuit_breaker and 
                receiver_name in self.circuit_breakers):
                circuit_breaker = self.circuit_breakers[receiver_name]
                if circuit_breaker.state == "open":
                    raise NotificationError(f"Circuit breaker open for {receiver_name}")
                    
            # Préparer le message
            message = await self._prepare_message(receiver, alert_context)
            
            # Envoyer avec retry
            result = await self._send_with_retry(receiver, message, alert_context)
            
            # Mettre à jour les métriques
            duration = time.time() - start_time
            await self._update_metrics(receiver_name, True, duration)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            await self._update_metrics(receiver_name, False, duration)
            
            logger.error(f"Failed to send to {receiver_name}: {e}")
            return NotificationResult(
                success=False,
                error_message=str(e),
                timestamp=datetime.utcnow(),
                duration=duration
            )
            
    @backoff.on_exception(
        backoff.expo,
        (NotificationError, aiohttp.ClientError),
        max_tries=3,
        max_time=300
    )
    async def _send_with_retry(
        self,
        receiver: ReceiverConfig,
        message: Dict[str, Any],
        alert_context: AlertContext
    ) -> NotificationResult:
        """Envoie avec logique de retry"""
        start_time = time.time()
        
        try:
            # Déléguer à la factory appropriée
            from .factories import get_receiver_factory
            
            factory = get_receiver_factory(receiver.channel_type)
            result = await factory.send_notification(
                receiver, message, alert_context, self.http_session
            )
            
            duration = time.time() - start_time
            return NotificationResult(
                success=True,
                timestamp=datetime.utcnow(),
                duration=duration,
                response_data=result
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Notification failed: {e}")
            raise NotificationError(f"Send failed: {e}")
            
    async def _prepare_message(
        self,
        receiver: ReceiverConfig,
        alert_context: AlertContext
    ) -> Dict[str, Any]:
        """Prépare le message pour le receiver"""
        try:
            # Résoudre les secrets
            config = await self.secret_manager.resolve_secrets(
                receiver.config, self.tenant_id
            )
            
            # Rendre le template
            template_data = {
                'alert': alert_context,
                'tenant_id': self.tenant_id,
                'receiver': receiver,
                'timestamp': datetime.utcnow(),
                'config': config
            }
            
            message = await self.template_renderer.render_template(
                receiver.channel_type,
                receiver.template_name or 'default',
                template_data
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Message preparation failed: {e}")
            raise TemplateRenderError(f"Template rendering failed: {e}")
            
    async def _filter_receivers_by_severity(
        self,
        receiver_names: List[str],
        severity: AlertSeverity
    ) -> List[str]:
        """Filtre les receivers selon la sévérité de l'alerte"""
        filtered = []
        
        for name in receiver_names:
            receiver = self.receivers.get(name)
            if not receiver:
                continue
                
            # Vérifier les critères de sévérité
            min_severity = getattr(receiver, 'min_severity', AlertSeverity.INFO)
            if severity.value <= min_severity.value:
                filtered.append(name)
                
        return filtered
        
    async def _handle_escalation(
        self,
        alert_context: AlertContext,
        initial_results: Dict[str, NotificationResult]
    ):
        """Gère l'escalade des alertes critiques"""
        # Vérifier si l'escalade est nécessaire
        failed_notifications = [
            name for name, result in initial_results.items()
            if not result.success
        ]
        
        if not failed_notifications:
            return
            
        # Chercher une politique d'escalade
        escalation_policy = self._get_escalation_policy(alert_context)
        if not escalation_policy:
            return
            
        try:
            await self._execute_escalation(
                alert_context, escalation_policy, failed_notifications
            )
            
            # Métrique d'escalade
            if self.enable_metrics:
                self.escalation_events.labels(
                    tenant_id=self.tenant_id,
                    receiver_name="escalation",
                    escalation_level=escalation_policy.level
                ).inc()
                
        except Exception as e:
            logger.error(f"Escalation failed: {e}")
            raise EscalationError(f"Escalation execution failed: {e}")
            
    def _get_escalation_policy(self, alert_context: AlertContext) -> Optional[EscalationPolicy]:
        """Trouve la politique d'escalade appropriée"""
        # Logique de sélection de politique basée sur le contexte
        for policy in self.escalation_policies.values():
            if (policy.severity_threshold <= alert_context.severity and
                policy.applies_to_tenant(self.tenant_id)):
                return policy
                
        return None
        
    async def _execute_escalation(
        self,
        alert_context: AlertContext,
        policy: EscalationPolicy,
        failed_receivers: List[str]
    ):
        """Exécute la politique d'escalade"""
        escalation_context = alert_context.model_copy()
        escalation_context.annotations['escalated_from'] = ', '.join(failed_receivers)
        escalation_context.annotations['escalation_policy'] = policy.name
        
        # Délai d'escalade
        if policy.delay_seconds > 0:
            await asyncio.sleep(policy.delay_seconds)
            
        # Envoyer aux receivers d'escalade
        await self.send_alert(escalation_context, policy.escalation_receivers)
        
    async def _update_metrics(
        self,
        receiver_name: str,
        success: bool,
        duration: float
    ):
        """Met à jour les métriques de performance"""
        metrics = self.receiver_metrics[receiver_name]
        
        if success:
            metrics.notifications_sent += 1
            metrics.last_success = datetime.utcnow()
        else:
            metrics.notifications_failed += 1
            metrics.last_failure = datetime.utcnow()
            
        # Moyenne mobile de la durée
        if metrics.avg_response_time == 0:
            metrics.avg_response_time = duration
        else:
            metrics.avg_response_time = (metrics.avg_response_time * 0.8 + duration * 0.2)
            
        # Métriques Prometheus
        if self.enable_metrics:
            receiver = self.receivers[receiver_name]
            status = "success" if success else "failure"
            
            self.notifications_total.labels(
                tenant_id=self.tenant_id,
                receiver_name=receiver_name,
                status=status,
                channel_type=receiver.channel_type
            ).inc()
            
            self.notification_duration.labels(
                tenant_id=self.tenant_id,
                receiver_name=receiver_name,
                channel_type=receiver.channel_type
            ).observe(duration)
            
            # Health status
            health_score = 1.0 if success else 0.0
            self.receiver_health.labels(
                tenant_id=self.tenant_id,
                receiver_name=receiver_name
            ).set(health_score)
            
    async def get_receiver_health(self, receiver_name: str) -> ReceiverHealth:
        """Retourne l'état de santé d'un receiver"""
        if receiver_name not in self.receiver_metrics:
            raise ValueError(f"Unknown receiver: {receiver_name}")
            
        metrics = self.receiver_metrics[receiver_name]
        
        # Calculer le taux de succès
        total_notifications = metrics.notifications_sent + metrics.notifications_failed
        success_rate = (
            metrics.notifications_sent / total_notifications
            if total_notifications > 0 else 0
        )
        
        # Déterminer l'état de santé
        is_healthy = (
            success_rate >= 0.95 and
            metrics.avg_response_time < 10.0 and
            (not metrics.last_failure or 
             metrics.last_failure < datetime.utcnow() - timedelta(hours=1))
        )
        
        return ReceiverHealth(
            receiver_name=receiver_name,
            is_healthy=is_healthy,
            success_rate=success_rate,
            avg_response_time=metrics.avg_response_time,
            last_success=metrics.last_success,
            last_failure=metrics.last_failure,
            circuit_breaker_state=metrics.circuit_breaker_state
        )
        
    async def test_receiver(self, receiver_name: str) -> NotificationResult:
        """Test la connectivité d'un receiver"""
        if receiver_name not in self.receivers:
            raise ValueError(f"Unknown receiver: {receiver_name}")
            
        test_alert = AlertContext(
            alert_name="TestAlert",
            severity=AlertSeverity.INFO,
            tenant_id=self.tenant_id,
            labels={'test': 'true'},
            annotations={
                'summary': 'Test notification',
                'description': 'This is a test notification'
            }
        )
        
        return await self._send_to_receiver(receiver_name, test_alert)

class ReceiverTemplate:
    """
    Gestionnaire de templates pour les notifications.
    Support Jinja2 avec fonctions personnalisées.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = template_dir or "templates"
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Ajouter des fonctions personnalisées
        self.env.globals.update({
            'format_timestamp': self._format_timestamp,
            'severity_color': self._get_severity_color,
            'truncate_text': self._truncate_text,
            'join_labels': self._join_labels
        })
        
    def _format_timestamp(self, timestamp: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
        """Formate un timestamp"""
        return timestamp.strftime(format_str)
        
    def _get_severity_color(self, severity: str) -> str:
        """Retourne une couleur selon la sévérité"""
        colors = {
            'critical': '#FF0000',
            'high': '#FF8C00',
            'medium': '#FFD700',
            'low': '#32CD32',
            'info': '#4169E1'
        }
        return colors.get(severity.lower(), '#808080')
        
    def _truncate_text(self, text: str, max_length: int = 100) -> str:
        """Tronque un texte"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
        
    def _join_labels(self, labels: Dict[str, str], separator: str = ", ") -> str:
        """Joint les labels en une chaîne"""
        return separator.join([f"{k}={v}" for k, v in labels.items()])
        
    async def render(
        self,
        template_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Rend un template avec le contexte donné"""
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            raise TemplateRenderError(f"Failed to render template {template_name}: {e}")
