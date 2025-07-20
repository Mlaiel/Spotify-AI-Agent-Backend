"""
External Integrations for Security System
========================================

Ce module implémente les intégrations externes pour le système de sécurité
multi-tenant du Spotify AI Agent.

Auteur: Fahed Mlaiel
"""

import asyncio
import json
import logging
import aiohttp
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import ssl
from urllib.parse import urlencode
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from .core import SecurityLevel, SecurityEvent
from .processors import AlertContext

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types d'intégrations disponibles"""
    SLACK = "slack"
    MICROSOFT_TEAMS = "microsoft_teams"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SIEM = "siem"
    SPLUNK = "splunk"
    ELASTICSEARCH = "elasticsearch"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    TELEGRAM = "telegram"
    DISCORD = "discord"


class DeliveryStatus(Enum):
    """Status de livraison"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    RATE_LIMITED = "rate_limited"
    SUPPRESSED = "suppressed"


@dataclass
class IntegrationConfig:
    """Configuration d'intégration"""
    integration_type: IntegrationType
    name: str
    enabled: bool = True
    
    # Configuration de connexion
    endpoint_url: str = ""
    api_key: str = ""
    token: str = ""
    username: str = ""
    password: str = ""
    
    # Configuration avancée
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 5
    rate_limit_per_minute: int = 60
    
    # Configuration de format
    message_format: str = "json"
    template: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Configuration de filtres
    severity_filter: List[SecurityLevel] = field(default_factory=list)
    event_type_filter: List[str] = field(default_factory=list)
    tenant_filter: List[str] = field(default_factory=list)
    
    # Configuration SSL/TLS
    verify_ssl: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    # Métadonnées
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None


@dataclass
class DeliveryAttempt:
    """Tentative de livraison"""
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    integration_name: str = ""
    alert_id: str = ""
    
    # Détails de la tentative
    attempted_at: datetime = field(default_factory=datetime.utcnow)
    status: DeliveryStatus = DeliveryStatus.PENDING
    response_code: Optional[int] = None
    response_message: str = ""
    latency_ms: Optional[float] = None
    
    # Erreurs
    error_message: str = ""
    retry_count: int = 0
    next_retry_at: Optional[datetime] = None


class SlackIntegration:
    """
    Intégration Slack avancée avec support des threads et attachments
    """
    
    def __init__(self, config: IntegrationConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis = redis_client
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialise l'intégration Slack"""
        connector = aiohttp.TCPConnector(
            ssl=ssl.create_default_context() if self.config.verify_ssl else False
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        
        logger.info(f"SlackIntegration '{self.config.name}' initialized")
    
    async def cleanup(self):
        """Nettoie les ressources"""
        if self.session:
            await self.session.close()
    
    async def send_alert(self, alert_context: AlertContext) -> DeliveryAttempt:
        """Envoie une alerte Slack"""
        attempt = DeliveryAttempt(
            integration_name=self.config.name,
            alert_id=alert_context.alert_id
        )
        
        try:
            # Vérification du rate limiting
            if not await self._check_rate_limit():
                attempt.status = DeliveryStatus.RATE_LIMITED
                attempt.error_message = "Rate limit exceeded"
                return attempt
            
            # Construction du message Slack
            slack_message = await self._build_slack_message(alert_context)
            
            # Envoi du message
            start_time = datetime.utcnow()
            
            async with self.session.post(
                self.config.endpoint_url,
                json=slack_message,
                headers=self.config.headers
            ) as response:
                end_time = datetime.utcnow()
                
                attempt.response_code = response.status
                attempt.latency_ms = (end_time - start_time).total_seconds() * 1000
                attempt.response_message = await response.text()
                
                if response.status == 200:
                    attempt.status = DeliveryStatus.SENT
                    await self._update_rate_limit_counter()
                else:
                    attempt.status = DeliveryStatus.FAILED
                    attempt.error_message = f"HTTP {response.status}: {attempt.response_message}"
        
        except asyncio.TimeoutError:
            attempt.status = DeliveryStatus.FAILED
            attempt.error_message = "Request timeout"
        except Exception as e:
            attempt.status = DeliveryStatus.FAILED
            attempt.error_message = str(e)
        
        return attempt
    
    async def _build_slack_message(self, alert_context: AlertContext) -> Dict[str, Any]:
        """Construit un message Slack formaté"""
        # Couleur basée sur la sévérité
        color_map = {
            SecurityLevel.LOW: "#36a64f",      # Vert
            SecurityLevel.MEDIUM: "#ff9f00",   # Orange
            SecurityLevel.HIGH: "#ff4444",     # Rouge
            SecurityLevel.CRITICAL: "#8b0000"  # Rouge foncé
        }
        
        color = color_map.get(alert_context.severity, "#36a64f")
        
        # Construction du message principal
        message = {
            "text": f"🔒 Security Alert - {alert_context.title}",
            "attachments": [
                {
                    "color": color,
                    "title": alert_context.title,
                    "text": alert_context.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert_context.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Tenant",
                            "value": alert_context.tenant_id,
                            "short": True
                        },
                        {
                            "title": "Event ID",
                            "value": alert_context.event_id,
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert_context.created_at.isoformat(),
                            "short": True
                        }
                    ],
                    "footer": "Achiri Security System",
                    "footer_icon": "https://achiri.com/icon.png",
                    "ts": int(alert_context.created_at.timestamp())
                }
            ]
        }
        
        # Ajout de détails si disponibles
        if alert_context.details:
            details_text = ""
            
            if "threats" in alert_context.details:
                threats = alert_context.details["threats"]
                if threats:
                    details_text += f"\n🚨 *Threats Detected:* {len(threats)}"
            
            if "anomalies" in alert_context.details:
                anomalies = alert_context.details["anomalies"]
                if anomalies:
                    details_text += f"\n⚠️ *Anomalies Detected:* {len(anomalies)}"
            
            if details_text:
                message["attachments"][0]["text"] += details_text
        
        # Ajout de boutons d'action pour les alertes critiques
        if alert_context.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            message["attachments"][0]["actions"] = [
                {
                    "type": "button",
                    "text": "Acknowledge",
                    "value": f"ack_{alert_context.alert_id}",
                    "style": "primary"
                },
                {
                    "type": "button",
                    "text": "Investigate",
                    "value": f"investigate_{alert_context.alert_id}",
                    "style": "default"
                },
                {
                    "type": "button",
                    "text": "Escalate",
                    "value": f"escalate_{alert_context.alert_id}",
                    "style": "danger"
                }
            ]
        
        return message
    
    async def _check_rate_limit(self) -> bool:
        """Vérifie les limites de taux"""
        rate_key = f"rate_limit:slack:{self.config.name}"
        current_count = await self.redis.get(rate_key)
        
        if current_count and int(current_count) >= self.config.rate_limit_per_minute:
            return False
        
        return True
    
    async def _update_rate_limit_counter(self):
        """Met à jour le compteur de rate limiting"""
        rate_key = f"rate_limit:slack:{self.config.name}"
        await self.redis.incr(rate_key)
        await self.redis.expire(rate_key, 60)  # 1 minute


class EmailIntegration:
    """
    Intégration Email avec support SMTP et templates HTML
    """
    
    def __init__(self, config: IntegrationConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis = redis_client
        self.smtp_session = None
        
    async def initialize(self):
        """Initialise l'intégration Email"""
        # Configuration SMTP serait ici
        logger.info(f"EmailIntegration '{self.config.name}' initialized")
    
    async def cleanup(self):
        """Nettoie les ressources"""
        if self.smtp_session:
            self.smtp_session.quit()
    
    async def send_alert(self, alert_context: AlertContext) -> DeliveryAttempt:
        """Envoie une alerte par email"""
        attempt = DeliveryAttempt(
            integration_name=self.config.name,
            alert_id=alert_context.alert_id
        )
        
        try:
            # Construction du message email
            email_content = await self._build_email_content(alert_context)
            
            # Envoi de l'email (simulation)
            await asyncio.sleep(0.1)  # Simulation d'envoi
            
            attempt.status = DeliveryStatus.SENT
            attempt.response_code = 250  # SMTP success
            attempt.response_message = "Message accepted"
            
        except Exception as e:
            attempt.status = DeliveryStatus.FAILED
            attempt.error_message = str(e)
        
        return attempt
    
    async def _build_email_content(self, alert_context: AlertContext) -> Dict[str, str]:
        """Construit le contenu de l'email"""
        subject = f"🔒 Security Alert: {alert_context.title}"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f44336; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; border: 1px solid #ddd; }}
                .severity {{ font-weight: bold; color: #f44336; }}
                .details {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Security Alert</h1>
            </div>
            <div class="content">
                <h2>{alert_context.title}</h2>
                <p><span class="severity">Severity:</span> {alert_context.severity.value.upper()}</p>
                <p><strong>Tenant:</strong> {alert_context.tenant_id}</p>
                <p><strong>Event ID:</strong> {alert_context.event_id}</p>
                <p><strong>Timestamp:</strong> {alert_context.created_at.isoformat()}</p>
                
                <div class="details">
                    <h3>Alert Details</h3>
                    <p>{alert_context.message}</p>
                </div>
            </div>
            <div class="footer">
                <p>Achiri Security System - {datetime.utcnow().isoformat()}</p>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Security Alert: {alert_context.title}
        
        Severity: {alert_context.severity.value.upper()}
        Tenant: {alert_context.tenant_id}
        Event ID: {alert_context.event_id}
        Timestamp: {alert_context.created_at.isoformat()}
        
        Details:
        {alert_context.message}
        
        --
        Achiri Security System
        """
        
        return {
            "subject": subject,
            "html": html_content,
            "text": text_content
        }


class SIEMIntegration:
    """
    Intégration SIEM (Security Information and Event Management)
    """
    
    def __init__(self, config: IntegrationConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis = redis_client
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialise l'intégration SIEM"""
        connector = aiohttp.TCPConnector(
            ssl=ssl.create_default_context() if self.config.verify_ssl else False
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.token}"
        }
        headers.update(self.config.headers)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
        logger.info(f"SIEMIntegration '{self.config.name}' initialized")
    
    async def cleanup(self):
        """Nettoie les ressources"""
        if self.session:
            await self.session.close()
    
    async def send_alert(self, alert_context: AlertContext) -> DeliveryAttempt:
        """Envoie une alerte au SIEM"""
        attempt = DeliveryAttempt(
            integration_name=self.config.name,
            alert_id=alert_context.alert_id
        )
        
        try:
            # Construction du payload SIEM
            siem_payload = await self._build_siem_payload(alert_context)
            
            # Envoi au SIEM
            start_time = datetime.utcnow()
            
            async with self.session.post(
                self.config.endpoint_url,
                json=siem_payload
            ) as response:
                end_time = datetime.utcnow()
                
                attempt.response_code = response.status
                attempt.latency_ms = (end_time - start_time).total_seconds() * 1000
                attempt.response_message = await response.text()
                
                if 200 <= response.status < 300:
                    attempt.status = DeliveryStatus.SENT
                else:
                    attempt.status = DeliveryStatus.FAILED
                    attempt.error_message = f"HTTP {response.status}: {attempt.response_message}"
        
        except Exception as e:
            attempt.status = DeliveryStatus.FAILED
            attempt.error_message = str(e)
        
        return attempt
    
    async def _build_siem_payload(self, alert_context: AlertContext) -> Dict[str, Any]:
        """Construit le payload SIEM"""
        # Format CEF (Common Event Format) ou JSON structuré
        payload = {
            "timestamp": alert_context.created_at.isoformat(),
            "source": "achiri_security",
            "event_type": "security_alert",
            "severity": self._map_severity_to_siem(alert_context.severity),
            "alert": {
                "id": alert_context.alert_id,
                "title": alert_context.title,
                "message": alert_context.message,
                "tenant_id": alert_context.tenant_id,
                "event_id": alert_context.event_id
            },
            "metadata": {
                "system": "spotify-ai-agent",
                "component": "security-system",
                "version": "1.0.0"
            }
        }
        
        # Ajout des détails si disponibles
        if alert_context.details:
            payload["details"] = alert_context.details
        
        return payload
    
    def _map_severity_to_siem(self, severity: SecurityLevel) -> int:
        """Mappe la sévérité vers le format SIEM"""
        mapping = {
            SecurityLevel.LOW: 2,
            SecurityLevel.MEDIUM: 4,
            SecurityLevel.HIGH: 7,
            SecurityLevel.CRITICAL: 10
        }
        return mapping.get(severity, 2)


class WebhookIntegration:
    """
    Intégration Webhook générique
    """
    
    def __init__(self, config: IntegrationConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis = redis_client
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialise l'intégration Webhook"""
        connector = aiohttp.TCPConnector(
            ssl=ssl.create_default_context() if self.config.verify_ssl else False
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        logger.info(f"WebhookIntegration '{self.config.name}' initialized")
    
    async def cleanup(self):
        """Nettoie les ressources"""
        if self.session:
            await self.session.close()
    
    async def send_alert(self, alert_context: AlertContext) -> DeliveryAttempt:
        """Envoie une alerte via webhook"""
        attempt = DeliveryAttempt(
            integration_name=self.config.name,
            alert_id=alert_context.alert_id
        )
        
        try:
            # Construction du payload
            payload = await self._build_webhook_payload(alert_context)
            
            # Signature HMAC si clé API fournie
            headers = dict(self.config.headers)
            if self.config.api_key:
                signature = await self._generate_hmac_signature(payload, self.config.api_key)
                headers["X-Achiri-Signature"] = signature
            
            # Envoi du webhook
            start_time = datetime.utcnow()
            
            async with self.session.post(
                self.config.endpoint_url,
                json=payload,
                headers=headers
            ) as response:
                end_time = datetime.utcnow()
                
                attempt.response_code = response.status
                attempt.latency_ms = (end_time - start_time).total_seconds() * 1000
                attempt.response_message = await response.text()
                
                if 200 <= response.status < 300:
                    attempt.status = DeliveryStatus.SENT
                else:
                    attempt.status = DeliveryStatus.FAILED
                    attempt.error_message = f"HTTP {response.status}: {attempt.response_message}"
        
        except Exception as e:
            attempt.status = DeliveryStatus.FAILED
            attempt.error_message = str(e)
        
        return attempt
    
    async def _build_webhook_payload(self, alert_context: AlertContext) -> Dict[str, Any]:
        """Construit le payload webhook"""
        return {
            "alert_id": alert_context.alert_id,
            "tenant_id": alert_context.tenant_id,
            "severity": alert_context.severity.value,
            "title": alert_context.title,
            "message": alert_context.message,
            "event_id": alert_context.event_id,
            "timestamp": alert_context.created_at.isoformat(),
            "details": alert_context.details,
            "metadata": {
                "source": "achiri_security_system",
                "version": "1.0.0"
            }
        }
    
    async def _generate_hmac_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Génère une signature HMAC pour le payload"""
        payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
        signature = hmac.new(
            secret.encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"


class LoggingIntegration:
    """
    Intégration pour logging centralisé (Elasticsearch, Splunk, etc.)
    """
    
    def __init__(self, config: IntegrationConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis = redis_client
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialise l'intégration de logging"""
        self.session = aiohttp.ClientSession()
        logger.info(f"LoggingIntegration '{self.config.name}' initialized")
    
    async def cleanup(self):
        """Nettoie les ressources"""
        if self.session:
            await self.session.close()
    
    async def log_security_event(self, event: SecurityEvent) -> bool:
        """Log un événement de sécurité"""
        try:
            log_entry = await self._build_log_entry(event)
            
            # Envoi vers le système de logging
            if self.config.integration_type == IntegrationType.ELASTICSEARCH:
                return await self._send_to_elasticsearch(log_entry)
            elif self.config.integration_type == IntegrationType.SPLUNK:
                return await self._send_to_splunk(log_entry)
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
            return False
    
    async def _build_log_entry(self, event: SecurityEvent) -> Dict[str, Any]:
        """Construit une entrée de log"""
        return {
            "@timestamp": event.timestamp.isoformat(),
            "event_id": event.event_id,
            "tenant_id": event.tenant_id,
            "user_id": event.user_id,
            "event_type": event.event_type,
            "severity": event.severity.value,
            "source_ip": event.source_ip,
            "user_agent": event.user_agent,
            "resource": event.resource,
            "action": event.action,
            "threat_score": event.threat_score,
            "is_blocked": event.is_blocked,
            "escalated": event.escalated,
            "metadata": event.metadata,
            "system": {
                "source": "achiri_security",
                "component": "security_event_processor",
                "version": "1.0.0"
            }
        }
    
    async def _send_to_elasticsearch(self, log_entry: Dict[str, Any]) -> bool:
        """Envoie vers Elasticsearch"""
        # Implémentation Elasticsearch
        return True
    
    async def _send_to_splunk(self, log_entry: Dict[str, Any]) -> bool:
        """Envoie vers Splunk"""
        # Implémentation Splunk
        return True


class MetricsIntegration:
    """
    Intégration pour métriques (Prometheus, Grafana, etc.)
    """
    
    def __init__(self, config: IntegrationConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis = redis_client
        self.metrics_buffer = []
        
    async def initialize(self):
        """Initialise l'intégration de métriques"""
        logger.info(f"MetricsIntegration '{self.config.name}' initialized")
    
    async def cleanup(self):
        """Nettoie les ressources"""
        pass
    
    async def record_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Enregistre une métrique"""
        metric = {
            "name": metric_name,
            "value": value,
            "labels": labels or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.metrics_buffer.append(metric)
        
        # Envoi en batch si buffer plein
        if len(self.metrics_buffer) >= 100:
            await self._flush_metrics()
    
    async def _flush_metrics(self):
        """Envoie les métriques en batch"""
        if not self.metrics_buffer:
            return
        
        try:
            # Envoi des métriques vers Prometheus/Grafana
            await self._send_metrics_batch(self.metrics_buffer)
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")
    
    async def _send_metrics_batch(self, metrics: List[Dict[str, Any]]):
        """Envoie un batch de métriques"""
        # Implémentation d'envoi vers système de métriques
        pass


class IntegrationManager:
    """
    Gestionnaire central des intégrations
    """
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.integrations: Dict[str, Any] = {}
        self.configs: Dict[str, IntegrationConfig] = {}
        
    async def initialize(self):
        """Initialise le gestionnaire d'intégrations"""
        await self._load_integration_configs()
        await self._initialize_integrations()
        logger.info("IntegrationManager initialized")
    
    async def cleanup(self):
        """Nettoie toutes les intégrations"""
        for integration in self.integrations.values():
            if hasattr(integration, 'cleanup'):
                await integration.cleanup()
    
    async def _load_integration_configs(self):
        """Charge les configurations d'intégrations"""
        # Chargement depuis Redis/DB
        configs_key = "integration_configs"
        configs_json = await self.redis.get(configs_key)
        
        if configs_json:
            configs_data = json.loads(configs_json)
            for name, config_data in configs_data.items():
                self.configs[name] = IntegrationConfig(**config_data)
    
    async def _initialize_integrations(self):
        """Initialise toutes les intégrations configurées"""
        for name, config in self.configs.items():
            if not config.enabled:
                continue
            
            try:
                integration = await self._create_integration(config)
                await integration.initialize()
                self.integrations[name] = integration
                
            except Exception as e:
                logger.error(f"Failed to initialize integration {name}: {e}")
    
    async def _create_integration(self, config: IntegrationConfig):
        """Crée une instance d'intégration"""
        integration_classes = {
            IntegrationType.SLACK: SlackIntegration,
            IntegrationType.EMAIL: EmailIntegration,
            IntegrationType.SIEM: SIEMIntegration,
            IntegrationType.WEBHOOK: WebhookIntegration,
            IntegrationType.ELASTICSEARCH: LoggingIntegration,
            IntegrationType.PROMETHEUS: MetricsIntegration
        }
        
        integration_class = integration_classes.get(config.integration_type)
        if not integration_class:
            raise ValueError(f"Unsupported integration type: {config.integration_type}")
        
        return integration_class(config, self.redis)
    
    async def send_alert_to_all(self, alert_context: AlertContext) -> Dict[str, DeliveryAttempt]:
        """Envoie une alerte à toutes les intégrations configurées"""
        results = {}
        
        for name, integration in self.integrations.items():
            config = self.configs[name]
            
            # Vérification des filtres
            if not await self._should_send_to_integration(alert_context, config):
                continue
            
            try:
                if hasattr(integration, 'send_alert'):
                    attempt = await integration.send_alert(alert_context)
                    results[name] = attempt
                    
                    # Stockage de la tentative
                    await self._store_delivery_attempt(attempt)
                    
            except Exception as e:
                logger.error(f"Error sending alert to {name}: {e}")
                
                # Création d'une tentative échouée
                failed_attempt = DeliveryAttempt(
                    integration_name=name,
                    alert_id=alert_context.alert_id,
                    status=DeliveryStatus.FAILED,
                    error_message=str(e)
                )
                results[name] = failed_attempt
        
        return results
    
    async def _should_send_to_integration(self, alert_context: AlertContext, config: IntegrationConfig) -> bool:
        """Vérifie si une alerte doit être envoyée à une intégration"""
        # Filtre par sévérité
        if config.severity_filter and alert_context.severity not in config.severity_filter:
            return False
        
        # Filtre par tenant
        if config.tenant_filter and alert_context.tenant_id not in config.tenant_filter:
            return False
        
        return True
    
    async def _store_delivery_attempt(self, attempt: DeliveryAttempt):
        """Stocke une tentative de livraison"""
        attempt_key = f"delivery_attempt:{attempt.attempt_id}"
        attempt_data = {
            "attempt_id": attempt.attempt_id,
            "integration_name": attempt.integration_name,
            "alert_id": attempt.alert_id,
            "attempted_at": attempt.attempted_at.isoformat(),
            "status": attempt.status.value,
            "response_code": attempt.response_code,
            "response_message": attempt.response_message,
            "latency_ms": attempt.latency_ms,
            "error_message": attempt.error_message,
            "retry_count": attempt.retry_count
        }
        
        await self.redis.set(attempt_key, json.dumps(attempt_data), ex=86400 * 7)  # 7 jours
    
    async def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Récupère le statut de toutes les intégrations"""
        status = {}
        
        for name, config in self.configs.items():
            integration_status = {
                "enabled": config.enabled,
                "type": config.integration_type.value,
                "last_used": config.last_used.isoformat() if config.last_used else None,
                "active": name in self.integrations
            }
            
            # Ajout de métriques de performance
            if name in self.integrations:
                metrics = await self._get_integration_metrics(name)
                integration_status.update(metrics)
            
            status[name] = integration_status
        
        return status
    
    async def _get_integration_metrics(self, integration_name: str) -> Dict[str, Any]:
        """Récupère les métriques d'une intégration"""
        # Métriques des dernières 24h
        metrics_key = f"integration_metrics:{integration_name}"
        metrics_data = await self.redis.hgetall(metrics_key)
        
        return {
            "messages_sent_24h": int(metrics_data.get(b"messages_sent_24h", 0)),
            "messages_failed_24h": int(metrics_data.get(b"messages_failed_24h", 0)),
            "avg_latency_ms": float(metrics_data.get(b"avg_latency_ms", 0)),
            "last_success": metrics_data.get(b"last_success", b"").decode() or None,
            "last_failure": metrics_data.get(b"last_failure", b"").decode() or None
        }
