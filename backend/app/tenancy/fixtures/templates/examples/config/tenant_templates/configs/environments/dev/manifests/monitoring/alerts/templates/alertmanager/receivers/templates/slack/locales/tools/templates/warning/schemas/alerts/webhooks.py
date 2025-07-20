"""
Schémas d'intégration webhook - Spotify AI Agent
Gestion avancée des webhooks entrants et sortants
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Literal
from uuid import UUID, uuid4
from enum import Enum
import json
import hmac
import hashlib
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator, computed_field, ConfigDict, HttpUrl

from . import (
    BaseSchema, TimestampMixin, TenantMixin, MetadataMixin,
    AlertLevel, AlertStatus, WarningCategory, Priority, Environment
)


class WebhookType(str, Enum):
    """Types de webhook"""
    INCOMING = "incoming"      # Webhook entrant (réception)
    OUTGOING = "outgoing"      # Webhook sortant (envoi)
    BIDIRECTIONAL = "bidirectional"  # Les deux


class WebhookEvent(str, Enum):
    """Événements de webhook"""
    ALERT_CREATED = "alert.created"
    ALERT_UPDATED = "alert.updated"
    ALERT_RESOLVED = "alert.resolved"
    ALERT_ESCALATED = "alert.escalated"
    ALERT_ACKNOWLEDGED = "alert.acknowledged"
    INCIDENT_CREATED = "incident.created"
    INCIDENT_UPDATED = "incident.updated"
    INCIDENT_RESOLVED = "incident.resolved"
    CORRELATION_DETECTED = "correlation.detected"
    NOTIFICATION_SENT = "notification.sent"
    WORKFLOW_EXECUTED = "workflow.executed"
    CUSTOM = "custom"


class WebhookStatus(str, Enum):
    """États de webhook"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"
    DEPRECATED = "deprecated"


class DeliveryStatus(str, Enum):
    """États de livraison"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRY = "retry"


class SecurityLevel(str, Enum):
    """Niveaux de sécurité"""
    NONE = "none"
    BASIC = "basic"
    HMAC = "hmac"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    MUTUAL_TLS = "mutual_tls"


class WebhookConfiguration(BaseModel):
    """Configuration de webhook"""
    
    # URL et méthode
    url: HttpUrl = Field(...)
    method: str = Field("POST", regex=r"^(GET|POST|PUT|PATCH|DELETE)$")
    
    # En-têtes HTTP
    headers: Dict[str, str] = Field(default_factory=dict)
    user_agent: str = Field("Spotify-AI-Agent-Webhook/1.0")
    
    # Authentification
    auth_type: str = Field("none")  # none, basic, bearer, api_key, oauth2
    auth_config: Dict[str, str] = Field(default_factory=dict)
    
    # Sécurité
    security_level: SecurityLevel = Field(SecurityLevel.HMAC)
    secret_key: Optional[str] = Field(None)
    signature_header: str = Field("X-Webhook-Signature")
    timestamp_header: str = Field("X-Webhook-Timestamp")
    
    # Timeouts et retry
    timeout_seconds: int = Field(30, ge=1, le=300)
    max_retries: int = Field(3, ge=0, le=10)
    retry_delay_seconds: int = Field(5, ge=1, le=3600)
    retry_exponential_backoff: bool = Field(True)
    
    # Formatage du payload
    payload_format: str = Field("json")  # json, xml, form
    custom_template: Optional[str] = Field(None)
    
    # Filtrage
    include_metadata: bool = Field(True)
    include_sensitive_data: bool = Field(False)
    field_filters: List[str] = Field(default_factory=list)
    
    @validator('url')
    def validate_url(cls, v):
        """Valide l'URL du webhook"""
        parsed = urlparse(str(v))
        if parsed.scheme not in ['http', 'https']:
            raise ValueError('URL must use http or https scheme')
        return v


class WebhookEndpoint(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Point de terminaison webhook"""
    
    # Informations de base
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    webhook_type: WebhookType = Field(...)
    
    # Configuration
    config: WebhookConfiguration = Field(...)
    
    # Événements écoutés/envoyés
    events: List[WebhookEvent] = Field(..., min_items=1)
    event_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Environnements
    environments: List[Environment] = Field(default_factory=list)
    
    # État et contrôle
    status: WebhookStatus = Field(WebhookStatus.ACTIVE)
    enabled: bool = Field(True)
    
    # Limitation de débit
    rate_limit_enabled: bool = Field(True)
    max_requests_per_minute: int = Field(60, ge=1, le=10000)
    max_requests_per_hour: int = Field(1000, ge=1, le=100000)
    
    # Santé et monitoring
    health_check_enabled: bool = Field(True)
    health_check_url: Optional[HttpUrl] = Field(None)
    health_check_interval_minutes: int = Field(15, ge=1, le=1440)
    last_health_check: Optional[datetime] = Field(None)
    health_status: str = Field("unknown")  # healthy, degraded, unhealthy, unknown
    
    # Métriques
    total_deliveries: int = Field(0, ge=0)
    successful_deliveries: int = Field(0, ge=0)
    failed_deliveries: int = Field(0, ge=0)
    last_delivery: Optional[datetime] = Field(None)
    avg_response_time_ms: Optional[float] = Field(None, ge=0)
    
    # Versioning
    version: str = Field("1.0.0")
    api_version: Optional[str] = Field(None)
    
    # Audit et sécurité
    created_by: Optional[UUID] = Field(None)
    last_modified_by: Optional[UUID] = Field(None)
    ip_whitelist: List[str] = Field(default_factory=list)
    
    # Tags et organisation
    tags: Set[str] = Field(default_factory=set)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    @computed_field
    @property
    def success_rate(self) -> float:
        """Taux de succès des livraisons"""
        if self.total_deliveries == 0:
            return 0.0
        return (self.successful_deliveries / self.total_deliveries) * 100

    @computed_field
    @property
    def is_healthy(self) -> bool:
        """Indique si le webhook est en bonne santé"""
        return (
            self.enabled and
            self.status == WebhookStatus.ACTIVE and
            self.health_status in ["healthy", "unknown"]
        )

    def generate_signature(self, payload: str, timestamp: Optional[str] = None) -> str:
        """Génère la signature HMAC pour le payload"""
        if not self.config.secret_key:
            return ""
        
        if timestamp is None:
            timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        
        # Créer la chaîne à signer
        sign_string = f"{timestamp}.{payload}"
        
        # Calculer la signature HMAC-SHA256
        signature = hmac.new(
            self.config.secret_key.encode('utf-8'),
            sign_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"

    def verify_signature(self, payload: str, signature: str, timestamp: str) -> bool:
        """Vérifie la signature d'un webhook entrant"""
        if not self.config.secret_key:
            return True  # Pas de vérification si pas de clé
        
        expected_signature = self.generate_signature(payload, timestamp)
        return hmac.compare_digest(signature, expected_signature)

    def format_payload(self, data: Dict[str, Any]) -> str:
        """Formate le payload selon la configuration"""
        # Filtrer les champs si nécessaire
        if self.config.field_filters:
            filtered_data = {
                k: v for k, v in data.items()
                if k in self.config.field_filters
            }
        else:
            filtered_data = data.copy()
        
        # Retirer les données sensibles si nécessaire
        if not self.config.include_sensitive_data:
            sensitive_fields = ['password', 'token', 'secret', 'key', 'credential']
            for field in sensitive_fields:
                if field in filtered_data:
                    filtered_data[field] = "***REDACTED***"
        
        # Ajouter les métadonnées si nécessaire
        if self.config.include_metadata:
            filtered_data['webhook_metadata'] = {
                'webhook_id': str(self.id),
                'webhook_name': self.name,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'version': self.version
            }
        
        # Formater selon le type demandé
        if self.config.payload_format == "json":
            return json.dumps(filtered_data, indent=2, default=str)
        elif self.config.payload_format == "xml":
            # Implémentation XML simplifiée
            return self._dict_to_xml(filtered_data)
        elif self.config.custom_template:
            # Utiliser un template personnalisé (Jinja2)
            import jinja2
            template = jinja2.Template(self.config.custom_template)
            return template.render(**filtered_data)
        else:
            return json.dumps(filtered_data, default=str)

    def _dict_to_xml(self, data: Dict[str, Any], root_tag: str = "webhook") -> str:
        """Convertit un dictionnaire en XML"""
        def _to_xml(obj, tag="item"):
            if isinstance(obj, dict):
                xml = f"<{tag}>"
                for k, v in obj.items():
                    xml += _to_xml(v, k)
                xml += f"</{tag}>"
                return xml
            elif isinstance(obj, list):
                xml = f"<{tag}>"
                for item in obj:
                    xml += _to_xml(item, "item")
                xml += f"</{tag}>"
                return xml
            else:
                return f"<{tag}>{str(obj)}</{tag}>"
        
        return f'<?xml version="1.0" encoding="UTF-8"?>{_to_xml(data, root_tag)}'


class WebhookDelivery(BaseSchema, TimestampMixin, TenantMixin):
    """Livraison de webhook"""
    
    delivery_id: UUID = Field(default_factory=uuid4)
    webhook_id: UUID = Field(...)
    event: WebhookEvent = Field(...)
    
    # Données de la requête
    payload: str = Field(...)
    headers: Dict[str, str] = Field(default_factory=dict)
    method: str = Field("POST")
    url: str = Field(...)
    
    # État de livraison
    status: DeliveryStatus = Field(DeliveryStatus.PENDING)
    attempt_count: int = Field(0, ge=0)
    max_attempts: int = Field(3, ge=1)
    
    # Temporisation
    scheduled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sent_at: Optional[datetime] = Field(None)
    delivered_at: Optional[datetime] = Field(None)
    next_retry_at: Optional[datetime] = Field(None)
    
    # Réponse
    response_status_code: Optional[int] = Field(None)
    response_headers: Dict[str, str] = Field(default_factory=dict)
    response_body: Optional[str] = Field(None)
    response_time_ms: Optional[float] = Field(None, ge=0)
    
    # Erreurs
    error_message: Optional[str] = Field(None)
    error_type: Optional[str] = Field(None)
    
    # Métadonnées
    alert_id: Optional[UUID] = Field(None)
    incident_id: Optional[UUID] = Field(None)
    correlation_id: Optional[UUID] = Field(None)
    
    @computed_field
    @property
    def is_successful(self) -> bool:
        """Indique si la livraison a réussi"""
        return (
            self.status == DeliveryStatus.DELIVERED and
            self.response_status_code is not None and
            200 <= self.response_status_code < 300
        )

    @computed_field
    @property
    def should_retry(self) -> bool:
        """Indique si une nouvelle tentative doit être faite"""
        return (
            self.status in [DeliveryStatus.FAILED, DeliveryStatus.TIMEOUT] and
            self.attempt_count < self.max_attempts and
            self.next_retry_at is not None and
            datetime.now(timezone.utc) >= self.next_retry_at
        )

    def mark_as_sent(self, response_code: int, response_body: Optional[str] = None,
                     response_time_ms: Optional[float] = None,
                     response_headers: Optional[Dict[str, str]] = None):
        """Marque la livraison comme envoyée"""
        self.sent_at = datetime.now(timezone.utc)
        self.response_status_code = response_code
        self.response_body = response_body
        self.response_time_ms = response_time_ms
        self.response_headers = response_headers or {}
        self.attempt_count += 1
        
        if 200 <= response_code < 300:
            self.status = DeliveryStatus.DELIVERED
            self.delivered_at = self.sent_at
        else:
            self.status = DeliveryStatus.FAILED
            self._schedule_retry()

    def mark_as_failed(self, error_message: str, error_type: Optional[str] = None):
        """Marque la livraison comme échouée"""
        self.status = DeliveryStatus.FAILED
        self.error_message = error_message
        self.error_type = error_type
        self.attempt_count += 1
        self._schedule_retry()

    def _schedule_retry(self):
        """Programme une nouvelle tentative"""
        if self.attempt_count < self.max_attempts:
            # Backoff exponentiel: 5s, 25s, 125s, etc.
            delay = 5 * (5 ** (self.attempt_count - 1))
            self.next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
            self.status = DeliveryStatus.RETRY


class WebhookSubscription(BaseSchema, TimestampMixin, TenantMixin):
    """Abonnement à des événements webhook"""
    
    subscription_id: UUID = Field(default_factory=uuid4)
    webhook_id: UUID = Field(...)
    
    # Événements souscrits
    subscribed_events: List[WebhookEvent] = Field(..., min_items=1)
    
    # Filtres d'événements
    event_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration
    active: bool = Field(True)
    batch_events: bool = Field(False)
    batch_size: int = Field(10, ge=1, le=1000)
    batch_timeout_seconds: int = Field(300, ge=1, le=3600)
    
    # Historique
    last_event_sent: Optional[datetime] = Field(None)
    total_events_sent: int = Field(0, ge=0)
    
    def matches_event(self, event: WebhookEvent, event_data: Dict[str, Any]) -> bool:
        """Vérifie si l'événement correspond aux filtres"""
        if event not in self.subscribed_events:
            return False
        
        # Appliquer les filtres
        for filter_key, filter_value in self.event_filters.items():
            if filter_key not in event_data:
                return False
            
            actual_value = event_data[filter_key]
            
            if isinstance(filter_value, list):
                if actual_value not in filter_value:
                    return False
            elif isinstance(filter_value, dict):
                # Filtres avancés (gt, lt, regex, etc.)
                operator = filter_value.get('operator', 'eq')
                expected = filter_value.get('value')
                
                if operator == 'eq' and actual_value != expected:
                    return False
                elif operator == 'gt' and actual_value <= expected:
                    return False
                elif operator == 'lt' and actual_value >= expected:
                    return False
                # Ajouter d'autres opérateurs selon les besoins
            else:
                if actual_value != filter_value:
                    return False
        
        return True


class WebhookMetrics(BaseSchema, TimestampMixin, TenantMixin):
    """Métriques de webhook"""
    
    metrics_id: UUID = Field(default_factory=uuid4)
    webhook_id: UUID = Field(...)
    
    # Période des métriques
    period_start: datetime = Field(...)
    period_end: datetime = Field(...)
    
    # Métriques de livraison
    total_deliveries: int = Field(0, ge=0)
    successful_deliveries: int = Field(0, ge=0)
    failed_deliveries: int = Field(0, ge=0)
    timeout_deliveries: int = Field(0, ge=0)
    
    # Métriques de performance
    avg_response_time_ms: Optional[float] = Field(None, ge=0)
    median_response_time_ms: Optional[float] = Field(None, ge=0)
    p95_response_time_ms: Optional[float] = Field(None, ge=0)
    p99_response_time_ms: Optional[float] = Field(None, ge=0)
    
    # Codes de réponse
    response_codes: Dict[str, int] = Field(default_factory=dict)
    
    # Métriques d'erreur
    error_types: Dict[str, int] = Field(default_factory=dict)
    retry_count: int = Field(0, ge=0)
    
    # Débit
    requests_per_minute: Optional[float] = Field(None, ge=0)
    requests_per_hour: Optional[float] = Field(None, ge=0)
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Taux de succès"""
        if self.total_deliveries == 0:
            return 0.0
        return (self.successful_deliveries / self.total_deliveries) * 100

    @computed_field
    @property
    def error_rate(self) -> float:
        """Taux d'erreur"""
        if self.total_deliveries == 0:
            return 0.0
        return (self.failed_deliveries / self.total_deliveries) * 100


__all__ = [
    'WebhookType', 'WebhookEvent', 'WebhookStatus', 'DeliveryStatus', 'SecurityLevel',
    'WebhookConfiguration', 'WebhookEndpoint', 'WebhookDelivery', 
    'WebhookSubscription', 'WebhookMetrics'
]
