"""
Modèles de données Pydantic pour les receivers d'alertes Alertmanager.

Ces modèles définissent la structure des données pour la configuration
des receivers, les contextes d'alertes, et les résultats de notifications.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
import uuid
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import SecretStr, EmailStr
import json

class ChannelType(str, Enum):
    """Types de canaux de notification supportés"""
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    DISCORD = "discord"
    OPSGENIE = "opsgenie"
    VICTOROPS = "victorops"
    PUSHOVER = "pushover"
    TELEGRAM = "telegram"

class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class EscalationLevel(str, Enum):
    """Niveaux d'escalade"""
    IMMEDIATE = "immediate"
    AFTER_5MIN = "after_5min"
    AFTER_15MIN = "after_15min"
    AFTER_30MIN = "after_30min"
    AFTER_1HOUR = "after_1hour"

class NotificationStatus(str, Enum):
    """Statuts des notifications"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"
    ESCALATED = "escalated"
    THROTTLED = "throttled"

class ReceiverConfig(BaseModel):
    """Configuration d'un receiver d'alertes"""
    
    name: str = Field(..., description="Nom unique du receiver")
    channel_type: ChannelType = Field(..., description="Type de canal de notification")
    enabled: bool = Field(default=True, description="Receiver activé ou non")
    
    # Configuration spécifique au canal
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration du canal")
    
    # Filtres d'alertes
    min_severity: AlertSeverity = Field(default=AlertSeverity.INFO, description="Sévérité minimale")
    label_selectors: Dict[str, str] = Field(default_factory=dict, description="Sélecteurs de labels")
    annotation_selectors: Dict[str, str] = Field(default_factory=dict, description="Sélecteurs d'annotations")
    
    # Templates
    template_name: Optional[str] = Field(None, description="Nom du template à utiliser")
    custom_template: Optional[str] = Field(None, description="Template personnalisé")
    
    # Retry et timeouts
    max_retry_attempts: int = Field(default=3, ge=0, le=10, description="Nombre max de tentatives")
    timeout_seconds: int = Field(default=30, ge=5, le=300, description="Timeout en secondes")
    retry_delay_seconds: int = Field(default=5, ge=1, le=60, description="Délai entre les tentatives")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000, description="Limite de taux par minute")
    burst_limit: int = Field(default=10, ge=1, le=100, description="Limite de burst")
    
    # Escalade
    escalation_policy: Optional[str] = Field(None, description="Politique d'escalade")
    escalation_delay_minutes: int = Field(default=15, ge=0, le=1440, description="Délai d'escalade")
    
    # Métadonnées
    description: Optional[str] = Field(None, description="Description du receiver")
    tags: List[str] = Field(default_factory=list, description="Tags pour catégorisation")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('config')
    def validate_config(cls, v, values):
        """Valide la configuration selon le type de canal"""
        channel_type = values.get('channel_type')
        if not channel_type:
            return v
            
        # Validation spécifique par type
        required_fields = {
            ChannelType.SLACK: ['webhook_url'],
            ChannelType.EMAIL: ['smtp_server', 'recipients'],
            ChannelType.PAGERDUTY: ['integration_key'],
            ChannelType.WEBHOOK: ['url'],
            ChannelType.TEAMS: ['webhook_url'],
            ChannelType.DISCORD: ['webhook_url']
        }
        
        if channel_type in required_fields:
            for field in required_fields[channel_type]:
                if field not in v:
                    raise ValueError(f"Missing required field '{field}' for {channel_type}")
                    
        return v
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReceiverConfig':
        """Crée une instance depuis un dictionnaire"""
        return cls(**data)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return self.dict(exclude_none=True)
        
    def matches_alert(self, alert_context: 'AlertContext') -> bool:
        """Vérifie si ce receiver correspond à l'alerte donnée"""
        # Vérifier la sévérité minimale
        severity_order = {s.value: i for i, s in enumerate(AlertSeverity)}
        if severity_order[alert_context.severity] > severity_order[self.min_severity]:
            return False
            
        # Vérifier les sélecteurs de labels
        for key, value in self.label_selectors.items():
            if key not in alert_context.labels or alert_context.labels[key] != value:
                return False
                
        # Vérifier les sélecteurs d'annotations
        for key, value in self.annotation_selectors.items():
            if key not in alert_context.annotations or alert_context.annotations[key] != value:
                return False
                
        return True

class SlackReceiverConfig(BaseModel):
    """Configuration spécifique pour Slack"""
    webhook_url: SecretStr = Field(..., description="URL du webhook Slack")
    channel: Optional[str] = Field(None, description="Canal Slack (optionnel si dans l'URL)")
    username: Optional[str] = Field("AlertManager", description="Nom d'utilisateur du bot")
    icon_emoji: Optional[str] = Field(":warning:", description="Emoji d'icône")
    icon_url: Optional[str] = Field(None, description="URL de l'icône")
    
    # Formatage des messages
    link_names: bool = Field(default=True, description="Lier les noms d'utilisateurs")
    mrkdwn: bool = Field(default=True, description="Activer le markdown")
    
    # Mentions
    mention_users: List[str] = Field(default_factory=list, description="Utilisateurs à mentionner")
    mention_groups: List[str] = Field(default_factory=list, description="Groupes à mentionner")
    mention_here: bool = Field(default=False, description="Mentionner @here")
    mention_channel: bool = Field(default=False, description="Mentionner @channel")

class EmailReceiverConfig(BaseModel):
    """Configuration spécifique pour Email"""
    smtp_server: str = Field(..., description="Serveur SMTP")
    smtp_port: int = Field(default=587, description="Port SMTP")
    username: Optional[str] = Field(None, description="Nom d'utilisateur SMTP")
    password: Optional[SecretStr] = Field(None, description="Mot de passe SMTP")
    
    # Configuration TLS/SSL
    use_tls: bool = Field(default=True, description="Utiliser TLS")
    use_ssl: bool = Field(default=False, description="Utiliser SSL")
    
    # Destinataires
    recipients: List[EmailStr] = Field(..., description="Liste des destinataires")
    cc_recipients: List[EmailStr] = Field(default_factory=list, description="Destinataires en copie")
    bcc_recipients: List[EmailStr] = Field(default_factory=list, description="Destinataires en copie cachée")
    
    # Formatage
    from_address: EmailStr = Field(..., description="Adresse d'expéditeur")
    from_name: Optional[str] = Field("AlertManager", description="Nom d'expéditeur")
    subject_template: Optional[str] = Field(None, description="Template du sujet")
    html_template: bool = Field(default=True, description="Utiliser HTML")

class PagerDutyReceiverConfig(BaseModel):
    """Configuration spécifique pour PagerDuty"""
    integration_key: SecretStr = Field(..., description="Clé d'intégration PagerDuty")
    service_key: Optional[SecretStr] = Field(None, description="Clé de service (legacy)")
    
    # Configuration de l'événement
    event_action: Literal["trigger", "acknowledge", "resolve"] = Field(default="trigger")
    dedup_key: Optional[str] = Field(None, description="Clé de déduplication")
    
    # Métadonnées
    client: Optional[str] = Field("AlertManager", description="Nom du client")
    client_url: Optional[str] = Field(None, description="URL du client")
    
    # Détails personnalisés
    custom_details: Dict[str, Any] = Field(default_factory=dict, description="Détails personnalisés")
    
    # Images et liens
    images: List[Dict[str, str]] = Field(default_factory=list, description="Images à inclure")
    links: List[Dict[str, str]] = Field(default_factory=list, description="Liens à inclure")

class WebhookReceiverConfig(BaseModel):
    """Configuration spécifique pour Webhook"""
    url: str = Field(..., description="URL du webhook")
    method: Literal["POST", "PUT", "PATCH"] = Field(default="POST", description="Méthode HTTP")
    
    # Headers
    headers: Dict[str, str] = Field(default_factory=dict, description="Headers HTTP")
    
    # Authentification
    auth_type: Optional[Literal["basic", "bearer", "api_key"]] = Field(None)
    username: Optional[str] = Field(None, description="Nom d'utilisateur (auth basic)")
    password: Optional[SecretStr] = Field(None, description="Mot de passe (auth basic)")
    token: Optional[SecretStr] = Field(None, description="Token (auth bearer)")
    api_key: Optional[SecretStr] = Field(None, description="Clé API")
    api_key_header: Optional[str] = Field("X-API-Key", description="Header de la clé API")
    
    # Formatage du payload
    payload_template: Optional[str] = Field(None, description="Template du payload")
    content_type: str = Field(default="application/json", description="Type de contenu")
    
    # Retry spécifique
    retry_status_codes: List[int] = Field(default_factory=lambda: [500, 502, 503, 504])

class AlertContext(BaseModel):
    """Contexte d'une alerte"""
    
    # Identificateurs
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID unique de l'alerte")
    alert_name: str = Field(..., description="Nom de l'alerte")
    tenant_id: str = Field(..., description="ID du tenant")
    
    # Métadonnées de l'alerte
    severity: AlertSeverity = Field(..., description="Sévérité de l'alerte")
    status: Literal["firing", "resolved"] = Field(default="firing", description="Statut de l'alerte")
    
    # Labels et annotations Prometheus
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels de l'alerte")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Annotations de l'alerte")
    
    # Timestamps
    starts_at: datetime = Field(default_factory=datetime.utcnow, description="Début de l'alerte")
    ends_at: Optional[datetime] = Field(None, description="Fin de l'alerte")
    
    # URLs
    generator_url: Optional[str] = Field(None, description="URL du générateur")
    silence_url: Optional[str] = Field(None, description="URL pour faire taire l'alerte")
    dashboard_url: Optional[str] = Field(None, description="URL du dashboard")
    
    # Contexte d'escalade
    escalation_level: int = Field(default=0, description="Niveau d'escalade actuel")
    escalated_from: List[str] = Field(default_factory=list, description="Receivers précédents")
    
    # Fingerprint pour déduplication
    fingerprint: str = Field(default="", description="Empreinte pour déduplication")
    
    @validator('fingerprint', always=True)
    def generate_fingerprint(cls, v, values):
        """Génère une empreinte pour la déduplication"""
        if v:
            return v
            
        # Générer à partir des labels et du nom
        alert_name = values.get('alert_name', '')
        labels = values.get('labels', {})
        
        key_parts = [alert_name]
        for key in sorted(labels.keys()):
            key_parts.append(f"{key}={labels[key]}")
            
        import hashlib
        fingerprint_str = "|".join(key_parts)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    def is_critical(self) -> bool:
        """Vérifie si l'alerte est critique"""
        return self.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]
        
    def get_duration(self) -> Optional[timedelta]:
        """Retourne la durée de l'alerte"""
        if self.ends_at:
            return self.ends_at - self.starts_at
        return datetime.utcnow() - self.starts_at
        
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour templates"""
        return {
            'id': self.alert_id,
            'name': self.alert_name,
            'severity': self.severity.value,
            'status': self.status,
            'labels': self.labels,
            'annotations': self.annotations,
            'starts_at': self.starts_at,
            'ends_at': self.ends_at,
            'duration': str(self.get_duration()) if self.get_duration() else None,
            'generator_url': self.generator_url,
            'silence_url': self.silence_url,
            'dashboard_url': self.dashboard_url,
            'fingerprint': self.fingerprint
        }

class NotificationResult(BaseModel):
    """Résultat d'une notification"""
    
    success: bool = Field(..., description="Succès de la notification")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration: float = Field(default=0.0, description="Durée en secondes")
    
    # Détails de l'erreur
    error_message: Optional[str] = Field(None, description="Message d'erreur")
    error_code: Optional[str] = Field(None, description="Code d'erreur")
    
    # Réponse du service
    response_status: Optional[int] = Field(None, description="Code de statut HTTP")
    response_data: Optional[Dict[str, Any]] = Field(None, description="Données de réponse")
    
    # Métadonnées
    receiver_name: Optional[str] = Field(None, description="Nom du receiver")
    channel_type: Optional[ChannelType] = Field(None, description="Type de canal")
    retry_attempt: int = Field(default=0, description="Numéro de la tentative")
    
    def is_retryable(self) -> bool:
        """Détermine si l'erreur est retriable"""
        if self.success:
            return False
            
        # Codes d'erreur non retriables
        non_retryable_codes = {400, 401, 403, 404, 422}
        if self.response_status in non_retryable_codes:
            return False
            
        # Codes retriables
        retryable_codes = {429, 500, 502, 503, 504}
        if self.response_status in retryable_codes:
            return True
            
        # Erreurs de réseau (pas de code de statut)
        if self.response_status is None:
            return True
            
        return False

class EscalationPolicy(BaseModel):
    """Politique d'escalade des alertes"""
    
    name: str = Field(..., description="Nom de la politique")
    description: Optional[str] = Field(None, description="Description")
    
    # Critères de déclenchement
    severity_threshold: AlertSeverity = Field(default=AlertSeverity.HIGH)
    tenant_filter: List[str] = Field(default_factory=list, description="Tenants concernés")
    label_filters: Dict[str, str] = Field(default_factory=dict)
    
    # Configuration de l'escalade
    escalation_receivers: List[str] = Field(..., description="Receivers d'escalade")
    delay_seconds: int = Field(default=300, description="Délai avant escalade")
    max_escalations: int = Field(default=3, description="Nombre max d'escalades")
    
    # Horaires
    business_hours_only: bool = Field(default=False)
    timezone: str = Field(default="UTC")
    
    def applies_to_tenant(self, tenant_id: str) -> bool:
        """Vérifie si la politique s'applique au tenant"""
        if not self.tenant_filter:
            return True
        return tenant_id in self.tenant_filter
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EscalationPolicy':
        """Crée depuis un dictionnaire"""
        return cls(**data)

class NotificationChannel(BaseModel):
    """Canal de notification générique"""
    
    name: str = Field(..., description="Nom du canal")
    type: ChannelType = Field(..., description="Type de canal")
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # État du canal
    enabled: bool = Field(default=True)
    healthy: bool = Field(default=True)
    last_success: Optional[datetime] = Field(None)
    last_failure: Optional[datetime] = Field(None)
    
    # Métriques
    total_sent: int = Field(default=0)
    total_failed: int = Field(default=0)
    avg_response_time: float = Field(default=0.0)
    
    def get_success_rate(self) -> float:
        """Calcule le taux de succès"""
        total = self.total_sent + self.total_failed
        if total == 0:
            return 1.0
        return self.total_sent / total

class ReceiverHealth(BaseModel):
    """État de santé d'un receiver"""
    
    receiver_name: str = Field(..., description="Nom du receiver")
    is_healthy: bool = Field(..., description="État de santé")
    
    # Métriques de performance
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Taux de succès")
    avg_response_time: float = Field(..., ge=0.0, description="Temps de réponse moyen")
    
    # Historique
    last_success: Optional[datetime] = Field(None)
    last_failure: Optional[datetime] = Field(None)
    
    # État du circuit breaker
    circuit_breaker_state: str = Field(default="closed")
    
    # Score de santé global
    health_score: float = Field(..., ge=0.0, le=1.0, description="Score de santé")
    
    @validator('health_score', always=True)
    def calculate_health_score(cls, v, values):
        """Calcule le score de santé"""
        if v != 0.0:  # Si déjà défini
            return v
            
        success_rate = values.get('success_rate', 0.0)
        avg_response_time = values.get('avg_response_time', 0.0)
        
        # Score basé sur le taux de succès (70%) et le temps de réponse (30%)
        response_score = max(0, 1 - (avg_response_time / 30.0))  # 30s = score 0
        
        return (success_rate * 0.7) + (response_score * 0.3)

class AlertBatch(BaseModel):
    """Lot d'alertes pour traitement par batch"""
    
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alerts: List[AlertContext] = Field(..., description="Liste des alertes")
    tenant_id: str = Field(..., description="ID du tenant")
    
    # Configuration du batch
    max_batch_size: int = Field(default=10, description="Taille max du batch")
    batch_timeout_seconds: int = Field(default=30, description="Timeout du batch")
    
    # Métadonnées
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(None)
    
    def is_expired(self) -> bool:
        """Vérifie si le batch a expiré"""
        if self.processed_at:
            return False
            
        expiry_time = self.created_at + timedelta(seconds=self.batch_timeout_seconds)
        return datetime.utcnow() > expiry_time
        
    def get_severity_distribution(self) -> Dict[AlertSeverity, int]:
        """Retourne la distribution des sévérités"""
        distribution = {severity: 0 for severity in AlertSeverity}
        for alert in self.alerts:
            distribution[alert.severity] += 1
        return distribution
