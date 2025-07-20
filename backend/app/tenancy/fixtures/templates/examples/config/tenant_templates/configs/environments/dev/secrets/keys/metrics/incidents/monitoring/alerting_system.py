# =============================================================================
# Monitoring & Observability - Alerting & Notification System Enterprise
# =============================================================================
# 
# Système d'alerting et de notification enterprise avec gestion intelligente,
# escalades automatiques et intégrations multi-canaux.
#
# Développé par l'équipe d'experts techniques:
# - Lead Developer + AI Architect (Architecture alerting et intelligence)
# - DevOps Senior Engineer (Intégrations et infrastructure)
# - Notification Specialist (Canaux et routing intelligent)
# - Reliability Engineer (SRE practices et escalades)
#
# Direction Technique: Fahed Mlaiel
# =============================================================================

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import time
import hashlib
import hmac
from pathlib import Path
import yaml
import re

# Imports pour notifications
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import aiohttp
import aioredis
from jinja2 import Template, Environment, FileSystemLoader

# Imports pour intégrations
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
import phonenumbers
from phonenumbers import NumberParseException

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MODÈLES D'ALERTING ET NOTIFICATION
# =============================================================================

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Statuts des alertes"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"

class NotificationType(Enum):
    """Types de notifications"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"
    TEAMS = "teams"
    TELEGRAM = "telegram"

class EscalationAction(Enum):
    """Actions d'escalade"""
    NOTIFY_TEAM = "notify_team"
    NOTIFY_MANAGER = "notify_manager"
    CREATE_INCIDENT = "create_incident"
    CALL_ONCALL = "call_oncall"
    INVOKE_RUNBOOK = "invoke_runbook"
    AUTO_REMEDIATION = "auto_remediation"

@dataclass
class ContactInfo:
    """Informations de contact"""
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    slack_user_id: Optional[str] = None
    telegram_user_id: Optional[str] = None
    timezone: str = "UTC"
    preferred_methods: List[NotificationType] = field(default_factory=list)

@dataclass
class NotificationRule:
    """Règle de notification"""
    id: str
    name: str
    enabled: bool = True
    filters: Dict[str, Any] = field(default_factory=dict)  # severity, tags, time_of_day
    channels: List[str] = field(default_factory=list)
    delay_seconds: int = 0
    repeat_interval_seconds: Optional[int] = None
    max_repeats: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EscalationLevel:
    """Niveau d'escalade"""
    level: int
    delay_minutes: int
    actions: List[EscalationAction]
    contacts: List[str] = field(default_factory=list)
    teams: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EscalationPolicy:
    """Politique d'escalade"""
    id: str
    name: str
    description: str
    levels: List[EscalationLevel]
    enabled: bool = True
    auto_resolve: bool = True
    auto_resolve_timeout_minutes: int = 60

@dataclass
class Alert:
    """Alerte système"""
    id: str
    rule_name: str
    status: AlertStatus
    severity: AlertSeverity
    title: str
    description: str
    started_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    tenant_id: str = ""
    source: str = ""
    fingerprint: str = ""
    escalation_policy_id: Optional[str] = None
    current_escalation_level: int = 0
    notification_count: int = 0
    last_notification_at: Optional[datetime] = None

@dataclass
class NotificationChannel:
    """Canal de notification"""
    id: str
    name: str
    type: NotificationType
    config: Dict[str, Any]
    enabled: bool = True
    rate_limit: Dict[str, int] = field(default_factory=dict)  # max_per_hour, max_per_day
    template_config: Dict[str, str] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NotificationMessage:
    """Message de notification"""
    id: str
    alert_id: str
    channel_id: str
    recipient: str
    subject: str
    content: str
    created_at: datetime
    sent_at: Optional[datetime] = None
    delivery_status: str = "pending"  # pending, sent, failed, delivered
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# GESTIONNAIRE D'ALERTING INTELLIGENT
# =============================================================================

class IntelligentAlertManager:
    """
    Gestionnaire d'alerting intelligent avec déduplication, groupement,
    filtrage contextuel et routing intelligent.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_rules: Dict[str, NotificationRule] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        self.contacts: Dict[str, ContactInfo] = {}
        
        # Connexions
        self.redis_client: Optional[aioredis.Redis] = None
        self.db_engine = None
        
        # Configuration
        self.deduplication_window = config.get('deduplication_window_minutes', 5)
        self.grouping_interval = config.get('grouping_interval_seconds', 30)
        self.max_alerts_per_group = config.get('max_alerts_per_group', 10)
        
        # État interne
        self.alert_groups: Dict[str, List[str]] = {}
        self.pending_notifications: List[NotificationMessage] = []
        
        logger.info("IntelligentAlertManager initialisé")

    async def initialize(self):
        """Initialisation du gestionnaire d'alerting"""
        try:
            # Connexions
            self.redis_client = aioredis.from_url(
                self.config['redis_url'],
                encoding='utf-8',
                decode_responses=True
            )
            
            self.db_engine = create_async_engine(self.config['database_url'])
            
            # Chargement de la configuration
            await self.load_notification_rules()
            await self.load_escalation_policies()
            await self.load_contacts()
            
            # Démarrage des tâches en arrière-plan
            asyncio.create_task(self.process_notification_queue())
            asyncio.create_task(self.process_escalations())
            asyncio.create_task(self.process_alert_grouping())
            
            logger.info("Gestionnaire d'alerting intelligent initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation AlertManager: {e}")
            raise

    async def load_notification_rules(self):
        """Chargement des règles de notification"""
        
        # Règles par défaut
        default_rules = [
            NotificationRule(
                id="critical_immediate",
                name="Critical Alerts - Immediate",
                filters={"severity": ["critical", "emergency"]},
                channels=["email_ops", "slack_alerts", "pagerduty_oncall"],
                delay_seconds=0,
                repeat_interval_seconds=300,  # 5 minutes
                max_repeats=10
            ),
            NotificationRule(
                id="warning_delayed",
                name="Warning Alerts - Delayed",
                filters={"severity": ["warning"]},
                channels=["email_ops", "slack_alerts"],
                delay_seconds=300,  # 5 minutes
                repeat_interval_seconds=900,  # 15 minutes
                max_repeats=3
            ),
            NotificationRule(
                id="business_hours_only",
                name="Info Alerts - Business Hours",
                filters={"severity": ["info"]},
                channels=["email_ops"],
                delay_seconds=0,
                conditions={"business_hours_only": True}
            ),
            NotificationRule(
                id="tenant_specific",
                name="Tenant Specific Alerts",
                filters={"tags": ["tenant:premium"]},
                channels=["email_premium_support", "slack_premium"],
                delay_seconds=0,
                repeat_interval_seconds=600,
                max_repeats=5
            )
        ]
        
        for rule in default_rules:
            self.notification_rules[rule.id] = rule
        
        logger.info(f"Chargé {len(self.notification_rules)} règles de notification")

    async def load_escalation_policies(self):
        """Chargement des politiques d'escalade"""
        
        # Politiques par défaut
        default_policies = [
            EscalationPolicy(
                id="standard_escalation",
                name="Standard Escalation Policy",
                description="Standard escalation for most alerts",
                levels=[
                    EscalationLevel(
                        level=1,
                        delay_minutes=0,
                        actions=[EscalationAction.NOTIFY_TEAM],
                        contacts=["oncall_primary"],
                        teams=["ops_team"]
                    ),
                    EscalationLevel(
                        level=2,
                        delay_minutes=15,
                        actions=[EscalationAction.NOTIFY_MANAGER],
                        contacts=["oncall_secondary", "team_lead"]
                    ),
                    EscalationLevel(
                        level=3,
                        delay_minutes=30,
                        actions=[EscalationAction.CREATE_INCIDENT, EscalationAction.CALL_ONCALL],
                        contacts=["incident_commander", "director_ops"]
                    )
                ]
            ),
            EscalationPolicy(
                id="critical_escalation",
                name="Critical Issue Escalation",
                description="Fast escalation for critical issues",
                levels=[
                    EscalationLevel(
                        level=1,
                        delay_minutes=0,
                        actions=[EscalationAction.NOTIFY_TEAM, EscalationAction.CREATE_INCIDENT],
                        contacts=["oncall_primary", "oncall_secondary"]
                    ),
                    EscalationLevel(
                        level=2,
                        delay_minutes=5,
                        actions=[EscalationAction.CALL_ONCALL, EscalationAction.NOTIFY_MANAGER],
                        contacts=["team_lead", "director_ops"]
                    ),
                    EscalationLevel(
                        level=3,
                        delay_minutes=10,
                        actions=[EscalationAction.INVOKE_RUNBOOK, EscalationAction.AUTO_REMEDIATION],
                        contacts=["incident_commander", "cto"]
                    )
                ],
                auto_resolve_timeout_minutes=30
            )
        ]
        
        for policy in default_policies:
            self.escalation_policies[policy.id] = policy
        
        logger.info(f"Chargé {len(self.escalation_policies)} politiques d'escalade")

    async def load_contacts(self):
        """Chargement des contacts"""
        
        # Contacts par défaut
        default_contacts = [
            ContactInfo(
                name="oncall_primary",
                email="oncall@company.com",
                phone="+1234567890",
                slack_user_id="U123456",
                preferred_methods=[NotificationType.SLACK, NotificationType.EMAIL, NotificationType.SMS]
            ),
            ContactInfo(
                name="team_lead",
                email="team.lead@company.com",
                slack_user_id="U789012",
                preferred_methods=[NotificationType.EMAIL, NotificationType.SLACK]
            ),
            ContactInfo(
                name="incident_commander",
                email="incident.commander@company.com",
                phone="+1987654321",
                preferred_methods=[NotificationType.EMAIL, NotificationType.SMS]
            )
        ]
        
        for contact in default_contacts:
            self.contacts[contact.name] = contact
        
        logger.info(f"Chargé {len(self.contacts)} contacts")

    async def process_alert(self, alert_data: Dict[str, Any]) -> Alert:
        """Traitement d'une nouvelle alerte"""
        try:
            # Création de l'alerte
            alert = Alert(
                id=str(uuid.uuid4()),
                rule_name=alert_data.get('rule_name', 'unknown'),
                status=AlertStatus.FIRING,
                severity=AlertSeverity(alert_data.get('severity', 'warning')),
                title=alert_data.get('title', 'Unknown Alert'),
                description=alert_data.get('description', ''),
                started_at=datetime.utcnow(),
                labels=alert_data.get('labels', {}),
                annotations=alert_data.get('annotations', {}),
                tenant_id=alert_data.get('tenant_id', ''),
                source=alert_data.get('source', 'monitoring')
            )
            
            # Génération du fingerprint pour déduplication
            alert.fingerprint = self.generate_fingerprint(alert)
            
            # Vérification de déduplication
            if await self.is_duplicate_alert(alert):
                logger.info(f"Alerte dupliquée ignorée: {alert.fingerprint}")
                return None
            
            # Assignation de la politique d'escalade
            alert.escalation_policy_id = await self.assign_escalation_policy(alert)
            
            # Stockage de l'alerte
            self.active_alerts[alert.id] = alert
            await self.store_alert_to_redis(alert)
            
            # Déclenchement des notifications
            await self.trigger_notifications(alert)
            
            # Ajout à un groupe si applicable
            await self.add_to_alert_group(alert)
            
            logger.info(f"Alerte traitée: {alert.id} - {alert.title}")
            
            return alert
            
        except Exception as e:
            logger.error(f"Erreur traitement alerte: {e}")
            raise

    def generate_fingerprint(self, alert: Alert) -> str:
        """Génération d'un fingerprint unique pour l'alerte"""
        # Combinaison des éléments clés pour l'unicité
        key_elements = [
            alert.rule_name,
            alert.tenant_id,
            str(sorted(alert.labels.items())),
            alert.source
        ]
        
        fingerprint_string = "|".join(key_elements)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]

    async def is_duplicate_alert(self, alert: Alert) -> bool:
        """Vérification si l'alerte est un duplicata"""
        try:
            # Recherche dans les alertes actives
            for existing_alert in self.active_alerts.values():
                if existing_alert.fingerprint == alert.fingerprint:
                    if existing_alert.status == AlertStatus.FIRING:
                        # Mise à jour du timestamp
                        existing_alert.started_at = alert.started_at
                        return True
            
            # Recherche dans Redis pour la fenêtre de déduplication
            redis_key = f"alert_fingerprint:{alert.fingerprint}"
            existing = await self.redis_client.get(redis_key)
            
            if existing:
                return True
            
            # Stockage du fingerprint avec TTL
            await self.redis_client.setex(
                redis_key,
                self.deduplication_window * 60,
                alert.id
            )
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur vérification duplicata: {e}")
            return False

    async def assign_escalation_policy(self, alert: Alert) -> str:
        """Assignation d'une politique d'escalade à l'alerte"""
        
        # Logique d'assignation basée sur la sévérité et les labels
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            return "critical_escalation"
        
        # Vérification des labels spéciaux
        if "escalation_policy" in alert.labels:
            policy_id = alert.labels["escalation_policy"]
            if policy_id in self.escalation_policies:
                return policy_id
        
        # Politique par défaut
        return "standard_escalation"

    async def trigger_notifications(self, alert: Alert):
        """Déclenchement des notifications pour une alerte"""
        try:
            # Application des règles de notification
            for rule_id, rule in self.notification_rules.items():
                if not rule.enabled:
                    continue
                
                if await self.alert_matches_rule(alert, rule):
                    await self.schedule_notifications(alert, rule)
            
        except Exception as e:
            logger.error(f"Erreur déclenchement notifications: {e}")

    async def alert_matches_rule(self, alert: Alert, rule: NotificationRule) -> bool:
        """Vérification si une alerte correspond à une règle"""
        
        # Filtre par sévérité
        if "severity" in rule.filters:
            if alert.severity.value not in rule.filters["severity"]:
                return False
        
        # Filtre par tags/labels
        if "tags" in rule.filters:
            rule_tags = rule.filters["tags"]
            alert_tags = [f"{k}:{v}" for k, v in alert.labels.items()]
            
            if not any(tag in alert_tags for tag in rule_tags):
                return False
        
        # Filtre par tenant
        if "tenant_id" in rule.filters:
            if alert.tenant_id not in rule.filters["tenant_id"]:
                return False
        
        # Conditions spéciales
        if rule.conditions:
            if not await self.evaluate_rule_conditions(alert, rule.conditions):
                return False
        
        return True

    async def evaluate_rule_conditions(self, alert: Alert, conditions: Dict[str, Any]) -> bool:
        """Évaluation des conditions spéciales d'une règle"""
        
        # Condition: heures ouvrables uniquement
        if conditions.get("business_hours_only", False):
            current_hour = datetime.utcnow().hour
            if not (9 <= current_hour <= 17):  # 9h-17h UTC
                return False
        
        # Condition: limites de taux
        if "rate_limit" in conditions:
            rate_limit = conditions["rate_limit"]
            if not await self.check_rate_limit(alert, rate_limit):
                return False
        
        # Condition: blackout periods
        if "blackout_periods" in conditions:
            if await self.is_in_blackout_period(conditions["blackout_periods"]):
                return False
        
        return True

    async def schedule_notifications(self, alert: Alert, rule: NotificationRule):
        """Planification des notifications selon une règle"""
        try:
            for channel_id in rule.channels:
                # Délai initial
                send_time = datetime.utcnow() + timedelta(seconds=rule.delay_seconds)
                
                # Création du message de notification
                message = await self.create_notification_message(alert, channel_id)
                message.metadata["rule_id"] = rule.id
                message.metadata["send_time"] = send_time.isoformat()
                message.metadata["repeat_interval"] = rule.repeat_interval_seconds
                message.metadata["max_repeats"] = rule.max_repeats
                
                # Ajout à la queue
                self.pending_notifications.append(message)
                
                logger.info(f"Notification planifiée: {message.id} via {channel_id}")
                
        except Exception as e:
            logger.error(f"Erreur planification notifications: {e}")

    async def create_notification_message(self, alert: Alert, channel_id: str) -> NotificationMessage:
        """Création d'un message de notification"""
        
        message = NotificationMessage(
            id=str(uuid.uuid4()),
            alert_id=alert.id,
            channel_id=channel_id,
            recipient="",  # Sera défini selon le canal
            subject=f"[{alert.severity.value.upper()}] {alert.title}",
            content="",  # Sera générée selon le template
            created_at=datetime.utcnow()
        )
        
        # Génération du contenu selon le type de canal
        await self.generate_message_content(message, alert)
        
        return message

    async def generate_message_content(self, message: NotificationMessage, alert: Alert):
        """Génération du contenu du message selon le canal"""
        
        # Template de base pour email/text
        basic_template = """
Alert: {{ alert.title }}
Severity: {{ alert.severity.value.upper() }}
Status: {{ alert.status.value }}
Started: {{ alert.started_at }}
{% if alert.description %}
Description: {{ alert.description }}
{% endif %}

Labels:
{% for key, value in alert.labels.items() %}
- {{ key }}: {{ value }}
{% endfor %}

{% if alert.annotations %}
Annotations:
{% for key, value in alert.annotations.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

Alert ID: {{ alert.id }}
Tenant: {{ alert.tenant_id }}
Source: {{ alert.source }}
        """
        
        # Rendu du template
        template = Template(basic_template)
        message.content = template.render(alert=alert)

    async def add_to_alert_group(self, alert: Alert):
        """Ajout d'une alerte à un groupe pour traitement par lot"""
        
        # Clé de groupement basée sur la règle et la sévérité
        group_key = f"{alert.rule_name}:{alert.severity.value}:{alert.tenant_id}"
        
        if group_key not in self.alert_groups:
            self.alert_groups[group_key] = []
        
        self.alert_groups[group_key].append(alert.id)
        
        # Limitation du nombre d'alertes par groupe
        if len(self.alert_groups[group_key]) > self.max_alerts_per_group:
            self.alert_groups[group_key] = self.alert_groups[group_key][-self.max_alerts_per_group:]

    async def process_notification_queue(self):
        """Traitement de la queue de notifications"""
        while True:
            try:
                current_time = datetime.utcnow()
                messages_to_send = []
                
                # Recherche des messages prêts à être envoyés
                for message in self.pending_notifications[:]:
                    send_time_str = message.metadata.get("send_time")
                    if send_time_str:
                        send_time = datetime.fromisoformat(send_time_str)
                        if current_time >= send_time:
                            messages_to_send.append(message)
                            self.pending_notifications.remove(message)
                
                # Envoi des messages
                for message in messages_to_send:
                    await self.send_notification_message(message)
                
                # Attente avant le prochain cycle
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Erreur traitement queue notifications: {e}")
                await asyncio.sleep(5)

    async def send_notification_message(self, message: NotificationMessage):
        """Envoi d'un message de notification"""
        try:
            # Simulation d'envoi (à remplacer par de vrais envois)
            logger.info(f"Envoi notification: {message.id} via {message.channel_id}")
            
            # Mise à jour du statut
            message.sent_at = datetime.utcnow()
            message.delivery_status = "sent"
            
            # Planification des répétitions si nécessaire
            await self.schedule_repeat_notification(message)
            
        except Exception as e:
            logger.error(f"Erreur envoi notification {message.id}: {e}")
            message.delivery_status = "failed"
            message.retry_count += 1

    async def schedule_repeat_notification(self, message: NotificationMessage):
        """Planification des notifications répétées"""
        
        repeat_interval = message.metadata.get("repeat_interval")
        max_repeats = message.metadata.get("max_repeats", 0)
        current_repeats = message.metadata.get("current_repeats", 0)
        
        if repeat_interval and current_repeats < max_repeats:
            # Vérification si l'alerte est toujours active
            alert = self.active_alerts.get(message.alert_id)
            if alert and alert.status == AlertStatus.FIRING:
                # Création d'une nouvelle notification répétée
                next_send_time = datetime.utcnow() + timedelta(seconds=repeat_interval)
                
                repeat_message = NotificationMessage(
                    id=str(uuid.uuid4()),
                    alert_id=message.alert_id,
                    channel_id=message.channel_id,
                    recipient=message.recipient,
                    subject=f"[REPEAT] {message.subject}",
                    content=message.content,
                    created_at=datetime.utcnow(),
                    metadata=message.metadata.copy()
                )
                
                repeat_message.metadata["send_time"] = next_send_time.isoformat()
                repeat_message.metadata["current_repeats"] = current_repeats + 1
                
                self.pending_notifications.append(repeat_message)

    async def process_escalations(self):
        """Traitement des escalades"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for alert in list(self.active_alerts.values()):
                    if alert.status != AlertStatus.FIRING:
                        continue
                    
                    if not alert.escalation_policy_id:
                        continue
                    
                    await self.check_escalation(alert, current_time)
                
                await asyncio.sleep(60)  # Vérification chaque minute
                
            except Exception as e:
                logger.error(f"Erreur traitement escalades: {e}")
                await asyncio.sleep(30)

    async def check_escalation(self, alert: Alert, current_time: datetime):
        """Vérification et exécution d'escalade pour une alerte"""
        
        policy = self.escalation_policies.get(alert.escalation_policy_id)
        if not policy or not policy.enabled:
            return
        
        # Calcul du temps écoulé depuis le début de l'alerte
        elapsed_minutes = (current_time - alert.started_at).total_seconds() / 60
        
        # Recherche du niveau d'escalade approprié
        for level in policy.levels:
            if level.level > alert.current_escalation_level and elapsed_minutes >= level.delay_minutes:
                await self.execute_escalation_level(alert, level, policy)
                alert.current_escalation_level = level.level

    async def execute_escalation_level(self, alert: Alert, level: EscalationLevel, policy: EscalationPolicy):
        """Exécution d'un niveau d'escalade"""
        try:
            logger.warning(f"Escalade niveau {level.level} pour alerte {alert.id}")
            
            for action in level.actions:
                await self.execute_escalation_action(alert, action, level)
            
            # Notification des contacts du niveau
            for contact_name in level.contacts:
                contact = self.contacts.get(contact_name)
                if contact:
                    await self.notify_contact_for_escalation(alert, contact, level)
            
        except Exception as e:
            logger.error(f"Erreur exécution escalade: {e}")

    async def execute_escalation_action(self, alert: Alert, action: EscalationAction, level: EscalationLevel):
        """Exécution d'une action d'escalade"""
        
        if action == EscalationAction.NOTIFY_TEAM:
            logger.info(f"Action: Notification équipe pour alerte {alert.id}")
            # Implémentation notification équipe
            
        elif action == EscalationAction.NOTIFY_MANAGER:
            logger.info(f"Action: Notification manager pour alerte {alert.id}")
            # Implémentation notification manager
            
        elif action == EscalationAction.CREATE_INCIDENT:
            logger.info(f"Action: Création incident pour alerte {alert.id}")
            # Implémentation création incident
            
        elif action == EscalationAction.CALL_ONCALL:
            logger.info(f"Action: Appel oncall pour alerte {alert.id}")
            # Implémentation appel téléphonique
            
        elif action == EscalationAction.INVOKE_RUNBOOK:
            logger.info(f"Action: Invocation runbook pour alerte {alert.id}")
            # Implémentation exécution runbook
            
        elif action == EscalationAction.AUTO_REMEDIATION:
            logger.info(f"Action: Remédiation automatique pour alerte {alert.id}")
            # Implémentation remédiation automatique

    async def notify_contact_for_escalation(self, alert: Alert, contact: ContactInfo, level: EscalationLevel):
        """Notification d'un contact dans le cadre d'une escalade"""
        
        # Création de messages d'escalade pour tous les canaux préférés du contact
        for notification_type in contact.preferred_methods:
            channel_id = f"{notification_type.value}_{contact.name}"
            
            escalation_message = NotificationMessage(
                id=str(uuid.uuid4()),
                alert_id=alert.id,
                channel_id=channel_id,
                recipient=self.get_contact_recipient(contact, notification_type),
                subject=f"[ESCALATION L{level.level}] {alert.title}",
                content=f"ESCALATION LEVEL {level.level}\n\n" + await self.generate_escalation_content(alert, level),
                created_at=datetime.utcnow(),
                metadata={"escalation_level": level.level, "contact": contact.name}
            )
            
            await self.send_notification_message(escalation_message)

    def get_contact_recipient(self, contact: ContactInfo, notification_type: NotificationType) -> str:
        """Récupération du destinataire selon le type de notification"""
        
        if notification_type == NotificationType.EMAIL:
            return contact.email or ""
        elif notification_type == NotificationType.SMS:
            return contact.phone or ""
        elif notification_type == NotificationType.SLACK:
            return contact.slack_user_id or ""
        elif notification_type == NotificationType.TELEGRAM:
            return contact.telegram_user_id or ""
        
        return ""

    async def generate_escalation_content(self, alert: Alert, level: EscalationLevel) -> str:
        """Génération du contenu d'escalade"""
        
        escalation_template = """
🚨 ALERT ESCALATION - LEVEL {{ level.level }} 🚨

Alert: {{ alert.title }}
Severity: {{ alert.severity.value.upper() }}
Duration: {{ duration_minutes }} minutes
Tenant: {{ alert.tenant_id }}

{{ alert.description }}

This alert has been escalated to level {{ level.level }} due to:
- No acknowledgment or resolution
- Continued firing for {{ duration_minutes }} minutes

Required Actions:
{% for action in level.actions %}
- {{ action.value }}
{% endfor %}

Alert Details:
- Alert ID: {{ alert.id }}
- Started: {{ alert.started_at }}
- Source: {{ alert.source }}
- Rule: {{ alert.rule_name }}

Please take immediate action to resolve this alert.
        """
        
        duration_minutes = int((datetime.utcnow() - alert.started_at).total_seconds() / 60)
        
        template = Template(escalation_template)
        return template.render(
            alert=alert,
            level=level,
            duration_minutes=duration_minutes
        )

    async def process_alert_grouping(self):
        """Traitement du groupement d'alertes"""
        while True:
            try:
                await asyncio.sleep(self.grouping_interval)
                
                # Traitement des groupes d'alertes
                for group_key, alert_ids in self.alert_groups.items():
                    if len(alert_ids) >= 3:  # Seuil pour notification groupée
                        await self.send_grouped_notification(group_key, alert_ids)
                
                # Nettoyage des groupes traités
                self.alert_groups.clear()
                
            except Exception as e:
                logger.error(f"Erreur traitement groupement: {e}")
                await asyncio.sleep(10)

    async def send_grouped_notification(self, group_key: str, alert_ids: List[str]):
        """Envoi d'une notification groupée"""
        try:
            alerts = [self.active_alerts[aid] for aid in alert_ids if aid in self.active_alerts]
            
            if not alerts:
                return
            
            # Création du message groupé
            grouped_message = NotificationMessage(
                id=str(uuid.uuid4()),
                alert_id="grouped",
                channel_id="email_ops",  # Canal par défaut pour les groupes
                recipient="ops@company.com",
                subject=f"[GROUPED] {len(alerts)} alerts - {group_key}",
                content=await self.generate_grouped_content(alerts),
                created_at=datetime.utcnow(),
                metadata={"group_key": group_key, "alert_count": len(alerts)}
            )
            
            await self.send_notification_message(grouped_message)
            
            logger.info(f"Notification groupée envoyée: {len(alerts)} alertes")
            
        except Exception as e:
            logger.error(f"Erreur notification groupée: {e}")

    async def generate_grouped_content(self, alerts: List[Alert]) -> str:
        """Génération du contenu pour notification groupée"""
        
        grouped_template = """
🔥 GROUPED ALERT NOTIFICATION 🔥

{{ alert_count }} alerts have been triggered in the same category:

{% for alert in alerts %}
{{ loop.index }}. [{{ alert.severity.value.upper() }}] {{ alert.title }}
   - Started: {{ alert.started_at }}
   - Tenant: {{ alert.tenant_id }}
   - ID: {{ alert.id }}
   
{% endfor %}

Common characteristics:
- Rule: {{ alerts[0].rule_name }}
- Source: {{ alerts[0].source }}

This grouping suggests a potential systemic issue.
Please investigate the root cause affecting multiple instances.
        """
        
        template = Template(grouped_template)
        return template.render(alerts=alerts, alert_count=len(alerts))

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Accusé de réception d'une alerte"""
        try:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return False
            
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            
            # Arrêt des notifications répétées
            await self.stop_repeat_notifications(alert_id)
            
            logger.info(f"Alerte accusée réception: {alert_id} par {acknowledged_by}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur accusé réception: {e}")
            return False

    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Résolution d'une alerte"""
        try:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return False
            
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            
            # Notification de résolution
            await self.send_resolution_notification(alert, resolved_by)
            
            # Déplacement vers l'historique
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            # Nettoyage Redis
            await self.cleanup_alert_from_redis(alert_id)
            
            logger.info(f"Alerte résolue: {alert_id} par {resolved_by}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur résolution alerte: {e}")
            return False

    async def send_resolution_notification(self, alert: Alert, resolved_by: str):
        """Envoi de notification de résolution"""
        
        resolution_message = NotificationMessage(
            id=str(uuid.uuid4()),
            alert_id=alert.id,
            channel_id="email_ops",
            recipient="ops@company.com",
            subject=f"[RESOLVED] {alert.title}",
            content=f"""
✅ ALERT RESOLVED ✅

Alert: {alert.title}
Severity: {alert.severity.value.upper()}
Resolved by: {resolved_by}
Duration: {(alert.resolved_at - alert.started_at).total_seconds() / 60:.1f} minutes

Resolution time: {alert.resolved_at}
Alert ID: {alert.id}
            """,
            created_at=datetime.utcnow(),
            metadata={"resolution": True, "resolved_by": resolved_by}
        )
        
        await self.send_notification_message(resolution_message)

    async def stop_repeat_notifications(self, alert_id: str):
        """Arrêt des notifications répétées pour une alerte"""
        
        # Suppression des notifications en attente pour cette alerte
        self.pending_notifications = [
            msg for msg in self.pending_notifications
            if msg.alert_id != alert_id
        ]

    async def store_alert_to_redis(self, alert: Alert):
        """Stockage d'une alerte dans Redis"""
        try:
            alert_data = {
                "id": alert.id,
                "title": alert.title,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "started_at": alert.started_at.isoformat(),
                "tenant_id": alert.tenant_id,
                "fingerprint": alert.fingerprint
            }
            
            await self.redis_client.hset(
                f"alert:{alert.id}",
                mapping=alert_data
            )
            
            # TTL pour nettoyage automatique
            await self.redis_client.expire(f"alert:{alert.id}", 86400)  # 24h
            
        except Exception as e:
            logger.error(f"Erreur stockage Redis: {e}")

    async def cleanup_alert_from_redis(self, alert_id: str):
        """Nettoyage d'une alerte de Redis"""
        try:
            await self.redis_client.delete(f"alert:{alert_id}")
        except Exception as e:
            logger.error(f"Erreur nettoyage Redis: {e}")

    async def get_active_alerts_summary(self) -> Dict[str, Any]:
        """Récupération d'un résumé des alertes actives"""
        
        summary = {
            "total_active": len(self.active_alerts),
            "by_severity": {},
            "by_tenant": {},
            "by_status": {},
            "oldest_alert": None,
            "escalated_alerts": 0
        }
        
        for alert in self.active_alerts.values():
            # Par sévérité
            severity = alert.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            
            # Par tenant
            tenant = alert.tenant_id or "unknown"
            summary["by_tenant"][tenant] = summary["by_tenant"].get(tenant, 0) + 1
            
            # Par statut
            status = alert.status.value
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            
            # Alerte la plus ancienne
            if not summary["oldest_alert"] or alert.started_at < summary["oldest_alert"]["started_at"]:
                summary["oldest_alert"] = {
                    "id": alert.id,
                    "title": alert.title,
                    "started_at": alert.started_at.isoformat()
                }
            
            # Alertes escaladées
            if alert.current_escalation_level > 0:
                summary["escalated_alerts"] += 1
        
        return summary

    async def shutdown(self):
        """Arrêt propre du gestionnaire"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.db_engine:
                await self.db_engine.dispose()
            
            logger.info("Gestionnaire d'alerting arrêté proprement")
            
        except Exception as e:
            logger.error(f"Erreur arrêt gestionnaire: {e}")

# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

async def main():
    """Exemple d'utilisation du système d'alerting"""
    
    config = {
        'redis_url': 'redis://localhost:6379/0',
        'database_url': 'postgresql+asyncpg://user:pass@localhost/alerting',
        'deduplication_window_minutes': 5,
        'grouping_interval_seconds': 30,
        'max_alerts_per_group': 10
    }
    
    # Initialisation du gestionnaire
    alert_manager = IntelligentAlertManager(config)
    await alert_manager.initialize()
    
    try:
        print("=== Démonstration du système d'alerting ===")
        
        # Simulation d'alertes
        test_alerts = [
            {
                "rule_name": "high_cpu_usage",
                "severity": "critical",
                "title": "High CPU Usage Detected",
                "description": "CPU usage above 90% for 5 minutes",
                "labels": {"instance": "web-01", "team": "ops"},
                "tenant_id": "tenant_123"
            },
            {
                "rule_name": "api_latency",
                "severity": "warning",
                "title": "API Latency High",
                "description": "API response time above 2 seconds",
                "labels": {"endpoint": "/api/incidents", "team": "dev"},
                "tenant_id": "tenant_123"
            },
            {
                "rule_name": "database_error",
                "severity": "emergency",
                "title": "Database Connection Failed",
                "description": "Cannot connect to primary database",
                "labels": {"database": "primary", "team": "dba"},
                "tenant_id": "tenant_456"
            }
        ]
        
        # Traitement des alertes
        alerts = []
        for alert_data in test_alerts:
            alert = await alert_manager.process_alert(alert_data)
            if alert:
                alerts.append(alert)
                print(f"Alerte créée: {alert.title} ({alert.severity.value})")
        
        # Attente pour voir les notifications et escalades
        print("\nAttente des notifications et escalades...")
        await asyncio.sleep(20)
        
        # Accusé de réception d'une alerte
        if alerts:
            await alert_manager.acknowledge_alert(alerts[0].id, "john.doe")
            print(f"Alerte accusée réception: {alerts[0].title}")
        
        # Résumé des alertes actives
        summary = await alert_manager.get_active_alerts_summary()
        print(f"\n=== Résumé des alertes actives ===")
        print(f"Total: {summary['total_active']}")
        print(f"Par sévérité: {summary['by_severity']}")
        print(f"Alertes escaladées: {summary['escalated_alerts']}")
        
        # Résolution d'une alerte
        if len(alerts) > 1:
            await alert_manager.resolve_alert(alerts[1].id, "automation")
            print(f"Alerte résolue: {alerts[1].title}")
        
        # Nouveau résumé
        summary = await alert_manager.get_active_alerts_summary()
        print(f"\nAprès résolution - Total actif: {summary['total_active']}")
        
    finally:
        await alert_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
