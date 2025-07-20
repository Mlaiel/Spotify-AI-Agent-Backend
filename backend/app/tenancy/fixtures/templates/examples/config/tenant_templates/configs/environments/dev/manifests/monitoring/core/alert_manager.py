# -*- coding: utf-8 -*-
"""
Alert Manager - Système d'Alerting Ultra-Avancé
==============================================

Gestionnaire d'alertes intelligent pour l'agent IA Spotify.
Gestion complète des alertes avec escalade automatique, auto-remédiation,
et intégration multi-canaux (Slack, Email, SMS, PagerDuty).

Fonctionnalités:
- Alerting intelligent avec ML pour détection d'anomalies
- Escalade automatique selon la sévérité
- Auto-remédiation avec scripts personnalisés
- Suppression des doublons et groupage d'alertes
- Intégration multi-canaux de notification
- Historique et audit trail complet
- SLA tracking et reporting

Auteur: Expert Team - Spécialiste Sécurité Backend + Architecte IA - Fahed Mlaiel
Version: 2.0.0
"""

import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import sqlite3
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import redis
from concurrent.futures import ThreadPoolExecutor
import subprocess

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Statuts d'alerte"""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"

class NotificationChannel(Enum):
    """Canaux de notification"""
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"

@dataclass
class Alert:
    """Modèle d'alerte"""
    id: str
    name: str
    description: str
    severity: str
    status: str
    source: str
    tenant_id: Optional[str]
    labels: Dict[str, str]
    annotations: Dict[str, str]
    created_at: float
    updated_at: float
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    escalated: bool = False
    notification_count: int = 0
    auto_resolved: bool = False

@dataclass
class NotificationRule:
    """Règle de notification"""
    name: str
    channels: List[str]
    severity_levels: List[str]
    tenant_filters: List[str]
    label_filters: Dict[str, str]
    cooldown_minutes: int
    escalation_minutes: int
    max_notifications: int
    enabled: bool = True

@dataclass
class RemediationAction:
    """Action de remédiation automatique"""
    name: str
    description: str
    script_path: str
    conditions: Dict[str, Any]
    timeout_seconds: int
    max_retries: int
    enabled: bool = True

class AlertManager:
    """
    Gestionnaire d'alertes ultra-avancé avec intelligence artificielle
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le gestionnaire d'alertes
        
        Args:
            config: Configuration du gestionnaire
        """
        self.config = config or self._default_config()
        self.alerts: Dict[str, Alert] = {}
        self.notification_rules: List[NotificationRule] = []
        self.remediation_actions: List[RemediationAction] = []
        self.lock = threading.RLock()
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Storage
        self.db_path = self.config.get('db_path', 'alerts.db')
        self.redis_client = self._init_redis()
        
        # Notification channels
        self.notification_handlers = {
            NotificationChannel.SLACK.value: self._send_slack_notification,
            NotificationChannel.EMAIL.value: self._send_email_notification,
            NotificationChannel.SMS.value: self._send_sms_notification,
            NotificationChannel.PAGERDUTY.value: self._send_pagerduty_notification,
            NotificationChannel.WEBHOOK.value: self._send_webhook_notification
        }
        
        # Background threads
        self.processor_thread = None
        self.escalation_thread = None
        
        # Initialisation
        self._init_database()
        self._load_rules()
        self._load_remediation_actions()
        
        logger.info("AlertManager initialisé avec succès")
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            'processing_interval': 5,
            'escalation_interval': 60,
            'max_alert_age_days': 30,
            'deduplication_window_minutes': 5,
            'auto_resolve_timeout_minutes': 60,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 1,
            'slack_webhook_url': '',
            'email_smtp_server': 'localhost',
            'email_smtp_port': 587,
            'email_username': '',
            'email_password': '',
            'sms_api_url': '',
            'sms_api_key': '',
            'pagerduty_api_key': '',
            'webhook_timeout': 30,
            'enable_auto_remediation': True,
            'enable_ml_anomaly_detection': True
        }
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialise la connexion Redis"""
        try:
            client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                db=self.config['redis_db'],
                decode_responses=True
            )
            client.ping()
            logger.info("Connexion Redis AlertManager établie")
            return client
        except Exception as e:
            logger.warning(f"Redis non disponible pour AlertManager: {e}")
            return None
    
    def _init_database(self):
        """Initialise la base de données SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table des alertes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    tenant_id TEXT,
                    labels TEXT,
                    annotations TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    resolved_at REAL,
                    acknowledged_at REAL,
                    acknowledged_by TEXT,
                    escalated INTEGER DEFAULT 0,
                    notification_count INTEGER DEFAULT 0,
                    auto_resolved INTEGER DEFAULT 0
                )
            ''')
            
            # Table des règles de notification
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notification_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Table des actions de remédiation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS remediation_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    script_path TEXT NOT NULL,
                    conditions TEXT,
                    timeout_seconds INTEGER DEFAULT 300,
                    max_retries INTEGER DEFAULT 3,
                    enabled INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Index pour les performances
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_tenant ON alerts(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at)')
            
            conn.commit()
            conn.close()
            
            logger.info("Base de données AlertManager initialisée")
            
        except Exception as e:
            logger.error(f"Erreur initialisation base de données: {e}")
    
    def start(self):
        """Démarre le gestionnaire d'alertes"""
        if self.running:
            logger.warning("AlertManager déjà en cours d'exécution")
            return
        
        self.running = True
        
        # Démarre les threads de traitement
        self.processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.escalation_thread = threading.Thread(target=self._escalation_loop, daemon=True)
        
        self.processor_thread.start()
        self.escalation_thread.start()
        
        logger.info("AlertManager démarré")
    
    def stop(self):
        """Arrête le gestionnaire d'alertes"""
        self.running = False
        
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
        
        if self.escalation_thread and self.escalation_thread.is_alive():
            self.escalation_thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
        logger.info("AlertManager arrêté")
    
    def trigger_alert(self, name: str, description: str = "",
                     severity: str = AlertSeverity.WARNING.value,
                     source: str = "system",
                     tenant_id: Optional[str] = None,
                     labels: Optional[Dict[str, str]] = None,
                     annotations: Optional[Dict[str, str]] = None) -> str:
        """
        Déclenche une nouvelle alerte
        
        Args:
            name: Nom de l'alerte
            description: Description détaillée
            severity: Niveau de sévérité
            source: Source de l'alerte
            tenant_id: ID du tenant
            labels: Labels de l'alerte
            annotations: Annotations additionnelles
            
        Returns:
            ID de l'alerte créée
        """
        try:
            # Génération de l'ID unique
            alert_id = self._generate_alert_id(name, labels, tenant_id)
            
            # Vérification de déduplication
            if self._is_duplicate_alert(alert_id):
                logger.debug(f"Alerte dupliquée ignorée: {alert_id}")
                return alert_id
            
            # Création de l'alerte
            now = time.time()
            alert = Alert(
                id=alert_id,
                name=name,
                description=description,
                severity=severity,
                status=AlertStatus.OPEN.value,
                source=source,
                tenant_id=tenant_id,
                labels=labels or {},
                annotations=annotations or {},
                created_at=now,
                updated_at=now
            )
            
            # Stockage
            with self.lock:
                self.alerts[alert_id] = alert
                self._save_alert_to_db(alert)
            
            # Traitement asynchrone
            self.executor.submit(self._process_new_alert, alert)
            
            logger.info(f"Alerte déclenchée: {name} [{severity}] - {alert_id}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Erreur lors du déclenchement d'alerte: {e}")
            raise
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """
        Acquitte une alerte
        
        Args:
            alert_id: ID de l'alerte
            acknowledged_by: Qui a acquitté l'alerte
            
        Returns:
            True si l'acquittement a réussi
        """
        try:
            with self.lock:
                if alert_id not in self.alerts:
                    logger.warning(f"Alerte introuvable: {alert_id}")
                    return False
                
                alert = self.alerts[alert_id]
                if alert.status != AlertStatus.OPEN.value:
                    logger.warning(f"Alerte déjà traitée: {alert_id}")
                    return False
                
                # Mise à jour
                alert.status = AlertStatus.ACKNOWLEDGED.value
                alert.acknowledged_at = time.time()
                alert.acknowledged_by = acknowledged_by
                alert.updated_at = time.time()
                
                self._save_alert_to_db(alert)
            
            logger.info(f"Alerte acquittée: {alert_id} par {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur acquittement alerte: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, auto_resolved: bool = False) -> bool:
        """
        Résout une alerte
        
        Args:
            alert_id: ID de l'alerte
            auto_resolved: True si résolution automatique
            
        Returns:
            True si la résolution a réussi
        """
        try:
            with self.lock:
                if alert_id not in self.alerts:
                    logger.warning(f"Alerte introuvable: {alert_id}")
                    return False
                
                alert = self.alerts[alert_id]
                if alert.status == AlertStatus.RESOLVED.value:
                    logger.debug(f"Alerte déjà résolue: {alert_id}")
                    return True
                
                # Mise à jour
                alert.status = AlertStatus.RESOLVED.value
                alert.resolved_at = time.time()
                alert.updated_at = time.time()
                alert.auto_resolved = auto_resolved
                
                self._save_alert_to_db(alert)
            
            logger.info(f"Alerte résolue: {alert_id} "
                       f"{'(auto)' if auto_resolved else '(manuel)'}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur résolution alerte: {e}")
            return False
    
    def silence_alert(self, alert_id: str, duration_minutes: int = 60) -> bool:
        """
        Met en silence une alerte
        
        Args:
            alert_id: ID de l'alerte
            duration_minutes: Durée du silence en minutes
            
        Returns:
            True si le silence a réussi
        """
        try:
            with self.lock:
                if alert_id not in self.alerts:
                    logger.warning(f"Alerte introuvable: {alert_id}")
                    return False
                
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.SILENCED.value
                alert.updated_at = time.time()
                
                # Stockage de la durée du silence
                alert.annotations['silenced_until'] = str(
                    time.time() + (duration_minutes * 60)
                )
                
                self._save_alert_to_db(alert)
            
            logger.info(f"Alerte mise en silence: {alert_id} pour {duration_minutes}min")
            return True
            
        except Exception as e:
            logger.error(f"Erreur silence alerte: {e}")
            return False
    
    def _generate_alert_id(self, name: str, labels: Optional[Dict[str, str]] = None,
                          tenant_id: Optional[str] = None) -> str:
        """Génère un ID unique pour l'alerte"""
        components = [name]
        
        if tenant_id:
            components.append(f"tenant:{tenant_id}")
        
        if labels:
            sorted_labels = sorted(labels.items())
            labels_str = ",".join([f"{k}:{v}" for k, v in sorted_labels])
            components.append(f"labels:{labels_str}")
        
        content = "|".join(components)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _is_duplicate_alert(self, alert_id: str) -> bool:
        """Vérifie si l'alerte est un doublon récent"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        if alert.status == AlertStatus.RESOLVED.value:
            return False
        
        # Fenêtre de déduplication
        dedup_window = self.config['deduplication_window_minutes'] * 60
        if time.time() - alert.created_at < dedup_window:
            return True
        
        return False
    
    def _process_new_alert(self, alert: Alert):
        """Traite une nouvelle alerte"""
        try:
            # Auto-remédiation
            if self.config['enable_auto_remediation']:
                self._attempt_auto_remediation(alert)
            
            # Notifications
            self._send_notifications(alert)
            
            # Détection d'anomalies ML
            if self.config['enable_ml_anomaly_detection']:
                self._analyze_alert_patterns(alert)
            
        except Exception as e:
            logger.error(f"Erreur traitement alerte {alert.id}: {e}")
    
    def _attempt_auto_remediation(self, alert: Alert):
        """Tente la remédiation automatique"""
        try:
            matching_actions = []
            
            for action in self.remediation_actions:
                if not action.enabled:
                    continue
                
                if self._matches_conditions(alert, action.conditions):
                    matching_actions.append(action)
            
            for action in matching_actions:
                logger.info(f"Exécution action remédiation: {action.name} pour {alert.id}")
                success = self._execute_remediation_action(action, alert)
                
                if success:
                    alert.annotations['auto_remediation'] = action.name
                    alert.annotations['auto_remediation_time'] = str(time.time())
                    self._save_alert_to_db(alert)
                    
                    # Auto-résolution si succès
                    self.executor.submit(self._check_auto_resolution, alert, action)
                    break
            
        except Exception as e:
            logger.error(f"Erreur auto-remédiation: {e}")
    
    def _execute_remediation_action(self, action: RemediationAction, alert: Alert) -> bool:
        """Exécute une action de remédiation"""
        try:
            # Préparation de l'environnement
            env = {
                'ALERT_ID': alert.id,
                'ALERT_NAME': alert.name,
                'ALERT_SEVERITY': alert.severity,
                'ALERT_TENANT': alert.tenant_id or '',
                'ALERT_SOURCE': alert.source
            }
            
            # Ajout des labels comme variables d'environnement
            for key, value in alert.labels.items():
                env[f'ALERT_LABEL_{key.upper()}'] = value
            
            # Exécution du script
            result = subprocess.run(
                [action.script_path],
                env=env,
                timeout=action.timeout_seconds,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Action remédiation réussie: {action.name}")
                return True
            else:
                logger.error(f"Action remédiation échouée: {action.name} - {result.stderr}")
                return False
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout action remédiation: {action.name}")
            return False
        except Exception as e:
            logger.error(f"Erreur exécution action: {e}")
            return False
    
    def _matches_conditions(self, alert: Alert, conditions: Dict[str, Any]) -> bool:
        """Vérifie si l'alerte correspond aux conditions"""
        try:
            for key, expected_value in conditions.items():
                if key == 'severity':
                    if alert.severity != expected_value:
                        return False
                elif key == 'name_pattern':
                    import re
                    if not re.match(expected_value, alert.name):
                        return False
                elif key == 'labels':
                    for label_key, label_value in expected_value.items():
                        if alert.labels.get(label_key) != label_value:
                            return False
                elif key == 'tenant_id':
                    if alert.tenant_id != expected_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur vérification conditions: {e}")
            return False
    
    def _send_notifications(self, alert: Alert):
        """Envoie les notifications pour une alerte"""
        try:
            matching_rules = self._get_matching_notification_rules(alert)
            
            for rule in matching_rules:
                # Vérification du cooldown
                if self._is_in_cooldown(alert, rule):
                    continue
                
                for channel in rule.channels:
                    if channel in self.notification_handlers:
                        self.executor.submit(
                            self._send_notification_with_retry,
                            channel, alert, rule
                        )
            
        except Exception as e:
            logger.error(f"Erreur envoi notifications: {e}")
    
    def _get_matching_notification_rules(self, alert: Alert) -> List[NotificationRule]:
        """Retourne les règles de notification correspondantes"""
        matching_rules = []
        
        for rule in self.notification_rules:
            if not rule.enabled:
                continue
            
            # Vérification sévérité
            if alert.severity not in rule.severity_levels:
                continue
            
            # Vérification tenant
            if rule.tenant_filters and alert.tenant_id not in rule.tenant_filters:
                continue
            
            # Vérification labels
            matches_labels = True
            for key, value in rule.label_filters.items():
                if alert.labels.get(key) != value:
                    matches_labels = False
                    break
            
            if matches_labels:
                matching_rules.append(rule)
        
        return matching_rules
    
    def _is_in_cooldown(self, alert: Alert, rule: NotificationRule) -> bool:
        """Vérifie si l'alerte est dans la période de cooldown"""
        cooldown_key = f"cooldown:{alert.id}:{rule.name}"
        
        if self.redis_client:
            last_notification = self.redis_client.get(cooldown_key)
            if last_notification:
                last_time = float(last_notification)
                if time.time() - last_time < rule.cooldown_minutes * 60:
                    return True
        
        return False
    
    def _send_notification_with_retry(self, channel: str, alert: Alert, rule: NotificationRule):
        """Envoie une notification avec retry"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                handler = self.notification_handlers[channel]
                success = handler(alert, rule)
                
                if success:
                    # Marque le cooldown
                    if self.redis_client:
                        cooldown_key = f"cooldown:{alert.id}:{rule.name}"
                        self.redis_client.setex(
                            cooldown_key,
                            rule.cooldown_minutes * 60,
                            str(time.time())
                        )
                    
                    # Incrémente le compteur
                    with self.lock:
                        alert.notification_count += 1
                        self._save_alert_to_db(alert)
                    
                    logger.info(f"Notification envoyée: {channel} pour {alert.id}")
                    break
                
            except Exception as e:
                logger.error(f"Erreur notification {channel} (tentative {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Échec définitif notification {channel} pour {alert.id}")
                else:
                    time.sleep(2 ** attempt)  # Backoff exponentiel
    
    def _send_slack_notification(self, alert: Alert, rule: NotificationRule) -> bool:
        """Envoie une notification Slack"""
        webhook_url = self.config.get('slack_webhook_url')
        if not webhook_url:
            logger.warning("Slack webhook URL non configurée")
            return False
        
        try:
            color_map = {
                AlertSeverity.INFO.value: "good",
                AlertSeverity.WARNING.value: "warning", 
                AlertSeverity.ERROR.value: "danger",
                AlertSeverity.CRITICAL.value: "danger"
            }
            
            message = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "title": f"🚨 {alert.name}",
                    "text": alert.description,
                    "fields": [
                        {"title": "Sévérité", "value": alert.severity.upper(), "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Tenant", "value": alert.tenant_id or "N/A", "short": True},
                        {"title": "Statut", "value": alert.status, "short": True}
                    ],
                    "footer": "Spotify AI Agent Monitoring",
                    "ts": int(alert.created_at)
                }]
            }
            
            response = requests.post(
                webhook_url,
                json=message,
                timeout=self.config['webhook_timeout']
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Erreur notification Slack: {e}")
            return False
    
    def _send_email_notification(self, alert: Alert, rule: NotificationRule) -> bool:
        """Envoie une notification par email"""
        try:
            smtp_server = self.config.get('email_smtp_server')
            smtp_port = self.config.get('email_smtp_port')
            username = self.config.get('email_username')
            password = self.config.get('email_password')
            
            if not all([smtp_server, username, password]):
                logger.warning("Configuration email incomplète")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = self.config.get('email_recipients', username)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.name}"
            
            body = f"""
Alerte: {alert.name}
Sévérité: {alert.severity.upper()}
Description: {alert.description}
Source: {alert.source}
Tenant: {alert.tenant_id or 'N/A'}
Heure: {datetime.fromtimestamp(alert.created_at)}

Labels:
{json.dumps(alert.labels, indent=2)}

Annotations:
{json.dumps(alert.annotations, indent=2)}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur notification email: {e}")
            return False
    
    def _send_sms_notification(self, alert: Alert, rule: NotificationRule) -> bool:
        """Envoie une notification SMS"""
        try:
            api_url = self.config.get('sms_api_url')
            api_key = self.config.get('sms_api_key')
            phone_numbers = self.config.get('sms_recipients', [])
            
            if not all([api_url, api_key, phone_numbers]):
                logger.warning("Configuration SMS incomplète")
                return False
            
            message = f"ALERTE [{alert.severity.upper()}]: {alert.name} - {alert.description[:100]}"
            
            for phone in phone_numbers:
                response = requests.post(
                    api_url,
                    headers={'Authorization': f'Bearer {api_key}'},
                    json={
                        'to': phone,
                        'message': message
                    },
                    timeout=self.config['webhook_timeout']
                )
                
                if response.status_code != 200:
                    logger.error(f"Erreur SMS vers {phone}: {response.text}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur notification SMS: {e}")
            return False
    
    def _send_pagerduty_notification(self, alert: Alert, rule: NotificationRule) -> bool:
        """Envoie une notification PagerDuty"""
        try:
            api_key = self.config.get('pagerduty_api_key')
            if not api_key:
                logger.warning("Clé API PagerDuty non configurée")
                return False
            
            event_data = {
                "routing_key": api_key,
                "event_action": "trigger",
                "dedup_key": alert.id,
                "payload": {
                    "summary": f"{alert.name}: {alert.description}",
                    "severity": alert.severity,
                    "source": alert.source,
                    "component": "spotify-ai-agent",
                    "group": alert.tenant_id or "system",
                    "class": "monitoring",
                    "custom_details": {
                        "alert_id": alert.id,
                        "tenant_id": alert.tenant_id,
                        "labels": alert.labels,
                        "annotations": alert.annotations
                    }
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=event_data,
                timeout=self.config['webhook_timeout']
            )
            
            return response.status_code == 202
            
        except Exception as e:
            logger.error(f"Erreur notification PagerDuty: {e}")
            return False
    
    def _send_webhook_notification(self, alert: Alert, rule: NotificationRule) -> bool:
        """Envoie une notification webhook générique"""
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                logger.warning("Webhook URL non configurée")
                return False
            
            payload = {
                "alert": asdict(alert),
                "rule": rule.name,
                "timestamp": time.time()
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=self.config['webhook_timeout'],
                headers={'Content-Type': 'application/json'}
            )
            
            return response.status_code in [200, 201, 202]
            
        except Exception as e:
            logger.error(f"Erreur notification webhook: {e}")
            return False
    
    def _processing_loop(self):
        """Boucle principale de traitement des alertes"""
        while self.running:
            try:
                self._process_pending_alerts()
                self._cleanup_old_alerts()
                time.sleep(self.config['processing_interval'])
            except Exception as e:
                logger.error(f"Erreur dans la boucle de traitement: {e}")
                time.sleep(5)
    
    def _escalation_loop(self):
        """Boucle de gestion des escalades"""
        while self.running:
            try:
                self._process_escalations()
                time.sleep(self.config['escalation_interval'])
            except Exception as e:
                logger.error(f"Erreur dans la boucle d'escalade: {e}")
                time.sleep(5)
    
    def _process_pending_alerts(self):
        """Traite les alertes en attente"""
        try:
            with self.lock:
                pending_alerts = [
                    alert for alert in self.alerts.values()
                    if alert.status == AlertStatus.OPEN.value
                ]
            
            for alert in pending_alerts:
                # Auto-résolution basée sur timeout
                auto_resolve_timeout = self.config['auto_resolve_timeout_minutes'] * 60
                if time.time() - alert.created_at > auto_resolve_timeout:
                    self.resolve_alert(alert.id, auto_resolved=True)
                
                # Vérification des silences expirés
                if alert.status == AlertStatus.SILENCED.value:
                    silenced_until = alert.annotations.get('silenced_until')
                    if silenced_until and float(silenced_until) < time.time():
                        alert.status = AlertStatus.OPEN.value
                        alert.updated_at = time.time()
                        del alert.annotations['silenced_until']
                        self._save_alert_to_db(alert)
                        
        except Exception as e:
            logger.error(f"Erreur traitement alertes pending: {e}")
    
    def _process_escalations(self):
        """Traite les escalades d'alertes"""
        try:
            with self.lock:
                escalation_candidates = [
                    alert for alert in self.alerts.values()
                    if (alert.status == AlertStatus.OPEN.value and 
                        not alert.escalated and
                        alert.severity in [AlertSeverity.ERROR.value, AlertSeverity.CRITICAL.value])
                ]
            
            for alert in escalation_candidates:
                # Vérifie si escalade nécessaire
                for rule in self.notification_rules:
                    if (rule.escalation_minutes > 0 and
                        time.time() - alert.created_at > rule.escalation_minutes * 60):
                        
                        self._escalate_alert(alert, rule)
                        break
                        
        except Exception as e:
            logger.error(f"Erreur traitement escalades: {e}")
    
    def _escalate_alert(self, alert: Alert, rule: NotificationRule):
        """Escalade une alerte"""
        try:
            alert.escalated = True
            alert.updated_at = time.time()
            alert.annotations['escalated_at'] = str(time.time())
            alert.annotations['escalated_rule'] = rule.name
            
            # Notification d'escalade
            escalated_alert = Alert(
                id=f"{alert.id}_escalated",
                name=f"ESCALATED: {alert.name}",
                description=f"Alerte escaladée après {rule.escalation_minutes} minutes: {alert.description}",
                severity=AlertSeverity.CRITICAL.value,
                status=AlertStatus.OPEN.value,
                source=alert.source,
                tenant_id=alert.tenant_id,
                labels=alert.labels.copy(),
                annotations=alert.annotations.copy(),
                created_at=time.time(),
                updated_at=time.time()
            )
            
            # Envoi notifications escalade
            self._send_notifications(escalated_alert)
            self._save_alert_to_db(alert)
            
            logger.warning(f"Alerte escaladée: {alert.id}")
            
        except Exception as e:
            logger.error(f"Erreur escalade alerte: {e}")
    
    def _save_alert_to_db(self, alert: Alert):
        """Sauvegarde une alerte en base de données"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO alerts 
                (id, name, description, severity, status, source, tenant_id,
                 labels, annotations, created_at, updated_at, resolved_at,
                 acknowledged_at, acknowledged_by, escalated, notification_count, auto_resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.name, alert.description, alert.severity,
                alert.status, alert.source, alert.tenant_id,
                json.dumps(alert.labels), json.dumps(alert.annotations),
                alert.created_at, alert.updated_at, alert.resolved_at,
                alert.acknowledged_at, alert.acknowledged_by,
                alert.escalated, alert.notification_count, alert.auto_resolved
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde alerte: {e}")
    
    def _load_rules(self):
        """Charge les règles de notification depuis la base"""
        try:
            # Règles par défaut si aucune en base
            default_rules = [
                NotificationRule(
                    name="critical_alerts",
                    channels=[NotificationChannel.SLACK.value, NotificationChannel.EMAIL.value],
                    severity_levels=[AlertSeverity.CRITICAL.value],
                    tenant_filters=[],
                    label_filters={},
                    cooldown_minutes=5,
                    escalation_minutes=15,
                    max_notifications=10
                ),
                NotificationRule(
                    name="error_alerts", 
                    channels=[NotificationChannel.SLACK.value],
                    severity_levels=[AlertSeverity.ERROR.value],
                    tenant_filters=[],
                    label_filters={},
                    cooldown_minutes=10,
                    escalation_minutes=30,
                    max_notifications=5
                ),
                NotificationRule(
                    name="warning_alerts",
                    channels=[NotificationChannel.SLACK.value],
                    severity_levels=[AlertSeverity.WARNING.value],
                    tenant_filters=[],
                    label_filters={},
                    cooldown_minutes=30,
                    escalation_minutes=0,
                    max_notifications=3
                )
            ]
            
            self.notification_rules = default_rules
            logger.info(f"Chargé {len(self.notification_rules)} règles de notification")
            
        except Exception as e:
            logger.error(f"Erreur chargement règles: {e}")
    
    def _load_remediation_actions(self):
        """Charge les actions de remédiation"""
        try:
            # Actions par défaut
            default_actions = [
                RemediationAction(
                    name="restart_service",
                    description="Redémarre un service en cas d'erreur critique",
                    script_path="/opt/scripts/restart_service.sh",
                    conditions={
                        "severity": AlertSeverity.CRITICAL.value,
                        "labels": {"service": "*"}
                    },
                    timeout_seconds=300,
                    max_retries=3
                ),
                RemediationAction(
                    name="scale_up",
                    description="Scale up automatique en cas de charge élevée",
                    script_path="/opt/scripts/scale_up.sh",
                    conditions={
                        "name_pattern": ".*high_load.*",
                        "severity": AlertSeverity.ERROR.value
                    },
                    timeout_seconds=600,
                    max_retries=1
                )
            ]
            
            self.remediation_actions = default_actions
            logger.info(f"Chargé {len(self.remediation_actions)} actions de remédiation")
            
        except Exception as e:
            logger.error(f"Erreur chargement actions remédiation: {e}")
    
    def _cleanup_old_alerts(self):
        """Nettoie les anciennes alertes"""
        try:
            cutoff_time = time.time() - (self.config['max_alert_age_days'] * 24 * 3600)
            
            with self.lock:
                old_alert_ids = [
                    alert_id for alert_id, alert in self.alerts.items()
                    if (alert.status == AlertStatus.RESOLVED.value and 
                        alert.resolved_at and alert.resolved_at < cutoff_time)
                ]
                
                for alert_id in old_alert_ids:
                    del self.alerts[alert_id]
            
            # Nettoyage base de données
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM alerts 
                WHERE status = 'resolved' AND resolved_at < ?
            ''', (cutoff_time,))
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Nettoyé {deleted_count} alertes anciennes")
                
        except Exception as e:
            logger.error(f"Erreur nettoyage alertes: {e}")
    
    def _analyze_alert_patterns(self, alert: Alert):
        """Analyse les patterns d'alertes avec ML"""
        try:
            # TODO: Implémentation ML pour détection d'anomalies
            # - Analyse fréquence des alertes
            # - Détection de patterns temporels
            # - Corrélation entre alertes
            # - Prédiction d'incidents
            pass
            
        except Exception as e:
            logger.error(f"Erreur analyse patterns ML: {e}")
    
    def get_alerts(self, status: Optional[str] = None,
                  severity: Optional[str] = None,
                  tenant_id: Optional[str] = None,
                  limit: int = 100) -> List[Dict]:
        """
        Récupère les alertes selon les critères
        
        Args:
            status: Filtre par statut
            severity: Filtre par sévérité  
            tenant_id: Filtre par tenant
            limit: Limite du nombre d'alertes
            
        Returns:
            Liste des alertes
        """
        try:
            with self.lock:
                alerts = list(self.alerts.values())
            
            # Filtrage
            if status:
                alerts = [a for a in alerts if a.status == status]
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            if tenant_id:
                alerts = [a for a in alerts if a.tenant_id == tenant_id]
            
            # Tri par date de création (plus récent en premier)
            alerts.sort(key=lambda x: x.created_at, reverse=True)
            
            # Limite
            alerts = alerts[:limit]
            
            return [asdict(alert) for alert in alerts]
            
        except Exception as e:
            logger.error(f"Erreur récupération alertes: {e}")
            return []
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des alertes"""
        try:
            with self.lock:
                alerts = list(self.alerts.values())
            
            stats = {
                'total': len(alerts),
                'by_status': {},
                'by_severity': {},
                'by_tenant': {},
                'avg_resolution_time': 0,
                'escalated_count': 0,
                'auto_resolved_count': 0
            }
            
            resolution_times = []
            
            for alert in alerts:
                # Par statut
                stats['by_status'][alert.status] = stats['by_status'].get(alert.status, 0) + 1
                
                # Par sévérité
                stats['by_severity'][alert.severity] = stats['by_severity'].get(alert.severity, 0) + 1
                
                # Par tenant
                tenant = alert.tenant_id or 'system'
                stats['by_tenant'][tenant] = stats['by_tenant'].get(tenant, 0) + 1
                
                # Temps de résolution
                if alert.resolved_at:
                    resolution_times.append(alert.resolved_at - alert.created_at)
                
                # Escalades
                if alert.escalated:
                    stats['escalated_count'] += 1
                
                # Auto-résolutions
                if alert.auto_resolved:
                    stats['auto_resolved_count'] += 1
            
            # Temps moyen de résolution
            if resolution_times:
                stats['avg_resolution_time'] = sum(resolution_times) / len(resolution_times)
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur stats alertes: {e}")
            return {}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Retourne le statut de santé du gestionnaire d'alertes"""
        return {
            'status': 'healthy' if self.running else 'unhealthy',
            'active_alerts': len([a for a in self.alerts.values() if a.status == AlertStatus.OPEN.value]),
            'total_alerts': len(self.alerts),
            'notification_rules': len(self.notification_rules),
            'remediation_actions': len(self.remediation_actions),
            'redis_connected': self.redis_client is not None and self._test_redis(),
            'processor_active': self.processor_thread and self.processor_thread.is_alive(),
            'escalation_active': self.escalation_thread and self.escalation_thread.is_alive(),
            'config': self.config
        }
    
    def _test_redis(self) -> bool:
        """Test la connexion Redis"""
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def _check_auto_resolution(self, alert: Alert, action: RemediationAction):
        """Vérifie si l'alerte peut être auto-résolue après remédiation"""
        try:
            # Attendre un délai pour vérifier si le problème persiste
            time.sleep(60)
            
            # Logique de vérification selon le type d'action
            # TODO: Implémenter des checks spécifiques par action
            
            # Pour l'instant, auto-résolution simple
            self.resolve_alert(alert.id, auto_resolved=True)
            
        except Exception as e:
            logger.error(f"Erreur vérification auto-résolution: {e}")

# Factory pour instance globale
_alert_manager_instance = None

def get_alert_manager(config: Optional[Dict] = None) -> AlertManager:
    """
    Retourne l'instance globale du gestionnaire d'alertes
    
    Args:
        config: Configuration optionnelle
        
    Returns:
        Instance d'AlertManager
    """
    global _alert_manager_instance
    
    if _alert_manager_instance is None:
        _alert_manager_instance = AlertManager(config)
    
    return _alert_manager_instance

# Fonctions de convenance
def trigger_alert(name: str, description: str = "", severity: str = AlertSeverity.WARNING.value,
                 source: str = "system", tenant_id: Optional[str] = None,
                 labels: Optional[Dict[str, str]] = None,
                 annotations: Optional[Dict[str, str]] = None) -> str:
    """Fonction de convenance pour déclencher une alerte"""
    manager = get_alert_manager()
    return manager.trigger_alert(name, description, severity, source, tenant_id, labels, annotations)

def acknowledge_alert(alert_id: str, acknowledged_by: str = "system") -> bool:
    """Fonction de convenance pour acquitter une alerte"""
    manager = get_alert_manager()
    return manager.acknowledge_alert(alert_id, acknowledged_by)

def resolve_alert(alert_id: str, auto_resolved: bool = False) -> bool:
    """Fonction de convenance pour résoudre une alerte"""
    manager = get_alert_manager()
    return manager.resolve_alert(alert_id, auto_resolved)
