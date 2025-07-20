"""
Moteur de monitoring et gestionnaire d'alertes avancé.

Ce module fournit un système complet de monitoring avec collecte de métriques,
génération d'alertes intelligentes et intégration multi-canal.
"""

import time
import json
import yaml
import asyncio
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..schemas.monitoring_schemas import MonitoringConfigSchema, MetricConfigSchema
from ..schemas.alert_schemas import AlertRuleSchema, AlertManagerConfigSchema
from ..schemas.slack_schemas import SlackConfigSchema, SlackMessageSchema


class MetricCollectionStatus(str, Enum):
    """États de collecte de métriques."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PAUSED = "paused"


class AlertState(str, Enum):
    """États des alertes."""
    FIRING = "firing"
    PENDING = "pending"
    RESOLVED = "resolved"
    SILENCED = "silenced"


class MonitoringTarget(BaseModel):
    """Cible de monitoring."""
    name: str
    url: str
    type: str = "prometheus"
    interval: int = 30
    timeout: int = 10
    labels: Dict[str, str] = Field(default_factory=dict)
    auth: Optional[Dict[str, str]] = None
    ssl_verify: bool = True


class MetricCollector:
    """Collecteur de métriques."""
    
    def __init__(self, name: str, target: MonitoringTarget):
        """Initialise le collecteur."""
        self.name = name
        self.target = target
        self.status = MetricCollectionStatus.INACTIVE
        self.last_collection = None
        self.error_count = 0
        self.metrics_cache: Dict[str, Any] = {}
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collecte les métriques depuis la cible."""
        try:
            self.status = MetricCollectionStatus.ACTIVE
            
            # Configuration de la requête
            headers = {'Accept': 'application/json'}
            auth = None
            
            if self.target.auth:
                if 'bearer_token' in self.target.auth:
                    headers['Authorization'] = f"Bearer {self.target.auth['bearer_token']}"
                elif 'username' in self.target.auth and 'password' in self.target.auth:
                    auth = (self.target.auth['username'], self.target.auth['password'])
            
            # Requête vers la cible
            response = requests.get(
                self.target.url,
                headers=headers,
                auth=auth,
                timeout=self.target.timeout,
                verify=self.target.ssl_verify
            )
            
            response.raise_for_status()
            
            # Parsing des métriques selon le type
            if self.target.type == "prometheus":
                metrics = self._parse_prometheus_metrics(response.text)
            elif self.target.type == "json":
                metrics = response.json()
            else:
                metrics = {"raw_data": response.text}
            
            # Enrichissement avec les labels
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict):
                    metric_data.update(self.target.labels)
            
            self.metrics_cache = metrics
            self.last_collection = datetime.now()
            self.error_count = 0
            
            return metrics
            
        except Exception as e:
            self.status = MetricCollectionStatus.ERROR
            self.error_count += 1
            raise Exception(f"Erreur de collecte pour {self.name}: {e}")
    
    def _parse_prometheus_metrics(self, content: str) -> Dict[str, Any]:
        """Parse les métriques Prometheus."""
        metrics = {}
        
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        metric_name = parts[0]
                        metric_value = float(parts[1])
                        
                        # Extraction des labels
                        if '{' in metric_name:
                            name_part = metric_name.split('{')[0]
                            labels_part = metric_name.split('{')[1].rstrip('}')
                            labels = self._parse_labels(labels_part)
                        else:
                            name_part = metric_name
                            labels = {}
                        
                        if name_part not in metrics:
                            metrics[name_part] = []
                        
                        metrics[name_part].append({
                            'value': metric_value,
                            'labels': labels,
                            'timestamp': datetime.now().isoformat()
                        })
                except ValueError:
                    continue
        
        return metrics
    
    def _parse_labels(self, labels_str: str) -> Dict[str, str]:
        """Parse les labels Prometheus."""
        labels = {}
        if not labels_str:
            return labels
        
        import re
        pattern = r'(\w+)="([^"]*)"'
        matches = re.findall(pattern, labels_str)
        
        for key, value in matches:
            labels[key] = value
        
        return labels


class AlertEvaluator:
    """Évaluateur d'alertes."""
    
    def __init__(self):
        """Initialise l'évaluateur."""
        self.rules: List[AlertRuleSchema] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
    
    def add_rule(self, rule: AlertRuleSchema):
        """Ajoute une règle d'alerte."""
        self.rules.append(rule)
    
    def evaluate_rules(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Évalue toutes les règles d'alerte contre les métriques."""
        triggered_alerts = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                alert_triggered = self._evaluate_rule(rule, metrics)
                if alert_triggered:
                    triggered_alerts.append(alert_triggered)
            except Exception as e:
                # Log de l'erreur mais continue l'évaluation
                print(f"Erreur lors de l'évaluation de la règle {rule.name}: {e}")
        
        return triggered_alerts
    
    def _evaluate_rule(self, rule: AlertRuleSchema, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Évalue une règle d'alerte spécifique."""
        # Évaluation de chaque condition
        for condition in rule.conditions:
            metric_name = condition.metric_name
            
            if metric_name not in metrics:
                continue
            
            metric_data = metrics[metric_name]
            
            # Extraction de la valeur selon le type de données
            if isinstance(metric_data, list) and metric_data:
                # Prise de la dernière valeur pour les séries temporelles
                current_value = metric_data[-1].get('value', 0)
            elif isinstance(metric_data, dict):
                current_value = metric_data.get('value', 0)
            elif isinstance(metric_data, (int, float)):
                current_value = metric_data
            else:
                continue
            
            # Évaluation du seuil
            threshold_value = condition.threshold.value
            operator = condition.threshold.operator
            
            alert_triggered = self._check_threshold(current_value, threshold_value, operator)
            
            if alert_triggered:
                alert_key = f"{rule.name}_{metric_name}"
                
                # Vérification si l'alerte existe déjà
                if alert_key in self.active_alerts:
                    # Mise à jour de l'alerte existante
                    self.active_alerts[alert_key]['last_seen'] = datetime.now()
                    self.active_alerts[alert_key]['value'] = current_value
                else:
                    # Nouvelle alerte
                    alert = {
                        'rule_name': rule.name,
                        'metric_name': metric_name,
                        'severity': rule.severity,
                        'value': current_value,
                        'threshold': threshold_value,
                        'operator': operator,
                        'state': AlertState.FIRING,
                        'started_at': datetime.now(),
                        'last_seen': datetime.now(),
                        'channels': [channel.name for channel in rule.channels],
                        'description': rule.description,
                        'tenant_id': rule.tenant_id,
                        'environment': rule.environment
                    }
                    
                    self.active_alerts[alert_key] = alert
                    self.alert_history.append(alert.copy())
                    
                    return alert
        
        return None
    
    def _check_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Vérifie si un seuil est dépassé."""
        if operator == "gt":
            return value > threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "lte":
            return value <= threshold
        elif operator == "eq":
            return value == threshold
        elif operator == "ne":
            return value != threshold
        else:
            return False
    
    def resolve_alert(self, alert_key: str):
        """Résout une alerte."""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert['state'] = AlertState.RESOLVED
            alert['resolved_at'] = datetime.now()
            
            # Déplacement vers l'historique
            self.alert_history.append(alert.copy())
            del self.active_alerts[alert_key]
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère les alertes actives avec filtrage optionnel."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        
        return sorted(alerts, key=lambda x: x['started_at'], reverse=True)


class NotificationChannel:
    """Canal de notification abstrait."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialise le canal."""
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.rate_limit = config.get('rate_limit', 60)  # messages par minute
        self.last_sent = {}
    
    async def send_notification(self, alert: Dict[str, Any]) -> bool:
        """Envoie une notification (à implémenter dans les sous-classes)."""
        raise NotImplementedError
    
    def _check_rate_limit(self, alert_key: str) -> bool:
        """Vérifie les limites de débit."""
        now = datetime.now()
        
        if alert_key in self.last_sent:
            time_diff = (now - self.last_sent[alert_key]).total_seconds()
            if time_diff < (60 / self.rate_limit):
                return False
        
        self.last_sent[alert_key] = now
        return True


class SlackNotificationChannel(NotificationChannel):
    """Canal de notification Slack."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialise le canal Slack."""
        super().__init__(name, config)
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'AlertBot')
        self.icon_emoji = config.get('icon_emoji', ':warning:')
    
    async def send_notification(self, alert: Dict[str, Any]) -> bool:
        """Envoie une notification Slack."""
        if not self.enabled or not self.webhook_url:
            return False
        
        alert_key = f"{alert['rule_name']}_{alert['metric_name']}"
        
        if not self._check_rate_limit(alert_key):
            return False
        
        # Construction du message Slack
        color = self._get_color_for_severity(alert['severity'])
        
        message = {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [{
                "color": color,
                "title": f"🚨 Alerte: {alert['rule_name']}",
                "text": alert.get('description', 'Aucune description'),
                "fields": [
                    {
                        "title": "Métrique",
                        "value": alert['metric_name'],
                        "short": True
                    },
                    {
                        "title": "Valeur",
                        "value": f"{alert['value']} (seuil: {alert['threshold']})",
                        "short": True
                    },
                    {
                        "title": "Sévérité",
                        "value": alert['severity'].upper(),
                        "short": True
                    },
                    {
                        "title": "Environnement",
                        "value": alert.get('environment', 'N/A'),
                        "short": True
                    }
                ],
                "footer": "Système de Monitoring",
                "ts": int(alert['started_at'].timestamp())
            }]
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Erreur envoi Slack: {e}")
            return False
    
    def _get_color_for_severity(self, severity: str) -> str:
        """Retourne la couleur selon la sévérité."""
        colors = {
            'critical': 'danger',
            'high': '#ff9900',
            'medium': 'warning',
            'low': '#0099ff',
            'info': 'good'
        }
        return colors.get(severity.lower(), '#cccccc')


class EmailNotificationChannel(NotificationChannel):
    """Canal de notification email."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialise le canal email."""
        super().__init__(name, config)
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_address = config.get('from_address')
        self.to_addresses = config.get('to_addresses', [])
    
    async def send_notification(self, alert: Dict[str, Any]) -> bool:
        """Envoie une notification email."""
        if not self.enabled or not self.smtp_server:
            return False
        
        alert_key = f"{alert['rule_name']}_{alert['metric_name']}"
        
        if not self._check_rate_limit(alert_key):
            return False
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Construction du message
            msg = MIMEMultipart()
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.to_addresses)
            msg['Subject'] = f"Alerte: {alert['rule_name']} - {alert['severity'].upper()}"
            
            body = f"""
            Alerte déclenchée: {alert['rule_name']}
            
            Métrique: {alert['metric_name']}
            Valeur actuelle: {alert['value']}
            Seuil configuré: {alert['threshold']}
            Sévérité: {alert['severity'].upper()}
            Environnement: {alert.get('environment', 'N/A')}
            Tenant: {alert.get('tenant_id', 'N/A')}
            
            Description: {alert.get('description', 'Aucune description')}
            
            Déclenchée le: {alert['started_at']}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Envoi
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"Erreur envoi email: {e}")
            return False


class AlertManager:
    """Gestionnaire d'alertes principal."""
    
    def __init__(self):
        """Initialise le gestionnaire d'alertes."""
        self.evaluator = AlertEvaluator()
        self.channels: Dict[str, NotificationChannel] = {}
        self.notification_queue = asyncio.Queue()
        self.running = False
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Ajoute un canal de notification."""
        self.channels[channel.name] = channel
    
    def configure_slack_channel(self, name: str, webhook_url: str, **kwargs):
        """Configure un canal Slack."""
        config = {
            'webhook_url': webhook_url,
            'enabled': True,
            **kwargs
        }
        channel = SlackNotificationChannel(name, config)
        self.add_notification_channel(channel)
    
    def configure_email_channel(self, name: str, smtp_config: Dict[str, Any]):
        """Configure un canal email."""
        channel = EmailNotificationChannel(name, smtp_config)
        self.add_notification_channel(channel)
    
    async def process_alert(self, alert: Dict[str, Any]):
        """Traite une alerte et envoie les notifications."""
        # Détermine les canaux de notification
        channels_to_notify = []
        
        if 'channels' in alert and alert['channels']:
            # Canaux spécifiés dans l'alerte
            for channel_name in alert['channels']:
                if channel_name in self.channels:
                    channels_to_notify.append(self.channels[channel_name])
        else:
            # Tous les canaux par défaut
            channels_to_notify = list(self.channels.values())
        
        # Envoi des notifications en parallèle
        tasks = []
        for channel in channels_to_notify:
            if channel.enabled:
                task = asyncio.create_task(channel.send_notification(alert))
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log des résultats
            success_count = sum(1 for r in results if r is True)
            total_count = len(results)
            
            print(f"Notifications envoyées: {success_count}/{total_count} pour {alert['rule_name']}")
    
    async def start_notification_processor(self):
        """Démarre le processeur de notifications."""
        self.running = True
        
        while self.running:
            try:
                # Récupération d'une alerte de la queue
                alert = await asyncio.wait_for(
                    self.notification_queue.get(),
                    timeout=1.0
                )
                
                # Traitement de l'alerte
                await self.process_alert(alert)
                self.notification_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Erreur dans le processeur de notifications: {e}")
    
    def stop_notification_processor(self):
        """Arrête le processeur de notifications."""
        self.running = False
    
    async def queue_alert(self, alert: Dict[str, Any]):
        """Ajoute une alerte à la queue de notification."""
        await self.notification_queue.put(alert)


class MonitoringEngine:
    """Moteur de monitoring principal."""
    
    def __init__(self):
        """Initialise le moteur de monitoring."""
        self.collectors: Dict[str, MetricCollector] = {}
        self.alert_manager = AlertManager()
        self.running = False
        self.collection_interval = 30
        self.metrics_storage: Dict[str, List[Dict[str, Any]]] = {}
        self.max_metrics_history = 1000
    
    def add_target(self, target: MonitoringTarget):
        """Ajoute une cible de monitoring."""
        collector = MetricCollector(target.name, target)
        self.collectors[target.name] = collector
    
    def add_alert_rule(self, rule: AlertRuleSchema):
        """Ajoute une règle d'alerte."""
        self.alert_manager.evaluator.add_rule(rule)
    
    def configure_notifications(self, config: Dict[str, Any]):
        """Configure les canaux de notification."""
        # Configuration Slack
        if 'slack' in config:
            slack_config = config['slack']
            self.alert_manager.configure_slack_channel(
                'slack',
                slack_config['webhook_url'],
                channel=slack_config.get('channel', '#alerts'),
                username=slack_config.get('username', 'AlertBot')
            )
        
        # Configuration Email
        if 'email' in config:
            email_config = config['email']
            self.alert_manager.configure_email_channel('email', email_config)
    
    async def start_monitoring(self):
        """Démarre le monitoring."""
        self.running = True
        
        # Démarrage du processeur de notifications
        notification_task = asyncio.create_task(
            self.alert_manager.start_notification_processor()
        )
        
        # Boucle principale de collecte
        try:
            while self.running:
                await self._collect_and_evaluate()
                await asyncio.sleep(self.collection_interval)
        finally:
            self.alert_manager.stop_notification_processor()
            await notification_task
    
    def stop_monitoring(self):
        """Arrête le monitoring."""
        self.running = False
    
    async def _collect_and_evaluate(self):
        """Collecte les métriques et évalue les alertes."""
        # Collecte des métriques en parallèle
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(collector.collect_metrics): name
                for name, collector in self.collectors.items()
                if collector.status != MetricCollectionStatus.PAUSED
            }
            
            all_metrics = {}
            
            for future in as_completed(futures):
                collector_name = futures[future]
                try:
                    metrics = future.result()
                    all_metrics.update(metrics)
                    
                    # Stockage des métriques
                    self._store_metrics(collector_name, metrics)
                    
                except Exception as e:
                    print(f"Erreur de collecte pour {collector_name}: {e}")
        
        # Évaluation des alertes
        if all_metrics:
            triggered_alerts = self.alert_manager.evaluator.evaluate_rules(all_metrics)
            
            # Envoi des alertes à la queue de notification
            for alert in triggered_alerts:
                await self.alert_manager.queue_alert(alert)
    
    def _store_metrics(self, collector_name: str, metrics: Dict[str, Any]):
        """Stocke les métriques avec limitation de l'historique."""
        timestamp = datetime.now()
        
        for metric_name, metric_value in metrics.items():
            key = f"{collector_name}_{metric_name}"
            
            if key not in self.metrics_storage:
                self.metrics_storage[key] = []
            
            self.metrics_storage[key].append({
                'timestamp': timestamp,
                'value': metric_value,
                'collector': collector_name
            })
            
            # Limitation de l'historique
            if len(self.metrics_storage[key]) > self.max_metrics_history:
                self.metrics_storage[key] = self.metrics_storage[key][-self.max_metrics_history:]
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        collector_name: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Récupère les métriques avec filtrage."""
        filtered_metrics = {}
        
        for key, values in self.metrics_storage.items():
            # Filtrage par nom de métrique
            if metric_name and metric_name not in key:
                continue
            
            # Filtrage par collecteur
            if collector_name and not key.startswith(collector_name):
                continue
            
            # Filtrage par date
            if since:
                values = [v for v in values if v['timestamp'] >= since]
            
            if values:
                filtered_metrics[key] = values
        
        return filtered_metrics
    
    def get_collector_status(self) -> Dict[str, Dict[str, Any]]:
        """Récupère le statut de tous les collecteurs."""
        status = {}
        
        for name, collector in self.collectors.items():
            status[name] = {
                'status': collector.status,
                'last_collection': collector.last_collection,
                'error_count': collector.error_count,
                'target_url': collector.target.url,
                'interval': collector.target.interval
            }
        
        return status
    
    def get_active_alerts_summary(self) -> Dict[str, Any]:
        """Récupère un résumé des alertes actives."""
        active_alerts = self.alert_manager.evaluator.get_active_alerts()
        
        summary = {
            'total_alerts': len(active_alerts),
            'by_severity': {},
            'by_environment': {},
            'oldest_alert': None,
            'newest_alert': None
        }
        
        if active_alerts:
            # Comptage par sévérité
            for alert in active_alerts:
                severity = alert['severity']
                summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
                
                environment = alert.get('environment', 'unknown')
                summary['by_environment'][environment] = summary['by_environment'].get(environment, 0) + 1
            
            # Alertes les plus anciennes et récentes
            sorted_alerts = sorted(active_alerts, key=lambda x: x['started_at'])
            summary['oldest_alert'] = sorted_alerts[0]
            summary['newest_alert'] = sorted_alerts[-1]
        
        return summary


# Factory functions
def create_monitoring_engine() -> MonitoringEngine:
    """Crée un moteur de monitoring."""
    return MonitoringEngine()


def create_alert_manager() -> AlertManager:
    """Crée un gestionnaire d'alertes."""
    return AlertManager()


async def setup_basic_monitoring(
    prometheus_url: str = "http://localhost:9090/metrics",
    slack_webhook: Optional[str] = None
) -> MonitoringEngine:
    """Configure un monitoring de base."""
    engine = create_monitoring_engine()
    
    # Ajout d'une cible Prometheus
    target = MonitoringTarget(
        name="prometheus",
        url=prometheus_url,
        type="prometheus",
        interval=30
    )
    engine.add_target(target)
    
    # Configuration des notifications
    notification_config = {}
    if slack_webhook:
        notification_config['slack'] = {
            'webhook_url': slack_webhook,
            'channel': '#monitoring'
        }
    
    if notification_config:
        engine.configure_notifications(notification_config)
    
    return engine
