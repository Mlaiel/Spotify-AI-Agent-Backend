#!/usr/bin/env python3
"""
Alert Manager for PagerDuty Integration

Gestionnaire avancé d'alertes pour l'intégration PagerDuty.
Fournit des fonctionnalités complètes de gestion des alertes,
routage intelligent, agrégation, et escalade automatique.

Fonctionnalités:
- Gestion centralisée des alertes
- Routage intelligent selon les règles
- Agrégation et déduplication
- Escalade automatique
- Templates de notification
- Statistiques et métriques
- Intégration multi-services

Version: 1.0.0
Auteur: Spotify AI Agent Team
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from utils.api_client import PagerDutyAPIClient, PagerDutyEventAction, PagerDutySeverity
from utils.validators import PagerDutyValidator, ValidationResult
from utils.formatters import MessageFormatter

console = Console()
logger = structlog.get_logger(__name__)

class AlertStatus(Enum):
    """Statuts d'alerte"""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FAILED = "failed"
    SUPPRESSED = "suppressed"

class AlertPriority(Enum):
    """Priorités d'alerte"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5

@dataclass
class Alert:
    """Représentation d'une alerte"""
    id: str
    title: str
    description: str
    severity: str
    source: str
    status: AlertStatus = AlertStatus.PENDING
    priority: AlertPriority = AlertPriority.MEDIUM
    created_at: datetime = None
    updated_at: datetime = None
    resolved_at: Optional[datetime] = None
    incident_key: Optional[str] = None
    tags: List[str] = None
    custom_details: Dict[str, Any] = None
    escalation_level: int = 0
    retry_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.tags is None:
            self.tags = []
        if self.custom_details is None:
            self.custom_details = {}

@dataclass
class AlertRule:
    """Règle de routage d'alerte"""
    name: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: int = 100
    enabled: bool = True
    throttle_minutes: int = 0
    escalation_policy: Optional[str] = None
    notification_template: Optional[str] = None

class AlertAggregator:
    """Agrégateur d'alertes pour éviter le spam"""
    
    def __init__(self, window_minutes: int = 5, max_alerts: int = 10):
        self.window_minutes = window_minutes
        self.max_alerts = max_alerts
        self.alert_groups = {}
    
    def should_aggregate(self, alert: Alert) -> bool:
        """Détermine si l'alerte doit être agrégée"""
        group_key = self._get_group_key(alert)
        now = datetime.now(timezone.utc)
        
        if group_key not in self.alert_groups:
            self.alert_groups[group_key] = {
                "alerts": [],
                "first_seen": now,
                "last_seen": now,
                "count": 0
            }
        
        group = self.alert_groups[group_key]
        
        # Nettoyer les anciennes alertes
        cutoff_time = now - timedelta(minutes=self.window_minutes)
        group["alerts"] = [
            a for a in group["alerts"] 
            if a.created_at > cutoff_time
        ]
        
        group["count"] = len(group["alerts"])
        group["last_seen"] = now
        
        return group["count"] >= self.max_alerts
    
    def add_alert(self, alert: Alert):
        """Ajoute une alerte au groupe"""
        group_key = self._get_group_key(alert)
        
        if group_key in self.alert_groups:
            self.alert_groups[group_key]["alerts"].append(alert)
            self.alert_groups[group_key]["count"] += 1
    
    def _get_group_key(self, alert: Alert) -> str:
        """Génère une clé de groupe pour l'alerte"""
        return f"{alert.source}:{alert.severity}"
    
    def get_aggregated_summary(self, group_key: str) -> Optional[Dict[str, Any]]:
        """Génère un résumé des alertes agrégées"""
        if group_key not in self.alert_groups:
            return None
        
        group = self.alert_groups[group_key]
        alerts = group["alerts"]
        
        if not alerts:
            return None
        
        return {
            "total_count": group["count"],
            "unique_sources": len(set(a.source for a in alerts)),
            "severity_counts": {
                severity: len([a for a in alerts if a.severity == severity])
                for severity in set(a.severity for a in alerts)
            },
            "time_window": {
                "start": group["first_seen"].isoformat(),
                "end": group["last_seen"].isoformat()
            },
            "sample_alerts": [asdict(a) for a in alerts[:3]]
        }

class RuleEngine:
    """Moteur de règles pour le routage des alertes"""
    
    def __init__(self, rules_file: Optional[str] = None):
        self.rules = []
        if rules_file:
            self.load_rules(rules_file)
    
    def load_rules(self, rules_file: str):
        """Charge les règles depuis un fichier"""
        try:
            with open(rules_file, 'r') as f:
                if rules_file.endswith('.json'):
                    rules_data = json.load(f)
                else:
                    rules_data = yaml.safe_load(f)
            
            self.rules = []
            for rule_data in rules_data.get("rules", []):
                rule = AlertRule(**rule_data)
                self.rules.append(rule)
            
            # Trier par priorité
            self.rules.sort(key=lambda r: r.priority)
            
            logger.info(f"Loaded {len(self.rules)} alert rules")
            
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            raise
    
    def evaluate_alert(self, alert: Alert) -> List[AlertRule]:
        """Évalue une alerte contre toutes les règles"""
        matching_rules = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if self._matches_conditions(alert, rule.conditions):
                matching_rules.append(rule)
        
        return matching_rules
    
    def _matches_conditions(self, alert: Alert, conditions: Dict[str, Any]) -> bool:
        """Vérifie si l'alerte correspond aux conditions"""
        for field, condition in conditions.items():
            if not self._evaluate_condition(alert, field, condition):
                return False
        
        return True
    
    def _evaluate_condition(self, alert: Alert, field: str, condition: Any) -> bool:
        """Évalue une condition individuelle"""
        # Récupérer la valeur du champ
        if hasattr(alert, field):
            value = getattr(alert, field)
        elif field in alert.custom_details:
            value = alert.custom_details[field]
        else:
            return False
        
        # Évaluer selon le type de condition
        if isinstance(condition, str):
            return str(value) == condition
        
        elif isinstance(condition, dict):
            operator = condition.get("operator", "eq")
            expected = condition.get("value")
            
            if operator == "eq":
                return value == expected
            elif operator == "ne":
                return value != expected
            elif operator == "in":
                return value in expected
            elif operator == "contains":
                return expected in str(value)
            elif operator == "regex":
                import re
                return bool(re.search(expected, str(value)))
            elif operator == "gt":
                return float(value) > float(expected)
            elif operator == "gte":
                return float(value) >= float(expected)
            elif operator == "lt":
                return float(value) < float(expected)
            elif operator == "lte":
                return float(value) <= float(expected)
        
        return False

class NotificationThrottler:
    """Gestionnaire de throttling des notifications"""
    
    def __init__(self):
        self.last_sent = {}
    
    def should_send(self, rule: AlertRule, alert: Alert) -> bool:
        """Détermine si la notification doit être envoyée"""
        if rule.throttle_minutes <= 0:
            return True
        
        throttle_key = f"{rule.name}:{alert.source}:{alert.severity}"
        now = datetime.now(timezone.utc)
        
        if throttle_key in self.last_sent:
            time_diff = now - self.last_sent[throttle_key]
            if time_diff.total_seconds() < (rule.throttle_minutes * 60):
                return False
        
        self.last_sent[throttle_key] = now
        return True

class PagerDutyAlertManager:
    """Gestionnaire principal d'alertes PagerDuty"""
    
    def __init__(
        self,
        api_key: str,
        integration_key: str,
        rules_file: Optional[str] = None,
        enable_aggregation: bool = True
    ):
        self.api_client = PagerDutyAPIClient(api_key, integration_key)
        self.rule_engine = RuleEngine(rules_file)
        self.aggregator = AlertAggregator() if enable_aggregation else None
        self.throttler = NotificationThrottler()
        self.formatter = MessageFormatter()
        
        self.alerts = {}  # Store des alertes
        self.metrics = {
            "alerts_received": 0,
            "alerts_sent": 0,
            "alerts_suppressed": 0,
            "alerts_failed": 0,
            "rules_matched": 0
        }
    
    async def process_alert(self, alert: Alert) -> bool:
        """Traite une alerte"""
        self.metrics["alerts_received"] += 1
        self.alerts[alert.id] = alert
        
        logger.info(f"Processing alert: {alert.id} - {alert.title}")
        
        try:
            # Vérifier l'agrégation
            if self.aggregator and self.aggregator.should_aggregate(alert):
                alert.status = AlertStatus.SUPPRESSED
                self.metrics["alerts_suppressed"] += 1
                logger.info(f"Alert {alert.id} suppressed due to aggregation")
                return True
            
            if self.aggregator:
                self.aggregator.add_alert(alert)
            
            # Évaluer les règles
            matching_rules = self.rule_engine.evaluate_alert(alert)
            
            if not matching_rules:
                logger.warning(f"No rules matched for alert {alert.id}")
                return False
            
            self.metrics["rules_matched"] += len(matching_rules)
            
            # Traiter chaque règle correspondante
            for rule in matching_rules:
                if not self.throttler.should_send(rule, alert):
                    logger.info(f"Alert {alert.id} throttled by rule {rule.name}")
                    continue
                
                success = await self._execute_rule_actions(alert, rule)
                if success:
                    alert.status = AlertStatus.SENT
                    self.metrics["alerts_sent"] += 1
                else:
                    alert.status = AlertStatus.FAILED
                    self.metrics["alerts_failed"] += 1
            
            alert.updated_at = datetime.now(timezone.utc)
            return True
            
        except Exception as e:
            logger.error(f"Failed to process alert {alert.id}: {e}")
            alert.status = AlertStatus.FAILED
            self.metrics["alerts_failed"] += 1
            return False
    
    async def _execute_rule_actions(self, alert: Alert, rule: AlertRule) -> bool:
        """Exécute les actions d'une règle"""
        success = True
        
        for action in rule.actions:
            action_type = action.get("type")
            
            try:
                if action_type == "pagerduty_event":
                    await self._send_pagerduty_event(alert, rule, action)
                elif action_type == "escalate":
                    await self._escalate_alert(alert, action)
                elif action_type == "notify":
                    await self._send_notification(alert, rule, action)
                else:
                    logger.warning(f"Unknown action type: {action_type}")
                    
            except Exception as e:
                logger.error(f"Failed to execute action {action_type}: {e}")
                success = False
        
        return success
    
    async def _send_pagerduty_event(self, alert: Alert, rule: AlertRule, action: Dict[str, Any]):
        """Envoie un événement à PagerDuty"""
        
        # Mapper la sévérité
        severity_mapping = {
            "critical": PagerDutySeverity.CRITICAL,
            "high": PagerDutySeverity.ERROR,
            "medium": PagerDutySeverity.WARNING,
            "low": PagerDutySeverity.INFO,
            "info": PagerDutySeverity.INFO
        }
        
        severity = severity_mapping.get(alert.severity.lower(), PagerDutySeverity.WARNING)
        
        # Formater le message
        summary = self.formatter.format_alert_message(
            alert.severity,
            alert.title,
            alert.source,
            alert.created_at,
            alert.custom_details
        )
        
        # Envoyer l'événement
        response = await self.api_client.send_event(
            action=PagerDutyEventAction.TRIGGER,
            summary=summary,
            source=alert.source,
            severity=severity,
            dedup_key=alert.id,
            timestamp=alert.created_at,
            custom_details={
                **alert.custom_details,
                "alert_id": alert.id,
                "rule_name": rule.name,
                "priority": alert.priority.name
            }
        )
        
        if response.status_code == 202:
            alert.incident_key = response.data.get("dedup_key")
            logger.info(f"PagerDuty event sent for alert {alert.id}")
        else:
            raise Exception(f"PagerDuty API error: {response.error}")
    
    async def _escalate_alert(self, alert: Alert, action: Dict[str, Any]):
        """Escalade une alerte"""
        alert.escalation_level += 1
        escalation_delay = action.get("delay_minutes", 30)
        
        logger.info(f"Alert {alert.id} escalated to level {alert.escalation_level}")
        
        # Programmer la ré-évaluation après le délai
        asyncio.create_task(self._schedule_escalation(alert, escalation_delay))
    
    async def _schedule_escalation(self, alert: Alert, delay_minutes: int):
        """Programme une escalade différée"""
        await asyncio.sleep(delay_minutes * 60)
        
        # Vérifier si l'alerte est toujours active
        if alert.id in self.alerts and alert.status not in [AlertStatus.RESOLVED, AlertStatus.ACKNOWLEDGED]:
            logger.info(f"Re-processing escalated alert {alert.id}")
            await self.process_alert(alert)
    
    async def _send_notification(self, alert: Alert, rule: AlertRule, action: Dict[str, Any]):
        """Envoie une notification personnalisée"""
        notification_type = action.get("notification_type", "email")
        recipients = action.get("recipients", [])
        
        message = self.formatter.format_alert_message(
            alert.severity,
            alert.title,
            alert.source,
            alert.created_at,
            alert.custom_details
        )
        
        logger.info(f"Sending {notification_type} notification for alert {alert.id} to {len(recipients)} recipients")
        
        # Ici, on pourrait intégrer avec d'autres systèmes de notification
        # Email, Slack, Teams, etc.
    
    async def acknowledge_alert(self, alert_id: str, from_email: str) -> bool:
        """Acknowledge une alerte"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        
        if alert.incident_key:
            try:
                response = await self.api_client.send_event(
                    action=PagerDutyEventAction.ACKNOWLEDGE,
                    summary=alert.title,
                    source=alert.source,
                    severity=PagerDutySeverity.INFO,
                    dedup_key=alert.incident_key
                )
                
                if response.status_code == 202:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.updated_at = datetime.now(timezone.utc)
                    logger.info(f"Alert {alert_id} acknowledged")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
        
        return False
    
    async def resolve_alert(self, alert_id: str, from_email: str) -> bool:
        """Résout une alerte"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        
        if alert.incident_key:
            try:
                response = await self.api_client.send_event(
                    action=PagerDutyEventAction.RESOLVE,
                    summary=alert.title,
                    source=alert.source,
                    severity=PagerDutySeverity.INFO,
                    dedup_key=alert.incident_key
                )
                
                if response.status_code == 202:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.now(timezone.utc)
                    alert.updated_at = alert.resolved_at
                    logger.info(f"Alert {alert_id} resolved")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to resolve alert {alert_id}: {e}")
        
        return False
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des alertes"""
        now = datetime.now(timezone.utc)
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        alerts_list = list(self.alerts.values())
        
        stats = {
            "total_alerts": len(alerts_list),
            "alerts_last_hour": len([a for a in alerts_list if a.created_at > last_hour]),
            "alerts_last_day": len([a for a in alerts_list if a.created_at > last_day]),
            "status_breakdown": {},
            "severity_breakdown": {},
            "source_breakdown": {},
            "metrics": self.metrics
        }
        
        # Breakdown par statut
        for status in AlertStatus:
            stats["status_breakdown"][status.value] = len([
                a for a in alerts_list if a.status == status
            ])
        
        # Breakdown par sévérité
        severities = set(a.severity for a in alerts_list)
        for severity in severities:
            stats["severity_breakdown"][severity] = len([
                a for a in alerts_list if a.severity == severity
            ])
        
        # Breakdown par source
        sources = set(a.source for a in alerts_list)
        for source in list(sources)[:10]:  # Top 10 sources
            stats["source_breakdown"][source] = len([
                a for a in alerts_list if a.source == source
            ])
        
        return stats
    
    async def cleanup_old_alerts(self, max_age_days: int = 7) -> int:
        """Nettoie les anciennes alertes"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        
        old_alert_ids = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.created_at < cutoff_date and alert.status == AlertStatus.RESOLVED
        ]
        
        for alert_id in old_alert_ids:
            del self.alerts[alert_id]
        
        logger.info(f"Cleaned up {len(old_alert_ids)} old alerts")
        return len(old_alert_ids)
    
    async def close(self):
        """Ferme les connexions"""
        await self.api_client.close()

async def main():
    """Fonction principale CLI"""
    parser = argparse.ArgumentParser(description="PagerDuty Alert Manager")
    parser.add_argument("--action", required=True,
                       choices=["send", "acknowledge", "resolve", "stats", "cleanup"],
                       help="Action à effectuer")
    parser.add_argument("--api-key", required=True, help="Clé API PagerDuty")
    parser.add_argument("--integration-key", help="Clé d'intégration PagerDuty")
    parser.add_argument("--rules-file", help="Fichier de règles")
    parser.add_argument("--alert-id", help="ID de l'alerte")
    parser.add_argument("--from-email", help="Email de l'expéditeur")
    
    # Paramètres pour créer une alerte
    parser.add_argument("--title", help="Titre de l'alerte")
    parser.add_argument("--description", help="Description de l'alerte")
    parser.add_argument("--severity", choices=["critical", "high", "medium", "low", "info"],
                       default="medium", help="Sévérité de l'alerte")
    parser.add_argument("--source", help="Source de l'alerte")
    parser.add_argument("--tags", nargs="*", help="Tags de l'alerte")
    parser.add_argument("--custom-details", help="Détails personnalisés (JSON)")
    
    args = parser.parse_args()
    
    try:
        manager = PagerDutyAlertManager(
            api_key=args.api_key,
            integration_key=args.integration_key,
            rules_file=args.rules_file
        )
        
        if args.action == "send":
            if not all([args.title, args.source]):
                console.print("[red]Title and source required for sending alert[/red]")
                return 1
            
            # Créer l'alerte
            alert = Alert(
                id=f"alert-{int(datetime.now().timestamp())}",
                title=args.title,
                description=args.description or "",
                severity=args.severity,
                source=args.source,
                tags=args.tags or [],
                custom_details=json.loads(args.custom_details) if args.custom_details else {}
            )
            
            success = await manager.process_alert(alert)
            
            if success:
                console.print(f"[green]Alert sent successfully: {alert.id}[/green]")
            else:
                console.print(f"[red]Failed to send alert: {alert.id}[/red]")
                return 1
        
        elif args.action == "acknowledge":
            if not args.alert_id or not args.from_email:
                console.print("[red]Alert ID and from-email required for acknowledge[/red]")
                return 1
            
            success = await manager.acknowledge_alert(args.alert_id, args.from_email)
            
            if success:
                console.print(f"[green]Alert {args.alert_id} acknowledged[/green]")
            else:
                console.print(f"[red]Failed to acknowledge alert {args.alert_id}[/red]")
                return 1
        
        elif args.action == "resolve":
            if not args.alert_id or not args.from_email:
                console.print("[red]Alert ID and from-email required for resolve[/red]")
                return 1
            
            success = await manager.resolve_alert(args.alert_id, args.from_email)
            
            if success:
                console.print(f"[green]Alert {args.alert_id} resolved[/green]")
            else:
                console.print(f"[red]Failed to resolve alert {args.alert_id}[/red]")
                return 1
        
        elif args.action == "stats":
            stats = manager.get_alert_stats()
            
            console.print(Panel.fit(
                f"Total Alerts: {stats['total_alerts']}\n"
                f"Last Hour: {stats['alerts_last_hour']}\n"
                f"Last Day: {stats['alerts_last_day']}",
                title="Alert Statistics"
            ))
            
            # Tableau des statuts
            status_table = Table(title="Status Breakdown")
            status_table.add_column("Status", style="bold")
            status_table.add_column("Count", justify="right")
            
            for status, count in stats["status_breakdown"].items():
                status_table.add_row(status.title(), str(count))
            
            console.print(status_table)
            
            # Tableau des sévérités
            if stats["severity_breakdown"]:
                severity_table = Table(title="Severity Breakdown")
                severity_table.add_column("Severity", style="bold")
                severity_table.add_column("Count", justify="right")
                
                for severity, count in stats["severity_breakdown"].items():
                    severity_table.add_row(severity.title(), str(count))
                
                console.print(severity_table)
        
        elif args.action == "cleanup":
            cleaned = await manager.cleanup_old_alerts()
            console.print(f"[green]Cleaned up {cleaned} old alerts[/green]")
        
        await manager.close()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
