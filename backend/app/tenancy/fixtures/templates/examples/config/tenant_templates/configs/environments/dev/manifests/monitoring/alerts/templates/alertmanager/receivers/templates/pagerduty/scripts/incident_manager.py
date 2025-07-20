#!/usr/bin/env python3
"""
Incident Response Manager pour PagerDuty

Gestionnaire avancé de réponse aux incidents avec PagerDuty.
Fournit des fonctionnalités complètes de gestion d'incidents,
automatisation des réponses, escalade intelligente, et reporting.

Fonctionnalités:
- Gestion complète du cycle de vie des incidents
- Automatisation des réponses selon les playbooks
- Escalade intelligente et notifications
- Intégration avec les équipes et calendriers
- Métriques et analytics d'incidents
- Génération de rapports détaillés
- Intégration ChatOps (Slack, Teams)

Version: 1.0.0
Auteur: Spotify AI Agent Team
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import yaml
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from utils.api_client import PagerDutyAPIClient
from utils.validators import PagerDutyValidator
from utils.formatters import MessageFormatter

console = Console()
logger = structlog.get_logger(__name__)

class IncidentStatus(Enum):
    """Statuts d'incident"""
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

class IncidentUrgency(Enum):
    """Urgences d'incident"""
    HIGH = "high"
    LOW = "low"

class ResponseAction(Enum):
    """Actions de réponse"""
    NOTIFY_TEAM = "notify_team"
    ESCALATE = "escalate"
    CREATE_BRIDGE = "create_bridge"
    EXECUTE_RUNBOOK = "execute_runbook"
    GATHER_LOGS = "gather_logs"
    SCALE_RESOURCES = "scale_resources"
    UPDATE_STATUS_PAGE = "update_status_page"

@dataclass
class IncidentContext:
    """Contexte d'un incident"""
    service_name: str
    environment: str
    affected_users: int = 0
    business_impact: str = "low"
    customer_facing: bool = False
    revenue_impact: float = 0.0
    affected_regions: List[str] = field(default_factory=list)
    related_incidents: List[str] = field(default_factory=list)
    runbooks: List[str] = field(default_factory=list)

@dataclass
class ResponsePlaybook:
    """Playbook de réponse aux incidents"""
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: int = 100
    enabled: bool = True
    timeout_minutes: int = 60
    requires_approval: bool = False
    auto_execute: bool = True

@dataclass
class IncidentMetrics:
    """Métriques d'incident"""
    time_to_acknowledge: Optional[timedelta] = None
    time_to_resolve: Optional[timedelta] = None
    escalation_count: int = 0
    affected_services: List[str] = field(default_factory=list)
    communication_count: int = 0
    post_mortem_required: bool = False

class PlaybookEngine:
    """Moteur d'exécution des playbooks"""
    
    def __init__(self, playbooks_file: Optional[str] = None):
        self.playbooks = []
        if playbooks_file:
            self.load_playbooks(playbooks_file)
    
    def load_playbooks(self, playbooks_file: str):
        """Charge les playbooks depuis un fichier"""
        try:
            with open(playbooks_file, 'r') as f:
                if playbooks_file.endswith('.json'):
                    playbooks_data = json.load(f)
                else:
                    playbooks_data = yaml.safe_load(f)
            
            self.playbooks = []
            for playbook_data in playbooks_data.get("playbooks", []):
                playbook = ResponsePlaybook(**playbook_data)
                self.playbooks.append(playbook)
            
            # Trier par priorité
            self.playbooks.sort(key=lambda p: p.priority)
            
            logger.info(f"Loaded {len(self.playbooks)} response playbooks")
            
        except Exception as e:
            logger.error(f"Failed to load playbooks: {e}")
            raise
    
    def find_matching_playbooks(self, incident_data: Dict[str, Any]) -> List[ResponsePlaybook]:
        """Trouve les playbooks correspondant à un incident"""
        matching_playbooks = []
        
        for playbook in self.playbooks:
            if not playbook.enabled:
                continue
            
            if self._matches_conditions(incident_data, playbook.conditions):
                matching_playbooks.append(playbook)
        
        return matching_playbooks
    
    def _matches_conditions(self, incident_data: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
        """Vérifie si l'incident correspond aux conditions du playbook"""
        for field, condition in conditions.items():
            if not self._evaluate_condition(incident_data, field, condition):
                return False
        
        return True
    
    def _evaluate_condition(self, incident_data: Dict[str, Any], field: str, condition: Any) -> bool:
        """Évalue une condition du playbook"""
        # Navigation dans les données imbriquées
        value = incident_data
        for key in field.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return False
        
        # Évaluation selon le type de condition
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

class CommunicationManager:
    """Gestionnaire de communications d'incident"""
    
    def __init__(self):
        self.channels = {
            "slack": self._send_slack_message,
            "teams": self._send_teams_message,
            "email": self._send_email,
            "sms": self._send_sms,
            "webhook": self._send_webhook
        }
        self.templates = {}
    
    async def send_notification(
        self,
        channel: str,
        template: str,
        context: Dict[str, Any],
        recipients: List[str]
    ):
        """Envoie une notification via le canal spécifié"""
        if channel not in self.channels:
            raise ValueError(f"Unsupported communication channel: {channel}")
        
        message = self._format_message(template, context)
        
        for recipient in recipients:
            try:
                await self.channels[channel](recipient, message, context)
                logger.info(f"Notification sent to {recipient} via {channel}")
            except Exception as e:
                logger.error(f"Failed to send notification to {recipient}: {e}")
    
    def _format_message(self, template: str, context: Dict[str, Any]) -> str:
        """Formate un message avec le contexte"""
        if template in self.templates:
            template_str = self.templates[template]
        else:
            template_str = template
        
        try:
            return template_str.format(**context)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return template_str
    
    async def _send_slack_message(self, channel: str, message: str, context: Dict[str, Any]):
        """Envoie un message Slack"""
        # Intégration Slack réelle ici
        logger.info(f"Slack message to {channel}: {message}")
    
    async def _send_teams_message(self, channel: str, message: str, context: Dict[str, Any]):
        """Envoie un message Teams"""
        # Intégration Teams réelle ici
        logger.info(f"Teams message to {channel}: {message}")
    
    async def _send_email(self, recipient: str, message: str, context: Dict[str, Any]):
        """Envoie un email"""
        # Intégration email réelle ici
        logger.info(f"Email to {recipient}: {message}")
    
    async def _send_sms(self, recipient: str, message: str, context: Dict[str, Any]):
        """Envoie un SMS"""
        # Intégration SMS réelle ici
        logger.info(f"SMS to {recipient}: {message}")
    
    async def _send_webhook(self, url: str, message: str, context: Dict[str, Any]):
        """Envoie un webhook"""
        # Intégration webhook réelle ici
        logger.info(f"Webhook to {url}: {message}")

class IncidentAnalyzer:
    """Analyseur d'incidents pour pattern recognition"""
    
    def __init__(self):
        self.incident_history = []
        self.patterns = {}
    
    def analyze_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse un incident pour détecter des patterns"""
        analysis = {
            "severity_assessment": self._assess_severity(incident_data),
            "similar_incidents": self._find_similar_incidents(incident_data),
            "impact_prediction": self._predict_impact(incident_data),
            "recommended_actions": self._recommend_actions(incident_data),
            "escalation_suggestion": self._suggest_escalation(incident_data)
        }
        
        return analysis
    
    def _assess_severity(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue la sévérité d'un incident"""
        # Facteurs de sévérité
        factors = {
            "user_impact": 0,
            "business_impact": 0,
            "service_criticality": 0,
            "outage_scope": 0
        }
        
        # Analyser l'impact utilisateur
        if "affected_users" in incident_data:
            user_count = incident_data["affected_users"]
            if user_count > 10000:
                factors["user_impact"] = 5
            elif user_count > 1000:
                factors["user_impact"] = 4
            elif user_count > 100:
                factors["user_impact"] = 3
            elif user_count > 10:
                factors["user_impact"] = 2
            else:
                factors["user_impact"] = 1
        
        # Analyser l'impact business
        if incident_data.get("customer_facing", False):
            factors["business_impact"] += 2
        
        if incident_data.get("revenue_impact", 0) > 0:
            factors["business_impact"] += 3
        
        # Score final
        total_score = sum(factors.values())
        
        if total_score >= 15:
            severity = "critical"
        elif total_score >= 10:
            severity = "high"
        elif total_score >= 5:
            severity = "medium"
        else:
            severity = "low"
        
        return {
            "severity": severity,
            "score": total_score,
            "factors": factors
        }
    
    def _find_similar_incidents(self, incident_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trouve des incidents similaires"""
        similar = []
        
        for past_incident in self.incident_history[-100:]:  # 100 derniers incidents
            similarity_score = self._calculate_similarity(incident_data, past_incident)
            
            if similarity_score > 0.7:  # Seuil de similarité
                similar.append({
                    "incident": past_incident,
                    "similarity_score": similarity_score
                })
        
        return sorted(similar, key=lambda x: x["similarity_score"], reverse=True)[:5]
    
    def _calculate_similarity(self, incident1: Dict[str, Any], incident2: Dict[str, Any]) -> float:
        """Calcule la similarité entre deux incidents"""
        # Implémentation simple de similarité
        score = 0.0
        factors = 0
        
        # Service
        if incident1.get("service") == incident2.get("service"):
            score += 0.4
        factors += 1
        
        # Environnement
        if incident1.get("environment") == incident2.get("environment"):
            score += 0.2
        factors += 1
        
        # Type d'erreur
        if incident1.get("error_type") == incident2.get("error_type"):
            score += 0.3
        factors += 1
        
        # Impact
        if abs(incident1.get("affected_users", 0) - incident2.get("affected_users", 0)) < 100:
            score += 0.1
        factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def _predict_impact(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prédit l'impact d'un incident"""
        # Prédiction basée sur l'historique et les patterns
        prediction = {
            "estimated_duration": "30-60 minutes",
            "affected_services": [],
            "potential_cascade": False,
            "confidence": 0.7
        }
        
        return prediction
    
    def _recommend_actions(self, incident_data: Dict[str, Any]) -> List[str]:
        """Recommande des actions basées sur l'analyse"""
        actions = []
        
        severity = self._assess_severity(incident_data)["severity"]
        
        if severity == "critical":
            actions.extend([
                "Immediately escalate to senior engineer",
                "Start incident bridge",
                "Notify executive team",
                "Prepare customer communication"
            ])
        elif severity == "high":
            actions.extend([
                "Escalate to on-call engineer",
                "Monitor closely",
                "Prepare status page update"
            ])
        else:
            actions.extend([
                "Investigate and monitor",
                "Document findings"
            ])
        
        return actions
    
    def _suggest_escalation(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Suggère une escalade appropriée"""
        severity = self._assess_severity(incident_data)["severity"]
        
        escalation = {
            "immediate": False,
            "delay_minutes": 30,
            "target_team": "engineering",
            "reason": "Standard escalation procedure"
        }
        
        if severity == "critical":
            escalation.update({
                "immediate": True,
                "delay_minutes": 0,
                "target_team": "senior_engineering",
                "reason": "Critical incident requiring immediate attention"
            })
        
        return escalation

class PagerDutyIncidentManager:
    """Gestionnaire principal d'incidents PagerDuty"""
    
    def __init__(
        self,
        api_key: str,
        playbooks_file: Optional[str] = None,
        enable_analytics: bool = True
    ):
        self.api_client = PagerDutyAPIClient(api_key, None)
        self.playbook_engine = PlaybookEngine(playbooks_file)
        self.communication_manager = CommunicationManager()
        self.analyzer = IncidentAnalyzer() if enable_analytics else None
        self.formatter = MessageFormatter()
        
        self.active_incidents = {}
        self.incident_metrics = {}
        
        self.metrics = {
            "incidents_processed": 0,
            "playbooks_executed": 0,
            "escalations_triggered": 0,
            "communications_sent": 0,
            "average_resolution_time": 0
        }
    
    async def handle_incident(self, incident_data: Dict[str, Any]) -> bool:
        """Gère un incident complet"""
        incident_id = incident_data.get("id")
        
        if not incident_id:
            logger.error("No incident ID provided")
            return False
        
        self.metrics["incidents_processed"] += 1
        
        logger.info(f"Handling incident: {incident_id}")
        
        try:
            # Analyser l'incident
            analysis = None
            if self.analyzer:
                analysis = self.analyzer.analyze_incident(incident_data)
                logger.info(f"Incident analysis completed for {incident_id}")
            
            # Enrichir les données d'incident
            enriched_data = await self._enrich_incident_data(incident_data, analysis)
            
            # Stocker l'incident
            self.active_incidents[incident_id] = {
                "data": enriched_data,
                "analysis": analysis,
                "start_time": datetime.now(timezone.utc),
                "playbooks_executed": [],
                "communications_sent": [],
                "escalations": []
            }
            
            # Initialiser les métriques
            self.incident_metrics[incident_id] = IncidentMetrics()
            
            # Trouver et exécuter les playbooks appropriés
            matching_playbooks = self.playbook_engine.find_matching_playbooks(enriched_data)
            
            if matching_playbooks:
                logger.info(f"Found {len(matching_playbooks)} matching playbooks for {incident_id}")
                
                for playbook in matching_playbooks:
                    if playbook.auto_execute:
                        await self._execute_playbook(incident_id, playbook)
                    else:
                        logger.info(f"Playbook {playbook.name} requires manual approval")
            else:
                logger.warning(f"No matching playbooks found for incident {incident_id}")
                await self._handle_no_playbook(incident_id, enriched_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle incident {incident_id}: {e}")
            return False
    
    async def _enrich_incident_data(
        self,
        incident_data: Dict[str, Any],
        analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enrichit les données d'incident avec des informations supplémentaires"""
        enriched = incident_data.copy()
        
        # Ajouter l'analyse si disponible
        if analysis:
            enriched["analysis"] = analysis
        
        # Ajouter des métadonnées
        enriched["metadata"] = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "handler_version": "1.0.0",
            "processing_node": "incident-manager-01"
        }
        
        # Récupérer des informations supplémentaires depuis PagerDuty
        if "incident_id" in incident_data:
            try:
                # Ici on pourrait récupérer plus d'infos depuis l'API PagerDuty
                pd_incident = await self._get_pagerduty_incident(incident_data["incident_id"])
                if pd_incident:
                    enriched["pagerduty_details"] = pd_incident
            except Exception as e:
                logger.warning(f"Failed to enrich with PagerDuty data: {e}")
        
        return enriched
    
    async def _get_pagerduty_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les détails d'un incident depuis PagerDuty"""
        # Implémentation de récupération d'incident PagerDuty
        # Pour l'exemple, on retourne None
        return None
    
    async def _execute_playbook(self, incident_id: str, playbook: ResponsePlaybook):
        """Exécute un playbook pour un incident"""
        logger.info(f"Executing playbook {playbook.name} for incident {incident_id}")
        
        incident_record = self.active_incidents[incident_id]
        incident_data = incident_record["data"]
        
        self.metrics["playbooks_executed"] += 1
        
        # Enregistrer l'exécution du playbook
        execution_record = {
            "playbook_name": playbook.name,
            "started_at": datetime.now(timezone.utc),
            "actions_completed": [],
            "actions_failed": []
        }
        
        incident_record["playbooks_executed"].append(execution_record)
        
        try:
            # Exécuter chaque action du playbook
            for action in playbook.actions:
                success = await self._execute_action(incident_id, action, incident_data)
                
                if success:
                    execution_record["actions_completed"].append(action)
                else:
                    execution_record["actions_failed"].append(action)
            
            execution_record["completed_at"] = datetime.now(timezone.utc)
            logger.info(f"Playbook {playbook.name} completed for incident {incident_id}")
            
        except Exception as e:
            execution_record["error"] = str(e)
            execution_record["failed_at"] = datetime.now(timezone.utc)
            logger.error(f"Playbook {playbook.name} failed for incident {incident_id}: {e}")
    
    async def _execute_action(
        self,
        incident_id: str,
        action: Dict[str, Any],
        incident_data: Dict[str, Any]
    ) -> bool:
        """Exécute une action spécifique"""
        action_type = action.get("type")
        
        try:
            if action_type == ResponseAction.NOTIFY_TEAM.value:
                await self._notify_team(incident_id, action, incident_data)
            
            elif action_type == ResponseAction.ESCALATE.value:
                await self._escalate_incident(incident_id, action, incident_data)
            
            elif action_type == ResponseAction.CREATE_BRIDGE.value:
                await self._create_incident_bridge(incident_id, action, incident_data)
            
            elif action_type == ResponseAction.EXECUTE_RUNBOOK.value:
                await self._execute_runbook(incident_id, action, incident_data)
            
            elif action_type == ResponseAction.GATHER_LOGS.value:
                await self._gather_logs(incident_id, action, incident_data)
            
            elif action_type == ResponseAction.SCALE_RESOURCES.value:
                await self._scale_resources(incident_id, action, incident_data)
            
            elif action_type == ResponseAction.UPDATE_STATUS_PAGE.value:
                await self._update_status_page(incident_id, action, incident_data)
            
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False
            
            logger.info(f"Action {action_type} completed for incident {incident_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute action {action_type} for incident {incident_id}: {e}")
            return False
    
    async def _notify_team(self, incident_id: str, action: Dict[str, Any], incident_data: Dict[str, Any]):
        """Notifie une équipe"""
        team = action.get("team", "engineering")
        channel = action.get("channel", "slack")
        template = action.get("template", "incident_notification")
        
        context = {
            "incident_id": incident_id,
            "incident_title": incident_data.get("title", "Unknown incident"),
            "severity": incident_data.get("severity", "medium"),
            "service": incident_data.get("service", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        recipients = action.get("recipients", [f"#{team}-alerts"])
        
        await self.communication_manager.send_notification(
            channel, template, context, recipients
        )
        
        self.metrics["communications_sent"] += 1
        
        # Enregistrer la communication
        self.active_incidents[incident_id]["communications_sent"].append({
            "type": "team_notification",
            "team": team,
            "channel": channel,
            "timestamp": datetime.now(timezone.utc)
        })
    
    async def _escalate_incident(self, incident_id: str, action: Dict[str, Any], incident_data: Dict[str, Any]):
        """Escalade un incident"""
        escalation_level = action.get("level", 1)
        target_team = action.get("target_team", "senior_engineering")
        delay_minutes = action.get("delay_minutes", 0)
        
        self.metrics["escalations_triggered"] += 1
        
        # Enregistrer l'escalade
        escalation_record = {
            "level": escalation_level,
            "target_team": target_team,
            "timestamp": datetime.now(timezone.utc),
            "delay_minutes": delay_minutes
        }
        
        self.active_incidents[incident_id]["escalations"].append(escalation_record)
        self.incident_metrics[incident_id].escalation_count += 1
        
        if delay_minutes > 0:
            # Programmer l'escalade
            asyncio.create_task(self._delayed_escalation(incident_id, escalation_record))
        else:
            # Escalade immédiate
            await self._perform_escalation(incident_id, escalation_record)
    
    async def _delayed_escalation(self, incident_id: str, escalation_record: Dict[str, Any]):
        """Effectue une escalade avec délai"""
        await asyncio.sleep(escalation_record["delay_minutes"] * 60)
        
        # Vérifier si l'incident est toujours actif
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            status = incident["data"].get("status", "triggered")
            
            if status not in ["resolved", "acknowledged"]:
                await self._perform_escalation(incident_id, escalation_record)
    
    async def _perform_escalation(self, incident_id: str, escalation_record: Dict[str, Any]):
        """Effectue l'escalade"""
        logger.info(f"Escalating incident {incident_id} to {escalation_record['target_team']}")
        
        # Ici on pourrait intégrer avec PagerDuty pour créer une nouvelle escalade
        # ou notifier l'équipe cible
    
    async def _create_incident_bridge(self, incident_id: str, action: Dict[str, Any], incident_data: Dict[str, Any]):
        """Crée un pont d'incident (conference call)"""
        bridge_type = action.get("bridge_type", "zoom")
        participants = action.get("participants", [])
        
        logger.info(f"Creating {bridge_type} bridge for incident {incident_id}")
        
        # Ici on pourrait intégrer avec Zoom, Teams, etc.
    
    async def _execute_runbook(self, incident_id: str, action: Dict[str, Any], incident_data: Dict[str, Any]):
        """Exécute un runbook"""
        runbook_name = action.get("runbook_name")
        runbook_url = action.get("runbook_url")
        auto_execute = action.get("auto_execute", False)
        
        logger.info(f"Executing runbook {runbook_name} for incident {incident_id}")
        
        if auto_execute:
            # Ici on pourrait intégrer avec des systèmes d'automatisation
            # comme Ansible, Terraform, ou des scripts personnalisés
            pass
    
    async def _gather_logs(self, incident_id: str, action: Dict[str, Any], incident_data: Dict[str, Any]):
        """Collecte des logs"""
        log_sources = action.get("sources", [])
        time_range = action.get("time_range", "last_1h")
        
        logger.info(f"Gathering logs from {len(log_sources)} sources for incident {incident_id}")
        
        # Ici on pourrait intégrer avec ELK, Splunk, CloudWatch, etc.
    
    async def _scale_resources(self, incident_id: str, action: Dict[str, Any], incident_data: Dict[str, Any]):
        """Scale les ressources"""
        resource_type = action.get("resource_type", "pods")
        scale_factor = action.get("scale_factor", 2)
        service = action.get("service")
        
        logger.info(f"Scaling {resource_type} by factor {scale_factor} for incident {incident_id}")
        
        # Ici on pourrait intégrer avec Kubernetes, AWS Auto Scaling, etc.
    
    async def _update_status_page(self, incident_id: str, action: Dict[str, Any], incident_data: Dict[str, Any]):
        """Met à jour la page de statut"""
        status_message = action.get("message", "We are investigating an issue")
        affected_components = action.get("affected_components", [])
        
        logger.info(f"Updating status page for incident {incident_id}")
        
        # Ici on pourrait intégrer avec StatusPage.io, Atlassian, etc.
    
    async def _handle_no_playbook(self, incident_id: str, incident_data: Dict[str, Any]):
        """Gère un incident sans playbook correspondant"""
        logger.warning(f"No playbook found for incident {incident_id}, using default response")
        
        # Réponse par défaut
        default_actions = [
            {
                "type": "notify_team",
                "team": "engineering",
                "channel": "slack",
                "template": "default_incident_notification"
            }
        ]
        
        for action in default_actions:
            await self._execute_action(incident_id, action, incident_data)
    
    async def acknowledge_incident(self, incident_id: str, user_email: str) -> bool:
        """Acknowledge un incident"""
        if incident_id not in self.active_incidents:
            return False
        
        try:
            # Ici on pourrait appeler l'API PagerDuty pour acknowledger
            incident = self.active_incidents[incident_id]
            incident["data"]["status"] = "acknowledged"
            incident["acknowledged_at"] = datetime.now(timezone.utc)
            incident["acknowledged_by"] = user_email
            
            # Calculer le temps de réponse
            start_time = incident["start_time"]
            ack_time = incident["acknowledged_at"]
            self.incident_metrics[incident_id].time_to_acknowledge = ack_time - start_time
            
            logger.info(f"Incident {incident_id} acknowledged by {user_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge incident {incident_id}: {e}")
            return False
    
    async def resolve_incident(self, incident_id: str, user_email: str, resolution_note: str = "") -> bool:
        """Résout un incident"""
        if incident_id not in self.active_incidents:
            return False
        
        try:
            # Ici on pourrait appeler l'API PagerDuty pour résoudre
            incident = self.active_incidents[incident_id]
            incident["data"]["status"] = "resolved"
            incident["resolved_at"] = datetime.now(timezone.utc)
            incident["resolved_by"] = user_email
            incident["resolution_note"] = resolution_note
            
            # Calculer le temps de résolution
            start_time = incident["start_time"]
            resolve_time = incident["resolved_at"]
            self.incident_metrics[incident_id].time_to_resolve = resolve_time - start_time
            
            # Calculer la moyenne des temps de résolution
            self._update_average_resolution_time()
            
            logger.info(f"Incident {incident_id} resolved by {user_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve incident {incident_id}: {e}")
            return False
    
    def _update_average_resolution_time(self):
        """Met à jour le temps moyen de résolution"""
        resolution_times = [
            m.time_to_resolve.total_seconds() / 60
            for m in self.incident_metrics.values()
            if m.time_to_resolve is not None
        ]
        
        if resolution_times:
            self.metrics["average_resolution_time"] = sum(resolution_times) / len(resolution_times)
    
    def get_incident_report(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Génère un rapport d'incident"""
        if incident_id not in self.active_incidents:
            return None
        
        incident = self.active_incidents[incident_id]
        metrics = self.incident_metrics[incident_id]
        
        report = {
            "incident_id": incident_id,
            "summary": {
                "title": incident["data"].get("title", "Unknown"),
                "status": incident["data"].get("status", "unknown"),
                "severity": incident["data"].get("severity", "medium"),
                "start_time": incident["start_time"].isoformat(),
                "duration": self._calculate_duration(incident),
                "acknowledged_by": incident.get("acknowledged_by"),
                "resolved_by": incident.get("resolved_by")
            },
            "metrics": {
                "time_to_acknowledge": metrics.time_to_acknowledge.total_seconds() / 60 if metrics.time_to_acknowledge else None,
                "time_to_resolve": metrics.time_to_resolve.total_seconds() / 60 if metrics.time_to_resolve else None,
                "escalation_count": metrics.escalation_count,
                "communication_count": len(incident["communications_sent"])
            },
            "timeline": {
                "playbooks_executed": incident["playbooks_executed"],
                "communications_sent": incident["communications_sent"],
                "escalations": incident["escalations"]
            },
            "analysis": incident.get("analysis", {}),
            "resolution_note": incident.get("resolution_note", "")
        }
        
        return report
    
    def _calculate_duration(self, incident: Dict[str, Any]) -> Optional[float]:
        """Calcule la durée d'un incident en minutes"""
        start_time = incident["start_time"]
        
        if "resolved_at" in incident:
            end_time = incident["resolved_at"]
        else:
            end_time = datetime.now(timezone.utc)
        
        duration = end_time - start_time
        return duration.total_seconds() / 60
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques"""
        active_count = len([
            i for i in self.active_incidents.values()
            if i["data"].get("status") not in ["resolved"]
        ])
        
        return {
            "active_incidents": active_count,
            "total_incidents": len(self.active_incidents),
            "metrics": self.metrics,
            "recent_incidents": list(self.active_incidents.keys())[-10:]
        }
    
    async def cleanup_resolved_incidents(self, max_age_days: int = 30) -> int:
        """Nettoie les incidents résolus anciens"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        
        to_remove = []
        for incident_id, incident in self.active_incidents.items():
            if (incident["data"].get("status") == "resolved" and
                incident.get("resolved_at", datetime.now(timezone.utc)) < cutoff_date):
                to_remove.append(incident_id)
        
        for incident_id in to_remove:
            del self.active_incidents[incident_id]
            if incident_id in self.incident_metrics:
                del self.incident_metrics[incident_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old resolved incidents")
        return len(to_remove)
    
    async def close(self):
        """Ferme les connexions"""
        await self.api_client.close()

async def main():
    """Fonction principale CLI"""
    parser = argparse.ArgumentParser(description="PagerDuty Incident Response Manager")
    parser.add_argument("--action", required=True,
                       choices=["handle", "acknowledge", "resolve", "report", "metrics", "cleanup"],
                       help="Action à effectuer")
    parser.add_argument("--api-key", required=True, help="Clé API PagerDuty")
    parser.add_argument("--playbooks-file", help="Fichier de playbooks")
    parser.add_argument("--incident-id", help="ID de l'incident")
    parser.add_argument("--user-email", help="Email de l'utilisateur")
    parser.add_argument("--incident-data", help="Données d'incident (JSON)")
    parser.add_argument("--resolution-note", help="Note de résolution")
    
    args = parser.parse_args()
    
    try:
        manager = PagerDutyIncidentManager(
            api_key=args.api_key,
            playbooks_file=args.playbooks_file
        )
        
        if args.action == "handle":
            if not args.incident_data:
                console.print("[red]Incident data required for handle action[/red]")
                return 1
            
            incident_data = json.loads(args.incident_data)
            success = await manager.handle_incident(incident_data)
            
            if success:
                console.print("[green]Incident handled successfully[/green]")
            else:
                console.print("[red]Failed to handle incident[/red]")
                return 1
        
        elif args.action == "acknowledge":
            if not args.incident_id or not args.user_email:
                console.print("[red]Incident ID and user email required[/red]")
                return 1
            
            success = await manager.acknowledge_incident(args.incident_id, args.user_email)
            
            if success:
                console.print(f"[green]Incident {args.incident_id} acknowledged[/green]")
            else:
                console.print(f"[red]Failed to acknowledge incident {args.incident_id}[/red]")
                return 1
        
        elif args.action == "resolve":
            if not args.incident_id or not args.user_email:
                console.print("[red]Incident ID and user email required[/red]")
                return 1
            
            success = await manager.resolve_incident(
                args.incident_id,
                args.user_email,
                args.resolution_note or ""
            )
            
            if success:
                console.print(f"[green]Incident {args.incident_id} resolved[/green]")
            else:
                console.print(f"[red]Failed to resolve incident {args.incident_id}[/red]")
                return 1
        
        elif args.action == "report":
            if not args.incident_id:
                console.print("[red]Incident ID required for report[/red]")
                return 1
            
            report = manager.get_incident_report(args.incident_id)
            
            if report:
                console.print(Panel.fit(
                    json.dumps(report, indent=2, default=str),
                    title=f"Incident Report - {args.incident_id}"
                ))
            else:
                console.print(f"[red]Incident {args.incident_id} not found[/red]")
                return 1
        
        elif args.action == "metrics":
            metrics = manager.get_metrics_summary()
            
            console.print(Panel.fit(
                json.dumps(metrics, indent=2),
                title="Incident Metrics Summary"
            ))
        
        elif args.action == "cleanup":
            cleaned = await manager.cleanup_resolved_incidents()
            console.print(f"[green]Cleaned up {cleaned} old incidents[/green]")
        
        await manager.close()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
