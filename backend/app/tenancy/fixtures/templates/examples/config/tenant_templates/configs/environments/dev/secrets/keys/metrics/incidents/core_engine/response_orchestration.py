# =============================================================================
# Response Orchestration - Workflows Automatisés Avancés
# =============================================================================
# 
# Module d'orchestration des réponses automatisées avec workflows configurables,
# gestion d'état avancée, et intégration avec les systèmes externes.
# Support pour workflows complexes, conditions, boucles et parallélisation.
#
# Auteur: Lead Developer + Microservices Architect
# Direction Technique: Fahed Mlaiel
# Version: 2.0.0 Enterprise
# =============================================================================

"""
Response Orchestration - Moteur de Workflows Automatisés

Ce module fournit un système complet d'orchestration des réponses avec:

Fonctionnalités Principales:
- Workflows configurables avec DSL (Domain Specific Language)
- Exécution parallèle et séquentielle des actions
- Gestion d'état distribuée avec checkpoints
- Intégrations externes (APIs, webhooks, notifications)

Composants:
- WorkflowEngine: Moteur principal d'exécution
- ActionExecutor: Exécuteur d'actions individuelles
- StateManager: Gestion d'état persistant
- IntegrationManager: Gestion des intégrations externes

Types de Workflows:
- Incident Response: Réponse automatique aux incidents
- Escalation: Escalade automatique selon règles
- Notification: Système de notification multi-canal
- Remediation: Actions de remédiation automatique
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import re
from abc import ABC, abstractmethod

# Imports pour intégrations externes
import aiohttp
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Imports du Core Engine
from . import core_registry, TenantContext, EngineStatus
from .incident_management import IncidentEvent, IncidentSeverity, IncidentCategory

logger = logging.getLogger(__name__)

# ===========================
# Configuration & Constants
# ===========================

DEFAULT_WORKFLOW_TIMEOUT = 300  # 5 minutes
MAX_WORKFLOW_RETRIES = 3
DEFAULT_CHECKPOINT_INTERVAL = 30  # secondes
MAX_PARALLEL_ACTIONS = 10

# Templates de workflows pré-définis
PREDEFINED_WORKFLOWS = {
    "incident_critical_response": {
        "name": "Critical Incident Response",
        "trigger": {"severity": ["critical", "emergency"]},
        "actions": [
            {"type": "notification", "channels": ["email", "slack", "sms"], "parallel": True},
            {"type": "create_ticket", "system": "jira", "priority": "highest"},
            {"type": "page_oncall", "escalation_level": 1},
            {"type": "start_war_room", "platform": "zoom"}
        ]
    },
    "security_incident_response": {
        "name": "Security Incident Response",
        "trigger": {"category": ["security"]},
        "actions": [
            {"type": "isolate_system", "target": "affected_systems"},
            {"type": "notification", "channels": ["security_team"], "urgent": True},
            {"type": "collect_forensics", "retention": "90_days"},
            {"type": "update_security_tools", "action": "block_indicators"}
        ]
    },
    "performance_degradation": {
        "name": "Performance Degradation Response",
        "trigger": {"category": ["performance"], "keywords": ["slow", "timeout", "latency"]},
        "actions": [
            {"type": "collect_metrics", "duration": "1_hour"},
            {"type": "scale_resources", "target": "auto_scaling_group", "factor": 1.5},
            {"type": "notification", "channels": ["dev_team"], "include_metrics": True},
            {"type": "create_incident", "template": "performance_investigation"}
        ]
    }
}

# ===========================
# Enums & Types
# ===========================

class WorkflowStatus(Enum):
    """Statuts de workflow"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class ActionType(Enum):
    """Types d'actions disponibles"""
    NOTIFICATION = "notification"
    API_CALL = "api_call"
    WEBHOOK = "webhook"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    CREATE_TICKET = "create_ticket"
    UPDATE_SYSTEM = "update_system"
    SCALE_RESOURCES = "scale_resources"
    COLLECT_METRICS = "collect_metrics"
    CUSTOM_SCRIPT = "custom_script"
    WAIT = "wait"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"

class ExecutionMode(Enum):
    """Modes d'exécution"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"

class TriggerType(Enum):
    """Types de déclencheurs"""
    INCIDENT_CREATED = "incident_created"
    INCIDENT_UPDATED = "incident_updated"
    SEVERITY_CHANGED = "severity_changed"
    STATUS_CHANGED = "status_changed"
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"

# ===========================
# Modèles de Données
# ===========================

@dataclass
class WorkflowAction:
    """Action individuelle dans un workflow"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ActionType = ActionType.NOTIFICATION
    name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 60
    retry_count: int = 0
    max_retries: int = 3
    is_critical: bool = False
    parallel: bool = False
    depends_on: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.name:
            self.name = f"{self.type.value}_{self.id[:8]}"

@dataclass
class WorkflowDefinition:
    """Définition d'un workflow"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    is_active: bool = True
    tenant_id: str = ""
    
    # Configuration du déclencheur
    trigger_type: TriggerType = TriggerType.MANUAL
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Actions du workflow
    actions: List[WorkflowAction] = field(default_factory=list)
    
    # Configuration d'exécution
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    timeout_seconds: int = DEFAULT_WORKFLOW_TIMEOUT
    max_retries: int = MAX_WORKFLOW_RETRIES
    checkpoint_enabled: bool = True
    
    # Métadonnées
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    
    def add_action(self, action: WorkflowAction) -> None:
        """Ajoute une action au workflow"""
        self.actions.append(action)
    
    def get_action(self, action_id: str) -> Optional[WorkflowAction]:
        """Récupère une action par son ID"""
        return next((action for action in self.actions if action.id == action_id), None)

@dataclass
class WorkflowExecution:
    """Instance d'exécution d'un workflow"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    workflow_definition: Optional[WorkflowDefinition] = None
    
    # Contexte d'exécution
    incident_id: Optional[str] = None
    tenant_id: str = ""
    triggered_by: str = ""
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    
    # État d'exécution
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_action_id: Optional[str] = None
    completed_actions: List[str] = field(default_factory=list)
    failed_actions: List[str] = field(default_factory=list)
    action_results: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    
    # Gestion d'erreurs
    error_message: Optional[str] = None
    retry_count: int = 0
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_running(self) -> bool:
        return self.status == WorkflowStatus.RUNNING
    
    @property
    def is_completed(self) -> bool:
        return self.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
    
    @property
    def success_rate(self) -> float:
        total_actions = len(self.completed_actions) + len(self.failed_actions)
        return (len(self.completed_actions) / total_actions * 100) if total_actions > 0 else 0.0

@dataclass
class ActionResult:
    """Résultat d'exécution d'une action"""
    action_id: str
    success: bool = False
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

# ===========================
# Exécuteurs d'Actions
# ===========================

class BaseActionExecutor(ABC):
    """Classe de base pour les exécuteurs d'actions"""
    
    def __init__(self, action_type: ActionType):
        self.action_type = action_type
        self.logger = logging.getLogger(f"{__name__}.{action_type.value}")
    
    @abstractmethod
    async def execute(self, action: WorkflowAction, context: Dict[str, Any]) -> ActionResult:
        """Exécute l'action"""
        pass
    
    def validate_parameters(self, action: WorkflowAction) -> bool:
        """Valide les paramètres de l'action"""
        return True

class NotificationExecutor(BaseActionExecutor):
    """Exécuteur pour les notifications"""
    
    def __init__(self):
        super().__init__(ActionType.NOTIFICATION)
    
    async def execute(self, action: WorkflowAction, context: Dict[str, Any]) -> ActionResult:
        start_time = datetime.utcnow()
        
        try:
            channels = action.parameters.get("channels", ["email"])
            message = action.parameters.get("message", "Notification from incident management system")
            recipients = action.parameters.get("recipients", [])
            urgent = action.parameters.get("urgent", False)
            
            results = {}
            
            for channel in channels:
                if channel == "email":
                    result = await self._send_email(message, recipients, urgent)
                    results["email"] = result
                elif channel == "slack":
                    result = await self._send_slack(message, recipients, urgent)
                    results["slack"] = result
                elif channel == "sms":
                    result = await self._send_sms(message, recipients, urgent)
                    results["sms"] = result
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ActionResult(
                action_id=action.id,
                success=True,
                result_data=results,
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Erreur lors de l'envoi de notification: {e}")
            
            return ActionResult(
                action_id=action.id,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )
    
    async def _send_email(self, message: str, recipients: List[str], urgent: bool) -> Dict[str, Any]:
        """Simule l'envoi d'email"""
        # Simulation d'envoi d'email
        await asyncio.sleep(0.5)  # Simulation de latence réseau
        
        return {
            "status": "sent",
            "recipients": recipients,
            "message_id": str(uuid.uuid4()),
            "urgent": urgent
        }
    
    async def _send_slack(self, message: str, recipients: List[str], urgent: bool) -> Dict[str, Any]:
        """Simule l'envoi vers Slack"""
        await asyncio.sleep(0.3)
        
        return {
            "status": "sent",
            "channels": recipients,
            "thread_id": str(uuid.uuid4()),
            "urgent": urgent
        }
    
    async def _send_sms(self, message: str, recipients: List[str], urgent: bool) -> Dict[str, Any]:
        """Simule l'envoi de SMS"""
        await asyncio.sleep(0.2)
        
        return {
            "status": "sent",
            "recipients": recipients,
            "message_ids": [str(uuid.uuid4()) for _ in recipients],
            "urgent": urgent
        }

class APICallExecutor(BaseActionExecutor):
    """Exécuteur pour les appels API"""
    
    def __init__(self):
        super().__init__(ActionType.API_CALL)
    
    async def execute(self, action: WorkflowAction, context: Dict[str, Any]) -> ActionResult:
        start_time = datetime.utcnow()
        
        try:
            url = action.parameters.get("url", "")
            method = action.parameters.get("method", "GET").upper()
            headers = action.parameters.get("headers", {})
            data = action.parameters.get("data", {})
            auth = action.parameters.get("auth", {})
            
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data if method in ["POST", "PUT", "PATCH"] else None,
                    timeout=aiohttp.ClientTimeout(total=action.timeout_seconds)
                ) as response:
                    response_data = await response.text()
                    
                    execution_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    return ActionResult(
                        action_id=action.id,
                        success=response.status < 400,
                        result_data={
                            "status_code": response.status,
                            "response_data": response_data,
                            "url": url,
                            "method": method
                        },
                        execution_time_seconds=execution_time
                    )
                    
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Erreur lors de l'appel API: {e}")
            
            return ActionResult(
                action_id=action.id,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )

class TicketExecutor(BaseActionExecutor):
    """Exécuteur pour la création de tickets"""
    
    def __init__(self):
        super().__init__(ActionType.CREATE_TICKET)
    
    async def execute(self, action: WorkflowAction, context: Dict[str, Any]) -> ActionResult:
        start_time = datetime.utcnow()
        
        try:
            system = action.parameters.get("system", "jira")
            title = action.parameters.get("title", "Incident Ticket")
            description = action.parameters.get("description", "")
            priority = action.parameters.get("priority", "medium")
            assignee = action.parameters.get("assignee", "")
            
            # Simulation de création de ticket
            await asyncio.sleep(1.0)  # Simulation de latence API
            
            ticket_id = f"{system.upper()}-{uuid.uuid4().hex[:8]}"
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ActionResult(
                action_id=action.id,
                success=True,
                result_data={
                    "ticket_id": ticket_id,
                    "system": system,
                    "title": title,
                    "priority": priority,
                    "assignee": assignee,
                    "url": f"https://{system}.company.com/ticket/{ticket_id}"
                },
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Erreur lors de la création de ticket: {e}")
            
            return ActionResult(
                action_id=action.id,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )

class SystemUpdateExecutor(BaseActionExecutor):
    """Exécuteur pour les mises à jour de système"""
    
    def __init__(self):
        super().__init__(ActionType.UPDATE_SYSTEM)
    
    async def execute(self, action: WorkflowAction, context: Dict[str, Any]) -> ActionResult:
        start_time = datetime.utcnow()
        
        try:
            target_system = action.parameters.get("target_system", "")
            action_type = action.parameters.get("action", "restart")
            parameters = action.parameters.get("parameters", {})
            
            # Simulation de mise à jour système
            await asyncio.sleep(2.0)  # Simulation d'opération système
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ActionResult(
                action_id=action.id,
                success=True,
                result_data={
                    "target_system": target_system,
                    "action": action_type,
                    "parameters": parameters,
                    "status": "completed",
                    "operation_id": str(uuid.uuid4())
                },
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Erreur lors de la mise à jour système: {e}")
            
            return ActionResult(
                action_id=action.id,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )

class WaitExecutor(BaseActionExecutor):
    """Exécuteur pour les attentes"""
    
    def __init__(self):
        super().__init__(ActionType.WAIT)
    
    async def execute(self, action: WorkflowAction, context: Dict[str, Any]) -> ActionResult:
        start_time = datetime.utcnow()
        
        try:
            wait_seconds = action.parameters.get("seconds", 30)
            reason = action.parameters.get("reason", "Waiting as configured")
            
            await asyncio.sleep(wait_seconds)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ActionResult(
                action_id=action.id,
                success=True,
                result_data={
                    "waited_seconds": wait_seconds,
                    "reason": reason
                },
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ActionResult(
                action_id=action.id,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )

# ===========================
# Gestionnaire d'État
# ===========================

class WorkflowStateManager:
    """Gestionnaire d'état des workflows"""
    
    def __init__(self):
        self.executions: Dict[str, WorkflowExecution] = {}
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Gestionnaire d'état des workflows initialisé")
    
    async def save_execution(self, execution: WorkflowExecution) -> None:
        """Sauvegarde l'état d'exécution"""
        self.executions[execution.id] = execution
        
        # Sauvegarde checkpoint si activé
        if execution.workflow_definition and execution.workflow_definition.checkpoint_enabled:
            await self.save_checkpoint(execution)
    
    async def load_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Charge l'état d'exécution"""
        return self.executions.get(execution_id)
    
    async def save_checkpoint(self, execution: WorkflowExecution) -> None:
        """Sauvegarde un checkpoint"""
        checkpoint_data = {
            "execution_id": execution.id,
            "status": execution.status.value,
            "current_action_id": execution.current_action_id,
            "completed_actions": execution.completed_actions.copy(),
            "failed_actions": execution.failed_actions.copy(),
            "action_results": execution.action_results.copy(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.checkpoints[execution.id] = checkpoint_data
        logger.debug(f"Checkpoint sauvegardé pour l'exécution {execution.id}")
    
    async def load_checkpoint(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Charge un checkpoint"""
        return self.checkpoints.get(execution_id)
    
    async def resume_from_checkpoint(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Reprend l'exécution depuis un checkpoint"""
        checkpoint = await self.load_checkpoint(execution_id)
        execution = await self.load_execution(execution_id)
        
        if checkpoint and execution:
            execution.status = WorkflowStatus(checkpoint["status"])
            execution.current_action_id = checkpoint["current_action_id"]
            execution.completed_actions = checkpoint["completed_actions"]
            execution.failed_actions = checkpoint["failed_actions"]
            execution.action_results = checkpoint["action_results"]
            
            logger.info(f"Exécution {execution_id} reprise depuis le checkpoint")
            return execution
        
        return None

# ===========================
# Moteur de Workflows
# ===========================

class WorkflowEngine:
    """Moteur principal d'exécution des workflows"""
    
    def __init__(self):
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.state_manager = WorkflowStateManager()
        self.running_executions: Dict[str, asyncio.Task] = {}
        
        # Enregistrement des exécuteurs d'actions
        self.action_executors = {
            ActionType.NOTIFICATION: NotificationExecutor(),
            ActionType.API_CALL: APICallExecutor(),
            ActionType.CREATE_TICKET: TicketExecutor(),
            ActionType.UPDATE_SYSTEM: SystemUpdateExecutor(),
            ActionType.WAIT: WaitExecutor()
        }
        
        # Chargement des workflows prédéfinis
        self._load_predefined_workflows()
        
        logger.info("Moteur de workflows initialisé")
    
    def _load_predefined_workflows(self):
        """Charge les workflows prédéfinis"""
        for workflow_id, config in PREDEFINED_WORKFLOWS.items():
            workflow = self._create_workflow_from_config(workflow_id, config)
            self.register_workflow(workflow)
    
    def _create_workflow_from_config(self, workflow_id: str, config: Dict[str, Any]) -> WorkflowDefinition:
        """Crée un workflow depuis une configuration"""
        workflow = WorkflowDefinition(
            id=workflow_id,
            name=config["name"],
            description=config.get("description", ""),
            trigger_type=TriggerType.INCIDENT_CREATED,
            trigger_conditions=config.get("trigger", {})
        )
        
        for i, action_config in enumerate(config.get("actions", [])):
            action = WorkflowAction(
                type=ActionType(action_config["type"]),
                name=action_config.get("name", f"Action {i+1}"),
                parameters=action_config,
                parallel=action_config.get("parallel", False)
            )
            workflow.add_action(action)
        
        return workflow
    
    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Enregistre un workflow"""
        self.workflow_definitions[workflow.id] = workflow
        logger.info(f"Workflow enregistré: {workflow.name} ({workflow.id})")
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Récupère un workflow par son ID"""
        return self.workflow_definitions.get(workflow_id)
    
    def list_workflows(self, tenant_id: Optional[str] = None) -> List[WorkflowDefinition]:
        """Liste les workflows disponibles"""
        workflows = list(self.workflow_definitions.values())
        
        if tenant_id:
            workflows = [w for w in workflows if w.tenant_id == tenant_id or w.tenant_id == ""]
        
        return [w for w in workflows if w.is_active]
    
    async def trigger_workflow(self, workflow_id: str, trigger_data: Dict[str, Any], 
                             incident: Optional[IncidentEvent] = None) -> Optional[WorkflowExecution]:
        """Déclenche l'exécution d'un workflow"""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            logger.error(f"Workflow non trouvé: {workflow_id}")
            return None
        
        # Vérification des conditions de déclenchement
        if not self._check_trigger_conditions(workflow, trigger_data, incident):
            logger.info(f"Conditions de déclenchement non remplies pour le workflow {workflow_id}")
            return None
        
        # Création de l'exécution
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_definition=workflow,
            incident_id=incident.id if incident else None,
            tenant_id=incident.tenant_id if incident else "",
            trigger_data=trigger_data,
            status=WorkflowStatus.PENDING
        )
        
        # Sauvegarde de l'état initial
        await self.state_manager.save_execution(execution)
        
        # Lancement de l'exécution asynchrone
        task = asyncio.create_task(self._execute_workflow(execution))
        self.running_executions[execution.id] = task
        
        logger.info(f"Workflow {workflow_id} déclenché (exécution {execution.id})")
        return execution
    
    def _check_trigger_conditions(self, workflow: WorkflowDefinition, 
                                 trigger_data: Dict[str, Any], 
                                 incident: Optional[IncidentEvent]) -> bool:
        """Vérifie les conditions de déclenchement"""
        conditions = workflow.trigger_conditions
        
        if not conditions:
            return True
        
        # Vérification des conditions sur l'incident
        if incident:
            # Condition sur la sévérité
            if "severity" in conditions:
                required_severities = conditions["severity"]
                if isinstance(required_severities, str):
                    required_severities = [required_severities]
                
                if incident.severity and incident.severity.name.lower() not in [s.lower() for s in required_severities]:
                    return False
            
            # Condition sur la catégorie
            if "category" in conditions:
                required_categories = conditions["category"]
                if isinstance(required_categories, str):
                    required_categories = [required_categories]
                
                if incident.category and incident.category.name.lower() not in [c.lower() for c in required_categories]:
                    return False
            
            # Condition sur les mots-clés
            if "keywords" in conditions:
                keywords = conditions["keywords"]
                if isinstance(keywords, str):
                    keywords = [keywords]
                
                text = f"{incident.title} {incident.description}".lower()
                if not any(keyword.lower() in text for keyword in keywords):
                    return False
        
        return True
    
    async def _execute_workflow(self, execution: WorkflowExecution) -> None:
        """Exécute un workflow"""
        workflow = execution.workflow_definition
        if not workflow:
            logger.error(f"Définition de workflow manquante pour l'exécution {execution.id}")
            return
        
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.utcnow()
            await self.state_manager.save_execution(execution)
            
            logger.info(f"Début de l'exécution du workflow {workflow.name} ({execution.id})")
            
            # Contexte d'exécution
            context = {
                "execution_id": execution.id,
                "workflow_id": workflow.id,
                "incident_id": execution.incident_id,
                "tenant_id": execution.tenant_id,
                "trigger_data": execution.trigger_data
            }
            
            # Exécution des actions selon le mode
            if workflow.execution_mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential(execution, context)
            elif workflow.execution_mode == ExecutionMode.PARALLEL:
                await self._execute_parallel(execution, context)
            elif workflow.execution_mode == ExecutionMode.CONDITIONAL:
                await self._execute_conditional(execution, context)
            
            # Finalisation
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.execution_time_seconds = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.info(f"Workflow {workflow.name} terminé avec succès en {execution.execution_time_seconds:.2f}s")
            
        except asyncio.TimeoutError:
            execution.status = WorkflowStatus.TIMEOUT
            execution.error_message = f"Timeout après {workflow.timeout_seconds} secondes"
            logger.error(f"Timeout du workflow {workflow.name}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            logger.error(f"Erreur lors de l'exécution du workflow {workflow.name}: {e}")
            
        finally:
            # Sauvegarde finale
            await self.state_manager.save_execution(execution)
            
            # Nettoyage
            if execution.id in self.running_executions:
                del self.running_executions[execution.id]
            
            # Mise à jour des métriques
            registry = core_registry
            if execution.status == WorkflowStatus.COMPLETED:
                registry.update_metrics(
                    completed_workflows=registry.get_metrics().completed_workflows + 1
                )
            else:
                registry.update_metrics(
                    failed_workflows=registry.get_metrics().failed_workflows + 1
                )
    
    async def _execute_sequential(self, execution: WorkflowExecution, context: Dict[str, Any]) -> None:
        """Exécution séquentielle des actions"""
        workflow = execution.workflow_definition
        
        for action in workflow.actions:
            if action.id in execution.completed_actions:
                continue  # Action déjà complétée (reprise depuis checkpoint)
            
            execution.current_action_id = action.id
            await self.state_manager.save_execution(execution)
            
            # Vérification des dépendances
            if action.depends_on:
                missing_deps = [dep for dep in action.depends_on if dep not in execution.completed_actions]
                if missing_deps:
                    logger.warning(f"Dépendances manquantes pour l'action {action.id}: {missing_deps}")
                    continue
            
            # Exécution de l'action
            result = await self._execute_action(action, context)
            execution.action_results[action.id] = result.result_data
            
            if result.success:
                execution.completed_actions.append(action.id)
                logger.info(f"Action {action.name} complétée avec succès")
            else:
                execution.failed_actions.append(action.id)
                logger.error(f"Action {action.name} échouée: {result.error_message}")
                
                if action.is_critical:
                    raise Exception(f"Action critique échouée: {action.name}")
            
            # Sauvegarde checkpoint
            await self.state_manager.save_checkpoint(execution)
    
    async def _execute_parallel(self, execution: WorkflowExecution, context: Dict[str, Any]) -> None:
        """Exécution parallèle des actions"""
        workflow = execution.workflow_definition
        
        # Groupement des actions parallèles
        parallel_groups = []
        current_group = []
        
        for action in workflow.actions:
            if action.parallel and current_group and not current_group[-1].parallel:
                # Fin du groupe parallèle
                parallel_groups.append(current_group)
                current_group = [action]
            elif action.parallel:
                current_group.append(action)
            else:
                if current_group:
                    parallel_groups.append(current_group)
                    current_group = []
                parallel_groups.append([action])
        
        if current_group:
            parallel_groups.append(current_group)
        
        # Exécution des groupes
        for group in parallel_groups:
            if len(group) == 1:
                # Action séquentielle
                action = group[0]
                execution.current_action_id = action.id
                result = await self._execute_action(action, context)
                execution.action_results[action.id] = result.result_data
                
                if result.success:
                    execution.completed_actions.append(action.id)
                else:
                    execution.failed_actions.append(action.id)
                    if action.is_critical:
                        raise Exception(f"Action critique échouée: {action.name}")
            else:
                # Actions parallèles
                tasks = []
                for action in group:
                    if action.id not in execution.completed_actions:
                        task = asyncio.create_task(self._execute_action(action, context))
                        tasks.append((action, task))
                
                # Attente de toutes les actions parallèles
                for action, task in tasks:
                    try:
                        result = await task
                        execution.action_results[action.id] = result.result_data
                        
                        if result.success:
                            execution.completed_actions.append(action.id)
                        else:
                            execution.failed_actions.append(action.id)
                            if action.is_critical:
                                # Annuler les autres tâches
                                for _, other_task in tasks:
                                    if not other_task.done():
                                        other_task.cancel()
                                raise Exception(f"Action critique échouée: {action.name}")
                    except Exception as e:
                        execution.failed_actions.append(action.id)
                        logger.error(f"Erreur lors de l'exécution de l'action {action.name}: {e}")
    
    async def _execute_conditional(self, execution: WorkflowExecution, context: Dict[str, Any]) -> None:
        """Exécution conditionnelle des actions"""
        # Implémentation basique de la logique conditionnelle
        # Dans un vrai système, ceci serait plus sophistiqué avec un parser de conditions
        
        workflow = execution.workflow_definition
        
        for action in workflow.actions:
            # Évaluation des conditions
            if action.conditions:
                if not self._evaluate_conditions(action.conditions, context, execution):
                    logger.info(f"Conditions non remplies pour l'action {action.name}, ignorée")
                    continue
            
            # Exécution de l'action
            execution.current_action_id = action.id
            result = await self._execute_action(action, context)
            execution.action_results[action.id] = result.result_data
            
            if result.success:
                execution.completed_actions.append(action.id)
            else:
                execution.failed_actions.append(action.id)
                if action.is_critical:
                    raise Exception(f"Action critique échouée: {action.name}")
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], 
                           context: Dict[str, Any], 
                           execution: WorkflowExecution) -> bool:
        """Évalue les conditions d'exécution d'une action"""
        # Implémentation simplifiée
        for condition_key, condition_value in conditions.items():
            if condition_key == "previous_action_success":
                previous_action_id = condition_value
                if previous_action_id in execution.failed_actions:
                    return False
            elif condition_key == "incident_severity":
                # Vérification de la sévérité de l'incident
                # Dans un vrai système, on récupérerait l'incident depuis le contexte
                pass
        
        return True
    
    async def _execute_action(self, action: WorkflowAction, context: Dict[str, Any]) -> ActionResult:
        """Exécute une action individuelle"""
        executor = self.action_executors.get(action.type)
        if not executor:
            return ActionResult(
                action_id=action.id,
                success=False,
                error_message=f"Exécuteur non trouvé pour le type d'action: {action.type.value}"
            )
        
        # Retry logic
        last_exception = None
        for attempt in range(action.max_retries + 1):
            try:
                # Exécution avec timeout
                result = await asyncio.wait_for(
                    executor.execute(action, context),
                    timeout=action.timeout_seconds
                )
                
                if result.success or attempt == action.max_retries:
                    result.retry_count = attempt
                    return result
                else:
                    logger.warning(f"Action {action.name} échouée (tentative {attempt + 1}/{action.max_retries + 1})")
                    await asyncio.sleep(2 ** attempt)  # Backoff exponentiel
                    
            except asyncio.TimeoutError:
                last_exception = f"Timeout après {action.timeout_seconds} secondes"
                logger.warning(f"Timeout de l'action {action.name} (tentative {attempt + 1}/{action.max_retries + 1})")
                
            except Exception as e:
                last_exception = str(e)
                logger.warning(f"Erreur lors de l'exécution de l'action {action.name}: {e} (tentative {attempt + 1}/{action.max_retries + 1})")
                
                if attempt < action.max_retries:
                    await asyncio.sleep(2 ** attempt)
        
        # Toutes les tentatives ont échoué
        return ActionResult(
            action_id=action.id,
            success=False,
            error_message=last_exception or "Toutes les tentatives ont échoué",
            retry_count=action.max_retries
        )
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Annule l'exécution d'un workflow"""
        if execution_id in self.running_executions:
            task = self.running_executions[execution_id]
            task.cancel()
            
            # Mise à jour de l'état
            execution = await self.state_manager.load_execution(execution_id)
            if execution:
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.utcnow()
                await self.state_manager.save_execution(execution)
            
            logger.info(f"Exécution de workflow {execution_id} annulée")
            return True
        
        return False
    
    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Récupère le statut d'exécution d'un workflow"""
        return await self.state_manager.load_execution(execution_id)
    
    async def list_active_executions(self) -> List[WorkflowExecution]:
        """Liste les exécutions actives"""
        active_executions = []
        for execution in self.state_manager.executions.values():
            if not execution.is_completed:
                active_executions.append(execution)
        return active_executions

# ===========================
# Response Orchestrator Principal
# ===========================

class ResponseOrchestrator:
    """Orchestrateur principal des réponses automatisées"""
    
    def __init__(self):
        self.workflow_engine = WorkflowEngine()
        self.auto_trigger_enabled = True
        
        logger.info("Orchestrateur de réponses initialisé")
    
    async def handle_incident_event(self, incident: IncidentEvent, event_type: str = "created") -> List[WorkflowExecution]:
        """Gère un événement d'incident et déclenche les workflows appropriés"""
        if not self.auto_trigger_enabled:
            return []
        
        triggered_executions = []
        
        # Recherche des workflows applicables
        applicable_workflows = self._find_applicable_workflows(incident, event_type)
        
        for workflow in applicable_workflows:
            trigger_data = {
                "event_type": event_type,
                "incident_data": incident.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            execution = await self.workflow_engine.trigger_workflow(
                workflow.id, 
                trigger_data, 
                incident
            )
            
            if execution:
                triggered_executions.append(execution)
        
        logger.info(f"Déclenché {len(triggered_executions)} workflows pour l'incident {incident.id}")
        return triggered_executions
    
    def _find_applicable_workflows(self, incident: IncidentEvent, event_type: str) -> List[WorkflowDefinition]:
        """Trouve les workflows applicables à un incident"""
        applicable = []
        
        workflows = self.workflow_engine.list_workflows(incident.tenant_id)
        
        for workflow in workflows:
            # Vérification du type de déclencheur
            if workflow.trigger_type == TriggerType.INCIDENT_CREATED and event_type == "created":
                if self.workflow_engine._check_trigger_conditions(workflow, {}, incident):
                    applicable.append(workflow)
            elif workflow.trigger_type == TriggerType.INCIDENT_UPDATED and event_type == "updated":
                if self.workflow_engine._check_trigger_conditions(workflow, {}, incident):
                    applicable.append(workflow)
        
        return applicable
    
    async def create_custom_workflow(self, workflow_def: Dict[str, Any]) -> WorkflowDefinition:
        """Crée un workflow personnalisé"""
        workflow = WorkflowDefinition(
            name=workflow_def.get("name", "Custom Workflow"),
            description=workflow_def.get("description", ""),
            tenant_id=workflow_def.get("tenant_id", ""),
            trigger_type=TriggerType(workflow_def.get("trigger_type", "manual")),
            trigger_conditions=workflow_def.get("trigger_conditions", {}),
            execution_mode=ExecutionMode(workflow_def.get("execution_mode", "sequential")),
            timeout_seconds=workflow_def.get("timeout_seconds", DEFAULT_WORKFLOW_TIMEOUT)
        )
        
        # Ajout des actions
        for action_def in workflow_def.get("actions", []):
            action = WorkflowAction(
                type=ActionType(action_def["type"]),
                name=action_def.get("name", ""),
                parameters=action_def.get("parameters", {}),
                conditions=action_def.get("conditions", {}),
                timeout_seconds=action_def.get("timeout_seconds", 60),
                max_retries=action_def.get("max_retries", 3),
                is_critical=action_def.get("is_critical", False),
                parallel=action_def.get("parallel", False)
            )
            workflow.add_action(action)
        
        self.workflow_engine.register_workflow(workflow)
        logger.info(f"Workflow personnalisé créé: {workflow.name}")
        
        return workflow
    
    async def execute_manual_workflow(self, workflow_id: str, context: Dict[str, Any]) -> Optional[WorkflowExecution]:
        """Exécute manuellement un workflow"""
        return await self.workflow_engine.trigger_workflow(workflow_id, context)
    
    def enable_auto_trigger(self, enabled: bool = True):
        """Active ou désactive le déclenchement automatique"""
        self.auto_trigger_enabled = enabled
        logger.info(f"Déclenchement automatique {'activé' if enabled else 'désactivé'}")

# ===========================
# Exports
# ===========================

__all__ = [
    "ResponseOrchestrator",
    "WorkflowEngine",
    "WorkflowDefinition",
    "WorkflowExecution", 
    "WorkflowAction",
    "ActionResult",
    "WorkflowStatus",
    "ActionType",
    "ExecutionMode",
    "TriggerType"
]

logger.info("Module Response Orchestration chargé")
