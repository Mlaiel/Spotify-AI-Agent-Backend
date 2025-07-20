"""
Système de workflow avancé pour le système de tenancy
Auteur: Fahed Mlaiel - Lead Dev & Architecte IA
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Callable, Type, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger(__name__)

class WorkflowStatus(Enum):
    """Statuts des workflows"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStatus(Enum):
    """Statuts des tâches"""
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"

class TaskType(Enum):
    """Types de tâches"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    CUSTOM = "custom"

@dataclass
class TaskResult:
    """Résultat d'exécution d'une tâche"""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class WorkflowContext:
    """Contexte d'exécution du workflow"""
    workflow_id: str
    tenant_id: str
    user_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, TaskResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

class BaseTask(ABC):
    """Tâche de base abstraite"""
    
    def __init__(self, task_id: str, name: str, description: str = ""):
        self.task_id = task_id
        self.name = name
        self.description = description
        self.dependencies: Set[str] = set()
        self.retry_count = 0
        self.max_retries = 3
        self.timeout = 300  # 5 minutes par défaut
        self.task_type = TaskType.CUSTOM
        
    def add_dependency(self, task_id: str) -> None:
        """Ajoute une dépendance"""
        self.dependencies.add(task_id)
        
    def remove_dependency(self, task_id: str) -> None:
        """Supprime une dépendance"""
        self.dependencies.discard(task_id)
        
    @abstractmethod
    async def execute(self, context: WorkflowContext) -> TaskResult:
        """Exécute la tâche"""
        pass
        
    async def can_execute(self, context: WorkflowContext) -> bool:
        """Vérifie si la tâche peut être exécutée"""
        for dep_id in self.dependencies:
            if dep_id not in context.results:
                return False
            if context.results[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
        
    async def should_retry(self, error: Exception) -> bool:
        """Détermine si la tâche doit être relancée"""
        return self.retry_count < self.max_retries

class TenantProvisioningTask(BaseTask):
    """Tâche de provisioning d'un tenant"""
    
    def __init__(self, task_id: str):
        super().__init__(task_id, "Tenant Provisioning", "Provisionne un nouveau tenant")
        self.task_type = TaskType.SEQUENTIAL
        
    async def execute(self, context: WorkflowContext) -> TaskResult:
        """Exécute le provisioning du tenant"""
        start_time = datetime.utcnow()
        
        try:
            tenant_config = context.data.get('tenant_config', {})
            
            # Simulation du provisioning
            steps = [
                self._create_database_schema,
                self._setup_storage,
                self._configure_security,
                self._initialize_features,
                self._setup_monitoring
            ]
            
            results = {}
            for step in steps:
                step_result = await step(tenant_config)
                results[step.__name__] = step_result
                
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                output=results,
                execution_time=execution_time,
                metadata={"tenant_id": context.tenant_id}
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error("Erreur lors du provisioning", error=str(e))
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

    async def _create_database_schema(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Crée le schéma de base de données"""
        await asyncio.sleep(1)  # Simulation
        return {"schema_created": True, "tables": ["users", "playlists", "tracks"]}
        
    async def _setup_storage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure le stockage"""
        await asyncio.sleep(0.5)  # Simulation
        return {"storage_configured": True, "quota_gb": config.get("storage_quota_gb", 10)}
        
    async def _configure_security(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure la sécurité"""
        await asyncio.sleep(0.3)  # Simulation
        return {"security_configured": True, "encryption": config.get("encryption_enabled", True)}
        
    async def _initialize_features(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialise les fonctionnalités"""
        await asyncio.sleep(0.7)  # Simulation
        return {"features_initialized": True, "features": config.get("features", [])}
        
    async def _setup_monitoring(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure le monitoring"""
        await asyncio.sleep(0.2)  # Simulation
        return {"monitoring_configured": True, "dashboards_created": True}

class TenantValidationTask(BaseTask):
    """Tâche de validation d'un tenant"""
    
    def __init__(self, task_id: str):
        super().__init__(task_id, "Tenant Validation", "Valide la configuration du tenant")
        
    async def execute(self, context: WorkflowContext) -> TaskResult:
        """Exécute la validation du tenant"""
        start_time = datetime.utcnow()
        
        try:
            tenant_config = context.data.get('tenant_config', {})
            
            # Validation des champs requis
            required_fields = ['tenant_id', 'name', 'email']
            missing_fields = [field for field in required_fields if field not in tenant_config]
            
            if missing_fields:
                raise ValueError(f"Champs manquants: {missing_fields}")
            
            # Validation de l'unicité de l'ID tenant
            if await self._tenant_exists(tenant_config['tenant_id']):
                raise ValueError(f"Tenant ID {tenant_config['tenant_id']} existe déjà")
            
            # Validation des quotas
            self._validate_quotas(tenant_config)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                output={"validation_passed": True},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

    async def _tenant_exists(self, tenant_id: str) -> bool:
        """Vérifie si un tenant existe déjà"""
        # Simulation - en production, vérifier en base
        await asyncio.sleep(0.1)
        return False
        
    def _validate_quotas(self, config: Dict[str, Any]) -> None:
        """Valide les quotas"""
        if 'api_quota_per_hour' in config:
            quota = config['api_quota_per_hour']
            if quota <= 0 or quota > 100000:
                raise ValueError("Quota API invalide")

class NotificationTask(BaseTask):
    """Tâche de notification"""
    
    def __init__(self, task_id: str, notification_type: str = "email"):
        super().__init__(task_id, "Notification", f"Envoie une notification {notification_type}")
        self.notification_type = notification_type
        
    async def execute(self, context: WorkflowContext) -> TaskResult:
        """Exécute l'envoi de notification"""
        start_time = datetime.utcnow()
        
        try:
            tenant_config = context.data.get('tenant_config', {})
            
            if self.notification_type == "email":
                result = await self._send_email(tenant_config, context)
            elif self.notification_type == "slack":
                result = await self._send_slack(tenant_config, context)
            else:
                raise ValueError(f"Type de notification non supporté: {self.notification_type}")
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                output=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

    async def _send_email(self, config: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """Envoie un email"""
        await asyncio.sleep(0.5)  # Simulation
        
        return {
            "email_sent": True,
            "recipient": config.get('email'),
            "subject": f"Tenant {config.get('tenant_id')} créé avec succès",
            "template": "tenant_creation_success"
        }
        
    async def _send_slack(self, config: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """Envoie une notification Slack"""
        await asyncio.sleep(0.2)  # Simulation
        
        return {
            "slack_sent": True,
            "channel": "#tenant-management",
            "message": f"Nouveau tenant créé: {config.get('tenant_id')}"
        }

class ConditionalTask(BaseTask):
    """Tâche conditionnelle"""
    
    def __init__(self, task_id: str, condition: Callable[[WorkflowContext], bool],
                 true_task: BaseTask, false_task: Optional[BaseTask] = None):
        super().__init__(task_id, "Conditional Task", "Exécute une tâche selon une condition")
        self.condition = condition
        self.true_task = true_task
        self.false_task = false_task
        self.task_type = TaskType.CONDITIONAL
        
    async def execute(self, context: WorkflowContext) -> TaskResult:
        """Exécute la tâche conditionnelle"""
        start_time = datetime.utcnow()
        
        try:
            if self.condition(context):
                result = await self.true_task.execute(context)
            elif self.false_task:
                result = await self.false_task.execute(context)
            else:
                result = TaskResult(
                    task_id=self.task_id,
                    status=TaskStatus.SKIPPED,
                    output={"condition_met": False}
                )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

class ParallelTask(BaseTask):
    """Tâche parallèle"""
    
    def __init__(self, task_id: str, tasks: List[BaseTask]):
        super().__init__(task_id, "Parallel Task", "Exécute plusieurs tâches en parallèle")
        self.tasks = tasks
        self.task_type = TaskType.PARALLEL
        
    async def execute(self, context: WorkflowContext) -> TaskResult:
        """Exécute les tâches en parallèle"""
        start_time = datetime.utcnow()
        
        try:
            # Exécution parallèle de toutes les tâches
            results = await asyncio.gather(
                *[task.execute(context) for task in self.tasks],
                return_exceptions=True
            )
            
            # Analyse des résultats
            completed_count = 0
            failed_count = 0
            task_results = {}
            
            for i, result in enumerate(results):
                task = self.tasks[i]
                
                if isinstance(result, Exception):
                    failed_count += 1
                    task_results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error=str(result)
                    )
                else:
                    if result.status == TaskStatus.COMPLETED:
                        completed_count += 1
                    elif result.status == TaskStatus.FAILED:
                        failed_count += 1
                    
                    task_results[task.task_id] = result
            
            # Détermination du statut global
            if failed_count == 0:
                status = TaskStatus.COMPLETED
            elif completed_count > 0:
                status = TaskStatus.COMPLETED  # Succès partiel
            else:
                status = TaskStatus.FAILED
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                task_id=self.task_id,
                status=status,
                output={
                    "parallel_results": task_results,
                    "completed_count": completed_count,
                    "failed_count": failed_count,
                    "total_tasks": len(self.tasks)
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )

class Workflow:
    """Workflow principal"""
    
    def __init__(self, workflow_id: str, name: str, description: str = ""):
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.tasks: Dict[str, BaseTask] = {}
        self.status = WorkflowStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.context: Optional[WorkflowContext] = None
        
    def add_task(self, task: BaseTask) -> None:
        """Ajoute une tâche au workflow"""
        self.tasks[task.task_id] = task
        
    def remove_task(self, task_id: str) -> None:
        """Supprime une tâche du workflow"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            
            # Supprime les dépendances vers cette tâche
            for task in self.tasks.values():
                task.remove_dependency(task_id)
    
    async def execute(self, context: WorkflowContext) -> Dict[str, Any]:
        """Exécute le workflow"""
        self.context = context
        self.status = WorkflowStatus.RUNNING
        self.started_at = datetime.utcnow()
        
        try:
            logger.info("Démarrage du workflow", workflow_id=self.workflow_id)
            
            # Tri topologique des tâches
            execution_order = self._topological_sort()
            
            # Exécution séquentielle selon les dépendances
            for task_id in execution_order:
                task = self.tasks[task_id]
                
                # Vérification des dépendances
                if not await task.can_execute(context):
                    logger.warning("Tâche non exécutable", task_id=task_id)
                    continue
                
                logger.info("Exécution de la tâche", task_id=task_id)
                result = await task.execute(context)
                context.results[task_id] = result
                
                if result.status == TaskStatus.FAILED:
                    logger.error("Échec de la tâche", task_id=task_id, error=result.error)
                    
                    # Gestion des échecs selon la stratégie
                    if not await self._handle_task_failure(task, result, context):
                        self.status = WorkflowStatus.FAILED
                        break
            
            # Détermination du statut final
            if self.status == WorkflowStatus.RUNNING:
                failed_tasks = [r for r in context.results.values() if r.status == TaskStatus.FAILED]
                if failed_tasks:
                    self.status = WorkflowStatus.FAILED
                else:
                    self.status = WorkflowStatus.COMPLETED
            
            self.completed_at = datetime.utcnow()
            
            execution_summary = {
                "workflow_id": self.workflow_id,
                "status": self.status.value,
                "execution_time": (self.completed_at - self.started_at).total_seconds(),
                "tasks_executed": len(context.results),
                "tasks_completed": len([r for r in context.results.values() if r.status == TaskStatus.COMPLETED]),
                "tasks_failed": len([r for r in context.results.values() if r.status == TaskStatus.FAILED]),
                "results": {task_id: result.__dict__ for task_id, result in context.results.items()}
            }
            
            logger.info("Workflow terminé", **execution_summary)
            return execution_summary
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            self.completed_at = datetime.utcnow()
            logger.error("Erreur lors de l'exécution du workflow", error=str(e))
            
            return {
                "workflow_id": self.workflow_id,
                "status": self.status.value,
                "error": str(e),
                "execution_time": (self.completed_at - self.started_at).total_seconds() if self.started_at else 0
            }

    def _topological_sort(self) -> List[str]:
        """Tri topologique des tâches selon leurs dépendances"""
        # Implémentation de l'algorithme de Kahn
        in_degree = {task_id: 0 for task_id in self.tasks}
        
        # Calcul des degrés entrants
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.task_id] += 1
        
        # File des tâches sans dépendances
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            task_id = queue.pop(0)
            result.append(task_id)
            
            # Mise à jour des degrés pour les tâches dépendantes
            for other_task in self.tasks.values():
                if task_id in other_task.dependencies:
                    in_degree[other_task.task_id] -= 1
                    if in_degree[other_task.task_id] == 0:
                        queue.append(other_task.task_id)
        
        if len(result) != len(self.tasks):
            raise ValueError("Dépendances cycliques détectées dans le workflow")
        
        return result

    async def _handle_task_failure(self, task: BaseTask, result: TaskResult, context: WorkflowContext) -> bool:
        """Gère l'échec d'une tâche"""
        if await task.should_retry(Exception(result.error or "Unknown error")):
            task.retry_count += 1
            logger.info("Nouvelle tentative de la tâche", task_id=task.task_id, retry=task.retry_count)
            
            # Nouvelle exécution
            new_result = await task.execute(context)
            context.results[task.task_id] = new_result
            
            return new_result.status != TaskStatus.FAILED
        
        return False

class WorkflowEngine:
    """Moteur de workflow avancé"""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Set[str] = set()
        self.workflow_templates: Dict[str, Callable[[], Workflow]] = {}
        
    def register_template(self, name: str, template_factory: Callable[[], Workflow]) -> None:
        """Enregistre un template de workflow"""
        self.workflow_templates[name] = template_factory
        
    async def create_workflow_from_template(self, template_name: str, 
                                          tenant_id: str, data: Dict[str, Any]) -> str:
        """Crée un workflow à partir d'un template"""
        if template_name not in self.workflow_templates:
            raise ValueError(f"Template {template_name} non trouvé")
        
        workflow = self.workflow_templates[template_name]()
        workflow.workflow_id = str(uuid.uuid4())
        
        self.workflows[workflow.workflow_id] = workflow
        
        return workflow.workflow_id
        
    async def execute_workflow(self, workflow_id: str, tenant_id: str, 
                             data: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Exécute un workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} non trouvé")
        
        if workflow_id in self.running_workflows:
            raise ValueError(f"Workflow {workflow_id} déjà en cours d'exécution")
        
        workflow = self.workflows[workflow_id]
        context = WorkflowContext(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            user_id=user_id,
            data=data
        )
        
        self.running_workflows.add(workflow_id)
        
        try:
            result = await workflow.execute(context)
            return result
        finally:
            self.running_workflows.discard(workflow_id)

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Récupère le statut d'un workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} non trouvé")
        
        workflow = self.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "task_count": len(workflow.tasks),
            "is_running": workflow_id in self.running_workflows
        }

# Templates de workflows prédéfinis
def create_tenant_provisioning_workflow() -> Workflow:
    """Crée un workflow de provisioning de tenant"""
    workflow = Workflow(
        workflow_id="",  # Sera défini lors de la création
        name="Tenant Provisioning",
        description="Workflow complet de création d'un tenant"
    )
    
    # Tâches du workflow
    validation_task = TenantValidationTask("validate_tenant")
    provisioning_task = TenantProvisioningTask("provision_tenant")
    email_notification = NotificationTask("notify_email", "email")
    slack_notification = NotificationTask("notify_slack", "slack")
    
    # Définition des dépendances
    provisioning_task.add_dependency("validate_tenant")
    email_notification.add_dependency("provision_tenant")
    slack_notification.add_dependency("provision_tenant")
    
    # Ajout des tâches
    workflow.add_task(validation_task)
    workflow.add_task(provisioning_task)
    workflow.add_task(email_notification)
    workflow.add_task(slack_notification)
    
    return workflow

# Instance globale du moteur de workflow
workflow_engine = WorkflowEngine()

# Enregistrement des templates
workflow_engine.register_template("tenant_provisioning", create_tenant_provisioning_workflow)
