"""
Schémas de workflows d'alertes - Spotify AI Agent
Automatisation avancée des workflows et processus d'alertes
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Literal
from uuid import UUID, uuid4
from enum import Enum
import json

from pydantic import BaseModel, Field, validator, computed_field, ConfigDict

from . import (
    BaseSchema, TimestampMixin, TenantMixin, MetadataMixin,
    AlertLevel, AlertStatus, WarningCategory, Priority, Environment
)


class WorkflowTrigger(str, Enum):
    """Déclencheurs de workflow"""
    ALERT_CREATED = "alert_created"
    ALERT_UPDATED = "alert_updated"
    ALERT_ESCALATED = "alert_escalated"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"
    ALERT_RESOLVED = "alert_resolved"
    ALERT_EXPIRED = "alert_expired"
    CORRELATION_DETECTED = "correlation_detected"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    SCHEDULE_TRIGGERED = "schedule_triggered"
    MANUAL_TRIGGER = "manual_trigger"
    WEBHOOK_RECEIVED = "webhook_received"


class ActionType(str, Enum):
    """Types d'actions de workflow"""
    SEND_NOTIFICATION = "send_notification"
    CREATE_TICKET = "create_ticket"
    UPDATE_TICKET = "update_ticket"
    EXECUTE_SCRIPT = "execute_script"
    CALL_WEBHOOK = "call_webhook"
    SEND_EMAIL = "send_email"
    CREATE_ALERT = "create_alert"
    UPDATE_ALERT = "update_alert"
    SUPPRESS_ALERT = "suppress_alert"
    ESCALATE_ALERT = "escalate_alert"
    RUN_AUTOMATION = "run_automation"
    WAIT_DELAY = "wait_delay"
    CONDITION_CHECK = "condition_check"
    PARALLEL_EXECUTION = "parallel_execution"
    LOOP = "loop"
    STOP_WORKFLOW = "stop_workflow"


class WorkflowStatus(str, Enum):
    """États de workflow"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    ARCHIVED = "archived"


class ExecutionStatus(str, Enum):
    """États d'exécution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class WorkflowCondition(BaseModel):
    """Condition de workflow"""
    
    field: str = Field(..., min_length=1, max_length=100)
    operator: str = Field(...)  # eq, ne, gt, lt, gte, lte, in, not_in, contains, regex
    value: Any = Field(...)
    negate: bool = Field(False)
    
    # Conditions composées
    logical_operator: Optional[Literal["AND", "OR"]] = Field(None)
    sub_conditions: List['WorkflowCondition'] = Field(default_factory=list)
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Évalue la condition"""
        field_value = context.get(self.field)
        
        if field_value is None:
            return False
        
        # Évaluation de base
        result = self._evaluate_single(field_value, self.operator, self.value)
        
        # Négation si nécessaire
        if self.negate:
            result = not result
        
        # Évaluation des sous-conditions
        if self.sub_conditions:
            sub_results = [cond.evaluate(context) for cond in self.sub_conditions]
            
            if self.logical_operator == "OR":
                sub_result = any(sub_results)
            else:  # AND par défaut
                sub_result = all(sub_results)
            
            # Combiner avec le résultat principal
            if self.logical_operator:
                result = result and sub_result if self.logical_operator == "AND" else result or sub_result
        
        return result
    
    def _evaluate_single(self, field_value: Any, operator: str, expected_value: Any) -> bool:
        """Évalue une condition simple"""
        if operator == "eq":
            return field_value == expected_value
        elif operator == "ne":
            return field_value != expected_value
        elif operator == "gt":
            return field_value > expected_value
        elif operator == "lt":
            return field_value < expected_value
        elif operator == "gte":
            return field_value >= expected_value
        elif operator == "lte":
            return field_value <= expected_value
        elif operator == "in":
            return field_value in expected_value
        elif operator == "not_in":
            return field_value not in expected_value
        elif operator == "contains":
            return expected_value in str(field_value)
        elif operator == "regex":
            import re
            return bool(re.search(expected_value, str(field_value)))
        else:
            return False


class WorkflowAction(BaseModel):
    """Action de workflow"""
    
    action_id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    action_type: ActionType = Field(...)
    
    # Configuration de l'action
    config: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Conditions d'exécution
    conditions: List[WorkflowCondition] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)  # IDs des actions précédentes
    
    # Gestion d'erreurs
    retry_attempts: int = Field(0, ge=0, le=10)
    retry_delay_seconds: int = Field(30, ge=1, le=3600)
    continue_on_error: bool = Field(False)
    timeout_seconds: Optional[int] = Field(None, ge=1, le=3600)
    
    # Ordre et flux
    order: int = Field(1, ge=1)
    parallel: bool = Field(False)
    
    def can_execute(self, context: Dict[str, Any]) -> bool:
        """Vérifie si l'action peut être exécutée"""
        # Vérifier les conditions
        if self.conditions:
            return all(condition.evaluate(context) for condition in self.conditions)
        return True


class WorkflowDefinition(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Définition de workflow d'alertes"""
    
    # Informations de base
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    version: str = Field("1.0.0")
    
    # Configuration du workflow
    triggers: List[WorkflowTrigger] = Field(..., min_items=1)
    trigger_conditions: List[WorkflowCondition] = Field(default_factory=list)
    
    # Actions du workflow
    actions: List[WorkflowAction] = Field(..., min_items=1)
    
    # Filtres d'activation
    severity_filter: List[AlertLevel] = Field(default_factory=list)
    category_filter: List[WarningCategory] = Field(default_factory=list)
    environment_filter: List[Environment] = Field(default_factory=list)
    
    # Configuration temporelle
    schedule: Optional[str] = Field(None)  # Expression cron pour déclenchement programmé
    timezone: str = Field("UTC")
    effective_from: Optional[datetime] = Field(None)
    effective_until: Optional[datetime] = Field(None)
    
    # Limitation et contrôle
    max_executions_per_hour: int = Field(100, ge=1, le=10000)
    max_concurrent_executions: int = Field(10, ge=1, le=100)
    cooldown_period_minutes: int = Field(0, ge=0, le=1440)
    
    # État et performance
    status: WorkflowStatus = Field(WorkflowStatus.DRAFT)
    enabled: bool = Field(False)
    
    # Métriques
    total_executions: int = Field(0, ge=0)
    successful_executions: int = Field(0, ge=0)
    failed_executions: int = Field(0, ge=0)
    last_execution: Optional[datetime] = Field(None)
    avg_execution_time_seconds: Optional[float] = Field(None, ge=0)
    
    # Audit et approval
    created_by: Optional[UUID] = Field(None)
    approved_by: Optional[UUID] = Field(None)
    approved_at: Optional[datetime] = Field(None)
    approval_required: bool = Field(True)
    
    # Tags et organisation
    tags: Set[str] = Field(default_factory=set)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    @validator('actions')
    def validate_actions(cls, v):
        """Valide les actions du workflow"""
        if not v:
            raise ValueError('At least one action is required')
        
        # Vérifier l'unicité des IDs d'action
        action_ids = [action.action_id for action in v]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError('Action IDs must be unique')
        
        # Vérifier les dépendances
        for action in v:
            for dep in action.depends_on:
                if dep not in action_ids:
                    raise ValueError(f'Action {action.action_id} depends on non-existent action {dep}')
        
        return v

    @computed_field
    @property
    def success_rate(self) -> float:
        """Taux de succès du workflow"""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100

    @computed_field
    @property
    def is_active(self) -> bool:
        """Indique si le workflow est actif"""
        if not self.enabled or self.status != WorkflowStatus.ACTIVE:
            return False
        
        now = datetime.now(timezone.utc)
        
        if self.effective_from and now < self.effective_from:
            return False
        
        if self.effective_until and now > self.effective_until:
            return False
        
        return True

    def can_trigger(self, trigger: WorkflowTrigger, context: Dict[str, Any]) -> bool:
        """Vérifie si le workflow peut être déclenché"""
        if not self.is_active:
            return False
        
        if trigger not in self.triggers:
            return False
        
        # Vérifier les conditions de déclenchement
        if self.trigger_conditions:
            return all(condition.evaluate(context) for condition in self.trigger_conditions)
        
        return True

    def get_executable_actions(self, context: Dict[str, Any], 
                              completed_actions: Set[str]) -> List[WorkflowAction]:
        """Obtient les actions prêtes à être exécutées"""
        executable = []
        
        for action in self.actions:
            # Vérifier si l'action peut être exécutée
            if not action.can_execute(context):
                continue
            
            # Vérifier les dépendances
            if action.depends_on:
                if not all(dep in completed_actions for dep in action.depends_on):
                    continue
            
            executable.append(action)
        
        # Trier par ordre
        executable.sort(key=lambda x: x.order)
        
        return executable


class WorkflowExecution(BaseSchema, TimestampMixin, TenantMixin):
    """Exécution d'un workflow"""
    
    execution_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID = Field(...)
    alert_id: Optional[UUID] = Field(None)
    
    # Déclenchement
    trigger: WorkflowTrigger = Field(...)
    trigger_context: Dict[str, Any] = Field(default_factory=dict)
    
    # État d'exécution
    status: ExecutionStatus = Field(ExecutionStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    # Actions exécutées
    completed_actions: Set[str] = Field(default_factory=set)
    failed_actions: Set[str] = Field(default_factory=set)
    skipped_actions: Set[str] = Field(default_factory=set)
    
    # Progression
    total_actions: int = Field(0, ge=0)
    current_action: Optional[str] = Field(None)
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    
    # Résultats
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(None)
    
    # Performance
    execution_duration_seconds: Optional[float] = Field(None, ge=0)
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Indique si l'exécution est terminée"""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.TIMEOUT
        ]

    @computed_field
    @property
    def is_running(self) -> bool:
        """Indique si l'exécution est en cours"""
        return self.status == ExecutionStatus.RUNNING

    def add_log_entry(self, action_id: str, message: str, level: str = "info", 
                     details: Optional[Dict[str, Any]] = None):
        """Ajoute une entrée au log d'exécution"""
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action_id': action_id,
            'level': level,
            'message': message,
            'details': details or {}
        }
        self.execution_log.append(entry)

    def mark_action_completed(self, action_id: str, success: bool = True, 
                             output: Optional[Dict[str, Any]] = None):
        """Marque une action comme terminée"""
        if success:
            self.completed_actions.add(action_id)
            if action_id in self.failed_actions:
                self.failed_actions.remove(action_id)
        else:
            self.failed_actions.add(action_id)
            if action_id in self.completed_actions:
                self.completed_actions.remove(action_id)
        
        if output:
            self.output_data[action_id] = output
        
        # Mettre à jour le progrès
        completed_count = len(self.completed_actions) + len(self.failed_actions) + len(self.skipped_actions)
        if self.total_actions > 0:
            self.progress_percentage = (completed_count / self.total_actions) * 100

    def get_execution_summary(self) -> Dict[str, Any]:
        """Obtient un résumé de l'exécution"""
        return {
            'execution_id': str(self.execution_id),
            'workflow_id': str(self.workflow_id),
            'status': self.status,
            'duration_seconds': self.execution_duration_seconds,
            'progress_percentage': self.progress_percentage,
            'total_actions': self.total_actions,
            'completed_actions': len(self.completed_actions),
            'failed_actions': len(self.failed_actions),
            'skipped_actions': len(self.skipped_actions),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message
        }


class WorkflowSchedule(BaseSchema, TimestampMixin, TenantMixin):
    """Planification de workflow"""
    
    schedule_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID = Field(...)
    name: str = Field(..., min_length=1, max_length=255)
    
    # Configuration de planification
    cron_expression: str = Field(...)
    timezone: str = Field("UTC")
    
    # État
    enabled: bool = Field(True)
    next_execution: Optional[datetime] = Field(None)
    last_execution: Optional[datetime] = Field(None)
    
    # Contexte par défaut
    default_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Limites
    max_executions: Optional[int] = Field(None, ge=1)
    execution_count: int = Field(0, ge=0)
    
    @validator('cron_expression')
    def validate_cron_expression(cls, v):
        """Valide l'expression cron"""
        try:
            from croniter import croniter
            croniter(v)
        except Exception:
            raise ValueError('Invalid cron expression')
        return v

    @computed_field
    @property
    def is_due(self) -> bool:
        """Indique si l'exécution est due"""
        if not self.enabled or not self.next_execution:
            return False
        
        return datetime.now(timezone.utc) >= self.next_execution

    def calculate_next_execution(self):
        """Calcule la prochaine exécution"""
        try:
            from croniter import croniter
            base = self.last_execution or datetime.now(timezone.utc)
            cron = croniter(self.cron_expression, base)
            self.next_execution = cron.get_next(datetime)
        except Exception:
            self.next_execution = None


class WorkflowTemplate(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Template de workflow réutilisable"""
    
    template_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Template du workflow
    workflow_template: Dict[str, Any] = Field(...)
    template_variables: Dict[str, Any] = Field(default_factory=dict)
    
    # Métadonnées
    category: str = Field(..., min_length=1, max_length=100)
    version: str = Field("1.0.0")
    author: Optional[UUID] = Field(None)
    
    # Usage
    usage_count: int = Field(0, ge=0)
    last_used: Optional[datetime] = Field(None)
    
    # État
    public: bool = Field(False)
    verified: bool = Field(False)
    
    # Tags
    tags: Set[str] = Field(default_factory=set)
    
    def instantiate(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Instancie un workflow à partir du template"""
        import copy
        
        workflow_data = copy.deepcopy(self.workflow_template)
        
        # Remplacer les variables dans le template
        def replace_variables(obj):
            if isinstance(obj, dict):
                return {k: replace_variables(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_variables(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                return variables.get(var_name, self.template_variables.get(var_name, obj))
            else:
                return obj
        
        return replace_variables(workflow_data)


# Rebuild des modèles avec forward references
WorkflowCondition.model_rebuild()


__all__ = [
    'WorkflowTrigger', 'ActionType', 'WorkflowStatus', 'ExecutionStatus',
    'WorkflowCondition', 'WorkflowAction', 'WorkflowDefinition', 'WorkflowExecution',
    'WorkflowSchedule', 'WorkflowTemplate'
]
