"""
Slack Escalation Manager - Gestionnaire intelligent d'escalade des alertes
Système multi-niveau avec SLA, planification et intégration PagerDuty
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

import aioredis
import asyncpg
from .slack_alert_manager import AlertData, AlertSeverity, AlertStatus


class EscalationLevel(str, Enum):
    """Niveaux d'escalade"""
    L1 = "l1"  # Support niveau 1
    L2 = "l2"  # Support niveau 2
    L3 = "l3"  # Experts techniques
    MANAGEMENT = "management"  # Management
    EXECUTIVE = "executive"   # Direction
    EXTERNAL = "external"     # Support externe


class EscalationTrigger(str, Enum):
    """Déclencheurs d'escalade"""
    TIME_BASED = "time_based"
    FAILURE_COUNT = "failure_count"
    SEVERITY_BASED = "severity_based"
    MANUAL = "manual"
    AI_PREDICTED = "ai_predicted"


class EscalationAction(str, Enum):
    """Actions d'escalade"""
    NOTIFY_SLACK = "notify_slack"
    SEND_EMAIL = "send_email"
    CREATE_TICKET = "create_ticket"
    CALL_PHONE = "call_phone"
    PAGE_ONCALL = "page_oncall"
    TRIGGER_RUNBOOK = "trigger_runbook"


@dataclass
class EscalationStep:
    """Étape d'escalade"""
    level: EscalationLevel
    delay_minutes: int
    target_users: List[str] = field(default_factory=list)
    target_channels: List[str] = field(default_factory=list)
    actions: List[EscalationAction] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    repeat_count: int = 1
    repeat_interval: int = 15  # minutes


@dataclass
class EscalationPolicy:
    """Politique d'escalade"""
    policy_id: str
    name: str
    description: str
    tenant_id: str
    steps: List[EscalationStep]
    max_escalation_time: int = 240  # minutes
    business_hours_only: bool = False
    severity_filter: List[AlertSeverity] = field(default_factory=list)
    service_filter: List[str] = field(default_factory=list)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EscalationExecution:
    """Exécution d'escalade"""
    execution_id: str
    alert_id: str
    policy_id: str
    tenant_id: str
    current_level: EscalationLevel
    current_step: int
    started_at: datetime
    next_escalation_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    status: str = "active"  # active, completed, cancelled, failed
    execution_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OnCallSchedule:
    """Planning de garde"""
    schedule_id: str
    name: str
    tenant_id: str
    timezone: str = "Europe/Paris"
    rotations: List[Dict[str, Any]] = field(default_factory=list)
    escalation_policy: Optional[str] = None
    active: bool = True


class SlackEscalationManager:
    """
    Gestionnaire d'escalade avancé avec:
    - Politiques d'escalade multi-niveau configurables
    - Intégration avec les plannings de garde
    - Escalade intelligente basée sur l'IA
    - Support des SLA et métriques de performance
    - Intégration PagerDuty et autres systèmes
    - Gestion des heures ouvrables et fuseaux horaires
    """

    def __init__(self):
        self.redis_pool = None
        self.postgres_pool = None
        self.logger = logging.getLogger(__name__)
        
        # Cache des politiques et plannings
        self.policies_cache = {}
        self.schedules_cache = {}
        
        # Exécutions actives
        self.active_escalations = {}
        
        # Configuration par défaut
        self.default_config = {
            "max_concurrent_escalations": 100,
            "default_escalation_timeout": 240,  # 4 heures
            "business_hours": {
                "start": "09:00",
                "end": "18:00",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
            },
            "emergency_override": True,
            "ai_prediction_enabled": True
        }
        
        # Métriques
        self.metrics = {
            "escalations_started": 0,
            "escalations_completed": 0,
            "escalations_cancelled": 0,
            "average_resolution_time": 0.0,
            "sla_breaches": 0
        }

    async def initialize(self, redis_pool: aioredis.Redis, postgres_pool: asyncpg.Pool):
        """Initialise le gestionnaire d'escalade"""
        self.redis_pool = redis_pool
        self.postgres_pool = postgres_pool
        
        try:
            # Création du schéma de base de données
            await self._create_database_schema()
            
            # Chargement des politiques et plannings
            await self._load_escalation_policies()
            await self._load_oncall_schedules()
            
            # Récupération des escalades actives
            await self._restore_active_escalations()
            
            # Démarrage du worker d'escalade
            asyncio.create_task(self._escalation_worker())
            
            # Démarrage du worker de monitoring SLA
            asyncio.create_task(self._sla_monitor_worker())
            
            self.logger.info("SlackEscalationManager initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise

    async def setup_escalation(
        self,
        alert_id: str,
        policy: str,
        tenant_id: str,
        custom_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Configure une escalade pour une alerte
        
        Args:
            alert_id: ID de l'alerte
            policy: ID de la politique d'escalade
            tenant_id: ID du tenant
            custom_context: Contexte personnalisé
            
        Returns:
            ID de l'exécution d'escalade
        """
        try:
            execution_id = str(uuid.uuid4())
            
            # Récupération de la politique
            escalation_policy = await self._get_escalation_policy(policy, tenant_id)
            if not escalation_policy:
                raise ValueError(f"Politique d'escalade {policy} non trouvée")
            
            # Vérification si une escalade est déjà en cours
            existing_execution = await self._get_active_escalation(alert_id)
            if existing_execution:
                self.logger.warning(f"Escalade déjà en cours pour l'alerte {alert_id}")
                return existing_execution.execution_id
            
            # Création de l'exécution d'escalade
            execution = EscalationExecution(
                execution_id=execution_id,
                alert_id=alert_id,
                policy_id=policy,
                tenant_id=tenant_id,
                current_level=escalation_policy.steps[0].level,
                current_step=0,
                started_at=datetime.utcnow()
            )
            
            # Calcul du prochain délai d'escalade
            first_step = escalation_policy.steps[0]
            execution.next_escalation_at = datetime.utcnow() + timedelta(minutes=first_step.delay_minutes)
            
            # Stockage en base et cache
            await self._store_escalation_execution(execution)
            self.active_escalations[execution_id] = execution
            
            # Log d'audit
            await self._log_escalation_event(
                execution_id,
                "escalation_started",
                {
                    "alert_id": alert_id,
                    "policy_id": policy,
                    "first_escalation_at": execution.next_escalation_at.isoformat()
                }
            )
            
            # Métriques
            self.metrics["escalations_started"] += 1
            
            self.logger.info(f"Escalade {execution_id} configurée pour l'alerte {alert_id}")
            
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration de l'escalade: {e}")
            raise

    async def cancel_escalation(self, alert_id: str) -> bool:
        """Annule l'escalade d'une alerte"""
        try:
            execution = await self._get_active_escalation(alert_id)
            if not execution:
                return False
            
            # Mise à jour du statut
            execution.status = "cancelled"
            execution.cancelled_at = datetime.utcnow()
            
            # Suppression du cache actif
            if execution.execution_id in self.active_escalations:
                del self.active_escalations[execution.execution_id]
            
            # Mise à jour en base
            await self._update_escalation_execution(execution)
            
            # Log d'audit
            await self._log_escalation_event(
                execution.execution_id,
                "escalation_cancelled",
                {"alert_id": alert_id, "cancelled_at": execution.cancelled_at.isoformat()}
            )
            
            # Métriques
            self.metrics["escalations_cancelled"] += 1
            
            self.logger.info(f"Escalade {execution.execution_id} annulée pour l'alerte {alert_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'annulation de l'escalade: {e}")
            return False

    async def escalate_manually(
        self,
        alert_id: str,
        target_level: EscalationLevel,
        user_id: str,
        reason: str
    ) -> bool:
        """Déclenche une escalade manuelle"""
        try:
            execution = await self._get_active_escalation(alert_id)
            if not execution:
                self.logger.warning(f"Aucune escalade active pour l'alerte {alert_id}")
                return False
            
            # Recherche du niveau cible dans la politique
            policy = await self._get_escalation_policy(execution.policy_id, execution.tenant_id)
            target_step_index = None
            
            for i, step in enumerate(policy.steps):
                if step.level == target_level:
                    target_step_index = i
                    break
            
            if target_step_index is None:
                raise ValueError(f"Niveau {target_level} non trouvé dans la politique")
            
            # Saut vers le niveau cible
            execution.current_step = target_step_index
            execution.current_level = target_level
            execution.next_escalation_at = datetime.utcnow()
            
            # Exécution immédiate
            await self._execute_escalation_step(execution, policy.steps[target_step_index])
            
            # Log d'audit
            await self._log_escalation_event(
                execution.execution_id,
                "manual_escalation",
                {
                    "user_id": user_id,
                    "target_level": target_level.value,
                    "reason": reason
                }
            )
            
            self.logger.info(f"Escalade manuelle vers {target_level} pour {alert_id} par {user_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'escalade manuelle: {e}")
            return False

    async def get_escalation_status(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Récupère le statut d'escalade d'une alerte"""
        try:
            execution = await self._get_active_escalation(alert_id)
            if not execution:
                return None
            
            policy = await self._get_escalation_policy(execution.policy_id, execution.tenant_id)
            
            # Calcul du temps écoulé et restant
            elapsed_time = datetime.utcnow() - execution.started_at
            time_to_next = None
            
            if execution.next_escalation_at:
                time_to_next = execution.next_escalation_at - datetime.utcnow()
                if time_to_next.total_seconds() < 0:
                    time_to_next = timedelta(0)
            
            return {
                "execution_id": execution.execution_id,
                "alert_id": alert_id,
                "policy_name": policy.name if policy else "Unknown",
                "current_level": execution.current_level.value,
                "current_step": execution.current_step + 1,
                "total_steps": len(policy.steps) if policy else 0,
                "status": execution.status,
                "elapsed_time": str(elapsed_time),
                "time_to_next_escalation": str(time_to_next) if time_to_next else None,
                "started_at": execution.started_at.isoformat(),
                "execution_log": execution.execution_log
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du statut: {e}")
            return None

    async def create_escalation_policy(self, policy: EscalationPolicy) -> bool:
        """Crée une nouvelle politique d'escalade"""
        try:
            # Validation de la politique
            await self._validate_escalation_policy(policy)
            
            # Stockage en base de données
            await self._store_escalation_policy(policy)
            
            # Mise à jour du cache
            cache_key = f"{policy.tenant_id}:{policy.policy_id}"
            self.policies_cache[cache_key] = policy
            
            self.logger.info(f"Politique d'escalade {policy.policy_id} créée pour {policy.tenant_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de la politique: {e}")
            return False

    async def create_oncall_schedule(self, schedule: OnCallSchedule) -> bool:
        """Crée un nouveau planning de garde"""
        try:
            # Validation du planning
            await self._validate_oncall_schedule(schedule)
            
            # Stockage en base de données
            await self._store_oncall_schedule(schedule)
            
            # Mise à jour du cache
            cache_key = f"{schedule.tenant_id}:{schedule.schedule_id}"
            self.schedules_cache[cache_key] = schedule
            
            self.logger.info(f"Planning de garde {schedule.schedule_id} créé pour {schedule.tenant_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du planning: {e}")
            return False

    async def get_current_oncall(self, schedule_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Récupère la personne de garde actuelle"""
        try:
            schedule = await self._get_oncall_schedule(schedule_id, tenant_id)
            if not schedule:
                return None
            
            current_time = datetime.utcnow()
            
            # Recherche de la rotation active
            for rotation in schedule.rotations:
                if self._is_rotation_active(rotation, current_time):
                    current_user = self._get_current_user_in_rotation(rotation, current_time)
                    
                    return {
                        "schedule_id": schedule_id,
                        "user_id": current_user["user_id"],
                        "user_name": current_user.get("name", "Unknown"),
                        "rotation_name": rotation.get("name", "Default"),
                        "shift_start": current_user.get("shift_start"),
                        "shift_end": current_user.get("shift_end"),
                        "contact_methods": current_user.get("contact_methods", [])
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de la garde: {e}")
            return None

    async def get_escalation_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère les métriques d'escalade"""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Métriques générales
                metrics_query = """
                SELECT 
                    COUNT(*) as total_escalations,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed,
                    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled,
                    COUNT(*) FILTER (WHERE status = 'active') as active,
                    AVG(EXTRACT(EPOCH FROM (COALESCE(completed_at, cancelled_at, NOW()) - started_at))/60) as avg_duration_minutes
                FROM escalation_executions 
                WHERE tenant_id = $1 AND started_at >= NOW() - INTERVAL '7 days'
                """
                
                metrics = await conn.fetchrow(metrics_query, tenant_id)
                
                # Métriques par niveau
                level_query = """
                SELECT current_level, COUNT(*) as count
                FROM escalation_executions 
                WHERE tenant_id = $1 AND started_at >= NOW() - INTERVAL '7 days'
                GROUP BY current_level
                ORDER BY count DESC
                """
                
                level_metrics = await conn.fetch(level_query, tenant_id)
                
                # Calcul du taux de respect des SLA
                sla_query = """
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE 
                        EXTRACT(EPOCH FROM (COALESCE(completed_at, cancelled_at, NOW()) - started_at))/60 <= 60
                    ) as within_sla
                FROM escalation_executions 
                WHERE tenant_id = $1 AND started_at >= NOW() - INTERVAL '7 days'
                """
                
                sla_metrics = await conn.fetchrow(sla_query, tenant_id)
                sla_compliance = (sla_metrics['within_sla'] / sla_metrics['total'] * 100) if sla_metrics['total'] > 0 else 0
                
                return {
                    "summary": dict(metrics),
                    "by_level": [dict(row) for row in level_metrics],
                    "sla_compliance_percent": round(sla_compliance, 2),
                    "system_metrics": self.metrics.copy()
                }
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des métriques: {e}")
            return {"error": str(e)}

    async def _escalation_worker(self):
        """Worker principal d'escalade"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Traitement des escalades en attente
                for execution_id, execution in list(self.active_escalations.items()):
                    if execution.next_escalation_at and execution.next_escalation_at <= current_time:
                        await self._process_escalation(execution)
                
                # Attente avant la prochaine vérification
                await asyncio.sleep(30)  # Vérification toutes les 30 secondes
                
            except Exception as e:
                self.logger.error(f"Erreur dans l'escalation worker: {e}")
                await asyncio.sleep(60)

    async def _process_escalation(self, execution: EscalationExecution):
        """Traite une escalade"""
        try:
            policy = await self._get_escalation_policy(execution.policy_id, execution.tenant_id)
            if not policy:
                self.logger.error(f"Politique {execution.policy_id} non trouvée")
                return
            
            current_step = policy.steps[execution.current_step]
            
            # Vérification des conditions métier (heures ouvrables, etc.)
            if not await self._check_escalation_conditions(current_step, execution.tenant_id):
                # Report de l'escalade
                execution.next_escalation_at = datetime.utcnow() + timedelta(minutes=15)
                await self._update_escalation_execution(execution)
                return
            
            # Exécution de l'étape d'escalade
            await self._execute_escalation_step(execution, current_step)
            
            # Passage à l'étape suivante ou fin
            if execution.current_step + 1 < len(policy.steps):
                execution.current_step += 1
                next_step = policy.steps[execution.current_step]
                execution.current_level = next_step.level
                execution.next_escalation_at = datetime.utcnow() + timedelta(minutes=next_step.delay_minutes)
            else:
                # Escalade terminée
                execution.status = "completed"
                execution.completed_at = datetime.utcnow()
                execution.next_escalation_at = None
                
                # Suppression du cache actif
                if execution.execution_id in self.active_escalations:
                    del self.active_escalations[execution.execution_id]
                
                # Métriques
                self.metrics["escalations_completed"] += 1
            
            # Mise à jour en base
            await self._update_escalation_execution(execution)
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de l'escalade {execution.execution_id}: {e}")

    async def _execute_escalation_step(self, execution: EscalationExecution, step: EscalationStep):
        """Exécute une étape d'escalade"""
        try:
            step_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": step.level.value,
                "actions": [],
                "success": True
            }
            
            # Exécution des actions
            for action in step.actions:
                try:
                    action_result = await self._execute_escalation_action(
                        action, execution, step
                    )
                    step_log["actions"].append({
                        "action": action.value,
                        "result": action_result,
                        "success": True
                    })
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'action {action}: {e}")
                    step_log["actions"].append({
                        "action": action.value,
                        "error": str(e),
                        "success": False
                    })
                    step_log["success"] = False
            
            # Ajout au log d'exécution
            execution.execution_log.append(step_log)
            
            # Log d'audit
            await self._log_escalation_event(
                execution.execution_id,
                "step_executed",
                {
                    "level": step.level.value,
                    "actions_count": len(step.actions),
                    "success": step_log["success"]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution de l'étape: {e}")

    async def _execute_escalation_action(
        self,
        action: EscalationAction,
        execution: EscalationExecution,
        step: EscalationStep
    ) -> Dict[str, Any]:
        """Exécute une action d'escalade spécifique"""
        if action == EscalationAction.NOTIFY_SLACK:
            return await self._notify_slack_escalation(execution, step)
        elif action == EscalationAction.SEND_EMAIL:
            return await self._send_email_escalation(execution, step)
        elif action == EscalationAction.PAGE_ONCALL:
            return await self._page_oncall(execution, step)
        elif action == EscalationAction.CREATE_TICKET:
            return await self._create_ticket(execution, step)
        elif action == EscalationAction.CALL_PHONE:
            return await self._call_phone(execution, step)
        elif action == EscalationAction.TRIGGER_RUNBOOK:
            return await self._trigger_runbook(execution, step)
        else:
            return {"status": "unknown_action", "action": action.value}

    async def _notify_slack_escalation(
        self,
        execution: EscalationExecution,
        step: EscalationStep
    ) -> Dict[str, Any]:
        """Envoie une notification Slack d'escalade"""
        # Import du formateur depuis le module parent
        from .slack_alert_formatter import SlackAlertFormatter
        
        formatter = SlackAlertFormatter()
        
        # Simulation - à intégrer avec SlackWebhookHandler
        notification_data = {
            "alert_id": execution.alert_id,
            "escalation_level": step.level.value,
            "target_users": step.target_users,
            "channels": step.target_channels
        }
        
        return {
            "status": "sent",
            "channels": step.target_channels,
            "users_notified": len(step.target_users)
        }

    async def _send_email_escalation(
        self,
        execution: EscalationExecution,
        step: EscalationStep
    ) -> Dict[str, Any]:
        """Envoie un email d'escalade"""
        # Simulation - à implémenter avec un service d'email
        return {
            "status": "sent",
            "recipients": step.target_users,
            "subject": f"Escalation Level {step.level.value} - Alert {execution.alert_id}"
        }

    async def _page_oncall(
        self,
        execution: EscalationExecution,
        step: EscalationStep
    ) -> Dict[str, Any]:
        """Page la personne de garde"""
        # Intégration avec PagerDuty ou autre service
        return {
            "status": "paged",
            "service": "pagerduty",
            "incident_id": f"escalation-{execution.execution_id}"
        }

    async def _create_ticket(
        self,
        execution: EscalationExecution,
        step: EscalationStep
    ) -> Dict[str, Any]:
        """Crée un ticket dans le système de ticketing"""
        # Intégration avec ServiceNow, Jira, etc.
        return {
            "status": "created",
            "ticket_id": f"INC-{execution.execution_id[:8]}",
            "system": "servicenow"
        }

    async def _call_phone(
        self,
        execution: EscalationExecution,
        step: EscalationStep
    ) -> Dict[str, Any]:
        """Déclenche un appel téléphonique"""
        # Intégration avec un service de téléphonie
        return {
            "status": "called",
            "numbers_called": len(step.target_users),
            "service": "twilio"
        }

    async def _trigger_runbook(
        self,
        execution: EscalationExecution,
        step: EscalationStep
    ) -> Dict[str, Any]:
        """Déclenche un runbook automatique"""
        # Intégration avec système d'automatisation
        return {
            "status": "triggered",
            "runbook_id": step.conditions.get("runbook_id"),
            "system": "ansible"
        }

    async def _check_escalation_conditions(self, step: EscalationStep, tenant_id: str) -> bool:
        """Vérifie les conditions pour exécuter une escalade"""
        # Vérification des heures ouvrables si configuré
        if step.conditions.get("business_hours_only", False):
            if not await self._is_business_hours(tenant_id):
                return False
        
        # Autres conditions personnalisées
        # À implémenter selon les besoins
        
        return True

    async def _is_business_hours(self, tenant_id: str) -> bool:
        """Vérifie si on est en heures ouvrables"""
        # Simulation - à implémenter selon la configuration tenant
        current_time = datetime.utcnow()
        current_hour = current_time.hour
        current_day = current_time.strftime("%A").lower()
        
        business_config = self.default_config["business_hours"]
        business_days = business_config["days"]
        start_hour = int(business_config["start"].split(":")[0])
        end_hour = int(business_config["end"].split(":")[0])
        
        return (current_day in business_days and 
                start_hour <= current_hour < end_hour)

    async def _get_escalation_policy(self, policy_id: str, tenant_id: str) -> Optional[EscalationPolicy]:
        """Récupère une politique d'escalade"""
        cache_key = f"{tenant_id}:{policy_id}"
        
        if cache_key not in self.policies_cache:
            # Chargement depuis la base de données
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM escalation_policies 
                    WHERE policy_id = $1 AND tenant_id = $2 AND active = true
                """, policy_id, tenant_id)
                
                if row:
                    policy_data = dict(row)
                    policy_data["steps"] = json.loads(policy_data["steps"])
                    policy_data["severity_filter"] = [AlertSeverity(s) for s in policy_data["severity_filter"]]
                    
                    policy = EscalationPolicy(**policy_data)
                    self.policies_cache[cache_key] = policy
                else:
                    return None
        
        return self.policies_cache.get(cache_key)

    async def _get_oncall_schedule(self, schedule_id: str, tenant_id: str) -> Optional[OnCallSchedule]:
        """Récupère un planning de garde"""
        cache_key = f"{tenant_id}:{schedule_id}"
        
        if cache_key not in self.schedules_cache:
            # Chargement depuis la base de données
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM oncall_schedules 
                    WHERE schedule_id = $1 AND tenant_id = $2 AND active = true
                """, schedule_id, tenant_id)
                
                if row:
                    schedule_data = dict(row)
                    schedule_data["rotations"] = json.loads(schedule_data["rotations"])
                    
                    schedule = OnCallSchedule(**schedule_data)
                    self.schedules_cache[cache_key] = schedule
                else:
                    return None
        
        return self.schedules_cache.get(cache_key)

    async def _get_active_escalation(self, alert_id: str) -> Optional[EscalationExecution]:
        """Récupère l'escalade active pour une alerte"""
        # Recherche dans le cache
        for execution in self.active_escalations.values():
            if execution.alert_id == alert_id and execution.status == "active":
                return execution
        
        # Recherche en base de données
        async with self.postgres_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM escalation_executions 
                WHERE alert_id = $1 AND status = 'active'
                ORDER BY started_at DESC LIMIT 1
            """, alert_id)
            
            if row:
                execution_data = dict(row)
                execution_data["execution_log"] = json.loads(execution_data.get("execution_log", "[]"))
                
                execution = EscalationExecution(**execution_data)
                self.active_escalations[execution.execution_id] = execution
                return execution
        
        return None

    def _is_rotation_active(self, rotation: Dict[str, Any], current_time: datetime) -> bool:
        """Vérifie si une rotation est active"""
        # Simulation - à implémenter selon la structure des rotations
        return True

    def _get_current_user_in_rotation(self, rotation: Dict[str, Any], current_time: datetime) -> Dict[str, Any]:
        """Récupère l'utilisateur actuel dans une rotation"""
        # Simulation - à implémenter selon la structure des rotations
        users = rotation.get("users", [])
        if users:
            return users[0]  # Premier utilisateur par défaut
        
        return {"user_id": "unknown", "name": "Unknown User"}

    async def _create_database_schema(self):
        """Crée le schéma de base de données"""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS escalation_policies (
            policy_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            tenant_id VARCHAR(255) NOT NULL,
            steps JSONB NOT NULL,
            max_escalation_time INTEGER DEFAULT 240,
            business_hours_only BOOLEAN DEFAULT false,
            severity_filter JSONB DEFAULT '[]',
            service_filter JSONB DEFAULT '[]',
            active BOOLEAN DEFAULT true,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS escalation_executions (
            execution_id VARCHAR(255) PRIMARY KEY,
            alert_id VARCHAR(255) NOT NULL,
            policy_id VARCHAR(255) NOT NULL,
            tenant_id VARCHAR(255) NOT NULL,
            current_level VARCHAR(50) NOT NULL,
            current_step INTEGER NOT NULL,
            started_at TIMESTAMP WITH TIME ZONE NOT NULL,
            next_escalation_at TIMESTAMP WITH TIME ZONE,
            completed_at TIMESTAMP WITH TIME ZONE,
            cancelled_at TIMESTAMP WITH TIME ZONE,
            status VARCHAR(50) DEFAULT 'active',
            execution_log JSONB DEFAULT '[]'
        );
        
        CREATE TABLE IF NOT EXISTS oncall_schedules (
            schedule_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            tenant_id VARCHAR(255) NOT NULL,
            timezone VARCHAR(100) DEFAULT 'Europe/Paris',
            rotations JSONB DEFAULT '[]',
            escalation_policy VARCHAR(255),
            active BOOLEAN DEFAULT true,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS escalation_audit (
            id SERIAL PRIMARY KEY,
            execution_id VARCHAR(255) NOT NULL,
            event_type VARCHAR(100) NOT NULL,
            event_data JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_escalation_policies_tenant ON escalation_policies(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_escalation_executions_alert ON escalation_executions(alert_id);
        CREATE INDEX IF NOT EXISTS idx_escalation_executions_status ON escalation_executions(status);
        CREATE INDEX IF NOT EXISTS idx_oncall_schedules_tenant ON oncall_schedules(tenant_id);
        """
        
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)

    async def _load_escalation_policies(self):
        """Charge toutes les politiques d'escalade"""
        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM escalation_policies WHERE active = true")
            
            for row in rows:
                try:
                    policy_data = dict(row)
                    policy_data["steps"] = [EscalationStep(**step) for step in json.loads(policy_data["steps"])]
                    policy_data["severity_filter"] = [AlertSeverity(s) for s in policy_data["severity_filter"]]
                    
                    policy = EscalationPolicy(**policy_data)
                    cache_key = f"{policy.tenant_id}:{policy.policy_id}"
                    self.policies_cache[cache_key] = policy
                    
                except Exception as e:
                    self.logger.warning(f"Erreur lors du chargement de la politique {row['policy_id']}: {e}")

    async def _load_oncall_schedules(self):
        """Charge tous les plannings de garde"""
        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM oncall_schedules WHERE active = true")
            
            for row in rows:
                try:
                    schedule_data = dict(row)
                    schedule_data["rotations"] = json.loads(schedule_data["rotations"])
                    
                    schedule = OnCallSchedule(**schedule_data)
                    cache_key = f"{schedule.tenant_id}:{schedule.schedule_id}"
                    self.schedules_cache[cache_key] = schedule
                    
                except Exception as e:
                    self.logger.warning(f"Erreur lors du chargement du planning {row['schedule_id']}: {e}")

    async def _restore_active_escalations(self):
        """Restaure les escalades actives au redémarrage"""
        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM escalation_executions WHERE status = 'active'")
            
            for row in rows:
                try:
                    execution_data = dict(row)
                    execution_data["execution_log"] = json.loads(execution_data.get("execution_log", "[]"))
                    
                    execution = EscalationExecution(**execution_data)
                    self.active_escalations[execution.execution_id] = execution
                    
                except Exception as e:
                    self.logger.warning(f"Erreur lors de la restauration de l'escalade {row['execution_id']}: {e}")

    async def _store_escalation_execution(self, execution: EscalationExecution):
        """Stocke une exécution d'escalade"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO escalation_executions (
                    execution_id, alert_id, policy_id, tenant_id, current_level,
                    current_step, started_at, next_escalation_at, status, execution_log
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                execution.execution_id, execution.alert_id, execution.policy_id,
                execution.tenant_id, execution.current_level.value, execution.current_step,
                execution.started_at, execution.next_escalation_at, execution.status,
                json.dumps(execution.execution_log)
            )

    async def _update_escalation_execution(self, execution: EscalationExecution):
        """Met à jour une exécution d'escalade"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                UPDATE escalation_executions SET
                    current_level = $1, current_step = $2, next_escalation_at = $3,
                    completed_at = $4, cancelled_at = $5, status = $6, execution_log = $7
                WHERE execution_id = $8
            """,
                execution.current_level.value, execution.current_step, execution.next_escalation_at,
                execution.completed_at, execution.cancelled_at, execution.status,
                json.dumps(execution.execution_log), execution.execution_id
            )

    async def _store_escalation_policy(self, policy: EscalationPolicy):
        """Stocke une politique d'escalade"""
        async with self.postgres_pool.acquire() as conn:
            steps_json = json.dumps([step.__dict__ for step in policy.steps], default=str)
            severity_filter_json = json.dumps([s.value for s in policy.severity_filter])
            
            await conn.execute("""
                INSERT INTO escalation_policies (
                    policy_id, name, description, tenant_id, steps,
                    max_escalation_time, business_hours_only, severity_filter,
                    service_filter, active, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (policy_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    steps = EXCLUDED.steps,
                    max_escalation_time = EXCLUDED.max_escalation_time,
                    business_hours_only = EXCLUDED.business_hours_only,
                    severity_filter = EXCLUDED.severity_filter,
                    service_filter = EXCLUDED.service_filter,
                    updated_at = NOW()
            """,
                policy.policy_id, policy.name, policy.description, policy.tenant_id,
                steps_json, policy.max_escalation_time, policy.business_hours_only,
                severity_filter_json, json.dumps(policy.service_filter),
                policy.active, policy.created_at
            )

    async def _store_oncall_schedule(self, schedule: OnCallSchedule):
        """Stocke un planning de garde"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO oncall_schedules (
                    schedule_id, name, tenant_id, timezone, rotations,
                    escalation_policy, active
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (schedule_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    timezone = EXCLUDED.timezone,
                    rotations = EXCLUDED.rotations,
                    escalation_policy = EXCLUDED.escalation_policy,
                    active = EXCLUDED.active
            """,
                schedule.schedule_id, schedule.name, schedule.tenant_id,
                schedule.timezone, json.dumps(schedule.rotations),
                schedule.escalation_policy, schedule.active
            )

    async def _log_escalation_event(
        self,
        execution_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """Log un événement d'escalade"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO escalation_audit (execution_id, event_type, event_data)
                VALUES ($1, $2, $3)
            """, execution_id, event_type, json.dumps(event_data))

    async def _validate_escalation_policy(self, policy: EscalationPolicy):
        """Valide une politique d'escalade"""
        if not policy.policy_id or not policy.name:
            raise ValueError("policy_id et name sont requis")
        
        if not policy.steps:
            raise ValueError("Au moins une étape d'escalade est requise")
        
        for i, step in enumerate(policy.steps):
            if step.delay_minutes < 0:
                raise ValueError(f"Délai invalide à l'étape {i}")
            
            if not step.target_users and not step.target_channels:
                raise ValueError(f"Cibles requises à l'étape {i}")

    async def _validate_oncall_schedule(self, schedule: OnCallSchedule):
        """Valide un planning de garde"""
        if not schedule.schedule_id or not schedule.name:
            raise ValueError("schedule_id et name sont requis")
        
        if not schedule.rotations:
            raise ValueError("Au moins une rotation est requise")

    async def _sla_monitor_worker(self):
        """Worker de monitoring des SLA"""
        while True:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                # Vérification des SLA pour les escalades actives
                current_time = datetime.utcnow()
                
                for execution in self.active_escalations.values():
                    elapsed_time = current_time - execution.started_at
                    
                    # SLA par défaut: résolution en 4 heures
                    sla_threshold = timedelta(hours=4)
                    
                    if elapsed_time > sla_threshold:
                        # Breach SLA détecté
                        await self._handle_sla_breach(execution)
                        self.metrics["sla_breaches"] += 1
                
            except Exception as e:
                self.logger.error(f"Erreur dans le SLA monitor: {e}")

    async def _handle_sla_breach(self, execution: EscalationExecution):
        """Gère une violation de SLA"""
        await self._log_escalation_event(
            execution.execution_id,
            "sla_breach",
            {
                "elapsed_time": str(datetime.utcnow() - execution.started_at),
                "alert_id": execution.alert_id
            }
        )
        
        # Notification immédiate au management
        # À implémenter selon les besoins
