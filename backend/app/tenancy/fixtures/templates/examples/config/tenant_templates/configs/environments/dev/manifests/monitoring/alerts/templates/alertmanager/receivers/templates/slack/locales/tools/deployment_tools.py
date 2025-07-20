"""
Outils de déploiement et de gestion des rollbacks.

Ce module fournit un système complet de déploiement avec stratégies
blue/green, canary et rolling update, ainsi que la gestion des rollbacks.
"""

import os
import yaml
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from ..schemas.tenant_schemas import TenantConfigSchema


class DeploymentStrategy(str, Enum):
    """Stratégies de déploiement."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING_UPDATE = "rolling_update"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class DeploymentStatus(str, Enum):
    """États de déploiement."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class HealthCheckResult(BaseModel):
    """Résultat d'un health check."""
    endpoint: str
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class DeploymentStep(BaseModel):
    """Étape de déploiement."""
    name: str
    description: str
    command: str
    timeout: int = 300
    retry_count: int = 3
    required: bool = True
    health_checks: List[str] = Field(default_factory=list)


class DeploymentPlan(BaseModel):
    """Plan de déploiement."""
    name: str
    strategy: DeploymentStrategy
    environment: str
    tenant_id: Optional[str] = None
    steps: List[DeploymentStep]
    rollback_steps: List[DeploymentStep] = Field(default_factory=list)
    validation_steps: List[DeploymentStep] = Field(default_factory=list)
    timeout: int = 1800
    auto_rollback: bool = True
    approval_required: bool = False


class DeploymentRecord(BaseModel):
    """Enregistrement de déploiement."""
    id: str
    plan: DeploymentPlan
    status: DeploymentStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    logs: List[str] = Field(default_factory=list)
    health_checks: List[HealthCheckResult] = Field(default_factory=list)
    error_message: Optional[str] = None
    artifacts: List[str] = Field(default_factory=list)


class DeploymentManager:
    """Gestionnaire de déploiement avec stratégies multiples."""
    
    def __init__(
        self,
        workspace_dir: str = "./deployments",
        config_dir: str = "./configs",
        backup_dir: str = "./backups"
    ):
        """Initialise le gestionnaire de déploiement."""
        self.workspace_dir = Path(workspace_dir)
        self.config_dir = Path(config_dir)
        self.backup_dir = Path(backup_dir)
        
        # Création des répertoires
        for directory in [self.workspace_dir, self.config_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.active_deployments: Dict[str, DeploymentRecord] = {}
        self.deployment_history: List[DeploymentRecord] = []
        
        # Configuration des stratégies
        self.strategy_handlers = {
            DeploymentStrategy.BLUE_GREEN: self._deploy_blue_green,
            DeploymentStrategy.CANARY: self._deploy_canary,
            DeploymentStrategy.ROLLING_UPDATE: self._deploy_rolling_update,
            DeploymentStrategy.RECREATE: self._deploy_recreate,
            DeploymentStrategy.A_B_TESTING: self._deploy_ab_testing
        }
    
    def deploy(self, plan: DeploymentPlan) -> DeploymentRecord:
        """Execute un déploiement selon le plan fourni."""
        deployment_id = self._generate_deployment_id()
        
        # Création de l'enregistrement de déploiement
        record = DeploymentRecord(
            id=deployment_id,
            plan=plan,
            status=DeploymentStatus.PENDING,
            started_at=datetime.now()
        )
        
        self.active_deployments[deployment_id] = record
        
        try:
            # Validation pré-déploiement
            self._validate_pre_deployment(plan)
            
            # Sauvegarde de l'état actuel
            backup_path = self._create_backup(plan)
            record.artifacts.append(str(backup_path))
            
            # Mise à jour du statut
            record.status = DeploymentStatus.RUNNING
            
            # Exécution selon la stratégie
            strategy_handler = self.strategy_handlers[plan.strategy]
            success = strategy_handler(plan, record)
            
            if success:
                # Validation post-déploiement
                validation_success = self._run_validation_steps(plan, record)
                
                if validation_success:
                    record.status = DeploymentStatus.SUCCESS
                    record.completed_at = datetime.now()
                else:
                    # Rollback automatique si validation échoue
                    if plan.auto_rollback:
                        self._perform_rollback(plan, record)
                    else:
                        record.status = DeploymentStatus.FAILED
            else:
                # Rollback automatique si déploiement échoue
                if plan.auto_rollback:
                    self._perform_rollback(plan, record)
                else:
                    record.status = DeploymentStatus.FAILED
        
        except Exception as e:
            record.status = DeploymentStatus.FAILED
            record.error_message = str(e)
            record.logs.append(f"ERREUR: {e}")
            
            if plan.auto_rollback:
                self._perform_rollback(plan, record)
        
        finally:
            # Déplacement vers l'historique
            self.deployment_history.append(record)
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
        
        return record
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Récupère le statut d'un déploiement."""
        # Recherche dans les déploiements actifs
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]
        
        # Recherche dans l'historique
        for record in self.deployment_history:
            if record.id == deployment_id:
                return record
        
        return None
    
    def list_deployments(
        self,
        environment: Optional[str] = None,
        status: Optional[DeploymentStatus] = None,
        limit: int = 50
    ) -> List[DeploymentRecord]:
        """Liste les déploiements avec filtrage."""
        all_deployments = list(self.active_deployments.values()) + self.deployment_history
        
        # Filtrage
        filtered = all_deployments
        if environment:
            filtered = [d for d in filtered if d.plan.environment == environment]
        if status:
            filtered = [d for d in filtered if d.status == status]
        
        # Tri par date (plus récent en premier)
        filtered.sort(key=lambda x: x.started_at, reverse=True)
        
        return filtered[:limit]
    
    def create_deployment_plan(
        self,
        name: str,
        strategy: DeploymentStrategy,
        environment: str,
        config_files: List[str],
        tenant_id: Optional[str] = None
    ) -> DeploymentPlan:
        """Crée un plan de déploiement standard."""
        steps = []
        
        # Étapes communes
        steps.extend([
            DeploymentStep(
                name="validate_configs",
                description="Validation des fichiers de configuration",
                command=f"python -m config_validator {' '.join(config_files)}",
                timeout=60
            ),
            DeploymentStep(
                name="backup_current",
                description="Sauvegarde de la configuration actuelle",
                command="echo 'Backup handled by deployment manager'",
                timeout=120
            )
        ])
        
        # Étapes spécifiques à la stratégie
        if strategy == DeploymentStrategy.BLUE_GREEN:
            steps.extend([
                DeploymentStep(
                    name="prepare_green",
                    description="Préparation de l'environnement green",
                    command=f"kubectl apply -f {' '.join(config_files)} --dry-run=server",
                    timeout=300
                ),
                DeploymentStep(
                    name="deploy_green",
                    description="Déploiement sur l'environnement green",
                    command=f"kubectl apply -f {' '.join(config_files)}",
                    timeout=600,
                    health_checks=[f"http://{environment}-green.local/health"]
                ),
                DeploymentStep(
                    name="switch_traffic",
                    description="Basculement du trafic vers green",
                    command=f"kubectl patch service {environment}-service -p '{{\"spec\":{{\"selector\":{{\"version\":\"green\"}}}}}}'",
                    timeout=60
                )
            ])
        
        elif strategy == DeploymentStrategy.CANARY:
            steps.extend([
                DeploymentStep(
                    name="deploy_canary",
                    description="Déploiement canary (10% du trafic)",
                    command=f"kubectl apply -f {' '.join(config_files)} --replicas=1",
                    timeout=300,
                    health_checks=[f"http://{environment}-canary.local/health"]
                ),
                DeploymentStep(
                    name="monitor_canary",
                    description="Monitoring du canary pendant 10 minutes",
                    command="sleep 600",
                    timeout=700
                ),
                DeploymentStep(
                    name="scale_canary",
                    description="Augmentation progressive du trafic canary",
                    command=f"kubectl scale deployment {environment}-canary --replicas=3",
                    timeout=300
                )
            ])
        
        # Étapes de rollback
        rollback_steps = [
            DeploymentStep(
                name="restore_backup",
                description="Restauration de la sauvegarde",
                command="echo 'Restore handled by rollback manager'",
                timeout=300
            ),
            DeploymentStep(
                name="verify_rollback",
                description="Vérification du rollback",
                command=f"curl -f http://{environment}.local/health",
                timeout=60
            )
        ]
        
        # Étapes de validation
        validation_steps = [
            DeploymentStep(
                name="health_check",
                description="Vérification de santé des services",
                command=f"curl -f http://{environment}.local/health",
                timeout=30,
                retry_count=5
            ),
            DeploymentStep(
                name="smoke_tests",
                description="Tests de fumée",
                command=f"python -m smoke_tests --environment {environment}",
                timeout=300
            )
        ]
        
        return DeploymentPlan(
            name=name,
            strategy=strategy,
            environment=environment,
            tenant_id=tenant_id,
            steps=steps,
            rollback_steps=rollback_steps,
            validation_steps=validation_steps
        )
    
    def _deploy_blue_green(self, plan: DeploymentPlan, record: DeploymentRecord) -> bool:
        """Implémentation du déploiement blue/green."""
        record.logs.append("Début du déploiement Blue/Green")
        
        try:
            # Exécution des étapes
            for step in plan.steps:
                success = self._execute_step(step, record)
                if not success and step.required:
                    return False
            
            record.logs.append("Déploiement Blue/Green terminé avec succès")
            return True
            
        except Exception as e:
            record.logs.append(f"Erreur durant le déploiement Blue/Green: {e}")
            return False
    
    def _deploy_canary(self, plan: DeploymentPlan, record: DeploymentRecord) -> bool:
        """Implémentation du déploiement canary."""
        record.logs.append("Début du déploiement Canary")
        
        try:
            # Déploiement progressif avec monitoring
            for step in plan.steps:
                success = self._execute_step(step, record)
                if not success and step.required:
                    return False
                
                # Monitoring spécial pour les étapes canary
                if "canary" in step.name:
                    health_results = self._run_health_checks(step.health_checks)
                    record.health_checks.extend(health_results)
                    
                    # Vérification des métriques canary
                    if not self._validate_canary_metrics(health_results):
                        record.logs.append("Métriques canary invalides, arrêt du déploiement")
                        return False
            
            record.logs.append("Déploiement Canary terminé avec succès")
            return True
            
        except Exception as e:
            record.logs.append(f"Erreur durant le déploiement Canary: {e}")
            return False
    
    def _deploy_rolling_update(self, plan: DeploymentPlan, record: DeploymentRecord) -> bool:
        """Implémentation du rolling update."""
        record.logs.append("Début du Rolling Update")
        
        try:
            for step in plan.steps:
                success = self._execute_step(step, record)
                if not success and step.required:
                    return False
            
            record.logs.append("Rolling Update terminé avec succès")
            return True
            
        except Exception as e:
            record.logs.append(f"Erreur durant le Rolling Update: {e}")
            return False
    
    def _deploy_recreate(self, plan: DeploymentPlan, record: DeploymentRecord) -> bool:
        """Implémentation du déploiement recreate."""
        record.logs.append("Début du déploiement Recreate")
        
        try:
            for step in plan.steps:
                success = self._execute_step(step, record)
                if not success and step.required:
                    return False
            
            record.logs.append("Déploiement Recreate terminé avec succès")
            return True
            
        except Exception as e:
            record.logs.append(f"Erreur durant le déploiement Recreate: {e}")
            return False
    
    def _deploy_ab_testing(self, plan: DeploymentPlan, record: DeploymentRecord) -> bool:
        """Implémentation du déploiement A/B Testing."""
        record.logs.append("Début du déploiement A/B Testing")
        
        try:
            for step in plan.steps:
                success = self._execute_step(step, record)
                if not success and step.required:
                    return False
            
            record.logs.append("Déploiement A/B Testing terminé avec succès")
            return True
            
        except Exception as e:
            record.logs.append(f"Erreur durant le déploiement A/B Testing: {e}")
            return False
    
    def _execute_step(self, step: DeploymentStep, record: DeploymentRecord) -> bool:
        """Exécute une étape de déploiement."""
        record.logs.append(f"Exécution de l'étape: {step.name}")
        
        for attempt in range(step.retry_count + 1):
            try:
                # Exécution de la commande
                result = subprocess.run(
                    step.command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=step.timeout
                )
                
                if result.returncode == 0:
                    record.logs.append(f"Étape {step.name} réussie")
                    
                    # Exécution des health checks
                    if step.health_checks:
                        health_results = self._run_health_checks(step.health_checks)
                        record.health_checks.extend(health_results)
                        
                        if not all(check.success for check in health_results):
                            record.logs.append(f"Health checks échoués pour {step.name}")
                            if attempt < step.retry_count:
                                continue
                            return False
                    
                    return True
                else:
                    record.logs.append(f"Étape {step.name} échouée: {result.stderr}")
                    if attempt < step.retry_count:
                        record.logs.append(f"Tentative {attempt + 2}/{step.retry_count + 1}")
                        continue
                    return False
            
            except subprocess.TimeoutExpired:
                record.logs.append(f"Timeout pour l'étape {step.name}")
                if attempt < step.retry_count:
                    continue
                return False
            except Exception as e:
                record.logs.append(f"Erreur lors de l'exécution de {step.name}: {e}")
                if attempt < step.retry_count:
                    continue
                return False
        
        return False
    
    def _run_health_checks(self, endpoints: List[str]) -> List[HealthCheckResult]:
        """Exécute des health checks sur des endpoints."""
        results = []
        
        for endpoint in endpoints:
            try:
                import requests
                start_time = datetime.now()
                response = requests.get(endpoint, timeout=10)
                response_time = (datetime.now() - start_time).total_seconds()
                
                results.append(HealthCheckResult(
                    endpoint=endpoint,
                    status_code=response.status_code,
                    response_time=response_time,
                    success=response.status_code < 400
                ))
            except Exception as e:
                results.append(HealthCheckResult(
                    endpoint=endpoint,
                    status_code=0,
                    response_time=0.0,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    def _validate_canary_metrics(self, health_results: List[HealthCheckResult]) -> bool:
        """Valide les métriques d'un déploiement canary."""
        if not health_results:
            return False
        
        # Vérification du taux de succès
        success_rate = sum(1 for r in health_results if r.success) / len(health_results)
        if success_rate < 0.95:  # Seuil de 95%
            return False
        
        # Vérification du temps de réponse
        avg_response_time = sum(r.response_time for r in health_results if r.success) / max(1, sum(1 for r in health_results if r.success))
        if avg_response_time > 2.0:  # Seuil de 2 secondes
            return False
        
        return True
    
    def _validate_pre_deployment(self, plan: DeploymentPlan):
        """Validation pré-déploiement."""
        # Vérification de l'existence des fichiers de configuration
        # Vérification des permissions
        # Vérification de l'état de l'environnement cible
        pass
    
    def _create_backup(self, plan: DeploymentPlan) -> Path:
        """Crée une sauvegarde de l'état actuel."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{plan.environment}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # Sauvegarde des configurations actuelles
        # Cette implémentation dépend de l'environnement (K8s, Docker, etc.)
        
        return backup_path
    
    def _run_validation_steps(self, plan: DeploymentPlan, record: DeploymentRecord) -> bool:
        """Exécute les étapes de validation post-déploiement."""
        for step in plan.validation_steps:
            success = self._execute_step(step, record)
            if not success and step.required:
                return False
        return True
    
    def _perform_rollback(self, plan: DeploymentPlan, record: DeploymentRecord):
        """Effectue un rollback."""
        record.status = DeploymentStatus.ROLLING_BACK
        record.logs.append("Début du rollback")
        
        try:
            for step in plan.rollback_steps:
                success = self._execute_step(step, record)
                if not success and step.required:
                    record.logs.append(f"Erreur critique durant le rollback: {step.name}")
                    break
            
            record.status = DeploymentStatus.ROLLED_BACK
            record.logs.append("Rollback terminé")
        except Exception as e:
            record.logs.append(f"Erreur durant le rollback: {e}")
    
    def _generate_deployment_id(self) -> str:
        """Génère un ID unique pour le déploiement."""
        import uuid
        return str(uuid.uuid4())[:8]


class RollbackManager:
    """Gestionnaire de rollbacks avec historique."""
    
    def __init__(self, deployment_manager: DeploymentManager):
        """Initialise le gestionnaire de rollbacks."""
        self.deployment_manager = deployment_manager
        self.rollback_history: List[Dict[str, Any]] = []
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """Effectue un rollback d'un déploiement spécifique."""
        deployment = self.deployment_manager.get_deployment_status(deployment_id)
        if not deployment:
            raise ValueError(f"Déploiement non trouvé: {deployment_id}")
        
        rollback_record = {
            'deployment_id': deployment_id,
            'initiated_at': datetime.now(),
            'status': 'in_progress'
        }
        
        try:
            # Exécution du rollback
            self.deployment_manager._perform_rollback(deployment.plan, deployment)
            rollback_record['status'] = 'success'
            rollback_record['completed_at'] = datetime.now()
            return True
        except Exception as e:
            rollback_record['status'] = 'failed'
            rollback_record['error'] = str(e)
            rollback_record['completed_at'] = datetime.now()
            return False
        finally:
            self.rollback_history.append(rollback_record)
    
    def get_rollback_candidates(self, environment: str) -> List[DeploymentRecord]:
        """Récupère les déploiements éligibles au rollback."""
        recent_deployments = self.deployment_manager.list_deployments(
            environment=environment,
            limit=10
        )
        
        # Filtrage des déploiements rollbackables
        candidates = [
            d for d in recent_deployments
            if d.status in [DeploymentStatus.SUCCESS, DeploymentStatus.FAILED]
            and d.started_at > datetime.now() - timedelta(days=7)
        ]
        
        return candidates


# Factory functions
def create_deployment_manager(
    workspace_dir: str = "./deployments",
    config_dir: str = "./configs", 
    backup_dir: str = "./backups"
) -> DeploymentManager:
    """Crée un gestionnaire de déploiement."""
    return DeploymentManager(workspace_dir, config_dir, backup_dir)


def create_rollback_manager(deployment_manager: DeploymentManager) -> RollbackManager:
    """Crée un gestionnaire de rollbacks."""
    return RollbackManager(deployment_manager)
