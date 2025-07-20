"""
Advanced DevOps Pipeline Manager

This module provides comprehensive DevOps pipeline management with
CI/CD automation, GitOps integration, and intelligent orchestration.

Version: 3.0.0
Developed by Spotify AI Agent Team
Lead Developer & AI Architect: Fahed Mlaiel
"""

import asyncio
import json
import yaml
import subprocess
import tempfile
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
import aiofiles
import aiohttp
from kubernetes import client, config as k8s_config
from git import Repo, Git
import docker
from jinja2 import Template, Environment, FileSystemLoader

from .enums import (
    DeploymentStrategy, DeploymentPhase, DeploymentStatus,
    EnvironmentType, CloudProvider, RuntimePlatform
)
from .constants import (
    DEFAULT_TIMEOUTS, ERROR_CODES, PIPELINE_STAGES,
    DEPLOYMENT_STRATEGIES, GIT_OPERATIONS
)
from .utils import (
    execute_command_async, validate_json_schema,
    compute_file_hash, create_secure_temp_dir
)

logger = structlog.get_logger(__name__)

# ============================================================================
# Pipeline Models and Enums
# ============================================================================

class PipelineStage(Enum):
    """Étapes du pipeline"""
    SOURCE = "source"
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    QUALITY_GATE = "quality_gate"
    PACKAGE = "package"
    DEPLOY_STAGING = "deploy_staging"
    INTEGRATION_TEST = "integration_test"
    DEPLOY_PRODUCTION = "deploy_production"
    MONITORING = "monitoring"
    CLEANUP = "cleanup"

class PipelineStatus(Enum):
    """Statuts du pipeline"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

@dataclass
class PipelineStep:
    """Étape individuelle du pipeline"""
    name: str
    command: str
    working_dir: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    timeout: int = 300
    retry_count: int = 0
    condition: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    
@dataclass
class PipelineExecution:
    """Exécution de pipeline"""
    id: str
    pipeline_name: str
    branch: str
    commit_sha: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GitOpsConfig:
    """Configuration GitOps"""
    repo_url: str
    branch: str = "main"
    path: str = "manifests"
    sync_interval: int = 300
    auto_sync: bool = True
    prune: bool = False
    self_heal: bool = True

@dataclass
class DeploymentTarget:
    """Cible de déploiement"""
    name: str
    environment: EnvironmentType
    platform: RuntimePlatform
    cluster_config: Dict[str, Any]
    namespace: Optional[str] = None
    resource_limits: Dict[str, str] = field(default_factory=dict)
    health_checks: List[Dict[str, Any]] = field(default_factory=list)

# ============================================================================
# Advanced DevOps Pipeline Manager
# ============================================================================

class AdvancedDevOpsPipelineManager:
    """Gestionnaire de pipeline DevOps avancé avec orchestration intelligente"""
    
    def __init__(self,
                 workspace_dir: str,
                 registry_url: Optional[str] = None,
                 git_credentials: Optional[Dict[str, str]] = None,
                 k8s_config_path: Optional[str] = None):
        
        self.workspace_dir = Path(workspace_dir)
        self.registry_url = registry_url
        self.git_credentials = git_credentials or {}
        self.k8s_config_path = k8s_config_path
        
        # Pipeline state
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        self.executions: Dict[str, PipelineExecution] = {}
        self.deployment_targets: Dict[str, DeploymentTarget] = {}
        self.gitops_configs: Dict[str, GitOpsConfig] = {}
        
        # Services
        self.docker_client = None
        self.k8s_client = None
        
        # Templates
        self.template_env = Environment(
            loader=FileSystemLoader(self.workspace_dir / "templates")
        )
        
        # Hooks
        self.pre_hooks: Dict[str, List[Callable]] = {}
        self.post_hooks: Dict[str, List[Callable]] = {}
        
        # Initialize services
        asyncio.create_task(self._initialize_services())
        
        logger.info("Advanced DevOps Pipeline Manager initialized")
    
    async def _initialize_services(self):
        """Initialise les services externes"""
        
        try:
            # Docker client
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
            
            # Kubernetes client
            if self.k8s_config_path:
                k8s_config.load_kube_config(config_file=self.k8s_config_path)
            else:
                k8s_config.load_incluster_config()
            
            self.k8s_client = client.ApiClient()
            logger.info("Kubernetes client initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize some services: {e}")
    
    async def create_pipeline(self,
                            name: str,
                            config: Dict[str, Any],
                            template: Optional[str] = None) -> bool:
        """Crée un nouveau pipeline"""
        
        try:
            # Validation de la configuration
            if not self._validate_pipeline_config(config):
                return False
            
            # Application du template si fourni
            if template:
                config = await self._apply_pipeline_template(config, template)
            
            # Création de la structure du pipeline
            pipeline = {
                "name": name,
                "config": config,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "version": "1.0.0",
                "stages": config.get("stages", []),
                "triggers": config.get("triggers", {}),
                "environment": config.get("environment", {}),
                "notifications": config.get("notifications", {})
            }
            
            self.pipelines[name] = pipeline
            
            # Création des hooks par défaut
            await self._setup_default_hooks(name)
            
            logger.info(f"Pipeline created: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create pipeline {name}: {e}")
            return False
    
    async def execute_pipeline(self,
                             pipeline_name: str,
                             branch: str = "main",
                             commit_sha: Optional[str] = None,
                             parameters: Optional[Dict[str, Any]] = None) -> str:
        """Exécute un pipeline"""
        
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_name}")
        
        # Génération de l'ID d'exécution
        execution_id = f"{pipeline_name}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Création de l'objet d'exécution
        execution = PipelineExecution(
            id=execution_id,
            pipeline_name=pipeline_name,
            branch=branch,
            commit_sha=commit_sha or "HEAD",
            status=PipelineStatus.PENDING,
            start_time=datetime.utcnow()
        )
        
        self.executions[execution_id] = execution
        
        # Exécution asynchrone
        asyncio.create_task(
            self._execute_pipeline_async(execution, parameters or {})
        )
        
        logger.info(f"Pipeline execution started: {execution_id}")
        return execution_id
    
    async def _execute_pipeline_async(self,
                                    execution: PipelineExecution,
                                    parameters: Dict[str, Any]):
        """Exécution asynchrone du pipeline"""
        
        try:
            execution.status = PipelineStatus.RUNNING
            pipeline = self.pipelines[execution.pipeline_name]
            
            # Préparation de l'environnement
            workspace = await self._prepare_workspace(execution)
            
            # Exécution des hooks pré-pipeline
            await self._execute_hooks("pre", execution.pipeline_name, execution)
            
            # Exécution des étapes
            for stage_config in pipeline["stages"]:
                stage_name = stage_config["name"]
                
                try:
                    execution.logs.append(f"Starting stage: {stage_name}")
                    
                    # Vérification des conditions
                    if not await self._check_stage_condition(stage_config, execution):
                        execution.logs.append(f"Stage skipped: {stage_name}")
                        continue
                    
                    # Exécution de l'étape
                    stage_result = await self._execute_stage(
                        stage_config, execution, workspace, parameters
                    )
                    
                    execution.steps.append({
                        "name": stage_name,
                        "status": "success" if stage_result else "failed",
                        "timestamp": datetime.utcnow(),
                        "duration": 0  # À calculer
                    })
                    
                    if not stage_result:
                        execution.status = PipelineStatus.FAILED
                        execution.logs.append(f"Stage failed: {stage_name}")
                        break
                    
                    execution.logs.append(f"Stage completed: {stage_name}")
                    
                except Exception as e:
                    execution.status = PipelineStatus.FAILED
                    execution.logs.append(f"Stage error in {stage_name}: {str(e)}")
                    logger.error(f"Stage error in {stage_name}: {e}")
                    break
            
            # Finalisation
            if execution.status == PipelineStatus.RUNNING:
                execution.status = PipelineStatus.SUCCESS
            
            execution.end_time = datetime.utcnow()
            
            # Exécution des hooks post-pipeline
            await self._execute_hooks("post", execution.pipeline_name, execution)
            
            # Nettoyage
            await self._cleanup_workspace(workspace)
            
            logger.info(f"Pipeline execution completed: {execution.id}")
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.end_time = datetime.utcnow()
            execution.logs.append(f"Pipeline error: {str(e)}")
            logger.error(f"Pipeline execution error: {e}")
    
    async def deploy_application(self,
                               application_name: str,
                               version: str,
                               target_name: str,
                               strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
                               config_overrides: Optional[Dict[str, Any]] = None) -> str:
        """Déploie une application"""
        
        if target_name not in self.deployment_targets:
            raise ValueError(f"Deployment target not found: {target_name}")
        
        target = self.deployment_targets[target_name]
        deployment_id = f"deploy-{application_name}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        try:
            logger.info(f"Starting deployment: {deployment_id}")
            
            # Préparation des manifestes
            manifests = await self._prepare_deployment_manifests(
                application_name, version, target, config_overrides or {}
            )
            
            # Exécution de la stratégie de déploiement
            if strategy == DeploymentStrategy.BLUE_GREEN:
                success = await self._deploy_blue_green(manifests, target)
            elif strategy == DeploymentStrategy.CANARY:
                success = await self._deploy_canary(manifests, target)
            elif strategy == DeploymentStrategy.ROLLING_UPDATE:
                success = await self._deploy_rolling_update(manifests, target)
            else:
                success = await self._deploy_recreate(manifests, target)
            
            if success:
                logger.info(f"Deployment successful: {deployment_id}")
            else:
                logger.error(f"Deployment failed: {deployment_id}")
            
            return deployment_id
            
        except Exception as e:
            logger.error(f"Deployment error: {e}")
            raise
    
    async def setup_gitops(self,
                         name: str,
                         config: GitOpsConfig) -> bool:
        """Configure GitOps pour un environnement"""
        
        try:
            self.gitops_configs[name] = config
            
            # Clonage du repository GitOps
            repo_dir = self.workspace_dir / "gitops" / name
            repo_dir.mkdir(parents=True, exist_ok=True)
            
            if not (repo_dir / ".git").exists():
                Repo.clone_from(config.repo_url, repo_dir)
                logger.info(f"GitOps repository cloned: {name}")
            
            # Configuration de la synchronisation automatique
            if config.auto_sync:
                asyncio.create_task(self._gitops_sync_loop(name))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup GitOps {name}: {e}")
            return False
    
    async def _gitops_sync_loop(self, gitops_name: str):
        """Boucle de synchronisation GitOps"""
        
        while gitops_name in self.gitops_configs:
            try:
                config = self.gitops_configs[gitops_name]
                repo_dir = self.workspace_dir / "gitops" / gitops_name
                
                # Pull des changements
                repo = Repo(repo_dir)
                origin = repo.remotes.origin
                origin.pull()
                
                # Application des changements
                manifests_dir = repo_dir / config.path
                if manifests_dir.exists():
                    await self._apply_gitops_manifests(manifests_dir, gitops_name)
                
                await asyncio.sleep(config.sync_interval)
                
            except Exception as e:
                logger.error(f"GitOps sync error for {gitops_name}: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def create_deployment_target(self,
                                     name: str,
                                     target_config: Dict[str, Any]) -> bool:
        """Crée une cible de déploiement"""
        
        try:
            target = DeploymentTarget(
                name=name,
                environment=EnvironmentType(target_config["environment"]),
                platform=RuntimePlatform(target_config["platform"]),
                cluster_config=target_config.get("cluster", {}),
                namespace=target_config.get("namespace"),
                resource_limits=target_config.get("resource_limits", {}),
                health_checks=target_config.get("health_checks", [])
            )
            
            self.deployment_targets[name] = target
            
            # Validation de la connectivité
            if await self._validate_deployment_target(target):
                logger.info(f"Deployment target created: {name}")
                return True
            else:
                del self.deployment_targets[name]
                return False
                
        except Exception as e:
            logger.error(f"Failed to create deployment target {name}: {e}")
            return False
    
    async def monitor_deployment(self,
                               deployment_id: str,
                               target_name: str) -> Dict[str, Any]:
        """Surveille un déploiement"""
        
        if target_name not in self.deployment_targets:
            raise ValueError(f"Deployment target not found: {target_name}")
        
        target = self.deployment_targets[target_name]
        
        # Récupération du statut selon la plateforme
        if target.platform == RuntimePlatform.KUBERNETES:
            return await self._monitor_k8s_deployment(deployment_id, target)
        elif target.platform == RuntimePlatform.DOCKER:
            return await self._monitor_docker_deployment(deployment_id, target)
        else:
            return {"status": "unknown", "message": "Platform not supported"}
    
    async def rollback_deployment(self,
                                deployment_id: str,
                                target_name: str,
                                revision: Optional[int] = None) -> bool:
        """Effectue un rollback de déploiement"""
        
        try:
            target = self.deployment_targets[target_name]
            
            if target.platform == RuntimePlatform.KUBERNETES:
                return await self._rollback_k8s_deployment(deployment_id, target, revision)
            elif target.platform == RuntimePlatform.DOCKER:
                return await self._rollback_docker_deployment(deployment_id, target, revision)
            
            return False
            
        except Exception as e:
            logger.error(f"Rollback failed for {deployment_id}: {e}")
            return False
    
    async def _prepare_workspace(self, execution: PipelineExecution) -> Path:
        """Prépare l'espace de travail pour l'exécution"""
        
        workspace = self.workspace_dir / "executions" / execution.id
        workspace.mkdir(parents=True, exist_ok=True)
        
        # Clonage du repository source si nécessaire
        pipeline = self.pipelines[execution.pipeline_name]
        if "source" in pipeline["config"]:
            source_config = pipeline["config"]["source"]
            if source_config.get("type") == "git":
                repo_url = source_config["url"]
                repo_dir = workspace / "source"
                
                Repo.clone_from(
                    repo_url,
                    repo_dir,
                    branch=execution.branch
                )
                
                execution.logs.append(f"Source cloned from {repo_url}")
        
        return workspace
    
    async def _execute_stage(self,
                           stage_config: Dict[str, Any],
                           execution: PipelineExecution,
                           workspace: Path,
                           parameters: Dict[str, Any]) -> bool:
        """Exécute une étape du pipeline"""
        
        stage_type = stage_config.get("type", "script")
        
        if stage_type == "script":
            return await self._execute_script_stage(stage_config, execution, workspace)
        elif stage_type == "docker":
            return await self._execute_docker_stage(stage_config, execution, workspace)
        elif stage_type == "kubernetes":
            return await self._execute_k8s_stage(stage_config, execution, workspace)
        elif stage_type == "test":
            return await self._execute_test_stage(stage_config, execution, workspace)
        elif stage_type == "security_scan":
            return await self._execute_security_scan_stage(stage_config, execution, workspace)
        else:
            execution.logs.append(f"Unknown stage type: {stage_type}")
            return False
    
    async def _execute_script_stage(self,
                                  stage_config: Dict[str, Any],
                                  execution: PipelineExecution,
                                  workspace: Path) -> bool:
        """Exécute une étape de script"""
        
        try:
            commands = stage_config.get("commands", [])
            working_dir = workspace / stage_config.get("working_dir", "source")
            env = {**os.environ, **stage_config.get("environment", {})}
            
            for command in commands:
                result = await execute_command_async(
                    command,
                    cwd=str(working_dir),
                    env=env,
                    timeout=stage_config.get("timeout", 300)
                )
                
                execution.logs.append(f"Command: {command}")
                execution.logs.append(f"Output: {result.stdout}")
                
                if result.returncode != 0:
                    execution.logs.append(f"Error: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            execution.logs.append(f"Script stage error: {str(e)}")
            return False
    
    async def _execute_docker_stage(self,
                                  stage_config: Dict[str, Any],
                                  execution: PipelineExecution,
                                  workspace: Path) -> bool:
        """Exécute une étape Docker"""
        
        try:
            action = stage_config.get("action", "build")
            
            if action == "build":
                return await self._docker_build(stage_config, execution, workspace)
            elif action == "push":
                return await self._docker_push(stage_config, execution, workspace)
            elif action == "run":
                return await self._docker_run(stage_config, execution, workspace)
            
            return False
            
        except Exception as e:
            execution.logs.append(f"Docker stage error: {str(e)}")
            return False
    
    async def _docker_build(self,
                          stage_config: Dict[str, Any],
                          execution: PipelineExecution,
                          workspace: Path) -> bool:
        """Build d'une image Docker"""
        
        try:
            dockerfile_path = workspace / "source" / stage_config.get("dockerfile", "Dockerfile")
            image_name = stage_config["image_name"]
            tag = stage_config.get("tag", "latest")
            
            # Build de l'image
            image, logs = self.docker_client.images.build(
                path=str(workspace / "source"),
                dockerfile=str(dockerfile_path),
                tag=f"{image_name}:{tag}",
                rm=True,
                forcerm=True
            )
            
            # Logging des résultats
            for log in logs:
                if 'stream' in log:
                    execution.logs.append(log['stream'].strip())
            
            execution.artifacts.append(f"{image_name}:{tag}")
            execution.logs.append(f"Image built: {image_name}:{tag}")
            
            return True
            
        except Exception as e:
            execution.logs.append(f"Docker build error: {str(e)}")
            return False
    
    async def _apply_gitops_manifests(self,
                                    manifests_dir: Path,
                                    gitops_name: str):
        """Applique les manifestes GitOps"""
        
        try:
            for manifest_file in manifests_dir.rglob("*.yaml"):
                with open(manifest_file, 'r') as f:
                    manifest = yaml.safe_load(f)
                
                if manifest:
                    await self._apply_k8s_manifest(manifest)
                    logger.debug(f"Applied manifest: {manifest_file}")
            
            logger.info(f"GitOps manifests applied for: {gitops_name}")
            
        except Exception as e:
            logger.error(f"Failed to apply GitOps manifests: {e}")
    
    def get_pipeline_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Retourne le statut d'une exécution de pipeline"""
        
        execution = self.executions.get(execution_id)
        if not execution:
            return None
        
        return {
            "id": execution.id,
            "pipeline_name": execution.pipeline_name,
            "status": execution.status.value,
            "branch": execution.branch,
            "commit_sha": execution.commit_sha,
            "start_time": execution.start_time,
            "end_time": execution.end_time,
            "duration": (
                (execution.end_time - execution.start_time).total_seconds()
                if execution.end_time else None
            ),
            "steps": execution.steps,
            "artifacts": execution.artifacts,
            "logs_count": len(execution.logs)
        }
    
    def get_deployment_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de déploiement"""
        
        return {
            "pipelines_count": len(self.pipelines),
            "executions_count": len(self.executions),
            "deployment_targets": len(self.deployment_targets),
            "gitops_configs": len(self.gitops_configs),
            "active_executions": len([
                e for e in self.executions.values()
                if e.status == PipelineStatus.RUNNING
            ])
        }
    
    def register_hook(self, 
                     hook_type: str, 
                     pipeline_name: str, 
                     handler: Callable):
        """Enregistre un hook de pipeline"""
        
        if hook_type not in ["pre", "post"]:
            raise ValueError("Hook type must be 'pre' or 'post'")
        
        hooks_dict = self.pre_hooks if hook_type == "pre" else self.post_hooks
        
        if pipeline_name not in hooks_dict:
            hooks_dict[pipeline_name] = []
        
        hooks_dict[pipeline_name].append(handler)
    
    async def _execute_hooks(self,
                           hook_type: str,
                           pipeline_name: str,
                           execution: PipelineExecution):
        """Exécute les hooks d'un pipeline"""
        
        hooks_dict = self.pre_hooks if hook_type == "pre" else self.post_hooks
        hooks = hooks_dict.get(pipeline_name, [])
        
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(execution)
                else:
                    hook(execution)
            except Exception as e:
                logger.error(f"Hook error in {pipeline_name}: {e}")
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        
        # Arrêt des boucles GitOps
        self.gitops_configs.clear()
        
        # Fermeture des clients
        if self.docker_client:
            self.docker_client.close()
        
        logger.info("DevOps Pipeline Manager cleaned up")

# ============================================================================
# Factory Functions
# ============================================================================

def create_pipeline_manager(
    workspace_dir: str,
    registry_url: Optional[str] = None,
    enable_gitops: bool = True
) -> AdvancedDevOpsPipelineManager:
    """Factory pour créer un gestionnaire de pipeline"""
    
    return AdvancedDevOpsPipelineManager(
        workspace_dir=workspace_dir,
        registry_url=registry_url
    )

# Export principal
__all__ = [
    "AdvancedDevOpsPipelineManager",
    "PipelineStage",
    "PipelineStatus",
    "PipelineStep",
    "PipelineExecution",
    "GitOpsConfig",
    "DeploymentTarget",
    "create_pipeline_manager"
]
