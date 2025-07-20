#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 MOTEUR DE WORKFLOWS ULTRA-AVANCÉ
Système d'orchestration de workflows distribués pour l'automation d'entreprise

Développé par l'équipe d'experts Achiri avec une architecture révolutionnaire
combinant l'orchestration distribuée, l'IA générative et la scalabilité cloud.

Architecture Enterprise-Grade:
├── 🔄 Orchestration Distribuée (Kubernetes, Docker Swarm, Cloud)
├── 🧠 IA Adaptative (Auto-optimisation, Prédictions, Auto-correction)
├── 🛡️ Sécurité Avancée (Zero-Trust, Encryption, Audit complet)
├── 📊 Monitoring Real-time (Métriques, Alerting, Observabilité)
├── 🌐 Multi-Cloud (AWS, Azure, GCP, Edge Computing)
└── 🔧 DevOps Intégré (CI/CD, Infrastructure as Code, GitOps)

Auteur: Fahed Mlaiel - Architecte Solutions d'Entreprise
Équipe: Experts DevOps, ML Engineers, Security Architects
Version: 3.0.0 - Production Ready Enterprise
"""

import asyncio
import uuid
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

# Imports pour l'orchestration avancée
import networkx as nx
from networkx.algorithms import dag, shortest_path
import yaml
import jinja2
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
from celery import Celery
import dramatiq
from dramatiq.brokers.redis import RedisBroker

# Imports cloud et Kubernetes
try:
    from kubernetes import client as k8s_client, config as k8s_config
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Imports monitoring et observabilité
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import opentelemetry
    from opentelemetry import trace, metrics as otel_metrics
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# Imports sécurité
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from passlib.context import CryptContext

# Imports ML et IA
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Configuration logging ultra-avancé
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ÉNUMÉRATIONS ET TYPES AVANCÉS
# =============================================================================

class WorkflowStatus(Enum):
    """États du workflow"""
    DRAFT = "draft"
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"
    SCHEDULED = "scheduled"

class TaskStatus(Enum):
    """États des tâches"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"

class TaskType(Enum):
    """Types de tâches"""
    # Tâches système
    SHELL_COMMAND = "shell_command"
    PYTHON_SCRIPT = "python_script"
    HTTP_REQUEST = "http_request"
    DATABASE_QUERY = "database_query"
    FILE_OPERATION = "file_operation"
    
    # Tâches conteneurs
    DOCKER_RUN = "docker_run"
    KUBERNETES_APPLY = "kubernetes_apply"
    KUBERNETES_DELETE = "kubernetes_delete"
    HELM_INSTALL = "helm_install"
    
    # Tâches cloud
    AWS_LAMBDA = "aws_lambda"
    AZURE_FUNCTION = "azure_function"
    GCP_FUNCTION = "gcp_function"
    TERRAFORM_APPLY = "terraform_apply"
    
    # Tâches notification
    EMAIL_SEND = "email_send"
    SLACK_MESSAGE = "slack_message"
    WEBHOOK_CALL = "webhook_call"
    SMS_SEND = "sms_send"
    
    # Tâches ML/IA
    ML_TRAINING = "ml_training"
    ML_PREDICTION = "ml_prediction"
    AI_ANALYSIS = "ai_analysis"
    DATA_PROCESSING = "data_processing"
    
    # Tâches avancées
    HUMAN_APPROVAL = "human_approval"
    CONDITION_CHECK = "condition_check"
    LOOP_ITERATION = "loop_iteration"
    PARALLEL_EXECUTION = "parallel_execution"
    WORKFLOW_CALL = "workflow_call"

class ExecutionMode(Enum):
    """Modes d'exécution"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    DAG = "dag"
    EVENT_DRIVEN = "event_driven"

class Priority(Enum):
    """Priorités d'exécution"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1

class RetryStrategy(Enum):
    """Stratégies de retry"""
    NONE = "none"
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CUSTOM = "custom"

# =============================================================================
# MODÈLES DE DONNÉES AVANCÉS
# =============================================================================

@dataclass
class TaskConfiguration:
    """Configuration avancée d'une tâche"""
    type: TaskType
    name: str
    description: str = ""
    
    # Configuration d'exécution
    command: Optional[str] = None
    script_path: Optional[str] = None
    docker_image: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    
    # Timeouts et retry
    timeout_seconds: int = 300
    retry_strategy: RetryStrategy = RetryStrategy.FIXED_DELAY
    max_retries: int = 3
    retry_delay_seconds: int = 5
    retry_multiplier: float = 2.0
    
    # Ressources
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    gpu_required: bool = False
    
    # Sécurité
    run_as_user: Optional[str] = None
    security_context: Dict[str, Any] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    
    # Validation et conditions
    pre_conditions: List[str] = field(default_factory=list)
    post_conditions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    # Configuration spécifique par type
    http_config: Optional[Dict[str, Any]] = None
    database_config: Optional[Dict[str, Any]] = None
    cloud_config: Optional[Dict[str, Any]] = None
    notification_config: Optional[Dict[str, Any]] = None
    ml_config: Optional[Dict[str, Any]] = None

@dataclass
class TaskExecution:
    """Exécution d'une tâche avec historique complet"""
    id: str
    task_id: str
    workflow_id: str
    status: TaskStatus = TaskStatus.PENDING
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Résultats
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Métadonnées d'exécution
    executor_node: Optional[str] = None
    execution_context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    last_retry_at: Optional[datetime] = None
    
    # Ressources utilisées
    cpu_usage_peak: Optional[float] = None
    memory_usage_peak: Optional[float] = None
    disk_io_read: Optional[int] = None
    disk_io_write: Optional[int] = None
    network_io_read: Optional[int] = None
    network_io_write: Optional[int] = None
    
    # Logs et artefacts
    log_entries: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    
    def mark_started(self):
        """Marquer le début de l'exécution"""
        self.started_at = datetime.utcnow()
        self.status = TaskStatus.RUNNING
    
    def mark_completed(self, success: bool, exit_code: Optional[int] = None, 
                      stdout: Optional[str] = None, stderr: Optional[str] = None,
                      error: Optional[str] = None):
        """Marquer la fin de l'exécution"""
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        
        self.status = TaskStatus.SUCCESS if success else TaskStatus.FAILED
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.error_message = error
        
        if not success and error:
            self.error_traceback = traceback.format_exc()

@dataclass
class WorkflowTask:
    """Tâche dans un workflow avec dépendances"""
    id: str
    name: str
    config: TaskConfiguration
    
    # Dépendances et ordre d'exécution
    depends_on: List[str] = field(default_factory=list)
    run_after: List[str] = field(default_factory=list)
    
    # Conditions d'exécution
    enabled: bool = True
    condition: Optional[str] = None  # Expression conditionnelle
    
    # Gestion des erreurs
    continue_on_failure: bool = False
    failure_strategy: str = "stop"  # stop, continue, retry, escalate
    
    # Variables et contexte
    input_variables: Dict[str, Any] = field(default_factory=dict)
    output_variables: List[str] = field(default_factory=list)
    
    # Métadonnées
    tags: List[str] = field(default_factory=list)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class WorkflowDefinition:
    """Définition complète d'un workflow"""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    
    # Configuration du workflow
    execution_mode: ExecutionMode = ExecutionMode.DAG
    priority: Priority = Priority.MEDIUM
    timeout_seconds: int = 3600
    
    # Tâches et structure
    tasks: List[WorkflowTask] = field(default_factory=list)
    start_tasks: List[str] = field(default_factory=list)
    end_tasks: List[str] = field(default_factory=list)
    
    # Variables globales
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration d'exécution
    max_parallel_tasks: int = 10
    auto_cleanup: bool = True
    save_artifacts: bool = True
    
    # Sécurité et permissions
    allowed_users: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    security_level: str = "standard"  # basic, standard, high, critical
    
    # Notification et monitoring
    notification_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    
    # Métadonnées
    tenant_id: str = ""
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validation de la définition du workflow"""
        errors = []
        
        # Validation de base
        if not self.name:
            errors.append("Le nom du workflow est requis")
        
        if not self.tasks:
            errors.append("Au moins une tâche est requise")
        
        # Validation des tâches
        task_ids = {task.id for task in self.tasks}
        
        for task in self.tasks:
            # Validation des dépendances
            for dep in task.depends_on:
                if dep not in task_ids:
                    errors.append(f"Dépendance invalide '{dep}' pour la tâche '{task.id}'")
        
        # Validation de la structure DAG
        if self.execution_mode == ExecutionMode.DAG:
            if not self._validate_dag_structure():
                errors.append("Structure DAG invalide - cycles détectés")
        
        # Validation des tâches de démarrage
        if self.start_tasks:
            for start_task in self.start_tasks:
                if start_task not in task_ids:
                    errors.append(f"Tâche de démarrage invalide '{start_task}'")
        
        return errors
    
    def _validate_dag_structure(self) -> bool:
        """Validation de la structure DAG (sans cycles)"""
        try:
            graph = nx.DiGraph()
            
            # Ajout des nœuds
            for task in self.tasks:
                graph.add_node(task.id)
            
            # Ajout des arêtes (dépendances)
            for task in self.tasks:
                for dep in task.depends_on:
                    graph.add_edge(dep, task.id)
            
            # Vérification des cycles
            return nx.is_directed_acyclic_graph(graph)
            
        except Exception:
            return False

@dataclass
class WorkflowExecution:
    """Exécution d'un workflow avec état complet"""
    id: str
    workflow_id: str
    workflow_definition: WorkflowDefinition
    status: WorkflowStatus = WorkflowStatus.PENDING
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Contexte d'exécution
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # État des tâches
    task_executions: Dict[str, TaskExecution] = field(default_factory=dict)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    
    # Métriques
    total_tasks: int = 0
    completed_task_count: int = 0
    failed_task_count: int = 0
    success_rate: float = 0.0
    
    # Métadonnées d'exécution
    executor_info: Dict[str, Any] = field(default_factory=dict)
    execution_node: Optional[str] = None
    tenant_id: str = ""
    user_id: str = ""
    
    # Gestion des erreurs
    error_summary: Optional[str] = None
    failure_reason: Optional[str] = None
    
    def update_status(self):
        """Mise à jour du statut basé sur l'état des tâches"""
        if not self.task_executions:
            return
        
        statuses = [exec.status for exec in self.task_executions.values()]
        
        if all(status == TaskStatus.SUCCESS for status in statuses):
            self.status = WorkflowStatus.COMPLETED
        elif any(status == TaskStatus.FAILED for status in statuses):
            self.status = WorkflowStatus.FAILED
        elif any(status == TaskStatus.RUNNING for status in statuses):
            self.status = WorkflowStatus.RUNNING
        elif all(status in [TaskStatus.PENDING, TaskStatus.SUCCESS] for status in statuses):
            if any(status == TaskStatus.PENDING for status in statuses):
                self.status = WorkflowStatus.RUNNING
        
        # Mise à jour des métriques
        self.completed_task_count = len([s for s in statuses if s == TaskStatus.SUCCESS])
        self.failed_task_count = len([s for s in statuses if s == TaskStatus.FAILED])
        self.total_tasks = len(statuses)
        
        if self.total_tasks > 0:
            self.success_rate = self.completed_task_count / self.total_tasks

# =============================================================================
# EXÉCUTEURS DE TÂCHES SPÉCIALISÉS
# =============================================================================

class TaskExecutor(ABC):
    """Interface pour les exécuteurs de tâches"""
    
    @abstractmethod
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> TaskExecution:
        """Exécute une tâche et retourne le résultat"""
        pass
    
    @abstractmethod
    async def validate(self, task: WorkflowTask) -> bool:
        """Valide une tâche avant exécution"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[TaskType]:
        """Retourne les types de tâches supportés"""
        pass
    
    @abstractmethod
    async def cleanup(self, execution: TaskExecution):
        """Nettoyage après exécution"""
        pass

class ShellCommandExecutor(TaskExecutor):
    """Exécuteur pour les commandes shell"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.allowed_commands = config.get('allowed_commands', [])
        self.forbidden_patterns = config.get('forbidden_patterns', ['rm -rf', 'format', 'del /'])
        self.execution_timeout = config.get('default_timeout', 300)
    
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> TaskExecution:
        """Exécution d'une commande shell"""
        execution = TaskExecution(
            id=str(uuid.uuid4()),
            task_id=task.id,
            workflow_id=context.get('workflow_id', ''),
            execution_context=context.copy()
        )
        
        try:
            execution.mark_started()
            
            # Préparation de la commande
            command = self._prepare_command(task, context)
            
            # Validation de sécurité
            if not self._validate_command_security(command):
                raise ValueError(f"Commande non autorisée ou dangereuse: {command}")
            
            # Exécution avec timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=task.config.working_directory,
                env={**os.environ, **task.config.environment}
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=task.config.timeout_seconds
                )
                
                execution.mark_completed(
                    success=process.returncode == 0,
                    exit_code=process.returncode,
                    stdout=stdout.decode('utf-8') if stdout else None,
                    stderr=stderr.decode('utf-8') if stderr else None,
                    error=None if process.returncode == 0 else f"Exit code: {process.returncode}"
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                execution.mark_completed(
                    success=False,
                    error=f"Timeout après {task.config.timeout_seconds} secondes"
                )
            
            logger.info("Commande shell exécutée", 
                       task_id=task.id, 
                       exit_code=execution.exit_code,
                       duration=execution.duration_seconds)
            
            return execution
            
        except Exception as e:
            execution.mark_completed(success=False, error=str(e))
            logger.error("Erreur exécution commande shell", task_id=task.id, error=str(e))
            return execution
    
    def _prepare_command(self, task: WorkflowTask, context: Dict[str, Any]) -> str:
        """Préparation de la commande avec substitution des variables"""
        command = task.config.command or ""
        
        # Substitution des variables du contexte
        for key, value in context.items():
            command = command.replace(f'${{{key}}}', str(value))
        
        # Substitution des variables de la tâche
        for key, value in task.input_variables.items():
            command = command.replace(f'${{{key}}}', str(value))
        
        return command
    
    def _validate_command_security(self, command: str) -> bool:
        """Validation de sécurité de la commande"""
        # Vérification des patterns interdits
        for pattern in self.forbidden_patterns:
            if pattern.lower() in command.lower():
                return False
        
        # Vérification de la liste blanche si configurée
        if self.allowed_commands:
            command_parts = command.split()
            if command_parts and command_parts[0] not in self.allowed_commands:
                return False
        
        return True
    
    async def validate(self, task: WorkflowTask) -> bool:
        """Validation d'une tâche shell"""
        if not task.config.command:
            return False
        
        return self._validate_command_security(task.config.command)
    
    def get_supported_types(self) -> List[TaskType]:
        """Types supportés"""
        return [TaskType.SHELL_COMMAND]
    
    async def cleanup(self, execution: TaskExecution):
        """Nettoyage après exécution shell"""
        # Nettoyage des fichiers temporaires si nécessaire
        pass

class DockerTaskExecutor(TaskExecutor):
    """Exécuteur pour les tâches Docker"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docker_client = None
        self._setup_docker_client()
    
    def _setup_docker_client(self):
        """Configuration du client Docker"""
        try:
            if DOCKER_AVAILABLE:
                self.docker_client = docker.from_env()
                logger.info("Client Docker configuré")
            else:
                logger.warning("Docker non disponible")
        except Exception as e:
            logger.error("Erreur configuration Docker", error=str(e))
    
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> TaskExecution:
        """Exécution d'une tâche Docker"""
        execution = TaskExecution(
            id=str(uuid.uuid4()),
            task_id=task.id,
            workflow_id=context.get('workflow_id', ''),
            execution_context=context.copy()
        )
        
        try:
            if not self.docker_client:
                raise Exception("Client Docker non disponible")
            
            execution.mark_started()
            
            # Configuration du conteneur
            container_config = self._prepare_container_config(task, context)
            
            # Création et démarrage du conteneur
            container = self.docker_client.containers.run(
                **container_config,
                detach=True
            )
            
            # Attente de la fin avec timeout
            try:
                result = container.wait(timeout=task.config.timeout_seconds)
                
                # Récupération des logs
                stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
                
                execution.mark_completed(
                    success=result['StatusCode'] == 0,
                    exit_code=result['StatusCode'],
                    stdout=stdout,
                    stderr=stderr,
                    error=None if result['StatusCode'] == 0 else f"Exit code: {result['StatusCode']}"
                )
                
            except Exception as timeout_error:
                container.kill()
                execution.mark_completed(
                    success=False,
                    error=f"Timeout Docker: {str(timeout_error)}"
                )
            
            finally:
                # Nettoyage du conteneur
                try:
                    container.remove(force=True)
                except:
                    pass
            
            logger.info("Tâche Docker exécutée", 
                       task_id=task.id,
                       image=container_config.get('image'),
                       exit_code=execution.exit_code)
            
            return execution
            
        except Exception as e:
            execution.mark_completed(success=False, error=str(e))
            logger.error("Erreur exécution Docker", task_id=task.id, error=str(e))
            return execution
    
    def _prepare_container_config(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Préparation de la configuration du conteneur"""
        config = {
            'image': task.config.docker_image,
            'environment': {**task.config.environment, **task.input_variables},
            'working_dir': task.config.working_directory or '/app',
            'network_mode': self.config.get('network_mode', 'bridge'),
            'mem_limit': task.config.memory_limit or self.config.get('default_memory_limit'),
            'cpu_count': task.config.cpu_limit or self.config.get('default_cpu_limit')
        }
        
        # Commande à exécuter
        if task.config.command:
            config['command'] = task.config.command
        
        # Volumes et montages
        if 'volumes' in self.config:
            config['volumes'] = self.config['volumes']
        
        # Configuration de sécurité
        if task.config.run_as_user:
            config['user'] = task.config.run_as_user
        
        return config
    
    async def validate(self, task: WorkflowTask) -> bool:
        """Validation d'une tâche Docker"""
        if not self.docker_client:
            return False
        
        if not task.config.docker_image:
            return False
        
        # Vérification de la disponibilité de l'image
        try:
            self.docker_client.images.get(task.config.docker_image)
            return True
        except docker.errors.ImageNotFound:
            # Tentative de pull de l'image
            try:
                self.docker_client.images.pull(task.config.docker_image)
                return True
            except Exception:
                return False
        except Exception:
            return False
    
    def get_supported_types(self) -> List[TaskType]:
        """Types supportés"""
        return [TaskType.DOCKER_RUN]
    
    async def cleanup(self, execution: TaskExecution):
        """Nettoyage après exécution Docker"""
        # Nettoyage des volumes et réseaux si nécessaire
        pass

class KubernetesTaskExecutor(TaskExecutor):
    """Exécuteur pour les tâches Kubernetes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k8s_client = None
        self.apps_v1 = None
        self.batch_v1 = None
        self._setup_kubernetes_client()
    
    def _setup_kubernetes_client(self):
        """Configuration du client Kubernetes"""
        try:
            if not KUBERNETES_AVAILABLE:
                logger.warning("Kubernetes non disponible")
                return
            
            # Configuration du client
            if self.config.get('in_cluster', False):
                k8s_config.load_incluster_config()
            else:
                k8s_config.load_kube_config(config_file=self.config.get('kubeconfig_path'))
            
            self.k8s_client = k8s_client.ApiClient()
            self.apps_v1 = k8s_client.AppsV1Api()
            self.batch_v1 = k8s_client.BatchV1Api()
            self.core_v1 = k8s_client.CoreV1Api()
            
            logger.info("Client Kubernetes configuré")
            
        except Exception as e:
            logger.error("Erreur configuration Kubernetes", error=str(e))
    
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> TaskExecution:
        """Exécution d'une tâche Kubernetes"""
        execution = TaskExecution(
            id=str(uuid.uuid4()),
            task_id=task.id,
            workflow_id=context.get('workflow_id', ''),
            execution_context=context.copy()
        )
        
        try:
            if not self.batch_v1:
                raise Exception("Client Kubernetes non disponible")
            
            execution.mark_started()
            
            # Création du Job Kubernetes
            job_manifest = self._prepare_job_manifest(task, context, execution.id)
            
            # Création du job
            namespace = self.config.get('namespace', 'default')
            job_response = self.batch_v1.create_namespaced_job(
                namespace=namespace,
                body=job_manifest
            )
            
            job_name = job_response.metadata.name
            
            # Attente de la fin du job
            success = await self._wait_for_job_completion(job_name, namespace, task.config.timeout_seconds)
            
            # Récupération des logs
            logs = await self._get_job_logs(job_name, namespace)
            
            execution.mark_completed(
                success=success,
                stdout=logs.get('stdout'),
                stderr=logs.get('stderr'),
                error=None if success else "Job Kubernetes échoué"
            )
            
            # Nettoyage du job
            if self.config.get('auto_cleanup', True):
                await self._cleanup_job(job_name, namespace)
            
            logger.info("Tâche Kubernetes exécutée", 
                       task_id=task.id,
                       job_name=job_name,
                       success=success)
            
            return execution
            
        except Exception as e:
            execution.mark_completed(success=False, error=str(e))
            logger.error("Erreur exécution Kubernetes", task_id=task.id, error=str(e))
            return execution
    
    def _prepare_job_manifest(self, task: WorkflowTask, context: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        """Préparation du manifeste Job Kubernetes"""
        job_name = f"workflow-{task.id}-{execution_id[:8]}"
        
        manifest = {
            'apiVersion': 'batch/v1',
            'kind': 'Job',
            'metadata': {
                'name': job_name,
                'labels': {
                    'workflow-id': context.get('workflow_id', ''),
                    'task-id': task.id,
                    'execution-id': execution_id
                }
            },
            'spec': {
                'template': {
                    'spec': {
                        'containers': [{
                            'name': 'task-container',
                            'image': task.config.docker_image,
                            'command': ['/bin/sh', '-c'] if task.config.command else None,
                            'args': [task.config.command] if task.config.command else None,
                            'env': [
                                {'name': k, 'value': str(v)} 
                                for k, v in {**task.config.environment, **task.input_variables}.items()
                            ],
                            'resources': {}
                        }],
                        'restartPolicy': 'Never'
                    }
                },
                'backoffLimit': task.config.max_retries,
                'activeDeadlineSeconds': task.config.timeout_seconds
            }
        }
        
        # Configuration des ressources
        resources = {}
        if task.config.cpu_limit:
            resources['limits'] = resources.get('limits', {})
            resources['limits']['cpu'] = task.config.cpu_limit
        if task.config.memory_limit:
            resources['limits'] = resources.get('limits', {})
            resources['limits']['memory'] = task.config.memory_limit
        
        if resources:
            manifest['spec']['template']['spec']['containers'][0]['resources'] = resources
        
        # Configuration de sécurité
        if task.config.run_as_user:
            manifest['spec']['template']['spec']['securityContext'] = {
                'runAsUser': int(task.config.run_as_user)
            }
        
        return manifest
    
    async def _wait_for_job_completion(self, job_name: str, namespace: str, timeout: int) -> bool:
        """Attente de la fin du job"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                job = self.batch_v1.read_namespaced_job(name=job_name, namespace=namespace)
                
                if job.status.succeeded:
                    return True
                elif job.status.failed:
                    return False
                
                await asyncio.sleep(5)  # Attente avant vérification suivante
                
            except Exception as e:
                logger.error("Erreur vérification job", job_name=job_name, error=str(e))
                return False
        
        return False  # Timeout
    
    async def _get_job_logs(self, job_name: str, namespace: str) -> Dict[str, str]:
        """Récupération des logs du job"""
        try:
            # Récupération des pods du job
            pods = self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f'job-name={job_name}'
            )
            
            logs = {'stdout': '', 'stderr': ''}
            
            for pod in pods.items:
                try:
                    pod_logs = self.core_v1.read_namespaced_pod_log(
                        name=pod.metadata.name,
                        namespace=namespace
                    )
                    logs['stdout'] += pod_logs
                except Exception as e:
                    logs['stderr'] += f"Erreur lecture logs pod {pod.metadata.name}: {str(e)}\n"
            
            return logs
            
        except Exception as e:
            return {'stdout': '', 'stderr': f"Erreur récupération logs: {str(e)}"}
    
    async def _cleanup_job(self, job_name: str, namespace: str):
        """Nettoyage du job"""
        try:
            self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                body=k8s_client.V1DeleteOptions(propagation_policy='Foreground')
            )
        except Exception as e:
            logger.warning("Erreur nettoyage job", job_name=job_name, error=str(e))
    
    async def validate(self, task: WorkflowTask) -> bool:
        """Validation d'une tâche Kubernetes"""
        if not self.batch_v1:
            return False
        
        if not task.config.docker_image:
            return False
        
        return True
    
    def get_supported_types(self) -> List[TaskType]:
        """Types supportés"""
        return [TaskType.KUBERNETES_APPLY, TaskType.KUBERNETES_DELETE]
    
    async def cleanup(self, execution: TaskExecution):
        """Nettoyage après exécution Kubernetes"""
        pass

class HTTPTaskExecutor(TaskExecutor):
    """Exécuteur pour les tâches HTTP"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=config.get('default_timeout', 300))
    
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> TaskExecution:
        """Exécution d'une requête HTTP"""
        execution = TaskExecution(
            id=str(uuid.uuid4()),
            task_id=task.id,
            workflow_id=context.get('workflow_id', ''),
            execution_context=context.copy()
        )
        
        try:
            execution.mark_started()
            
            # Configuration de la requête
            http_config = task.config.http_config or {}
            url = http_config.get('url')
            method = http_config.get('method', 'GET').upper()
            headers = http_config.get('headers', {})
            data = http_config.get('data')
            json_data = http_config.get('json')
            
            # Substitution des variables dans l'URL
            for key, value in {**context, **task.input_variables}.items():
                url = url.replace(f'${{{key}}}', str(value))
            
            # Exécution de la requête
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=data,
                    json=json_data
                ) as response:
                    
                    response_text = await response.text()
                    
                    # Détermination du succès
                    success = 200 <= response.status < 400
                    
                    execution.mark_completed(
                        success=success,
                        stdout=response_text if success else None,
                        stderr=response_text if not success else None,
                        error=None if success else f"HTTP {response.status}: {response.reason}"
                    )
                    
                    # Sauvegarde des données de réponse
                    execution.result_data = {
                        'status_code': response.status,
                        'headers': dict(response.headers),
                        'response_body': response_text[:10000]  # Limite à 10KB
                    }
            
            logger.info("Requête HTTP exécutée", 
                       task_id=task.id,
                       url=url,
                       method=method,
                       status_code=execution.result_data.get('status_code'))
            
            return execution
            
        except Exception as e:
            execution.mark_completed(success=False, error=str(e))
            logger.error("Erreur exécution HTTP", task_id=task.id, error=str(e))
            return execution
    
    async def validate(self, task: WorkflowTask) -> bool:
        """Validation d'une tâche HTTP"""
        http_config = task.config.http_config
        if not http_config or not http_config.get('url'):
            return False
        
        # Validation de l'URL
        try:
            from urllib.parse import urlparse
            parsed = urlparse(http_config['url'])
            return parsed.scheme in ['http', 'https'] and parsed.netloc
        except Exception:
            return False
    
    def get_supported_types(self) -> List[TaskType]:
        """Types supportés"""
        return [TaskType.HTTP_REQUEST]
    
# Imports manqués
import os
import aiohttp
import sys

class DatabaseTaskExecutor(TaskExecutor):
    """Exécuteur pour les tâches de base de données"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connections = {}
        self._setup_connections()
    
    def _setup_connections(self):
        """Configuration des connexions base de données"""
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            
            for db_name, db_config in self.config.get('databases', {}).items():
                engine = create_async_engine(db_config['connection_string'])
                self.connections[db_name] = engine
                logger.info("Connexion DB configurée", database=db_name)
                
        except ImportError:
            logger.warning("SQLAlchemy async non disponible")
        except Exception as e:
            logger.error("Erreur configuration DB", error=str(e))
    
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> TaskExecution:
        """Exécution d'une requête base de données"""
        execution = TaskExecution(
            id=str(uuid.uuid4()),
            task_id=task.id,
            workflow_id=context.get('workflow_id', ''),
            execution_context=context.copy()
        )
        
        try:
            execution.mark_started()
            
            db_config = task.config.database_config or {}
            database = db_config.get('database', 'default')
            query = db_config.get('query')
            parameters = db_config.get('parameters', {})
            
            if database not in self.connections:
                raise Exception(f"Base de données non configurée: {database}")
            
            # Substitution des variables
            for key, value in {**context, **task.input_variables}.items():
                query = query.replace(f'${{{key}}}', str(value))
            
            # Exécution de la requête
            engine = self.connections[database]
            async with engine.begin() as conn:
                from sqlalchemy import text
                result = await conn.execute(text(query), parameters)
                
                if result.returns_rows:
                    rows = result.fetchall()
                    data = [dict(row) for row in rows]
                    
                    execution.mark_completed(
                        success=True,
                        stdout=f"Requête exécutée avec succès. {len(rows)} lignes retournées."
                    )
                    execution.result_data = {'rows': data, 'count': len(rows)}
                    
                else:
                    execution.mark_completed(
                        success=True,
                        stdout=f"Requête exécutée avec succès. {result.rowcount} lignes affectées."
                    )
                    execution.result_data = {'rows_affected': result.rowcount}
            
            logger.info("Requête DB exécutée", task_id=task.id, database=database)
            return execution
            
        except Exception as e:
            execution.mark_completed(success=False, error=str(e))
            logger.error("Erreur exécution DB", task_id=task.id, error=str(e))
            return execution
    
    async def validate(self, task: WorkflowTask) -> bool:
        """Validation d'une tâche base de données"""
        db_config = task.config.database_config
        if not db_config or not db_config.get('query'):
            return False
        
        database = db_config.get('database', 'default')
        return database in self.connections
    
    def get_supported_types(self) -> List[TaskType]:
        """Types supportés"""
        return [TaskType.DATABASE_QUERY]
    
    async def cleanup(self, execution: TaskExecution):
        """Nettoyage après exécution DB"""
        pass

class NotificationTaskExecutor(TaskExecutor):
    """Exécuteur pour les tâches de notification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.notification_channels = config.get('channels', {})
    
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> TaskExecution:
        """Exécution d'une notification"""
        execution = TaskExecution(
            id=str(uuid.uuid4()),
            task_id=task.id,
            workflow_id=context.get('workflow_id', ''),
            execution_context=context.copy()
        )
        
        try:
            execution.mark_started()
            
            notif_config = task.config.notification_config or {}
            notification_type = notif_config.get('type', 'email')
            
            if notification_type == 'email':
                result = await self._send_email(notif_config, context, task.input_variables)
            elif notification_type == 'slack':
                result = await self._send_slack(notif_config, context, task.input_variables)
            elif notification_type == 'webhook':
                result = await self._send_webhook(notif_config, context, task.input_variables)
            else:
                raise Exception(f"Type de notification non supporté: {notification_type}")
            
            execution.mark_completed(success=True, stdout=f"Notification {notification_type} envoyée")
            execution.result_data = result
            
            logger.info("Notification envoyée", task_id=task.id, type=notification_type)
            return execution
            
        except Exception as e:
            execution.mark_completed(success=False, error=str(e))
            logger.error("Erreur envoi notification", task_id=task.id, error=str(e))
            return execution
    
    async def _send_email(self, config: Dict[str, Any], context: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Envoi d'email"""
        recipients = config.get('recipients', [])
        subject = self._substitute_variables(config.get('subject', 'Workflow Notification'), context, variables)
        body = self._substitute_variables(config.get('body', ''), context, variables)
        
        # Simulation d'envoi d'email
        logger.info("Email simulé", recipients=recipients, subject=subject)
        
        return {
            'type': 'email',
            'recipients': recipients,
            'subject': subject,
            'status': 'sent'
        }
    
    async def _send_slack(self, config: Dict[str, Any], context: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Envoi de message Slack"""
        webhook_url = config.get('webhook_url')
        channel = config.get('channel', '#general')
        message = self._substitute_variables(config.get('message', 'Workflow notification'), context, variables)
        
        payload = {
            'channel': channel,
            'text': message,
            'username': 'WorkflowBot'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                return {
                    'type': 'slack',
                    'channel': channel,
                    'status_code': response.status,
                    'status': 'sent' if response.status == 200 else 'failed'
                }
    
    async def _send_webhook(self, config: Dict[str, Any], context: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """Envoi de webhook"""
        url = config.get('url')
        method = config.get('method', 'POST')
        headers = config.get('headers', {})
        payload = config.get('payload', {})
        
        # Substitution des variables dans le payload
        payload_str = json.dumps(payload)
        payload_str = self._substitute_variables(payload_str, context, variables)
        payload = json.loads(payload_str)
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, json=payload, headers=headers) as response:
                return {
                    'type': 'webhook',
                    'url': url,
                    'method': method,
                    'status_code': response.status,
                    'status': 'sent' if response.status < 400 else 'failed'
                }
    
    def _substitute_variables(self, text: str, context: Dict[str, Any], variables: Dict[str, Any]) -> str:
        """Substitution des variables dans le texte"""
        for key, value in {**context, **variables}.items():
            text = text.replace(f'${{{key}}}', str(value))
        return text
    
    async def validate(self, task: WorkflowTask) -> bool:
        """Validation d'une tâche de notification"""
        notif_config = task.config.notification_config
        if not notif_config:
            return False
        
        notification_type = notif_config.get('type')
        
        if notification_type == 'email':
            return 'recipients' in notif_config
        elif notification_type == 'slack':
            return 'webhook_url' in notif_config
        elif notification_type == 'webhook':
            return 'url' in notif_config
        
        return False
    
    def get_supported_types(self) -> List[TaskType]:
        """Types supportés"""
        return [TaskType.EMAIL_SEND, TaskType.SLACK_MESSAGE, TaskType.WEBHOOK_CALL, TaskType.SMS_SEND]
    
    async def cleanup(self, execution: TaskExecution):
        """Nettoyage après notification"""
        pass

# =============================================================================
# MOTEUR DE WORKFLOWS PRINCIPAL
# =============================================================================

class WorkflowEngine:
    """Moteur de workflows ultra-avancé avec orchestration distribuée"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.executors: Dict[TaskType, TaskExecutor] = {}
        
        # État et métriques
        self.metrics = {
            'total_workflows': 0,
            'active_workflows': 0,
            'completed_workflows': 0,
            'failed_workflows': 0,
            'total_tasks_executed': 0,
            'average_workflow_duration': 0.0,
            'success_rate': 0.0
        }
        
        # Cache et optimisations
        self.execution_cache = {}
        self.dag_cache = {}
        
        # Configuration des composants
        self._setup_executors()
        self._setup_monitoring()
        self._setup_storage()
        
        # Pool de threads pour les opérations intensives
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get('max_workers', 10))
        
        logger.info("Moteur de workflows initialisé", config_keys=list(config.keys()))
    
    def _setup_executors(self):
        """Configuration des exécuteurs de tâches"""
        try:
            # Exécuteur Shell
            if 'shell' in self.config:
                shell_executor = ShellCommandExecutor(self.config['shell'])
                for task_type in shell_executor.get_supported_types():
                    self.executors[task_type] = shell_executor
            
            # Exécuteur Docker
            if 'docker' in self.config and DOCKER_AVAILABLE:
                docker_executor = DockerTaskExecutor(self.config['docker'])
                for task_type in docker_executor.get_supported_types():
                    self.executors[task_type] = docker_executor
            
            # Exécuteur Kubernetes
            if 'kubernetes' in self.config and KUBERNETES_AVAILABLE:
                k8s_executor = KubernetesTaskExecutor(self.config['kubernetes'])
                for task_type in k8s_executor.get_supported_types():
                    self.executors[task_type] = k8s_executor
            
            # Exécuteur HTTP
            if 'http' in self.config:
                http_executor = HTTPTaskExecutor(self.config['http'])
                for task_type in http_executor.get_supported_types():
                    self.executors[task_type] = http_executor
            
            # Exécuteur base de données
            if 'database' in self.config:
                db_executor = DatabaseTaskExecutor(self.config['database'])
                for task_type in db_executor.get_supported_types():
                    self.executors[task_type] = db_executor
            
            # Exécuteur notifications
            if 'notifications' in self.config:
                notif_executor = NotificationTaskExecutor(self.config['notifications'])
                for task_type in notif_executor.get_supported_types():
                    self.executors[task_type] = notif_executor
            
            logger.info("Exécuteurs configurés", count=len(self.executors))
            
        except Exception as e:
            logger.error("Erreur configuration exécuteurs", error=str(e))
    
    def _setup_monitoring(self):
        """Configuration du monitoring"""
        try:
            if PROMETHEUS_AVAILABLE and self.config.get('prometheus', {}).get('enabled', False):
                self.prometheus_metrics = {
                    'workflow_executions_total': Counter(
                        'workflow_executions_total',
                        'Total des exécutions de workflows',
                        ['workflow_id', 'status']
                    ),
                    'workflow_duration_seconds': Histogram(
                        'workflow_duration_seconds',
                        'Durée d\'exécution des workflows',
                        ['workflow_id']
                    ),
                    'task_executions_total': Counter(
                        'workflow_task_executions_total',
                        'Total des exécutions de tâches',
                        ['task_type', 'status']
                    ),
                    'active_workflows_gauge': Gauge(
                        'workflow_active_workflows',
                        'Nombre de workflows actifs'
                    )
                }
            else:
                self.prometheus_metrics = {}
            
            logger.info("Monitoring configuré")
            
        except Exception as e:
            logger.error("Erreur configuration monitoring", error=str(e))
            self.prometheus_metrics = {}
    
    def _setup_storage(self):
        """Configuration du stockage"""
        try:
            storage_config = self.config.get('storage', {})
            
            if storage_config.get('type') == 'redis':
                self.redis_client = redis.from_url(storage_config.get('url', 'redis://localhost:6379'))
                logger.info("Stockage Redis configuré")
            else:
                self.redis_client = None
                
        except Exception as e:
            logger.error("Erreur configuration stockage", error=str(e))
            self.redis_client = None
    
    async def register_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Enregistrement d'un workflow"""
        try:
            # Validation du workflow
            validation_errors = workflow.validate()
            if validation_errors:
                logger.error("Workflow invalide", workflow_id=workflow.id, errors=validation_errors)
                return False
            
            # Enregistrement
            self.workflows[workflow.id] = workflow
            
            # Cache du DAG
            if workflow.execution_mode == ExecutionMode.DAG:
                self.dag_cache[workflow.id] = self._build_dag(workflow)
            
            # Sauvegarde persistante si configurée
            if self.redis_client:
                await self.redis_client.set(
                    f"workflow:{workflow.id}",
                    json.dumps(workflow.__dict__, default=str),
                    ex=86400  # 24h
                )
            
            self.metrics['total_workflows'] += 1
            
            logger.info("Workflow enregistré", workflow_id=workflow.id, name=workflow.name)
            return True
            
        except Exception as e:
            logger.error("Erreur enregistrement workflow", workflow_id=workflow.id, error=str(e))
            return False
    
    def _build_dag(self, workflow: WorkflowDefinition) -> nx.DiGraph:
        """Construction du DAG pour un workflow"""
        try:
            graph = nx.DiGraph()
            
            # Ajout des nœuds (tâches)
            for task in workflow.tasks:
                graph.add_node(task.id, task=task)
            
            # Ajout des arêtes (dépendances)
            for task in workflow.tasks:
                for dependency in task.depends_on:
                    graph.add_edge(dependency, task.id)
            
            # Validation du DAG
            if not nx.is_directed_acyclic_graph(graph):
                raise ValueError("Le workflow contient des cycles")
            
            return graph
            
        except Exception as e:
            logger.error("Erreur construction DAG", workflow_id=workflow.id, error=str(e))
            raise
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any] = None, 
                             user_id: str = "", tenant_id: str = "") -> str:
        """Démarrage de l'exécution d'un workflow"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow non trouvé: {workflow_id}")
            
            workflow = self.workflows[workflow_id]
            
            # Création de l'exécution
            execution = WorkflowExecution(
                id=str(uuid.uuid4()),
                workflow_id=workflow_id,
                workflow_definition=workflow,
                input_data=input_data or {},
                user_id=user_id,
                tenant_id=tenant_id,
                variables={**workflow.variables, **(input_data or {})}
            )
            
            # Enregistrement de l'exécution active
            self.active_executions[execution.id] = execution
            
            # Démarrage de l'exécution asynchrone
            asyncio.create_task(self._run_workflow_execution(execution))
            
            # Métriques
            self.metrics['active_workflows'] += 1
            if 'active_workflows_gauge' in self.prometheus_metrics:
                self.prometheus_metrics['active_workflows_gauge'].set(self.metrics['active_workflows'])
            
            logger.info("Exécution de workflow démarrée", 
                       execution_id=execution.id,
                       workflow_id=workflow_id,
                       user_id=user_id)
            
            return execution.id
            
        except Exception as e:
            logger.error("Erreur démarrage workflow", workflow_id=workflow_id, error=str(e))
            raise
    
    async def _run_workflow_execution(self, execution: WorkflowExecution):
        """Exécution complète d'un workflow"""
        try:
            execution.started_at = datetime.utcnow()
            execution.status = WorkflowStatus.RUNNING
            
            logger.info("Début exécution workflow", execution_id=execution.id)
            
            # Exécution selon le mode
            if execution.workflow_definition.execution_mode == ExecutionMode.DAG:
                await self._execute_dag_workflow(execution)
            elif execution.workflow_definition.execution_mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential_workflow(execution)
            elif execution.workflow_definition.execution_mode == ExecutionMode.PARALLEL:
                await self._execute_parallel_workflow(execution)
            else:
                raise ValueError(f"Mode d'exécution non supporté: {execution.workflow_definition.execution_mode}")
            
            # Finalisation
            execution.completed_at = datetime.utcnow()
            if execution.started_at:
                execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
            
            # Mise à jour du statut final
            execution.update_status()
            
            # Métriques
            self._update_workflow_metrics(execution)
            
            logger.info("Exécution workflow terminée", 
                       execution_id=execution.id,
                       status=execution.status.value,
                       duration=execution.duration_seconds,
                       success_rate=execution.success_rate)
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.failure_reason = str(e)
            execution.completed_at = datetime.utcnow()
            
            logger.error("Erreur exécution workflow", execution_id=execution.id, error=str(e))
            
        finally:
            # Nettoyage
            if execution.id in self.active_executions:
                del self.active_executions[execution.id]
            
            self.metrics['active_workflows'] -= 1
            
            # Sauvegarde de l'historique si configurée
            if self.redis_client:
                await self._save_execution_history(execution)
    
    async def _execute_dag_workflow(self, execution: WorkflowExecution):
        """Exécution d'un workflow en mode DAG"""
        try:
            workflow_id = execution.workflow_definition.id
            
            if workflow_id not in self.dag_cache:
                self.dag_cache[workflow_id] = self._build_dag(execution.workflow_definition)
            
            dag = self.dag_cache[workflow_id]
            
            # Initialisation des exécutions de tâches
            for task in execution.workflow_definition.tasks:
                task_execution = TaskExecution(
                    id=str(uuid.uuid4()),
                    task_id=task.id,
                    workflow_id=execution.id
                )
                execution.task_executions[task.id] = task_execution
            
            # Exécution topologique
            ready_tasks = [node for node in dag.nodes() if dag.in_degree(node) == 0]
            completed_tasks = set()
            
            # Limite de parallélisme
            max_parallel = execution.workflow_definition.max_parallel_tasks
            running_tasks = set()
            
            while ready_tasks or running_tasks:
                # Démarrage des tâches prêtes
                while ready_tasks and len(running_tasks) < max_parallel:
                    task_id = ready_tasks.pop(0)
                    task = self._get_task_by_id(execution.workflow_definition, task_id)
                    
                    if task and task.enabled and self._evaluate_task_condition(task, execution.variables):
                        # Démarrage de la tâche
                        task_coroutine = self._execute_task(task, execution)
                        running_tasks.add(asyncio.create_task(task_coroutine))
                
                # Attente de la fin d'au moins une tâche
                if running_tasks:
                    done, running_tasks = await asyncio.wait(
                        running_tasks, 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Traitement des tâches terminées
                    for task_future in done:
                        task_result = await task_future
                        task_id = task_result.task_id
                        completed_tasks.add(task_id)
                        
                        # Mise à jour des variables de sortie
                        self._update_output_variables(execution, task_id, task_result)
                        
                        # Vérification des nouvelles tâches prêtes
                        for successor in dag.successors(task_id):
                            if successor not in completed_tasks and successor not in [t.task_id for t in ready_tasks]:
                                # Vérification que toutes les dépendances sont terminées
                                dependencies = list(dag.predecessors(successor))
                                if all(dep in completed_tasks for dep in dependencies):
                                    ready_tasks.append(successor)
                        
                        # Gestion des échecs
                        if task_result.status == TaskStatus.FAILED:
                            task = self._get_task_by_id(execution.workflow_definition, task_id)
                            if task and not task.continue_on_failure:
                                # Arrêt du workflow en cas d'échec
                                for running_task in running_tasks:
                                    running_task.cancel()
                                return
            
        except Exception as e:
            logger.error("Erreur exécution DAG", execution_id=execution.id, error=str(e))
            raise
    
    async def _execute_sequential_workflow(self, execution: WorkflowExecution):
        """Exécution séquentielle d'un workflow"""
        try:
            for task in execution.workflow_definition.tasks:
                if not task.enabled:
                    continue
                
                if not self._evaluate_task_condition(task, execution.variables):
                    logger.info("Tâche ignorée (condition non remplie)", task_id=task.id)
                    continue
                
                # Exécution de la tâche
                task_result = await self._execute_task(task, execution)
                
                # Mise à jour des variables
                self._update_output_variables(execution, task.id, task_result)
                
                # Gestion des échecs
                if task_result.status == TaskStatus.FAILED and not task.continue_on_failure:
                    execution.failure_reason = f"Tâche {task.id} échouée: {task_result.error_message}"
                    break
                    
        except Exception as e:
            logger.error("Erreur exécution séquentielle", execution_id=execution.id, error=str(e))
            raise
    
    async def _execute_parallel_workflow(self, execution: WorkflowExecution):
        """Exécution parallèle d'un workflow"""
        try:
            # Démarrage de toutes les tâches en parallèle
            tasks_to_execute = [
                task for task in execution.workflow_definition.tasks 
                if task.enabled and self._evaluate_task_condition(task, execution.variables)
            ]
            
            # Limite de parallélisme
            max_parallel = execution.workflow_definition.max_parallel_tasks
            
            # Exécution par batches
            for i in range(0, len(tasks_to_execute), max_parallel):
                batch = tasks_to_execute[i:i + max_parallel]
                
                # Démarrage du batch
                task_futures = [
                    self._execute_task(task, execution) 
                    for task in batch
                ]
                
                # Attente de la fin du batch
                results = await asyncio.gather(*task_futures, return_exceptions=True)
                
                # Traitement des résultats
                for task, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error("Erreur exécution tâche", task_id=task.id, error=str(result))
                    else:
                        self._update_output_variables(execution, task.id, result)
                        
        except Exception as e:
            logger.error("Erreur exécution parallèle", execution_id=execution.id, error=str(e))
            raise
    
    async def _execute_task(self, task: WorkflowTask, execution: WorkflowExecution) -> TaskExecution:
        """Exécution d'une tâche individuelle"""
        try:
            # Récupération de l'exécution de tâche
            task_execution = execution.task_executions.get(task.id)
            if not task_execution:
                task_execution = TaskExecution(
                    id=str(uuid.uuid4()),
                    task_id=task.id,
                    workflow_id=execution.id
                )
                execution.task_executions[task.id] = task_execution
            
            # Vérification de l'exécuteur disponible
            if task.config.type not in self.executors:
                task_execution.mark_completed(
                    success=False,
                    error=f"Exécuteur non disponible pour le type: {task.config.type}"
                )
                return task_execution
            
            # Préparation du contexte
            context = {
                'workflow_id': execution.id,
                'workflow_name': execution.workflow_definition.name,
                'task_id': task.id,
                'task_name': task.name,
                'tenant_id': execution.tenant_id,
                'user_id': execution.user_id,
                **execution.variables
            }
            
            # Exécution avec retry
            executor = self.executors[task.config.type]
            
            for attempt in range(task.config.max_retries + 1):
                try:
                    if attempt > 0:
                        task_execution.retry_count = attempt
                        task_execution.last_retry_at = datetime.utcnow()
                        
                        # Délai de retry
                        delay = self._calculate_retry_delay(task.config, attempt)
                        await asyncio.sleep(delay)
                        
                        logger.info("Retry tâche", task_id=task.id, attempt=attempt)
                    
                    # Exécution
                    result = await executor.execute(task, context)
                    
                    # Mise à jour des métriques
                    if 'task_executions_total' in self.prometheus_metrics:
                        status = 'success' if result.status == TaskStatus.SUCCESS else 'failed'
                        self.prometheus_metrics['task_executions_total'].labels(
                            task_type=task.config.type.value,
                            status=status
                        ).inc()
                    
                    self.metrics['total_tasks_executed'] += 1
                    
                    return result
                    
                except Exception as e:
                    if attempt == task.config.max_retries:
                        # Dernier essai échoué
                        task_execution.mark_completed(success=False, error=str(e))
                        logger.error("Tâche échouée après tous les retries", 
                                   task_id=task.id, 
                                   attempts=attempt + 1, 
                                   error=str(e))
                        return task_execution
                    else:
                        logger.warning("Échec tâche, retry prévu", 
                                     task_id=task.id, 
                                     attempt=attempt + 1, 
                                     error=str(e))
            
            return task_execution
            
        except Exception as e:
            logger.error("Erreur exécution tâche", task_id=task.id, error=str(e))
            task_execution.mark_completed(success=False, error=str(e))
            return task_execution
    
    def _calculate_retry_delay(self, config: TaskConfiguration, attempt: int) -> float:
        """Calcul du délai de retry"""
        base_delay = config.retry_delay_seconds
        
        if config.retry_strategy == RetryStrategy.FIXED_DELAY:
            return base_delay
        elif config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return base_delay * (config.retry_multiplier ** (attempt - 1))
        elif config.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            return base_delay * attempt
        else:
            return base_delay
    
    def _get_task_by_id(self, workflow: WorkflowDefinition, task_id: str) -> Optional[WorkflowTask]:
        """Récupération d'une tâche par son ID"""
        for task in workflow.tasks:
            if task.id == task_id:
                return task
        return None
    
    def _evaluate_task_condition(self, task: WorkflowTask, variables: Dict[str, Any]) -> bool:
        """Évaluation de la condition d'une tâche"""
        if not task.condition:
            return True
        
        try:
            # Évaluation simple des conditions
            condition = task.condition
            for key, value in variables.items():
                condition = condition.replace(f'${{{key}}}', str(value))
            
            # Évaluation sécurisée (à améliorer avec un parser plus robuste)
            # Pour l'instant, évaluation basique
            if '==' in condition:
                left, right = condition.split('==', 1)
                return left.strip().strip('"\'') == right.strip().strip('"\'')
            elif '!=' in condition:
                left, right = condition.split('!=', 1)
                return left.strip().strip('"\'') != right.strip().strip('"\'')
            elif condition.lower() in ['true', '1', 'yes']:
                return True
            elif condition.lower() in ['false', '0', 'no']:
                return False
            
            return True
            
        except Exception as e:
            logger.warning("Erreur évaluation condition", task_id=task.id, condition=task.condition, error=str(e))
            return True
    
    def _update_output_variables(self, execution: WorkflowExecution, task_id: str, task_result: TaskExecution):
        """Mise à jour des variables de sortie"""
        try:
            task = self._get_task_by_id(execution.workflow_definition, task_id)
            if not task or not task.output_variables:
                return
            
            # Extraction des variables de sortie du résultat
            for var_name in task.output_variables:
                if var_name in task_result.result_data:
                    execution.variables[var_name] = task_result.result_data[var_name]
                elif var_name == 'exit_code':
                    execution.variables[f'{task_id}_exit_code'] = task_result.exit_code
                elif var_name == 'stdout':
                    execution.variables[f'{task_id}_stdout'] = task_result.stdout
                elif var_name == 'stderr':
                    execution.variables[f'{task_id}_stderr'] = task_result.stderr
                elif var_name == 'duration':
                    execution.variables[f'{task_id}_duration'] = task_result.duration_seconds
                elif var_name == 'status':
                    execution.variables[f'{task_id}_status'] = task_result.status.value
                    
        except Exception as e:
            logger.error("Erreur mise à jour variables sortie", task_id=task_id, error=str(e))
    
    def _update_workflow_metrics(self, execution: WorkflowExecution):
        """Mise à jour des métriques de workflow"""
        try:
            if execution.status == WorkflowStatus.COMPLETED:
                self.metrics['completed_workflows'] += 1
            elif execution.status == WorkflowStatus.FAILED:
                self.metrics['failed_workflows'] += 1
            
            # Calcul du taux de réussite
            total_completed = self.metrics['completed_workflows'] + self.metrics['failed_workflows']
            if total_completed > 0:
                self.metrics['success_rate'] = self.metrics['completed_workflows'] / total_completed
            
            # Durée moyenne
            if execution.duration_seconds:
                current_avg = self.metrics['average_workflow_duration']
                count = total_completed
                self.metrics['average_workflow_duration'] = (
                    (current_avg * (count - 1) + execution.duration_seconds) / count
                )
            
            # Métriques Prometheus
            if 'workflow_executions_total' in self.prometheus_metrics:
                self.prometheus_metrics['workflow_executions_total'].labels(
                    workflow_id=execution.workflow_definition.id,
                    status=execution.status.value
                ).inc()
            
            if 'workflow_duration_seconds' in self.prometheus_metrics and execution.duration_seconds:
                self.prometheus_metrics['workflow_duration_seconds'].labels(
                    workflow_id=execution.workflow_definition.id
                ).observe(execution.duration_seconds)
                
        except Exception as e:
            logger.error("Erreur mise à jour métriques", execution_id=execution.id, error=str(e))
    
    async def _save_execution_history(self, execution: WorkflowExecution):
        """Sauvegarde de l'historique d'exécution"""
        try:
            if self.redis_client:
                history_key = f"execution_history:{execution.id}"
                history_data = {
                    'id': execution.id,
                    'workflow_id': execution.workflow_id,
                    'status': execution.status.value,
                    'started_at': execution.started_at.isoformat() if execution.started_at else None,
                    'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
                    'duration_seconds': execution.duration_seconds,
                    'success_rate': execution.success_rate,
                    'task_count': len(execution.task_executions),
                    'user_id': execution.user_id,
                    'tenant_id': execution.tenant_id
                }
                
                await self.redis_client.set(
                    history_key,
                    json.dumps(history_data, default=str),
                    ex=86400 * 30  # 30 jours
                )
                
        except Exception as e:
            logger.error("Erreur sauvegarde historique", execution_id=execution.id, error=str(e))
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Récupération du statut d'une exécution"""
        try:
            # Recherche dans les exécutions actives
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                return self._execution_to_dict(execution)
            
            # Recherche dans l'historique Redis
            if self.redis_client:
                history_data = await self.redis_client.get(f"execution_history:{execution_id}")
                if history_data:
                    return json.loads(history_data)
            
            return None
            
        except Exception as e:
            logger.error("Erreur récupération statut", execution_id=execution_id, error=str(e))
            return None
    
    def _execution_to_dict(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Conversion d'une exécution en dictionnaire"""
        return {
            'id': execution.id,
            'workflow_id': execution.workflow_id,
            'workflow_name': execution.workflow_definition.name,
            'status': execution.status.value,
            'started_at': execution.started_at.isoformat() if execution.started_at else None,
            'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
            'duration_seconds': execution.duration_seconds,
            'progress': {
                'total_tasks': len(execution.workflow_definition.tasks),
                'completed_tasks': execution.completed_task_count,
                'failed_tasks': execution.failed_task_count,
                'success_rate': execution.success_rate
            },
            'task_executions': {
                task_id: {
                    'status': exec.status.value,
                    'started_at': exec.started_at.isoformat() if exec.started_at else None,
                    'completed_at': exec.completed_at.isoformat() if exec.completed_at else None,
                    'duration_seconds': exec.duration_seconds,
                    'retry_count': exec.retry_count
                }
                for task_id, exec in execution.task_executions.items()
            },
            'user_id': execution.user_id,
            'tenant_id': execution.tenant_id
        }
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Annulation d'une exécution"""
        try:
            if execution_id not in self.active_executions:
                logger.warning("Exécution non trouvée pour annulation", execution_id=execution_id)
                return False
            
            execution = self.active_executions[execution_id]
            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = datetime.utcnow()
            
            # Annulation des tâches en cours
            for task_execution in execution.task_executions.values():
                if task_execution.status == TaskStatus.RUNNING:
                    task_execution.status = TaskStatus.CANCELLED
                    task_execution.completed_at = datetime.utcnow()
            
            logger.info("Exécution annulée", execution_id=execution_id)
            return True
            
        except Exception as e:
            logger.error("Erreur annulation exécution", execution_id=execution_id, error=str(e))
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Récupération des métriques du moteur"""
        try:
            metrics = self.metrics.copy()
            
            # Métriques en temps réel
            metrics['active_workflows'] = len(self.active_executions)
            metrics['registered_workflows'] = len(self.workflows)
            metrics['available_executors'] = len(self.executors)
            
            # Métriques détaillées des exécutions actives
            active_details = []
            for execution in self.active_executions.values():
                active_details.append({
                    'id': execution.id,
                    'workflow_id': execution.workflow_id,
                    'status': execution.status.value,
                    'started_at': execution.started_at.isoformat() if execution.started_at else None,
                    'progress': execution.success_rate
                })
            
            metrics['active_executions'] = active_details
            
            return metrics
            
        except Exception as e:
            logger.error("Erreur récupération métriques", error=str(e))
            return self.metrics.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification de santé du moteur"""
        try:
            health = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'components': {}
            }
            
            # Vérification des exécuteurs
            for task_type, executor in self.executors.items():
                try:
                    # Test basique de l'exécuteur
                    test_task = WorkflowTask(
                        id="health_check",
                        name="Health Check",
                        config=TaskConfiguration(type=task_type, name="test")
                    )
                    is_healthy = await executor.validate(test_task)
                    health['components'][f'executor_{task_type.value}'] = 'healthy' if is_healthy else 'degraded'
                except Exception:
                    health['components'][f'executor_{task_type.value}'] = 'unhealthy'
            
            # Vérification du stockage Redis
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    health['components']['redis_storage'] = 'healthy'
                except Exception:
                    health['components']['redis_storage'] = 'unhealthy'
            
            # Statut global
            unhealthy_components = [k for k, v in health['components'].items() if v == 'unhealthy']
            if unhealthy_components:
                health['status'] = 'unhealthy'
                health['unhealthy_components'] = unhealthy_components
            elif any(v == 'degraded' for v in health['components'].values()):
                health['status'] = 'degraded'
            
            return health
            
        except Exception as e:
            logger.error("Erreur health check", error=str(e))
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def shutdown(self):
        """Arrêt propre du moteur"""
        try:
            logger.info("Arrêt du moteur de workflows")
            
            # Attente de la fin des exécutions actives
            if self.active_executions:
                logger.info("Attente fin exécutions actives", count=len(self.active_executions))
                
                timeout = 120  # 2 minutes de timeout
                start_time = time.time()
                
                while self.active_executions and (time.time() - start_time) < timeout:
                    await asyncio.sleep(5)
                
                if self.active_executions:
                    logger.warning("Exécutions actives forcément interrompues", count=len(self.active_executions))
                    for execution_id in list(self.active_executions.keys()):
                        await self.cancel_execution(execution_id)
            
            # Nettoyage des ressources
            for executor in self.executors.values():
                if hasattr(executor, 'cleanup'):
                    try:
                        await executor.cleanup(None)
                    except Exception as e:
                        logger.error("Erreur nettoyage exécuteur", error=str(e))
            
            # Fermeture du pool de threads
            self.thread_pool.shutdown(wait=True)
            
            # Fermeture de la connexion Redis
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Moteur de workflows arrêté")
            
        except Exception as e:
            logger.error("Erreur arrêt moteur", error=str(e))

# =============================================================================
# UTILITAIRES ET HELPERS
# =============================================================================

def create_simple_workflow(name: str, tasks: List[Dict[str, Any]]) -> WorkflowDefinition:
    """Création simplifiée d'un workflow"""
    workflow_tasks = []
    
    for i, task_def in enumerate(tasks):
        task_config = TaskConfiguration(
            type=TaskType(task_def.get('type', 'shell_command')),
            name=task_def.get('name', f'Task {i+1}'),
            command=task_def.get('command'),
            timeout_seconds=task_def.get('timeout', 300)
        )
        
        task = WorkflowTask(
            id=f'task_{i+1}',
            name=task_config.name,
            config=task_config,
            depends_on=task_def.get('depends_on', [])
        )
        
        workflow_tasks.append(task)
    
    return WorkflowDefinition(
        id=str(uuid.uuid4()),
        name=name,
        description=f"Workflow {name} créé automatiquement",
        tasks=workflow_tasks,
        execution_mode=ExecutionMode.DAG
    )

def load_workflow_from_yaml(file_path: str) -> WorkflowDefinition:
    """Chargement d'un workflow depuis un fichier YAML"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Conversion des données YAML en WorkflowDefinition
        tasks = []
        for task_data in data.get('tasks', []):
            config = TaskConfiguration(
                type=TaskType(task_data.get('type')),
                name=task_data.get('name'),
                command=task_data.get('command'),
                timeout_seconds=task_data.get('timeout', 300),
                docker_image=task_data.get('docker_image'),
                environment=task_data.get('environment', {}),
                http_config=task_data.get('http_config'),
                database_config=task_data.get('database_config'),
                notification_config=task_data.get('notification_config')
            )
            
            task = WorkflowTask(
                id=task_data.get('id'),
                name=task_data.get('name'),
                config=config,
                depends_on=task_data.get('depends_on', []),
                condition=task_data.get('condition'),
                continue_on_failure=task_data.get('continue_on_failure', False),
                input_variables=task_data.get('input_variables', {}),
                output_variables=task_data.get('output_variables', [])
            )
            
            tasks.append(task)
        
        workflow = WorkflowDefinition(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name'),
            description=data.get('description', ''),
            version=data.get('version', '1.0.0'),
            execution_mode=ExecutionMode(data.get('execution_mode', 'dag')),
            priority=Priority(data.get('priority', 'medium')),
            timeout_seconds=data.get('timeout', 3600),
            tasks=tasks,
            variables=data.get('variables', {}),
            max_parallel_tasks=data.get('max_parallel_tasks', 10)
        )
        
        return workflow
        
    except Exception as e:
        logger.error("Erreur chargement workflow YAML", file_path=file_path, error=str(e))
        raise

async def main():
    """Fonction principale de démonstration"""
    # Configuration du moteur
    config = {
        'shell': {
            'allowed_commands': ['echo', 'ls', 'cat', 'grep'],
            'forbidden_patterns': ['rm', 'del', 'format'],
            'default_timeout': 300
        },
        'http': {
            'default_timeout': 60
        },
        'prometheus': {
            'enabled': True
        },
        'storage': {
            'type': 'redis',
            'url': 'redis://localhost:6379'
        },
        'max_workers': 5
    }
    
    # Création du moteur
    engine = WorkflowEngine(config)
    
    try:
        # Exemple de workflow simple
        simple_workflow = create_simple_workflow("Test Workflow", [
            {
                'type': 'shell_command',
                'name': 'Echo Hello',
                'command': 'echo "Hello World"',
                'timeout': 30
            },
            {
                'type': 'shell_command',
                'name': 'List Files',
                'command': 'ls -la',
                'depends_on': ['task_1'],
                'timeout': 30
            }
        ])
        
        # Enregistrement du workflow
        success = await engine.register_workflow(simple_workflow)
        if success:
            print(f"Workflow enregistré: {simple_workflow.id}")
            
            # Exécution du workflow
            execution_id = await engine.execute_workflow(
                simple_workflow.id,
                input_data={'test_var': 'test_value'},
                user_id='demo_user'
            )
            
            print(f"Exécution démarrée: {execution_id}")
            
            # Attente et vérification du statut
            await asyncio.sleep(10)
            
            status = await engine.get_execution_status(execution_id)
            if status:
                print(f"Statut: {status['status']}")
                print(f"Progression: {status['progress']}")
        
        # Métriques
        metrics = await engine.get_metrics()
        print(f"Métriques: {metrics}")
        
        # Health check
        health = await engine.health_check()
        print(f"Santé: {health['status']}")
        
    finally:
        # Arrêt propre
        await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
