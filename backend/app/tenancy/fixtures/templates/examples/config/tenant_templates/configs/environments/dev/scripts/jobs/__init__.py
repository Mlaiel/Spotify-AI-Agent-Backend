#!/usr/bin/env python3
"""
Spotify AI Agent - Système de Gestion de Jobs Enterprise Ultra-Avancé
====================================================================

Module de gestion avancée des jobs Kubernetes pour l'agent IA Spotify
Développé par Fahed Mlaiel - Lead Infrastructure Engineering

Ce module fournit une infrastructure complète et industrialisée pour :
- Orchestration de jobs Kubernetes multi-tenant
- Gestion de cycle de vie avancée des tâches
- Monitoring et observabilité en temps réel
- Sécurité de niveau enterprise (PCI-DSS, SOX, GDPR, HIPAA, ISO27001)
- Automation CI/CD et déploiement zero-downtime
- Support multi-cloud et haute disponibilité

Architecture:
- ML Training Jobs: Formation de modèles IA avec GPUs
- Data ETL Jobs: Pipeline de données temps réel avec Kafka/Spark
- Security Scan Jobs: Analyse de sécurité et conformité
- Billing Report Jobs: Rapports financiers multi-devises
- Tenant Backup Jobs: Sauvegarde et migration zéro interruption

Version: 7.2.1-enterprise
License: Proprietary - Spotify AI Agent Platform
Author: Fahed Mlaiel <fahed.mlaiel@spotify-ai-agent.com>
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import uuid
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
import concurrent.futures
import threading
import signal
import resource
import psutil
import aiofiles
import aiohttp
import asyncpg
import redis.asyncio as redis
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
import structlog

# Configuration des logs structurés
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Métriques Prometheus
METRICS_REGISTRY = CollectorRegistry()
JOB_EXECUTIONS = Counter('spotify_ai_job_executions_total', 'Total job executions', ['job_type', 'tenant', 'status'], registry=METRICS_REGISTRY)
JOB_DURATION = Histogram('spotify_ai_job_duration_seconds', 'Job execution duration', ['job_type', 'tenant'], registry=METRICS_REGISTRY)
ACTIVE_JOBS = Gauge('spotify_ai_active_jobs', 'Currently active jobs', ['job_type', 'tenant'], registry=METRICS_REGISTRY)
RESOURCE_USAGE = Gauge('spotify_ai_job_resources', 'Job resource usage', ['job_type', 'tenant', 'resource'], registry=METRICS_REGISTRY)

class JobType(Enum):
    """Types de jobs supportés dans le système enterprise"""
    ML_TRAINING = "ml_training"
    DATA_ETL = "data_etl"
    SECURITY_SCAN = "security_scan"
    BILLING_REPORT = "billing_report"
    TENANT_BACKUP = "tenant_backup"
    CUSTOM = "custom"

class JobStatus(Enum):
    """États possibles des jobs"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class Priority(Enum):
    """Niveaux de priorité des jobs"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class TenantTier(Enum):
    """Niveaux de service tenant"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"

@dataclass
class ResourceLimits:
    """Configuration des limites de ressources"""
    cpu_request: str = "100m"
    cpu_limit: str = "1000m"
    memory_request: str = "128Mi"
    memory_limit: str = "1Gi"
    storage_request: str = "1Gi"
    storage_limit: str = "10Gi"
    gpu_count: int = 0
    gpu_type: str = "nvidia.com/gpu"

@dataclass
class SecurityContext:
    """Contexte de sécurité pour les jobs"""
    run_as_user: int = 1000
    run_as_group: int = 3000
    fs_group: int = 2000
    read_only_root_filesystem: bool = True
    allow_privilege_escalation: bool = False
    capabilities_drop: List[str] = field(default_factory=lambda: ["ALL"])
    seccomp_profile: str = "runtime/default"

@dataclass
class ComplianceConfig:
    """Configuration de conformité enterprise"""
    pci_dss_level: str = "level_1"
    sox_compliance: bool = True
    gdpr_compliance: bool = True
    hipaa_compliance: bool = True
    iso27001_compliance: bool = True
    data_residency_requirements: List[str] = field(default_factory=list)
    audit_logging_enabled: bool = True
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True

@dataclass
class MonitoringConfig:
    """Configuration du monitoring et observabilité"""
    prometheus_enabled: bool = True
    jaeger_enabled: bool = True
    elastic_apm_enabled: bool = True
    grafana_dashboard_enabled: bool = True
    alertmanager_enabled: bool = True
    log_level: str = "INFO"
    metrics_retention: str = "30d"
    traces_retention: str = "7d"

@dataclass
class TenantConfig:
    """Configuration tenant enterprise"""
    tenant_id: str
    tenant_name: str
    tenant_tier: TenantTier
    namespace: str
    resource_quota: Dict[str, str]
    network_policies: List[str] = field(default_factory=list)
    rbac_rules: List[str] = field(default_factory=list)
    backup_policy: str = "daily"
    disaster_recovery_enabled: bool = True

@dataclass
class JobDefinition:
    """Définition complète d'un job enterprise"""
    job_id: str
    job_name: str
    job_type: JobType
    tenant_config: TenantConfig
    priority: Priority = Priority.NORMAL
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    security_context: SecurityContext = field(default_factory=SecurityContext)
    compliance_config: ComplianceConfig = field(default_factory=ComplianceConfig)
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    init_containers: List[Dict[str, Any]] = field(default_factory=list)
    sidecar_containers: List[Dict[str, Any]] = field(default_factory=list)
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    affinity: Dict[str, Any] = field(default_factory=dict)
    restart_policy: str = "OnFailure"
    backoff_limit: int = 3
    active_deadline_seconds: int = 3600
    ttl_seconds_after_finished: int = 7200
    parallelism: int = 1
    completions: int = 1
    suspend: bool = False
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

class JobExecutionContext:
    """Contexte d'exécution d'un job avec state management"""
    
    def __init__(self, job_definition: JobDefinition):
        self.job_definition = job_definition
        self.execution_id = str(uuid.uuid4())
        self.status = JobStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.kubernetes_job_name: Optional[str] = None
        self.error_message: Optional[str] = None
        self.logs: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.checkpoints: List[Dict[str, Any]] = []
        self.resource_usage: Dict[str, float] = {}

class KubernetesJobManager:
    """Gestionnaire avancé de jobs Kubernetes avec support enterprise"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.logger = structlog.get_logger()
        self.kubeconfig_path = kubeconfig_path
        self.k8s_client: Optional[client.BatchV1Api] = None
        self.k8s_core_client: Optional[client.CoreV1Api] = None
        self.redis_client: Optional[redis.Redis] = None
        self.active_jobs: Dict[str, JobExecutionContext] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.shutdown_event = asyncio.Event()
        
        # Métriques
        self.metrics_server_port = int(os.getenv('METRICS_PORT', '9090'))
        self.metrics_registry = METRICS_REGISTRY
        
        # Configuration
        self.max_concurrent_jobs = int(os.getenv('MAX_CONCURRENT_JOBS', '50'))
        self.job_timeout_default = int(os.getenv('JOB_TIMEOUT_DEFAULT', '3600'))
        
    async def initialize(self):
        """Initialisation du gestionnaire de jobs"""
        try:
            await self._init_kubernetes_client()
            await self._init_redis_client()
            await self._start_metrics_server()
            await self._start_job_workers()
            self.logger.info("Job manager initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize job manager", error=str(e))
            raise

    async def _init_kubernetes_client(self):
        """Initialisation du client Kubernetes"""
        try:
            if self.kubeconfig_path:
                config.load_kube_config(config_file=self.kubeconfig_path)
            else:
                config.load_incluster_config()
            
            self.k8s_client = client.BatchV1Api()
            self.k8s_core_client = client.CoreV1Api()
            self.logger.info("Kubernetes client initialized")
        except Exception as e:
            self.logger.error("Failed to initialize Kubernetes client", error=str(e))
            raise

    async def _init_redis_client(self):
        """Initialisation du client Redis pour le state management"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            self.logger.info("Redis client initialized")
        except Exception as e:
            self.logger.error("Failed to initialize Redis client", error=str(e))
            raise

    async def _start_metrics_server(self):
        """Démarrage du serveur de métriques Prometheus"""
        try:
            start_http_server(self.metrics_server_port, registry=self.metrics_registry)
            self.logger.info("Metrics server started", port=self.metrics_server_port)
        except Exception as e:
            self.logger.error("Failed to start metrics server", error=str(e))
            raise

    async def _start_job_workers(self):
        """Démarrage des workers pour traitement des jobs"""
        for i in range(5):  # 5 workers par défaut
            asyncio.create_task(self._job_worker(f"worker-{i}"))
        self.logger.info("Job workers started")

    async def _job_worker(self, worker_name: str):
        """Worker pour traitement asynchrone des jobs"""
        while not self.shutdown_event.is_set():
            try:
                # Timeout pour éviter le blocage indefini
                job_context = await asyncio.wait_for(
                    self.job_queue.get(), timeout=1.0
                )
                await self._execute_job(job_context)
                self.job_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error("Worker error", worker=worker_name, error=str(e))

    async def submit_job(self, job_definition: JobDefinition) -> str:
        """Soumission d'un nouveau job à la queue"""
        try:
            # Validation du job
            await self._validate_job_definition(job_definition)
            
            # Création du contexte d'exécution
            job_context = JobExecutionContext(job_definition)
            
            # Enregistrement dans Redis
            await self._save_job_context(job_context)
            
            # Ajout à la queue
            await self.job_queue.put(job_context)
            
            # Métriques
            JOB_EXECUTIONS.labels(
                job_type=job_definition.job_type.value,
                tenant=job_definition.tenant_config.tenant_id,
                status="submitted"
            ).inc()
            
            self.logger.info(
                "Job submitted successfully",
                job_id=job_definition.job_id,
                execution_id=job_context.execution_id
            )
            
            return job_context.execution_id
            
        except Exception as e:
            self.logger.error("Failed to submit job", error=str(e))
            raise

    async def _validate_job_definition(self, job_definition: JobDefinition):
        """Validation complète de la définition du job"""
        # Validation du tenant
        if not job_definition.tenant_config.tenant_id:
            raise ValueError("Tenant ID is required")
        
        # Validation des ressources
        if job_definition.resource_limits.gpu_count > 8:
            raise ValueError("Maximum 8 GPUs per job")
        
        # Validation de la sécurité
        if job_definition.priority == Priority.EMERGENCY:
            if not job_definition.compliance_config.sox_compliance:
                raise ValueError("Emergency jobs require SOX compliance")
        
        # Validation namespace Kubernetes
        namespace = job_definition.tenant_config.namespace
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.worker_pool,
                self.k8s_core_client.read_namespace,
                namespace
            )
        except ApiException as e:
            if e.status == 404:
                raise ValueError(f"Namespace {namespace} does not exist")
            raise

    async def _execute_job(self, job_context: JobExecutionContext):
        """Exécution d'un job avec monitoring complet"""
        try:
            job_context.status = JobStatus.RUNNING
            job_context.start_time = datetime.utcnow()
            
            # Mise à jour des métriques
            ACTIVE_JOBS.labels(
                job_type=job_context.job_definition.job_type.value,
                tenant=job_context.job_definition.tenant_config.tenant_id
            ).inc()
            
            # Sauvegarde du state
            await self._save_job_context(job_context)
            
            # Génération du manifest Kubernetes
            k8s_manifest = await self._generate_kubernetes_manifest(job_context)
            
            # Déploiement du job
            await self._deploy_kubernetes_job(job_context, k8s_manifest)
            
            # Monitoring de l'exécution
            await self._monitor_job_execution(job_context)
            
            self.logger.info(
                "Job execution completed",
                execution_id=job_context.execution_id,
                status=job_context.status.value,
                duration=self._calculate_duration(job_context)
            )
            
        except Exception as e:
            job_context.status = JobStatus.FAILED
            job_context.error_message = str(e)
            job_context.end_time = datetime.utcnow()
            
            self.logger.error(
                "Job execution failed",
                execution_id=job_context.execution_id,
                error=str(e),
                traceback=traceback.format_exc()
            )
            
        finally:
            # Nettoyage des métriques
            ACTIVE_JOBS.labels(
                job_type=job_context.job_definition.job_type.value,
                tenant=job_context.job_definition.tenant_config.tenant_id
            ).dec()
            
            # Enregistrement des métriques finales
            if job_context.start_time and job_context.end_time:
                duration = (job_context.end_time - job_context.start_time).total_seconds()
                JOB_DURATION.labels(
                    job_type=job_context.job_definition.job_type.value,
                    tenant=job_context.job_definition.tenant_config.tenant_id
                ).observe(duration)
            
            JOB_EXECUTIONS.labels(
                job_type=job_context.job_definition.job_type.value,
                tenant=job_context.job_definition.tenant_config.tenant_id,
                status=job_context.status.value
            ).inc()
            
            # Sauvegarde finale du state
            await self._save_job_context(job_context)

    async def _generate_kubernetes_manifest(self, job_context: JobExecutionContext) -> Dict[str, Any]:
        """Génération du manifest Kubernetes optimisé"""
        job_def = job_context.job_definition
        
        # Nom unique du job Kubernetes
        k8s_job_name = f"{job_def.job_name}-{job_context.execution_id[:8]}"
        job_context.kubernetes_job_name = k8s_job_name
        
        # Sélection du template selon le type de job
        template_path = self._get_job_template_path(job_def.job_type)
        
        # Chargement et personnalisation du template
        manifest = await self._load_and_customize_template(template_path, job_context)
        
        return manifest

    def _get_job_template_path(self, job_type: JobType) -> Path:
        """Récupération du chemin du template selon le type de job"""
        base_path = Path(__file__).parent / "manifests" / "jobs"
        
        template_mapping = {
            JobType.ML_TRAINING: "ml-training-job.yaml",
            JobType.DATA_ETL: "data-etl-job.yaml",
            JobType.SECURITY_SCAN: "security-scan-job.yaml",
            JobType.BILLING_REPORT: "billing-reporting-job.yaml",
            JobType.TENANT_BACKUP: "tenant-backup-job.yaml"
        }
        
        template_file = template_mapping.get(job_type, "generic-job.yaml")
        return base_path / template_file

    async def _load_and_customize_template(self, template_path: Path, job_context: JobExecutionContext) -> Dict[str, Any]:
        """Chargement et personnalisation du template YAML"""
        try:
            async with aiofiles.open(template_path, 'r') as f:
                template_content = await f.read()
            
            # Variables de substitution
            variables = self._build_template_variables(job_context)
            
            # Substitution des variables
            for key, value in variables.items():
                template_content = template_content.replace(f"{{{{ .Values.{key} }}}}", str(value))
            
            # Parse YAML
            manifest = yaml.safe_load(template_content)
            
            # Personnalisations avancées
            await self._apply_advanced_customizations(manifest, job_context)
            
            return manifest
            
        except Exception as e:
            self.logger.error("Failed to load template", template_path=str(template_path), error=str(e))
            raise

    def _build_template_variables(self, job_context: JobExecutionContext) -> Dict[str, Any]:
        """Construction des variables pour substitution dans les templates"""
        job_def = job_context.job_definition
        
        return {
            "job.id": job_def.job_id,
            "job.name": job_def.job_name,
            "job.type": job_def.job_type.value,
            "execution.id": job_context.execution_id,
            "tenant.id": job_def.tenant_config.tenant_id,
            "tenant.namespace": job_def.tenant_config.namespace,
            "tenant.tier": job_def.tenant_config.tenant_tier.value,
            "resources.cpu.request": job_def.resource_limits.cpu_request,
            "resources.cpu.limit": job_def.resource_limits.cpu_limit,
            "resources.memory.request": job_def.resource_limits.memory_request,
            "resources.memory.limit": job_def.resource_limits.memory_limit,
            "resources.gpu.count": job_def.resource_limits.gpu_count,
            "security.user": job_def.security_context.run_as_user,
            "security.group": job_def.security_context.run_as_group,
            "backup.id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _apply_advanced_customizations(self, manifest: Dict[str, Any], job_context: JobExecutionContext):
        """Application de personnalisations avancées au manifest"""
        job_def = job_context.job_definition
        
        # Injection des labels et annotations
        if "metadata" not in manifest:
            manifest["metadata"] = {}
        
        # Labels enterprise
        enterprise_labels = {
            "app": "spotify-ai-agent",
            "job-id": job_def.job_id,
            "execution-id": job_context.execution_id,
            "tenant-id": job_def.tenant_config.tenant_id,
            "job-type": job_def.job_type.value,
            "priority": job_def.priority.name.lower(),
            "tier": job_def.tenant_config.tenant_tier.value,
            "managed-by": "spotify-ai-job-manager",
            "version": "7.2.1"
        }
        
        manifest["metadata"].setdefault("labels", {}).update(enterprise_labels)
        
        # Annotations de conformité
        compliance_annotations = {
            "compliance.spotify-ai/pci-dss": str(job_def.compliance_config.pci_dss_level),
            "compliance.spotify-ai/sox": str(job_def.compliance_config.sox_compliance),
            "compliance.spotify-ai/gdpr": str(job_def.compliance_config.gdpr_compliance),
            "compliance.spotify-ai/hipaa": str(job_def.compliance_config.hipaa_compliance),
            "compliance.spotify-ai/iso27001": str(job_def.compliance_config.iso27001_compliance),
            "monitoring.spotify-ai/prometheus": str(job_def.monitoring_config.prometheus_enabled),
            "monitoring.spotify-ai/jaeger": str(job_def.monitoring_config.jaeger_enabled),
            "security.spotify-ai/encryption-at-rest": str(job_def.compliance_config.encryption_at_rest),
            "security.spotify-ai/encryption-in-transit": str(job_def.compliance_config.encryption_in_transit)
        }
        
        manifest["metadata"].setdefault("annotations", {}).update(compliance_annotations)

    async def _deploy_kubernetes_job(self, job_context: JobExecutionContext, manifest: Dict[str, Any]):
        """Déploiement du job sur Kubernetes avec gestion d'erreurs avancée"""
        try:
            namespace = job_context.job_definition.tenant_config.namespace
            
            # Déploiement via l'API Kubernetes
            response = await asyncio.get_event_loop().run_in_executor(
                self.worker_pool,
                self.k8s_client.create_namespaced_job,
                namespace,
                client.V1Job(**manifest)
            )
            
            self.logger.info(
                "Kubernetes job deployed",
                job_name=response.metadata.name,
                namespace=namespace,
                execution_id=job_context.execution_id
            )
            
        except ApiException as e:
            error_msg = f"Kubernetes API error: {e.status} - {e.reason}"
            self.logger.error("Failed to deploy job", error=error_msg)
            raise Exception(error_msg)
        except Exception as e:
            self.logger.error("Unexpected error during deployment", error=str(e))
            raise

    async def _monitor_job_execution(self, job_context: JobExecutionContext):
        """Monitoring en temps réel de l'exécution du job"""
        namespace = job_context.job_definition.tenant_config.namespace
        job_name = job_context.kubernetes_job_name
        timeout = job_context.job_definition.active_deadline_seconds
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Récupération du status du job
                job_status = await asyncio.get_event_loop().run_in_executor(
                    self.worker_pool,
                    self.k8s_client.read_namespaced_job_status,
                    job_name,
                    namespace
                )
                
                # Analyse du status
                if job_status.status.succeeded:
                    job_context.status = JobStatus.SUCCEEDED
                    job_context.end_time = datetime.utcnow()
                    break
                elif job_status.status.failed:
                    job_context.status = JobStatus.FAILED
                    job_context.end_time = datetime.utcnow()
                    job_context.error_message = "Kubernetes job failed"
                    break
                
                # Collecte des métriques de ressources
                await self._collect_resource_metrics(job_context, namespace, job_name)
                
                # Attente avant la prochaine vérification
                await asyncio.sleep(30)
                
            except ApiException as e:
                if e.status == 404:
                    job_context.status = JobStatus.FAILED
                    job_context.error_message = "Job not found in Kubernetes"
                    break
                else:
                    self.logger.warning("Monitoring error", error=str(e))
                    await asyncio.sleep(10)
            except Exception as e:
                self.logger.error("Unexpected monitoring error", error=str(e))
                await asyncio.sleep(10)
        
        # Timeout check
        if job_context.status == JobStatus.RUNNING:
            job_context.status = JobStatus.TIMEOUT
            job_context.end_time = datetime.utcnow()
            job_context.error_message = f"Job timeout after {timeout} seconds"

    async def _collect_resource_metrics(self, job_context: JobExecutionContext, namespace: str, job_name: str):
        """Collecte des métriques de ressources pour le job"""
        try:
            # Récupération des pods du job
            pods = await asyncio.get_event_loop().run_in_executor(
                self.worker_pool,
                self.k8s_core_client.list_namespaced_pod,
                namespace,
                label_selector=f"job-name={job_name}"
            )
            
            total_cpu = 0
            total_memory = 0
            
            for pod in pods.items:
                if pod.status.phase == "Running":
                    # Simulation de collecte de métriques (normalement via metrics-server)
                    total_cpu += 0.5  # CPU units
                    total_memory += 256  # MB
            
            # Mise à jour des métriques
            job_context.resource_usage.update({
                "cpu_usage": total_cpu,
                "memory_usage": total_memory,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Export vers Prometheus
            RESOURCE_USAGE.labels(
                job_type=job_context.job_definition.job_type.value,
                tenant=job_context.job_definition.tenant_config.tenant_id,
                resource="cpu"
            ).set(total_cpu)
            
            RESOURCE_USAGE.labels(
                job_type=job_context.job_definition.job_type.value,
                tenant=job_context.job_definition.tenant_config.tenant_id,
                resource="memory"
            ).set(total_memory)
            
        except Exception as e:
            self.logger.warning("Failed to collect resource metrics", error=str(e))

    async def _save_job_context(self, job_context: JobExecutionContext):
        """Sauvegarde du contexte du job dans Redis"""
        try:
            context_data = {
                "job_id": job_context.job_definition.job_id,
                "execution_id": job_context.execution_id,
                "status": job_context.status.value,
                "start_time": job_context.start_time.isoformat() if job_context.start_time else None,
                "end_time": job_context.end_time.isoformat() if job_context.end_time else None,
                "kubernetes_job_name": job_context.kubernetes_job_name,
                "error_message": job_context.error_message,
                "resource_usage": job_context.resource_usage,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(
                f"job_context:{job_context.execution_id}",
                86400,  # 24 heures TTL
                json.dumps(context_data)
            )
            
        except Exception as e:
            self.logger.error("Failed to save job context", error=str(e))

    def _calculate_duration(self, job_context: JobExecutionContext) -> Optional[float]:
        """Calcul de la durée d'exécution du job"""
        if job_context.start_time and job_context.end_time:
            return (job_context.end_time - job_context.start_time).total_seconds()
        return None

    async def get_job_status(self, execution_id: str) -> Optional[JobExecutionContext]:
        """Récupération du status d'un job par son execution_id"""
        try:
            context_data = await self.redis_client.get(f"job_context:{execution_id}")
            if context_data:
                data = json.loads(context_data)
                # Reconstruction simplifiée du contexte (pour status seulement)
                return {
                    "execution_id": data["execution_id"],
                    "status": data["status"],
                    "start_time": data["start_time"],
                    "end_time": data["end_time"],
                    "error_message": data["error_message"],
                    "resource_usage": data["resource_usage"]
                }
            return None
        except Exception as e:
            self.logger.error("Failed to get job status", execution_id=execution_id, error=str(e))
            return None

    async def cancel_job(self, execution_id: str) -> bool:
        """Annulation d'un job en cours d'exécution"""
        try:
            # Récupération du contexte
            job_context = await self.get_job_status(execution_id)
            if not job_context:
                return False
            
            if job_context["status"] not in ["running", "pending"]:
                return False
            
            # Suppression du job Kubernetes si déployé
            kubernetes_job_name = job_context.get("kubernetes_job_name")
            if kubernetes_job_name:
                # Logic pour supprimer le job Kubernetes
                pass
            
            # Mise à jour du status
            job_context["status"] = "cancelled"
            job_context["end_time"] = datetime.utcnow().isoformat()
            
            # Sauvegarde
            await self.redis_client.setex(
                f"job_context:{execution_id}",
                86400,
                json.dumps(job_context)
            )
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to cancel job", execution_id=execution_id, error=str(e))
            return False

    async def list_jobs(self, tenant_id: Optional[str] = None, status: Optional[JobStatus] = None) -> List[Dict[str, Any]]:
        """Liste des jobs avec filtres optionnels"""
        try:
            # Scan des clés Redis pour les contextes de jobs
            keys = []
            async for key in self.redis_client.scan_iter(match="job_context:*"):
                keys.append(key)
            
            jobs = []
            for key in keys:
                context_data = await self.redis_client.get(key)
                if context_data:
                    job_data = json.loads(context_data)
                    
                    # Application des filtres
                    if tenant_id and job_data.get("tenant_id") != tenant_id:
                        continue
                    if status and job_data.get("status") != status.value:
                        continue
                    
                    jobs.append(job_data)
            
            return jobs
            
        except Exception as e:
            self.logger.error("Failed to list jobs", error=str(e))
            return []

    async def cleanup_completed_jobs(self, older_than_hours: int = 24):
        """Nettoyage des jobs terminés anciens"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            
            # Scan et suppression des anciens jobs
            deleted_count = 0
            async for key in self.redis_client.scan_iter(match="job_context:*"):
                context_data = await self.redis_client.get(key)
                if context_data:
                    job_data = json.loads(context_data)
                    end_time_str = job_data.get("end_time")
                    
                    if end_time_str:
                        end_time = datetime.fromisoformat(end_time_str)
                        if end_time < cutoff_time:
                            await self.redis_client.delete(key)
                            deleted_count += 1
            
            self.logger.info("Cleanup completed", deleted_jobs=deleted_count)
            return deleted_count
            
        except Exception as e:
            self.logger.error("Failed to cleanup jobs", error=str(e))
            return 0

    async def shutdown(self):
        """Arrêt propre du gestionnaire de jobs"""
        self.logger.info("Shutting down job manager...")
        
        # Signal d'arrêt
        self.shutdown_event.set()
        
        # Attente de la fin des jobs en cours
        await self.job_queue.join()
        
        # Fermeture des connexions
        if self.redis_client:
            await self.redis_client.close()
        
        # Arrêt du worker pool
        self.worker_pool.shutdown(wait=True)
        
        self.logger.info("Job manager shutdown completed")


class SpotifyAIJobManager:
    """Gestionnaire principal des jobs Spotify AI Agent - Interface utilisateur"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = structlog.get_logger()
        self.config_path = config_path
        self.kubernetes_manager: Optional[KubernetesJobManager] = None
        
    async def initialize(self):
        """Initialisation du gestionnaire principal"""
        try:
            self.kubernetes_manager = KubernetesJobManager()
            await self.kubernetes_manager.initialize()
            self.logger.info("Spotify AI Job Manager initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize job manager", error=str(e))
            raise

    async def create_ml_training_job(
        self,
        tenant_id: str,
        model_name: str,
        dataset_path: str,
        gpu_count: int = 1,
        priority: Priority = Priority.NORMAL
    ) -> str:
        """Création d'un job de formation ML"""
        
        # Configuration du tenant
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            tenant_name=f"tenant-{tenant_id}",
            tenant_tier=TenantTier.ENTERPRISE,
            namespace=f"spotify-ai-{tenant_id}",
            resource_quota={"cpu": "10", "memory": "20Gi", "nvidia.com/gpu": str(gpu_count)}
        )
        
        # Configuration des ressources
        resource_limits = ResourceLimits(
            cpu_request="2000m",
            cpu_limit="4000m",
            memory_request="8Gi",
            memory_limit="16Gi",
            gpu_count=gpu_count
        )
        
        # Définition du job
        job_def = JobDefinition(
            job_id=str(uuid.uuid4()),
            job_name=f"ml-training-{model_name}",
            job_type=JobType.ML_TRAINING,
            tenant_config=tenant_config,
            priority=priority,
            resource_limits=resource_limits,
            environment_variables={
                "MODEL_NAME": model_name,
                "DATASET_PATH": dataset_path,
                "GPU_COUNT": str(gpu_count),
                "DISTRIBUTED_TRAINING": "true" if gpu_count > 1 else "false"
            }
        )
        
        return await self.kubernetes_manager.submit_job(job_def)

    async def create_data_etl_job(
        self,
        tenant_id: str,
        source_config: Dict[str, Any],
        destination_config: Dict[str, Any],
        transformation_script: str,
        priority: Priority = Priority.NORMAL
    ) -> str:
        """Création d'un job ETL de données"""
        
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            tenant_name=f"tenant-{tenant_id}",
            tenant_tier=TenantTier.ENTERPRISE,
            namespace=f"spotify-ai-{tenant_id}",
            resource_quota={"cpu": "8", "memory": "16Gi"}
        )
        
        resource_limits = ResourceLimits(
            cpu_request="1500m",
            cpu_limit="3000m",
            memory_request="4Gi",
            memory_limit="8Gi"
        )
        
        job_def = JobDefinition(
            job_id=str(uuid.uuid4()),
            job_name=f"data-etl-{tenant_id}",
            job_type=JobType.DATA_ETL,
            tenant_config=tenant_config,
            priority=priority,
            resource_limits=resource_limits,
            environment_variables={
                "SOURCE_CONFIG": json.dumps(source_config),
                "DESTINATION_CONFIG": json.dumps(destination_config),
                "TRANSFORMATION_SCRIPT": transformation_script,
                "PARALLEL_STREAMS": "4"
            }
        )
        
        return await self.kubernetes_manager.submit_job(job_def)

    async def create_security_scan_job(
        self,
        tenant_id: str,
        scan_targets: List[str],
        compliance_frameworks: List[str],
        priority: Priority = Priority.HIGH
    ) -> str:
        """Création d'un job de scan de sécurité"""
        
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            tenant_name=f"tenant-{tenant_id}",
            tenant_tier=TenantTier.ENTERPRISE,
            namespace=f"spotify-ai-{tenant_id}",
            resource_quota={"cpu": "6", "memory": "12Gi"}
        )
        
        compliance_config = ComplianceConfig(
            pci_dss_level="level_1",
            sox_compliance=True,
            gdpr_compliance=True,
            hipaa_compliance=True,
            iso27001_compliance=True
        )
        
        job_def = JobDefinition(
            job_id=str(uuid.uuid4()),
            job_name=f"security-scan-{tenant_id}",
            job_type=JobType.SECURITY_SCAN,
            tenant_config=tenant_config,
            priority=priority,
            compliance_config=compliance_config,
            environment_variables={
                "SCAN_TARGETS": json.dumps(scan_targets),
                "COMPLIANCE_FRAMEWORKS": json.dumps(compliance_frameworks),
                "SCAN_DEPTH": "comprehensive"
            }
        )
        
        return await self.kubernetes_manager.submit_job(job_def)

    async def create_billing_report_job(
        self,
        tenant_id: str,
        report_period: str,
        report_type: str,
        currency: str = "USD",
        priority: Priority = Priority.NORMAL
    ) -> str:
        """Création d'un job de rapport de facturation"""
        
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            tenant_name=f"tenant-{tenant_id}",
            tenant_tier=TenantTier.ENTERPRISE,
            namespace=f"spotify-ai-{tenant_id}",
            resource_quota={"cpu": "4", "memory": "8Gi"}
        )
        
        job_def = JobDefinition(
            job_id=str(uuid.uuid4()),
            job_name=f"billing-report-{tenant_id}",
            job_type=JobType.BILLING_REPORT,
            tenant_config=tenant_config,
            priority=priority,
            environment_variables={
                "REPORT_PERIOD": report_period,
                "REPORT_TYPE": report_type,
                "CURRENCY": currency,
                "COMPLIANCE_LEVEL": "enterprise"
            }
        )
        
        return await self.kubernetes_manager.submit_job(job_def)

    async def create_tenant_backup_job(
        self,
        tenant_id: str,
        backup_type: str = "full",
        retention_days: int = 30,
        priority: Priority = Priority.HIGH
    ) -> str:
        """Création d'un job de sauvegarde tenant"""
        
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            tenant_name=f"tenant-{tenant_id}",
            tenant_tier=TenantTier.ENTERPRISE,
            namespace=f"spotify-ai-{tenant_id}",
            resource_quota={"cpu": "6", "memory": "12Gi", "storage": "1Ti"}
        )
        
        job_def = JobDefinition(
            job_id=str(uuid.uuid4()),
            job_name=f"tenant-backup-{tenant_id}",
            job_type=JobType.TENANT_BACKUP,
            tenant_config=tenant_config,
            priority=priority,
            environment_variables={
                "BACKUP_TYPE": backup_type,
                "RETENTION_DAYS": str(retention_days),
                "ENCRYPTION_ENABLED": "true",
                "COMPRESSION_ENABLED": "true"
            }
        )
        
        return await self.kubernetes_manager.submit_job(job_def)

    # Méthodes de gestion et monitoring
    async def get_job_status(self, execution_id: str):
        """Récupération du status d'un job"""
        return await self.kubernetes_manager.get_job_status(execution_id)

    async def cancel_job(self, execution_id: str) -> bool:
        """Annulation d'un job"""
        return await self.kubernetes_manager.cancel_job(execution_id)

    async def list_jobs(self, tenant_id: Optional[str] = None, status: Optional[JobStatus] = None):
        """Liste des jobs avec filtres"""
        return await self.kubernetes_manager.list_jobs(tenant_id, status)

    async def cleanup_old_jobs(self, older_than_hours: int = 24):
        """Nettoyage des anciens jobs"""
        return await self.kubernetes_manager.cleanup_completed_jobs(older_than_hours)

    async def shutdown(self):
        """Arrêt propre du gestionnaire"""
        if self.kubernetes_manager:
            await self.kubernetes_manager.shutdown()


# Configuration du signal handler pour arrêt propre
def signal_handler(signum, frame):
    """Handler pour arrêt propre sur signal système"""
    logger.info("Received shutdown signal", signal=signum)
    # Logic d'arrêt ici
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Interface CLI pour tests et administration
async def main():
    """Fonction principale pour tests CLI"""
    if len(sys.argv) < 2:
        print("Usage: python job_manager.py <command> [args...]")
        print("Commands: create-ml, create-etl, create-security, create-billing, create-backup, status, list, cleanup")
        return
    
    command = sys.argv[1]
    
    # Initialisation du gestionnaire
    job_manager = SpotifyAIJobManager()
    await job_manager.initialize()
    
    try:
        if command == "create-ml":
            tenant_id = sys.argv[2] if len(sys.argv) > 2 else "demo-tenant"
            model_name = sys.argv[3] if len(sys.argv) > 3 else "demo-model"
            execution_id = await job_manager.create_ml_training_job(
                tenant_id=tenant_id,
                model_name=model_name,
                dataset_path="/data/training",
                gpu_count=1
            )
            print(f"ML Training job created: {execution_id}")
            
        elif command == "create-etl":
            tenant_id = sys.argv[2] if len(sys.argv) > 2 else "demo-tenant"
            execution_id = await job_manager.create_data_etl_job(
                tenant_id=tenant_id,
                source_config={"type": "kafka", "topic": "raw-data"},
                destination_config={"type": "s3", "bucket": "processed-data"},
                transformation_script="transform.py"
            )
            print(f"Data ETL job created: {execution_id}")
            
        elif command == "status":
            execution_id = sys.argv[2] if len(sys.argv) > 2 else ""
            if execution_id:
                status = await job_manager.get_job_status(execution_id)
                print(f"Job Status: {json.dumps(status, indent=2)}")
            else:
                print("Execution ID required")
                
        elif command == "list":
            jobs = await job_manager.list_jobs()
            print(f"Active Jobs: {json.dumps(jobs, indent=2)}")
            
        elif command == "cleanup":
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            deleted = await job_manager.cleanup_old_jobs(hours)
            print(f"Cleaned up {deleted} old jobs")
            
        else:
            print(f"Unknown command: {command}")
            
    finally:
        await job_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())


# Exportation des classes principales pour utilisation en tant que module
__all__ = [
    'SpotifyAIJobManager',
    'KubernetesJobManager',
    'JobDefinition',
    'JobExecutionContext',
    'JobType',
    'JobStatus',
    'Priority',
    'TenantTier',
    'ResourceLimits',
    'SecurityContext',
    'ComplianceConfig',
    'MonitoringConfig',
    'TenantConfig'
]

# Métadonnées du module
__version__ = "7.2.1"
__author__ = "Fahed Mlaiel"
__email__ = "fahed.mlaiel@spotify-ai-agent.com"
__license__ = "Proprietary - Spotify AI Agent Platform"
__description__ = "Système de gestion de jobs enterprise ultra-avancé pour l'agent IA Spotify"
