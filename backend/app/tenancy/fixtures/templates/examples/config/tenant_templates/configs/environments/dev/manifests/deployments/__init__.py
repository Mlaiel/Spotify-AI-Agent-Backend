"""
Deployments Module - Advanced Kubernetes Application Deployments
===============================================================

Module ultra-avancé de gestion des déploiements d'applications pour l'environnement
de développement du système multi-tenant Spotify AI Agent.

Features:
- Advanced deployment strategies (Blue-Green, Canary, Rolling)
- Multi-tenant resource isolation and scaling
- AI/ML model serving deployments
- Observability and monitoring integration
- Security hardening and compliance
- Auto-scaling and resource optimization
- Chaos engineering integration
- Service mesh ready configurations

Architectures supportées:
- Microservices distribuées
- Event-driven architecture
- CQRS/Event Sourcing
- AI/ML pipeline deployments
- Real-time streaming services
"""

from typing import Dict, List, Optional, Any, Union, Callable
import yaml
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid

from ...__init__ import ManifestGenerator, DEFAULT_LABELS

__version__ = "2.0.0"
__author__ = "Fahed Mlaiel - Lead Architect & AI Specialist"
__maintainer__ = "Fahed Mlaiel Development Team"
__email__ = "fahed.mlaiel@spotify-ai-agent.com"
__status__ = "Production Ready"

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    """Stratégies de déploiement supportées."""
    ROLLING_UPDATE = "RollingUpdate"
    RECREATE = "Recreate"
    BLUE_GREEN = "BlueGreen"
    CANARY = "Canary"
    A_B_TESTING = "ABTesting"

class TenantTier(Enum):
    """Niveaux de tenant supportés."""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"

class ServiceType(Enum):
    """Types de services déployables."""
    BACKEND_API = "backend-api"
    ML_SERVICE = "ml-service"
    STREAMING_SERVICE = "streaming-service"
    ANALYTICS_SERVICE = "analytics-service"
    NOTIFICATION_SERVICE = "notification-service"
    AUTH_SERVICE = "auth-service"
    TENANT_SERVICE = "tenant-service"
    BILLING_SERVICE = "billing-service"

@dataclass
class DeploymentConfig:
    """Configuration avancée pour les déploiements."""
    name: str
    service_type: ServiceType
    tenant_tier: TenantTier
    replicas: int = 3
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    enable_hpa: bool = True
    enable_pdb: bool = True
    enable_network_policies: bool = True
    enable_security_context: bool = True
    enable_monitoring: bool = True
    enable_tracing: bool = True
    enable_chaos_engineering: bool = False
    custom_annotations: Dict[str, str] = field(default_factory=dict)
    custom_labels: Dict[str, str] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    config_maps: List[str] = field(default_factory=list)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    metrics_port: int = 9000
    service_port: int = 8000

class AdvancedDeploymentManager:
    """Gestionnaire avancé des déploiements Kubernetes multi-tenant."""
    
    def __init__(self, namespace: str = "spotify-ai-agent-dev"):
        self.namespace = namespace
        self.manifest_generator = ManifestGenerator(namespace)
        self.deployment_id = str(uuid.uuid4())[:8]
        self.created_at = datetime.utcnow()
        
        # Configuration avancée
        self.resource_quotas = self._load_resource_quotas()
        self.security_policies = self._load_security_policies()
        self.monitoring_config = self._load_monitoring_config()
        
        logger.info(f"AdvancedDeploymentManager initialized for namespace: {namespace}")

    def _load_resource_quotas(self) -> Dict[TenantTier, Dict[str, str]]:
        """Charge les quotas de ressources par niveau de tenant."""
        return {
            TenantTier.FREE: {
                "cpu": "200m",
                "memory": "256Mi",
                "storage": "1Gi",
                "pods": "2",
                "replicas": "1"
            },
            TenantTier.PREMIUM: {
                "cpu": "1000m",
                "memory": "2Gi",
                "storage": "10Gi",
                "pods": "10",
                "replicas": "3"
            },
            TenantTier.ENTERPRISE: {
                "cpu": "4000m",
                "memory": "8Gi",
                "storage": "100Gi",
                "pods": "50",
                "replicas": "5"
            },
            TenantTier.ENTERPRISE_PLUS: {
                "cpu": "16000m",
                "memory": "32Gi",
                "storage": "1Ti",
                "pods": "200",
                "replicas": "10"
            }
        }

    def _load_security_policies(self) -> Dict[str, Any]:
        """Charge les politiques de sécurité avancées."""
        return {
            "pod_security_standards": "restricted",
            "network_policies": True,
            "rbac_enabled": True,
            "admission_controllers": ["PodSecurity", "ResourceQuota", "LimitRanger"],
            "image_scanning": True,
            "runtime_security": True,
            "encryption_at_rest": True,
            "encryption_in_transit": True
        }

    def _load_monitoring_config(self) -> Dict[str, Any]:
        """Charge la configuration de monitoring avancée."""
        return {
            "prometheus": {
                "enabled": True,
                "scrape_interval": "15s",
                "retention": "30d"
            },
            "grafana": {
                "enabled": True,
                "dashboards": ["deployment", "performance", "security"]
            },
            "jaeger": {
                "enabled": True,
                "sampling_rate": 0.1
            },
            "elk": {
                "enabled": True,
                "retention": "7d"
            },
            "alertmanager": {
                "enabled": True,
                "slack_webhook": True,
                "email_alerts": True
            }
        }

    def create_advanced_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Crée un déploiement avancé avec toutes les fonctionnalités enterprise."""
        base_labels = {
            **DEFAULT_LABELS,
            "app": config.name,
            "service-type": config.service_type.value,
            "tenant-tier": config.tenant_tier.value,
            "deployment-id": self.deployment_id,
            "version": "v2.0.0",
            **config.custom_labels
        }
        
        base_annotations = {
            "deployment.kubernetes.io/revision": "1",
            "created-by": "Fahed Mlaiel - Lead Architect",
            "deployment-strategy": config.strategy.value,
            "tenant-tier": config.tenant_tier.value,
            "monitoring-enabled": str(config.enable_monitoring),
            "security-hardened": str(config.enable_security_context),
            "chaos-engineering": str(config.enable_chaos_engineering),
            **config.custom_annotations
        }

        # Génération du déploiement principal
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": config.name,
                "namespace": self.namespace,
                "labels": base_labels,
                "annotations": base_annotations
            },
            "spec": {
                "replicas": config.replicas,
                "strategy": self._get_deployment_strategy(config.strategy),
                "selector": {
                    "matchLabels": {
                        "app": config.name,
                        "environment": "development"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": base_labels,
                        "annotations": self._get_pod_annotations(config)
                    },
                    "spec": self._get_pod_spec(config)
                }
            }
        }
        
        return deployment

    def _get_deployment_strategy(self, strategy: DeploymentStrategy) -> Dict[str, Any]:
        """Génère la stratégie de déploiement selon le type."""
        strategies = {
            DeploymentStrategy.ROLLING_UPDATE: {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxUnavailable": "25%",
                    "maxSurge": "25%"
                }
            },
            DeploymentStrategy.RECREATE: {
                "type": "Recreate"
            },
            DeploymentStrategy.BLUE_GREEN: {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxUnavailable": 0,
                    "maxSurge": "100%"
                }
            },
            DeploymentStrategy.CANARY: {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxUnavailable": 0,
                    "maxSurge": "10%"
                }
            }
        }
        return strategies.get(strategy, strategies[DeploymentStrategy.ROLLING_UPDATE])

    def _get_pod_annotations(self, config: DeploymentConfig) -> Dict[str, str]:
        """Génère les annotations du pod."""
        annotations = {
            "created-at": self.created_at.isoformat(),
            "config-hash": self._generate_config_hash(config),
        }
        
        if config.enable_monitoring:
            annotations.update({
                "prometheus.io/scrape": "true",
                "prometheus.io/port": str(config.metrics_port),
                "prometheus.io/path": "/metrics"
            })
        
        if config.enable_tracing:
            annotations.update({
                "sidecar.jaegertracing.io/inject": "true",
                "jaeger.io/sampling-type": "probabilistic",
                "jaeger.io/sampling-param": "0.1"
            })
            
        return annotations

    def _get_pod_spec(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Génère la spécification complète du pod."""
        pod_spec = {
            "serviceAccountName": f"{config.name}-sa",
            "automountServiceAccountToken": True,
            "terminationGracePeriodSeconds": 30,
            "restartPolicy": "Always",
            "containers": [self._get_main_container(config)]
        }
        
        # Contexte de sécurité avancé
        if config.enable_security_context:
            pod_spec["securityContext"] = self._get_security_context()
        
        # Volumes personnalisés
        if config.volumes:
            pod_spec["volumes"] = config.volumes
            
        # Sidecar containers pour monitoring et tracing
        sidecars = []
        
        if config.enable_monitoring:
            sidecars.append(self._get_monitoring_sidecar())
            
        if config.enable_tracing:
            sidecars.append(self._get_tracing_sidecar())
            
        if sidecars:
            pod_spec["containers"].extend(sidecars)
            
        # Node affinity pour l'optimisation des performances
        pod_spec["affinity"] = self._get_affinity_rules(config)
        
        # Toleration pour les nœuds spécialisés
        pod_spec["tolerations"] = self._get_tolerations(config)
        
        return pod_spec

    def _get_main_container(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Génère le container principal de l'application."""
        container = {
            "name": config.service_type.value,
            "image": f"spotify-ai-agent/{config.service_type.value}:dev-latest",
            "imagePullPolicy": "Always",
            "ports": [
                {
                    "containerPort": config.service_port,
                    "name": "http",
                    "protocol": "TCP"
                },
                {
                    "containerPort": config.metrics_port,
                    "name": "metrics",
                    "protocol": "TCP"
                }
            ],
            "env": self._get_environment_variables(config),
            "resources": self._get_resource_requirements(config),
            "livenessProbe": self._get_liveness_probe(config),
            "readinessProbe": self._get_readiness_probe(config),
            "startupProbe": self._get_startup_probe(config),
            "lifecycle": self._get_lifecycle_hooks(config)
        }
        
        # Contexte de sécurité du container
        if config.enable_security_context:
            container["securityContext"] = {
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": True,
                "runAsNonRoot": True,
                "runAsUser": 10001,
                "capabilities": {
                    "drop": ["ALL"],
                    "add": ["NET_BIND_SERVICE"]
                }
            }
            
        # Volume mounts
        if config.volumes:
            container["volumeMounts"] = [
                {"name": vol["name"], "mountPath": vol.get("mountPath", f"/mnt/{vol['name']}")}
                for vol in config.volumes
            ]
            
        return container

    def _get_environment_variables(self, config: DeploymentConfig) -> List[Dict[str, Any]]:
        """Génère les variables d'environnement avancées."""
        base_env = [
            {"name": "ENVIRONMENT", "value": "development"},
            {"name": "SERVICE_NAME", "value": config.name},
            {"name": "SERVICE_TYPE", "value": config.service_type.value},
            {"name": "TENANT_TIER", "value": config.tenant_tier.value},
            {"name": "DEPLOYMENT_ID", "value": self.deployment_id},
            {"name": "NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}},
            {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
            {"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
            {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
            {"name": "CPU_REQUEST", "valueFrom": {"resourceFieldRef": {"resource": "requests.cpu"}}},
            {"name": "MEMORY_REQUEST", "valueFrom": {"resourceFieldRef": {"resource": "requests.memory"}}},
        ]
        
        # Variables personnalisées
        for key, value in config.env_vars.items():
            base_env.append({"name": key, "value": value})
            
        # Secrets
        for secret in config.secrets:
            base_env.append({
                "name": f"{secret.upper()}_SECRET",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": secret,
                        "key": "value"
                    }
                }
            })
            
        # ConfigMaps
        for configmap in config.config_maps:
            base_env.append({
                "name": f"{configmap.upper()}_CONFIG",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": configmap,
                        "key": "value"
                    }
                }
            })
            
        return base_env

    def _generate_config_hash(self, config: DeploymentConfig) -> str:
        """Génère un hash de la configuration pour le rolling update."""
        config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def create_ml_service_deployment(self, tenant_tier: TenantTier = TenantTier.PREMIUM) -> Dict[str, Any]:
        """Crée un déploiement spécialisé pour les services ML/AI."""
        config = DeploymentConfig(
            name="spotify-ai-ml-service",
            service_type=ServiceType.ML_SERVICE,
            tenant_tier=tenant_tier,
            replicas=self.resource_quotas[tenant_tier]["replicas"],
            cpu_request="500m",
            cpu_limit="2000m",
            memory_request="1Gi",
            memory_limit="4Gi",
            enable_hpa=True,
            enable_monitoring=True,
            enable_tracing=True,
            custom_annotations={
                "ml-framework": "pytorch-tensorflow",
                "gpu-required": "false",
                "model-serving": "true"
            },
            env_vars={
                "ML_MODEL_PATH": "/models",
                "INFERENCE_BATCH_SIZE": "32",
                "MODEL_CACHE_SIZE": "1000"
            }
        )
        
        return self.create_advanced_deployment(config)

    def create_streaming_service_deployment(self, tenant_tier: TenantTier = TenantTier.ENTERPRISE) -> Dict[str, Any]:
        """Crée un déploiement pour les services de streaming en temps réel."""
        config = DeploymentConfig(
            name="spotify-ai-streaming-service",
            service_type=ServiceType.STREAMING_SERVICE,
            tenant_tier=tenant_tier,
            replicas=int(self.resource_quotas[tenant_tier]["replicas"]) * 2,  # Plus de répliques pour le streaming
            cpu_request="1000m",
            cpu_limit="4000m",
            memory_request="2Gi",
            memory_limit="8Gi",
            enable_hpa=True,
            enable_monitoring=True,
            enable_tracing=True,
            custom_annotations={
                "streaming-protocol": "websocket-sse",
                "max-connections": "10000",
                "buffer-size": "1MB"
            },
            env_vars={
                "STREAMING_BUFFER_SIZE": "1048576",
                "MAX_CONCURRENT_STREAMS": "1000",
                "HEARTBEAT_INTERVAL": "30"
            }
        )
        
        return self.create_advanced_deployment(config)

    async def deploy_multi_tenant_stack(self, tenant_tiers: List[TenantTier]) -> List[Dict[str, Any]]:
        """Déploie une stack complète multi-tenant de manière asynchrone."""
        deployments = []
        
        async def create_tenant_deployment(tier: TenantTier) -> List[Dict[str, Any]]:
            """Crée tous les déploiements pour un niveau de tenant."""
            tenant_deployments = []
            
            # Backend API
            backend_config = DeploymentConfig(
                name=f"spotify-ai-backend-{tier.value}",
                service_type=ServiceType.BACKEND_API,
                tenant_tier=tier,
                replicas=int(self.resource_quotas[tier]["replicas"])
            )
            tenant_deployments.append(self.create_advanced_deployment(backend_config))
            
            # ML Service (Premium+)
            if tier in [TenantTier.PREMIUM, TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
                tenant_deployments.append(self.create_ml_service_deployment(tier))
            
            # Streaming Service (Enterprise+)
            if tier in [TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
                tenant_deployments.append(self.create_streaming_service_deployment(tier))
            
            return tenant_deployments
        
        # Déploiement parallèle pour tous les tiers
        tasks = [create_tenant_deployment(tier) for tier in tenant_tiers]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            deployments.extend(result)
            
        logger.info(f"Deployed {len(deployments)} advanced deployments for {len(tenant_tiers)} tenant tiers")
        return deployments

    def generate_deployment_manifest_files(self, output_dir: str = "./manifests") -> None:
        """Génère tous les fichiers de manifestes de déploiement."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Déploiements pour tous les tiers
        for tier in TenantTier:
            tier_dir = output_path / tier.value
            tier_dir.mkdir(exist_ok=True)
            
            # Backend
            backend_deployment = self.create_advanced_deployment(
                DeploymentConfig(
                    name=f"spotify-ai-backend-{tier.value}",
                    service_type=ServiceType.BACKEND_API,
                    tenant_tier=tier
                )
            )
            
            with open(tier_dir / f"backend-deployment-{tier.value}.yaml", "w") as f:
                yaml.dump(backend_deployment, f, default_flow_style=False, indent=2)
            
            # Services additionnels selon le tier
            if tier in [TenantTier.PREMIUM, TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
                ml_deployment = self.create_ml_service_deployment(tier)
                with open(tier_dir / f"ml-service-deployment-{tier.value}.yaml", "w") as f:
                    yaml.dump(ml_deployment, f, default_flow_style=False, indent=2)
            
            if tier in [TenantTier.ENTERPRISE, TenantTier.ENTERPRISE_PLUS]:
                streaming_deployment = self.create_streaming_service_deployment(tier)
                with open(tier_dir / f"streaming-service-deployment-{tier.value}.yaml", "w") as f:
                    yaml.dump(streaming_deployment, f, default_flow_style=False, indent=2)
        
        logger.info(f"Generated all deployment manifests in {output_dir}")

# Gestionnaire global des déploiements avancés
deployment_manager = AdvancedDeploymentManager()

# Déploiement rapide pour développement
def quick_dev_deployment() -> Dict[str, Any]:
    """Crée un déploiement rapide pour le développement."""
    config = DeploymentConfig(
        name="spotify-ai-quick-dev",
        service_type=ServiceType.BACKEND_API,
        tenant_tier=TenantTier.FREE,
        replicas=1,
        enable_monitoring=False,
        enable_tracing=False,
        enable_security_context=False
    )
    return deployment_manager.create_advanced_deployment(config)

class DeploymentManager:
    """Gestionnaire des déploiements Kubernetes pour les microservices (Legacy)."""
    
    def __init__(self, namespace: str = "spotify-ai-agent-dev"):
        self.namespace = namespace
        self.manifest_generator = ManifestGenerator(namespace)
        logger.warning("Using legacy DeploymentManager. Consider upgrading to AdvancedDeploymentManager")
    
    def create_main_app_deployment(self) -> Dict[str, Any]:
        """Crée le déploiement principal de l'application Spotify AI Agent."""
        return {
            **self.manifest_generator.generate_base_manifest(
                "Deployment", 
                "spotify-ai-agent-backend",
                {"app.kubernetes.io/component": "backend-api"}
            ),
            "spec": {
                "replicas": 3,
                "selector": {
                    "matchLabels": {
                        "app": "spotify-ai-agent-backend",
                        "environment": "development"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            **DEFAULT_LABELS,
                            "app": "spotify-ai-agent-backend",
                            "version": "v1.0.0"
                        }
                    },
                    "spec": {
                        "serviceAccountName": "spotify-ai-agent-backend",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 2000
                        },
                        "containers": [{
                            "name": "backend",
                            "image": "spotify-ai-agent/backend:dev-latest",
                            "ports": [
                                {"containerPort": 8000, "name": "http"},
                                {"containerPort": 9000, "name": "metrics"}
                            ]
                        }]
                    }
                }
            }
        }

    def _get_security_context(self) -> Dict[str, Any]:
        """Contexte de sécurité avancé pour les pods."""
        return {
            "runAsNonRoot": True,
            "runAsUser": 10001,
            "runAsGroup": 10001,
            "fsGroup": 10001,
            "seccompProfile": {"type": "RuntimeDefault"},
            "sysctls": [
                {"name": "net.core.somaxconn", "value": "1024"},
                {"name": "net.ipv4.tcp_keepalive_time", "value": "600"}
            ]
        }

    def _get_resource_requirements(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Exigences de ressources basées sur le tier du tenant."""
        return {
            "requests": {
                "cpu": config.cpu_request,
                "memory": config.memory_request,
                "ephemeral-storage": "100Mi"
            },
            "limits": {
                "cpu": config.cpu_limit,
                "memory": config.memory_limit,
                "ephemeral-storage": "1Gi"
            }
        }

    def _get_liveness_probe(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Sonde de vivacité avancée."""
        return {
            "httpGet": {
                "path": config.health_check_path,
                "port": config.service_port,
                "scheme": "HTTP"
            },
            "initialDelaySeconds": 30,
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "successThreshold": 1,
            "failureThreshold": 3
        }

    def _get_readiness_probe(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Sonde de disponibilité avancée."""
        return {
            "httpGet": {
                "path": config.readiness_check_path,
                "port": config.service_port,
                "scheme": "HTTP"
            },
            "initialDelaySeconds": 5,
            "periodSeconds": 5,
            "timeoutSeconds": 3,
            "successThreshold": 1,
            "failureThreshold": 3
        }

    def _get_startup_probe(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Sonde de démarrage pour les applications lentes."""
        return {
            "httpGet": {
                "path": config.health_check_path,
                "port": config.service_port,
                "scheme": "HTTP"
            },
            "initialDelaySeconds": 10,
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "successThreshold": 1,
            "failureThreshold": 30
        }

    def _get_lifecycle_hooks(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Hooks de cycle de vie pour graceful shutdown."""
        return {
            "preStop": {
                "exec": {
                    "command": ["/bin/sh", "-c", "sleep 15"]
                }
            }
        }

    def _get_affinity_rules(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Règles d'affinité pour l'optimisation des performances."""
        return {
            "podAntiAffinity": {
                "preferredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "weight": 100,
                        "podAffinityTerm": {
                            "labelSelector": {
                                "matchExpressions": [
                                    {
                                        "key": "app",
                                        "operator": "In",
                                        "values": [config.name]
                                    }
                                ]
                            },
                            "topologyKey": "kubernetes.io/hostname"
                        }
                    }
                ]
            },
            "nodeAffinity": {
                "preferredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "weight": 50,
                        "preference": {
                            "matchExpressions": [
                                {
                                    "key": "node-type",
                                    "operator": "In",
                                    "values": ["compute-optimized"]
                                }
                            ]
                        }
                    }
                ]
            }
        }

    def _get_tolerations(self, config: DeploymentConfig) -> List[Dict[str, Any]]:
        """Tolérances pour les nœuds spécialisés."""
        base_tolerations = [
            {
                "key": "node.kubernetes.io/not-ready",
                "operator": "Exists",
                "effect": "NoExecute",
                "tolerationSeconds": 300
            },
            {
                "key": "node.kubernetes.io/unreachable",
                "operator": "Exists",
                "effect": "NoExecute",
                "tolerationSeconds": 300
            }
        ]
        
        # Tolérances spéciales pour les services ML
        if config.service_type == ServiceType.ML_SERVICE:
            base_tolerations.append({
                "key": "gpu-node",
                "operator": "Equal",
                "value": "true",
                "effect": "NoSchedule"
            })
        
        return base_tolerations

    def _get_monitoring_sidecar(self) -> Dict[str, Any]:
        """Container sidecar pour le monitoring avancé."""
        return {
            "name": "monitoring-agent",
            "image": "prometheus/node-exporter:latest",
            "ports": [{"containerPort": 9100, "name": "node-metrics"}],
            "resources": {
                "requests": {"cpu": "10m", "memory": "32Mi"},
                "limits": {"cpu": "50m", "memory": "64Mi"}
            },
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": True,
                "runAsNonRoot": True,
                "runAsUser": 65534
            }
        }

    def _get_tracing_sidecar(self) -> Dict[str, Any]:
        """Container sidecar pour le tracing distribué."""
        return {
            "name": "jaeger-agent",
            "image": "jaegertracing/jaeger-agent:latest",
            "ports": [
                {"containerPort": 5775, "protocol": "UDP"},
                {"containerPort": 6831, "protocol": "UDP"},
                {"containerPort": 6832, "protocol": "UDP"},
                {"containerPort": 5778, "protocol": "TCP"}
            ],
            "env": [
                {"name": "REPORTER_GRPC_HOST_PORT", "value": "jaeger-collector:14250"}
            ],
            "resources": {
                "requests": {"cpu": "10m", "memory": "32Mi"},
                "limits": {"cpu": "50m", "memory": "64Mi"}
            }
        }
                            "runAsUser": 1000,
                            "fsGroup": 2000
                        },
                        "containers": [{
                            "name": "backend",
                            "image": "spotify-ai-agent/backend:dev-latest",
                            "imagePullPolicy": "Always",
                            "ports": [
                                {"containerPort": 8000, "name": "http"},
                                {"containerPort": 9000, "name": "metrics"}
                            ],
                            "env": [
                                {"name": "ENVIRONMENT", "value": "development"},
                                {"name": "NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}},
                                {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
                                {"name": "DATABASE_URL", "valueFrom": {"secretKeyRef": {"name": "database-credentials", "key": "url"}}},
                                {"name": "REDIS_URL", "valueFrom": {"secretKeyRef": {"name": "redis-credentials", "key": "url"}}},
                                {"name": "SPOTIFY_CLIENT_ID", "valueFrom": {"secretKeyRef": {"name": "spotify-credentials", "key": "client_id"}}},
                                {"name": "SPOTIFY_CLIENT_SECRET", "valueFrom": {"secretKeyRef": {"name": "spotify-credentials", "key": "client_secret"}}},
                                {"name": "JWT_SECRET_KEY", "valueFrom": {"secretKeyRef": {"name": "jwt-secrets", "key": "secret_key"}}},
                                {"name": "OPENAI_API_KEY", "valueFrom": {"secretKeyRef": {"name": "ai-credentials", "key": "openai_api_key"}}}
                            ],
                            "envFrom": [
                                {"configMapRef": {"name": "spotify-ai-agent-config"}},
                                {"configMapRef": {"name": "ml-model-config"}}
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "512Mi",
                                    "cpu": "250m"
                                },
                                "limits": {
                                    "memory": "2Gi",
                                    "cpu": "1000m"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                                "timeoutSeconds": 3,
                                "failureThreshold": 3
                            },
                            "volumeMounts": [
                                {"name": "app-config", "mountPath": "/app/config", "readOnly": True},
                                {"name": "ml-models", "mountPath": "/app/models", "readOnly": True},
                                {"name": "temp-storage", "mountPath": "/tmp"},
                                {"name": "logs", "mountPath": "/app/logs"}
                            ]
                        }],
                        "volumes": [
                            {"name": "app-config", "configMap": {"name": "spotify-ai-agent-config"}},
                            {"name": "ml-models", "persistentVolumeClaim": {"claimName": "ml-models-pvc"}},
                            {"name": "temp-storage", "emptyDir": {"sizeLimit": "1Gi"}},
                            {"name": "logs", "persistentVolumeClaim": {"claimName": "logs-pvc"}}
                        ],
                        "affinity": {
                            "podAntiAffinity": {
                                "preferredDuringSchedulingIgnoredDuringExecution": [{
                                    "weight": 100,
                                    "podAffinityTerm": {
                                        "labelSelector": {
                                            "matchLabels": {"app": "spotify-ai-agent-backend"}
                                        },
                                        "topologyKey": "kubernetes.io/hostname"
                                    }
                                }]
                            }
                        },
                        "tolerations": [
                            {"key": "node.kubernetes.io/not-ready", "operator": "Exists", "effect": "NoExecute", "tolerationSeconds": 300},
                            {"key": "node.kubernetes.io/unreachable", "operator": "Exists", "effect": "NoExecute", "tolerationSeconds": 300}
                        ]
                    }
                },
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxUnavailable": 1,
                        "maxSurge": 1
                    }
                }
            }
        }
    
    def create_ml_service_deployment(self) -> Dict[str, Any]:
        """Crée le déploiement du service de Machine Learning."""
        return {
            **self.manifest_generator.generate_base_manifest(
                "Deployment", 
                "spotify-ai-agent-ml",
                {"app.kubernetes.io/component": "ml-service"}
            ),
            "spec": {
                "replicas": 2,
                "selector": {
                    "matchLabels": {
                        "app": "spotify-ai-agent-ml",
                        "environment": "development"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            **DEFAULT_LABELS,
                            "app": "spotify-ai-agent-ml",
                            "version": "v1.0.0"
                        }
                    },
                    "spec": {
                        "serviceAccountName": "spotify-ai-agent-ml",
                        "containers": [{
                            "name": "ml-service",
                            "image": "spotify-ai-agent/ml-service:dev-latest",
                            "ports": [
                                {"containerPort": 8001, "name": "grpc"},
                                {"containerPort": 9001, "name": "metrics"}
                            ],
                            "env": [
                                {"name": "ENVIRONMENT", "value": "development"},
                                {"name": "ML_MODEL_PATH", "value": "/app/models"},
                                {"name": "TENSORFLOW_SERVING_PORT", "value": "8501"},
                                {"name": "PYTORCH_MODEL_PATH", "value": "/app/models/pytorch"},
                                {"name": "HUGGINGFACE_CACHE_DIR", "value": "/app/cache/huggingface"}
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "2Gi",
                                    "cpu": "500m",
                                    "nvidia.com/gpu": "0"
                                },
                                "limits": {
                                    "memory": "8Gi",
                                    "cpu": "2000m",
                                    "nvidia.com/gpu": "1"
                                }
                            },
                            "volumeMounts": [
                                {"name": "ml-models", "mountPath": "/app/models"},
                                {"name": "ml-cache", "mountPath": "/app/cache"},
                                {"name": "temp-ml-storage", "mountPath": "/tmp/ml"}
                            ]
                        }],
                        "volumes": [
                            {"name": "ml-models", "persistentVolumeClaim": {"claimName": "ml-models-pvc"}},
                            {"name": "ml-cache", "persistentVolumeClaim": {"claimName": "ml-cache-pvc"}},
                            {"name": "temp-ml-storage", "emptyDir": {"sizeLimit": "5Gi"}}
                        ],
                        "nodeSelector": {
                            "node-type": "gpu-enabled"
                        }
                    }
                }
            }
        }

# Classes d'export
__all__ = ['DeploymentManager']
