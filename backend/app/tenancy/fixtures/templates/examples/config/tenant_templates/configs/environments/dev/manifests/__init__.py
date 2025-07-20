"""
Kubernetes Manifests Module for Development Environment
======================================================

Ce module contient tous les manifests Kubernetes pour l'environnement de développement
du système multi-tenant Spotify AI Agent.

Architecture:
- Deployments pour les microservices
- Services pour l'exposition des services
- ConfigMaps pour la configuration
- Secrets pour les données sensibles
- PersistentVolumes pour le stockage
- NetworkPolicies pour la sécurité réseau
- RBAC pour les autorisations
- HPA pour l'auto-scaling
- Ingress pour l'exposition externe

Modules:
- deployments: Déploiements des applications
- services: Services Kubernetes
- configs: ConfigMaps et configurations
- secrets: Gestion des secrets
- storage: Volumes persistants
- networking: Politiques réseau et ingress
- security: RBAC et politiques de sécurité
- monitoring: Métriques et observabilité
- autoscaling: Auto-scaling horizontal
- jobs: Jobs et CronJobs

Équipe de développement dirigée par Fahed Mlaiel:
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import logging

__version__ = "1.0.0"
__author__ = "Fahed Mlaiel"
__team__ = "Spotify AI Agent Development Team"

# Configuration des logs
logger = logging.getLogger(__name__)

# Constantes pour les manifests
MANIFEST_TYPES = {
    'DEPLOYMENT': 'Deployment',
    'SERVICE': 'Service',
    'CONFIGMAP': 'ConfigMap',
    'SECRET': 'Secret',
    'INGRESS': 'Ingress',
    'PERSISTENT_VOLUME': 'PersistentVolume',
    'PERSISTENT_VOLUME_CLAIM': 'PersistentVolumeClaim',
    'NETWORK_POLICY': 'NetworkPolicy',
    'SERVICE_ACCOUNT': 'ServiceAccount',
    'CLUSTER_ROLE': 'ClusterRole',
    'CLUSTER_ROLE_BINDING': 'ClusterRoleBinding',
    'HORIZONTAL_POD_AUTOSCALER': 'HorizontalPodAutoscaler',
    'JOB': 'Job',
    'CRON_JOB': 'CronJob'
}

# Namespace par défaut pour le développement
DEFAULT_NAMESPACE = "spotify-ai-agent-dev"

# Labels standards pour tous les manifests
DEFAULT_LABELS = {
    "app.kubernetes.io/name": "spotify-ai-agent",
    "app.kubernetes.io/instance": "dev",
    "app.kubernetes.io/version": "v1.0.0",
    "app.kubernetes.io/component": "backend",
    "app.kubernetes.io/part-of": "spotify-ai-agent",
    "app.kubernetes.io/managed-by": "helm",
    "environment": "development",
    "team": "spotify-ai-agent-dev"
}

class ManifestGenerator:
    """Générateur de manifests Kubernetes pour l'environnement de développement."""
    
    def __init__(self, namespace: str = DEFAULT_NAMESPACE):
        self.namespace = namespace
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_base_manifest(self, 
                             kind: str, 
                             name: str, 
                             labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Génère la structure de base d'un manifest Kubernetes."""
        base_labels = DEFAULT_LABELS.copy()
        if labels:
            base_labels.update(labels)
        
        return {
            "apiVersion": self._get_api_version(kind),
            "kind": kind,
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "labels": base_labels,
                "annotations": {
                    "created-by": "spotify-ai-agent-manifest-generator",
                    "team": "Fahed Mlaiel Development Team"
                }
            }
        }
    
    def _get_api_version(self, kind: str) -> str:
        """Retourne la version d'API appropriée pour le type de ressource."""
        api_versions = {
            "Deployment": "apps/v1",
            "Service": "v1",
            "ConfigMap": "v1",
            "Secret": "v1",
            "Ingress": "networking.k8s.io/v1",
            "PersistentVolume": "v1",
            "PersistentVolumeClaim": "v1",
            "NetworkPolicy": "networking.k8s.io/v1",
            "ServiceAccount": "v1",
            "ClusterRole": "rbac.authorization.k8s.io/v1",
            "ClusterRoleBinding": "rbac.authorization.k8s.io/v1",
            "HorizontalPodAutoscaler": "autoscaling/v2",
            "Job": "batch/v1",
            "CronJob": "batch/v1"
        }
        return api_versions.get(kind, "v1")

def load_manifest_template(template_path: Path) -> Dict[str, Any]:
    """Charge un template de manifest depuis un fichier YAML."""
    try:
        with open(template_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du template {template_path}: {e}")
        raise

def save_manifest(manifest: Dict[str, Any], output_path: Path) -> None:
    """Sauvegarde un manifest dans un fichier YAML."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(manifest, file, default_flow_style=False, indent=2)
        logger.info(f"Manifest sauvegardé dans {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du manifest {output_path}: {e}")
        raise

# Exportation des classes et fonctions principales
__all__ = [
    'ManifestGenerator',
    'load_manifest_template',
    'save_manifest',
    'MANIFEST_TYPES',
    'DEFAULT_NAMESPACE',
    'DEFAULT_LABELS'
]
