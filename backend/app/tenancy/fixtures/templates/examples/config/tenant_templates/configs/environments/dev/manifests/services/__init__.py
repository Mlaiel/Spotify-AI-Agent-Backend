"""
Services Module - Kubernetes Service Definitions
===============================================

Gestion des services Kubernetes pour l'exposition des applications
du système multi-tenant Spotify AI Agent.
"""

from typing import Dict, List, Optional, Any
from ...__init__ import ManifestGenerator, DEFAULT_LABELS

__version__ = "1.0.0"
__author__ = "Fahed Mlaiel Development Team"

class ServiceManager:
    """Gestionnaire des services Kubernetes."""
    
    def __init__(self, namespace: str = "spotify-ai-agent-dev"):
        self.namespace = namespace
        self.manifest_generator = ManifestGenerator(namespace)
    
    def create_backend_service(self) -> Dict[str, Any]:
        """Crée le service pour l'API backend."""
        return {
            **self.manifest_generator.generate_base_manifest(
                "Service", 
                "spotify-ai-agent-backend",
                {"app.kubernetes.io/component": "backend-api"}
            ),
            "spec": {
                "type": "ClusterIP",
                "ports": [
                    {
                        "name": "http",
                        "port": 80,
                        "targetPort": 8000,
                        "protocol": "TCP"
                    },
                    {
                        "name": "metrics",
                        "port": 9000,
                        "targetPort": 9000,
                        "protocol": "TCP"
                    }
                ],
                "selector": {
                    "app": "spotify-ai-agent-backend",
                    "environment": "development"
                },
                "sessionAffinity": "None"
            }
        }
    
    def create_ml_service(self) -> Dict[str, Any]:
        """Crée le service pour le microservice ML."""
        return {
            **self.manifest_generator.generate_base_manifest(
                "Service", 
                "spotify-ai-agent-ml",
                {"app.kubernetes.io/component": "ml-service"}
            ),
            "spec": {
                "type": "ClusterIP",
                "ports": [
                    {
                        "name": "grpc",
                        "port": 8001,
                        "targetPort": 8001,
                        "protocol": "TCP"
                    },
                    {
                        "name": "metrics",
                        "port": 9001,
                        "targetPort": 9001,
                        "protocol": "TCP"
                    },
                    {
                        "name": "tf-serving",
                        "port": 8501,
                        "targetPort": 8501,
                        "protocol": "TCP"
                    }
                ],
                "selector": {
                    "app": "spotify-ai-agent-ml",
                    "environment": "development"
                }
            }
        }

# Classes d'export
__all__ = ['ServiceManager']
