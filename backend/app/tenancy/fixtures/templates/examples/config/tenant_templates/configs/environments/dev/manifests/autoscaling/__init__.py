"""
Module d'autoscaling avancé pour Spotify AI Agent
Gestion intelligente de la scalabilité horizontale et verticale des services
Architecture microservices avec support multi-tenant
"""

from .config_manager import AutoscalingConfigManager
from .hpa_controller import HorizontalPodAutoscaler
from .vpa_controller import VerticalPodAutoscaler
from .metrics_collector import MetricsCollector
from .scaling_policies import ScalingPolicyEngine
from .tenant_scaler import TenantAwareScaler
from .resource_optimizer import ResourceOptimizer
from .cost_optimizer import CostOptimizer

__version__ = "1.0.0"
__author__ = "Spotify AI Agent Team"

__all__ = [
    "AutoscalingConfigManager",
    "HorizontalPodAutoscaler", 
    "VerticalPodAutoscaler",
    "MetricsCollector",
    "ScalingPolicyEngine",
    "TenantAwareScaler",
    "ResourceOptimizer",
    "CostOptimizer"
]
