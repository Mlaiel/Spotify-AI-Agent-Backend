"""
Gestionnaire centralisé de configuration pour l'autoscaling multi-tenant
Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ScalingMode(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    INTELLIGENT = "intelligent"
    COST_OPTIMIZED = "cost_optimized"

class TenantTier(Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

@dataclass
class ResourceLimits:
    """Limites de ressources par tenant"""
    min_cpu: float
    max_cpu: float
    min_memory: str
    max_memory: str
    min_replicas: int
    max_replicas: int
    max_nodes: int
    storage_limit: str

@dataclass
class ScalingPolicy:
    """Politique de scaling par service"""
    target_cpu_utilization: int
    target_memory_utilization: int
    scale_up_stabilization: int
    scale_down_stabilization: int
    scale_up_percent: int
    scale_down_percent: int
    metrics_server_delay: int

@dataclass
class TenantConfig:
    """Configuration complète par tenant"""
    tenant_id: str
    tier: TenantTier
    scaling_mode: ScalingMode
    resource_limits: ResourceLimits
    scaling_policies: Dict[str, ScalingPolicy]
    priority: int
    cost_budget: float
    sla_requirements: Dict[str, Any]

class AutoscalingConfigManager:
    """Gestionnaire avancé des configurations d'autoscaling"""
    
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else Path(__file__).parent / "configs"
        self.configs: Dict[str, TenantConfig] = {}
        self.global_config: Dict[str, Any] = {}
        self.default_policies: Dict[str, ScalingPolicy] = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Charge toutes les configurations depuis les fichiers"""
        try:
            # Configuration globale
            global_config_path = self.config_path / "global-config.yaml"
            if global_config_path.exists():
                with open(global_config_path, 'r') as f:
                    self.global_config = yaml.safe_load(f)
            
            # Politiques par défaut
            default_policies_path = self.config_path / "default-policies.yaml"
            if default_policies_path.exists():
                with open(default_policies_path, 'r') as f:
                    policies_data = yaml.safe_load(f)
                    for service, policy_data in policies_data.items():
                        self.default_policies[service] = ScalingPolicy(**policy_data)
            
            # Configurations par tenant
            tenant_configs_path = self.config_path / "tenant-configs"
            if tenant_configs_path.exists():
                for tenant_file in tenant_configs_path.glob("*.yaml"):
                    with open(tenant_file, 'r') as f:
                        tenant_data = yaml.safe_load(f)
                        tenant_config = self._parse_tenant_config(tenant_data)
                        self.configs[tenant_config.tenant_id] = tenant_config
                        
            logger.info(f"Loaded {len(self.configs)} tenant configurations")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise
    
    def _parse_tenant_config(self, data: Dict[str, Any]) -> TenantConfig:
        """Parse et valide la configuration d'un tenant"""
        resource_limits = ResourceLimits(**data["resource_limits"])
        
        scaling_policies = {}
        for service, policy_data in data.get("scaling_policies", {}).items():
            scaling_policies[service] = ScalingPolicy(**policy_data)
        
        return TenantConfig(
            tenant_id=data["tenant_id"],
            tier=TenantTier(data["tier"]),
            scaling_mode=ScalingMode(data["scaling_mode"]),
            resource_limits=resource_limits,
            scaling_policies=scaling_policies,
            priority=data.get("priority", 1),
            cost_budget=data.get("cost_budget", 1000.0),
            sla_requirements=data.get("sla_requirements", {})
        )
    
    def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Récupère la configuration d'un tenant"""
        return self.configs.get(tenant_id)
    
    def get_scaling_policy(self, tenant_id: str, service: str) -> ScalingPolicy:
        """Récupère la politique de scaling pour un service/tenant"""
        tenant_config = self.get_tenant_config(tenant_id)
        
        if tenant_config and service in tenant_config.scaling_policies:
            return tenant_config.scaling_policies[service]
        
        # Fallback vers politique par défaut
        if service in self.default_policies:
            return self.default_policies[service]
        
        # Politique ultra-conservative par défaut
        return ScalingPolicy(
            target_cpu_utilization=70,
            target_memory_utilization=80,
            scale_up_stabilization=300,
            scale_down_stabilization=600,
            scale_up_percent=50,
            scale_down_percent=25,
            metrics_server_delay=60
        )
    
    def update_tenant_config(self, tenant_id: str, config: TenantConfig):
        """Met à jour la configuration d'un tenant"""
        self.configs[tenant_id] = config
        self._save_tenant_config(config)
    
    def _save_tenant_config(self, config: TenantConfig):
        """Sauvegarde la configuration d'un tenant"""
        tenant_configs_path = self.config_path / "tenant-configs"
        tenant_configs_path.mkdir(parents=True, exist_ok=True)
        
        config_file = tenant_configs_path / f"{config.tenant_id}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
    
    def get_resource_limits(self, tenant_id: str) -> ResourceLimits:
        """Récupère les limites de ressources pour un tenant"""
        tenant_config = self.get_tenant_config(tenant_id)
        if tenant_config:
            return tenant_config.resource_limits
        
        # Limites par défaut très restrictives
        return ResourceLimits(
            min_cpu=0.1,
            max_cpu=2.0,
            min_memory="128Mi",
            max_memory="2Gi",
            min_replicas=1,
            max_replicas=5,
            max_nodes=2,
            storage_limit="10Gi"
        )
    
    def is_scaling_allowed(self, tenant_id: str, current_cost: float) -> bool:
        """Vérifie si le scaling est autorisé selon le budget"""
        tenant_config = self.get_tenant_config(tenant_id)
        if not tenant_config:
            return False
        
        return current_cost < tenant_config.cost_budget * 0.9  # 90% du budget
    
    def get_tenant_priority(self, tenant_id: str) -> int:
        """Récupère la priorité d'un tenant"""
        tenant_config = self.get_tenant_config(tenant_id)
        return tenant_config.priority if tenant_config else 1
    
    def get_scaling_mode(self, tenant_id: str) -> ScalingMode:
        """Récupère le mode de scaling d'un tenant"""
        tenant_config = self.get_tenant_config(tenant_id)
        return tenant_config.scaling_mode if tenant_config else ScalingMode.CONSERVATIVE
    
    async def reload_configurations(self):
        """Recharge les configurations depuis les fichiers"""
        self._load_configurations()
        logger.info("Configurations reloaded successfully")
    
    def validate_configuration(self, tenant_id: str) -> List[str]:
        """Valide la configuration d'un tenant et retourne les erreurs"""
        errors = []
        tenant_config = self.get_tenant_config(tenant_id)
        
        if not tenant_config:
            errors.append(f"No configuration found for tenant {tenant_id}")
            return errors
        
        # Validation des limites de ressources
        limits = tenant_config.resource_limits
        if limits.min_replicas > limits.max_replicas:
            errors.append("min_replicas cannot be greater than max_replicas")
        
        if limits.min_cpu > limits.max_cpu:
            errors.append("min_cpu cannot be greater than max_cpu")
        
        # Validation des politiques de scaling
        for service, policy in tenant_config.scaling_policies.items():
            if policy.target_cpu_utilization > 100:
                errors.append(f"Invalid CPU target for service {service}")
            
            if policy.scale_up_percent <= 0:
                errors.append(f"Invalid scale up percentage for service {service}")
        
        return errors
    
    def get_all_tenants(self) -> List[str]:
        """Retourne la liste de tous les tenants configurés"""
        return list(self.configs.keys())
    
    def export_config(self, tenant_id: str) -> Dict[str, Any]:
        """Exporte la configuration d'un tenant en format dict"""
        tenant_config = self.get_tenant_config(tenant_id)
        return asdict(tenant_config) if tenant_config else {}
    
    def import_config(self, tenant_id: str, config_data: Dict[str, Any]):
        """Importe une configuration de tenant depuis un dict"""
        tenant_config = self._parse_tenant_config(config_data)
        self.update_tenant_config(tenant_id, tenant_config)
