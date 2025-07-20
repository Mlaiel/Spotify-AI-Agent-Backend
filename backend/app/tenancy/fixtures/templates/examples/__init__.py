"""
Enterprise Template Management System
Advanced Multi-Tenant Template Engine with Dynamic Configuration

This module provides the core template management system for the Spotify AI Agent platform,
handling tenant and user template provisioning, configuration management, and dynamic
template generation with enterprise-grade features.

Lead Developer & AI Architect: Fahed Mlaiel
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from contextlib import asynccontextmanager
import uuid
import hashlib
import secrets

# Import tenant and user management
from .tenant import TenantManager, TenantProfile, TenantTier
from .user import UserManager, UserProfile, UserTier

# Configure logging
logger = logging.getLogger(__name__)

class TemplateType(str, Enum):
    """Template type enumeration"""
    TENANT = "tenant"
    USER = "user"
    INTEGRATION = "integration"
    WORKFLOW = "workflow"
    ANALYTICS = "analytics"
    SECURITY = "security"
    COMPLIANCE = "compliance"

class TemplateStatus(str, Enum):
    """Template status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class DeploymentEnvironment(str, Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"

@dataclass
class TemplateMetadata:
    """Template metadata and versioning information"""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    template_type: TemplateType = TemplateType.TENANT
    status: TemplateStatus = TemplateStatus.DRAFT
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT
    
    # Versioning and lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    updated_by: str = "system"
    version_history: List[str] = field(default_factory=list)
    
    # Dependencies and compatibility
    dependencies: List[str] = field(default_factory=list)
    compatible_versions: List[str] = field(default_factory=list)
    minimum_platform_version: str = "1.0.0"
    maximum_platform_version: str = "*"
    
    # Security and compliance
    security_level: str = "standard"
    compliance_frameworks: List[str] = field(default_factory=list)
    encryption_required: bool = False
    audit_required: bool = False
    
    # Feature flags and configuration
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    default_values: Dict[str, Any] = field(default_factory=dict)
    
    # Performance and scaling
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    scaling_parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Deployment and distribution
    deployment_strategy: str = "rolling"
    rollback_strategy: str = "automatic"
    distribution_channels: List[str] = field(default_factory=list)
    target_regions: List[str] = field(default_factory=list)

@dataclass
class TemplateConfiguration:
    """Template configuration with dynamic parameters"""
    template_id: str
    configuration_data: Dict[str, Any] = field(default_factory=dict)
    environment_specific: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    feature_overrides: Dict[str, bool] = field(default_factory=dict)
    
    # Dynamic configuration
    variable_bindings: Dict[str, str] = field(default_factory=dict)
    conditional_logic: List[Dict[str, Any]] = field(default_factory=list)
    inheritance_chain: List[str] = field(default_factory=list)
    
    # Validation and constraints
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    constraint_checks: List[Dict[str, Any]] = field(default_factory=list)
    transformation_pipeline: List[Dict[str, Any]] = field(default_factory=list)

class TemplateEngine:
    """Advanced template engine with dynamic configuration and multi-tenant support"""
    
    def __init__(self):
        self.tenant_manager = TenantManager()
        self.user_manager = UserManager()
        self.template_cache = {}
        self.configuration_cache = {}
        self.template_registry = {}
        
    async def initialize(self):
        """Initialize template engine with registry and cache"""
        try:
            # Load template registry
            await self._load_template_registry()
            
            # Initialize cache
            await self._initialize_cache()
            
            # Load default templates
            await self._load_default_templates()
            
            logger.info("Template engine initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize template engine", error=str(e))
            raise
    
    async def _load_template_registry(self):
        """Load template registry from configuration"""
        # Mock template registry
        self.template_registry = {
            "tenant_free": {
                "metadata": TemplateMetadata(
                    name="Free Tenant Template",
                    template_type=TemplateType.TENANT,
                    status=TemplateStatus.ACTIVE
                ),
                "path": "tenant/free_tenant_init.json"
            },
            "tenant_professional": {
                "metadata": TemplateMetadata(
                    name="Professional Tenant Template",
                    template_type=TemplateType.TENANT,
                    status=TemplateStatus.ACTIVE
                ),
                "path": "tenant/professional_tenant_init.json"
            },
            "user_free": {
                "metadata": TemplateMetadata(
                    name="Free User Profile",
                    template_type=TemplateType.USER,
                    status=TemplateStatus.ACTIVE
                ),
                "path": "user/free_user_profile.json"
            },
            "user_premium": {
                "metadata": TemplateMetadata(
                    name="Premium User Profile",
                    template_type=TemplateType.USER,
                    status=TemplateStatus.ACTIVE
                ),
                "path": "user/premium_user_profile.json"
            }
        }
        
        logger.info("Template registry loaded", templates=len(self.template_registry))
    
    async def _initialize_cache(self):
        """Initialize template and configuration cache"""
        # Mock cache initialization
        self.template_cache = {}
        self.configuration_cache = {}
        
        logger.info("Template cache initialized")
    
    async def _load_default_templates(self):
        """Load default templates into cache"""
        for template_id, template_info in self.template_registry.items():
            try:
                template_path = Path(__file__).parent / template_info["path"]
                if template_path.exists():
                    with open(template_path) as f:
                        template_data = json.load(f)
                    
                    self.template_cache[template_id] = {
                        "metadata": template_info["metadata"],
                        "data": template_data,
                        "loaded_at": datetime.now(timezone.utc)
                    }
                    
                    logger.debug("Template loaded", template_id=template_id)
                    
            except Exception as e:
                logger.warning("Failed to load template", template_id=template_id, error=str(e))
        
        logger.info("Default templates loaded", count=len(self.template_cache))
    
    async def create_tenant_from_template(self, template_id: str, tenant_data: Dict[str, Any]) -> TenantProfile:
        """Create tenant from template with dynamic configuration"""
        try:
            # Get template
            template = await self._get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Validate template type
            if template["metadata"].template_type != TemplateType.TENANT:
                raise ValueError(f"Template {template_id} is not a tenant template")
            
            # Apply template configuration
            tenant_config = await self._apply_template_configuration(template, tenant_data)
            
            # Determine tenant tier from template
            tier_mapping = {
                "tenant_free": TenantTier.FREE,
                "tenant_professional": TenantTier.PROFESSIONAL,
                "tenant_enterprise": TenantTier.ENTERPRISE,
                "tenant_custom": TenantTier.CUSTOM
            }
            
            tenant_tier = tier_mapping.get(template_id, TenantTier.FREE)
            
            # Create tenant using tenant manager
            tenant_profile = await self.tenant_manager.create_tenant(
                name=tenant_data.get("name", "New Tenant"),
                tier=tenant_tier,
                admin_email=tenant_data.get("admin_email", "admin@example.com"),
                configuration=tenant_config
            )
            
            # Apply template-specific configurations
            await self._apply_tenant_template_features(tenant_profile, template)
            
            logger.info("Tenant created from template", 
                       tenant_id=tenant_profile.tenant_id, 
                       template_id=template_id)
            
            return tenant_profile
            
        except Exception as e:
            logger.error("Failed to create tenant from template", 
                        template_id=template_id, error=str(e))
            raise
    
    async def create_user_from_template(self, template_id: str, user_data: Dict[str, Any]) -> UserProfile:
        """Create user from template with dynamic configuration"""
        try:
            # Get template
            template = await self._get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Validate template type
            if template["metadata"].template_type != TemplateType.USER:
                raise ValueError(f"Template {template_id} is not a user template")
            
            # Apply template configuration
            user_config = await self._apply_template_configuration(template, user_data)
            
            # Determine user tier from template
            tier_mapping = {
                "user_free": UserTier.FREE,
                "user_premium": UserTier.PREMIUM,
                "user_enterprise": UserTier.ENTERPRISE,
                "user_vip": UserTier.VIP
            }
            
            user_tier = tier_mapping.get(template_id, UserTier.FREE)
            
            # Create user using user manager
            user_profile = await self.user_manager.create_user(
                email=user_data.get("email", "user@example.com"),
                password=user_data.get("password", "temp_password"),
                tier=user_tier,
                profile_data=user_config
            )
            
            # Apply template-specific configurations
            await self._apply_user_template_features(user_profile, template)
            
            logger.info("User created from template", 
                       user_id=user_profile.user_id, 
                       template_id=template_id)
            
            return user_profile
            
        except Exception as e:
            logger.error("Failed to create user from template", 
                        template_id=template_id, error=str(e))
            raise
    
    async def _get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template from cache or load from storage"""
        # Check cache first
        if template_id in self.template_cache:
            template = self.template_cache[template_id]
            
            # Check if template is still fresh (1 hour cache)
            cache_age = datetime.now(timezone.utc) - template["loaded_at"]
            if cache_age < timedelta(hours=1):
                return template
        
        # Load from storage
        if template_id in self.template_registry:
            template_info = self.template_registry[template_id]
            template_path = Path(__file__).parent / template_info["path"]
            
            if template_path.exists():
                with open(template_path) as f:
                    template_data = json.load(f)
                
                template = {
                    "metadata": template_info["metadata"],
                    "data": template_data,
                    "loaded_at": datetime.now(timezone.utc)
                }
                
                # Update cache
                self.template_cache[template_id] = template
                return template
        
        return None
    
    async def _apply_template_configuration(self, template: Dict[str, Any], 
                                          input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply template configuration with variable substitution"""
        template_data = template["data"].copy()
        
        # Apply variable substitution
        config = await self._substitute_variables(template_data, input_data)
        
        # Apply conditional logic
        config = await self._apply_conditional_logic(config, input_data)
        
        # Apply transformations
        config = await self._apply_transformations(config, input_data)
        
        # Validate configuration
        await self._validate_configuration(config, template["metadata"])
        
        return config
    
    async def _substitute_variables(self, template_data: Dict[str, Any], 
                                   input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute variables in template data"""
        def substitute_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Simple variable substitution
                for key, value in input_data.items():
                    placeholder = f"{{{{{key}}}}}"
                    if placeholder in obj:
                        obj = obj.replace(placeholder, str(value))
                return obj
            else:
                return obj
        
        return substitute_recursive(template_data)
    
    async def _apply_conditional_logic(self, config: Dict[str, Any], 
                                     input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conditional logic to configuration"""
        # Mock conditional logic application
        # In real implementation, this would process conditional statements
        return config
    
    async def _apply_transformations(self, config: Dict[str, Any], 
                                   input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data transformations to configuration"""
        # Mock transformation application
        # In real implementation, this would apply data transformations
        return config
    
    async def _validate_configuration(self, config: Dict[str, Any], 
                                    metadata: TemplateMetadata):
        """Validate configuration against schema"""
        # Mock validation
        # In real implementation, this would validate against JSON schema
        pass
    
    async def _apply_tenant_template_features(self, tenant_profile: TenantProfile, 
                                            template: Dict[str, Any]):
        """Apply template-specific features to tenant"""
        template_data = template["data"]
        
        # Apply configuration from template
        if "configuration" in template_data:
            config_data = template_data["configuration"]
            
            # Update tenant limits
            if "limits" in config_data:
                for key, value in config_data["limits"].items():
                    if hasattr(tenant_profile.limits, key):
                        setattr(tenant_profile.limits, key, value)
            
            # Update feature flags
            if "feature_flags" in config_data:
                tenant_profile.feature_flags.update(config_data["feature_flags"])
        
        logger.info("Template features applied to tenant", tenant_id=tenant_profile.tenant_id)
    
    async def _apply_user_template_features(self, user_profile: UserProfile, 
                                          template: Dict[str, Any]):
        """Apply template-specific features to user"""
        template_data = template["data"]
        
        # Apply configuration from template
        if "configuration" in template_data:
            config_data = template_data["configuration"]
            
            # Update user limits
            if "limits" in config_data:
                for key, value in config_data["limits"].items():
                    if hasattr(user_profile.limits, key):
                        setattr(user_profile.limits, key, value)
            
            # Update feature flags
            if "feature_flags" in config_data:
                user_profile.feature_flags.update(config_data["feature_flags"])
            
            # Update AI preferences
            if "ai_preferences" in config_data:
                for key, value in config_data["ai_preferences"].items():
                    if hasattr(user_profile.ai_preferences, key):
                        setattr(user_profile.ai_preferences, key, value)
        
        logger.info("Template features applied to user", user_id=user_profile.user_id)
    
    async def get_available_templates(self, template_type: Optional[TemplateType] = None,
                                    environment: Optional[DeploymentEnvironment] = None) -> List[Dict[str, Any]]:
        """Get list of available templates with filtering"""
        templates = []
        
        for template_id, template_info in self.template_registry.items():
            metadata = template_info["metadata"]
            
            # Apply filters
            if template_type and metadata.template_type != template_type:
                continue
            
            if environment and metadata.environment != environment:
                continue
            
            templates.append({
                "template_id": template_id,
                "name": metadata.name,
                "description": metadata.description,
                "type": metadata.template_type.value,
                "version": metadata.version,
                "status": metadata.status.value,
                "environment": metadata.environment.value,
                "created_at": metadata.created_at.isoformat(),
                "updated_at": metadata.updated_at.isoformat()
            })
        
        return templates
    
    async def validate_template(self, template_id: str) -> Dict[str, Any]:
        """Validate template structure and configuration"""
        template = await self._get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        validation_result = {
            "template_id": template_id,
            "valid": True,
            "errors": [],
            "warnings": [],
            "metadata_valid": True,
            "data_valid": True,
            "schema_valid": True
        }
        
        try:
            # Validate metadata
            metadata = template["metadata"]
            if not metadata.name:
                validation_result["errors"].append("Template name is required")
                validation_result["metadata_valid"] = False
            
            # Validate template data
            template_data = template["data"]
            if not isinstance(template_data, dict):
                validation_result["errors"].append("Template data must be a dictionary")
                validation_result["data_valid"] = False
            
            # Check for required fields based on template type
            if metadata.template_type == TemplateType.TENANT:
                required_fields = ["limits", "feature_flags", "security_settings"]
                for field in required_fields:
                    if field not in template_data.get("configuration", {}):
                        validation_result["warnings"].append(f"Recommended field '{field}' missing")
            
            elif metadata.template_type == TemplateType.USER:
                required_fields = ["limits", "feature_flags", "ai_preferences"]
                for field in required_fields:
                    if field not in template_data.get("configuration", {}):
                        validation_result["warnings"].append(f"Recommended field '{field}' missing")
            
            # Overall validation status
            validation_result["valid"] = (
                validation_result["metadata_valid"] and 
                validation_result["data_valid"] and 
                validation_result["schema_valid"] and 
                len(validation_result["errors"]) == 0
            )
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    async def get_template_usage_statistics(self) -> Dict[str, Any]:
        """Get template usage statistics and analytics"""
        stats = {
            "total_templates": len(self.template_registry),
            "templates_by_type": {},
            "templates_by_status": {},
            "templates_by_environment": {},
            "cache_statistics": {
                "cached_templates": len(self.template_cache),
                "cache_hit_rate": 0.95,  # Mock value
                "average_load_time": 0.15  # Mock value in seconds
            },
            "usage_metrics": {
                "tenant_creations_24h": 50,  # Mock value
                "user_creations_24h": 200,   # Mock value
                "template_validations_24h": 25  # Mock value
            }
        }
        
        # Calculate distributions
        for template_info in self.template_registry.values():
            metadata = template_info["metadata"]
            
            # By type
            type_key = metadata.template_type.value
            stats["templates_by_type"][type_key] = stats["templates_by_type"].get(type_key, 0) + 1
            
            # By status
            status_key = metadata.status.value
            stats["templates_by_status"][status_key] = stats["templates_by_status"].get(status_key, 0) + 1
            
            # By environment
            env_key = metadata.environment.value
            stats["templates_by_environment"][env_key] = stats["templates_by_environment"].get(env_key, 0) + 1
        
        return stats

# Global template engine instance
template_engine = TemplateEngine()

# Factory functions for easy template-based creation
async def create_tenant_from_template(template_id: str, **kwargs) -> TenantProfile:
    """Create tenant from template"""
    return await template_engine.create_tenant_from_template(template_id, kwargs)

async def create_user_from_template(template_id: str, **kwargs) -> UserProfile:
    """Create user from template"""
    return await template_engine.create_user_from_template(template_id, kwargs)

# Export main components
__all__ = [
    'TemplateType', 'TemplateStatus', 'DeploymentEnvironment',
    'TemplateMetadata', 'TemplateConfiguration', 'TemplateEngine',
    'template_engine', 'create_tenant_from_template', 'create_user_from_template'
]
