#!/usr/bin/env python3
"""
Spotify AI Agent - Fixture Templates Package
===========================================

Enterprise template management system providing:
- Dynamic template generation and rendering
- Multi-tenant template isolation
- Template inheritance and composition
- Real-time template compilation
- Advanced template caching and optimization
- Template versioning and migration
- Security-first template processing

Author: Expert Development Team (Fahed Mlaiel)
"""

from .engine import TemplateEngine, TemplateRenderer
from .manager import TemplateManager, TemplateCache
from .generators import (
    TenantTemplateGenerator,
    UserTemplateGenerator,
    ContentTemplateGenerator,
    AISessionTemplateGenerator,
    CollaborationTemplateGenerator
)
from .validators import TemplateValidator, TemplateSecurityChecker
from .loaders import TemplateLoader, RemoteTemplateLoader
from .processors import TemplateProcessor, TemplateOptimizer
from .migrations import TemplateMigrator, TemplateVersionManager

# Template registry for programmatic access
TEMPLATE_CATEGORIES = {
    "tenant": {
        "name": "Tenant Templates",
        "description": "Templates for tenant initialization and configuration",
        "generator": TenantTemplateGenerator,
        "templates": [
            "tenant_init.json",
            "tenant_config.yaml", 
            "tenant_permissions.json",
            "tenant_billing.json",
            "tenant_integrations.yaml"
        ]
    },
    "user": {
        "name": "User Templates", 
        "description": "Templates for user profiles and preferences",
        "generator": UserTemplateGenerator,
        "templates": [
            "user_profile.json",
            "user_preferences.json",
            "user_settings.yaml",
            "user_roles.json",
            "user_onboarding.json"
        ]
    },
    "content": {
        "name": "Content Templates",
        "description": "Templates for content generation and management",
        "generator": ContentTemplateGenerator,
        "templates": [
            "content_types.json",
            "content_metadata.yaml",
            "content_workflows.json",
            "content_ai_prompts.json",
            "content_analytics.yaml"
        ]
    },
    "ai_session": {
        "name": "AI Session Templates",
        "description": "Templates for AI interaction sessions",
        "generator": AISessionTemplateGenerator,
        "templates": [
            "session_config.json",
            "session_prompts.yaml",
            "session_workflows.json",
            "session_analytics.json",
            "session_responses.yaml"
        ]
    },
    "collaboration": {
        "name": "Collaboration Templates",
        "description": "Templates for collaborative features",
        "generator": CollaborationTemplateGenerator,
        "templates": [
            "collaboration_spaces.json",
            "collaboration_workflows.yaml",
            "collaboration_permissions.json",
            "collaboration_notifications.json",
            "collaboration_analytics.yaml"
        ]
    }
}

# Template metadata for management
TEMPLATE_METADATA = {
    "version": "1.0.0",
    "schema_version": "2024.1",
    "supported_formats": ["json", "yaml", "jinja2", "liquid"],
    "template_engine": "advanced_jinja2",
    "caching_enabled": True,
    "security_scanning": True,
    "migration_support": True
}

__all__ = [
    "TemplateEngine",
    "TemplateRenderer", 
    "TemplateManager",
    "TemplateCache",
    "TenantTemplateGenerator",
    "UserTemplateGenerator", 
    "ContentTemplateGenerator",
    "AISessionTemplateGenerator",
    "CollaborationTemplateGenerator",
    "TemplateValidator",
    "TemplateSecurityChecker",
    "TemplateLoader",
    "RemoteTemplateLoader",
    "TemplateProcessor",
    "TemplateOptimizer",
    "TemplateMigrator",
    "TemplateVersionManager",
    "TEMPLATE_CATEGORIES",
    "TEMPLATE_METADATA"
]
