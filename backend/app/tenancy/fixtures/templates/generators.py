#!/usr/bin/env python3
"""
Spotify AI Agent - Template Generators
=====================================

Specialized template generators for different entity types:
- Tenant templates (initialization, configuration, billing)
- User templates (profiles, preferences, settings)
- Content templates (types, workflows, analytics)
- AI Session templates (configurations, prompts, workflows)
- Collaboration templates (spaces, permissions, workflows)

Author: Expert Development Team
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4

from app.tenancy.fixtures.utils import FixtureUtils

logger = logging.getLogger(__name__)


class BaseTemplateGenerator(ABC):
    """Base class for all template generators."""
    
    def __init__(self):
        self.template_version = "1.0.0"
        self.schema_version = "2024.1"
        self.generated_at = datetime.now(timezone.utc)
    
    @abstractmethod
    def generate_template(self, **kwargs) -> Dict[str, Any]:
        """Generate template content."""
        pass
    
    def _add_metadata(self, template: Dict[str, Any], template_type: str) -> Dict[str, Any]:
        """Add standard metadata to template."""
        template["_metadata"] = {
            "template_type": template_type,
            "template_version": self.template_version,
            "schema_version": self.schema_version,
            "generated_at": self.generated_at.isoformat(),
            "generator": self.__class__.__name__
        }
        return template
    
    def _generate_id(self, prefix: str = "") -> str:
        """Generate unique ID with optional prefix."""
        return f"{prefix}{uuid4()}" if prefix else str(uuid4())


class TenantTemplateGenerator(BaseTemplateGenerator):
    """Generator for tenant-related templates."""
    
    def generate_tenant_init_template(
        self,
        tier: str = "starter",
        features: Optional[List[str]] = None,
        integrations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate tenant initialization template."""
        
        # Define tier-specific configurations
        tier_configs = {
            "starter": {
                "max_users": 10,
                "storage_gb": 5,
                "ai_sessions_per_month": 100,
                "api_rate_limit": 100,
                "features": ["basic_ai", "simple_collaboration", "standard_support"]
            },
            "professional": {
                "max_users": 100,
                "storage_gb": 50,
                "ai_sessions_per_month": 1000,
                "api_rate_limit": 1000,
                "features": ["advanced_ai", "full_collaboration", "priority_support", "analytics"]
            },
            "enterprise": {
                "max_users": -1,  # Unlimited
                "storage_gb": 500,
                "ai_sessions_per_month": -1,  # Unlimited
                "api_rate_limit": 10000,
                "features": ["premium_ai", "enterprise_collaboration", "24_7_support", "advanced_analytics", "custom_integrations", "sso"]
            }
        }
        
        config = tier_configs.get(tier, tier_configs["starter"])
        
        # Override features if provided
        if features:
            config["features"] = features
        
        template = {
            "tenant_id": "{{ tenant_id }}",
            "tenant_name": "{{ tenant_name | default(tenant_id) }}",
            "tier": tier,
            "configuration": {
                "limits": {
                    "max_users": config["max_users"],
                    "storage_gb": config["storage_gb"],
                    "ai_sessions_per_month": config["ai_sessions_per_month"],
                    "api_rate_limit_per_hour": config["api_rate_limit"]
                },
                "features": {
                    "enabled": config["features"],
                    "disabled": []
                },
                "security": {
                    "password_policy": {
                        "min_length": 8,
                        "require_special_chars": True,
                        "require_numbers": True,
                        "require_uppercase": True,
                        "max_age_days": 90
                    },
                    "session_timeout_minutes": 480,
                    "mfa_required": tier == "enterprise",
                    "ip_whitelist": [],
                    "audit_logging": tier in ["professional", "enterprise"]
                },
                "integrations": {
                    "spotify": {
                        "enabled": True,
                        "client_id": "{{ spotify_client_id | default('') }}",
                        "scopes": ["user-read-private", "user-read-email", "playlist-read-private"]
                    },
                    "slack": {
                        "enabled": "slack" in (integrations or []),
                        "webhook_url": "{{ slack_webhook_url | default('') }}"
                    },
                    "teams": {
                        "enabled": "teams" in (integrations or []),
                        "webhook_url": "{{ teams_webhook_url | default('') }}"
                    }
                }
            },
            "database": {
                "schema_name": "tenant_{{ tenant_id }}",
                "isolation_level": "schema",
                "backup_enabled": True,
                "backup_retention_days": 30 if tier == "starter" else 90
            },
            "storage": {
                "bucket_name": "tenant-{{ tenant_id }}-storage",
                "encryption_enabled": True,
                "versioning_enabled": tier in ["professional", "enterprise"]
            },
            "ai_configuration": {
                "model_access": {
                    "gpt-3.5-turbo": True,
                    "gpt-4": tier in ["professional", "enterprise"],
                    "claude-3": tier == "enterprise",
                    "custom_models": tier == "enterprise"
                },
                "rate_limits": {
                    "requests_per_minute": 10 if tier == "starter" else 50 if tier == "professional" else 200,
                    "tokens_per_day": 10000 if tier == "starter" else 100000 if tier == "professional" else -1
                },
                "features": {
                    "context_memory": tier in ["professional", "enterprise"],
                    "custom_prompts": True,
                    "conversation_export": True,
                    "ai_analytics": tier in ["professional", "enterprise"]
                }
            },
            "created_at": "{{ current_timestamp() }}",
            "expires_at": None,
            "status": "active"
        }
        
        return self._add_metadata(template, "tenant_init")
    
    def generate_tenant_config_template(self) -> Dict[str, Any]:
        """Generate tenant configuration template."""
        template = {
            "tenant_id": "{{ tenant_id }}",
            "display_name": "{{ tenant_display_name | default(tenant_id | title) }}",
            "logo_url": "{{ tenant_logo_url | default('') }}",
            "primary_color": "{{ tenant_primary_color | default('#1DB954') }}",
            "secondary_color": "{{ tenant_secondary_color | default('#191414') }}",
            "timezone": "{{ tenant_timezone | default('UTC') }}",
            "locale": "{{ tenant_locale | default('en_US') }}",
            "contact": {
                "admin_email": "{{ admin_email }}",
                "support_email": "{{ support_email | default(admin_email) }}",
                "billing_email": "{{ billing_email | default(admin_email) }}",
                "phone": "{{ contact_phone | default('') }}",
                "website": "{{ tenant_website | default('') }}"
            },
            "branding": {
                "custom_domain": "{{ custom_domain | default('') }}",
                "favicon_url": "{{ favicon_url | default('') }}",
                "theme": {
                    "dark_mode_enabled": "{{ dark_mode | default(true) }}",
                    "custom_css": "{{ custom_css | default('') }}"
                }
            },
            "notifications": {
                "email_enabled": True,
                "slack_enabled": "{{ feature_enabled('slack_integration') }}",
                "webhook_enabled": "{{ feature_enabled('webhook_notifications') }}",
                "preferences": {
                    "system_alerts": True,
                    "usage_reports": True,
                    "security_notifications": True,
                    "billing_notifications": True
                }
            },
            "compliance": {
                "gdpr_enabled": "{{ gdpr_required | default(false) }}",
                "ccpa_enabled": "{{ ccpa_required | default(false) }}",
                "data_retention_days": "{{ data_retention_days | default(365) }}",
                "audit_log_retention_days": "{{ audit_retention_days | default(90) }}"
            }
        }
        
        return self._add_metadata(template, "tenant_config")
    
    def generate_tenant_permissions_template(self) -> Dict[str, Any]:
        """Generate tenant permissions template."""
        template = {
            "tenant_id": "{{ tenant_id }}",
            "roles": {
                "admin": {
                    "name": "Administrator",
                    "description": "Full access to all tenant features",
                    "permissions": [
                        "tenant.manage",
                        "users.manage",
                        "content.manage",
                        "ai.manage",
                        "collaboration.manage",
                        "analytics.view",
                        "settings.manage",
                        "billing.manage"
                    ],
                    "inherits": [],
                    "system_role": True
                },
                "manager": {
                    "name": "Manager",
                    "description": "Manage users and content",
                    "permissions": [
                        "users.manage",
                        "content.manage",
                        "ai.use",
                        "collaboration.manage",
                        "analytics.view"
                    ],
                    "inherits": ["user"],
                    "system_role": True
                },
                "user": {
                    "name": "User",
                    "description": "Standard user access",
                    "permissions": [
                        "content.create",
                        "content.view",
                        "ai.use",
                        "collaboration.participate",
                        "profile.manage"
                    ],
                    "inherits": [],
                    "system_role": True
                },
                "guest": {
                    "name": "Guest",
                    "description": "Limited read-only access",
                    "permissions": [
                        "content.view",
                        "collaboration.view"
                    ],
                    "inherits": [],
                    "system_role": True
                }
            },
            "permissions": {
                "tenant.manage": {
                    "name": "Manage Tenant",
                    "description": "Full tenant administration",
                    "category": "administration"
                },
                "users.manage": {
                    "name": "Manage Users",
                    "description": "Create, update, delete users",
                    "category": "user_management"
                },
                "content.manage": {
                    "name": "Manage Content",
                    "description": "Full content management",
                    "category": "content"
                },
                "content.create": {
                    "name": "Create Content",
                    "description": "Create new content",
                    "category": "content"
                },
                "content.view": {
                    "name": "View Content",
                    "description": "View content",
                    "category": "content"
                },
                "ai.manage": {
                    "name": "Manage AI",
                    "description": "Configure AI settings",
                    "category": "ai"
                },
                "ai.use": {
                    "name": "Use AI",
                    "description": "Use AI features",
                    "category": "ai"
                },
                "collaboration.manage": {
                    "name": "Manage Collaboration",
                    "description": "Manage collaboration spaces",
                    "category": "collaboration"
                },
                "collaboration.participate": {
                    "name": "Participate in Collaboration",
                    "description": "Join and participate in collaborations",
                    "category": "collaboration"
                },
                "collaboration.view": {
                    "name": "View Collaborations",
                    "description": "View collaboration content",
                    "category": "collaboration"
                },
                "analytics.view": {
                    "name": "View Analytics",
                    "description": "Access analytics and reports",
                    "category": "analytics"
                },
                "settings.manage": {
                    "name": "Manage Settings",
                    "description": "Configure tenant settings",
                    "category": "settings"
                },
                "billing.manage": {
                    "name": "Manage Billing",
                    "description": "Access billing information",
                    "category": "billing"
                },
                "profile.manage": {
                    "name": "Manage Profile",
                    "description": "Manage own profile",
                    "category": "profile"
                }
            },
            "default_role": "user",
            "role_assignment_rules": {
                "auto_assign_admin": False,
                "email_domain_roles": {},
                "invite_default_role": "user"
            }
        }
        
        return self._add_metadata(template, "tenant_permissions")
    
    def generate_tenant_billing_template(self) -> Dict[str, Any]:
        """Generate tenant billing configuration template."""
        template = {
            "tenant_id": "{{ tenant_id }}",
            "billing_config": {
                "currency": "{{ billing_currency | default('USD') }}",
                "billing_cycle": "{{ billing_cycle | default('monthly') }}",
                "billing_date": "{{ billing_date | default(1) }}",
                "payment_method": {
                    "type": "{{ payment_method_type | default('credit_card') }}",
                    "details": {}
                },
                "tax_info": {
                    "tax_id": "{{ tax_id | default('') }}",
                    "tax_rate": "{{ tax_rate | default(0) }}",
                    "tax_region": "{{ tax_region | default('') }}"
                }
            },
            "subscription": {
                "plan_id": "{{ plan_id }}",
                "status": "active",
                "started_at": "{{ current_timestamp() }}",
                "next_billing_date": "{{ next_billing_date }}",
                "auto_renew": "{{ auto_renew | default(true) }}"
            },
            "usage_tracking": {
                "track_users": True,
                "track_storage": True,
                "track_ai_sessions": True,
                "track_api_calls": True,
                "billing_alerts": {
                    "usage_threshold_percent": 80,
                    "overage_alerts": True,
                    "cost_alerts": True
                }
            },
            "pricing": {
                "base_price": "{{ base_price }}",
                "user_price": "{{ user_price | default(0) }}",
                "storage_price_per_gb": "{{ storage_price | default(0.10) }}",
                "ai_session_price": "{{ ai_session_price | default(0.01) }}",
                "overage_rates": {
                    "users": "{{ user_overage_rate | default(5.00) }}",
                    "storage_gb": "{{ storage_overage_rate | default(0.20) }}",
                    "ai_sessions": "{{ ai_overage_rate | default(0.02) }}"
                }
            }
        }
        
        return self._add_metadata(template, "tenant_billing")
    
    def generate_tenant_integrations_template(self) -> Dict[str, Any]:
        """Generate tenant integrations template."""
        template = {
            "tenant_id": "{{ tenant_id }}",
            "integrations": {
                "spotify": {
                    "enabled": True,
                    "config": {
                        "client_id": "{{ spotify_client_id }}",
                        "client_secret": "{{ spotify_client_secret | encrypt_sensitive }}",
                        "redirect_uri": "{{ spotify_redirect_uri }}",
                        "scopes": [
                            "user-read-private",
                            "user-read-email",
                            "playlist-read-private",
                            "playlist-modify-public",
                            "playlist-modify-private"
                        ]
                    },
                    "features": {
                        "playlist_analysis": True,
                        "recommendation_engine": True,
                        "social_features": "{{ feature_enabled('spotify_social') }}"
                    }
                },
                "slack": {
                    "enabled": "{{ feature_enabled('slack_integration') }}",
                    "config": {
                        "webhook_url": "{{ slack_webhook_url | encrypt_sensitive }}",
                        "channel": "{{ slack_channel | default('#general') }}",
                        "bot_token": "{{ slack_bot_token | encrypt_sensitive }}"
                    },
                    "notifications": {
                        "ai_sessions": True,
                        "collaboration_updates": True,
                        "system_alerts": True
                    }
                },
                "microsoft_teams": {
                    "enabled": "{{ feature_enabled('teams_integration') }}",
                    "config": {
                        "webhook_url": "{{ teams_webhook_url | encrypt_sensitive }}",
                        "tenant_id": "{{ ms_tenant_id | default('') }}"
                    },
                    "notifications": {
                        "ai_sessions": True,
                        "collaboration_updates": True
                    }
                },
                "google_workspace": {
                    "enabled": "{{ feature_enabled('google_workspace') }}",
                    "config": {
                        "client_id": "{{ google_client_id }}",
                        "client_secret": "{{ google_client_secret | encrypt_sensitive }}",
                        "domain": "{{ google_domain | default('') }}"
                    },
                    "features": {
                        "sso": True,
                        "calendar_integration": True,
                        "drive_integration": "{{ feature_enabled('google_drive') }}"
                    }
                },
                "webhook": {
                    "enabled": "{{ feature_enabled('webhook_notifications') }}",
                    "config": {
                        "url": "{{ webhook_url }}",
                        "secret": "{{ webhook_secret | encrypt_sensitive }}",
                        "events": [
                            "user.created",
                            "ai_session.completed",
                            "collaboration.created"
                        ]
                    }
                }
            },
            "sso": {
                "enabled": "{{ feature_enabled('sso') }}",
                "provider": "{{ sso_provider | default('google') }}",
                "config": {
                    "domain": "{{ sso_domain }}",
                    "auto_provision": "{{ sso_auto_provision | default(true) }}",
                    "default_role": "{{ sso_default_role | default('user') }}"
                }
            }
        }
        
        return self._add_metadata(template, "tenant_integrations")


class UserTemplateGenerator(BaseTemplateGenerator):
    """Generator for user-related templates."""
    
    def generate_user_profile_template(self) -> Dict[str, Any]:
        """Generate user profile template."""
        template = {
            "user_id": "{{ user_id | default(generate_uuid()) }}",
            "tenant_id": "{{ tenant_id }}",
            "email": "{{ email }}",
            "username": "{{ username | default(email.split('@')[0]) }}",
            "display_name": "{{ display_name | default(username) }}",
            "first_name": "{{ first_name | default('') }}",
            "last_name": "{{ last_name | default('') }}",
            "avatar_url": "{{ avatar_url | default('') }}",
            "role": "{{ role | default('user') }}",
            "status": "active",
            "profile": {
                "bio": "{{ bio | default('') }}",
                "location": "{{ location | default('') }}",
                "website": "{{ website | default('') }}",
                "social_links": {
                    "spotify": "{{ spotify_profile | default('') }}",
                    "twitter": "{{ twitter_handle | default('') }}",
                    "linkedin": "{{ linkedin_profile | default('') }}"
                },
                "music_preferences": {
                    "favorite_genres": "{{ favorite_genres | default([]) | list }}",
                    "favorite_artists": "{{ favorite_artists | default([]) | list }}",
                    "listening_habits": {
                        "primary_platform": "spotify",
                        "average_daily_hours": "{{ daily_listening_hours | default(2) }}",
                        "preferred_discovery_method": "{{ discovery_method | default('recommendations') }}"
                    }
                }
            },
            "account": {
                "created_at": "{{ current_timestamp() }}",
                "last_login": null,
                "login_count": 0,
                "email_verified": false,
                "phone_verified": false,
                "mfa_enabled": false,
                "password_last_changed": "{{ current_timestamp() }}"
            },
            "permissions": {
                "role": "{{ role | default('user') }}",
                "custom_permissions": [],
                "restrictions": []
            },
            "onboarding": {
                "completed": false,
                "current_step": "profile_setup",
                "steps_completed": [],
                "started_at": "{{ current_timestamp() }}"
            }
        }
        
        return self._add_metadata(template, "user_profile")
    
    def generate_user_preferences_template(self) -> Dict[str, Any]:
        """Generate user preferences template."""
        template = {
            "user_id": "{{ user_id }}",
            "tenant_id": "{{ tenant_id }}",
            "preferences": {
                "interface": {
                    "theme": "{{ theme | default('dark') }}",
                    "language": "{{ language | default('en') }}",
                    "timezone": "{{ timezone | default('UTC') }}",
                    "date_format": "{{ date_format | default('YYYY-MM-DD') }}",
                    "time_format": "{{ time_format | default('24h') }}",
                    "density": "{{ ui_density | default('comfortable') }}"
                },
                "notifications": {
                    "email": {
                        "enabled": "{{ email_notifications | default(true) }}",
                        "frequency": "{{ email_frequency | default('immediate') }}",
                        "types": {
                            "ai_session_complete": true,
                            "collaboration_invite": true,
                            "system_updates": false,
                            "marketing": false
                        }
                    },
                    "push": {
                        "enabled": "{{ push_notifications | default(true) }}",
                        "types": {
                            "mentions": true,
                            "ai_responses": true,
                            "collaboration_updates": true
                        }
                    },
                    "in_app": {
                        "enabled": true,
                        "sound": "{{ notification_sound | default(true) }}",
                        "badge": true
                    }
                },
                "ai": {
                    "default_model": "{{ default_ai_model | default('gpt-3.5-turbo') }}",
                    "response_style": "{{ ai_response_style | default('balanced') }}",
                    "creativity_level": "{{ ai_creativity | default(0.7) }}",
                    "max_response_length": "{{ max_ai_response | default(500) }}",
                    "context_memory": "{{ ai_context_memory | default(true) }}",
                    "auto_save_sessions": "{{ auto_save_ai | default(true) }}"
                },
                "collaboration": {
                    "auto_join_spaces": "{{ auto_join_spaces | default(false) }}",
                    "default_visibility": "{{ default_visibility | default('private') }}",
                    "allow_mentions": "{{ allow_mentions | default(true) }}",
                    "share_activity": "{{ share_activity | default(true) }}"
                },
                "privacy": {
                    "profile_visibility": "{{ profile_visibility | default('tenant') }}",
                    "activity_tracking": "{{ activity_tracking | default(true) }}",
                    "analytics_participation": "{{ analytics_participation | default(true) }}",
                    "data_sharing": "{{ data_sharing | default(false) }}"
                },
                "music": {
                    "spotify_integration": "{{ spotify_integration | default(true) }}",
                    "auto_import_playlists": "{{ auto_import_playlists | default(true) }}",
                    "share_listening_activity": "{{ share_listening | default(true) }}",
                    "recommendation_preferences": {
                        "discovery_mode": "{{ discovery_mode | default('balanced') }}",
                        "explicit_content": "{{ allow_explicit | default(true) }}",
                        "diversity_preference": "{{ diversity_pref | default(0.5) }}"
                    }
                }
            },
            "last_updated": "{{ current_timestamp() }}"
        }
        
        return self._add_metadata(template, "user_preferences")
    
    def generate_user_settings_template(self) -> Dict[str, Any]:
        """Generate user settings template."""
        template = {
            "user_id": "{{ user_id }}",
            "tenant_id": "{{ tenant_id }}",
            "settings": {
                "security": {
                    "two_factor_auth": {
                        "enabled": "{{ mfa_enabled | default(false) }}",
                        "method": "{{ mfa_method | default('app') }}",
                        "backup_codes_generated": false
                    },
                    "login_alerts": "{{ login_alerts | default(true) }}",
                    "session_timeout": "{{ session_timeout | default(480) }}",
                    "password_requirements": {
                        "min_length": 8,
                        "require_special_chars": true,
                        "require_numbers": true
                    }
                },
                "api": {
                    "api_key_enabled": "{{ api_access | default(false) }}",
                    "rate_limit": "{{ api_rate_limit | default(100) }}",
                    "allowed_ips": "{{ api_allowed_ips | default([]) | list }}"
                },
                "data": {
                    "export_format": "{{ export_format | default('json') }}",
                    "auto_backup": "{{ auto_backup | default(false) }}",
                    "retention_period": "{{ data_retention | default(365) }}"
                },
                "integrations": {
                    "spotify": {
                        "connected": false,
                        "scopes": [],
                        "last_sync": null
                    },
                    "google": {
                        "connected": false,
                        "scopes": [],
                        "last_sync": null
                    }
                }
            }
        }
        
        return self._add_metadata(template, "user_settings")
    
    def generate_user_roles_template(self) -> Dict[str, Any]:
        """Generate user roles template."""
        template = {
            "user_id": "{{ user_id }}",
            "tenant_id": "{{ tenant_id }}",
            "role_assignments": [
                {
                    "role": "{{ primary_role | default('user') }}",
                    "scope": "tenant",
                    "assigned_at": "{{ current_timestamp() }}",
                    "assigned_by": "{{ assigned_by | default('system') }}",
                    "expires_at": null,
                    "conditions": {}
                }
            ],
            "permission_overrides": {
                "granted": "{{ additional_permissions | default([]) | list }}",
                "denied": "{{ denied_permissions | default([]) | list }}"
            },
            "context_roles": {
                "workspaces": {},
                "projects": {},
                "collaborations": {}
            },
            "role_history": [
                {
                    "role": "{{ primary_role | default('user') }}",
                    "action": "assigned",
                    "timestamp": "{{ current_timestamp() }}",
                    "reason": "initial_assignment"
                }
            ]
        }
        
        return self._add_metadata(template, "user_roles")
    
    def generate_user_onboarding_template(self) -> Dict[str, Any]:
        """Generate user onboarding template."""
        template = {
            "user_id": "{{ user_id }}",
            "tenant_id": "{{ tenant_id }}",
            "onboarding_flow": {
                "flow_id": "default_user_onboarding",
                "version": "1.0",
                "started_at": "{{ current_timestamp() }}",
                "completed_at": null,
                "current_step": 1,
                "total_steps": 6
            },
            "steps": [
                {
                    "step_id": 1,
                    "name": "welcome",
                    "title": "Welcome to {{ tenant_name | default('Spotify AI Agent') }}",
                    "description": "Let's get you started with a quick tour",
                    "type": "introduction",
                    "completed": false,
                    "skippable": false,
                    "data": {
                        "welcome_message": "Welcome to your AI-powered music experience!",
                        "video_url": "",
                        "estimated_time": "2 minutes"
                    }
                },
                {
                    "step_id": 2,
                    "name": "profile_setup",
                    "title": "Set Up Your Profile",
                    "description": "Tell us a bit about yourself and your music preferences",
                    "type": "form",
                    "completed": false,
                    "skippable": true,
                    "data": {
                        "fields": [
                            "display_name",
                            "favorite_genres",
                            "listening_habits"
                        ]
                    }
                },
                {
                    "step_id": 3,
                    "name": "spotify_connection",
                    "title": "Connect Your Spotify",
                    "description": "Link your Spotify account for personalized recommendations",
                    "type": "integration",
                    "completed": false,
                    "skippable": true,
                    "data": {
                        "integration": "spotify",
                        "required_scopes": [
                            "user-read-private",
                            "playlist-read-private"
                        ]
                    }
                },
                {
                    "step_id": 4,
                    "name": "ai_introduction",
                    "title": "Meet Your AI Assistant",
                    "description": "Learn how to interact with your AI music assistant",
                    "type": "tutorial",
                    "completed": false,
                    "skippable": false,
                    "data": {
                        "tutorial_steps": [
                            "Basic AI interactions",
                            "Music discovery features",
                            "Playlist generation"
                        ]
                    }
                },
                {
                    "step_id": 5,
                    "name": "collaboration_intro",
                    "title": "Collaboration Features",
                    "description": "Discover how to collaborate with others",
                    "type": "feature_tour",
                    "completed": false,
                    "skippable": true,
                    "data": {
                        "features": [
                            "Shared playlists",
                            "Music discussions",
                            "Collaborative discovery"
                        ]
                    }
                },
                {
                    "step_id": 6,
                    "name": "completion",
                    "title": "You're All Set!",
                    "description": "Start exploring your AI-powered music experience",
                    "type": "completion",
                    "completed": false,
                    "skippable": false,
                    "data": {
                        "next_actions": [
                            "Start your first AI session",
                            "Explore the music library",
                            "Join a collaboration space"
                        ]
                    }
                }
            ],
            "progress": {
                "completion_percentage": 0,
                "steps_completed": 0,
                "estimated_time_remaining": "15 minutes"
            }
        }
        
        return self._add_metadata(template, "user_onboarding")


class ContentTemplateGenerator(BaseTemplateGenerator):
    """Generator for content-related templates."""
    
    def generate_content_types_template(self) -> Dict[str, Any]:
        """Generate content types configuration template."""
        template = {
            "tenant_id": "{{ tenant_id }}",
            "content_types": {
                "playlist": {
                    "name": "Playlist",
                    "description": "Curated music playlists",
                    "fields": {
                        "title": {
                            "type": "string",
                            "required": True,
                            "max_length": 200
                        },
                        "description": {
                            "type": "text",
                            "required": False,
                            "max_length": 1000
                        },
                        "tracks": {
                            "type": "array",
                            "items": "track_reference",
                            "required": True
                        },
                        "cover_image": {
                            "type": "image_url",
                            "required": False
                        },
                        "mood": {
                            "type": "enum",
                            "values": ["happy", "sad", "energetic", "relaxed", "focus"],
                            "required": False
                        },
                        "genre": {
                            "type": "array",
                            "items": "string",
                            "required": False
                        }
                    },
                    "ai_fields": {
                        "auto_description": {
                            "enabled": True,
                            "prompt": "Generate a description for this playlist based on the tracks"
                        },
                        "mood_detection": {
                            "enabled": True,
                            "model": "audio_analysis"
                        }
                    }
                },
                "track_analysis": {
                    "name": "Track Analysis",
                    "description": "AI-powered track analysis and insights",
                    "fields": {
                        "track_id": {
                            "type": "string",
                            "required": True
                        },
                        "spotify_id": {
                            "type": "string",
                            "required": False
                        },
                        "audio_features": {
                            "type": "object",
                            "properties": {
                                "danceability": "float",
                                "energy": "float",
                                "valence": "float",
                                "tempo": "float"
                            }
                        },
                        "ai_insights": {
                            "type": "object",
                            "properties": {
                                "mood": "string",
                                "genre_prediction": "array",
                                "similarity_scores": "object"
                            }
                        }
                    }
                },
                "music_review": {
                    "name": "Music Review",
                    "description": "User-generated music reviews and ratings",
                    "fields": {
                        "subject_type": {
                            "type": "enum",
                            "values": ["track", "album", "artist", "playlist"],
                            "required": True
                        },
                        "subject_id": {
                            "type": "string",
                            "required": True
                        },
                        "rating": {
                            "type": "integer",
                            "min": 1,
                            "max": 5,
                            "required": True
                        },
                        "review_text": {
                            "type": "text",
                            "max_length": 2000,
                            "required": False
                        },
                        "tags": {
                            "type": "array",
                            "items": "string",
                            "required": False
                        }
                    },
                    "ai_fields": {
                        "sentiment_analysis": {
                            "enabled": True,
                            "model": "sentiment_classifier"
                        },
                        "topic_extraction": {
                            "enabled": True,
                            "model": "topic_model"
                        }
                    }
                }
            },
            "workflow_rules": {
                "auto_categorization": {
                    "enabled": True,
                    "confidence_threshold": 0.8
                },
                "content_moderation": {
                    "enabled": True,
                    "auto_flag_inappropriate": True
                },
                "ai_enhancement": {
                    "auto_generate_metadata": True,
                    "auto_tag_content": True
                }
            }
        }
        
        return self._add_metadata(template, "content_types")


class AISessionTemplateGenerator(BaseTemplateGenerator):
    """Generator for AI session templates."""
    
    def generate_session_config_template(self) -> Dict[str, Any]:
        """Generate AI session configuration template."""
        template = {
            "tenant_id": "{{ tenant_id }}",
            "session_config": {
                "default_model": "{{ default_model | default('gpt-3.5-turbo') }}",
                "max_tokens": "{{ max_tokens | default(1000) }}",
                "temperature": "{{ temperature | default(0.7) }}",
                "timeout_seconds": "{{ timeout | default(30) }}",
                "context_window": "{{ context_window | default(10) }}",
                "memory_enabled": "{{ memory_enabled | default(true) }}"
            },
            "model_configurations": {
                "gpt-3.5-turbo": {
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                },
                "gpt-4": {
                    "max_tokens": 2000,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                }
            },
            "safety_settings": {
                "content_filter": True,
                "profanity_filter": True,
                "personal_info_detection": True,
                "max_session_duration": 3600
            }
        }
        
        return self._add_metadata(template, "session_config")


class CollaborationTemplateGenerator(BaseTemplateGenerator):
    """Generator for collaboration templates."""
    
    def generate_collaboration_spaces_template(self) -> Dict[str, Any]:
        """Generate collaboration spaces template."""
        template = {
            "tenant_id": "{{ tenant_id }}",
            "space_types": {
                "music_discovery": {
                    "name": "Music Discovery",
                    "description": "Collaborative music discovery and sharing",
                    "features": ["playlist_sharing", "track_recommendations", "group_listening"],
                    "default_permissions": {
                        "create_playlists": True,
                        "share_tracks": True,
                        "comment": True,
                        "react": True
                    }
                },
                "creative_project": {
                    "name": "Creative Project",
                    "description": "Collaborative music creation and production",
                    "features": ["project_management", "file_sharing", "version_control", "feedback_system"],
                    "default_permissions": {
                        "upload_files": True,
                        "edit_content": True,
                        "manage_versions": False,
                        "approve_changes": False
                    }
                }
            },
            "default_settings": {
                "visibility": "private",
                "auto_join": False,
                "moderation": "community",
                "ai_assistance": True
            }
        }
        
        return self._add_metadata(template, "collaboration_spaces")


# Factory function to get appropriate generator
def get_template_generator(category: str) -> BaseTemplateGenerator:
    """Get template generator for specified category."""
    generators = {
        "tenant": TenantTemplateGenerator,
        "user": UserTemplateGenerator,
        "content": ContentTemplateGenerator,
        "ai_session": AISessionTemplateGenerator,
        "collaboration": CollaborationTemplateGenerator
    }
    
    generator_class = generators.get(category)
    if not generator_class:
        raise ValueError(f"Unknown template category: {category}")
    
    return generator_class()
