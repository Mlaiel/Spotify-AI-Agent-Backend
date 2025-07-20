"""
Enterprise Configuration Profiles for Spotify AI Agent Alert Algorithms

This module provides pre-configured profiles for different deployment scenarios,
algorithm specializations, and business requirements for music streaming platforms.

Author: Fahed Mlaiel (Expert Backend Developer & ML Engineer)
Version: 2.0.0 (Enterprise Edition)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import json


class DeploymentProfile(Enum):
    """Deployment profiles for different scenarios"""
    HIGH_VOLUME = "high_volume"           # For peak traffic periods
    LOW_LATENCY = "low_latency"          # For real-time critical systems
    HIGH_ACCURACY = "high_accuracy"      # For business-critical decisions
    COST_OPTIMIZED = "cost_optimized"    # For resource-constrained environments
    EXPERIMENTAL = "experimental"        # For testing new algorithms
    DISASTER_RECOVERY = "disaster_recovery"  # For failover scenarios


class GeographicProfile(Enum):
    """Geographic deployment profiles"""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST_AFRICA = "middle_east_africa"
    GLOBAL = "global"


class BusinessProfile(Enum):
    """Business-focused configuration profiles"""
    REVENUE_FOCUSED = "revenue_focused"      # Optimize for revenue protection
    USER_EXPERIENCE = "user_experience"     # Optimize for user satisfaction
    CONTENT_QUALITY = "content_quality"     # Optimize for content delivery
    FRAUD_DETECTION = "fraud_detection"     # Optimize for security
    GROWTH_OPTIMIZATION = "growth_optimization"  # Optimize for user acquisition


@dataclass
class ConfigurationProfile:
    """Complete configuration profile"""
    name: str
    description: str
    deployment_type: DeploymentProfile
    geographic_scope: GeographicProfile
    business_focus: BusinessProfile
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    performance_sla: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)


class ConfigurationProfileManager:
    """Manager for configuration profiles"""
    
    def __init__(self):
        self.profiles: Dict[str, ConfigurationProfile] = {}
        self._initialize_default_profiles()
    
    def _initialize_default_profiles(self):
        """Initialize default configuration profiles"""
        
        # High Volume Profile - For peak traffic periods (album releases, concerts)
        self.profiles["high_volume_global"] = ConfigurationProfile(
            name="High Volume Global",
            description="Optimized for handling peak traffic during major music events",
            deployment_type=DeploymentProfile.HIGH_VOLUME,
            geographic_scope=GeographicProfile.GLOBAL,
            business_focus=BusinessProfile.USER_EXPERIENCE,
            config_overrides={
                "anomaly_detection": {
                    "models": {
                        "isolation_forest": {
                            "contamination": 0.02,  # Lower threshold during peaks
                            "n_estimators": 500,    # Higher accuracy
                            "max_samples": 1024,
                            "n_jobs": -1           # Use all available cores
                        }
                    }
                },
                "caching": {
                    "ttl_strategies": {
                        "realtime_alerts": 30,     # Faster refresh during peaks
                        "ml_predictions": 300,     # Shorter cache for dynamic conditions
                        "user_behavior": 1800
                    },
                    "redis": {
                        "connection_pool_size": 100  # Increased connections
                    }
                },
                "streaming": {
                    "kafka": {
                        "batch_size": 2000,        # Larger batches for throughput
                        "max_poll_records": 5000
                    }
                }
            },
            performance_sla={
                "max_latency_ms": 50,
                "min_throughput_rps": 10000,
                "target_availability": 0.9999
            },
            resource_limits={
                "max_memory_gb": 32,
                "max_cpu_cores": 16,
                "max_storage_gb": 500
            }
        )
        
        # Low Latency Profile - For real-time critical systems
        self.profiles["low_latency_critical"] = ConfigurationProfile(
            name="Low Latency Critical",
            description="Optimized for ultra-low latency real-time audio streaming",
            deployment_type=DeploymentProfile.LOW_LATENCY,
            geographic_scope=GeographicProfile.GLOBAL,
            business_focus=BusinessProfile.USER_EXPERIENCE,
            config_overrides={
                "anomaly_detection": {
                    "models": {
                        "isolation_forest": {
                            "contamination": 0.05,
                            "n_estimators": 100,    # Fewer estimators for speed
                            "max_samples": 256      # Smaller samples for speed
                        }
                    }
                },
                "predictive_alerting": {
                    "models": {
                        "prophet": {
                            "prediction_horizon_hours": 1  # Shorter predictions
                        }
                    }
                },
                "caching": {
                    "local_cache": {
                        "max_size": 20000,      # Larger local cache
                        "default_ttl": 60       # Shorter TTL
                    }
                },
                "monitoring": {
                    "collection_interval": 10   # More frequent monitoring
                }
            },
            performance_sla={
                "max_latency_ms": 5,
                "min_throughput_rps": 5000,
                "target_availability": 0.99999
            }
        )
        
        # High Accuracy Profile - For business-critical decisions
        self.profiles["high_accuracy_business"] = ConfigurationProfile(
            name="High Accuracy Business Critical",
            description="Maximum accuracy for revenue-critical business decisions",
            deployment_type=DeploymentProfile.HIGH_ACCURACY,
            geographic_scope=GeographicProfile.GLOBAL,
            business_focus=BusinessProfile.REVENUE_FOCUSED,
            config_overrides={
                "anomaly_detection": {
                    "models": {
                        "isolation_forest": {
                            "contamination": 0.01,  # Very low false positive rate
                            "n_estimators": 1000,   # Maximum accuracy
                            "max_samples": 2048     # Large samples
                        },
                        "autoencoder": {
                            "architecture": {
                                "hidden_layers": [128, 64, 32, 16, 8, 16, 32, 64, 128]
                            },
                            "training": {
                                "epochs": 200,      # Extended training
                                "validation_split": 0.3
                            }
                        }
                    }
                },
                "severity_classification": {
                    "models": {
                        "xgboost": {
                            "n_estimators": 500,
                            "max_depth": 12,
                            "learning_rate": 0.05   # Slower learning for accuracy
                        }
                    }
                },
                "alert_correlation": {
                    "correlation_methods": {
                        "statistical": {
                            "pearson_threshold": 0.8,    # Higher correlation threshold
                            "time_window_minutes": 60    # Longer analysis window
                        }
                    }
                }
            },
            performance_sla={
                "min_accuracy": 0.99,
                "max_false_positive_rate": 0.001,
                "min_precision": 0.98
            }
        )
        
        # Cost Optimized Profile - For resource-constrained environments
        self.profiles["cost_optimized"] = ConfigurationProfile(
            name="Cost Optimized",
            description="Minimal resource usage while maintaining acceptable performance",
            deployment_type=DeploymentProfile.COST_OPTIMIZED,
            geographic_scope=GeographicProfile.GLOBAL,
            business_focus=BusinessProfile.GROWTH_OPTIMIZATION,
            config_overrides={
                "anomaly_detection": {
                    "models": {
                        "isolation_forest": {
                            "contamination": 0.1,
                            "n_estimators": 50,     # Minimal estimators
                            "max_samples": 128,     # Small samples
                            "n_jobs": 1            # Single thread
                        }
                    }
                },
                "caching": {
                    "ttl_strategies": {
                        "realtime_alerts": 300,     # Longer cache times
                        "ml_predictions": 3600,
                        "user_behavior": 21600
                    },
                    "redis": {
                        "connection_pool_size": 5   # Minimal connections
                    }
                },
                "monitoring": {
                    "collection_interval": 300     # Less frequent monitoring
                }
            },
            performance_sla={
                "max_latency_ms": 500,
                "min_throughput_rps": 100,
                "target_availability": 0.99
            },
            resource_limits={
                "max_memory_gb": 4,
                "max_cpu_cores": 2,
                "max_storage_gb": 50
            }
        )
        
        # Revenue Focused Profile - Optimized for revenue protection
        self.profiles["revenue_protection"] = ConfigurationProfile(
            name="Revenue Protection",
            description="Optimized to prevent revenue loss and detect payment issues",
            deployment_type=DeploymentProfile.HIGH_ACCURACY,
            geographic_scope=GeographicProfile.GLOBAL,
            business_focus=BusinessProfile.REVENUE_FOCUSED,
            config_overrides={
                "business_rules": {
                    "revenue_protection": {
                        "critical_revenue_threshold_per_minute": 100,  # Very sensitive
                        "user_churn_prevention_threshold": 0.01,
                        "premium_user_priority_multiplier": 5.0
                    }
                },
                "severity_classification": {
                    "business_rules": {
                        "severity_mapping": {
                            "critical": {
                                "revenue_impact_threshold": 1000,  # Lower threshold
                                "user_impact_threshold": 10000
                            }
                        }
                    }
                },
                "monitoring": {
                    "custom_metrics": {
                        "business_metrics": [
                            "revenue_per_minute",
                            "payment_failure_rate",
                            "subscription_churn_rate",
                            "premium_conversion_rate"
                        ]
                    }
                }
            },
            performance_sla={
                "max_revenue_detection_latency_ms": 10,
                "min_churn_prediction_accuracy": 0.95
            }
        )
        
        # Geographic-specific profiles
        
        # North America Profile
        self.profiles["north_america_optimized"] = ConfigurationProfile(
            name="North America Optimized",
            description="Optimized for North American market characteristics",
            deployment_type=DeploymentProfile.HIGH_VOLUME,
            geographic_scope=GeographicProfile.NORTH_AMERICA,
            business_focus=BusinessProfile.USER_EXPERIENCE,
            config_overrides={
                "music_streaming": {
                    "quality_tiers": {
                        "premium": {
                            "max_bitrate": 320,     # High quality for premium market
                            "spatial_audio": True   # Advanced features
                        }
                    },
                    "geographic_config": {
                        "regions": [{
                            "name": "North America",
                            "countries": ["US", "CA", "MX"],
                            "cdn_nodes": 25,        # High CDN density
                            "peak_hours": "18:00-23:00",
                            "local_content_cache": True
                        }]
                    }
                },
                "business_rules": {
                    "user_experience": {
                        "max_buffering_tolerance_seconds": 1,  # Lower tolerance
                        "search_result_relevance_threshold": 0.9
                    }
                }
            }
        )
        
        # Europe Profile - GDPR compliant
        self.profiles["europe_gdpr_compliant"] = ConfigurationProfile(
            name="Europe GDPR Compliant",
            description="Optimized for European market with GDPR compliance",
            deployment_type=DeploymentProfile.HIGH_ACCURACY,
            geographic_scope=GeographicProfile.EUROPE,
            business_focus=BusinessProfile.USER_EXPERIENCE,
            config_overrides={
                "security": {
                    "gdpr_compliance": {
                        "data_anonymization": True,
                        "right_to_deletion": True,
                        "consent_management": True,
                        "data_portability": True
                    },
                    "audit": {
                        "detailed_logging": True,
                        "retention_days": 2555,  # 7 years for GDPR
                        "personal_data_tracking": True
                    }
                },
                "business_rules": {
                    "content_licensing": {
                        "regional_licensing_check": True,
                        "eu_copyright_directive_compliance": True
                    }
                }
            }
        )
    
    def get_profile(self, profile_name: str) -> Optional[ConfigurationProfile]:
        """Get configuration profile by name"""
        return self.profiles.get(profile_name)
    
    def list_profiles(self) -> List[str]:
        """List all available profile names"""
        return list(self.profiles.keys())
    
    def get_profiles_by_deployment_type(self, deployment_type: DeploymentProfile) -> List[ConfigurationProfile]:
        """Get profiles filtered by deployment type"""
        return [profile for profile in self.profiles.values() 
                if profile.deployment_type == deployment_type]
    
    def get_profiles_by_geographic_scope(self, geographic_scope: GeographicProfile) -> List[ConfigurationProfile]:
        """Get profiles filtered by geographic scope"""
        return [profile for profile in self.profiles.values() 
                if profile.geographic_scope == geographic_scope]
    
    def get_profiles_by_business_focus(self, business_focus: BusinessProfile) -> List[ConfigurationProfile]:
        """Get profiles filtered by business focus"""
        return [profile for profile in self.profiles.values() 
                if profile.business_focus == business_focus]
    
    def create_custom_profile(self, 
                            name: str,
                            description: str,
                            deployment_type: DeploymentProfile,
                            geographic_scope: GeographicProfile,
                            business_focus: BusinessProfile,
                            config_overrides: Dict[str, Any]) -> ConfigurationProfile:
        """Create a custom configuration profile"""
        profile = ConfigurationProfile(
            name=name,
            description=description,
            deployment_type=deployment_type,
            geographic_scope=geographic_scope,
            business_focus=business_focus,
            config_overrides=config_overrides
        )
        
        self.profiles[name] = profile
        return profile
    
    def apply_profile_to_config(self, base_config: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
        """Apply configuration profile overrides to base configuration"""
        profile = self.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Profile not found: {profile_name}")
        
        # Deep merge configuration overrides
        result_config = self._deep_merge(base_config.copy(), profile.config_overrides)
        
        # Add profile metadata
        result_config['profile_metadata'] = {
            'name': profile.name,
            'description': profile.description,
            'deployment_type': profile.deployment_type.value,
            'geographic_scope': profile.geographic_scope.value,
            'business_focus': profile.business_focus.value,
            'performance_sla': profile.performance_sla,
            'resource_limits': profile.resource_limits
        }
        
        return result_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def export_profile(self, profile_name: str) -> str:
        """Export profile configuration as JSON"""
        profile = self.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Profile not found: {profile_name}")
        
        export_data = {
            'name': profile.name,
            'description': profile.description,
            'deployment_type': profile.deployment_type.value,
            'geographic_scope': profile.geographic_scope.value,
            'business_focus': profile.business_focus.value,
            'config_overrides': profile.config_overrides,
            'performance_sla': profile.performance_sla,
            'resource_limits': profile.resource_limits,
            'monitoring_config': profile.monitoring_config
        }
        
        return json.dumps(export_data, indent=2)
    
    def import_profile(self, profile_json: str) -> ConfigurationProfile:
        """Import profile from JSON"""
        data = json.loads(profile_json)
        
        profile = ConfigurationProfile(
            name=data['name'],
            description=data['description'],
            deployment_type=DeploymentProfile(data['deployment_type']),
            geographic_scope=GeographicProfile(data['geographic_scope']),
            business_focus=BusinessProfile(data['business_focus']),
            config_overrides=data.get('config_overrides', {}),
            performance_sla=data.get('performance_sla', {}),
            resource_limits=data.get('resource_limits', {}),
            monitoring_config=data.get('monitoring_config', {})
        )
        
        self.profiles[profile.name] = profile
        return profile


# Global profile manager instance
_profile_manager: Optional[ConfigurationProfileManager] = None


def get_profile_manager() -> ConfigurationProfileManager:
    """Get global configuration profile manager instance"""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ConfigurationProfileManager()
    return _profile_manager


def get_recommended_profile(deployment_scenario: str) -> Optional[str]:
    """Get recommended profile for deployment scenario"""
    scenario_mapping = {
        'album_release': 'high_volume_global',
        'concert_streaming': 'high_volume_global',
        'daily_operations': 'cost_optimized',
        'revenue_critical': 'revenue_protection',
        'real_time_streaming': 'low_latency_critical',
        'business_analytics': 'high_accuracy_business',
        'european_deployment': 'europe_gdpr_compliant',
        'north_american_deployment': 'north_america_optimized'
    }
    
    return scenario_mapping.get(deployment_scenario.lower())


# Export all classes and functions
__all__ = [
    'DeploymentProfile',
    'GeographicProfile', 
    'BusinessProfile',
    'ConfigurationProfile',
    'ConfigurationProfileManager',
    'get_profile_manager',
    'get_recommended_profile'
]
