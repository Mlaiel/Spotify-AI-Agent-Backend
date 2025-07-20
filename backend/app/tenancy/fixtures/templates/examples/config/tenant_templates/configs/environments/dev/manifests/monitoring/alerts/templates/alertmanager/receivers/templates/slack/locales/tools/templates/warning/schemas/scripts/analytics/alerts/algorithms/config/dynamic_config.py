"""
Dynamic Configuration Management for Spotify AI Agent Alert Algorithms

This module provides dynamic configuration capabilities including hot reloading,
A/B testing configurations, feature flags, and runtime configuration updates
for music streaming platform operations.

Author: Fahed Mlaiel (Expert Backend Developer & ML Engineer)
Version: 2.0.0 (Enterprise Edition)
"""

import asyncio
import json
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Union, Set
from threading import Lock, RLock
import logging
import weakref
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigChangeType(Enum):
    """Types of configuration changes"""
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    RELOAD = "reload"


class FeatureFlag(Enum):
    """Feature flags for dynamic feature toggling"""
    ENHANCED_ANOMALY_DETECTION = "enhanced_anomaly_detection"
    REAL_TIME_RECOMMENDATIONS = "real_time_recommendations"
    ADVANCED_AUDIO_ANALYTICS = "advanced_audio_analytics"
    PREDICTIVE_CHURN_DETECTION = "predictive_churn_detection"
    DYNAMIC_BITRATE_OPTIMIZATION = "dynamic_bitrate_optimization"
    GEOGRAPHIC_LOAD_BALANCING = "geographic_load_balancing"
    PREMIUM_USER_PRIORITIZATION = "premium_user_prioritization"
    EXPERIMENTAL_ML_MODELS = "experimental_ml_models"
    ENHANCED_SECURITY_SCANNING = "enhanced_security_scanning"
    ADVANCED_CORRELATION_ENGINE = "advanced_correlation_engine"


@dataclass
class ConfigChangeEvent:
    """Event representing a configuration change"""
    change_type: ConfigChangeType
    config_path: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestConfiguration:
    """A/B testing configuration"""
    test_name: str
    description: str
    start_date: datetime
    end_date: datetime
    traffic_split: Dict[str, float]  # variant -> percentage
    configurations: Dict[str, Dict[str, Any]]  # variant -> config
    success_metrics: List[str]
    enabled: bool = True
    user_assignment_hash_key: str = "user_id"


@dataclass
class FeatureFlagConfig:
    """Feature flag configuration"""
    flag: FeatureFlag
    enabled: bool
    rollout_percentage: float = 100.0
    target_groups: List[str] = field(default_factory=list)
    geographic_restrictions: List[str] = field(default_factory=list)
    user_segment_restrictions: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    dependencies: List[FeatureFlag] = field(default_factory=list)


class ConfigurationChangeListener(ABC):
    """Abstract base class for configuration change listeners"""
    
    @abstractmethod
    async def on_config_changed(self, event: ConfigChangeEvent) -> None:
        """Handle configuration change event"""
        pass


class DynamicConfigurationManager:
    """Advanced dynamic configuration manager with hot reloading and A/B testing"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self._configs: Dict[str, Any] = {}
        self._config_lock = RLock()
        self._listeners: List[weakref.WeakMethod] = []
        self._file_watchers: Dict[str, float] = {}
        self._ab_tests: Dict[str, ABTestConfiguration] = {}
        self._feature_flags: Dict[FeatureFlag, FeatureFlagConfig] = {}
        self._config_history: List[ConfigChangeEvent] = []
        self._max_history_size = 1000
        self._watch_enabled = False
        self._watch_task: Optional[asyncio.Task] = None
        
        # Initialize default feature flags
        self._initialize_feature_flags()
        
        logger.info("Dynamic configuration manager initialized")
    
    def _initialize_feature_flags(self):
        """Initialize default feature flags"""
        default_flags = {
            FeatureFlag.ENHANCED_ANOMALY_DETECTION: FeatureFlagConfig(
                flag=FeatureFlag.ENHANCED_ANOMALY_DETECTION,
                enabled=True,
                rollout_percentage=100.0,
                target_groups=["premium", "family"]
            ),
            FeatureFlag.REAL_TIME_RECOMMENDATIONS: FeatureFlagConfig(
                flag=FeatureFlag.REAL_TIME_RECOMMENDATIONS,
                enabled=True,
                rollout_percentage=80.0
            ),
            FeatureFlag.ADVANCED_AUDIO_ANALYTICS: FeatureFlagConfig(
                flag=FeatureFlag.ADVANCED_AUDIO_ANALYTICS,
                enabled=True,
                rollout_percentage=50.0,
                target_groups=["premium", "family", "student"]
            ),
            FeatureFlag.PREDICTIVE_CHURN_DETECTION: FeatureFlagConfig(
                flag=FeatureFlag.PREDICTIVE_CHURN_DETECTION,
                enabled=True,
                rollout_percentage=100.0
            ),
            FeatureFlag.DYNAMIC_BITRATE_OPTIMIZATION: FeatureFlagConfig(
                flag=FeatureFlag.DYNAMIC_BITRATE_OPTIMIZATION,
                enabled=True,
                rollout_percentage=75.0
            ),
            FeatureFlag.EXPERIMENTAL_ML_MODELS: FeatureFlagConfig(
                flag=FeatureFlag.EXPERIMENTAL_ML_MODELS,
                enabled=False,
                rollout_percentage=5.0,
                target_groups=["internal_users", "beta_testers"]
            )
        }
        
        self._feature_flags.update(default_flags)
    
    def register_change_listener(self, listener: ConfigurationChangeListener):
        """Register a configuration change listener"""
        weak_method = weakref.WeakMethod(listener.on_config_changed)
        self._listeners.append(weak_method)
        logger.debug(f"Registered configuration change listener: {listener.__class__.__name__}")
    
    async def _notify_listeners(self, event: ConfigChangeEvent):
        """Notify all listeners of configuration change"""
        # Clean up dead weak references
        self._listeners = [ref for ref in self._listeners if ref() is not None]
        
        for weak_method in self._listeners:
            method = weak_method()
            if method:
                try:
                    await method(event)
                except Exception as e:
                    logger.error(f"Error notifying configuration listener: {e}")
    
    def _record_change(self, event: ConfigChangeEvent):
        """Record configuration change in history"""
        with self._config_lock:
            self._config_history.append(event)
            
            # Maintain history size limit
            if len(self._config_history) > self._max_history_size:
                self._config_history = self._config_history[-self._max_history_size:]
    
    async def update_config(self, config_path: str, value: Any, source: str = "manual") -> bool:
        """Update configuration value dynamically"""
        try:
            with self._config_lock:
                # Get current value
                old_value = self._get_config_value(config_path)
                
                # Update value
                self._set_config_value(config_path, value)
                
                # Create change event
                event = ConfigChangeEvent(
                    change_type=ConfigChangeType.UPDATE,
                    config_path=config_path,
                    old_value=old_value,
                    new_value=value,
                    source=source
                )
                
                # Record change and notify
                self._record_change(event)
                await self._notify_listeners(event)
                
                logger.info(f"Configuration updated: {config_path} = {value}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating configuration {config_path}: {e}")
            return False
    
    def _get_config_value(self, config_path: str) -> Any:
        """Get configuration value by path (e.g., 'anomaly_detection.models.isolation_forest.contamination')"""
        path_parts = config_path.split('.')
        current = self._configs
        
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _set_config_value(self, config_path: str, value: Any):
        """Set configuration value by path"""
        path_parts = config_path.split('.')
        current = self._configs
        
        # Navigate to parent
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set final value
        current[path_parts[-1]] = value
    
    async def reload_config(self, config_name: str) -> bool:
        """Reload configuration from file"""
        try:
            config_file = self._find_config_file(config_name)
            if not config_file:
                logger.error(f"Configuration file not found: {config_name}")
                return False
            
            # Load new configuration
            new_config = self._load_config_from_file(config_file)
            
            with self._config_lock:
                old_config = self._configs.get(config_name, {})
                self._configs[config_name] = new_config
                
                # Create reload event
                event = ConfigChangeEvent(
                    change_type=ConfigChangeType.RELOAD,
                    config_path=config_name,
                    old_value=old_config,
                    new_value=new_config,
                    source="file_reload"
                )
                
                self._record_change(event)
                await self._notify_listeners(event)
                
                logger.info(f"Configuration reloaded: {config_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error reloading configuration {config_name}: {e}")
            return False
    
    def _find_config_file(self, config_name: str) -> Optional[Path]:
        """Find configuration file with various extensions"""
        for ext in ['yaml', 'yml', 'json']:
            config_file = self.config_dir / f"{config_name}.{ext}"
            if config_file.exists():
                return config_file
        return None
    
    def _load_config_from_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from file"""
        import yaml
        
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                return json.load(f)
        
        raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
    
    # Feature Flag Management
    
    def set_feature_flag(self, flag: FeatureFlag, enabled: bool, rollout_percentage: float = 100.0):
        """Set feature flag state"""
        with self._config_lock:
            if flag in self._feature_flags:
                old_config = self._feature_flags[flag]
                self._feature_flags[flag].enabled = enabled
                self._feature_flags[flag].rollout_percentage = rollout_percentage
            else:
                old_config = None
                self._feature_flags[flag] = FeatureFlagConfig(
                    flag=flag,
                    enabled=enabled,
                    rollout_percentage=rollout_percentage
                )
            
            logger.info(f"Feature flag updated: {flag.value} = {enabled} ({rollout_percentage}%)")
    
    def is_feature_enabled(self, flag: FeatureFlag, user_context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if feature flag is enabled for given context"""
        with self._config_lock:
            if flag not in self._feature_flags:
                return False
            
            flag_config = self._feature_flags[flag]
            
            # Check if flag is globally disabled
            if not flag_config.enabled:
                return False
            
            # Check date restrictions
            now = datetime.now()
            if flag_config.start_date and now < flag_config.start_date:
                return False
            if flag_config.end_date and now > flag_config.end_date:
                return False
            
            # Check dependencies
            for dependency in flag_config.dependencies:
                if not self.is_feature_enabled(dependency, user_context):
                    return False
            
            # Check user context restrictions
            if user_context:
                # Geographic restrictions
                if flag_config.geographic_restrictions:
                    user_region = user_context.get('region', '')
                    if user_region not in flag_config.geographic_restrictions:
                        return False
                
                # User segment restrictions
                if flag_config.target_groups:
                    user_segment = user_context.get('user_segment', '')
                    if user_segment not in flag_config.target_groups:
                        return False
                
                # Rollout percentage (based on user hash)
                if flag_config.rollout_percentage < 100.0:
                    user_id = user_context.get('user_id', '')
                    user_hash = int(hashlib.md5(f"{flag.value}:{user_id}".encode()).hexdigest(), 16)
                    rollout_threshold = (user_hash % 100) + 1
                    if rollout_threshold > flag_config.rollout_percentage:
                        return False
            
            return True
    
    def get_feature_flags_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all feature flags"""
        with self._config_lock:
            return {
                flag.value: {
                    'enabled': config.enabled,
                    'rollout_percentage': config.rollout_percentage,
                    'target_groups': config.target_groups,
                    'geographic_restrictions': config.geographic_restrictions,
                    'user_segment_restrictions': config.user_segment_restrictions,
                    'dependencies': [dep.value for dep in config.dependencies]
                }
                for flag, config in self._feature_flags.items()
            }
    
    # A/B Testing Management
    
    def create_ab_test(self, ab_test: ABTestConfiguration) -> bool:
        """Create A/B test configuration"""
        try:
            with self._config_lock:
                # Validate traffic split
                total_traffic = sum(ab_test.traffic_split.values())
                if abs(total_traffic - 100.0) > 0.01:
                    raise ValueError(f"Traffic split must sum to 100%, got {total_traffic}")
                
                self._ab_tests[ab_test.test_name] = ab_test
                
                logger.info(f"A/B test created: {ab_test.test_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating A/B test {ab_test.test_name}: {e}")
            return False
    
    def get_ab_test_variant(self, test_name: str, user_context: Dict[str, Any]) -> Optional[str]:
        """Get A/B test variant for user"""
        with self._config_lock:
            if test_name not in self._ab_tests:
                return None
            
            ab_test = self._ab_tests[test_name]
            
            # Check if test is enabled and within date range
            if not ab_test.enabled:
                return None
            
            now = datetime.now()
            if now < ab_test.start_date or now > ab_test.end_date:
                return None
            
            # Determine variant based on user hash
            user_identifier = user_context.get(ab_test.user_assignment_hash_key, '')
            user_hash = int(hashlib.md5(f"{test_name}:{user_identifier}".encode()).hexdigest(), 16)
            hash_percentage = (user_hash % 100) + 1
            
            # Assign variant based on traffic split
            cumulative_percentage = 0
            for variant, percentage in ab_test.traffic_split.items():
                cumulative_percentage += percentage
                if hash_percentage <= cumulative_percentage:
                    return variant
            
            return None
    
    def get_ab_test_config(self, test_name: str, variant: str) -> Optional[Dict[str, Any]]:
        """Get configuration for A/B test variant"""
        with self._config_lock:
            if test_name not in self._ab_tests:
                return None
            
            ab_test = self._ab_tests[test_name]
            return ab_test.configurations.get(variant)
    
    def get_active_ab_tests(self) -> List[str]:
        """Get list of active A/B test names"""
        now = datetime.now()
        with self._config_lock:
            return [
                test_name for test_name, ab_test in self._ab_tests.items()
                if ab_test.enabled and ab_test.start_date <= now <= ab_test.end_date
            ]
    
    # Configuration History and Monitoring
    
    def get_config_history(self, limit: int = 100) -> List[ConfigChangeEvent]:
        """Get configuration change history"""
        with self._config_lock:
            return self._config_history[-limit:]
    
    def get_config_diff(self, config_path: str, start_time: datetime, end_time: datetime) -> List[ConfigChangeEvent]:
        """Get configuration changes within time range"""
        with self._config_lock:
            return [
                event for event in self._config_history
                if event.config_path == config_path and start_time <= event.timestamp <= end_time
            ]
    
    async def start_file_watching(self):
        """Start watching configuration files for changes"""
        if self._watch_enabled:
            return
        
        self._watch_enabled = True
        self._watch_task = asyncio.create_task(self._file_watch_loop())
        logger.info("Started configuration file watching")
    
    async def stop_file_watching(self):
        """Stop watching configuration files"""
        self._watch_enabled = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped configuration file watching")
    
    async def _file_watch_loop(self):
        """File watching loop"""
        while self._watch_enabled:
            try:
                await self._check_file_changes()
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in file watch loop: {e}")
                await asyncio.sleep(10)  # Longer delay on error
    
    async def _check_file_changes(self):
        """Check for configuration file changes"""
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            current_mtime = config_file.stat().st_mtime
            
            if config_name in self._file_watchers:
                if current_mtime > self._file_watchers[config_name]:
                    logger.info(f"Configuration file changed: {config_file}")
                    await self.reload_config(config_name)
            
            self._file_watchers[config_name] = current_mtime
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export complete configuration state"""
        with self._config_lock:
            return {
                'configs': self._configs,
                'feature_flags': {
                    flag.value: {
                        'enabled': config.enabled,
                        'rollout_percentage': config.rollout_percentage,
                        'target_groups': config.target_groups
                    }
                    for flag, config in self._feature_flags.items()
                },
                'ab_tests': {
                    name: {
                        'description': test.description,
                        'traffic_split': test.traffic_split,
                        'start_date': test.start_date.isoformat(),
                        'end_date': test.end_date.isoformat(),
                        'enabled': test.enabled
                    }
                    for name, test in self._ab_tests.items()
                },
                'export_timestamp': datetime.now().isoformat()
            }


# Global dynamic configuration manager
_dynamic_config_manager: Optional[DynamicConfigurationManager] = None


def get_dynamic_config_manager() -> DynamicConfigurationManager:
    """Get global dynamic configuration manager instance"""
    global _dynamic_config_manager
    if _dynamic_config_manager is None:
        _dynamic_config_manager = DynamicConfigurationManager()
    return _dynamic_config_manager


def is_feature_enabled(flag: FeatureFlag, user_context: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to check feature flag"""
    manager = get_dynamic_config_manager()
    return manager.is_feature_enabled(flag, user_context)


async def update_config_value(config_path: str, value: Any) -> bool:
    """Convenience function to update configuration value"""
    manager = get_dynamic_config_manager()
    return await manager.update_config(config_path, value)


# Export all classes and functions
__all__ = [
    'ConfigChangeType',
    'FeatureFlag',
    'ConfigChangeEvent',
    'ABTestConfiguration',
    'FeatureFlagConfig',
    'ConfigurationChangeListener',
    'DynamicConfigurationManager',
    'get_dynamic_config_manager',
    'is_feature_enabled',
    'update_config_value'
]
