# -*- coding: utf-8 -*-
"""
Ultra-Advanced Core Alerts Module - Enterprise-Grade Alert Processing System
==========================================================================

Ce module fournit la base fondamentale pour un système de traitement d'alertes
multi-tenant de qualité industrielle avec intelligence artificielle, corrélation
temps réel, analyse prédictive et capacités de remédiation automatisée.

Composants Principaux:
- AlertEngine: Traitement d'alertes haute performance avec architecture distribuée  
- RuleEngine: Évaluation de règles enrichie par ML avec logique floue et patterns temporels
- NotificationHub: Système de notification multi-canal avec routage intelligent
- EscalationManager: Escalation pilotée par IA avec rotation d'équipes et gestion SLA
- CorrelationEngine: Corrélation d'événements avancée avec reconnaissance de motifs
- SuppressionManager: Suppression d'alertes intelligente avec déduplication basée ML
- RemediationEngine: Remédiation automatisée avec orchestration de workflows
- AnalyticsEngine: Analytics temps réel avec insights prédictifs
- MetricsCollector: Collecte et agrégation avancée de métriques
- StateManager: Gestion d'état distribuée pour le cycle de vie des alertes
- ComplianceManager: Vérification automatisée de conformité et rapports
- SecurityManager: Couche de sécurité avec chiffrement et pistes d'audit

Fonctionnalités Enterprise:
- Isolation multi-tenant avec séparation complète des données
- Scaling horizontal avec traitement distribué
- Analytics temps réel et insights alimentés par ML
- Génération automatisée de conformité et pistes d'audit
- Design sécurisé avec chiffrement end-to-end
- Monitoring SLA et application automatisée
- Disaster recovery et haute disponibilité
- Monitoring et observabilité avancés

Version: 5.0.0
Auteur: Fahed Mlaiel (Lead Dev + Architecte IA)
Architecture: Microservices avec Design Event-Driven
"""

import asyncio
import logging
import time
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import (
    Dict, List, Optional, Any, Union, Callable, Protocol, 
    TypeVar, Generic, AsyncIterator, Set, Tuple
)
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum, auto
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import weakref
import gc

# External dependencies for advanced features
try:
    import redis
    import psycopg2
    import motor.motor_asyncio
    import aiohttp
    import prometheus_client
    import cryptography
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    import numpy as np
    import pandas as pd
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    logging.warning("Advanced features disabled - install optional dependencies for full functionality")

# Configure logging with structured output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/spotify/alerts-core.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
AlertID = str
TenantID = str
RuleID = str
NotificationID = str

class AlertSeverity(Enum):
    """Alert severity levels with numeric ordering."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1

class AlertStatus(Enum):
    """Alert lifecycle status."""
    NEW = auto()
    PROCESSING = auto()
    ACKNOWLEDGED = auto()
    ESCALATED = auto()
    RESOLVED = auto()
    SUPPRESSED = auto()
    EXPIRED = auto()

class NotificationChannel(Enum):
    """Supported notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    SMS = "sms"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SERVICENOW = "servicenow"
    TELEGRAM = "telegram"

class ProcessingPriority(Enum):
    """Processing priority levels."""
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BULK = 5

# Core data structures
@dataclass
class AlertMetadata:
    """Enhanced alert metadata with versioning and lineage."""
    tenant_id: TenantID
    alert_id: AlertID
    version: int = 1
    parent_alert_id: Optional[AlertID] = None
    correlation_id: Optional[str] = None
    source_system: str = "spotify-ai-agent"
    tags: Dict[str, Any] = field(default_factory=dict)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: Optional[int] = None
    compliance_flags: Set[str] = field(default_factory=set)

@dataclass
class AlertContext:
    """Rich context information for alerts."""
    environment: str = "production"
    service_name: str = ""
    component: str = ""
    instance_id: str = ""
    cluster: str = ""
    region: str = ""
    availability_zone: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    traces: List[str] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    related_alerts: List[AlertID] = field(default_factory=list)
    business_impact: Optional[str] = None
    affected_users: Optional[int] = None
    estimated_cost: Optional[float] = None

@dataclass
class Alert:
    """Core alert data structure with comprehensive information."""
    metadata: AlertMetadata
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.NEW
    context: AlertContext = field(default_factory=AlertContext)
    conditions: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    escalated_at: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None
    auto_resolve_after: Optional[timedelta] = None
    ml_confidence: float = 0.0
    anomaly_score: float = 0.0
    correlation_score: float = 0.0
    business_priority: int = 3  # 1=highest, 5=lowest
    
    def __post_init__(self):
        """Post-initialization validation and defaults."""
        if not self.metadata.alert_id:
            self.metadata.alert_id = str(uuid.uuid4())
        
        # Update timestamp
        self.metadata.updated_at = datetime.utcnow()
        
        # Validate severity and priority alignment
        if self.severity == AlertSeverity.CRITICAL and self.business_priority > 2:
            self.business_priority = 1
    
    @property
    def age(self) -> timedelta:
        """Calculate alert age."""
        return datetime.utcnow() - self.metadata.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if alert has expired."""
        if self.metadata.ttl_seconds:
            return self.age.total_seconds() > self.metadata.ttl_seconds
        return False
    
    @property
    def unique_key(self) -> str:
        """Generate unique key for deduplication."""
        key_parts = [
            self.metadata.tenant_id,
            self.title,
            self.context.service_name,
            self.context.component,
            str(sorted(self.labels.items()))
        ]
        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()

import logging
import time
from typing import Dict, List, Optional, Any, Union, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Protocol definitions for dependency injection
class AlertProcessor(Protocol):
    """Protocol for alert processing components."""
    
    async def process_alert(self, alert: Alert) -> bool:
        """Process a single alert."""
        ...
    
    async def process_batch(self, alerts: List[Alert]) -> List[bool]:
        """Process multiple alerts in batch."""
        ...

class RuleEvaluator(Protocol):
    """Protocol for rule evaluation components."""
    
    async def evaluate_rules(self, alert: Alert, rules: List[Any]) -> List[bool]:
        """Evaluate rules against an alert."""
        ...

class NotificationSender(Protocol):
    """Protocol for notification sending components."""
    
    async def send_notification(self, alert: Alert, channel: NotificationChannel, recipients: List[str]) -> bool:
        """Send notification through specified channel."""
        ...

# Advanced configuration system
@dataclass
class CoreAlertsConfig:
    """Comprehensive configuration for core alerts system."""
    
    # Processing configuration
    max_concurrent_alerts: int = 10000
    batch_processing_size: int = 100
    processing_timeout_seconds: int = 60
    retry_attempts: int = 5
    retry_backoff_multiplier: float = 2.0
    max_retry_delay_seconds: int = 300
    
    # Queue configuration
    queue_size_limit: int = 100000
    priority_queue_enabled: bool = True
    dead_letter_queue_enabled: bool = True
    queue_persistence_enabled: bool = True
    
    # ML and analytics
    enable_ml_processing: bool = True
    ml_model_update_interval_hours: int = 6
    anomaly_detection_threshold: float = 0.8
    correlation_window_minutes: int = 30
    pattern_learning_enabled: bool = True
    
    # Notifications
    notification_rate_limit_per_minute: int = 1000
    notification_batch_size: int = 50
    notification_retry_attempts: int = 3
    notification_timeout_seconds: int = 30
    
    # Escalation
    default_escalation_timeout_minutes: int = 15
    max_escalation_levels: int = 5
    escalation_path_validation: bool = True
    
    # Compliance and security
    encryption_enabled: bool = True
    audit_trail_enabled: bool = True
    gdpr_compliance_mode: bool = True
    data_retention_days: int = 90
    anonymization_enabled: bool = True
    
    # Performance and monitoring
    metrics_collection_enabled: bool = True
    detailed_profiling_enabled: bool = False
    performance_monitoring_interval_seconds: int = 60
    health_check_interval_seconds: int = 30
    
    # High availability
    cluster_mode_enabled: bool = False
    replication_factor: int = 3
    auto_failover_enabled: bool = True
    backup_interval_minutes: int = 30
    
    # Storage configuration
    redis_url: str = "redis://localhost:6379/0"
    postgresql_url: str = "postgresql://localhost:5432/spotify_alerts"
    mongodb_url: str = "mongodb://localhost:27017/spotify_alerts"
    
    # External integrations
    prometheus_enabled: bool = True
    jaeger_tracing_enabled: bool = True
    elasticsearch_logging_enabled: bool = False

# Global configuration instance
config = CoreAlertsConfig()

# Metrics collection
class MetricsCollector:
    """Advanced metrics collection for monitoring and observability."""
    
    def __init__(self):
        self.counters = {}
        self.histograms = {}
        self.gauges = {}
        self._lock = threading.Lock()
        
        if ADVANCED_FEATURES_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.alert_processed_total = prometheus_client.Counter(
            'alerts_processed_total', 
            'Total number of alerts processed',
            ['tenant_id', 'severity', 'status']
        )
        
        self.alert_processing_duration = prometheus_client.Histogram(
            'alert_processing_duration_seconds',
            'Time spent processing alerts',
            ['tenant_id', 'severity']
        )
        
        self.active_alerts_gauge = prometheus_client.Gauge(
            'active_alerts_total',
            'Number of active alerts',
            ['tenant_id', 'severity']
        )
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None, value: float = 1):
        """Increment a counter metric."""
        with self._lock:
            key = f"{name}:{labels or {}}"
            self.counters[key] = self.counters.get(key, 0) + value
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram observation."""
        with self._lock:
            key = f"{name}:{labels or {}}"
            if key not in self.histograms:
                self.histograms[key] = []
            self.histograms[key].append(value)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge value."""
        with self._lock:
            key = f"{name}:{labels or {}}"
            self.gauges[key] = value

# Global metrics instance
metrics = MetricsCollector()

# Performance monitoring
class PerformanceMonitor:
    """Real-time performance monitoring and optimization."""
    
    def __init__(self):
        self.start_time = time.time()
        self.processed_alerts = 0
        self.processing_times = []
        self.error_count = 0
        self.memory_usage = []
        self._lock = threading.Lock()
    
    @asynccontextmanager
    async def measure_processing_time(self, alert: Alert):
        """Context manager to measure processing time."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            with self._lock:
                self.processing_times.append(duration)
                self.processed_alerts += 1
            
            # Record metrics
            metrics.observe_histogram(
                'alert_processing_duration',
                duration,
                {
                    'tenant_id': alert.metadata.tenant_id,
                    'severity': alert.severity.name
                }
            )
    
    def record_error(self):
        """Record processing error."""
        with self._lock:
            self.error_count += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self._lock:
            uptime = time.time() - self.start_time
            avg_processing_time = (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0
            )
            
            return {
                'uptime_seconds': uptime,
                'processed_alerts': self.processed_alerts,
                'processing_rate_per_second': self.processed_alerts / uptime if uptime > 0 else 0,
                'average_processing_time_ms': avg_processing_time * 1000,
                'error_count': self.error_count,
                'error_rate': self.error_count / max(self.processed_alerts, 1),
                'memory_usage_mb': self._get_memory_usage()
            }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

# Global performance monitor
performance_monitor = PerformanceMonitor()

# State management
class StateManager:
    """Distributed state management for alert system."""
    
    def __init__(self):
        self.local_cache = {}
        self.redis_client = None
        self._lock = threading.Lock()
        
        if ADVANCED_FEATURES_AVAILABLE:
            self._init_redis_connection()
    
    def _init_redis_connection(self):
        """Initialize Redis connection for distributed state."""
        try:
            self.redis_client = redis.Redis.from_url(config.redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established for state management")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using local state only.")
    
    async def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        # Try local cache first
        with self._lock:
            if key in self.local_cache:
                return self.local_cache[key]
        
        # Try Redis if available
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    decoded_value = json.loads(value.decode())
                    # Cache locally
                    with self._lock:
                        self.local_cache[key] = decoded_value
                    return decoded_value
            except Exception as e:
                logger.warning(f"Redis get failed for key {key}: {e}")
        
        return default
    
    async def set_state(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set state value."""
        # Update local cache
        with self._lock:
            self.local_cache[key] = value
        
        # Update Redis if available
        if self.redis_client:
            try:
                serialized_value = json.dumps(value, default=str)
                if ttl_seconds:
                    self.redis_client.setex(key, ttl_seconds, serialized_value)
                else:
                    self.redis_client.set(key, serialized_value)
            except Exception as e:
                logger.warning(f"Redis set failed for key {key}: {e}")

# Global state manager
state_manager = StateManager()

# Component registry for dependency injection
class ComponentRegistry:
    """Registry for dependency injection and component management."""
    
    def __init__(self):
        self._components = {}
        self._singletons = {}
        self._lock = threading.Lock()
    
    def register(self, interface: type, implementation: type, singleton: bool = True):
        """Register a component implementation."""
        with self._lock:
            self._components[interface] = (implementation, singleton)
    
    def get(self, interface: type, *args, **kwargs):
        """Get component instance."""
        with self._lock:
            if interface not in self._components:
                raise ValueError(f"No implementation registered for {interface}")
            
            implementation, singleton = self._components[interface]
            
            if singleton:
                if interface not in self._singletons:
                    self._singletons[interface] = implementation(*args, **kwargs)
                return self._singletons[interface]
            else:
                return implementation(*args, **kwargs)
    
    def clear(self):
        """Clear all registrations."""
        with self._lock:
            self._components.clear()
            self._singletons.clear()

# Global component registry
registry = ComponentRegistry()

# Health check system
class HealthChecker:
    """Comprehensive health checking for the alert system."""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
        self._lock = threading.Lock()
    
    def register_check(self, name: str, check_func: Callable[[], bool], interval_seconds: int = 60):
        """Register a health check."""
        with self._lock:
            self.checks[name] = (check_func, interval_seconds)
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        current_time = time.time()
        
        for name, (check_func, interval) in self.checks.items():
            last_check = self.last_check_time.get(name, 0)
            
            if current_time - last_check >= interval:
                try:
                    result = check_func()
                    results[name] = {
                        'status': 'healthy' if result else 'unhealthy',
                        'last_check': current_time,
                        'details': None
                    }
                except Exception as e:
                    results[name] = {
                        'status': 'error',
                        'last_check': current_time,
                        'details': str(e)
                    }
                
                with self._lock:
                    self.last_check_time[name] = current_time
            else:
                # Use cached result
                results[name] = {
                    'status': 'cached',
                    'last_check': last_check,
                    'details': 'Using cached result'
                }
        
        return results

# Global health checker
health_checker = HealthChecker()

# Initialize default health checks
def _init_default_health_checks():
    """Initialize default system health checks."""
    
    def check_memory_usage():
        """Check system memory usage."""
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90
    
    def check_disk_usage():
        """Check disk usage."""
        import psutil
        disk = psutil.disk_usage('/')
        return disk.percent < 85
    
    def check_redis_connection():
        """Check Redis connectivity."""
        if state_manager.redis_client:
            try:
                state_manager.redis_client.ping()
                return True
            except:
                return False
        return True  # If Redis not configured, consider healthy
    
    health_checker.register_check('memory_usage', check_memory_usage, 30)
    health_checker.register_check('disk_usage', check_disk_usage, 60)
    health_checker.register_check('redis_connection', check_redis_connection, 15)

# Initialize health checks
_init_default_health_checks()

# Utility functions
def generate_correlation_id() -> str:
    """Generate a unique correlation ID."""
    return f"corr_{int(time.time())}_{uuid.uuid4().hex[:8]}"

def serialize_alert(alert: Alert) -> Dict[str, Any]:
    """Serialize alert to dictionary."""
    return asdict(alert)

def deserialize_alert(data: Dict[str, Any]) -> Alert:
    """Deserialize alert from dictionary."""
    # Handle datetime fields
    for field in ['created_at', 'updated_at', 'resolved_at', 'acknowledged_at', 'escalated_at', 'suppressed_until']:
        if field in data.get('metadata', {}) and isinstance(data['metadata'][field], str):
            data['metadata'][field] = datetime.fromisoformat(data['metadata'][field])
    
    return Alert(**data)

def calculate_alert_hash(alert: Alert) -> str:
    """Calculate hash for alert deduplication."""
    return alert.unique_key

async def validate_alert(alert: Alert) -> List[str]:
    """Validate alert and return list of validation errors."""
    errors = []
    
    if not alert.title.strip():
        errors.append("Alert title cannot be empty")
    
    if not alert.description.strip():
        errors.append("Alert description cannot be empty")
    
    if not alert.metadata.tenant_id:
        errors.append("Tenant ID is required")
    
    if alert.severity not in AlertSeverity:
        errors.append("Invalid alert severity")
    
    return errors

# Export public interface
__all__ = [
    # Core classes
    'Alert',
    'AlertMetadata', 
    'AlertContext',
    'AlertSeverity',
    'AlertStatus',
    'NotificationChannel',
    'ProcessingPriority',
    
    # Protocols
    'AlertProcessor',
    'RuleEvaluator', 
    'NotificationSender',
    
    # Configuration
    'CoreAlertsConfig',
    'config',
    
    # Global instances
    'metrics',
    'performance_monitor',
    'state_manager',
    'registry',
    'health_checker',
    
    # Utilities
    'generate_correlation_id',
    'serialize_alert',
    'deserialize_alert', 
    'calculate_alert_hash',
    'validate_alert'
]

# Module initialization
logger.info("Core alerts module initialized successfully")
logger.info(f"Advanced features available: {ADVANCED_FEATURES_AVAILABLE}")
logger.info(f"Configuration: {config}")

# Cleanup on module unload
import atexit

def _cleanup():
    """Cleanup resources on module unload."""
    if state_manager.redis_client:
        state_manager.redis_client.close()
    logger.info("Core alerts module cleanup completed")

atexit.register(_cleanup)

# Interfaces pour les moteurs
class AlertEngineInterface(Protocol):
    """Interface pour le moteur d'alertes"""
    def process_alert(self, alert: Any) -> bool: ...
    def get_processing_stats(self) -> Dict[str, Any]: ...

class RuleEngineInterface(Protocol):
    """Interface pour le moteur de règles"""
    def evaluate_rules(self, event: Any) -> List[Any]: ...
    def add_rule(self, rule: Any) -> bool: ...
    def remove_rule(self, rule_id: str) -> bool: ...

class NotificationInterface(Protocol):
    """Interface pour les notifications"""
    def send_notification(self, alert: Any, channel: str) -> bool: ...
    def get_channel_status(self, channel: str) -> Dict[str, Any]: ...

class EscalationInterface(Protocol):
    """Interface pour l'escalade"""
    def check_escalation(self, alert: Any) -> bool: ...
    def escalate_alert(self, alert: Any) -> bool: ...

class CorrelationInterface(Protocol):
    """Interface pour la corrélation"""
    def correlate_alert(self, alert: Any) -> Optional[str]: ...
    def get_correlation_clusters(self) -> List[Dict[str, Any]]: ...

class SuppressionInterface(Protocol):
    """Interface pour la suppression"""
    def should_suppress(self, alert: Any) -> bool: ...
    def add_suppression_rule(self, rule: Any) -> bool: ...

class RemediationInterface(Protocol):
    """Interface pour la remédiation"""
    def remediate_alert(self, alert: Any) -> bool: ...
    def rollback_remediation(self, action_id: str) -> bool: ...

class AnalyticsInterface(Protocol):
    """Interface pour les analytics"""
    def record_metric(self, name: str, value: float, labels: Dict[str, str]) -> None: ...
    def generate_report(self, timeframe_hours: int) -> Dict[str, Any]: ...

# Classes de base abstraites
class BaseEngine(ABC):
    """Classe de base pour tous les moteurs"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le moteur de base
        
        Args:
            config: Configuration du moteur
        """
        self.config = config
        self.is_running = False
        self.metrics = {}
        self.start_time = time.time()
        
        logger.info(f"{self.__class__.__name__} initialisé")
    
    @abstractmethod
    def start(self) -> bool:
        """Démarre le moteur"""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Arrête le moteur"""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Vérification de santé du moteur"""
        pass
    
    def get_uptime(self) -> float:
        """Retourne l'uptime en secondes"""
        return time.time() - self.start_time
    
    def update_metric(self, name: str, value: Union[int, float]):
        """Met à jour une métrique"""
        self.metrics[name] = {
            'value': value,
            'timestamp': time.time()
        }

# Import des moteurs (avec fallback)
try:
    from .alert_engine import AlertEngine
    logger.info("AlertEngine importé")
except ImportError:
    logger.warning("AlertEngine non disponible, utilisation de mock")
    class AlertEngine(BaseEngine):
        def start(self): return True
        def stop(self): return True
        def health_check(self): return {'status': 'mock'}
        def process_alert(self, alert): return True
        def get_processing_stats(self): return {}

try:
    from .rule_engine import RuleEngine
    logger.info("RuleEngine importé")
except ImportError:
    logger.warning("RuleEngine non disponible, utilisation de mock")
    class RuleEngine(BaseEngine):
        def start(self): return True
        def stop(self): return True
        def health_check(self): return {'status': 'mock'}
        def evaluate_rules(self, event): return []
        def add_rule(self, rule): return True
        def remove_rule(self, rule_id): return True

try:
    from .notification_hub import NotificationHub
    logger.info("NotificationHub importé")
except ImportError:
    logger.warning("NotificationHub non disponible, utilisation de mock")
    class NotificationHub(BaseEngine):
        def start(self): return True
        def stop(self): return True
        def health_check(self): return {'status': 'mock'}
        def send_notification(self, alert, channel): return True
        def get_channel_status(self, channel): return {}

try:
    from .escalation_manager import EscalationManager
    logger.info("EscalationManager importé")
except ImportError:
    logger.warning("EscalationManager non disponible, utilisation de mock")
    class EscalationManager(BaseEngine):
        def start(self): return True
        def stop(self): return True
        def health_check(self): return {'status': 'mock'}
        def check_escalation(self, alert): return False
        def escalate_alert(self, alert): return True

try:
    from .correlation_engine import CorrelationEngine
    logger.info("CorrelationEngine importé")
except ImportError:
    logger.warning("CorrelationEngine non disponible, utilisation de mock")
    class CorrelationEngine(BaseEngine):
        def start(self): return True
        def stop(self): return True
        def health_check(self): return {'status': 'mock'}
        def correlate_alert(self, alert): return None
        def get_correlation_clusters(self): return []

try:
    from .suppression_manager import SuppressionManager
    logger.info("SuppressionManager importé")
except ImportError:
    logger.warning("SuppressionManager non disponible, utilisation de mock")
    class SuppressionManager(BaseEngine):
        def start(self): return True
        def stop(self): return True
        def health_check(self): return {'status': 'mock'}
        def should_suppress(self, alert): return False
        def add_suppression_rule(self, rule): return True

try:
    from .remediation_engine import RemediationEngine
    logger.info("RemediationEngine importé")
except ImportError:
    logger.warning("RemediationEngine non disponible, utilisation de mock")
    class RemediationEngine(BaseEngine):
        def start(self): return True
        def stop(self): return True
        def health_check(self): return {'status': 'mock'}
        def remediate_alert(self, alert): return True
        def rollback_remediation(self, action_id): return True

try:
    from .analytics_engine import AnalyticsEngine
    logger.info("AnalyticsEngine importé")
except ImportError:
    logger.warning("AnalyticsEngine non disponible, utilisation de mock")
    class AnalyticsEngine(BaseEngine):
        def start(self): return True
        def stop(self): return True
        def health_check(self): return {'status': 'mock'}
        def record_metric(self, name, value, labels): pass
        def generate_report(self, timeframe_hours): return {}

# Factory pour créer les moteurs
class EngineFactory:
    """Factory pour créer et configurer les moteurs"""
    
    @staticmethod
    def create_alert_engine(config: Dict[str, Any]) -> AlertEngine:
        """Crée un AlertEngine"""
        return AlertEngine(config.get('alert_engine', {}))
    
    @staticmethod
    def create_rule_engine(config: Dict[str, Any]) -> RuleEngine:
        """Crée un RuleEngine"""
        return RuleEngine(config.get('rule_engine', {}))
    
    @staticmethod
    def create_notification_hub(config: Dict[str, Any]) -> NotificationHub:
        """Crée un NotificationHub"""
        return NotificationHub(config.get('notification_hub', {}))
    
    @staticmethod
    def create_escalation_manager(config: Dict[str, Any]) -> EscalationManager:
        """Crée un EscalationManager"""
        return EscalationManager(config.get('escalation_manager', {}))
    
    @staticmethod
    def create_correlation_engine(config: Dict[str, Any]) -> CorrelationEngine:
        """Crée un CorrelationEngine"""
        return CorrelationEngine(config.get('correlation_engine', {}))
    
    @staticmethod
    def create_suppression_manager(config: Dict[str, Any]) -> SuppressionManager:
        """Crée un SuppressionManager"""
        return SuppressionManager(config.get('suppression_manager', {}))
    
    @staticmethod
    def create_remediation_engine(config: Dict[str, Any]) -> RemediationEngine:
        """Crée un RemediationEngine"""
        return RemediationEngine(config.get('remediation_engine', {}))
    
    @staticmethod
    def create_analytics_engine(config: Dict[str, Any]) -> AnalyticsEngine:
        """Crée un AnalyticsEngine"""
        return AnalyticsEngine(config.get('analytics_engine', {}))
    
    @staticmethod
    def create_all_engines(config: Dict[str, Any]) -> Dict[str, BaseEngine]:
        """Crée tous les moteurs"""
        engines = {
            'alert_engine': EngineFactory.create_alert_engine(config),
            'rule_engine': EngineFactory.create_rule_engine(config),
            'notification_hub': EngineFactory.create_notification_hub(config),
            'escalation_manager': EngineFactory.create_escalation_manager(config),
            'correlation_engine': EngineFactory.create_correlation_engine(config),
            'suppression_manager': EngineFactory.create_suppression_manager(config),
            'remediation_engine': EngineFactory.create_remediation_engine(config),
            'analytics_engine': EngineFactory.create_analytics_engine(config)
        }
        
        logger.info(f"Créé {len(engines)} moteurs d'alertes")
        return engines

# Gestionnaire d'orchestration des moteurs
class EngineOrchestrator:
    """Orchestrateur pour gérer tous les moteurs"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise l'orchestrateur
        
        Args:
            config: Configuration complète
        """
        self.config = config
        self.engines = EngineFactory.create_all_engines(config)
        self.is_running = False
        
        logger.info("EngineOrchestrator initialisé")
    
    def start_all(self) -> bool:
        """Démarre tous les moteurs"""
        try:
            success_count = 0
            for name, engine in self.engines.items():
                try:
                    if engine.start():
                        success_count += 1
                        logger.info(f"Moteur {name} démarré avec succès")
                    else:
                        logger.error(f"Échec démarrage moteur {name}")
                except Exception as e:
                    logger.error(f"Erreur démarrage moteur {name}: {e}")
            
            self.is_running = success_count == len(self.engines)
            
            if self.is_running:
                logger.info("Tous les moteurs démarrés avec succès")
            else:
                logger.warning(f"Seulement {success_count}/{len(self.engines)} moteurs démarrés")
            
            return self.is_running
            
        except Exception as e:
            logger.error(f"Erreur démarrage orchestrateur: {e}")
            return False
    
    def stop_all(self) -> bool:
        """Arrête tous les moteurs"""
        try:
            success_count = 0
            for name, engine in self.engines.items():
                try:
                    if engine.stop():
                        success_count += 1
                        logger.info(f"Moteur {name} arrêté avec succès")
                    else:
                        logger.error(f"Échec arrêt moteur {name}")
                except Exception as e:
                    logger.error(f"Erreur arrêt moteur {name}: {e}")
            
            self.is_running = False
            
            if success_count == len(self.engines):
                logger.info("Tous les moteurs arrêtés avec succès")
                return True
            else:
                logger.warning(f"Seulement {success_count}/{len(self.engines)} moteurs arrêtés")
                return False
                
        except Exception as e:
            logger.error(f"Erreur arrêt orchestrateur: {e}")
            return False
    
    def health_check_all(self) -> Dict[str, Any]:
        """Vérification de santé de tous les moteurs"""
        health_status = {
            'overall_status': 'healthy',
            'engines': {},
            'timestamp': time.time()
        }
        
        unhealthy_count = 0
        
        for name, engine in self.engines.items():
            try:
                engine_health = engine.health_check()
                health_status['engines'][name] = engine_health
                
                if engine_health.get('status') != 'healthy':
                    unhealthy_count += 1
                    
            except Exception as e:
                health_status['engines'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                unhealthy_count += 1
        
        # Statut global
        if unhealthy_count == 0:
            health_status['overall_status'] = 'healthy'
        elif unhealthy_count < len(self.engines):
            health_status['overall_status'] = 'degraded'
        else:
            health_status['overall_status'] = 'unhealthy'
        
        health_status['healthy_engines'] = len(self.engines) - unhealthy_count
        health_status['total_engines'] = len(self.engines)
        
        return health_status
    
    def get_engine(self, name: str) -> Optional[BaseEngine]:
        """Récupère un moteur par nom"""
        return self.engines.get(name)
    
    def restart_engine(self, name: str) -> bool:
        """Redémarre un moteur spécifique"""
        engine = self.engines.get(name)
        if not engine:
            logger.error(f"Moteur {name} introuvable")
            return False
        
        try:
            logger.info(f"Redémarrage moteur {name}")
            engine.stop()
            return engine.start()
        except Exception as e:
            logger.error(f"Erreur redémarrage moteur {name}: {e}")
            return False

# Export des principales classes
__all__ = [
    'CORE_ALERTS_CONFIG',
    'BaseEngine',
    'AlertEngineInterface',
    'RuleEngineInterface', 
    'NotificationInterface',
    'EscalationInterface',
    'CorrelationInterface',
    'SuppressionInterface',
    'RemediationInterface',
    'AnalyticsInterface',
    'AlertEngine',
    'RuleEngine',
    'NotificationHub',
    'EscalationManager',
    'CorrelationEngine',
    'SuppressionManager',
    'RemediationEngine',
    'AnalyticsEngine',
    'EngineFactory',
    'EngineOrchestrator'
]

# Instance globale de l'orchestrateur
_orchestrator_instance = None

def get_engine_orchestrator(config: Optional[Dict[str, Any]] = None) -> EngineOrchestrator:
    """
    Retourne l'instance globale de l'orchestrateur
    
    Args:
        config: Configuration optionnelle
        
    Returns:
        Instance de l'orchestrateur
    """
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        final_config = config or CORE_ALERTS_CONFIG
        _orchestrator_instance = EngineOrchestrator(final_config)
    
    return _orchestrator_instance

def start_alerts_engines(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Fonction de convenance pour démarrer tous les moteurs
    
    Args:
        config: Configuration optionnelle
        
    Returns:
        True si démarrage réussi
    """
    orchestrator = get_engine_orchestrator(config)
    return orchestrator.start_all()

def stop_alerts_engines() -> bool:
    """
    Fonction de convenance pour arrêter tous les moteurs
    
    Returns:
        True si arrêt réussi
    """
    global _orchestrator_instance
    
    if _orchestrator_instance:
        return _orchestrator_instance.stop_all()
    
    return True

logger.info("Module core alerts initialisé - Orchestration des moteurs d'alertes")
