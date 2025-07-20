"""
🔒 Data Isolation Module - Architecture Multi-Tenant Ultra-Avancée
================================================================

Module central d'isolation des données pour l'architecture multi-tenant.
Implémente les meilleures pratiques de sécurité et de performance.

Développé par l'équipe d'experts :
- Lead Dev + Architecte IA : Fahed Mlaiel
- Développeur Backend Senior (Python/FastAPI/Django) : Fahed Mlaiel  
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face) : Fahed Mlaiel
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB) : Fahed Mlaiel
- Spécialiste Sécurité Backend : Fahed Mlaiel
- Architecte Microservices : Fahed Mlaiel

Version: 2.0.0
License: Enterprise
"""

from .core.tenant_context import TenantContext, TenantContextManager
from .core.data_partition import DataPartition, PartitionStrategy
from .core.isolation_engine import IsolationEngine, IsolationLevel
from .core.tenant_resolver import TenantResolver, ResolutionStrategy

from .strategies.database_level import DatabaseLevelStrategy
from .strategies.schema_level import SchemaLevelStrategy  
from .strategies.row_level import RowLevelStrategy
from .strategies.hybrid_strategy import HybridStrategy
from .strategies.ai_driven_strategy import AIDrivenStrategy
from .strategies.analytics_driven_strategy import AnalyticsDrivenStrategy
from .strategies.performance_optimized_strategy import PerformanceOptimizedStrategy
from .strategies.predictive_scaling_strategy import PredictiveScalingStrategy
from .strategies.real_time_adaptive_strategy import RealTimeAdaptiveStrategy
from .strategies.blockchain_security_strategy import BlockchainSecurityStrategy
from .strategies.edge_computing_strategy import EdgeComputingStrategy
from .strategies.event_driven_strategy import EventDrivenStrategy
from .strategies.ultra_advanced_orchestrator import UltraAdvancedStrategyOrchestrator, OrchestratorConfig

from .middleware.tenant_middleware import TenantMiddleware
from .middleware.security_middleware import SecurityMiddleware
from .middleware.monitoring_middleware import MonitoringMiddleware

from .managers.connection_manager import ConnectionManager
from .managers.cache_manager import CacheManager
from .managers.security_manager import SecurityManager
from .managers.compliance_manager import ComplianceManager

from .decorators.tenant_aware import tenant_aware, require_tenant
from .decorators.data_isolation import data_isolation, isolation_level
from .decorators.security_decorators import secure_tenant_access, audit_access

from .validators.tenant_validator import TenantValidator
from .validators.data_validator import DataValidator
from .validators.security_validator import SecurityValidator

from .filters.tenant_filter import TenantFilter, AutoTenantFilter
from .filters.security_filter import SecurityFilter
from .filters.compliance_filter import ComplianceFilter

from .encryption.tenant_encryption import TenantEncryption
from .encryption.field_encryption import FieldEncryption
from .encryption.key_manager import KeyManager

from .monitoring.isolation_monitor import IsolationMonitor
from .monitoring.performance_monitor import PerformanceMonitor
from .monitoring.security_monitor import SecurityMonitor

from .utils.tenant_utils import TenantUtils
from .utils.security_utils import SecurityUtils
from .utils.performance_utils import PerformanceUtils

from .exceptions import (
    DataIsolationError,
    TenantNotFoundError,
    SecurityViolationError,
    IsolationLevelError,
    PartitionError,
    EncryptionError
)

__version__ = "2.0.0"
__author__ = "Fahed Mlaiel"

__all__ = [
    # Core components
    "TenantContext",
    "TenantContextManager", 
    "DataPartition",
    "PartitionStrategy",
    "IsolationEngine",
    "IsolationLevel",
    "TenantResolver",
    "ResolutionStrategy",
    
    # Isolation strategies
    "DatabaseLevelStrategy",
    "SchemaLevelStrategy",
    "RowLevelStrategy", 
    "HybridStrategy",
    "AIDrivenStrategy",
    "AnalyticsDrivenStrategy", 
    "PerformanceOptimizedStrategy",
    "PredictiveScalingStrategy",
    "RealTimeAdaptiveStrategy",
    "BlockchainSecurityStrategy",
    "EdgeComputingStrategy",
    "EventDrivenStrategy",
    "UltraAdvancedStrategyOrchestrator",
    "OrchestratorConfig",
    
    # Middleware
    "TenantMiddleware",
    "SecurityMiddleware",
    "MonitoringMiddleware",
    
    # Managers
    "ConnectionManager",
    "CacheManager",
    "SecurityManager",
    "ComplianceManager",
    
    # Decorators
    "tenant_aware",
    "require_tenant",
    "data_isolation",
    "isolation_level",
    "secure_tenant_access",
    "audit_access",
    
    # Validators
    "TenantValidator",
    "DataValidator", 
    "SecurityValidator",
    
    # Filters
    "TenantFilter",
    "AutoTenantFilter",
    "SecurityFilter",
    "ComplianceFilter",
    
    # Encryption
    "TenantEncryption",
    "FieldEncryption",
    "KeyManager",
    
    # Monitoring
    "IsolationMonitor",
    "PerformanceMonitor",
    "SecurityMonitor",
    
    # Utils
    "TenantUtils",
    "SecurityUtils",
    "PerformanceUtils",
    
    # Exceptions
    "DataIsolationError",
    "TenantNotFoundError",
    "SecurityViolationError",
    "IsolationLevelError",
    "PartitionError",
    "EncryptionError"
]
