"""
🏗️ Data Isolation Core Module - Module Central d'Isolation des Données
======================================================================

Module ultra-avancé pour l'isolation des données multi-tenant avec architecture
enterprise-grade, sécurité paranoid-level et performance optimisée.

Architecture:
- Tenant Context Management (Gestion des contextes)
- Isolation Engine (Moteur d'isolation)
- Data Partitioning (Partitionnement des données)
- Tenant Resolution (Résolution des tenants)
- Security Context (Contexte de sécurité)
- Performance Optimization (Optimisation des performances)
- Real-time Monitoring (Surveillance en temps réel)

Features:
✅ Multi-tenant data isolation
✅ Real-time context switching
✅ Performance-optimized queries
✅ Security policy enforcement
✅ Compliance monitoring
✅ Audit trail integration
✅ Cache-aware operations
✅ Horizontal scaling support

Author: Lead Dev + Architecte IA - Fahed Mlaiel
"""

from .tenant_context import (
    TenantContext,
    TenantType,
    TenantMetadata,
    SecurityContext,
    IsolationLevel
)

from .isolation_engine import (
    IsolationEngine,
    EngineState,
    PerformanceMode,
    IsolationMetrics
)

from .data_partition import (
    DataPartition,
    PartitionConfig,
    PartitionType,
    PartitionStrategy,
    PartitionManager
)

from .tenant_resolver import (
    TenantResolver,
    ResolutionStrategy,
    ResolutionSource,
    ResolutionConfig
)

from .compliance_engine import (
    ComplianceEngine,
    ComplianceLevel,
    ComplianceRule,
    AuditEvent
)

from .security_policy_engine import (
    SecurityPolicyEngine,
    PolicyType,
    SecurityPolicy,
    PolicyEnforcement
)

from .performance_optimizer import (
    PerformanceOptimizer,
    OptimizationStrategy,
    PerformanceMetrics,
    QueryOptimizer
)

from .context_manager import (
    ContextManager,
    ContextSwitcher,
    ContextState,
    ContextValidator
)

__all__ = [
    # Core Components
    "TenantContext",
    "TenantType", 
    "TenantMetadata",
    "SecurityContext",
    "IsolationLevel",
    
    # Isolation Engine
    "IsolationEngine",
    "EngineState",
    "PerformanceMode",
    "IsolationMetrics",
    
    # Data Partitioning
    "DataPartition",
    "PartitionConfig",
    "PartitionType",
    "PartitionStrategy",
    "PartitionManager",
    
    # Tenant Resolution
    "TenantResolver",
    "ResolutionStrategy",
    "ResolutionSource", 
    "ResolutionConfig",
    
    # Compliance
    "ComplianceEngine",
    "ComplianceLevel",
    "ComplianceRule",
    "AuditEvent",
    
    # Security
    "SecurityPolicyEngine",
    "PolicyType",
    "SecurityPolicy",
    "PolicyEnforcement",
    
    # Performance
    "PerformanceOptimizer",
    "OptimizationStrategy",
    "PerformanceMetrics",
    "QueryOptimizer",
    
    # Context Management
    "ContextManager",
    "ContextSwitcher",
    "ContextState",
    "ContextValidator"
]

# Version Information
__version__ = "2.0.0"
__author__ = "Fahed Mlaiel"
__email__ = "fahed.mlaiel@spotify-ai-agent.com"
__status__ = "Production"
