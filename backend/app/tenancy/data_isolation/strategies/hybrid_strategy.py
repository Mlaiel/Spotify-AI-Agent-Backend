"""
🚀 Hybrid Strategy - Stratégie d'Isolation Hybride Ultra-Avancée
===============================================================

Stratégie hybride combinant Database, Schema et Row Level Security
pour une isolation optimale selon le tenant et le type de données.

Author: Architecte Microservices - Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from ..core.tenant_context import TenantContext, TenantType, IsolationLevel
from ..core.isolation_engine import IsolationStrategy, EngineConfig
from .database_level import DatabaseLevelStrategy, DatabaseConfig
from .schema_level import SchemaLevelStrategy, SchemaConfig
from .row_level import RowLevelStrategy, RLSConfig
from ..exceptions import DataIsolationError, IsolationLevelError


class HybridMode(Enum):
    """Modes de fonctionnement hybride"""
    TENANT_TYPE_BASED = "tenant_type_based"      # Basé sur le type de tenant
    DATA_SENSITIVITY = "data_sensitivity"        # Basé sur la sensibilité des données
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # Optimisé pour la performance
    SECURITY_FIRST = "security_first"            # Sécurité maximale
    ADAPTIVE = "adaptive"                        # Adaptatif selon la charge


class DataClassification(Enum):
    """Classification des données"""
    PUBLIC = "public"              # Données publiques
    INTERNAL = "internal"          # Données internes
    CONFIDENTIAL = "confidential"  # Données confidentielles
    RESTRICTED = "restricted"      # Données restreintes
    TOP_SECRET = "top_secret"      # Données ultra-sensibles


@dataclass
class HybridRule:
    """Règle pour la stratégie hybride"""
    name: str
    condition: Dict[str, Any]  # Conditions pour appliquer cette règle
    strategy: str              # database, schema, ou row_level
    priority: int = 100        # Priorité (plus bas = plus prioritaire)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridConfig:
    """Configuration de la stratégie hybride"""
    mode: HybridMode = HybridMode.ADAPTIVE
    
    # Strategies configurations
    database_config: Optional[DatabaseConfig] = None
    schema_config: Optional[SchemaConfig] = None
    rls_config: Optional[RLSConfig] = None
    
    # Hybrid rules
    custom_rules: List[HybridRule] = field(default_factory=list)
    
    # Tenant type mappings
    tenant_type_strategies: Dict[TenantType, str] = field(default_factory=lambda: {
        TenantType.SPOTIFY_ARTIST: "schema",
        TenantType.RECORD_LABEL: "database",
        TenantType.MUSIC_PRODUCER: "schema",
        TenantType.DISTRIBUTOR: "database",
        TenantType.ENTERPRISE: "database",
        TenantType.PLATFORM: "row_level"
    })
    
    # Data classification mappings
    data_classification_strategies: Dict[DataClassification, str] = field(default_factory=lambda: {
        DataClassification.PUBLIC: "row_level",
        DataClassification.INTERNAL: "schema",
        DataClassification.CONFIDENTIAL: "schema",
        DataClassification.RESTRICTED: "database",
        DataClassification.TOP_SECRET: "database"
    })
    
    # Performance thresholds
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "response_time_ms": 100.0,
        "cpu_usage_percent": 80.0,
        "memory_usage_percent": 85.0,
        "concurrent_queries": 1000
    })
    
    # Security settings
    enforce_encryption: bool = True
    audit_strategy_selection: bool = True
    allow_strategy_fallback: bool = True
    monitor_performance: bool = True


class HybridDecisionEngine:
    """Moteur de décision pour la stratégie hybride"""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.logger = logging.getLogger("hybrid.decision_engine")
        
        # Decision statistics
        self.strategy_usage_stats: Dict[str, int] = {
            "database": 0,
            "schema": 0,
            "row_level": 0
        }
        
        # Performance metrics per strategy
        self.strategy_performance: Dict[str, Dict[str, float]] = {
            "database": {"avg_response_time": 0.0, "success_rate": 1.0},
            "schema": {"avg_response_time": 0.0, "success_rate": 1.0},
            "row_level": {"avg_response_time": 0.0, "success_rate": 1.0}
        }
        
        # Current system load
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "concurrent_queries": 0,
            "avg_response_time": 0.0
        }
    
    def decide_strategy(
        self, 
        context: TenantContext,
        operation: str,
        target: Any,
        **kwargs
    ) -> str:
        """Décide quelle stratégie utiliser"""
        try:
            # Check custom rules first (highest priority)
            strategy = self._check_custom_rules(context, operation, target, **kwargs)
            if strategy:
                self.logger.debug(f"Strategy selected by custom rule: {strategy}")
                return strategy
            
            # Apply mode-specific logic
            if self.config.mode == HybridMode.TENANT_TYPE_BASED:
                strategy = self._tenant_type_based_decision(context)
            elif self.config.mode == HybridMode.DATA_SENSITIVITY:
                strategy = self._data_sensitivity_based_decision(target, **kwargs)
            elif self.config.mode == HybridMode.PERFORMANCE_OPTIMIZED:
                strategy = self._performance_optimized_decision(context, operation)
            elif self.config.mode == HybridMode.SECURITY_FIRST:
                strategy = self._security_first_decision(context, target)
            elif self.config.mode == HybridMode.ADAPTIVE:
                strategy = self._adaptive_decision(context, operation, target, **kwargs)
            else:
                strategy = "schema"  # Default fallback
            
            # Track usage
            self.strategy_usage_stats[strategy] = self.strategy_usage_stats.get(strategy, 0) + 1
            
            if self.config.audit_strategy_selection:
                self.logger.info(f"Selected strategy '{strategy}' for tenant {context.tenant_id}, operation: {operation}")
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Decision engine error: {e}")
            return "schema"  # Safe fallback
    
    def _check_custom_rules(
        self, 
        context: TenantContext,
        operation: str,
        target: Any,
        **kwargs
    ) -> Optional[str]:
        """Vérifie les règles personnalisées"""
        # Sort rules by priority
        sorted_rules = sorted(self.config.custom_rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            if self._rule_matches(rule, context, operation, target, **kwargs):
                self.logger.debug(f"Rule '{rule.name}' matched, using strategy: {rule.strategy}")
                return rule.strategy
        
        return None
    
    def _rule_matches(
        self, 
        rule: HybridRule,
        context: TenantContext,
        operation: str,
        target: Any,
        **kwargs
    ) -> bool:
        """Vérifie si une règle correspond"""
        conditions = rule.condition
        
        # Check tenant conditions
        if "tenant_id" in conditions:
            if context.tenant_id not in conditions["tenant_id"]:
                return False
        
        if "tenant_type" in conditions:
            if context.tenant_type.value not in conditions["tenant_type"]:
                return False
        
        # Check operation conditions
        if "operation" in conditions:
            if operation not in conditions["operation"]:
                return False
        
        # Check target conditions
        if "target_type" in conditions:
            target_type = type(target).__name__.lower()
            if target_type not in conditions["target_type"]:
                return False
        
        # Check time-based conditions
        if "time_range" in conditions:
            current_hour = datetime.now().hour
            time_range = conditions["time_range"]
            if not (time_range[0] <= current_hour <= time_range[1]):
                return False
        
        return True
    
    def _tenant_type_based_decision(self, context: TenantContext) -> str:
        """Décision basée sur le type de tenant"""
        return self.config.tenant_type_strategies.get(
            context.tenant_type, 
            "schema"
        )
    
    def _data_sensitivity_based_decision(self, target: Any, **kwargs) -> str:
        """Décision basée sur la sensibilité des données"""
        # Determine data classification
        data_class = self._classify_data(target, **kwargs)
        return self.config.data_classification_strategies.get(
            data_class,
            "schema"
        )
    
    def _classify_data(self, target: Any, **kwargs) -> DataClassification:
        """Classifie les données selon leur sensibilité"""
        target_str = str(target).lower()
        
        # Check for highly sensitive data patterns
        sensitive_patterns = [
            "password", "token", "secret", "key", "payment", 
            "credit_card", "ssn", "personal", "private"
        ]
        
        restricted_patterns = [
            "financial", "revenue", "contract", "legal", "internal"
        ]
        
        if any(pattern in target_str for pattern in sensitive_patterns):
            return DataClassification.RESTRICTED
        elif any(pattern in target_str for pattern in restricted_patterns):
            return DataClassification.CONFIDENTIAL
        elif "user" in target_str or "profile" in target_str:
            return DataClassification.INTERNAL
        else:
            return DataClassification.PUBLIC
    
    def _performance_optimized_decision(
        self, 
        context: TenantContext,
        operation: str
    ) -> str:
        """Décision optimisée pour la performance"""
        # Check current system load
        if self.system_metrics["cpu_usage"] > self.config.performance_thresholds["cpu_usage_percent"]:
            # High CPU load - use lighter strategy
            return "row_level"
        
        if self.system_metrics["concurrent_queries"] > self.config.performance_thresholds["concurrent_queries"]:
            # High query load - use connection pooling friendly strategy
            return "schema"
        
        # Choose based on operation type
        if operation in ["select", "read"]:
            # Read operations - optimize for speed
            best_strategy = min(
                self.strategy_performance.keys(),
                key=lambda s: self.strategy_performance[s]["avg_response_time"]
            )
            return best_strategy
        else:
            # Write operations - balance performance and isolation
            return "schema"
    
    def _security_first_decision(self, context: TenantContext, target: Any) -> str:
        """Décision privilégiant la sécurité"""
        # Always use the most secure strategy for enterprise tenants
        if context.tenant_type == TenantType.ENTERPRISE:
            return "database"
        
        # For sensitive data, use database isolation
        if self._classify_data(target) in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            return "database"
        
        # Default to schema level for good security/performance balance
        return "schema"
    
    def _adaptive_decision(
        self, 
        context: TenantContext,
        operation: str,
        target: Any,
        **kwargs
    ) -> str:
        """Décision adaptative combinant plusieurs facteurs"""
        # Calculate scores for each strategy
        scores = {
            "database": self._calculate_strategy_score("database", context, operation, target),
            "schema": self._calculate_strategy_score("schema", context, operation, target),
            "row_level": self._calculate_strategy_score("row_level", context, operation, target)
        }
        
        # Select strategy with highest score
        best_strategy = max(scores.keys(), key=lambda s: scores[s])
        
        self.logger.debug(f"Adaptive decision scores: {scores}, selected: {best_strategy}")
        return best_strategy
    
    def _calculate_strategy_score(
        self, 
        strategy: str,
        context: TenantContext,
        operation: str,
        target: Any
    ) -> float:
        """Calcule le score d'une stratégie"""
        score = 0.0
        
        # Performance factor (40% weight)
        performance_score = self._get_performance_score(strategy)
        score += performance_score * 0.4
        
        # Security factor (30% weight)
        security_score = self._get_security_score(strategy, context, target)
        score += security_score * 0.3
        
        # Compatibility factor (20% weight)
        compatibility_score = self._get_compatibility_score(strategy, context, operation)
        score += compatibility_score * 0.2
        
        # Load factor (10% weight)
        load_score = self._get_load_score(strategy)
        score += load_score * 0.1
        
        return score
    
    def _get_performance_score(self, strategy: str) -> float:
        """Score de performance d'une stratégie (0-1)"""
        perf = self.strategy_performance.get(strategy, {"avg_response_time": 100.0, "success_rate": 1.0})
        
        # Lower response time = higher score
        response_score = max(0, 1 - (perf["avg_response_time"] / 1000.0))
        success_score = perf["success_rate"]
        
        return (response_score + success_score) / 2
    
    def _get_security_score(self, strategy: str, context: TenantContext, target: Any) -> float:
        """Score de sécurité d'une stratégie (0-1)"""
        # Security ranking: database > schema > row_level
        security_scores = {
            "database": 1.0,
            "schema": 0.7,
            "row_level": 0.5
        }
        
        base_score = security_scores.get(strategy, 0.5)
        
        # Adjust based on tenant type and data sensitivity
        if context.tenant_type == TenantType.ENTERPRISE:
            if strategy == "database":
                base_score *= 1.2
        
        data_class = self._classify_data(target)
        if data_class in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            if strategy == "database":
                base_score *= 1.1
        
        return min(1.0, base_score)
    
    def _get_compatibility_score(self, strategy: str, context: TenantContext, operation: str) -> float:
        """Score de compatibilité d'une stratégie (0-1)"""
        # Some operations work better with certain strategies
        compatibility_matrix = {
            "database": {"bulk_insert": 0.9, "analytics": 0.8, "reporting": 0.9},
            "schema": {"crud": 0.9, "api": 0.9, "web": 0.8},
            "row_level": {"read": 0.9, "select": 0.9, "multi_tenant": 0.8}
        }
        
        return compatibility_matrix.get(strategy, {}).get(operation, 0.7)
    
    def _get_load_score(self, strategy: str) -> float:
        """Score basé sur la charge système actuelle (0-1)"""
        # Database isolation requires more resources
        if strategy == "database":
            if self.system_metrics["memory_usage"] > 80:
                return 0.3
            return 0.7
        
        # Schema isolation is balanced
        if strategy == "schema":
            return 0.8
        
        # Row level is lightest
        return 0.9
    
    def update_performance_metrics(self, strategy: str, response_time: float, success: bool):
        """Met à jour les métriques de performance"""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {"avg_response_time": 0.0, "success_rate": 1.0}
        
        # Update average response time (exponential moving average)
        current_avg = self.strategy_performance[strategy]["avg_response_time"]
        self.strategy_performance[strategy]["avg_response_time"] = (
            current_avg * 0.9 + response_time * 0.1
        )
        
        # Update success rate
        current_rate = self.strategy_performance[strategy]["success_rate"]
        self.strategy_performance[strategy]["success_rate"] = (
            current_rate * 0.95 + (1.0 if success else 0.0) * 0.05
        )
    
    def update_system_metrics(self, metrics: Dict[str, float]):
        """Met à jour les métriques système"""
        self.system_metrics.update(metrics)
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de décision"""
        total_decisions = sum(self.strategy_usage_stats.values())
        
        return {
            "total_decisions": total_decisions,
            "strategy_distribution": {
                strategy: (count / total_decisions * 100) if total_decisions > 0 else 0
                for strategy, count in self.strategy_usage_stats.items()
            },
            "strategy_performance": dict(self.strategy_performance),
            "current_system_metrics": dict(self.system_metrics),
            "mode": self.config.mode.value
        }


class HybridStrategy(IsolationStrategy):
    """
    Stratégie d'isolation hybride ultra-avancée
    
    Features:
    - Combinaison intelligente de 3 stratégies d'isolation
    - Moteur de décision adaptatif
    - Optimisation automatique selon la charge
    - Monitoring de performance en temps réel
    - Règles personnalisables
    - Fallback automatique
    - Audit complet des décisions
    """
    
    def __init__(self, hybrid_config: Optional[HybridConfig] = None):
        self.hybrid_config = hybrid_config or HybridConfig()
        self.logger = logging.getLogger("isolation.hybrid")
        
        # Initialize decision engine
        self.decision_engine = HybridDecisionEngine(self.hybrid_config)
        
        # Initialize sub-strategies
        self.strategies: Dict[str, IsolationStrategy] = {}
        
        # Performance monitoring
        self.strategy_metrics: Dict[str, List[Dict[str, Any]]] = {
            "database": [],
            "schema": [],
            "row_level": []
        }
        
        # Fallback tracking
        self.fallback_stats = {
            "total_fallbacks": 0,
            "fallback_reasons": {}
        }
        
        self.logger.info("Hybrid isolation strategy initialized")
    
    async def initialize(self, config: EngineConfig):
        """Initialise la stratégie hybride"""
        try:
            # Initialize sub-strategies
            await self._initialize_sub_strategies(config)
            
            # Start performance monitoring
            if self.hybrid_config.monitor_performance:
                asyncio.create_task(self._performance_monitoring_loop())
            
            self.logger.info("Hybrid strategy ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid strategy: {e}")
            raise DataIsolationError(f"Hybrid initialization failed: {e}")
    
    async def _initialize_sub_strategies(self, config: EngineConfig):
        """Initialise les sous-stratégies"""
        try:
            # Database level strategy
            db_config = self.hybrid_config.database_config or DatabaseConfig()
            self.strategies["database"] = DatabaseLevelStrategy(db_config)
            await self.strategies["database"].initialize(config)
            
            # Schema level strategy
            schema_config = self.hybrid_config.schema_config or SchemaConfig()
            self.strategies["schema"] = SchemaLevelStrategy(schema_config)
            await self.strategies["schema"].initialize(config)
            
            # Row level strategy
            rls_config = self.hybrid_config.rls_config or RLSConfig()
            self.strategies["row_level"] = RowLevelStrategy(rls_config)
            await self.strategies["row_level"].initialize(config)
            
            self.logger.info("All sub-strategies initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sub-strategies: {e}")
            raise
    
    async def apply_isolation(
        self, 
        context: TenantContext, 
        target: Any,
        **kwargs
    ) -> Any:
        """Applique l'isolation hybride"""
        operation = kwargs.get("operation", "query")
        start_time = datetime.now(timezone.utc)
        
        try:
            # Decide which strategy to use
            selected_strategy = self.decision_engine.decide_strategy(
                context, operation, target, **kwargs
            )
            
            # Apply the selected strategy
            strategy_instance = self.strategies[selected_strategy]
            result = await strategy_instance.apply_isolation(context, target, **kwargs)
            
            # Record success metrics
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.decision_engine.update_performance_metrics(
                selected_strategy, response_time, True
            )
            
            self._record_strategy_usage(selected_strategy, response_time, True)
            
            return result
            
        except Exception as e:
            # Try fallback if enabled
            if self.hybrid_config.allow_strategy_fallback:
                fallback_result = await self._try_fallback_strategies(
                    context, target, operation, **kwargs
                )
                if fallback_result is not None:
                    return fallback_result
            
            # Record failure metrics
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            selected_strategy = getattr(self, '_last_selected_strategy', 'unknown')
            self.decision_engine.update_performance_metrics(
                selected_strategy, response_time, False
            )
            
            self.logger.error(f"Hybrid isolation failed for tenant {context.tenant_id}: {e}")
            raise DataIsolationError(f"Hybrid isolation failed: {e}")
    
    async def _try_fallback_strategies(
        self, 
        context: TenantContext,
        target: Any,
        operation: str,
        **kwargs
    ) -> Optional[Any]:
        """Essaie les stratégies de fallback"""
        fallback_order = ["schema", "row_level", "database"]
        
        for strategy_name in fallback_order:
            try:
                self.logger.warning(f"Trying fallback strategy: {strategy_name}")
                
                strategy_instance = self.strategies[strategy_name]
                result = await strategy_instance.apply_isolation(context, target, **kwargs)
                
                # Record fallback usage
                self.fallback_stats["total_fallbacks"] += 1
                reason = f"fallback_to_{strategy_name}"
                self.fallback_stats["fallback_reasons"][reason] = (
                    self.fallback_stats["fallback_reasons"].get(reason, 0) + 1
                )
                
                self.logger.info(f"Fallback to {strategy_name} succeeded")
                return result
                
            except Exception as e:
                self.logger.warning(f"Fallback strategy {strategy_name} also failed: {e}")
                continue
        
        self.logger.error("All fallback strategies failed")
        return None
    
    def _record_strategy_usage(self, strategy: str, response_time: float, success: bool):
        """Enregistre l'utilisation d'une stratégie"""
        record = {
            "timestamp": datetime.now(timezone.utc),
            "strategy": strategy,
            "response_time_ms": response_time,
            "success": success
        }
        
        self.strategy_metrics[strategy].append(record)
        
        # Keep only last 1000 records per strategy
        if len(self.strategy_metrics[strategy]) > 1000:
            self.strategy_metrics[strategy] = self.strategy_metrics[strategy][-1000:]
    
    async def _performance_monitoring_loop(self):
        """Boucle de monitoring des performances"""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Collect system metrics (mock implementation)
                system_metrics = await self._collect_system_metrics()
                self.decision_engine.update_system_metrics(system_metrics)
                
                # Log performance summary
                stats = self.decision_engine.get_decision_stats()
                self.logger.debug(f"Performance stats: {stats}")
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
    
    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collecte les métriques système"""
        # This would integrate with actual system monitoring
        # For now, return mock data
        import random
        return {
            "cpu_usage": random.uniform(20, 90),
            "memory_usage": random.uniform(30, 85),
            "concurrent_queries": random.randint(10, 500),
            "avg_response_time": random.uniform(10, 200)
        }
    
    async def validate_access(self, context: TenantContext, resource: str) -> bool:
        """Valide l'accès avec la stratégie appropriée"""
        try:
            # Decide strategy for validation
            selected_strategy = self.decision_engine.decide_strategy(
                context, "validate", resource
            )
            
            strategy_instance = self.strategies[selected_strategy]
            return await strategy_instance.validate_access(context, resource)
            
        except Exception as e:
            self.logger.error(f"Access validation failed: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Vérification de santé hybride"""
        try:
            # Check all sub-strategies
            health_results = {}
            for name, strategy in self.strategies.items():
                health_results[name] = await strategy.health_check()
            
            # At least one strategy must be healthy
            if not any(health_results.values()):
                self.logger.error("All sub-strategies are unhealthy")
                return False
            
            # Log unhealthy strategies
            unhealthy = [name for name, healthy in health_results.items() if not healthy]
            if unhealthy:
                self.logger.warning(f"Unhealthy strategies: {unhealthy}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Hybrid health check failed: {e}")
            return False
    
    def add_custom_rule(self, rule: HybridRule):
        """Ajoute une règle personnalisée"""
        self.hybrid_config.custom_rules.append(rule)
        # Sort by priority
        self.hybrid_config.custom_rules.sort(key=lambda r: r.priority)
        self.logger.info(f"Added custom rule: {rule.name}")
    
    def remove_custom_rule(self, rule_name: str):
        """Supprime une règle personnalisée"""
        self.hybrid_config.custom_rules = [
            rule for rule in self.hybrid_config.custom_rules 
            if rule.name != rule_name
        ]
        self.logger.info(f"Removed custom rule: {rule_name}")
    
    async def get_strategy_statistics(self) -> Dict[str, Any]:
        """Obtient les statistiques détaillées"""
        stats = {
            "hybrid_config": {
                "mode": self.hybrid_config.mode.value,
                "custom_rules_count": len(self.hybrid_config.custom_rules),
                "fallback_enabled": self.hybrid_config.allow_strategy_fallback
            },
            "decision_engine": self.decision_engine.get_decision_stats(),
            "fallback_stats": dict(self.fallback_stats),
            "strategy_health": {}
        }
        
        # Add health status for each strategy
        for name, strategy in self.strategies.items():
            stats["strategy_health"][name] = await strategy.health_check()
        
        # Add detailed metrics for each strategy
        for name, metrics in self.strategy_metrics.items():
            if metrics:
                recent_metrics = metrics[-10:]  # Last 10 operations
                avg_response_time = sum(m["response_time_ms"] for m in recent_metrics) / len(recent_metrics)
                success_rate = sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics)
                
                stats[f"{name}_recent_performance"] = {
                    "avg_response_time_ms": avg_response_time,
                    "success_rate": success_rate,
                    "sample_size": len(recent_metrics)
                }
        
        return stats
    
    async def cleanup(self):
        """Nettoie les ressources hybrides"""
        self.logger.info("Cleaning up hybrid strategy...")
        
        # Cleanup all sub-strategies
        for name, strategy in self.strategies.items():
            try:
                await strategy.cleanup()
                self.logger.debug(f"Cleaned up {name} strategy")
            except Exception as e:
                self.logger.error(f"Error cleaning up {name} strategy: {e}")
        
        # Clear metrics
        for strategy_metrics in self.strategy_metrics.values():
            strategy_metrics.clear()
        
        self.logger.info("Hybrid strategy cleanup completed")
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la stratégie hybride"""
        return {
            "strategy_type": "hybrid",
            "mode": self.hybrid_config.mode.value,
            "active_sub_strategies": list(self.strategies.keys()),
            "custom_rules": len(self.hybrid_config.custom_rules),
            "decision_stats": self.decision_engine.get_decision_stats(),
            "fallback_stats": dict(self.fallback_stats),
            "total_operations": sum(
                len(metrics) for metrics in self.strategy_metrics.values()
            )
        }
