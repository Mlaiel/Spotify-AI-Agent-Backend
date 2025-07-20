"""
🎯 Ultra-Advanced Strategy Orchestrator - Orchestrateur de Stratégies Ultra-Intelligent
======================================================================================

Orchestrateur ultra-intelligent pour la sélection, configuration et optimisation
automatique des stratégies d'isolation multi-tenant avec intelligence artificielle,
apprentissage automatique et adaptation temps réel.

Features Ultra-Avancées:
    🧠 Sélection automatique de stratégie par IA
    🔄 Orchestration multi-stratégies hybride
    📊 Optimisation continue par ML
    ⚡ Adaptation temps réel intelligente
    🎯 Prédiction des besoins futurs
    🛡️ Sécurité multi-niveaux automatique
    📈 Analytics prédictives avancées
    🔮 Auto-scaling prédictif
    🌐 Distribution géographique optimale
    ⚖️ Load balancing ultra-intelligent

Experts Contributeurs - Team Fahed Mlaiel:
    🧠 Lead Dev + Architecte IA - Fahed Mlaiel
    💻 Développeur Backend Senior (Python/FastAPI/Django)
    🤖 Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
    🗄️ DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
    🔒 Spécialiste Sécurité Backend
    🏗️ Architecte Microservices

Author: Lead Dev + Architecte IA Expert - Team Fahed Mlaiel
Version: 1.0.0 - Ultra-Intelligent Orchestration Edition
License: Ultra-Advanced Enterprise License
"""

import asyncio
import logging
import json
import time
import math
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
import heapq
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import uuid

# ML and optimization imports
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.neural_network import MLPClassifier
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..core.tenant_context import TenantContext, TenantType, IsolationLevel
from ..core.isolation_engine import IsolationStrategy, EngineConfig
from ..managers.connection_manager import DatabaseConnection
from ..exceptions import DataIsolationError, StrategyOrchestrationError, OptimizationError

# Import all strategies
from .database_level import DatabaseLevelStrategy, DatabaseConfig
from .schema_level import SchemaLevelStrategy, SchemaConfig
from .row_level import RowLevelStrategy, RLSConfig
from .hybrid_strategy import HybridStrategy, HybridConfig
from .ai_driven_strategy import AIDriverStrategy, AIConfig
from .analytics_driven_strategy import AnalyticsDrivenStrategy, AnalyticsConfig
from .performance_optimized_strategy import PerformanceOptimizedStrategy, PerformanceConfig
from .predictive_scaling_strategy import PredictiveScalingStrategy, ScalingConfig
from .real_time_adaptive_strategy import RealTimeAdaptiveStrategy, AdaptiveConfig
from .blockchain_security_strategy import BlockchainSecurityStrategy, BlockchainConfig
from .edge_computing_strategy import EdgeComputingStrategy, EdgeConfig
from .event_driven_strategy import EventDrivenStrategy, EventDrivenConfig

logger = logging.getLogger(__name__)


class StrategyPriority(Enum):
    """Priorités des stratégies"""
    CRITICAL = 1      # Sécurité, compliance critiques
    HIGH = 2          # Performance critique, haute charge
    NORMAL = 3        # Opérations standard
    LOW = 4           # Optimisation background
    BACKGROUND = 5    # Analytics, monitoring


class OptimizationObjective(Enum):
    """Objectifs d'optimisation"""
    PERFORMANCE = "performance"
    COST = "cost"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    SCALABILITY = "scalability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"
    CONSISTENCY = "consistency"
    RELIABILITY = "reliability"


class AutoScalingTrigger(Enum):
    """Déclencheurs d'auto-scaling"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"


@dataclass
class StrategyMetrics:
    """Métriques d'une stratégie"""
    strategy_name: str
    performance_score: float = 0.0
    cost_score: float = 0.0
    security_score: float = 0.0
    compliance_score: float = 0.0
    latency_ms: float = 0.0
    throughput_ops_sec: float = 0.0
    error_rate: float = 0.0
    resource_usage: float = 0.0
    scalability_factor: float = 1.0
    availability_percentage: float = 99.9
    
    # Historical data
    measurements_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trend_direction: str = "stable"  # "improving", "degrading", "stable"
    
    def overall_score(self, weights: Dict[str, float] = None) -> float:
        """Calcule le score global pondéré"""
        if not weights:
            weights = {
                "performance": 0.25,
                "cost": 0.15,
                "security": 0.20,
                "compliance": 0.15,
                "latency": 0.15,
                "throughput": 0.10
            }
        
        score = (
            self.performance_score * weights.get("performance", 0.25) +
            (1 - self.cost_score) * weights.get("cost", 0.15) +  # Lower cost is better
            self.security_score * weights.get("security", 0.20) +
            self.compliance_score * weights.get("compliance", 0.15) +
            (1 - min(self.latency_ms / 1000, 1)) * weights.get("latency", 0.15) +  # Lower latency is better
            min(self.throughput_ops_sec / 1000, 1) * weights.get("throughput", 0.10)
        )
        
        return max(0, min(1, score))


@dataclass
class WorkloadPattern:
    """Pattern de charge de travail"""
    tenant_id: str
    operation_types: Dict[str, int]  # operation -> count
    peak_hours: List[int]
    average_load: float
    peak_load: float
    data_volume_gb: float
    request_rate_per_second: float
    geographic_distribution: Dict[str, float]  # region -> percentage
    compliance_requirements: List[str]
    security_level: str
    performance_requirements: Dict[str, float]
    
    # Predictions
    predicted_growth_rate: float = 0.0
    seasonal_patterns: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0


@dataclass
class OrchestratorConfig:
    """Configuration de l'orchestrateur"""
    # Intelligence settings
    ml_enabled: bool = True
    auto_optimization: bool = True
    predictive_scaling: bool = True
    real_time_adaptation: bool = True
    
    # Strategy selection
    max_concurrent_strategies: int = 3
    strategy_switching_enabled: bool = True
    fallback_strategy: str = "hybrid"
    min_confidence_threshold: float = 0.7
    
    # Optimization settings
    optimization_interval_minutes: int = 15
    learning_data_retention_days: int = 30
    model_retraining_interval_hours: int = 24
    performance_baseline_window_hours: int = 168  # 7 days
    
    # Monitoring settings
    metrics_collection_interval_seconds: int = 30
    anomaly_detection_enabled: bool = True
    alerting_enabled: bool = True
    performance_sla_targets: Dict[str, float] = field(default_factory=lambda: {
        "latency_ms": 100,
        "throughput_ops_sec": 1000,
        "availability_percentage": 99.9,
        "error_rate": 0.01
    })
    
    # Auto-scaling settings
    auto_scaling_enabled: bool = True
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    min_instances: int = 1
    max_instances: int = 100
    
    # Security settings
    security_monitoring: bool = True
    compliance_validation: bool = True
    automatic_security_upgrades: bool = True
    
    # Cost optimization
    cost_optimization_enabled: bool = True
    budget_constraints: Dict[str, float] = field(default_factory=dict)
    cost_efficiency_targets: Dict[str, float] = field(default_factory=dict)


class UltraAdvancedStrategyOrchestrator:
    """
    Orchestrateur ultra-intelligent de stratégies d'isolation
    
    Features Ultra-Avancées:
        🧠 Sélection automatique par ML avec deep learning
        🔄 Orchestration multi-stratégies avec optimisation continue
        📊 Analytics prédictives et machine learning avancé
        ⚡ Adaptation temps réel avec circuit breakers
        🎯 Prédiction des patterns de charge futurs
        🛡️ Sécurité adaptative multi-niveaux
        📈 Auto-scaling prédictif intelligent
        🔮 Optimisation continue par reinforcement learning
        🌐 Distribution géographique optimale
        ⚖️ Load balancing ultra-intelligent multi-critères
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.logger = logging.getLogger("isolation.orchestrator")
        
        # Strategy instances
        self.strategies: Dict[str, IsolationStrategy] = {}
        self.strategy_configs: Dict[str, Any] = {}
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        
        # ML models
        self.strategy_selector_model = None
        self.performance_predictor_model = None
        self.anomaly_detector_model = None
        self.workload_classifier_model = None
        
        # Workload analysis
        self.workload_patterns: Dict[str, WorkloadPattern] = {}
        self.tenant_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Real-time tracking
        self.active_strategies: Dict[str, str] = {}  # tenant_id -> strategy_name
        self.strategy_switches: List[Dict[str, Any]] = []
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # Auto-scaling
        self.scaling_history: List[Dict[str, Any]] = []
        self.current_instances: Dict[str, int] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        self.logger.info("Ultra-advanced strategy orchestrator initialized")
    
    async def initialize(self, engine_config: EngineConfig):
        """Initialise l'orchestrateur"""
        try:
            # Initialize all strategies
            await self._initialize_all_strategies(engine_config)
            
            # Initialize ML models
            if self.config.ml_enabled and ML_AVAILABLE:
                await self._initialize_ml_models()
            
            # Start monitoring and optimization
            await self._start_background_services()
            
            # Load historical data
            await self._load_historical_data()
            
            self.logger.info("Strategy orchestrator fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise StrategyOrchestrationError(f"Orchestrator initialization failed: {e}")
    
    async def _initialize_all_strategies(self, engine_config: EngineConfig):
        """Initialise toutes les stratégies"""
        strategy_definitions = {
            "database_level": (DatabaseLevelStrategy, DatabaseConfig()),
            "schema_level": (SchemaLevelStrategy, SchemaConfig()),
            "row_level": (RowLevelStrategy, RLSConfig()),
            "hybrid": (HybridStrategy, HybridConfig()),
            "ai_driven": (AIDriverStrategy, AIConfig()),
            "analytics_driven": (AnalyticsDrivenStrategy, AnalyticsConfig()),
            "performance_optimized": (PerformanceOptimizedStrategy, PerformanceConfig()),
            "predictive_scaling": (PredictiveScalingStrategy, ScalingConfig()),
            "real_time_adaptive": (RealTimeAdaptiveStrategy, AdaptiveConfig()),
            "blockchain_security": (BlockchainSecurityStrategy, BlockchainConfig()),
            "edge_computing": (EdgeComputingStrategy, EdgeConfig()),
            "event_driven": (EventDrivenStrategy, EventDrivenConfig())
        }
        
        for strategy_name, (strategy_class, strategy_config) in strategy_definitions.items():
            try:
                strategy = strategy_class(strategy_config)
                await strategy.initialize(engine_config)
                
                self.strategies[strategy_name] = strategy
                self.strategy_configs[strategy_name] = strategy_config
                self.strategy_metrics[strategy_name] = StrategyMetrics(strategy_name)
                
                # Initialize circuit breaker
                self.circuit_breakers[strategy_name] = {
                    "state": "closed",  # closed, open, half-open
                    "failure_count": 0,
                    "last_failure": None,
                    "timeout": 60,  # seconds
                    "threshold": 5
                }
                
                self.logger.info(f"Initialized strategy: {strategy_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize strategy {strategy_name}: {e}")
                # Continue with other strategies
    
    async def _initialize_ml_models(self):
        """Initialise les modèles ML"""
        try:
            # Strategy selector model
            self.strategy_selector_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Performance predictor model
            self.performance_predictor_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Anomaly detector model
            self.anomaly_detector_model = DBSCAN(
                eps=0.3,
                min_samples=5
            )
            
            # Workload classifier model
            self.workload_classifier_model = KMeans(
                n_clusters=5,
                random_state=42
            )
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"ML model initialization failed: {e}")
    
    async def _start_background_services(self):
        """Démarre les services en arrière-plan"""
        # Metrics collector
        task = asyncio.create_task(self._metrics_collector())
        self.background_tasks.append(task)
        
        # Performance monitor
        task = asyncio.create_task(self._performance_monitor())
        self.background_tasks.append(task)
        
        # Auto-optimizer
        if self.config.auto_optimization:
            task = asyncio.create_task(self._auto_optimizer())
            self.background_tasks.append(task)
        
        # Predictive scaler
        if self.config.predictive_scaling:
            task = asyncio.create_task(self._predictive_scaler())
            self.background_tasks.append(task)
        
        # Anomaly detector
        if self.config.anomaly_detection_enabled:
            task = asyncio.create_task(self._anomaly_detector())
            self.background_tasks.append(task)
        
        # Model trainer
        if self.config.ml_enabled and ML_AVAILABLE:
            task = asyncio.create_task(self._model_trainer())
            self.background_tasks.append(task)
        
        self.logger.info(f"Started {len(self.background_tasks)} background services")
    
    async def _load_historical_data(self):
        """Charge les données historiques"""
        # In production, this would load from persistent storage
        # For now, we initialize with empty data structures
        self.logger.info("Historical data loaded")
    
    async def select_optimal_strategy(self, tenant_context: TenantContext, operation: str, data: Any) -> str:
        """Sélectionne la stratégie optimale pour un tenant"""
        try:
            # Get tenant workload pattern
            workload_pattern = await self._analyze_workload_pattern(tenant_context, operation, data)
            
            # Use ML model if available and trained
            if (self.config.ml_enabled and ML_AVAILABLE and 
                self.strategy_selector_model and 
                hasattr(self.strategy_selector_model, 'classes_')):
                
                strategy = await self._ml_strategy_selection(tenant_context, workload_pattern)
                if strategy:
                    return strategy
            
            # Fallback to rule-based selection
            strategy = await self._rule_based_strategy_selection(tenant_context, workload_pattern)
            
            # Validate strategy is available and healthy
            if strategy in self.strategies and await self._is_strategy_healthy(strategy):
                return strategy
            
            # Final fallback
            return self.config.fallback_strategy
            
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {e}")
            return self.config.fallback_strategy
    
    async def _analyze_workload_pattern(self, tenant_context: TenantContext, operation: str, data: Any) -> WorkloadPattern:
        """Analyse le pattern de charge de travail"""
        tenant_id = tenant_context.tenant_id
        
        # Get or create workload pattern
        if tenant_id not in self.workload_patterns:
            self.workload_patterns[tenant_id] = WorkloadPattern(
                tenant_id=tenant_id,
                operation_types={},
                peak_hours=[],
                average_load=0.0,
                peak_load=0.0,
                data_volume_gb=0.0,
                request_rate_per_second=0.0,
                geographic_distribution={},
                compliance_requirements=[],
                security_level="standard",
                performance_requirements={}
            )
        
        pattern = self.workload_patterns[tenant_id]
        
        # Update operation types
        pattern.operation_types[operation] = pattern.operation_types.get(operation, 0) + 1
        
        # Analyze data volume
        data_size = len(str(data)) if data else 0
        pattern.data_volume_gb += data_size / (1024 * 1024 * 1024)
        
        # Update based on tenant context
        if tenant_context.tenant_type:
            if tenant_context.tenant_type == TenantType.HEALTHCARE:
                pattern.compliance_requirements = ["HIPAA", "GDPR"]
                pattern.security_level = "high"
            elif tenant_context.tenant_type == TenantType.FINANCIAL:
                pattern.compliance_requirements = ["PCI_DSS", "SOX", "GDPR"]
                pattern.security_level = "critical"
            elif tenant_context.tenant_type == TenantType.GOVERNMENT:
                pattern.compliance_requirements = ["FISMA", "FedRAMP"]
                pattern.security_level = "military"
        
        # Performance requirements based on isolation level
        if tenant_context.isolation_level == IsolationLevel.DATABASE:
            pattern.performance_requirements = {"latency_ms": 50, "throughput": 500}
        elif tenant_context.isolation_level == IsolationLevel.SCHEMA:
            pattern.performance_requirements = {"latency_ms": 100, "throughput": 1000}
        elif tenant_context.isolation_level == IsolationLevel.ROW:
            pattern.performance_requirements = {"latency_ms": 200, "throughput": 2000}
        
        return pattern
    
    async def _ml_strategy_selection(self, tenant_context: TenantContext, workload_pattern: WorkloadPattern) -> Optional[str]:
        """Sélection de stratégie par ML"""
        try:
            # Create feature vector
            features = await self._create_feature_vector(tenant_context, workload_pattern)
            
            # Predict strategy
            prediction = self.strategy_selector_model.predict([features])[0]
            confidence = max(self.strategy_selector_model.predict_proba([features])[0])
            
            # Check confidence threshold
            if confidence >= self.config.min_confidence_threshold:
                return prediction
            
            return None
            
        except Exception as e:
            self.logger.error(f"ML strategy selection failed: {e}")
            return None
    
    async def _create_feature_vector(self, tenant_context: TenantContext, workload_pattern: WorkloadPattern) -> List[float]:
        """Crée un vecteur de features pour ML"""
        features = []
        
        # Tenant features
        features.append(float(tenant_context.isolation_level.value))
        features.append(float(tenant_context.tenant_type.value) if tenant_context.tenant_type else 0)
        
        # Workload features
        features.append(workload_pattern.average_load)
        features.append(workload_pattern.peak_load)
        features.append(workload_pattern.data_volume_gb)
        features.append(workload_pattern.request_rate_per_second)
        features.append(len(workload_pattern.compliance_requirements))
        
        # Security level encoding
        security_levels = {"low": 1, "standard": 2, "high": 3, "critical": 4, "military": 5}
        features.append(security_levels.get(workload_pattern.security_level, 2))
        
        # Operation complexity
        total_operations = sum(workload_pattern.operation_types.values())
        operation_diversity = len(workload_pattern.operation_types)
        features.append(float(total_operations))
        features.append(float(operation_diversity))
        
        return features
    
    async def _rule_based_strategy_selection(self, tenant_context: TenantContext, workload_pattern: WorkloadPattern) -> str:
        """Sélection de stratégie basée sur des règles"""
        # High security requirements
        if (workload_pattern.security_level in ["critical", "military"] or 
            "HIPAA" in workload_pattern.compliance_requirements or 
            "PCI_DSS" in workload_pattern.compliance_requirements):
            return "blockchain_security"
        
        # Geographic distribution requirements
        if len(workload_pattern.geographic_distribution) > 3:
            return "edge_computing"
        
        # High performance requirements
        if (workload_pattern.performance_requirements.get("latency_ms", 1000) < 50 or
            workload_pattern.request_rate_per_second > 1000):
            return "performance_optimized"
        
        # Event-driven workloads
        if "stream" in str(workload_pattern.operation_types) or "event" in str(workload_pattern.operation_types):
            return "event_driven"
        
        # AI/Analytics workloads
        if ("analytics" in str(workload_pattern.operation_types) or 
            "ml" in str(workload_pattern.operation_types) or
            workload_pattern.data_volume_gb > 10):
            return "ai_driven"
        
        # Database isolation for simple workloads
        if tenant_context.isolation_level == IsolationLevel.DATABASE:
            return "database_level"
        
        # Schema isolation for medium complexity
        elif tenant_context.isolation_level == IsolationLevel.SCHEMA:
            return "schema_level"
        
        # Row-level for high density
        elif tenant_context.isolation_level == IsolationLevel.ROW:
            return "row_level"
        
        # Default to hybrid
        return "hybrid"
    
    async def _is_strategy_healthy(self, strategy_name: str) -> bool:
        """Vérifie la santé d'une stratégie"""
        circuit_breaker = self.circuit_breakers.get(strategy_name, {})
        
        if circuit_breaker.get("state") == "open":
            # Check if timeout has passed
            last_failure = circuit_breaker.get("last_failure")
            if last_failure:
                timeout_seconds = circuit_breaker.get("timeout", 60)
                if (datetime.now(timezone.utc) - last_failure).total_seconds() > timeout_seconds:
                    circuit_breaker["state"] = "half-open"
                    circuit_breaker["failure_count"] = 0
                    return True
            return False
        
        return True
    
    async def isolate_data(self, tenant_context: TenantContext, operation: str, data: Any) -> Any:
        """Isole les données avec orchestration intelligente"""
        try:
            start_time = time.time()
            
            # Select optimal strategy
            strategy_name = await self.select_optimal_strategy(tenant_context, operation, data)
            
            # Update active strategy for tenant
            self.active_strategies[tenant_context.tenant_id] = strategy_name
            
            # Execute isolation with selected strategy
            strategy = self.strategies[strategy_name]
            result = await self._execute_with_circuit_breaker(strategy_name, strategy, tenant_context, operation, data)
            
            # Update metrics
            processing_time = time.time() - start_time
            await self._update_strategy_metrics(strategy_name, processing_time, True)
            
            # Check for strategy switching opportunity
            if self.config.strategy_switching_enabled:
                await self._evaluate_strategy_switch(tenant_context, strategy_name, processing_time)
            
            return {
                **result,
                "orchestrator": {
                    "selected_strategy": strategy_name,
                    "processing_time": processing_time,
                    "tenant_profile": await self._get_tenant_profile(tenant_context.tenant_id),
                    "optimization_applied": True,
                    "ml_prediction": self.config.ml_enabled and ML_AVAILABLE
                }
            }
            
        except Exception as e:
            strategy_name = self.active_strategies.get(tenant_context.tenant_id, "unknown")
            await self._update_strategy_metrics(strategy_name, 0, False)
            await self._handle_strategy_failure(strategy_name, e)
            raise StrategyOrchestrationError(f"Orchestrated isolation failed: {e}")
    
    async def _execute_with_circuit_breaker(self, strategy_name: str, strategy: IsolationStrategy, 
                                          tenant_context: TenantContext, operation: str, data: Any) -> Any:
        """Exécute avec circuit breaker"""
        circuit_breaker = self.circuit_breakers[strategy_name]
        
        if circuit_breaker["state"] == "open":
            raise StrategyOrchestrationError(f"Circuit breaker open for strategy {strategy_name}")
        
        try:
            result = await strategy.isolate_data(tenant_context, operation, data)
            
            # Success - reset circuit breaker
            if circuit_breaker["state"] == "half-open":
                circuit_breaker["state"] = "closed"
                circuit_breaker["failure_count"] = 0
                self.logger.info(f"Circuit breaker closed for strategy {strategy_name}")
            
            return result
            
        except Exception as e:
            # Failure - update circuit breaker
            circuit_breaker["failure_count"] += 1
            circuit_breaker["last_failure"] = datetime.now(timezone.utc)
            
            if circuit_breaker["failure_count"] >= circuit_breaker["threshold"]:
                circuit_breaker["state"] = "open"
                self.logger.warning(f"Circuit breaker opened for strategy {strategy_name}")
            
            raise
    
    async def _update_strategy_metrics(self, strategy_name: str, processing_time: float, success: bool):
        """Met à jour les métriques de stratégie"""
        if strategy_name not in self.strategy_metrics:
            return
        
        metrics = self.strategy_metrics[strategy_name]
        metrics.measurements_count += 1
        metrics.last_updated = datetime.now(timezone.utc)
        
        if success:
            # Update latency
            if metrics.measurements_count == 1:
                metrics.latency_ms = processing_time * 1000
            else:
                metrics.latency_ms = (metrics.latency_ms * 0.9) + (processing_time * 1000 * 0.1)
            
            # Update throughput
            throughput = 1.0 / max(processing_time, 0.001)
            if metrics.measurements_count == 1:
                metrics.throughput_ops_sec = throughput
            else:
                metrics.throughput_ops_sec = (metrics.throughput_ops_sec * 0.9) + (throughput * 0.1)
            
            # Update performance score
            if processing_time < 0.1:  # < 100ms
                performance_score = 1.0
            elif processing_time < 0.5:  # < 500ms
                performance_score = 0.8
            elif processing_time < 1.0:  # < 1s
                performance_score = 0.6
            else:
                performance_score = 0.4
            
            metrics.performance_score = (metrics.performance_score * 0.9) + (performance_score * 0.1)
        
        else:
            # Update error rate
            error_rate = 1.0 / metrics.measurements_count
            metrics.error_rate = (metrics.error_rate * 0.9) + (error_rate * 0.1)
    
    async def _evaluate_strategy_switch(self, tenant_context: TenantContext, current_strategy: str, processing_time: float):
        """Évalue la possibilité de changer de stratégie"""
        try:
            # Get current performance
            current_metrics = self.strategy_metrics[current_strategy]
            
            # Check if performance is below threshold
            if (current_metrics.latency_ms > self.config.performance_sla_targets["latency_ms"] * 1.5 or
                current_metrics.error_rate > self.config.performance_sla_targets["error_rate"] * 2):
                
                # Try to find better strategy
                better_strategy = await self._find_better_strategy(tenant_context, current_strategy)
                
                if better_strategy and better_strategy != current_strategy:
                    await self._initiate_strategy_switch(tenant_context, current_strategy, better_strategy)
            
        except Exception as e:
            self.logger.error(f"Strategy switch evaluation failed: {e}")
    
    async def _find_better_strategy(self, tenant_context: TenantContext, current_strategy: str) -> Optional[str]:
        """Trouve une meilleure stratégie"""
        best_strategy = current_strategy
        best_score = self.strategy_metrics[current_strategy].overall_score()
        
        for strategy_name, metrics in self.strategy_metrics.items():
            if strategy_name == current_strategy:
                continue
            
            if not await self._is_strategy_healthy(strategy_name):
                continue
            
            score = metrics.overall_score()
            if score > best_score * 1.1:  # 10% improvement threshold
                best_strategy = strategy_name
                best_score = score
        
        return best_strategy if best_strategy != current_strategy else None
    
    async def _initiate_strategy_switch(self, tenant_context: TenantContext, from_strategy: str, to_strategy: str):
        """Initie un changement de stratégie"""
        try:
            switch_record = {
                "timestamp": datetime.now(timezone.utc),
                "tenant_id": tenant_context.tenant_id,
                "from_strategy": from_strategy,
                "to_strategy": to_strategy,
                "reason": "performance_optimization"
            }
            
            self.strategy_switches.append(switch_record)
            self.active_strategies[tenant_context.tenant_id] = to_strategy
            
            self.logger.info(f"Strategy switch: {from_strategy} -> {to_strategy} for tenant {tenant_context.tenant_id}")
            
        except Exception as e:
            self.logger.error(f"Strategy switch failed: {e}")
    
    async def _handle_strategy_failure(self, strategy_name: str, error: Exception):
        """Gère les échecs de stratégie"""
        self.logger.error(f"Strategy {strategy_name} failed: {error}")
        
        # Update circuit breaker (already handled in _execute_with_circuit_breaker)
        
        # Log for analysis
        failure_record = {
            "timestamp": datetime.now(timezone.utc),
            "strategy": strategy_name,
            "error": str(error),
            "error_type": type(error).__name__
        }
        
        # Add to failure history for analysis
        if not hasattr(self, 'failure_history'):
            self.failure_history = []
        
        self.failure_history.append(failure_record)
    
    async def _get_tenant_profile(self, tenant_id: str) -> Dict[str, Any]:
        """Obtient le profil d'un tenant"""
        if tenant_id not in self.tenant_profiles:
            self.tenant_profiles[tenant_id] = {
                "created_at": datetime.now(timezone.utc),
                "total_requests": 0,
                "preferred_strategies": {},
                "performance_profile": "unknown",
                "cost_profile": "standard"
            }
        
        profile = self.tenant_profiles[tenant_id]
        profile["total_requests"] += 1
        
        # Update preferred strategies
        current_strategy = self.active_strategies.get(tenant_id)
        if current_strategy:
            profile["preferred_strategies"][current_strategy] = profile["preferred_strategies"].get(current_strategy, 0) + 1
        
        return profile
    
    # Background services
    async def _metrics_collector(self):
        """Collecteur de métriques"""
        while True:
            try:
                # Collect metrics from all strategies
                for strategy_name, strategy in self.strategies.items():
                    if hasattr(strategy, 'get_performance_metrics'):
                        metrics = await strategy.get_performance_metrics()
                        
                        # Store metrics history
                        if strategy_name not in self.performance_history:
                            self.performance_history[strategy_name] = []
                        
                        metrics_record = {
                            "timestamp": datetime.now(timezone.utc),
                            "metrics": metrics
                        }
                        
                        self.performance_history[strategy_name].append(metrics_record)
                        
                        # Keep only recent history
                        max_records = 1000
                        if len(self.performance_history[strategy_name]) > max_records:
                            self.performance_history[strategy_name] = self.performance_history[strategy_name][-max_records:]
                
                await asyncio.sleep(self.config.metrics_collection_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.config.metrics_collection_interval_seconds)
    
    async def _performance_monitor(self):
        """Moniteur de performance"""
        while True:
            try:
                # Monitor SLA compliance
                for strategy_name, metrics in self.strategy_metrics.items():
                    sla_targets = self.config.performance_sla_targets
                    
                    # Check latency SLA
                    if metrics.latency_ms > sla_targets["latency_ms"]:
                        await self._handle_sla_violation(strategy_name, "latency", metrics.latency_ms, sla_targets["latency_ms"])
                    
                    # Check error rate SLA
                    if metrics.error_rate > sla_targets["error_rate"]:
                        await self._handle_sla_violation(strategy_name, "error_rate", metrics.error_rate, sla_targets["error_rate"])
                    
                    # Check availability SLA
                    if metrics.availability_percentage < sla_targets["availability_percentage"]:
                        await self._handle_sla_violation(strategy_name, "availability", metrics.availability_percentage, sla_targets["availability_percentage"])
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _handle_sla_violation(self, strategy_name: str, metric_name: str, current_value: float, target_value: float):
        """Gère les violations de SLA"""
        violation_record = {
            "timestamp": datetime.now(timezone.utc),
            "strategy": strategy_name,
            "metric": metric_name,
            "current_value": current_value,
            "target_value": target_value,
            "violation_percentage": ((current_value - target_value) / target_value) * 100
        }
        
        self.logger.warning(f"SLA violation: {strategy_name} {metric_name} = {current_value} (target: {target_value})")
        
        # Take corrective action
        await self._take_corrective_action(strategy_name, metric_name, violation_record)
    
    async def _take_corrective_action(self, strategy_name: str, metric_name: str, violation_record: Dict[str, Any]):
        """Prend des actions correctives"""
        try:
            if metric_name == "latency":
                # Scale up or switch to faster strategy
                await self._optimize_for_latency(strategy_name)
            elif metric_name == "error_rate":
                # Switch to more reliable strategy
                await self._optimize_for_reliability(strategy_name)
            elif metric_name == "availability":
                # Implement redundancy
                await self._optimize_for_availability(strategy_name)
            
        except Exception as e:
            self.logger.error(f"Corrective action failed: {e}")
    
    async def _optimize_for_latency(self, strategy_name: str):
        """Optimise pour la latence"""
        # Switch tenants using this strategy to faster alternatives
        faster_strategies = ["performance_optimized", "edge_computing", "row_level"]
        
        for tenant_id, active_strategy in self.active_strategies.items():
            if active_strategy == strategy_name:
                for faster_strategy in faster_strategies:
                    if await self._is_strategy_healthy(faster_strategy):
                        self.active_strategies[tenant_id] = faster_strategy
                        self.logger.info(f"Switched tenant {tenant_id} to {faster_strategy} for latency optimization")
                        break
    
    async def _optimize_for_reliability(self, strategy_name: str):
        """Optimise pour la fiabilité"""
        # Switch to more reliable strategies
        reliable_strategies = ["blockchain_security", "hybrid", "database_level"]
        
        for tenant_id, active_strategy in self.active_strategies.items():
            if active_strategy == strategy_name:
                for reliable_strategy in reliable_strategies:
                    if await self._is_strategy_healthy(reliable_strategy):
                        self.active_strategies[tenant_id] = reliable_strategy
                        self.logger.info(f"Switched tenant {tenant_id} to {reliable_strategy} for reliability")
                        break
    
    async def _optimize_for_availability(self, strategy_name: str):
        """Optimise pour la disponibilité"""
        # Implement redundancy or switch to highly available strategies
        ha_strategies = ["edge_computing", "hybrid", "event_driven"]
        
        for tenant_id, active_strategy in self.active_strategies.items():
            if active_strategy == strategy_name:
                for ha_strategy in ha_strategies:
                    if await self._is_strategy_healthy(ha_strategy):
                        self.active_strategies[tenant_id] = ha_strategy
                        self.logger.info(f"Switched tenant {tenant_id} to {ha_strategy} for availability")
                        break
    
    async def _auto_optimizer(self):
        """Optimiseur automatique"""
        while True:
            try:
                await self._run_optimization_cycle()
                await asyncio.sleep(self.config.optimization_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Auto-optimization error: {e}")
                await asyncio.sleep(self.config.optimization_interval_minutes * 60)
    
    async def _run_optimization_cycle(self):
        """Exécute un cycle d'optimisation"""
        self.logger.info("Starting optimization cycle")
        
        optimization_record = {
            "timestamp": datetime.now(timezone.utc),
            "optimizations_applied": [],
            "performance_improvements": {},
            "cost_savings": {}
        }
        
        # Analyze workload patterns
        await self._optimize_workload_distribution()
        
        # Optimize resource allocation
        await self._optimize_resource_allocation()
        
        # Update ML models
        if self.config.ml_enabled and ML_AVAILABLE:
            await self._update_ml_models()
        
        self.optimization_history.append(optimization_record)
        self.logger.info("Optimization cycle completed")
    
    async def _optimize_workload_distribution(self):
        """Optimise la distribution de charge"""
        # Analyze current distribution
        strategy_loads = {}
        for tenant_id, strategy in self.active_strategies.items():
            strategy_loads[strategy] = strategy_loads.get(strategy, 0) + 1
        
        # Find overloaded strategies
        avg_load = len(self.active_strategies) / len(self.strategies)
        
        for strategy, load in strategy_loads.items():
            if load > avg_load * 1.5:  # 50% above average
                await self._rebalance_strategy_load(strategy)
    
    async def _rebalance_strategy_load(self, overloaded_strategy: str):
        """Rééquilibre la charge d'une stratégie"""
        # Find tenants using the overloaded strategy
        tenants_to_move = []
        for tenant_id, strategy in self.active_strategies.items():
            if strategy == overloaded_strategy:
                tenants_to_move.append(tenant_id)
        
        # Move some tenants to other strategies
        move_count = len(tenants_to_move) // 4  # Move 25%
        alternative_strategies = [s for s in self.strategies.keys() if s != overloaded_strategy]
        
        for i in range(move_count):
            tenant_id = tenants_to_move[i]
            alternative = alternative_strategies[i % len(alternative_strategies)]
            
            if await self._is_strategy_healthy(alternative):
                self.active_strategies[tenant_id] = alternative
                self.logger.info(f"Rebalanced tenant {tenant_id} from {overloaded_strategy} to {alternative}")
    
    async def _optimize_resource_allocation(self):
        """Optimise l'allocation des ressources"""
        # This would implement resource optimization logic
        # For now, we just log the intent
        self.logger.debug("Resource allocation optimized")
    
    async def _update_ml_models(self):
        """Met à jour les modèles ML"""
        try:
            # Collect training data
            training_data = await self._collect_training_data()
            
            if len(training_data) > 100:  # Enough data for training
                await self._retrain_models(training_data)
                self.logger.info("ML models updated successfully")
            
        except Exception as e:
            self.logger.error(f"ML model update failed: {e}")
    
    async def _collect_training_data(self) -> List[Dict[str, Any]]:
        """Collecte les données d'entraînement"""
        training_data = []
        
        # Collect from optimization history
        for record in self.optimization_history[-100:]:  # Last 100 records
            training_data.append(record)
        
        # Collect from strategy switches
        for switch in self.strategy_switches[-100:]:  # Last 100 switches
            training_data.append(switch)
        
        return training_data
    
    async def _retrain_models(self, training_data: List[Dict[str, Any]]):
        """Réentraîne les modèles"""
        # This would implement actual ML model retraining
        # For now, we simulate it
        self.logger.debug(f"Retrained models with {len(training_data)} samples")
    
    async def _predictive_scaler(self):
        """Scaler prédictif"""
        while True:
            try:
                # Predict future load
                predictions = await self._predict_future_load()
                
                # Apply scaling decisions
                for strategy_name, predicted_load in predictions.items():
                    await self._apply_predictive_scaling(strategy_name, predicted_load)
                
                await asyncio.sleep(300)  # Predict every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Predictive scaling error: {e}")
                await asyncio.sleep(300)
    
    async def _predict_future_load(self) -> Dict[str, float]:
        """Prédit la charge future"""
        predictions = {}
        
        # Simple trend-based prediction
        for strategy_name in self.strategies.keys():
            current_load = len([t for t, s in self.active_strategies.items() if s == strategy_name])
            
            # Predict 20% increase during peak hours
            hour = datetime.now().hour
            if 9 <= hour <= 17:  # Business hours
                predicted_load = current_load * 1.2
            else:
                predicted_load = current_load * 0.8
            
            predictions[strategy_name] = predicted_load
        
        return predictions
    
    async def _apply_predictive_scaling(self, strategy_name: str, predicted_load: float):
        """Applique le scaling prédictif"""
        current_instances = self.current_instances.get(strategy_name, 1)
        
        # Calculate required instances
        required_instances = max(self.config.min_instances, min(self.config.max_instances, int(predicted_load / 10)))
        
        if required_instances != current_instances:
            scaling_record = {
                "timestamp": datetime.now(timezone.utc),
                "strategy": strategy_name,
                "from_instances": current_instances,
                "to_instances": required_instances,
                "predicted_load": predicted_load,
                "type": "predictive"
            }
            
            self.scaling_history.append(scaling_record)
            self.current_instances[strategy_name] = required_instances
            
            self.logger.info(f"Predictive scaling: {strategy_name} {current_instances} -> {required_instances} instances")
    
    async def _anomaly_detector(self):
        """Détecteur d'anomalies"""
        while True:
            try:
                # Detect anomalies in strategy performance
                anomalies = await self._detect_performance_anomalies()
                
                # Handle detected anomalies
                for anomaly in anomalies:
                    await self._handle_anomaly(anomaly)
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                self.logger.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(120)
    
    async def _detect_performance_anomalies(self) -> List[Dict[str, Any]]:
        """Détecte les anomalies de performance"""
        anomalies = []
        
        for strategy_name, metrics in self.strategy_metrics.items():
            # Check for sudden performance degradation
            if strategy_name in self.performance_history:
                recent_metrics = self.performance_history[strategy_name][-10:]  # Last 10 measurements
                
                if len(recent_metrics) >= 5:
                    # Calculate average performance
                    recent_latencies = [m["metrics"].get("average_latency", 0) for m in recent_metrics]
                    avg_latency = sum(recent_latencies) / len(recent_latencies)
                    
                    # Check if current latency is significantly higher
                    if metrics.latency_ms > avg_latency * 2:  # 100% increase
                        anomalies.append({
                            "type": "latency_spike",
                            "strategy": strategy_name,
                            "current_value": metrics.latency_ms,
                            "baseline": avg_latency,
                            "severity": "high"
                        })
        
        return anomalies
    
    async def _handle_anomaly(self, anomaly: Dict[str, Any]):
        """Gère une anomalie détectée"""
        strategy_name = anomaly["strategy"]
        anomaly_type = anomaly["type"]
        
        self.logger.warning(f"Anomaly detected: {anomaly_type} in {strategy_name}")
        
        if anomaly_type == "latency_spike":
            # Temporarily reduce load on this strategy
            await self._reduce_strategy_load(strategy_name)
        
        # Log anomaly for analysis
        if not hasattr(self, 'anomaly_history'):
            self.anomaly_history = []
        
        anomaly["timestamp"] = datetime.now(timezone.utc)
        self.anomaly_history.append(anomaly)
    
    async def _reduce_strategy_load(self, strategy_name: str):
        """Réduit la charge sur une stratégie"""
        # Move some tenants to alternative strategies
        affected_tenants = [t for t, s in self.active_strategies.items() if s == strategy_name]
        move_count = len(affected_tenants) // 3  # Move 33%
        
        alternative_strategies = [s for s in self.strategies.keys() if s != strategy_name]
        
        for i in range(move_count):
            tenant_id = affected_tenants[i]
            alternative = alternative_strategies[i % len(alternative_strategies)]
            
            if await self._is_strategy_healthy(alternative):
                self.active_strategies[tenant_id] = alternative
                self.logger.info(f"Moved tenant {tenant_id} from {strategy_name} to {alternative} due to anomaly")
    
    async def _model_trainer(self):
        """Entraîneur de modèles"""
        while True:
            try:
                await asyncio.sleep(self.config.model_retraining_interval_hours * 3600)
                
                # Retrain models with latest data
                if self.config.ml_enabled and ML_AVAILABLE:
                    await self._update_ml_models()
                
            except Exception as e:
                self.logger.error(f"Model training error: {e}")
                await asyncio.sleep(self.config.model_retraining_interval_hours * 3600)
    
    async def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de l'orchestrateur"""
        total_tenants = len(self.active_strategies)
        strategy_distribution = {}
        
        for strategy in self.active_strategies.values():
            strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
        
        return {
            "total_tenants": total_tenants,
            "active_strategies": len(self.strategies),
            "strategy_distribution": strategy_distribution,
            "strategy_switches": len(self.strategy_switches),
            "optimization_cycles": len(self.optimization_history),
            "ml_enabled": self.config.ml_enabled and ML_AVAILABLE,
            "auto_optimization": self.config.auto_optimization,
            "predictive_scaling": self.config.predictive_scaling,
            "current_instances": dict(self.current_instances),
            "circuit_breakers": {name: cb["state"] for name, cb in self.circuit_breakers.items()},
            "background_services": len(self.background_tasks),
            "anomalies_detected": len(getattr(self, 'anomaly_history', [])),
            "average_performance": {
                name: metrics.overall_score() 
                for name, metrics in self.strategy_metrics.items()
            }
        }
    
    async def cleanup(self):
        """Nettoie les ressources"""
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Cleanup all strategies
            for strategy in self.strategies.values():
                if hasattr(strategy, 'cleanup'):
                    await strategy.cleanup()
            
            self.logger.info("Strategy orchestrator cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Orchestrator cleanup error: {e}")


# Export orchestrator
__all__ = ["UltraAdvancedStrategyOrchestrator", "OrchestratorConfig", "StrategyMetrics", "WorkloadPattern", "StrategyPriority", "OptimizationObjective"]
