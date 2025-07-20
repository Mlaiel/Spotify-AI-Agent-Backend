"""
Analytics Module - Enterprise Grade AI-Powered Analytics Engine
==================================================================

Système d'analyse avancé pour l'agent Spotify IA avec:
- Machine Learning en temps réel
- Détection d'anomalies IA
- Analytics prédictifs
- Monitoring intelligent
- Alertes contextuelles
- Analyse comportementale

Architecture:
- Core Engine: Moteur principal d'analytics
- ML Pipeline: Pipeline de machine learning
- Alert System: Système d'alertes intelligent
- Storage Engine: Stockage haute performance
- Dashboard API: Interface de visualisation
- Quantum Analytics: Calculs quantiques avancés

Technologies:
- TensorFlow/PyTorch pour ML
- Redis/PostgreSQL pour stockage
- WebSockets pour temps réel
- Prometheus/Grafana pour monitoring
- Apache Kafka pour streaming
- Apache Spark pour big data

Auteur: Architecture développée selon les spécifications métier
Version: 1.0.0 - Production Ready
"""

from typing import Optional, Dict, Any, List
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Core components
from .core import (
    AnalyticsEngine,
    MetricsCollector,
    DataProcessor,
    RealTimeAnalyzer
)

# ML components
from .ml import (
    MLModelManager,
    PredictiveAnalyzer,
    AnomalyDetector,
    BehaviorAnalyzer
)

# Storage components
from .storage import (
    AnalyticsStorage,
    MetricsStorage,
    CacheManager,
    DataWarehouse
)

# Dashboard components
from .dashboard import (
    DashboardEngine,
    VisualizationGenerator,
    ReportGenerator,
    MetricsDashboard
)

# Alerts components
from .alerts import (
    AlertEngine,
    NotificationManager,
    EscalationManager,
    AlertAnalyzer
)

# Utils components
from .utils import (
    DataValidator,
    PerformanceOptimizer,
    SecurityManager,
    ConfigManager
)

# Configuration
from .config import (
    AnalyticsConfig,
    MLConfig,
    StorageConfig,
    AlertConfig
)

# Models
from .models import (
    AnalyticsMetric,
    MLModel,
    Alert,
    Dashboard,
    Tenant,
    User
)

class AnalyticsLevel(Enum):
    """Niveaux d'analyse disponibles"""
    BASIC = "basic"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"
    QUANTUM = "quantum"

class AnalyticsMode(Enum):
    """Modes d'exécution des analytics"""
    REALTIME = "realtime"
    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"

@dataclass
class AnalyticsContext:
    """Contexte d'exécution des analytics"""
    tenant_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    level: AnalyticsLevel = AnalyticsLevel.ADVANCED
    mode: AnalyticsMode = AnalyticsMode.HYBRID
    config: Optional[Dict[str, Any]] = None

class AnalyticsFactory:
    """Factory pour créer des instances d'analytics"""
    
    _instances: Dict[str, Any] = {}
    _config: Optional[AnalyticsConfig] = None
    
    @classmethod
    async def create_engine(
        cls,
        context: AnalyticsContext,
        config: Optional[AnalyticsConfig] = None
    ) -> AnalyticsEngine:
        """Crée une instance du moteur d'analytics"""
        key = f"{context.tenant_id}_{context.level.value}_{context.mode.value}"
        
        if key not in cls._instances:
            engine_config = config or cls._config or AnalyticsConfig()
            
            engine = AnalyticsEngine(
                context=context,
                config=engine_config
            )
            
            await engine.initialize()
            cls._instances[key] = engine
            
        return cls._instances[key]
    
    @classmethod
    async def create_ml_analyzer(
        cls,
        context: AnalyticsContext,
        model_type: str = "default"
    ) -> PredictiveAnalyzer:
        """Crée un analyseur ML"""
        analyzer = PredictiveAnalyzer(
            context=context,
            model_type=model_type
        )
        await analyzer.initialize()
        return analyzer
    
    @classmethod
    async def create_dashboard(
        cls,
        context: AnalyticsContext,
        dashboard_type: str = "executive"
    ) -> DashboardEngine:
        """Crée un dashboard"""
        dashboard = DashboardEngine(
            context=context,
            dashboard_type=dashboard_type
        )
        await dashboard.initialize()
        return dashboard

class AnalyticsManager:
    """Gestionnaire principal des analytics"""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()
        self.engines: Dict[str, AnalyticsEngine] = {}
        self.logger = logging.getLogger(__name__)
        self._initialized = False
    
    async def initialize(self):
        """Initialise le gestionnaire d'analytics"""
        if self._initialized:
            return
        
        try:
            # Initialiser les composants de base
            await self._initialize_storage()
            await self._initialize_ml_models()
            await self._initialize_alert_system()
            await self._initialize_monitoring()
            
            self._initialized = True
            self.logger.info("Analytics Manager initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def _initialize_storage(self):
        """Initialise les systèmes de stockage"""
        storage = AnalyticsStorage(self.config.storage)
        await storage.initialize()
    
    async def _initialize_ml_models(self):
        """Initialise les modèles ML"""
        ml_manager = MLModelManager(self.config.ml)
        await ml_manager.load_models()
    
    async def _initialize_alert_system(self):
        """Initialise le système d'alertes"""
        alert_engine = AlertEngine(self.config.alerts)
        await alert_engine.initialize()
    
    async def _initialize_monitoring(self):
        """Initialise le monitoring"""
        monitor = MetricsCollector(self.config.monitoring)
        await monitor.start()
    
    async def get_engine(self, context: AnalyticsContext) -> AnalyticsEngine:
        """Récupère ou crée un moteur d'analytics"""
        return await AnalyticsFactory.create_engine(context, self.config)
    
    async def process_event(self, event: Dict[str, Any], context: AnalyticsContext):
        """Traite un événement analytics"""
        engine = await self.get_engine(context)
        return await engine.process_event(event)
    
    async def generate_insights(self, context: AnalyticsContext) -> Dict[str, Any]:
        """Génère des insights analytics"""
        engine = await self.get_engine(context)
        return await engine.generate_insights()
    
    async def shutdown(self):
        """Arrête proprement tous les composants"""
        for engine in self.engines.values():
            await engine.shutdown()
        
        self._initialized = False
        self.logger.info("Analytics Manager arrêté")

# Instance globale
_analytics_manager: Optional[AnalyticsManager] = None

async def get_analytics_manager(config: Optional[AnalyticsConfig] = None) -> AnalyticsManager:
    """Récupère l'instance globale du gestionnaire d'analytics"""
    global _analytics_manager
    
    if _analytics_manager is None:
        _analytics_manager = AnalyticsManager(config)
        await _analytics_manager.initialize()
    
    return _analytics_manager

# Fonctions utilitaires
async def quick_analyze(
    data: Dict[str, Any],
    tenant_id: str,
    analysis_type: str = "comprehensive"
) -> Dict[str, Any]:
    """Analyse rapide de données"""
    context = AnalyticsContext(
        tenant_id=tenant_id,
        level=AnalyticsLevel.ADVANCED,
        mode=AnalyticsMode.REALTIME
    )
    
    manager = await get_analytics_manager()
    engine = await manager.get_engine(context)
    
    return await engine.quick_analyze(data, analysis_type)

async def generate_report(
    tenant_id: str,
    report_type: str = "summary",
    time_range: Optional[tuple] = None
) -> Dict[str, Any]:
    """Génère un rapport analytics"""
    context = AnalyticsContext(
        tenant_id=tenant_id,
        level=AnalyticsLevel.ENTERPRISE
    )
    
    dashboard = await AnalyticsFactory.create_dashboard(context, report_type)
    return await dashboard.generate_report(time_range)

# Exports principaux
__all__ = [
    # Core classes
    'AnalyticsEngine',
    'AnalyticsManager',
    'AnalyticsFactory',
    'AnalyticsContext',
    
    # Enums
    'AnalyticsLevel',
    'AnalyticsMode',
    
    # Components
    'MLModelManager',
    'PredictiveAnalyzer',
    'AnomalyDetector',
    'DashboardEngine',
    'AlertEngine',
    'AnalyticsStorage',
    
    # Models
    'AnalyticsMetric',
    'MLModel',
    'Alert',
    'Dashboard',
    
    # Utils
    'get_analytics_manager',
    'quick_analyze',
    'generate_report',
    
    # Config
    'AnalyticsConfig',
    'MLConfig',
    'StorageConfig',
    'AlertConfig'
]

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

# Core modules
from .core import AnalyticsEngine, MetricsCollector, AlertManager
from .processors import (
    RealTimeProcessor, 
    BatchProcessor, 
    StreamProcessor,
    MLProcessor
)
from .storage import (
    TimeSeriesStorage, 
    MetricsStorage, 
    EventStorage,
    CacheStorage
)
from .models import (
    Metric, 
    Event, 
    Alert, 
    Dashboard,
    Tenant,
    User,
    Session
)
from .dashboard import (
    DashboardManager, 
    VisualizationEngine, 
    ReportGenerator
)
from .ml import (
    AnomalyDetector, 
    PredictiveAnalytics, 
    RecommendationEngine,
    BehaviorAnalyzer
)

# Configuration
from .config import AnalyticsConfig, Settings
from .utils import Logger, Validator, Formatter

__version__ = "2.0.0"
__author__ = "Fahed Mlaiel"

# Exports publics
__all__ = [
    # Core
    "AnalyticsEngine",
    "MetricsCollector", 
    "AlertManager",
    
    # Processors
    "RealTimeProcessor",
    "BatchProcessor", 
    "StreamProcessor",
    "MLProcessor",
    
    # Storage
    "TimeSeriesStorage",
    "MetricsStorage", 
    "EventStorage",
    "CacheStorage",
    
    # Models
    "Metric", 
    "Event", 
    "Alert", 
    "Dashboard",
    "Tenant", 
    "User", 
    "Session",
    
    # Dashboard
    "DashboardManager", 
    "VisualizationEngine", 
    "ReportGenerator",
    
    # ML
    "AnomalyDetector", 
    "PredictiveAnalytics", 
    "RecommendationEngine",
    "BehaviorAnalyzer",
    
    # Utils
    "AnalyticsConfig", 
    "Settings",
    "Logger", 
    "Validator", 
    "Formatter"
]

# Logger configuration
logger = logging.getLogger(__name__)

@dataclass
class AnalyticsModule:
    """Module principal d'analytics."""
    
    engine: AnalyticsEngine
    config: AnalyticsConfig
    processors: Dict[str, Any]
    storage: Dict[str, Any]
    ml_models: Dict[str, Any]
    
    def __post_init__(self):
        """Initialisation post-création."""
        self.logger = Logger(__name__)
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialise le module d'analytics."""
        try:
            # Initialiser les composants
            await self.engine.start()
            await self._initialize_processors()
            await self._initialize_storage()
            await self._initialize_ml_models()
            
            self.is_initialized = True
            self.logger.info("Module d'analytics initialisé avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            return False
    
    async def _initialize_processors(self):
        """Initialise les processeurs."""
        for name, processor in self.processors.items():
            await processor.start()
            self.logger.debug(f"Processeur {name} initialisé")
    
    async def _initialize_storage(self):
        """Initialise les systèmes de stockage."""
        for name, storage in self.storage.items():
            await storage.connect()
            self.logger.debug(f"Storage {name} connecté")
    
    async def _initialize_ml_models(self):
        """Initialise les modèles ML."""
        for name, model in self.ml_models.items():
            await model.load()
            self.logger.debug(f"Modèle ML {name} chargé")

# Instance globale (singleton)
_analytics_instance: Optional[AnalyticsModule] = None

async def get_analytics() -> AnalyticsModule:
    """Retourne l'instance singleton du module d'analytics."""
    global _analytics_instance
    
    if _analytics_instance is None:
        config = AnalyticsConfig()
        engine = AnalyticsEngine(config)
        
        processors = {
            "realtime": RealTimeProcessor(config),
            "batch": BatchProcessor(config),
            "stream": StreamProcessor(config),
            "ml": MLProcessor(config)
        }
        
        storage = {
            "timeseries": TimeSeriesStorage(config),
            "metrics": MetricsStorage(config),
            "events": EventStorage(config),
            "cache": CacheStorage(config)
        }
        
        ml_models = {
            "anomaly": AnomalyDetector(config),
            "predictive": PredictiveAnalytics(config),
            "recommendations": RecommendationEngine(config),
            "behavior": BehaviorAnalyzer(config)
        }
        
        _analytics_instance = AnalyticsModule(
            engine=engine,
            config=config,
            processors=processors,
            storage=storage,
            ml_models=ml_models
        )
        
        await _analytics_instance.initialize()
    
    return _analytics_instance

async def shutdown_analytics():
    """Arrête proprement le module d'analytics."""
    global _analytics_instance
    
    if _analytics_instance and _analytics_instance.is_initialized:
        await _analytics_instance.engine.stop()
        _analytics_instance = None
        logger.info("Module d'analytics arrêté")
