"""
Factory pattern pour l'instanciation dynamique des algorithmes de monitoring.

Ce module implémente le pattern Factory pour :
- Création dynamique d'instances d'algorithmes
- Configuration automatique selon l'environnement
- Injection de dépendances (cache, métriques, etc.)
- Gestion du cycle de vie des modèles
- Support pour les algorithmes hot-swappables

Architecture extensible pour nouveaux algorithmes.
"""

import asyncio
import importlib
from typing import Dict, Any, Optional, Type, Union, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging

from .config import (
    AlgorithmType, 
    Environment, 
    ConfigurationManager,
    get_config
)
from .utils import (
    PrometheusMetricsManager,
    RedisCache,
    DataProcessor,
    ModelVersionManager
)

logger = logging.getLogger(__name__)

@dataclass
class AlgorithmDependencies:
    """Dépendances injectées dans les algorithmes."""
    metrics_manager: PrometheusMetricsManager
    cache: RedisCache
    data_processor: DataProcessor
    model_version_manager: ModelVersionManager
    config_manager: ConfigurationManager

class BaseAlgorithm(ABC):
    """Classe de base pour tous les algorithmes."""
    
    def __init__(self, 
                 algorithm_type: AlgorithmType,
                 model_name: str,
                 dependencies: AlgorithmDependencies):
        self.algorithm_type = algorithm_type
        self.model_name = model_name
        self.dependencies = dependencies
        self.config = dependencies.config_manager.get_algorithm_config(algorithm_type)
        self.model_config = self.config.models.get(model_name)
        self.is_initialized = False
        self.last_training_time: Optional[datetime] = None
        
        # Validation de la configuration
        if not self.model_config:
            raise ValueError(f"No configuration found for model {model_name}")
        
        if not self.model_config.enabled:
            raise ValueError(f"Model {model_name} is disabled")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialise l'algorithme."""
        pass
    
    @abstractmethod
    async def process(self, data: Any, **kwargs) -> Any:
        """Traite les données."""
        pass
    
    @abstractmethod
    async def train(self, training_data: Any, **kwargs) -> None:
        """Entraîne le modèle."""
        pass
    
    async def cleanup(self) -> None:
        """Nettoie les ressources."""
        pass
    
    @property
    def cache_key_prefix(self) -> str:
        """Préfixe pour les clés de cache."""
        return f"{self.algorithm_type.value}:{self.model_name}"

class AlgorithmFactory:
    """Factory pour créer des instances d'algorithmes."""
    
    def __init__(self, dependencies: AlgorithmDependencies):
        self.dependencies = dependencies
        self.algorithm_registry: Dict[str, Type[BaseAlgorithm]] = {}
        self.active_instances: Dict[str, BaseAlgorithm] = {}
        self._register_default_algorithms()
    
    def _register_default_algorithms(self) -> None:
        """Enregistre les algorithmes par défaut."""
        
        # Import dynamique pour éviter les dépendances circulaires
        try:
            # Anomaly Detection
            from .anomaly_detection import (
                IsolationForestDetector,
                OneClassSVMDetector,
                LSTMAutoencoderDetector,
                EnsembleAnomalyDetector
            )
            
            self.register_algorithm('isolation_forest', IsolationForestDetector)
            self.register_algorithm('one_class_svm', OneClassSVMDetector)
            self.register_algorithm('lstm_autoencoder', LSTMAutoencoderDetector)
            self.register_algorithm('ensemble_anomaly', EnsembleAnomalyDetector)
            
        except ImportError as e:
            logger.warning(f"Could not import anomaly detection algorithms: {e}")
        
        try:
            # Alert Classification
            from .alert_classification import (
                AlertClassifier,
                SeverityPredictor,
                BusinessImpactAnalyzer,
                EnsembleClassifier
            )
            
            self.register_algorithm('alert_classifier', AlertClassifier)
            self.register_algorithm('severity_predictor', SeverityPredictor)
            self.register_algorithm('business_impact_analyzer', BusinessImpactAnalyzer)
            self.register_algorithm('ensemble_classifier', EnsembleClassifier)
            
        except ImportError as e:
            logger.warning(f"Could not import alert classification algorithms: {e}")
        
        try:
            # Correlation Engine
            from .correlation_engine import (
                MetricCorrelationAnalyzer,
                EventCorrelationAnalyzer,
                CausalityDetector,
                CorrelationEngine
            )
            
            self.register_algorithm('metric_correlation', MetricCorrelationAnalyzer)
            self.register_algorithm('event_correlation', EventCorrelationAnalyzer)
            self.register_algorithm('causality_detector', CausalityDetector)
            self.register_algorithm('correlation_engine', CorrelationEngine)
            
        except ImportError as e:
            logger.warning(f"Could not import correlation engine algorithms: {e}")
        
        try:
            # Prediction Models
            from .prediction_models import (
                IncidentPredictor,
                CapacityForecaster,
                TrendAnalyzer
            )
            
            self.register_algorithm('incident_predictor', IncidentPredictor)
            self.register_algorithm('capacity_forecaster', CapacityForecaster)
            self.register_algorithm('trend_analyzer', TrendAnalyzer)
            
        except ImportError as e:
            logger.warning(f"Could not import prediction models: {e}")
        
        try:
            # Behavioral Analysis
            from .behavioral_analysis import (
                UserBehaviorAnalyzer,
                SystemBehaviorProfiler,
                DeviationDetector,
                BehaviorAnalysisEngine
            )
            
            self.register_algorithm('user_behavior_analyzer', UserBehaviorAnalyzer)
            self.register_algorithm('system_behavior_profiler', SystemBehaviorProfiler)
            self.register_algorithm('deviation_detector', DeviationDetector)
            self.register_algorithm('behavior_analysis_engine', BehaviorAnalysisEngine)
            
        except ImportError as e:
            logger.warning(f"Could not import behavioral analysis algorithms: {e}")
    
    def register_algorithm(self, name: str, algorithm_class: Type[BaseAlgorithm]) -> None:
        """Enregistre une classe d'algorithme."""
        
        if not issubclass(algorithm_class, BaseAlgorithm):
            raise ValueError(f"Algorithm class must inherit from BaseAlgorithm")
        
        self.algorithm_registry[name] = algorithm_class
        logger.info(f"Registered algorithm: {name}")
    
    def unregister_algorithm(self, name: str) -> None:
        """Désenregistre un algorithme."""
        
        if name in self.algorithm_registry:
            del self.algorithm_registry[name]
            logger.info(f"Unregistered algorithm: {name}")
    
    async def create_algorithm(self, 
                              algorithm_type: AlgorithmType,
                              model_name: str,
                              force_recreate: bool = False) -> BaseAlgorithm:
        """Crée une instance d'algorithme."""
        
        instance_key = f"{algorithm_type.value}:{model_name}"
        
        # Réutilisation d'instance existante
        if not force_recreate and instance_key in self.active_instances:
            instance = self.active_instances[instance_key]
            if instance.is_initialized:
                return instance
            else:
                # Instance existe mais pas initialisée
                await instance.initialize()
                return instance
        
        # Vérification de l'enregistrement
        if model_name not in self.algorithm_registry:
            raise ValueError(f"Algorithm {model_name} not registered")
        
        # Création de l'instance
        algorithm_class = self.algorithm_registry[model_name]
        
        try:
            instance = algorithm_class(
                algorithm_type=algorithm_type,
                model_name=model_name,
                dependencies=self.dependencies
            )
            
            # Initialisation
            await instance.initialize()
            
            # Stockage de l'instance
            self.active_instances[instance_key] = instance
            
            logger.info(f"Created and initialized algorithm: {instance_key}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create algorithm {instance_key}: {e}")
            raise
    
    async def get_algorithm(self, 
                           algorithm_type: AlgorithmType,
                           model_name: str) -> Optional[BaseAlgorithm]:
        """Récupère une instance d'algorithme existante."""
        
        instance_key = f"{algorithm_type.value}:{model_name}"
        return self.active_instances.get(instance_key)
    
    async def destroy_algorithm(self, 
                               algorithm_type: AlgorithmType,
                               model_name: str) -> None:
        """Détruit une instance d'algorithme."""
        
        instance_key = f"{algorithm_type.value}:{model_name}"
        
        if instance_key in self.active_instances:
            instance = self.active_instances[instance_key]
            
            try:
                await instance.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup of {instance_key}: {e}")
            
            del self.active_instances[instance_key]
            logger.info(f"Destroyed algorithm: {instance_key}")
    
    async def list_active_algorithms(self) -> List[str]:
        """Liste les algorithmes actifs."""
        return list(self.active_instances.keys())
    
    async def reload_algorithm(self, 
                              algorithm_type: AlgorithmType,
                              model_name: str) -> BaseAlgorithm:
        """Recharge un algorithme (hot-reload)."""
        
        # Destruction de l'instance existante
        await self.destroy_algorithm(algorithm_type, model_name)
        
        # Création d'une nouvelle instance
        return await self.create_algorithm(algorithm_type, model_name, force_recreate=True)
    
    async def reload_all_algorithms(self) -> None:
        """Recharge tous les algorithmes actifs."""
        
        active_keys = list(self.active_instances.keys())
        
        for instance_key in active_keys:
            algorithm_type_str, model_name = instance_key.split(':', 1)
            algorithm_type = AlgorithmType(algorithm_type_str)
            
            try:
                await self.reload_algorithm(algorithm_type, model_name)
                logger.info(f"Reloaded algorithm: {instance_key}")
            except Exception as e:
                logger.error(f"Failed to reload algorithm {instance_key}: {e}")
    
    def get_registered_algorithms(self) -> List[str]:
        """Retourne la liste des algorithmes enregistrés."""
        return list(self.algorithm_registry.keys())

class AlgorithmManager:
    """Gestionnaire global des algorithmes."""
    
    def __init__(self, dependencies: AlgorithmDependencies):
        self.dependencies = dependencies
        self.factory = AlgorithmFactory(dependencies)
        self.algorithm_configs: Dict[AlgorithmType, Dict[str, Any]] = {}
        self._load_configurations()
    
    def _load_configurations(self) -> None:
        """Charge les configurations de tous les algorithmes."""
        
        for algorithm_type in AlgorithmType:
            try:
                config = get_config(algorithm_type)
                self.algorithm_configs[algorithm_type] = config
                logger.info(f"Loaded configuration for {algorithm_type.value}")
            except Exception as e:
                logger.error(f"Failed to load config for {algorithm_type.value}: {e}")
    
    async def initialize_enabled_algorithms(self) -> None:
        """Initialise tous les algorithmes activés."""
        
        for algorithm_type, config in self.algorithm_configs.items():
            for model_name, model_config in config.models.items():
                if model_config.enabled:
                    try:
                        await self.factory.create_algorithm(algorithm_type, model_name)
                        logger.info(f"Initialized {algorithm_type.value}:{model_name}")
                    except Exception as e:
                        logger.error(
                            f"Failed to initialize {algorithm_type.value}:{model_name}: {e}"
                        )
    
    async def process_with_algorithm(self, 
                                   algorithm_type: AlgorithmType,
                                   model_name: str,
                                   data: Any,
                                   **kwargs) -> Any:
        """Traite des données avec un algorithme spécifique."""
        
        algorithm = await self.factory.get_algorithm(algorithm_type, model_name)
        
        if not algorithm:
            # Tentative de création si pas trouvé
            algorithm = await self.factory.create_algorithm(algorithm_type, model_name)
        
        return await algorithm.process(data, **kwargs)
    
    async def train_algorithm(self, 
                             algorithm_type: AlgorithmType,
                             model_name: str,
                             training_data: Any,
                             **kwargs) -> None:
        """Entraîne un algorithme spécifique."""
        
        algorithm = await self.factory.get_algorithm(algorithm_type, model_name)
        
        if not algorithm:
            algorithm = await self.factory.create_algorithm(algorithm_type, model_name)
        
        await algorithm.train(training_data, **kwargs)
    
    async def batch_process(self, 
                           requests: List[Dict[str, Any]]) -> List[Any]:
        """Traite plusieurs requêtes en parallèle."""
        
        tasks = []
        
        for request in requests:
            algorithm_type = AlgorithmType(request['algorithm_type'])
            model_name = request['model_name']
            data = request['data']
            kwargs = request.get('kwargs', {})
            
            task = self.process_with_algorithm(
                algorithm_type, model_name, data, **kwargs
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_algorithm_status(self) -> Dict[str, Dict[str, Any]]:
        """Retourne le statut de tous les algorithmes."""
        
        status = {}
        active_algorithms = await self.factory.list_active_algorithms()
        
        for instance_key in active_algorithms:
            algorithm_type_str, model_name = instance_key.split(':', 1)
            algorithm = await self.factory.get_algorithm(
                AlgorithmType(algorithm_type_str), model_name
            )
            
            if algorithm:
                status[instance_key] = {
                    'is_initialized': algorithm.is_initialized,
                    'last_training_time': algorithm.last_training_time,
                    'config_enabled': algorithm.model_config.enabled if algorithm.model_config else False,
                    'model_config': algorithm.model_config.__dict__ if algorithm.model_config else None
                }
        
        return status
    
    async def shutdown(self) -> None:
        """Arrêt propre de tous les algorithmes."""
        
        active_algorithms = await self.factory.list_active_algorithms()
        
        for instance_key in active_algorithms:
            algorithm_type_str, model_name = instance_key.split(':', 1)
            algorithm_type = AlgorithmType(algorithm_type_str)
            
            try:
                await self.factory.destroy_algorithm(algorithm_type, model_name)
            except Exception as e:
                logger.error(f"Error shutting down {instance_key}: {e}")
        
        logger.info("Algorithm manager shutdown complete")

# Factory globale pour faciliter l'utilisation
_global_factory: Optional[AlgorithmFactory] = None
_global_manager: Optional[AlgorithmManager] = None

def get_algorithm_factory(dependencies: Optional[AlgorithmDependencies] = None) -> AlgorithmFactory:
    """Retourne la factory globale d'algorithmes."""
    
    global _global_factory
    
    if _global_factory is None:
        if dependencies is None:
            raise ValueError("Dependencies required for first initialization")
        _global_factory = AlgorithmFactory(dependencies)
    
    return _global_factory

def get_algorithm_manager(dependencies: Optional[AlgorithmDependencies] = None) -> AlgorithmManager:
    """Retourne le gestionnaire global d'algorithmes."""
    
    global _global_manager
    
    if _global_manager is None:
        if dependencies is None:
            raise ValueError("Dependencies required for first initialization")
        _global_manager = AlgorithmManager(dependencies)
    
    return _global_manager

async def create_default_dependencies() -> AlgorithmDependencies:
    """Crée les dépendances par défaut."""
    
    from .utils import METRICS_MANAGER, MODEL_VERSION_MANAGER
    import redis.asyncio as aioredis
    
    # Configuration Redis
    redis_client = aioredis.from_url("redis://localhost:6379/0")
    cache = RedisCache(redis_client)
    
    # Autres dépendances
    data_processor = DataProcessor()
    config_manager = ConfigurationManager()
    
    return AlgorithmDependencies(
        metrics_manager=METRICS_MANAGER,
        cache=cache,
        data_processor=data_processor,
        model_version_manager=MODEL_VERSION_MANAGER,
        config_manager=config_manager
    )

# Export des classes principales
__all__ = [
    'BaseAlgorithm',
    'AlgorithmFactory',
    'AlgorithmManager',
    'AlgorithmDependencies',
    'get_algorithm_factory',
    'get_algorithm_manager',
    'create_default_dependencies'
]
