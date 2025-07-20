# 🎵 Core ML Analytics Engine
# ============================
# 
# Orchestrateur principal du système ML Analytics
# Architecture enterprise avec patterns avancés
#
# 🎖️ Expert: Lead Dev + Architecte IA

"""
🧠 Core ML Analytics Engine
===========================

Central orchestrator for the ML Analytics system providing:
- Model lifecycle management
- Pipeline orchestration
- Resource optimization
- Performance monitoring
- Fault tolerance and recovery
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Callable, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
import threading
import weakref
from abc import ABC, abstractmethod
import traceback
import sys
from pathlib import Path

# ML Analytics imports
from .config import MLAnalyticsConfig, ModelConfig, PipelineConfig
from .exceptions import (
    MLAnalyticsError, ModelNotFoundError, PipelineError, 
    InferenceError, TrainingError
)
from .monitoring import MLMetricsCollector, PerformanceTracker
from .utils import (
    ensure_async, retry_with_backoff, circuit_breaker,
    validate_input_data, normalize_tensor_data
)


class MLEngineState(IntEnum):
    """États du moteur ML Analytics"""
    INITIALIZING = 0
    READY = 1
    TRAINING = 2
    INFERRING = 3
    OPTIMIZING = 4
    ERROR = 5
    SHUTTING_DOWN = 6


class ModelType(Enum):
    """Types de modèles ML supportés"""
    RECOMMENDATION = "recommendation"
    AUDIO_ANALYSIS = "audio_analysis"
    SENTIMENT_NLP = "sentiment_nlp"
    USER_BEHAVIOR = "user_behavior"
    CONTENT_FILTERING = "content_filtering"
    TREND_PREDICTION = "trend_prediction"
    PERSONALIZATION = "personalization"


@dataclass
class MLEngineConfig:
    """Configuration du moteur ML Analytics"""
    max_concurrent_inferences: int = 100
    model_cache_size: int = 50
    auto_optimization: bool = True
    monitoring_enabled: bool = True
    metrics_collection_interval: float = 30.0
    health_check_interval: float = 60.0
    model_warmup_timeout: float = 300.0
    inference_timeout: float = 30.0
    training_timeout: float = 3600.0
    cleanup_interval: float = 900.0  # 15 minutes
    checkpoint_interval: float = 1800.0  # 30 minutes


class IMLModel(Protocol):
    """Interface pour les modèles ML"""
    
    async def predict(self, input_data: Any) -> Any:
        """Prédiction asynchrone"""
        ...
    
    async def train(self, training_data: Any) -> Dict[str, Any]:
        """Entraînement asynchrone"""
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Métadonnées du modèle"""
        ...
    
    def is_ready(self) -> bool:
        """État de préparation du modèle"""
        ...


class MLModelManager:
    """Gestionnaire de modèles ML avec cache intelligent"""
    
    def __init__(self, config: MLEngineConfig):
        self.config = config
        self.models: Dict[str, IMLModel] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.model_usage_stats: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.MLModelManager")
        
    async def register_model(
        self, 
        model_id: str, 
        model: IMLModel, 
        model_type: ModelType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Enregistrer un nouveau modèle"""
        try:
            with self.lock:
                self.models[model_id] = model
                self.model_metadata[model_id] = {
                    "type": model_type.value,
                    "registered_at": datetime.utcnow().isoformat(),
                    "metadata": metadata or {},
                    "ready": await ensure_async(model.is_ready)()
                }
                self.model_usage_stats[model_id] = {
                    "inference_count": 0,
                    "training_count": 0,
                    "error_count": 0,
                    "last_used": None
                }
                
            self.logger.info(f"Modèle {model_id} enregistré avec succès", 
                           extra={"model_type": model_type.value})
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement du modèle {model_id}: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[IMLModel]:
        """Récupérer un modèle par ID"""
        with self.lock:
            model = self.models.get(model_id)
            if model:
                self._update_usage_stats(model_id, "access")
            return model
    
    async def predict(self, model_id: str, input_data: Any) -> Any:
        """Prédiction via modèle spécifique"""
        model = await self.get_model(model_id)
        if not model:
            raise ModelNotFoundError(f"Modèle {model_id} non trouvé")
        
        try:
            with self.lock:
                self._update_usage_stats(model_id, "inference_start")
            
            result = await asyncio.wait_for(
                model.predict(input_data),
                timeout=self.config.inference_timeout
            )
            
            with self.lock:
                self._update_usage_stats(model_id, "inference_success")
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout lors de l'inférence pour {model_id}")
            with self.lock:
                self._update_usage_stats(model_id, "error")
            raise InferenceError(f"Timeout d'inférence pour {model_id}")
            
        except Exception as e:
            self.logger.error(f"Erreur d'inférence pour {model_id}: {e}")
            with self.lock:
                self._update_usage_stats(model_id, "error")
            raise InferenceError(f"Erreur d'inférence: {e}")
    
    def _update_usage_stats(self, model_id: str, event_type: str):
        """Mettre à jour les statistiques d'usage"""
        if model_id not in self.model_usage_stats:
            return
            
        stats = self.model_usage_stats[model_id]
        now = datetime.utcnow().isoformat()
        
        if event_type == "inference_start":
            stats["inference_count"] += 1
            stats["last_used"] = now
        elif event_type == "training":
            stats["training_count"] += 1
        elif event_type == "error":
            stats["error_count"] += 1
        elif event_type == "access":
            stats["last_used"] = now
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Statistiques globales des modèles"""
        with self.lock:
            return {
                "total_models": len(self.models),
                "model_types": list(set(
                    meta.get("type") for meta in self.model_metadata.values()
                )),
                "ready_models": sum(
                    1 for meta in self.model_metadata.values() 
                    if meta.get("ready", False)
                ),
                "usage_stats": dict(self.model_usage_stats)
            }


class MLPipelineOrchestrator:
    """Orchestrateur de pipelines ML complexes"""
    
    def __init__(self, config: MLEngineConfig):
        self.config = config
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        self.pipeline_history: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.MLPipelineOrchestrator")
        
    async def execute_pipeline(
        self,
        pipeline_id: str,
        steps: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Exécuter un pipeline ML"""
        pipeline_context = {
            "pipeline_id": pipeline_id,
            "started_at": datetime.utcnow(),
            "status": "running",
            "steps_completed": 0,
            "total_steps": len(steps),
            "context": context or {},
            "results": {}
        }
        
        try:
            with self.lock:
                self.active_pipelines[pipeline_id] = pipeline_context
            
            self.logger.info(f"Démarrage du pipeline {pipeline_id} avec {len(steps)} étapes")
            
            # Exécution séquentielle des étapes
            for step_idx, step in enumerate(steps):
                step_result = await self._execute_pipeline_step(
                    pipeline_id, step_idx, step, pipeline_context
                )
                
                pipeline_context["results"][f"step_{step_idx}"] = step_result
                pipeline_context["steps_completed"] = step_idx + 1
                
                # Point de contrôle
                if step_idx % 5 == 0:  # Checkpoint tous les 5 steps
                    await self._checkpoint_pipeline(pipeline_id, pipeline_context)
            
            # Finalisation
            pipeline_context["status"] = "completed"
            pipeline_context["completed_at"] = datetime.utcnow()
            
            self.logger.info(f"Pipeline {pipeline_id} terminé avec succès")
            return pipeline_context
            
        except Exception as e:
            pipeline_context["status"] = "failed"
            pipeline_context["error"] = str(e)
            pipeline_context["failed_at"] = datetime.utcnow()
            
            self.logger.error(f"Échec du pipeline {pipeline_id}: {e}")
            raise PipelineError(f"Échec du pipeline: {e}")
            
        finally:
            with self.lock:
                if pipeline_id in self.active_pipelines:
                    self.pipeline_history.append(
                        dict(self.active_pipelines[pipeline_id])
                    )
                    del self.active_pipelines[pipeline_id]
    
    async def _execute_pipeline_step(
        self,
        pipeline_id: str,
        step_idx: int,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Exécuter une étape de pipeline"""
        step_type = step.get("type")
        step_config = step.get("config", {})
        
        self.logger.debug(f"Exécution étape {step_idx} ({step_type}) pour {pipeline_id}")
        
        # Simulation d'exécution - à implémenter selon les types d'étapes
        await asyncio.sleep(0.1)  # Simulation de traitement
        
        return {
            "step_type": step_type,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _checkpoint_pipeline(self, pipeline_id: str, context: Dict[str, Any]):
        """Créer un point de contrôle du pipeline"""
        self.logger.debug(f"Checkpoint pipeline {pipeline_id} - étape {context['steps_completed']}")
        # Implémentation du checkpoint (sauvegarde état, etc.)
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Statut d'un pipeline"""
        with self.lock:
            return self.active_pipelines.get(pipeline_id)
    
    def get_active_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Pipelines actifs"""
        with self.lock:
            return dict(self.active_pipelines)


class MLAnalyticsEngine:
    """Moteur principal ML Analytics Enterprise"""
    
    def __init__(self, config: Optional[MLEngineConfig] = None):
        self.config = config or MLEngineConfig()
        self.state = MLEngineState.INITIALIZING
        self.model_manager = MLModelManager(self.config)
        self.pipeline_orchestrator = MLPipelineOrchestrator(self.config)
        self.metrics_collector = MLMetricsCollector() if self.config.monitoring_enabled else None
        self.performance_tracker = PerformanceTracker()
        
        # Threading et async
        self.lock = threading.RLock()
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = threading.Event()
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.MLAnalyticsEngine")
        
        # Health check
        self.last_health_check = datetime.utcnow()
        self.health_status = {"status": "unknown", "checks": {}}
    
    async def initialize(self) -> bool:
        """Initialisation du moteur ML Analytics"""
        try:
            self.logger.info("Initialisation du moteur ML Analytics...")
            
            # Initialisation des composants
            await self._initialize_components()
            
            # Démarrage des tâches de fond
            if self.config.monitoring_enabled:
                self._start_background_tasks()
            
            self.state = MLEngineState.READY
            self.logger.info("Moteur ML Analytics initialisé avec succès")
            return True
            
        except Exception as e:
            self.state = MLEngineState.ERROR
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            return False
    
    async def _initialize_components(self):
        """Initialisation des composants internes"""
        # Initialisation des métriques
        if self.metrics_collector:
            await self.metrics_collector.initialize()
        
        # Initialisation du tracker de performance
        await self.performance_tracker.initialize()
        
        # Chargement des modèles par défaut
        await self._load_default_models()
    
    async def _load_default_models(self):
        """Chargement des modèles par défaut"""
        # À implémenter : chargement des modèles depuis la configuration
        self.logger.info("Chargement des modèles par défaut...")
    
    def _start_background_tasks(self):
        """Démarrage des tâches de fond"""
        try:
            # Tâche de collecte de métriques
            if self.metrics_collector:
                task = asyncio.create_task(self._metrics_collection_loop())
                self.background_tasks.append(task)
            
            # Tâche de health check
            task = asyncio.create_task(self._health_check_loop())
            self.background_tasks.append(task)
            
            # Tâche de nettoyage
            task = asyncio.create_task(self._cleanup_loop())
            self.background_tasks.append(task)
            
            self.logger.info(f"Démarrage de {len(self.background_tasks)} tâches de fond")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du démarrage des tâches de fond: {e}")
    
    async def _metrics_collection_loop(self):
        """Boucle de collecte de métriques"""
        while not self.shutdown_event.is_set():
            try:
                if self.metrics_collector:
                    await self.metrics_collector.collect_metrics()
                await asyncio.sleep(self.config.metrics_collection_interval)
            except Exception as e:
                self.logger.error(f"Erreur dans la collecte de métriques: {e}")
                await asyncio.sleep(5.0)
    
    async def _health_check_loop(self):
        """Boucle de vérification de santé"""
        while not self.shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Erreur dans le health check: {e}")
                await asyncio.sleep(10.0)
    
    async def _cleanup_loop(self):
        """Boucle de nettoyage périodique"""
        while not self.shutdown_event.is_set():
            try:
                await self._perform_cleanup()
                await asyncio.sleep(self.config.cleanup_interval)
            except Exception as e:
                self.logger.error(f"Erreur dans le nettoyage: {e}")
                await asyncio.sleep(30.0)
    
    async def _perform_health_check(self):
        """Vérification de santé du système"""
        checks = {}
        
        # Vérification de l'état du gestionnaire de modèles
        try:
            model_stats = self.model_manager.get_model_stats()
            checks["model_manager"] = {
                "status": "healthy",
                "models_count": model_stats["total_models"],
                "ready_models": model_stats["ready_models"]
            }
        except Exception as e:
            checks["model_manager"] = {"status": "unhealthy", "error": str(e)}
        
        # Vérification des pipelines actifs
        try:
            active_pipelines = self.pipeline_orchestrator.get_active_pipelines()
            checks["pipeline_orchestrator"] = {
                "status": "healthy",
                "active_pipelines": len(active_pipelines)
            }
        except Exception as e:
            checks["pipeline_orchestrator"] = {"status": "unhealthy", "error": str(e)}
        
        # Vérification de la mémoire et des ressources
        checks["resources"] = await self._check_resources()
        
        # Mise à jour du statut global
        all_healthy = all(
            check.get("status") == "healthy" 
            for check in checks.values()
        )
        
        self.health_status = {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
        
        self.last_health_check = datetime.utcnow()
    
    async def _check_resources(self) -> Dict[str, Any]:
        """Vérification des ressources système"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "status": "healthy",
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "available_memory_gb": memory.available / (1024**3)
            }
        except ImportError:
            return {"status": "unknown", "error": "psutil not available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _perform_cleanup(self):
        """Nettoyage périodique"""
        self.logger.debug("Exécution du nettoyage périodique...")
        
        # Nettoyage de l'historique des pipelines (garder seulement les 1000 derniers)
        with self.pipeline_orchestrator.lock:
            if len(self.pipeline_orchestrator.pipeline_history) > 1000:
                self.pipeline_orchestrator.pipeline_history = \
                    self.pipeline_orchestrator.pipeline_history[-1000:]
        
        # Nettoyage des métriques anciennes
        if self.metrics_collector:
            await self.metrics_collector.cleanup_old_metrics()
    
    # API publique
    async def predict(self, model_id: str, input_data: Any) -> Any:
        """Prédiction via modèle"""
        if self.state != MLEngineState.READY:
            raise MLAnalyticsError(f"Moteur non prêt (état: {self.state.name})")
        
        return await self.model_manager.predict(model_id, input_data)
    
    async def execute_pipeline(
        self,
        pipeline_id: str,
        steps: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Exécution de pipeline ML"""
        if self.state != MLEngineState.READY:
            raise MLAnalyticsError(f"Moteur non prêt (état: {self.state.name})")
        
        return await self.pipeline_orchestrator.execute_pipeline(
            pipeline_id, steps, context
        )
    
    async def register_model(
        self,
        model_id: str,
        model: IMLModel,
        model_type: ModelType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Enregistrement d'un nouveau modèle"""
        return await self.model_manager.register_model(
            model_id, model, model_type, metadata
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Status global du moteur"""
        return {
            "state": self.state.name,
            "health": self.health_status,
            "model_stats": self.model_manager.get_model_stats(),
            "active_pipelines": len(self.pipeline_orchestrator.get_active_pipelines()),
            "background_tasks": len(self.background_tasks),
            "last_health_check": self.last_health_check.isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.last_health_check).total_seconds()
        }
    
    async def shutdown(self):
        """Arrêt propre du moteur"""
        self.logger.info("Début de l'arrêt du moteur ML Analytics...")
        self.state = MLEngineState.SHUTTING_DOWN
        
        # Signaler l'arrêt aux tâches de fond
        self.shutdown_event.set()
        
        # Attendre la fin des tâches de fond
        if self.background_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.background_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Timeout lors de l'arrêt des tâches de fond")
        
        self.logger.info("Moteur ML Analytics arrêté")


# Instance globale (singleton pattern)
_ml_engine_instance: Optional[MLAnalyticsEngine] = None
_engine_lock = threading.Lock()


def get_ml_engine(config: Optional[MLEngineConfig] = None) -> MLAnalyticsEngine:
    """Récupération de l'instance globale du moteur ML"""
    global _ml_engine_instance
    
    with _engine_lock:
        if _ml_engine_instance is None:
            _ml_engine_instance = MLAnalyticsEngine(config)
        return _ml_engine_instance


async def initialize_ml_engine(config: Optional[MLEngineConfig] = None) -> bool:
    """Initialisation globale du moteur ML"""
    engine = get_ml_engine(config)
    return await engine.initialize()


# Exports publics
__all__ = [
    'MLAnalyticsEngine',
    'MLEngineConfig', 
    'ModelType',
    'MLEngineState',
    'IMLModel',
    'get_ml_engine',
    'initialize_ml_engine'
]
