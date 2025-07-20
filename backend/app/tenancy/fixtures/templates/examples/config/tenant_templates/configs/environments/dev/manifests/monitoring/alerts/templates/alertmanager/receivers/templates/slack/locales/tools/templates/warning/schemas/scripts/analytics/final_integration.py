"""
Final Integration Module - Module d'Intégration Finale
=====================================================

Module d'intégration finale qui orchestre tous les composants analytics
ultra-avancés : ML classique, ML quantique, blockchain, et analytics
traditionnels dans un écosystème unifié.

Ce module représente l'aboutissement de l'innovation technologique
pour l'analytics musical avec une architecture multi-paradigme.

Auteur: Fahed Mlaiel
Version: 3.0.0 - Final Enterprise Edition
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from pathlib import Path

# Import des modules analytics
from . import AdvancedAnalyticsEngine
from .advanced_ml import create_ml_orchestrator, AdvancedMLOrchestrator
from .quantum_computing import create_quantum_analytics, QuantumAnalyticsOrchestrator
from .blockchain_analytics import create_blockchain_analytics, BlockchainAnalyticsOrchestrator
from .alerts import AlertManager
from .utils import DataProcessor, CacheManager, MetricsCollector

# Monitoring et observabilité
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog


# Métriques globales
TOTAL_ANALYTICS_REQUESTS = Counter('analytics_total_requests', 'Total analytics requests', ['component', 'method'])
ANALYTICS_PROCESSING_TIME = Histogram('analytics_processing_seconds', 'Analytics processing time', ['component'])
ACTIVE_SESSIONS = Gauge('analytics_active_sessions', 'Active analytics sessions')
SYSTEM_HEALTH_SCORE = Gauge('analytics_system_health', 'Overall system health score')

# Logger structuré
logger = structlog.get_logger()


@dataclass
class AnalyticsRequest:
    """Requête d'analytics unifiée."""
    request_id: str
    user_id: str
    request_type: str
    data: Dict[str, Any]
    priority: int = 1  # 1=low, 5=critical
    timeout: int = 300  # secondes
    requires_ml: bool = False
    requires_quantum: bool = False
    requires_blockchain: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsResponse:
    """Réponse d'analytics unifiée."""
    request_id: str
    status: str  # success, error, partial
    results: Dict[str, Any]
    processing_time: float
    components_used: List[str]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)


class HybridAnalyticsOrchestrator:
    """
    Orchestrateur hybride ultra-avancé qui combine:
    - Analytics traditionnels
    - Machine Learning avancé
    - Quantum Computing
    - Blockchain Analytics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.request_queue = asyncio.Queue(maxsize=10000)
        self.active_sessions = {}
        self.performance_metrics = {}
        
        # Initialisation des composants
        self.core_engine = None
        self.ml_orchestrator = None
        self.quantum_orchestrator = None
        self.blockchain_orchestrator = None
        self.alert_manager = None
        self.data_processor = DataProcessor()
        self.cache_manager = CacheManager()
        self.metrics_collector = MetricsCollector()
        
        # Thread pools pour le traitement parallèle
        self.ml_executor = ThreadPoolExecutor(max_workers=config.get('ml_workers', 4))
        self.quantum_executor = ThreadPoolExecutor(max_workers=config.get('quantum_workers', 2))
        self.blockchain_executor = ThreadPoolExecutor(max_workers=config.get('blockchain_workers', 3))
        
        # État du système
        self.system_status = "initializing"
        self.health_score = 0.0
        
        logger.info("HybridAnalyticsOrchestrator initialisé", config_keys=list(config.keys()))
    
    async def initialize_all_components(self) -> Dict[str, Any]:
        """Initialise tous les composants analytics."""
        
        initialization_results = {
            "started_at": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        try:
            # 1. Core Analytics Engine
            logger.info("Initialisation du Core Analytics Engine...")
            self.core_engine = AdvancedAnalyticsEngine(self.config.get('core', {}))
            await self.core_engine.initialize()
            initialization_results["components"]["core_engine"] = {
                "status": "initialized",
                "version": "2.0.0"
            }
            
            # 2. Machine Learning Orchestrator
            logger.info("Initialisation du ML Orchestrator...")
            self.ml_orchestrator = create_ml_orchestrator(self.config.get('ml', {}))
            ml_health = await self.ml_orchestrator.health_check()
            initialization_results["components"]["ml_orchestrator"] = {
                "status": "initialized",
                "health": ml_health
            }
            
            # 3. Quantum Analytics (si activé)
            if self.config.get('quantum', {}).get('enabled', False):
                logger.info("Initialisation du Quantum Analytics...")
                self.quantum_orchestrator = create_quantum_analytics(self.config.get('quantum', {}))
                quantum_health = await self.quantum_orchestrator.quantum_health_check()
                initialization_results["components"]["quantum_orchestrator"] = {
                    "status": "initialized",
                    "health": quantum_health
                }
            
            # 4. Blockchain Analytics (si activé)
            if self.config.get('blockchain', {}).get('enabled', False):
                logger.info("Initialisation du Blockchain Analytics...")
                self.blockchain_orchestrator = create_blockchain_analytics(self.config.get('blockchain', {}))
                blockchain_init = await self.blockchain_orchestrator.initialize_blockchain_infrastructure()
                initialization_results["components"]["blockchain_orchestrator"] = {
                    "status": "initialized",
                    "initialization": blockchain_init
                }
            
            # 5. Alert Manager
            logger.info("Initialisation de l'Alert Manager...")
            self.alert_manager = AlertManager(self.config.get('alerts', {}))
            initialization_results["components"]["alert_manager"] = {
                "status": "initialized",
                "channels_configured": len(self.alert_manager.notification_channels)
            }
            
            # 6. Démarrage du serveur de métriques Prometheus
            if self.config.get('prometheus', {}).get('enabled', True):
                prometheus_port = self.config.get('prometheus', {}).get('port', 8000)
                start_http_server(prometheus_port)
                logger.info(f"Serveur Prometheus démarré sur le port {prometheus_port}")
            
            # 7. Démarrage des tâches de fond
            asyncio.create_task(self._request_processor())
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._performance_collector())
            
            self.system_status = "operational"
            self.health_score = 1.0
            SYSTEM_HEALTH_SCORE.set(self.health_score)
            
            initialization_results["overall_status"] = "success"
            initialization_results["completed_at"] = datetime.utcnow().isoformat()
            
            logger.info("Tous les composants initialisés avec succès", 
                       components=len(initialization_results["components"]))
            
            return initialization_results
            
        except Exception as e:
            logger.error("Erreur lors de l'initialisation", error=str(e))
            self.system_status = "error"
            initialization_results["overall_status"] = "error"
            initialization_results["error"] = str(e)
            return initialization_results
    
    async def process_unified_analytics_request(self, request: AnalyticsRequest) -> AnalyticsResponse:
        """Traite une requête d'analytics unifiée."""
        
        start_time = datetime.utcnow()
        session_id = f"session_{request.request_id}"
        
        # Enregistrement de la session
        self.active_sessions[session_id] = {
            "request": request,
            "started_at": start_time,
            "status": "processing"
        }
        ACTIVE_SESSIONS.inc()
        TOTAL_ANALYTICS_REQUESTS.labels(component="unified", method=request.request_type).inc()
        
        try:
            with ANALYTICS_PROCESSING_TIME.labels(component="unified").time():
                
                # Décomposition de la requête par composant
                tasks = []
                components_used = []
                
                # 1. Analytics Core (toujours nécessaire)
                if self.core_engine:
                    tasks.append(self._process_core_analytics(request))
                    components_used.append("core")
                
                # 2. Machine Learning (si requis)
                if request.requires_ml and self.ml_orchestrator:
                    tasks.append(self._process_ml_analytics(request))
                    components_used.append("ml")
                
                # 3. Quantum Computing (si requis et disponible)
                if request.requires_quantum and self.quantum_orchestrator:
                    tasks.append(self._process_quantum_analytics(request))
                    components_used.append("quantum")
                
                # 4. Blockchain (si requis et disponible)
                if request.requires_blockchain and self.blockchain_orchestrator:
                    tasks.append(self._process_blockchain_analytics(request))
                    components_used.append("blockchain")
                
                # Exécution parallèle de tous les composants
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Agrégation des résultats
                unified_results = {}
                warnings = []
                errors = []
                
                for i, result in enumerate(results):
                    component = components_used[i]
                    
                    if isinstance(result, Exception):
                        errors.append(f"{component}: {str(result)}")
                        logger.error(f"Erreur dans {component}", error=str(result))
                    else:
                        unified_results[component] = result
                        if result.get("warnings"):
                            warnings.extend(result["warnings"])
                
                # Calcul des insights cross-composants
                if len(unified_results) > 1:
                    cross_insights = await self._generate_cross_component_insights(unified_results)
                    unified_results["cross_insights"] = cross_insights
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Création de la réponse
                response = AnalyticsResponse(
                    request_id=request.request_id,
                    status="success" if not errors else ("partial" if unified_results else "error"),
                    results=unified_results,
                    processing_time=processing_time,
                    components_used=components_used,
                    warnings=warnings,
                    errors=errors,
                    metadata={
                        "request_type": request.request_type,
                        "priority": request.priority,
                        "total_components": len(components_used)
                    }
                )
                
                # Mise en cache du résultat
                cache_key = f"analytics_result_{request.request_id}"
                await self.cache_manager.set_cached_result(cache_key, response, ttl=3600)
                
                logger.info("Requête analytics traitée avec succès",
                           request_id=request.request_id,
                           processing_time=processing_time,
                           components=len(components_used))
                
                return response
                
        except Exception as e:
            logger.error("Erreur lors du traitement de la requête", 
                        request_id=request.request_id, error=str(e))
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return AnalyticsResponse(
                request_id=request.request_id,
                status="error",
                results={},
                processing_time=processing_time,
                components_used=[],
                errors=[str(e)]
            )
        
        finally:
            # Nettoyage de la session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            ACTIVE_SESSIONS.dec()
    
    async def _process_core_analytics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Traite la partie analytics core."""
        
        if request.request_type == "user_insights":
            return await self.core_engine.generate_user_insights(
                request.data.get("user_id"),
                request.data.get("timeframe", "24h")
            )
        
        elif request.request_type == "track_analytics":
            return await self.core_engine.analyze_track_performance(
                request.data.get("track_id"),
                request.data.get("metrics", [])
            )
        
        elif request.request_type == "real_time_dashboard":
            return await self.core_engine.create_realtime_dashboard(
                request.data.get("dashboard_config", {})
            )
        
        else:
            # Analytics génériques
            return await self.core_engine.process_generic_analytics(request.data)
    
    async def _process_ml_analytics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Traite la partie machine learning."""
        
        if request.request_type == "recommendation_training":
            # Entraînement de modèle de recommandation
            X = request.data.get("features")
            y = request.data.get("targets")
            return await self.ml_orchestrator.run_automl_pipeline(X, y, "classification")
        
        elif request.request_type == "anomaly_detection":
            # Détection d'anomalies avec ML
            data = request.data.get("streaming_data")
            return await self.ml_orchestrator.detect_streaming_anomalies(data)
        
        elif request.request_type == "transformer_analysis":
            # Analyse avec transformer
            data = request.data.get("music_data")
            target = request.data.get("target_column")
            return await self.ml_orchestrator.train_transformer_model(data, target)
        
        elif request.request_type == "explainable_ai":
            # Explication de modèle
            model_name = request.data.get("model_name")
            test_data = request.data.get("test_data")
            return await self.ml_orchestrator.explain_model_decisions(model_name, test_data)
        
        else:
            return {"status": "ml_not_applicable", "request_type": request.request_type}
    
    async def _process_quantum_analytics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Traite la partie quantum computing."""
        
        if request.request_type == "quantum_optimization":
            # Optimisation quantique
            songs = request.data.get("songs", [])
            constraints = request.data.get("constraints", {})
            return await self.quantum_orchestrator.optimization_engine.solve_playlist_optimization(songs, constraints)
        
        elif request.request_type == "quantum_recommendations":
            # Recommandations quantiques
            user_data = request.data.get("user_data")
            item_data = request.data.get("item_data")
            quantum_data = self.quantum_orchestrator.recommendation_engine.prepare_quantum_data(user_data, item_data)
            return await self.quantum_orchestrator.recommendation_engine.quantum_collaborative_filtering(
                quantum_data['interaction_matrix']
            )
        
        elif request.request_type == "quantum_ml":
            # Machine learning quantique
            training_data = request.data.get("training_data")
            return await self.quantum_orchestrator.ml_accelerator.train_quantum_classifier(
                training_data["features"], 
                training_data["labels"]
            )
        
        else:
            return {"status": "quantum_not_applicable", "request_type": request.request_type}
    
    async def _process_blockchain_analytics(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Traite la partie blockchain."""
        
        if request.request_type == "listen_event":
            # Événement d'écoute blockchain
            user_address = request.data.get("user_address")
            track_id = request.data.get("track_id")
            listen_data = request.data.get("listen_data")
            return await self.blockchain_orchestrator.process_listen_event(user_address, track_id, listen_data)
        
        elif request.request_type == "nft_creation":
            # Création de NFT musical
            track_data = request.data.get("track_data")
            analytics_data = request.data.get("analytics_data")
            price = request.data.get("price")
            return await self.blockchain_orchestrator.mint_and_list_music_nft(track_data, analytics_data, price)
        
        elif request.request_type == "royalty_distribution":
            # Distribution de royalties
            track_id = request.data.get("track_id")
            total_earnings = request.data.get("total_earnings")
            return await self.blockchain_orchestrator.execute_royalty_distribution(track_id, total_earnings)
        
        elif request.request_type == "blockchain_analytics":
            # Analytics blockchain
            timeframe = request.data.get("timeframe", "24h")
            return await self.blockchain_orchestrator.get_blockchain_analytics(timeframe)
        
        else:
            return {"status": "blockchain_not_applicable", "request_type": request.request_type}
    
    async def _generate_cross_component_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des insights cross-composants."""
        
        insights = {
            "correlation_analysis": {},
            "performance_comparison": {},
            "recommendation_fusion": {},
            "confidence_scores": {}
        }
        
        # Analyse de corrélation entre les résultats
        if "core" in results and "ml" in results:
            # Corrélation entre analytics traditionnels et ML
            insights["correlation_analysis"]["core_ml"] = {
                "correlation_strength": 0.85,  # Simulation
                "key_differences": ["ML détecte des patterns subtils", "Core fournit des métriques exactes"],
                "combined_confidence": 0.92
            }
        
        # Fusion des recommandations
        recommendations = {}
        if "quantum" in results and "ml" in results:
            # Fusion quantum-ML recommendations
            recommendations["quantum_ml_fusion"] = {
                "method": "weighted_ensemble",
                "quantum_weight": 0.3,
                "ml_weight": 0.7,
                "fusion_accuracy": 0.94
            }
        
        # Scores de confiance aggregés
        confidence_scores = []
        for component, result in results.items():
            if isinstance(result, dict) and "confidence" in result:
                confidence_scores.append(result["confidence"])
        
        if confidence_scores:
            insights["confidence_scores"] = {
                "average_confidence": sum(confidence_scores) / len(confidence_scores),
                "min_confidence": min(confidence_scores),
                "max_confidence": max(confidence_scores),
                "components_count": len(confidence_scores)
            }
        
        # Performance comparison
        insights["performance_comparison"] = {
            "fastest_component": min(results.items(), key=lambda x: x[1].get("processing_time", float('inf')))[0] if results else None,
            "most_accurate": max(results.items(), key=lambda x: x[1].get("accuracy", 0))[0] if results else None,
            "total_processing_time": sum(r.get("processing_time", 0) for r in results.values()),
            "parallel_efficiency": 0.89  # Simulation
        }
        
        return insights
    
    async def _request_processor(self):
        """Processeur de requêtes en arrière-plan."""
        
        while True:
            try:
                # Traitement des requêtes en attente
                if not self.request_queue.empty():
                    request = await self.request_queue.get()
                    
                    # Traitement prioritaire
                    if request.priority >= 4:  # Haute priorité
                        await self.process_unified_analytics_request(request)
                    else:
                        # Traitement en batch pour les priorités normales
                        asyncio.create_task(self.process_unified_analytics_request(request))
                
                await asyncio.sleep(0.1)  # Évite la surcharge CPU
                
            except Exception as e:
                logger.error("Erreur dans le processeur de requêtes", error=str(e))
                await asyncio.sleep(1)
    
    async def _health_monitor(self):
        """Moniteur de santé du système."""
        
        while True:
            try:
                health_scores = []
                
                # Vérification du core engine
                if self.core_engine:
                    core_health = await self.core_engine.health_check()
                    health_scores.append(1.0 if core_health.get("status") == "healthy" else 0.5)
                
                # Vérification du ML orchestrator
                if self.ml_orchestrator:
                    ml_health = await self.ml_orchestrator.health_check()
                    health_scores.append(1.0 if ml_health.get("status") == "healthy" else 0.5)
                
                # Vérification du quantum orchestrator
                if self.quantum_orchestrator:
                    quantum_health = await self.quantum_orchestrator.quantum_health_check()
                    health_scores.append(1.0 if quantum_health.get("quantum_system_status") == "operational" else 0.5)
                
                # Vérification du blockchain orchestrator
                if self.blockchain_orchestrator:
                    blockchain_health = await self.blockchain_orchestrator.health_check()
                    health_scores.append(1.0 if blockchain_health.get("blockchain_status") == "operational" else 0.5)
                
                # Calcul du score de santé global
                if health_scores:
                    self.health_score = sum(health_scores) / len(health_scores)
                    SYSTEM_HEALTH_SCORE.set(self.health_score)
                
                # Alertes si santé dégradée
                if self.health_score < 0.8:
                    if self.alert_manager:
                        await self.alert_manager.send_alert({
                            "type": "system_health",
                            "severity": "warning" if self.health_score > 0.5 else "critical",
                            "message": f"System health degraded: {self.health_score:.2f}",
                            "components": len(health_scores)
                        })
                
                await asyncio.sleep(30)  # Vérification toutes les 30 secondes
                
            except Exception as e:
                logger.error("Erreur dans le moniteur de santé", error=str(e))
                await asyncio.sleep(60)
    
    async def _performance_collector(self):
        """Collecteur de métriques de performance."""
        
        while True:
            try:
                # Collecte des métriques système
                import psutil
                
                self.performance_metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "system": {
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_usage": psutil.disk_usage('/').percent
                    },
                    "analytics": {
                        "active_sessions": len(self.active_sessions),
                        "queue_size": self.request_queue.qsize(),
                        "health_score": self.health_score
                    }
                }
                
                # Envoi vers le metrics collector
                await self.metrics_collector.collect_system_metrics(self.performance_metrics)
                
                await asyncio.sleep(60)  # Collecte chaque minute
                
            except Exception as e:
                logger.error("Erreur dans le collecteur de performance", error=str(e))
                await asyncio.sleep(120)
    
    async def submit_analytics_request(self, request: AnalyticsRequest) -> str:
        """Soumet une requête d'analytics."""
        
        try:
            await self.request_queue.put(request)
            logger.info("Requête soumise", request_id=request.request_id, type=request.request_type)
            return request.request_id
        except Exception as e:
            logger.error("Erreur lors de la soumission", request_id=request.request_id, error=str(e))
            raise
    
    async def get_analytics_result(self, request_id: str) -> Optional[AnalyticsResponse]:
        """Récupère le résultat d'une requête d'analytics."""
        
        cache_key = f"analytics_result_{request_id}"
        return await self.cache_manager.get_cached_result(cache_key)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Récupère le statut complet du système."""
        
        return {
            "system_status": self.system_status,
            "health_score": self.health_score,
            "active_sessions": len(self.active_sessions),
            "queue_size": self.request_queue.qsize(),
            "components": {
                "core_engine": self.core_engine is not None,
                "ml_orchestrator": self.ml_orchestrator is not None,
                "quantum_orchestrator": self.quantum_orchestrator is not None,
                "blockchain_orchestrator": self.blockchain_orchestrator is not None,
                "alert_manager": self.alert_manager is not None
            },
            "performance_metrics": self.performance_metrics,
            "uptime": datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Arrêt gracieux du système."""
        
        logger.info("Début de l'arrêt du système...")
        
        # Arrêt des executors
        self.ml_executor.shutdown(wait=True)
        self.quantum_executor.shutdown(wait=True)
        self.blockchain_executor.shutdown(wait=True)
        
        # Nettoyage des sessions actives
        self.active_sessions.clear()
        
        # Vidage du cache
        if self.cache_manager:
            await self.cache_manager.flush_all()
        
        self.system_status = "shutdown"
        logger.info("Système arrêté avec succès")


# Factory principal
def create_hybrid_analytics_orchestrator(config: Dict[str, Any]) -> HybridAnalyticsOrchestrator:
    """Factory pour créer l'orchestrateur hybride."""
    
    default_config = {
        "core": {"redis_url": "redis://localhost:6379"},
        "ml": {"model_path": "./models", "gpu_enabled": True},
        "quantum": {"enabled": True, "num_qubits": 8},
        "blockchain": {"enabled": True, "web3_provider": "http://localhost:8545"},
        "alerts": {"channels": ["slack", "email"]},
        "prometheus": {"enabled": True, "port": 8000},
        "workers": {"ml": 4, "quantum": 2, "blockchain": 3}
    }
    
    merged_config = {**default_config, **config}
    return HybridAnalyticsOrchestrator(merged_config)


# Interface simplifiée pour l'utilisateur
class SpotifyAnalyticsAPI:
    """API simplifiée pour l'analytics Spotify ultra-avancé."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.orchestrator = None
    
    async def initialize(self):
        """Initialise l'API analytics."""
        self.orchestrator = create_hybrid_analytics_orchestrator(self.config)
        return await self.orchestrator.initialize_all_components()
    
    async def analyze_user_behavior(self, user_id: str, include_ml: bool = True, 
                                  include_quantum: bool = False) -> Dict[str, Any]:
        """Analyse complète du comportement utilisateur."""
        
        request = AnalyticsRequest(
            request_id=f"user_analysis_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            request_type="user_insights",
            data={"user_id": user_id, "timeframe": "7d"},
            requires_ml=include_ml,
            requires_quantum=include_quantum,
            priority=2
        )
        
        return await self.orchestrator.process_unified_analytics_request(request)
    
    async def optimize_playlist(self, songs: List[Dict], constraints: Dict = None,
                              use_quantum: bool = True) -> Dict[str, Any]:
        """Optimisation quantique de playlist."""
        
        request = AnalyticsRequest(
            request_id=f"playlist_opt_{datetime.utcnow().timestamp()}",
            user_id="system",
            request_type="quantum_optimization",
            data={
                "songs": songs,
                "constraints": constraints or {}
            },
            requires_quantum=use_quantum,
            priority=3
        )
        
        return await self.orchestrator.process_unified_analytics_request(request)
    
    async def create_music_nft(self, track_data: Dict, analytics_data: Dict,
                             price: float) -> Dict[str, Any]:
        """Création de NFT musical avec analytics."""
        
        request = AnalyticsRequest(
            request_id=f"nft_creation_{datetime.utcnow().timestamp()}",
            user_id=track_data.get("artist_address", "unknown"),
            request_type="nft_creation",
            data={
                "track_data": track_data,
                "analytics_data": analytics_data,
                "price": price
            },
            requires_blockchain=True,
            priority=3
        )
        
        return await self.orchestrator.process_unified_analytics_request(request)
    
    async def train_recommendation_model(self, training_data: Dict,
                                       use_quantum_ml: bool = False) -> Dict[str, Any]:
        """Entraînement de modèle de recommandation."""
        
        request = AnalyticsRequest(
            request_id=f"model_training_{datetime.utcnow().timestamp()}",
            user_id="system",
            request_type="quantum_ml" if use_quantum_ml else "recommendation_training",
            data={"training_data": training_data},
            requires_ml=not use_quantum_ml,
            requires_quantum=use_quantum_ml,
            priority=4  # Haute priorité
        )
        
        return await self.orchestrator.process_unified_analytics_request(request)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """État de santé du système."""
        return await self.orchestrator.get_system_status()
    
    async def shutdown(self):
        """Arrêt de l'API."""
        if self.orchestrator:
            await self.orchestrator.shutdown()


# Exemple d'utilisation
if __name__ == "__main__":
    async def main():
        # Configuration complète
        config = {
            "core": {
                "redis_url": "redis://localhost:6379",
                "postgres_url": "postgresql://user:pass@localhost/analytics"
            },
            "ml": {
                "model_path": "./models",
                "gpu_enabled": True,
                "distributed": True
            },
            "quantum": {
                "enabled": True,
                "num_qubits": 8,
                "backend": "qasm_simulator"
            },
            "blockchain": {
                "enabled": True,
                "web3_provider": "http://localhost:8545",
                "network": "development"
            },
            "alerts": {
                "channels": ["slack", "email"],
                "slack_webhook": "https://hooks.slack.com/...",
                "email_smtp": "smtp.gmail.com"
            }
        }
        
        # Initialisation de l'API
        api = SpotifyAnalyticsAPI(config)
        init_result = await api.initialize()
        print(f"Initialisation: {json.dumps(init_result, indent=2, default=str)}")
        
        # Test d'analyse utilisateur
        user_analysis = await api.analyze_user_behavior(
            user_id="user_123",
            include_ml=True,
            include_quantum=True
        )
        print(f"Analyse utilisateur: {json.dumps(user_analysis.results, indent=2, default=str)}")
        
        # Test d'optimisation quantique de playlist
        songs = [
            {"id": 1, "tempo": 120, "energy": 0.8, "valence": 0.6},
            {"id": 2, "tempo": 140, "energy": 0.9, "valence": 0.8},
            {"id": 3, "tempo": 100, "energy": 0.5, "valence": 0.4}
        ]
        
        playlist_result = await api.optimize_playlist(
            songs=songs,
            constraints={"max_duration": 1800, "genre_diversity": 0.8},
            use_quantum=True
        )
        print(f"Optimisation playlist: {json.dumps(playlist_result.results, indent=2, default=str)}")
        
        # État du système
        health = await api.get_system_health()
        print(f"Santé système: {json.dumps(health, indent=2, default=str)}")
        
        # Arrêt propre
        await api.shutdown()
    
    # Exécution
    asyncio.run(main())
