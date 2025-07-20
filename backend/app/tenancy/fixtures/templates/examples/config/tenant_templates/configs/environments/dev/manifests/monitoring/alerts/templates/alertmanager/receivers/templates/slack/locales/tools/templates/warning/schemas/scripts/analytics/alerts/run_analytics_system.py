#!/usr/bin/env python3
"""
Script principal de démarrage du système d'analytics d'alertes
Démarrage orchestré avec monitoring de santé et auto-recovery
"""

import asyncio
import signal
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import uvloop
import psutil

# Configuration du chemin
sys.path.append(str(Path(__file__).parent))

from config.analytics_config import get_analytics_config, EnvironmentType
from alert_analytics_engine import create_analytics_engine
from anomaly_detector import create_anomaly_detector
from correlation_analyzer import CorrelationAnalyzer
from processors.stream_processor import create_stream_processor

logger = logging.getLogger(__name__)

class AnalyticsSystemOrchestrator:
    """
    Orchestrateur principal du système d'analytics
    
    Responsabilités:
    - Démarrage coordonné des composants
    - Monitoring de santé des services
    - Auto-recovery en cas de panne
    - Arrêt gracieux du système
    - Métriques et monitoring
    """
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = get_analytics_config()
        if config_override:
            # Application des overrides de configuration
            for key, value in config_override.items():
                setattr(self.config, key, value)
        
        self.components = {}
        self.health_tasks = []
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Configuration logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configuration du système de logging"""
        log_level = getattr(logging, self.config.log_level.value)
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        if self.config.environment == EnvironmentType.DEVELOPMENT:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('analytics_system.log')
            ]
        )
        
        # Configuration spécifique pour les modules
        logging.getLogger('kafka').setLevel(logging.WARNING)
        logging.getLogger('asyncpg').setLevel(logging.INFO)
        
    async def initialize_components(self):
        """Initialisation de tous les composants"""
        logger.info("Initialisation des composants du système d'analytics...")
        
        try:
            # 1. Analytics Engine (cœur du système)
            logger.info("Initialisation Analytics Engine...")
            self.components['analytics_engine'] = await create_analytics_engine(
                self.config.export_for_ml_models()
            )
            
            # 2. Détecteur d'anomalies
            if self.config.ml_config.anomaly_detection_enabled:
                logger.info("Initialisation Détecteur d'anomalies...")
                self.components['anomaly_detector'] = await create_anomaly_detector(
                    self.config.ml_config.dict()
                )
            
            # 3. Analyseur de corrélation
            if self.config.ml_config.correlation_enabled:
                logger.info("Initialisation Analyseur de corrélation...")
                correlation_analyzer = CorrelationAnalyzer(self.config.ml_config.dict())
                await correlation_analyzer.initialize()
                self.components['correlation_analyzer'] = correlation_analyzer
            
            # 4. Processeur de streaming
            if self.config.streaming_config.streaming_enabled:
                logger.info("Initialisation Processeur de streaming...")
                self.components['stream_processor'] = await create_stream_processor(
                    self.config.streaming_config.dict()
                )
            
            logger.info("Tous les composants initialisés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des composants: {e}")
            raise
    
    async def start_system(self):
        """Démarrage du système complet"""
        if self.is_running:
            logger.warning("Le système est déjà en cours d'exécution")
            return
        
        try:
            logger.info("Démarrage du système d'analytics Spotify AI Agent...")
            
            # Vérifications préliminaires
            await self._perform_health_checks()
            
            # Démarrage des composants
            await self._start_components()
            
            # Démarrage du monitoring
            await self._start_health_monitoring()
            
            # Configuration des signaux
            self._setup_signal_handlers()
            
            self.is_running = True
            logger.info("Système d'analytics démarré avec succès")
            
            # Boucle principale
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Erreur lors du démarrage: {e}")
            await self.shutdown_system()
            raise
    
    async def _perform_health_checks(self):
        """Vérifications de santé préliminaires"""
        logger.info("Vérifications de santé du système...")
        
        # Vérification mémoire disponible
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            logger.warning(f"Mémoire faible: {memory.percent}% utilisée")
        
        # Vérification espace disque
        disk = psutil.disk_usage('/')
        if disk.percent > 85:
            logger.warning(f"Espace disque faible: {disk.percent}% utilisé")
        
        # Vérification connectivité base de données
        try:
            import asyncpg
            conn = await asyncpg.connect(self.config.database_url)
            await conn.execute('SELECT 1')
            await conn.close()
            logger.info("Connexion base de données: OK")
        except Exception as e:
            logger.error(f"Erreur connexion base de données: {e}")
            raise
        
        # Vérification Redis
        try:
            import aioredis
            redis = await aioredis.from_url(self.config.redis_url)
            await redis.ping()
            await redis.close()
            logger.info("Connexion Redis: OK")
        except Exception as e:
            logger.error(f"Erreur connexion Redis: {e}")
            raise
        
        logger.info("Toutes les vérifications de santé passées")
    
    async def _start_components(self):
        """Démarrage séquentiel des composants"""
        
        # Démarrage du processeur de streaming en premier
        if 'stream_processor' in self.components:
            logger.info("Démarrage du processeur de streaming...")
            stream_task = asyncio.create_task(
                self.components['stream_processor'].start_processing()
            )
            self.health_tasks.append(stream_task)
        
        # Les autres composants sont déjà initialisés et prêts
        logger.info("Tous les composants démarrés")
    
    async def _start_health_monitoring(self):
        """Démarrage du monitoring de santé"""
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.health_tasks.append(health_task)
        
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.health_tasks.append(metrics_task)
        
        logger.info("Monitoring de santé démarré")
    
    async def _health_monitoring_loop(self):
        """Boucle de monitoring de santé"""
        while self.is_running:
            try:
                # Vérification de l'état des composants
                component_health = await self._check_components_health()
                
                # Auto-recovery si nécessaire
                if not all(component_health.values()):
                    await self._perform_auto_recovery(component_health)
                
                # Vérification des ressources système
                await self._check_system_resources()
                
                # Attente avant prochaine vérification
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Erreur dans le monitoring de santé: {e}")
                await asyncio.sleep(10)
    
    async def _check_components_health(self) -> Dict[str, bool]:
        """Vérification de l'état de santé des composants"""
        health_status = {}
        
        for component_name, component in self.components.items():
            try:
                if hasattr(component, 'get_health_status'):
                    health = await component.get_health_status()
                    health_status[component_name] = health.get('healthy', True)
                else:
                    # Vérification basique si pas de méthode health
                    health_status[component_name] = True
                    
            except Exception as e:
                logger.error(f"Erreur vérification santé {component_name}: {e}")
                health_status[component_name] = False
        
        return health_status
    
    async def _perform_auto_recovery(self, health_status: Dict[str, bool]):
        """Auto-recovery des composants défaillants"""
        for component_name, is_healthy in health_status.items():
            if not is_healthy:
                logger.warning(f"Tentative de recovery du composant: {component_name}")
                
                try:
                    component = self.components[component_name]
                    
                    # Tentative de redémarrage
                    if hasattr(component, 'restart'):
                        await component.restart()
                    elif hasattr(component, 'close') and hasattr(component, 'initialize'):
                        await component.close()
                        await component.initialize()
                    
                    logger.info(f"Recovery réussie pour {component_name}")
                    
                except Exception as e:
                    logger.error(f"Échec recovery {component_name}: {e}")
    
    async def _check_system_resources(self):
        """Vérification des ressources système"""
        # Mémoire
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            logger.critical(f"Mémoire critique: {memory.percent}%")
            # Trigger garbage collection agressif
            import gc
            gc.collect()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            logger.warning(f"CPU élevé: {cpu_percent}%")
        
        # Connexions réseau
        connections = psutil.net_connections()
        if len(connections) > 1000:
            logger.warning(f"Nombreuses connexions: {len(connections)}")
    
    async def _metrics_collection_loop(self):
        """Boucle de collecte de métriques"""
        while self.is_running:
            try:
                # Collecte métriques système
                system_metrics = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent,
                    'network_connections': len(psutil.net_connections())
                }
                
                # Collecte métriques composants
                component_metrics = {}
                for name, component in self.components.items():
                    if hasattr(component, 'get_metrics'):
                        try:
                            metrics = await component.get_metrics()
                            component_metrics[name] = metrics
                        except Exception as e:
                            logger.error(f"Erreur collecte métriques {name}: {e}")
                
                # Log des métriques (en production, envoyer à un système de métriques)
                if self.config.debug:
                    logger.debug(f"Métriques système: {system_metrics}")
                    logger.debug(f"Métriques composants: {component_metrics}")
                
                await asyncio.sleep(self.config.streaming_config.metrics_interval_seconds)
                
            except Exception as e:
                logger.error(f"Erreur collecte métriques: {e}")
                await asyncio.sleep(30)
    
    async def _main_loop(self):
        """Boucle principale du système"""
        try:
            logger.info("Système en fonctionnement - En attente d'arrêt...")
            await self.shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("Interruption clavier détectée")
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale: {e}")
        finally:
            await self.shutdown_system()
    
    def _setup_signal_handlers(self):
        """Configuration des gestionnaires de signaux"""
        def signal_handler(signum, frame):
            logger.info(f"Signal {signum} reçu, arrêt du système...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)
    
    async def shutdown_system(self):
        """Arrêt gracieux du système"""
        if not self.is_running:
            return
        
        logger.info("Arrêt du système d'analytics en cours...")
        self.is_running = False
        
        try:
            # Arrêt des tâches de monitoring
            for task in self.health_tasks:
                if not task.done():
                    task.cancel()
            
            if self.health_tasks:
                await asyncio.gather(*self.health_tasks, return_exceptions=True)
            
            # Arrêt des composants
            for component_name, component in self.components.items():
                try:
                    logger.info(f"Arrêt du composant: {component_name}")
                    if hasattr(component, 'stop_processing'):
                        await component.stop_processing()
                    elif hasattr(component, 'close'):
                        await component.close()
                except Exception as e:
                    logger.error(f"Erreur arrêt {component_name}: {e}")
            
            logger.info("Système d'analytics arrêté proprement")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {e}")

async def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Spotify AI Agent - Système d\'Analytics d\'Alertes')
    parser.add_argument('--config', help='Fichier de configuration', default=None)
    parser.add_argument('--environment', help='Environnement (dev/staging/prod)', default='development')
    parser.add_argument('--debug', action='store_true', help='Mode debug')
    parser.add_argument('--profile', action='store_true', help='Profiling des performances')
    
    args = parser.parse_args()
    
    # Configuration overrides
    config_override = {}
    if args.environment:
        env_map = {
            'dev': EnvironmentType.DEVELOPMENT,
            'development': EnvironmentType.DEVELOPMENT,
            'staging': EnvironmentType.STAGING,
            'prod': EnvironmentType.PRODUCTION,
            'production': EnvironmentType.PRODUCTION
        }
        if args.environment.lower() in env_map:
            config_override['environment'] = env_map[args.environment.lower()]
    
    if args.debug:
        config_override['debug'] = True
        config_override['log_level'] = 'DEBUG'
    
    # Optimisation des performances avec uvloop
    if sys.platform != 'win32':
        uvloop.install()
    
    # Profiling si demandé
    if args.profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
    
    try:
        # Création et démarrage de l'orchestrateur
        orchestrator = AnalyticsSystemOrchestrator(config_override)
        await orchestrator.initialize_components()
        await orchestrator.start_system()
        
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        sys.exit(1)
    finally:
        if args.profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 fonctions

if __name__ == '__main__':
    asyncio.run(main())
