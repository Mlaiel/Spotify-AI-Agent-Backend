#!/usr/bin/env python3
"""
Analytics Main Script - Script Principal d'Analytics
===================================================

Ce script principal fournit une interface en ligne de commande
pour gérer et utiliser le système d'analytics Spotify AI Agent.

Fonctionnalités:
- Démarrage/arrêt du système analytics
- Ingestion de données en temps réel
- Génération de rapports
- Entraînement de modèles ML
- Gestion des alertes
- Monitoring du système

Usage:
    python analytics_main.py [command] [options]

Commands:
    start           - Démarre le système analytics
    stop            - Arrête le système analytics
    ingest          - Ingestion de données
    train           - Entraîne les modèles ML
    predict         - Fait des prédictions
    report          - Génère des rapports
    alert           - Gère les alertes
    status          - Affiche le statut du système
"""

import asyncio
import sys
import argparse
import json
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Analytics modules
from config import AnalyticsConfig, get_config, create_development_config, create_production_config
from core import AnalyticsEngine, MetricsCollector, AlertManager
from storage import StorageManager
from ml import ModelManager, prepare_metric_features
from models import Metric, Event, Alert, create_metric, create_event
from utils import Logger, Timer, Formatter
from processors import create_standard_pipeline, create_batch_pipeline


class AnalyticsApp:
    """Application principale d'analytics."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = Logger("AnalyticsApp")
        self.engine: Optional[AnalyticsEngine] = None
        self.storage_manager: Optional[StorageManager] = None
        self.model_manager: Optional[ModelManager] = None
        self.is_running = False
        
        # Gestion des signaux
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Gestionnaire de signaux pour arrêt propre."""
        self.logger.info(f"Signal {signum} reçu, arrêt en cours...")
        if self.is_running:
            asyncio.create_task(self.stop())
    
    async def start(self):
        """Démarre l'application analytics."""
        if self.is_running:
            self.logger.warning("Application déjà en cours d'exécution")
            return
        
        try:
            self.logger.info("Démarrage de l'application analytics...")
            
            # Initialiser les composants
            self.engine = AnalyticsEngine(self.config)
            self.storage_manager = StorageManager(self.config)
            self.model_manager = ModelManager(self.config)
            
            # Démarrer les services
            await self.storage_manager.connect_all()
            await self.model_manager.load_all_models()
            await self.engine.start()
            
            self.is_running = True
            self.logger.info("Application analytics démarrée avec succès")
            
            # Afficher le statut
            await self.show_status()
            
        except Exception as e:
            self.logger.error(f"Erreur démarrage application: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Arrête l'application analytics."""
        if not self.is_running:
            return
        
        try:
            self.logger.info("Arrêt de l'application analytics...")
            
            # Arrêter les services
            if self.engine:
                await self.engine.stop()
            
            if self.model_manager:
                await self.model_manager.save_all_models()
            
            if self.storage_manager:
                await self.storage_manager.disconnect_all()
            
            self.is_running = False
            self.logger.info("Application analytics arrêtée")
            
        except Exception as e:
            self.logger.error(f"Erreur arrêt application: {e}")
    
    async def show_status(self):
        """Affiche le statut du système."""
        print("\n" + "="*60)
        print("           SPOTIFY AI ANALYTICS - STATUS")
        print("="*60)
        
        if not self.is_running:
            print("❌ Système arrêté")
            return
        
        print("✅ Système en cours d'exécution")
        print()
        
        # Statut du moteur
        if self.engine:
            engine_status = self.engine.get_status()
            print(f"🚀 Moteur Analytics:")
            print(f"   État: {engine_status['state']}")
            print(f"   Uptime: {Formatter.format_duration(engine_status['uptime_seconds'])}")
            print(f"   Alertes actives: {engine_status['active_alerts']}")
            print()
        
        # Statut du stockage
        if self.storage_manager:
            health_status = await self.storage_manager.health_check_all()
            print("💾 Systèmes de stockage:")
            for storage_name, is_healthy in health_status.items():
                status = "✅" if is_healthy else "❌"
                print(f"   {storage_name}: {status}")
            print()
        
        # Statut des modèles ML
        if self.model_manager:
            model_stats = self.model_manager.get_all_model_stats()
            print("🧠 Modèles Machine Learning:")
            for model_name, stats in model_stats.items():
                status = "✅" if stats['is_trained'] else "⚠️"
                print(f"   {model_name}: {status} ({stats['feature_count']} features)")
            print()
        
        print("="*60)
    
    async def ingest_sample_data(self, count: int = 100):
        """Ingestion de données d'exemple."""
        if not self.is_running or not self.engine:
            raise RuntimeError("Système non démarré")
        
        self.logger.info(f"Ingestion de {count} métriques d'exemple...")
        
        metrics = []
        for i in range(count):
            metric = create_metric(
                name=f"test_metric_{i % 10}",
                value=float(i * 10 + (i % 7) * 5),
                tenant_id="demo_tenant",
                tags={
                    "source": f"server_{i % 3}",
                    "environment": "development",
                    "service": f"service_{i % 5}"
                }
            )
            metrics.append(metric)
        
        # Traitement en batch
        collector = self.engine.metrics_collector
        for metric in metrics:
            await collector.collect_metric(
                tenant_id=metric.tenant_id,
                metric_name=metric.name,
                value=metric.value,
                tags=metric.tags
            )
        
        self.logger.info(f"Ingestion terminée: {count} métriques")
    
    async def train_models(self):
        """Entraîne tous les modèles ML."""
        if not self.model_manager:
            raise RuntimeError("Gestionnaire de modèles non initialisé")
        
        self.logger.info("Démarrage de l'entraînement des modèles...")
        
        # Générer des données d'exemple pour l'entraînement
        sample_metrics = []
        for i in range(1000):
            metric = create_metric(
                name=f"training_metric_{i % 20}",
                value=float(i + (i % 13) * 2.5),
                tenant_id="training_tenant",
                tags={"source": f"node_{i % 10}"}
            )
            sample_metrics.append(metric)
        
        # Préparer les features
        training_data = prepare_metric_features(sample_metrics)
        
        # Entraîner le détecteur d'anomalies
        try:
            anomaly_model = self.model_manager.get_model('anomaly_detector')
            if anomaly_model:
                await anomaly_model.train(training_data)
                self.logger.info("✅ Détecteur d'anomalies entraîné")
        except Exception as e:
            self.logger.error(f"❌ Erreur entraînement détecteur d'anomalies: {e}")
        
        # Entraîner l'analyseur comportemental
        try:
            behavior_model = self.model_manager.get_model('behavior_analyzer')
            if behavior_model:
                await behavior_model.train(training_data)
                self.logger.info("✅ Analyseur comportemental entraîné")
        except Exception as e:
            self.logger.error(f"❌ Erreur entraînement analyseur comportemental: {e}")
        
        # Sauvegarder les modèles
        await self.model_manager.save_all_models()
        self.logger.info("Entraînement terminé et modèles sauvegardés")
    
    async def generate_report(self, tenant_id: str = "demo_tenant"):
        """Génère un rapport d'analytics."""
        if not self.is_running:
            raise RuntimeError("Système non démarré")
        
        print("\n" + "="*60)
        print(f"      RAPPORT ANALYTICS - {tenant_id.upper()}")
        print("="*60)
        print(f"Généré le: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print()
        
        # Métriques générales
        print("📊 MÉTRIQUES GÉNÉRALES")
        print("-" * 30)
        
        if self.storage_manager:
            storage_stats = self.storage_manager.get_all_stats()
            for storage_name, stats in storage_stats.items():
                print(f"{storage_name}:")
                print(f"  Écritures: {Formatter.format_number(stats['total_writes'])}")
                print(f"  Lectures: {Formatter.format_number(stats['total_reads'])}")
                print(f"  Erreurs: {stats['total_errors']}")
                print()
        
        # Modèles ML
        print("🧠 MODÈLES MACHINE LEARNING")
        print("-" * 30)
        
        if self.model_manager:
            model_stats = self.model_manager.get_all_model_stats()
            for model_name, stats in model_stats.items():
                print(f"{model_name}:")
                print(f"  Entraîné: {'✅' if stats['is_trained'] else '❌'}")
                print(f"  Features: {stats['feature_count']}")
                if stats['metrics']['accuracy'] > 0:
                    print(f"  Précision: {stats['metrics']['accuracy']:.3f}")
                print()
        
        # Alertes
        print("🚨 ALERTES")
        print("-" * 30)
        
        if self.engine and self.engine.alert_manager:
            active_alerts = len(self.engine.alert_manager.active_alerts)
            print(f"Alertes actives: {active_alerts}")
            
            if active_alerts > 0:
                for alert_id, alert in list(self.engine.alert_manager.active_alerts.items())[:5]:
                    print(f"  • {alert.name} ({alert.severity})")
        
        print("\n" + "="*60)
    
    async def test_predictions(self):
        """Teste les prédictions ML."""
        if not self.model_manager:
            raise RuntimeError("Gestionnaire de modèles non initialisé")
        
        self.logger.info("Test des prédictions ML...")
        
        # Données de test
        test_metrics = []
        for i in range(50):
            metric = create_metric(
                name=f"test_prediction_{i % 5}",
                value=float(i * 15 + (i % 11) * 3),
                tenant_id="test_tenant",
                tags={"source": f"test_node_{i % 3}"}
            )
            test_metrics.append(metric)
        
        test_data = prepare_metric_features(test_metrics)
        
        # Test détection d'anomalies
        anomaly_model = self.model_manager.get_model('anomaly_detector')
        if anomaly_model and anomaly_model.is_trained:
            try:
                prediction = await anomaly_model.predict(test_data)
                anomaly_count = sum(1 for r in prediction.prediction if r['is_anomaly'])
                self.logger.info(f"✅ Détection d'anomalies: {anomaly_count} anomalies sur {len(test_data)} échantillons")
            except Exception as e:
                self.logger.error(f"❌ Erreur prédiction anomalies: {e}")
        
        # Test analyse comportementale
        behavior_model = self.model_manager.get_model('behavior_analyzer')
        if behavior_model and behavior_model.is_trained:
            try:
                prediction = await behavior_model.predict(test_data)
                cluster_dist = prediction.metadata['cluster_distribution']
                self.logger.info(f"✅ Analyse comportementale: distribution des clusters {cluster_dist}")
            except Exception as e:
                self.logger.error(f"❌ Erreur analyse comportementale: {e}")


async def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Spotify AI Analytics System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
    python analytics_main.py start --env development
    python analytics_main.py ingest --count 1000
    python analytics_main.py train
    python analytics_main.py report --tenant demo_tenant
    python analytics_main.py status
        """
    )
    
    parser.add_argument(
        'command',
        choices=['start', 'stop', 'ingest', 'train', 'predict', 'report', 'status'],
        help='Commande à exécuter'
    )
    
    parser.add_argument(
        '--env',
        choices=['development', 'production', 'testing'],
        default='development',
        help='Environnement d\'exécution'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help='Nombre d\'éléments à traiter'
    )
    
    parser.add_argument(
        '--tenant',
        default='demo_tenant',
        help='ID du tenant'
    )
    
    parser.add_argument(
        '--config',
        help='Chemin vers le fichier de configuration'
    )
    
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Exécuter en mode daemon'
    )
    
    args = parser.parse_args()
    
    # Configuration
    if args.config:
        config = AnalyticsConfig()
        config.load_from_file(args.config)
    elif args.env == 'production':
        config = create_production_config()
    else:
        config = create_development_config()
    
    # Validation de la configuration
    errors = config.validate_config()
    if errors:
        print("❌ Erreurs de configuration:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)
    
    # Application
    app = AnalyticsApp(config)
    
    try:
        if args.command == 'start':
            await app.start()
            
            if args.daemon:
                # Mode daemon - continuer à tourner
                try:
                    while app.is_running:
                        await asyncio.sleep(60)
                        # Vérifications périodiques
                        if not await app.storage_manager.health_check_all():
                            app.logger.warning("Problème de santé détecté")
                except KeyboardInterrupt:
                    pass
            
            await app.stop()
        
        elif args.command == 'status':
            await app.start()
            await app.show_status()
            await app.stop()
        
        elif args.command == 'ingest':
            await app.start()
            await app.ingest_sample_data(args.count)
            await app.stop()
        
        elif args.command == 'train':
            await app.start()
            await app.train_models()
            await app.stop()
        
        elif args.command == 'predict':
            await app.start()
            await app.test_predictions()
            await app.stop()
        
        elif args.command == 'report':
            await app.start()
            await app.generate_report(args.tenant)
            await app.stop()
        
        elif args.command == 'stop':
            # Arrêt d'une instance en cours (nécessiterait un système de PID)
            print("Commande stop non implémentée (utiliser Ctrl+C)")
    
    except KeyboardInterrupt:
        print("\n🛑 Interruption utilisateur")
        await app.stop()
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        await app.stop()
        sys.exit(1)


if __name__ == "__main__":
    # Configuration logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Bannière
    print("""
    ██████╗ ██████╗  ██████╗ ████████╗██╗███████╗██╗   ██╗
    ██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝██║██╔════╝╚██╗ ██╔╝
    ██████╔╝██████╔╝██║   ██║   ██║   ██║█████╗   ╚████╔╝ 
    ██╔══██╗██╔═══╝ ██║   ██║   ██║   ██║██╔══╝    ╚██╔╝  
    ██████╔╝██║     ╚██████╔╝   ██║   ██║██║        ██║   
    ╚═════╝ ╚═╝      ╚═════╝    ╚═╝   ╚═╝╚═╝        ╚═╝   
                                                          
        ███████╗██████╗  ██████╗ ████████╗██╗███████╗██╗   ██╗
        ██╔════╝██╔══██╗██╔═══██╗╚══██╔══╝██║██╔════╝╚██╗ ██╔╝
        ███████╗██████╔╝██║   ██║   ██║   ██║█████╗   ╚████╔╝ 
        ╚════██║██╔═══╝ ██║   ██║   ██║   ██║██╔══╝    ╚██╔╝  
        ███████║██║     ╚██████╔╝   ██║   ██║██║        ██║   
        ╚══════╝╚═╝      ╚═════╝    ╚═╝   ╚═╝╚═╝        ╚═╝   
                                                              
                     🎵 AI ANALYTICS SYSTEM 🎵
                        Version 2.0.0
                     By Fahed Mlaiel & Team
    """)
    
    # Exécution
    asyncio.run(main())
