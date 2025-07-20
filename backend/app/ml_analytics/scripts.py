# 🎵 ML Analytics Scripts
# =======================
# 
# Scripts d'automatisation et de maintenance ML Analytics
# Outils enterprise pour gestion et opérations
#
# 🎖️ Expert: Lead Dev + Architecte IA

"""
🔧 ML Analytics Scripts & Automation
====================================

Comprehensive automation scripts for ML Analytics:
- Model training and deployment automation
- Data pipeline orchestration
- Performance monitoring and optimization
- Maintenance and cleanup operations
- Backup and disaster recovery
"""

import asyncio
import argparse
import logging
import sys
import os
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import schedule
import time
import shutil
import subprocess
from dataclasses import dataclass

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_analytics_scripts.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Imports relatifs
try:
    from .core import MLAnalyticsEngine
    from .config import MLAnalyticsConfig
    from .monitoring import ml_monitor
    from .utils import MemoryManager, DataProcessor
except ImportError:
    # Import fallback pour exécution standalone
    logger.warning("Imports relatifs échoués, utilisation des imports absolus")


@dataclass
class ScriptConfig:
    """Configuration des scripts"""
    script_name: str
    environment: str = "development"
    config_path: Optional[str] = None
    dry_run: bool = False
    verbose: bool = False
    log_level: str = "INFO"


class MLAnalyticsScriptRunner:
    """Runner principal pour les scripts ML Analytics"""
    
    def __init__(self, config: ScriptConfig):
        self.config = config
        self.setup_logging()
        self.ml_config = MLAnalyticsConfig()
        self.engine = None
        
    def setup_logging(self):
        """Configuration du logging"""
        level = getattr(logging, self.config.log_level.upper())
        logging.getLogger().setLevel(level)
        
        if self.config.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    async def initialize(self):
        """Initialisation du runner"""
        self.engine = MLAnalyticsEngine()
        await self.engine.initialize()
        logger.info(f"Script {self.config.script_name} initialisé")
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        if self.engine:
            await self.engine.cleanup()
        logger.info(f"Script {self.config.script_name} terminé")


class ModelTrainingScript(MLAnalyticsScriptRunner):
    """Script d'entraînement de modèles"""
    
    async def train_recommendation_model(
        self,
        data_path: str,
        model_params: Dict[str, Any],
        output_path: str
    ):
        """Entraînement du modèle de recommandation"""
        logger.info(f"Début de l'entraînement du modèle de recommandation")
        
        if self.config.dry_run:
            logger.info("[DRY RUN] Simulation de l'entraînement")
            return
        
        try:
            # Chargement des données
            logger.info(f"Chargement des données depuis {data_path}")
            training_data = pd.read_csv(data_path)
            
            # Validation des données
            logger.info("Validation des données d'entraînement")
            if training_data.empty:
                raise ValueError("Données d'entraînement vides")
            
            # Préparation des données
            processor = DataProcessor()
            processed_data, stats = processor.normalize_features(
                training_data.select_dtypes(include=[np.number]).values
            )
            
            # Récupération du modèle
            model = await self.engine.get_model("spotify_recommendation")
            
            # Configuration de l'entraînement
            training_config = {
                **model_params,
                "data_stats": stats,
                "training_size": len(training_data),
                "started_at": datetime.utcnow().isoformat()
            }
            
            # Entraînement
            logger.info("Début de l'entraînement...")
            training_result = await model.train(
                training_data=processed_data,
                config=training_config
            )
            
            # Sauvegarde
            model_path = Path(output_path) / f"recommendation_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            await model.save(str(model_path))
            
            # Rapport de résultats
            logger.info(f"Entraînement terminé avec succès")
            logger.info(f"Modèle sauvegardé: {model_path}")
            logger.info(f"Métriques: {training_result.get('metrics', {})}")
            
            return {
                "status": "success",
                "model_path": str(model_path),
                "metrics": training_result.get("metrics", {}),
                "training_time": training_result.get("training_time", 0)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}")
            raise
    
    async def train_audio_analysis_model(
        self,
        data_path: str,
        model_params: Dict[str, Any],
        output_path: str
    ):
        """Entraînement du modèle d'analyse audio"""
        logger.info(f"Début de l'entraînement du modèle d'analyse audio")
        
        if self.config.dry_run:
            logger.info("[DRY RUN] Simulation de l'entraînement audio")
            return
        
        try:
            # Récupération du modèle d'analyse audio
            model = await self.engine.get_model("audio_analysis")
            
            # Configuration spécifique pour l'analyse audio
            audio_config = {
                **model_params,
                "sample_rate": 22050,
                "n_mfcc": 13,
                "n_fft": 2048,
                "hop_length": 512
            }
            
            # Entraînement avec données audio
            training_result = await model.train_from_audio_files(
                data_directory=data_path,
                config=audio_config
            )
            
            # Sauvegarde du modèle audio
            model_path = Path(output_path) / f"audio_analysis_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            await model.save(str(model_path))
            
            logger.info(f"Modèle d'analyse audio sauvegardé: {model_path}")
            
            return {
                "status": "success",
                "model_path": str(model_path),
                "metrics": training_result.get("metrics", {})
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement audio: {e}")
            raise


class DataPipelineScript(MLAnalyticsScriptRunner):
    """Script de gestion des pipelines de données"""
    
    async def run_etl_pipeline(
        self,
        source_config: Dict[str, Any],
        transformation_config: Dict[str, Any],
        destination_config: Dict[str, Any]
    ):
        """Exécution d'un pipeline ETL"""
        logger.info("Début du pipeline ETL")
        
        if self.config.dry_run:
            logger.info("[DRY RUN] Simulation du pipeline ETL")
            return
        
        try:
            # Extract
            logger.info("Phase d'extraction...")
            extracted_data = await self._extract_data(source_config)
            
            # Transform
            logger.info("Phase de transformation...")
            transformed_data = await self._transform_data(
                extracted_data, transformation_config
            )
            
            # Load
            logger.info("Phase de chargement...")
            load_result = await self._load_data(
                transformed_data, destination_config
            )
            
            logger.info(f"Pipeline ETL terminé avec succès")
            logger.info(f"Enregistrements traités: {len(transformed_data)}")
            
            return {
                "status": "success",
                "records_processed": len(transformed_data),
                "load_result": load_result
            }
            
        except Exception as e:
            logger.error(f"Erreur dans le pipeline ETL: {e}")
            raise
    
    async def _extract_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Extraction de données"""
        source_type = config.get("type", "csv")
        
        if source_type == "csv":
            return pd.read_csv(config["path"])
        elif source_type == "database":
            # Connexion base de données
            connection_string = config["connection_string"]
            query = config["query"]
            return pd.read_sql(query, connection_string)
        elif source_type == "api":
            # Récupération via API
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(config["url"]) as response:
                    data = await response.json()
                    return pd.DataFrame(data)
        else:
            raise ValueError(f"Type de source non supporté: {source_type}")
    
    async def _transform_data(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Transformation de données"""
        transformed = data.copy()
        
        # Nettoyage des valeurs manquantes
        if config.get("handle_missing", False):
            strategy = config.get("missing_strategy", "drop")
            processor = DataProcessor()
            transformed = processor.handle_missing_values(transformed, strategy)
        
        # Normalisation
        if config.get("normalize", False):
            numeric_columns = transformed.select_dtypes(include=[np.number]).columns
            if not numeric_columns.empty:
                processor = DataProcessor()
                normalized, _ = processor.normalize_features(
                    transformed[numeric_columns].values
                )
                transformed[numeric_columns] = normalized
        
        # Filtrage
        filters = config.get("filters", [])
        for filter_config in filters:
            column = filter_config["column"]
            operator = filter_config["operator"]
            value = filter_config["value"]
            
            if operator == "equals":
                transformed = transformed[transformed[column] == value]
            elif operator == "greater_than":
                transformed = transformed[transformed[column] > value]
            elif operator == "less_than":
                transformed = transformed[transformed[column] < value]
        
        return transformed
    
    async def _load_data(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Chargement de données"""
        destination_type = config.get("type", "csv")
        
        if destination_type == "csv":
            data.to_csv(config["path"], index=False)
            return {"records_written": len(data), "path": config["path"]}
        
        elif destination_type == "database":
            # Chargement en base de données
            connection_string = config["connection_string"]
            table_name = config["table_name"]
            data.to_sql(table_name, connection_string, if_exists="append", index=False)
            return {"records_written": len(data), "table": table_name}
        
        else:
            raise ValueError(f"Type de destination non supporté: {destination_type}")


class MaintenanceScript(MLAnalyticsScriptRunner):
    """Script de maintenance et optimisation"""
    
    async def optimize_models(self):
        """Optimisation des modèles"""
        logger.info("Début de l'optimisation des modèles")
        
        if self.config.dry_run:
            logger.info("[DRY RUN] Simulation de l'optimisation")
            return
        
        try:
            models = await self.engine.get_all_models()
            optimization_results = {}
            
            for model_id, model_info in models.items():
                logger.info(f"Optimisation du modèle: {model_id}")
                
                # Analyse des performances
                metrics = model_info.get("metrics", {})
                
                # Optimisation selon le type de modèle
                if "recommendation" in model_id:
                    result = await self._optimize_recommendation_model(model_id)
                elif "audio" in model_id:
                    result = await self._optimize_audio_model(model_id)
                else:
                    result = await self._optimize_generic_model(model_id)
                
                optimization_results[model_id] = result
            
            logger.info("Optimisation des modèles terminée")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation: {e}")
            raise
    
    async def _optimize_recommendation_model(self, model_id: str) -> Dict[str, Any]:
        """Optimisation spécifique pour les modèles de recommandation"""
        model = await self.engine.get_model(model_id)
        
        # Optimisation des hyperparamètres
        optimized_params = {
            "learning_rate": 0.001,
            "batch_size": 128,
            "regularization": 0.01
        }
        
        # Application des optimisations
        await model.update_parameters(optimized_params)
        
        return {
            "status": "optimized",
            "parameters_updated": optimized_params,
            "performance_improvement": "5%"  # Exemple
        }
    
    async def _optimize_audio_model(self, model_id: str) -> Dict[str, Any]:
        """Optimisation spécifique pour les modèles audio"""
        model = await self.engine.get_model(model_id)
        
        # Optimisation des paramètres audio
        optimized_params = {
            "n_mfcc": 13,
            "n_fft": 2048,
            "hop_length": 512
        }
        
        await model.update_audio_parameters(optimized_params)
        
        return {
            "status": "optimized",
            "audio_parameters_updated": optimized_params
        }
    
    async def _optimize_generic_model(self, model_id: str) -> Dict[str, Any]:
        """Optimisation générique"""
        return {
            "status": "no_optimization_available",
            "model_id": model_id
        }
    
    async def cleanup_old_models(self, days_old: int = 30):
        """Nettoyage des anciens modèles"""
        logger.info(f"Nettoyage des modèles de plus de {days_old} jours")
        
        if self.config.dry_run:
            logger.info("[DRY RUN] Simulation du nettoyage")
            return
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            cleaned_count = 0
            
            models = await self.engine.get_all_models()
            
            for model_id, model_info in models.items():
                last_used = model_info.get("last_used")
                if last_used and datetime.fromisoformat(last_used) < cutoff_date:
                    logger.info(f"Suppression du modèle ancien: {model_id}")
                    await self.engine.delete_model(model_id)
                    cleaned_count += 1
            
            logger.info(f"Nettoyage terminé: {cleaned_count} modèles supprimés")
            return {"models_cleaned": cleaned_count}
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
            raise
    
    async def backup_models(self, backup_path: str):
        """Sauvegarde des modèles"""
        logger.info(f"Sauvegarde des modèles vers {backup_path}")
        
        if self.config.dry_run:
            logger.info("[DRY RUN] Simulation de la sauvegarde")
            return
        
        try:
            backup_dir = Path(backup_path) / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            models = await self.engine.get_all_models()
            backed_up_count = 0
            
            for model_id, model_info in models.items():
                model_backup_path = backup_dir / f"{model_id}.backup"
                
                # Sauvegarde du modèle
                model = await self.engine.get_model(model_id)
                await model.save(str(model_backup_path))
                
                # Sauvegarde des métadonnées
                metadata_path = backup_dir / f"{model_id}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(model_info, f, indent=2, default=str)
                
                backed_up_count += 1
                logger.info(f"Modèle sauvegardé: {model_id}")
            
            logger.info(f"Sauvegarde terminée: {backed_up_count} modèles")
            return {
                "backup_path": str(backup_dir),
                "models_backed_up": backed_up_count
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise


class MonitoringScript(MLAnalyticsScriptRunner):
    """Script de monitoring et alertes"""
    
    async def run_health_checks(self):
        """Exécution des contrôles de santé"""
        logger.info("Exécution des contrôles de santé")
        
        try:
            # Démarrage du monitoring si nécessaire
            if not ml_monitor.running:
                await ml_monitor.start_monitoring()
            
            # Exécution des contrôles
            health_results = await ml_monitor.health_monitor.run_all_checks()
            
            # Rapport des résultats
            healthy_checks = sum(1 for result in health_results.values() if result.status == "healthy")
            total_checks = len(health_results)
            
            logger.info(f"Contrôles de santé: {healthy_checks}/{total_checks} sains")
            
            # Génération d'alertes si nécessaire
            for check_name, result in health_results.items():
                if result.status != "healthy":
                    ml_monitor.alert_manager.create_alert(
                        alert_id=f"health_check_{check_name}",
                        severity="warning",
                        title=f"Contrôle de santé échoué: {check_name}",
                        message=result.message,
                        source="health_check_script"
                    )
            
            return {
                "total_checks": total_checks,
                "healthy_checks": healthy_checks,
                "unhealthy_checks": total_checks - healthy_checks,
                "results": {name: result.to_dict() for name, result in health_results.items()}
            }
            
        except Exception as e:
            logger.error(f"Erreur lors des contrôles de santé: {e}")
            raise
    
    async def generate_performance_report(self, output_path: str):
        """Génération d'un rapport de performance"""
        logger.info(f"Génération du rapport de performance: {output_path}")
        
        try:
            # Collecte des métriques
            performance_stats = performance_monitor.get_statistics()
            system_health = ml_monitor.get_monitoring_status()
            
            # Génération du rapport
            report = {
                "generated_at": datetime.utcnow().isoformat(),
                "performance_metrics": performance_stats,
                "system_health": system_health,
                "recommendations": self._generate_performance_recommendations(performance_stats)
            }
            
            # Sauvegarde du rapport
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Rapport de performance généré: {output_path}")
            return report
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {e}")
            raise
    
    def _generate_performance_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Génération de recommandations de performance"""
        recommendations = []
        
        # Analyse des temps d'exécution
        execution_times = stats.get("execution_time", {})
        if execution_times.get("mean", 0) > 5:
            recommendations.append("Optimiser les temps d'exécution (moyenne > 5s)")
        
        # Analyse de l'utilisation mémoire
        memory_usage = stats.get("memory_usage", {})
        if memory_usage.get("max", 0) > 1000:  # > 1GB
            recommendations.append("Optimiser l'utilisation mémoire (pic > 1GB)")
        
        # Analyse du taux d'erreur
        success_rate = stats.get("success_rate", 1.0)
        if success_rate < 0.95:
            recommendations.append("Améliorer la fiabilité (taux de succès < 95%)")
        
        if not recommendations:
            recommendations.append("Performances optimales, aucune recommandation")
        
        return recommendations


def create_cli():
    """Création de l'interface en ligne de commande"""
    parser = argparse.ArgumentParser(description="Scripts ML Analytics")
    
    # Arguments globaux
    parser.add_argument("--config", type=str, help="Chemin du fichier de configuration")
    parser.add_argument("--environment", type=str, default="development", choices=["development", "staging", "production"])
    parser.add_argument("--dry-run", action="store_true", help="Mode simulation")
    parser.add_argument("--verbose", action="store_true", help="Mode verbeux")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")
    
    # Commande d'entraînement
    train_parser = subparsers.add_parser("train", help="Entraînement de modèles")
    train_parser.add_argument("--model-type", type=str, required=True, choices=["recommendation", "audio", "all"])
    train_parser.add_argument("--data-path", type=str, required=True, help="Chemin des données d'entraînement")
    train_parser.add_argument("--output-path", type=str, required=True, help="Chemin de sortie du modèle")
    train_parser.add_argument("--params", type=str, help="Paramètres du modèle (JSON)")
    
    # Commande de pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Exécution de pipelines")
    pipeline_parser.add_argument("--config-file", type=str, required=True, help="Fichier de configuration du pipeline")
    
    # Commande de maintenance
    maintenance_parser = subparsers.add_parser("maintenance", help="Opérations de maintenance")
    maintenance_parser.add_argument("--action", type=str, required=True, 
                                  choices=["optimize", "cleanup", "backup"])
    maintenance_parser.add_argument("--days-old", type=int, default=30, help="Ancienneté en jours")
    maintenance_parser.add_argument("--backup-path", type=str, help="Chemin de sauvegarde")
    
    # Commande de monitoring
    monitoring_parser = subparsers.add_parser("monitoring", help="Monitoring et alertes")
    monitoring_parser.add_argument("--action", type=str, required=True,
                                 choices=["health-check", "performance-report"])
    monitoring_parser.add_argument("--output", type=str, help="Chemin de sortie du rapport")
    
    return parser


async def main():
    """Fonction principale"""
    parser = create_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Configuration
    script_config = ScriptConfig(
        script_name=args.command,
        environment=args.environment,
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose,
        log_level=args.log_level
    )
    
    try:
        if args.command == "train":
            # Entraînement de modèles
            runner = ModelTrainingScript(script_config)
            await runner.initialize()
            
            try:
                params = json.loads(args.params) if args.params else {}
                
                if args.model_type == "recommendation":
                    result = await runner.train_recommendation_model(
                        args.data_path, params, args.output_path
                    )
                elif args.model_type == "audio":
                    result = await runner.train_audio_analysis_model(
                        args.data_path, params, args.output_path
                    )
                elif args.model_type == "all":
                    # Entraînement de tous les modèles
                    rec_result = await runner.train_recommendation_model(
                        args.data_path, params, args.output_path
                    )
                    audio_result = await runner.train_audio_analysis_model(
                        args.data_path, params, args.output_path
                    )
                    result = {"recommendation": rec_result, "audio": audio_result}
                
                logger.info(f"Résultat de l'entraînement: {result}")
            
            finally:
                await runner.cleanup()
        
        elif args.command == "pipeline":
            # Exécution de pipeline
            runner = DataPipelineScript(script_config)
            await runner.initialize()
            
            try:
                # Chargement de la configuration du pipeline
                with open(args.config_file, 'r') as f:
                    if args.config_file.endswith('.yaml') or args.config_file.endswith('.yml'):
                        pipeline_config = yaml.safe_load(f)
                    else:
                        pipeline_config = json.load(f)
                
                result = await runner.run_etl_pipeline(
                    pipeline_config["source"],
                    pipeline_config["transformation"],
                    pipeline_config["destination"]
                )
                
                logger.info(f"Résultat du pipeline: {result}")
            
            finally:
                await runner.cleanup()
        
        elif args.command == "maintenance":
            # Opérations de maintenance
            runner = MaintenanceScript(script_config)
            await runner.initialize()
            
            try:
                if args.action == "optimize":
                    result = await runner.optimize_models()
                elif args.action == "cleanup":
                    result = await runner.cleanup_old_models(args.days_old)
                elif args.action == "backup":
                    if not args.backup_path:
                        raise ValueError("--backup-path requis pour la sauvegarde")
                    result = await runner.backup_models(args.backup_path)
                
                logger.info(f"Résultat de la maintenance: {result}")
            
            finally:
                await runner.cleanup()
        
        elif args.command == "monitoring":
            # Monitoring et alertes
            runner = MonitoringScript(script_config)
            await runner.initialize()
            
            try:
                if args.action == "health-check":
                    result = await runner.run_health_checks()
                elif args.action == "performance-report":
                    if not args.output:
                        args.output = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    result = await runner.generate_performance_report(args.output)
                
                logger.info(f"Résultat du monitoring: {result}")
            
            finally:
                await runner.cleanup()
        
        logger.info(f"Script {args.command} exécuté avec succès")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


# Exports publics
__all__ = [
    'ScriptConfig',
    'MLAnalyticsScriptRunner',
    'ModelTrainingScript',
    'DataPipelineScript',
    'MaintenanceScript',
    'MonitoringScript',
    'create_cli',
    'main'
]
