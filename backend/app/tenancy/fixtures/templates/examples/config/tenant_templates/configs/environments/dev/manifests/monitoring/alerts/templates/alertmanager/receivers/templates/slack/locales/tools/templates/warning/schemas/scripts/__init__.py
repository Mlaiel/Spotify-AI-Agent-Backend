#!/usr/bin/env python3
"""
Tenancy Schema Scripts Module
============================

Module de scripts automatisés pour la gestion complète des schémas tenancy.
Architecture industrielle avec automation complète et intelligence opérationnelle.

Ce module fournit:
- Scripts de déploiement automatisé
- Outils de migration et synchronisation
- Automation de monitoring et alerting
- Scripts de maintenance et optimisation
- Outils de backup et restauration
- Automation de compliance et audit
- Scripts de performance et scaling
- Outils de diagnostic et debugging

Tous les scripts sont production-ready avec gestion d'erreurs complète,
logging avancé, métriques intégrées et support multi-environnement.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Version du module scripts
__version__ = "1.0.0"
__author__ = "Spotify AI Agent Team"
__maintainer__ = "DevOps & Platform Engineering Team"

# Configuration des scripts
SCRIPTS_CONFIG = {
    "version": __version__,
    "supported_environments": ["dev", "staging", "prod"],
    "default_log_level": "INFO",
    "max_concurrent_operations": 10,
    "timeout_seconds": 300,
    "retry_attempts": 3,
    "backup_retention_days": 30
}

# Modules disponibles
AVAILABLE_MODULES = {
    "deployment": "Scripts de déploiement automatisé",
    "migration": "Outils de migration et synchronisation",
    "monitoring": "Scripts de monitoring et alerting",
    "maintenance": "Outils de maintenance et optimisation",
    "backup": "Scripts de backup et restauration",
    "compliance": "Automation de compliance et audit",
    "performance": "Scripts de performance et scaling",
    "diagnostics": "Outils de diagnostic et debugging",
    "security": "Scripts de sécurité et audit",
    "analytics": "Outils d'analyse et reporting"
}

# Scripts principaux
MAIN_SCRIPTS = {
    "deploy_tenant": "deployment.tenant_deployer",
    "migrate_schemas": "migration.schema_migrator", 
    "setup_monitoring": "monitoring.monitoring_setup",
    "run_maintenance": "maintenance.maintenance_runner",
    "backup_data": "backup.backup_manager",
    "audit_compliance": "compliance.compliance_auditor",
    "analyze_performance": "performance.performance_analyzer",
    "diagnose_issues": "diagnostics.issue_diagnostician",
    "scan_security": "security.security_scanner",
    "generate_reports": "analytics.report_generator"
}

# Utilitaires partagés
SHARED_UTILITIES = {
    "config_manager": "Configuration centralisée",
    "logger_factory": "Factory de logging avancé",
    "metrics_collector": "Collecteur de métriques",
    "error_handler": "Gestionnaire d'erreurs global",
    "notification_sender": "Envoi de notifications",
    "environment_detector": "Détection d'environnement",
    "dependency_checker": "Vérification des dépendances",
    "health_checker": "Vérification de santé système"
}


def get_script_info(script_name: str) -> Optional[Dict[str, Any]]:
    """
    Retourne les informations d'un script.
    
    Args:
        script_name: Nom du script
        
    Returns:
        Dictionnaire avec les informations du script ou None
    """
    if script_name in MAIN_SCRIPTS:
        return {
            "name": script_name,
            "module": MAIN_SCRIPTS[script_name],
            "description": AVAILABLE_MODULES.get(script_name.split('_')[0], "Script personnalisé"),
            "version": __version__
        }
    return None


def list_available_scripts() -> List[Dict[str, str]]:
    """
    Liste tous les scripts disponibles.
    
    Returns:
        Liste des scripts avec leurs descriptions
    """
    scripts = []
    for script_name, module_path in MAIN_SCRIPTS.items():
        category = script_name.split('_')[0]
        scripts.append({
            "name": script_name,
            "category": category,
            "module": module_path,
            "description": AVAILABLE_MODULES.get(category, "Script personnalisé")
        })
    return scripts


def get_module_status() -> Dict[str, Any]:
    """
    Retourne le statut du module scripts.
    
    Returns:
        Dictionnaire avec les informations de statut
    """
    return {
        "version": __version__,
        "author": __author__,
        "maintainer": __maintainer__,
        "total_scripts": len(MAIN_SCRIPTS),
        "total_modules": len(AVAILABLE_MODULES),
        "utilities": len(SHARED_UTILITIES),
        "config": SCRIPTS_CONFIG,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform
    }


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure le logging pour les scripts.
    
    Args:
        log_level: Niveau de logging
        
    Returns:
        Logger configuré
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/tmp/tenancy_scripts.log')
        ]
    )
    return logging.getLogger(__name__)


# Logger par défaut
logger = setup_logging(SCRIPTS_CONFIG["default_log_level"])

# Export des principales fonctions et classes
__all__ = [
    # Configuration
    "SCRIPTS_CONFIG",
    "AVAILABLE_MODULES", 
    "MAIN_SCRIPTS",
    "SHARED_UTILITIES",
    
    # Fonctions utilitaires
    "get_script_info",
    "list_available_scripts", 
    "get_module_status",
    "setup_logging",
    
    # Logger
    "logger",
    
    # Métadonnées
    "__version__",
    "__author__",
    "__maintainer__"
]


# Message d'initialisation
logger.info(f"Tenancy Scripts Module v{__version__} initialized")
logger.info(f"Available scripts: {len(MAIN_SCRIPTS)}")
logger.info(f"Available modules: {len(AVAILABLE_MODULES)}")
logger.info(f"Shared utilities: {len(SHARED_UTILITIES)}")

# Vérification de l'environnement au chargement
try:
    import json
    import yaml
    import asyncio
    import aiofiles
    import httpx
    import psutil
    logger.info("All required dependencies are available")
except ImportError as e:
    logger.warning(f"Optional dependency missing: {e}")
    logger.info("Some scripts may have limited functionality")

# Détection de l'environnement
current_env = "dev"  # Valeur par défaut
if "prod" in str(Path.cwd()):
    current_env = "prod"
elif "staging" in str(Path.cwd()):
    current_env = "staging"

logger.info(f"Detected environment: {current_env}")
SCRIPTS_CONFIG["current_environment"] = current_env
            "alert_id": "alert_123",
            "title": "Test Alert",
            "message": "This is a test alert",
            "level": "warning",
            "category": "performance",
            "tenant_id": "test_tenant"
        }
        
        # Données de test pour NotificationMessage
        notification_test_data = {
            "channel": "slack",
            "recipients": ["user@example.com"],
            "body": "Test notification",
            "tenant_id": "test_tenant"
        }
        
        # Données de test pour MLModel
        ml_model_test_data = {
            "name": "test_model",
            "model_type": "classification",
            "framework": "tensorflow",
            "version": "1.0.0",
            "tenant_id": "test_tenant"
        }
        
        schemas_to_test = [
            (AlertInstance, alert_test_data),
            (NotificationMessage, notification_test_data),
            (MLModel, ml_model_test_data)
        ]
        
        all_valid = True
        for schema_class, test_data in schemas_to_test:
            if not self.validate_schema_file(schema_class, test_data):
                all_valid = False
        
        return all_valid


class SchemaDocumentationGenerator:
    """Générateur de documentation pour les schémas"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_schema_docs(self, schema_class: Type[BaseModel]) -> str:
        """Génère la documentation d'un schéma"""
        schema = schema_class.schema()
        
        doc = f"# {schema_class.__name__}\n\n"
        
        if schema.get('description'):
            doc += f"{schema['description']}\n\n"
        
        doc += "## Propriétés\n\n"
        
        properties = schema.get('properties', {})
        for prop_name, prop_info in properties.items():
            doc += f"### {prop_name}\n\n"
            doc += f"- **Type**: {prop_info.get('type', 'unknown')}\n"
            
            if prop_info.get('description'):
                doc += f"- **Description**: {prop_info['description']}\n"
            
            if 'default' in prop_info:
                doc += f"- **Défaut**: `{prop_info['default']}`\n"
            
            if prop_info.get('enum'):
                doc += f"- **Valeurs possibles**: {', '.join(prop_info['enum'])}\n"
            
            doc += "\n"
        
        # Exemple
        if hasattr(schema_class.Config, 'schema_extra'):
            example = schema_class.Config.schema_extra.get('example')
            if example:
                doc += "## Exemple\n\n"
                doc += "```json\n"
                doc += json.dumps(example, indent=2, ensure_ascii=False)
                doc += "\n```\n\n"
        
        return doc
    
    def generate_all_docs(self):
        """Génère la documentation pour tous les schémas"""
        from ..alerts import AlertInstance, AlertRule, AlertGroup
        from ..notifications import NotificationMessage, NotificationTemplate
        from ..ml import MLModel, AlertPrediction
        
        schemas = [
            AlertInstance, AlertRule, AlertGroup,
            NotificationMessage, NotificationTemplate,
            MLModel, AlertPrediction
        ]
        
        for schema_class in schemas:
            doc_content = self.generate_schema_docs(schema_class)
            doc_file = self.output_dir / f"{schema_class.__name__.lower()}.md"
            doc_file.write_text(doc_content, encoding='utf-8')
            print(f"📝 Generated documentation for {schema_class.__name__}")


class OpenAPIExporter:
    """Exporteur OpenAPI pour les schémas"""
    
    def __init__(self, output_file: Path):
        self.output_file = Path(output_file)
    
    def export_openapi_spec(self, schemas: List[Type[BaseModel]]) -> Dict[str, Any]:
        """Exporte la spécification OpenAPI"""
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Spotify AI Agent - Alert Schemas",
                "version": "1.0.0",
                "description": "Schémas de validation pour le système d'alerting"
            },
            "components": {
                "schemas": {}
            }
        }
        
        for schema_class in schemas:
            schema_name = schema_class.__name__
            schema_def = schema_class.schema()
            openapi_spec["components"]["schemas"][schema_name] = schema_def
        
        # Sauvegarde
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(openapi_spec, f, indent=2, ensure_ascii=False)
        
        return openapi_spec


class SchemaMetricsCollector:
    """Collecteur de métriques sur les schémas"""
    
    def __init__(self):
        self.metrics = {}
    
    def analyze_schema_complexity(self, schema_class: Type[BaseModel]) -> Dict[str, Any]:
        """Analyse la complexité d'un schéma"""
        schema = schema_class.schema()
        
        properties = schema.get('properties', {})
        required_fields = schema.get('required', [])
        
        metrics = {
            "name": schema_class.__name__,
            "total_fields": len(properties),
            "required_fields": len(required_fields),
            "optional_fields": len(properties) - len(required_fields),
            "nested_objects": 0,
            "enum_fields": 0,
            "array_fields": 0,
            "validation_rules": 0
        }
        
        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get('type')
            
            if prop_type == 'object':
                metrics["nested_objects"] += 1
            elif prop_type == 'array':
                metrics["array_fields"] += 1
            elif 'enum' in prop_info:
                metrics["enum_fields"] += 1
            
            # Compte les règles de validation
            if 'minimum' in prop_info or 'maximum' in prop_info:
                metrics["validation_rules"] += 1
            if 'pattern' in prop_info:
                metrics["validation_rules"] += 1
            if 'minLength' in prop_info or 'maxLength' in prop_info:
                metrics["validation_rules"] += 1
        
        # Score de complexité
        complexity_score = (
            metrics["total_fields"] * 1 +
            metrics["nested_objects"] * 3 +
            metrics["validation_rules"] * 2 +
            metrics["enum_fields"] * 1
        )
        
        metrics["complexity_score"] = complexity_score
        
        if complexity_score < 20:
            metrics["complexity_level"] = "simple"
        elif complexity_score < 50:
            metrics["complexity_level"] = "moderate"
        else:
            metrics["complexity_level"] = "complex"
        
        return metrics
    
    def generate_metrics_report(self, schemas: List[Type[BaseModel]]) -> Dict[str, Any]:
        """Génère un rapport de métriques"""
        schema_metrics = []
        
        for schema_class in schemas:
            metrics = self.analyze_schema_complexity(schema_class)
            schema_metrics.append(metrics)
        
        # Statistiques globales
        total_schemas = len(schema_metrics)
        total_fields = sum(m["total_fields"] for m in schema_metrics)
        avg_complexity = sum(m["complexity_score"] for m in schema_metrics) / total_schemas
        
        complexity_distribution = {
            "simple": len([m for m in schema_metrics if m["complexity_level"] == "simple"]),
            "moderate": len([m for m in schema_metrics if m["complexity_level"] == "moderate"]),
            "complex": len([m for m in schema_metrics if m["complexity_level"] == "complex"])
        }
        
        report = {
            "generation_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_schemas": total_schemas,
                "total_fields": total_fields,
                "average_fields_per_schema": total_fields / total_schemas,
                "average_complexity_score": avg_complexity,
                "complexity_distribution": complexity_distribution
            },
            "schemas": schema_metrics
        }
        
        return report


async def run_schema_validation():
    """Exécute la validation complète des schémas"""
    print("🔍 Démarrage de la validation des schémas...")
    
    validator = SchemaValidator()
    success = validator.validate_all_schemas()
    
    if success:
        print("✅ Tous les schémas sont valides!")
        return 0
    else:
        print(f"❌ {len(validator.validation_errors)} erreurs de validation trouvées")
        return 1


async def generate_documentation():
    """Génère la documentation des schémas"""
    print("📚 Génération de la documentation...")
    
    docs_dir = Path(__file__).parent.parent / "docs" / "schemas"
    generator = SchemaDocumentationGenerator(docs_dir)
    generator.generate_all_docs()
    
    print(f"📚 Documentation générée dans {docs_dir}")


async def export_openapi():
    """Exporte la spécification OpenAPI"""
    print("📋 Export de la spécification OpenAPI...")
    
    from ..alerts import AlertInstance, AlertRule
    from ..notifications import NotificationMessage
    from ..ml import MLModel
    
    schemas = [AlertInstance, AlertRule, NotificationMessage, MLModel]
    
    output_file = Path(__file__).parent.parent / "openapi.json"
    exporter = OpenAPIExporter(output_file)
    spec = exporter.export_openapi_spec(schemas)
    
    print(f"📋 Spécification OpenAPI exportée vers {output_file}")


async def analyze_metrics():
    """Analyse les métriques des schémas"""
    print("📊 Analyse des métriques des schémas...")
    
    from ..alerts import AlertInstance, AlertRule, AlertGroup
    from ..notifications import NotificationMessage, NotificationTemplate
    from ..ml import MLModel, AlertPrediction
    
    schemas = [
        AlertInstance, AlertRule, AlertGroup,
        NotificationMessage, NotificationTemplate,
        MLModel, AlertPrediction
    ]
    
    collector = SchemaMetricsCollector()
    report = collector.generate_metrics_report(schemas)
    
    # Sauvegarde du rapport
    report_file = Path(__file__).parent.parent / "metrics_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Rapport de métriques généré dans {report_file}")
    
    # Affichage du résumé
    summary = report["summary"]
    print(f"\n📈 Résumé:")
    print(f"  - Schémas totaux: {summary['total_schemas']}")
    print(f"  - Champs totaux: {summary['total_fields']}")
    print(f"  - Moyenne champs/schéma: {summary['average_fields_per_schema']:.1f}")
    print(f"  - Score complexité moyen: {summary['average_complexity_score']:.1f}")
    print(f"  - Distribution complexité: {summary['complexity_distribution']}")


async def main():
    """Point d'entrée principal"""
    if len(sys.argv) < 2:
        print("Usage: python schemas_utils.py <command>")
        print("Commands: validate, docs, openapi, metrics, all")
        return 1
    
    command = sys.argv[1]
    
    if command == "validate":
        return await run_schema_validation()
    elif command == "docs":
        await generate_documentation()
    elif command == "openapi":
        await export_openapi()
    elif command == "metrics":
        await analyze_metrics()
    elif command == "all":
        print("🚀 Exécution de toutes les tâches...")
        result = await run_schema_validation()
        if result == 0:
            await generate_documentation()
            await export_openapi()
            await analyze_metrics()
            print("✅ Toutes les tâches terminées avec succès!")
        else:
            print("❌ Validation échouée, arrêt des autres tâches")
            return result
    else:
        print(f"❌ Commande inconnue: {command}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
