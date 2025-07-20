#!/usr/bin/env python3
"""
Script de validation des schémas de configuration tenant.

Ce script valide automatiquement toutes les configurations tenant
en utilisant les schémas JSON et les classes Pydantic.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging
from datetime import datetime

# Import des schémas
try:
    from schemas.tenant_schema import TenantConfigSchema
    from schemas.monitoring_schema import MonitoringConfigSchema
    from schemas.security_schema import SecurityPolicySchema
    from schemas.environment_schema import EnvironmentConfigSchema
    from schemas.localization_schema import LocalizationSchema
except ImportError:
    # Fallback pour l'exécution standalone
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from tenant_schema import TenantConfigSchema
    from monitoring_schema import MonitoringConfigSchema
    from security_schema import SecurityPolicySchema
    from environment_schema import EnvironmentConfigSchema
    from localization_schema import LocalizationSchema


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('validation.log')
    ]
)
logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validateur de configurations tenant."""
    
    def __init__(self, config_dir: str = None):
        """
        Initialise le validateur.
        
        Args:
            config_dir: Répertoire contenant les configurations à valider
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.validated_count = 0
        
        # Mapping des types de schémas
        self.schema_types = {
            'tenant': TenantConfigSchema,
            'monitoring': MonitoringConfigSchema,
            'security': SecurityPolicySchema,
            'environment': EnvironmentConfigSchema,
            'localization': LocalizationSchema
        }
    
    def detect_config_type(self, config_data: Dict[str, Any]) -> Optional[str]:
        """
        Détecte automatiquement le type de configuration.
        
        Args:
            config_data: Données de configuration
            
        Returns:
            Type de configuration détecté ou None
        """
        # Détection basée sur les clés présentes
        if 'tenant_id' in config_data and 'metadata' in config_data:
            return 'tenant'
        elif 'prometheus' in config_data and 'alertmanager' in config_data:
            return 'monitoring'
        elif 'authentication' in config_data and 'authorization' in config_data:
            return 'security'
        elif 'environment' in config_data and 'infrastructure' in config_data:
            return 'environment'
        elif 'default_locale' in config_data and 'translations' in config_data:
            return 'localization'
        
        return None
    
    def validate_single_config(self, config_path: Path) -> bool:
        """
        Valide une configuration individuelle.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            
        Returns:
            True si la validation réussit, False sinon
        """
        try:
            logger.info(f"Validation de {config_path}")
            
            # Chargement du fichier
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                elif config_path.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Format de fichier non supporté: {config_path.suffix}")
            
            # Détection du type
            config_type = self.detect_config_type(config_data)
            if not config_type:
                self.warnings.append({
                    'file': str(config_path),
                    'message': 'Type de configuration non détecté automatiquement',
                    'timestamp': datetime.utcnow().isoformat()
                })
                return False
            
            # Validation avec le schéma approprié
            schema_class = self.schema_types[config_type]
            validated_config = schema_class(**config_data)
            
            logger.info(f"✅ {config_path} - Type: {config_type} - Validation réussie")
            self.validated_count += 1
            return True
            
        except Exception as e:
            error_info = {
                'file': str(config_path),
                'error': str(e),
                'type': type(e).__name__,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.errors.append(error_info)
            logger.error(f"❌ {config_path} - Erreur: {e}")
            return False
    
    def validate_directory(self, recursive: bool = True) -> Dict[str, Any]:
        """
        Valide tous les fichiers de configuration dans un répertoire.
        
        Args:
            recursive: Si True, parcourt récursivement les sous-répertoires
            
        Returns:
            Rapport de validation
        """
        logger.info(f"Début de la validation dans {self.config_dir}")
        
        # Patterns de fichiers à valider
        patterns = ['*.json', '*.yaml', '*.yml']
        config_files = []
        
        for pattern in patterns:
            if recursive:
                config_files.extend(self.config_dir.rglob(pattern))
            else:
                config_files.extend(self.config_dir.glob(pattern))
        
        # Filtrage des fichiers (exclusion des fichiers de schéma)
        config_files = [
            f for f in config_files 
            if not f.name.endswith('_schema.json') and 
               not f.name.startswith('.')
        ]
        
        logger.info(f"Trouvé {len(config_files)} fichiers de configuration")
        
        # Validation de chaque fichier
        start_time = datetime.utcnow()
        for config_file in config_files:
            self.validate_single_config(config_file)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Génération du rapport
        report = {
            'summary': {
                'total_files': len(config_files),
                'validated_successfully': self.validated_count,
                'errors': len(self.errors),
                'warnings': len(self.warnings),
                'success_rate': round((self.validated_count / len(config_files)) * 100, 2) if config_files else 0,
                'duration_seconds': duration,
                'timestamp': end_time.isoformat()
            },
            'errors': self.errors,
            'warnings': self.warnings
        }
        
        return report
    
    def generate_report(self, report: Dict[str, Any], output_file: str = None) -> None:
        """
        Génère un rapport de validation.
        
        Args:
            report: Données du rapport
            output_file: Fichier de sortie (optionnel)
        """
        summary = report['summary']
        
        print("\n" + "="*60)
        print("           RAPPORT DE VALIDATION")
        print("="*60)
        print(f"Fichiers traités: {summary['total_files']}")
        print(f"Validés avec succès: {summary['validated_successfully']}")
        print(f"Erreurs: {summary['errors']}")
        print(f"Avertissements: {summary['warnings']}")
        print(f"Taux de réussite: {summary['success_rate']}%")
        print(f"Durée: {summary['duration_seconds']:.2f}s")
        
        if report['errors']:
            print(f"\n❌ ERREURS ({len(report['errors'])}):")
            for error in report['errors']:
                print(f"  - {error['file']}: {error['error']}")
        
        if report['warnings']:
            print(f"\n⚠️  AVERTISSEMENTS ({len(report['warnings'])}):")
            for warning in report['warnings']:
                print(f"  - {warning['file']}: {warning['message']}")
        
        if summary['errors'] == 0:
            print(f"\n🎉 Toutes les configurations sont valides!")
        
        # Sauvegarde du rapport JSON
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Rapport sauvegardé dans {output_file}")


def create_sample_configs():
    """Crée des configurations d'exemple pour les tests."""
    
    # Configuration tenant d'exemple
    tenant_config = {
        "tenant_id": "spotify-ai-example",
        "metadata": {
            "name": "Spotify AI Example",
            "description": "Configuration d'exemple pour tenant",
            "owner": {
                "user_id": "user_123",
                "email": "admin@example.com",
                "name": "Admin User"
            },
            "tags": ["example", "demo"]
        },
        "environments": {
            "development": {
                "enabled": True,
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "name": "spotify_ai_dev",
                    "schema": "public",
                    "ssl_mode": "prefer"
                },
                "monitoring": {
                    "enabled": True,
                    "logging": {
                        "level": "DEBUG",
                        "structured": True
                    }
                }
            }
        },
        "features": {
            "ai_processing": {
                "enabled": True,
                "models": ["gpt-3.5-turbo"],
                "rate_limits": {
                    "requests_per_minute": 100
                }
            },
            "spotify_integration": {
                "enabled": True,
                "api_version": "v1",
                "scopes": ["user-read-private"]
            }
        },
        "security": {
            "authentication": {
                "provider": "jwt",
                "session_timeout": 3600
            },
            "encryption": {
                "at_rest": True,
                "in_transit": True,
                "algorithm": "AES-256"
            }
        },
        "billing": {
            "plan": "free",
            "billing_cycle": "monthly"
        }
    }
    
    # Sauvegarde
    with open('example_tenant_config.json', 'w', encoding='utf-8') as f:
        json.dump(tenant_config, f, indent=2, ensure_ascii=False)
    
    logger.info("Configuration d'exemple créée: example_tenant_config.json")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Validateur de configurations tenant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python validate_configs.py                          # Valide le répertoire courant
  python validate_configs.py -d /path/to/configs     # Valide un répertoire spécifique
  python validate_configs.py --create-examples       # Crée des exemples de configuration
  python validate_configs.py -r report.json          # Sauvegarde le rapport en JSON
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='.',
        help='Répertoire contenant les configurations à valider (défaut: répertoire courant)'
    )
    
    parser.add_argument(
        '-r', '--report',
        type=str,
        help='Fichier de sortie pour le rapport JSON'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='Validation récursive des sous-répertoires (défaut: activé)'
    )
    
    parser.add_argument(
        '--create-examples',
        action='store_true',
        help='Crée des configurations d\'exemple'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mode verbeux'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.create_examples:
        create_sample_configs()
        return
    
    # Validation
    validator = ConfigValidator(args.directory)
    report = validator.validate_directory(args.recursive)
    validator.generate_report(report, args.report)
    
    # Code de sortie
    exit_code = 0 if report['summary']['errors'] == 0 else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
