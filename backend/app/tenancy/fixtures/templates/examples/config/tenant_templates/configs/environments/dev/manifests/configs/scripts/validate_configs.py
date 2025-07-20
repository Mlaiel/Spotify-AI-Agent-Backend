#!/usr/bin/env python3
"""
Configuration Validation Script
==============================

Script pour valider toutes les configurations du système Spotify AI Agent.
Vérifie la cohérence, la sécurité et la conformité des configurations.

Author: Fahed Mlaiel
Team: Spotify AI Agent Development Team
Version: 1.0.0

Usage:
    python validate_configs.py [options]
    
Examples:
    python validate_configs.py --config-dir ./configs/
    python validate_configs.py --environment production --strict
    python validate_configs.py --check-security --check-performance
"""

import argparse
import json
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import re

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation_schemas import (
    ConfigurationValidator,
    ApplicationConfigSchema,
    DatabaseConfigSchema,
    SecurityConfigSchema,
    MLConfigSchema,
    MonitoringConfigSchema
)
from security_policies import SecurityPolicyManager
from feature_flags import FeatureFlagManager

class ConfigurationValidationError(Exception):
    """Exception pour les erreurs de validation."""
    pass

class ComprehensiveValidator:
    """Validateur complet de configurations."""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.validator = ConfigurationValidator()
        self.security_manager = SecurityPolicyManager(environment)
        self.feature_manager = FeatureFlagManager(environment)
        self.errors = []
        self.warnings = []
        self.infos = []
    
    def validate_config_files(self, config_dir: Path) -> Dict[str, Any]:
        """Valide tous les fichiers de configuration dans un répertoire."""
        results = {
            "total_files": 0,
            "valid_files": 0,
            "files_with_errors": 0,
            "files_with_warnings": 0,
            "details": {}
        }
        
        # Recherche des fichiers de configuration
        config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml")) + list(config_dir.glob("*.json"))
        results["total_files"] = len(config_files)
        
        for config_file in config_files:
            file_result = self._validate_single_file(config_file)
            results["details"][str(config_file)] = file_result
            
            if file_result["valid"]:
                results["valid_files"] += 1
            if file_result["errors"]:
                results["files_with_errors"] += 1
            if file_result["warnings"]:
                results["files_with_warnings"] += 1
        
        return results
    
    def _validate_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Valide un seul fichier de configuration."""
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "infos": [],
            "config_type": "unknown",
            "content": None
        }
        
        try:
            # Lecture du fichier
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    content = yaml.safe_load(f)
                else:
                    content = json.load(f)
            
            result["content"] = content
            
            # Détection du type de configuration
            config_type = self._detect_config_type(file_path.name, content)
            result["config_type"] = config_type
            
            # Validation selon le type
            if config_type in ["application", "database", "security", "ml", "monitoring"]:
                is_valid, errors = self.validator.validate_config(config_type, content)
                result["valid"] = is_valid
                result["errors"] = errors
            else:
                result["warnings"].append(f"Type de configuration non reconnu: {config_type}")
            
            # Validations supplémentaires
            self._validate_security_aspects(content, result)
            self._validate_performance_aspects(content, result)
            self._validate_business_rules(content, result)
            
        except yaml.YAMLError as e:
            result["errors"].append(f"Erreur YAML: {e}")
        except json.JSONDecodeError as e:
            result["errors"].append(f"Erreur JSON: {e}")
        except Exception as e:
            result["errors"].append(f"Erreur générale: {e}")
        
        return result
    
    def _detect_config_type(self, filename: str, content: Any) -> str:
        """Détecte automatiquement le type de configuration."""
        filename_lower = filename.lower()
        
        # Détection par nom de fichier
        if "configmap" in filename_lower or "application" in filename_lower:
            return "application"
        elif "database" in filename_lower or "db" in filename_lower:
            return "database"
        elif "security" in filename_lower or "auth" in filename_lower:
            return "security"
        elif "ml" in filename_lower or "machine" in filename_lower or "ai" in filename_lower:
            return "ml"
        elif "monitoring" in filename_lower or "metrics" in filename_lower:
            return "monitoring"
        
        # Détection par contenu
        if isinstance(content, dict):
            keys = set(str(k).lower() for k in content.keys())
            
            if any(k.startswith("db_") or k.startswith("database_") for k in keys):
                return "database"
            elif any(k.startswith("jwt_") or k.startswith("oauth_") or k.startswith("auth_") for k in keys):
                return "security"
            elif any(k.startswith("ml_") or k.startswith("ai_") or k.startswith("model_") for k in keys):
                return "ml"
            elif any(k.startswith("prometheus_") or k.startswith("grafana_") or k.startswith("monitoring_") for k in keys):
                return "monitoring"
            elif any(k in ["debug", "log_level", "environment", "port", "host"] for k in keys):
                return "application"
        
        return "unknown"
    
    def _validate_security_aspects(self, content: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Valide les aspects de sécurité."""
        if not isinstance(content, dict):
            return
        
        # Vérification des mots de passe faibles
        for key, value in content.items():
            if isinstance(value, str):
                if "password" in key.lower() and value in ["password", "123456", "admin", "test"]:
                    result["errors"].append(f"Mot de passe faible détecté: {key}")
                
                if "secret" in key.lower() and len(value) < 16:
                    result["warnings"].append(f"Secret potentiellement faible: {key} (longueur < 16)")
        
        # Vérification des configurations de sécurité
        if "https_only" in content and str(content["https_only"]).lower() == "false":
            if self.environment == "production":
                result["errors"].append("HTTPS_ONLY doit être activé en production")
            else:
                result["warnings"].append("HTTPS_ONLY désactivé - OK pour dev/staging")
        
        if "debug" in content and str(content["debug"]).lower() == "true":
            if self.environment == "production":
                result["errors"].append("DEBUG ne doit pas être activé en production")
            else:
                result["infos"].append("Mode DEBUG activé - normal pour dev/staging")
        
        # Vérification des CORS
        if "allowed_origins" in content:
            origins = str(content["allowed_origins"])
            if "*" in origins and self.environment == "production":
                result["errors"].append("CORS wildcard (*) ne doit pas être utilisé en production")
    
    def _validate_performance_aspects(self, content: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Valide les aspects de performance."""
        if not isinstance(content, dict):
            return
        
        # Vérification des pools de connexions
        if "db_pool_size" in content:
            pool_size = int(content.get("db_pool_size", 0))
            if pool_size > 1000:
                result["warnings"].append(f"Taille de pool DB très élevée: {pool_size}")
            elif pool_size < 5:
                result["warnings"].append(f"Taille de pool DB très faible: {pool_size}")
        
        # Vérification des timeouts
        if "worker_timeout" in content:
            timeout = int(content.get("worker_timeout", 0))
            if timeout > 300:
                result["warnings"].append(f"Timeout worker très élevé: {timeout}s")
            elif timeout < 10:
                result["warnings"].append(f"Timeout worker très faible: {timeout}s")
        
        # Vérification de la taille du cache
        if "cache_max_size" in content:
            cache_size = int(content.get("cache_max_size", 0))
            if cache_size > 1000000:
                result["warnings"].append(f"Taille de cache très élevée: {cache_size}")
        
        # Vérification des workers
        if "max_workers" in content:
            workers = int(content.get("max_workers", 0))
            if workers > 32:
                result["warnings"].append(f"Nombre de workers très élevé: {workers}")
            elif workers < 1:
                result["errors"].append("Le nombre de workers doit être au moins 1")
    
    def _validate_business_rules(self, content: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Valide les règles métier."""
        if not isinstance(content, dict):
            return
        
        # Vérification de cohérence des environnements
        if "environment" in content:
            env = content["environment"]
            if env != self.environment:
                result["warnings"].append(f"Environnement dans config ({env}) != environnement attendu ({self.environment})")
        
        # Vérification des feature flags
        for key, value in content.items():
            if key.startswith("FEATURE_") and key.endswith("_ENABLED"):
                feature_key = key[8:-8].lower().replace("_", ".")
                if not self.feature_manager.get_flag(feature_key):
                    result["warnings"].append(f"Feature flag non défini: {feature_key}")
        
        # Vérification des URLs
        for key, value in content.items():
            if isinstance(value, str) and any(keyword in key.lower() for keyword in ["url", "uri", "endpoint"]):
                if value.startswith("http://") and self.environment == "production":
                    result["warnings"].append(f"URL HTTP en production: {key}")
    
    def validate_environment_consistency(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Valide la cohérence entre les différentes configurations."""
        result = {
            "consistent": True,
            "errors": [],
            "warnings": [],
            "checks_performed": []
        }
        
        # Vérification de cohérence des environnements
        environments = set()
        for config_name, config_data in configs.items():
            if isinstance(config_data, dict) and "environment" in config_data:
                environments.add(config_data["environment"])
        
        if len(environments) > 1:
            result["errors"].append(f"Environnements incohérents détectés: {environments}")
            result["consistent"] = False
        
        result["checks_performed"].append("environment_consistency")
        
        # Vérification de cohérence des bases de données
        db_hosts = set()
        for config_name, config_data in configs.items():
            if isinstance(config_data, dict):
                for key, value in config_data.items():
                    if "db_host" in key.lower():
                        db_hosts.add(value)
        
        if len(db_hosts) > 2:  # Primary + replica acceptable
            result["warnings"].append(f"Plusieurs hôtes de DB détectés: {db_hosts}")
        
        result["checks_performed"].append("database_consistency")
        
        # Vérification de cohérence des secrets
        secret_keys = set()
        for config_name, config_data in configs.items():
            if isinstance(config_data, dict):
                for key in config_data.keys():
                    if any(secret_word in key.lower() for secret_word in ["secret", "key", "token", "password"]):
                        secret_keys.add(key)
        
        if len(secret_keys) == 0:
            result["warnings"].append("Aucun secret détecté - vérifiez la configuration de sécurité")
        
        result["checks_performed"].append("secrets_consistency")
        
        return result
    
    def validate_security_compliance(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Valide la conformité aux standards de sécurité."""
        result = {
            "compliant": True,
            "security_score": 0,
            "max_score": 100,
            "issues": [],
            "recommendations": []
        }
        
        score = 0
        
        # Vérification HTTPS (10 points)
        https_configured = False
        for config_data in configs.values():
            if isinstance(config_data, dict):
                if config_data.get("https_only") == "true":
                    https_configured = True
                    score += 10
                    break
        
        if not https_configured and self.environment == "production":
            result["issues"].append("HTTPS non configuré en production")
            result["recommendations"].append("Activez HTTPS_ONLY=true en production")
        
        # Vérification authentification forte (15 points)
        jwt_configured = False
        oauth_configured = False
        for config_data in configs.values():
            if isinstance(config_data, dict):
                if any("jwt" in key.lower() for key in config_data.keys()):
                    jwt_configured = True
                    score += 10
                if config_data.get("oauth_enabled") == "true":
                    oauth_configured = True
                    score += 5
                    break
        
        if not jwt_configured:
            result["issues"].append("Authentification JWT non configurée")
            result["recommendations"].append("Configurez l'authentification JWT")
        
        # Vérification rate limiting (10 points)
        rate_limit_configured = False
        for config_data in configs.values():
            if isinstance(config_data, dict):
                if config_data.get("rate_limit_enabled") == "true":
                    rate_limit_configured = True
                    score += 10
                    break
        
        if not rate_limit_configured:
            result["issues"].append("Rate limiting non configuré")
            result["recommendations"].append("Activez le rate limiting pour prévenir les abus")
        
        # Vérification audit logging (15 points)
        audit_configured = False
        for config_data in configs.values():
            if isinstance(config_data, dict):
                if config_data.get("audit_log_enabled") == "true":
                    audit_configured = True
                    score += 15
                    break
        
        if not audit_configured:
            result["issues"].append("Audit logging non configuré")
            result["recommendations"].append("Activez l'audit logging pour la traçabilité")
        
        # Vérification chiffrement (20 points)
        encryption_configured = False
        for config_data in configs.values():
            if isinstance(config_data, dict):
                if (config_data.get("encryption_at_rest") == "true" or 
                    config_data.get("encryption_in_transit") == "true"):
                    encryption_configured = True
                    score += 20
                    break
        
        if not encryption_configured:
            result["issues"].append("Chiffrement non configuré")
            result["recommendations"].append("Configurez le chiffrement des données")
        
        # Vérification monitoring sécurité (10 points)
        security_monitoring = False
        for config_data in configs.values():
            if isinstance(config_data, dict):
                if (config_data.get("security_monitoring_enabled") == "true" or
                    config_data.get("real_time_alerts") == "true"):
                    security_monitoring = True
                    score += 10
                    break
        
        # Vérification CSRF protection (5 points)
        csrf_protection = False
        for config_data in configs.values():
            if isinstance(config_data, dict):
                if config_data.get("csrf_protection") == "true":
                    csrf_protection = True
                    score += 5
                    break
        
        # Vérification session security (10 points)
        session_security = False
        for config_data in configs.values():
            if isinstance(config_data, dict):
                if (config_data.get("secure_cookies") == "true" or
                    config_data.get("session_timeout")):
                    session_security = True
                    score += 10
                    break
        
        # Vérification password policy (5 points)
        password_policy = False
        for config_data in configs.values():
            if isinstance(config_data, dict):
                if any("password_min_length" in key.lower() for key in config_data.keys()):
                    password_policy = True
                    score += 5
                    break
        
        result["security_score"] = score
        result["compliant"] = score >= 70  # 70% minimum pour être compliant
        
        if result["security_score"] < 70:
            result["issues"].append(f"Score de sécurité insuffisant: {score}/100")
        
        return result
    
    def generate_validation_report(self, 
                                 file_results: Dict[str, Any],
                                 consistency_results: Dict[str, Any],
                                 security_results: Dict[str, Any]) -> str:
        """Génère un rapport de validation complet."""
        report = []
        report.append("=" * 80)
        report.append("RAPPORT DE VALIDATION DES CONFIGURATIONS")
        report.append("=" * 80)
        report.append(f"Généré le: {datetime.now().isoformat()}")
        report.append(f"Environnement: {self.environment}")
        report.append("")
        
        # Résumé des fichiers
        report.append("RÉSUMÉ DES FICHIERS")
        report.append("-" * 40)
        report.append(f"Total fichiers: {file_results['total_files']}")
        report.append(f"Fichiers valides: {file_results['valid_files']}")
        report.append(f"Fichiers avec erreurs: {file_results['files_with_errors']}")
        report.append(f"Fichiers avec avertissements: {file_results['files_with_warnings']}")
        report.append("")
        
        # Détails des erreurs
        if file_results['files_with_errors'] > 0:
            report.append("ERREURS DÉTECTÉES")
            report.append("-" * 40)
            for file_path, details in file_results['details'].items():
                if details['errors']:
                    report.append(f"📁 {file_path}")
                    for error in details['errors']:
                        report.append(f"   ❌ {error}")
            report.append("")
        
        # Cohérence des configurations
        report.append("COHÉRENCE DES CONFIGURATIONS")
        report.append("-" * 40)
        if consistency_results['consistent']:
            report.append("✅ Configurations cohérentes")
        else:
            report.append("❌ Incohérences détectées:")
            for error in consistency_results['errors']:
                report.append(f"   • {error}")
        
        for warning in consistency_results['warnings']:
            report.append(f"   ⚠️ {warning}")
        report.append("")
        
        # Conformité sécurité
        report.append("CONFORMITÉ SÉCURITÉ")
        report.append("-" * 40)
        report.append(f"Score de sécurité: {security_results['security_score']}/100")
        
        if security_results['compliant']:
            report.append("✅ Conforme aux standards de sécurité")
        else:
            report.append("❌ Non conforme aux standards de sécurité")
        
        if security_results['issues']:
            report.append("\nProblèmes de sécurité:")
            for issue in security_results['issues']:
                report.append(f"   • {issue}")
        
        if security_results['recommendations']:
            report.append("\nRecommandations:")
            for rec in security_results['recommendations']:
                report.append(f"   💡 {rec}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Validateur de configurations Spotify AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python validate_configs.py --config-dir ./configs/
  python validate_configs.py --environment production --strict
  python validate_configs.py --check-security --output-report validation_report.txt
        """
    )
    
    parser.add_argument(
        "--config-dir", "-d",
        type=Path,
        default=Path("./configs"),
        help="Répertoire contenant les configurations à valider"
    )
    
    parser.add_argument(
        "--environment", "-e",
        choices=["local", "development", "staging", "production"],
        default="development",
        help="Environnement cible (défaut: development)"
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Mode strict: les avertissements sont traités comme des erreurs"
    )
    
    parser.add_argument(
        "--check-security",
        action="store_true",
        help="Effectuer une vérification de sécurité approfondie"
    )
    
    parser.add_argument(
        "--check-performance",
        action="store_true",
        help="Effectuer une vérification de performance"
    )
    
    parser.add_argument(
        "--output-report", "-o",
        type=Path,
        help="Fichier de sortie pour le rapport de validation"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json", "yaml"],
        default="text",
        help="Format du rapport (défaut: text)"
    )
    
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Échouer si des avertissements sont détectés"
    )
    
    args = parser.parse_args()
    
    try:
        # Vérification de l'existence du répertoire
        if not args.config_dir.exists():
            print(f"❌ Erreur: Le répertoire {args.config_dir} n'existe pas")
            sys.exit(1)
        
        # Création du validateur
        validator = ComprehensiveValidator(args.environment)
        
        print(f"🔍 Validation des configurations dans: {args.config_dir}")
        print(f"📍 Environnement: {args.environment}")
        
        # Validation des fichiers
        file_results = validator.validate_config_files(args.config_dir)
        
        # Chargement de toutes les configurations pour les validations croisées
        all_configs = {}
        for file_path, details in file_results["details"].items():
            if details["content"]:
                all_configs[file_path] = details["content"]
        
        # Validation de cohérence
        consistency_results = validator.validate_environment_consistency(all_configs)
        
        # Validation de sécurité
        security_results = validator.validate_security_compliance(all_configs)
        
        # Génération du rapport
        report = validator.generate_validation_report(
            file_results, consistency_results, security_results
        )
        
        # Affichage du rapport
        print("\n" + report)
        
        # Sauvegarde du rapport si demandé
        if args.output_report:
            with open(args.output_report, 'w', encoding='utf-8') as f:
                if args.format == "json":
                    json.dump({
                        "file_results": file_results,
                        "consistency_results": consistency_results,
                        "security_results": security_results
                    }, f, indent=2, default=str)
                elif args.format == "yaml":
                    yaml.dump({
                        "file_results": file_results,
                        "consistency_results": consistency_results,
                        "security_results": security_results
                    }, f, default_flow_style=False)
                else:
                    f.write(report)
            
            print(f"\n📄 Rapport sauvegardé: {args.output_report}")
        
        # Détermination du code de sortie
        exit_code = 0
        
        if file_results["files_with_errors"] > 0:
            exit_code = 1
        elif not consistency_results["consistent"]:
            exit_code = 1
        elif not security_results["compliant"] and args.check_security:
            exit_code = 1
        elif (file_results["files_with_warnings"] > 0 and 
              (args.strict or args.fail_on_warnings)):
            exit_code = 1
        
        if exit_code == 0:
            print("\n✅ Validation réussie!")
        else:
            print("\n❌ Validation échouée!")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ Erreur lors de la validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
