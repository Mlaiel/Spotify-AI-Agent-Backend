#!/usr/bin/env python3
"""
Configuration Generation Script
==============================

Script pour générer automatiquement toutes les configurations 
de l'environnement de développement Spotify AI Agent.

Author: Fahed Mlaiel
Team: Spotify AI Agent Development Team
Version: 1.0.0

Usage:
    python generate_configs.py [options]
    
Examples:
    python generate_configs.py --environment dev
    python generate_configs.py --environment dev --output ./generated/
    python generate_configs.py --profile high_throughput --format yaml
"""

import argparse
import json
import yaml
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from __init__ import (
    ConfigMapManager, 
    EnvironmentTier, 
    ConfigMapUtils,
    ConfigurationValidator
)
from environment_profiles import EnvironmentProfileManager, EnvironmentType
from feature_flags import FeatureFlagManager
from security_policies import SecurityPolicyManager
from performance_tuning import PerformanceTuningManager, PerformanceProfile

class ConfigurationGenerator:
    """Générateur de configurations complètes."""
    
    def __init__(self, 
                 environment: str = "development",
                 namespace: str = "spotify-ai-agent-dev",
                 output_dir: str = "./generated"):
        self.environment = environment
        self.namespace = namespace
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisation des gestionnaires
        self.config_manager = ConfigMapManager(
            namespace=namespace,
            environment=EnvironmentTier(environment)
        )
        self.profile_manager = EnvironmentProfileManager()
        self.feature_manager = FeatureFlagManager(environment)
        self.security_manager = SecurityPolicyManager(environment)
        self.performance_manager = PerformanceTuningManager(
            environment=environment
        )
        self.validator = ConfigurationValidator()
    
    def generate_all_configs(self, 
                           performance_profile: str = "balanced",
                           include_secrets: bool = False) -> Dict[str, Any]:
        """Génère toutes les configurations."""
        print(f"🚀 Génération des configurations pour l'environnement: {self.environment}")
        
        # Configuration du profil de performance
        perf_profile = PerformanceProfile(performance_profile)
        self.performance_manager = PerformanceTuningManager(
            profile=perf_profile,
            environment=self.environment
        )
        
        configs = {}
        
        # 1. ConfigMaps Kubernetes
        print("📋 Génération des ConfigMaps...")
        configs["configmaps"] = self._generate_configmaps()
        
        # 2. Profils d'environnement
        print("🏗️ Génération des profils d'environnement...")
        configs["environment_profiles"] = self._generate_environment_profiles()
        
        # 3. Feature Flags
        print("🎚️ Génération des feature flags...")
        configs["feature_flags"] = self._generate_feature_flags()
        
        # 4. Politiques de sécurité
        print("🔒 Génération des politiques de sécurité...")
        configs["security_policies"] = self._generate_security_policies()
        
        # 5. Configuration de performance
        print("⚡ Génération des configurations de performance...")
        configs["performance_tuning"] = self._generate_performance_configs()
        
        # 6. Variables d'environnement
        print("🌍 Génération des variables d'environnement...")
        configs["environment_variables"] = self._generate_env_vars()
        
        # 7. Secrets (optionnel)
        if include_secrets:
            print("🔐 Génération des templates de secrets...")
            configs["secrets_templates"] = self._generate_secrets_templates()
        
        # 8. Métadonnées de génération
        configs["metadata"] = self._generate_metadata(performance_profile)
        
        print("✅ Génération terminée avec succès!")
        return configs
    
    def _generate_configmaps(self) -> Dict[str, Any]:
        """Génère les ConfigMaps Kubernetes."""
        configmaps = self.config_manager.generate_all_configs()
        
        # Validation des ConfigMaps
        all_valid = True
        for i, config in enumerate(configmaps):
            try:
                self.validator.validate_config(
                    list(config["data"].keys())[0].split("_")[0].lower(),
                    config["data"]
                )
            except Exception as e:
                print(f"⚠️ Avertissement: Validation échouée pour ConfigMap {i}: {e}")
                all_valid = False
        
        if all_valid:
            print("✅ Toutes les ConfigMaps sont valides")
        
        return {
            "manifests": configmaps,
            "count": len(configmaps),
            "valid": all_valid
        }
    
    def _generate_environment_profiles(self) -> Dict[str, Any]:
        """Génère les profils d'environnement."""
        profile = self.profile_manager.get_profile(self.environment)
        
        return {
            "active_profile": profile.name,
            "description": profile.description,
            "configs": {
                "application": profile.application_config,
                "database": profile.database_config,
                "security": profile.security_config,
                "ml": profile.ml_config,
                "monitoring": profile.monitoring_config
            },
            "available_profiles": self.profile_manager.list_profiles()
        }
    
    def _generate_feature_flags(self) -> Dict[str, Any]:
        """Génère les feature flags."""
        flags = self.feature_manager.list_flags()
        summary = self.feature_manager.get_flags_summary()
        config = self.feature_manager.export_to_config()
        
        return {
            "summary": summary,
            "flags": [
                {
                    "key": flag.key,
                    "name": flag.name,
                    "description": flag.description,
                    "status": flag.status.value,
                    "target_audience": flag.target_audience.value,
                    "percentage": flag.percentage,
                    "dependencies": flag.dependencies,
                    "metadata": flag.metadata
                }
                for flag in flags
            ],
            "environment_variables": config
        }
    
    def _generate_security_policies(self) -> Dict[str, Any]:
        """Génère les politiques de sécurité."""
        security_config = self.security_manager.export_security_config()
        
        # Exemples de validation
        password_validation = self.security_manager.validate_password("TestPassword123!")
        
        return {
            "policies": {
                "password": self.security_manager.get_policy("password_policy").__dict__,
                "session": self.security_manager.get_policy("session_policy").__dict__,
                "rate_limit": self.security_manager.get_policy("rate_limit_policy").__dict__,
                "encryption": self.security_manager.get_policy("encryption_policy").__dict__,
                "audit": self.security_manager.get_policy("audit_policy").__dict__
            },
            "environment_variables": security_config,
            "compliance": {
                standard.name: requirements
                for standard, requirements in self.security_manager.compliance_requirements.items()
            },
            "validation_example": password_validation
        }
    
    def _generate_performance_configs(self) -> Dict[str, Any]:
        """Génère les configurations de performance."""
        env_vars = self.performance_manager.export_to_env_vars()
        k8s_resources = self.performance_manager.get_kubernetes_resources()
        recommendations = self.performance_manager.get_performance_recommendations()
        
        return {
            "profile": self.performance_manager.profile.value,
            "environment": self.performance_manager.environment,
            "configurations": {
                "cache": self.performance_manager.get_configuration("cache").__dict__,
                "database": self.performance_manager.get_configuration("database").__dict__,
                "webserver": self.performance_manager.get_configuration("webserver").__dict__,
                "ml": self.performance_manager.get_configuration("ml").__dict__,
                "network": self.performance_manager.get_configuration("network").__dict__,
                "resources": self.performance_manager.get_configuration("resources").__dict__
            },
            "environment_variables": env_vars,
            "kubernetes_resources": k8s_resources,
            "recommendations": recommendations
        }
    
    def _generate_env_vars(self) -> Dict[str, str]:
        """Génère toutes les variables d'environnement."""
        all_env_vars = {}
        
        # Variables des feature flags
        all_env_vars.update(self.feature_manager.export_to_config())
        
        # Variables de sécurité
        all_env_vars.update(self.security_manager.export_security_config())
        
        # Variables de performance
        all_env_vars.update(self.performance_manager.export_to_env_vars())
        
        # Variables du profil d'environnement
        profile = self.profile_manager.get_profile(self.environment)
        for config_type in ["application", "database", "security", "ml", "monitoring"]:
            config_data = getattr(profile, f"{config_type}_config")
            for key, value in config_data.items():
                all_env_vars[f"{config_type.upper()}_{key}"] = str(value)
        
        return all_env_vars
    
    def _generate_secrets_templates(self) -> Dict[str, Any]:
        """Génère les templates de secrets."""
        return {
            "note": "Les valeurs ci-dessous sont des exemples. Utilisez des valeurs sécurisées en production.",
            "database_secrets": {
                "DB_PASSWORD": "<generate-secure-password>",
                "DB_READ_PASSWORD": "<generate-secure-password>",
                "REDIS_PASSWORD": "<generate-secure-password>",
                "MONGO_PASSWORD": "<generate-secure-password>",
                "ELASTICSEARCH_PASSWORD": "<generate-secure-password>"
            },
            "auth_secrets": {
                "JWT_SECRET_KEY": "<generate-256-bit-key>",
                "OAUTH_CLIENT_SECRETS": {
                    "GOOGLE_CLIENT_SECRET": "<google-oauth-secret>",
                    "SPOTIFY_CLIENT_SECRET": "<spotify-oauth-secret>",
                    "GITHUB_CLIENT_SECRET": "<github-oauth-secret>"
                },
                "ENCRYPTION_KEY": "<generate-encryption-key>",
                "SESSION_SECRET": "<generate-session-secret>"
            },
            "external_services": {
                "SPOTIFY_API_KEY": "<spotify-api-key>",
                "SENDGRID_API_KEY": "<sendgrid-api-key>",
                "TWILIO_AUTH_TOKEN": "<twilio-auth-token>",
                "STRIPE_SECRET_KEY": "<stripe-secret-key>"
            },
            "monitoring_secrets": {
                "GRAFANA_ADMIN_PASSWORD": "<grafana-admin-password>",
                "PROMETHEUS_PASSWORD": "<prometheus-password>",
                "ALERTMANAGER_WEBHOOK_SECRET": "<webhook-secret>"
            }
        }
    
    def _generate_metadata(self, performance_profile: str) -> Dict[str, Any]:
        """Génère les métadonnées de génération."""
        return {
            "generated_at": datetime.now().isoformat(),
            "generator_version": "1.0.0",
            "environment": self.environment,
            "namespace": self.namespace,
            "performance_profile": performance_profile,
            "author": "Fahed Mlaiel",
            "team": "Spotify AI Agent Development Team",
            "total_config_items": len(self._generate_env_vars()),
            "feature_flags_count": len(self.feature_manager.list_flags()),
            "security_policies_count": len(self.security_manager.policies),
            "compliance_standards": list(self.security_manager.compliance_requirements.keys())
        }
    
    def save_configs(self, 
                    configs: Dict[str, Any], 
                    format_type: str = "yaml") -> List[str]:
        """Sauvegarde les configurations dans des fichiers."""
        saved_files = []
        
        if format_type.lower() == "yaml":
            # Sauvegarde en YAML
            for config_name, config_data in configs.items():
                if config_name == "metadata":
                    continue
                
                file_path = self.output_dir / f"{config_name}.yaml"
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                saved_files.append(str(file_path))
        
        elif format_type.lower() == "json":
            # Sauvegarde en JSON
            for config_name, config_data in configs.items():
                if config_name == "metadata":
                    continue
                
                file_path = self.output_dir / f"{config_name}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False, 
                             default=str)
                saved_files.append(str(file_path))
        
        # Sauvegarde du fichier .env
        env_file_path = self.output_dir / ".env"
        with open(env_file_path, 'w', encoding='utf-8') as f:
            f.write("# Configuration générée automatiquement\n")
            f.write(f"# Générée le: {datetime.now().isoformat()}\n")
            f.write(f"# Environnement: {self.environment}\n")
            f.write(f"# Namespace: {self.namespace}\n\n")
            
            for key, value in configs.get("environment_variables", {}).items():
                f.write(f"{key}={value}\n")
        saved_files.append(str(env_file_path))
        
        # Sauvegarde des métadonnées
        metadata_path = self.output_dir / "generation_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(configs.get("metadata", {}), f, indent=2, 
                     ensure_ascii=False, default=str)
        saved_files.append(str(metadata_path))
        
        return saved_files
    
    def generate_docker_compose_override(self, configs: Dict[str, Any]) -> str:
        """Génère un fichier docker-compose.override.yml."""
        env_vars = configs.get("environment_variables", {})
        
        override_content = {
            "version": "3.8",
            "services": {
                "spotify-ai-agent": {
                    "environment": env_vars
                }
            }
        }
        
        file_path = self.output_dir / "docker-compose.override.yml"
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(override_content, f, default_flow_style=False)
        
        return str(file_path)

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Générateur de configurations Spotify AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python generate_configs.py --environment development
  python generate_configs.py --environment production --performance high_throughput
  python generate_configs.py --output ./configs/ --format json --include-secrets
        """
    )
    
    parser.add_argument(
        "--environment", "-e",
        choices=["local", "development", "staging", "production"],
        default="development",
        help="Environnement cible (défaut: development)"
    )
    
    parser.add_argument(
        "--namespace", "-n",
        default="spotify-ai-agent-dev",
        help="Namespace Kubernetes (défaut: spotify-ai-agent-dev)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="./generated",
        help="Répertoire de sortie (défaut: ./generated)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["yaml", "json"],
        default="yaml",
        help="Format de sortie (défaut: yaml)"
    )
    
    parser.add_argument(
        "--performance", "-p",
        choices=["low_latency", "high_throughput", "balanced", 
                "memory_optimized", "cpu_optimized", "cost_optimized"],
        default="balanced",
        help="Profil de performance (défaut: balanced)"
    )
    
    parser.add_argument(
        "--include-secrets",
        action="store_true",
        help="Inclure les templates de secrets"
    )
    
    parser.add_argument(
        "--docker-compose",
        action="store_true",
        help="Générer un fichier docker-compose.override.yml"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Valider uniquement sans générer de fichiers"
    )
    
    args = parser.parse_args()
    
    try:
        # Création du générateur
        generator = ConfigurationGenerator(
            environment=args.environment,
            namespace=args.namespace,
            output_dir=args.output
        )
        
        # Génération des configurations
        configs = generator.generate_all_configs(
            performance_profile=args.performance,
            include_secrets=args.include_secrets
        )
        
        if args.validate_only:
            print("✅ Validation terminée avec succès!")
            return
        
        # Sauvegarde des fichiers
        saved_files = generator.save_configs(configs, args.format)
        
        # Génération du docker-compose override
        if args.docker_compose:
            override_file = generator.generate_docker_compose_override(configs)
            saved_files.append(override_file)
        
        # Rapport de génération
        print("\n🎉 Génération terminée avec succès!")
        print(f"📁 Répertoire de sortie: {args.output}")
        print(f"📄 Fichiers générés: {len(saved_files)}")
        print("\n📋 Fichiers créés:")
        for file_path in saved_files:
            print(f"  • {file_path}")
        
        # Résumé des configurations
        metadata = configs.get("metadata", {})
        print(f"\n📊 Résumé:")
        print(f"  • Variables d'environnement: {metadata.get('total_config_items', 0)}")
        print(f"  • Feature flags: {metadata.get('feature_flags_count', 0)}")
        print(f"  • Politiques de sécurité: {metadata.get('security_policies_count', 0)}")
        print(f"  • Standards de conformité: {len(metadata.get('compliance_standards', []))}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
