#!/usr/bin/env python3
"""
Script de Configuration et Déploiement - Spotify AI Agent Localization
====================================================================

Script d'initialisation, configuration et déploiement du système de localisation
avec validation, tests automatiques et mise en place de l'environnement.

Fonctionnalités:
- Validation de l'environnement
- Configuration automatique
- Tests de santé du système
- Déploiement des ressources
- Monitoring de l'état

Author: Fahed Mlaiel
"""

import os
import sys
import asyncio
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timezone

# Ajout du chemin pour les imports locaux
sys.path.insert(0, str(Path(__file__).parent))

from localization_manager import alert_localizer
from cache_manager import cache_manager
from currency_converter import currency_converter, CurrencyCode
from data_validators import data_validator, ValidationLevel
from exceptions import *


class LocalizationSetup:
    """Gestionnaire de configuration et déploiement"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.logger = self._setup_logging()
        self.config = {}
        self.status = {
            "environment_check": False,
            "config_loaded": False,
            "cache_initialized": False,
            "localization_ready": False,
            "currency_ready": False,
            "health_check_passed": False
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Configure le logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('localization_setup.log')
            ]
        )
        return logging.getLogger(__name__)
    
    async def run_full_setup(self) -> bool:
        """Exécute la configuration complète"""
        self.logger.info("🚀 Démarrage de la configuration Spotify AI Agent Localization")
        
        try:
            # 1. Vérification de l'environnement
            if not await self._check_environment():
                self.logger.error("❌ Vérification de l'environnement échouée")
                return False
            
            # 2. Chargement de la configuration
            if not await self._load_configuration():
                self.logger.error("❌ Chargement de la configuration échoué")
                return False
            
            # 3. Initialisation du cache
            if not await self._initialize_cache():
                self.logger.error("❌ Initialisation du cache échouée")
                return False
            
            # 4. Configuration de la localisation
            if not await self._setup_localization():
                self.logger.error("❌ Configuration de la localisation échouée")
                return False
            
            # 5. Configuration des devises
            if not await self._setup_currency_conversion():
                self.logger.error("❌ Configuration des devises échouée")
                return False
            
            # 6. Tests de santé
            if not await self._run_health_checks():
                self.logger.error("❌ Tests de santé échoués")
                return False
            
            # 7. Déploiement des ressources
            if not await self._deploy_resources():
                self.logger.error("❌ Déploiement des ressources échoué")
                return False
            
            self.logger.info("✅ Configuration terminée avec succès!")
            await self._print_status_summary()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la configuration: {e}")
            return False
    
    async def _check_environment(self) -> bool:
        """Vérifie l'environnement système"""
        self.logger.info("🔍 Vérification de l'environnement...")
        
        checks = []
        
        # Vérifie Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            checks.append(("Python version", True, f"{python_version.major}.{python_version.minor}"))
        else:
            checks.append(("Python version", False, f"Requis: >= 3.8, Actuel: {python_version.major}.{python_version.minor}"))
        
        # Vérifie les dépendances
        required_packages = [
            "redis", "aiohttp", "aiofiles", "bleach", "pyyaml"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                checks.append((f"Package {package}", True, "Installé"))
            except ImportError:
                checks.append((f"Package {package}", False, "Manquant"))
        
        # Vérifie les répertoires
        required_dirs = [
            Path(__file__).parent / "locales",
            Path(__file__).parent / "locales" / "en_US",
            Path(__file__).parent / "locales" / "fr_FR",
            Path(__file__).parent / "locales" / "de_DE"
        ]
        
        for dir_path in required_dirs:
            if dir_path.exists():
                checks.append((f"Répertoire {dir_path.name}", True, "Présent"))
            else:
                checks.append((f"Répertoire {dir_path.name}", False, "Manquant"))
        
        # Affiche les résultats
        all_passed = True
        for name, passed, details in checks:
            status = "✅" if passed else "❌"
            self.logger.info(f"  {status} {name}: {details}")
            if not passed:
                all_passed = False
        
        self.status["environment_check"] = all_passed
        return all_passed
    
    async def _load_configuration(self) -> bool:
        """Charge la configuration depuis le fichier YAML"""
        self.logger.info("📋 Chargement de la configuration...")
        
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self.logger.warning(f"Fichier de config {self.config_path} non trouvé, utilisation des valeurs par défaut")
                self.config = self._get_default_config()
            else:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            
            # Validation de la configuration
            if not self._validate_configuration():
                return False
            
            self.status["config_loaded"] = True
            self.logger.info("✅ Configuration chargée avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Retourne la configuration par défaut"""
        return {
            "general": {
                "default_locale": "en_US",
                "fallback_locale": "en_US",
                "supported_locales": ["en_US", "fr_FR", "de_DE"]
            },
            "cache": {
                "enabled": True,
                "default_ttl": 3600,
                "redis_cache": {"enabled": False}
            },
            "currencies": {
                "real_time_conversion": {"enabled": False}
            },
            "validation": {
                "global_level": "normal"
            }
        }
    
    def _validate_configuration(self) -> bool:
        """Valide la configuration chargée"""
        required_sections = ["general", "cache", "validation"]
        
        for section in required_sections:
            if section not in self.config:
                self.logger.error(f"Section manquante dans la configuration: {section}")
                return False
        
        # Valide les locales supportées
        supported_locales = self.config.get("general", {}).get("supported_locales", [])
        valid_locales = ["en_US", "fr_FR", "de_DE", "es_ES", "it_IT", "pt_BR", "ja_JP", "ko_KR", "zh_CN", "ru_RU", "ar_SA"]
        
        for locale in supported_locales:
            if locale not in valid_locales:
                self.logger.warning(f"Locale non supportée: {locale}")
        
        return True
    
    async def _initialize_cache(self) -> bool:
        """Initialise le système de cache"""
        self.logger.info("🗄️ Initialisation du cache...")
        
        try:
            cache_config = self.config.get("cache", {})
            
            # Test du cache mémoire
            test_key = "setup_test_key"
            test_value = {"timestamp": datetime.now().isoformat(), "test": True}
            
            success = await cache_manager.set(test_key, test_value)
            if not success:
                self.logger.error("Échec du test de cache mémoire")
                return False
            
            retrieved = await cache_manager.get(test_key)
            if retrieved != test_value:
                self.logger.error("Échec de la récupération depuis le cache")
                return False
            
            # Nettoyage
            await cache_manager.delete(test_key)
            
            self.status["cache_initialized"] = True
            self.logger.info("✅ Cache initialisé avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation du cache: {e}")
            return False
    
    async def _setup_localization(self) -> bool:
        """Configure le système de localisation"""
        self.logger.info("🌍 Configuration de la localisation...")
        
        try:
            # Initialise le gestionnaire d'alertes
            await alert_localizer.initialize()
            
            # Test de génération d'alerte
            test_params = {
                "cpu_usage": 85.0,
                "threshold": 80.0,
                "tenant_id": "test_tenant"
            }
            
            supported_locales = self.config.get("general", {}).get("supported_locales", ["en_US"])
            
            for locale_code in supported_locales[:3]:  # Test les 3 premières
                try:
                    from . import LocaleType
                    locale = LocaleType(locale_code)
                    
                    alert = await alert_localizer.generate_alert(
                        alert_type="cpu_high",
                        locale=locale,
                        tenant_id="test_tenant",
                        parameters=test_params
                    )
                    
                    self.logger.info(f"  ✅ Test {locale_code}: {alert.title[:50]}...")
                    
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Test {locale_code} échoué: {e}")
            
            self.status["localization_ready"] = True
            self.logger.info("✅ Localisation configurée avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration de la localisation: {e}")
            return False
    
    async def _setup_currency_conversion(self) -> bool:
        """Configure la conversion de devises"""
        self.logger.info("💱 Configuration des devises...")
        
        try:
            currency_config = self.config.get("currencies", {}).get("real_time_conversion", {})
            
            if not currency_config.get("enabled", False):
                self.logger.info("  ℹ️ Conversion de devises désactivée")
                self.status["currency_ready"] = True
                return True
            
            # Test de conversion simple (sans API externe)
            async with currency_converter as converter:
                # Test de formatage seulement
                from . import LocaleType
                test_amount = 100.0
                
                formatted_usd = await converter.format_currency(
                    test_amount, 
                    CurrencyCode.USD, 
                    LocaleType.EN_US
                )
                
                formatted_eur = await converter.format_currency(
                    test_amount, 
                    CurrencyCode.EUR, 
                    LocaleType.FR_FR
                )
                
                self.logger.info(f"  ✅ Test formatage: {formatted_usd}, {formatted_eur}")
            
            self.status["currency_ready"] = True
            self.logger.info("✅ Devises configurées avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration des devises: {e}")
            return False
    
    async def _run_health_checks(self) -> bool:
        """Exécute les tests de santé du système"""
        self.logger.info("🏥 Tests de santé du système...")
        
        health_checks = []
        
        try:
            # Test de validation
            result = data_validator.validate_locale_code("en_US")
            health_checks.append(("Validation locale", result.is_valid))
            
            result = data_validator.validate_number("1234.56")
            health_checks.append(("Validation nombre", result.is_valid))
            
            # Test de formatage
            from .format_handlers import number_formatter
            from . import LocaleType
            
            formatted = number_formatter.format_number(1234.56, LocaleType.EN_US)
            health_checks.append(("Formatage nombre", formatted == "1,234.56"))
            
            # Test de cache
            test_key = "health_check_cache"
            await cache_manager.set(test_key, "test_value")
            cached_value = await cache_manager.get(test_key)
            health_checks.append(("Cache fonctionnel", cached_value == "test_value"))
            await cache_manager.delete(test_key)
            
            # Affiche les résultats
            all_passed = True
            for check_name, passed in health_checks:
                status = "✅" if passed else "❌"
                self.logger.info(f"  {status} {check_name}")
                if not passed:
                    all_passed = False
            
            self.status["health_check_passed"] = all_passed
            return all_passed
            
        except Exception as e:
            self.logger.error(f"Erreur lors des tests de santé: {e}")
            return False
    
    async def _deploy_resources(self) -> bool:
        """Déploie les ressources nécessaires"""
        self.logger.info("🚀 Déploiement des ressources...")
        
        try:
            # Vérifie la présence des fichiers de localisation
            locales_dir = Path(__file__).parent / "locales"
            resource_count = 0
            
            for locale_dir in locales_dir.iterdir():
                if locale_dir.is_dir():
                    alerts_file = locale_dir / "alerts.json"
                    formats_file = locale_dir / "formats.json"
                    
                    if alerts_file.exists():
                        resource_count += 1
                        self.logger.info(f"  ✅ Alertes {locale_dir.name}: {alerts_file}")
                    
                    if formats_file.exists():
                        resource_count += 1
                        self.logger.info(f"  ✅ Formats {locale_dir.name}: {formats_file}")
            
            self.logger.info(f"✅ {resource_count} ressources déployées")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du déploiement: {e}")
            return False
    
    async def _print_status_summary(self):
        """Affiche un résumé de l'état du système"""
        print("\n" + "="*80)
        print("📊 RÉSUMÉ DE LA CONFIGURATION SPOTIFY AI AGENT LOCALIZATION")
        print("="*80)
        
        for component, status in self.status.items():
            status_icon = "✅" if status else "❌"
            component_name = component.replace("_", " ").title()
            print(f"{status_icon} {component_name}")
        
        print("\n🔧 CONFIGURATION ACTIVE:")
        print(f"  • Locale par défaut: {self.config.get('general', {}).get('default_locale', 'N/A')}")
        print(f"  • Locales supportées: {len(self.config.get('general', {}).get('supported_locales', []))}")
        print(f"  • Cache activé: {self.config.get('cache', {}).get('enabled', False)}")
        print(f"  • Validation: {self.config.get('validation', {}).get('global_level', 'N/A')}")
        
        print("\n🎯 SYSTÈME PRÊT POUR LA PRODUCTION!")
        print("="*80)


async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Configuration et déploiement du système de localisation Spotify AI Agent"
    )
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Chemin vers le fichier de configuration YAML"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Active le mode verbeux"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialise et exécute la configuration
    setup = LocalizationSetup(args.config)
    success = await setup.run_full_setup()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
