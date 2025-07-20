"""
Script d'Exemple d'Utilisation - Spotify AI Agent Localization
============================================================

Démontre l'utilisation complète du système de localisation des alertes
avec exemples pratiques pour chaque fonctionnalité du module.

Author: Fahed Mlaiel
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# Import des modules de localisation
from . import LocaleType, locale_manager
from .localization_manager import alert_localizer, AlertSeverity, AlertCategory
from .format_handlers import number_formatter, datetime_formatter
from .currency_converter import currency_converter, CurrencyCode
from .cache_manager import cache_manager, cached
from .data_validators import data_validator, ValidationLevel
from .exceptions import *


class LocalizationDemo:
    """Classe de démonstration du système de localisation"""
    
    def __init__(self):
        self.demo_tenant_id = "spotify-artist-demo-001"
        
    async def run_complete_demo(self):
        """Exécute une démonstration complète"""
        print("🎵 SPOTIFY AI AGENT - DEMONSTRATION SYSTÈME DE LOCALISATION")
        print("=" * 80)
        
        await self.demo_basic_localization()
        await self.demo_alert_generation()
        await self.demo_number_formatting()
        await self.demo_currency_conversion()
        await self.demo_datetime_formatting()
        await self.demo_data_validation()
        await self.demo_cache_operations()
        await self.demo_error_handling()
        
        print("\n✅ Démonstration terminée avec succès!")
    
    async def demo_basic_localization(self):
        """Démontre la localisation de base"""
        print("\n📍 1. LOCALISATION DE BASE")
        print("-" * 40)
        
        # Configuration des locales
        locales_to_test = [LocaleType.EN_US, LocaleType.FR_FR, LocaleType.DE_DE]
        
        for locale in locales_to_test:
            locale_manager.set_current_locale(locale)
            
            # Formatage d'un nombre
            number = 1234567.89
            formatted = locale_manager.format_number(number)
            
            print(f"{locale.value}: {number} → {formatted}")
    
    async def demo_alert_generation(self):
        """Démontre la génération d'alertes localisées"""
        print("\n🚨 2. GÉNÉRATION D'ALERTES LOCALISÉES")
        print("-" * 40)
        
        # Initialise le localisateur
        await alert_localizer.initialize()
        
        # Paramètres d'alerte exemple
        alert_params = {
            "cpu_usage": 87.5,
            "threshold": 80.0,
            "tenant_id": self.demo_tenant_id,
            "instance_id": "api-server-001"
        }
        
        # Génère l'alerte dans différentes langues
        locales = [LocaleType.EN_US, LocaleType.FR_FR, LocaleType.DE_DE]
        
        for locale in locales:
            try:
                alert = await alert_localizer.generate_alert(
                    alert_type="cpu_high",
                    locale=locale,
                    tenant_id=self.demo_tenant_id,
                    parameters=alert_params
                )
                
                print(f"\n{locale.value}:")
                print(f"  Titre: {alert.title}")
                print(f"  Message: {alert.message}")
                print(f"  Action: {alert.action}")
                
            except Exception as e:
                print(f"Erreur pour {locale.value}: {e}")
    
    async def demo_number_formatting(self):
        """Démontre le formatage de nombres"""
        print("\n🔢 3. FORMATAGE DE NOMBRES")
        print("-" * 40)
        
        test_numbers = [1234.56, 1234567.89, 0.00123, 99.9]
        locales = [LocaleType.EN_US, LocaleType.FR_FR, LocaleType.DE_DE]
        
        for number in test_numbers:
            print(f"\nNombre: {number}")
            for locale in locales:
                formatted = number_formatter.format_number(number, locale)
                percentage = number_formatter.format_percentage(number, locale)
                bytes_size = number_formatter.format_bytes(int(number * 1000000), locale)
                
                print(f"  {locale.value}: {formatted} | {percentage} | {bytes_size}")
    
    async def demo_currency_conversion(self):
        """Démontre la conversion de devises"""
        print("\n💱 4. CONVERSION DE DEVISES")
        print("-" * 40)
        
        async with currency_converter as converter:
            try:
                # Conversions d'exemple
                conversions = [
                    (100.0, CurrencyCode.USD, CurrencyCode.EUR),
                    (1000.0, CurrencyCode.EUR, CurrencyCode.USD),
                    (50.0, CurrencyCode.GBP, CurrencyCode.USD)
                ]
                
                for amount, from_curr, to_curr in conversions:
                    try:
                        converted = await converter.convert(amount, from_curr, to_curr)
                        
                        # Formate selon différentes locales
                        for locale in [LocaleType.EN_US, LocaleType.FR_FR, LocaleType.DE_DE]:
                            original_formatted = await converter.format_currency(amount, from_curr, locale)
                            converted_formatted = await converter.format_currency(converted, to_curr, locale)
                            
                            print(f"{locale.value}: {original_formatted} → {converted_formatted}")
                        
                        print()
                        
                    except Exception as e:
                        print(f"Erreur de conversion {from_curr.value}→{to_curr.value}: {e}")
                        
            except Exception as e:
                print(f"Erreur d'initialisation du convertisseur: {e}")
    
    async def demo_datetime_formatting(self):
        """Démontre le formatage de dates et heures"""
        print("\n📅 5. FORMATAGE DATE/HEURE")
        print("-" * 40)
        
        now = datetime.now(timezone.utc)
        past_time = now - timedelta(hours=2, minutes=30)
        
        locales = [LocaleType.EN_US, LocaleType.FR_FR, LocaleType.DE_DE]
        
        for locale in locales:
            print(f"\n{locale.value}:")
            
            # Formatage de date/heure
            formatted_now = datetime_formatter.format_datetime(now, locale, 'datetime')
            formatted_date = datetime_formatter.format_datetime(now, locale, 'short_date')
            formatted_time = datetime_formatter.format_datetime(now, locale, 'short_time')
            
            print(f"  DateTime: {formatted_now}")
            print(f"  Date: {formatted_date}")
            print(f"  Time: {formatted_time}")
            
            # Temps relatif
            relative = datetime_formatter.format_relative_time(past_time, locale)
            print(f"  Relatif: {relative}")
    
    async def demo_data_validation(self):
        """Démontre la validation de données"""
        print("\n🔍 6. VALIDATION DE DONNÉES")
        print("-" * 40)
        
        # Test de validation de locale
        test_locales = ["en_US", "fr_FR", "invalid_locale", "de_DE"]
        
        for locale_code in test_locales:
            result = data_validator.validate_locale_code(locale_code)
            status = "✅" if result.is_valid else "❌"
            print(f"{status} Locale '{locale_code}': {result.errors if result.errors else 'Valide'}")
        
        # Test de validation de nombres
        test_numbers = ["1,234.56", "1 234,56", "1.234,56", "invalid", "-123.45"]
        
        print("\nValidation de nombres:")
        for number_str in test_numbers:
            result = data_validator.validate_number(number_str)
            status = "✅" if result.is_valid else "❌"
            cleaned = result.cleaned_value if result.is_valid else "N/A"
            print(f"{status} '{number_str}' → {cleaned}")
        
        # Test de validation de template
        template = "CPU usage: {cpu_usage}% on tenant '{tenant_id}'"
        result = data_validator.validate_template_string(template)
        status = "✅" if result.is_valid else "❌"
        print(f"\n{status} Template: {template}")
    
    async def demo_cache_operations(self):
        """Démontre les opérations de cache"""
        print("\n🗄️ 7. OPÉRATIONS DE CACHE")
        print("-" * 40)
        
        # Test des opérations de cache de base
        test_key = "demo_cache_key"
        test_value = {"message": "Hello from cache", "timestamp": datetime.now().isoformat()}
        
        # Set
        success = await cache_manager.set(test_key, test_value, ttl=timedelta(minutes=5))
        print(f"Cache SET: {'✅' if success else '❌'}")
        
        # Get
        cached_value = await cache_manager.get(test_key)
        print(f"Cache GET: {'✅' if cached_value else '❌'} - {cached_value}")
        
        # Métriques
        metrics = cache_manager.get_metrics()
        print(f"Métriques: Hits={metrics.hits}, Misses={metrics.misses}, Ratio={metrics.hit_ratio:.2%}")
        
        # Test du décorateur de cache
        @cached(ttl=timedelta(seconds=30))
        async def expensive_computation(value: int) -> int:
            """Fonction coûteuse mise en cache"""
            await asyncio.sleep(0.1)  # Simule du travail
            return value * value
        
        # Premier appel (calcul)
        start_time = asyncio.get_event_loop().time()
        result1 = await expensive_computation(42)
        duration1 = asyncio.get_event_loop().time() - start_time
        
        # Deuxième appel (cache)
        start_time = asyncio.get_event_loop().time()
        result2 = await expensive_computation(42)
        duration2 = asyncio.get_event_loop().time() - start_time
        
        print(f"Calcul: {result1} en {duration1*1000:.1f}ms")
        print(f"Cache: {result2} en {duration2*1000:.1f}ms")
    
    async def demo_error_handling(self):
        """Démontre la gestion d'erreurs"""
        print("\n⚠️ 8. GESTION D'ERREURS")
        print("-" * 40)
        
        # Test d'erreurs de validation
        try:
            invalid_result = data_validator.validate_locale_code("invalid")
            if not invalid_result.is_valid:
                raise LocaleValidationError(
                    "Locale invalide détectée",
                    invalid_locale="invalid",
                    supported_locales=["en_US", "fr_FR", "de_DE"]
                )
        except LocaleValidationError as e:
            print(f"❌ {e.category.value}: {e.message}")
            print(f"   Contexte: {e.context}")
        
        # Test d'erreur de formatage
        try:
            raise NumberFormatError(
                "Impossible de formater le nombre",
                value="not_a_number",
                locale=LocaleType.FR_FR
            )
        except NumberFormatError as e:
            print(f"❌ {e.category.value}: {e.message}")
            print(f"   Sévérité: {e.severity.value}")
        
        # Test de gestion automatique d'exception
        try:
            @handle_localization_exceptions("demo_operation")
            def problematic_function():
                raise ValueError("Erreur de démonstration")
            
            problematic_function()
            
        except LocalizationBaseException as e:
            print(f"❌ Exception gérée: {e.message}")
            print(f"   Catégorie: {e.category.value}")


async def main():
    """Fonction principale de démonstration"""
    demo = LocalizationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
