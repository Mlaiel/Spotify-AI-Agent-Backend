#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires pour le module i18n Slack ultra-avancé

Ce module contient une suite complète de tests pour valider:
- Système de traduction avec cache Redis
- Détection intelligente de langue avec IA
- Formatage culturel adaptatif
- Gestion de configuration avancée
- Performance et monitoring
- Sécurité et conformité

Auteur: Expert Team  
Version: 2.0.0
"""

import pytest
import asyncio
import json
import yaml
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from pathlib import Path
import tempfile
import shutil

# Import des modules à tester
from . import (
    AdvancedTranslationManager,
    SmartLanguageDetector, 
    AdvancedCulturalFormatter,
    ConfigManager,
    I18nConfig,
    Environment
)


class TestAdvancedTranslationManager:
    """Tests pour le gestionnaire de traduction avancé"""
    
    @pytest.fixture
    async def translation_manager(self):
        """Fixture pour le gestionnaire de traduction"""
        config = I18nConfig()
        config.cache.enabled = False  # Désactiver Redis pour les tests
        
        manager = AdvancedTranslationManager(config)
        yield manager
        await manager.close()
    
    @pytest.fixture
    def sample_translations(self):
        """Fixture avec des traductions d'exemple"""
        return {
            "meta": {
                "version": "2.0.0",
                "supported_languages": ["en", "fr", "de", "es"]
            },
            "translations": {
                "en": {
                    "alerts": {
                        "high_cpu": "High CPU usage detected: {cpu_usage}%",
                        "memory_leak": "Memory leak detected in {service}",
                        "disk_full": "Disk space critical: {available_space} remaining"
                    },
                    "actions": {
                        "investigate": "Investigate",
                        "acknowledge": "Acknowledge", 
                        "escalate": "Escalate"
                    }
                },
                "fr": {
                    "alerts": {
                        "high_cpu": "Utilisation CPU élevée détectée: {cpu_usage}%",
                        "memory_leak": "Fuite mémoire détectée dans {service}",
                        "disk_full": "Espace disque critique: {available_space} restant"
                    },
                    "actions": {
                        "investigate": "Investiguer",
                        "acknowledge": "Acquitter",
                        "escalate": "Escalader"
                    }
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_basic_translation(self, translation_manager, sample_translations):
        """Test traduction de base"""
        # Mock des données de traduction
        with patch.object(translation_manager, '_load_translations') as mock_load:
            mock_load.return_value = sample_translations
            await translation_manager._initialize()
            
            # Test traduction simple
            result = await translation_manager.translate(
                key="alerts.high_cpu",
                language="fr",
                context={"cpu_usage": 85}
            )
            
            assert result.text == "Utilisation CPU élevée détectée: 85%"
            assert result.language == "fr"
            assert result.confidence > 0.9
    
    @pytest.mark.asyncio
    async def test_fallback_translation(self, translation_manager, sample_translations):
        """Test fallback vers langue par défaut"""
        with patch.object(translation_manager, '_load_translations') as mock_load:
            mock_load.return_value = sample_translations
            await translation_manager._initialize()
            
            # Test avec langue non supportée
            result = await translation_manager.translate(
                key="alerts.high_cpu",
                language="zh",  # Langue non supportée
                context={"cpu_usage": 85}
            )
            
            # Doit fallback vers anglais
            assert result.text == "High CPU usage detected: 85%"
            assert result.language == "en"
            assert result.fallback_used is True
    
    @pytest.mark.asyncio
    async def test_missing_key_handling(self, translation_manager, sample_translations):
        """Test gestion des clés manquantes"""
        with patch.object(translation_manager, '_load_translations') as mock_load:
            mock_load.return_value = sample_translations
            await translation_manager._initialize()
            
            result = await translation_manager.translate(
                key="alerts.nonexistent_key",
                language="fr"
            )
            
            # Doit retourner la clé avec formatage
            assert "nonexistent_key" in result.text
            assert result.confidence < 0.5
    
    @pytest.mark.asyncio
    async def test_context_interpolation(self, translation_manager, sample_translations):
        """Test interpolation des variables de contexte"""
        with patch.object(translation_manager, '_load_translations') as mock_load:
            mock_load.return_value = sample_translations
            await translation_manager._initialize()
            
            complex_context = {
                "service": "web-backend",
                "cpu_usage": 92.5,
                "timestamp": "2024-01-15T10:30:00Z",
                "severity": "critical"
            }
            
            result = await translation_manager.translate(
                key="alerts.memory_leak",
                language="en",
                context=complex_context
            )
            
            assert "web-backend" in result.text
            assert result.context_used == complex_context
    
    @pytest.mark.asyncio
    async def test_ai_enhancement(self, translation_manager, sample_translations):
        """Test amélioration IA des traductions"""
        with patch.object(translation_manager, '_load_translations') as mock_load:
            mock_load.return_value = sample_translations
            
            # Mock du service IA
            with patch.object(translation_manager, '_enhance_with_ai') as mock_ai:
                mock_ai.return_value = "Traduction améliorée par IA"
                
                await translation_manager._initialize()
                
                result = await translation_manager.translate(
                    key="alerts.high_cpu",
                    language="fr",
                    context={"cpu_usage": 85},
                    enhance_with_ai=True
                )
                
                assert result.ai_enhanced is True
                mock_ai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """Test intégration du cache Redis"""
        config = I18nConfig()
        config.cache.enabled = True
        
        # Mock Redis
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            # Configuration du cache
            mock_redis_instance.get.return_value = None  # Cache miss
            mock_redis_instance.setex.return_value = True
            
            manager = AdvancedTranslationManager(config)
            await manager._initialize()
            
            result = await manager.translate("test.key", "en")
            
            # Vérifier appels Redis
            mock_redis_instance.get.assert_called()
            mock_redis_instance.setex.assert_called()
            
            await manager.close()
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, translation_manager, sample_translations):
        """Test collecte des métriques de performance"""
        with patch.object(translation_manager, '_load_translations') as mock_load:
            mock_load.return_value = sample_translations
            await translation_manager._initialize()
            
            # Effectuer plusieurs traductions
            for i in range(10):
                await translation_manager.translate(
                    key="alerts.high_cpu",
                    language="fr",
                    context={"cpu_usage": 80 + i}
                )
            
            metrics = await translation_manager.get_metrics()
            
            assert metrics["total_requests"] == 10
            assert metrics["average_latency_ms"] > 0
            assert "cache_hit_ratio" in metrics


class TestSmartLanguageDetector:
    """Tests pour le détecteur de langue intelligent"""
    
    @pytest.fixture
    def language_detector(self):
        """Fixture pour le détecteur de langue"""
        config = I18nConfig()
        return SmartLanguageDetector(config)
    
    @pytest.mark.asyncio
    async def test_content_analysis_detection(self, language_detector):
        """Test détection par analyse de contenu"""
        texts = {
            "Hello world, this is a test message": "en",
            "Bonjour monde, ceci est un message de test": "fr", 
            "Hallo Welt, das ist eine Testnachricht": "de",
            "Hola mundo, este es un mensaje de prueba": "es"
        }
        
        for text, expected_lang in texts.items():
            result = await language_detector.detect_language(
                content=text,
                methods=["content_analysis"]
            )
            
            assert result.detected_language == expected_lang
            assert result.confidence > 0.7
    
    @pytest.mark.asyncio
    async def test_user_preference_detection(self, language_detector):
        """Test détection basée sur les préférences utilisateur"""
        user_context = {
            "user_id": "user123",
            "preferred_language": "fr",
            "browser_languages": ["fr-FR", "en-US"],
            "timezone": "Europe/Paris"
        }
        
        result = await language_detector.detect_language(
            content="Mixed content avec français",
            user_context=user_context,
            methods=["user_preference"]
        )
        
        assert result.detected_language == "fr"
        assert result.method_used == "user_preference"
        assert result.confidence > 0.8
    
    @pytest.mark.asyncio
    async def test_geographic_detection(self, language_detector):
        """Test détection géographique"""
        geo_context = {
            "country_code": "DE",
            "region": "Bayern",
            "city": "Munich",
            "ip_address": "192.168.1.1"
        }
        
        result = await language_detector.detect_language(
            content="Some text",
            geo_context=geo_context,
            methods=["geographic"]
        )
        
        assert result.detected_language == "de"
        assert result.method_used == "geographic"
    
    @pytest.mark.asyncio
    async def test_multi_method_detection(self, language_detector):
        """Test détection avec méthodes multiples"""
        content = "Hello, this is a test message"
        user_context = {"preferred_language": "fr"}
        
        result = await language_detector.detect_language(
            content=content,
            user_context=user_context,
            methods=["content_analysis", "user_preference"]
        )
        
        # Devrait privilégier content_analysis pour ce texte anglais
        assert result.detected_language == "en"
        assert len(result.method_scores) == 2
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, language_detector):
        """Test système de scoring de confiance"""
        # Texte très clair en français
        clear_french = "Bonjour, comment allez-vous aujourd'hui? J'espère que tout va bien."
        
        result = await language_detector.detect_language(
            content=clear_french,
            methods=["content_analysis"]
        )
        
        assert result.confidence > 0.9
        assert result.confidence_level == "very_high"
        
        # Texte ambigu
        ambiguous = "OK yes no"
        
        result = await language_detector.detect_language(
            content=ambiguous,
            methods=["content_analysis"]
        )
        
        assert result.confidence < 0.7
        assert result.confidence_level in ["low", "medium"]
    
    @pytest.mark.asyncio
    async def test_user_profile_learning(self, language_detector):
        """Test apprentissage du profil utilisateur"""
        user_id = "user123"
        
        # Simuler plusieurs détections pour un utilisateur
        detections = [
            ("Bonjour", "fr"),
            ("Hello", "en"), 
            ("Guten Tag", "de"),
            ("Bonjour encore", "fr"),
            ("Salut", "fr")
        ]
        
        for content, expected in detections:
            await language_detector.detect_language(
                content=content,
                user_context={"user_id": user_id}
            )
        
        # Vérifier le profil utilisateur
        profile = await language_detector.get_user_profile(user_id)
        
        assert profile is not None
        assert profile["most_used_language"] == "fr"  # Plus fréquent
        assert len(profile["detection_history"]) == 5


class TestAdvancedCulturalFormatter:
    """Tests pour le formateur culturel avancé"""
    
    @pytest.fixture
    def cultural_formatter(self):
        """Fixture pour le formateur culturel"""
        config = I18nConfig()
        return AdvancedCulturalFormatter(config)
    
    def test_datetime_formatting(self, cultural_formatter):
        """Test formatage des dates selon la culture"""
        from datetime import datetime
        
        dt = datetime(2024, 1, 15, 14, 30, 0)
        
        # Format français
        fr_formatted = cultural_formatter.format_datetime(dt, "fr")
        assert "15/01/2024" in fr_formatted or "15 janvier 2024" in fr_formatted
        
        # Format américain
        en_formatted = cultural_formatter.format_datetime(dt, "en")
        assert "01/15/2024" in en_formatted or "January 15, 2024" in en_formatted
        
        # Format allemand
        de_formatted = cultural_formatter.format_datetime(dt, "de")
        assert "15.01.2024" in de_formatted or "15. Januar 2024" in de_formatted
    
    def test_number_formatting(self, cultural_formatter):
        """Test formatage des nombres selon la culture"""
        number = 1234567.89
        
        # Format français (espace comme séparateur de milliers, virgule décimale)
        fr_formatted = cultural_formatter.format_number(number, "fr")
        assert "1 234 567,89" in fr_formatted
        
        # Format américain (virgule comme séparateur de milliers, point décimal)
        en_formatted = cultural_formatter.format_number(number, "en")
        assert "1,234,567.89" in en_formatted
        
        # Format allemand (point comme séparateur de milliers, virgule décimale)
        de_formatted = cultural_formatter.format_number(number, "de")
        assert "1.234.567,89" in de_formatted
    
    def test_currency_formatting(self, cultural_formatter):
        """Test formatage des devises"""
        amount = 1234.56
        
        # Euro français
        fr_eur = cultural_formatter.format_currency(amount, "EUR", "fr")
        assert "€" in fr_eur
        assert "1 234,56" in fr_eur
        
        # Dollar américain
        en_usd = cultural_formatter.format_currency(amount, "USD", "en")
        assert "$" in en_usd
        assert "1,234.56" in en_usd
    
    def test_address_formatting(self, cultural_formatter):
        """Test formatage des adresses"""
        address = {
            "street": "123 Main Street",
            "city": "Paris",
            "postal_code": "75001",
            "country": "France"
        }
        
        # Format français
        fr_address = cultural_formatter.format_address(address, "fr")
        assert "75001 Paris" in fr_address
        
        # Format américain
        address_us = {
            "street": "123 Main St",
            "city": "New York",
            "state": "NY",
            "postal_code": "10001",
            "country": "USA"
        }
        
        en_address = cultural_formatter.format_address(address_us, "en")
        assert "New York, NY 10001" in en_address
    
    def test_rtl_support(self, cultural_formatter):
        """Test support des langues RTL (droite vers gauche)"""
        text = "Hello world"
        
        # Texte arabe (RTL)
        ar_formatted = cultural_formatter.format_text(text, "ar")
        assert ar_formatted["direction"] == "rtl"
        assert ar_formatted["text_align"] == "right"
        
        # Texte hébreu (RTL)
        he_formatted = cultural_formatter.format_text(text, "he")
        assert he_formatted["direction"] == "rtl"
        
        # Texte français (LTR)
        fr_formatted = cultural_formatter.format_text(text, "fr")
        assert fr_formatted["direction"] == "ltr"
        assert fr_formatted["text_align"] == "left"
    
    def test_emoji_adaptation(self, cultural_formatter):
        """Test adaptation des emojis selon la culture"""
        # Emoji de salutation
        greeting_emoji = cultural_formatter.get_cultural_emoji("greeting", "ja")
        assert greeting_emoji in ["🙏", "🙇‍♂️", "🙇‍♀️"]  # Emojis japonais typiques
        
        # Emoji de succès
        success_emoji = cultural_formatter.get_cultural_emoji("success", "en")
        assert success_emoji in ["✅", "👍", "🎉"]
        
        # Emoji d'erreur
        error_emoji = cultural_formatter.get_cultural_emoji("error", "fr")
        assert error_emoji in ["❌", "⚠️", "🚫"]
    
    def test_color_adaptation(self, cultural_formatter):
        """Test adaptation des couleurs selon la culture"""
        # Couleur de succès
        success_colors = cultural_formatter.get_cultural_colors("success", "zh")
        assert "red" in success_colors  # Rouge = chance en Chine
        
        # Couleur d'erreur en occidental
        error_colors = cultural_formatter.get_cultural_colors("error", "en")
        assert "red" in error_colors
        
        # Couleur de danger en culturel
        warning_colors = cultural_formatter.get_cultural_colors("warning", "ja")
        assert "yellow" in warning_colors or "orange" in warning_colors


class TestConfigManager:
    """Tests pour le gestionnaire de configuration"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Répertoire temporaire pour les tests"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_environment_detection(self):
        """Test détection automatique de l'environnement"""
        # Test avec variable d'environnement
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = ConfigManager()
            assert manager.environment == Environment.PRODUCTION
        
        with patch.dict(os.environ, {"ENV": "staging"}):
            manager = ConfigManager()
            assert manager.environment == Environment.STAGING
        
        # Test sans variable (défaut)
        with patch.dict(os.environ, {}, clear=True):
            manager = ConfigManager()
            assert manager.environment == Environment.DEVELOPMENT
    
    def test_config_file_loading(self, temp_config_dir):
        """Test chargement depuis fichier YAML"""
        config_file = temp_config_dir / "config.yaml"
        config_data = {
            "debug": True,
            "redis": {
                "host": "redis.example.com",
                "port": 6380
            },
            "ai": {
                "enabled": True,
                "model": "gpt-4"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigManager(config_path=str(config_file))
        config = manager.load_config()
        
        assert config.debug is True
        assert config.redis.host == "redis.example.com"
        assert config.redis.port == 6380
        assert config.ai.model == "gpt-4"
    
    def test_environment_variables_override(self):
        """Test override par variables d'environnement"""
        env_vars = {
            "REDIS_HOST": "env-redis.com",
            "REDIS_PORT": "7000",
            "AI_ENABLED": "false",
            "CACHE_DEFAULT_TTL": "7200",
            "DEBUG": "true"
        }
        
        with patch.dict(os.environ, env_vars):
            manager = ConfigManager()
            config = manager.load_config()
            
            assert config.redis.host == "env-redis.com"
            assert config.redis.port == 7000
            assert config.ai.enabled is False
            assert config.cache.default_ttl == 7200
            assert config.debug is True
    
    def test_configuration_validation(self):
        """Test validation de la configuration"""
        manager = ConfigManager()
        config = manager.load_config()
        
        # Configuration invalide
        config.redis.port = -1
        config.cache.default_ttl = 0
        config.ai.enabled = True
        config.ai.api_key = None
        
        # La validation doit corriger
        manager._validate_config(config)
        
        assert config.redis.port == 6379  # Corrigé
        assert config.cache.default_ttl == 3600  # Corrigé
        assert config.ai.enabled is False  # Désactivé car pas de clé
    
    def test_environment_specific_defaults(self):
        """Test valeurs par défaut selon l'environnement"""
        # Development
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            manager = ConfigManager()
            config = manager.load_config()
            
            assert config.debug is True
            assert config.log_level.value == "DEBUG"
            assert config.ai.enabled is False
        
        # Production
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = ConfigManager()
            config = manager.load_config()
            
            assert config.debug is False
            assert config.log_level.value == "WARNING"
            assert config.performance_mode == "fast"
            assert config.max_concurrent_requests == 500
    
    def test_config_export(self):
        """Test export de configuration"""
        manager = ConfigManager()
        config = manager.load_config()
        
        # Export YAML
        yaml_export = manager.export_config("yaml")
        assert "redis:" in yaml_export
        assert "cache:" in yaml_export
        
        # Export JSON
        json_export = manager.export_config("json")
        json_data = json.loads(json_export)
        assert "redis" in json_data
        assert "cache" in json_data
    
    def test_environment_validation_report(self):
        """Test rapport de validation d'environnement"""
        manager = ConfigManager()
        
        # Mock Redis pour le test
        with patch('redis.Redis') as mock_redis:
            mock_redis.return_value.ping.return_value = True
            
            report = manager.validate_environment()
            
            assert report["environment"] == "development"
            assert any("Redis OK" in check for check in report["checks"])
    
    def test_config_watcher(self):
        """Test système de watchers"""
        manager = ConfigManager()
        
        # Mock du callback
        callback = Mock()
        manager.add_config_watcher(callback)
        
        # Rechargement de config
        config = manager.reload_config()
        
        # Vérifier appel du callback
        callback.assert_called_once_with(config)


class TestIntegration:
    """Tests d'intégration du système complet"""
    
    @pytest.fixture
    async def full_system(self):
        """Fixture pour le système complet"""
        config = I18nConfig()
        config.cache.enabled = False  # Simplifier pour les tests
        config.ai.enabled = False
        
        translation_manager = AdvancedTranslationManager(config)
        language_detector = SmartLanguageDetector(config)
        cultural_formatter = AdvancedCulturalFormatter(config)
        
        await translation_manager._initialize()
        
        yield {
            "config": config,
            "translation": translation_manager,
            "detection": language_detector,
            "formatting": cultural_formatter
        }
        
        await translation_manager.close()
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, full_system):
        """Test workflow complet: détection → traduction → formatage"""
        
        # 1. Détection de langue
        detection_result = await full_system["detection"].detect_language(
            content="Alerte: CPU élevé détecté",
            user_context={"user_id": "user123"}
        )
        
        detected_lang = detection_result.detected_language
        assert detected_lang == "fr"
        
        # 2. Traduction
        translation_result = await full_system["translation"].translate(
            key="alerts.high_cpu",
            language=detected_lang,
            context={"cpu_usage": 85}
        )
        
        assert "85" in translation_result.text
        
        # 3. Formatage culturel
        formatted_result = full_system["formatting"].format_text(
            translation_result.text,
            detected_lang
        )
        
        assert formatted_result["direction"] == "ltr"
        assert formatted_result["language"] == detected_lang
    
    @pytest.mark.asyncio
    async def test_error_handling_chain(self, full_system):
        """Test gestion d'erreurs dans la chaîne complète"""
        
        # Clé de traduction inexistante
        translation_result = await full_system["translation"].translate(
            key="nonexistent.key",
            language="fr"
        )
        
        # Doit gérer gracieusement
        assert translation_result.text is not None
        assert translation_result.confidence < 0.5
        
        # Formatage doit toujours marcher
        formatted = full_system["formatting"].format_text(
            translation_result.text,
            "fr"
        )
        
        assert formatted is not None
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, full_system):
        """Test performance sous charge"""
        import time
        
        start_time = time.time()
        
        # Simulation de charge
        tasks = []
        for i in range(50):
            task = full_system["translation"].translate(
                key="alerts.high_cpu",
                language="fr",
                context={"cpu_usage": 80 + i % 20}
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Vérifications de performance
        assert len(results) == 50
        assert all(r.text for r in results)
        assert duration < 5.0  # Moins de 5 secondes pour 50 requêtes
        
        # Vérifier métriques
        metrics = await full_system["translation"].get_metrics()
        assert metrics["total_requests"] >= 50
        assert metrics["average_latency_ms"] < 100  # Moins de 100ms en moyenne


# Configuration des tests
pytest_plugins = ["pytest_asyncio"]

# Fixtures globales pour tous les tests
@pytest.fixture(scope="session")
def event_loop():
    """Event loop pour les tests async"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Marks personnalisés pour organiser les tests
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.slow,  # Pour les tests longs
    pytest.mark.integration  # Pour les tests d'intégration
]

# Configuration des timeouts
def pytest_configure(config):
    """Configuration globale des tests"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")

if __name__ == "__main__":
    # Lancement des tests en mode standalone
    pytest.main([__file__, "-v", "--tb=short"])
