#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'internationalisation (i18n) pour les alertes Slack

Ce module fournit une infrastructure complète pour la gestion
multilingue des notifications d'alertes via Slack, incluant:

- Traductions automatiques et adaptatives
- Formatage culturel des dates, nombres et devises
- Détection automatique de la langue utilisateur
- Mise en cache intelligente des traductions
- Génération automatique de contenu adaptatif
- Support pour les langues RTL (Right-to-Left)
- Contextualisation des messages selon la gravité
- Pluralisation intelligente
- Fallback automatique entre langues
- Intégration avec l'IA pour améliorer les traductions

Auteur: Expert Team
Version: 2.0.0
"""

import logging
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from datetime import datetime

__version__ = "2.0.0"
__author__ = "Expert Team"
__email__ = "expert.team@spotify-ai-agent.com"

# Configuration du logging pour le module i18n
logger = logging.getLogger(__name__)

# Métadonnées du module
MODULE_METADATA = {
    "name": "slack_i18n",
    "version": __version__,
    "description": "Module d'internationalisation avancé pour alertes Slack",
    "author": __author__,
    "license": "MIT",
    "python_requires": ">=3.9",
    "dependencies": [
        "pyyaml>=6.0",
        "babel>=2.12.0",
        "python-i18n>=0.3.9",
        "polyglot>=16.7.4",
        "langdetect>=1.0.9",
        "redis>=4.5.0",
        "asyncio-throttle>=1.0.2",
        "jinja2>=3.1.0"
    ],
    "features": [
        "Traductions multilingues (15+ langues)",
        "Formatage culturel automatique", 
        "Détection automatique de langue",
        "Cache Redis intégré",
        "Support RTL",
        "Pluralisation intelligente",
        "Fallback automatique",
        "Intégration IA",
        "Templates adaptatifs",
        "Métriques en temps réel"
    ]
}

# Langues supportées avec leurs métadonnées
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "native_name": "English",
        "code": "en",
        "rtl": False,
        "fallback": None,
        "priority": 1,
        "regions": ["US", "GB", "CA", "AU", "NZ"],
        "encoding": "UTF-8",
        "pluralization_rules": "simple"
    },
    "fr": {
        "name": "French",
        "native_name": "Français",
        "code": "fr",
        "rtl": False,
        "fallback": "en",
        "priority": 2,
        "regions": ["FR", "CA", "BE", "CH"],
        "encoding": "UTF-8",
        "pluralization_rules": "french"
    },
    "de": {
        "name": "German",
        "native_name": "Deutsch",
        "code": "de",
        "rtl": False,
        "fallback": "en",
        "priority": 3,
        "regions": ["DE", "AT", "CH"],
        "encoding": "UTF-8",
        "pluralization_rules": "german"
    },
    "es": {
        "name": "Spanish",
        "native_name": "Español",
        "code": "es",
        "rtl": False,
        "fallback": "en",
        "priority": 4,
        "regions": ["ES", "MX", "AR", "CO", "PE"],
        "encoding": "UTF-8",
        "pluralization_rules": "spanish"
    },
    "pt": {
        "name": "Portuguese",
        "native_name": "Português",
        "code": "pt",
        "rtl": False,
        "fallback": "en",
        "priority": 5,
        "regions": ["PT", "BR"],
        "encoding": "UTF-8",
        "pluralization_rules": "portuguese"
    },
    "it": {
        "name": "Italian",
        "native_name": "Italiano",
        "code": "it",
        "rtl": False,
        "fallback": "en",
        "priority": 6,
        "regions": ["IT"],
        "encoding": "UTF-8",
        "pluralization_rules": "italian"
    },
    "ru": {
        "name": "Russian",
        "native_name": "Русский",
        "code": "ru",
        "rtl": False,
        "fallback": "en",
        "priority": 7,
        "regions": ["RU", "BY", "KZ"],
        "encoding": "UTF-8",
        "pluralization_rules": "russian"
    },
    "zh": {
        "name": "Chinese",
        "native_name": "中文",
        "code": "zh",
        "rtl": False,
        "fallback": "en",
        "priority": 8,
        "regions": ["CN", "TW", "HK"],
        "encoding": "UTF-8",
        "pluralization_rules": "chinese"
    },
    "ja": {
        "name": "Japanese",
        "native_name": "日本語",
        "code": "ja",
        "rtl": False,
        "fallback": "en",
        "priority": 9,
        "regions": ["JP"],
        "encoding": "UTF-8",
        "pluralization_rules": "japanese"
    },
    "ar": {
        "name": "Arabic",
        "native_name": "العربية",
        "code": "ar",
        "rtl": True,
        "fallback": "en",
        "priority": 10,
        "regions": ["SA", "AE", "EG", "MA"],
        "encoding": "UTF-8",
        "pluralization_rules": "arabic"
    }
}

# Configuration par défaut
DEFAULT_CONFIG = {
    "default_language": "en",
    "fallback_language": "en",
    "auto_detect": True,
    "cache_enabled": True,
    "cache_ttl": 3600,  # 1 heure
    "translation_cache_size": 10000,
    "enable_ai_enhancement": True,
    "pluralization_enabled": True,
    "rtl_support": True,
    "cultural_formatting": True,
    "emoji_support": True,
    "timezone_aware": True,
    "performance_monitoring": True
}

# Événements du cycle de vie du module
class I18nEvents:
    """Gestionnaire d'événements pour le système i18n"""
    
    LANGUAGE_DETECTED = "language_detected"
    TRANSLATION_LOADED = "translation_loaded"
    TRANSLATION_FAILED = "translation_failed"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    FALLBACK_USED = "fallback_used"
    AI_ENHANCEMENT_APPLIED = "ai_enhancement_applied"
    PERFORMANCE_ALERT = "performance_alert"

# Types personnalisés
TranslationKey = str
LanguageCode = str
TranslationValue = Union[str, Dict[str, Any]]
ContextData = Dict[str, Any]
CultureConfig = Dict[str, Any]

# Interfaces pour l'extensibilité
class TranslationProvider:
    """Interface pour les fournisseurs de traduction"""
    
    def get_translation(self, key: TranslationKey, language: LanguageCode, 
                       context: Optional[ContextData] = None) -> Optional[TranslationValue]:
        raise NotImplementedError
    
    def set_translation(self, key: TranslationKey, language: LanguageCode, 
                       value: TranslationValue, context: Optional[ContextData] = None) -> bool:
        raise NotImplementedError

class CultureProvider:
    """Interface pour les fournisseurs de configuration culturelle"""
    
    def get_culture_config(self, language: LanguageCode) -> CultureConfig:
        raise NotImplementedError
    
    def format_datetime(self, dt: datetime, language: LanguageCode, 
                       format_type: str = "datetime") -> str:
        raise NotImplementedError
    
    def format_number(self, number: Union[int, float], language: LanguageCode, 
                     format_type: str = "decimal") -> str:
        raise NotImplementedError

# Décorateurs utilitaires
def translation_cached(ttl: int = 3600):
    """Décorateur pour mettre en cache les traductions"""
    def decorator(func: Callable) -> Callable:
        func._cached = True
        func._cache_ttl = ttl
        return func
    return decorator

def language_fallback(fallback_lang: str = "en"):
    """Décorateur pour gérer le fallback automatique"""
    def decorator(func: Callable) -> Callable:
        func._fallback_language = fallback_lang
        return func
    return decorator

def performance_monitored(metric_name: str):
    """Décorateur pour monitorer les performances"""
    def decorator(func: Callable) -> Callable:
        func._monitored = True
        func._metric_name = metric_name
        return func
    return decorator

# Configuration du module au chargement
def _initialize_module():
    """Initialise le module i18n avec la configuration par défaut"""
    logger.info(f"Initialisation du module i18n v{__version__}")
    logger.debug(f"Langues supportées: {list(SUPPORTED_LANGUAGES.keys())}")
    logger.debug(f"Configuration par défaut chargée: {DEFAULT_CONFIG}")

# Auto-initialisation
_initialize_module()

# Exports publics
__all__ = [
    "MODULE_METADATA",
    "SUPPORTED_LANGUAGES", 
    "DEFAULT_CONFIG",
    "I18nEvents",
    "TranslationKey",
    "LanguageCode", 
    "TranslationValue",
    "ContextData",
    "CultureConfig",
    "TranslationProvider",
    "CultureProvider",
    "translation_cached",
    "language_fallback", 
    "performance_monitored"
]
