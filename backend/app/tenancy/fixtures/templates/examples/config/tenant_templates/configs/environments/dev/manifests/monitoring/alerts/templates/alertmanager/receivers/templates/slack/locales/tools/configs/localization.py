"""
Gestionnaire de localisation avancé pour le système de monitoring Slack.

Ce module fournit un système complet de localisation avec:
- Support de multiples langues avec fallback intelligent
- Cache multi-niveaux pour les performances
- Hot-reload des traductions en temps réel
- Formatage contextuel (dates, nombres, devises)
- Pluralisation automatique selon les règles linguistiques
- Templates Jinja2 avec fonctions de localisation intégrées

Architecture:
    - Factory pattern pour la création des localiseurs
    - Strategy pattern pour les différentes langues
    - Observer pattern pour les mises à jour de traductions
    - Lazy loading des ressources linguistiques
    - Cache distribué pour les environnements multi-instances

Fonctionnalités:
    - Traduction automatique avec contexte
    - Formatage intelligent des dates et heures
    - Support des fuseaux horaires
    - Pluralisation selon les règles CLDR
    - Interpolation de variables dans les messages
    - Validation des traductions manquantes

Auteur: Équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
"""

import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from weakref import WeakSet

import babel
import babel.dates
import babel.numbers
from babel.core import Locale
from babel.plural import PluralRule
from jinja2 import Environment, FileSystemLoader, Template

from .cache_manager import CacheManager
from .metrics import MetricsCollector


class LocaleCode(Enum):
    """Codes de langues supportées."""
    FRENCH = "fr_FR"
    ENGLISH = "en_US"
    GERMAN = "de_DE"
    SPANISH = "es_ES"
    ITALIAN = "it_IT"


class MessageCategory(Enum):
    """Catégories de messages."""
    ALERT = "alert"
    NOTIFICATION = "notification"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"
    SYSTEM = "system"
    USER = "user"


@dataclass
class LocalizationContext:
    """Contexte de localisation."""
    locale: str
    timezone: str = "UTC"
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "locale": self.locale,
            "timezone": self.timezone,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "metadata": self.metadata
        }


@dataclass
class TranslationEntry:
    """Entrée de traduction."""
    key: str
    message: str
    category: MessageCategory
    locale: str
    context: Optional[str] = None
    pluralization_rules: Optional[Dict[str, str]] = None
    variables: Optional[List[str]] = None
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc)


@dataclass
class TranslationStats:
    """Statistiques de traduction."""
    total_keys: int = 0
    translated_keys: int = 0
    missing_keys: int = 0
    coverage_percentage: float = 0.0
    last_update: Optional[datetime] = None
    
    def calculate_coverage(self) -> None:
        """Calcule le taux de couverture."""
        if self.total_keys > 0:
            self.coverage_percentage = (self.translated_keys / self.total_keys) * 100
        else:
            self.coverage_percentage = 0.0


class ITranslationProvider(ABC):
    """Interface pour les fournisseurs de traductions."""
    
    @abstractmethod
    def load_translations(self, locale: str) -> Dict[str, TranslationEntry]:
        """Charge les traductions pour une locale."""
        pass
    
    @abstractmethod
    def save_translation(self, locale: str, key: str, entry: TranslationEntry) -> bool:
        """Sauvegarde une traduction."""
        pass
    
    @abstractmethod
    def get_available_locales(self) -> List[str]:
        """Retourne les locales disponibles."""
        pass


class FileTranslationProvider(ITranslationProvider):
    """Fournisseur de traductions basé sur des fichiers."""
    
    def __init__(self, translations_dir: Path):
        self.translations_dir = Path(translations_dir)
        self._cache: Dict[str, Dict[str, TranslationEntry]] = {}
        self._file_timestamps: Dict[str, float] = {}
        self._metrics = MetricsCollector()
        
        # Création du répertoire si nécessaire
        self.translations_dir.mkdir(parents=True, exist_ok=True)
    
    def load_translations(self, locale: str) -> Dict[str, TranslationEntry]:
        """Charge les traductions depuis un fichier JSON."""
        file_path = self.translations_dir / f"{locale}.json"
        
        if not file_path.exists():
            self._create_default_translation_file(locale, file_path)
            return {}
        
        # Vérification de modification du fichier
        current_timestamp = file_path.stat().st_mtime
        if (locale in self._cache and 
            locale in self._file_timestamps and
            current_timestamp == self._file_timestamps[locale]):
            return self._cache[locale]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            translations = {}
            for key, entry_data in data.items():
                entry = TranslationEntry(
                    key=key,
                    message=entry_data.get("message", ""),
                    category=MessageCategory(entry_data.get("category", "info")),
                    locale=locale,
                    context=entry_data.get("context"),
                    pluralization_rules=entry_data.get("pluralization_rules"),
                    variables=entry_data.get("variables"),
                    last_updated=datetime.fromisoformat(
                        entry_data.get("last_updated", datetime.now().isoformat())
                    )
                )
                translations[key] = entry
            
            self._cache[locale] = translations
            self._file_timestamps[locale] = current_timestamp
            
            self._metrics.increment("translation_file_loaded")
            return translations
            
        except Exception as e:
            self._metrics.increment("translation_file_load_error")
            # Log l'erreur et retourner un dictionnaire vide
            return {}
    
    def save_translation(self, locale: str, key: str, entry: TranslationEntry) -> bool:
        """Sauvegarde une traduction dans le fichier."""
        try:
            file_path = self.translations_dir / f"{locale}.json"
            
            # Chargement des traductions existantes
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {}
            
            # Ajout/mise à jour de l'entrée
            data[key] = {
                "message": entry.message,
                "category": entry.category.value,
                "context": entry.context,
                "pluralization_rules": entry.pluralization_rules,
                "variables": entry.variables,
                "last_updated": entry.last_updated.isoformat()
            }
            
            # Sauvegarde
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Invalidation du cache
            if locale in self._cache:
                self._cache[locale][key] = entry
            
            self._metrics.increment("translation_saved")
            return True
            
        except Exception as e:
            self._metrics.increment("translation_save_error")
            return False
    
    def get_available_locales(self) -> List[str]:
        """Retourne les locales disponibles."""
        locales = []
        for file_path in self.translations_dir.glob("*.json"):
            locale = file_path.stem
            locales.append(locale)
        return locales
    
    def _create_default_translation_file(self, locale: str, file_path: Path) -> None:
        """Crée un fichier de traduction par défaut."""
        default_translations = self._get_default_translations(locale)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(default_translations, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Ignore les erreurs de création
    
    def _get_default_translations(self, locale: str) -> Dict[str, Any]:
        """Retourne les traductions par défaut pour une locale."""
        templates = {
            "fr_FR": {
                "alert.critical.title": {
                    "message": "🚨 Alerte Critique",
                    "category": "alert",
                    "context": "Titre d'alerte critique"
                },
                "alert.warning.title": {
                    "message": "⚠️ Avertissement",
                    "category": "alert",
                    "context": "Titre d'avertissement"
                },
                "alert.info.title": {
                    "message": "ℹ️ Information",
                    "category": "alert",
                    "context": "Titre d'information"
                },
                "alert.resolved.title": {
                    "message": "✅ Résolu",
                    "category": "alert",
                    "context": "Titre de résolution"
                },
                "system.startup": {
                    "message": "Système démarré avec succès",
                    "category": "system"
                },
                "system.shutdown": {
                    "message": "Arrêt du système en cours",
                    "category": "system"
                }
            },
            "en_US": {
                "alert.critical.title": {
                    "message": "🚨 Critical Alert",
                    "category": "alert",
                    "context": "Critical alert title"
                },
                "alert.warning.title": {
                    "message": "⚠️ Warning",
                    "category": "alert",
                    "context": "Warning title"
                },
                "alert.info.title": {
                    "message": "ℹ️ Information",
                    "category": "alert",
                    "context": "Information title"
                },
                "alert.resolved.title": {
                    "message": "✅ Resolved",
                    "category": "alert",
                    "context": "Resolution title"
                },
                "system.startup": {
                    "message": "System started successfully",
                    "category": "system"
                },
                "system.shutdown": {
                    "message": "System shutdown in progress",
                    "category": "system"
                }
            },
            "de_DE": {
                "alert.critical.title": {
                    "message": "🚨 Kritischer Alarm",
                    "category": "alert",
                    "context": "Kritischer Alarm Titel"
                },
                "alert.warning.title": {
                    "message": "⚠️ Warnung",
                    "category": "alert",
                    "context": "Warnung Titel"
                },
                "alert.info.title": {
                    "message": "ℹ️ Information",
                    "category": "alert",
                    "context": "Information Titel"
                },
                "alert.resolved.title": {
                    "message": "✅ Gelöst",
                    "category": "alert",
                    "context": "Lösung Titel"
                },
                "system.startup": {
                    "message": "System erfolgreich gestartet",
                    "category": "system"
                },
                "system.shutdown": {
                    "message": "System-Herunterfahren läuft",
                    "category": "system"
                }
            },
            "es_ES": {
                "alert.critical.title": {
                    "message": "🚨 Alerta Crítica",
                    "category": "alert",
                    "context": "Título de alerta crítica"
                },
                "alert.warning.title": {
                    "message": "⚠️ Advertencia",
                    "category": "alert",
                    "context": "Título de advertencia"
                },
                "alert.info.title": {
                    "message": "ℹ️ Información",
                    "category": "alert",
                    "context": "Título de información"
                },
                "alert.resolved.title": {
                    "message": "✅ Resuelto",
                    "category": "alert",
                    "context": "Título de resolución"
                },
                "system.startup": {
                    "message": "Sistema iniciado con éxito",
                    "category": "system"
                },
                "system.shutdown": {
                    "message": "Apagado del sistema en progreso",
                    "category": "system"
                }
            },
            "it_IT": {
                "alert.critical.title": {
                    "message": "🚨 Allarme Critico",
                    "category": "alert",
                    "context": "Titolo allarme critico"
                },
                "alert.warning.title": {
                    "message": "⚠️ Avvertimento",
                    "category": "alert",
                    "context": "Titolo avvertimento"
                },
                "alert.info.title": {
                    "message": "ℹ️ Informazione",
                    "category": "alert",
                    "context": "Titolo informazione"
                },
                "alert.resolved.title": {
                    "message": "✅ Risolto",
                    "category": "alert",
                    "context": "Titolo risoluzione"
                },
                "system.startup": {
                    "message": "Sistema avviato con successo",
                    "category": "system"
                },
                "system.shutdown": {
                    "message": "Spegnimento del sistema in corso",
                    "category": "system"
                }
            }
        }
        
        return templates.get(locale, templates["en_US"])


class LocalizationEngine:
    """
    Moteur de localisation principal.
    
    Gère la traduction, le formatage et la localisation complète
    des messages avec support du cache et du hot-reload.
    """
    
    def __init__(self,
                 default_locale: str = "fr_FR",
                 fallback_locale: str = "en_US",
                 cache_ttl: int = 3600,
                 translations_dir: Optional[Path] = None,
                 enable_hot_reload: bool = True):
        
        self.default_locale = default_locale
        self.fallback_locale = fallback_locale
        self._cache_ttl = cache_ttl
        self._enable_hot_reload = enable_hot_reload
        
        # Fournisseur de traductions
        if translations_dir is None:
            translations_dir = Path(__file__).parent / "translations"
        
        self._provider = FileTranslationProvider(translations_dir)
        
        # Cache et performance
        self._cache_manager = CacheManager()
        self._metrics = MetricsCollector()
        
        # Babel locales
        self._babel_locales: Dict[str, Locale] = {}
        self._plural_rules: Dict[str, PluralRule] = {}
        
        # Hooks pour les changements
        self._translation_hooks: WeakSet[Callable[[str, str], None]] = WeakSet()
        
        # Templates Jinja2
        self._template_env = self._setup_template_environment()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialisation
        self._initialize_locales()
    
    def translate(self, 
                 key: str,
                 locale: Optional[str] = None,
                 context: Optional[LocalizationContext] = None,
                 variables: Optional[Dict[str, Any]] = None,
                 default_message: Optional[str] = None,
                 count: Optional[int] = None) -> str:
        """
        Traduit un message avec support de la pluralisation et interpolation.
        
        Args:
            key: Clé de traduction
            locale: Locale cible (défaut: locale par défaut)
            context: Contexte de localisation
            variables: Variables pour l'interpolation
            default_message: Message par défaut si la traduction n'existe pas
            count: Nombre pour la pluralisation
            
        Returns:
            Message traduit et formaté
        """
        start_time = time.time()
        
        # Résolution de la locale
        target_locale = locale or (context.locale if context else self.default_locale)
        
        # Recherche de la traduction
        translation = self._get_translation(key, target_locale)
        
        if not translation:
            # Tentative avec la locale de fallback
            translation = self._get_translation(key, self.fallback_locale)
            
            if not translation:
                # Message par défaut ou clé
                message = default_message or key
                self._metrics.increment("translation_missing")
                self._log_missing_translation(key, target_locale)
            else:
                message = translation.message
                self._metrics.increment("translation_fallback_used")
        else:
            message = translation.message
            self._metrics.increment("translation_found")
        
        # Gestion de la pluralisation
        if count is not None and translation and translation.pluralization_rules:
            message = self._apply_pluralization(message, translation, count, target_locale)
        
        # Interpolation des variables
        if variables:
            message = self._interpolate_variables(message, variables, target_locale, context)
        
        # Métriques
        translation_time = (time.time() - start_time) * 1000
        self._metrics.histogram("translation_time_ms", translation_time)
        
        return message
    
    def format_datetime(self,
                       dt: datetime,
                       format_type: str = "medium",
                       locale: Optional[str] = None,
                       timezone: Optional[str] = None) -> str:
        """
        Formate une date/heure selon la locale.
        
        Args:
            dt: Date/heure à formater
            format_type: Type de format (short, medium, long, full)
            locale: Locale cible
            timezone: Fuseau horaire cible
            
        Returns:
            Date/heure formatée
        """
        target_locale = locale or self.default_locale
        babel_locale = self._get_babel_locale(target_locale)
        
        try:
            # Conversion de fuseau horaire si nécessaire
            if timezone and dt.tzinfo is None:
                import pytz
                tz = pytz.timezone(timezone)
                dt = tz.localize(dt)
            elif timezone:
                import pytz
                tz = pytz.timezone(timezone)
                dt = dt.astimezone(tz)
            
            return babel.dates.format_datetime(dt, format_type, locale=babel_locale)
            
        except Exception:
            # Fallback en cas d'erreur
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def format_number(self,
                     number: Union[int, float],
                     locale: Optional[str] = None,
                     decimal_places: Optional[int] = None) -> str:
        """
        Formate un nombre selon la locale.
        
        Args:
            number: Nombre à formater
            locale: Locale cible
            decimal_places: Nombre de décimales
            
        Returns:
            Nombre formaté
        """
        target_locale = locale or self.default_locale
        babel_locale = self._get_babel_locale(target_locale)
        
        try:
            if decimal_places is not None and isinstance(number, float):
                format_pattern = f"#,##0.{'0' * decimal_places}"
                return babel.numbers.format_decimal(number, format=format_pattern, locale=babel_locale)
            else:
                return babel.numbers.format_decimal(number, locale=babel_locale)
                
        except Exception:
            # Fallback en cas d'erreur
            if decimal_places is not None and isinstance(number, float):
                return f"{number:.{decimal_places}f}"
            else:
                return str(number)
    
    def format_currency(self,
                       amount: Union[int, float],
                       currency: str = "EUR",
                       locale: Optional[str] = None) -> str:
        """
        Formate un montant en devise selon la locale.
        
        Args:
            amount: Montant à formater
            currency: Code devise (ISO 4217)
            locale: Locale cible
            
        Returns:
            Montant formaté avec devise
        """
        target_locale = locale or self.default_locale
        babel_locale = self._get_babel_locale(target_locale)
        
        try:
            return babel.numbers.format_currency(amount, currency, locale=babel_locale)
        except Exception:
            # Fallback en cas d'erreur
            return f"{amount} {currency}"
    
    def get_translation_stats(self, locale: str) -> TranslationStats:
        """
        Retourne les statistiques de traduction pour une locale.
        
        Args:
            locale: Locale cible
            
        Returns:
            Statistiques de traduction
        """
        translations = self._provider.load_translations(locale)
        
        stats = TranslationStats()
        stats.translated_keys = len(translations)
        stats.last_update = datetime.now(timezone.utc)
        
        # Calcul du total de clés (union de toutes les locales)
        all_keys = set()
        for available_locale in self._provider.get_available_locales():
            locale_translations = self._provider.load_translations(available_locale)
            all_keys.update(locale_translations.keys())
        
        stats.total_keys = len(all_keys)
        stats.missing_keys = stats.total_keys - stats.translated_keys
        stats.calculate_coverage()
        
        return stats
    
    def add_translation_hook(self, hook: Callable[[str, str], None]) -> None:
        """Ajoute un hook pour les changements de traduction."""
        self._translation_hooks.add(hook)
    
    def reload_translations(self, locale: Optional[str] = None) -> None:
        """
        Recharge les traductions depuis le stockage.
        
        Args:
            locale: Locale spécifique à recharger (toutes si None)
        """
        if locale:
            locales_to_reload = [locale]
        else:
            locales_to_reload = self._provider.get_available_locales()
        
        for loc in locales_to_reload:
            # Invalidation du cache
            cache_key = f"translations:{loc}"
            self._cache_manager.delete(cache_key)
            
            # Rechargement
            self._provider.load_translations(loc)
            
            # Notification des hooks
            for hook in self._translation_hooks:
                try:
                    hook(loc, "reloaded")
                except Exception:
                    pass  # Ignore les erreurs des hooks
    
    def _get_translation(self, key: str, locale: str) -> Optional[TranslationEntry]:
        """Récupère une traduction avec cache."""
        cache_key = f"translations:{locale}"
        
        # Tentative de récupération depuis le cache
        cached_translations = self._cache_manager.get(cache_key)
        
        if cached_translations is None:
            # Chargement depuis le provider
            translations = self._provider.load_translations(locale)
            self._cache_manager.set(cache_key, translations, ttl=self._cache_ttl)
            cached_translations = translations
        
        return cached_translations.get(key)
    
    def _apply_pluralization(self, 
                           message: str, 
                           translation: TranslationEntry, 
                           count: int, 
                           locale: str) -> str:
        """Applique les règles de pluralisation."""
        if not translation.pluralization_rules:
            return message
        
        # Obtention de la règle de pluralisation
        plural_rule = self._get_plural_rule(locale)
        
        try:
            # Détermination de la forme plurielle
            plural_form = plural_rule(count)
            
            # Recherche de la forme appropriée
            if plural_form in translation.pluralization_rules:
                return translation.pluralization_rules[plural_form]
            elif "other" in translation.pluralization_rules:
                return translation.pluralization_rules["other"]
            else:
                return message
                
        except Exception:
            return message
    
    def _interpolate_variables(self, 
                             message: str, 
                             variables: Dict[str, Any], 
                             locale: str,
                             context: Optional[LocalizationContext] = None) -> str:
        """Interpole les variables dans le message."""
        try:
            # Préparation des variables avec formatage
            formatted_vars = {}
            
            for key, value in variables.items():
                if isinstance(value, datetime):
                    formatted_vars[key] = self.format_datetime(value, locale=locale)
                elif isinstance(value, (int, float)) and key.endswith(('_amount', '_price', '_cost')):
                    formatted_vars[key] = self.format_currency(value, locale=locale)
                elif isinstance(value, (int, float)):
                    formatted_vars[key] = self.format_number(value, locale=locale)
                else:
                    formatted_vars[key] = str(value)
            
            # Utilisation de Jinja2 pour l'interpolation
            template = self._template_env.from_string(message)
            return template.render(**formatted_vars)
            
        except Exception:
            # Fallback vers une interpolation simple
            try:
                return message.format(**variables)
            except Exception:
                return message
    
    def _get_babel_locale(self, locale: str) -> Locale:
        """Récupère une instance Babel Locale."""
        if locale not in self._babel_locales:
            try:
                self._babel_locales[locale] = Locale.parse(locale)
            except Exception:
                # Fallback vers la locale par défaut
                self._babel_locales[locale] = Locale.parse(self.fallback_locale)
        
        return self._babel_locales[locale]
    
    def _get_plural_rule(self, locale: str) -> PluralRule:
        """Récupère les règles de pluralisation pour une locale."""
        if locale not in self._plural_rules:
            try:
                babel_locale = self._get_babel_locale(locale)
                self._plural_rules[locale] = babel_locale.plural_form
            except Exception:
                # Fallback vers les règles anglaises
                self._plural_rules[locale] = Locale.parse("en").plural_form
        
        return self._plural_rules[locale]
    
    def _setup_template_environment(self) -> Environment:
        """Configure l'environnement Jinja2."""
        env = Environment()
        
        # Ajout de fonctions personnalisées
        env.globals['format_datetime'] = self.format_datetime
        env.globals['format_number'] = self.format_number
        env.globals['format_currency'] = self.format_currency
        env.globals['translate'] = self.translate
        
        return env
    
    def _initialize_locales(self) -> None:
        """Initialise les locales supportées."""
        supported_locales = [
            LocaleCode.FRENCH.value,
            LocaleCode.ENGLISH.value,
            LocaleCode.GERMAN.value,
            LocaleCode.SPANISH.value,
            LocaleCode.ITALIAN.value
        ]
        
        for locale in supported_locales:
            try:
                self._get_babel_locale(locale)
                self._get_plural_rule(locale)
            except Exception:
                continue  # Ignore les erreurs d'initialisation
    
    def _log_missing_translation(self, key: str, locale: str) -> None:
        """Log une traduction manquante."""
        # Ici, nous pourrions intégrer avec un système de logging
        # ou une base de données pour tracker les traductions manquantes
        pass
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Statistiques du moteur de localisation."""
        available_locales = self._provider.get_available_locales()
        
        return {
            "default_locale": self.default_locale,
            "fallback_locale": self.fallback_locale,
            "available_locales": available_locales,
            "locale_count": len(available_locales),
            "cache_enabled": True,
            "hot_reload_enabled": self._enable_hot_reload,
            "metrics": self._metrics.get_all_metrics()
        }


# Instance globale singleton
_global_localization_engine: Optional[LocalizationEngine] = None
_engine_lock = threading.Lock()


def get_localization_engine(**kwargs) -> LocalizationEngine:
    """
    Récupère l'instance globale du moteur de localisation.
    
    Returns:
        Instance singleton du LocalizationEngine
    """
    global _global_localization_engine
    
    if _global_localization_engine is None:
        with _engine_lock:
            if _global_localization_engine is None:
                _global_localization_engine = LocalizationEngine(**kwargs)
    
    return _global_localization_engine


# API publique simplifiée
def translate(key: str, 
              locale: Optional[str] = None,
              variables: Optional[Dict[str, Any]] = None,
              **kwargs) -> str:
    """API simplifiée pour la traduction."""
    engine = get_localization_engine()
    return engine.translate(key, locale=locale, variables=variables, **kwargs)


def format_datetime(dt: datetime, 
                   locale: Optional[str] = None,
                   **kwargs) -> str:
    """API simplifiée pour le formatage de date."""
    engine = get_localization_engine()
    return engine.format_datetime(dt, locale=locale, **kwargs)


def format_number(number: Union[int, float], 
                 locale: Optional[str] = None,
                 **kwargs) -> str:
    """API simplifiée pour le formatage de nombre."""
    engine = get_localization_engine()
    return engine.format_number(number, locale=locale, **kwargs)


def format_currency(amount: Union[int, float], 
                   currency: str = "EUR",
                   locale: Optional[str] = None,
                   **kwargs) -> str:
    """API simplifiée pour le formatage de devise."""
    engine = get_localization_engine()
    return engine.format_currency(amount, currency=currency, locale=locale, **kwargs)
