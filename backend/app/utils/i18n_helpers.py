"""
Internationalization Helpers for Spotify AI Agent
================================================

Système complet d'internationalisation industrialisé pour l'agent Spotify.
Support multi-langues, RTL/LTR, formatage locale, détection automatique.
"""

import json
import logging
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import asyncio
from functools import lru_cache
import aiofiles
import locale

logger = logging.getLogger(__name__)

# === Configuration des langues supportées ===
SUPPORTED_LANGUAGES = {
    'en': {
        'name': 'English',
        'native': 'English',
        'rtl': False,
        'decimal_sep': '.',
        'thousands_sep': ',',
        'currency_symbol': '$',
        'date_format': '%Y-%m-%d',
        'time_format': '%H:%M:%S'
    },
    'fr': {
        'name': 'French',
        'native': 'Français',
        'rtl': False,
        'decimal_sep': ',',
        'thousands_sep': ' ',
        'currency_symbol': '€',
        'date_format': '%d/%m/%Y',
        'time_format': '%H:%M:%S'
    },
    'es': {
        'name': 'Spanish',
        'native': 'Español',
        'rtl': False,
        'decimal_sep': ',',
        'thousands_sep': '.',
        'currency_symbol': '€',
        'date_format': '%d/%m/%Y',
        'time_format': '%H:%M:%S'
    },
    'de': {
        'name': 'German',
        'native': 'Deutsch',
        'rtl': False,
        'decimal_sep': ',',
        'thousands_sep': '.',
        'currency_symbol': '€',
        'date_format': '%d.%m.%Y',
        'time_format': '%H:%M:%S'
    },
    'it': {
        'name': 'Italian',
        'native': 'Italiano',
        'rtl': False,
        'decimal_sep': ',',
        'thousands_sep': '.',
        'currency_symbol': '€',
        'date_format': '%d/%m/%Y',
        'time_format': '%H:%M:%S'
    },
    'pt': {
        'name': 'Portuguese',
        'native': 'Português',
        'rtl': False,
        'decimal_sep': ',',
        'thousands_sep': '.',
        'currency_symbol': 'R$',
        'date_format': '%d/%m/%Y',
        'time_format': '%H:%M:%S'
    },
    'ru': {
        'name': 'Russian',
        'native': 'Русский',
        'rtl': False,
        'decimal_sep': ',',
        'thousands_sep': ' ',
        'currency_symbol': '₽',
        'date_format': '%d.%m.%Y',
        'time_format': '%H:%M:%S'
    },
    'ja': {
        'name': 'Japanese',
        'native': '日本語',
        'rtl': False,
        'decimal_sep': '.',
        'thousands_sep': ',',
        'currency_symbol': '¥',
        'date_format': '%Y/%m/%d',
        'time_format': '%H:%M:%S'
    },
    'ko': {
        'name': 'Korean',
        'native': '한국어',
        'rtl': False,
        'decimal_sep': '.',
        'thousands_sep': ',',
        'currency_symbol': '₩',
        'date_format': '%Y.%m.%d',
        'time_format': '%H:%M:%S'
    },
    'zh': {
        'name': 'Chinese',
        'native': '中文',
        'rtl': False,
        'decimal_sep': '.',
        'thousands_sep': ',',
        'currency_symbol': '¥',
        'date_format': '%Y年%m月%d日',
        'time_format': '%H:%M:%S'
    },
    'ar': {
        'name': 'Arabic',
        'native': 'العربية',
        'rtl': True,
        'decimal_sep': '.',
        'thousands_sep': ',',
        'currency_symbol': 'ر.س',
        'date_format': '%d/%m/%Y',
        'time_format': '%H:%M:%S'
    },
    'he': {
        'name': 'Hebrew',
        'native': 'עברית',
        'rtl': True,
        'decimal_sep': '.',
        'thousands_sep': ',',
        'currency_symbol': '₪',
        'date_format': '%d/%m/%Y',
        'time_format': '%H:%M:%S'
    },
    'hi': {
        'name': 'Hindi',
        'native': 'हिन्दी',
        'rtl': False,
        'decimal_sep': '.',
        'thousands_sep': ',',
        'currency_symbol': '₹',
        'date_format': '%d/%m/%Y',
        'time_format': '%H:%M:%S'
    },
    'tr': {
        'name': 'Turkish',
        'native': 'Türkçe',
        'rtl': False,
        'decimal_sep': ',',
        'thousands_sep': '.',
        'currency_symbol': '₺',
        'date_format': '%d.%m.%Y',
        'time_format': '%H:%M:%S'
    },
    'nl': {
        'name': 'Dutch',
        'native': 'Nederlands',
        'rtl': False,
        'decimal_sep': ',',
        'thousands_sep': '.',
        'currency_symbol': '€',
        'date_format': '%d-%m-%Y',
        'time_format': '%H:%M:%S'
    },
    'sv': {
        'name': 'Swedish',
        'native': 'Svenska',
        'rtl': False,
        'decimal_sep': ',',
        'thousands_sep': ' ',
        'currency_symbol': 'kr',
        'date_format': '%Y-%m-%d',
        'time_format': '%H:%M:%S'
    }
}

# Langues RTL
RTL_LANGUAGES = {lang for lang, config in SUPPORTED_LANGUAGES.items() if config['rtl']}

# Fallback par défaut
DEFAULT_LANGUAGE = 'en'

# === Exceptions ===
class I18nError(Exception):
    """Exception de base pour erreurs d'internationalisation."""
    pass

class TranslationNotFoundError(I18nError):
    """Exception pour traductions manquantes."""
    pass

class UnsupportedLanguageError(I18nError):
    """Exception pour langues non supportées."""
    pass

# === Gestionnaire principal d'internationalisation ===
class I18nManager:
    """
    Gestionnaire principal d'internationalisation avec cache et optimisations.
    """
    
    def __init__(self, translations_dir: str = None, default_lang: str = DEFAULT_LANGUAGE):
        self.translations_dir = Path(translations_dir) if translations_dir else None
        self.default_lang = default_lang
        self._translations: Dict[str, Dict[str, Any]] = {}
        self._pluralization_rules: Dict[str, callable] = {}
        self._cache: Dict[str, str] = {}
        self._load_pluralization_rules()
    
    async def load_translations(self, language: str = None) -> bool:
        """
        Charge les traductions depuis les fichiers JSON.
        
        Args:
            language: Code langue à charger (None pour toutes)
            
        Returns:
            True si chargement réussi
        """
        if not self.translations_dir or not self.translations_dir.exists():
            logger.warning("Translations directory not found, using embedded translations")
            self._load_embedded_translations()
            return True
        
        languages_to_load = [language] if language else SUPPORTED_LANGUAGES.keys()
        
        for lang in languages_to_load:
            translation_file = self.translations_dir / f"{lang}.json"
            
            if translation_file.exists():
                try:
                    async with aiofiles.open(translation_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        self._translations[lang] = json.loads(content)
                        logger.info(f"Loaded translations for {lang}")
                except Exception as e:
                    logger.error(f"Error loading translations for {lang}: {e}")
                    return False
            else:
                logger.warning(f"Translation file not found for {lang}")
        
        return True
    
    def _load_embedded_translations(self):
        """Charge les traductions intégrées de base."""
        embedded_translations = {
            'en': {
                'common': {
                    'welcome': 'Welcome',
                    'error': 'Error',
                    'success': 'Success',
                    'loading': 'Loading...',
                    'save': 'Save',
                    'cancel': 'Cancel',
                    'delete': 'Delete',
                    'edit': 'Edit',
                    'search': 'Search',
                    'filter': 'Filter',
                    'sort': 'Sort'
                },
                'music': {
                    'track': 'Track',
                    'album': 'Album',
                    'artist': 'Artist',
                    'playlist': 'Playlist',
                    'genre': 'Genre',
                    'duration': 'Duration',
                    'play': 'Play',
                    'pause': 'Pause',
                    'skip': 'Skip',
                    'repeat': 'Repeat',
                    'shuffle': 'Shuffle'
                },
                'ai': {
                    'generating': 'Generating...',
                    'analyzing': 'Analyzing...',
                    'processing': 'Processing...',
                    'recommendations': 'Recommendations',
                    'mood_analysis': 'Mood Analysis',
                    'lyrics_generation': 'Lyrics Generation'
                },
                'errors': {
                    'network_error': 'Network connection error',
                    'invalid_input': 'Invalid input provided',
                    'permission_denied': 'Permission denied',
                    'rate_limit_exceeded': 'Rate limit exceeded',
                    'service_unavailable': 'Service temporarily unavailable'
                },
                'time': {
                    'now': 'now',
                    'seconds_ago': '{count} seconds ago',
                    'minutes_ago': '{count} minutes ago',
                    'hours_ago': '{count} hours ago',
                    'days_ago': '{count} days ago'
                }
            },
            'fr': {
                'common': {
                    'welcome': 'Bienvenue',
                    'error': 'Erreur',
                    'success': 'Succès',
                    'loading': 'Chargement...',
                    'save': 'Enregistrer',
                    'cancel': 'Annuler',
                    'delete': 'Supprimer',
                    'edit': 'Modifier',
                    'search': 'Rechercher',
                    'filter': 'Filtrer',
                    'sort': 'Trier'
                },
                'music': {
                    'track': 'Piste',
                    'album': 'Album',
                    'artist': 'Artiste',
                    'playlist': 'Liste de lecture',
                    'genre': 'Genre',
                    'duration': 'Durée',
                    'play': 'Jouer',
                    'pause': 'Pause',
                    'skip': 'Passer',
                    'repeat': 'Répéter',
                    'shuffle': 'Aléatoire'
                },
                'ai': {
                    'generating': 'Génération...',
                    'analyzing': 'Analyse...',
                    'processing': 'Traitement...',
                    'recommendations': 'Recommandations',
                    'mood_analysis': 'Analyse de l\'humeur',
                    'lyrics_generation': 'Génération de paroles'
                },
                'errors': {
                    'network_error': 'Erreur de connexion réseau',
                    'invalid_input': 'Entrée invalide fournie',
                    'permission_denied': 'Permission refusée',
                    'rate_limit_exceeded': 'Limite de taux dépassée',
                    'service_unavailable': 'Service temporairement indisponible'
                },
                'time': {
                    'now': 'maintenant',
                    'seconds_ago': 'il y a {count} secondes',
                    'minutes_ago': 'il y a {count} minutes',
                    'hours_ago': 'il y a {count} heures',
                    'days_ago': 'il y a {count} jours'
                }
            },
            'es': {
                'common': {
                    'welcome': 'Bienvenido',
                    'error': 'Error',
                    'success': 'Éxito',
                    'loading': 'Cargando...',
                    'save': 'Guardar',
                    'cancel': 'Cancelar',
                    'delete': 'Eliminar',
                    'edit': 'Editar',
                    'search': 'Buscar',
                    'filter': 'Filtrar',
                    'sort': 'Ordenar'
                },
                'music': {
                    'track': 'Pista',
                    'album': 'Álbum',
                    'artist': 'Artista',
                    'playlist': 'Lista de reproducción',
                    'genre': 'Género',
                    'duration': 'Duración',
                    'play': 'Reproducir',
                    'pause': 'Pausar',
                    'skip': 'Saltar',
                    'repeat': 'Repetir',
                    'shuffle': 'Aleatorio'
                }
            }
        }
        
        self._translations.update(embedded_translations)
    
    def _load_pluralization_rules(self):
        """Charge les règles de pluralisation par langue."""
        self._pluralization_rules = {
            'en': lambda n: 0 if n == 1 else 1,
            'fr': lambda n: 0 if n <= 1 else 1,
            'es': lambda n: 0 if n == 1 else 1,
            'de': lambda n: 0 if n == 1 else 1,
            'it': lambda n: 0 if n == 1 else 1,
            'pt': lambda n: 0 if n == 1 else 1,
            'ru': lambda n: 0 if n % 10 == 1 and n % 100 != 11 else (1 if 2 <= n % 10 <= 4 and (n % 100 < 10 or n % 100 >= 20) else 2),
            'ja': lambda n: 0,  # Pas de pluriel en japonais
            'ko': lambda n: 0,  # Pas de pluriel en coréen
            'zh': lambda n: 0,  # Pas de pluriel en chinois
            'ar': lambda n: 0 if n == 0 else (1 if n == 1 else (2 if n == 2 else (3 if 3 <= n <= 10 else (4 if 11 <= n <= 99 else 5)))),
            'he': lambda n: 0 if n == 1 else (1 if n == 2 else 2),
            'hi': lambda n: 0 if n <= 1 else 1,
            'tr': lambda n: 0 if n == 1 else 1,
            'nl': lambda n: 0 if n == 1 else 1,
            'sv': lambda n: 0 if n == 1 else 1
        }
    
    def translate(self, key: str, language: str = None, **kwargs) -> str:
        """
        Traduit une clé dans la langue spécifiée.
        
        Args:
            key: Clé de traduction (ex: 'common.welcome')
            language: Code langue cible
            **kwargs: Variables à interpoler
            
        Returns:
            Texte traduit avec interpolation
        """
        language = language or self.default_lang
        
        # Vérification cache
        cache_key = f"{language}:{key}:{str(kwargs)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Récupération de la traduction
        translation = self._get_translation(key, language)
        
        # Interpolation des variables
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation interpolation failed for {key}: {e}")
        
        # Mise en cache
        self._cache[cache_key] = translation
        
        return translation
    
    def _get_translation(self, key: str, language: str) -> str:
        """Récupère une traduction depuis les données chargées."""
        # Vérification langue supportée
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language {language}, using {self.default_lang}")
            language = self.default_lang
        
        # Navigation dans l'arbre de traductions
        keys = key.split('.')
        translation_data = self._translations.get(language, {})
        
        for k in keys:
            if isinstance(translation_data, dict) and k in translation_data:
                translation_data = translation_data[k]
            else:
                # Fallback vers langue par défaut
                if language != self.default_lang:
                    return self._get_translation(key, self.default_lang)
                else:
                    logger.warning(f"Translation not found: {key}")
                    return f"[{key}]"
        
        return str(translation_data)
    
    def translate_plural(self, key: str, count: int, language: str = None, **kwargs) -> str:
        """
        Traduit avec gestion de la pluralisation.
        
        Args:
            key: Clé de base (ex: 'time.minutes_ago')
            count: Nombre pour déterminer la forme plurielle
            language: Code langue cible
            **kwargs: Variables supplémentaires
            
        Returns:
            Texte traduit avec forme plurielle correcte
        """
        language = language or self.default_lang
        
        # Détermination de la forme plurielle
        plural_rule = self._pluralization_rules.get(language, lambda n: 0 if n == 1 else 1)
        plural_form = plural_rule(count)
        
        # Construction de la clé plurielle
        plural_key = f"{key}.{plural_form}" if plural_form > 0 else key
        
        # Ajout du count aux variables
        kwargs['count'] = count
        
        return self.translate(plural_key, language, **kwargs)
    
    def get_language_info(self, language: str) -> Dict[str, Any]:
        """Retourne les informations d'une langue."""
        return SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE])
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """Retourne la liste des langues supportées."""
        return [
            {"code": code, **info}
            for code, info in SUPPORTED_LANGUAGES.items()
        ]
    
    def clear_cache(self):
        """Vide le cache de traductions."""
        self._cache.clear()

# === Détecteur de langue ===
class LanguageDetector:
    """
    Détecteur de langue basé sur plusieurs heuristiques.
    """
    
    def __init__(self):
        self._language_patterns = self._build_language_patterns()
    
    def _build_language_patterns(self) -> Dict[str, List[str]]:
        """Construit des patterns de détection par langue."""
        return {
            'en': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with'],
            'fr': ['le', 'de', 'et', 'à', 'un', 'une', 'ce', 'que', 'qui', 'dans'],
            'es': ['el', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no'],
            'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
            'it': ['il', 'di', 'che', 'e', 'la', 'per', 'un', 'in', 'con', 'da'],
            'pt': ['o', 'de', 'que', 'e', 'do', 'a', 'em', 'um', 'para', 'é'],
            'ru': ['в', 'и', 'не', 'на', 'я', 'быть', 'тот', 'он', 'оно', 'с'],
            'ja': ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し'],
            'ko': ['이', '의', '가', '을', '는', '에', '와', '로', '으로', '도'],
            'zh': ['的', '一', '是', '在', '不', '了', '有', '和', '人', '这'],
            'ar': ['في', 'من', 'إلى', 'على', 'أن', 'هذا', 'أو', 'كان', 'التي', 'عن'],
            'tr': ['bir', 've', 'bu', 'da', 'de', 'o', 'için', 'ile', 'var', 'daha']
        }
    
    def detect_language(self, text: str, fallback: str = DEFAULT_LANGUAGE) -> str:
        """
        Détecte la langue d'un texte.
        
        Args:
            text: Texte à analyser
            fallback: Langue par défaut si détection échoue
            
        Returns:
            Code langue détecté
        """
        if not text or len(text.strip()) < 10:
            return fallback
        
        text_lower = text.lower()
        scores = {}
        
        # Score basé sur les mots communs
        for lang, patterns in self._language_patterns.items():
            score = 0
            for pattern in patterns:
                score += text_lower.count(pattern)
            scores[lang] = score
        
        # Détection par caractères spéciaux
        if re.search(r'[الأإؤئءآ]', text):
            scores['ar'] = scores.get('ar', 0) + 10
        if re.search(r'[ñáéíóúü]', text):
            scores['es'] = scores.get('es', 0) + 5
        if re.search(r'[àâäéèêëïîôöùûüÿç]', text):
            scores['fr'] = scores.get('fr', 0) + 5
        if re.search(r'[äöüß]', text):
            scores['de'] = scores.get('de', 0) + 5
        if re.search(r'[ひらがなカタカナ]', text):
            scores['ja'] = scores.get('ja', 0) + 10
        if re.search(r'[가-힣]', text):
            scores['ko'] = scores.get('ko', 0) + 10
        if re.search(r'[一-龯]', text):
            scores['zh'] = scores.get('zh', 0) + 10
        if re.search(r'[а-яё]', text):
            scores['ru'] = scores.get('ru', 0) + 5
        
        # Retourne la langue avec le meilleur score
        if scores:
            detected = max(scores, key=scores.get)
            if scores[detected] > 0:
                return detected
        
        return fallback
    
    async def detect_from_headers(self, accept_language: str) -> str:
        """
        Détecte la langue préférée depuis les headers HTTP.
        
        Args:
            accept_language: Header Accept-Language
            
        Returns:
            Code langue détecté
        """
        if not accept_language:
            return DEFAULT_LANGUAGE
        
        # Parse du header Accept-Language
        languages = []
        for lang_range in accept_language.split(','):
            parts = lang_range.strip().split(';')
            lang = parts[0].strip().lower()
            
            # Extraction du code langue principal
            lang_code = lang.split('-')[0]
            
            # Calcul du poids (q-value)
            weight = 1.0
            if len(parts) > 1:
                q_part = parts[1].strip()
                if q_part.startswith('q='):
                    try:
                        weight = float(q_part[2:])
                    except ValueError:
                        weight = 1.0
            
            if lang_code in SUPPORTED_LANGUAGES:
                languages.append((lang_code, weight))
        
        # Tri par poids décroissant
        languages.sort(key=lambda x: x[1], reverse=True)
        
        return languages[0][0] if languages else DEFAULT_LANGUAGE

# === Formatteur de nombres et dates ===
class LocaleFormatter:
    """
    Formatteur pour nombres, devises et dates selon la locale.
    """
    
    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self.language = language
        self.config = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE])
    
    def format_number(self, number: Union[int, float, Decimal], decimals: int = 2) -> str:
        """
        Formate un nombre selon la locale.
        
        Args:
            number: Nombre à formater
            decimals: Nombre de décimales
            
        Returns:
            Nombre formaté
        """
        if isinstance(number, str):
            try:
                number = float(number)
            except ValueError:
                return str(number)
        
        # Formatage avec décimales
        formatted = f"{number:.{decimals}f}"
        
        # Séparation partie entière et décimale
        if '.' in formatted:
            integer_part, decimal_part = formatted.split('.')
        else:
            integer_part, decimal_part = formatted, ""
        
        # Application du séparateur de milliers
        if len(integer_part) > 3:
            thousands_sep = self.config['thousands_sep']
            reversed_int = integer_part[::-1]
            grouped = [reversed_int[i:i+3] for i in range(0, len(reversed_int), 3)]
            integer_part = thousands_sep.join(grouped)[::-1]
        
        # Assemblage final
        if decimal_part and int(decimal_part) > 0:
            decimal_sep = self.config['decimal_sep']
            return f"{integer_part}{decimal_sep}{decimal_part}"
        else:
            return integer_part
    
    def format_currency(self, amount: Union[int, float, Decimal], currency: str = None) -> str:
        """
        Formate un montant en devise selon la locale.
        
        Args:
            amount: Montant à formater
            currency: Code devise (None pour devise par défaut de la locale)
            
        Returns:
            Montant formaté avec symbole devise
        """
        currency_symbol = currency or self.config['currency_symbol']
        formatted_amount = self.format_number(amount, 2)
        
        # Position du symbole selon la langue
        if self.language in ['en']:
            return f"{currency_symbol}{formatted_amount}"
        else:
            return f"{formatted_amount} {currency_symbol}"
    
    def format_percentage(self, value: Union[int, float], decimals: int = 1) -> str:
        """Formate un pourcentage."""
        formatted = self.format_number(value, decimals)
        return f"{formatted} %"
    
    def format_date(self, date: datetime, format_type: str = 'date') -> str:
        """
        Formate une date selon la locale.
        
        Args:
            date: Date à formater
            format_type: Type de format ('date', 'time', 'datetime')
            
        Returns:
            Date formatée
        """
        if format_type == 'date':
            format_str = self.config['date_format']
        elif format_type == 'time':
            format_str = self.config['time_format']
        else:  # datetime
            format_str = f"{self.config['date_format']} {self.config['time_format']}"
        
        return date.strftime(format_str)
    
    def format_relative_time(self, past_date: datetime, i18n_manager: I18nManager) -> str:
        """
        Formate un temps relatif (ex: "il y a 2 heures").
        
        Args:
            past_date: Date passée
            i18n_manager: Manager d'internationalisation
            
        Returns:
            Temps relatif formaté
        """
        now = datetime.now(past_date.tzinfo)
        delta = now - past_date
        
        if delta.days > 0:
            return i18n_manager.translate_plural('time.days_ago', delta.days, self.language)
        elif delta.seconds >= 3600:
            hours = delta.seconds // 3600
            return i18n_manager.translate_plural('time.hours_ago', hours, self.language)
        elif delta.seconds >= 60:
            minutes = delta.seconds // 60
            return i18n_manager.translate_plural('time.minutes_ago', minutes, self.language)
        elif delta.seconds > 0:
            return i18n_manager.translate_plural('time.seconds_ago', delta.seconds, self.language)
        else:
            return i18n_manager.translate('time.now', self.language)

# === Gestionnaire de direction de texte ===
class TextDirectionManager:
    """
    Gestionnaire pour la direction du texte (LTR/RTL).
    """
    
    @staticmethod
    def is_rtl_language(language: str) -> bool:
        """Vérifie si une langue est RTL."""
        return language in RTL_LANGUAGES
    
    @staticmethod
    def get_text_direction(language: str) -> str:
        """Retourne la direction du texte pour une langue."""
        return 'rtl' if TextDirectionManager.is_rtl_language(language) else 'ltr'
    
    @staticmethod
    def format_mixed_text(text: str, language: str) -> str:
        """
        Formate du texte mixte LTR/RTL.
        
        Args:
            text: Texte à formater
            language: Langue principale
            
        Returns:
            Texte avec marqueurs de direction appropriés
        """
        if not TextDirectionManager.is_rtl_language(language):
            return text
        
        # Ajout de marqueurs Unicode pour RTL
        rtl_mark = '\u200F'  # Right-to-Left Mark
        ltr_mark = '\u200E'  # Left-to-Right Mark
        
        # Détection de segments LTR dans un contexte RTL
        import re
        
        # Patterns pour détecter du contenu LTR (nombres, URLs, etc.)
        ltr_patterns = [
            r'\b\d+\b',  # Nombres
            r'https?://[^\s]+',  # URLs
            r'[a-zA-Z]+@[a-zA-Z]+\.[a-zA-Z]+',  # Emails
            r'[A-Za-z]{2,}'  # Mots latins
        ]
        
        formatted_text = text
        for pattern in ltr_patterns:
            formatted_text = re.sub(
                pattern,
                lambda m: f"{ltr_mark}{m.group()}{rtl_mark}",
                formatted_text
            )
        
        return f"{rtl_mark}{formatted_text}"

# === Validateur de contenu international ===
class InternationalContentValidator:
    """
    Validateur pour contenu international et culturellement approprié.
    """
    
    def __init__(self):
        self._cultural_sensitivities = self._load_cultural_sensitivities()
    
    def _load_cultural_sensitivities(self) -> Dict[str, List[str]]:
        """Charge les sensibilités culturelles par région."""
        return {
            'general': ['politics', 'religion', 'adult_content'],
            'middle_east': ['alcohol', 'pork', 'gambling'],
            'china': ['tibet', 'taiwan', 'tiananmen'],
            'germany': ['nazi', 'holocaust_denial'],
            'france': ['cult', 'scientology'],
            'india': ['beef', 'cow'],
            'japan': ['whale', 'dolphin'],
        }
    
    def validate_content_appropriateness(
        self,
        content: str,
        target_languages: List[str],
        content_type: str = 'general'
    ) -> Dict[str, Any]:
        """
        Valide l'appropriété culturelle d'un contenu.
        
        Args:
            content: Contenu à valider
            target_languages: Langues cibles
            content_type: Type de contenu
            
        Returns:
            Dict avec résultats de validation
        """
        result = {
            'appropriate': True,
            'warnings': [],
            'suggestions': [],
            'blocked_regions': []
        }
        
        content_lower = content.lower()
        
        # Vérification des sensibilités générales
        for sensitivity in self._cultural_sensitivities['general']:
            if sensitivity in content_lower:
                result['warnings'].append(f"Content may contain {sensitivity}-related material")
        
        # Vérification par région linguistique
        region_mapping = {
            'ar': ['middle_east'],
            'zh': ['china'],
            'de': ['germany'],
            'fr': ['france'],
            'hi': ['india'],
            'ja': ['japan']
        }
        
        for lang in target_languages:
            regions = region_mapping.get(lang, [])
            for region in regions:
                if region in self._cultural_sensitivities:
                    for sensitivity in self._cultural_sensitivities[region]:
                        if sensitivity in content_lower:
                            result['blocked_regions'].append(region)
                            result['warnings'].append(
                                f"Content may be inappropriate for {region} due to {sensitivity}"
                            )
        
        # Détermination du statut final
        if result['blocked_regions']:
            result['appropriate'] = False
            result['suggestions'].append("Consider creating region-specific content versions")
        
        return result
    
    def suggest_alternatives(self, problematic_content: str, language: str) -> List[str]:
        """Suggère des alternatives pour contenu problématique."""
        alternatives = []
        
        # Suggestions basiques (à étendre avec IA/NLP)
        replacements = {
            'forbidden': ['not allowed', 'restricted', 'unavailable'],
            'banned': ['prohibited', 'not permitted', 'blocked'],
            'illegal': ['unauthorized', 'not compliant', 'against policy']
        }
        
        for word, alts in replacements.items():
            if word in problematic_content.lower():
                for alt in alts:
                    alternatives.append(problematic_content.replace(word, alt))
        
        return alternatives[:3]  # Limite à 3 suggestions

# === Cache intelligent pour traductions ===
class TranslationCache:
    """
    Cache intelligent pour traductions avec TTL et invalidation.
    """
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[str, datetime, int]] = {}
    
    def get(self, key: str) -> Optional[str]:
        """Récupère une traduction du cache."""
        if key in self._cache:
            value, timestamp, ttl = self._cache[key]
            if (datetime.now() - timestamp).seconds < ttl:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: str, ttl: int = None):
        """Met en cache une traduction."""
        ttl = ttl or self.default_ttl
        
        # Éviction si cache plein
        if len(self._cache) >= self.max_size:
            # Supprime l'entrée la plus ancienne
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        self._cache[key] = (value, datetime.now(), ttl)
    
    def invalidate_pattern(self, pattern: str):
        """Invalide les entrées correspondant à un pattern."""
        import fnmatch
        keys_to_delete = [
            key for key in self._cache.keys()
            if fnmatch.fnmatch(key, pattern)
        ]
        for key in keys_to_delete:
            del self._cache[key]
    
    def clear(self):
        """Vide le cache."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'usage_percentage': (len(self._cache) / self.max_size) * 100
        }

# === Fonctions utilitaires ===
def get_browser_language(accept_language_header: str) -> str:
    """Extrait la langue préférée du navigateur."""
    detector = LanguageDetector()
    return asyncio.run(detector.detect_from_headers(accept_language_header))

def normalize_language_code(lang_code: str) -> str:
    """Normalise un code langue."""
    if not lang_code:
        return DEFAULT_LANGUAGE
    
    # Extraction du code principal (ex: en-US -> en)
    normalized = lang_code.lower().split('-')[0].split('_')[0]
    
    return normalized if normalized in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE

@lru_cache(maxsize=1000)
def get_currency_symbol(language: str) -> str:
    """Retourne le symbole de devise pour une langue."""
    config = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE])
    return config['currency_symbol']

def create_translation_template(keys: List[str], languages: List[str] = None) -> Dict[str, Dict]:
    """
    Crée un template de traduction pour les clés spécifiées.
    
    Args:
        keys: Liste des clés de traduction
        languages: Langues cibles (None pour toutes)
        
    Returns:
        Template de traduction
    """
    languages = languages or list(SUPPORTED_LANGUAGES.keys())
    template = {}
    
    for lang in languages:
        template[lang] = {}
        for key in keys:
            # Navigation dans l'arbre de clés
            current = template[lang]
            key_parts = key.split('.')
            
            for part in key_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Valeur placeholder
            current[key_parts[-1]] = f"[{key}]"
    
    return template

# === Instance globale pour faciliter l'utilisation ===
_global_i18n_manager = None

def get_i18n_manager(translations_dir: str = None) -> I18nManager:
    """Retourne l'instance globale du gestionnaire I18n."""
    global _global_i18n_manager
    
    if _global_i18n_manager is None:
        _global_i18n_manager = I18nManager(translations_dir)
        # Chargement asynchrone en arrière-plan
        asyncio.create_task(_global_i18n_manager.load_translations())
    
    return _global_i18n_manager

def t(key: str, language: str = None, **kwargs) -> str:
    """Fonction raccourci pour traduction."""
    return get_i18n_manager().translate(key, language, **kwargs)

def tn(key: str, count: int, language: str = None, **kwargs) -> str:
    """Fonction raccourci pour traduction avec pluriel."""
    return get_i18n_manager().translate_plural(key, count, language, **kwargs)