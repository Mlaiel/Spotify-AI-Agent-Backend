"""
Advanced Translation and Internationalization Manager

This module provides sophisticated multi-language support for email templates
including automatic translation, RTL language support, locale-specific formatting,
and AI-powered translation optimization.

Version: 3.0.0
Developed by Spotify AI Agent Team
"""

import re
import json
import asyncio
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
import aiofiles
import aiohttp
import babel
from babel import Locale
from babel.dates import format_datetime, format_date, format_time
from babel.numbers import format_currency, format_decimal, format_percent
from babel.messages import Catalog
from babel.support import Translations
import gettext

logger = structlog.get_logger(__name__)

# ============================================================================
# Translation Configuration Classes
# ============================================================================

class LanguageDirection(Enum):
    """Direction d'écriture"""
    LTR = "ltr"  # Left to Right
    RTL = "rtl"  # Right to Left

class TranslationProvider(Enum):
    """Fournisseurs de traduction"""
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    AMAZON = "amazon"
    DEEPL = "deepl"
    INTERNAL = "internal"

@dataclass
class LanguageConfig:
    """Configuration d'une langue"""
    code: str  # ISO 639-1 code (en, fr, es, etc.)
    name: str
    native_name: str
    direction: LanguageDirection = LanguageDirection.LTR
    locale: Optional[str] = None  # Full locale (en_US, fr_FR, etc.)
    fallback: Optional[str] = "en"
    enabled: bool = True
    emoji_flag: Optional[str] = None

@dataclass
class TranslationEntry:
    """Entrée de traduction"""
    key: str
    translations: Dict[str, str]
    context: Optional[str] = None
    plurals: Optional[Dict[str, Dict[str, str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TranslationRequest:
    """Requête de traduction"""
    text: str
    source_language: str
    target_language: str
    context: Optional[str] = None
    preserve_html: bool = False
    preserve_variables: bool = True

# ============================================================================
# Advanced Translation Manager
# ============================================================================

class AdvancedTranslationManager:
    """Gestionnaire de traductions avancé"""
    
    def __init__(self,
                 translations_dir: str,
                 default_language: str = "en",
                 fallback_language: str = "en",
                 auto_translate: bool = False,
                 translation_provider: TranslationProvider = TranslationProvider.INTERNAL):
        
        self.translations_dir = Path(translations_dir)
        self.default_language = default_language
        self.fallback_language = fallback_language
        self.auto_translate = auto_translate
        self.translation_provider = translation_provider
        
        # Configuration des langues
        self.languages: Dict[str, LanguageConfig] = {}
        self.translations: Dict[str, Dict[str, str]] = {}
        self.catalogs: Dict[str, Catalog] = {}
        
        # Cache et optimisations
        self.translation_cache: Dict[str, str] = {}
        self.interpolation_cache: Dict[str, Callable] = {}
        
        # Formats localisés
        self.locale_cache: Dict[str, Locale] = {}
        
        # Session HTTP pour API de traduction
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Initialize
        asyncio.create_task(self._initialize())
        
        logger.info("Advanced Translation Manager initialized")
    
    async def _initialize(self):
        """Initialisation du gestionnaire"""
        
        # Création des répertoires
        await self._ensure_directories()
        
        # Chargement de la configuration des langues
        await self._load_language_config()
        
        # Chargement des traductions
        await self._load_translations()
        
        # Initialisation des locales
        await self._initialize_locales()
        
        # Session HTTP pour traductions automatiques
        if self.auto_translate:
            self.http_session = aiohttp.ClientSession()
        
        logger.info("Translation Manager initialization completed")
    
    async def _ensure_directories(self):
        """Assure que les répertoires nécessaires existent"""
        
        directories = [
            self.translations_dir,
            self.translations_dir / "locales",
            self.translations_dir / "po",
            self.translations_dir / "json",
            self.translations_dir / "templates"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def _load_language_config(self):
        """Charge la configuration des langues"""
        
        # Configuration par défaut
        default_languages = [
            LanguageConfig("en", "English", "English", emoji_flag="🇺🇸"),
            LanguageConfig("fr", "French", "Français", emoji_flag="🇫🇷"),
            LanguageConfig("es", "Spanish", "Español", emoji_flag="🇪🇸"),
            LanguageConfig("de", "German", "Deutsch", emoji_flag="🇩🇪"),
            LanguageConfig("it", "Italian", "Italiano", emoji_flag="🇮🇹"),
            LanguageConfig("pt", "Portuguese", "Português", emoji_flag="🇵🇹"),
            LanguageConfig("nl", "Dutch", "Nederlands", emoji_flag="🇳🇱"),
            LanguageConfig("ru", "Russian", "Русский", emoji_flag="🇷🇺"),
            LanguageConfig("ja", "Japanese", "日本語", emoji_flag="🇯🇵"),
            LanguageConfig("ko", "Korean", "한국어", emoji_flag="🇰🇷"),
            LanguageConfig("zh", "Chinese", "中文", emoji_flag="🇨🇳"),
            LanguageConfig("ar", "Arabic", "العربية", LanguageDirection.RTL, emoji_flag="🇸🇦"),
            LanguageConfig("he", "Hebrew", "עברית", LanguageDirection.RTL, emoji_flag="🇮🇱"),
            LanguageConfig("hi", "Hindi", "हिन्दी", emoji_flag="🇮🇳"),
            LanguageConfig("th", "Thai", "ไทย", emoji_flag="🇹🇭"),
            LanguageConfig("vi", "Vietnamese", "Tiếng Việt", emoji_flag="🇻🇳")
        ]
        
        for lang in default_languages:
            self.languages[lang.code] = lang
        
        # Chargement de la configuration personnalisée si elle existe
        config_file = self.translations_dir / "languages.json"
        if config_file.exists():
            try:
                async with aiofiles.open(config_file, 'r', encoding='utf-8') as f:
                    custom_config = json.loads(await f.read())
                    
                for lang_code, lang_data in custom_config.items():
                    direction = LanguageDirection(lang_data.get('direction', 'ltr'))
                    
                    self.languages[lang_code] = LanguageConfig(
                        code=lang_code,
                        name=lang_data['name'],
                        native_name=lang_data['native_name'],
                        direction=direction,
                        locale=lang_data.get('locale'),
                        fallback=lang_data.get('fallback', 'en'),
                        enabled=lang_data.get('enabled', True),
                        emoji_flag=lang_data.get('emoji_flag')
                    )
                    
            except Exception as e:
                logger.error(f"Failed to load custom language config: {e}")
    
    async def _load_translations(self):
        """Charge les traductions"""
        
        # Chargement des fichiers JSON
        json_dir = self.translations_dir / "json"
        if json_dir.exists():
            for lang_file in json_dir.glob("*.json"):
                lang_code = lang_file.stem
                try:
                    async with aiofiles.open(lang_file, 'r', encoding='utf-8') as f:
                        translations = json.loads(await f.read())
                        self.translations[lang_code] = translations
                        
                except Exception as e:
                    logger.error(f"Failed to load translations for {lang_code}: {e}")
        
        # Chargement des fichiers PO
        po_dir = self.translations_dir / "po"
        if po_dir.exists():
            for po_file in po_dir.glob("*.po"):
                lang_code = po_file.stem
                try:
                    catalog = Catalog()
                    
                    with open(po_file, 'rb') as f:
                        catalog.update(pofile.read_po(f))
                    
                    self.catalogs[lang_code] = catalog
                    
                    # Conversion en dict pour compatibilité
                    if lang_code not in self.translations:
                        self.translations[lang_code] = {}
                    
                    for message in catalog:
                        if message.id and message.string:
                            self.translations[lang_code][message.id] = message.string
                            
                except Exception as e:
                    logger.error(f"Failed to load PO file for {lang_code}: {e}")
    
    async def _initialize_locales(self):
        """Initialise les locales Babel"""
        
        for lang_code, lang_config in self.languages.items():
            try:
                locale_code = lang_config.locale or lang_code
                locale = Locale.parse(locale_code)
                self.locale_cache[lang_code] = locale
                
            except Exception as e:
                logger.warning(f"Failed to initialize locale for {lang_code}: {e}")
                # Fallback vers locale anglaise
                self.locale_cache[lang_code] = Locale.parse('en')
    
    def translate(self,
                 key: str,
                 language: str,
                 variables: Optional[Dict[str, Any]] = None,
                 context: Optional[str] = None,
                 fallback: Optional[str] = None) -> str:
        """Traduit une clé dans une langue donnée"""
        
        # Cache key
        cache_key = f"{key}:{language}:{context}"
        if cache_key in self.translation_cache:
            translated = self.translation_cache[cache_key]
        else:
            # Recherche de la traduction
            translated = self._find_translation(key, language, context, fallback)
            self.translation_cache[cache_key] = translated
        
        # Interpolation des variables
        if variables:
            translated = self._interpolate_variables(translated, variables, language)
        
        return translated
    
    def _find_translation(self,
                         key: str,
                         language: str,
                         context: Optional[str] = None,
                         fallback: Optional[str] = None) -> str:
        """Trouve une traduction pour une clé"""
        
        # Tentative avec la langue demandée
        if language in self.translations:
            if key in self.translations[language]:
                return self.translations[language][key]
        
        # Tentative avec la langue fallback de la langue
        lang_config = self.languages.get(language)
        if lang_config and lang_config.fallback:
            if lang_config.fallback in self.translations:
                if key in self.translations[lang_config.fallback]:
                    return self.translations[lang_config.fallback][key]
        
        # Tentative avec la langue fallback globale
        if self.fallback_language in self.translations:
            if key in self.translations[self.fallback_language]:
                return self.translations[self.fallback_language][key]
        
        # Utilisation du fallback fourni
        if fallback:
            return fallback
        
        # Retour de la clé comme dernier recours
        logger.warning(f"Translation not found: {key} in {language}")
        return key
    
    def _interpolate_variables(self,
                             text: str,
                             variables: Dict[str, Any],
                             language: str) -> str:
        """Interpole les variables dans le texte"""
        
        # Formatage des variables selon la locale
        formatted_vars = {}
        locale = self.locale_cache.get(language, Locale.parse('en'))
        
        for var_name, var_value in variables.items():
            if isinstance(var_value, datetime):
                formatted_vars[var_name] = format_datetime(var_value, locale=locale)
            elif isinstance(var_value, date):
                formatted_vars[var_name] = format_date(var_value, locale=locale)
            elif isinstance(var_value, (int, float)):
                formatted_vars[var_name] = format_decimal(var_value, locale=locale)
            else:
                formatted_vars[var_name] = str(var_value)
        
        # Interpolation simple
        try:
            return text.format(**formatted_vars)
        except KeyError as e:
            logger.warning(f"Missing variable in translation: {e}")
            return text
        except Exception as e:
            logger.error(f"Variable interpolation failed: {e}")
            return text
    
    async def auto_translate(self,
                           text: str,
                           source_language: str,
                           target_language: str,
                           context: Optional[str] = None) -> Optional[str]:
        """Traduit automatiquement un texte"""
        
        if not self.auto_translate or not self.http_session:
            return None
        
        request = TranslationRequest(
            text=text,
            source_language=source_language,
            target_language=target_language,
            context=context
        )
        
        if self.translation_provider == TranslationProvider.GOOGLE:
            return await self._translate_google(request)
        elif self.translation_provider == TranslationProvider.MICROSOFT:
            return await self._translate_microsoft(request)
        elif self.translation_provider == TranslationProvider.DEEPL:
            return await self._translate_deepl(request)
        else:
            return None
    
    async def _translate_google(self, request: TranslationRequest) -> Optional[str]:
        """Traduction via Google Translate API"""
        
        try:
            # Note: Nécessite une clé API Google Cloud
            url = "https://translation.googleapis.com/language/translate/v2"
            
            params = {
                'q': request.text,
                'source': request.source_language,
                'target': request.target_language,
                'format': 'html' if request.preserve_html else 'text'
            }
            
            async with self.http_session.post(url, json=params) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['data']['translations'][0]['translatedText']
                    
        except Exception as e:
            logger.error(f"Google translation failed: {e}")
        
        return None
    
    async def _translate_microsoft(self, request: TranslationRequest) -> Optional[str]:
        """Traduction via Microsoft Translator"""
        
        try:
            # Note: Nécessite une clé API Azure
            url = "https://api.cognitive.microsofttranslator.com/translate"
            
            params = {
                'api-version': '3.0',
                'from': request.source_language,
                'to': request.target_language
            }
            
            body = [{'text': request.text}]
            
            async with self.http_session.post(url, params=params, json=body) as response:
                if response.status == 200:
                    result = await response.json()
                    return result[0]['translations'][0]['text']
                    
        except Exception as e:
            logger.error(f"Microsoft translation failed: {e}")
        
        return None
    
    async def _translate_deepl(self, request: TranslationRequest) -> Optional[str]:
        """Traduction via DeepL API"""
        
        try:
            # Note: Nécessite une clé API DeepL
            url = "https://api.deepl.com/v2/translate"
            
            data = {
                'text': request.text,
                'source_lang': request.source_language.upper(),
                'target_lang': request.target_language.upper(),
                'tag_handling': 'html' if request.preserve_html else None
            }
            
            async with self.http_session.post(url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['translations'][0]['text']
                    
        except Exception as e:
            logger.error(f"DeepL translation failed: {e}")
        
        return None
    
    def format_currency(self,
                       amount: float,
                       currency: str,
                       language: str) -> str:
        """Formate une devise selon la locale"""
        
        locale = self.locale_cache.get(language, Locale.parse('en'))
        
        try:
            return format_currency(amount, currency, locale=locale)
        except Exception as e:
            logger.error(f"Currency formatting failed: {e}")
            return f"{amount} {currency}"
    
    def format_number(self,
                     number: Union[int, float],
                     language: str) -> str:
        """Formate un nombre selon la locale"""
        
        locale = self.locale_cache.get(language, Locale.parse('en'))
        
        try:
            return format_decimal(number, locale=locale)
        except Exception as e:
            logger.error(f"Number formatting failed: {e}")
            return str(number)
    
    def format_percentage(self,
                         value: float,
                         language: str) -> str:
        """Formate un pourcentage selon la locale"""
        
        locale = self.locale_cache.get(language, Locale.parse('en'))
        
        try:
            return format_percent(value, locale=locale)
        except Exception as e:
            logger.error(f"Percentage formatting failed: {e}")
            return f"{value * 100}%"
    
    def format_datetime(self,
                       dt: datetime,
                       language: str,
                       format_type: str = "medium") -> str:
        """Formate une date/heure selon la locale"""
        
        locale = self.locale_cache.get(language, Locale.parse('en'))
        
        try:
            return format_datetime(dt, format=format_type, locale=locale)
        except Exception as e:
            logger.error(f"DateTime formatting failed: {e}")
            return str(dt)
    
    def get_language_direction(self, language: str) -> LanguageDirection:
        """Obtient la direction d'écriture d'une langue"""
        
        lang_config = self.languages.get(language)
        return lang_config.direction if lang_config else LanguageDirection.LTR
    
    def is_rtl_language(self, language: str) -> bool:
        """Vérifie si une langue s'écrit de droite à gauche"""
        
        return self.get_language_direction(language) == LanguageDirection.RTL
    
    def get_available_languages(self, enabled_only: bool = True) -> List[LanguageConfig]:
        """Obtient la liste des langues disponibles"""
        
        languages = list(self.languages.values())
        
        if enabled_only:
            languages = [lang for lang in languages if lang.enabled]
        
        return sorted(languages, key=lambda x: x.name)
    
    def get_language_name(self, language: str, in_language: Optional[str] = None) -> str:
        """Obtient le nom d'une langue"""
        
        lang_config = self.languages.get(language)
        if not lang_config:
            return language
        
        # Nom dans la langue demandée ou nom natif
        if in_language and in_language != language:
            # Tentative de traduction du nom de langue
            key = f"language.{language}"
            translated = self._find_translation(key, in_language, fallback=lang_config.name)
            return translated if translated != key else lang_config.name
        
        return lang_config.native_name
    
    async def add_translation_entry(self,
                                  key: str,
                                  translations: Dict[str, str],
                                  context: Optional[str] = None):
        """Ajoute une nouvelle entrée de traduction"""
        
        for language, translation in translations.items():
            if language not in self.translations:
                self.translations[language] = {}
            
            self.translations[language][key] = translation
        
        # Sauvegarde
        await self._save_translations()
        
        logger.info(f"Added translation entry: {key}")
    
    async def _save_translations(self):
        """Sauvegarde les traductions"""
        
        json_dir = self.translations_dir / "json"
        
        for language, translations in self.translations.items():
            try:
                file_path = json_dir / f"{language}.json"
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(translations, ensure_ascii=False, indent=2))
                    
            except Exception as e:
                logger.error(f"Failed to save translations for {language}: {e}")
    
    async def extract_translatable_strings(self, template_content: str) -> List[str]:
        """Extrait les chaînes traduisibles d'un template"""
        
        # Patterns pour différents formats de traduction
        patterns = [
            r't\([\'"]([^\'"]+)[\'"]\)',  # t('string')
            r'translate\([\'"]([^\'"]+)[\'"]\)',  # translate('string')
            r'\{\{\s*[\'"]([^\'"]+)[\'"]\s*\|\s*trans\s*\}\}',  # {{ 'string' | trans }}
            r'trans\([\'"]([^\'"]+)[\'"]\)',  # trans('string')
        ]
        
        strings = set()
        
        for pattern in patterns:
            matches = re.findall(pattern, template_content)
            strings.update(matches)
        
        return list(strings)
    
    async def generate_translation_template(self,
                                          source_language: str,
                                          target_languages: List[str],
                                          template_content: str) -> Dict[str, Dict[str, str]]:
        """Génère un template de traduction"""
        
        # Extraction des chaînes
        translatable_strings = await self.extract_translatable_strings(template_content)
        
        template = {}
        
        for target_lang in target_languages:
            template[target_lang] = {}
            
            for string in translatable_strings:
                if self.auto_translate:
                    # Traduction automatique
                    translated = await self.auto_translate(
                        string,
                        source_language,
                        target_lang
                    )
                    template[target_lang][string] = translated or ""
                else:
                    template[target_lang][string] = ""
        
        return template
    
    def apply_rtl_fixes(self, html_content: str, language: str) -> str:
        """Applique les corrections RTL au HTML"""
        
        if not self.is_rtl_language(language):
            return html_content
        
        # Ajout de l'attribut dir
        html_content = re.sub(
            r'<html([^>]*)>',
            r'<html\1 dir="rtl">',
            html_content
        )
        
        # Ajout de classes RTL pour le CSS
        html_content = re.sub(
            r'<body([^>]*)class="([^"]*)"',
            r'<body\1class="\2 rtl"',
            html_content
        )
        
        if 'class=' not in html_content:
            html_content = re.sub(
                r'<body([^>]*)>',
                r'<body\1 class="rtl">',
                html_content
            )
        
        return html_content
    
    async def validate_translations(self, language: str) -> Dict[str, Any]:
        """Valide les traductions d'une langue"""
        
        validation_results = {
            "valid": True,
            "missing_keys": [],
            "empty_translations": [],
            "invalid_interpolations": [],
            "warnings": []
        }
        
        if language not in self.translations:
            validation_results["valid"] = False
            validation_results["missing_keys"] = ["Language not found"]
            return validation_results
        
        # Référence (généralement l'anglais)
        reference_lang = self.fallback_language
        if reference_lang not in self.translations:
            reference_lang = next(iter(self.translations.keys()))
        
        reference_keys = set(self.translations[reference_lang].keys())
        current_keys = set(self.translations[language].keys())
        
        # Clés manquantes
        missing = reference_keys - current_keys
        validation_results["missing_keys"] = list(missing)
        
        if missing:
            validation_results["valid"] = False
        
        # Traductions vides
        for key, value in self.translations[language].items():
            if not value.strip():
                validation_results["empty_translations"].append(key)
        
        # Validation des interpolations
        for key in reference_keys.intersection(current_keys):
            ref_vars = set(re.findall(r'\{([^}]+)\}', self.translations[reference_lang][key]))
            current_vars = set(re.findall(r'\{([^}]+)\}', self.translations[language][key]))
            
            if ref_vars != current_vars:
                validation_results["invalid_interpolations"].append({
                    "key": key,
                    "expected": list(ref_vars),
                    "found": list(current_vars)
                })
        
        return validation_results
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        
        if self.http_session:
            await self.http_session.close()
        
        await self._save_translations()

# ============================================================================
# Template Translation Helper
# ============================================================================

class TemplateTranslationHelper:
    """Helper pour traductions dans les templates"""
    
    def __init__(self, translation_manager: AdvancedTranslationManager):
        self.translation_manager = translation_manager
    
    def t(self, key: str, language: str, **kwargs) -> str:
        """Fonction de traduction courte"""
        return self.translation_manager.translate(key, language, kwargs)
    
    def trans(self, key: str, language: str, **kwargs) -> str:
        """Alias pour translate"""
        return self.translation_manager.translate(key, language, kwargs)
    
    def format_currency(self, amount: float, currency: str, language: str) -> str:
        """Format devise"""
        return self.translation_manager.format_currency(amount, currency, language)
    
    def format_number(self, number: Union[int, float], language: str) -> str:
        """Format nombre"""
        return self.translation_manager.format_number(number, language)
    
    def format_date(self, dt: datetime, language: str) -> str:
        """Format date"""
        return self.translation_manager.format_datetime(dt, language)

# ============================================================================
# Factory Functions
# ============================================================================

def create_translation_manager(
    translations_dir: str,
    default_language: str = "en",
    auto_translate: bool = False
) -> AdvancedTranslationManager:
    """Factory pour créer un gestionnaire de traductions"""
    
    return AdvancedTranslationManager(
        translations_dir=translations_dir,
        default_language=default_language,
        auto_translate=auto_translate
    )

def create_language_config(
    code: str,
    name: str,
    native_name: str,
    is_rtl: bool = False
) -> LanguageConfig:
    """Crée une configuration de langue"""
    
    direction = LanguageDirection.RTL if is_rtl else LanguageDirection.LTR
    
    return LanguageConfig(
        code=code,
        name=name,
        native_name=native_name,
        direction=direction
    )

# Export des classes principales
__all__ = [
    "AdvancedTranslationManager",
    "TemplateTranslationHelper",
    "LanguageConfig",
    "LanguageDirection",
    "TranslationProvider",
    "create_translation_manager",
    "create_language_config"
]
