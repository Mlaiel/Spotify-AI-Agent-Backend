#!/usr/bin/env python3
"""
Spotify AI Agent - Advanced Locale & Translation Manager
========================================================

Enterprise-grade internationalization and localization system providing:
- Multi-language support with dynamic translation
- Cultural adaptation and localization
- Real-time language switching
- Translation caching and optimization
- Advanced pluralization and formatting
- Regional customization support

Author: Fahed Mlaiel (Lead Developer + AI Architect)
Team: Expert Development Team
"""

import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import yaml
import redis
from babel import Locale, dates, numbers
from babel.core import get_global
from babel.messages.pofile import read_po, write_po
from babel.messages.catalog import Catalog
from babel.support import Translations
import aiofiles
import aiohttp
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)


class SupportedLanguage(str, Enum):
    """Supported languages"""
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"


class TranslationProvider(str, Enum):
    """Translation service providers"""
    LOCAL = "local"
    GOOGLE = "google"
    DEEPL = "deepl"
    AZURE = "azure"
    AWS = "aws"


@dataclass
class TranslationEntry:
    """Translation entry data"""
    key: str
    value: str
    language: str
    context: Optional[str] = None
    pluralization: Optional[Dict[str, str]] = None
    last_updated: datetime = None
    metadata: Dict[str, Any] = None


@dataclass
class LocaleConfig:
    """Locale configuration"""
    language: str
    country: str
    timezone: str
    date_format: str
    time_format: str
    number_format: str
    currency: str
    first_day_of_week: int  # 0=Monday, 6=Sunday


class LocaleManagerConfig(BaseModel):
    """Locale manager configuration"""
    default_language: SupportedLanguage = Field(default=SupportedLanguage.ENGLISH)
    supported_languages: List[SupportedLanguage] = Field(default_factory=lambda: [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.GERMAN,
        SupportedLanguage.FRENCH
    ])
    translation_provider: TranslationProvider = Field(default=TranslationProvider.LOCAL)
    cache_ttl: int = Field(default=3600, ge=300)  # 1 hour
    auto_translate: bool = Field(default=False)
    fallback_language: SupportedLanguage = Field(default=SupportedLanguage.ENGLISH)
    translation_api_key: Optional[str] = Field(default=None)
    redis_url: str = Field(default="redis://localhost:6379")
    locales_path: str = Field(default="./locales")


class LocaleManager:
    """
    Advanced locale and internationalization manager
    """
    
    def __init__(self, config: LocaleManagerConfig):
        self.config = config
        self.redis_client = redis.from_url(config.redis_url, decode_responses=True)
        
        # Translation storage
        self.translations: Dict[str, Dict[str, TranslationEntry]] = {}
        self.locale_configs: Dict[str, LocaleConfig] = {}
        
        # Translation provider
        self.translation_provider = self._init_translation_provider()
        
        # Babel locales
        self.babel_locales: Dict[str, Locale] = {}
        
        logger.info("LocaleManager initialized")

    def _init_translation_provider(self):
        """Initialize translation provider"""
        if self.config.translation_provider == TranslationProvider.GOOGLE:
            return GoogleTranslationProvider(self.config.translation_api_key)
        elif self.config.translation_provider == TranslationProvider.DEEPL:
            return DeepLTranslationProvider(self.config.translation_api_key)
        elif self.config.translation_provider == TranslationProvider.AZURE:
            return AzureTranslationProvider(self.config.translation_api_key)
        else:
            return LocalTranslationProvider()

    async def load_translations(self) -> bool:
        """Load all translations from files and cache"""
        try:
            # Load from files
            await self._load_translations_from_files()
            
            # Load from cache
            await self._load_translations_from_cache()
            
            # Load locale configurations
            await self._load_locale_configs()
            
            # Initialize Babel locales
            self._init_babel_locales()
            
            logger.info(f"Translations loaded for {len(self.translations)} languages")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load translations: {e}")
            return False

    async def _load_translations_from_files(self):
        """Load translations from locale files"""
        locales_path = Path(self.config.locales_path)
        if not locales_path.exists():
            locales_path.mkdir(parents=True)
            return
        
        for lang_dir in locales_path.iterdir():
            if lang_dir.is_dir() and lang_dir.name in [lang.value for lang in self.config.supported_languages]:
                language = lang_dir.name
                self.translations[language] = {}
                
                # Load JSON files
                for json_file in lang_dir.glob("*.json"):
                    await self._load_json_translations(json_file, language)
                
                # Load PO files
                for po_file in lang_dir.glob("*.po"):
                    await self._load_po_translations(po_file, language)

    async def _load_json_translations(self, file_path: Path, language: str):
        """Load translations from JSON file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                
                for key, value in data.items():
                    if isinstance(value, str):
                        entry = TranslationEntry(
                            key=key,
                            value=value,
                            language=language,
                            last_updated=datetime.utcnow()
                        )
                    elif isinstance(value, dict):
                        # Handle complex translations with context/pluralization
                        entry = TranslationEntry(
                            key=key,
                            value=value.get('value', ''),
                            language=language,
                            context=value.get('context'),
                            pluralization=value.get('pluralization'),
                            last_updated=datetime.utcnow(),
                            metadata=value.get('metadata', {})
                        )
                    else:
                        continue
                    
                    self.translations[language][key] = entry
                    
        except Exception as e:
            logger.error(f"Failed to load JSON translations from {file_path}: {e}")

    async def _load_po_translations(self, file_path: Path, language: str):
        """Load translations from PO file"""
        try:
            with open(file_path, 'rb') as f:
                catalog = read_po(f)
                
                for message in catalog:
                    if message.id and message.string:
                        entry = TranslationEntry(
                            key=message.id,
                            value=message.string,
                            language=language,
                            context=message.context,
                            last_updated=datetime.utcnow()
                        )
                        self.translations[language][message.id] = entry
                        
        except Exception as e:
            logger.error(f"Failed to load PO translations from {file_path}: {e}")

    async def _load_translations_from_cache(self):
        """Load translations from Redis cache"""
        try:
            for language in self.config.supported_languages:
                cache_key = f"translations:{language.value}"
                cached_data = self.redis_client.get(cache_key)
                
                if cached_data:
                    data = json.loads(cached_data)
                    if language.value not in self.translations:
                        self.translations[language.value] = {}
                    
                    for key, entry_data in data.items():
                        entry = TranslationEntry(**entry_data)
                        self.translations[language.value][key] = entry
                        
        except Exception as e:
            logger.error(f"Failed to load translations from cache: {e}")

    async def _load_locale_configs(self):
        """Load locale configuration files"""
        config_file = Path(self.config.locales_path) / "locale_configs.yaml"
        
        if config_file.exists():
            try:
                async with aiofiles.open(config_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = yaml.safe_load(content)
                    
                    for lang, config_data in data.items():
                        if lang in [l.value for l in self.config.supported_languages]:
                            self.locale_configs[lang] = LocaleConfig(**config_data)
                            
            except Exception as e:
                logger.error(f"Failed to load locale configs: {e}")
        
        # Create default configs for missing languages
        for language in self.config.supported_languages:
            if language.value not in self.locale_configs:
                self.locale_configs[language.value] = self._create_default_locale_config(language.value)

    def _create_default_locale_config(self, language: str) -> LocaleConfig:
        """Create default locale configuration"""
        configs = {
            "en": LocaleConfig("en", "US", "UTC", "%Y-%m-%d", "%H:%M:%S", "#,##0.##", "USD", 0),
            "de": LocaleConfig("de", "DE", "Europe/Berlin", "%d.%m.%Y", "%H:%M", "#.##0,##", "EUR", 0),
            "fr": LocaleConfig("fr", "FR", "Europe/Paris", "%d/%m/%Y", "%H:%M", "# ##0,##", "EUR", 0),
            "es": LocaleConfig("es", "ES", "Europe/Madrid", "%d/%m/%Y", "%H:%M", "#.##0,##", "EUR", 0),
            "it": LocaleConfig("it", "IT", "Europe/Rome", "%d/%m/%Y", "%H:%M", "#.##0,##", "EUR", 0),
        }
        
        return configs.get(language, configs["en"])

    def _init_babel_locales(self):
        """Initialize Babel locale objects"""
        for language in self.config.supported_languages:
            try:
                self.babel_locales[language.value] = Locale.parse(language.value)
            except Exception as e:
                logger.error(f"Failed to initialize Babel locale for {language.value}: {e}")

    async def translate(
        self, 
        key: str, 
        language: str, 
        context: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        count: Optional[int] = None
    ) -> str:
        """Translate a key to the specified language"""
        
        # Normalize language
        language = self._normalize_language(language)
        
        # Try to get from cache/memory
        translation = await self._get_translation_from_cache(key, language, context)
        
        if not translation:
            # Try auto-translation if enabled
            if self.config.auto_translate:
                translation = await self._auto_translate(key, language, context)
            else:
                # Fall back to default language
                translation = await self._get_translation_from_cache(
                    key, 
                    self.config.fallback_language.value, 
                    context
                )
        
        if not translation:
            # Return key if no translation found
            return key
        
        # Handle pluralization
        if count is not None and translation.pluralization:
            translation_text = self._handle_pluralization(translation, count, language)
        else:
            translation_text = translation.value
        
        # Handle variable substitution
        if variables:
            translation_text = self._substitute_variables(translation_text, variables)
        
        return translation_text

    async def _get_translation_from_cache(
        self, 
        key: str, 
        language: str, 
        context: Optional[str] = None
    ) -> Optional[TranslationEntry]:
        """Get translation from memory or Redis cache"""
        
        # Check memory first
        if language in self.translations and key in self.translations[language]:
            entry = self.translations[language][key]
            if context is None or entry.context == context:
                return entry
        
        # Check Redis cache
        cache_key = f"translation:{language}:{key}"
        if context:
            cache_key += f":{context}"
        
        cached_data = self.redis_client.get(cache_key)
        if cached_data:
            entry_data = json.loads(cached_data)
            return TranslationEntry(**entry_data)
        
        return None

    async def _auto_translate(
        self, 
        key: str, 
        target_language: str, 
        context: Optional[str] = None
    ) -> Optional[TranslationEntry]:
        """Auto-translate using translation provider"""
        
        # Get source text (from default language)
        source_entry = await self._get_translation_from_cache(
            key, 
            self.config.fallback_language.value, 
            context
        )
        
        if not source_entry:
            return None
        
        try:
            # Translate using provider
            translated_text = await self.translation_provider.translate(
                source_entry.value,
                target_language,
                self.config.fallback_language.value
            )
            
            if translated_text:
                # Create translation entry
                entry = TranslationEntry(
                    key=key,
                    value=translated_text,
                    language=target_language,
                    context=context,
                    last_updated=datetime.utcnow(),
                    metadata={"auto_translated": True}
                )
                
                # Cache the translation
                await self._cache_translation(entry)
                
                return entry
                
        except Exception as e:
            logger.error(f"Auto-translation failed for {key} to {target_language}: {e}")
        
        return None

    async def _cache_translation(self, entry: TranslationEntry):
        """Cache translation in memory and Redis"""
        
        # Store in memory
        if entry.language not in self.translations:
            self.translations[entry.language] = {}
        
        self.translations[entry.language][entry.key] = entry
        
        # Store in Redis
        cache_key = f"translation:{entry.language}:{entry.key}"
        if entry.context:
            cache_key += f":{entry.context}"
        
        self.redis_client.setex(
            cache_key,
            self.config.cache_ttl,
            json.dumps(asdict(entry), default=str)
        )

    def _handle_pluralization(self, entry: TranslationEntry, count: int, language: str) -> str:
        """Handle pluralization based on language rules"""
        if not entry.pluralization:
            return entry.value
        
        # Get plural rule for language
        babel_locale = self.babel_locales.get(language)
        if babel_locale:
            plural_form = babel_locale.plural_form(count)
            
            # Map plural forms to keys
            if plural_form == 1:  # Singular
                return entry.pluralization.get('one', entry.value)
            else:  # Plural
                return entry.pluralization.get('other', entry.value)
        
        # Fallback to simple English rules
        if count == 1:
            return entry.pluralization.get('one', entry.value)
        else:
            return entry.pluralization.get('other', entry.value)

    def _substitute_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in translation text"""
        try:
            return text.format(**variables)
        except Exception as e:
            logger.error(f"Variable substitution failed: {e}")
            return text

    def _normalize_language(self, language: str) -> str:
        """Normalize language code"""
        # Handle common variations
        language = language.lower().split('-')[0].split('_')[0]
        
        # Map to supported languages
        language_map = {
            'en': 'en',
            'de': 'de', 
            'fr': 'fr',
            'es': 'es',
            'it': 'it',
            'pt': 'pt',
            'nl': 'nl',
            'zh': 'zh',
            'ja': 'ja',
            'ko': 'ko',
            'ru': 'ru',
            'ar': 'ar'
        }
        
        return language_map.get(language, self.config.default_language.value)

    async def add_translation(
        self, 
        key: str, 
        value: str, 
        language: str,
        context: Optional[str] = None,
        pluralization: Optional[Dict[str, str]] = None
    ) -> bool:
        """Add or update a translation"""
        try:
            entry = TranslationEntry(
                key=key,
                value=value,
                language=language,
                context=context,
                pluralization=pluralization,
                last_updated=datetime.utcnow()
            )
            
            await self._cache_translation(entry)
            
            # Save to file
            await self._save_translation_to_file(entry)
            
            logger.info(f"Translation added: {key} for {language}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add translation: {e}")
            return False

    async def _save_translation_to_file(self, entry: TranslationEntry):
        """Save translation to file"""
        file_path = Path(self.config.locales_path) / entry.language / "translations.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing translations
        translations = {}
        if file_path.exists():
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                translations = json.loads(content)
        
        # Add/update translation
        if entry.pluralization or entry.context or entry.metadata:
            translations[entry.key] = {
                'value': entry.value,
                'context': entry.context,
                'pluralization': entry.pluralization,
                'metadata': entry.metadata
            }
        else:
            translations[entry.key] = entry.value
        
        # Save back to file
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(translations, indent=2, ensure_ascii=False))

    def format_date(self, date_obj: datetime, language: str, format_type: str = "medium") -> str:
        """Format date according to locale"""
        babel_locale = self.babel_locales.get(language)
        if babel_locale:
            return dates.format_date(date_obj.date(), format=format_type, locale=babel_locale)
        return date_obj.strftime("%Y-%m-%d")

    def format_time(self, time_obj: datetime, language: str, format_type: str = "medium") -> str:
        """Format time according to locale"""
        babel_locale = self.babel_locales.get(language)
        if babel_locale:
            return dates.format_time(time_obj.time(), format=format_type, locale=babel_locale)
        return time_obj.strftime("%H:%M:%S")

    def format_datetime(self, datetime_obj: datetime, language: str, format_type: str = "medium") -> str:
        """Format datetime according to locale"""
        babel_locale = self.babel_locales.get(language)
        if babel_locale:
            return dates.format_datetime(datetime_obj, format=format_type, locale=babel_locale)
        return datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

    def format_number(self, number: Union[int, float], language: str) -> str:
        """Format number according to locale"""
        babel_locale = self.babel_locales.get(language)
        if babel_locale:
            return numbers.format_number(number, locale=babel_locale)
        return str(number)

    def format_currency(self, amount: Union[int, float], currency: str, language: str) -> str:
        """Format currency according to locale"""
        babel_locale = self.babel_locales.get(language)
        if babel_locale:
            return numbers.format_currency(amount, currency, locale=babel_locale)
        return f"{currency} {amount}"

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return [lang.value for lang in self.config.supported_languages]

    def get_locale_config(self, language: str) -> Optional[LocaleConfig]:
        """Get locale configuration for language"""
        return self.locale_configs.get(language)


class MultiLanguageHandler:
    """
    High-level handler for multi-language content
    """
    
    def __init__(self, locale_manager: LocaleManager):
        self.locale_manager = locale_manager
        self.user_languages: Dict[str, str] = {}  # user_id -> language
    
    def set_user_language(self, user_id: str, language: str):
        """Set preferred language for user"""
        normalized_lang = self.locale_manager._normalize_language(language)
        self.user_languages[user_id] = normalized_lang
    
    def get_user_language(self, user_id: str) -> str:
        """Get user's preferred language"""
        return self.user_languages.get(
            user_id, 
            self.locale_manager.config.default_language.value
        )
    
    async def translate_for_user(
        self, 
        user_id: str, 
        key: str, 
        variables: Optional[Dict[str, Any]] = None,
        count: Optional[int] = None
    ) -> str:
        """Translate content for specific user"""
        language = self.get_user_language(user_id)
        return await self.locale_manager.translate(key, language, variables=variables, count=count)
    
    async def format_message_for_user(
        self,
        user_id: str,
        template_key: str,
        data: Dict[str, Any]
    ) -> str:
        """Format a complete message for user"""
        language = self.get_user_language(user_id)
        
        # Translate template
        template = await self.locale_manager.translate(template_key, language)
        
        # Format dates, numbers, etc. according to locale
        formatted_data = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                formatted_data[key] = self.locale_manager.format_datetime(value, language)
            elif isinstance(value, (int, float)) and key.endswith('_amount'):
                currency = data.get(f"{key}_currency", "USD")
                formatted_data[key] = self.locale_manager.format_currency(value, currency, language)
            elif isinstance(value, (int, float)):
                formatted_data[key] = self.locale_manager.format_number(value, language)
            else:
                formatted_data[key] = value
        
        # Substitute variables
        return self.locale_manager._substitute_variables(template, formatted_data)


class TranslationEngine:
    """
    Advanced translation engine with AI capabilities
    """
    
    def __init__(self, locale_manager: LocaleManager):
        self.locale_manager = locale_manager
    
    async def batch_translate(
        self, 
        keys: List[str], 
        target_language: str,
        source_language: Optional[str] = None
    ) -> Dict[str, str]:
        """Translate multiple keys in batch"""
        if source_language is None:
            source_language = self.locale_manager.config.fallback_language.value
        
        results = {}
        
        # Group translations for efficiency
        for key in keys:
            translation = await self.locale_manager.translate(key, target_language)
            results[key] = translation
        
        return results
    
    async def suggest_translations(
        self, 
        source_text: str, 
        target_language: str
    ) -> List[str]:
        """Get translation suggestions"""
        if hasattr(self.locale_manager.translation_provider, 'get_suggestions'):
            return await self.locale_manager.translation_provider.get_suggestions(
                source_text, target_language
            )
        return []


# Translation Provider Implementations

class LocalTranslationProvider:
    """Local file-based translation provider"""
    
    async def translate(self, text: str, target_lang: str, source_lang: str) -> Optional[str]:
        # Local provider doesn't do automatic translation
        return None


class GoogleTranslationProvider:
    """Google Translate API provider"""
    
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
    
    async def translate(self, text: str, target_lang: str, source_lang: str) -> Optional[str]:
        if not self.api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://translation.googleapis.com/language/translate/v2?key={self.api_key}"
                data = {
                    'q': text,
                    'source': source_lang,
                    'target': target_lang
                }
                
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['data']['translations'][0]['translatedText']
        except Exception as e:
            logger.error(f"Google translation failed: {e}")
        
        return None


class DeepLTranslationProvider:
    """DeepL API provider"""
    
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
    
    async def translate(self, text: str, target_lang: str, source_lang: str) -> Optional[str]:
        if not self.api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api-free.deepl.com/v2/translate"
                headers = {"Authorization": f"DeepL-Auth-Key {self.api_key}"}
                data = {
                    'text': text,
                    'source_lang': source_lang.upper(),
                    'target_lang': target_lang.upper()
                }
                
                async with session.post(url, headers=headers, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['translations'][0]['text']
        except Exception as e:
            logger.error(f"DeepL translation failed: {e}")
        
        return None


class AzureTranslationProvider:
    """Azure Translator API provider"""
    
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
    
    async def translate(self, text: str, target_lang: str, source_lang: str) -> Optional[str]:
        if not self.api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.cognitive.microsofttranslator.com/translate"
                headers = {
                    "Ocp-Apim-Subscription-Key": self.api_key,
                    "Content-Type": "application/json"
                }
                params = {
                    'api-version': '3.0',
                    'from': source_lang,
                    'to': target_lang
                }
                data = [{'text': text}]
                
                async with session.post(url, headers=headers, params=params, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result[0]['translations'][0]['text']
        except Exception as e:
            logger.error(f"Azure translation failed: {e}")
        
        return None
