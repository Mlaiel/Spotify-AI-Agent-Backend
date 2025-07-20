"""
Advanced Locale Manager for Multi-Language Slack Alert System.

This module provides comprehensive internationalization support for Slack alerts
with advanced features including cultural formatting, pluralization, and
intelligent fallback mechanisms.

Features:
- 15+ language support with cultural formatting
- Advanced pluralization rules
- Variable interpolation and formatting
- Intelligent fallback mechanisms
- Performance-optimized translation caching
- Real-time locale switching
- Cultural-aware date/time/number formatting

Author: Fahed Mlaiel
Version: 1.0.0
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from babel import Locale, dates, numbers, core
from babel.support import Format
from babel.plural import PluralRule
import yaml

from .constants import (
    DEFAULT_LOCALE,
    SUPPORTED_LOCALES,
    FALLBACK_LOCALES,
    LOCALE_CACHE_TTL
)
from .exceptions import (
    LocaleError,
    TranslationNotFoundError,
    InvalidLocaleError,
    FormatError
)
from .performance_monitor import PerformanceMonitor


@dataclass
class LocaleConfig:
    """Configuration for a specific locale."""
    code: str
    name: str
    native_name: str
    fallback: str
    rtl: bool = False
    date_format: str = 'medium'
    time_format: str = 'medium'
    number_format: str = 'decimal'
    currency_code: str = 'USD'
    timezone: str = 'UTC'
    enabled: bool = True


@dataclass
class TranslationMessage:
    """A translated message with metadata."""
    key: str
    locale: str
    message: str
    plural_forms: Optional[Dict[str, str]] = None
    variables: Optional[List[str]] = None
    context: Optional[str] = None
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()


class LocaleManager:
    """
    Advanced locale manager for multi-language support.
    
    Provides comprehensive internationalization features including
    translation management, cultural formatting, and performance optimization.
    """

    def __init__(
        self,
        locales_path: Optional[str] = None,
        cache_ttl: int = LOCALE_CACHE_TTL,
        enable_monitoring: bool = True,
        preload_locales: bool = True
    ):
        """
        Initialize the locale manager.
        
        Args:
            locales_path: Path to locale files directory
            cache_ttl: Cache time-to-live in seconds
            enable_monitoring: Enable performance monitoring
            preload_locales: Preload all locales on initialization
        """
        self.locales_path = Path(locales_path) if locales_path else Path(__file__).parent / "locales"
        self.cache_ttl = cache_ttl
        self.enable_monitoring = enable_monitoring
        self.preload_locales = preload_locales
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor() if enable_monitoring else None
        
        # Locale configurations
        self.locale_configs: Dict[str, LocaleConfig] = {}
        
        # Translation cache
        self.translation_cache: Dict[str, Dict[str, TranslationMessage]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Babel locale objects cache
        self.babel_locales: Dict[str, Locale] = {}
        
        # Plural rules cache
        self.plural_rules: Dict[str, PluralRule] = {}
        
        # Format objects cache
        self.formatters: Dict[str, Dict[str, Format]] = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize the locale manager."""
        try:
            # Load locale configurations
            await self._load_locale_configs()
            
            # Initialize Babel locales
            await self._initialize_babel_locales()
            
            # Load plural rules
            await self._load_plural_rules()
            
            # Initialize formatters
            await self._initialize_formatters()
            
            # Preload translations if enabled
            if self.preload_locales:
                await self._preload_translations()
            
            # Initialize performance monitoring
            if self.performance_monitor:
                await self.performance_monitor.initialize()
            
            self.logger.info("LocaleManager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LocaleManager: {e}")
            raise LocaleError(f"Initialization failed: {e}")

    async def get_message(
        self,
        key: str,
        locale: str = DEFAULT_LOCALE,
        variables: Optional[Dict[str, Any]] = None,
        plural_count: Optional[int] = None,
        context: Optional[str] = None,
        fallback_message: Optional[str] = None
    ) -> str:
        """
        Get a translated message with variable interpolation.
        
        Args:
            key: Translation key
            locale: Target locale code
            variables: Variables for interpolation
            plural_count: Count for pluralization
            context: Message context for disambiguation
            fallback_message: Fallback message if translation not found
            
        Returns:
            Formatted translated message
            
        Raises:
            TranslationNotFoundError: If translation not found and no fallback
            LocaleError: If locale is invalid
        """
        if self.performance_monitor:
            timer = self.performance_monitor.start_timer("get_message")
        
        try:
            # Validate locale
            normalized_locale = await self._normalize_locale(locale)
            
            # Get translation message
            translation = await self._get_translation(
                key, normalized_locale, context, fallback_message
            )
            
            if not translation:
                if fallback_message:
                    translation = TranslationMessage(
                        key=key,
                        locale=normalized_locale,
                        message=fallback_message
                    )
                else:
                    raise TranslationNotFoundError(f"Translation not found for key: {key}")
            
            # Handle pluralization
            message = translation.message
            if plural_count is not None and translation.plural_forms:
                message = await self._get_plural_form(
                    translation, normalized_locale, plural_count
                )
            
            # Interpolate variables
            if variables:
                message = await self._interpolate_variables(
                    message, variables, normalized_locale
                )
            
            return message
            
        except Exception as e:
            self.logger.error(f"Failed to get message for key {key}: {e}")
            if fallback_message:
                return fallback_message
            raise
        finally:
            if self.performance_monitor and 'timer' in locals():
                self.performance_monitor.end_timer(timer)

    async def format_datetime(
        self,
        dt: datetime,
        locale: str = DEFAULT_LOCALE,
        format_type: str = 'medium',
        timezone_name: Optional[str] = None
    ) -> str:
        """
        Format datetime according to locale conventions.
        
        Args:
            dt: Datetime to format
            locale: Target locale code
            format_type: Format type (short, medium, long, full)
            timezone_name: Target timezone name
            
        Returns:
            Formatted datetime string
        """
        try:
            normalized_locale = await self._normalize_locale(locale)
            babel_locale = self.babel_locales.get(normalized_locale)
            
            if not babel_locale:
                babel_locale = Locale.parse(normalized_locale)
                self.babel_locales[normalized_locale] = babel_locale
            
            # Convert timezone if specified
            if timezone_name:
                target_tz = timezone.fromisoformat(timezone_name)
                dt = dt.replace(tzinfo=timezone.utc).astimezone(target_tz)
            
            # Format datetime
            formatted = dates.format_datetime(
                dt,
                format=format_type,
                locale=babel_locale
            )
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Failed to format datetime: {e}")
            return dt.isoformat()

    async def format_date(
        self,
        date_obj: datetime,
        locale: str = DEFAULT_LOCALE,
        format_type: str = 'medium'
    ) -> str:
        """Format date according to locale conventions."""
        try:
            normalized_locale = await self._normalize_locale(locale)
            babel_locale = self.babel_locales.get(normalized_locale)
            
            if not babel_locale:
                babel_locale = Locale.parse(normalized_locale)
                self.babel_locales[normalized_locale] = babel_locale
            
            formatted = dates.format_date(
                date_obj,
                format=format_type,
                locale=babel_locale
            )
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Failed to format date: {e}")
            return date_obj.strftime('%Y-%m-%d')

    async def format_time(
        self,
        time_obj: datetime,
        locale: str = DEFAULT_LOCALE,
        format_type: str = 'medium'
    ) -> str:
        """Format time according to locale conventions."""
        try:
            normalized_locale = await self._normalize_locale(locale)
            babel_locale = self.babel_locales.get(normalized_locale)
            
            if not babel_locale:
                babel_locale = Locale.parse(normalized_locale)
                self.babel_locales[normalized_locale] = babel_locale
            
            formatted = dates.format_time(
                time_obj,
                format=format_type,
                locale=babel_locale
            )
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Failed to format time: {e}")
            return time_obj.strftime('%H:%M:%S')

    async def format_number(
        self,
        number: Union[int, float, Decimal],
        locale: str = DEFAULT_LOCALE,
        format_type: str = 'decimal'
    ) -> str:
        """
        Format number according to locale conventions.
        
        Args:
            number: Number to format
            locale: Target locale code
            format_type: Format type (decimal, currency, percent, scientific)
            
        Returns:
            Formatted number string
        """
        try:
            normalized_locale = await self._normalize_locale(locale)
            babel_locale = self.babel_locales.get(normalized_locale)
            
            if not babel_locale:
                babel_locale = Locale.parse(normalized_locale)
                self.babel_locales[normalized_locale] = babel_locale
            
            if format_type == 'decimal':
                formatted = numbers.format_decimal(number, locale=babel_locale)
            elif format_type == 'currency':
                locale_config = self.locale_configs.get(normalized_locale)
                currency = locale_config.currency_code if locale_config else 'USD'
                formatted = numbers.format_currency(number, currency, locale=babel_locale)
            elif format_type == 'percent':
                formatted = numbers.format_percent(number, locale=babel_locale)
            elif format_type == 'scientific':
                formatted = numbers.format_scientific(number, locale=babel_locale)
            else:
                formatted = numbers.format_decimal(number, locale=babel_locale)
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Failed to format number: {e}")
            return str(number)

    async def get_locale_info(self, locale: str = DEFAULT_LOCALE) -> Dict[str, Any]:
        """
        Get comprehensive locale information.
        
        Args:
            locale: Locale code
            
        Returns:
            Dictionary with locale information
        """
        try:
            normalized_locale = await self._normalize_locale(locale)
            locale_config = self.locale_configs.get(normalized_locale)
            babel_locale = self.babel_locales.get(normalized_locale)
            
            if not babel_locale:
                babel_locale = Locale.parse(normalized_locale)
                self.babel_locales[normalized_locale] = babel_locale
            
            info = {
                'code': normalized_locale,
                'display_name': babel_locale.display_name,
                'english_name': babel_locale.english_name,
                'language': babel_locale.language,
                'territory': babel_locale.territory,
                'script': babel_locale.script,
                'variant': babel_locale.variant,
                'text_direction': 'rtl' if babel_locale.text_direction == 'rtl' else 'ltr',
                'number_symbols': dict(babel_locale.number_symbols) if babel_locale.number_symbols else {},
                'decimal_formats': dict(babel_locale.decimal_formats) if babel_locale.decimal_formats else {},
                'currency_formats': dict(babel_locale.currency_formats) if babel_locale.currency_formats else {},
                'percent_formats': dict(babel_locale.percent_formats) if babel_locale.percent_formats else {},
                'scientific_formats': dict(babel_locale.scientific_formats) if babel_locale.scientific_formats else {}
            }
            
            # Add custom locale config if available
            if locale_config:
                info.update({
                    'native_name': locale_config.native_name,
                    'fallback': locale_config.fallback,
                    'timezone': locale_config.timezone,
                    'currency_code': locale_config.currency_code,
                    'enabled': locale_config.enabled
                })
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get locale info for {locale}: {e}")
            raise LocaleError(f"Failed to get locale info: {e}")

    async def get_supported_locales(self) -> List[Dict[str, str]]:
        """Get list of supported locales with display names."""
        locales = []
        
        for locale_code in SUPPORTED_LOCALES:
            try:
                babel_locale = self.babel_locales.get(locale_code)
                if not babel_locale:
                    babel_locale = Locale.parse(locale_code)
                    self.babel_locales[locale_code] = babel_locale
                
                locale_config = self.locale_configs.get(locale_code)
                
                locales.append({
                    'code': locale_code,
                    'name': babel_locale.display_name,
                    'english_name': babel_locale.english_name,
                    'native_name': locale_config.native_name if locale_config else babel_locale.display_name,
                    'enabled': locale_config.enabled if locale_config else True
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to get info for locale {locale_code}: {e}")
        
        return sorted(locales, key=lambda x: x['english_name'])

    async def validate_locale(self, locale: str) -> bool:
        """Validate if a locale is supported."""
        try:
            normalized_locale = await self._normalize_locale(locale)
            return normalized_locale in SUPPORTED_LOCALES
        except Exception:
            return False

    async def detect_locale_from_accept_language(
        self,
        accept_language: str
    ) -> str:
        """
        Detect best matching locale from Accept-Language header.
        
        Args:
            accept_language: HTTP Accept-Language header value
            
        Returns:
            Best matching supported locale code
        """
        try:
            # Parse Accept-Language header
            locales_with_quality = []
            
            for item in accept_language.split(','):
                item = item.strip()
                if ';q=' in item:
                    locale_code, quality = item.split(';q=')
                    quality = float(quality)
                else:
                    locale_code = item
                    quality = 1.0
                
                # Normalize locale code
                locale_code = locale_code.strip().replace('-', '_')
                
                locales_with_quality.append((locale_code, quality))
            
            # Sort by quality score
            locales_with_quality.sort(key=lambda x: x[1], reverse=True)
            
            # Find best matching supported locale
            for locale_code, _ in locales_with_quality:
                # Exact match
                if locale_code in SUPPORTED_LOCALES:
                    return locale_code
                
                # Language-only match (e.g., 'en' -> 'en_US')
                language = locale_code.split('_')[0]
                for supported_locale in SUPPORTED_LOCALES:
                    if supported_locale.startswith(f"{language}_"):
                        return supported_locale
            
            # Fallback to default locale
            return DEFAULT_LOCALE
            
        except Exception as e:
            self.logger.warning(f"Failed to detect locale from Accept-Language: {e}")
            return DEFAULT_LOCALE

    async def reload_translations(self, locale: Optional[str] = None) -> None:
        """Reload translations from files."""
        try:
            if locale:
                # Reload specific locale
                await self._load_translations_for_locale(locale)
                self.logger.info(f"Reloaded translations for locale: {locale}")
            else:
                # Reload all locales
                self.translation_cache.clear()
                self.cache_timestamps.clear()
                await self._preload_translations()
                self.logger.info("Reloaded all translations")
                
        except Exception as e:
            self.logger.error(f"Failed to reload translations: {e}")
            raise LocaleError(f"Translation reload failed: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get translation cache statistics."""
        stats = {
            'cached_locales': len(self.translation_cache),
            'total_translations': sum(len(translations) for translations in self.translation_cache.values()),
            'cache_timestamps': {locale: timestamp.isoformat() for locale, timestamp in self.cache_timestamps.items()},
            'babel_locales_cached': len(self.babel_locales),
            'formatters_cached': len(self.formatters)
        }
        
        # Calculate cache hit/miss ratio if monitoring is enabled
        if self.performance_monitor:
            metrics = await self.performance_monitor.get_metrics()
            if 'translation_cache_hits' in metrics and 'translation_cache_misses' in metrics:
                hits = metrics['translation_cache_hits']
                misses = metrics['translation_cache_misses']
                total = hits + misses
                stats['cache_hit_ratio'] = hits / total if total > 0 else 0
        
        return stats

    # Private helper methods
    
    async def _normalize_locale(self, locale: str) -> str:
        """Normalize locale code to standard format."""
        try:
            # Replace hyphens with underscores
            normalized = locale.replace('-', '_')
            
            # Handle common variants
            locale_mappings = {
                'en': 'en_US',
                'es': 'es_ES',
                'fr': 'fr_FR',
                'de': 'de_DE',
                'it': 'it_IT',
                'pt': 'pt_BR',
                'zh': 'zh_CN',
                'ja': 'ja_JP',
                'ko': 'ko_KR',
                'ru': 'ru_RU',
                'ar': 'ar_SA',
                'hi': 'hi_IN'
            }
            
            if normalized in locale_mappings:
                normalized = locale_mappings[normalized]
            
            # Validate format
            parts = normalized.split('_')
            if len(parts) >= 2:
                language = parts[0].lower()
                territory = parts[1].upper()
                normalized = f"{language}_{territory}"
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"Failed to normalize locale {locale}: {e}")
            return DEFAULT_LOCALE

    async def _get_translation(
        self,
        key: str,
        locale: str,
        context: Optional[str] = None,
        fallback_message: Optional[str] = None
    ) -> Optional[TranslationMessage]:
        """Get translation with fallback mechanism."""
        # Try exact locale
        translation = await self._get_translation_from_cache(key, locale, context)
        if translation:
            return translation
        
        # Try fallback locales
        fallback_locales = FALLBACK_LOCALES.get(locale, [DEFAULT_LOCALE])
        for fallback_locale in fallback_locales:
            translation = await self._get_translation_from_cache(key, fallback_locale, context)
            if translation:
                return translation
        
        # Try language-only fallback
        language = locale.split('_')[0]
        if language != locale:
            for supported_locale in SUPPORTED_LOCALES:
                if supported_locale.startswith(f"{language}_"):
                    translation = await self._get_translation_from_cache(key, supported_locale, context)
                    if translation:
                        return translation
        
        return None

    async def _get_translation_from_cache(
        self,
        key: str,
        locale: str,
        context: Optional[str] = None
    ) -> Optional[TranslationMessage]:
        """Get translation from cache or load from file."""
        # Check if locale is cached
        if locale not in self.translation_cache:
            await self._load_translations_for_locale(locale)
        
        translations = self.translation_cache.get(locale, {})
        
        # Build cache key with context
        cache_key = f"{key}#{context}" if context else key
        
        return translations.get(cache_key)

    async def _load_translations_for_locale(self, locale: str) -> None:
        """Load translations for a specific locale from file."""
        try:
            locale_file = self.locales_path / f"{locale}.yaml"
            
            if not locale_file.exists():
                self.logger.warning(f"Translation file not found: {locale_file}")
                return
            
            with open(locale_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            translations = {}
            self._flatten_translations(data, translations, locale)
            
            self.translation_cache[locale] = translations
            self.cache_timestamps[locale] = datetime.utcnow()
            
            self.logger.debug(f"Loaded {len(translations)} translations for locale {locale}")
            
        except Exception as e:
            self.logger.error(f"Failed to load translations for locale {locale}: {e}")

    def _flatten_translations(
        self,
        data: Dict[str, Any],
        result: Dict[str, TranslationMessage],
        locale: str,
        prefix: str = ""
    ) -> None:
        """Recursively flatten nested translation data."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                if 'message' in value:
                    # Translation object with metadata
                    translation = TranslationMessage(
                        key=full_key,
                        locale=locale,
                        message=value['message'],
                        plural_forms=value.get('plural_forms'),
                        variables=value.get('variables'),
                        context=value.get('context')
                    )
                    
                    cache_key = f"{full_key}#{translation.context}" if translation.context else full_key
                    result[cache_key] = translation
                else:
                    # Nested object
                    self._flatten_translations(value, result, locale, full_key)
            elif isinstance(value, str):
                # Simple string translation
                translation = TranslationMessage(
                    key=full_key,
                    locale=locale,
                    message=value
                )
                result[full_key] = translation

    async def _get_plural_form(
        self,
        translation: TranslationMessage,
        locale: str,
        count: int
    ) -> str:
        """Get appropriate plural form for count."""
        if not translation.plural_forms:
            return translation.message
        
        try:
            # Get plural rule for locale
            plural_rule = self.plural_rules.get(locale)
            if not plural_rule:
                plural_rule = PluralRule({'one': 'n = 1'})  # Default rule
                self.plural_rules[locale] = plural_rule
            
            # Determine plural form
            plural_form = plural_rule(count)
            
            # Get message for plural form
            if plural_form in translation.plural_forms:
                return translation.plural_forms[plural_form]
            elif 'other' in translation.plural_forms:
                return translation.plural_forms['other']
            else:
                return translation.message
                
        except Exception as e:
            self.logger.warning(f"Failed to get plural form: {e}")
            return translation.message

    async def _interpolate_variables(
        self,
        message: str,
        variables: Dict[str, Any],
        locale: str
    ) -> str:
        """Interpolate variables in message with locale-aware formatting."""
        try:
            # Format variables according to locale
            formatted_variables = {}
            
            for key, value in variables.items():
                if isinstance(value, datetime):
                    formatted_variables[key] = await self.format_datetime(value, locale)
                elif isinstance(value, (int, float, Decimal)):
                    formatted_variables[key] = await self.format_number(value, locale)
                else:
                    formatted_variables[key] = str(value)
            
            # Use safe string formatting
            return message.format(**formatted_variables)
            
        except Exception as e:
            self.logger.warning(f"Failed to interpolate variables: {e}")
            return message

    async def _load_locale_configs(self) -> None:
        """Load locale configurations from file."""
        try:
            config_file = self.locales_path / "locales.yaml"
            
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                for locale_code, config_data in data.items():
                    self.locale_configs[locale_code] = LocaleConfig(
                        code=locale_code,
                        name=config_data.get('name', locale_code),
                        native_name=config_data.get('native_name', locale_code),
                        fallback=config_data.get('fallback', DEFAULT_LOCALE),
                        rtl=config_data.get('rtl', False),
                        date_format=config_data.get('date_format', 'medium'),
                        time_format=config_data.get('time_format', 'medium'),
                        number_format=config_data.get('number_format', 'decimal'),
                        currency_code=config_data.get('currency_code', 'USD'),
                        timezone=config_data.get('timezone', 'UTC'),
                        enabled=config_data.get('enabled', True)
                    )
            
        except Exception as e:
            self.logger.warning(f"Failed to load locale configs: {e}")

    async def _initialize_babel_locales(self) -> None:
        """Initialize Babel locale objects."""
        for locale_code in SUPPORTED_LOCALES:
            try:
                babel_locale = Locale.parse(locale_code)
                self.babel_locales[locale_code] = babel_locale
            except Exception as e:
                self.logger.warning(f"Failed to initialize Babel locale {locale_code}: {e}")

    async def _load_plural_rules(self) -> None:
        """Load plural rules for supported locales."""
        for locale_code in SUPPORTED_LOCALES:
            try:
                # Babel provides plural rules
                locale_obj = Locale.parse(locale_code)
                if hasattr(locale_obj, 'plural_form'):
                    self.plural_rules[locale_code] = locale_obj.plural_form
            except Exception as e:
                self.logger.warning(f"Failed to load plural rules for {locale_code}: {e}")

    async def _initialize_formatters(self) -> None:
        """Initialize format objects for performance."""
        for locale_code in SUPPORTED_LOCALES:
            try:
                babel_locale = self.babel_locales.get(locale_code)
                if babel_locale:
                    self.formatters[locale_code] = {
                        'date': Format(babel_locale),
                        'number': Format(babel_locale),
                        'currency': Format(babel_locale)
                    }
            except Exception as e:
                self.logger.warning(f"Failed to initialize formatters for {locale_code}: {e}")

    async def _preload_translations(self) -> None:
        """Preload translations for all supported locales."""
        tasks = []
        for locale_code in SUPPORTED_LOCALES:
            tasks.append(self._load_translations_for_locale(locale_code))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info(f"Preloaded translations for {len(self.translation_cache)} locales")

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self.performance_monitor:
                await self.performance_monitor.close()
            
            self.logger.info("LocaleManager closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing LocaleManager: {e}")
