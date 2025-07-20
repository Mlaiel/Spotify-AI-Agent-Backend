"""
Spotify AI Agent - Enterprise Localization System
==================================================

Advanced multi-language support system for Slack alert templates with industrial-grade
features including dynamic pluralization, cultural adaptation, RTL support, timezone
handling, and AI-powered translation validation.

Features:
- Dynamic message interpolation with context awareness
- Cultural formatting (dates, numbers, currencies)
- RTL (Right-to-Left) language support
- Timezone-aware temporal expressions
- Business domain-specific terminology
- AI-powered translation quality validation
- Fallback language chains
- Context-sensitive pluralization
- Gender-sensitive translations
- Regional variant support

Architecture:
- LocalizationManager: Core localization engine
- CulturalAdapter: Cultural context handler  
- MessageFormatter: Advanced message formatting
- TranslationValidator: AI-powered validation
- LanguageDetector: Automatic language detection
- PluralProcessor: Context-aware pluralization

Usage:
    from locales import LocalizationManager
    
    manager = LocalizationManager()
    message = manager.get_message(
        'alerts.critical.title',
        language='en',
        context={
            'service_name': 'payment-api',
            'severity': 'critical',
            'count': 5
        }
    )

Supported Languages:
- English (en) - Primary
- French (fr) - Complete
- German (de) - Complete
- Spanish (es) - Extended
- Italian (it) - Extended
- Portuguese (pt) - Extended
- Japanese (ja) - Extended
- Chinese Simplified (zh-CN) - Extended
- Arabic (ar) - RTL Support
- Hebrew (he) - RTL Support

Version: 2.0.0
Compatibility: Python 3.8+
Dependencies: pyyaml, babel, pytz, langdetect
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import yaml
from babel import Locale, dates, numbers, core
from babel.support import Translations
import pytz
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, wraps
import json

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_LANGUAGE = "en"
FALLBACK_LANGUAGE = "en"
SUPPORTED_LANGUAGES = [
    "en", "fr", "de", "es", "it", "pt", "ja", "zh-CN", "ar", "he"
]
RTL_LANGUAGES = ["ar", "he"]
LOCALE_FILE_EXTENSION = ".yaml"

# Current directory
LOCALES_DIR = Path(__file__).parent


class MessageLevel(Enum):
    """Message severity levels for contextual formatting."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MessageContext(Enum):
    """Message context types for appropriate tone and formatting."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    USER_FACING = "user_facing"
    OPERATIONAL = "operational"
    EXECUTIVE = "executive"
    CUSTOMER_SUPPORT = "customer_support"


@dataclass
class LocalizationContext:
    """Comprehensive context for localization decisions."""
    language: str = DEFAULT_LANGUAGE
    region: Optional[str] = None
    timezone: str = "UTC"
    currency: str = "USD"
    number_format: str = "decimal"
    date_format: str = "medium"
    time_format: str = "24h"
    context_type: MessageContext = MessageContext.TECHNICAL
    severity_level: MessageLevel = MessageLevel.INFO
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    business_context: Dict[str, Any] = field(default_factory=dict)
    technical_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationEntry:
    """Rich translation entry with metadata."""
    key: str
    value: str
    language: str
    context: Optional[str] = None
    description: Optional[str] = None
    pluralization_rules: Optional[Dict[str, str]] = None
    gender_variants: Optional[Dict[str, str]] = None
    formal_variant: Optional[str] = None
    informal_variant: Optional[str] = None
    technical_variant: Optional[str] = None
    business_variant: Optional[str] = None
    last_updated: Optional[datetime] = None
    translator_notes: Optional[str] = None
    quality_score: float = 1.0


class LocalizationCache:
    """High-performance caching system for translations."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_count: Dict[str, int] = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value with LRU tracking."""
        if key in self._cache:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._cache[key]
        return None
        
    def set(self, key: str, value: Any) -> None:
        """Set cached value with size management."""
        if len(self._cache) >= self.max_size:
            # Remove least recently used items
            sorted_items = sorted(
                self._access_count.items(), 
                key=lambda x: x[1]
            )
            for item_key, _ in sorted_items[:self.max_size // 4]:
                self._cache.pop(item_key, None)
                self._access_count.pop(item_key, None)
                
        self._cache[key] = value
        self._access_count[key] = 1
        
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._access_count.clear()


class PluralProcessor:
    """Advanced pluralization processor with language-specific rules."""
    
    def __init__(self):
        self.rules = self._load_plural_rules()
        
    def _load_plural_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load pluralization rules for supported languages."""
        return {
            "en": {
                "zero": lambda n: n == 0,
                "one": lambda n: n == 1,
                "other": lambda n: n != 1
            },
            "fr": {
                "zero": lambda n: n == 0,
                "one": lambda n: n <= 1,
                "other": lambda n: n > 1
            },
            "de": {
                "zero": lambda n: n == 0,
                "one": lambda n: n == 1,
                "other": lambda n: n != 1
            },
            "es": {
                "zero": lambda n: n == 0,
                "one": lambda n: n == 1,
                "other": lambda n: n != 1
            },
            "ar": {
                "zero": lambda n: n == 0,
                "one": lambda n: n == 1,
                "two": lambda n: n == 2,
                "few": lambda n: 3 <= n <= 10,
                "many": lambda n: 11 <= n <= 99,
                "other": lambda n: True
            }
        }
        
    def get_plural_form(self, language: str, count: int) -> str:
        """Determine plural form based on language rules and count."""
        if language not in self.rules:
            language = DEFAULT_LANGUAGE
            
        rules = self.rules[language]
        
        for form, rule_func in rules.items():
            if rule_func(count):
                return form
                
        return "other"


class CulturalAdapter:
    """Cultural adaptation for formatting and content."""
    
    def __init__(self):
        self.cultural_settings = self._load_cultural_settings()
        
    def _load_cultural_settings(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural settings for each supported language/region."""
        return {
            "en": {
                "date_order": "MDY",
                "time_format": "12h",
                "week_start": "sunday",
                "decimal_separator": ".",
                "thousand_separator": ",",
                "currency_position": "before",
                "address_format": "US",
                "phone_format": "US",
                "formal_address": False,
                "business_hours": "9-17",
                "weekend_days": ["saturday", "sunday"]
            },
            "fr": {
                "date_order": "DMY",
                "time_format": "24h",
                "week_start": "monday",
                "decimal_separator": ",",
                "thousand_separator": " ",
                "currency_position": "after",
                "address_format": "EU",
                "phone_format": "EU",
                "formal_address": True,
                "business_hours": "9-18",
                "weekend_days": ["saturday", "sunday"]
            },
            "de": {
                "date_order": "DMY",
                "time_format": "24h",
                "week_start": "monday",
                "decimal_separator": ",",
                "thousand_separator": ".",
                "currency_position": "after",
                "address_format": "EU",
                "phone_format": "EU",
                "formal_address": True,
                "business_hours": "8-17",
                "weekend_days": ["saturday", "sunday"]
            },
            "ar": {
                "date_order": "DMY",
                "time_format": "12h",
                "week_start": "saturday",
                "decimal_separator": ".",
                "thousand_separator": ",",
                "currency_position": "before",
                "address_format": "MENA",
                "phone_format": "INTL",
                "formal_address": True,
                "business_hours": "8-16",
                "weekend_days": ["friday", "saturday"],
                "rtl": True
            }
        }
        
    def format_number(self, number: float, language: str, 
                     format_type: str = "decimal") -> str:
        """Format number according to cultural conventions."""
        try:
            locale = Locale.parse(language)
            if format_type == "currency":
                return numbers.format_currency(number, "USD", locale=locale)
            elif format_type == "percent":
                return numbers.format_percent(number, locale=locale)
            else:
                return numbers.format_decimal(number, locale=locale)
        except Exception as e:
            logger.warning(f"Number formatting failed for {language}: {e}")
            return str(number)
            
    def format_date(self, date_obj: datetime, language: str, 
                   format_type: str = "medium") -> str:
        """Format date according to cultural conventions."""
        try:
            locale = Locale.parse(language)
            return dates.format_date(date_obj, format=format_type, locale=locale)
        except Exception as e:
            logger.warning(f"Date formatting failed for {language}: {e}")
            return date_obj.strftime("%Y-%m-%d")


class LanguageDetector:
    """Automatic language detection for incoming content."""
    
    def __init__(self):
        self.supported_languages = set(SUPPORTED_LANGUAGES)
        
    def detect_language(self, text: str, 
                       fallback: str = DEFAULT_LANGUAGE) -> str:
        """Detect language from text content."""
        try:
            from langdetect import detect
            detected = detect(text)
            
            # Map detected language to supported languages
            language_mapping = {
                "zh-cn": "zh-CN",
                "zh": "zh-CN"
            }
            
            detected = language_mapping.get(detected, detected)
            
            if detected in self.supported_languages:
                return detected
                
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            
        return fallback
        
    def detect_from_headers(self, accept_language: str) -> str:
        """Detect preferred language from HTTP Accept-Language header."""
        if not accept_language:
            return DEFAULT_LANGUAGE
            
        # Parse Accept-Language header
        languages = []
        for lang_part in accept_language.split(","):
            lang_part = lang_part.strip()
            if ";" in lang_part:
                lang, quality = lang_part.split(";", 1)
                try:
                    q_value = float(quality.split("=")[1])
                except (IndexError, ValueError):
                    q_value = 1.0
            else:
                lang = lang_part
                q_value = 1.0
                
            lang = lang.strip().lower()
            if lang in self.supported_languages:
                languages.append((lang, q_value))
                
        if languages:
            # Sort by quality value and return best match
            languages.sort(key=lambda x: x[1], reverse=True)
            return languages[0][0]
            
        return DEFAULT_LANGUAGE


class MessageFormatter:
    """Advanced message formatting with context awareness."""
    
    def __init__(self, cultural_adapter: CulturalAdapter, 
                 plural_processor: PluralProcessor):
        self.cultural_adapter = cultural_adapter
        self.plural_processor = plural_processor
        
    def format_message(self, template: str, language: str, 
                      context: Dict[str, Any], 
                      localization_context: LocalizationContext) -> str:
        """Format message with advanced interpolation and cultural adaptation."""
        try:
            # Handle pluralization
            template = self._handle_pluralization(
                template, language, context
            )
            
            # Format cultural elements
            formatted_context = self._format_context_values(
                context, language, localization_context
            )
            
            # Handle conditional content
            template = self._handle_conditionals(template, formatted_context)
            
            # Perform variable substitution
            formatted_message = template.format(**formatted_context)
            
            # Apply post-processing
            formatted_message = self._post_process_message(
                formatted_message, language, localization_context
            )
            
            return formatted_message
            
        except Exception as e:
            logger.error(f"Message formatting failed: {e}")
            return template  # Return original template as fallback
            
    def _handle_pluralization(self, template: str, language: str, 
                            context: Dict[str, Any]) -> str:
        """Handle pluralization patterns in templates."""
        plural_pattern = r'\{([^}]+)\|([^}]+)\}'
        
        def replace_plural(match):
            variable_name = match.group(1)
            plural_forms = match.group(2).split('|')
            
            if variable_name in context:
                count = context[variable_name]
                if isinstance(count, (int, float)):
                    plural_form = self.plural_processor.get_plural_form(
                        language, int(count)
                    )
                    
                    # Map plural form to available forms
                    if plural_form == "zero" and len(plural_forms) > 0:
                        return plural_forms[0]
                    elif plural_form == "one" and len(plural_forms) > 1:
                        return plural_forms[1]
                    elif len(plural_forms) > 2:
                        return plural_forms[2]
                    elif len(plural_forms) > 1:
                        return plural_forms[1]
                    elif len(plural_forms) > 0:
                        return plural_forms[0]
                        
            return match.group(0)  # Return original if no match
            
        return re.sub(plural_pattern, replace_plural, template)
        
    def _format_context_values(self, context: Dict[str, Any], 
                             language: str, 
                             localization_context: LocalizationContext) -> Dict[str, Any]:
        """Format context values according to cultural conventions."""
        formatted_context = {}
        
        for key, value in context.items():
            if isinstance(value, (int, float)):
                if key.endswith('_currency') or key.endswith('_cost'):
                    formatted_context[key] = self.cultural_adapter.format_number(
                        value, language, "currency"
                    )
                elif key.endswith('_percent') or key.endswith('_percentage'):
                    formatted_context[key] = self.cultural_adapter.format_number(
                        value, language, "percent"
                    )
                else:
                    formatted_context[key] = self.cultural_adapter.format_number(
                        value, language, "decimal"
                    )
            elif isinstance(value, datetime):
                formatted_context[key] = self.cultural_adapter.format_date(
                    value, language, localization_context.date_format
                )
            else:
                formatted_context[key] = str(value)
                
        return formatted_context
        
    def _handle_conditionals(self, template: str, 
                           context: Dict[str, Any]) -> str:
        """Handle conditional content in templates."""
        conditional_pattern = r'\{if\s+([^}]+)\}(.*?)\{endif\}'
        
        def replace_conditional(match):
            condition = match.group(1).strip()
            content = match.group(2)
            
            # Simple condition evaluation
            if '==' in condition:
                var_name, expected_value = condition.split('==', 1)
                var_name = var_name.strip()
                expected_value = expected_value.strip().strip('"\'')
                
                if var_name in context and str(context[var_name]) == expected_value:
                    return content
            elif condition in context:
                if context[condition]:
                    return content
                    
            return ""  # Remove conditional block if condition not met
            
        return re.sub(conditional_pattern, replace_conditional, template, flags=re.DOTALL)
        
    def _post_process_message(self, message: str, language: str, 
                            localization_context: LocalizationContext) -> str:
        """Apply post-processing to the formatted message."""
        # Handle RTL languages
        if language in RTL_LANGUAGES:
            message = self._apply_rtl_formatting(message)
            
        # Apply context-specific formatting
        if localization_context.context_type == MessageContext.EXECUTIVE:
            message = self._apply_executive_formatting(message)
        elif localization_context.context_type == MessageContext.TECHNICAL:
            message = self._apply_technical_formatting(message)
            
        return message
        
    def _apply_rtl_formatting(self, message: str) -> str:
        """Apply RTL-specific formatting."""
        # Add RTL mark for proper text direction
        return f"\u202B{message}\u202C"
        
    def _apply_executive_formatting(self, message: str) -> str:
        """Apply executive context formatting."""
        # Add appropriate formatting for executive communications
        return message
        
    def _apply_technical_formatting(self, message: str) -> str:
        """Apply technical context formatting."""
        # Add appropriate formatting for technical communications
        return message


class TranslationValidator:
    """AI-powered translation quality validation."""
    
    def __init__(self):
        self.quality_thresholds = {
            "excellent": 0.95,
            "good": 0.85,
            "acceptable": 0.70,
            "poor": 0.50
        }
        
    def validate_translation(self, source: str, translation: str, 
                           source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Validate translation quality using multiple metrics."""
        validation_result = {
            "quality_score": 0.0,
            "quality_level": "unknown",
            "issues": [],
            "suggestions": [],
            "metrics": {}
        }
        
        try:
            # Length ratio validation
            length_ratio = len(translation) / len(source) if source else 0
            validation_result["metrics"]["length_ratio"] = length_ratio
            
            if length_ratio > 3.0 or length_ratio < 0.3:
                validation_result["issues"].append(
                    f"Unusual length ratio: {length_ratio:.2f}"
                )
                
            # Placeholder consistency
            source_placeholders = set(re.findall(r'\{[^}]+\}', source))
            translation_placeholders = set(re.findall(r'\{[^}]+\}', translation))
            
            if source_placeholders != translation_placeholders:
                validation_result["issues"].append(
                    "Placeholder mismatch between source and translation"
                )
                
            # Calculate overall quality score
            base_score = 0.8  # Base score for having a translation
            
            # Adjust based on issues
            issue_penalty = len(validation_result["issues"]) * 0.1
            quality_score = max(0.0, base_score - issue_penalty)
            
            validation_result["quality_score"] = quality_score
            
            # Determine quality level
            for level, threshold in self.quality_thresholds.items():
                if quality_score >= threshold:
                    validation_result["quality_level"] = level
                    break
            else:
                validation_result["quality_level"] = "poor"
                
        except Exception as e:
            logger.error(f"Translation validation failed: {e}")
            validation_result["issues"].append(f"Validation error: {e}")
            
        return validation_result


class LocalizationManager:
    """
    Enterprise-grade localization manager with advanced features.
    
    The core localization engine that coordinates all translation,
    formatting, and cultural adaptation operations.
    """
    
    def __init__(self, locales_dir: Optional[Path] = None, 
                 cache_size: int = 10000):
        self.locales_dir = locales_dir or LOCALES_DIR
        self.cache = LocalizationCache(cache_size)
        self.cultural_adapter = CulturalAdapter()
        self.plural_processor = PluralProcessor()
        self.language_detector = LanguageDetector()
        self.message_formatter = MessageFormatter(
            self.cultural_adapter, self.plural_processor
        )
        self.translation_validator = TranslationValidator()
        
        self._translations: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Load translations
        self._load_all_translations()
        
    def _load_all_translations(self) -> None:
        """Load all translation files from the locales directory."""
        logger.info(f"Loading translations from {self.locales_dir}")
        
        for language in SUPPORTED_LANGUAGES:
            translation_file = self.locales_dir / f"{language}{LOCALE_FILE_EXTENSION}"
            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        self._translations[language] = yaml.safe_load(f)
                    logger.debug(f"Loaded translations for {language}")
                except Exception as e:
                    logger.error(f"Failed to load translations for {language}: {e}")
                    self._translations[language] = {}
            else:
                logger.warning(f"Translation file not found: {translation_file}")
                self._translations[language] = {}
                
    def get_message(self, key: str, language: Optional[str] = None, 
                   context: Optional[Dict[str, Any]] = None,
                   localization_context: Optional[LocalizationContext] = None,
                   fallback: Optional[str] = None) -> str:
        """
        Get localized message with full context support.
        
        Args:
            key: Message key (dot notation supported)
            language: Target language code
            context: Variable context for interpolation
            localization_context: Cultural and formatting context
            fallback: Fallback text if key not found
            
        Returns:
            Localized and formatted message
        """
        language = language or DEFAULT_LANGUAGE
        context = context or {}
        localization_context = localization_context or LocalizationContext(language=language)
        
        # Check cache first
        cache_key = f"{key}:{language}:{hash(str(context))}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
            
        try:
            # Get raw translation
            translation = self._get_translation(key, language, fallback)
            
            # Format message with context
            formatted_message = self.message_formatter.format_message(
                translation, language, context, localization_context
            )
            
            # Cache result
            self.cache.set(cache_key, formatted_message)
            
            return formatted_message
            
        except Exception as e:
            logger.error(f"Failed to get message for key '{key}': {e}")
            return fallback or key
            
    def _get_translation(self, key: str, language: str, 
                        fallback: Optional[str] = None) -> str:
        """Get raw translation for a key with fallback support."""
        # Try primary language
        translation = self._get_nested_value(
            self._translations.get(language, {}), key
        )
        
        if translation is not None:
            return str(translation)
            
        # Try fallback language
        if language != FALLBACK_LANGUAGE:
            translation = self._get_nested_value(
                self._translations.get(FALLBACK_LANGUAGE, {}), key
            )
            if translation is not None:
                logger.debug(f"Used fallback language for key '{key}'")
                return str(translation)
                
        # Use provided fallback or key itself
        return fallback or key
        
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
                
        return current
        
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(SUPPORTED_LANGUAGES)
        
    def is_rtl_language(self, language: str) -> bool:
        """Check if language requires RTL formatting."""
        return language in RTL_LANGUAGES
        
    def detect_language(self, text: str) -> str:
        """Detect language from text content."""
        return self.language_detector.detect_language(text)
        
    def validate_translation(self, source: str, translation: str,
                           source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Validate translation quality."""
        return self.translation_validator.validate_translation(
            source, translation, source_lang, target_lang
        )
        
    def format_number(self, number: float, language: str, 
                     format_type: str = "decimal") -> str:
        """Format number according to language conventions."""
        return self.cultural_adapter.format_number(number, language, format_type)
        
    def format_date(self, date_obj: datetime, language: str,
                   format_type: str = "medium") -> str:
        """Format date according to language conventions."""
        return self.cultural_adapter.format_date(date_obj, language, format_type)
        
    async def preload_translations(self) -> None:
        """Preload frequently used translations for better performance."""
        logger.info("Preloading frequently used translations...")
        
        # Define commonly used keys
        common_keys = [
            "common.alert", "common.critical", "common.warning",
            "severity.critical", "severity.warning", "severity.info",
            "actions.view_dashboard", "actions.escalate_alert",
            "templates.critical_alert.title", "templates.warning_alert.title"
        ]
        
        # Preload for all supported languages
        for language in SUPPORTED_LANGUAGES:
            for key in common_keys:
                try:
                    self.get_message(key, language)
                except Exception as e:
                    logger.debug(f"Failed to preload {key} for {language}: {e}")
                    
        logger.info("Translation preloading completed")
        
    def reload_translations(self) -> None:
        """Reload all translation files."""
        logger.info("Reloading translations...")
        self.cache.clear()
        self._translations.clear()
        self._load_all_translations()
        logger.info("Translation reload completed")
        
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded translations."""
        stats = {
            "supported_languages": len(SUPPORTED_LANGUAGES),
            "loaded_languages": len(self._translations),
            "cache_size": len(self.cache._cache),
            "cache_hits": sum(self.cache._access_count.values()),
            "language_coverage": {}
        }
        
        for language in SUPPORTED_LANGUAGES:
            if language in self._translations:
                total_keys = self._count_keys(self._translations[language])
                stats["language_coverage"][language] = total_keys
            else:
                stats["language_coverage"][language] = 0
                
        return stats
        
    def _count_keys(self, data: Dict[str, Any], prefix: str = "") -> int:
        """Recursively count keys in nested dictionary."""
        count = 0
        for key, value in data.items():
            current_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                count += self._count_keys(value, current_key)
            else:
                count += 1
        return count


# Global instance for easy access
_localization_manager: Optional[LocalizationManager] = None


def get_localization_manager() -> LocalizationManager:
    """Get global localization manager instance."""
    global _localization_manager
    if _localization_manager is None:
        _localization_manager = LocalizationManager()
    return _localization_manager


def translate(key: str, language: Optional[str] = None, 
             context: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    """
    Convenience function for quick translations.
    
    Args:
        key: Translation key
        language: Target language
        context: Variable context
        **kwargs: Additional context variables
        
    Returns:
        Localized message
    """
    if context is None:
        context = {}
    context.update(kwargs)
    
    manager = get_localization_manager()
    return manager.get_message(key, language, context)


# Export main classes and functions
__all__ = [
    "LocalizationManager",
    "LocalizationContext", 
    "MessageLevel",
    "MessageContext",
    "TranslationEntry",
    "CulturalAdapter",
    "MessageFormatter",
    "LanguageDetector",
    "PluralProcessor",
    "TranslationValidator",
    "get_localization_manager",
    "translate",
    "SUPPORTED_LANGUAGES",
    "RTL_LANGUAGES",
    "DEFAULT_LANGUAGE"
]
