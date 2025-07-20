"""
Advanced internationalization and localization schemas for multi-tenant systems.

This module provides comprehensive i18n/l10n support with AI-powered translation,
cultural adaptation, and dynamic content localization for alerting systems.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator
from pydantic.types import EmailStr


class SupportedLocale(str, Enum):
    """Supported locales with regional variants."""
    EN_US = "en_US"  # English (United States)
    EN_GB = "en_GB"  # English (United Kingdom)
    EN_CA = "en_CA"  # English (Canada)
    EN_AU = "en_AU"  # English (Australia)
    FR_FR = "fr_FR"  # French (France)
    FR_CA = "fr_CA"  # French (Canada)
    DE_DE = "de_DE"  # German (Germany)
    DE_AT = "de_AT"  # German (Austria)
    DE_CH = "de_CH"  # German (Switzerland)
    ES_ES = "es_ES"  # Spanish (Spain)
    ES_MX = "es_MX"  # Spanish (Mexico)
    ES_AR = "es_AR"  # Spanish (Argentina)
    IT_IT = "it_IT"  # Italian (Italy)
    PT_BR = "pt_BR"  # Portuguese (Brazil)
    PT_PT = "pt_PT"  # Portuguese (Portugal)
    NL_NL = "nl_NL"  # Dutch (Netherlands)
    SV_SE = "sv_SE"  # Swedish (Sweden)
    NO_NO = "no_NO"  # Norwegian (Norway)
    DA_DK = "da_DK"  # Danish (Denmark)
    FI_FI = "fi_FI"  # Finnish (Finland)
    PL_PL = "pl_PL"  # Polish (Poland)
    RU_RU = "ru_RU"  # Russian (Russia)
    JA_JP = "ja_JP"  # Japanese (Japan)
    KO_KR = "ko_KR"  # Korean (South Korea)
    ZH_CN = "zh_CN"  # Chinese (Simplified)
    ZH_TW = "zh_TW"  # Chinese (Traditional)
    AR_SA = "ar_SA"  # Arabic (Saudi Arabia)
    HE_IL = "he_IL"  # Hebrew (Israel)
    HI_IN = "hi_IN"  # Hindi (India)
    TH_TH = "th_TH"  # Thai (Thailand)
    VI_VN = "vi_VN"  # Vietnamese (Vietnam)


class TextDirection(str, Enum):
    """Text direction for proper UI rendering."""
    LTR = "ltr"  # Left-to-right
    RTL = "rtl"  # Right-to-left
    TTB = "ttb"  # Top-to-bottom (rare)


class DateFormat(str, Enum):
    """Standard date formats by region."""
    ISO_8601 = "YYYY-MM-DD"           # ISO standard
    US_FORMAT = "MM/DD/YYYY"          # US format
    EU_FORMAT = "DD/MM/YYYY"          # European format
    UK_FORMAT = "DD/MM/YYYY"          # UK format
    GERMAN_FORMAT = "DD.MM.YYYY"      # German format
    JAPANESE_FORMAT = "YYYY年MM月DD日"  # Japanese format
    CHINESE_FORMAT = "YYYY年MM月DD日"   # Chinese format


class TimeFormat(str, Enum):
    """Standard time formats."""
    FORMAT_24H = "HH:mm:ss"     # 24-hour format
    FORMAT_12H = "hh:mm:ss a"   # 12-hour format with AM/PM
    FORMAT_12H_SHORT = "h:mm a" # Short 12-hour format


class NumberFormat(str, Enum):
    """Number formatting styles."""
    DECIMAL_POINT = "1,234.56"    # US/UK style
    DECIMAL_COMMA = "1.234,56"    # European style
    SPACE_SEPARATOR = "1 234,56"  # French style
    INDIAN_STYLE = "1,23,456.78"  # Indian numbering


class CurrencyPosition(str, Enum):
    """Currency symbol positioning."""
    BEFORE = "before"  # $100.00
    AFTER = "after"    # 100.00$
    BEFORE_WITH_SPACE = "before_space"  # $ 100.00
    AFTER_WITH_SPACE = "after_space"    # 100.00 $


class LocaleConfigurationSchema(BaseModel):
    """Core locale configuration schema."""
    locale: SupportedLocale = Field(..., description="Primary locale identifier")
    language_code: str = Field(..., description="ISO 639-1 language code")
    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    script_code: Optional[str] = Field(None, description="ISO 15924 script code")
    
    # Display properties
    display_name: str = Field(..., description="Localized display name")
    native_name: str = Field(..., description="Native language name")
    text_direction: TextDirection = Field(TextDirection.LTR, description="Text direction")
    
    # Formatting preferences
    date_format: DateFormat = Field(..., description="Preferred date format")
    time_format: TimeFormat = Field(..., description="Preferred time format")
    number_format: NumberFormat = Field(..., description="Number formatting style")
    
    # Currency settings
    currency_code: str = Field(..., description="ISO 4217 currency code")
    currency_symbol: str = Field(..., description="Currency symbol")
    currency_position: CurrencyPosition = Field(..., description="Currency symbol position")
    
    # Timezone settings
    default_timezone: str = Field(..., description="Default timezone identifier")
    dst_supported: bool = Field(True, description="Daylight saving time support")
    
    # Calendar settings
    first_day_of_week: int = Field(1, ge=0, le=6, description="First day of week (0=Sunday)")
    calendar_type: str = Field("gregorian", description="Calendar system")
    
    class Config:
        schema_extra = {
            "example": {
                "locale": "en_US",
                "language_code": "en",
                "country_code": "US",
                "display_name": "English (United States)",
                "native_name": "English (United States)",
                "text_direction": "ltr",
                "date_format": "MM/DD/YYYY",
                "time_format": "hh:mm:ss a",
                "number_format": "1,234.56",
                "currency_code": "USD",
                "currency_symbol": "$",
                "currency_position": "before",
                "default_timezone": "America/New_York",
                "dst_supported": True,
                "first_day_of_week": 0,
                "calendar_type": "gregorian"
            }
        }


class TranslationSchema(BaseModel):
    """Schema for managing translations."""
    translation_id: UUID = Field(..., description="Unique translation identifier")
    key: str = Field(..., description="Translation key")
    source_locale: SupportedLocale = Field(..., description="Source locale")
    target_locale: SupportedLocale = Field(..., description="Target locale")
    source_text: str = Field(..., description="Original text")
    translated_text: str = Field(..., description="Translated text")
    
    # Translation metadata
    context: Optional[str] = Field(None, description="Translation context")
    category: str = Field("general", description="Translation category")
    priority: int = Field(1, ge=1, le=5, description="Translation priority")
    
    # Quality assurance
    reviewed: bool = Field(False, description="Translation has been reviewed")
    reviewer: Optional[EmailStr] = Field(None, description="Reviewer email")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score")
    
    # AI translation metadata
    ai_generated: bool = Field(False, description="AI-generated translation")
    ai_model: Optional[str] = Field(None, description="AI model used")
    ai_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI confidence score")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = Field(None, description="Review timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "translation_id": "550e8400-e29b-41d4-a716-446655440000",
                "key": "alert.cpu.warning.title",
                "source_locale": "en_US",
                "target_locale": "fr_FR",
                "source_text": "CPU usage warning",
                "translated_text": "Avertissement d'utilisation du processeur",
                "context": "System monitoring alert",
                "category": "alerts",
                "priority": 3,
                "reviewed": True,
                "reviewer": "translator@spotify.com",
                "quality_score": 0.95,
                "ai_generated": False
            }
        }


class MessageLocalizationSchema(BaseModel):
    """Schema for message localization in alerts."""
    message_id: UUID = Field(..., description="Unique message identifier")
    template_key: str = Field(..., description="Message template key")
    
    # Localized content
    localizations: Dict[SupportedLocale, Dict[str, Any]] = Field(
        ..., description="Localized message content by locale"
    )
    
    # Fallback configuration
    fallback_locale: SupportedLocale = Field(SupportedLocale.EN_US, description="Fallback locale")
    fallback_strategy: str = Field("cascade", description="Fallback strategy")
    
    # Personalization
    personalized: bool = Field(False, description="Message supports personalization")
    personalization_fields: List[str] = Field([], description="Personalizable fields")
    
    # Adaptive content
    adaptive_content: bool = Field(False, description="Content adapts to user preferences")
    adaptation_rules: Dict[str, Any] = Field({}, description="Content adaptation rules")
    
    # Accessibility
    accessibility_features: Dict[str, bool] = Field({}, description="Accessibility features")
    screen_reader_optimized: bool = Field(True, description="Optimized for screen readers")
    
    class Config:
        schema_extra = {
            "example": {
                "message_id": "660e8400-e29b-41d4-a716-446655440001",
                "template_key": "alert_notification_critical",
                "localizations": {
                    "en_US": {
                        "title": "🚨 Critical Alert: {alert_name}",
                        "message": "A critical issue has been detected in {service_name}. Immediate action required.",
                        "action_button": "View Details"
                    },
                    "fr_FR": {
                        "title": "🚨 Alerte Critique: {alert_name}",
                        "message": "Un problème critique a été détecté dans {service_name}. Action immédiate requise.",
                        "action_button": "Voir les Détails"
                    }
                },
                "fallback_locale": "en_US",
                "fallback_strategy": "cascade",
                "personalized": True,
                "personalization_fields": ["user_name", "preferred_timezone"],
                "accessibility_features": {
                    "high_contrast": True,
                    "large_text": True,
                    "voice_notification": True
                }
            }
        }


class CulturalAdaptationSchema(BaseModel):
    """Schema for cultural adaptation of content."""
    locale: SupportedLocale = Field(..., description="Target locale")
    
    # Cultural preferences
    color_preferences: Dict[str, str] = Field({}, description="Cultural color associations")
    icon_preferences: Dict[str, str] = Field({}, description="Preferred icons by context")
    communication_style: str = Field("direct", description="Communication style preference")
    
    # Business practices
    business_hours: Dict[str, str] = Field({}, description="Standard business hours")
    holiday_calendar: List[str] = Field([], description="Cultural holidays")
    work_week_pattern: List[int] = Field([1, 2, 3, 4, 5], description="Working days")
    
    # Notification preferences
    urgency_expressions: Dict[str, str] = Field({}, description="Urgency level expressions")
    politeness_level: str = Field("formal", description="Preferred politeness level")
    
    # Regulatory considerations
    data_protection_notices: bool = Field(True, description="Include data protection notices")
    consent_requirements: List[str] = Field([], description="Required consent types")
    
    class Config:
        schema_extra = {
            "example": {
                "locale": "ja_JP",
                "color_preferences": {
                    "warning": "#FF9800",
                    "error": "#F44336",
                    "success": "#4CAF50"
                },
                "icon_preferences": {
                    "warning": "⚠️",
                    "error": "❌",
                    "success": "✅"
                },
                "communication_style": "polite",
                "business_hours": {
                    "start": "09:00",
                    "end": "18:00"
                },
                "holiday_calendar": ["new_year", "golden_week", "obon"],
                "work_week_pattern": [1, 2, 3, 4, 5],
                "urgency_expressions": {
                    "low": "お知らせ",
                    "medium": "注意",
                    "high": "緊急",
                    "critical": "重大"
                },
                "politeness_level": "very_formal"
            }
        }


class AITranslationConfigSchema(BaseModel):
    """Schema for AI-powered translation configuration."""
    enabled: bool = Field(True, description="Enable AI translation")
    primary_model: str = Field("gpt-4", description="Primary translation model")
    fallback_model: str = Field("claude-3", description="Fallback translation model")
    
    # Model configuration
    model_config: Dict[str, Any] = Field({}, description="Model-specific configuration")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum confidence threshold")
    
    # Quality assurance
    auto_review_enabled: bool = Field(True, description="Enable automatic review")
    human_review_threshold: float = Field(0.9, description="Threshold for human review requirement")
    quality_gates: List[str] = Field([], description="Quality gate checks")
    
    # Performance optimization
    caching_enabled: bool = Field(True, description="Enable translation caching")
    batch_processing: bool = Field(True, description="Enable batch processing")
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    
    # Monitoring
    metrics_collection: bool = Field(True, description="Collect translation metrics")
    error_tracking: bool = Field(True, description="Track translation errors")
    
    class Config:
        schema_extra = {
            "example": {
                "enabled": True,
                "primary_model": "gpt-4",
                "fallback_model": "claude-3",
                "model_config": {
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "context_window": 4000
                },
                "confidence_threshold": 0.8,
                "auto_review_enabled": True,
                "human_review_threshold": 0.9,
                "quality_gates": ["grammar_check", "terminology_check", "cultural_check"],
                "caching_enabled": True,
                "batch_processing": True,
                "metrics_collection": True
            }
        }


class LocalizationConfigurationSchema(BaseModel):
    """Master schema for localization configuration."""
    config_id: UUID = Field(..., description="Unique configuration identifier")
    name: str = Field(..., description="Configuration name")
    description: str = Field(..., description="Configuration description")
    
    # Supported locales
    supported_locales: List[SupportedLocale] = Field(..., description="Supported locales")
    default_locale: SupportedLocale = Field(SupportedLocale.EN_US, description="Default locale")
    
    # Locale configurations
    locale_configs: Dict[SupportedLocale, LocaleConfigurationSchema] = Field(
        ..., description="Locale-specific configurations"
    )
    
    # Cultural adaptations
    cultural_adaptations: Dict[SupportedLocale, CulturalAdaptationSchema] = Field(
        {}, description="Cultural adaptation settings"
    )
    
    # AI translation
    ai_translation: AITranslationConfigSchema = Field(..., description="AI translation configuration")
    
    # Content management
    auto_fallback: bool = Field(True, description="Enable automatic fallback")
    dynamic_loading: bool = Field(True, description="Enable dynamic content loading")
    lazy_loading: bool = Field(True, description="Enable lazy loading of translations")
    
    # Performance settings
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")
    preload_locales: List[SupportedLocale] = Field([], description="Locales to preload")
    
    # Compliance
    gdpr_compliant: bool = Field(True, description="GDPR compliance enabled")
    data_retention_days: int = Field(365, description="Data retention period")
    
    # Monitoring
    usage_tracking: bool = Field(True, description="Track locale usage")
    performance_monitoring: bool = Field(True, description="Monitor performance")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('supported_locales')
    def validate_supported_locales(cls, v, values):
        """Validate that default locale is in supported locales."""
        default_locale = values.get('default_locale')
        if default_locale and default_locale not in v:
            v.append(default_locale)
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "config_id": "770e8400-e29b-41d4-a716-446655440002",
                "name": "Spotify AI Agent Localization",
                "description": "Comprehensive localization for global deployment",
                "supported_locales": ["en_US", "fr_FR", "de_DE", "ja_JP", "es_ES"],
                "default_locale": "en_US",
                "auto_fallback": True,
                "dynamic_loading": True,
                "cache_ttl": 3600,
                "preload_locales": ["en_US", "fr_FR"],
                "gdpr_compliant": True,
                "usage_tracking": True
            }
        }


# Export all schemas
__all__ = [
    "SupportedLocale",
    "TextDirection",
    "DateFormat",
    "TimeFormat",
    "NumberFormat",
    "CurrencyPosition",
    "LocaleConfigurationSchema",
    "TranslationSchema",
    "MessageLocalizationSchema",
    "CulturalAdaptationSchema",
    "AITranslationConfigSchema",
    "LocalizationConfigurationSchema"
]
