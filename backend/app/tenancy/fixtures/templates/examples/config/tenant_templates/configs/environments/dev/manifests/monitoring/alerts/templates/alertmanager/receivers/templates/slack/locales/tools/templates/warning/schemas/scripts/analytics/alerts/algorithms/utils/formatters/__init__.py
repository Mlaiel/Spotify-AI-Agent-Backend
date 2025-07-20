"""
Spotify AI Agent - Ultra-Advanced Formatters Module
=================================================

Industrialized, enterprise-grade formatters for multi-channel communications,
business intelligence, real-time analytics, and comprehensive data presentation.

This module provides ultra-advanced formatting capabilities for:
- Multi-channel alert systems (Slack, Email, SMS, Teams, PagerDuty)
- Business intelligence dashboards and executive reports
- Real-time streaming analytics and metrics visualization
- Multi-language localization with cultural adaptation
- Audio/video content processing and interactive media
- Advanced metrics formatting for multiple platforms
- Template-driven content generation with dynamic adaptation
- AI/ML model performance monitoring and explainability
- Security incident response and threat intelligence
- Advanced template engine with dynamic content generation

Author: Fahed Mlaiel & Spotify Engineering Team
Version: 2.1.0
License: MIT
"""

from typing import Dict, Any, Optional, List
import logging

# Core formatter classes
from .alert_formatter import (
    BaseAlertFormatter,
    SlackAlertFormatter,
    EmailAlertFormatter,
    SMSAlertFormatter,
    TeamsAlertFormatter,
    PagerDutyAlertFormatter,
    create_alert_formatter
)

from .metrics_formatter import (
    BaseMetricsFormatter,
    PrometheusMetricsFormatter,
    GrafanaMetricsFormatter,
    InfluxDBMetricsFormatter,
    ElasticsearchMetricsFormatter,
    DatadogMetricsFormatter,
    create_metrics_formatter
)

from .business_formatter import (
    BaseBusinessFormatter,
    BusinessIntelligenceFormatter,
    SpotifyArtistFormatter,
    PlaylistAnalyticsFormatter,
    create_business_formatter
)

from .streaming_formatter import (
    BaseStreamingFormatter,
    RealTimeAnalyticsFormatter,
    LiveUserMetricsFormatter,
    StreamingMLMetricsFormatter,
    create_streaming_formatter
)

from .media_formatter import (
    BaseMediaFormatter,
    AudioVisualizationFormatter,
    VideoContentFormatter,
    create_media_formatter
)

from .localization_formatter import (
    BaseLocalizationFormatter,
    SpotifyMultiLanguageFormatter,
    AccessibilityLocalizationFormatter,
    create_localization_formatter
)

from .ai_model_formatter import (
    BaseAIModelFormatter,
    SpotifyRecommendationModelFormatter,
    GeneralMLModelFormatter,
    create_ai_model_formatter
)

from .security_formatter import (
    BaseSecurityFormatter,
    SecurityIncidentFormatter,
    VulnerabilityFormatter,
    create_security_formatter
)

from .template_engine_formatter import (
    SpotifyTemplateEngine,
    create_template_formatter
)

# Version information
__version__ = "2.1.0"
__author__ = "Fahed Mlaiel & Spotify Engineering Team"
__license__ = "MIT"

# Formatter registry for factory pattern
FORMATTER_REGISTRY: Dict[str, Any] = {
    # Alert formatters
    'alert': BaseAlertFormatter,
    'slack': SlackAlertFormatter,
    'email': EmailAlertFormatter,
    'sms': SMSAlertFormatter,
    'teams': TeamsAlertFormatter,
    'pagerduty': PagerDutyAlertFormatter,
    
    # Metrics formatters
    'metrics': BaseMetricsFormatter,
    'prometheus': PrometheusMetricsFormatter,
    'grafana': GrafanaMetricsFormatter,
    'influxdb': InfluxDBMetricsFormatter,
    'elasticsearch': ElasticsearchMetricsFormatter,
    'datadog': DatadogMetricsFormatter,
    
    # Business intelligence formatters
    'business': BaseBusinessFormatter,
    'bi': BusinessIntelligenceFormatter,
    'artist': SpotifyArtistFormatter,
    'playlist': PlaylistAnalyticsFormatter,
    
    # Streaming formatters
    'streaming': BaseStreamingFormatter,
    'realtime': RealTimeAnalyticsFormatter,
    'live_metrics': LiveUserMetricsFormatter,
    'ml_streaming': StreamingMLMetricsFormatter,
    
    # Media formatters
    'media': BaseMediaFormatter,
    'audio': AudioVisualizationFormatter,
    'video': VideoContentFormatter,
    
    # Localization formatters
    'localization': BaseLocalizationFormatter,
    'i18n': SpotifyMultiLanguageFormatter,
    'l10n': SpotifyMultiLanguageFormatter,
    'accessibility': AccessibilityLocalizationFormatter,
    'a11y': AccessibilityLocalizationFormatter,
    
    # AI/ML model formatters
    'ai_model': BaseAIModelFormatter,
    'ml_model': GeneralMLModelFormatter,
    'recommendation': SpotifyRecommendationModelFormatter,
    'model_performance': BaseAIModelFormatter,
    
    # Security formatters
    'security': BaseSecurityFormatter,
    'incident': SecurityIncidentFormatter,
    'vulnerability': VulnerabilityFormatter,
    'security_incident': SecurityIncidentFormatter,
    'threat': SecurityIncidentFormatter,
    
    # Template engine
    'template': SpotifyTemplateEngine,
    'template_engine': SpotifyTemplateEngine,
    'dynamic_content': SpotifyTemplateEngine,
}

# Factory functions registry
FACTORY_FUNCTIONS: Dict[str, Any] = {
    'alert': create_alert_formatter,
    'metrics': create_metrics_formatter,
    'business': create_business_formatter,
    'streaming': create_streaming_formatter,
    'media': create_media_formatter,
    'localization': create_localization_formatter,
    'ai_model': create_ai_model_formatter,
    'security': create_security_formatter,
    'template': create_template_formatter,
}


def get_formatter(formatter_type: str, 
                 tenant_id: str, 
                 config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Factory function to get appropriate formatter instance.
    
    Args:
        formatter_type: Type of formatter to create
        tenant_id: Tenant identifier
        config: Optional configuration dictionary
        
    Returns:
        Configured formatter instance
        
    Raises:
        ValueError: If formatter type is not supported
    """
    if formatter_type not in FORMATTER_REGISTRY:
        available_types = ", ".join(sorted(FORMATTER_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported formatter type: {formatter_type}. "
            f"Available types: {available_types}"
        )
    
    formatter_class = FORMATTER_REGISTRY[formatter_type]
    return formatter_class(tenant_id, config or {})


def get_factory_function(category: str) -> Optional[Any]:
    """
    Get factory function for formatter category.
    
    Args:
        category: Formatter category (alert, metrics, business, etc.)
        
    Returns:
        Factory function or None if not found
    """
    return FACTORY_FUNCTIONS.get(category)


def list_available_formatters() -> Dict[str, List[str]]:
    """
    List all available formatters organized by category.
    
    Returns:
        Dictionary with categories and their available formatters
    """
    categories = {
        'alert': ['alert', 'slack', 'email', 'sms', 'teams', 'pagerduty'],
        'metrics': ['metrics', 'prometheus', 'grafana', 'influxdb', 'elasticsearch', 'datadog'],
        'business': ['business', 'bi', 'artist', 'playlist'],
        'streaming': ['streaming', 'realtime', 'live_metrics', 'ml_streaming'],
        'media': ['media', 'audio', 'video'],
        'localization': ['localization', 'i18n', 'l10n', 'accessibility', 'a11y'],
        'ai_model': ['ai_model', 'ml_model', 'recommendation', 'model_performance'],
        'security': ['security', 'incident', 'vulnerability', 'security_incident', 'threat'],
        'template': ['template', 'template_engine', 'dynamic_content']
    }
    return categories


def get_formatter_metadata() -> Dict[str, Any]:
    """
    Get metadata about the formatters module.
    
    Returns:
        Dictionary containing module metadata
    """
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "total_formatters": len(FORMATTER_REGISTRY),
        "categories": len(FACTORY_FUNCTIONS),
        "supported_outputs": [
            "Slack messages", "Email HTML/text", "SMS text", "Teams cards",
            "PagerDuty incidents", "Prometheus metrics", "Grafana panels",
            "InfluxDB line protocol", "Elasticsearch documents", "Datadog metrics",
            "Business intelligence reports", "Artist analytics", "Playlist metrics",
            "Real-time streaming data", "Audio visualizations", "Video analytics",
            "Multi-language content", "Accessibility features", "AI model reports",
            "Security incidents", "Vulnerability assessments", "Dynamic templates"
        ],
        "features": [
            "Multi-channel communication", "Real-time analytics", "Business intelligence",
            "Multi-language support", "Accessibility compliance", "Template engine",
            "A/B testing support", "Caching and performance optimization",
            "Security and sanitization", "AI/ML model monitoring",
            "Threat intelligence", "Dynamic content generation"
        ]
    }


# Module-level exports
__all__ = [
    # Core classes
    'BaseAlertFormatter', 'SlackAlertFormatter', 'EmailAlertFormatter',
    'SMSAlertFormatter', 'TeamsAlertFormatter', 'PagerDutyAlertFormatter',
    'BaseMetricsFormatter', 'PrometheusMetricsFormatter', 'GrafanaMetricsFormatter',
    'InfluxDBMetricsFormatter', 'ElasticsearchMetricsFormatter', 'DatadogMetricsFormatter',
    'BaseBusinessFormatter', 'BusinessIntelligenceFormatter', 'SpotifyArtistFormatter',
    'PlaylistAnalyticsFormatter', 'BaseStreamingFormatter', 'RealTimeAnalyticsFormatter',
    'LiveUserMetricsFormatter', 'StreamingMLMetricsFormatter', 'BaseMediaFormatter',
    'AudioVisualizationFormatter', 'VideoContentFormatter', 'BaseLocalizationFormatter',
    'SpotifyMultiLanguageFormatter', 'AccessibilityLocalizationFormatter',
    'BaseAIModelFormatter', 'SpotifyRecommendationModelFormatter', 'GeneralMLModelFormatter',
    'BaseSecurityFormatter', 'SecurityIncidentFormatter', 'VulnerabilityFormatter',
    'SpotifyTemplateEngine',
    
    # Factory functions
    'create_alert_formatter', 'create_metrics_formatter', 'create_business_formatter',
    'create_streaming_formatter', 'create_media_formatter', 'create_localization_formatter',
    'create_ai_model_formatter', 'create_security_formatter', 'create_template_formatter',
    
    # Utility functions
    'get_formatter', 'get_factory_function', 'list_available_formatters',
    'get_formatter_metadata',
    
    # Registries
    'FORMATTER_REGISTRY', 'FACTORY_FUNCTIONS',
    
    # Metadata
    '__version__', '__author__', '__license__'
]

# Initialize logging for the module
logging.getLogger(__name__).info(
    f"Spotify AI Agent Formatters Module v{__version__} initialized with "
    f"{len(FORMATTER_REGISTRY)} formatters across {len(FACTORY_FUNCTIONS)} categories"
)

# Core formatters
from .alert_formatter import (
    AlertFormatter,
    SlackAlertFormatter,
    EmailAlertFormatter,
    SMSAlertFormatter,
    TeamsAlertFormatter,
    PagerDutyAlertFormatter
)

# Metrics formatters
from .metrics_formatter import (
    MetricsFormatter,
    PrometheusMetricsFormatter,
    GrafanaMetricsFormatter,
    InfluxDBMetricsFormatter,
    ElasticsearchMetricsFormatter,
    CustomMetricsFormatter
)

# Business Intelligence formatters
from .business_formatter import (
    BusinessIntelligenceFormatter,
    SpotifyArtistFormatter,
    PlaylistAnalyticsFormatter,
    RevenueReportFormatter,
    UserEngagementFormatter,
    MLModelPerformanceFormatter
)

# Streaming and real-time formatters
from .streaming_formatter import (
    StreamingFormatter,
    RealTimeEventFormatter,
    WebSocketMessageFormatter,
    SSEFormatter,
    MQTTMessageFormatter,
    KafkaEventFormatter
)

# Rich media formatters
from .media_formatter import (
    MediaFormatter,
    AudioTrackFormatter,
    PlaylistFormatter,
    ArtistProfileFormatter,
    AlbumFormatter,
    PodcastFormatter,
    VideoContentFormatter
)

# Localization formatters
from .localization_formatter import (
    LocalizationFormatter,
    MultiLanguageFormatter,
    CurrencyFormatter,
    DateTimeFormatter,
    RegionalFormatter
)

# AI/ML specific formatters
from .ai_formatter import (
    AIFormatter,
    ModelPredictionFormatter,
    RecommendationFormatter,
    SentimentAnalysisFormatter,
    AudioFeatureFormatter,
    NLPFormatter,
    ComputerVisionFormatter
)

# Security and compliance formatters
from .security_formatter import (
    SecurityFormatter,
    AuditLogFormatter,
    ComplianceReportFormatter,
    GDPRFormatter,
    SOCFormatter,
    PenTestReportFormatter
)

# Template engine and utilities
from .template_formatter import (
    TemplateFormatter,
    JinjaTemplateFormatter,
    MustacheTemplateFormatter,
    CustomTemplateFormatter,
    TemplateCache,
    TemplateRegistry
)

# Export all formatters
__all__ = [
    # Core alerts
    'AlertFormatter',
    'SlackAlertFormatter',
    'EmailAlertFormatter',
    'SMSAlertFormatter',
    'TeamsAlertFormatter',
    'PagerDutyAlertFormatter',
    
    # Metrics
    'MetricsFormatter',
    'PrometheusMetricsFormatter',
    'GrafanaMetricsFormatter',
    'InfluxDBMetricsFormatter',
    'ElasticsearchMetricsFormatter',
    'CustomMetricsFormatter',
    
    # Business Intelligence
    'BusinessIntelligenceFormatter',
    'SpotifyArtistFormatter',
    'PlaylistAnalyticsFormatter',
    'RevenueReportFormatter',
    'UserEngagementFormatter',
    'MLModelPerformanceFormatter',
    
    # Streaming
    'StreamingFormatter',
    'RealTimeEventFormatter',
    'WebSocketMessageFormatter',
    'SSEFormatter',
    'MQTTMessageFormatter',
    'KafkaEventFormatter',
    
    # Rich media
    'MediaFormatter',
    'AudioTrackFormatter',
    'PlaylistFormatter',
    'ArtistProfileFormatter',
    'AlbumFormatter',
    'PodcastFormatter',
    'VideoContentFormatter',
    
    # Localization
    'LocalizationFormatter',
    'MultiLanguageFormatter',
    'CurrencyFormatter',
    'DateTimeFormatter',
    'RegionalFormatter',
    
    # AI/ML
    'AIFormatter',
    'ModelPredictionFormatter',
    'RecommendationFormatter',
    'SentimentAnalysisFormatter',
    'AudioFeatureFormatter',
    'NLPFormatter',
    'ComputerVisionFormatter',
    
    # Security
    'SecurityFormatter',
    'AuditLogFormatter',
    'ComplianceReportFormatter',
    'GDPRFormatter',
    'SOCFormatter',
    'PenTestReportFormatter',
    
    # Templates
    'TemplateFormatter',
    'JinjaTemplateFormatter',
    'MustacheTemplateFormatter',
    'CustomTemplateFormatter',
    'TemplateCache',
    'TemplateRegistry'
]

# Module metadata
__version__ = "2.1.0"
__author__ = "Fahed Mlaiel"
__description__ = "Ultra-advanced formatting system for Spotify AI Agent"
__license__ = "Proprietary"

# Configuration constants
DEFAULT_ENCODING = "utf-8"
DEFAULT_LOCALE = "en_US"
MAX_MESSAGE_LENGTH = 4096
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10MB

# Supported output formats
SUPPORTED_FORMATS = [
    'json',
    'yaml', 
    'xml',
    'html',
    'markdown',
    'plain_text',
    'slack_blocks',
    'teams_adaptive_cards',
    'prometheus_metrics',
    'grafana_dashboard',
    'email_template',
    'sms_message',
    'rich_media',
    'audio_metadata',
    'playlist_data'
]

# Available languages for localization
SUPPORTED_LANGUAGES = [
    'en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'sv', 'no', 'da', 'fi',
    'pl', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi', 'tr', 'he', 'cs', 'hu'
]

# Formatter registry for dynamic loading
FORMATTER_REGISTRY = {
    'alert': AlertFormatter,
    'slack': SlackAlertFormatter,
    'email': EmailAlertFormatter,
    'sms': SMSAlertFormatter,
    'teams': TeamsAlertFormatter,
    'pagerduty': PagerDutyAlertFormatter,
    'metrics': MetricsFormatter,
    'prometheus': PrometheusMetricsFormatter,
    'grafana': GrafanaMetricsFormatter,
    'influxdb': InfluxDBMetricsFormatter,
    'elasticsearch': ElasticsearchMetricsFormatter,
    'business': BusinessIntelligenceFormatter,
    'spotify_artist': SpotifyArtistFormatter,
    'playlist': PlaylistAnalyticsFormatter,
    'revenue': RevenueReportFormatter,
    'engagement': UserEngagementFormatter,
    'ml_performance': MLModelPerformanceFormatter,
    'streaming': StreamingFormatter,
    'realtime': RealTimeEventFormatter,
    'websocket': WebSocketMessageFormatter,
    'sse': SSEFormatter,
    'mqtt': MQTTMessageFormatter,
    'kafka': KafkaEventFormatter,
    'media': MediaFormatter,
    'audio': AudioTrackFormatter,
    'album': AlbumFormatter,
    'podcast': PodcastFormatter,
    'video': VideoContentFormatter,
    'localization': LocalizationFormatter,
    'multilang': MultiLanguageFormatter,
    'currency': CurrencyFormatter,
    'datetime': DateTimeFormatter,
    'regional': RegionalFormatter,
    'ai': AIFormatter,
    'prediction': ModelPredictionFormatter,
    'recommendation': RecommendationFormatter,
    'sentiment': SentimentAnalysisFormatter,
    'audio_features': AudioFeatureFormatter,
    'nlp': NLPFormatter,
    'computer_vision': ComputerVisionFormatter,
    'security': SecurityFormatter,
    'audit': AuditLogFormatter,
    'compliance': ComplianceReportFormatter,
    'gdpr': GDPRFormatter,
    'soc': SOCFormatter,
    'pentest': PenTestReportFormatter,
    'template': TemplateFormatter,
    'jinja': JinjaTemplateFormatter,
    'mustache': MustacheTemplateFormatter,
    'custom_template': CustomTemplateFormatter
}


def get_formatter(formatter_type: str, **kwargs):
    """
    Factory function to get a formatter instance.
    
    Args:
        formatter_type: Type of formatter to create
        **kwargs: Additional arguments for formatter initialization
        
    Returns:
        Formatter instance
        
    Raises:
        ValueError: If formatter type is not supported
    """
    if formatter_type not in FORMATTER_REGISTRY:
        raise ValueError(f"Unsupported formatter type: {formatter_type}")
        
    formatter_class = FORMATTER_REGISTRY[formatter_type]
    return formatter_class(**kwargs)


def list_formatters():
    """List all available formatter types."""
    return list(FORMATTER_REGISTRY.keys())


def validate_format_support(output_format: str) -> bool:
    """Check if output format is supported."""
    return output_format in SUPPORTED_FORMATS


def validate_language_support(language: str) -> bool:
    """Check if language is supported for localization."""
    return language in SUPPORTED_LANGUAGES
