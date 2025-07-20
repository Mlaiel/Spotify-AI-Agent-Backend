"""
External API Integrations Module
================================

Ultra-advanced external API integrations for seamless connectivity with
music streaming platforms, social media APIs, payment gateways, and analytics services.

This module provides comprehensive API integration capabilities for:
- Music streaming platforms (Spotify, Apple Music, YouTube Music, Deezer)
- Social media platforms (Twitter, Instagram, TikTok, Facebook, LinkedIn)
- Payment gateways (Stripe, PayPal, Square, Braintree)
- Analytics platforms (Google Analytics, Mixpanel, Amplitude, Segment)
- Communication services (Twilio, SendGrid, Firebase)
- Content delivery networks (Cloudflare, AWS CloudFront)

Integration Features:
- OAuth 2.0 and API key authentication
- Rate limiting and request throttling
- Automatic retry with exponential backoff
- Circuit breaker pattern for fault tolerance
- Response caching and optimization
- Multi-tenant API key management
- Real-time webhook handling
- Comprehensive error handling and logging

Author: Expert Team - Lead Dev + AI Architect, Backend Senior, API Specialist
Version: 2.1.0
"""

from typing import Dict, List, Any, Optional
from ..factory import IntegrationDiscovery
from .. import IntegrationType

# Import all external API integration modules
from .spotify_integration import SpotifyIntegration
from .apple_music_integration import AppleMusicIntegration
from .youtube_music_integration import YouTubeMusicIntegration
from .social_media_integration import (
    TwitterIntegration,
    InstagramIntegration,
    TikTokIntegration,
    FacebookIntegration,
    LinkedInIntegration
)
from .payment_integration import (
    StripeIntegration,
    PayPalIntegration,
    SquareIntegration,
    BraintreeIntegration
)
from .analytics_integration import (
    GoogleAnalyticsIntegration,
    MixpanelIntegration,
    AmplitudeIntegration,
    SegmentIntegration
)

# Integration registry for external APIs
EXTERNAL_API_INTEGRATIONS = {
    # Music Streaming APIs
    "spotify": SpotifyIntegration,
    "spotify_api": SpotifyIntegration,
    "spotify_web_api": SpotifyIntegration,
    
    "apple_music": AppleMusicIntegration,
    "apple_music_api": AppleMusicIntegration,
    
    "youtube_music": YouTubeMusicIntegration,
    "youtube_music_api": YouTubeMusicIntegration,
    "ytmusic": YouTubeMusicIntegration,
    
    # Social Media APIs
    "twitter": TwitterIntegration,
    "twitter_api": TwitterIntegration,
    "twitter_v2": TwitterIntegration,
    
    "instagram": InstagramIntegration,
    "instagram_api": InstagramIntegration,
    "instagram_graph": InstagramIntegration,
    
    "tiktok": TikTokIntegration,
    "tiktok_api": TikTokIntegration,
    "tiktok_for_developers": TikTokIntegration,
    
    "facebook": FacebookIntegration,
    "facebook_api": FacebookIntegration,
    "facebook_graph": FacebookIntegration,
    
    "linkedin": LinkedInIntegration,
    "linkedin_api": LinkedInIntegration,
    
    # Payment APIs
    "stripe": StripeIntegration,
    "stripe_api": StripeIntegration,
    "stripe_payment": StripeIntegration,
    
    "paypal": PayPalIntegration,
    "paypal_api": PayPalIntegration,
    "paypal_payment": PayPalIntegration,
    
    "square": SquareIntegration,
    "square_api": SquareIntegration,
    "square_payment": SquareIntegration,
    
    "braintree": BraintreeIntegration,
    "braintree_api": BraintreeIntegration,
    "braintree_payment": BraintreeIntegration,
    
    # Analytics APIs
    "google_analytics": GoogleAnalyticsIntegration,
    "ga4": GoogleAnalyticsIntegration,
    "google_analytics_4": GoogleAnalyticsIntegration,
    
    "mixpanel": MixpanelIntegration,
    "mixpanel_api": MixpanelIntegration,
    
    "amplitude": AmplitudeIntegration,
    "amplitude_api": AmplitudeIntegration,
    
    "segment": SegmentIntegration,
    "segment_api": SegmentIntegration,
}

# Export all integration classes
__all__ = [
    # Music streaming integrations
    "SpotifyIntegration",
    "AppleMusicIntegration", 
    "YouTubeMusicIntegration",
    
    # Social media integrations
    "TwitterIntegration",
    "InstagramIntegration",
    "TikTokIntegration",
    "FacebookIntegration",
    "LinkedInIntegration",
    
    # Payment integrations
    "StripeIntegration",
    "PayPalIntegration",
    "SquareIntegration",
    "BraintreeIntegration",
    
    # Analytics integrations
    "GoogleAnalyticsIntegration",
    "MixpanelIntegration",
    "AmplitudeIntegration",
    "SegmentIntegration",
    
    # Registry
    "EXTERNAL_API_INTEGRATIONS"
]

def get_available_integrations() -> Dict[str, str]:
    """Get list of available external API integrations."""
    return {
        name: integration_class.__doc__.split('\n')[0] if integration_class.__doc__ else "External API Integration"
        for name, integration_class in EXTERNAL_API_INTEGRATIONS.items()
    }

def get_integration_by_type(integration_type: str) -> Optional[type]:
    """Get integration class by type name."""
    return EXTERNAL_API_INTEGRATIONS.get(integration_type.lower())

# Module metadata
__version__ = "2.1.0"
__description__ = "Ultra-advanced external API integrations for Spotify AI Agent"
