"""
Social Media Integrations Module
===============================

Ultra-advanced social media integrations with comprehensive platform support,
real-time features, and enterprise-grade functionality.

This module provides integrations with major social media platforms including:
- Twitter/X API v2 with real-time streaming
- Instagram Graph API and Basic Display API
- Facebook Graph API with comprehensive features
- TikTok for Developers API
- LinkedIn API for business networking
- YouTube Data API v3
- Discord bot integration
- Telegram bot API
- WhatsApp Business API
- Snapchat Marketing API
- Pinterest API for Developers
- Reddit API (PRAW)
- Twitch API for live streaming

Features:
- Multi-platform authentication (OAuth 2.0, API keys, webhooks)
- Real-time content streaming and webhooks
- Content publishing and management
- Analytics and insights extraction
- User engagement tracking
- Hashtag and trend monitoring
- Content moderation and filtering
- Cross-platform content syndication
- Automated social media campaigns
- Influencer identification and tracking

Author: Expert Team - Lead Dev + AI Architect, Social Media Specialist
Version: 2.1.0
"""

from typing import Dict, List, Any, Optional, Type
import structlog

from .. import BaseIntegration, IntegrationConfig
from ..factory import IntegrationDependency

logger = structlog.get_logger(__name__)


# Social Media Integration Registry
SOCIAL_MEDIA_INTEGRATIONS: Dict[str, Type[BaseIntegration]] = {}


def register_social_media_integration(name: str):
    """Decorator to register social media integrations."""
    def decorator(cls):
        SOCIAL_MEDIA_INTEGRATIONS[name] = cls
        return cls
    return decorator


# Import and register all social media integrations
try:
    from .twitter_integration import TwitterIntegration
    SOCIAL_MEDIA_INTEGRATIONS['twitter'] = TwitterIntegration
    SOCIAL_MEDIA_INTEGRATIONS['x'] = TwitterIntegration  # Alias for X
except ImportError:
    logger.warning("Twitter integration not available")

try:
    from .instagram_integration import InstagramIntegration
    SOCIAL_MEDIA_INTEGRATIONS['instagram'] = InstagramIntegration
except ImportError:
    logger.warning("Instagram integration not available")

try:
    from .facebook_integration import FacebookIntegration
    SOCIAL_MEDIA_INTEGRATIONS['facebook'] = FacebookIntegration
except ImportError:
    logger.warning("Facebook integration not available")

try:
    from .tiktok_integration import TikTokIntegration
    SOCIAL_MEDIA_INTEGRATIONS['tiktok'] = TikTokIntegration
except ImportError:
    logger.warning("TikTok integration not available")

try:
    from .linkedin_integration import LinkedInIntegration
    SOCIAL_MEDIA_INTEGRATIONS['linkedin'] = LinkedInIntegration
except ImportError:
    logger.warning("LinkedIn integration not available")

try:
    from .youtube_integration import YouTubeIntegration
    SOCIAL_MEDIA_INTEGRATIONS['youtube'] = YouTubeIntegration
except ImportError:
    logger.warning("YouTube integration not available")

try:
    from .discord_integration import DiscordIntegration
    SOCIAL_MEDIA_INTEGRATIONS['discord'] = DiscordIntegration
except ImportError:
    logger.warning("Discord integration not available")

try:
    from .telegram_integration import TelegramIntegration
    SOCIAL_MEDIA_INTEGRATIONS['telegram'] = TelegramIntegration
except ImportError:
    logger.warning("Telegram integration not available")

try:
    from .whatsapp_integration import WhatsAppIntegration
    SOCIAL_MEDIA_INTEGRATIONS['whatsapp'] = WhatsAppIntegration
except ImportError:
    logger.warning("WhatsApp integration not available")

try:
    from .snapchat_integration import SnapchatIntegration
    SOCIAL_MEDIA_INTEGRATIONS['snapchat'] = SnapchatIntegration
except ImportError:
    logger.warning("Snapchat integration not available")

try:
    from .pinterest_integration import PinterestIntegration
    SOCIAL_MEDIA_INTEGRATIONS['pinterest'] = PinterestIntegration
except ImportError:
    logger.warning("Pinterest integration not available")

try:
    from .reddit_integration import RedditIntegration
    SOCIAL_MEDIA_INTEGRATIONS['reddit'] = RedditIntegration
except ImportError:
    logger.warning("Reddit integration not available")

try:
    from .twitch_integration import TwitchIntegration
    SOCIAL_MEDIA_INTEGRATIONS['twitch'] = TwitchIntegration
except ImportError:
    logger.warning("Twitch integration not available")


class SocialMediaManager:
    """Centralized manager for all social media integrations."""
    
    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}
        self.logger = logger.bind(component="social_media_manager")
    
    async def initialize_integration(self, platform: str, config: IntegrationConfig, tenant_id: str) -> bool:
        """Initialize a specific social media integration."""
        if platform not in SOCIAL_MEDIA_INTEGRATIONS:
            self.logger.error(f"Social media platform '{platform}' not supported")
            return False
        
        try:
            integration_class = SOCIAL_MEDIA_INTEGRATIONS[platform]
            integration = integration_class(config, tenant_id)
            
            success = await integration.initialize()
            if success:
                self.integrations[platform] = integration
                self.logger.info(f"Initialized {platform} integration for tenant {tenant_id}")
                return True
            else:
                self.logger.error(f"Failed to initialize {platform} integration")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing {platform} integration: {str(e)}")
            return False
    
    async def get_integration(self, platform: str) -> Optional[BaseIntegration]:
        """Get a specific social media integration."""
        return self.integrations.get(platform)
    
    async def cleanup_all(self) -> None:
        """Cleanup all social media integrations."""
        for platform, integration in self.integrations.items():
            try:
                await integration.cleanup()
                self.logger.info(f"Cleaned up {platform} integration")
            except Exception as e:
                self.logger.error(f"Error cleaning up {platform} integration: {str(e)}")
        
        self.integrations.clear()
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all active integrations."""
        health_status = {}
        
        for platform, integration in self.integrations.items():
            try:
                status = await integration.health_check()
                health_status[platform] = status
            except Exception as e:
                health_status[platform] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": str(datetime.now(timezone.utc))
                }
        
        return health_status
    
    def get_supported_platforms(self) -> List[str]:
        """Get list of supported social media platforms."""
        return list(SOCIAL_MEDIA_INTEGRATIONS.keys())
    
    def get_active_platforms(self) -> List[str]:
        """Get list of currently active social media integrations."""
        return list(self.integrations.keys())


# Export all available integration classes and utilities
__all__ = [
    'SocialMediaManager',
    'SOCIAL_MEDIA_INTEGRATIONS',
    'register_social_media_integration'
] + [cls.__name__ for cls in SOCIAL_MEDIA_INTEGRATIONS.values()]
