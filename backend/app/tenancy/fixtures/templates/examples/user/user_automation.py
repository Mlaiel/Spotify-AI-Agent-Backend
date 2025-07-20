#!/usr/bin/env python3
"""
Enterprise User Management Automation System
Advanced User Lifecycle Management, Provisioning, and Analytics

This automation system provides comprehensive user management capabilities including:
- Automated user provisioning and deprovisioning
- Tier-based feature management and upgrades
- Security policy enforcement and compliance
- Analytics and reporting automation
- Integration management and synchronization
- Performance monitoring and optimization
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import click
import aiohttp
import asyncpg
import aioredis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway
import structlog

# Import user management components
try:
    from . import (
        UserManager, UserProfile, UserTier, UserStatus,
        create_free_user_profile, create_premium_user_profile,
        create_enterprise_user_profile, create_vip_user_profile
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from __init__ import (
        UserManager, UserProfile, UserTier, UserStatus,
        create_free_user_profile, create_premium_user_profile,
        create_enterprise_user_profile, create_vip_user_profile
    )

# Configure structured logging
logger = structlog.get_logger(__name__)

# Metrics
user_automation_counter = Counter('user_automation_operations_total', 'User automation operations', ['operation', 'status'])
user_provisioning_duration = Histogram('user_provisioning_duration_seconds', 'User provisioning duration')
user_migration_counter = Counter('user_migrations_total', 'User tier migrations', ['from_tier', 'to_tier', 'status'])
active_automation_tasks = Gauge('active_automation_tasks', 'Active automation tasks')

@dataclass
class AutomationConfig:
    """Configuration for user automation system"""
    database_url: str = "postgresql://localhost/spotify_ai_agent"
    redis_url: str = "redis://localhost:6379"
    prometheus_gateway: str = "localhost:9091"
    notification_webhook: str = ""
    
    # Automation settings
    batch_size: int = 100
    max_concurrent_operations: int = 10
    retry_attempts: int = 3
    timeout_seconds: int = 300
    
    # Scheduling
    provisioning_schedule: str = "0 */6 * * *"  # Every 6 hours
    analytics_schedule: str = "0 2 * * *"      # Daily at 2 AM
    cleanup_schedule: str = "0 1 * * 0"        # Weekly on Sunday at 1 AM
    
    # Feature flags
    enable_auto_provisioning: bool = True
    enable_auto_deprovisioning: bool = True
    enable_tier_recommendations: bool = True
    enable_usage_analytics: bool = True
    enable_security_monitoring: bool = True
    enable_compliance_reporting: bool = True

class UserAutomationEngine:
    """Advanced user automation and lifecycle management"""
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.user_manager = UserManager()
        self.db_engine = None
        self.redis_client = None
        self.session_factory = None
        self.metrics_registry = CollectorRegistry()
        
    async def initialize(self):
        """Initialize automation engine with database and cache connections"""
        try:
            # Initialize database connection
            self.db_engine = create_async_engine(self.config.database_url)
            self.session_factory = sessionmaker(
                self.db_engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Initialize Redis connection
            self.redis_client = await aioredis.from_url(self.config.redis_url)
            
            # Test connections
            await self._test_connections()
            
            logger.info("User automation engine initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize automation engine", error=str(e))
            raise
    
    async def _test_connections(self):
        """Test database and cache connections"""
        # Test database
        async with self.session_factory() as session:
            result = await session.execute("SELECT 1")
            assert result.scalar() == 1
        
        # Test Redis
        await self.redis_client.ping()
        
        logger.info("All connections tested successfully")
    
    async def run_user_provisioning(self) -> Dict[str, Any]:
        """Automated user provisioning and onboarding"""
        with user_provisioning_duration.time():
            logger.info("Starting automated user provisioning")
            
            stats = {
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "errors": []
            }
            
            try:
                # Get pending user registrations
                pending_users = await self._get_pending_users()
                
                # Process users in batches
                for batch in self._create_batches(pending_users, self.config.batch_size):
                    batch_results = await self._process_user_batch(batch)
                    stats["processed"] += len(batch)
                    stats["successful"] += batch_results["successful"]
                    stats["failed"] += batch_results["failed"]
                    stats["errors"].extend(batch_results["errors"])
                
                # Update metrics
                user_automation_counter.labels(operation='provisioning', status='success').inc(stats["successful"])
                user_automation_counter.labels(operation='provisioning', status='failed').inc(stats["failed"])
                
                logger.info("User provisioning completed", stats=stats)
                return stats
                
            except Exception as e:
                logger.error("User provisioning failed", error=str(e))
                user_automation_counter.labels(operation='provisioning', status='error').inc()
                raise
    
    async def _get_pending_users(self) -> List[Dict[str, Any]]:
        """Get users pending provisioning"""
        async with self.session_factory() as session:
            # Mock query - would fetch from actual database
            return [
                {
                    "user_id": f"user_{i}",
                    "email": f"user{i}@example.com",
                    "tier": UserTier.FREE,
                    "registration_data": {}
                }
                for i in range(1, 6)  # Mock 5 pending users
            ]
    
    async def _process_user_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of users for provisioning"""
        batch_stats = {"successful": 0, "failed": 0, "errors": []}
        
        # Create semaphore for concurrent processing
        semaphore = asyncio.Semaphore(self.config.max_concurrent_operations)
        
        async def process_single_user(user_data: Dict[str, Any]) -> bool:
            async with semaphore:
                try:
                    # Create user profile based on tier
                    tier = user_data["tier"]
                    email = user_data["email"]
                    
                    if tier == UserTier.FREE:
                        profile = await create_free_user_profile(email, "temp_password")
                    elif tier == UserTier.PREMIUM:
                        profile = await create_premium_user_profile(email, "temp_password")
                    elif tier == UserTier.ENTERPRISE:
                        profile = await create_enterprise_user_profile(email, "temp_password")
                    elif tier == UserTier.VIP:
                        profile = await create_vip_user_profile(email, "temp_password")
                    else:
                        profile = await create_free_user_profile(email, "temp_password")
                    
                    # Initialize user workspace
                    await self._initialize_user_workspace(profile)
                    
                    # Send welcome notifications
                    await self._send_welcome_notifications(profile)
                    
                    # Setup integrations
                    await self._setup_default_integrations(profile)
                    
                    logger.info("User provisioned successfully", user_id=profile.user_id)
                    return True
                    
                except Exception as e:
                    error_msg = f"Failed to provision user {user_data.get('email', 'unknown')}: {str(e)}"
                    batch_stats["errors"].append(error_msg)
                    logger.error("User provisioning failed", error=error_msg)
                    return False
        
        # Process all users in batch concurrently
        results = await asyncio.gather(
            *[process_single_user(user_data) for user_data in batch],
            return_exceptions=True
        )
        
        # Count successful and failed operations
        for result in results:
            if isinstance(result, Exception):
                batch_stats["failed"] += 1
            elif result:
                batch_stats["successful"] += 1
            else:
                batch_stats["failed"] += 1
        
        return batch_stats
    
    async def _initialize_user_workspace(self, profile: UserProfile):
        """Initialize user workspace with default configurations"""
        # Create default playlists
        await self._create_default_playlists(profile)
        
        # Setup AI preferences
        await self._configure_ai_preferences(profile)
        
        # Initialize analytics tracking
        await self._setup_analytics_tracking(profile)
        
        # Cache user profile
        await self._cache_user_profile(profile)
    
    async def _create_default_playlists(self, profile: UserProfile):
        """Create default playlists for new user"""
        default_playlists = [
            {"name": "My Favorites", "description": "Your favorite tracks"},
            {"name": "Discover Weekly", "description": "AI-curated weekly discoveries"},
            {"name": "Focus Music", "description": "Music for concentration"}
        ]
        
        for playlist_data in default_playlists:
            # Mock playlist creation
            logger.info("Created default playlist", 
                       user_id=profile.user_id, 
                       playlist=playlist_data["name"])
    
    async def _configure_ai_preferences(self, profile: UserProfile):
        """Configure AI preferences based on user tier"""
        # Set tier-specific AI configuration
        if profile.tier == UserTier.FREE:
            profile.ai_preferences.personalization_level = "basic"
            profile.ai_preferences.learning_rate = 0.05
        elif profile.tier == UserTier.PREMIUM:
            profile.ai_preferences.personalization_level = "advanced"
            profile.ai_preferences.learning_rate = 0.1
        elif profile.tier in [UserTier.ENTERPRISE, UserTier.VIP]:
            profile.ai_preferences.personalization_level = "expert"
            profile.ai_preferences.learning_rate = 0.15
        
        logger.info("AI preferences configured", user_id=profile.user_id, tier=profile.tier.value)
    
    async def _setup_analytics_tracking(self, profile: UserProfile):
        """Setup analytics tracking for user"""
        # Initialize tracking configuration
        tracking_config = {
            "user_id": profile.user_id,
            "tier": profile.tier.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tracking_enabled": profile.analytics_settings.tracking_enabled
        }
        
        # Store in analytics system
        await self.redis_client.setex(
            f"analytics:user:{profile.user_id}",
            86400,  # 24 hours
            json.dumps(tracking_config)
        )
        
        logger.info("Analytics tracking configured", user_id=profile.user_id)
    
    async def _cache_user_profile(self, profile: UserProfile):
        """Cache user profile in Redis"""
        profile_data = profile.to_dict()
        await self.redis_client.setex(
            f"user:profile:{profile.user_id}",
            3600,  # 1 hour
            json.dumps(profile_data)
        )
        
        logger.info("User profile cached", user_id=profile.user_id)
    
    async def _send_welcome_notifications(self, profile: UserProfile):
        """Send welcome notifications to new user"""
        # Mock notification sending
        notifications = [
            {
                "type": "email",
                "template": "welcome_email",
                "user_id": profile.user_id,
                "tier": profile.tier.value
            },
            {
                "type": "push",
                "template": "welcome_push",
                "user_id": profile.user_id,
                "tier": profile.tier.value
            }
        ]
        
        for notification in notifications:
            logger.info("Welcome notification sent", 
                       user_id=profile.user_id, 
                       type=notification["type"])
    
    async def _setup_default_integrations(self, profile: UserProfile):
        """Setup default integrations for user tier"""
        # Enable tier-appropriate integrations
        if profile.tier in [UserTier.PREMIUM, UserTier.ENTERPRISE, UserTier.VIP]:
            # Enable premium integrations
            integrations = ["spotify", "last_fm", "discord"]
        else:
            # Enable basic integrations
            integrations = ["spotify"]
        
        for integration in integrations:
            logger.info("Integration enabled", 
                       user_id=profile.user_id, 
                       integration=integration)
    
    def _create_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Create batches from list of items"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    async def run_tier_migration_analysis(self) -> Dict[str, Any]:
        """Analyze users for potential tier upgrades"""
        logger.info("Starting tier migration analysis")
        
        analysis_results = {
            "analyzed_users": 0,
            "upgrade_candidates": [],
            "downgrade_candidates": [],
            "recommendations": []
        }
        
        try:
            # Get active users for analysis
            active_users = await self._get_active_users()
            analysis_results["analyzed_users"] = len(active_users)
            
            for user in active_users:
                # Analyze usage patterns
                usage_analysis = await self._analyze_user_usage(user)
                
                # Generate tier recommendations
                recommendation = await self._generate_tier_recommendation(user, usage_analysis)
                
                if recommendation["action"] == "upgrade":
                    analysis_results["upgrade_candidates"].append(recommendation)
                elif recommendation["action"] == "downgrade":
                    analysis_results["downgrade_candidates"].append(recommendation)
                
                analysis_results["recommendations"].append(recommendation)
            
            # Store analysis results
            await self._store_analysis_results(analysis_results)
            
            logger.info("Tier migration analysis completed", results=analysis_results)
            return analysis_results
            
        except Exception as e:
            logger.error("Tier migration analysis failed", error=str(e))
            raise
    
    async def _get_active_users(self) -> List[Dict[str, Any]]:
        """Get active users for analysis"""
        # Mock query - would fetch from actual database
        return [
            {
                "user_id": f"user_{i}",
                "current_tier": UserTier.FREE,
                "usage_data": {
                    "daily_sessions": 5,
                    "ai_requests_per_day": 45,
                    "playlists_created": 8,
                    "features_used": ["recommendations", "analysis"]
                }
            }
            for i in range(1, 11)  # Mock 10 active users
        ]
    
    async def _analyze_user_usage(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user usage patterns"""
        usage_data = user["usage_data"]
        
        analysis = {
            "engagement_score": 0.0,
            "feature_utilization": 0.0,
            "growth_trend": "stable",
            "limit_exceeded": [],
            "underutilized_features": []
        }
        
        # Calculate engagement score
        daily_sessions = usage_data.get("daily_sessions", 0)
        ai_requests = usage_data.get("ai_requests_per_day", 0)
        playlists = usage_data.get("playlists_created", 0)
        
        analysis["engagement_score"] = min((daily_sessions * 0.2 + ai_requests * 0.01 + playlists * 0.1), 1.0)
        
        # Check if limits are being exceeded
        current_tier = user["current_tier"]
        if current_tier == UserTier.FREE:
            if ai_requests > 40:  # Close to limit of 50
                analysis["limit_exceeded"].append("ai_requests")
            if playlists > 8:  # Close to limit of 10
                analysis["limit_exceeded"].append("playlists")
        
        return analysis
    
    async def _generate_tier_recommendation(self, user: Dict[str, Any], usage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tier upgrade/downgrade recommendation"""
        current_tier = user["current_tier"]
        engagement_score = usage_analysis["engagement_score"]
        limits_exceeded = usage_analysis["limit_exceeded"]
        
        recommendation = {
            "user_id": user["user_id"],
            "current_tier": current_tier.value,
            "recommended_tier": current_tier.value,
            "action": "maintain",
            "confidence": 0.5,
            "reasons": [],
            "estimated_value": 0.0
        }
        
        # Upgrade logic
        if current_tier == UserTier.FREE and (engagement_score > 0.7 or limits_exceeded):
            recommendation["recommended_tier"] = UserTier.PREMIUM.value
            recommendation["action"] = "upgrade"
            recommendation["confidence"] = 0.8
            recommendation["reasons"] = [
                "High engagement score",
                "Approaching usage limits",
                "Active feature usage"
            ]
            recommendation["estimated_value"] = 25.0  # Revenue potential
        
        # Downgrade logic (for trial users not engaging)
        elif current_tier == UserTier.PREMIUM and engagement_score < 0.2:
            recommendation["recommended_tier"] = UserTier.FREE.value
            recommendation["action"] = "downgrade"
            recommendation["confidence"] = 0.6
            recommendation["reasons"] = [
                "Low engagement",
                "Minimal feature usage",
                "Cost optimization opportunity"
            ]
        
        return recommendation
    
    async def _store_analysis_results(self, results: Dict[str, Any]):
        """Store analysis results for reporting"""
        # Store in Redis with expiration
        await self.redis_client.setex(
            f"analysis:tier_migration:{datetime.now().strftime('%Y%m%d')}",
            86400 * 7,  # Keep for 7 days
            json.dumps(results, default=str)
        )
        
        logger.info("Analysis results stored")
    
    async def run_usage_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive usage analytics"""
        logger.info("Starting usage analytics generation")
        
        analytics = {
            "period": "daily",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "user_metrics": {},
            "tier_breakdown": {},
            "feature_usage": {},
            "performance_metrics": {}
        }
        
        try:
            # User metrics
            analytics["user_metrics"] = await self._calculate_user_metrics()
            
            # Tier breakdown
            analytics["tier_breakdown"] = await self._calculate_tier_breakdown()
            
            # Feature usage
            analytics["feature_usage"] = await self._calculate_feature_usage()
            
            # Performance metrics
            analytics["performance_metrics"] = await self._calculate_performance_metrics()
            
            # Store analytics
            await self._store_analytics(analytics)
            
            # Send to monitoring systems
            await self._push_metrics_to_prometheus(analytics)
            
            logger.info("Usage analytics completed", metrics_count=len(analytics))
            return analytics
            
        except Exception as e:
            logger.error("Usage analytics failed", error=str(e))
            raise
    
    async def _calculate_user_metrics(self) -> Dict[str, Any]:
        """Calculate user-related metrics"""
        return {
            "total_users": 1000,
            "active_users_24h": 750,
            "active_users_7d": 950,
            "new_registrations_24h": 25,
            "churned_users_7d": 15,
            "retention_rate_7d": 0.94,
            "retention_rate_30d": 0.85
        }
    
    async def _calculate_tier_breakdown(self) -> Dict[str, Any]:
        """Calculate tier distribution metrics"""
        return {
            "free": {"count": 700, "percentage": 70.0},
            "premium": {"count": 250, "percentage": 25.0},
            "enterprise": {"count": 40, "percentage": 4.0},
            "vip": {"count": 10, "percentage": 1.0}
        }
    
    async def _calculate_feature_usage(self) -> Dict[str, Any]:
        """Calculate feature usage metrics"""
        return {
            "recommendations": {"users": 800, "requests": 15000},
            "ai_composer": {"users": 200, "generations": 3000},
            "playlist_creation": {"users": 600, "playlists": 8000},
            "analytics": {"users": 150, "views": 2000},
            "integrations": {"spotify": 400, "apple_music": 100}
        }
    
    async def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate system performance metrics"""
        return {
            "avg_response_time_ms": 150,
            "api_success_rate": 0.995,
            "error_rate": 0.005,
            "uptime_percentage": 99.9,
            "cache_hit_rate": 0.85
        }
    
    async def _store_analytics(self, analytics: Dict[str, Any]):
        """Store analytics data"""
        # Store in time-series format
        timestamp = datetime.now().strftime('%Y%m%d_%H')
        await self.redis_client.setex(
            f"analytics:usage:{timestamp}",
            86400 * 30,  # Keep for 30 days
            json.dumps(analytics, default=str)
        )
        
        logger.info("Analytics data stored")
    
    async def _push_metrics_to_prometheus(self, analytics: Dict[str, Any]):
        """Push metrics to Prometheus gateway"""
        try:
            # Update Prometheus metrics
            user_metrics = analytics["user_metrics"]
            
            # Create temporary gauges for metrics
            registry = CollectorRegistry()
            
            total_users_gauge = Gauge('total_users', 'Total users', registry=registry)
            active_users_gauge = Gauge('active_users_24h', 'Active users 24h', registry=registry)
            
            total_users_gauge.set(user_metrics["total_users"])
            active_users_gauge.set(user_metrics["active_users_24h"])
            
            # Push to Prometheus gateway (mock)
            logger.info("Metrics pushed to Prometheus", gateway=self.config.prometheus_gateway)
            
        except Exception as e:
            logger.warning("Failed to push metrics to Prometheus", error=str(e))
    
    async def cleanup_old_data(self) -> Dict[str, Any]:
        """Cleanup old user data and temporary files"""
        logger.info("Starting data cleanup")
        
        cleanup_stats = {
            "deleted_sessions": 0,
            "archived_users": 0,
            "cleaned_cache": 0,
            "freed_storage_mb": 0
        }
        
        try:
            # Clean expired sessions
            cleanup_stats["deleted_sessions"] = await self._cleanup_expired_sessions()
            
            # Archive inactive users
            cleanup_stats["archived_users"] = await self._archive_inactive_users()
            
            # Clean cache
            cleanup_stats["cleaned_cache"] = await self._cleanup_cache()
            
            # Calculate freed storage
            cleanup_stats["freed_storage_mb"] = cleanup_stats["deleted_sessions"] * 0.1  # Mock calculation
            
            logger.info("Data cleanup completed", stats=cleanup_stats)
            return cleanup_stats
            
        except Exception as e:
            logger.error("Data cleanup failed", error=str(e))
            raise
    
    async def _cleanup_expired_sessions(self) -> int:
        """Cleanup expired user sessions"""
        # Mock cleanup - would scan and delete expired sessions
        expired_sessions = 150
        logger.info("Expired sessions cleaned", count=expired_sessions)
        return expired_sessions
    
    async def _archive_inactive_users(self) -> int:
        """Archive users inactive for long periods"""
        # Mock archival - would identify and archive inactive users
        archived_users = 25
        logger.info("Inactive users archived", count=archived_users)
        return archived_users
    
    async def _cleanup_cache(self) -> int:
        """Cleanup old cache entries"""
        # Mock cache cleanup
        cache_keys_cleaned = 500
        logger.info("Cache entries cleaned", count=cache_keys_cleaned)
        return cache_keys_cleaned
    
    async def shutdown(self):
        """Shutdown automation engine gracefully"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.db_engine:
                await self.db_engine.dispose()
            
            logger.info("User automation engine shutdown completed")
            
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))

# CLI Interface
@click.group()
@click.option('--config', default='config.json', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """User Management Automation CLI"""
    # Load configuration
    config_path = Path(config)
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        automation_config = AutomationConfig(**config_data)
    else:
        automation_config = AutomationConfig()
    
    ctx.ensure_object(dict)
    ctx.obj['config'] = automation_config

@cli.command()
@click.pass_context
def provision_users(ctx):
    """Run user provisioning automation"""
    async def _run():
        config = ctx.obj['config']
        engine = UserAutomationEngine(config)
        
        try:
            await engine.initialize()
            results = await engine.run_user_provisioning()
            click.echo(f"Provisioning completed: {json.dumps(results, indent=2)}")
        finally:
            await engine.shutdown()
    
    asyncio.run(_run())

@cli.command()
@click.pass_context
def analyze_tiers(ctx):
    """Run tier migration analysis"""
    async def _run():
        config = ctx.obj['config']
        engine = UserAutomationEngine(config)
        
        try:
            await engine.initialize()
            results = await engine.run_tier_migration_analysis()
            click.echo(f"Analysis completed: {json.dumps(results, indent=2, default=str)}")
        finally:
            await engine.shutdown()
    
    asyncio.run(_run())

@cli.command()
@click.pass_context
def generate_analytics(ctx):
    """Generate usage analytics"""
    async def _run():
        config = ctx.obj['config']
        engine = UserAutomationEngine(config)
        
        try:
            await engine.initialize()
            results = await engine.run_usage_analytics()
            click.echo(f"Analytics generated: {json.dumps(results, indent=2, default=str)}")
        finally:
            await engine.shutdown()
    
    asyncio.run(_run())

@cli.command()
@click.pass_context
def cleanup_data(ctx):
    """Cleanup old user data"""
    async def _run():
        config = ctx.obj['config']
        engine = UserAutomationEngine(config)
        
        try:
            await engine.initialize()
            results = await engine.cleanup_old_data()
            click.echo(f"Cleanup completed: {json.dumps(results, indent=2)}")
        finally:
            await engine.shutdown()
    
    asyncio.run(_run())

@cli.command()
@click.option('--operations', multiple=True, default=['provision', 'analyze', 'analytics'],
              help='Operations to run')
@click.pass_context
def run_all(ctx, operations):
    """Run all automation operations"""
    async def _run():
        config = ctx.obj['config']
        engine = UserAutomationEngine(config)
        
        try:
            await engine.initialize()
            
            results = {}
            
            if 'provision' in operations:
                results['provisioning'] = await engine.run_user_provisioning()
            
            if 'analyze' in operations:
                results['tier_analysis'] = await engine.run_tier_migration_analysis()
            
            if 'analytics' in operations:
                results['usage_analytics'] = await engine.run_usage_analytics()
            
            if 'cleanup' in operations:
                results['cleanup'] = await engine.cleanup_old_data()
            
            click.echo(f"All operations completed: {json.dumps(results, indent=2, default=str)}")
            
        finally:
            await engine.shutdown()
    
    asyncio.run(_run())

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run CLI
    cli()
