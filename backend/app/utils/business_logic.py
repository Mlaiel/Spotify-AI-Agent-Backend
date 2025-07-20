"""
Enterprise Business Logic
========================
Advanced business logic utilities for Spotify AI Agent streaming platform.

Expert Team Implementation:
- Lead Developer + AI Architect: AI-powered business intelligence and decision engines
- Senior Backend Developer: High-performance business process automation
- DBA & Data Engineer: Business analytics and data-driven insights
- ML Engineer: Predictive business modeling and recommendation engines
- Microservices Architect: Distributed business process orchestration
- Security Specialist: Business compliance and audit trails
"""

import asyncio
import logging
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Set
from abc import ABC, abstractmethod
from enum import Enum
import statistics
import uuid
from decimal import Decimal, ROUND_HALF_UP
import re

logger = logging.getLogger(__name__)

# === Business Types and Enums ===
class UserTier(Enum):
    """User subscription tiers."""
    FREE = "free"
    PREMIUM = "premium"
    FAMILY = "family"
    STUDENT = "student"
    ARTIST = "artist"
    ENTERPRISE = "enterprise"

class ContentType(Enum):
    """Content types in the platform."""
    SONG = "song"
    ALBUM = "album"
    PLAYLIST = "playlist"
    PODCAST = "podcast"
    AUDIOBOOK = "audiobook"

class RecommendationType(Enum):
    """Types of recommendations."""
    DISCOVER_WEEKLY = "discover_weekly"
    DAILY_MIX = "daily_mix"
    RELEASE_RADAR = "release_radar"
    SIMILAR_ARTISTS = "similar_artists"
    MOOD_BASED = "mood_based"
    ACTIVITY_BASED = "activity_based"

class PlaybackQuality(Enum):
    """Audio playback quality levels."""
    LOW = "low"        # 96 kbps
    NORMAL = "normal"  # 160 kbps
    HIGH = "high"      # 320 kbps
    LOSSLESS = "lossless"  # FLAC

class BusinessEventType(Enum):
    """Business event types for analytics."""
    USER_SIGNUP = "user_signup"
    SUBSCRIPTION_START = "subscription_start"
    SUBSCRIPTION_CANCEL = "subscription_cancel"
    SONG_PLAY = "song_play"
    SONG_SKIP = "song_skip"
    PLAYLIST_CREATE = "playlist_create"
    ARTIST_FOLLOW = "artist_follow"
    CONTENT_LIKE = "content_like"
    PREMIUM_FEATURE_USE = "premium_feature_use"

@dataclass
class User:
    """User business entity."""
    user_id: str
    email: str
    username: str
    tier: UserTier
    country: str
    registration_date: datetime
    last_active: datetime
    preferences: Dict[str, Any] = field(default_factory=dict)
    subscription_expires: Optional[datetime] = None
    total_listening_time_minutes: int = 0
    favorite_genres: List[str] = field(default_factory=list)
    is_active: bool = True

@dataclass
class Content:
    """Content business entity."""
    content_id: str
    title: str
    content_type: ContentType
    artist_id: Optional[str] = None
    album_id: Optional[str] = None
    duration_seconds: int = 0
    release_date: Optional[datetime] = None
    genres: List[str] = field(default_factory=list)
    popularity_score: float = 0.0
    is_explicit: bool = False
    is_available: bool = True
    play_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PlaybackSession:
    """User playback session."""
    session_id: str
    user_id: str
    content_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_played_seconds: int = 0
    quality: PlaybackQuality = PlaybackQuality.NORMAL
    device_type: str = "unknown"
    was_skipped: bool = False
    completion_percentage: float = 0.0

@dataclass
class BusinessEvent:
    """Business event for analytics."""
    event_id: str
    event_type: BusinessEventType
    user_id: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    content_id: Optional[str] = None

@dataclass
class Recommendation:
    """Content recommendation."""
    recommendation_id: str
    user_id: str
    content_id: str
    recommendation_type: RecommendationType
    score: float
    reason: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    was_clicked: bool = False
    was_played: bool = False

@dataclass
class BusinessMetrics:
    """Business performance metrics."""
    timestamp: datetime
    total_users: int = 0
    active_users_daily: int = 0
    active_users_monthly: int = 0
    premium_conversion_rate: float = 0.0
    average_session_duration_minutes: float = 0.0
    content_engagement_rate: float = 0.0
    revenue_per_user: Decimal = Decimal('0.00')
    churn_rate: float = 0.0

# === Business Rules Engine ===
class BusinessRule(ABC):
    """Abstract business rule."""
    
    def __init__(self, rule_id: str, description: str, priority: int = 1):
        self.rule_id = rule_id
        self.description = description
        self.priority = priority
        self.enabled = True
    
    @abstractmethod
    async def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule against context."""
        pass
    
    @abstractmethod
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rule action to context."""
        pass

class SubscriptionRule(BusinessRule):
    """Rules for subscription management."""
    
    async def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if user can access premium features."""
        user = context.get('user')
        if not user:
            return False
        
        # Check subscription status
        if user.tier == UserTier.FREE:
            return False
        
        # Check subscription expiry
        if user.subscription_expires and user.subscription_expires < datetime.now():
            return False
        
        return True
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply subscription restrictions."""
        user = context.get('user')
        content = context.get('content')
        
        if not await self.evaluate(context):
            # Downgrade quality for free users
            if 'playback_quality' in context:
                context['playback_quality'] = PlaybackQuality.NORMAL
            
            # Add ads for free users
            context['insert_ads'] = True
            context['ads_frequency'] = 3  # Every 3 songs
        
        return context

class ContentAccessRule(BusinessRule):
    """Rules for content access control."""
    
    async def evaluate(self, context: Dict[str, Any]) -> bool:
        """Check if user can access content."""
        user = context.get('user')
        content = context.get('content')
        
        if not user or not content:
            return False
        
        # Check content availability
        if not content.is_available:
            return False
        
        # Check geographic restrictions
        if 'restricted_countries' in content.metadata:
            if user.country in content.metadata['restricted_countries']:
                return False
        
        # Check explicit content
        if content.is_explicit and user.preferences.get('filter_explicit', False):
            return False
        
        return True
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply content access restrictions."""
        if not await self.evaluate(context):
            context['access_denied'] = True
            context['denial_reason'] = "Content not available in your region or filtered by preferences"
        
        return context

class BusinessRulesEngine:
    """Engine for evaluating and applying business rules."""
    
    def __init__(self):
        self.rules = {}
        self.rule_categories = defaultdict(list)
        self.execution_stats = defaultdict(int)
    
    def add_rule(self, rule: BusinessRule, category: str = "general"):
        """Add business rule to engine."""
        self.rules[rule.rule_id] = rule
        self.rule_categories[category].append(rule.rule_id)
        
        logger.info(f"Added business rule '{rule.rule_id}' to category '{category}'")
    
    def remove_rule(self, rule_id: str):
        """Remove business rule from engine."""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            del self.rules[rule_id]
            
            # Remove from categories
            for category_rules in self.rule_categories.values():
                if rule_id in category_rules:
                    category_rules.remove(rule_id)
            
            logger.info(f"Removed business rule '{rule_id}'")
            return True
        return False
    
    async def evaluate_rules(self, context: Dict[str, Any], category: str = None) -> List[str]:
        """Evaluate rules and return list of applicable rule IDs."""
        applicable_rules = []
        
        rules_to_evaluate = []
        if category:
            rules_to_evaluate = [self.rules[rule_id] for rule_id in self.rule_categories.get(category, [])]
        else:
            rules_to_evaluate = list(self.rules.values())
        
        # Sort by priority
        rules_to_evaluate.sort(key=lambda r: r.priority, reverse=True)
        
        for rule in rules_to_evaluate:
            if not rule.enabled:
                continue
            
            try:
                if await rule.evaluate(context):
                    applicable_rules.append(rule.rule_id)
                    self.execution_stats[rule.rule_id] += 1
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
        
        return applicable_rules
    
    async def apply_rules(self, context: Dict[str, Any], rule_ids: List[str] = None) -> Dict[str, Any]:
        """Apply specified rules to context."""
        if rule_ids is None:
            rule_ids = await self.evaluate_rules(context)
        
        for rule_id in rule_ids:
            if rule_id in self.rules and self.rules[rule_id].enabled:
                try:
                    context = await self.rules[rule_id].apply(context)
                except Exception as e:
                    logger.error(f"Error applying rule {rule_id}: {e}")
        
        return context

# === Recommendation Engine ===
class RecommendationEngine:
    """AI-powered content recommendation engine."""
    
    def __init__(self):
        self.user_preferences = defaultdict(dict)
        self.content_similarity = defaultdict(dict)
        self.collaborative_data = defaultdict(set)
        self.recommendation_cache = {}
        
    async def generate_recommendations(self, 
                                     user: User, 
                                     recommendation_type: RecommendationType,
                                     limit: int = 20) -> List[Recommendation]:
        """Generate personalized recommendations for user."""
        cache_key = f"{user.user_id}_{recommendation_type.value}_{limit}"
        
        # Check cache first
        if cache_key in self.recommendation_cache:
            cached_time, recommendations = self.recommendation_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < 3600:  # 1 hour cache
                return recommendations
        
        try:
            if recommendation_type == RecommendationType.DISCOVER_WEEKLY:
                recommendations = await self._generate_discover_weekly(user, limit)
            
            elif recommendation_type == RecommendationType.DAILY_MIX:
                recommendations = await self._generate_daily_mix(user, limit)
            
            elif recommendation_type == RecommendationType.MOOD_BASED:
                recommendations = await self._generate_mood_based(user, limit)
            
            elif recommendation_type == RecommendationType.SIMILAR_ARTISTS:
                recommendations = await self._generate_similar_artists(user, limit)
            
            else:
                recommendations = await self._generate_generic_recommendations(user, limit)
            
            # Cache recommendations
            self.recommendation_cache[cache_key] = (datetime.now(), recommendations)
            
            logger.info(f"Generated {len(recommendations)} {recommendation_type.value} recommendations for user {user.user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations for user {user.user_id}: {e}")
            return []
    
    async def _generate_discover_weekly(self, user: User, limit: int) -> List[Recommendation]:
        """Generate Discover Weekly recommendations."""
        recommendations = []
        
        # Analyze user's listening history
        user_genres = user.favorite_genres or []
        
        # Find similar users (collaborative filtering)
        similar_users = await self._find_similar_users(user)
        
        # Get content liked by similar users but not played by current user
        candidate_content = await self._get_collaborative_candidates(user, similar_users)
        
        # Score and rank candidates
        for content_id in candidate_content[:limit]:
            score = await self._calculate_recommendation_score(user, content_id, "discover_weekly")
            
            recommendation = Recommendation(
                recommendation_id=str(uuid.uuid4()),
                user_id=user.user_id,
                content_id=content_id,
                recommendation_type=RecommendationType.DISCOVER_WEEKLY,
                score=score,
                reason="Based on your listening history and similar users",
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=7)
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_daily_mix(self, user: User, limit: int) -> List[Recommendation]:
        """Generate Daily Mix recommendations."""
        recommendations = []
        
        # Get user's most played genres
        favorite_genres = user.favorite_genres[:3] if user.favorite_genres else ["pop"]
        
        for genre in favorite_genres:
            # Get popular content in this genre
            genre_content = await self._get_popular_content_by_genre(genre, limit // len(favorite_genres))
            
            for content_id in genre_content:
                score = await self._calculate_recommendation_score(user, content_id, "daily_mix")
                
                recommendation = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    user_id=user.user_id,
                    content_id=content_id,
                    recommendation_type=RecommendationType.DAILY_MIX,
                    score=score,
                    reason=f"Based on your love for {genre} music",
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=1)
                )
                recommendations.append(recommendation)
        
        return recommendations[:limit]
    
    async def _generate_mood_based(self, user: User, limit: int) -> List[Recommendation]:
        """Generate mood-based recommendations."""
        recommendations = []
        
        # Determine user's current mood (simplified)
        current_hour = datetime.now().hour
        
        if 6 <= current_hour <= 9:
            mood = "energetic"
        elif 9 <= current_hour <= 17:
            mood = "focus"
        elif 17 <= current_hour <= 22:
            mood = "relaxed"
        else:
            mood = "chill"
        
        # Get content matching mood
        mood_content = await self._get_content_by_mood(mood, limit)
        
        for content_id in mood_content:
            score = await self._calculate_recommendation_score(user, content_id, "mood_based")
            
            recommendation = Recommendation(
                recommendation_id=str(uuid.uuid4()),
                user_id=user.user_id,
                content_id=content_id,
                recommendation_type=RecommendationType.MOOD_BASED,
                score=score,
                reason=f"Perfect for your {mood} mood",
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=6)
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_similar_artists(self, user: User, limit: int) -> List[Recommendation]:
        """Generate recommendations based on similar artists."""
        recommendations = []
        
        # Get user's favorite artists (would come from listening history)
        favorite_artists = ["artist_1", "artist_2"]  # Placeholder
        
        for artist_id in favorite_artists:
            similar_artists = await self._find_similar_artists(artist_id)
            
            for similar_artist_id in similar_artists[:limit//len(favorite_artists)]:
                # Get popular content from similar artist
                artist_content = await self._get_artist_content(similar_artist_id, 3)
                
                for content_id in artist_content:
                    score = await self._calculate_recommendation_score(user, content_id, "similar_artists")
                    
                    recommendation = Recommendation(
                        recommendation_id=str(uuid.uuid4()),
                        user_id=user.user_id,
                        content_id=content_id,
                        recommendation_type=RecommendationType.SIMILAR_ARTISTS,
                        score=score,
                        reason=f"Because you like similar artists",
                        created_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(days=3)
                    )
                    recommendations.append(recommendation)
        
        return recommendations[:limit]
    
    async def _generate_generic_recommendations(self, user: User, limit: int) -> List[Recommendation]:
        """Generate generic recommendations."""
        recommendations = []
        
        # Use trending content as fallback
        trending_content = await self._get_trending_content(limit)
        
        for content_id in trending_content:
            score = await self._calculate_recommendation_score(user, content_id, "generic")
            
            recommendation = Recommendation(
                recommendation_id=str(uuid.uuid4()),
                user_id=user.user_id,
                content_id=content_id,
                recommendation_type=RecommendationType.DISCOVER_WEEKLY,
                score=score,
                reason="Trending now",
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=1)
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _find_similar_users(self, user: User) -> List[str]:
        """Find users with similar preferences."""
        # Simplified similarity based on favorite genres
        similar_users = []
        
        # In a real implementation, this would use collaborative filtering
        # or machine learning to find similar users
        
        return similar_users[:50]  # Return top 50 similar users
    
    async def _get_collaborative_candidates(self, user: User, similar_users: List[str]) -> List[str]:
        """Get content candidates from collaborative filtering."""
        candidates = []
        
        # Get content liked by similar users
        for similar_user_id in similar_users:
            user_content = self.collaborative_data.get(similar_user_id, set())
            candidates.extend(list(user_content))
        
        # Remove duplicates and content already consumed by user
        user_content = self.collaborative_data.get(user.user_id, set())
        candidates = list(set(candidates) - user_content)
        
        return candidates
    
    async def _get_popular_content_by_genre(self, genre: str, limit: int) -> List[str]:
        """Get popular content in specific genre."""
        # Placeholder implementation
        return [f"content_{genre}_{i}" for i in range(limit)]
    
    async def _get_content_by_mood(self, mood: str, limit: int) -> List[str]:
        """Get content matching mood."""
        # Placeholder implementation
        return [f"content_{mood}_{i}" for i in range(limit)]
    
    async def _find_similar_artists(self, artist_id: str) -> List[str]:
        """Find artists similar to given artist."""
        # Placeholder implementation
        return [f"similar_artist_{i}" for i in range(10)]
    
    async def _get_artist_content(self, artist_id: str, limit: int) -> List[str]:
        """Get content from specific artist."""
        # Placeholder implementation
        return [f"content_{artist_id}_{i}" for i in range(limit)]
    
    async def _get_trending_content(self, limit: int) -> List[str]:
        """Get currently trending content."""
        # Placeholder implementation
        return [f"trending_content_{i}" for i in range(limit)]
    
    async def _calculate_recommendation_score(self, user: User, content_id: str, context: str) -> float:
        """Calculate recommendation score for user-content pair."""
        base_score = 0.5
        
        # Adjust score based on user preferences
        # This is a simplified scoring system
        
        # Genre matching
        # content_genres = await self._get_content_genres(content_id)
        # genre_match = len(set(user.favorite_genres) & set(content_genres))
        # base_score += genre_match * 0.1
        
        # Popularity boost
        # content_popularity = await self._get_content_popularity(content_id)
        # base_score += content_popularity * 0.2
        
        # Recency boost for newer content
        # content_age_days = await self._get_content_age_days(content_id)
        # if content_age_days < 30:
        #     base_score += 0.1
        
        # Add some randomness for diversity
        import random
        base_score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_score))
    
    def update_user_interaction(self, user_id: str, content_id: str, interaction_type: str):
        """Update user interaction data for future recommendations."""
        if interaction_type in ['play', 'like', 'save']:
            self.collaborative_data[user_id].add(content_id)
        elif interaction_type in ['skip', 'dislike']:
            self.collaborative_data[user_id].discard(content_id)

# === Business Analytics ===
class BusinessAnalytics:
    """Business intelligence and analytics engine."""
    
    def __init__(self):
        self.events = deque(maxlen=100000)  # Store recent events
        self.user_metrics = defaultdict(dict)
        self.content_metrics = defaultdict(dict)
        self.daily_metrics = defaultdict(dict)
        
    async def record_event(self, event: BusinessEvent):
        """Record business event for analytics."""
        self.events.append(event)
        
        # Update real-time metrics
        await self._update_realtime_metrics(event)
        
        logger.debug(f"Recorded business event: {event.event_type.value} for user {event.user_id}")
    
    async def _update_realtime_metrics(self, event: BusinessEvent):
        """Update real-time metrics from event."""
        today = datetime.now().date()
        
        # User metrics
        if event.event_type == BusinessEventType.USER_SIGNUP:
            self.daily_metrics[today]['new_signups'] = self.daily_metrics[today].get('new_signups', 0) + 1
        
        elif event.event_type == BusinessEventType.SONG_PLAY:
            self.user_metrics[event.user_id]['total_plays'] = self.user_metrics[event.user_id].get('total_plays', 0) + 1
            self.content_metrics[event.content_id]['play_count'] = self.content_metrics[event.content_id].get('play_count', 0) + 1
        
        elif event.event_type == BusinessEventType.SUBSCRIPTION_START:
            self.daily_metrics[today]['new_subscriptions'] = self.daily_metrics[today].get('new_subscriptions', 0) + 1
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for specific user."""
        user_events = [e for e in self.events if e.user_id == user_id]
        
        # Calculate user metrics
        total_plays = len([e for e in user_events if e.event_type == BusinessEventType.SONG_PLAY])
        total_skips = len([e for e in user_events if e.event_type == BusinessEventType.SONG_SKIP])
        skip_rate = (total_skips / total_plays) if total_plays > 0 else 0
        
        # Get listening patterns
        play_events = [e for e in user_events if e.event_type == BusinessEventType.SONG_PLAY]
        listening_hours = defaultdict(int)
        
        for event in play_events:
            hour = event.timestamp.hour
            listening_hours[hour] += 1
        
        peak_hour = max(listening_hours.items(), key=lambda x: x[1])[0] if listening_hours else 12
        
        return {
            'user_id': user_id,
            'total_plays': total_plays,
            'total_skips': total_skips,
            'skip_rate': skip_rate,
            'peak_listening_hour': peak_hour,
            'listening_distribution': dict(listening_hours),
            'engagement_score': self._calculate_engagement_score(user_events)
        }
    
    async def get_content_analytics(self, content_id: str) -> Dict[str, Any]:
        """Get analytics for specific content."""
        content_events = [e for e in self.events if e.content_id == content_id]
        
        total_plays = len([e for e in content_events if e.event_type == BusinessEventType.SONG_PLAY])
        total_skips = len([e for e in content_events if e.event_type == BusinessEventType.SONG_SKIP])
        unique_listeners = len(set(e.user_id for e in content_events))
        
        # Calculate completion rate (would need session data)
        completion_rate = 0.75  # Placeholder
        
        return {
            'content_id': content_id,
            'total_plays': total_plays,
            'total_skips': total_skips,
            'unique_listeners': unique_listeners,
            'completion_rate': completion_rate,
            'popularity_trend': self._calculate_popularity_trend(content_events)
        }
    
    async def get_business_metrics(self, date_range: Tuple[datetime, datetime] = None) -> BusinessMetrics:
        """Get comprehensive business metrics."""
        if date_range is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
        else:
            start_date, end_date = date_range
        
        # Filter events by date range
        period_events = [
            e for e in self.events 
            if start_date <= e.timestamp <= end_date
        ]
        
        # Calculate metrics
        total_users = len(set(e.user_id for e in period_events))
        
        # Daily active users (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        daily_active_events = [e for e in period_events if e.timestamp >= yesterday]
        active_users_daily = len(set(e.user_id for e in daily_active_events))
        
        # Monthly active users
        last_month = datetime.now() - timedelta(days=30)
        monthly_active_events = [e for e in period_events if e.timestamp >= last_month]
        active_users_monthly = len(set(e.user_id for e in monthly_active_events))
        
        # Premium conversion (simplified)
        signup_events = [e for e in period_events if e.event_type == BusinessEventType.USER_SIGNUP]
        subscription_events = [e for e in period_events if e.event_type == BusinessEventType.SUBSCRIPTION_START]
        conversion_rate = (len(subscription_events) / len(signup_events)) if signup_events else 0
        
        # Session metrics
        play_events = [e for e in period_events if e.event_type == BusinessEventType.SONG_PLAY]
        avg_session_duration = 25.5  # Placeholder (would calculate from session data)
        
        # Engagement rate
        total_interactions = len([e for e in period_events if e.event_type in [
            BusinessEventType.CONTENT_LIKE, BusinessEventType.PLAYLIST_CREATE, 
            BusinessEventType.ARTIST_FOLLOW
        ]])
        engagement_rate = (total_interactions / len(period_events)) if period_events else 0
        
        return BusinessMetrics(
            timestamp=datetime.now(),
            total_users=total_users,
            active_users_daily=active_users_daily,
            active_users_monthly=active_users_monthly,
            premium_conversion_rate=conversion_rate,
            average_session_duration_minutes=avg_session_duration,
            content_engagement_rate=engagement_rate,
            revenue_per_user=Decimal('4.99'),  # Placeholder
            churn_rate=0.05  # Placeholder
        )
    
    def _calculate_engagement_score(self, user_events: List[BusinessEvent]) -> float:
        """Calculate user engagement score."""
        if not user_events:
            return 0.0
        
        # Weighted scoring for different events
        event_weights = {
            BusinessEventType.SONG_PLAY: 1.0,
            BusinessEventType.CONTENT_LIKE: 2.0,
            BusinessEventType.PLAYLIST_CREATE: 3.0,
            BusinessEventType.ARTIST_FOLLOW: 2.5,
            BusinessEventType.SONG_SKIP: -0.5
        }
        
        total_score = sum(event_weights.get(event.event_type, 0) for event in user_events)
        
        # Normalize by time period and event count
        days_active = (max(e.timestamp for e in user_events) - min(e.timestamp for e in user_events)).days + 1
        score_per_day = total_score / days_active
        
        return max(0.0, min(10.0, score_per_day))  # Scale 0-10
    
    def _calculate_popularity_trend(self, content_events: List[BusinessEvent]) -> str:
        """Calculate content popularity trend."""
        if len(content_events) < 10:
            return "insufficient_data"
        
        # Group events by day and count plays
        daily_plays = defaultdict(int)
        for event in content_events:
            if event.event_type == BusinessEventType.SONG_PLAY:
                day = event.timestamp.date()
                daily_plays[day] += 1
        
        if len(daily_plays) < 3:
            return "stable"
        
        # Calculate trend
        sorted_days = sorted(daily_plays.keys())
        recent_plays = sum(daily_plays[day] for day in sorted_days[-3:])
        earlier_plays = sum(daily_plays[day] for day in sorted_days[:3])
        
        if recent_plays > earlier_plays * 1.2:
            return "trending_up"
        elif recent_plays < earlier_plays * 0.8:
            return "trending_down"
        else:
            return "stable"

# === Pricing Engine ===
class PricingEngine:
    """Dynamic pricing and subscription management."""
    
    def __init__(self):
        self.base_prices = {
            UserTier.PREMIUM: Decimal('9.99'),
            UserTier.FAMILY: Decimal('14.99'),
            UserTier.STUDENT: Decimal('4.99'),
            UserTier.ARTIST: Decimal('7.99')
        }
        
        self.regional_multipliers = {
            'US': Decimal('1.0'),
            'EU': Decimal('0.85'),
            'IN': Decimal('0.3'),
            'BR': Decimal('0.4'),
            'default': Decimal('0.8')
        }
    
    def calculate_subscription_price(self, 
                                   tier: UserTier, 
                                   country: str,
                                   promotional_discount: float = 0.0) -> Decimal:
        """Calculate subscription price with regional and promotional adjustments."""
        if tier == UserTier.FREE:
            return Decimal('0.00')
        
        base_price = self.base_prices.get(tier, Decimal('9.99'))
        regional_multiplier = self.regional_multipliers.get(country, self.regional_multipliers['default'])
        
        # Apply regional pricing
        adjusted_price = base_price * regional_multiplier
        
        # Apply promotional discount
        if promotional_discount > 0:
            discount_amount = adjusted_price * Decimal(str(promotional_discount))
            adjusted_price -= discount_amount
        
        # Round to 2 decimal places
        return adjusted_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def calculate_family_plan_savings(self, country: str) -> Dict[str, Decimal]:
        """Calculate savings for family plan vs individual plans."""
        individual_price = self.calculate_subscription_price(UserTier.PREMIUM, country)
        family_price = self.calculate_subscription_price(UserTier.FAMILY, country)
        
        max_family_members = 6
        individual_total = individual_price * max_family_members
        savings = individual_total - family_price
        savings_percentage = (savings / individual_total) * 100
        
        return {
            'individual_price': individual_price,
            'family_price': family_price,
            'max_members': max_family_members,
            'total_individual_cost': individual_total,
            'savings_amount': savings,
            'savings_percentage': savings_percentage.quantize(Decimal('0.1'))
        }

# === Content Curation ===
class ContentCurator:
    """AI-powered content curation and playlist generation."""
    
    def __init__(self):
        self.curation_algorithms = {
            'trending': self._curate_trending,
            'mood_based': self._curate_mood_based,
            'genre_mix': self._curate_genre_mix,
            'discovery': self._curate_discovery
        }
    
    async def curate_playlist(self, 
                            algorithm: str,
                            target_duration_minutes: int = 60,
                            user_context: Dict[str, Any] = None) -> List[str]:
        """Curate playlist using specified algorithm."""
        if algorithm not in self.curation_algorithms:
            algorithm = 'trending'
        
        curate_func = self.curation_algorithms[algorithm]
        content_ids = await curate_func(target_duration_minutes, user_context or {})
        
        logger.info(f"Curated playlist with {len(content_ids)} tracks using {algorithm} algorithm")
        return content_ids
    
    async def _curate_trending(self, duration_minutes: int, context: Dict[str, Any]) -> List[str]:
        """Curate trending content."""
        # Get trending tracks (placeholder)
        trending_tracks = [f"trending_track_{i}" for i in range(duration_minutes // 3)]
        return trending_tracks
    
    async def _curate_mood_based(self, duration_minutes: int, context: Dict[str, Any]) -> List[str]:
        """Curate based on mood."""
        mood = context.get('mood', 'neutral')
        
        # Get mood-appropriate tracks (placeholder)
        mood_tracks = [f"{mood}_track_{i}" for i in range(duration_minutes // 3)]
        return mood_tracks
    
    async def _curate_genre_mix(self, duration_minutes: int, context: Dict[str, Any]) -> List[str]:
        """Curate genre mix playlist."""
        genres = context.get('genres', ['pop', 'rock', 'jazz'])
        tracks_per_genre = (duration_minutes // 3) // len(genres)
        
        mixed_tracks = []
        for genre in genres:
            genre_tracks = [f"{genre}_track_{i}" for i in range(tracks_per_genre)]
            mixed_tracks.extend(genre_tracks)
        
        # Shuffle for variety
        import random
        random.shuffle(mixed_tracks)
        return mixed_tracks
    
    async def _curate_discovery(self, duration_minutes: int, context: Dict[str, Any]) -> List[str]:
        """Curate discovery playlist with lesser-known tracks."""
        # Get discovery tracks (placeholder)
        discovery_tracks = [f"discovery_track_{i}" for i in range(duration_minutes // 3)]
        return discovery_tracks

# === Factory Functions ===
def create_business_rules_engine() -> BusinessRulesEngine:
    """Create business rules engine with default rules."""
    engine = BusinessRulesEngine()
    
    # Add default rules
    engine.add_rule(SubscriptionRule("subscription_access", "Check subscription access"), "subscription")
    engine.add_rule(ContentAccessRule("content_access", "Check content access"), "content")
    
    return engine

def create_recommendation_engine() -> RecommendationEngine:
    """Create recommendation engine instance."""
    return RecommendationEngine()

def create_business_analytics() -> BusinessAnalytics:
    """Create business analytics instance."""
    return BusinessAnalytics()

def create_pricing_engine() -> PricingEngine:
    """Create pricing engine instance."""
    return PricingEngine()

def create_content_curator() -> ContentCurator:
    """Create content curator instance."""
    return ContentCurator()

# === Export Classes ===
__all__ = [
    'BusinessRulesEngine', 'RecommendationEngine', 'BusinessAnalytics', 
    'PricingEngine', 'ContentCurator',
    'UserTier', 'ContentType', 'RecommendationType', 'PlaybackQuality', 'BusinessEventType',
    'User', 'Content', 'PlaybackSession', 'BusinessEvent', 'Recommendation', 'BusinessMetrics',
    'BusinessRule', 'SubscriptionRule', 'ContentAccessRule',
    'create_business_rules_engine', 'create_recommendation_engine', 'create_business_analytics',
    'create_pricing_engine', 'create_content_curator'
]
