"""
Enterprise User Management System
Advanced Multi-Tier User Profile Management with AI Integration

This module provides comprehensive user management capabilities including:
- Multi-tier user profiles (Free, Premium, Enterprise, VIP)
- Advanced authentication and authorization
- AI-powered personalization and recommendations
- Behavioral analytics and insights
- Security and compliance frameworks
- Real-time synchronization and caching
- Advanced integrations and automations
"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
import aioredis
import bcrypt
import jwt
from cryptography.fernet import Fernet
from pydantic import BaseModel, EmailStr, validator, Field
import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import elasticsearch
from kafka import KafkaProducer
import opentelemetry
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Configure structured logging
logger = structlog.get_logger(__name__)

# Metrics
user_creation_counter = Counter('user_creations_total', 'Total user creations', ['tier', 'status'])
user_operation_duration = Histogram('user_operation_duration_seconds', 'User operation duration', ['operation'])
active_users_gauge = Gauge('active_users_total', 'Total active users', ['tier'])
security_events_counter = Counter('security_events_total', 'Security events', ['event_type', 'severity'])

class UserTier(str, Enum):
    """User tier enumeration with privilege levels"""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    VIP = "vip"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class UserStatus(str, Enum):
    """User account status"""
    PENDING_VERIFICATION = "pending_verification"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    BANNED = "banned"
    ARCHIVED = "archived"

class AuthenticationMethod(str, Enum):
    """Authentication methods"""
    PASSWORD = "password"
    OAUTH = "oauth"
    SAML = "saml"
    LDAP = "ldap"
    BIOMETRIC = "biometric"
    HARDWARE_TOKEN = "hardware_token"
    MAGIC_LINK = "magic_link"

class SecurityLevel(IntEnum):
    """Security clearance levels"""
    PUBLIC = 0
    RESTRICTED = 1
    CONFIDENTIAL = 2
    SECRET = 3
    TOP_SECRET = 4

class AIPersonalizationLevel(str, Enum):
    """AI personalization levels"""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"
    CUSTOM = "custom"

@dataclass
class UserLimits:
    """User tier limits and quotas"""
    max_playlists: int = 10
    max_songs_per_playlist: int = 100
    max_ai_requests_per_day: int = 50
    max_api_calls_per_hour: int = 100
    max_storage_mb: int = 100
    max_concurrent_sessions: int = 1
    max_integrations: int = 3
    max_custom_models: int = 0
    max_collaboration_users: int = 5
    max_export_operations_per_day: int = 5
    max_backup_retention_days: int = 30
    can_use_premium_features: bool = False
    can_access_analytics: bool = False
    can_use_ai_composer: bool = False
    can_create_custom_algorithms: bool = False
    can_access_enterprise_features: bool = False
    priority_support: bool = False
    dedicated_resources: bool = False

@dataclass
class SecuritySettings:
    """User security configuration"""
    require_mfa: bool = False
    mfa_methods: List[str] = field(default_factory=lambda: ["email"])
    session_timeout_minutes: int = 1440
    max_failed_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    password_expiry_days: int = 90
    require_password_change: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    geo_restrictions: List[str] = field(default_factory=list)
    device_trust_required: bool = False
    biometric_enabled: bool = False
    hardware_token_required: bool = False
    risk_score_threshold: float = 0.7
    anomaly_detection_enabled: bool = True
    session_recording_enabled: bool = False
    encryption_level: str = "AES-256-GCM"

@dataclass
class AIPreferences:
    """AI personalization preferences"""
    personalization_level: AIPersonalizationLevel = AIPersonalizationLevel.BASIC
    recommendation_types: List[str] = field(default_factory=lambda: ["music", "artists"])
    learning_rate: float = 0.1
    privacy_mode: bool = False
    explicit_content_filter: bool = True
    mood_detection_enabled: bool = True
    context_awareness_enabled: bool = True
    cross_platform_sync: bool = True
    predictive_caching: bool = False
    custom_model_training: bool = False
    federated_learning_participation: bool = False
    bias_mitigation_enabled: bool = True
    explanation_level: str = "detailed"
    feedback_collection_enabled: bool = True
    a_b_testing_participation: bool = True

@dataclass
class IntegrationSettings:
    """External integrations configuration"""
    spotify_connected: bool = False
    spotify_premium: bool = False
    spotify_scopes: List[str] = field(default_factory=list)
    apple_music_connected: bool = False
    youtube_music_connected: bool = False
    amazon_music_connected: bool = False
    social_media_connections: Dict[str, bool] = field(default_factory=dict)
    webhook_endpoints: List[str] = field(default_factory=list)
    api_keys: Dict[str, str] = field(default_factory=dict)
    third_party_analytics: bool = False
    data_sharing_consent: Dict[str, bool] = field(default_factory=dict)
    sync_preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalyticsSettings:
    """User analytics and tracking preferences"""
    tracking_enabled: bool = True
    detailed_analytics: bool = False
    behavioral_tracking: bool = True
    performance_monitoring: bool = True
    usage_analytics: bool = True
    error_reporting: bool = True
    crash_reporting: bool = True
    feature_usage_tracking: bool = True
    a_b_test_participation: bool = True
    heatmap_tracking: bool = False
    session_replay_enabled: bool = False
    funnel_analysis_enabled: bool = True
    cohort_analysis_enabled: bool = False
    retention_analysis_enabled: bool = True
    churn_prediction_enabled: bool = False
    lifetime_value_tracking: bool = False

@dataclass
class ComplianceSettings:
    """Data protection and compliance settings"""
    gdpr_consent: bool = False
    ccpa_consent: bool = False
    coppa_compliant: bool = False
    data_retention_days: int = 365
    right_to_deletion: bool = True
    data_portability: bool = True
    consent_timestamp: Optional[datetime] = None
    privacy_policy_version: str = "1.0"
    terms_of_service_version: str = "1.0"
    cookie_consent: Dict[str, bool] = field(default_factory=dict)
    marketing_consent: bool = False
    profiling_consent: bool = False
    third_party_sharing_consent: bool = False
    data_processing_purposes: List[str] = field(default_factory=list)
    lawful_basis: str = "consent"
    data_controller_info: Dict[str, str] = field(default_factory=dict)

@dataclass
class UserProfile:
    """Comprehensive user profile with all features"""
    # Basic Information
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    email: str = ""
    username: str = ""
    display_name: str = ""
    first_name: str = ""
    last_name: str = ""
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    timezone: str = "UTC"
    language: str = "en"
    date_of_birth: Optional[datetime] = None
    phone_number: Optional[str] = None
    
    # Account Management
    tier: UserTier = UserTier.FREE
    status: UserStatus = UserStatus.PENDING_VERIFICATION
    email_verified: bool = False
    phone_verified: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None
    subscription_start: Optional[datetime] = None
    subscription_end: Optional[datetime] = None
    trial_expires_at: Optional[datetime] = None
    
    # Authentication & Security
    password_hash: Optional[str] = None
    password_salt: str = field(default_factory=lambda: secrets.token_hex(32))
    authentication_methods: List[AuthenticationMethod] = field(default_factory=lambda: [AuthenticationMethod.PASSWORD])
    mfa_secret: Optional[str] = None
    mfa_backup_codes: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None
    security_questions: Dict[str, str] = field(default_factory=dict)
    
    # Configuration Objects
    limits: UserLimits = field(default_factory=UserLimits)
    security_settings: SecuritySettings = field(default_factory=SecuritySettings)
    ai_preferences: AIPreferences = field(default_factory=AIPreferences)
    integration_settings: IntegrationSettings = field(default_factory=IntegrationSettings)
    analytics_settings: AnalyticsSettings = field(default_factory=AnalyticsSettings)
    compliance_settings: ComplianceSettings = field(default_factory=ComplianceSettings)
    
    # Usage Statistics
    total_login_count: int = 0
    total_session_duration: int = 0  # seconds
    total_api_calls: int = 0
    total_ai_requests: int = 0
    total_playlists_created: int = 0
    total_songs_analyzed: int = 0
    total_recommendations_received: int = 0
    total_data_exported: int = 0
    
    # Preferences
    theme: str = "auto"
    notifications_enabled: bool = True
    notification_preferences: Dict[str, bool] = field(default_factory=lambda: {
        "email": True, "push": True, "sms": False, "in_app": True
    })
    privacy_settings: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    experimental_features: bool = False
    
    # Social Features
    followers_count: int = 0
    following_count: int = 0
    public_profile: bool = False
    social_sharing_enabled: bool = True
    collaboration_enabled: bool = True
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    internal_notes: str = ""
    external_ids: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary with datetime serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO format
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    def is_premium(self) -> bool:
        """Check if user has premium tier or higher"""
        premium_tiers = {UserTier.PREMIUM, UserTier.ENTERPRISE, UserTier.VIP, UserTier.ADMIN, UserTier.SUPER_ADMIN}
        return self.tier in premium_tiers
    
    def is_enterprise(self) -> bool:
        """Check if user has enterprise tier or higher"""
        enterprise_tiers = {UserTier.ENTERPRISE, UserTier.VIP, UserTier.ADMIN, UserTier.SUPER_ADMIN}
        return self.tier in enterprise_tiers
    
    def is_admin(self) -> bool:
        """Check if user has admin privileges"""
        admin_tiers = {UserTier.ADMIN, UserTier.SUPER_ADMIN}
        return self.tier in admin_tiers
    
    def can_access_feature(self, feature: str) -> bool:
        """Check if user can access specific feature based on tier"""
        tier_features = {
            UserTier.FREE: {"basic_recommendations", "playlist_creation", "basic_analytics"},
            UserTier.PREMIUM: {"advanced_recommendations", "unlimited_playlists", "ai_composer", "premium_analytics"},
            UserTier.ENTERPRISE: {"custom_algorithms", "api_access", "team_collaboration", "advanced_integrations"},
            UserTier.VIP: {"priority_support", "beta_features", "custom_development", "dedicated_resources"},
            UserTier.ADMIN: {"user_management", "system_configuration", "audit_logs", "security_management"},
            UserTier.SUPER_ADMIN: {"full_system_access", "infrastructure_management", "compliance_tools"}
        }
        
        user_features = set()
        for tier in UserTier:
            user_features.update(tier_features.get(tier, set()))
            if tier == self.tier:
                break
        
        return feature in user_features

class UserSecurityManager:
    """Advanced security management for users"""
    
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.failed_attempts_cache = {}
        self.suspicious_activities = {}
        
    async def hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Hash password with bcrypt and salt"""
        if not salt:
            salt = secrets.token_hex(32)
        
        # Combine password with salt
        salted_password = f"{password}{salt}".encode('utf-8')
        
        # Hash with bcrypt
        hashed = bcrypt.hashpw(salted_password, bcrypt.gensalt(rounds=12))
        
        return hashed.decode('utf-8'), salt
    
    async def verify_password(self, password: str, salt: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        salted_password = f"{password}{salt}".encode('utf-8')
        return bcrypt.checkpw(salted_password, hashed_password.encode('utf-8'))
    
    async def generate_mfa_secret(self) -> str:
        """Generate MFA secret for TOTP"""
        return secrets.token_urlsafe(32)
    
    async def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate MFA backup codes"""
        return [secrets.token_hex(8) for _ in range(count)]
    
    async def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    async def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    async def calculate_risk_score(self, user: UserProfile, context: Dict[str, Any]) -> float:
        """Calculate user risk score based on behavior and context"""
        risk_factors = {
            'new_device': 0.3,
            'unusual_location': 0.4,
            'unusual_time': 0.2,
            'multiple_failed_attempts': 0.5,
            'suspicious_ip': 0.6,
            'tor_usage': 0.8,
            'vpn_usage': 0.3,
            'automated_behavior': 0.7
        }
        
        total_risk = 0.0
        for factor, weight in risk_factors.items():
            if context.get(factor, False):
                total_risk += weight
        
        # Normalize risk score to 0-1 range
        return min(total_risk, 1.0)
    
    async def detect_anomalies(self, user: UserProfile, activity: Dict[str, Any]) -> List[str]:
        """Detect anomalous user behavior"""
        anomalies = []
        
        # Time-based anomalies
        current_hour = datetime.now().hour
        if activity.get('login_hour') and abs(activity['login_hour'] - current_hour) > 6:
            anomalies.append('unusual_login_time')
        
        # Location-based anomalies
        if activity.get('location') and activity['location'] != user.location:
            anomalies.append('unusual_location')
        
        # Frequency anomalies
        if activity.get('requests_per_minute', 0) > 100:
            anomalies.append('high_request_frequency')
        
        return anomalies

class UserAnalyticsEngine:
    """Advanced analytics and insights for users"""
    
    def __init__(self):
        self.behavioral_patterns = {}
        self.usage_metrics = {}
        
    async def track_user_behavior(self, user_id: str, event: str, metadata: Dict[str, Any]):
        """Track user behavior events"""
        timestamp = datetime.now(timezone.utc)
        
        behavior_event = {
            'user_id': user_id,
            'event': event,
            'timestamp': timestamp.isoformat(),
            'metadata': metadata,
            'session_id': metadata.get('session_id'),
            'device_info': metadata.get('device_info'),
            'location': metadata.get('location')
        }
        
        # Store in analytics database
        await self._store_analytics_event(behavior_event)
        
        # Update real-time metrics
        await self._update_metrics(user_id, event, metadata)
    
    async def _store_analytics_event(self, event: Dict[str, Any]):
        """Store analytics event in database"""
        # Implementation would store in time-series database like InfluxDB or TimescaleDB
        logger.info("Analytics event stored", event=event)
    
    async def _update_metrics(self, user_id: str, event: str, metadata: Dict[str, Any]):
        """Update real-time metrics"""
        # Update Prometheus metrics
        if event == 'login':
            active_users_gauge.inc()
        elif event == 'user_created':
            tier = metadata.get('tier', 'unknown')
            user_creation_counter.labels(tier=tier, status='success').inc()
    
    async def generate_user_insights(self, user: UserProfile) -> Dict[str, Any]:
        """Generate personalized insights for user"""
        insights = {
            'listening_patterns': await self._analyze_listening_patterns(user),
            'usage_trends': await self._analyze_usage_trends(user),
            'recommendations': await self._generate_recommendations(user),
            'optimization_suggestions': await self._suggest_optimizations(user),
            'engagement_score': await self._calculate_engagement_score(user)
        }
        
        return insights
    
    async def _analyze_listening_patterns(self, user: UserProfile) -> Dict[str, Any]:
        """Analyze user's listening patterns"""
        return {
            'preferred_genres': ['electronic', 'ambient', 'classical'],
            'peak_listening_hours': [8, 12, 18, 22],
            'average_session_duration': 45,  # minutes
            'mood_preferences': ['focused', 'relaxed', 'energetic'],
            'seasonal_trends': {'spring': 'upbeat', 'summer': 'energetic', 'fall': 'mellow', 'winter': 'ambient'}
        }
    
    async def _analyze_usage_trends(self, user: UserProfile) -> Dict[str, Any]:
        """Analyze user's usage trends"""
        return {
            'daily_usage_average': 2.5,  # hours
            'feature_adoption_rate': 0.75,
            'engagement_trend': 'increasing',
            'churn_risk': 'low',
            'growth_trajectory': 'positive'
        }
    
    async def _generate_recommendations(self, user: UserProfile) -> List[Dict[str, Any]]:
        """Generate personalized recommendations"""
        return [
            {
                'type': 'feature',
                'title': 'Try AI Composer',
                'description': 'Create unique compositions based on your listening history',
                'confidence': 0.85
            },
            {
                'type': 'content',
                'title': 'Discover Similar Artists',
                'description': 'Based on your love for ambient music',
                'confidence': 0.92
            }
        ]
    
    async def _suggest_optimizations(self, user: UserProfile) -> List[str]:
        """Suggest profile optimizations"""
        suggestions = []
        
        if not user.ai_preferences.mood_detection_enabled:
            suggestions.append("Enable mood detection for better recommendations")
        
        if not user.integration_settings.spotify_connected:
            suggestions.append("Connect Spotify for richer music analysis")
        
        return suggestions
    
    async def _calculate_engagement_score(self, user: UserProfile) -> float:
        """Calculate user engagement score"""
        factors = {
            'login_frequency': 0.3,
            'feature_usage': 0.25,
            'content_creation': 0.2,
            'social_interaction': 0.15,
            'feedback_provided': 0.1
        }
        
        # Calculate based on user activity (mock calculation)
        score = 0.0
        if user.total_login_count > 10:
            score += factors['login_frequency']
        if user.total_playlists_created > 5:
            score += factors['content_creation']
        
        return min(score, 1.0)

class UserManager:
    """Enterprise user management system"""
    
    def __init__(self):
        self.security_manager = UserSecurityManager()
        self.analytics_engine = UserAnalyticsEngine()
        self.cache = {}  # Redis cache would be implemented here
        self.event_queue = []  # Kafka/RabbitMQ would be implemented here
        
    async def create_user(self, email: str, password: str, tier: UserTier = UserTier.FREE, 
                         profile_data: Dict[str, Any] = None) -> UserProfile:
        """Create new user with comprehensive setup"""
        with user_operation_duration.labels(operation='create').time():
            try:
                # Create base profile
                profile = UserProfile(
                    email=email,
                    tier=tier,
                    username=email.split('@')[0],  # Default username from email
                )
                
                # Apply profile data if provided
                if profile_data:
                    for key, value in profile_data.items():
                        if hasattr(profile, key):
                            setattr(profile, key, value)
                
                # Set tier-specific limits
                profile.limits = await self._get_tier_limits(tier)
                
                # Hash password
                if password:
                    profile.password_hash, profile.password_salt = await self.security_manager.hash_password(password)
                
                # Generate MFA secret
                profile.mfa_secret = await self.security_manager.generate_mfa_secret()
                profile.mfa_backup_codes = await self.security_manager.generate_backup_codes()
                
                # Set trial period for premium tiers
                if tier in [UserTier.PREMIUM, UserTier.ENTERPRISE]:
                    profile.trial_expires_at = datetime.now(timezone.utc) + timedelta(days=14)
                
                # Initialize feature flags based on tier
                profile.feature_flags = await self._get_tier_features(tier)
                
                # Track creation event
                await self.analytics_engine.track_user_behavior(
                    profile.user_id, 'user_created', 
                    {'tier': tier.value, 'source': 'registration'}
                )
                
                # Store in database (mock implementation)
                await self._store_user_profile(profile)
                
                # Send welcome notifications
                await self._send_welcome_notifications(profile)
                
                # Update metrics
                user_creation_counter.labels(tier=tier.value, status='success').inc()
                
                logger.info("User created successfully", user_id=profile.user_id, tier=tier.value)
                return profile
                
            except Exception as e:
                user_creation_counter.labels(tier=tier.value, status='error').inc()
                logger.error("Failed to create user", error=str(e))
                raise
    
    async def authenticate_user(self, email: str, password: str, context: Dict[str, Any] = None) -> Optional[UserProfile]:
        """Authenticate user with advanced security checks"""
        try:
            # Load user profile
            profile = await self._load_user_by_email(email)
            if not profile:
                return None
            
            # Check account status
            if profile.status != UserStatus.ACTIVE:
                security_events_counter.labels(event_type='login_blocked', severity='medium').inc()
                return None
            
            # Check if account is locked
            if profile.locked_until and profile.locked_until > datetime.now(timezone.utc):
                security_events_counter.labels(event_type='login_locked', severity='high').inc()
                return None
            
            # Verify password
            password_valid = await self.security_manager.verify_password(
                password, profile.password_salt, profile.password_hash
            )
            
            if not password_valid:
                await self._handle_failed_login(profile)
                return None
            
            # Calculate risk score
            risk_score = await self.security_manager.calculate_risk_score(profile, context or {})
            
            # Check if MFA is required
            if profile.security_settings.require_mfa or risk_score > profile.security_settings.risk_score_threshold:
                # Return partial authentication requiring MFA
                profile.requires_mfa = True
                return profile
            
            # Successful authentication
            await self._handle_successful_login(profile, context or {})
            
            return profile
            
        except Exception as e:
            logger.error("Authentication failed", email=email, error=str(e))
            return None
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> UserProfile:
        """Update user profile with validation and security checks"""
        profile = await self._load_user_by_id(user_id)
        if not profile:
            raise ValueError(f"User {user_id} not found")
        
        # Validate updates
        validated_updates = await self._validate_profile_updates(profile, updates)
        
        # Apply updates
        for key, value in validated_updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.updated_at = datetime.now(timezone.utc)
        
        # Store updated profile
        await self._store_user_profile(profile)
        
        # Track update event
        await self.analytics_engine.track_user_behavior(
            user_id, 'profile_updated', 
            {'fields_updated': list(validated_updates.keys())}
        )
        
        return profile
    
    async def upgrade_user_tier(self, user_id: str, new_tier: UserTier) -> UserProfile:
        """Upgrade user to new tier with feature migration"""
        profile = await self._load_user_by_id(user_id)
        if not profile:
            raise ValueError(f"User {user_id} not found")
        
        old_tier = profile.tier
        
        # Update tier and limits
        profile.tier = new_tier
        profile.limits = await self._get_tier_limits(new_tier)
        profile.feature_flags = await self._get_tier_features(new_tier)
        
        # Set subscription dates
        profile.subscription_start = datetime.now(timezone.utc)
        if new_tier in [UserTier.PREMIUM, UserTier.ENTERPRISE]:
            profile.subscription_end = datetime.now(timezone.utc) + timedelta(days=365)
        
        # Store updated profile
        await self._store_user_profile(profile)
        
        # Track upgrade event
        await self.analytics_engine.track_user_behavior(
            user_id, 'tier_upgraded', 
            {'old_tier': old_tier.value, 'new_tier': new_tier.value}
        )
        
        # Send upgrade notifications
        await self._send_upgrade_notifications(profile, old_tier)
        
        return profile
    
    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user insights and analytics"""
        profile = await self._load_user_by_id(user_id)
        if not profile:
            raise ValueError(f"User {user_id} not found")
        
        insights = await self.analytics_engine.generate_user_insights(profile)
        return insights
    
    async def _get_tier_limits(self, tier: UserTier) -> UserLimits:
        """Get limits configuration for user tier"""
        tier_configs = {
            UserTier.FREE: UserLimits(
                max_playlists=10,
                max_songs_per_playlist=100,
                max_ai_requests_per_day=50,
                max_api_calls_per_hour=100,
                max_storage_mb=100,
                max_concurrent_sessions=1,
                max_integrations=3,
                max_custom_models=0,
                can_use_premium_features=False,
                can_access_analytics=False
            ),
            UserTier.PREMIUM: UserLimits(
                max_playlists=100,
                max_songs_per_playlist=1000,
                max_ai_requests_per_day=500,
                max_api_calls_per_hour=1000,
                max_storage_mb=1000,
                max_concurrent_sessions=3,
                max_integrations=10,
                max_custom_models=3,
                can_use_premium_features=True,
                can_access_analytics=True,
                can_use_ai_composer=True
            ),
            UserTier.ENTERPRISE: UserLimits(
                max_playlists=1000,
                max_songs_per_playlist=10000,
                max_ai_requests_per_day=5000,
                max_api_calls_per_hour=10000,
                max_storage_mb=10000,
                max_concurrent_sessions=10,
                max_integrations=50,
                max_custom_models=20,
                can_use_premium_features=True,
                can_access_analytics=True,
                can_use_ai_composer=True,
                can_create_custom_algorithms=True,
                can_access_enterprise_features=True,
                priority_support=True
            ),
            UserTier.VIP: UserLimits(
                max_playlists=-1,  # Unlimited
                max_songs_per_playlist=-1,
                max_ai_requests_per_day=-1,
                max_api_calls_per_hour=-1,
                max_storage_mb=-1,
                max_concurrent_sessions=-1,
                max_integrations=-1,
                max_custom_models=-1,
                can_use_premium_features=True,
                can_access_analytics=True,
                can_use_ai_composer=True,
                can_create_custom_algorithms=True,
                can_access_enterprise_features=True,
                priority_support=True,
                dedicated_resources=True
            )
        }
        
        return tier_configs.get(tier, UserLimits())
    
    async def _get_tier_features(self, tier: UserTier) -> Dict[str, bool]:
        """Get feature flags for user tier"""
        base_features = {
            "basic_recommendations": True,
            "playlist_creation": True,
            "music_analysis": True,
            "basic_export": True
        }
        
        premium_features = {
            "advanced_recommendations": True,
            "ai_composer": True,
            "unlimited_playlists": True,
            "advanced_analytics": True,
            "api_access": True,
            "custom_integrations": True
        }
        
        enterprise_features = {
            "team_collaboration": True,
            "custom_algorithms": True,
            "white_label": True,
            "dedicated_support": True,
            "advanced_security": True,
            "compliance_tools": True
        }
        
        admin_features = {
            "user_management": True,
            "system_configuration": True,
            "audit_logs": True,
            "security_management": True,
            "billing_management": True
        }
        
        features = base_features.copy()
        
        if tier in [UserTier.PREMIUM, UserTier.ENTERPRISE, UserTier.VIP, UserTier.ADMIN, UserTier.SUPER_ADMIN]:
            features.update(premium_features)
        
        if tier in [UserTier.ENTERPRISE, UserTier.VIP, UserTier.ADMIN, UserTier.SUPER_ADMIN]:
            features.update(enterprise_features)
        
        if tier in [UserTier.ADMIN, UserTier.SUPER_ADMIN]:
            features.update(admin_features)
        
        return features
    
    async def _handle_failed_login(self, profile: UserProfile):
        """Handle failed login attempt"""
        profile.failed_login_attempts += 1
        
        # Lock account if too many failed attempts
        if profile.failed_login_attempts >= profile.security_settings.max_failed_login_attempts:
            profile.locked_until = datetime.now(timezone.utc) + timedelta(
                minutes=profile.security_settings.lockout_duration_minutes
            )
            security_events_counter.labels(event_type='account_locked', severity='high').inc()
        
        await self._store_user_profile(profile)
    
    async def _handle_successful_login(self, profile: UserProfile, context: Dict[str, Any]):
        """Handle successful login"""
        now = datetime.now(timezone.utc)
        
        # Reset failed attempts
        profile.failed_login_attempts = 0
        profile.locked_until = None
        
        # Update login statistics
        profile.total_login_count += 1
        profile.last_login_at = now
        profile.last_active_at = now
        
        # Store updated profile
        await self._store_user_profile(profile)
        
        # Track login event
        await self.analytics_engine.track_user_behavior(
            profile.user_id, 'login', context
        )
    
    async def _validate_profile_updates(self, profile: UserProfile, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Validate profile updates"""
        validated = {}
        
        # Email validation
        if 'email' in updates:
            email = updates['email']
            if '@' in email and '.' in email:
                validated['email'] = email
                validated['email_verified'] = False  # Require re-verification
        
        # Username validation
        if 'username' in updates:
            username = updates['username']
            if len(username) >= 3 and username.isalnum():
                validated['username'] = username
        
        # Basic field updates
        safe_fields = ['display_name', 'first_name', 'last_name', 'bio', 'website', 'location', 'timezone', 'language']
        for field in safe_fields:
            if field in updates:
                validated[field] = updates[field]
        
        return validated
    
    async def _store_user_profile(self, profile: UserProfile):
        """Store user profile in database"""
        # Mock implementation - would store in actual database
        logger.info("User profile stored", user_id=profile.user_id)
    
    async def _load_user_by_id(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile by ID"""
        # Mock implementation - would load from actual database
        return None
    
    async def _load_user_by_email(self, email: str) -> Optional[UserProfile]:
        """Load user profile by email"""
        # Mock implementation - would load from actual database
        return None
    
    async def _send_welcome_notifications(self, profile: UserProfile):
        """Send welcome notifications to new user"""
        logger.info("Welcome notifications sent", user_id=profile.user_id)
    
    async def _send_upgrade_notifications(self, profile: UserProfile, old_tier: UserTier):
        """Send tier upgrade notifications"""
        logger.info("Upgrade notifications sent", 
                   user_id=profile.user_id, 
                   old_tier=old_tier.value, 
                   new_tier=profile.tier.value)

# Factory functions for creating user profiles by tier
async def create_free_user_profile(email: str, password: str, **kwargs) -> UserProfile:
    """Create free tier user profile"""
    manager = UserManager()
    return await manager.create_user(email, password, UserTier.FREE, kwargs)

async def create_premium_user_profile(email: str, password: str, **kwargs) -> UserProfile:
    """Create premium tier user profile"""
    manager = UserManager()
    return await manager.create_user(email, password, UserTier.PREMIUM, kwargs)

async def create_enterprise_user_profile(email: str, password: str, **kwargs) -> UserProfile:
    """Create enterprise tier user profile"""
    manager = UserManager()
    return await manager.create_user(email, password, UserTier.ENTERPRISE, kwargs)

async def create_vip_user_profile(email: str, password: str, **kwargs) -> UserProfile:
    """Create VIP tier user profile"""
    manager = UserManager()
    return await manager.create_user(email, password, UserTier.VIP, kwargs)

# Export main classes and functions
__all__ = [
    'UserTier', 'UserStatus', 'AuthenticationMethod', 'SecurityLevel', 'AIPersonalizationLevel',
    'UserLimits', 'SecuritySettings', 'AIPreferences', 'IntegrationSettings', 
    'AnalyticsSettings', 'ComplianceSettings', 'UserProfile',
    'UserSecurityManager', 'UserAnalyticsEngine', 'UserManager',
    'create_free_user_profile', 'create_premium_user_profile', 
    'create_enterprise_user_profile', 'create_vip_user_profile'
]
