"""
Modèles de données pour le système de notifications avancées
============================================================

Modèles SQLAlchemy pour la persistance des notifications, templates,
et métriques avec support des relations complexes et audit trail.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
from enum import Enum, IntEnum
import json

from sqlalchemy import (
    Column, String, DateTime, JSON, Boolean, Integer, 
    ForeignKey, Text, Float, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.sql import func

Base = declarative_base()


class NotificationStatus(IntEnum):
    """Statuts des notifications avec priorités numériques"""
    PENDING = 1
    PROCESSING = 2
    SENT = 3
    DELIVERED = 4
    READ = 5
    FAILED = 6
    RETRYING = 7
    EXPIRED = 8
    CANCELLED = 9


class NotificationPriority(IntEnum):
    """Priorités des notifications"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class ChannelType(str, Enum):
    """Types de canaux de notification"""
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    DISCORD = "discord"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    DATABASE = "database"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"


class NotificationTemplate(Base):
    """Template de notification avec versioning et A/B testing"""
    __tablename__ = "notification_templates"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    channel_type = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False, default="1.0.0")
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Template content
    subject_template = Column(Text)
    body_template = Column(Text, nullable=False)
    html_template = Column(Text)
    metadata_template = Column(JSONB, default=dict)
    
    # Localization
    locale = Column(String(10), nullable=False, default="en")
    fallback_locale = Column(String(10), default="en")
    
    # A/B Testing
    ab_test_group = Column(String(50))
    ab_test_weight = Column(Float, default=1.0)
    
    # Analytics
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    avg_delivery_time = Column(Float, default=0.0)
    
    # Audit
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String(255))
    tenant_id = Column(String(255), nullable=False, index=True)
    
    # Relationships
    notifications = relationship("Notification", back_populates="template")
    
    __table_args__ = (
        Index('ix_template_channel_locale', 'channel_type', 'locale'),
        UniqueConstraint('name', 'version', 'tenant_id', name='uq_template_version'),
    )


class NotificationRule(Base):
    """Règles de routage et escalade des notifications"""
    __tablename__ = "notification_rules"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Conditions
    conditions = Column(JSONB, nullable=False)  # JSON query conditions
    priority_threshold = Column(Integer, default=1)
    
    # Actions
    channels = Column(JSONB, nullable=False)  # List of channel configs
    escalation_rules = Column(JSONB, default=list)
    retry_config = Column(JSONB, default=dict)
    
    # Rate limiting
    rate_limit_per_minute = Column(Integer, default=100)
    rate_limit_per_hour = Column(Integer, default=1000)
    burst_allowance = Column(Integer, default=10)
    
    # Scheduling
    active_hours = Column(JSONB)  # Time windows when rule is active
    timezone = Column(String(50), default="UTC")
    
    # Status
    is_active = Column(Boolean, default=True)
    execution_count = Column(Integer, default=0)
    last_executed = Column(DateTime(timezone=True))
    
    # Audit
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    tenant_id = Column(String(255), nullable=False, index=True)
    
    __table_args__ = (
        Index('ix_rule_tenant_active', 'tenant_id', 'is_active'),
    )


class Notification(Base):
    """Notification principale avec tracking complet"""
    __tablename__ = "notifications"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Basic info
    title = Column(String(500), nullable=False)
    message = Column(Text, nullable=False)
    priority = Column(Integer, nullable=False, default=NotificationPriority.NORMAL)
    status = Column(Integer, nullable=False, default=NotificationStatus.PENDING)
    
    # Recipient info
    recipient_id = Column(String(255), nullable=False, index=True)
    recipient_type = Column(String(50), nullable=False)  # user, group, role, etc.
    recipient_metadata = Column(JSONB, default=dict)
    
    # Channel info
    channel_type = Column(String(50), nullable=False)
    channel_config = Column(JSONB, default=dict)
    
    # Template
    template_id = Column(PG_UUID(as_uuid=True), ForeignKey("notification_templates.id"))
    template_data = Column(JSONB, default=dict)  # Variables for template
    
    # Delivery tracking
    scheduled_at = Column(DateTime(timezone=True))
    sent_at = Column(DateTime(timezone=True))
    delivered_at = Column(DateTime(timezone=True))
    read_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    
    # Retry management
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    next_retry_at = Column(DateTime(timezone=True))
    retry_backoff = Column(Float, default=1.0)  # Exponential backoff multiplier
    
    # External references
    external_id = Column(String(500))  # ID from external service (Slack message ID, etc.)
    correlation_id = Column(String(255), index=True)  # For tracking related notifications
    parent_notification_id = Column(PG_UUID(as_uuid=True), ForeignKey("notifications.id"))
    
    # Metadata
    metadata = Column(JSONB, default=dict)
    tags = Column(JSONB, default=list)
    source_system = Column(String(100))
    source_event = Column(String(255))
    
    # Analytics
    delivery_time_ms = Column(Integer)  # Time to deliver in milliseconds
    read_time_ms = Column(Integer)  # Time to read after delivery
    interaction_count = Column(Integer, default=0)
    
    # Audit
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    tenant_id = Column(String(255), nullable=False, index=True)
    
    # Relationships
    template = relationship("NotificationTemplate", back_populates="notifications")
    children = relationship("Notification", backref=backref("parent", remote_side=[id]))
    delivery_attempts = relationship("NotificationDeliveryAttempt", back_populates="notification")
    
    __table_args__ = (
        Index('ix_notification_recipient_status', 'recipient_id', 'status'),
        Index('ix_notification_tenant_created', 'tenant_id', 'created_at'),
        Index('ix_notification_correlation', 'correlation_id'),
        Index('ix_notification_scheduled', 'scheduled_at'),
    )


class NotificationDeliveryAttempt(Base):
    """Tentatives de livraison avec détails d'erreur"""
    __tablename__ = "notification_delivery_attempts"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    notification_id = Column(PG_UUID(as_uuid=True), ForeignKey("notifications.id"), nullable=False)
    
    attempt_number = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False)  # success, failed, timeout, rate_limited
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    completed_at = Column(DateTime(timezone=True))
    duration_ms = Column(Integer)
    
    # Details
    channel_response = Column(JSONB)  # Response from channel API
    error_code = Column(String(100))
    error_message = Column(Text)
    error_details = Column(JSONB)
    
    # Network details
    endpoint_url = Column(String(1000))
    http_status_code = Column(Integer)
    response_time_ms = Column(Integer)
    
    # Relationships
    notification = relationship("Notification", back_populates="delivery_attempts")
    
    __table_args__ = (
        Index('ix_attempt_notification', 'notification_id', 'attempt_number'),
        Index('ix_attempt_status_time', 'status', 'started_at'),
    )


class NotificationMetrics(Base):
    """Métriques agrégées pour analytics et reporting"""
    __tablename__ = "notification_metrics"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Time bucket (hourly aggregation)
    time_bucket = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Dimensions
    tenant_id = Column(String(255), nullable=False, index=True)
    channel_type = Column(String(50), nullable=False)
    priority = Column(Integer, nullable=False)
    recipient_type = Column(String(50), nullable=False)
    
    # Metrics
    total_sent = Column(Integer, default=0)
    total_delivered = Column(Integer, default=0)
    total_failed = Column(Integer, default=0)
    total_read = Column(Integer, default=0)
    
    avg_delivery_time_ms = Column(Float, default=0.0)
    avg_read_time_ms = Column(Float, default=0.0)
    p95_delivery_time_ms = Column(Float, default=0.0)
    p99_delivery_time_ms = Column(Float, default=0.0)
    
    retry_rate = Column(Float, default=0.0)  # Percentage of notifications that required retry
    success_rate = Column(Float, default=0.0)  # Percentage of successful deliveries
    read_rate = Column(Float, default=0.0)  # Percentage of notifications read
    
    # Error breakdown
    error_breakdown = Column(JSONB, default=dict)  # Error code -> count mapping
    
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    __table_args__ = (
        Index('ix_metrics_bucket_tenant', 'time_bucket', 'tenant_id'),
        Index('ix_metrics_channel_priority', 'channel_type', 'priority'),
        UniqueConstraint(
            'time_bucket', 'tenant_id', 'channel_type', 'priority', 'recipient_type',
            name='uq_metrics_dimensions'
        ),
    )


class NotificationPreference(Base):
    """Préférences utilisateur pour les notifications"""
    __tablename__ = "notification_preferences"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    user_id = Column(String(255), nullable=False, index=True)
    tenant_id = Column(String(255), nullable=False, index=True)
    
    # Channel preferences
    enabled_channels = Column(JSONB, nullable=False, default=list)
    disabled_channels = Column(JSONB, default=list)
    
    # Priority filters
    min_priority = Column(Integer, default=NotificationPriority.LOW)
    priority_overrides = Column(JSONB, default=dict)  # Category -> priority mapping
    
    # Quiet hours
    quiet_hours_start = Column(String(5))  # HH:MM format
    quiet_hours_end = Column(String(5))
    quiet_hours_timezone = Column(String(50), default="UTC")
    
    # Frequency limits
    max_notifications_per_hour = Column(Integer, default=50)
    max_notifications_per_day = Column(Integer, default=200)
    
    # Content preferences
    language = Column(String(10), default="en")
    template_style = Column(String(50), default="default")
    include_attachments = Column(Boolean, default=True)
    
    # Smart features
    enable_ai_filtering = Column(Boolean, default=False)
    enable_smart_batching = Column(Boolean, default=True)
    enable_sentiment_analysis = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('user_id', 'tenant_id', name='uq_user_tenant_preferences'),
    )


class NotificationQueue(Base):
    """Queue de notifications pour processing asynchrone"""
    __tablename__ = "notification_queue"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Queue info
    queue_name = Column(String(100), nullable=False, index=True)
    priority_score = Column(Float, nullable=False, index=True)  # For priority-based ordering
    
    # Payload
    notification_data = Column(JSONB, nullable=False)
    processing_metadata = Column(JSONB, default=dict)
    
    # Status
    status = Column(String(50), nullable=False, default="pending")  # pending, processing, completed, failed
    worker_id = Column(String(255))  # ID of worker processing this item
    
    # Timing
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    scheduled_for = Column(DateTime(timezone=True), index=True)
    started_processing_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Retry management
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    next_retry_at = Column(DateTime(timezone=True))
    
    # Error tracking
    last_error = Column(Text)
    error_count = Column(Integer, default=0)
    
    tenant_id = Column(String(255), nullable=False, index=True)
    
    __table_args__ = (
        Index('ix_queue_priority_status', 'priority_score', 'status'),
        Index('ix_queue_scheduled', 'scheduled_for'),
        Index('ix_queue_tenant_status', 'tenant_id', 'status'),
    )
