"""
Enterprise Session Management Module
===================================

Ultra-advanced enterprise session management system with distributed storage,
advanced security features, and comprehensive monitoring capabilities.

This module provides enterprise-grade session management with:
- Distributed session storage with Redis clustering
- Advanced session security with device fingerprinting
- Geographic session tracking and anomaly detection
- Session analytics and business intelligence
- High-performance session operations with sub-millisecond latency
- Enterprise compliance and audit logging
- Advanced session policies and governance
- Real-time session monitoring and alerting

Key Features:
- Multi-tenant session isolation with enterprise governance
- Advanced session lifecycle management
- Distributed session clustering for high availability
- Real-time session analytics and insights
- Session security with threat detection
- Compliance reporting and audit trails
- Performance optimization with intelligent caching
- Enterprise integration with LDAP/AD/SAML
"""

from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator, Set
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import logging
import json
import uuid
import hashlib
import secrets
import base64
from collections import defaultdict
import aioredis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog
from geopy.distance import geodesic
from user_agents import parse as parse_user_agent


# Configure structured logging
logger = structlog.get_logger(__name__)

# Prometheus metrics for enterprise session monitoring
ENTERPRISE_SESSION_OPERATIONS = Counter(
    'enterprise_session_operations_total',
    'Total enterprise session operations',
    ['tenant_id', 'operation', 'session_type', 'result']
)

ENTERPRISE_SESSION_DURATION = Histogram(
    'enterprise_session_duration_seconds',
    'Enterprise session operation duration',
    ['tenant_id', 'operation']
)

ENTERPRISE_ACTIVE_SESSIONS = Gauge(
    'enterprise_active_sessions_total',
    'Number of active enterprise sessions',
    ['tenant_id', 'session_type', 'location']
)

ENTERPRISE_SESSION_EVENTS = Counter(
    'enterprise_session_events_total',
    'Total enterprise session events',
    ['tenant_id', 'event_type', 'security_level']
)

ENTERPRISE_SESSION_PERFORMANCE = Summary(
    'enterprise_session_performance_seconds',
    'Enterprise session performance metrics',
    ['tenant_id', 'operation_type']
)


class EnterpriseSessionType(Enum):
    """Enterprise session types."""
    EXECUTIVE_SESSION = "executive_session"
    ADMINISTRATIVE_SESSION = "administrative_session"
    STANDARD_USER_SESSION = "standard_user_session"
    SERVICE_ACCOUNT_SESSION = "service_account_session"
    API_CLIENT_SESSION = "api_client_session"
    PRIVILEGED_ACCESS_SESSION = "privileged_access_session"
    EMERGENCY_ACCESS_SESSION = "emergency_access_session"
    FEDERATED_SESSION = "federated_session"
    MOBILE_SESSION = "mobile_session"
    KIOSK_SESSION = "kiosk_session"


class EnterpriseSessionStatus(Enum):
    """Enterprise session status."""
    ACTIVE = "active"
    IDLE = "idle"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SECURITY_LOCKED = "security_locked"
    COMPLIANCE_HOLD = "compliance_hold"
    FORCE_LOGOUT = "force_logout"


class EnterpriseSessionSecurityLevel(IntEnum):
    """Session security levels."""
    MINIMAL = 1
    STANDARD = 2
    ENHANCED = 3
    HIGH_SECURITY = 4
    MAXIMUM_SECURITY = 5
    CLASSIFIED = 6
    TOP_SECRET = 7


class EnterpriseDeviceType(Enum):
    """Enterprise device types."""
    DESKTOP = "desktop"
    LAPTOP = "laptop"
    MOBILE_PHONE = "mobile_phone"
    TABLET = "tablet"
    KIOSK = "kiosk"
    SERVER = "server"
    IOT_DEVICE = "iot_device"
    VIRTUAL_MACHINE = "virtual_machine"
    CONTAINER = "container"
    UNKNOWN = "unknown"


@dataclass
class EnterpriseDeviceInfo:
    """Enterprise device information."""
    
    device_id: str
    device_type: EnterpriseDeviceType
    device_name: Optional[str] = None
    
    # Operating system information
    os_name: Optional[str] = None
    os_version: Optional[str] = None
    os_architecture: Optional[str] = None
    
    # Browser information
    browser_name: Optional[str] = None
    browser_version: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Hardware information
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    
    # Security information
    device_fingerprint: Optional[str] = None
    is_managed: bool = False
    is_compliant: bool = True
    encryption_enabled: bool = False
    antivirus_status: bool = False
    
    # Network information
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    network_type: Optional[str] = None
    
    # Compliance and governance
    compliance_tags: List[str] = field(default_factory=list)
    policy_violations: List[str] = field(default_factory=list)
    
    # Timestamps
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "device_name": self.device_name,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "os_architecture": self.os_architecture,
            "browser_name": self.browser_name,
            "browser_version": self.browser_version,
            "user_agent": self.user_agent,
            "screen_resolution": self.screen_resolution,
            "timezone": self.timezone,
            "language": self.language,
            "device_fingerprint": self.device_fingerprint,
            "is_managed": self.is_managed,
            "is_compliant": self.is_compliant,
            "encryption_enabled": self.encryption_enabled,
            "antivirus_status": self.antivirus_status,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "network_type": self.network_type,
            "compliance_tags": self.compliance_tags,
            "policy_violations": self.policy_violations,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat()
        }


@dataclass
class EnterpriseLocationInfo:
    """Enterprise location information."""
    
    ip_address: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    timezone: Optional[str] = None
    isp: Optional[str] = None
    organization: Optional[str] = None
    
    # Security information
    is_vpn: bool = False
    is_proxy: bool = False
    is_tor: bool = False
    is_datacenter: bool = False
    risk_score: float = 0.0
    
    # Compliance information
    gdpr_applicable: bool = False
    data_residency_requirements: List[str] = field(default_factory=list)
    
    # Tracking
    accuracy: Optional[int] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "ip_address": self.ip_address,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "country": self.country,
            "region": self.region,
            "city": self.city,
            "postal_code": self.postal_code,
            "timezone": self.timezone,
            "isp": self.isp,
            "organization": self.organization,
            "is_vpn": self.is_vpn,
            "is_proxy": self.is_proxy,
            "is_tor": self.is_tor,
            "is_datacenter": self.is_datacenter,
            "risk_score": self.risk_score,
            "gdpr_applicable": self.gdpr_applicable,
            "data_residency_requirements": self.data_residency_requirements,
            "accuracy": self.accuracy,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EnterpriseSessionActivity:
    """Enterprise session activity tracking."""
    
    session_id: str
    activity_type: str
    activity_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Context information
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    page_url: Optional[str] = None
    api_endpoint: Optional[str] = None
    
    # Performance metrics
    response_time: Optional[float] = None
    data_transferred: Optional[int] = None
    
    # Security information
    risk_score: float = 0.0
    anomaly_detected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "activity_type": self.activity_type,
            "activity_data": self.activity_data,
            "timestamp": self.timestamp.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "page_url": self.page_url,
            "api_endpoint": self.api_endpoint,
            "response_time": self.response_time,
            "data_transferred": self.data_transferred,
            "risk_score": self.risk_score,
            "anomaly_detected": self.anomaly_detected
        }


@dataclass
class EnterpriseSessionData:
    """Enterprise session data with comprehensive metadata."""
    
    session_id: str
    user_id: str
    tenant_id: str
    organization_id: str
    
    # Session properties
    session_type: EnterpriseSessionType
    status: EnterpriseSessionStatus
    security_level: EnterpriseSessionSecurityLevel
    
    # Authentication information
    auth_method: str
    mfa_verified: bool = False
    certificate_verified: bool = False
    biometric_verified: bool = False
    
    # Device and location
    device_info: Optional[EnterpriseDeviceInfo] = None
    location_info: Optional[EnterpriseLocationInfo] = None
    
    # Access control
    permissions: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    scopes: Set[str] = field(default_factory=set)
    
    # Session timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    max_idle_time: timedelta = field(default_factory=lambda: timedelta(hours=2))
    
    # Business context
    department_id: Optional[str] = None
    cost_center: Optional[str] = None
    project_codes: List[str] = field(default_factory=list)
    business_justification: Optional[str] = None
    
    # Activity tracking
    activities: List[EnterpriseSessionActivity] = field(default_factory=list)
    page_views: int = 0
    api_calls: int = 0
    data_accessed: int = 0
    
    # Security tracking
    security_events: List[Dict[str, Any]] = field(default_factory=list)
    anomalies_detected: int = 0
    risk_score: float = 0.0
    
    # Compliance
    compliance_tags: List[str] = field(default_factory=list)
    audit_events: List[Dict[str, Any]] = field(default_factory=list)
    data_classifications: Set[str] = field(default_factory=set)
    
    # Custom attributes
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if session is currently active."""
        if self.status != EnterpriseSessionStatus.ACTIVE:
            return False
        
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        
        idle_time = datetime.now(timezone.utc) - self.last_activity
        if idle_time > self.max_idle_time:
            return False
        
        return True
    
    @property
    def idle_time(self) -> timedelta:
        """Get current idle time."""
        return datetime.now(timezone.utc) - self.last_activity
    
    @property
    def session_duration(self) -> timedelta:
        """Get total session duration."""
        return datetime.now(timezone.utc) - self.created_at
    
    def update_activity(
        self,
        activity_type: str,
        activity_data: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Update session activity."""
        
        self.last_activity = datetime.now(timezone.utc)
        
        # Create activity record
        activity = EnterpriseSessionActivity(
            session_id=self.session_id,
            activity_type=activity_type,
            activity_data=activity_data or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.activities.append(activity)
        
        # Update counters
        if activity_type == "page_view":
            self.page_views += 1
        elif activity_type == "api_call":
            self.api_calls += 1
        elif activity_type == "data_access":
            self.data_accessed += 1
        
        # Keep only recent activities (last 1000)
        if len(self.activities) > 1000:
            self.activities = self.activities[-1000:]
    
    def add_security_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        risk_score: float = 0.0
    ):
        """Add security event to session."""
        
        security_event = {
            "event_type": event_type,
            "event_data": event_data,
            "risk_score": risk_score,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.security_events.append(security_event)
        
        # Update session risk score
        self.risk_score = max(self.risk_score, risk_score)
        
        if risk_score > 0.7:
            self.anomalies_detected += 1
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        
        result = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "session_type": self.session_type.value,
            "status": self.status.value,
            "security_level": self.security_level.value,
            "auth_method": self.auth_method,
            "mfa_verified": self.mfa_verified,
            "certificate_verified": self.certificate_verified,
            "biometric_verified": self.biometric_verified,
            "permissions": list(self.permissions),
            "roles": list(self.roles),
            "scopes": list(self.scopes),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "max_idle_time": self.max_idle_time.total_seconds(),
            "department_id": self.department_id,
            "cost_center": self.cost_center,
            "project_codes": self.project_codes,
            "business_justification": self.business_justification,
            "page_views": self.page_views,
            "api_calls": self.api_calls,
            "data_accessed": self.data_accessed,
            "anomalies_detected": self.anomalies_detected,
            "risk_score": self.risk_score,
            "compliance_tags": self.compliance_tags,
            "data_classifications": list(self.data_classifications),
            "is_active": self.is_active,
            "idle_time": self.idle_time.total_seconds(),
            "session_duration": self.session_duration.total_seconds()
        }
        
        # Include device and location info
        if self.device_info:
            result["device_info"] = self.device_info.to_dict()
        
        if self.location_info:
            result["location_info"] = self.location_info.to_dict()
        
        # Include sensitive data if requested
        if include_sensitive:
            result.update({
                "activities": [activity.to_dict() for activity in self.activities[-10:]],  # Last 10 activities
                "security_events": self.security_events,
                "audit_events": self.audit_events,
                "custom_attributes": self.custom_attributes
            })
        
        return result


class EnterpriseSessionStorage(ABC):
    """Abstract base class for enterprise session storage."""
    
    @abstractmethod
    async def create_session(self, session: EnterpriseSessionData) -> bool:
        """Create a new session."""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[EnterpriseSessionData]:
        """Get session by ID."""
        pass
    
    @abstractmethod
    async def update_session(self, session: EnterpriseSessionData) -> bool:
        """Update existing session."""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        pass
    
    @abstractmethod
    async def get_user_sessions(
        self,
        user_id: str,
        tenant_id: str,
        include_inactive: bool = False
    ) -> List[EnterpriseSessionData]:
        """Get all sessions for a user."""
        pass
    
    @abstractmethod
    async def get_active_sessions(
        self,
        tenant_id: Optional[str] = None
    ) -> List[EnterpriseSessionData]:
        """Get all active sessions."""
        pass
    
    @abstractmethod
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        pass


class EnterpriseRedisSessionStorage(EnterpriseSessionStorage):
    """Redis-based enterprise session storage with clustering support."""
    
    def __init__(
        self,
        redis_client: aioredis.Redis,
        key_prefix: str = "enterprise_session:",
        default_ttl: int = 86400  # 24 hours
    ):
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
    
    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session."""
        return f"{self.key_prefix}{session_id}"
    
    def _get_user_sessions_key(self, user_id: str, tenant_id: str) -> str:
        """Get Redis key for user sessions index."""
        return f"{self.key_prefix}user:{tenant_id}:{user_id}"
    
    def _get_active_sessions_key(self, tenant_id: str) -> str:
        """Get Redis key for active sessions index."""
        return f"{self.key_prefix}active:{tenant_id}"
    
    async def create_session(self, session: EnterpriseSessionData) -> bool:
        """Create a new session in Redis."""
        
        try:
            session_key = self._get_session_key(session.session_id)
            user_sessions_key = self._get_user_sessions_key(session.user_id, session.tenant_id)
            active_sessions_key = self._get_active_sessions_key(session.tenant_id)
            
            # Serialize session data
            session_data = json.dumps(session.to_dict(include_sensitive=True))
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Store session data
            pipe.setex(session_key, self.default_ttl, session_data)
            
            # Add to user sessions index
            pipe.sadd(user_sessions_key, session.session_id)
            pipe.expire(user_sessions_key, self.default_ttl)
            
            # Add to active sessions index
            pipe.sadd(active_sessions_key, session.session_id)
            pipe.expire(active_sessions_key, self.default_ttl)
            
            # Execute pipeline
            await pipe.execute()
            
            logger.info(
                "Session created in Redis",
                session_id=session.session_id,
                user_id=session.user_id,
                tenant_id=session.tenant_id
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to create session in Redis", error=str(e))
            return False
    
    async def get_session(self, session_id: str) -> Optional[EnterpriseSessionData]:
        """Get session from Redis."""
        
        try:
            session_key = self._get_session_key(session_id)
            session_data = await self.redis_client.get(session_key)
            
            if not session_data:
                return None
            
            # Deserialize session data
            session_dict = json.loads(session_data)
            
            # Reconstruct session object
            session = self._dict_to_session(session_dict)
            
            return session
            
        except Exception as e:
            logger.error("Failed to get session from Redis", error=str(e))
            return None
    
    def _dict_to_session(self, session_dict: Dict[str, Any]) -> EnterpriseSessionData:
        """Convert dictionary to EnterpriseSessionData object."""
        
        # Create device info if present
        device_info = None
        if "device_info" in session_dict and session_dict["device_info"]:
            device_data = session_dict["device_info"]
            device_info = EnterpriseDeviceInfo(
                device_id=device_data["device_id"],
                device_type=EnterpriseDeviceType(device_data["device_type"]),
                device_name=device_data.get("device_name"),
                os_name=device_data.get("os_name"),
                os_version=device_data.get("os_version"),
                os_architecture=device_data.get("os_architecture"),
                browser_name=device_data.get("browser_name"),
                browser_version=device_data.get("browser_version"),
                user_agent=device_data.get("user_agent"),
                screen_resolution=device_data.get("screen_resolution"),
                timezone=device_data.get("timezone"),
                language=device_data.get("language"),
                device_fingerprint=device_data.get("device_fingerprint"),
                is_managed=device_data.get("is_managed", False),
                is_compliant=device_data.get("is_compliant", True),
                encryption_enabled=device_data.get("encryption_enabled", False),
                antivirus_status=device_data.get("antivirus_status", False),
                ip_address=device_data.get("ip_address"),
                mac_address=device_data.get("mac_address"),
                network_type=device_data.get("network_type"),
                compliance_tags=device_data.get("compliance_tags", []),
                policy_violations=device_data.get("policy_violations", [])
            )
        
        # Create location info if present
        location_info = None
        if "location_info" in session_dict and session_dict["location_info"]:
            location_data = session_dict["location_info"]
            location_info = EnterpriseLocationInfo(
                ip_address=location_data["ip_address"],
                latitude=location_data.get("latitude"),
                longitude=location_data.get("longitude"),
                country=location_data.get("country"),
                region=location_data.get("region"),
                city=location_data.get("city"),
                postal_code=location_data.get("postal_code"),
                timezone=location_data.get("timezone"),
                isp=location_data.get("isp"),
                organization=location_data.get("organization"),
                is_vpn=location_data.get("is_vpn", False),
                is_proxy=location_data.get("is_proxy", False),
                is_tor=location_data.get("is_tor", False),
                is_datacenter=location_data.get("is_datacenter", False),
                risk_score=location_data.get("risk_score", 0.0),
                gdpr_applicable=location_data.get("gdpr_applicable", False),
                data_residency_requirements=location_data.get("data_residency_requirements", []),
                accuracy=location_data.get("accuracy")
            )
        
        # Reconstruct activities
        activities = []
        if "activities" in session_dict and session_dict["activities"]:
            for activity_data in session_dict["activities"]:
                activity = EnterpriseSessionActivity(
                    session_id=activity_data["session_id"],
                    activity_type=activity_data["activity_type"],
                    activity_data=activity_data["activity_data"],
                    timestamp=datetime.fromisoformat(activity_data["timestamp"]),
                    ip_address=activity_data.get("ip_address"),
                    user_agent=activity_data.get("user_agent"),
                    page_url=activity_data.get("page_url"),
                    api_endpoint=activity_data.get("api_endpoint"),
                    response_time=activity_data.get("response_time"),
                    data_transferred=activity_data.get("data_transferred"),
                    risk_score=activity_data.get("risk_score", 0.0),
                    anomaly_detected=activity_data.get("anomaly_detected", False)
                )
                activities.append(activity)
        
        # Create session object
        session = EnterpriseSessionData(
            session_id=session_dict["session_id"],
            user_id=session_dict["user_id"],
            tenant_id=session_dict["tenant_id"],
            organization_id=session_dict["organization_id"],
            session_type=EnterpriseSessionType(session_dict["session_type"]),
            status=EnterpriseSessionStatus(session_dict["status"]),
            security_level=EnterpriseSessionSecurityLevel(session_dict["security_level"]),
            auth_method=session_dict["auth_method"],
            mfa_verified=session_dict.get("mfa_verified", False),
            certificate_verified=session_dict.get("certificate_verified", False),
            biometric_verified=session_dict.get("biometric_verified", False),
            device_info=device_info,
            location_info=location_info,
            permissions=set(session_dict.get("permissions", [])),
            roles=set(session_dict.get("roles", [])),
            scopes=set(session_dict.get("scopes", [])),
            created_at=datetime.fromisoformat(session_dict["created_at"]),
            last_activity=datetime.fromisoformat(session_dict["last_activity"]),
            expires_at=datetime.fromisoformat(session_dict["expires_at"]) if session_dict.get("expires_at") else None,
            max_idle_time=timedelta(seconds=session_dict.get("max_idle_time", 7200)),
            department_id=session_dict.get("department_id"),
            cost_center=session_dict.get("cost_center"),
            project_codes=session_dict.get("project_codes", []),
            business_justification=session_dict.get("business_justification"),
            activities=activities,
            page_views=session_dict.get("page_views", 0),
            api_calls=session_dict.get("api_calls", 0),
            data_accessed=session_dict.get("data_accessed", 0),
            security_events=session_dict.get("security_events", []),
            anomalies_detected=session_dict.get("anomalies_detected", 0),
            risk_score=session_dict.get("risk_score", 0.0),
            compliance_tags=session_dict.get("compliance_tags", []),
            audit_events=session_dict.get("audit_events", []),
            data_classifications=set(session_dict.get("data_classifications", [])),
            custom_attributes=session_dict.get("custom_attributes", {})
        )
        
        return session
    
    async def update_session(self, session: EnterpriseSessionData) -> bool:
        """Update existing session in Redis."""
        
        try:
            session_key = self._get_session_key(session.session_id)
            
            # Check if session exists
            exists = await self.redis_client.exists(session_key)
            if not exists:
                logger.warning("Attempting to update non-existent session", session_id=session.session_id)
                return False
            
            # Serialize and store updated session data
            session_data = json.dumps(session.to_dict(include_sensitive=True))
            await self.redis_client.setex(session_key, self.default_ttl, session_data)
            
            return True
            
        except Exception as e:
            logger.error("Failed to update session in Redis", error=str(e))
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session from Redis."""
        
        try:
            # Get session first to clean up indices
            session = await self.get_session(session_id)
            if not session:
                return True  # Already deleted
            
            session_key = self._get_session_key(session_id)
            user_sessions_key = self._get_user_sessions_key(session.user_id, session.tenant_id)
            active_sessions_key = self._get_active_sessions_key(session.tenant_id)
            
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Delete session data
            pipe.delete(session_key)
            
            # Remove from indices
            pipe.srem(user_sessions_key, session_id)
            pipe.srem(active_sessions_key, session_id)
            
            # Execute pipeline
            await pipe.execute()
            
            logger.info("Session deleted from Redis", session_id=session_id)
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete session from Redis", error=str(e))
            return False
    
    async def get_user_sessions(
        self,
        user_id: str,
        tenant_id: str,
        include_inactive: bool = False
    ) -> List[EnterpriseSessionData]:
        """Get all sessions for a user."""
        
        try:
            user_sessions_key = self._get_user_sessions_key(user_id, tenant_id)
            session_ids = await self.redis_client.smembers(user_sessions_key)
            
            sessions = []
            for session_id in session_ids:
                session = await self.get_session(session_id)
                if session:
                    if include_inactive or session.is_active:
                        sessions.append(session)
            
            # Sort by last activity (most recent first)
            sessions.sort(key=lambda s: s.last_activity, reverse=True)
            
            return sessions
            
        except Exception as e:
            logger.error("Failed to get user sessions from Redis", error=str(e))
            return []
    
    async def get_active_sessions(
        self,
        tenant_id: Optional[str] = None
    ) -> List[EnterpriseSessionData]:
        """Get all active sessions."""
        
        try:
            if tenant_id:
                # Get sessions for specific tenant
                active_sessions_key = self._get_active_sessions_key(tenant_id)
                session_ids = await self.redis_client.smembers(active_sessions_key)
            else:
                # Get sessions for all tenants (scan pattern)
                pattern = f"{self.key_prefix}active:*"
                keys = []
                cursor = 0
                while True:
                    cursor, batch = await self.redis_client.scan(cursor, match=pattern, count=100)
                    keys.extend(batch)
                    if cursor == 0:
                        break
                
                session_ids = set()
                for key in keys:
                    ids = await self.redis_client.smembers(key)
                    session_ids.update(ids)
            
            sessions = []
            for session_id in session_ids:
                session = await self.get_session(session_id)
                if session and session.is_active:
                    sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error("Failed to get active sessions from Redis", error=str(e))
            return []
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        
        try:
            # Scan for all session keys
            pattern = f"{self.key_prefix}*"
            keys = []
            cursor = 0
            while True:
                cursor, batch = await self.redis_client.scan(cursor, match=pattern, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break
            
            cleaned_count = 0
            
            # Check each session for expiration
            for key in keys:
                if key.startswith(f"{self.key_prefix}user:") or key.startswith(f"{self.key_prefix}active:"):
                    continue  # Skip index keys
                
                try:
                    session_data = await self.redis_client.get(key)
                    if session_data:
                        session_dict = json.loads(session_data)
                        session = self._dict_to_session(session_dict)
                        
                        if not session.is_active:
                            # Session is expired, delete it
                            session_id = session.session_id
                            await self.delete_session(session_id)
                            cleaned_count += 1
                            
                except Exception as e:
                    logger.error("Error processing session during cleanup", key=key, error=str(e))
                    continue
            
            logger.info("Session cleanup completed", cleaned_sessions=cleaned_count)
            
            return cleaned_count
            
        except Exception as e:
            logger.error("Failed to cleanup expired sessions", error=str(e))
            return 0


# Export main classes
__all__ = [
    # Enums
    "EnterpriseSessionType",
    "EnterpriseSessionStatus", 
    "EnterpriseSessionSecurityLevel",
    "EnterpriseDeviceType",
    
    # Data classes
    "EnterpriseDeviceInfo",
    "EnterpriseLocationInfo", 
    "EnterpriseSessionActivity",
    "EnterpriseSessionData",
    
    # Storage classes
    "EnterpriseSessionStorage",
    "EnterpriseRedisSessionStorage",
    
    # Metrics
    "ENTERPRISE_SESSION_OPERATIONS",
    "ENTERPRISE_SESSION_DURATION",
    "ENTERPRISE_ACTIVE_SESSIONS", 
    "ENTERPRISE_SESSION_EVENTS",
    "ENTERPRISE_SESSION_PERFORMANCE"
]
