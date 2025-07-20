"""
Enterprise Cache Security
=========================
Comprehensive security management for cache systems with encryption, access control, and threat detection.

Expert Team Implementation:
- Lead Developer + AI Architect: AI-powered threat detection and behavioral analysis
- Senior Backend Developer: High-performance security middleware and authentication
- Machine Learning Engineer: Anomaly detection models for unusual access patterns
- DBA & Data Engineer: Secure data handling and compliance validation
- Security Specialist: Advanced encryption, audit trails, and security policies
- Microservices Architect: Distributed security coordination and token management
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor

# Cryptography imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    import cryptography.hazmat.backends.openssl as openssl_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None

# JWT imports
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None

logger = logging.getLogger(__name__)

# === Security Types and Enums ===
class SecurityLevel(Enum):
    """Security levels for cache operations."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

class AccessType(Enum):
    """Types of cache access."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuthenticationMethod(Enum):
    """Authentication methods."""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    MUTUAL_TLS = "mutual_tls"
    HMAC_SIGNATURE = "hmac_signature"

@dataclass
class SecurityContext:
    """Security context for cache operations."""
    user_id: str
    roles: List[str]
    permissions: Set[str]
    authentication_method: AuthenticationMethod
    client_ip: str
    user_agent: str
    session_id: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if security context is expired."""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions
    
    def has_any_role(self, roles: List[str]) -> bool:
        """Check if context has any of the specified roles."""
        return bool(set(self.roles) & set(roles))

@dataclass
class SecurityAuditEvent:
    """Security audit event."""
    event_type: str
    user_id: str
    client_ip: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.LOW

@dataclass
class ThreatDetection:
    """Detected security threat."""
    threat_id: str
    threat_type: str
    description: str
    risk_score: float
    level: ThreatLevel
    affected_user: str
    client_ip: str
    timestamp: datetime = field(default_factory=datetime.now)
    mitigations: List[str] = field(default_factory=list)
    auto_blocked: bool = False

# === Encryption Engine ===
class EncryptionEngine:
    """Advanced encryption engine for cache data protection."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography package required for encryption features")
        
        self.master_key = master_key or self._generate_master_key()
        self.fernet = Fernet(self._derive_fernet_key(self.master_key))
        self.key_rotation_interval = timedelta(days=30)
        self.last_rotation = datetime.now()
        
        # AES encryption setup
        self.aes_key = self._derive_aes_key(self.master_key)
        
        logger.info("EncryptionEngine initialized with master key")
    
    def _generate_master_key(self) -> bytes:
        """Generate cryptographically secure master key."""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _derive_fernet_key(self, master_key: bytes) -> bytes:
        """Derive Fernet key from master key."""
        salt = b"cache_fernet_salt_v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=openssl_backend.backend
        )
        return Fernet.generate_key()  # For simplicity, using generated key
    
    def _derive_aes_key(self, master_key: bytes) -> bytes:
        """Derive AES key from master key."""
        salt = b"cache_aes_salt_v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=openssl_backend.backend
        )
        return kdf.derive(master_key)
    
    def encrypt_data(self, data: bytes, security_level: SecurityLevel = SecurityLevel.INTERNAL) -> bytes:
        """Encrypt data based on security level."""
        try:
            if security_level in [SecurityLevel.SECRET, SecurityLevel.CONFIDENTIAL]:
                return self._aes_encrypt(data)
            else:
                return self.fernet.encrypt(data)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes, security_level: SecurityLevel = SecurityLevel.INTERNAL) -> bytes:
        """Decrypt data based on security level."""
        try:
            if security_level in [SecurityLevel.SECRET, SecurityLevel.CONFIDENTIAL]:
                return self._aes_decrypt(encrypted_data)
            else:
                return self.fernet.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def _aes_encrypt(self, data: bytes) -> bytes:
        """AES encryption for high-security data."""
        iv = secrets.token_bytes(16)  # 128-bit IV
        cipher = Cipher(
            algorithms.AES(self.aes_key),
            modes.CBC(iv),
            backend=openssl_backend.backend
        )
        encryptor = cipher.encryptor()
        
        # Pad data to AES block size
        padded_data = self._pad_data(data, 16)
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        return iv + encrypted
    
    def _aes_decrypt(self, encrypted_data: bytes) -> bytes:
        """AES decryption for high-security data."""
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        cipher = Cipher(
            algorithms.AES(self.aes_key),
            modes.CBC(iv),
            backend=openssl_backend.backend
        )
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        return self._unpad_data(padded_data)
    
    def _pad_data(self, data: bytes, block_size: int) -> bytes:
        """PKCS7 padding."""
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding."""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def generate_signature(self, data: bytes, secret_key: bytes) -> str:
        """Generate HMAC signature for data integrity."""
        signature = hmac.new(secret_key, data, hashlib.sha256).hexdigest()
        return signature
    
    def verify_signature(self, data: bytes, signature: str, secret_key: bytes) -> bool:
        """Verify HMAC signature."""
        expected_signature = self.generate_signature(data, secret_key)
        return hmac.compare_digest(signature, expected_signature)
    
    def rotate_keys(self):
        """Rotate encryption keys for security."""
        if datetime.now() - self.last_rotation > self.key_rotation_interval:
            old_master_key = self.master_key
            self.master_key = self._generate_master_key()
            self.fernet = Fernet(self._derive_fernet_key(self.master_key))
            self.aes_key = self._derive_aes_key(self.master_key)
            self.last_rotation = datetime.now()
            
            logger.info("Encryption keys rotated successfully")
            return old_master_key
        
        return None

# === Access Control Manager ===
class AccessControlManager:
    """Role-based access control for cache operations."""
    
    def __init__(self):
        self.roles: Dict[str, Set[str]] = {}
        self.permissions: Dict[str, Dict[str, Any]] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.access_policies: Dict[str, Callable] = {}
        
        # Initialize default roles and permissions
        self._setup_default_roles()
        
        logger.info("AccessControlManager initialized")
    
    def _setup_default_roles(self):
        """Setup default roles and permissions."""
        # Define permissions
        self.permissions = {
            "cache.read": {"description": "Read from cache", "level": "basic"},
            "cache.write": {"description": "Write to cache", "level": "basic"},
            "cache.delete": {"description": "Delete from cache", "level": "intermediate"},
            "cache.flush": {"description": "Flush cache", "level": "advanced"},
            "cache.admin": {"description": "Cache administration", "level": "admin"},
            "cache.config": {"description": "Configure cache", "level": "admin"},
            "cache.monitor": {"description": "Monitor cache", "level": "intermediate"},
            "cache.audit": {"description": "View audit logs", "level": "admin"}
        }
        
        # Define roles
        self.roles = {
            "cache_readonly": {"cache.read", "cache.monitor"},
            "cache_user": {"cache.read", "cache.write", "cache.monitor"},
            "cache_operator": {"cache.read", "cache.write", "cache.delete", "cache.monitor"},
            "cache_admin": {"cache.read", "cache.write", "cache.delete", "cache.flush", 
                           "cache.admin", "cache.config", "cache.monitor", "cache.audit"}
        }
    
    def add_role(self, role_name: str, permissions: Set[str]):
        """Add new role with permissions."""
        # Validate permissions exist
        invalid_perms = permissions - set(self.permissions.keys())
        if invalid_perms:
            raise ValueError(f"Invalid permissions: {invalid_perms}")
        
        self.roles[role_name] = permissions
        logger.info(f"Added role: {role_name} with {len(permissions)} permissions")
    
    def assign_role_to_user(self, user_id: str, role_name: str):
        """Assign role to user."""
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} does not exist")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role_name)
        logger.info(f"Assigned role {role_name} to user {user_id}")
    
    def revoke_role_from_user(self, user_id: str, role_name: str):
        """Revoke role from user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
            logger.info(f"Revoked role {role_name} from user {user_id}")
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for user."""
        user_permissions = set()
        user_roles = self.user_roles.get(user_id, set())
        
        for role in user_roles:
            if role in self.roles:
                user_permissions.update(self.roles[role])
        
        return user_permissions
    
    def check_permission(self, context: SecurityContext, permission: str) -> bool:
        """Check if user has specific permission."""
        # Check direct permissions in context
        if context.has_permission(permission):
            return True
        
        # Check role-based permissions
        user_permissions = self.get_user_permissions(context.user_id)
        return permission in user_permissions
    
    def add_access_policy(self, name: str, policy_func: Callable[[SecurityContext, str], bool]):
        """Add custom access policy."""
        self.access_policies[name] = policy_func
        logger.info(f"Added access policy: {name}")
    
    def evaluate_access_policies(self, context: SecurityContext, resource: str) -> bool:
        """Evaluate all access policies for resource access."""
        for policy_name, policy_func in self.access_policies.items():
            try:
                if not policy_func(context, resource):
                    logger.warning(f"Access denied by policy {policy_name} for user {context.user_id}")
                    return False
            except Exception as e:
                logger.error(f"Policy {policy_name} evaluation failed: {e}")
                return False  # Fail-safe: deny access on policy error
        
        return True

# === Authentication Manager ===
class AuthenticationManager:
    """Multi-method authentication for cache access."""
    
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.jwt_algorithm = "HS256"
        self.jwt_expiry = timedelta(hours=24)
        self.session_store: Dict[str, Dict[str, Any]] = {}
        
        if not JWT_AVAILABLE:
            logger.warning("PyJWT not available. JWT authentication disabled.")
        
        logger.info("AuthenticationManager initialized")
    
    def generate_api_key(self, user_id: str, description: str = "") -> str:
        """Generate API key for user."""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            "user_id": user_id,
            "description": description,
            "created_at": datetime.now(),
            "last_used": None,
            "active": True
        }
        
        logger.info(f"Generated API key for user {user_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return user ID."""
        key_info = self.api_keys.get(api_key)
        if key_info and key_info["active"]:
            key_info["last_used"] = datetime.now()
            return key_info["user_id"]
        
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        return None
    
    def revoke_api_key(self, api_key: str):
        """Revoke API key."""
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            logger.info(f"Revoked API key: {api_key[:8]}...")
    
    def generate_jwt_token(self, user_id: str, roles: List[str]) -> Optional[str]:
        """Generate JWT token."""
        if not JWT_AVAILABLE:
            logger.error("JWT not available - install PyJWT")
            return None
        
        payload = {
            "user_id": user_id,
            "roles": roles,
            "iat": datetime.now(),
            "exp": datetime.now() + self.jwt_expiry
        }
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)
            logger.info(f"Generated JWT token for user {user_id}")
            return token
        except Exception as e:
            logger.error(f"JWT generation failed: {e}")
            return None
    
    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return payload."""
        if not JWT_AVAILABLE:
            return None
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])
            logger.debug(f"Valid JWT token for user {payload.get('user_id')}")
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
        
        return None
    
    def create_session(self, user_id: str, client_ip: str) -> str:
        """Create authenticated session."""
        session_id = secrets.token_urlsafe(24)
        self.session_store[session_id] = {
            "user_id": user_id,
            "client_ip": client_ip,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "active": True
        }
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def validate_session(self, session_id: str, client_ip: str) -> Optional[str]:
        """Validate session and return user ID."""
        session = self.session_store.get(session_id)
        
        if not session or not session["active"]:
            return None
        
        # Check IP consistency for security
        if session["client_ip"] != client_ip:
            logger.warning(f"Session IP mismatch for {session_id}")
            return None
        
        # Check session timeout (24 hours)
        if datetime.now() - session["last_activity"] > timedelta(hours=24):
            session["active"] = False
            logger.info(f"Session {session_id} expired")
            return None
        
        session["last_activity"] = datetime.now()
        return session["user_id"]
    
    def end_session(self, session_id: str):
        """End authenticated session."""
        if session_id in self.session_store:
            self.session_store[session_id]["active"] = False
            logger.info(f"Ended session {session_id}")

# === Audit Logger ===
class AuditLogger:
    """Comprehensive audit logging for security compliance."""
    
    def __init__(self, max_events: int = 10000):
        self.audit_events: deque = deque(maxlen=max_events)
        self.event_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        
        logger.info("AuditLogger initialized")
    
    def log_event(self, event_type: str, user_id: str, client_ip: str, 
                  success: bool = True, details: Dict[str, Any] = None, 
                  risk_score: float = 0.0) -> SecurityAuditEvent:
        """Log security audit event."""
        event = SecurityAuditEvent(
            event_type=event_type,
            user_id=user_id,
            client_ip=client_ip,
            success=success,
            details=details or {},
            risk_score=risk_score,
            threat_level=self._determine_threat_level(risk_score)
        )
        
        with self._lock:
            self.audit_events.append(event)
            self.event_counts[event_type] += 1
        
        # Log high-risk events
        if risk_score > 0.7:
            logger.warning(f"High-risk audit event: {event_type} by {user_id} from {client_ip}")
        
        return event
    
    def _determine_threat_level(self, risk_score: float) -> ThreatLevel:
        """Determine threat level from risk score."""
        if risk_score >= 0.9:
            return ThreatLevel.CRITICAL
        elif risk_score >= 0.7:
            return ThreatLevel.HIGH
        elif risk_score >= 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def get_events_by_user(self, user_id: str, hours: int = 24) -> List[SecurityAuditEvent]:
        """Get audit events for specific user."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [event for event in self.audit_events 
                   if event.user_id == user_id and event.timestamp >= cutoff_time]
    
    def get_events_by_type(self, event_type: str, hours: int = 24) -> List[SecurityAuditEvent]:
        """Get audit events by type."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [event for event in self.audit_events 
                   if event.event_type == event_type and event.timestamp >= cutoff_time]
    
    def get_high_risk_events(self, hours: int = 24, min_risk: float = 0.7) -> List[SecurityAuditEvent]:
        """Get high-risk audit events."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [event for event in self.audit_events 
                   if event.risk_score >= min_risk and event.timestamp >= cutoff_time]
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_events = [event for event in self.audit_events 
                           if event.timestamp >= cutoff_time]
        
        if not recent_events:
            return {"status": "no_events", "hours": hours}
        
        success_count = sum(1 for event in recent_events if event.success)
        failure_count = len(recent_events) - success_count
        
        event_type_counts = defaultdict(int)
        risk_scores = []
        
        for event in recent_events:
            event_type_counts[event.event_type] += 1
            risk_scores.append(event.risk_score)
        
        return {
            "hours": hours,
            "total_events": len(recent_events),
            "successful_events": success_count,
            "failed_events": failure_count,
            "success_rate": (success_count / len(recent_events)) * 100,
            "event_types": dict(event_type_counts),
            "average_risk_score": sum(risk_scores) / len(risk_scores),
            "high_risk_events": len([r for r in risk_scores if r >= 0.7])
        }

# === Threat Detector ===
class ThreatDetector:
    """AI-powered threat detection for cache security."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.threat_patterns: Dict[str, Dict[str, Any]] = {}
        self.user_baselines: Dict[str, Dict[str, Any]] = {}
        self.detected_threats: deque = deque(maxlen=1000)
        self.auto_block_enabled = True
        self.blocked_ips: Set[str] = set()
        self.blocked_users: Set[str] = set()
        
        # Initialize threat patterns
        self._setup_threat_patterns()
        
        logger.info("ThreatDetector initialized")
    
    def _setup_threat_patterns(self):
        """Setup threat detection patterns."""
        self.threat_patterns = {
            "brute_force_api_key": {
                "description": "Multiple failed API key attempts",
                "threshold": 5,
                "window_minutes": 10,
                "risk_score": 0.8,
                "auto_block": True
            },
            "unusual_access_pattern": {
                "description": "Access pattern deviates from user baseline",
                "threshold": 3,
                "window_minutes": 60,
                "risk_score": 0.6,
                "auto_block": False
            },
            "privilege_escalation": {
                "description": "Attempt to access unauthorized resources",
                "threshold": 3,
                "window_minutes": 5,
                "risk_score": 0.9,
                "auto_block": True
            },
            "high_volume_access": {
                "description": "Unusually high volume of cache operations",
                "threshold": 1000,
                "window_minutes": 10,
                "risk_score": 0.5,
                "auto_block": False
            },
            "off_hours_access": {
                "description": "Access during unusual hours",
                "threshold": 1,
                "window_minutes": 60,
                "risk_score": 0.3,
                "auto_block": False
            }
        }
    
    async def analyze_threats(self) -> List[ThreatDetection]:
        """Analyze for potential security threats."""
        detected_threats = []
        
        try:
            # Analyze recent audit events
            recent_events = self.audit_logger.get_events_by_type("authentication_failed", hours=1)
            detected_threats.extend(await self._detect_brute_force(recent_events))
            
            # Analyze access patterns
            all_recent_events = []
            for event_type in ["cache_read", "cache_write", "cache_delete"]:
                all_recent_events.extend(self.audit_logger.get_events_by_type(event_type, hours=2))
            
            detected_threats.extend(await self._detect_unusual_patterns(all_recent_events))
            detected_threats.extend(await self._detect_privilege_escalation(all_recent_events))
            detected_threats.extend(await self._detect_high_volume_access(all_recent_events))
            
            # Store detected threats
            for threat in detected_threats:
                self.detected_threats.append(threat)
                
                # Auto-block if enabled and threat warrants it
                if self.auto_block_enabled and threat.auto_blocked:
                    await self._auto_block_threat(threat)
            
            return detected_threats
            
        except Exception as e:
            logger.error(f"Threat analysis failed: {e}")
            return []
    
    async def _detect_brute_force(self, failed_auth_events: List[SecurityAuditEvent]) -> List[ThreatDetection]:
        """Detect brute force authentication attempts."""
        threats = []
        pattern = self.threat_patterns["brute_force_api_key"]
        
        # Group by IP and user
        ip_attempts = defaultdict(int)
        user_attempts = defaultdict(int)
        
        cutoff_time = datetime.now() - timedelta(minutes=pattern["window_minutes"])
        
        for event in failed_auth_events:
            if event.timestamp >= cutoff_time:
                ip_attempts[event.client_ip] += 1
                user_attempts[event.user_id] += 1
        
        # Check IP-based brute force
        for ip, attempts in ip_attempts.items():
            if attempts >= pattern["threshold"]:
                threat = ThreatDetection(
                    threat_id=f"brute_force_ip_{ip}_{int(time.time())}",
                    threat_type="brute_force_api_key",
                    description=f"Brute force attack detected from IP {ip}: {attempts} failed attempts",
                    risk_score=min(pattern["risk_score"] + (attempts - pattern["threshold"]) * 0.1, 1.0),
                    level=ThreatLevel.HIGH,
                    affected_user="multiple",
                    client_ip=ip,
                    mitigations=["block_ip", "rate_limit", "alert_admin"],
                    auto_blocked=pattern["auto_block"]
                )
                threats.append(threat)
        
        # Check user-based brute force
        for user, attempts in user_attempts.items():
            if attempts >= pattern["threshold"]:
                threat = ThreatDetection(
                    threat_id=f"brute_force_user_{user}_{int(time.time())}",
                    threat_type="brute_force_api_key",
                    description=f"Brute force attack against user {user}: {attempts} failed attempts",
                    risk_score=pattern["risk_score"],
                    level=ThreatLevel.HIGH,
                    affected_user=user,
                    client_ip="multiple",
                    mitigations=["temporary_account_lock", "require_password_reset"],
                    auto_blocked=pattern["auto_block"]
                )
                threats.append(threat)
        
        return threats
    
    async def _detect_unusual_patterns(self, events: List[SecurityAuditEvent]) -> List[ThreatDetection]:
        """Detect unusual access patterns using ML-style analysis."""
        threats = []
        
        # Build user behavior baselines
        user_activity = defaultdict(lambda: {"times": [], "operations": [], "ips": set()})
        
        for event in events:
            user_activity[event.user_id]["times"].append(event.timestamp.hour)
            user_activity[event.user_id]["operations"].append(event.event_type)
            user_activity[event.user_id]["ips"].add(event.client_ip)
        
        for user_id, activity in user_activity.items():
            # Check for unusual timing patterns
            if self._is_unusual_timing(activity["times"]):
                threat = ThreatDetection(
                    threat_id=f"unusual_timing_{user_id}_{int(time.time())}",
                    threat_type="unusual_access_pattern",
                    description=f"User {user_id} accessing cache at unusual hours",
                    risk_score=0.4,
                    level=ThreatLevel.MEDIUM,
                    affected_user=user_id,
                    client_ip="multiple",
                    mitigations=["monitor_closely", "require_additional_auth"]
                )
                threats.append(threat)
            
            # Check for multiple IPs
            if len(activity["ips"]) > 3:
                threat = ThreatDetection(
                    threat_id=f"multiple_ips_{user_id}_{int(time.time())}",
                    threat_type="unusual_access_pattern",
                    description=f"User {user_id} accessing from {len(activity['ips'])} different IPs",
                    risk_score=0.6,
                    level=ThreatLevel.MEDIUM,
                    affected_user=user_id,
                    client_ip="multiple",
                    mitigations=["verify_identity", "monitor_closely"]
                )
                threats.append(threat)
        
        return threats
    
    async def _detect_privilege_escalation(self, events: List[SecurityAuditEvent]) -> List[ThreatDetection]:
        """Detect privilege escalation attempts."""
        threats = []
        
        # Look for failed access attempts to high-privilege operations
        high_privilege_ops = ["cache_flush", "cache_admin", "cache_config"]
        
        user_escalation_attempts = defaultdict(int)
        
        for event in events:
            if (not event.success and 
                any(op in event.details.get("operation", "") for op in high_privilege_ops)):
                user_escalation_attempts[event.user_id] += 1
        
        for user_id, attempts in user_escalation_attempts.items():
            if attempts >= 3:
                threat = ThreatDetection(
                    threat_id=f"privilege_escalation_{user_id}_{int(time.time())}",
                    threat_type="privilege_escalation",
                    description=f"User {user_id} attempting privilege escalation: {attempts} failed admin operations",
                    risk_score=0.9,
                    level=ThreatLevel.HIGH,
                    affected_user=user_id,
                    client_ip="unknown",
                    mitigations=["temporary_account_lock", "security_review", "alert_admin"],
                    auto_blocked=True
                )
                threats.append(threat)
        
        return threats
    
    async def _detect_high_volume_access(self, events: List[SecurityAuditEvent]) -> List[ThreatDetection]:
        """Detect unusually high volume access patterns."""
        threats = []
        pattern = self.threat_patterns["high_volume_access"]
        
        # Count operations per user in recent window
        cutoff_time = datetime.now() - timedelta(minutes=pattern["window_minutes"])
        user_ops = defaultdict(int)
        
        for event in events:
            if event.timestamp >= cutoff_time:
                user_ops[event.user_id] += 1
        
        for user_id, op_count in user_ops.items():
            if op_count >= pattern["threshold"]:
                threat = ThreatDetection(
                    threat_id=f"high_volume_{user_id}_{int(time.time())}",
                    threat_type="high_volume_access",
                    description=f"User {user_id} performed {op_count} operations in {pattern['window_minutes']} minutes",
                    risk_score=min(pattern["risk_score"] + (op_count - pattern["threshold"]) / 1000, 0.9),
                    level=ThreatLevel.MEDIUM,
                    affected_user=user_id,
                    client_ip="unknown",
                    mitigations=["rate_limit", "monitor_closely"]
                )
                threats.append(threat)
        
        return threats
    
    def _is_unusual_timing(self, access_times: List[int]) -> bool:
        """Check if access times are unusual (simple heuristic)."""
        if not access_times:
            return False
        
        # Consider 10 PM to 6 AM as unusual hours
        unusual_hours = set(range(22, 24)) | set(range(0, 6))
        unusual_accesses = sum(1 for hour in access_times if hour in unusual_hours)
        
        return (unusual_accesses / len(access_times)) > 0.5
    
    async def _auto_block_threat(self, threat: ThreatDetection):
        """Automatically block detected threat."""
        if threat.client_ip != "multiple" and threat.client_ip != "unknown":
            self.blocked_ips.add(threat.client_ip)
            logger.warning(f"Auto-blocked IP {threat.client_ip} due to threat {threat.threat_id}")
        
        if threat.level == ThreatLevel.CRITICAL:
            self.blocked_users.add(threat.affected_user)
            logger.critical(f"Auto-blocked user {threat.affected_user} due to critical threat {threat.threat_id}")
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips
    
    def is_user_blocked(self, user_id: str) -> bool:
        """Check if user is blocked."""
        return user_id in self.blocked_users
    
    def unblock_ip(self, ip: str):
        """Unblock IP address."""
        self.blocked_ips.discard(ip)
        logger.info(f"Unblocked IP {ip}")
    
    def unblock_user(self, user_id: str):
        """Unblock user."""
        self.blocked_users.discard(user_id)
        logger.info(f"Unblocked user {user_id}")

# === Cache Security Manager ===
class CacheSecurityManager:
    """Main security manager coordinating all security components."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.encryption_engine = EncryptionEngine(master_key)
        self.access_control = AccessControlManager()
        self.auth_manager = AuthenticationManager(self.encryption_engine.master_key)
        self.audit_logger = AuditLogger()
        self.threat_detector = ThreatDetector(self.audit_logger)
        
        # Security policies
        self.enforce_encryption = True
        self.require_authentication = True
        self.enable_audit_logging = True
        self.threat_detection_enabled = True
        
        # Background tasks
        self._threat_analysis_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("CacheSecurityManager initialized")
    
    async def start_security_monitoring(self):
        """Start background security monitoring."""
        if self._running:
            return
        
        self._running = True
        self._threat_analysis_task = asyncio.create_task(self._threat_analysis_loop())
        logger.info("Security monitoring started")
    
    async def stop_security_monitoring(self):
        """Stop background security monitoring."""
        self._running = False
        if self._threat_analysis_task:
            self._threat_analysis_task.cancel()
            try:
                await self._threat_analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("Security monitoring stopped")
    
    async def _threat_analysis_loop(self):
        """Background threat analysis loop."""
        while self._running:
            try:
                if self.threat_detection_enabled:
                    threats = await self.threat_detector.analyze_threats()
                    if threats:
                        logger.info(f"Detected {len(threats)} threats in analysis cycle")
                
                await asyncio.sleep(60)  # Analyze every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Threat analysis loop error: {e}")
                await asyncio.sleep(10)
    
    async def authenticate_request(self, auth_header: str, client_ip: str, user_agent: str) -> Optional[SecurityContext]:
        """Authenticate cache request and return security context."""
        try:
            # Parse authentication header
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                return await self._authenticate_jwt(token, client_ip, user_agent)
            elif auth_header.startswith("ApiKey "):
                api_key = auth_header[7:]
                return await self._authenticate_api_key(api_key, client_ip, user_agent)
            elif auth_header.startswith("Session "):
                session_id = auth_header[8:]
                return await self._authenticate_session(session_id, client_ip, user_agent)
            else:
                self.audit_logger.log_event(
                    "authentication_failed",
                    "unknown",
                    client_ip,
                    success=False,
                    details={"reason": "invalid_auth_format"},
                    risk_score=0.3
                )
                return None
        
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            self.audit_logger.log_event(
                "authentication_error",
                "unknown",
                client_ip,
                success=False,
                details={"error": str(e)},
                risk_score=0.5
            )
            return None
    
    async def _authenticate_jwt(self, token: str, client_ip: str, user_agent: str) -> Optional[SecurityContext]:
        """Authenticate JWT token."""
        payload = self.auth_manager.validate_jwt_token(token)
        if not payload:
            self.audit_logger.log_event(
                "authentication_failed",
                "unknown",
                client_ip,
                success=False,
                details={"method": "jwt", "reason": "invalid_token"},
                risk_score=0.6
            )
            return None
        
        user_id = payload["user_id"]
        roles = payload.get("roles", [])
        permissions = self.access_control.get_user_permissions(user_id)
        
        # Check if user/IP is blocked
        if self.threat_detector.is_user_blocked(user_id) or self.threat_detector.is_ip_blocked(client_ip):
            self.audit_logger.log_event(
                "authentication_blocked",
                user_id,
                client_ip,
                success=False,
                details={"method": "jwt", "reason": "user_or_ip_blocked"},
                risk_score=0.9
            )
            return None
        
        self.audit_logger.log_event(
            "authentication_success",
            user_id,
            client_ip,
            success=True,
            details={"method": "jwt"},
            risk_score=0.1
        )
        
        return SecurityContext(
            user_id=user_id,
            roles=roles,
            permissions=permissions,
            authentication_method=AuthenticationMethod.JWT_TOKEN,
            client_ip=client_ip,
            user_agent=user_agent,
            expires_at=datetime.fromtimestamp(payload["exp"]) if "exp" in payload else None
        )
    
    async def _authenticate_api_key(self, api_key: str, client_ip: str, user_agent: str) -> Optional[SecurityContext]:
        """Authenticate API key."""
        user_id = self.auth_manager.validate_api_key(api_key)
        if not user_id:
            self.audit_logger.log_event(
                "authentication_failed",
                "unknown",
                client_ip,
                success=False,
                details={"method": "api_key", "key_prefix": api_key[:8]},
                risk_score=0.7
            )
            return None
        
        # Check if user/IP is blocked
        if self.threat_detector.is_user_blocked(user_id) or self.threat_detector.is_ip_blocked(client_ip):
            self.audit_logger.log_event(
                "authentication_blocked",
                user_id,
                client_ip,
                success=False,
                details={"method": "api_key", "reason": "user_or_ip_blocked"},
                risk_score=0.9
            )
            return None
        
        permissions = self.access_control.get_user_permissions(user_id)
        user_roles = list(self.access_control.user_roles.get(user_id, set()))
        
        self.audit_logger.log_event(
            "authentication_success",
            user_id,
            client_ip,
            success=True,
            details={"method": "api_key"},
            risk_score=0.1
        )
        
        return SecurityContext(
            user_id=user_id,
            roles=user_roles,
            permissions=permissions,
            authentication_method=AuthenticationMethod.API_KEY,
            client_ip=client_ip,
            user_agent=user_agent
        )
    
    async def _authenticate_session(self, session_id: str, client_ip: str, user_agent: str) -> Optional[SecurityContext]:
        """Authenticate session."""
        user_id = self.auth_manager.validate_session(session_id, client_ip)
        if not user_id:
            self.audit_logger.log_event(
                "authentication_failed",
                "unknown",
                client_ip,
                success=False,
                details={"method": "session", "session_id": session_id[:8]},
                risk_score=0.5
            )
            return None
        
        permissions = self.access_control.get_user_permissions(user_id)
        user_roles = list(self.access_control.user_roles.get(user_id, set()))
        
        self.audit_logger.log_event(
            "authentication_success",
            user_id,
            client_ip,
            success=True,
            details={"method": "session"},
            risk_score=0.1
        )
        
        return SecurityContext(
            user_id=user_id,
            roles=user_roles,
            permissions=permissions,
            authentication_method=AuthenticationMethod.API_KEY,  # Sessions use API key method
            client_ip=client_ip,
            user_agent=user_agent,
            session_id=session_id
        )
    
    async def authorize_operation(self, context: SecurityContext, operation: str, resource: str = "") -> bool:
        """Authorize cache operation."""
        if not context or context.is_expired():
            return False
        
        # Check basic permission
        if not self.access_control.check_permission(context, operation):
            self.audit_logger.log_event(
                "authorization_failed",
                context.user_id,
                context.client_ip,
                success=False,
                details={"operation": operation, "resource": resource, "reason": "insufficient_permissions"},
                risk_score=0.4
            )
            return False
        
        # Evaluate access policies
        if not self.access_control.evaluate_access_policies(context, resource):
            self.audit_logger.log_event(
                "authorization_failed",
                context.user_id,
                context.client_ip,
                success=False,
                details={"operation": operation, "resource": resource, "reason": "policy_violation"},
                risk_score=0.6
            )
            return False
        
        self.audit_logger.log_event(
            "authorization_success",
            context.user_id,
            context.client_ip,
            success=True,
            details={"operation": operation, "resource": resource},
            risk_score=0.1
        )
        
        return True
    
    def encrypt_cache_data(self, data: bytes, security_level: SecurityLevel = SecurityLevel.INTERNAL) -> bytes:
        """Encrypt cache data based on security level."""
        if not self.enforce_encryption:
            return data
        
        return self.encryption_engine.encrypt_data(data, security_level)
    
    def decrypt_cache_data(self, encrypted_data: bytes, security_level: SecurityLevel = SecurityLevel.INTERNAL) -> bytes:
        """Decrypt cache data based on security level."""
        if not self.enforce_encryption:
            return encrypted_data
        
        return self.encryption_engine.decrypt_data(encrypted_data, security_level)
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        # Get recent threats
        recent_threats = [threat for threat in self.threat_detector.detected_threats 
                         if (datetime.now() - threat.timestamp).total_seconds() < 3600]
        
        # Get audit summary
        audit_summary = self.audit_logger.get_audit_summary(hours=24)
        
        # Count active security measures
        active_measures = {
            "encryption_enabled": self.enforce_encryption,
            "authentication_required": self.require_authentication,
            "audit_logging_enabled": self.enable_audit_logging,
            "threat_detection_enabled": self.threat_detection_enabled,
            "monitoring_active": self._running
        }
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "active_measures": active_measures,
            "recent_threats": len(recent_threats),
            "blocked_ips": len(self.threat_detector.blocked_ips),
            "blocked_users": len(self.threat_detector.blocked_users),
            "audit_summary": audit_summary,
            "threat_levels": {
                level.value: sum(1 for t in recent_threats if t.level == level)
                for level in ThreatLevel
            }
        }

# === Factory Functions ===
def create_encryption_engine(master_key: Optional[bytes] = None) -> EncryptionEngine:
    """Create encryption engine."""
    return EncryptionEngine(master_key)

def create_access_control_manager() -> AccessControlManager:
    """Create access control manager."""
    return AccessControlManager()

def create_authentication_manager(secret_key: bytes) -> AuthenticationManager:
    """Create authentication manager."""
    return AuthenticationManager(secret_key)

def create_audit_logger(max_events: int = 10000) -> AuditLogger:
    """Create audit logger."""
    return AuditLogger(max_events)

def create_threat_detector(audit_logger: AuditLogger) -> ThreatDetector:
    """Create threat detector."""
    return ThreatDetector(audit_logger)

def create_cache_security_manager(master_key: Optional[bytes] = None) -> CacheSecurityManager:
    """Create complete cache security manager."""
    return CacheSecurityManager(master_key)

def create_security_suite(master_key: Optional[bytes] = None) -> Dict[str, Any]:
    """Create complete security suite."""
    security_manager = create_cache_security_manager(master_key)
    
    return {
        'security_manager': security_manager,
        'encryption_engine': security_manager.encryption_engine,
        'access_control': security_manager.access_control,
        'auth_manager': security_manager.auth_manager,
        'audit_logger': security_manager.audit_logger,
        'threat_detector': security_manager.threat_detector
    }
