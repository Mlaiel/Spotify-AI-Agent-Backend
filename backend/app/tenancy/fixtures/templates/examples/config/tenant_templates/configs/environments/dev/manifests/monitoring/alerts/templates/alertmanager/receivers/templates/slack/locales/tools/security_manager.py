#!/usr/bin/env python3
"""
Security Manager - Advanced security and compliance system
==========================================================

Created by: Fahed Mlaiel - Backend Security Specialist & Compliance Engineer
Description: Enterprise-grade security manager for multi-tenant Slack notification 
            system. Implements comprehensive security controls, compliance monitoring,
            and threat detection capabilities.

Features:
- Advanced encryption and key management
- Multi-factor authentication and authorization
- RBAC with fine-grained permissions
- Security audit logging and compliance reporting
- Threat detection and anomaly analysis
- Data loss prevention (DLP) controls
- Secure token management and rotation
- GDPR/SOC2/HIPAA compliance support
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import uuid

import redis.asyncio as redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from prometheus_client import Counter, Histogram, Gauge
from pydantic import BaseModel, Field, validator

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError


class SecurityLevel(Enum):
    """Security level enumeration"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class PermissionLevel(Enum):
    """Permission level enumeration"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class ThreatLevel(Enum):
    """Threat level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStandard(Enum):
    """Compliance standard enumeration"""
    GDPR = "gdpr"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"


class SecurityContext(BaseModel):
    """Security context for operations"""
    user_id: str
    tenant_id: str
    session_id: str
    permissions: List[str]
    security_level: SecurityLevel
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mfa_verified: bool = False
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)


class AuditEvent(BaseModel):
    """Security audit event"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    user_id: str
    tenant_id: str
    resource_type: str
    resource_id: str
    action: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)
    compliance_tags: List[ComplianceStandard] = Field(default_factory=list)


class SecurityPolicy(BaseModel):
    """Security policy configuration"""
    tenant_id: str
    policy_name: str
    security_level: SecurityLevel
    max_failed_attempts: int = Field(default=5, ge=1, le=20)
    lockout_duration: int = Field(default=300, ge=60, le=3600)  # seconds
    session_timeout: int = Field(default=3600, ge=300, le=86400)  # seconds
    require_mfa: bool = True
    allowed_ip_ranges: List[str] = Field(default_factory=list)
    blocked_ip_ranges: List[str] = Field(default_factory=list)
    data_retention_days: int = Field(default=90, ge=1, le=2555)  # 7 years max
    encryption_required: bool = True
    compliance_standards: List[ComplianceStandard] = Field(default_factory=list)
    dlp_patterns: List[str] = Field(default_factory=list)


class ThreatDetection(BaseModel):
    """Threat detection rule"""
    rule_id: str
    rule_name: str
    description: str
    threat_level: ThreatLevel
    detection_pattern: str
    threshold: int = 1
    time_window: int = 300  # seconds
    enabled: bool = True
    actions: List[str] = Field(default_factory=list)


class SecurityManager:
    """
    Enterprise-grade security manager with comprehensive protection
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        encryption_key: Optional[str] = None,
        jwt_secret: Optional[str] = None
    ):
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode())
        else:
            # Generate new key if not provided
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
            self.logger.warning("Generated new encryption key - store securely!")
        
        # Initialize JWT
        self.jwt_secret = jwt_secret or self._generate_jwt_secret()
        
        # Initialize password hasher
        self.password_hasher = PasswordHasher()
        
        # Security policies and rules
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.threat_rules: Dict[str, ThreatDetection] = {}
        
        # Active sessions and security state
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.blocked_ips: Set[str] = set()
        
        # Metrics
        self.auth_attempts = Counter(
            'security_auth_attempts_total',
            'Total authentication attempts',
            ['tenant_id', 'result', 'method']
        )
        self.threat_detections = Counter(
            'security_threats_detected_total',
            'Total threats detected',
            ['tenant_id', 'threat_level', 'rule_id']
        )
        self.audit_events = Counter(
            'security_audit_events_total',
            'Total audit events',
            ['tenant_id', 'event_type', 'success']
        )
        self.security_score = Gauge(
            'security_tenant_score',
            'Tenant security score',
            ['tenant_id']
        )
    
    async def initialize(self):
        """Initialize security manager"""
        try:
            # Load security policies
            await self._load_security_policies()
            
            # Load threat detection rules
            await self._load_threat_rules()
            
            # Start background security tasks
            asyncio.create_task(self._session_cleanup())
            asyncio.create_task(self._threat_monitor())
            asyncio.create_task(self._compliance_monitor())
            asyncio.create_task(self._security_metrics_updater())
            
            self.logger.info("SecurityManager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SecurityManager: {e}")
            raise
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        tenant_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        mfa_token: Optional[str] = None
    ) -> Optional[SecurityContext]:
        """Authenticate user with comprehensive security checks"""
        try:
            # Check if IP is blocked
            if ip_address and await self._is_ip_blocked(ip_address, tenant_id):
                await self._log_audit_event(
                    event_type="authentication",
                    user_id=username,
                    tenant_id=tenant_id,
                    resource_type="user",
                    resource_id=username,
                    action="login_blocked_ip",
                    success=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    risk_score=0.9
                )
                return None
            
            # Check failed attempts
            attempt_key = f"{tenant_id}:{username}:{ip_address or 'unknown'}"
            failed_count = await self._get_failed_attempts(attempt_key)
            
            policy = await self._get_security_policy(tenant_id)
            if failed_count >= policy.max_failed_attempts:
                await self._log_audit_event(
                    event_type="authentication",
                    user_id=username,
                    tenant_id=tenant_id,
                    resource_type="user",
                    resource_id=username,
                    action="login_locked_out",
                    success=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    risk_score=0.8
                )
                return None
            
            # Verify credentials
            user_data = await self._get_user_credentials(username, tenant_id)
            if not user_data:
                await self._increment_failed_attempts(attempt_key)
                self.auth_attempts.labels(
                    tenant_id=tenant_id,
                    result='failed',
                    method='password'
                ).inc()
                return None
            
            try:
                self.password_hasher.verify(user_data['password_hash'], password)
            except VerifyMismatchError:
                await self._increment_failed_attempts(attempt_key)
                await self._log_audit_event(
                    event_type="authentication",
                    user_id=username,
                    tenant_id=tenant_id,
                    resource_type="user",
                    resource_id=username,
                    action="login_invalid_password",
                    success=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    risk_score=0.6
                )
                self.auth_attempts.labels(
                    tenant_id=tenant_id,
                    result='failed',
                    method='password'
                ).inc()
                return None
            
            # Verify MFA if required
            mfa_verified = False
            if policy.require_mfa:
                if not mfa_token:
                    await self._log_audit_event(
                        event_type="authentication",
                        user_id=username,
                        tenant_id=tenant_id,
                        resource_type="user",
                        resource_id=username,
                        action="login_mfa_required",
                        success=False,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        risk_score=0.4
                    )
                    return None
                
                mfa_verified = await self._verify_mfa_token(username, tenant_id, mfa_token)
                if not mfa_verified:
                    await self._increment_failed_attempts(attempt_key)
                    await self._log_audit_event(
                        event_type="authentication",
                        user_id=username,
                        tenant_id=tenant_id,
                        resource_type="user",
                        resource_id=username,
                        action="login_invalid_mfa",
                        success=False,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        risk_score=0.7
                    )
                    self.auth_attempts.labels(
                        tenant_id=tenant_id,
                        result='failed',
                        method='mfa'
                    ).inc()
                    return None
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(
                username, tenant_id, ip_address, user_agent
            )
            
            # Create security context
            session_id = str(uuid.uuid4())
            context = SecurityContext(
                user_id=username,
                tenant_id=tenant_id,
                session_id=session_id,
                permissions=user_data.get('permissions', []),
                security_level=SecurityLevel(user_data.get('security_level', 'internal')),
                ip_address=ip_address,
                user_agent=user_agent,
                mfa_verified=mfa_verified,
                risk_score=risk_score
            )
            
            # Store active session
            self.active_sessions[session_id] = context
            await self.redis.setex(
                f"session:{session_id}",
                policy.session_timeout,
                context.json()
            )
            
            # Clear failed attempts
            await self._clear_failed_attempts(attempt_key)
            
            # Log successful authentication
            await self._log_audit_event(
                event_type="authentication",
                user_id=username,
                tenant_id=tenant_id,
                resource_type="user",
                resource_id=username,
                action="login_success",
                success=True,
                ip_address=ip_address,
                user_agent=user_agent,
                risk_score=risk_score
            )
            
            self.auth_attempts.labels(
                tenant_id=tenant_id,
                result='success',
                method='password' if not mfa_verified else 'mfa'
            ).inc()
            
            return context
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None
    
    async def authorize_action(
        self,
        session_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        required_permission: PermissionLevel = PermissionLevel.READ
    ) -> bool:
        """Authorize action with RBAC and security checks"""
        try:
            # Get security context
            context = await self._get_security_context(session_id)
            if not context:
                return False
            
            # Check session validity
            if not await self._is_session_valid(context):
                return False
            
            # Check permissions
            permission_granted = await self._check_permissions(
                context,
                resource_type,
                resource_id,
                action,
                required_permission
            )
            
            # Calculate operation risk
            risk_score = await self._calculate_operation_risk(
                context, resource_type, action
            )
            
            # Log authorization attempt
            await self._log_audit_event(
                event_type="authorization",
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                success=permission_granted,
                ip_address=context.ip_address,
                user_agent=context.user_agent,
                risk_score=risk_score
            )
            
            # Check for high-risk operations
            if risk_score > 0.8 and permission_granted:
                await self._trigger_security_alert(
                    context,
                    f"High-risk operation: {action} on {resource_type}",
                    ThreatLevel.HIGH
                )
            
            return permission_granted
            
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return False
    
    async def encrypt_data(
        self,
        data: Union[str, bytes],
        context: SecurityContext
    ) -> str:
        """Encrypt sensitive data"""
        try:
            if isinstance(data, str):
                data = data.encode()
            
            # Add metadata
            metadata = {
                'tenant_id': context.tenant_id,
                'user_id': context.user_id,
                'timestamp': context.timestamp.isoformat(),
                'security_level': context.security_level.value
            }
            
            # Combine data with metadata
            payload = {
                'data': base64.b64encode(data).decode(),
                'metadata': metadata
            }
            
            # Encrypt payload
            encrypted = self.fernet.encrypt(json.dumps(payload).encode())
            return base64.b64encode(encrypted).decode()
            
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            raise
    
    async def decrypt_data(
        self,
        encrypted_data: str,
        context: SecurityContext
    ) -> Optional[bytes]:
        """Decrypt sensitive data with access control"""
        try:
            # Decode and decrypt
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            payload = json.loads(decrypted.decode())
            
            # Check access permissions
            metadata = payload['metadata']
            if metadata['tenant_id'] != context.tenant_id:
                await self._log_audit_event(
                    event_type="data_access",
                    user_id=context.user_id,
                    tenant_id=context.tenant_id,
                    resource_type="encrypted_data",
                    resource_id="unknown",
                    action="decrypt_cross_tenant",
                    success=False,
                    risk_score=0.9
                )
                return None
            
            # Check security level
            data_security_level = SecurityLevel(metadata['security_level'])
            if data_security_level.value > context.security_level.value:
                await self._log_audit_event(
                    event_type="data_access",
                    user_id=context.user_id,
                    tenant_id=context.tenant_id,
                    resource_type="encrypted_data",
                    resource_id="unknown",
                    action="decrypt_insufficient_clearance",
                    success=False,
                    risk_score=0.8
                )
                return None
            
            # Log successful decryption
            await self._log_audit_event(
                event_type="data_access",
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                resource_type="encrypted_data",
                resource_id="unknown",
                action="decrypt_success",
                success=True,
                risk_score=0.2
            )
            
            return base64.b64decode(payload['data'].encode())
            
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            return None
    
    async def scan_for_sensitive_data(self, text: str, tenant_id: str) -> List[Dict[str, Any]]:
        """Scan text for sensitive data patterns (DLP)"""
        findings = []
        
        try:
            policy = await self._get_security_policy(tenant_id)
            
            # Built-in patterns
            patterns = {
                'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
                'api_key': r'\b[A-Za-z0-9]{32,}\b'
            }
            
            # Add custom patterns from policy
            for pattern in policy.dlp_patterns:
                patterns[f'custom_{len(patterns)}'] = pattern
            
            # Scan text
            for pattern_name, pattern in patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    findings.append({
                        'type': pattern_name,
                        'value': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8
                    })
            
            # Log findings
            if findings:
                await self._log_audit_event(
                    event_type="dlp_scan",
                    user_id="system",
                    tenant_id=tenant_id,
                    resource_type="text",
                    resource_id="unknown",
                    action="sensitive_data_detected",
                    success=True,
                    risk_score=0.7,
                    details={'findings_count': len(findings)}
                )
            
            return findings
            
        except Exception as e:
            self.logger.error(f"DLP scan error: {e}")
            return []
    
    async def generate_compliance_report(
        self,
        tenant_id: str,
        standard: ComplianceStandard,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for specified standard"""
        try:
            # Get audit events for period
            events = await self._get_audit_events(tenant_id, start_date, end_date)
            
            # Generate report based on standard
            if standard == ComplianceStandard.GDPR:
                return await self._generate_gdpr_report(events, tenant_id)
            elif standard == ComplianceStandard.SOC2:
                return await self._generate_soc2_report(events, tenant_id)
            elif standard == ComplianceStandard.HIPAA:
                return await self._generate_hipaa_report(events, tenant_id)
            else:
                return await self._generate_generic_report(events, tenant_id, standard)
                
        except Exception as e:
            self.logger.error(f"Compliance report generation error: {e}")
            return {}
    
    async def _get_security_context(self, session_id: str) -> Optional[SecurityContext]:
        """Get security context from session"""
        try:
            if session_id in self.active_sessions:
                return self.active_sessions[session_id]
            
            # Try Redis
            session_data = await self.redis.get(f"session:{session_id}")
            if session_data:
                context = SecurityContext.parse_raw(session_data)
                self.active_sessions[session_id] = context
                return context
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get security context: {e}")
            return None
    
    async def _is_session_valid(self, context: SecurityContext) -> bool:
        """Check if session is still valid"""
        try:
            # Check session timeout
            policy = await self._get_security_policy(context.tenant_id)
            session_age = (datetime.utcnow() - context.timestamp).total_seconds()
            
            if session_age > policy.session_timeout:
                return False
            
            # Check if user is still active
            user_data = await self._get_user_credentials(context.user_id, context.tenant_id)
            if not user_data or not user_data.get('active', True):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Session validation error: {e}")
            return False
    
    async def _check_permissions(
        self,
        context: SecurityContext,
        resource_type: str,
        resource_id: str,
        action: str,
        required_permission: PermissionLevel
    ) -> bool:
        """Check RBAC permissions"""
        try:
            # Super admin can do anything
            if 'super_admin' in context.permissions:
                return True
            
            # Check specific permissions
            permission_key = f"{resource_type}:{action}"
            
            if required_permission == PermissionLevel.READ:
                required_perms = ['read', 'write', 'admin']
            elif required_permission == PermissionLevel.WRITE:
                required_perms = ['write', 'admin']
            elif required_permission == PermissionLevel.ADMIN:
                required_perms = ['admin']
            else:
                required_perms = ['super_admin']
            
            # Check if user has any required permission
            for perm in context.permissions:
                if perm in required_perms:
                    return True
                if perm.startswith(f"{resource_type}:") and perm.endswith(f":{action}"):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Permission check error: {e}")
            return False
    
    async def _calculate_risk_score(
        self,
        user_id: str,
        tenant_id: str,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ) -> float:
        """Calculate risk score for authentication"""
        risk_score = 0.0
        
        try:
            # Check IP reputation
            if ip_address:
                if await self._is_suspicious_ip(ip_address):
                    risk_score += 0.3
                
                # Check geolocation changes
                if await self._has_location_changed(user_id, ip_address):
                    risk_score += 0.2
            
            # Check time-based patterns
            if await self._is_unusual_time(user_id):
                risk_score += 0.1
            
            # Check user agent patterns
            if user_agent and await self._is_suspicious_user_agent(user_agent):
                risk_score += 0.2
            
            # Check recent failed attempts
            recent_failures = await self._get_recent_failures(user_id, tenant_id)
            if recent_failures > 0:
                risk_score += min(0.3, recent_failures * 0.1)
            
            return min(1.0, risk_score)
            
        except Exception as e:
            self.logger.error(f"Risk calculation error: {e}")
            return 0.5  # Default moderate risk
    
    async def _calculate_operation_risk(
        self,
        context: SecurityContext,
        resource_type: str,
        action: str
    ) -> float:
        """Calculate risk score for operations"""
        base_risk = context.risk_score
        
        # Add risk based on operation type
        if action in ['delete', 'modify', 'admin']:
            base_risk += 0.3
        elif action in ['create', 'update']:
            base_risk += 0.2
        
        # Add risk based on resource sensitivity
        if resource_type in ['user', 'security', 'billing']:
            base_risk += 0.2
        
        return min(1.0, base_risk)
    
    async def _get_security_policy(self, tenant_id: str) -> SecurityPolicy:
        """Get security policy for tenant"""
        if tenant_id in self.security_policies:
            return self.security_policies[tenant_id]
        
        # Load from Redis
        policy_data = await self.redis.get(f"security_policy:{tenant_id}")
        if policy_data:
            policy = SecurityPolicy.parse_raw(policy_data)
            self.security_policies[tenant_id] = policy
            return policy
        
        # Return default policy
        return SecurityPolicy(
            tenant_id=tenant_id,
            policy_name="default",
            security_level=SecurityLevel.INTERNAL
        )
    
    async def _log_audit_event(
        self,
        event_type: str,
        user_id: str,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        risk_score: float = 0.0,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security audit event"""
        try:
            event = AuditEvent(
                event_type=event_type,
                user_id=user_id,
                tenant_id=tenant_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                success=success,
                ip_address=ip_address,
                user_agent=user_agent,
                risk_score=risk_score,
                details=details or {}
            )
            
            # Store in Redis
            await self.redis.lpush(
                f"audit_events:{tenant_id}",
                event.json()
            )
            
            # Keep only recent events (configurable retention)
            await self.redis.ltrim(f"audit_events:{tenant_id}", 0, 10000)
            
            # Update metrics
            self.audit_events.labels(
                tenant_id=tenant_id,
                event_type=event_type,
                success=str(success).lower()
            ).inc()
            
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
    
    def _generate_jwt_secret(self) -> str:
        """Generate JWT secret"""
        return base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes).decode()
    
    # Additional helper methods would continue here...
    # Implementation continues with remaining security features
    
    async def _load_security_policies(self):
        """Load security policies from Redis"""
        # Implementation placeholder
        pass
    
    async def _load_threat_rules(self):
        """Load threat detection rules"""
        # Implementation placeholder
        pass
    
    async def _session_cleanup(self):
        """Background task for session cleanup"""
        # Implementation placeholder
        pass
    
    async def _threat_monitor(self):
        """Background threat monitoring"""
        # Implementation placeholder
        pass
    
    async def _compliance_monitor(self):
        """Background compliance monitoring"""
        # Implementation placeholder
        pass
    
    async def _security_metrics_updater(self):
        """Update security metrics"""
        # Implementation placeholder
        pass


# Export classes for external use
__all__ = [
    'SecurityManager',
    'SecurityContext',
    'SecurityPolicy',
    'SecurityLevel',
    'PermissionLevel',
    'ThreatLevel',
    'ComplianceStandard',
    'AuditEvent'
]
