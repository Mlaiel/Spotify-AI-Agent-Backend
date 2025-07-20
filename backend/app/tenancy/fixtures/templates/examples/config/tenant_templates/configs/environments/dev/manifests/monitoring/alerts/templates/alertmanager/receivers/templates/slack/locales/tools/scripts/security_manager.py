#!/usr/bin/env python3
"""
Spotify AI Agent - Advanced Security & Compliance Manager
=========================================================

Enterprise-grade security and compliance management system providing:
- Multi-layered security controls and monitoring
- Compliance framework implementation (GDPR, SOX, PCI, etc.)
- Advanced audit logging and forensics
- Real-time threat detection and response
- Security policy enforcement
- Access control and identity management

Author: Fahed Mlaiel (Lead Developer + AI Architect)
Team: Expert Development Team
"""

import hashlib
import hmac
import secrets
import logging
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import redis
import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from pydantic import BaseModel, Field, validator
import asyncpg
from sqlalchemy import create_engine, text
import bcrypt

# Configure logging
logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(str, Enum):
    """Compliance frameworks"""
    GDPR = "gdpr"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    SOC2 = "soc2"


class ThreatLevel(str, Enum):
    """Threat levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditEventType(str, Enum):
    """Audit event types"""
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    COMPLIANCE_CHECK = "compliance_check"


@dataclass
class SecurityEvent:
    """Security event data"""
    event_id: str
    event_type: str
    severity: SecurityLevel
    source_ip: str
    user_id: Optional[str]
    tenant_id: str
    resource: str
    action: str
    timestamp: datetime
    details: Dict[str, Any]
    risk_score: float


@dataclass
class AuditLogEntry:
    """Audit log entry"""
    log_id: str
    event_type: AuditEventType
    user_id: Optional[str]
    tenant_id: str
    resource: str
    action: str
    result: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class CompliancePolicy:
    """Compliance policy definition"""
    policy_id: str
    name: str
    framework: ComplianceFramework
    description: str
    requirements: List[str]
    controls: List[str]
    automated_checks: List[str]
    manual_checks: List[str]
    frequency: str  # cron expression
    enabled: bool


class SecurityConfig(BaseModel):
    """Security manager configuration"""
    encryption_key: str = Field(..., min_length=32)
    jwt_secret: str = Field(..., min_length=32)
    audit_retention_days: int = Field(default=2555, ge=365)  # 7 years
    session_timeout: int = Field(default=3600, ge=300)  # 1 hour
    max_login_attempts: int = Field(default=5, ge=3)
    password_policy: Dict[str, Any] = Field(default_factory=lambda: {
        "min_length": 12,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_symbols": True,
        "prevent_reuse": 12
    })
    ip_whitelist: List[str] = Field(default_factory=list)
    rate_limits: Dict[str, int] = Field(default_factory=lambda: {
        "api_requests_per_minute": 1000,
        "login_attempts_per_hour": 10,
        "password_reset_per_day": 3
    })
    compliance_frameworks: List[ComplianceFramework] = Field(default_factory=lambda: [
        ComplianceFramework.GDPR
    ])


class SecurityManager:
    """
    Advanced security manager with enterprise-grade features
    """
    
    def __init__(self, config: SecurityConfig, redis_url: str, database_url: str):
        self.config = config
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.database_url = database_url
        
        # Initialize encryption
        self.fernet = Fernet(config.encryption_key.encode()[:44])
        
        # Security event storage
        self.security_events: List[SecurityEvent] = []
        self.audit_logs: List[AuditLogEntry] = []
        
        # Threat detection
        self.threat_indicators: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: Set[str] = set()
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("SecurityManager initialized")

    async def authenticate_user(
        self, 
        username: str, 
        password: str, 
        ip_address: str,
        user_agent: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Authenticate user with advanced security checks"""
        
        try:
            # Check IP whitelist if configured
            if self.config.ip_whitelist and ip_address not in self.config.ip_whitelist:
                await self._log_security_event(
                    event_type="authentication_blocked",
                    severity=SecurityLevel.MEDIUM,
                    source_ip=ip_address,
                    details={"reason": "IP not whitelisted", "username": username}
                )
                return None
            
            # Check rate limiting
            if not await self._check_rate_limit("login", ip_address):
                await self._log_security_event(
                    event_type="rate_limit_exceeded",
                    severity=SecurityLevel.HIGH,
                    source_ip=ip_address,
                    details={"type": "login", "username": username}
                )
                return None
            
            # Check failed login attempts
            failed_attempts = await self._get_failed_login_attempts(username, ip_address)
            if failed_attempts >= self.config.max_login_attempts:
                await self._log_security_event(
                    event_type="account_locked",
                    severity=SecurityLevel.HIGH,
                    source_ip=ip_address,
                    details={"username": username, "attempts": failed_attempts}
                )
                return None
            
            # Verify credentials
            user_data = await self._verify_credentials(username, password)
            if not user_data:
                await self._record_failed_login(username, ip_address)
                await self._log_audit_event(
                    event_type=AuditEventType.LOGIN,
                    user_id=username,
                    action="login_failed",
                    result="failure",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    metadata={"reason": "invalid_credentials"}
                )
                return None
            
            # Generate session
            session_data = await self._create_session(user_data, ip_address, user_agent)
            
            # Clear failed attempts
            await self._clear_failed_login_attempts(username, ip_address)
            
            # Log successful authentication
            await self._log_audit_event(
                event_type=AuditEventType.LOGIN,
                user_id=user_data["user_id"],
                action="login_success",
                result="success",
                ip_address=ip_address,
                user_agent=user_agent,
                metadata={"session_id": session_data["session_id"]}
            )
            
            return session_data
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await self._log_security_event(
                event_type="authentication_error",
                severity=SecurityLevel.HIGH,
                source_ip=ip_address,
                details={"error": str(e), "username": username}
            )
            return None

    async def _verify_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials against database"""
        try:
            conn = await asyncpg.connect(self.database_url)
            
            query = """
            SELECT user_id, username, password_hash, tenant_id, roles, is_active,
                   last_password_change, failed_login_attempts
            FROM users 
            WHERE username = $1 AND is_active = true
            """
            
            user_record = await conn.fetchrow(query, username)
            await conn.close()
            
            if not user_record:
                return None
            
            # Verify password
            if bcrypt.checkpw(password.encode('utf-8'), user_record['password_hash'].encode('utf-8')):
                return {
                    "user_id": user_record['user_id'],
                    "username": user_record['username'],
                    "tenant_id": user_record['tenant_id'],
                    "roles": user_record['roles'],
                    "last_password_change": user_record['last_password_change']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Credential verification failed: {e}")
            return None

    async def _create_session(
        self, 
        user_data: Dict[str, Any], 
        ip_address: str,
        user_agent: str
    ) -> Dict[str, Any]:
        """Create authenticated session"""
        
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=self.config.session_timeout)
        
        # Create JWT token
        token_payload = {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "tenant_id": user_data["tenant_id"],
            "roles": user_data["roles"],
            "session_id": session_id,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp())
        }
        
        token = jwt.encode(token_payload, self.config.jwt_secret, algorithm="HS256")
        
        # Store session
        session_data = {
            "session_id": session_id,
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "tenant_id": user_data["tenant_id"],
            "roles": user_data["roles"],
            "ip_address": ip_address,
            "user_agent": user_agent,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "last_activity": now.isoformat()
        }
        
        # Store in Redis
        self.redis_client.setex(
            f"session:{session_id}",
            self.config.session_timeout,
            json.dumps(session_data, default=str)
        )
        
        # Store in memory
        self.active_sessions[session_id] = session_data
        
        return {
            "session_id": session_id,
            "token": token,
            "expires_at": expires_at.isoformat(),
            "user": user_data
        }

    async def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate session token"""
        try:
            # Decode JWT
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=["HS256"])
            session_id = payload["session_id"]
            
            # Check session exists
            session_data = self.redis_client.get(f"session:{session_id}")
            if not session_data:
                return None
            
            session = json.loads(session_data)
            
            # Update last activity
            session["last_activity"] = datetime.utcnow().isoformat()
            self.redis_client.setex(
                f"session:{session_id}",
                self.config.session_timeout,
                json.dumps(session, default=str)
            )
            
            return session
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired JWT token")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return None

    async def check_access_permission(
        self, 
        user_id: str, 
        resource: str, 
        action: str,
        tenant_id: str
    ) -> bool:
        """Check if user has permission for resource/action"""
        try:
            conn = await asyncpg.connect(self.database_url)
            
            query = """
            SELECT COUNT(*) FROM user_permissions up
            JOIN roles r ON up.role_id = r.role_id
            WHERE up.user_id = $1 
            AND up.tenant_id = $4
            AND (r.permissions @> $2 OR r.permissions @> $3)
            """
            
            count = await conn.fetchval(
                query, 
                user_id, 
                json.dumps([f"{resource}:*"]),
                json.dumps([f"{resource}:{action}"]),
                tenant_id
            )
            
            await conn.close()
            
            has_permission = count > 0
            
            # Log access check
            await self._log_audit_event(
                event_type=AuditEventType.ACCESS_GRANTED if has_permission else AuditEventType.ACCESS_DENIED,
                user_id=user_id,
                tenant_id=tenant_id,
                resource=resource,
                action=action,
                result="granted" if has_permission else "denied",
                ip_address="system",
                user_agent="",
                metadata={"permission_check": True}
            )
            
            return has_permission
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False

    async def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()

    async def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    async def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    async def _check_rate_limit(self, action: str, identifier: str) -> bool:
        """Check rate limiting"""
        key = f"rate_limit:{action}:{identifier}"
        current = self.redis_client.get(key)
        
        if action == "login":
            limit = self.config.rate_limits.get("login_attempts_per_hour", 10)
            window = 3600
        elif action == "api":
            limit = self.config.rate_limits.get("api_requests_per_minute", 1000)
            window = 60
        else:
            return True
        
        if current is None:
            self.redis_client.setex(key, window, 1)
            return True
        
        if int(current) >= limit:
            return False
        
        self.redis_client.incr(key)
        return True

    async def _get_failed_login_attempts(self, username: str, ip_address: str) -> int:
        """Get failed login attempts count"""
        key = f"failed_logins:{username}:{ip_address}"
        attempts = self.redis_client.get(key)
        return int(attempts) if attempts else 0

    async def _record_failed_login(self, username: str, ip_address: str):
        """Record failed login attempt"""
        key = f"failed_logins:{username}:{ip_address}"
        self.redis_client.incr(key)
        self.redis_client.expire(key, 3600)  # 1 hour window

    async def _clear_failed_login_attempts(self, username: str, ip_address: str):
        """Clear failed login attempts"""
        key = f"failed_logins:{username}:{ip_address}"
        self.redis_client.delete(key)

    async def _log_security_event(
        self,
        event_type: str,
        severity: SecurityLevel,
        source_ip: str,
        user_id: Optional[str] = None,
        tenant_id: str = "system",
        resource: str = "",
        action: str = "",
        details: Dict[str, Any] = None
    ):
        """Log security event"""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            tenant_id=tenant_id,
            resource=resource,
            action=action,
            timestamp=datetime.utcnow(),
            details=details or {},
            risk_score=self._calculate_risk_score(severity, event_type, details or {})
        )
        
        # Store in memory
        self.security_events.append(event)
        
        # Store in Redis
        key = f"security_event:{event.event_id}"
        self.redis_client.setex(key, 86400 * 30, json.dumps(asdict(event), default=str))  # 30 days
        
        # Store in database
        await self._store_security_event_db(event)
        
        # Check for threats
        await self._analyze_threat(event)

    async def _log_audit_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        action: str,
        result: str,
        ip_address: str,
        user_agent: str,
        tenant_id: str = "system",
        resource: str = "",
        metadata: Dict[str, Any] = None
    ):
        """Log audit event"""
        entry = AuditLogEntry(
            log_id=str(uuid.uuid4()),
            event_type=event_type,
            user_id=user_id,
            tenant_id=tenant_id,
            resource=resource,
            action=action,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Store in memory
        self.audit_logs.append(entry)
        
        # Store in database
        await self._store_audit_log_db(entry)

    async def _store_security_event_db(self, event: SecurityEvent):
        """Store security event in database"""
        try:
            conn = await asyncpg.connect(self.database_url)
            
            query = """
            INSERT INTO security_events (
                event_id, event_type, severity, source_ip, user_id, 
                tenant_id, resource, action, timestamp, details, risk_score
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """
            
            await conn.execute(
                query,
                event.event_id,
                event.event_type,
                event.severity.value,
                event.source_ip,
                event.user_id,
                event.tenant_id,
                event.resource,
                event.action,
                event.timestamp,
                json.dumps(event.details),
                event.risk_score
            )
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store security event: {e}")

    async def _store_audit_log_db(self, entry: AuditLogEntry):
        """Store audit log in database"""
        try:
            conn = await asyncpg.connect(self.database_url)
            
            query = """
            INSERT INTO audit_logs (
                log_id, event_type, user_id, tenant_id, resource, 
                action, result, ip_address, user_agent, timestamp, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """
            
            await conn.execute(
                query,
                entry.log_id,
                entry.event_type.value,
                entry.user_id,
                entry.tenant_id,
                entry.resource,
                entry.action,
                entry.result,
                entry.ip_address,
                entry.user_agent,
                entry.timestamp,
                json.dumps(entry.metadata)
            )
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store audit log: {e}")

    def _calculate_risk_score(
        self, 
        severity: SecurityLevel, 
        event_type: str, 
        details: Dict[str, Any]
    ) -> float:
        """Calculate risk score for security event"""
        base_scores = {
            SecurityLevel.LOW: 2.0,
            SecurityLevel.MEDIUM: 5.0,
            SecurityLevel.HIGH: 8.0,
            SecurityLevel.CRITICAL: 10.0
        }
        
        score = base_scores.get(severity, 5.0)
        
        # Adjust based on event type
        if event_type in ["authentication_blocked", "rate_limit_exceeded"]:
            score *= 1.2
        elif event_type in ["account_locked", "authentication_error"]:
            score *= 1.5
        
        # Adjust based on details
        if details.get("attempts", 0) > 10:
            score *= 1.3
        
        return min(score, 10.0)

    async def _analyze_threat(self, event: SecurityEvent):
        """Analyze security event for threats"""
        # Simple threat detection logic
        if event.risk_score >= 8.0:
            await self._handle_high_risk_event(event)
        
        # Track suspicious IPs
        if event.source_ip not in self.threat_indicators:
            self.threat_indicators[event.source_ip] = {
                "events": [],
                "risk_score": 0.0,
                "first_seen": event.timestamp,
                "last_seen": event.timestamp
            }
        
        indicators = self.threat_indicators[event.source_ip]
        indicators["events"].append(event.event_id)
        indicators["risk_score"] += event.risk_score
        indicators["last_seen"] = event.timestamp
        
        # Block IP if risk is too high
        if indicators["risk_score"] >= 50.0:
            await self._block_ip(event.source_ip)

    async def _handle_high_risk_event(self, event: SecurityEvent):
        """Handle high-risk security events"""
        logger.warning(f"High-risk security event detected: {event.event_type}")
        
        # Could trigger alerts, notifications, etc.
        # For now, just log it
        
    async def _block_ip(self, ip_address: str):
        """Block suspicious IP address"""
        self.blocked_ips.add(ip_address)
        
        # Store in Redis
        self.redis_client.setex(f"blocked_ip:{ip_address}", 86400, "blocked")  # 24 hours
        
        logger.warning(f"IP address blocked: {ip_address}")


class ComplianceChecker:
    """
    Compliance framework checker and auditor
    """
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.compliance_policies: Dict[str, CompliancePolicy] = {}
        
    async def load_compliance_policies(self, policies_path: str):
        """Load compliance policies from file"""
        try:
            async with aiofiles.open(policies_path, 'r') as f:
                content = await f.read()
                policies_data = json.loads(content)
                
                for policy_data in policies_data:
                    policy = CompliancePolicy(**policy_data)
                    self.compliance_policies[policy.policy_id] = policy
                    
            logger.info(f"Loaded {len(self.compliance_policies)} compliance policies")
            
        except Exception as e:
            logger.error(f"Failed to load compliance policies: {e}")

    async def run_compliance_check(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Run compliance check for framework"""
        results = {
            "framework": framework.value,
            "timestamp": datetime.utcnow().isoformat(),
            "policies_checked": 0,
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for policy in self.compliance_policies.values():
            if policy.framework == framework and policy.enabled:
                check_result = await self._check_policy(policy)
                results["policies_checked"] += 1
                
                if check_result["passed"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                results["details"].append(check_result)
        
        return results

    async def _check_policy(self, policy: CompliancePolicy) -> Dict[str, Any]:
        """Check individual compliance policy"""
        # Implement specific policy checks
        # This is a simplified example
        
        result = {
            "policy_id": policy.policy_id,
            "policy_name": policy.name,
            "passed": True,
            "issues": [],
            "recommendations": []
        }
        
        # Example checks
        if policy.framework == ComplianceFramework.GDPR:
            # Check data retention policies
            if self.security_manager.config.audit_retention_days < 365:
                result["passed"] = False
                result["issues"].append("Audit logs must be retained for at least 1 year")
        
        return result


class AuditLogger:
    """
    Advanced audit logging system
    """
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
    
    async def generate_audit_report(
        self, 
        start_date: datetime, 
        end_date: datetime,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        
        try:
            conn = await asyncpg.connect(self.security_manager.database_url)
            
            # Base query
            where_clause = "WHERE timestamp BETWEEN $1 AND $2"
            params = [start_date, end_date]
            
            if tenant_id:
                where_clause += " AND tenant_id = $3"
                params.append(tenant_id)
            
            # Get audit statistics
            stats_query = f"""
            SELECT 
                event_type,
                result,
                COUNT(*) as count
            FROM audit_logs 
            {where_clause}
            GROUP BY event_type, result
            ORDER BY count DESC
            """
            
            stats = await conn.fetch(stats_query, *params)
            
            # Get security events
            security_query = f"""
            SELECT 
                event_type,
                severity,
                COUNT(*) as count,
                AVG(risk_score) as avg_risk_score
            FROM security_events 
            {where_clause}
            GROUP BY event_type, severity
            ORDER BY count DESC
            """
            
            security_stats = await conn.fetch(security_query, *params)
            
            await conn.close()
            
            report = {
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "tenant_id": tenant_id,
                "audit_statistics": [dict(row) for row in stats],
                "security_statistics": [dict(row) for row in security_stats],
                "summary": {
                    "total_audit_events": sum(row["count"] for row in stats),
                    "total_security_events": sum(row["count"] for row in security_stats),
                    "high_risk_events": sum(
                        row["count"] for row in security_stats 
                        if row["avg_risk_score"] and row["avg_risk_score"] >= 8.0
                    )
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate audit report: {e}")
            return {}

    async def export_audit_logs(
        self, 
        start_date: datetime, 
        end_date: datetime,
        format: str = "json"
    ) -> str:
        """Export audit logs in specified format"""
        
        try:
            conn = await asyncpg.connect(self.security_manager.database_url)
            
            query = """
            SELECT * FROM audit_logs 
            WHERE timestamp BETWEEN $1 AND $2
            ORDER BY timestamp DESC
            """
            
            logs = await conn.fetch(query, start_date, end_date)
            await conn.close()
            
            if format == "json":
                return json.dumps([dict(log) for log in logs], default=str, indent=2)
            elif format == "csv":
                # Convert to CSV format
                import csv
                import io
                
                output = io.StringIO()
                if logs:
                    writer = csv.DictWriter(output, fieldnames=logs[0].keys())
                    writer.writeheader()
                    for log in logs:
                        writer.writerow(dict(log))
                
                return output.getvalue()
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to export audit logs: {e}")
            return ""
