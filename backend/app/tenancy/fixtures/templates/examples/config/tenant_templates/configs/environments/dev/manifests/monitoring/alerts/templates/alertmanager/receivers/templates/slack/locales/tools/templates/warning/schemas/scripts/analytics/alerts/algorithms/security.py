"""
Advanced Security and Compliance Framework for Spotify AI Agent.

This module implements enterprise-grade security measures for the
monitoring and alerting algorithms, ensuring data protection and
regulatory compliance.

Features:
- End-to-end encryption for sensitive data
- GDPR/CCPA compliance mechanisms
- Advanced threat detection and prevention
- Audit logging and compliance reporting
- Role-based access control (RBAC)
- Data anonymization and pseudonymization
"""

import asyncio
import hashlib
import hmac
import secrets
import base64
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis.asyncio as aioredis
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class AccessPermission(Enum):
    """Access permissions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    AUDIT = "audit"

class ComplianceRegulation(Enum):
    """Compliance regulations."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"

@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    roles: List[str]
    permissions: List[AccessPermission]
    security_level: SecurityLevel
    session_id: str
    ip_address: str
    timestamp: datetime
    mfa_verified: bool = False
    
@dataclass
class AuditEvent:
    """Audit event for compliance logging."""
    event_id: str
    user_id: str
    action: str
    resource: str
    timestamp: datetime
    success: bool
    details: Dict[str, Any]
    security_context: SecurityContext
    compliance_tags: List[ComplianceRegulation] = field(default_factory=list)

@dataclass
class EncryptionKey:
    """Encryption key metadata."""
    key_id: str
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime]
    security_level: SecurityLevel
    usage_count: int = 0

class EncryptionManager:
    """Advanced encryption and key management."""
    
    def __init__(self):
        self.keys: Dict[str, EncryptionKey] = {}
        self.active_keys: Dict[SecurityLevel, str] = {}
        self._master_key = self._generate_master_key()
        
        # Initialize keys for each security level
        for level in SecurityLevel:
            key_id = self._generate_key_for_level(level)
            self.active_keys[level] = key_id
    
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key."""
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        return kdf.derive(password)
    
    def _generate_key_for_level(self, level: SecurityLevel) -> str:
        """Generate encryption key for security level."""
        key_id = f"key_{level.value}_{int(time.time())}"
        
        # Generate Fernet key
        fernet_key = Fernet.generate_key()
        
        # Create key metadata
        key_metadata = EncryptionKey(
            key_id=key_id,
            algorithm="Fernet",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=90),
            security_level=level
        )
        
        self.keys[key_id] = key_metadata
        
        # Store encrypted key (in production, use HSM or key vault)
        self._store_encrypted_key(key_id, fernet_key)
        
        return key_id
    
    def _store_encrypted_key(self, key_id: str, key_data: bytes):
        """Store encrypted key securely."""
        # In production: use AWS KMS, Azure Key Vault, or HashiCorp Vault
        # For demo: encrypt with master key
        master_fernet = Fernet(base64.urlsafe_b64encode(self._master_key))
        encrypted_key = master_fernet.encrypt(key_data)
        
        # Store in secure location (mock storage)
        self._key_storage = getattr(self, '_key_storage', {})
        self._key_storage[key_id] = encrypted_key
    
    def _retrieve_key(self, key_id: str) -> bytes:
        """Retrieve and decrypt key."""
        if key_id not in self._key_storage:
            raise ValueError(f"Key {key_id} not found")
        
        master_fernet = Fernet(base64.urlsafe_b64encode(self._master_key))
        encrypted_key = self._key_storage[key_id]
        
        return master_fernet.decrypt(encrypted_key)
    
    def encrypt_data(self, data: Union[str, bytes], 
                    security_level: SecurityLevel) -> Tuple[str, str]:
        """Encrypt data with appropriate key."""
        
        key_id = self.active_keys[security_level]
        key_data = self._retrieve_key(key_id)
        
        fernet = Fernet(key_data)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted_data = fernet.encrypt(data)
        
        # Update key usage
        self.keys[key_id].usage_count += 1
        
        return base64.urlsafe_b64encode(encrypted_data).decode(), key_id
    
    def decrypt_data(self, encrypted_data: str, key_id: str) -> bytes:
        """Decrypt data with specified key."""
        
        if key_id not in self.keys:
            raise ValueError(f"Key {key_id} not found")
        
        key_data = self._retrieve_key(key_id)
        fernet = Fernet(key_data)
        
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        
        return fernet.decrypt(encrypted_bytes)
    
    def rotate_keys(self):
        """Rotate encryption keys."""
        
        for level in SecurityLevel:
            # Generate new key
            new_key_id = self._generate_key_for_level(level)
            
            # Update active key
            old_key_id = self.active_keys[level]
            self.active_keys[level] = new_key_id
            
            logger.info(f"Rotated key for level {level.value}: {old_key_id} -> {new_key_id}")

class DataAnonymizer:
    """Data anonymization and pseudonymization."""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.anonymization_salt = secrets.token_bytes(32)
    
    def anonymize_user_id(self, user_id: str) -> str:
        """Anonymize user ID for analytics."""
        
        # Create deterministic hash for consistency
        hash_input = f"{user_id}{self.anonymization_salt.hex()}"
        hashed = hashlib.sha256(hash_input.encode()).hexdigest()
        
        return f"anon_{hashed[:16]}"
    
    def pseudonymize_data(self, data: Dict[str, Any], 
                         security_context: SecurityContext) -> Dict[str, Any]:
        """Pseudonymize sensitive data fields."""
        
        pseudonymized = data.copy()
        
        # Define sensitive fields
        sensitive_fields = {
            'user_id', 'email', 'ip_address', 'session_id',
            'payment_info', 'personal_data'
        }
        
        for field in sensitive_fields:
            if field in pseudonymized:
                original_value = str(pseudonymized[field])
                
                # Create pseudonym
                pseudonym_key = f"pseudonym_{field}_{original_value}"
                pseudonym_hash = hashlib.sha256(
                    f"{pseudonym_key}{self.anonymization_salt.hex()}".encode()
                ).hexdigest()
                
                pseudonymized[field] = f"pseudo_{pseudonym_hash[:12]}"
        
        return pseudonymized
    
    def apply_gdpr_anonymization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GDPR-compliant anonymization."""
        
        anonymized = {}
        
        # Remove direct identifiers
        gdpr_identifiers = {
            'name', 'email', 'phone', 'address', 'passport_number',
            'national_id', 'ip_address', 'device_id'
        }
        
        for key, value in data.items():
            if key in gdpr_identifiers:
                continue  # Remove completely
            elif key == 'user_id':
                anonymized['user_hash'] = self.anonymize_user_id(str(value))
            elif key in ['timestamp', 'date']:
                # Truncate to reduce granularity
                if isinstance(value, datetime):
                    anonymized[key] = value.replace(
                        minute=0, second=0, microsecond=0
                    )
                else:
                    anonymized[key] = value
            else:
                anonymized[key] = value
        
        return anonymized

class AccessControlManager:
    """Role-based access control system."""
    
    def __init__(self):
        self.roles: Dict[str, Dict[str, Any]] = self._initialize_roles()
        self.user_roles: Dict[str, List[str]] = {}
        self.resource_permissions: Dict[str, SecurityLevel] = {}
    
    def _initialize_roles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default roles and permissions."""
        
        return {
            'viewer': {
                'permissions': [AccessPermission.READ],
                'security_levels': [SecurityLevel.PUBLIC, SecurityLevel.INTERNAL],
                'description': 'Read-only access to public and internal data'
            },
            'analyst': {
                'permissions': [AccessPermission.READ, AccessPermission.EXECUTE],
                'security_levels': [
                    SecurityLevel.PUBLIC, SecurityLevel.INTERNAL, 
                    SecurityLevel.CONFIDENTIAL
                ],
                'description': 'Analytics access with execution rights'
            },
            'engineer': {
                'permissions': [
                    AccessPermission.READ, AccessPermission.WRITE, 
                    AccessPermission.EXECUTE
                ],
                'security_levels': [
                    SecurityLevel.PUBLIC, SecurityLevel.INTERNAL, 
                    SecurityLevel.CONFIDENTIAL
                ],
                'description': 'Engineering access for system operations'
            },
            'admin': {
                'permissions': [
                    AccessPermission.READ, AccessPermission.WRITE, 
                    AccessPermission.EXECUTE, AccessPermission.ADMIN
                ],
                'security_levels': [
                    SecurityLevel.PUBLIC, SecurityLevel.INTERNAL, 
                    SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED
                ],
                'description': 'Administrative access to most resources'
            },
            'security_admin': {
                'permissions': [
                    AccessPermission.READ, AccessPermission.WRITE, 
                    AccessPermission.EXECUTE, AccessPermission.ADMIN,
                    AccessPermission.AUDIT
                ],
                'security_levels': list(SecurityLevel),
                'description': 'Full security and audit access'
            }
        }
    
    def assign_role(self, user_id: str, role: str):
        """Assign role to user."""
        
        if role not in self.roles:
            raise ValueError(f"Role {role} not defined")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        
        if role not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role)
            logger.info(f"Assigned role {role} to user {user_id}")
    
    def check_permission(self, security_context: SecurityContext,
                        resource: str, permission: AccessPermission) -> bool:
        """Check if user has permission for resource."""
        
        # Get resource security level
        resource_level = self.resource_permissions.get(
            resource, SecurityLevel.INTERNAL
        )
        
        # Check user permissions
        user_permissions = set()
        user_security_levels = set()
        
        for role in security_context.roles:
            if role in self.roles:
                role_data = self.roles[role]
                user_permissions.update(role_data['permissions'])
                user_security_levels.update(role_data['security_levels'])
        
        # Check permission
        has_permission = permission in user_permissions
        has_security_level = resource_level in user_security_levels
        
        return has_permission and has_security_level
    
    def get_accessible_resources(self, security_context: SecurityContext) -> List[str]:
        """Get list of resources accessible to user."""
        
        accessible_levels = set()
        for role in security_context.roles:
            if role in self.roles:
                accessible_levels.update(self.roles[role]['security_levels'])
        
        accessible_resources = []
        for resource, level in self.resource_permissions.items():
            if level in accessible_levels:
                accessible_resources.append(resource)
        
        return accessible_resources

class ComplianceManager:
    """Compliance and audit management."""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis_client = redis_client
        self.audit_events: List[AuditEvent] = []
        
        # Compliance metrics
        self.audit_events_counter = Counter(
            'security_audit_events_total',
            'Total audit events',
            ['action', 'resource', 'success']
        )
        
        self.compliance_violations_counter = Counter(
            'security_compliance_violations_total',
            'Compliance violations',
            ['regulation', 'violation_type']
        )
    
    async def log_audit_event(self, event: AuditEvent):
        """Log audit event for compliance."""
        
        # Store in memory
        self.audit_events.append(event)
        
        # Store in Redis with TTL based on regulation requirements
        ttl = self._get_retention_period(event.compliance_tags)
        
        audit_data = {
            'event_id': event.event_id,
            'user_id': event.user_id,
            'action': event.action,
            'resource': event.resource,
            'timestamp': event.timestamp.isoformat(),
            'success': event.success,
            'details': event.details,
            'security_context': {
                'user_id': event.security_context.user_id,
                'roles': event.security_context.roles,
                'permissions': [p.value for p in event.security_context.permissions],
                'security_level': event.security_context.security_level.value,
                'ip_address': event.security_context.ip_address,
                'mfa_verified': event.security_context.mfa_verified
            },
            'compliance_tags': [reg.value for reg in event.compliance_tags]
        }
        
        await self.redis_client.setex(
            f"audit:{event.event_id}",
            ttl,
            json.dumps(audit_data)
        )
        
        # Add to daily audit log
        daily_key = f"audit_log:{event.timestamp.strftime('%Y%m%d')}"
        await self.redis_client.lpush(daily_key, event.event_id)
        await self.redis_client.expire(daily_key, ttl)
        
        # Update metrics
        self.audit_events_counter.labels(
            action=event.action,
            resource=event.resource,
            success=str(event.success)
        ).inc()
        
        logger.info(f"Audit event logged: {event.event_id}")
    
    def _get_retention_period(self, regulations: List[ComplianceRegulation]) -> int:
        """Get data retention period based on regulations."""
        
        retention_periods = {
            ComplianceRegulation.GDPR: 7 * 365 * 24 * 3600,    # 7 years
            ComplianceRegulation.CCPA: 2 * 365 * 24 * 3600,    # 2 years
            ComplianceRegulation.HIPAA: 6 * 365 * 24 * 3600,   # 6 years
            ComplianceRegulation.SOX: 7 * 365 * 24 * 3600,     # 7 years
            ComplianceRegulation.PCI_DSS: 1 * 365 * 24 * 3600  # 1 year
        }
        
        if not regulations:
            return 365 * 24 * 3600  # 1 year default
        
        # Return the longest retention period required
        return max(
            retention_periods.get(reg, 365 * 24 * 3600)
            for reg in regulations
        )
    
    async def check_gdpr_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check GDPR compliance of data processing."""
        
        compliance_report = {
            'compliant': True,
            'violations': [],
            'recommendations': []
        }
        
        # Check for personal data without consent
        personal_data_fields = {
            'email', 'name', 'phone', 'address', 'ip_address'
        }
        
        found_personal_data = set(data.keys()) & personal_data_fields
        
        if found_personal_data:
            if not data.get('consent_given', False):
                compliance_report['compliant'] = False
                compliance_report['violations'].append({
                    'type': 'missing_consent',
                    'fields': list(found_personal_data),
                    'regulation': 'GDPR Article 6'
                })
        
        # Check data minimization
        if len(data) > 20:  # Arbitrary threshold
            compliance_report['recommendations'].append({
                'type': 'data_minimization',
                'message': 'Consider reducing data collection to minimum necessary',
                'regulation': 'GDPR Article 5(1)(c)'
            })
        
        # Check for data retention policy
        if 'created_at' in data:
            created_at = datetime.fromisoformat(data['created_at'])
            if (datetime.now() - created_at).days > 365:
                compliance_report['recommendations'].append({
                    'type': 'data_retention',
                    'message': 'Review if data retention exceeds necessary period',
                    'regulation': 'GDPR Article 5(1)(e)'
                })
        
        return compliance_report
    
    async def generate_compliance_report(self, 
                                       regulation: ComplianceRegulation,
                                       start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specified period."""
        
        # Get audit events for period
        period_events = []
        
        for days_back in range((end_date - start_date).days + 1):
            check_date = start_date + timedelta(days=days_back)
            daily_key = f"audit_log:{check_date.strftime('%Y%m%d')}"
            
            event_ids = await self.redis_client.lrange(daily_key, 0, -1)
            
            for event_id in event_ids:
                event_data = await self.redis_client.get(f"audit:{event_id}")
                if event_data:
                    event = json.loads(event_data)
                    if regulation.value in event.get('compliance_tags', []):
                        period_events.append(event)
        
        # Generate report
        report = {
            'regulation': regulation.value,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_events': len(period_events),
            'successful_operations': sum(1 for e in period_events if e['success']),
            'failed_operations': sum(1 for e in period_events if not e['success']),
            'unique_users': len(set(e['user_id'] for e in period_events)),
            'most_common_actions': self._get_action_frequency(period_events),
            'security_incidents': self._identify_security_incidents(period_events),
            'compliance_violations': []
        }
        
        return report
    
    def _get_action_frequency(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get frequency of actions in events."""
        
        action_counts = {}
        for event in events:
            action = event['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Sort by frequency
        return dict(
            sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        )
    
    def _identify_security_incidents(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential security incidents."""
        
        incidents = []
        
        # Failed login attempts
        failed_logins = [e for e in events if e['action'] == 'login' and not e['success']]
        
        # Group by user
        user_failures = {}
        for event in failed_logins:
            user_id = event['user_id']
            user_failures[user_id] = user_failures.get(user_id, 0) + 1
        
        # Flag users with many failures
        for user_id, failure_count in user_failures.items():
            if failure_count >= 5:
                incidents.append({
                    'type': 'potential_brute_force',
                    'user_id': user_id,
                    'failure_count': failure_count,
                    'severity': 'high' if failure_count >= 10 else 'medium'
                })
        
        return incidents

class ThreatDetectionEngine:
    """Advanced threat detection and prevention."""
    
    def __init__(self):
        self.threat_patterns = self._initialize_threat_patterns()
        self.suspicious_activities = []
        
        # Threat detection metrics
        self.threats_detected_counter = Counter(
            'security_threats_detected_total',
            'Threats detected',
            ['threat_type', 'severity']
        )
    
    def _initialize_threat_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize threat detection patterns."""
        
        return {
            'brute_force': {
                'description': 'Multiple failed authentication attempts',
                'threshold': 5,
                'time_window': 300,  # 5 minutes
                'severity': 'high'
            },
            'data_exfiltration': {
                'description': 'Unusual data access patterns',
                'threshold': 100,
                'time_window': 3600,  # 1 hour
                'severity': 'critical'
            },
            'privilege_escalation': {
                'description': 'Unauthorized access to restricted resources',
                'threshold': 1,
                'time_window': 0,
                'severity': 'critical'
            },
            'anomalous_api_usage': {
                'description': 'Unusual API call patterns',
                'threshold': 1000,
                'time_window': 3600,
                'severity': 'medium'
            }
        }
    
    async def analyze_security_event(self, event: AuditEvent) -> List[Dict[str, Any]]:
        """Analyze event for security threats."""
        
        threats = []
        
        # Check for failed authentication
        if event.action == 'login' and not event.success:
            threat = await self._check_brute_force(event)
            if threat:
                threats.append(threat)
        
        # Check for privilege escalation
        if event.action in ['role_assignment', 'permission_grant']:
            threat = await self._check_privilege_escalation(event)
            if threat:
                threats.append(threat)
        
        # Check for data access anomalies
        if event.action in ['data_access', 'export_data']:
            threat = await self._check_data_exfiltration(event)
            if threat:
                threats.append(threat)
        
        # Record detected threats
        for threat in threats:
            self.threats_detected_counter.labels(
                threat_type=threat['type'],
                severity=threat['severity']
            ).inc()
        
        return threats
    
    async def _check_brute_force(self, event: AuditEvent) -> Optional[Dict[str, Any]]:
        """Check for brute force attacks."""
        
        pattern = self.threat_patterns['brute_force']
        
        # Count recent failed attempts from same IP
        recent_failures = sum(
            1 for activity in self.suspicious_activities
            if (
                activity.get('ip_address') == event.security_context.ip_address and
                activity.get('action') == 'login_failed' and
                (event.timestamp - activity.get('timestamp', datetime.min)).total_seconds() <= pattern['time_window']
            )
        )
        
        if recent_failures >= pattern['threshold']:
            return {
                'type': 'brute_force',
                'severity': pattern['severity'],
                'details': {
                    'ip_address': event.security_context.ip_address,
                    'failure_count': recent_failures,
                    'user_id': event.user_id
                }
            }
        
        return None
    
    async def _check_privilege_escalation(self, event: AuditEvent) -> Optional[Dict[str, Any]]:
        """Check for privilege escalation attempts."""
        
        # Check if user is trying to grant themselves higher privileges
        details = event.details
        
        if (details.get('target_user') == event.user_id and
            details.get('granted_role') in ['admin', 'security_admin']):
            
            return {
                'type': 'privilege_escalation',
                'severity': 'critical',
                'details': {
                    'user_id': event.user_id,
                    'attempted_role': details.get('granted_role'),
                    'self_granted': True
                }
            }
        
        return None
    
    async def _check_data_exfiltration(self, event: AuditEvent) -> Optional[Dict[str, Any]]:
        """Check for potential data exfiltration."""
        
        pattern = self.threat_patterns['data_exfiltration']
        
        # Count recent data access from same user
        recent_access_count = sum(
            1 for activity in self.suspicious_activities
            if (
                activity.get('user_id') == event.user_id and
                activity.get('action') in ['data_access', 'export_data'] and
                (event.timestamp - activity.get('timestamp', datetime.min)).total_seconds() <= pattern['time_window']
            )
        )
        
        if recent_access_count >= pattern['threshold']:
            return {
                'type': 'data_exfiltration',
                'severity': pattern['severity'],
                'details': {
                    'user_id': event.user_id,
                    'access_count': recent_access_count,
                    'time_window_hours': pattern['time_window'] / 3600
                }
            }
        
        return None

# Global security components
ENCRYPTION_MANAGER = EncryptionManager()
ACCESS_CONTROL_MANAGER = AccessControlManager()
THREAT_DETECTION_ENGINE = ThreatDetectionEngine()

def create_security_context(user_id: str, roles: List[str], 
                          ip_address: str, session_id: str,
                          mfa_verified: bool = False) -> SecurityContext:
    """Create security context for operations."""
    
    # Get permissions from roles
    permissions = []
    security_level = SecurityLevel.PUBLIC
    
    for role in roles:
        if role in ACCESS_CONTROL_MANAGER.roles:
            role_data = ACCESS_CONTROL_MANAGER.roles[role]
            permissions.extend(role_data['permissions'])
            
            # Get highest security level
            role_levels = role_data['security_levels']
            for level in [SecurityLevel.TOP_SECRET, SecurityLevel.RESTRICTED, 
                         SecurityLevel.CONFIDENTIAL, SecurityLevel.INTERNAL, 
                         SecurityLevel.PUBLIC]:
                if level in role_levels and level.value > security_level.value:
                    security_level = level
                    break
    
    return SecurityContext(
        user_id=user_id,
        roles=roles,
        permissions=list(set(permissions)),
        security_level=security_level,
        session_id=session_id,
        ip_address=ip_address,
        timestamp=datetime.now(),
        mfa_verified=mfa_verified
    )

__all__ = [
    'SecurityLevel',
    'AccessPermission',
    'ComplianceRegulation',
    'SecurityContext',
    'AuditEvent',
    'EncryptionManager',
    'DataAnonymizer',
    'AccessControlManager',
    'ComplianceManager',
    'ThreatDetectionEngine',
    'ENCRYPTION_MANAGER',
    'ACCESS_CONTROL_MANAGER',
    'THREAT_DETECTION_ENGINE',
    'create_security_context'
]
