#!/usr/bin/env python3
"""
Enterprise Database Security Audit Script
==========================================

This script provides comprehensive security auditing for multi-database
enterprise environments with advanced threat detection, compliance checking,
and security recommendation engine.

Features:
- Multi-database security assessment
- Compliance framework validation (SOX, GDPR, HIPAA, PCI-DSS)
- Vulnerability scanning and threat detection
- Access control and privilege audit
- Data encryption and privacy validation
- Security configuration assessment
- Automated security hardening recommendations
- Security policy enforcement validation
"""

import asyncio
import json
import logging
import time
import sys
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import argparse
import yaml
import ipaddress
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from connection_manager import ConnectionManager
from security_validator import SecurityValidator, EncryptionManager
from __init__ import DatabaseType, ConfigurationLoader


class SecurityLevel(Enum):
    """Security audit level enumeration"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    PARANOID = "paranoid"


class ComplianceFramework(Enum):
    """Compliance framework enumeration"""
    SOX = "sox"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST = "nist"


class SeverityLevel(Enum):
    """Security issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityCategory(Enum):
    """Vulnerability category enumeration"""
    ACCESS_CONTROL = "access_control"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"
    CONFIGURATION = "configuration"
    NETWORK_SECURITY = "network_security"
    DATA_PRIVACY = "data_privacy"
    AUDIT_LOGGING = "audit_logging"
    BACKUP_SECURITY = "backup_security"
    INJECTION = "injection"


@dataclass
class SecurityFinding:
    """Security audit finding data structure"""
    finding_id: str
    title: str
    description: str
    category: VulnerabilityCategory
    severity: SeverityLevel
    
    # Technical details
    database_type: DatabaseType
    affected_objects: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Risk assessment
    risk_score: float = 0.0  # 0-10 scale
    exploitability: str = "unknown"  # low, medium, high
    impact: str = "unknown"  # low, medium, high
    
    # Remediation
    remediation_steps: List[str] = field(default_factory=list)
    remediation_sql: Optional[str] = None
    remediation_config: Optional[Dict[str, Any]] = None
    remediation_effort: str = "unknown"  # low, medium, high
    
    # Compliance
    compliance_violations: List[ComplianceFramework] = field(default_factory=list)
    regulatory_notes: str = ""
    
    # Metadata
    tenant_id: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)
    false_positive: bool = False
    acknowledged: bool = False


@dataclass
class SecurityReport:
    """Security audit report data structure"""
    report_id: str
    tenant_id: str
    audit_timestamp: datetime
    
    # Audit scope
    databases_audited: List[DatabaseType]
    security_level: SecurityLevel
    compliance_frameworks: List[ComplianceFramework]
    
    # Results summary
    total_findings: int
    findings_by_severity: Dict[SeverityLevel, int]
    findings_by_category: Dict[VulnerabilityCategory, int]
    
    # Risk assessment
    overall_risk_score: float
    security_posture: str  # excellent, good, fair, poor, critical
    
    # Findings
    findings: List[SecurityFinding]
    
    # Compliance status
    compliance_status: Dict[ComplianceFramework, str]  # compliant, non_compliant, partial
    
    # Recommendations
    immediate_actions: List[str]
    security_roadmap: List[str]
    
    # Metadata
    audit_duration_seconds: float
    next_audit_recommended: datetime


class DatabaseSecurityAuditor:
    """
    Comprehensive security auditor for enterprise database environments
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.config_loader = ConfigurationLoader()
        self.security_validators: Dict[DatabaseType, SecurityValidator] = {}
        self.encryption_manager = EncryptionManager(config.get('encryption', {}))
        
        # Audit configuration
        self.security_level = SecurityLevel(config.get('security_level', 'standard'))
        self.compliance_frameworks = [
            ComplianceFramework(fw) for fw in config.get('compliance_frameworks', ['gdpr'])
        ]
        
        # Security policies and rules
        self.security_policies = self._load_security_policies()
        self.vulnerability_rules = self._load_vulnerability_rules()
        
        # Known vulnerabilities database
        self.known_vulnerabilities = self._load_known_vulnerabilities()
    
    async def conduct_security_audit(self, 
                                   database_types: Optional[List[DatabaseType]] = None,
                                   tenant_id: Optional[str] = None,
                                   security_level: Optional[SecurityLevel] = None,
                                   compliance_frameworks: Optional[List[ComplianceFramework]] = None) -> SecurityReport:
        """Conduct comprehensive security audit"""
        
        start_time = time.time()
        audit_timestamp = datetime.now()
        
        security_level = security_level or self.security_level
        compliance_frameworks = compliance_frameworks or self.compliance_frameworks
        
        self.logger.info(f"Starting security audit - Level: {security_level.value}, Tenant: {tenant_id}")
        
        # Determine databases to audit
        if database_types is None:
            database_types = self.config_loader.list_available_configurations()
        
        all_findings = []
        
        # Audit each database type
        for db_type in database_types:
            try:
                findings = await self._audit_database_security(
                    db_type, tenant_id, security_level, compliance_frameworks
                )
                all_findings.extend(findings)
                
            except Exception as e:
                self.logger.error(f"Security audit failed for {db_type.value}: {e}")
                # Create finding for audit failure
                finding = SecurityFinding(
                    finding_id=f"audit_failure_{db_type.value}",
                    title=f"Security audit failed for {db_type.value}",
                    description=f"Unable to complete security audit: {str(e)}",
                    category=VulnerabilityCategory.CONFIGURATION,
                    severity=SeverityLevel.HIGH,
                    database_type=db_type,
                    tenant_id=tenant_id or ""
                )
                all_findings.append(finding)
        
        # Conduct cross-database security analysis
        cross_db_findings = await self._conduct_cross_database_analysis(
            all_findings, database_types, security_level
        )
        all_findings.extend(cross_db_findings)
        
        # Generate security report
        execution_time = time.time() - start_time
        
        report = SecurityReport(
            report_id=f"audit_{int(audit_timestamp.timestamp())}",
            tenant_id=tenant_id or "global",
            audit_timestamp=audit_timestamp,
            databases_audited=database_types,
            security_level=security_level,
            compliance_frameworks=compliance_frameworks,
            total_findings=len(all_findings),
            findings_by_severity=self._count_by_severity(all_findings),
            findings_by_category=self._count_by_category(all_findings),
            overall_risk_score=self._calculate_overall_risk_score(all_findings),
            security_posture=self._assess_security_posture(all_findings),
            findings=all_findings,
            compliance_status=self._assess_compliance_status(all_findings, compliance_frameworks),
            immediate_actions=self._generate_immediate_actions(all_findings),
            security_roadmap=self._generate_security_roadmap(all_findings),
            audit_duration_seconds=round(execution_time, 2),
            next_audit_recommended=audit_timestamp + timedelta(days=30)
        )
        
        self.logger.info(f"Security audit completed - {len(all_findings)} findings, risk score: {report.overall_risk_score:.1f}")
        
        return report
    
    async def _audit_database_security(self, 
                                     db_type: DatabaseType,
                                     tenant_id: Optional[str],
                                     security_level: SecurityLevel,
                                     compliance_frameworks: List[ComplianceFramework]) -> List[SecurityFinding]:
        """Audit security for specific database type"""
        
        findings = []
        
        try:
            # Load database configuration
            config = self.config_loader.load_configuration(db_type, tenant_id)
            
            # Authentication and authorization audit
            auth_findings = await self._audit_authentication_authorization(config, security_level)
            findings.extend(auth_findings)
            
            # Access control audit
            access_findings = await self._audit_access_control(config, security_level)
            findings.extend(access_findings)
            
            # Encryption audit
            encryption_findings = await self._audit_encryption(config, security_level)
            findings.extend(encryption_findings)
            
            # Configuration security audit
            config_findings = await self._audit_configuration_security(config, security_level)
            findings.extend(config_findings)
            
            # Network security audit
            network_findings = await self._audit_network_security(config, security_level)
            findings.extend(network_findings)
            
            # Data privacy audit
            privacy_findings = await self._audit_data_privacy(config, security_level, compliance_frameworks)
            findings.extend(privacy_findings)
            
            # Audit logging audit
            logging_findings = await self._audit_logging_configuration(config, security_level)
            findings.extend(logging_findings)
            
            # Backup security audit
            backup_findings = await self._audit_backup_security(config, security_level)
            findings.extend(backup_findings)
            
            # Vulnerability scanning
            vuln_findings = await self._scan_vulnerabilities(config, security_level)
            findings.extend(vuln_findings)
            
            # Database-specific security checks
            if db_type == DatabaseType.POSTGRESQL:
                pg_findings = await self._audit_postgresql_security(config, security_level)
                findings.extend(pg_findings)
            elif db_type == DatabaseType.MONGODB:
                mongo_findings = await self._audit_mongodb_security(config, security_level)
                findings.extend(mongo_findings)
            elif db_type == DatabaseType.REDIS:
                redis_findings = await self._audit_redis_security(config, security_level)
                findings.extend(redis_findings)
            
        except Exception as e:
            self.logger.error(f"Database security audit failed for {db_type.value}: {e}")
            raise
        
        return findings
    
    async def _audit_authentication_authorization(self, config, security_level: SecurityLevel) -> List[SecurityFinding]:
        """Audit authentication and authorization mechanisms"""
        
        findings = []
        
        # Check for default credentials
        if await self._check_default_credentials(config):
            finding = SecurityFinding(
                finding_id=f"auth_default_creds_{config.type.value}",
                title="Default credentials detected",
                description="Database is using default or weak credentials",
                category=VulnerabilityCategory.AUTHENTICATION,
                severity=SeverityLevel.CRITICAL,
                database_type=config.type,
                risk_score=9.5,
                exploitability="high",
                impact="high",
                remediation_steps=[
                    "Change default passwords immediately",
                    "Implement strong password policy",
                    "Enable multi-factor authentication if available",
                    "Rotate credentials regularly"
                ],
                compliance_violations=[ComplianceFramework.GDPR, ComplianceFramework.SOX],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        # Check password strength policy
        if not await self._check_password_policy(config):
            finding = SecurityFinding(
                finding_id=f"auth_weak_password_policy_{config.type.value}",
                title="Weak password policy",
                description="Password policy does not meet security standards",
                category=VulnerabilityCategory.AUTHENTICATION,
                severity=SeverityLevel.HIGH,
                database_type=config.type,
                risk_score=7.0,
                exploitability="medium",
                impact="high",
                remediation_steps=[
                    "Implement minimum password length (12+ characters)",
                    "Require password complexity (uppercase, lowercase, numbers, symbols)",
                    "Enable password expiration",
                    "Prevent password reuse"
                ],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        # Check for excessive privileges
        excessive_privs = await self._check_excessive_privileges(config)
        if excessive_privs:
            finding = SecurityFinding(
                finding_id=f"auth_excessive_privileges_{config.type.value}",
                title="Excessive user privileges detected",
                description="Users have more privileges than necessary (principle of least privilege violation)",
                category=VulnerabilityCategory.AUTHORIZATION,
                severity=SeverityLevel.HIGH,
                database_type=config.type,
                affected_objects=excessive_privs,
                risk_score=6.5,
                exploitability="medium",
                impact="high",
                remediation_steps=[
                    "Review and reduce user privileges",
                    "Implement role-based access control",
                    "Regular privilege audits",
                    "Remove unnecessary administrative access"
                ],
                compliance_violations=[ComplianceFramework.SOX, ComplianceFramework.ISO27001],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _audit_access_control(self, config, security_level: SecurityLevel) -> List[SecurityFinding]:
        """Audit access control mechanisms"""
        
        findings = []
        
        # Check for anonymous access
        if await self._check_anonymous_access(config):
            finding = SecurityFinding(
                finding_id=f"access_anonymous_{config.type.value}",
                title="Anonymous access enabled",
                description="Database allows anonymous/unauthenticated access",
                category=VulnerabilityCategory.ACCESS_CONTROL,
                severity=SeverityLevel.CRITICAL,
                database_type=config.type,
                risk_score=9.0,
                exploitability="high",
                impact="high",
                remediation_steps=[
                    "Disable anonymous access",
                    "Require authentication for all connections",
                    "Implement proper user management",
                    "Review and update access control lists"
                ],
                compliance_violations=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.PCI_DSS],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        # Check network access restrictions
        network_issues = await self._check_network_access_restrictions(config)
        if network_issues:
            finding = SecurityFinding(
                finding_id=f"access_network_{config.type.value}",
                title="Insufficient network access restrictions",
                description="Database accepts connections from unrestricted networks",
                category=VulnerabilityCategory.ACCESS_CONTROL,
                severity=SeverityLevel.HIGH,
                database_type=config.type,
                evidence={"network_issues": network_issues},
                risk_score=7.5,
                exploitability="high",
                impact="medium",
                remediation_steps=[
                    "Restrict database access to specific IP ranges",
                    "Use VPN or private networks",
                    "Implement firewall rules",
                    "Disable public internet access"
                ],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _audit_encryption(self, config, security_level: SecurityLevel) -> List[SecurityFinding]:
        """Audit encryption implementation"""
        
        findings = []
        
        # Check encryption in transit
        if not await self._check_encryption_in_transit(config):
            finding = SecurityFinding(
                finding_id=f"crypto_transit_{config.type.value}",
                title="Encryption in transit not enabled",
                description="Database connections are not encrypted (no TLS/SSL)",
                category=VulnerabilityCategory.ENCRYPTION,
                severity=SeverityLevel.HIGH,
                database_type=config.type,
                risk_score=8.0,
                exploitability="high",
                impact="high",
                remediation_steps=[
                    "Enable TLS/SSL for all connections",
                    "Use strong cipher suites",
                    "Implement certificate validation",
                    "Disable weak encryption protocols"
                ],
                compliance_violations=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.PCI_DSS],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        # Check encryption at rest
        if not await self._check_encryption_at_rest(config):
            finding = SecurityFinding(
                finding_id=f"crypto_rest_{config.type.value}",
                title="Encryption at rest not enabled",
                description="Database files and backups are not encrypted",
                category=VulnerabilityCategory.ENCRYPTION,
                severity=SeverityLevel.HIGH,
                database_type=config.type,
                risk_score=7.5,
                exploitability="medium",
                impact="high",
                remediation_steps=[
                    "Enable transparent data encryption (TDE)",
                    "Encrypt database files and tablespaces",
                    "Encrypt backups and log files",
                    "Implement proper key management"
                ],
                compliance_violations=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        # Check key management
        key_issues = await self._check_key_management(config)
        if key_issues:
            finding = SecurityFinding(
                finding_id=f"crypto_keys_{config.type.value}",
                title="Poor key management practices",
                description="Encryption key management does not follow best practices",
                category=VulnerabilityCategory.ENCRYPTION,
                severity=SeverityLevel.MEDIUM,
                database_type=config.type,
                evidence={"key_issues": key_issues},
                risk_score=6.0,
                exploitability="medium",
                impact="medium",
                remediation_steps=[
                    "Implement proper key rotation",
                    "Use hardware security modules (HSM)",
                    "Separate key management from data",
                    "Implement key escrow procedures"
                ],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _audit_configuration_security(self, config, security_level: SecurityLevel) -> List[SecurityFinding]:
        """Audit database configuration security"""
        
        findings = []
        
        # Check for insecure configuration options
        insecure_configs = await self._check_insecure_configurations(config)
        for config_issue in insecure_configs:
            finding = SecurityFinding(
                finding_id=f"config_{config_issue['name']}_{config.type.value}",
                title=f"Insecure configuration: {config_issue['name']}",
                description=config_issue['description'],
                category=VulnerabilityCategory.CONFIGURATION,
                severity=SeverityLevel(config_issue['severity']),
                database_type=config.type,
                evidence={"current_value": config_issue['current_value']},
                risk_score=config_issue['risk_score'],
                remediation_steps=config_issue['remediation_steps'],
                remediation_config={config_issue['name']: config_issue['recommended_value']},
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _audit_network_security(self, config, security_level: SecurityLevel) -> List[SecurityFinding]:
        """Audit network security configuration"""
        
        findings = []
        
        # Check for open ports
        open_ports = await self._check_open_ports(config)
        if open_ports:
            finding = SecurityFinding(
                finding_id=f"network_open_ports_{config.type.value}",
                title="Unnecessary open ports detected",
                description="Database has unnecessary network ports exposed",
                category=VulnerabilityCategory.NETWORK_SECURITY,
                severity=SeverityLevel.MEDIUM,
                database_type=config.type,
                affected_objects=open_ports,
                risk_score=5.5,
                exploitability="medium",
                impact="medium",
                remediation_steps=[
                    "Close unnecessary ports",
                    "Use non-standard ports if possible",
                    "Implement port-based access controls",
                    "Monitor network traffic"
                ],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _audit_data_privacy(self, config, security_level: SecurityLevel, 
                                compliance_frameworks: List[ComplianceFramework]) -> List[SecurityFinding]:
        """Audit data privacy and protection measures"""
        
        findings = []
        
        # Check for PII without encryption
        unencrypted_pii = await self._check_unencrypted_pii(config)
        if unencrypted_pii:
            finding = SecurityFinding(
                finding_id=f"privacy_unencrypted_pii_{config.type.value}",
                title="Personally Identifiable Information (PII) not encrypted",
                description="PII data found without proper encryption or masking",
                category=VulnerabilityCategory.DATA_PRIVACY,
                severity=SeverityLevel.HIGH,
                database_type=config.type,
                affected_objects=unencrypted_pii,
                risk_score=8.5,
                exploitability="medium",
                impact="high",
                remediation_steps=[
                    "Encrypt sensitive PII columns",
                    "Implement data masking",
                    "Use tokenization for PII",
                    "Implement data classification"
                ],
                compliance_violations=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
                regulatory_notes="GDPR Article 32 requires appropriate technical measures for data protection",
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        # Check data retention policies
        if not await self._check_data_retention_policies(config):
            finding = SecurityFinding(
                finding_id=f"privacy_retention_{config.type.value}",
                title="Data retention policies not implemented",
                description="No automated data retention and deletion policies found",
                category=VulnerabilityCategory.DATA_PRIVACY,
                severity=SeverityLevel.MEDIUM,
                database_type=config.type,
                risk_score=6.0,
                remediation_steps=[
                    "Implement automated data retention policies",
                    "Define data lifecycle management",
                    "Implement right to be forgotten procedures",
                    "Regular data purging schedules"
                ],
                compliance_violations=[ComplianceFramework.GDPR] if ComplianceFramework.GDPR in compliance_frameworks else [],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _audit_logging_configuration(self, config, security_level: SecurityLevel) -> List[SecurityFinding]:
        """Audit audit logging configuration"""
        
        findings = []
        
        # Check if audit logging is enabled
        if not await self._check_audit_logging_enabled(config):
            finding = SecurityFinding(
                finding_id=f"logging_disabled_{config.type.value}",
                title="Audit logging not enabled",
                description="Database audit logging is not properly configured",
                category=VulnerabilityCategory.AUDIT_LOGGING,
                severity=SeverityLevel.HIGH,
                database_type=config.type,
                risk_score=7.0,
                exploitability="low",
                impact="high",
                remediation_steps=[
                    "Enable comprehensive audit logging",
                    "Log authentication attempts",
                    "Log privilege changes",
                    "Log data access and modifications"
                ],
                compliance_violations=[ComplianceFramework.SOX, ComplianceFramework.PCI_DSS],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        # Check log retention
        log_retention_issues = await self._check_log_retention(config)
        if log_retention_issues:
            finding = SecurityFinding(
                finding_id=f"logging_retention_{config.type.value}",
                title="Insufficient log retention period",
                description="Audit logs are not retained for sufficient period",
                category=VulnerabilityCategory.AUDIT_LOGGING,
                severity=SeverityLevel.MEDIUM,
                database_type=config.type,
                evidence={"retention_issues": log_retention_issues},
                risk_score=5.0,
                remediation_steps=[
                    "Increase log retention period",
                    "Implement log archiving",
                    "Secure log storage",
                    "Regular log backup procedures"
                ],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _audit_backup_security(self, config, security_level: SecurityLevel) -> List[SecurityFinding]:
        """Audit backup security measures"""
        
        findings = []
        
        # Check backup encryption
        if not await self._check_backup_encryption(config):
            finding = SecurityFinding(
                finding_id=f"backup_encryption_{config.type.value}",
                title="Backups not encrypted",
                description="Database backups are stored without encryption",
                category=VulnerabilityCategory.BACKUP_SECURITY,
                severity=SeverityLevel.HIGH,
                database_type=config.type,
                risk_score=7.5,
                exploitability="medium",
                impact="high",
                remediation_steps=[
                    "Enable backup encryption",
                    "Use strong encryption algorithms",
                    "Secure backup storage locations",
                    "Implement backup access controls"
                ],
                compliance_violations=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _scan_vulnerabilities(self, config, security_level: SecurityLevel) -> List[SecurityFinding]:
        """Scan for known vulnerabilities"""
        
        findings = []
        
        # Check database version for known vulnerabilities
        version_vulns = await self._check_version_vulnerabilities(config)
        for vuln in version_vulns:
            finding = SecurityFinding(
                finding_id=f"vuln_{vuln['cve_id']}_{config.type.value}",
                title=f"Known vulnerability: {vuln['cve_id']}",
                description=vuln['description'],
                category=VulnerabilityCategory.CONFIGURATION,
                severity=SeverityLevel(vuln['severity']),
                database_type=config.type,
                evidence={"cve_id": vuln['cve_id'], "version": vuln['version']},
                risk_score=vuln['risk_score'],
                exploitability=vuln['exploitability'],
                impact=vuln['impact'],
                remediation_steps=vuln['remediation_steps'],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _audit_postgresql_security(self, config, security_level: SecurityLevel) -> List[SecurityFinding]:
        """PostgreSQL-specific security audit"""
        
        findings = []
        
        # Check for pg_hba.conf misconfigurations
        hba_issues = await self._check_postgresql_hba_config(config)
        for issue in hba_issues:
            finding = SecurityFinding(
                finding_id=f"pg_hba_{issue['type']}",
                title=f"PostgreSQL pg_hba.conf issue: {issue['title']}",
                description=issue['description'],
                category=VulnerabilityCategory.ACCESS_CONTROL,
                severity=SeverityLevel(issue['severity']),
                database_type=config.type,
                evidence={"hba_line": issue['line']},
                risk_score=issue['risk_score'],
                remediation_steps=issue['remediation_steps'],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        # Check for superuser accounts
        superusers = await self._check_postgresql_superusers(config)
        if len(superusers) > 1:  # More than just postgres user
            finding = SecurityFinding(
                finding_id="pg_excessive_superusers",
                title="Excessive PostgreSQL superuser accounts",
                description="Multiple superuser accounts present",
                category=VulnerabilityCategory.AUTHORIZATION,
                severity=SeverityLevel.MEDIUM,
                database_type=config.type,
                affected_objects=superusers,
                risk_score=6.0,
                remediation_steps=[
                    "Reduce number of superuser accounts",
                    "Use role-based access instead",
                    "Regular superuser audit",
                    "Implement least privilege principle"
                ],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _audit_mongodb_security(self, config, security_level: SecurityLevel) -> List[SecurityFinding]:
        """MongoDB-specific security audit"""
        
        findings = []
        
        # Check for authentication disabled
        if not await self._check_mongodb_auth_enabled(config):
            finding = SecurityFinding(
                finding_id="mongo_auth_disabled",
                title="MongoDB authentication disabled",
                description="MongoDB is running without authentication enabled",
                category=VulnerabilityCategory.AUTHENTICATION,
                severity=SeverityLevel.CRITICAL,
                database_type=config.type,
                risk_score=9.5,
                exploitability="high",
                impact="high",
                remediation_steps=[
                    "Enable authentication",
                    "Create admin user",
                    "Configure authorization",
                    "Restart MongoDB with --auth"
                ],
                compliance_violations=[ComplianceFramework.GDPR, ComplianceFramework.SOX],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _audit_redis_security(self, config, security_level: SecurityLevel) -> List[SecurityFinding]:
        """Redis-specific security audit"""
        
        findings = []
        
        # Check for no password
        if not await self._check_redis_auth_enabled(config):
            finding = SecurityFinding(
                finding_id="redis_no_auth",
                title="Redis authentication not configured",
                description="Redis is accessible without authentication",
                category=VulnerabilityCategory.AUTHENTICATION,
                severity=SeverityLevel.HIGH,
                database_type=config.type,
                risk_score=8.0,
                exploitability="high",
                impact="medium",
                remediation_steps=[
                    "Set requirepass in redis.conf",
                    "Use strong password",
                    "Consider ACL for fine-grained access",
                    "Restrict network access"
                ],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        # Check for dangerous commands enabled
        dangerous_commands = await self._check_redis_dangerous_commands(config)
        if dangerous_commands:
            finding = SecurityFinding(
                finding_id="redis_dangerous_commands",
                title="Dangerous Redis commands enabled",
                description="Redis has dangerous commands that should be disabled",
                category=VulnerabilityCategory.CONFIGURATION,
                severity=SeverityLevel.MEDIUM,
                database_type=config.type,
                affected_objects=dangerous_commands,
                risk_score=5.5,
                remediation_steps=[
                    "Disable or rename dangerous commands",
                    "Use command ACLs",
                    "Monitor command usage",
                    "Implement command filtering"
                ],
                tenant_id=config.tenant_id or ""
            )
            findings.append(finding)
        
        return findings
    
    async def _conduct_cross_database_analysis(self, 
                                             findings: List[SecurityFinding],
                                             database_types: List[DatabaseType],
                                             security_level: SecurityLevel) -> List[SecurityFinding]:
        """Conduct cross-database security analysis"""
        
        cross_findings = []
        
        # Check for consistent security policies across databases
        if len(database_types) > 1:
            inconsistencies = self._check_security_policy_consistency(findings, database_types)
            if inconsistencies:
                finding = SecurityFinding(
                    finding_id="cross_db_policy_inconsistency",
                    title="Inconsistent security policies across databases",
                    description="Security configurations vary significantly between databases",
                    category=VulnerabilityCategory.CONFIGURATION,
                    severity=SeverityLevel.MEDIUM,
                    database_type=database_types[0],  # Representative
                    evidence={"inconsistencies": inconsistencies},
                    risk_score=5.0,
                    remediation_steps=[
                        "Standardize security policies",
                        "Implement centralized security management",
                        "Regular security policy reviews",
                        "Automate security configuration deployment"
                    ]
                )
                cross_findings.append(finding)
        
        return cross_findings
    
    # Helper methods for security checks (simulation)
    
    async def _check_default_credentials(self, config) -> bool:
        """Check for default credentials (simulated)"""
        # In production, this would check actual credentials
        return False  # Assume no default credentials for simulation
    
    async def _check_password_policy(self, config) -> bool:
        """Check password policy strength (simulated)"""
        return True  # Assume good password policy for simulation
    
    async def _check_excessive_privileges(self, config) -> List[str]:
        """Check for excessive user privileges (simulated)"""
        return []  # Assume no excessive privileges for simulation
    
    async def _check_anonymous_access(self, config) -> bool:
        """Check for anonymous access (simulated)"""
        return False  # Assume no anonymous access for simulation
    
    async def _check_network_access_restrictions(self, config) -> List[str]:
        """Check network access restrictions (simulated)"""
        return []  # Assume proper restrictions for simulation
    
    async def _check_encryption_in_transit(self, config) -> bool:
        """Check encryption in transit (simulated)"""
        return True  # Assume encryption enabled for simulation
    
    async def _check_encryption_at_rest(self, config) -> bool:
        """Check encryption at rest (simulated)"""
        return True  # Assume encryption enabled for simulation
    
    async def _check_key_management(self, config) -> List[str]:
        """Check key management practices (simulated)"""
        return []  # Assume good key management for simulation
    
    async def _check_insecure_configurations(self, config) -> List[Dict[str, Any]]:
        """Check for insecure configurations (simulated)"""
        # Simulate some common configuration issues
        if config.type == DatabaseType.POSTGRESQL:
            return [
                {
                    'name': 'log_statement',
                    'description': 'SQL statement logging not comprehensive enough',
                    'current_value': 'none',
                    'recommended_value': 'all',
                    'severity': 'medium',
                    'risk_score': 4.0,
                    'remediation_steps': [
                        'Set log_statement = all in postgresql.conf',
                        'Restart PostgreSQL service',
                        'Monitor log volume'
                    ]
                }
            ]
        return []
    
    async def _check_open_ports(self, config) -> List[str]:
        """Check for unnecessary open ports (simulated)"""
        return []  # Assume no unnecessary ports for simulation
    
    async def _check_unencrypted_pii(self, config) -> List[str]:
        """Check for unencrypted PII (simulated)"""
        return []  # Assume no unencrypted PII for simulation
    
    async def _check_data_retention_policies(self, config) -> bool:
        """Check data retention policies (simulated)"""
        return True  # Assume policies exist for simulation
    
    async def _check_audit_logging_enabled(self, config) -> bool:
        """Check if audit logging is enabled (simulated)"""
        return True  # Assume logging enabled for simulation
    
    async def _check_log_retention(self, config) -> List[str]:
        """Check log retention configuration (simulated)"""
        return []  # Assume proper retention for simulation
    
    async def _check_backup_encryption(self, config) -> bool:
        """Check backup encryption (simulated)"""
        return True  # Assume encryption enabled for simulation
    
    async def _check_version_vulnerabilities(self, config) -> List[Dict[str, Any]]:
        """Check for known version vulnerabilities (simulated)"""
        return []  # Assume no known vulnerabilities for simulation
    
    async def _check_postgresql_hba_config(self, config) -> List[Dict[str, Any]]:
        """Check PostgreSQL pg_hba.conf (simulated)"""
        return []  # Assume proper configuration for simulation
    
    async def _check_postgresql_superusers(self, config) -> List[str]:
        """Check PostgreSQL superusers (simulated)"""
        return ['postgres']  # Default superuser only
    
    async def _check_mongodb_auth_enabled(self, config) -> bool:
        """Check MongoDB authentication (simulated)"""
        return True  # Assume authentication enabled for simulation
    
    async def _check_redis_auth_enabled(self, config) -> bool:
        """Check Redis authentication (simulated)"""
        return True  # Assume authentication enabled for simulation
    
    async def _check_redis_dangerous_commands(self, config) -> List[str]:
        """Check Redis dangerous commands (simulated)"""
        return []  # Assume no dangerous commands for simulation
    
    def _check_security_policy_consistency(self, findings: List[SecurityFinding], 
                                         database_types: List[DatabaseType]) -> List[str]:
        """Check security policy consistency across databases"""
        # Analyze findings for consistency patterns
        inconsistencies = []
        
        # Group findings by category
        findings_by_category = defaultdict(list)
        for finding in findings:
            findings_by_category[finding.category].append(finding)
        
        # Check for inconsistent security levels
        for category, category_findings in findings_by_category.items():
            severities = {f.severity for f in category_findings}
            if len(severities) > 1:
                inconsistencies.append(f"{category.value}: Inconsistent severity levels across databases")
        
        return inconsistencies
    
    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security policies configuration"""
        return {
            'password_policy': {
                'min_length': 12,
                'require_complexity': True,
                'max_age_days': 90
            },
            'access_control': {
                'default_deny': True,
                'require_mfa': False,
                'session_timeout_minutes': 30
            }
        }
    
    def _load_vulnerability_rules(self) -> List[Dict[str, Any]]:
        """Load vulnerability detection rules"""
        return [
            {
                'rule_id': 'default_creds',
                'name': 'Default Credentials',
                'description': 'Detect default or weak credentials',
                'severity': 'critical'
            }
        ]
    
    def _load_known_vulnerabilities(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load known vulnerabilities database"""
        return {
            'postgresql': [
                {
                    'cve_id': 'CVE-2021-32027',
                    'description': 'Buffer overflow vulnerability',
                    'affected_versions': ['< 13.3'],
                    'severity': 'high',
                    'risk_score': 7.5
                }
            ],
            'mongodb': [],
            'redis': []
        }
    
    def _count_by_severity(self, findings: List[SecurityFinding]) -> Dict[SeverityLevel, int]:
        """Count findings by severity level"""
        counts = {severity: 0 for severity in SeverityLevel}
        for finding in findings:
            counts[finding.severity] += 1
        return counts
    
    def _count_by_category(self, findings: List[SecurityFinding]) -> Dict[VulnerabilityCategory, int]:
        """Count findings by category"""
        counts = {category: 0 for category in VulnerabilityCategory}
        for finding in findings:
            counts[finding.category] += 1
        return counts
    
    def _calculate_overall_risk_score(self, findings: List[SecurityFinding]) -> float:
        """Calculate overall risk score (0-10)"""
        if not findings:
            return 0.0
        
        # Weight by severity
        severity_weights = {
            SeverityLevel.CRITICAL: 10.0,
            SeverityLevel.HIGH: 7.0,
            SeverityLevel.MEDIUM: 4.0,
            SeverityLevel.LOW: 2.0,
            SeverityLevel.INFO: 1.0
        }
        
        total_weighted_score = sum(
            severity_weights.get(finding.severity, 1.0) for finding in findings
        )
        
        # Normalize to 0-10 scale
        max_possible_score = len(findings) * 10.0
        risk_score = (total_weighted_score / max_possible_score) * 10.0
        
        return min(10.0, risk_score)
    
    def _assess_security_posture(self, findings: List[SecurityFinding]) -> str:
        """Assess overall security posture"""
        if not findings:
            return "excellent"
        
        critical_count = sum(1 for f in findings if f.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SeverityLevel.HIGH)
        
        if critical_count > 0:
            return "critical"
        elif high_count > 3:
            return "poor"
        elif high_count > 0:
            return "fair"
        else:
            return "good"
    
    def _assess_compliance_status(self, findings: List[SecurityFinding], 
                                frameworks: List[ComplianceFramework]) -> Dict[ComplianceFramework, str]:
        """Assess compliance status for each framework"""
        status = {}
        
        for framework in frameworks:
            violations = [
                f for f in findings 
                if framework in f.compliance_violations
            ]
            
            if not violations:
                status[framework] = "compliant"
            elif len(violations) > 5:
                status[framework] = "non_compliant"
            else:
                status[framework] = "partial"
        
        return status
    
    def _generate_immediate_actions(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate immediate action items"""
        actions = []
        
        critical_findings = [f for f in findings if f.severity == SeverityLevel.CRITICAL]
        for finding in critical_findings[:3]:  # Top 3 critical issues
            actions.append(f"URGENT: {finding.title}")
        
        return actions
    
    def _generate_security_roadmap(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate security improvement roadmap"""
        roadmap = []
        
        # Group by category and prioritize
        category_counts = self._count_by_category(findings)
        
        if category_counts[VulnerabilityCategory.AUTHENTICATION] > 0:
            roadmap.append("Strengthen authentication mechanisms")
        
        if category_counts[VulnerabilityCategory.ENCRYPTION] > 0:
            roadmap.append("Implement comprehensive encryption strategy")
        
        if category_counts[VulnerabilityCategory.ACCESS_CONTROL] > 0:
            roadmap.append("Review and enhance access controls")
        
        roadmap.append("Implement regular security auditing")
        roadmap.append("Establish security monitoring and alerting")
        
        return roadmap


async def main():
    """Main function for command-line execution"""
    
    parser = argparse.ArgumentParser(description='Database Security Audit Script')
    parser.add_argument('--databases', nargs='+', 
                       choices=[db.value for db in DatabaseType],
                       help='Database types to audit')
    parser.add_argument('--tenant', type=str, help='Tenant ID to audit')
    parser.add_argument('--level', type=str, default='standard',
                       choices=[level.value for level in SecurityLevel],
                       help='Security audit level')
    parser.add_argument('--compliance', nargs='+',
                       choices=[fw.value for fw in ComplianceFramework],
                       default=['gdpr'],
                       help='Compliance frameworks to check')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--format', type=str, default='json',
                       choices=['json', 'yaml'],
                       help='Output format')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    
    # Parse parameters
    database_types = None
    if args.databases:
        database_types = [DatabaseType(db) for db in args.databases]
    
    security_level = SecurityLevel(args.level)
    compliance_frameworks = [ComplianceFramework(fw) for fw in args.compliance]
    
    # Create security auditor
    auditor = DatabaseSecurityAuditor(config)
    
    # Run security audit
    try:
        report = await auditor.conduct_security_audit(
            database_types=database_types,
            tenant_id=args.tenant,
            security_level=security_level,
            compliance_frameworks=compliance_frameworks
        )
        
        # Convert to serializable format
        report_dict = {
            'report_id': report.report_id,
            'tenant_id': report.tenant_id,
            'audit_timestamp': report.audit_timestamp.isoformat(),
            'databases_audited': [db.value for db in report.databases_audited],
            'security_level': report.security_level.value,
            'compliance_frameworks': [fw.value for fw in report.compliance_frameworks],
            'total_findings': report.total_findings,
            'findings_by_severity': {k.value: v for k, v in report.findings_by_severity.items()},
            'findings_by_category': {k.value: v for k, v in report.findings_by_category.items()},
            'overall_risk_score': report.overall_risk_score,
            'security_posture': report.security_posture,
            'findings': [
                {
                    'finding_id': f.finding_id,
                    'title': f.title,
                    'description': f.description,
                    'category': f.category.value,
                    'severity': f.severity.value,
                    'database_type': f.database_type.value,
                    'risk_score': f.risk_score,
                    'remediation_steps': f.remediation_steps,
                    'compliance_violations': [cv.value for cv in f.compliance_violations]
                }
                for f in report.findings
            ],
            'compliance_status': {k.value: v for k, v in report.compliance_status.items()},
            'immediate_actions': report.immediate_actions,
            'security_roadmap': report.security_roadmap,
            'audit_duration_seconds': report.audit_duration_seconds,
            'next_audit_recommended': report.next_audit_recommended.isoformat()
        }
        
        # Output results
        if args.format == 'yaml':
            output = yaml.dump(report_dict, default_flow_style=False)
        else:
            output = json.dumps(report_dict, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Security audit report written to {args.output}")
        else:
            print(output)
        
        # Summary
        print(f"\n🔒 Security Audit Summary:")
        print(f"   • Security Posture: {report.security_posture.upper()}")
        print(f"   • Risk Score: {report.overall_risk_score:.1f}/10")
        print(f"   • Total Findings: {report.total_findings}")
        print(f"   • Critical: {report.findings_by_severity[SeverityLevel.CRITICAL]}")
        print(f"   • High: {report.findings_by_severity[SeverityLevel.HIGH]}")
        print(f"   • Medium: {report.findings_by_severity[SeverityLevel.MEDIUM]}")
        
        if report.immediate_actions:
            print(f"\n⚠️  Immediate Actions Required:")
            for action in report.immediate_actions:
                print(f"   • {action}")
        
        # Exit with appropriate code
        if report.overall_risk_score > 7.0:
            sys.exit(2)  # High risk
        elif report.overall_risk_score > 4.0:
            sys.exit(1)  # Medium risk
        else:
            sys.exit(0)  # Low risk
        
    except Exception as e:
        print(f"Security audit failed: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == '__main__':
    asyncio.run(main())
