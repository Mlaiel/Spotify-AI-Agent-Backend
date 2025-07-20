"""
Spotify AI Agent - Security Event & Threat Formatters
====================================================

Ultra-advanced security event formatting system for threat intelligence,
incident response, compliance reporting, and security analytics.

This module handles sophisticated formatting for:
- Security incidents and threat intelligence reports
- Vulnerability assessments and penetration testing results
- Compliance reports (GDPR, SOX, PCI-DSS, ISO 27001)
- Authentication and authorization audit logs
- Network security events and intrusion detection
- Data loss prevention (DLP) and privacy impact assessments
- Security awareness training and phishing simulation results
- DevSecOps security pipeline and SAST/DAST reports
- Zero-trust architecture monitoring and behavioral analytics

Author: Fahed Mlaiel & Spotify Security Team
Version: 2.1.0
"""

import asyncio
import json
import hashlib
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
import re
from urllib.parse import urlparse

logger = structlog.get_logger(__name__)


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    DATA_BREACH = "data_breach"
    MALWARE_DETECTION = "malware_detection"
    NETWORK_INTRUSION = "network_intrusion"
    VULNERABILITY_DISCOVERED = "vulnerability_discovered"
    PHISHING_ATTEMPT = "phishing_attempt"
    INSIDER_THREAT = "insider_threat"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SYSTEM_COMPROMISE = "system_compromise"
    COMPLIANCE_VIOLATION = "compliance_violation"


class SeverityLevel(Enum):
    """Security event severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class ComplianceFramework(Enum):
    """Compliance frameworks."""
    GDPR = "gdpr"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    NIST = "nist"
    SOC2 = "soc2"
    CCPA = "ccpa"


class ThreatCategory(Enum):
    """Threat categories."""
    APT = "advanced_persistent_threat"
    RANSOMWARE = "ransomware"
    PHISHING = "phishing"
    MALWARE = "malware"
    DDOS = "ddos"
    SQL_INJECTION = "sql_injection"
    XSS = "cross_site_scripting"
    CSRF = "cross_site_request_forgery"
    SOCIAL_ENGINEERING = "social_engineering"
    ZERO_DAY = "zero_day"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    
    event_id: str
    event_type: SecurityEventType
    severity: SeverityLevel
    timestamp: datetime
    source_ip: str
    target_system: str
    user_id: Optional[str] = None
    description: str = ""
    threat_category: Optional[ThreatCategory] = None
    affected_assets: List[str] = field(default_factory=list)
    indicators_of_compromise: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "source_ip": self.source_ip,
            "target_system": self.target_system,
            "user_id": self.user_id,
            "description": self.description,
            "threat_category": self.threat_category.value if self.threat_category else None,
            "affected_assets": self.affected_assets,
            "indicators_of_compromise": self.indicators_of_compromise,
            "mitigation_actions": self.mitigation_actions,
            "compliance_frameworks": [f.value for f in self.compliance_frameworks],
            "metadata": self.metadata
        }


@dataclass
class VulnerabilityAssessment:
    """Vulnerability assessment results."""
    
    vulnerability_id: str
    cvss_score: float
    cve_id: Optional[str] = None
    affected_systems: List[str] = field(default_factory=list)
    description: str = ""
    impact_assessment: str = ""
    remediation_steps: List[str] = field(default_factory=list)
    exploit_available: bool = False
    patch_available: bool = False
    business_impact: str = "medium"
    discovery_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "vulnerability_id": self.vulnerability_id,
            "cvss_score": self.cvss_score,
            "cve_id": self.cve_id,
            "affected_systems": self.affected_systems,
            "description": self.description,
            "impact_assessment": self.impact_assessment,
            "remediation_steps": self.remediation_steps,
            "exploit_available": self.exploit_available,
            "patch_available": self.patch_available,
            "business_impact": self.business_impact,
            "discovery_date": self.discovery_date.isoformat() if self.discovery_date else None
        }


@dataclass
class ComplianceReport:
    """Compliance assessment report."""
    
    framework: ComplianceFramework
    assessment_date: datetime
    compliance_score: float
    passed_controls: int
    failed_controls: int
    total_controls: int
    critical_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    remediation_timeline: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "framework": self.framework.value,
            "assessment_date": self.assessment_date.isoformat(),
            "compliance_score": self.compliance_score,
            "passed_controls": self.passed_controls,
            "failed_controls": self.failed_controls,
            "total_controls": self.total_controls,
            "critical_findings": self.critical_findings,
            "recommendations": self.recommendations,
            "remediation_timeline": self.remediation_timeline
        }


@dataclass
class FormattedSecurityReport:
    """Container for formatted security report."""
    
    report_type: str
    content: str
    executive_summary: str
    technical_details: str
    recommendations: List[str] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_type": self.report_type,
            "content": self.content,
            "executive_summary": self.executive_summary,
            "technical_details": self.technical_details,
            "recommendations": self.recommendations,
            "visualizations": self.visualizations,
            "metrics": self.metrics,
            "alerts": self.alerts,
            "metadata": self.metadata
        }


class BaseSecurityFormatter:
    """Base class for security event formatters."""
    
    def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
        self.tenant_id = tenant_id
        self.config = config or {}
        self.logger = logger.bind(tenant_id=tenant_id, formatter=self.__class__.__name__)
        
        # Security thresholds
        self.severity_icons = {
            SeverityLevel.CRITICAL: "🚨",
            SeverityLevel.HIGH: "🔴",
            SeverityLevel.MEDIUM: "🟡",
            SeverityLevel.LOW: "🟢",
            SeverityLevel.INFORMATIONAL: "ℹ️"
        }
        
        self.cvss_thresholds = {
            'critical': 9.0,
            'high': 7.0,
            'medium': 4.0,
            'low': 0.1
        }
        
        # Compliance score thresholds
        self.compliance_thresholds = {
            'excellent': 95.0,
            'good': 85.0,
            'needs_improvement': 70.0,
            'poor': 50.0
        }
    
    def get_severity_icon(self, severity: SeverityLevel) -> str:
        """Get icon for severity level."""
        return self.severity_icons.get(severity, "❓")
    
    def get_cvss_severity(self, score: float) -> Tuple[str, str]:
        """Get CVSS severity level and icon."""
        if score >= self.cvss_thresholds['critical']:
            return "Critical", "🚨"
        elif score >= self.cvss_thresholds['high']:
            return "High", "🔴"
        elif score >= self.cvss_thresholds['medium']:
            return "Medium", "🟡"
        elif score >= self.cvss_thresholds['low']:
            return "Low", "🟢"
        else:
            return "None", "ℹ️"
    
    def get_compliance_status(self, score: float) -> Tuple[str, str]:
        """Get compliance status and icon."""
        if score >= self.compliance_thresholds['excellent']:
            return "Excellent", "🟢"
        elif score >= self.compliance_thresholds['good']:
            return "Good", "🟡"
        elif score >= self.compliance_thresholds['needs_improvement']:
            return "Needs Improvement", "🟠"
        elif score >= self.compliance_thresholds['poor']:
            return "Poor", "🔴"
        else:
            return "Critical", "🚨"
    
    def anonymize_ip(self, ip_address: str) -> str:
        """Anonymize IP address for privacy."""
        try:
            if '.' in ip_address:  # IPv4
                parts = ip_address.split('.')
                return f"{parts[0]}.{parts[1]}.xxx.xxx"
            elif ':' in ip_address:  # IPv6
                parts = ip_address.split(':')
                return f"{parts[0]}:{parts[1]}:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx"
            else:
                return "xxx.xxx.xxx.xxx"
        except:
            return "xxx.xxx.xxx.xxx"
    
    def mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in text."""
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '****@****.***', text)
        
        # Credit card numbers
        text = re.sub(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', '**** **** **** ****', text)
        
        # Social Security Numbers
        text = re.sub(r'\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b', '***-**-****', text)
        
        # Phone numbers
        text = re.sub(r'\b\d{3}[\s\-]?\d{3}[\s\-]?\d{4}\b', '***-***-****', text)
        
        return text
    
    async def format_security_report(self, data: Any) -> FormattedSecurityReport:
        """Format security report - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement format_security_report")


class SecurityIncidentFormatter(BaseSecurityFormatter):
    """Formatter for security incidents and threat intelligence."""
    
    async def format_security_report(self, security_event: SecurityEvent) -> FormattedSecurityReport:
        """Format comprehensive security incident report."""
        
        # Generate threat intelligence analysis
        threat_analysis = await self._analyze_threat_intelligence(security_event)
        
        # Create incident timeline
        timeline = await self._create_incident_timeline(security_event)
        
        # Generate impact assessment
        impact_assessment = await self._assess_incident_impact(security_event)
        
        # Create visualizations
        visualizations = await self._create_incident_visualizations(security_event, threat_analysis)
        
        # Generate recommendations
        recommendations = await self._generate_incident_recommendations(security_event, impact_assessment)
        
        # Create alerts
        alerts = await self._create_incident_alerts(security_event)
        
        # Format executive summary
        executive_summary = await self._format_executive_summary(security_event, impact_assessment)
        
        # Format technical details
        technical_details = await self._format_technical_details(security_event, threat_analysis)
        
        # Format main content
        content = await self._format_incident_content(security_event, timeline, impact_assessment)
        
        metrics = {
            "incident_severity": security_event.severity.value,
            "threat_score": threat_analysis.get('threat_score', 0),
            "impact_score": impact_assessment.get('impact_score', 0),
            "response_time_minutes": impact_assessment.get('response_time_minutes', 0),
            "affected_systems_count": len(security_event.affected_assets),
            "ioc_count": len(security_event.indicators_of_compromise)
        }
        
        metadata = {
            "report_type": "security_incident",
            "incident_id": security_event.event_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tenant_id": self.tenant_id,
            "event_type": security_event.event_type.value,
            "severity": security_event.severity.value,
            "threat_category": security_event.threat_category.value if security_event.threat_category else None
        }
        
        return FormattedSecurityReport(
            report_type="security_incident",
            content=content,
            executive_summary=executive_summary,
            technical_details=technical_details,
            recommendations=recommendations,
            visualizations=visualizations,
            metrics=metrics,
            alerts=alerts,
            metadata=metadata
        )
    
    async def _analyze_threat_intelligence(self, security_event: SecurityEvent) -> Dict[str, Any]:
        """Analyze threat intelligence for the security event."""
        
        analysis = {
            "threat_score": 0,
            "threat_actors": [],
            "attack_vectors": [],
            "ttp_analysis": {},  # Tactics, Techniques, Procedures
            "attribution": {},
            "campaign_indicators": [],
            "geographic_origin": "Unknown"
        }
        
        # Calculate threat score based on severity and type
        severity_scores = {
            SeverityLevel.CRITICAL: 100,
            SeverityLevel.HIGH: 80,
            SeverityLevel.MEDIUM: 60,
            SeverityLevel.LOW: 40,
            SeverityLevel.INFORMATIONAL: 20
        }
        
        base_score = severity_scores.get(security_event.severity, 50)
        
        # Adjust based on event type
        event_type_multipliers = {
            SecurityEventType.DATA_BREACH: 1.5,
            SecurityEventType.SYSTEM_COMPROMISE: 1.4,
            SecurityEventType.MALWARE_DETECTION: 1.3,
            SecurityEventType.NETWORK_INTRUSION: 1.2,
            SecurityEventType.PRIVILEGE_ESCALATION: 1.2,
            SecurityEventType.AUTHENTICATION_FAILURE: 0.8
        }
        
        multiplier = event_type_multipliers.get(security_event.event_type, 1.0)
        analysis['threat_score'] = min(100, int(base_score * multiplier))
        
        # Analyze attack vectors based on event type
        if security_event.event_type == SecurityEventType.PHISHING_ATTEMPT:
            analysis['attack_vectors'] = ["Email", "Social Engineering", "Credential Harvesting"]
            analysis['threat_actors'] = ["Cybercriminal Groups", "APT Groups"]
        elif security_event.event_type == SecurityEventType.MALWARE_DETECTION:
            analysis['attack_vectors'] = ["Malicious Downloads", "Email Attachments", "Drive-by Downloads"]
            analysis['threat_actors'] = ["Ransomware Groups", "Banking Trojans", "State-sponsored APT"]
        elif security_event.event_type == SecurityEventType.NETWORK_INTRUSION:
            analysis['attack_vectors'] = ["Vulnerability Exploitation", "Lateral Movement", "Command & Control"]
            analysis['threat_actors'] = ["APT Groups", "Insider Threats", "Cybercriminal Organizations"]
        
        # TTP Analysis (MITRE ATT&CK Framework)
        analysis['ttp_analysis'] = {
            "tactics": self._get_mitre_tactics(security_event.event_type),
            "techniques": self._get_mitre_techniques(security_event.event_type),
            "procedures": self._get_common_procedures(security_event.event_type)
        }
        
        # Geographic analysis based on source IP
        if security_event.source_ip:
            analysis['geographic_origin'] = await self._analyze_ip_geolocation(security_event.source_ip)
        
        return analysis
    
    def _get_mitre_tactics(self, event_type: SecurityEventType) -> List[str]:
        """Get MITRE ATT&CK tactics for event type."""
        tactics_mapping = {
            SecurityEventType.AUTHENTICATION_FAILURE: ["Credential Access", "Initial Access"],
            SecurityEventType.MALWARE_DETECTION: ["Execution", "Persistence", "Defense Evasion"],
            SecurityEventType.NETWORK_INTRUSION: ["Initial Access", "Lateral Movement", "Command and Control"],
            SecurityEventType.DATA_EXFILTRATION: ["Collection", "Exfiltration", "Command and Control"],
            SecurityEventType.PRIVILEGE_ESCALATION: ["Privilege Escalation", "Persistence"],
            SecurityEventType.SYSTEM_COMPROMISE: ["Persistence", "Defense Evasion", "Impact"]
        }
        return tactics_mapping.get(event_type, ["Unknown"])
    
    def _get_mitre_techniques(self, event_type: SecurityEventType) -> List[str]:
        """Get MITRE ATT&CK techniques for event type."""
        techniques_mapping = {
            SecurityEventType.AUTHENTICATION_FAILURE: ["T1110 - Brute Force", "T1078 - Valid Accounts"],
            SecurityEventType.MALWARE_DETECTION: ["T1059 - Command Line Interface", "T1105 - Ingress Tool Transfer"],
            SecurityEventType.NETWORK_INTRUSION: ["T1190 - Exploit Public-Facing Application", "T1021 - Remote Services"],
            SecurityEventType.DATA_EXFILTRATION: ["T1041 - Exfiltration Over C2 Channel", "T1048 - Exfiltration Over Alternative Protocol"],
            SecurityEventType.PRIVILEGE_ESCALATION: ["T1068 - Exploitation for Privilege Escalation", "T1055 - Process Injection"]
        }
        return techniques_mapping.get(event_type, ["Unknown"])
    
    def _get_common_procedures(self, event_type: SecurityEventType) -> List[str]:
        """Get common procedures for event type."""
        procedures_mapping = {
            SecurityEventType.PHISHING_ATTEMPT: ["Spear phishing emails", "Credential harvesting forms", "Malicious attachments"],
            SecurityEventType.MALWARE_DETECTION: ["Dropper execution", "Registry modification", "Process hollowing"],
            SecurityEventType.NETWORK_INTRUSION: ["Port scanning", "Vulnerability exploitation", "Backdoor installation"]
        }
        return procedures_mapping.get(event_type, ["Standard attack procedures"])
    
    async def _analyze_ip_geolocation(self, ip_address: str) -> str:
        """Analyze IP geolocation (simulated)."""
        # In production, this would use actual geolocation services
        high_risk_countries = ["Unknown", "Anonymized", "TOR Exit Node"]
        return "High-risk region" if any(risk in ip_address for risk in high_risk_countries) else "Standard region"
    
    async def _create_incident_timeline(self, security_event: SecurityEvent) -> List[Dict[str, Any]]:
        """Create incident timeline."""
        
        timeline = []
        base_time = security_event.timestamp
        
        # Initial detection
        timeline.append({
            "timestamp": base_time.isoformat(),
            "event": "Initial Detection",
            "description": f"{security_event.event_type.value.replace('_', ' ').title()} detected",
            "source": "Security Monitoring System",
            "severity": security_event.severity.value
        })
        
        # Analysis phase
        timeline.append({
            "timestamp": (base_time + timedelta(minutes=5)).isoformat(),
            "event": "Analysis Started",
            "description": "Security team initiated incident analysis",
            "source": "SOC Team",
            "severity": "informational"
        })
        
        # Containment (if high severity)
        if security_event.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            timeline.append({
                "timestamp": (base_time + timedelta(minutes=15)).isoformat(),
                "event": "Containment Actions",
                "description": "Immediate containment measures implemented",
                "source": "Incident Response Team",
                "severity": "medium"
            })
        
        # Investigation
        timeline.append({
            "timestamp": (base_time + timedelta(minutes=30)).isoformat(),
            "event": "Investigation",
            "description": "Detailed forensic investigation in progress",
            "source": "Digital Forensics Team",
            "severity": "informational"
        })
        
        return timeline
    
    async def _assess_incident_impact(self, security_event: SecurityEvent) -> Dict[str, Any]:
        """Assess incident impact."""
        
        impact = {
            "impact_score": 0,
            "business_impact": "Low",
            "financial_impact": 0,
            "operational_impact": "Minimal",
            "reputation_impact": "None",
            "regulatory_impact": "None",
            "affected_users": 0,
            "data_at_risk": False,
            "service_disruption": False,
            "response_time_minutes": 10
        }
        
        # Calculate impact score
        severity_scores = {
            SeverityLevel.CRITICAL: 90,
            SeverityLevel.HIGH: 70,
            SeverityLevel.MEDIUM: 50,
            SeverityLevel.LOW: 30,
            SeverityLevel.INFORMATIONAL: 10
        }
        
        base_impact = severity_scores.get(security_event.severity, 30)
        
        # Adjust based on affected assets
        asset_count = len(security_event.affected_assets)
        if asset_count > 10:
            base_impact += 20
        elif asset_count > 5:
            base_impact += 10
        
        # Adjust based on event type
        if security_event.event_type in [SecurityEventType.DATA_BREACH, SecurityEventType.SYSTEM_COMPROMISE]:
            base_impact += 15
            impact['data_at_risk'] = True
            impact['business_impact'] = "High"
            impact['operational_impact'] = "Significant"
        
        impact['impact_score'] = min(100, base_impact)
        
        # Estimate financial impact (simplified)
        if impact['impact_score'] > 80:
            impact['financial_impact'] = 100000  # $100k
            impact['reputation_impact'] = "Significant"
        elif impact['impact_score'] > 60:
            impact['financial_impact'] = 50000   # $50k
            impact['reputation_impact'] = "Moderate"
        elif impact['impact_score'] > 40:
            impact['financial_impact'] = 10000   # $10k
            impact['reputation_impact'] = "Minor"
        
        # Check compliance impact
        if security_event.compliance_frameworks:
            impact['regulatory_impact'] = "Potential compliance violations"
        
        return impact
    
    async def _create_incident_visualizations(self, security_event: SecurityEvent, 
                                            threat_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create visualizations for security incident."""
        
        visualizations = []
        
        # Threat score gauge
        threat_gauge = {
            "type": "gauge",
            "title": "Threat Score",
            "data": {
                "datasets": [{
                    "data": [threat_analysis['threat_score'], 100 - threat_analysis['threat_score']],
                    "backgroundColor": ["#FF6B6B", "#E0E0E0"],
                    "borderWidth": 0
                }]
            },
            "options": {
                "responsive": True,
                "circumference": 180,
                "rotation": 270,
                "plugins": {
                    "title": {"display": True, "text": f"Threat Score: {threat_analysis['threat_score']}/100"},
                    "legend": {"display": False}
                }
            }
        }
        visualizations.append(threat_gauge)
        
        # Event timeline
        if len(security_event.affected_assets) > 0:
            timeline_chart = {
                "type": "bar",
                "title": "Affected Systems",
                "data": {
                    "labels": security_event.affected_assets[:10],  # Limit to 10 for readability
                    "datasets": [{
                        "label": "Impact Level",
                        "data": [security_event.severity.value == "critical" and 4 or 
                                security_event.severity.value == "high" and 3 or
                                security_event.severity.value == "medium" and 2 or 1] * len(security_event.affected_assets[:10]),
                        "backgroundColor": "#FF6B6B",
                        "borderColor": "#FF5252",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": "Systems Impact Assessment"}},
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "max": 4,
                            "ticks": {
                                "stepSize": 1,
                                "callback": "function(value) { const levels = ['', 'Low', 'Medium', 'High', 'Critical']; return levels[value]; }"
                            }
                        }
                    }
                }
            }
            visualizations.append(timeline_chart)
        
        # MITRE ATT&CK heatmap (simplified)
        tactics = threat_analysis['ttp_analysis']['tactics']
        if tactics:
            mitre_chart = {
                "type": "doughnut",
                "title": "MITRE ATT&CK Tactics",
                "data": {
                    "labels": tactics,
                    "datasets": [{
                        "data": [1] * len(tactics),  # Equal distribution for detected tactics
                        "backgroundColor": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
                        "borderWidth": 2,
                        "borderColor": "#FFFFFF"
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {"display": True, "text": "Detected Attack Tactics"},
                        "legend": {"position": "bottom"}
                    }
                }
            }
            visualizations.append(mitre_chart)
        
        return visualizations
    
    async def _generate_incident_recommendations(self, security_event: SecurityEvent, 
                                               impact_assessment: Dict[str, Any]) -> List[str]:
        """Generate incident response recommendations."""
        
        recommendations = []
        
        # Immediate response recommendations
        if security_event.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            recommendations.append("🚨 Activate incident response team immediately")
            recommendations.append("🔒 Implement emergency containment measures for affected systems")
            recommendations.append("📞 Notify executive leadership and legal team")
        
        # Event-specific recommendations
        if security_event.event_type == SecurityEventType.DATA_BREACH:
            recommendations.append("🔐 Conduct immediate data impact assessment")
            recommendations.append("📋 Prepare breach notification procedures")
            recommendations.append("🛡️ Enhance data loss prevention controls")
        
        elif security_event.event_type == SecurityEventType.MALWARE_DETECTION:
            recommendations.append("🦠 Isolate infected systems from network")
            recommendations.append("🔍 Perform full malware analysis and signature creation")
            recommendations.append("📡 Update endpoint detection and response rules")
        
        elif security_event.event_type == SecurityEventType.NETWORK_INTRUSION:
            recommendations.append("🌐 Review and update network segmentation")
            recommendations.append("🔍 Conduct network traffic analysis")
            recommendations.append("🛡️ Strengthen perimeter security controls")
        
        # General security improvements
        recommendations.append("📊 Enhance monitoring and detection capabilities")
        recommendations.append("🎓 Provide targeted security awareness training")
        recommendations.append("🔄 Review and update incident response procedures")
        
        # Compliance recommendations
        if security_event.compliance_frameworks:
            recommendations.append("📋 Conduct compliance impact assessment")
            recommendations.append("📝 Prepare regulatory reporting documentation")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    async def _create_incident_alerts(self, security_event: SecurityEvent) -> List[Dict[str, Any]]:
        """Create alerts for security incident."""
        
        alerts = []
        
        # High-severity incident alert
        if security_event.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            alerts.append({
                "type": "high_severity_incident",
                "severity": security_event.severity.value,
                "title": f"{security_event.event_type.value.replace('_', ' ').title()} Detected",
                "message": f"High-severity security incident requiring immediate attention",
                "affected_systems": len(security_event.affected_assets),
                "recommended_action": "Activate incident response team",
                "escalation_required": True
            })
        
        # Compliance alert
        if security_event.compliance_frameworks:
            alerts.append({
                "type": "compliance_impact",
                "severity": "medium",
                "title": "Potential Compliance Violation",
                "message": f"Incident may impact {', '.join([f.value.upper() for f in security_event.compliance_frameworks])} compliance",
                "frameworks": [f.value for f in security_event.compliance_frameworks],
                "recommended_action": "Review compliance requirements and prepare documentation"
            })
        
        # Data breach alert
        if security_event.event_type == SecurityEventType.DATA_BREACH:
            alerts.append({
                "type": "data_breach",
                "severity": "critical",
                "title": "Data Breach Detected",
                "message": "Potential unauthorized access to sensitive data",
                "recommended_action": "Activate data breach response procedures",
                "legal_notification_required": True
            })
        
        return alerts
    
    async def _format_executive_summary(self, security_event: SecurityEvent, 
                                      impact_assessment: Dict[str, Any]) -> str:
        """Format executive summary for security incident."""
        
        severity_icon = self.get_severity_icon(security_event.severity)
        event_title = security_event.event_type.value.replace('_', ' ').title()
        
        summary = f"""
## {severity_icon} Executive Summary - {event_title}

**Incident ID**: {security_event.event_id}  
**Severity**: {security_event.severity.value.title()}  
**Detection Time**: {security_event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Business Impact**: {impact_assessment['business_impact']}

### Key Points

• **Threat Level**: {impact_assessment['impact_score']}/100
• **Affected Systems**: {len(security_event.affected_assets)} systems impacted
• **Financial Impact**: ${impact_assessment['financial_impact']:,} estimated
• **Response Status**: {'Active incident response' if security_event.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH] else 'Monitoring and analysis'}

### Immediate Actions Required

"""
        
        if security_event.severity == SeverityLevel.CRITICAL:
            summary += "• **Critical**: Immediate C-suite notification and emergency response activation\n"
            summary += "• **Critical**: All affected systems must be isolated within 30 minutes\n"
        elif security_event.severity == SeverityLevel.HIGH:
            summary += "• **High**: Incident response team activation within 1 hour\n"
            summary += "• **High**: Stakeholder notification and impact assessment\n"
        else:
            summary += "• **Standard**: Continue monitoring and analysis procedures\n"
            summary += "• **Standard**: Document lessons learned and improve controls\n"
        
        if security_event.compliance_frameworks:
            summary += f"• **Compliance**: Review {', '.join([f.value.upper() for f in security_event.compliance_frameworks])} requirements\n"
        
        return summary.strip()
    
    async def _format_technical_details(self, security_event: SecurityEvent, 
                                       threat_analysis: Dict[str, Any]) -> str:
        """Format technical details for security incident."""
        
        details = f"""
## 🔍 Technical Analysis

### Event Details
**Event Type**: {security_event.event_type.value.replace('_', ' ').title()}  
**Source IP**: {self.anonymize_ip(security_event.source_ip)}  
**Target System**: {security_event.target_system}  
**User Account**: {security_event.user_id or 'N/A'}  

### Threat Intelligence
**Threat Score**: {threat_analysis['threat_score']}/100  
**Geographic Origin**: {threat_analysis['geographic_origin']}  
**Threat Actors**: {', '.join(threat_analysis['threat_actors']) if threat_analysis['threat_actors'] else 'Unknown'}

### MITRE ATT&CK Framework Mapping
**Tactics**: {', '.join(threat_analysis['ttp_analysis']['tactics'])}  
**Techniques**: {', '.join(threat_analysis['ttp_analysis']['techniques'])}

### Indicators of Compromise (IOCs)
"""
        
        if security_event.indicators_of_compromise:
            for ioc in security_event.indicators_of_compromise:
                details += f"• `{self.mask_sensitive_data(ioc)}`\n"
        else:
            details += "• No IOCs identified\n"
        
        details += "\n### Affected Assets\n"
        if security_event.affected_assets:
            for asset in security_event.affected_assets:
                details += f"• {asset}\n"
        else:
            details += "• No assets specifically identified\n"
        
        return details.strip()
    
    async def _format_incident_content(self, security_event: SecurityEvent, 
                                     timeline: List[Dict[str, Any]],
                                     impact_assessment: Dict[str, Any]) -> str:
        """Format main incident content."""
        
        event_title = security_event.event_type.value.replace('_', ' ').title()
        severity_icon = self.get_severity_icon(security_event.severity)
        
        content = f"""
# {severity_icon} Security Incident Report - {event_title}

**Incident ID**: {security_event.event_id}  
**Severity**: {security_event.severity.value.title()}  
**Detected**: {security_event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Status**: {'Active' if security_event.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH] else 'Monitoring'}

## 📋 Incident Description

{security_event.description or f"A {event_title.lower()} was detected on {security_event.target_system} from source {self.anonymize_ip(security_event.source_ip)}."}

## ⏱️ Incident Timeline

"""
        
        for event in timeline:
            event_time = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            content += f"**{event_time.strftime('%H:%M:%S')}** - {event['event']}: {event['description']}\n"
        
        content += f"""

## 📊 Impact Assessment

**Overall Impact Score**: {impact_assessment['impact_score']}/100  
**Business Impact**: {impact_assessment['business_impact']}  
**Operational Impact**: {impact_assessment['operational_impact']}  
**Financial Impact**: ${impact_assessment['financial_impact']:,}  
**Response Time**: {impact_assessment['response_time_minutes']} minutes

## 🛡️ Mitigation Actions Taken

"""
        
        if security_event.mitigation_actions:
            for action in security_event.mitigation_actions:
                content += f"• {action}\n"
        else:
            content += "• Mitigation actions in progress\n"
        
        content += """

## 📈 Next Steps

1. **Continue Monitoring**: Enhanced monitoring of affected systems
2. **Forensic Analysis**: Complete digital forensics investigation
3. **Remediation**: Implement security improvements based on findings
4. **Documentation**: Update incident response procedures
5. **Communication**: Provide updates to stakeholders

---
*This report is generated automatically by the Spotify AI Security System. For questions, contact the Security Operations Center.*
        """
        
        return content.strip()


class VulnerabilityFormatter(BaseSecurityFormatter):
    """Formatter for vulnerability assessments and penetration testing results."""
    
    async def format_security_report(self, vulnerability: VulnerabilityAssessment) -> FormattedSecurityReport:
        """Format vulnerability assessment report."""
        
        # Analyze vulnerability risk
        risk_analysis = await self._analyze_vulnerability_risk(vulnerability)
        
        # Create remediation plan
        remediation_plan = await self._create_remediation_plan(vulnerability, risk_analysis)
        
        # Generate visualizations
        visualizations = await self._create_vulnerability_visualizations(vulnerability, risk_analysis)
        
        # Create recommendations
        recommendations = await self._generate_vulnerability_recommendations(vulnerability, risk_analysis)
        
        # Format executive summary
        executive_summary = await self._format_vulnerability_executive_summary(vulnerability, risk_analysis)
        
        # Format technical details
        technical_details = await self._format_vulnerability_technical_details(vulnerability)
        
        # Format main content
        content = await self._format_vulnerability_content(vulnerability, risk_analysis, remediation_plan)
        
        alerts = []
        if vulnerability.cvss_score >= 7.0:
            alerts.append({
                "type": "high_severity_vulnerability",
                "severity": "high",
                "message": f"High-severity vulnerability (CVSS {vulnerability.cvss_score}) requires immediate attention",
                "recommended_action": "Prioritize patching within 72 hours"
            })
        
        metrics = {
            "cvss_score": vulnerability.cvss_score,
            "affected_systems_count": len(vulnerability.affected_systems),
            "exploit_available": vulnerability.exploit_available,
            "patch_available": vulnerability.patch_available,
            "risk_score": risk_analysis['risk_score']
        }
        
        metadata = {
            "report_type": "vulnerability_assessment",
            "vulnerability_id": vulnerability.vulnerability_id,
            "cve_id": vulnerability.cve_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tenant_id": self.tenant_id
        }
        
        return FormattedSecurityReport(
            report_type="vulnerability_assessment",
            content=content,
            executive_summary=executive_summary,
            technical_details=technical_details,
            recommendations=recommendations,
            visualizations=visualizations,
            metrics=metrics,
            alerts=alerts,
            metadata=metadata
        )
    
    async def _analyze_vulnerability_risk(self, vulnerability: VulnerabilityAssessment) -> Dict[str, Any]:
        """Analyze vulnerability risk factors."""
        
        risk_analysis = {
            "risk_score": 0,
            "severity_level": "Low",
            "exploitability": "Low",
            "business_risk": "Low",
            "remediation_urgency": "Standard",
            "risk_factors": []
        }
        
        # Base risk from CVSS score
        cvss_score = vulnerability.cvss_score
        severity_level, _ = self.get_cvss_severity(cvss_score)
        risk_analysis['severity_level'] = severity_level
        
        # Calculate risk score (0-100)
        base_risk = min(100, int(cvss_score * 10))
        
        # Adjust for exploit availability
        if vulnerability.exploit_available:
            base_risk += 15
            risk_analysis['exploitability'] = "High"
            risk_analysis['risk_factors'].append("Public exploit available")
        
        # Adjust for patch availability
        if not vulnerability.patch_available:
            base_risk += 10
            risk_analysis['risk_factors'].append("No patch available")
        
        # Adjust for affected systems count
        system_count = len(vulnerability.affected_systems)
        if system_count > 10:
            base_risk += 15
            risk_analysis['risk_factors'].append(f"High system exposure ({system_count} systems)")
        elif system_count > 5:
            base_risk += 10
            risk_analysis['risk_factors'].append(f"Medium system exposure ({system_count} systems)")
        
        # Business impact assessment
        if vulnerability.business_impact == "high":
            base_risk += 20
            risk_analysis['business_risk'] = "High"
            risk_analysis['risk_factors'].append("High business impact")
        elif vulnerability.business_impact == "medium":
            base_risk += 10
            risk_analysis['business_risk'] = "Medium"
        
        risk_analysis['risk_score'] = min(100, base_risk)
        
        # Determine remediation urgency
        if risk_analysis['risk_score'] >= 80:
            risk_analysis['remediation_urgency'] = "Critical (24-48 hours)"
        elif risk_analysis['risk_score'] >= 60:
            risk_analysis['remediation_urgency'] = "High (1 week)"
        elif risk_analysis['risk_score'] >= 40:
            risk_analysis['remediation_urgency'] = "Medium (1 month)"
        else:
            risk_analysis['remediation_urgency'] = "Standard (next maintenance window)"
        
        return risk_analysis
    
    async def _create_remediation_plan(self, vulnerability: VulnerabilityAssessment, 
                                     risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create remediation plan for vulnerability."""
        
        plan = {
            "immediate_actions": [],
            "short_term_actions": [],
            "long_term_actions": [],
            "timeline": {},
            "resources_required": [],
            "risk_mitigation": []
        }
        
        # Immediate actions for high-risk vulnerabilities
        if risk_analysis['risk_score'] >= 70:
            plan['immediate_actions'] = [
                "Isolate affected systems if possible",
                "Apply emergency patches if available",
                "Implement compensating controls",
                "Monitor affected systems closely"
            ]
        
        # Short-term remediation
        if vulnerability.patch_available:
            plan['short_term_actions'].append("Apply vendor patches during maintenance window")
        else:
            plan['short_term_actions'].append("Implement workarounds and compensating controls")
        
        plan['short_term_actions'].extend([
            "Update vulnerability scanning configurations",
            "Verify remediation effectiveness",
            "Update security documentation"
        ])
        
        # Long-term actions
        plan['long_term_actions'] = [
            "Review and improve vulnerability management processes",
            "Enhance security awareness training",
            "Implement additional security controls",
            "Regular security assessments"
        ]
        
        # Timeline based on urgency
        urgency = risk_analysis['remediation_urgency']
        if "Critical" in urgency:
            plan['timeline'] = {
                "immediate": "0-48 hours",
                "short_term": "1 week",
                "long_term": "1 month"
            }
        elif "High" in urgency:
            plan['timeline'] = {
                "immediate": "1 week",
                "short_term": "2 weeks",
                "long_term": "1 month"
            }
        else:
            plan['timeline'] = {
                "immediate": "Next maintenance window",
                "short_term": "1 month",
                "long_term": "3 months"
            }
        
        # Resources required
        plan['resources_required'] = [
            "System administrators for patching",
            "Network team for firewall changes",
            "Security team for validation",
            "Change management approval"
        ]
        
        return plan
    
    async def _create_vulnerability_visualizations(self, vulnerability: VulnerabilityAssessment,
                                                 risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create visualizations for vulnerability assessment."""
        
        visualizations = []
        
        # CVSS score gauge
        cvss_gauge = {
            "type": "gauge",
            "title": "CVSS Score",
            "data": {
                "datasets": [{
                    "data": [vulnerability.cvss_score, 10 - vulnerability.cvss_score],
                    "backgroundColor": ["#FF6B6B", "#E0E0E0"],
                    "borderWidth": 0
                }]
            },
            "options": {
                "responsive": True,
                "circumference": 180,
                "rotation": 270,
                "plugins": {
                    "title": {"display": True, "text": f"CVSS Score: {vulnerability.cvss_score}/10"},
                    "legend": {"display": False}
                }
            }
        }
        visualizations.append(cvss_gauge)
        
        # Risk factors chart
        if risk_analysis['risk_factors']:
            risk_chart = {
                "type": "doughnut",
                "title": "Risk Factors",
                "data": {
                    "labels": risk_analysis['risk_factors'],
                    "datasets": [{
                        "data": [1] * len(risk_analysis['risk_factors']),
                        "backgroundColor": ["#FF6B6B", "#FFA726", "#FFCA28", "#66BB6A", "#42A5F5"],
                        "borderWidth": 2,
                        "borderColor": "#FFFFFF"
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {"display": True, "text": "Identified Risk Factors"},
                        "legend": {"position": "bottom"}
                    }
                }
            }
            visualizations.append(risk_chart)
        
        return visualizations
    
    async def _generate_vulnerability_recommendations(self, vulnerability: VulnerabilityAssessment,
                                                    risk_analysis: Dict[str, Any]) -> List[str]:
        """Generate vulnerability remediation recommendations."""
        
        recommendations = []
        
        # Priority recommendations based on CVSS score
        if vulnerability.cvss_score >= 9.0:
            recommendations.append("🚨 Critical vulnerability - apply emergency patches immediately")
            recommendations.append("🔒 Isolate affected systems until patched")
        elif vulnerability.cvss_score >= 7.0:
            recommendations.append("🔴 High-severity vulnerability - prioritize patching within 72 hours")
            recommendations.append("🛡️ Implement compensating controls if immediate patching not possible")
        
        # Specific remediation recommendations
        if vulnerability.patch_available:
            recommendations.append("📦 Apply vendor-provided patches during next maintenance window")
        else:
            recommendations.append("🔧 Implement workarounds until official patches are available")
        
        # Exploit-specific recommendations
        if vulnerability.exploit_available:
            recommendations.append("⚠️ Public exploits available - monitor for exploitation attempts")
            recommendations.append("📊 Enhance monitoring and detection for this vulnerability")
        
        # System-specific recommendations
        system_count = len(vulnerability.affected_systems)
        if system_count > 5:
            recommendations.append(f"🏢 Multiple systems affected ({system_count}) - coordinate patching effort")
        
        # General security improvements
        recommendations.append("🔍 Review vulnerability management processes")
        recommendations.append("📋 Update security baselines and hardening standards")
        recommendations.append("🎓 Provide targeted security training to relevant teams")
        
        return recommendations[:6]
    
    async def _format_vulnerability_executive_summary(self, vulnerability: VulnerabilityAssessment,
                                                    risk_analysis: Dict[str, Any]) -> str:
        """Format executive summary for vulnerability."""
        
        severity_level, severity_icon = self.get_cvss_severity(vulnerability.cvss_score)
        
        summary = f"""
## {severity_icon} Vulnerability Assessment Summary

**Vulnerability ID**: {vulnerability.vulnerability_id}  
**CVE ID**: {vulnerability.cve_id or 'Not assigned'}  
**CVSS Score**: {vulnerability.cvss_score}/10 ({severity_level})  
**Risk Score**: {risk_analysis['risk_score']}/100  
**Remediation Urgency**: {risk_analysis['remediation_urgency']}

### Business Impact

• **Affected Systems**: {len(vulnerability.affected_systems)} systems at risk
• **Business Impact Level**: {vulnerability.business_impact.title()}
• **Exploit Available**: {'Yes - High Risk' if vulnerability.exploit_available else 'No'}
• **Patch Available**: {'Yes' if vulnerability.patch_available else 'No - Workarounds needed'}

### Key Risk Factors

"""
        
        if risk_analysis['risk_factors']:
            for factor in risk_analysis['risk_factors']:
                summary += f"• {factor}\n"
        else:
            summary += "• Standard vulnerability risk profile\n"
        
        summary += f"""

### Immediate Actions Required

"""
        
        if vulnerability.cvss_score >= 9.0:
            summary += "• **Critical**: Emergency patching within 24 hours\n"
            summary += "• **Critical**: Isolate affected systems immediately\n"
        elif vulnerability.cvss_score >= 7.0:
            summary += "• **High**: Prioritize patching within 72 hours\n"
            summary += "• **High**: Implement compensating controls\n"
        else:
            summary += "• **Standard**: Include in next maintenance cycle\n"
            summary += "• **Standard**: Continue monitoring for exploitation\n"
        
        return summary.strip()
    
    async def _format_vulnerability_technical_details(self, vulnerability: VulnerabilityAssessment) -> str:
        """Format technical details for vulnerability."""
        
        details = f"""
## 🔍 Technical Details

### Vulnerability Information
**Vulnerability ID**: {vulnerability.vulnerability_id}  
**CVE ID**: {vulnerability.cve_id or 'Not assigned'}  
**CVSS Score**: {vulnerability.cvss_score}/10  
**Discovery Date**: {vulnerability.discovery_date.strftime('%Y-%m-%d') if vulnerability.discovery_date else 'Unknown'}

### Description
{vulnerability.description or 'No detailed description available.'}

### Impact Assessment
{vulnerability.impact_assessment or 'Impact assessment in progress.'}

### Affected Systems
"""
        
        if vulnerability.affected_systems:
            for system in vulnerability.affected_systems:
                details += f"• {system}\n"
        else:
            details += "• No specific systems identified\n"
        
        details += "\n### Remediation Steps\n"
        if vulnerability.remediation_steps:
            for i, step in enumerate(vulnerability.remediation_steps, 1):
                details += f"{i}. {step}\n"
        else:
            details += "1. Remediation steps being developed\n"
        
        return details.strip()
    
    async def _format_vulnerability_content(self, vulnerability: VulnerabilityAssessment,
                                          risk_analysis: Dict[str, Any],
                                          remediation_plan: Dict[str, Any]) -> str:
        """Format main vulnerability content."""
        
        severity_level, severity_icon = self.get_cvss_severity(vulnerability.cvss_score)
        
        content = f"""
# {severity_icon} Vulnerability Report - {vulnerability.vulnerability_id}

**CVE ID**: {vulnerability.cve_id or 'Pending assignment'}  
**CVSS Score**: {vulnerability.cvss_score}/10 ({severity_level})  
**Assessment Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}  
**Business Impact**: {vulnerability.business_impact.title()}

## 📋 Vulnerability Summary

{vulnerability.description or 'A security vulnerability has been identified that requires attention.'}

**Key Details:**
• **Affected Systems**: {len(vulnerability.affected_systems)} systems
• **Public Exploit**: {'Available' if vulnerability.exploit_available else 'Not available'}
• **Vendor Patch**: {'Available' if vulnerability.patch_available else 'Not available'}
• **Risk Score**: {risk_analysis['risk_score']}/100

## 🎯 Remediation Plan

### Immediate Actions ({remediation_plan['timeline'].get('immediate', 'TBD')})
"""
        
        for action in remediation_plan['immediate_actions']:
            content += f"• {action}\n"
        
        content += f"\n### Short-term Actions ({remediation_plan['timeline'].get('short_term', 'TBD')})\n"
        for action in remediation_plan['short_term_actions']:
            content += f"• {action}\n"
        
        content += f"\n### Long-term Actions ({remediation_plan['timeline'].get('long_term', 'TBD')})\n"
        for action in remediation_plan['long_term_actions']:
            content += f"• {action}\n"
        
        content += """

## 📊 Risk Assessment

"""
        content += f"**Overall Risk**: {risk_analysis['risk_score']}/100\n"
        content += f"**Remediation Urgency**: {risk_analysis['remediation_urgency']}\n"
        content += f"**Business Risk**: {risk_analysis['business_risk']}\n"
        content += f"**Exploitability**: {risk_analysis['exploitability']}\n"
        
        content += """

## 👥 Required Resources

"""
        for resource in remediation_plan['resources_required']:
            content += f"• {resource}\n"
        
        content += """

---
*This vulnerability assessment was generated by the Spotify AI Security System. For technical questions, contact the Security Engineering team.*
        """
        
        return content.strip()


# Factory function for creating security formatters
def create_security_formatter(
    formatter_type: str,
    tenant_id: str,
    config: Optional[Dict[str, Any]] = None
) -> BaseSecurityFormatter:
    """
    Factory function to create security formatters.
    
    Args:
        formatter_type: Type of formatter ('incident', 'vulnerability', 'compliance')
        tenant_id: Tenant identifier
        config: Configuration dictionary
        
    Returns:
        Configured security formatter instance
    """
    formatters = {
        'incident': SecurityIncidentFormatter,
        'security_incident': SecurityIncidentFormatter,
        'threat': SecurityIncidentFormatter,
        'vulnerability': VulnerabilityFormatter,
        'vuln': VulnerabilityFormatter,
        'pentest': VulnerabilityFormatter,
        'security': SecurityIncidentFormatter
    }
    
    if formatter_type not in formatters:
        raise ValueError(f"Unsupported security formatter type: {formatter_type}")
    
    formatter_class = formatters[formatter_type]
    return formatter_class(tenant_id, config or {})
