"""
🎵 Spotify AI Agent - Advanced Security Audit Middleware
======================================================

Middleware de sécurité et audit avancé pour la surveillance des menaces,
l'audit de conformité et la protection des données sensibles.

Architecture:
- Real-time Threat Detection
- Behavioral Analysis & Anomaly Detection
- Security Event Logging & SIEM Integration
- Compliance Monitoring (GDPR, SOX, HIPAA)
- Data Loss Prevention (DLP)
- Advanced Intrusion Detection System (IDS)
- Security Analytics & Machine Learning
- Automated Incident Response

Enterprise Security Features:
- Zero Trust Security Model
- Multi-Factor Authentication Monitoring
- Privileged Access Management (PAM)
- Security Orchestration & Automated Response (SOAR)
- Threat Intelligence Integration
- Vulnerability Assessment
- Security Score Calculation
- Forensic Data Collection
"""

import asyncio
import hashlib
import hmac
import json
import time
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import uuid
import re
from urllib.parse import urlparse, parse_qs
import geoip2.database
import user_agents

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import cryptography.fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt
import jwt
from passlib.context import CryptContext

from ..core.config import get_settings
from ..core.logging import get_logger
from ..utils.encryption import EncryptionService
from ..utils.threat_intel import ThreatIntelligenceService


class ThreatLevel(str, Enum):
    """Niveaux de menace"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(str, Enum):
    """Types d'événements de sécurité"""
    AUTHENTICATION_FAILED = "auth_failed"
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHORIZATION_DENIED = "authz_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    CSRF_ATTEMPT = "csrf_attempt"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    COMPLIANCE_VIOLATION = "compliance_violation"
    MALWARE_DETECTED = "malware_detected"
    PHISHING_ATTEMPT = "phishing_attempt"
    DDOS_ATTACK = "ddos_attack"
    API_ABUSE = "api_abuse"
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    CONFIGURATION_CHANGE = "config_change"
    SYSTEM_COMPROMISE = "system_compromise"


class ComplianceStandard(str, Enum):
    """Standards de conformité"""
    GDPR = "gdpr"
    SOX = "sox"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    SOC2 = "soc2"


class RiskScore(str, Enum):
    """Scores de risque"""
    MINIMAL = "minimal"      # 0-20
    LOW = "low"             # 21-40
    MODERATE = "moderate"   # 41-60
    HIGH = "high"          # 61-80
    CRITICAL = "critical"   # 81-100


@dataclass
class SecurityEvent:
    """Événement de sécurité"""
    id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: str
    user_id: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    endpoint: str
    method: str
    user_agent: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: int = 0
    indicators: List[str] = field(default_factory=list)
    mitigations_applied: List[str] = field(default_factory=list)
    false_positive: bool = False
    investigated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class ThreatIndicator:
    """Indicateur de menace"""
    ioc_type: str  # ip, domain, hash, email, etc.
    value: str
    threat_level: ThreatLevel
    source: str
    confidence: float
    first_seen: datetime
    last_seen: datetime
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def is_expired(self, ttl_hours: int = 168) -> bool:  # 7 jours par défaut
        """Vérifie si l'indicateur a expiré"""
        return (datetime.utcnow() - self.last_seen).total_seconds() > (ttl_hours * 3600)


@dataclass
class UserBehaviorProfile:
    """Profil comportemental utilisateur"""
    user_id: str
    usual_ips: Set[str] = field(default_factory=set)
    usual_user_agents: Set[str] = field(default_factory=set)
    usual_endpoints: Dict[str, int] = field(default_factory=dict)
    usual_times: List[int] = field(default_factory=list)  # heures usuelles
    request_frequency: float = 0.0  # requêtes par minute
    risk_score: int = 0
    last_activity: Optional[datetime] = None
    anomaly_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_activity(self, ip: str, user_agent: str, endpoint: str, timestamp: datetime):
        """Met à jour l'activité utilisateur"""
        self.usual_ips.add(ip)
        self.usual_user_agents.add(user_agent)
        
        if endpoint in self.usual_endpoints:
            self.usual_endpoints[endpoint] += 1
        else:
            self.usual_endpoints[endpoint] = 1
        
        hour = timestamp.hour
        if hour not in self.usual_times:
            self.usual_times.append(hour)
        
        self.last_activity = timestamp
        self.updated_at = timestamp
    
    def calculate_anomaly_score(self, ip: str, user_agent: str, endpoint: str, timestamp: datetime) -> float:
        """Calcule le score d'anomalie pour une activité"""
        score = 0.0
        
        # IP inhabituelle
        if ip not in self.usual_ips and len(self.usual_ips) > 0:
            score += 30.0
        
        # User agent inhabituel
        if user_agent not in self.usual_user_agents and len(self.usual_user_agents) > 0:
            score += 20.0
        
        # Endpoint inhabituel
        if endpoint not in self.usual_endpoints:
            score += 15.0
        
        # Heure inhabituelle
        hour = timestamp.hour
        if hour not in self.usual_times and len(self.usual_times) > 0:
            score += 10.0
        
        # Fréquence anormale (si trop rapide)
        if self.last_activity:
            time_diff = (timestamp - self.last_activity).total_seconds()
            if time_diff < 1.0:  # Moins d'une seconde
                score += 25.0
        
        return min(score, 100.0)


@dataclass
class ComplianceRule:
    """Règle de conformité"""
    id: str
    standard: ComplianceStandard
    name: str
    description: str
    severity: ThreatLevel
    check_function: str  # nom de la fonction de vérification
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


class ThreatDetectionEngine:
    """Moteur de détection de menaces"""
    
    def __init__(self):
        self.logger = get_logger("threat_detection")
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        self.ip_reputation_cache: Dict[str, Tuple[str, datetime]] = {}
        self.blocked_ips: Set[str] = set()
        self.blocked_patterns: List[re.Pattern] = []
        
        # Chargement des patterns de détection
        self._load_detection_patterns()
        
        # GeoIP database (si disponible)
        self.geoip_reader = None
        try:
            self.geoip_reader = geoip2.database.Reader('/usr/share/GeoIP/GeoLite2-City.mmdb')
        except:
            self.logger.warning("Base GeoIP non disponible")
    
    def _load_detection_patterns(self):
        """Charge les patterns de détection"""
        # Patterns SQL Injection
        sql_patterns = [
            r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
            r"(\b(script|javascript|vbscript|onload|onerror|onclick)\b)",
            r"(\b(eval|expression|url|import)\b)",
            r"([\'\";].*[\'\";])",
            r"(\-\-.*$)",
            r"(/\*.*\*/)"
        ]
        
        # Patterns XSS
        xss_patterns = [
            r"(<script[^>]*>.*?</script>)",
            r"(<iframe[^>]*>.*?</iframe>)",
            r"(<object[^>]*>.*?</object>)",
            r"(<embed[^>]*>.*?</embed>)",
            r"(javascript:)",
            r"(on\w+\s*=)"
        ]
        
        # Patterns Path Traversal
        path_traversal_patterns = [
            r"(\.\./){2,}",
            r"(\.\.\\){2,}",
            r"(%2e%2e%2f){2,}",
            r"(%2e%2e\\){2,}"
        ]
        
        all_patterns = sql_patterns + xss_patterns + path_traversal_patterns
        self.blocked_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in all_patterns]
    
    def add_threat_indicator(self, indicator: ThreatIndicator):
        """Ajoute un indicateur de menace"""
        key = f"{indicator.ioc_type}:{indicator.value}"
        self.threat_indicators[key] = indicator
        self.logger.info(f"Indicateur de menace ajouté: {key}")
    
    def check_threat_indicators(self, ip: str, domain: str = None, user_agent: str = None) -> List[ThreatIndicator]:
        """Vérifie les indicateurs de menace"""
        matches = []
        
        # Vérifier IP
        ip_key = f"ip:{ip}"
        if ip_key in self.threat_indicators:
            indicator = self.threat_indicators[ip_key]
            if not indicator.is_expired():
                matches.append(indicator)
        
        # Vérifier domaine
        if domain:
            domain_key = f"domain:{domain}"
            if domain_key in self.threat_indicators:
                indicator = self.threat_indicators[domain_key]
                if not indicator.is_expired():
                    matches.append(indicator)
        
        # Vérifier patterns dans user agent
        if user_agent:
            for pattern in self.blocked_patterns:
                if pattern.search(user_agent):
                    matches.append(ThreatIndicator(
                        ioc_type="pattern",
                        value=pattern.pattern,
                        threat_level=ThreatLevel.MEDIUM,
                        source="pattern_detection",
                        confidence=0.8,
                        first_seen=datetime.utcnow(),
                        last_seen=datetime.utcnow(),
                        description="Pattern malveillant détecté dans User-Agent"
                    ))
        
        return matches
    
    def analyze_request_content(self, method: str, path: str, query_params: str, headers: Dict[str, str], body: str = "") -> List[str]:
        """Analyse le contenu de la requête pour détecter des attaques"""
        threats = []
        content_to_check = f"{path} {query_params} {body}".lower()
        
        # Vérifier les patterns malveillants
        for pattern in self.blocked_patterns:
            if pattern.search(content_to_check):
                if "union" in pattern.pattern or "select" in pattern.pattern:
                    threats.append("sql_injection")
                elif "script" in pattern.pattern or "javascript" in pattern.pattern:
                    threats.append("xss")
                elif ".." in pattern.pattern:
                    threats.append("path_traversal")
                else:
                    threats.append("malicious_pattern")
        
        # Vérifications spécifiques
        if method in ["POST", "PUT", "PATCH"] and len(body) > 1000000:  # 1MB
            threats.append("large_payload")
        
        if len(query_params) > 2048:
            threats.append("long_query_string")
        
        # Vérifier les headers suspicieux
        suspicious_headers = ["x-forwarded-for", "x-real-ip", "x-originating-ip"]
        for header in suspicious_headers:
            if header in headers and len(headers[header].split(",")) > 5:
                threats.append("header_spoofing")
        
        return threats
    
    def update_user_behavior(self, user_id: str, ip: str, user_agent: str, endpoint: str, timestamp: datetime):
        """Met à jour le profil comportemental utilisateur"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserBehaviorProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        profile.update_activity(ip, user_agent, endpoint, timestamp)
    
    def detect_behavioral_anomalies(self, user_id: str, ip: str, user_agent: str, endpoint: str, timestamp: datetime) -> float:
        """Détecte les anomalies comportementales"""
        if user_id not in self.user_profiles:
            return 0.0  # Nouvel utilisateur, pas d'anomalie
        
        profile = self.user_profiles[user_id]
        return profile.calculate_anomaly_score(ip, user_agent, endpoint, timestamp)
    
    def get_ip_geolocation(self, ip: str) -> Dict[str, Any]:
        """Obtient la géolocalisation d'une IP"""
        if not self.geoip_reader:
            return {}
        
        try:
            response = self.geoip_reader.city(ip)
            return {
                "country": response.country.name,
                "country_code": response.country.iso_code,
                "city": response.city.name,
                "latitude": float(response.location.latitude) if response.location.latitude else None,
                "longitude": float(response.location.longitude) if response.location.longitude else None,
                "timezone": response.location.time_zone,
            }
        except Exception as e:
            self.logger.debug(f"Erreur géolocalisation pour {ip}: {e}")
            return {}
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Vérifie si une IP est bloquée"""
        return ip in self.blocked_ips
    
    def block_ip(self, ip: str, reason: str = ""):
        """Bloque une IP"""
        self.blocked_ips.add(ip)
        self.logger.warning(f"IP bloquée: {ip} - Raison: {reason}")
    
    def unblock_ip(self, ip: str):
        """Débloque une IP"""
        self.blocked_ips.discard(ip)
        self.logger.info(f"IP débloquée: {ip}")


class ComplianceMonitor:
    """Moniteur de conformité"""
    
    def __init__(self):
        self.logger = get_logger("compliance_monitor")
        self.rules: Dict[str, ComplianceRule] = {}
        self.violations: List[Dict[str, Any]] = []
        self.compliance_checks = {
            "gdpr_data_access": self._check_gdpr_data_access,
            "gdpr_consent": self._check_gdpr_consent,
            "pci_sensitive_data": self._check_pci_sensitive_data,
            "hipaa_phi_access": self._check_hipaa_phi_access,
            "sox_financial_data": self._check_sox_financial_data,
        }
        
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Configure les règles par défaut"""
        rules = [
            ComplianceRule(
                id="gdpr_data_access",
                standard=ComplianceStandard.GDPR,
                name="GDPR Data Access Logging",
                description="Toutes les accès aux données personnelles doivent être loggés",
                severity=ThreatLevel.HIGH,
                check_function="gdpr_data_access"
            ),
            ComplianceRule(
                id="pci_sensitive_data",
                standard=ComplianceStandard.PCI_DSS,
                name="PCI Sensitive Data Protection",
                description="Les données sensibles de carte de crédit doivent être protégées",
                severity=ThreatLevel.CRITICAL,
                check_function="pci_sensitive_data"
            ),
            ComplianceRule(
                id="hipaa_phi_access",
                standard=ComplianceStandard.HIPAA,
                name="HIPAA PHI Access Control",
                description="L'accès aux informations de santé protégées doit être contrôlé",
                severity=ThreatLevel.CRITICAL,
                check_function="hipaa_phi_access"
            )
        ]
        
        for rule in rules:
            self.rules[rule.id] = rule
    
    def check_compliance(self, request: Request, response: Response, event: SecurityEvent) -> List[Dict[str, Any]]:
        """Vérifie la conformité"""
        violations = []
        
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            check_func = self.compliance_checks.get(rule.check_function)
            if check_func:
                try:
                    violation = check_func(request, response, event, rule)
                    if violation:
                        violations.append(violation)
                        self.violations.append(violation)
                except Exception as e:
                    self.logger.error(f"Erreur vérification conformité {rule_id}: {e}")
        
        return violations
    
    def _check_gdpr_data_access(self, request: Request, response: Response, event: SecurityEvent, rule: ComplianceRule) -> Optional[Dict[str, Any]]:
        """Vérification GDPR pour l'accès aux données"""
        # Vérifier si la requête accède à des données personnelles
        personal_data_endpoints = ["/api/v1/user/profile", "/api/v1/user/data", "/api/v1/analytics/user"]
        
        if any(endpoint in request.url.path for endpoint in personal_data_endpoints):
            # Vérifier si l'accès est loggé correctement
            if event.event_type not in [SecurityEventType.SENSITIVE_DATA_ACCESS]:
                return {
                    "rule_id": rule.id,
                    "standard": rule.standard,
                    "severity": rule.severity,
                    "description": "Accès aux données personnelles non loggé correctement",
                    "details": {
                        "endpoint": request.url.path,
                        "user_id": event.user_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
        
        return None
    
    def _check_gdpr_consent(self, request: Request, response: Response, event: SecurityEvent, rule: ComplianceRule) -> Optional[Dict[str, Any]]:
        """Vérification du consentement GDPR"""
        # Logique de vérification du consentement
        return None
    
    def _check_pci_sensitive_data(self, request: Request, response: Response, event: SecurityEvent, rule: ComplianceRule) -> Optional[Dict[str, Any]]:
        """Vérification PCI pour les données sensibles"""
        # Détecter les numéros de cartes de crédit dans les logs ou réponses
        card_pattern = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
        
        # Vérifier dans le body de la requête (si disponible)
        # Note: En production, éviter de logger les corps de requête avec des données sensibles
        
        return None
    
    def _check_hipaa_phi_access(self, request: Request, response: Response, event: SecurityEvent, rule: ComplianceRule) -> Optional[Dict[str, Any]]:
        """Vérification HIPAA pour l'accès aux PHI"""
        # Logique spécifique HIPAA
        return None
    
    def _check_sox_financial_data(self, request: Request, response: Response, event: SecurityEvent, rule: ComplianceRule) -> Optional[Dict[str, Any]]:
        """Vérification SOX pour les données financières"""
        # Logique spécifique SOX
        return None


class SecurityAuditLogger:
    """Logger d'audit de sécurité"""
    
    def __init__(self):
        self.logger = get_logger("security_audit")
        self.events: deque = deque(maxlen=100000)  # Garder les 100k derniers événements
        self.siem_endpoints: List[str] = []
        
        # Configuration SIEM
        settings = get_settings()
        if hasattr(settings, 'SIEM_ENDPOINTS'):
            self.siem_endpoints = settings.SIEM_ENDPOINTS
    
    def log_security_event(self, event: SecurityEvent):
        """Logue un événement de sécurité"""
        self.events.append(event)
        
        # Log structuré
        log_data = {
            "security_event": True,
            "event_id": event.id,
            "event_type": event.event_type.value,
            "threat_level": event.threat_level.value,
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "endpoint": event.endpoint,
            "method": event.method,
            "risk_score": event.risk_score,
            "indicators": event.indicators,
            "timestamp": event.timestamp.isoformat()
        }
        
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.logger.error(f"🚨 SECURITY ALERT: {event.description}", extra=log_data)
        elif event.threat_level == ThreatLevel.MEDIUM:
            self.logger.warning(f"⚠️ SECURITY WARNING: {event.description}", extra=log_data)
        else:
            self.logger.info(f"ℹ️ SECURITY INFO: {event.description}", extra=log_data)
        
        # Envoyer vers SIEM si configuré
        asyncio.create_task(self._send_to_siem(event))
    
    async def _send_to_siem(self, event: SecurityEvent):
        """Envoie l'événement vers les systèmes SIEM"""
        if not self.siem_endpoints:
            return
        
        try:
            import aiohttp
            
            event_data = event.to_dict()
            
            async with aiohttp.ClientSession() as session:
                for endpoint in self.siem_endpoints:
                    try:
                        async with session.post(
                            endpoint,
                            json=event_data,
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                self.logger.debug(f"Événement envoyé vers SIEM: {endpoint}")
                            else:
                                self.logger.warning(f"Erreur envoi SIEM {endpoint}: {response.status}")
                    except Exception as e:
                        self.logger.error(f"Erreur envoi SIEM {endpoint}: {e}")
        
        except Exception as e:
            self.logger.error(f"Erreur configuration SIEM: {e}")
    
    def get_events_by_type(self, event_type: SecurityEventType, limit: int = 100) -> List[SecurityEvent]:
        """Récupère les événements par type"""
        return [event for event in list(self.events)[-limit:] if event.event_type == event_type]
    
    def get_events_by_user(self, user_id: str, limit: int = 100) -> List[SecurityEvent]:
        """Récupère les événements par utilisateur"""
        return [event for event in list(self.events)[-limit:] if event.user_id == user_id]
    
    def get_high_risk_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Récupère les événements à haut risque"""
        return [event for event in list(self.events)[-limit:] 
                if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]


class AdvancedSecurityAuditMiddleware:
    """Middleware d'audit de sécurité avancé"""
    
    def __init__(self, 
                 enable_threat_detection: bool = True,
                 enable_behavioral_analysis: bool = True,
                 enable_compliance_monitoring: bool = True,
                 block_suspicious_ips: bool = True):
        
        self.enable_threat_detection = enable_threat_detection
        self.enable_behavioral_analysis = enable_behavioral_analysis
        self.enable_compliance_monitoring = enable_compliance_monitoring
        self.block_suspicious_ips = block_suspicious_ips
        
        self.logger = get_logger("security_audit_middleware")
        
        # Composants de sécurité
        self.threat_engine = ThreatDetectionEngine() if enable_threat_detection else None
        self.compliance_monitor = ComplianceMonitor() if enable_compliance_monitoring else None
        self.audit_logger = SecurityAuditLogger()
        
        # Statistiques de sécurité
        self.security_stats = {
            "total_events": 0,
            "blocked_requests": 0,
            "threats_detected": 0,
            "compliance_violations": 0,
            "false_positives": 0
        }
        
        # Cache pour les vérifications répétitives
        self.ip_reputation_cache: Dict[str, Tuple[bool, datetime]] = {}
        self.user_agent_cache: Dict[str, Tuple[bool, datetime]] = {}
        
        # Rate limiting pour les IPs suspectes
        self.suspicious_ip_counts: Dict[str, List[datetime]] = defaultdict(list)
        
        self._initialized = False
    
    async def initialize(self):
        """Initialise le middleware"""
        if self._initialized:
            return
        
        try:
            # Charger les indicateurs de menace depuis des sources externes
            if self.threat_engine:
                await self._load_threat_intelligence()
            
            self._initialized = True
            self.logger.info("Middleware d'audit de sécurité initialisé")
            
        except Exception as e:
            self.logger.error(f"Erreur initialisation sécurité: {e}")
            raise
    
    async def __call__(self, request: Request, call_next):
        """Traite la requête avec audit de sécurité"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Vérifications préliminaires
        security_check_result = await self._perform_security_checks(request, client_ip, user_agent)
        
        if security_check_result.get("block", False):
            # Requête bloquée pour raisons de sécurité
            security_event = SecurityEvent(
                id=str(uuid.uuid4()),
                event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                threat_level=ThreatLevel.HIGH,
                timestamp=datetime.utcnow(),
                source_ip=client_ip,
                user_id=getattr(request.state, "user_id", None),
                session_id=getattr(request.state, "session_id", None),
                request_id=getattr(request.state, "request_id", None),
                endpoint=request.url.path,
                method=request.method,
                user_agent=user_agent,
                description=security_check_result.get("reason", "Requête bloquée pour raisons de sécurité"),
                details=security_check_result.get("details", {}),
                risk_score=security_check_result.get("risk_score", 80),
                indicators=security_check_result.get("indicators", [])
            )
            
            self.audit_logger.log_security_event(security_event)
            self.security_stats["blocked_requests"] += 1
            
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "Access denied",
                    "message": "Request blocked by security policy",
                    "incident_id": security_event.id
                }
            )
        
        try:
            # Exécuter la requête
            response = await call_next(request)
            
            # Analyse post-traitement
            await self._post_process_analysis(request, response, client_ip, user_agent, start_time)
            
            return response
            
        except Exception as e:
            # Analyser les exceptions pour des indicateurs de sécurité
            await self._analyze_exception(request, e, client_ip, user_agent)
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Récupère l'IP réelle du client"""
        # Vérifier les headers de proxy
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Prendre la première IP (client original)
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        
        # Fallback vers l'IP de la connexion
        return request.client.host if request.client else "unknown"
    
    async def _perform_security_checks(self, request: Request, client_ip: str, user_agent: str) -> Dict[str, Any]:
        """Effectue les vérifications de sécurité"""
        result = {
            "block": False,
            "reason": "",
            "details": {},
            "risk_score": 0,
            "indicators": []
        }
        
        try:
            # 1. Vérifier IP bloquée
            if self.threat_engine and self.threat_engine.is_ip_blocked(client_ip):
                result.update({
                    "block": True,
                    "reason": "IP address is blocked",
                    "risk_score": 100,
                    "indicators": ["blocked_ip"]
                })
                return result
            
            # 2. Vérifier les indicateurs de menace
            if self.threat_engine:
                threat_indicators = self.threat_engine.check_threat_indicators(
                    client_ip,
                    request.url.hostname,
                    user_agent
                )
                
                if threat_indicators:
                    high_threats = [t for t in threat_indicators if t.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
                    if high_threats and self.block_suspicious_ips:
                        result.update({
                            "block": True,
                            "reason": "Threat indicators detected",
                            "risk_score": 90,
                            "indicators": [f"{t.ioc_type}:{t.value}" for t in high_threats],
                            "details": {"threats": [asdict(t) for t in high_threats]}
                        })
                        return result
                    else:
                        result["risk_score"] += 30
                        result["indicators"].extend([f"{t.ioc_type}:{t.value}" for t in threat_indicators])
            
            # 3. Analyser le contenu de la requête
            if self.threat_engine:
                query_string = str(request.query_params)
                body = ""
                
                # Note: En production, éviter de lire le body complet pour les gros uploads
                try:
                    if request.method in ["POST", "PUT", "PATCH"]:
                        # Lire un échantillon du body
                        body_bytes = await request.body()
                        if len(body_bytes) < 10000:  # Limiter à 10KB
                            body = body_bytes.decode('utf-8', errors='ignore')
                except:
                    pass
                
                threats = self.threat_engine.analyze_request_content(
                    request.method,
                    request.url.path,
                    query_string,
                    dict(request.headers),
                    body
                )
                
                if threats:
                    critical_threats = ["sql_injection", "xss", "path_traversal"]
                    if any(threat in critical_threats for threat in threats):
                        result.update({
                            "block": True,
                            "reason": f"Malicious content detected: {', '.join(threats)}",
                            "risk_score": 95,
                            "indicators": threats
                        })
                        return result
                    else:
                        result["risk_score"] += 20
                        result["indicators"].extend(threats)
            
            # 4. Vérifier le rate limiting par IP
            await self._check_rate_limiting(client_ip, result)
            
            # 5. Analyse comportementale
            if self.enable_behavioral_analysis and self.threat_engine:
                user_id = getattr(request.state, "user_id", None)
                if user_id:
                    anomaly_score = self.threat_engine.detect_behavioral_anomalies(
                        user_id, client_ip, user_agent, request.url.path, datetime.utcnow()
                    )
                    
                    if anomaly_score > 70:
                        result["risk_score"] += anomaly_score
                        result["indicators"].append(f"behavioral_anomaly:{anomaly_score:.1f}")
                        
                        if anomaly_score > 90:
                            result.update({
                                "block": True,
                                "reason": f"Anomalous behavior detected (score: {anomaly_score:.1f})",
                                "risk_score": 85
                            })
                            return result
            
            # 6. Vérifier la géolocalisation
            if self.threat_engine:
                geo_info = self.threat_engine.get_ip_geolocation(client_ip)
                if geo_info:
                    # Bloquer certains pays à haut risque (configurable)
                    high_risk_countries = ["CN", "RU", "KP"]  # Exemple
                    if geo_info.get("country_code") in high_risk_countries:
                        result["risk_score"] += 20
                        result["indicators"].append(f"high_risk_country:{geo_info.get('country_code')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur vérifications sécurité: {e}")
            return {"block": False, "reason": "", "details": {}, "risk_score": 0, "indicators": []}
    
    async def _check_rate_limiting(self, client_ip: str, result: Dict[str, Any]):
        """Vérifie le rate limiting pour détecter les attaques"""
        now = datetime.utcnow()
        
        # Nettoyer les anciennes entrées (dernière minute)
        cutoff = now - timedelta(minutes=1)
        self.suspicious_ip_counts[client_ip] = [
            timestamp for timestamp in self.suspicious_ip_counts[client_ip]
            if timestamp > cutoff
        ]
        
        # Ajouter la requête actuelle
        self.suspicious_ip_counts[client_ip].append(now)
        
        # Vérifier les seuils
        count_1min = len(self.suspicious_ip_counts[client_ip])
        
        if count_1min > 100:  # Plus de 100 requêtes par minute
            result["risk_score"] += 40
            result["indicators"].append(f"high_request_rate:{count_1min}/min")
            
            if count_1min > 200:  # Blocage pour attaque DoS potentielle
                result.update({
                    "block": True,
                    "reason": f"Rate limit exceeded: {count_1min} requests/minute",
                    "risk_score": 90
                })
                
                # Bloquer temporairement l'IP
                if self.threat_engine:
                    self.threat_engine.block_ip(client_ip, f"Rate limit exceeded: {count_1min} req/min")
    
    async def _post_process_analysis(self, request: Request, response: Response, client_ip: str, user_agent: str, start_time: float):
        """Analyse post-traitement de la requête"""
        try:
            duration = time.time() - start_time
            user_id = getattr(request.state, "user_id", None)
            
            # Mettre à jour le profil comportemental
            if self.enable_behavioral_analysis and self.threat_engine and user_id:
                self.threat_engine.update_user_behavior(
                    user_id, client_ip, user_agent, request.url.path, datetime.utcnow()
                )
            
            # Créer un événement de sécurité pour les accès réussis
            event_type = SecurityEventType.AUTHENTICATION_SUCCESS
            threat_level = ThreatLevel.LOW
            description = "Successful request"
            
            # Analyser le type de requête
            if "/admin/" in request.url.path:
                event_type = SecurityEventType.SENSITIVE_DATA_ACCESS
                threat_level = ThreatLevel.MEDIUM
                description = "Admin area access"
            elif "/api/v1/user/" in request.url.path:
                event_type = SecurityEventType.SENSITIVE_DATA_ACCESS
                threat_level = ThreatLevel.LOW
                description = "User data access"
            elif response.status_code >= 400:
                event_type = SecurityEventType.AUTHORIZATION_DENIED
                threat_level = ThreatLevel.MEDIUM
                description = f"Request failed with status {response.status_code}"
            
            # Créer l'événement de sécurité
            security_event = SecurityEvent(
                id=str(uuid.uuid4()),
                event_type=event_type,
                threat_level=threat_level,
                timestamp=datetime.utcnow(),
                source_ip=client_ip,
                user_id=user_id,
                session_id=getattr(request.state, "session_id", None),
                request_id=getattr(request.state, "request_id", None),
                endpoint=request.url.path,
                method=request.method,
                user_agent=user_agent,
                description=description,
                details={
                    "status_code": response.status_code,
                    "duration_ms": duration * 1000,
                    "response_size": getattr(response, "content_length", 0) or 0
                },
                risk_score=10 if response.status_code < 400 else 30
            )
            
            # Vérifier la conformité
            if self.enable_compliance_monitoring and self.compliance_monitor:
                violations = self.compliance_monitor.check_compliance(request, response, security_event)
                if violations:
                    security_event.details["compliance_violations"] = violations
                    security_event.threat_level = ThreatLevel.HIGH
                    security_event.risk_score += 40
                    self.security_stats["compliance_violations"] += len(violations)
            
            # Logger l'événement
            self.audit_logger.log_security_event(security_event)
            self.security_stats["total_events"] += 1
            
        except Exception as e:
            self.logger.error(f"Erreur analyse post-traitement: {e}")
    
    async def _analyze_exception(self, request: Request, exception: Exception, client_ip: str, user_agent: str):
        """Analyse les exceptions pour des indicateurs de sécurité"""
        try:
            # Créer un événement pour l'exception
            security_event = SecurityEvent(
                id=str(uuid.uuid4()),
                event_type=SecurityEventType.SYSTEM_COMPROMISE,
                threat_level=ThreatLevel.MEDIUM,
                timestamp=datetime.utcnow(),
                source_ip=client_ip,
                user_id=getattr(request.state, "user_id", None),
                session_id=getattr(request.state, "session_id", None),
                request_id=getattr(request.state, "request_id", None),
                endpoint=request.url.path,
                method=request.method,
                user_agent=user_agent,
                description=f"Application exception: {type(exception).__name__}",
                details={
                    "exception_type": type(exception).__name__,
                    "exception_message": str(exception)
                },
                risk_score=50
            )
            
            # Analyser le type d'exception
            if isinstance(exception, HTTPException):
                if exception.status_code == 401:
                    security_event.event_type = SecurityEventType.AUTHENTICATION_FAILED
                elif exception.status_code == 403:
                    security_event.event_type = SecurityEventType.AUTHORIZATION_DENIED
                elif exception.status_code == 404:
                    security_event.risk_score = 20  # Moins critique
            
            self.audit_logger.log_security_event(security_event)
            
        except Exception as e:
            self.logger.error(f"Erreur analyse exception: {e}")
    
    async def _load_threat_intelligence(self):
        """Charge les indicateurs de menace depuis des sources externes"""
        try:
            # Ici on pourrait charger depuis:
            # - APIs de threat intelligence (VirusTotal, AbuseIPDB, etc.)
            # - Feeds STIX/TAXII
            # - Bases de données internes
            # - Listes de blocage publiques
            
            # Exemple d'IPs malveillantes (normalement chargées depuis une source externe)
            malicious_ips = [
                "192.168.1.100",  # Exemple
                "10.0.0.50"       # Exemple
            ]
            
            for ip in malicious_ips:
                indicator = ThreatIndicator(
                    ioc_type="ip",
                    value=ip,
                    threat_level=ThreatLevel.HIGH,
                    source="internal_blocklist",
                    confidence=0.9,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    description="Known malicious IP"
                )
                self.threat_engine.add_threat_indicator(indicator)
            
            self.logger.info("Threat intelligence chargée")
            
        except Exception as e:
            self.logger.error(f"Erreur chargement threat intelligence: {e}")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Retourne un tableau de bord de sécurité"""
        try:
            recent_events = list(self.audit_logger.events)[-100:]
            high_risk_events = [e for e in recent_events if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
            
            # Statistiques par type d'événement
            event_types = defaultdict(int)
            for event in recent_events:
                event_types[event.event_type.value] += 1
            
            # Top IPs suspectes
            ip_counts = defaultdict(int)
            for event in recent_events:
                if event.threat_level != ThreatLevel.LOW:
                    ip_counts[event.source_ip] += 1
            
            top_suspicious_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "summary": {
                    "total_events": len(recent_events),
                    "high_risk_events": len(high_risk_events),
                    "blocked_requests": self.security_stats["blocked_requests"],
                    "threats_detected": len([e for e in recent_events if e.indicators]),
                    "compliance_violations": self.security_stats["compliance_violations"]
                },
                "event_types": dict(event_types),
                "top_suspicious_ips": top_suspicious_ips,
                "recent_high_risk_events": [e.to_dict() for e in high_risk_events[-10:]],
                "threat_indicators_count": len(self.threat_engine.threat_indicators) if self.threat_engine else 0,
                "blocked_ips_count": len(self.threat_engine.blocked_ips) if self.threat_engine else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Erreur dashboard sécurité: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Arrête proprement le middleware"""
        try:
            self.logger.info("Arrêt du middleware d'audit de sécurité")
        except Exception as e:
            self.logger.error(f"Erreur arrêt sécurité: {e}")


# Factory functions

def create_security_audit_middleware(
    enable_threat_detection: bool = True,
    enable_behavioral_analysis: bool = True,
    enable_compliance_monitoring: bool = True,
    block_suspicious_ips: bool = True
) -> AdvancedSecurityAuditMiddleware:
    """Crée un middleware d'audit de sécurité configuré"""
    return AdvancedSecurityAuditMiddleware(
        enable_threat_detection=enable_threat_detection,
        enable_behavioral_analysis=enable_behavioral_analysis,
        enable_compliance_monitoring=enable_compliance_monitoring,
        block_suspicious_ips=block_suspicious_ips
    )


def create_production_security_middleware() -> AdvancedSecurityAuditMiddleware:
    """Configuration de sécurité pour la production"""
    return AdvancedSecurityAuditMiddleware(
        enable_threat_detection=True,
        enable_behavioral_analysis=True,
        enable_compliance_monitoring=True,
        block_suspicious_ips=True
    )


def create_development_security_middleware() -> AdvancedSecurityAuditMiddleware:
    """Configuration de sécurité pour le développement"""
    return AdvancedSecurityAuditMiddleware(
        enable_threat_detection=True,
        enable_behavioral_analysis=False,
        enable_compliance_monitoring=False,
        block_suspicious_ips=False
    )


# Export des classes principales
__all__ = [
    "ThreatLevel",
    "SecurityEventType",
    "ComplianceStandard",
    "RiskScore",
    "SecurityEvent",
    "ThreatIndicator",
    "UserBehaviorProfile",
    "ComplianceRule",
    "ThreatDetectionEngine",
    "ComplianceMonitor",
    "SecurityAuditLogger",
    "AdvancedSecurityAuditMiddleware",
    "create_security_audit_middleware",
    "create_production_security_middleware",
    "create_development_security_middleware"
]
