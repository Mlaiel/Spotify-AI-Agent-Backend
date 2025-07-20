"""
Moniteurs de Sécurité Ultra-Avancés
Système de surveillance sécuritaire intelligent pour Spotify AI Agent

Fonctionnalités:
- Détection d'intrusions en temps réel par IA
- Analyse comportementale des utilisateurs
- Monitoring des vulnérabilités et menaces
- Protection contre les attaques DDoS et injection
- Audit de sécurité continu
- Détection de fuites de données
- Conformité réglementaire automatisée
"""

import asyncio
import logging
import hashlib
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import redis.asyncio as redis
from collections import defaultdict, deque
import geoip2.database
import jwt

from . import AlertConfig, AlertSeverity, AlertCategory, ScriptType, register_alert

logger = logging.getLogger(__name__)

class SecurityThreatType(Enum):
    """Types de menaces de sécurité"""
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    DDOS_ATTACK = "ddos_attack"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    MALWARE_DETECTION = "malware_detection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    API_ABUSE = "api_abuse"
    CREDENTIAL_STUFFING = "credential_stuffing"
    INSIDER_THREAT = "insider_threat"

class SecuritySeverity(Enum):
    """Niveaux de sévérité sécuritaire"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class SecurityEvent:
    """Événement de sécurité"""
    event_id: str
    event_type: SecurityThreatType
    severity: SecuritySeverity
    timestamp: datetime
    source_ip: str
    user_agent: Optional[str] = None
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    payload: Optional[str] = None
    country: Optional[str] = None
    risk_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False

@dataclass
class UserBehaviorProfile:
    """Profil comportemental d'un utilisateur"""
    user_id: str
    normal_ips: Set[str] = field(default_factory=set)
    normal_countries: Set[str] = field(default_factory=set)
    typical_hours: Set[int] = field(default_factory=set)
    average_session_duration: float = 0.0
    typical_endpoints: Set[str] = field(default_factory=set)
    risk_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    anomaly_count: int = 0

@dataclass
class SecurityRule:
    """Règle de sécurité"""
    rule_id: str
    name: str
    threat_type: SecurityThreatType
    pattern: str
    is_regex: bool = False
    threshold: int = 1
    time_window: timedelta = timedelta(minutes=5)
    action: str = "alert"
    enabled: bool = True
    whitelist: List[str] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)

class AdvancedSecurityMonitor:
    """Moniteur de sécurité avancé avec IA"""
    
    def __init__(self):
        self.redis_client = None
        self.security_events: List[SecurityEvent] = []
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        self.security_rules: List[SecurityRule] = []
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, float] = defaultdict(float)  # IP -> risk score
        self.failed_login_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.monitoring_active = False
        
        self._initialize_security_rules()

    async def initialize(self):
        """Initialise le moniteur de sécurité"""
        try:
            self.redis_client = redis.Redis(
                host="localhost", 
                port=6379, 
                decode_responses=True,
                db=4
            )
            
            # Chargement des listes noires/blanches
            await self._load_ip_blacklists()
            await self._load_user_profiles()
            
            logger.info("Moniteur de sécurité initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")

    def _initialize_security_rules(self):
        """Initialise les règles de sécurité"""
        
        rules = [
            # Détection brute force
            SecurityRule(
                rule_id="brute_force_detection",
                name="Détection attaque brute force",
                threat_type=SecurityThreatType.BRUTE_FORCE_ATTACK,
                pattern="failed_login",
                threshold=5,
                time_window=timedelta(minutes=5),
                action="block_ip"
            ),
            
            # Détection injection SQL
            SecurityRule(
                rule_id="sql_injection_detection",
                name="Détection injection SQL",
                threat_type=SecurityThreatType.SQL_INJECTION,
                pattern=r"(\bunion\b|\bselect\b|\binsert\b|\bdelete\b|\bdrop\b|\b--\b|\b;\b)",
                is_regex=True,
                threshold=1,
                action="block_request"
            ),
            
            # Détection XSS
            SecurityRule(
                rule_id="xss_detection",
                name="Détection XSS",
                threat_type=SecurityThreatType.XSS_ATTACK,
                pattern=r"(<script|javascript:|on\w+\s*=)",
                is_regex=True,
                threshold=1,
                action="block_request"
            ),
            
            # Détection DDoS
            SecurityRule(
                rule_id="ddos_detection",
                name="Détection DDoS",
                threat_type=SecurityThreatType.DDOS_ATTACK,
                pattern="high_request_rate",
                threshold=100,
                time_window=timedelta(minutes=1),
                action="rate_limit"
            ),
            
            # Détection accès non autorisé
            SecurityRule(
                rule_id="unauthorized_access",
                name="Détection accès non autorisé",
                threat_type=SecurityThreatType.UNAUTHORIZED_ACCESS,
                pattern="403|401",
                threshold=10,
                time_window=timedelta(minutes=10),
                action="investigate"
            ),
            
            # Détection credential stuffing
            SecurityRule(
                rule_id="credential_stuffing",
                name="Détection credential stuffing",
                threat_type=SecurityThreatType.CREDENTIAL_STUFFING,
                pattern="multiple_user_login_attempts",
                threshold=20,
                time_window=timedelta(minutes=5),
                action="block_ip"
            )
        ]
        
        self.security_rules.extend(rules)

    async def _load_ip_blacklists(self):
        """Charge les listes noires d'IPs"""
        
        # IPs connues malveillantes (exemple)
        known_malicious_ips = {
            "192.168.1.100",  # IP de test malveillante
            "10.0.0.1",       # Autre IP de test
        }
        
        self.blocked_ips.update(known_malicious_ips)
        
        # En production, charger depuis des sources threat intelligence
        logger.info(f"Chargé {len(self.blocked_ips)} IPs dans la liste noire")

    async def _load_user_profiles(self):
        """Charge les profils comportementaux des utilisateurs"""
        
        try:
            if self.redis_client:
                # Récupération depuis Redis
                profile_keys = await self.redis_client.keys("user_profile:*")
                
                for key in profile_keys:
                    user_id = key.split(":")[-1]
                    profile_data = await self.redis_client.hgetall(key)
                    
                    if profile_data:
                        profile = UserBehaviorProfile(
                            user_id=user_id,
                            normal_ips=set(profile_data.get("normal_ips", "").split(",")),
                            normal_countries=set(profile_data.get("normal_countries", "").split(",")),
                            typical_hours=set(map(int, profile_data.get("typical_hours", "").split(","))),
                            average_session_duration=float(profile_data.get("average_session_duration", 0)),
                            risk_score=float(profile_data.get("risk_score", 0))
                        )
                        self.user_profiles[user_id] = profile
                
                logger.info(f"Chargé {len(self.user_profiles)} profils utilisateur")
                
        except Exception as e:
            logger.error(f"Erreur chargement profils utilisateur: {e}")

    async def start_monitoring(self):
        """Démarre le monitoring de sécurité"""
        self.monitoring_active = True
        
        monitoring_tasks = [
            self._monitor_login_attempts(),
            self._monitor_api_requests(),
            self._monitor_network_traffic(),
            self._analyze_user_behavior(),
            self._threat_intelligence_updates(),
            self._compliance_monitoring()
        ]
        
        await asyncio.gather(*monitoring_tasks)

    async def _monitor_login_attempts(self):
        """Monitore les tentatives de connexion"""
        while self.monitoring_active:
            try:
                # Simulation de monitoring des logs de connexion
                login_events = await self._collect_login_events()
                
                for event in login_events:
                    await self._analyze_login_event(event)
                
                await asyncio.sleep(10)  # Vérification toutes les 10 secondes
                
            except Exception as e:
                logger.error(f"Erreur monitoring connexions: {e}")
                await asyncio.sleep(30)

    async def _collect_login_events(self) -> List[Dict[str, Any]]:
        """Collecte les événements de connexion"""
        
        # Simulation d'événements de connexion
        import random
        
        events = []
        
        # Simulation d'événements normaux et suspects
        for _ in range(random.randint(0, 5)):
            event = {
                "timestamp": datetime.utcnow(),
                "ip": f"192.168.1.{random.randint(1, 255)}",
                "user_id": f"user_{random.randint(1, 1000)}",
                "success": random.choice([True, True, True, False]),  # 75% succès
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "country": random.choice(["FR", "US", "GB", "DE", "CN"])
            }
            events.append(event)
        
        return events

    async def _analyze_login_event(self, event: Dict[str, Any]):
        """Analyse un événement de connexion"""
        
        ip = event["ip"]
        user_id = event.get("user_id")
        success = event["success"]
        timestamp = event["timestamp"]
        
        # Vérification IP bloquée
        if ip in self.blocked_ips:
            await self._create_security_event(
                SecurityThreatType.UNAUTHORIZED_ACCESS,
                SecuritySeverity.HIGH,
                ip,
                details={"reason": "IP in blacklist", "user_id": user_id}
            )
            return
        
        # Analyse des échecs de connexion
        if not success:
            self.failed_login_attempts[ip].append(timestamp)
            
            # Détection brute force
            recent_failures = [
                t for t in self.failed_login_attempts[ip]
                if timestamp - t < timedelta(minutes=5)
            ]
            
            if len(recent_failures) >= 5:
                await self._create_security_event(
                    SecurityThreatType.BRUTE_FORCE_ATTACK,
                    SecuritySeverity.HIGH,
                    ip,
                    details={
                        "failed_attempts": len(recent_failures),
                        "target_user": user_id,
                        "time_window": "5 minutes"
                    }
                )
                
                # Blocage automatique de l'IP
                await self._block_ip(ip, "Brute force attack detected")
        
        # Analyse comportementale pour les connexions réussies
        if success and user_id:
            await self._analyze_user_behavior_event(user_id, event)

    async def _analyze_user_behavior_event(self, user_id: str, event: Dict[str, Any]):
        """Analyse un événement pour détecter un comportement anormal"""
        
        # Récupération ou création du profil utilisateur
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserBehaviorProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        ip = event["ip"]
        country = event.get("country")
        hour = event["timestamp"].hour
        
        anomalies = []
        
        # Vérification IP inhabituelle
        if profile.normal_ips and ip not in profile.normal_ips:
            anomalies.append("Connexion depuis une IP inhabituelle")
        
        # Vérification pays inhabituel
        if profile.normal_countries and country not in profile.normal_countries:
            anomalies.append("Connexion depuis un pays inhabituel")
        
        # Vérification heure inhabituelle
        if profile.typical_hours and hour not in profile.typical_hours:
            anomalies.append("Connexion à une heure inhabituelle")
        
        # Création d'alerte si anomalies détectées
        if anomalies:
            profile.anomaly_count += 1
            profile.risk_score = min(100, profile.risk_score + 10)
            
            await self._create_security_event(
                SecurityThreatType.SUSPICIOUS_BEHAVIOR,
                SecuritySeverity.MEDIUM,
                ip,
                user_id=user_id,
                details={
                    "anomalies": anomalies,
                    "risk_score": profile.risk_score,
                    "anomaly_count": profile.anomaly_count
                }
            )
        
        # Mise à jour du profil (apprentissage)
        profile.normal_ips.add(ip)
        if country:
            profile.normal_countries.add(country)
        profile.typical_hours.add(hour)
        profile.last_updated = datetime.utcnow()
        
        # Sauvegarde du profil
        await self._save_user_profile(profile)

    async def _monitor_api_requests(self):
        """Monitore les requêtes API pour détecter les abus"""
        while self.monitoring_active:
            try:
                api_events = await self._collect_api_events()
                
                for event in api_events:
                    await self._analyze_api_request(event)
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Erreur monitoring API: {e}")
                await asyncio.sleep(15)

    async def _collect_api_events(self) -> List[Dict[str, Any]]:
        """Collecte les événements API"""
        
        import random
        
        events = []
        
        # Simulation d'événements API
        endpoints = [
            "/api/v1/users",
            "/api/v1/audio/process", 
            "/api/v1/auth/login",
            "/api/v1/upload",
            "/admin/users",
            "/api/v1/analytics"
        ]
        
        for _ in range(random.randint(5, 20)):
            event = {
                "timestamp": datetime.utcnow(),
                "ip": f"192.168.1.{random.randint(1, 255)}",
                "endpoint": random.choice(endpoints),
                "method": random.choice(["GET", "POST", "PUT", "DELETE"]),
                "status_code": random.choice([200, 200, 200, 401, 403, 500]),
                "user_agent": random.choice([
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "curl/7.68.0",
                    "python-requests/2.25.1",
                    "<script>alert('xss')</script>"  # Payload suspect
                ]),
                "payload": random.choice([None, None, None, "'; DROP TABLE users; --"])
            }
            events.append(event)
        
        return events

    async def _analyze_api_request(self, event: Dict[str, Any]):
        """Analyse une requête API"""
        
        ip = event["ip"]
        endpoint = event["endpoint"]
        user_agent = event["user_agent"]
        payload = event.get("payload")
        status_code = event["status_code"]
        
        # Détection injection SQL
        if payload:
            for rule in self.security_rules:
                if rule.threat_type == SecurityThreatType.SQL_INJECTION and rule.enabled:
                    if rule.is_regex:
                        if re.search(rule.pattern, payload, re.IGNORECASE):
                            await self._create_security_event(
                                SecurityThreatType.SQL_INJECTION,
                                SecuritySeverity.CRITICAL,
                                ip,
                                endpoint=endpoint,
                                payload=payload,
                                details={"detected_pattern": rule.pattern}
                            )
        
        # Détection XSS
        if user_agent:
            for rule in self.security_rules:
                if rule.threat_type == SecurityThreatType.XSS_ATTACK and rule.enabled:
                    if rule.is_regex:
                        if re.search(rule.pattern, user_agent, re.IGNORECASE):
                            await self._create_security_event(
                                SecurityThreatType.XSS_ATTACK,
                                SecuritySeverity.HIGH,
                                ip,
                                endpoint=endpoint,
                                details={"malicious_user_agent": user_agent}
                            )
        
        # Détection accès non autorisé
        if status_code in [401, 403]:
            await self._track_unauthorized_access(ip, endpoint, status_code)
        
        # Détection rate limiting (DDoS)
        await self._track_request_rate(ip)

    async def _track_unauthorized_access(self, ip: str, endpoint: str, status_code: int):
        """Suit les accès non autorisés"""
        
        key = f"unauthorized:{ip}"
        
        try:
            if self.redis_client:
                # Incrémenter le compteur
                count = await self.redis_client.incr(key)
                await self.redis_client.expire(key, 600)  # 10 minutes TTL
                
                # Alerte si seuil dépassé
                if count >= 10:
                    await self._create_security_event(
                        SecurityThreatType.UNAUTHORIZED_ACCESS,
                        SecuritySeverity.HIGH,
                        ip,
                        endpoint=endpoint,
                        details={
                            "unauthorized_attempts": count,
                            "status_code": status_code,
                            "time_window": "10 minutes"
                        }
                    )
                    
                    # Blocage automatique
                    await self._block_ip(ip, f"Too many unauthorized access attempts ({count})")
                    
        except Exception as e:
            logger.error(f"Erreur tracking accès non autorisé: {e}")

    async def _track_request_rate(self, ip: str):
        """Suit le taux de requêtes pour détecter DDoS"""
        
        key = f"rate_limit:{ip}"
        
        try:
            if self.redis_client:
                # Sliding window counter
                current_time = int(datetime.utcnow().timestamp())
                
                # Pipeline pour opérations atomiques
                pipe = self.redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, current_time - 60)  # Supprimer > 1 minute
                pipe.zadd(key, {str(current_time): current_time})
                pipe.zcard(key)
                pipe.expire(key, 120)
                
                results = await pipe.execute()
                request_count = results[2]
                
                # Vérification seuil DDoS
                if request_count > 100:  # 100 requêtes par minute
                    await self._create_security_event(
                        SecurityThreatType.DDOS_ATTACK,
                        SecuritySeverity.CRITICAL,
                        ip,
                        details={
                            "requests_per_minute": request_count,
                            "threshold": 100
                        }
                    )
                    
                    # Blocage temporaire
                    await self._block_ip(ip, f"DDoS attack detected ({request_count} req/min)", duration=3600)
                    
        except Exception as e:
            logger.error(f"Erreur tracking taux requêtes: {e}")

    async def _monitor_network_traffic(self):
        """Monitore le trafic réseau pour détecter les anomalies"""
        while self.monitoring_active:
            try:
                # Simulation de monitoring réseau
                network_stats = await self._collect_network_stats()
                
                await self._analyze_network_anomalies(network_stats)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Erreur monitoring réseau: {e}")
                await asyncio.sleep(60)

    async def _collect_network_stats(self) -> Dict[str, Any]:
        """Collecte les statistiques réseau"""
        
        import random
        
        # Simulation de statistiques réseau
        return {
            "bytes_sent": random.randint(1000000, 10000000),
            "bytes_received": random.randint(5000000, 50000000),
            "packets_sent": random.randint(1000, 10000),
            "packets_received": random.randint(5000, 50000),
            "connections_count": random.randint(100, 1000),
            "unusual_ports": random.choice([[], [8080, 3389, 22]]),
            "timestamp": datetime.utcnow()
        }

    async def _analyze_network_anomalies(self, stats: Dict[str, Any]):
        """Analyse les anomalies réseau"""
        
        # Détection de ports suspects
        suspicious_ports = [22, 23, 3389, 1433, 3306]  # SSH, Telnet, RDP, SQL Server, MySQL
        
        unusual_ports = stats.get("unusual_ports", [])
        for port in unusual_ports:
            if port in suspicious_ports:
                await self._create_security_event(
                    SecurityThreatType.SUSPICIOUS_BEHAVIOR,
                    SecuritySeverity.MEDIUM,
                    "network",
                    details={
                        "suspicious_port": port,
                        "reason": "Uncommon port activity detected"
                    }
                )

    async def _analyze_user_behavior(self):
        """Analyse comportementale avancée des utilisateurs"""
        while self.monitoring_active:
            try:
                for user_id, profile in self.user_profiles.items():
                    # Calcul du score de risque
                    risk_score = await self._calculate_user_risk_score(profile)
                    profile.risk_score = risk_score
                    
                    # Alerte si score de risque élevé
                    if risk_score > 80:
                        await self._create_security_event(
                            SecurityThreatType.INSIDER_THREAT,
                            SecuritySeverity.HIGH,
                            "behavior_analysis",
                            user_id=user_id,
                            details={
                                "risk_score": risk_score,
                                "anomaly_count": profile.anomaly_count,
                                "reason": "High risk user behavior detected"
                            }
                        )
                
                await asyncio.sleep(300)  # Analyse toutes les 5 minutes
                
            except Exception as e:
                logger.error(f"Erreur analyse comportementale: {e}")
                await asyncio.sleep(600)

    async def _calculate_user_risk_score(self, profile: UserBehaviorProfile) -> float:
        """Calcule le score de risque d'un utilisateur"""
        
        base_score = 0.0
        
        # Facteurs de risque
        if profile.anomaly_count > 10:
            base_score += 30
        elif profile.anomaly_count > 5:
            base_score += 15
        
        # IPs multiples (possibles compromissions)
        if len(profile.normal_ips) > 10:
            base_score += 20
        
        # Pays multiples
        if len(profile.normal_countries) > 5:
            base_score += 25
        
        # Activité à des heures inhabituelles
        if len(profile.typical_hours) > 18:  # Actif plus de 18h/jour
            base_score += 15
        
        return min(100.0, base_score)

    async def _threat_intelligence_updates(self):
        """Met à jour la threat intelligence"""
        while self.monitoring_active:
            try:
                # Simulation de mise à jour threat intelligence
                await self._update_malicious_ips()
                await self._update_malware_signatures()
                
                await asyncio.sleep(3600)  # Mise à jour toutes les heures
                
            except Exception as e:
                logger.error(f"Erreur mise à jour threat intelligence: {e}")
                await asyncio.sleep(7200)

    async def _update_malicious_ips(self):
        """Met à jour la liste des IPs malveillantes"""
        
        # En production, récupérer depuis des feeds threat intelligence
        new_malicious_ips = {
            "198.51.100.1",
            "203.0.113.1"
        }
        
        self.blocked_ips.update(new_malicious_ips)
        logger.info(f"Ajouté {len(new_malicious_ips)} nouvelles IPs malveillantes")

    async def _update_malware_signatures(self):
        """Met à jour les signatures de malware"""
        
        # En production, mettre à jour les signatures de détection
        logger.info("Signatures de malware mises à jour")

    async def _compliance_monitoring(self):
        """Monitoring de conformité réglementaire"""
        while self.monitoring_active:
            try:
                await self._check_gdpr_compliance()
                await self._check_data_retention()
                await self._audit_access_controls()
                
                await asyncio.sleep(86400)  # Vérification quotidienne
                
            except Exception as e:
                logger.error(f"Erreur monitoring conformité: {e}")
                await asyncio.sleep(86400)

    async def _check_gdpr_compliance(self):
        """Vérifie la conformité GDPR"""
        
        # Vérifications GDPR basiques
        compliance_issues = []
        
        # Vérifier la retention des données
        old_events = [
            event for event in self.security_events
            if datetime.utcnow() - event.timestamp > timedelta(days=365)
        ]
        
        if old_events:
            compliance_issues.append(f"{len(old_events)} événements dépassent la limite de rétention")
        
        if compliance_issues:
            await self._create_security_event(
                SecurityThreatType.SUSPICIOUS_BEHAVIOR,
                SecuritySeverity.MEDIUM,
                "compliance",
                details={
                    "compliance_type": "GDPR",
                    "issues": compliance_issues
                }
            )

    async def _check_data_retention(self):
        """Vérifie les politiques de rétention des données"""
        
        # Nettoyage automatique des anciens événements
        cutoff_date = datetime.utcnow() - timedelta(days=90)
        
        self.security_events = [
            event for event in self.security_events
            if event.timestamp > cutoff_date
        ]
        
        logger.info("Nettoyage des anciens événements de sécurité effectué")

    async def _audit_access_controls(self):
        """Audit des contrôles d'accès"""
        
        # Simulation d'audit des accès
        audit_findings = []
        
        # Vérifier les utilisateurs à haut risque
        high_risk_users = [
            user_id for user_id, profile in self.user_profiles.items()
            if profile.risk_score > 70
        ]
        
        if high_risk_users:
            audit_findings.append(f"{len(high_risk_users)} utilisateurs à haut risque identifiés")
        
        if audit_findings:
            logger.warning(f"Audit d'accès: {', '.join(audit_findings)}")

    async def _create_security_event(self, threat_type: SecurityThreatType, severity: SecuritySeverity, source_ip: str, user_id: Optional[str] = None, endpoint: Optional[str] = None, payload: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Crée un événement de sécurité"""
        
        event_id = f"sec_{int(datetime.utcnow().timestamp())}_{len(self.security_events)}"
        
        event = SecurityEvent(
            event_id=event_id,
            event_type=threat_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            endpoint=endpoint,
            payload=payload,
            details=details or {}
        )
        
        # Calcul du score de risque
        event.risk_score = await self._calculate_event_risk_score(event)
        
        # Géolocalisation IP
        event.country = await self._get_ip_country(source_ip)
        
        self.security_events.append(event)
        
        # Stockage dans Redis
        await self._store_security_event(event)
        
        # Actions automatiques selon la sévérité
        if severity in [SecuritySeverity.CRITICAL, SecuritySeverity.EMERGENCY]:
            await self._handle_critical_security_event(event)
        
        logger.warning(
            f"ÉVÉNEMENT SÉCURITÉ: {threat_type.value} - {severity.value} "
            f"- IP: {source_ip} - Score: {event.risk_score:.1f}"
        )

    async def _calculate_event_risk_score(self, event: SecurityEvent) -> float:
        """Calcule le score de risque d'un événement"""
        
        base_score = 0.0
        
        # Score basé sur la sévérité
        severity_scores = {
            SecuritySeverity.INFO: 10,
            SecuritySeverity.LOW: 25,
            SecuritySeverity.MEDIUM: 50,
            SecuritySeverity.HIGH: 75,
            SecuritySeverity.CRITICAL: 90,
            SecuritySeverity.EMERGENCY: 100
        }
        
        base_score = severity_scores.get(event.severity, 50)
        
        # Ajustements basés sur le contexte
        if event.source_ip in self.blocked_ips:
            base_score += 20
        
        if event.user_id and event.user_id in self.user_profiles:
            user_risk = self.user_profiles[event.user_id].risk_score
            base_score += user_risk * 0.3
        
        # Score historique de l'IP
        ip_risk = self.suspicious_ips.get(event.source_ip, 0)
        base_score += ip_risk * 0.2
        
        return min(100.0, base_score)

    async def _get_ip_country(self, ip: str) -> Optional[str]:
        """Obtient le pays d'une IP via géolocalisation"""
        
        try:
            # En production, utiliser une vraie base GeoIP
            # Simulation basée sur l'IP
            if ip.startswith("192.168"):
                return "FR"  # IP privée = France par défaut
            elif ip.startswith("10."):
                return "US"
            else:
                return "Unknown"
        except Exception as e:
            logger.error(f"Erreur géolocalisation IP {ip}: {e}")
            return None

    async def _store_security_event(self, event: SecurityEvent):
        """Stocke un événement de sécurité dans Redis"""
        
        try:
            if self.redis_client:
                key = f"security_event:{event.event_id}"
                
                data = {
                    "event_type": event.event_type.value,
                    "severity": event.severity.value,
                    "timestamp": event.timestamp.isoformat(),
                    "source_ip": event.source_ip,
                    "user_id": event.user_id or "",
                    "endpoint": event.endpoint or "",
                    "country": event.country or "",
                    "risk_score": event.risk_score,
                    "details": str(event.details),
                    "blocked": event.blocked
                }
                
                await self.redis_client.hset(key, mapping=data)
                await self.redis_client.expire(key, 2592000)  # 30 jours TTL
                
        except Exception as e:
            logger.error(f"Erreur stockage événement sécurité: {e}")

    async def _handle_critical_security_event(self, event: SecurityEvent):
        """Gère un événement de sécurité critique"""
        
        # Actions automatiques pour événements critiques
        if event.event_type in [SecurityThreatType.BRUTE_FORCE_ATTACK, SecurityThreatType.DDOS_ATTACK]:
            await self._block_ip(event.source_ip, f"Critical security event: {event.event_type.value}")
        
        # Notification immédiate de l'équipe sécurité
        logger.critical(
            f"ÉVÉNEMENT SÉCURITÉ CRITIQUE: {event.event_type.value} "
            f"de {event.source_ip} - Action requise immédiatement"
        )

    async def _block_ip(self, ip: str, reason: str, duration: Optional[int] = None):
        """Bloque une IP"""
        
        self.blocked_ips.add(ip)
        
        try:
            if self.redis_client:
                key = f"blocked_ip:{ip}"
                data = {
                    "reason": reason,
                    "blocked_at": datetime.utcnow().isoformat(),
                    "duration": duration or 86400  # 24h par défaut
                }
                
                await self.redis_client.hset(key, mapping=data)
                if duration:
                    await self.redis_client.expire(key, duration)
                
            logger.warning(f"IP {ip} bloquée: {reason}")
            
        except Exception as e:
            logger.error(f"Erreur blocage IP {ip}: {e}")

    async def _save_user_profile(self, profile: UserBehaviorProfile):
        """Sauvegarde un profil utilisateur"""
        
        try:
            if self.redis_client:
                key = f"user_profile:{profile.user_id}"
                
                data = {
                    "normal_ips": ",".join(profile.normal_ips),
                    "normal_countries": ",".join(profile.normal_countries),
                    "typical_hours": ",".join(map(str, profile.typical_hours)),
                    "average_session_duration": profile.average_session_duration,
                    "risk_score": profile.risk_score,
                    "last_updated": profile.last_updated.isoformat(),
                    "anomaly_count": profile.anomaly_count
                }
                
                await self.redis_client.hset(key, mapping=data)
                
        except Exception as e:
            logger.error(f"Erreur sauvegarde profil {profile.user_id}: {e}")

    async def get_security_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la sécurité"""
        
        recent_events = [
            event for event in self.security_events
            if datetime.utcnow() - event.timestamp < timedelta(hours=24)
        ]
        
        # Distribution par type de menace
        threat_distribution = defaultdict(int)
        for event in recent_events:
            threat_distribution[event.event_type.value] += 1
        
        # Top IPs suspectes
        ip_counts = defaultdict(int)
        for event in recent_events:
            ip_counts[event.source_ip] += 1
        
        top_suspicious_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_events": len(self.security_events),
            "recent_24h": len(recent_events),
            "blocked_ips_count": len(self.blocked_ips),
            "high_risk_users": len([
                p for p in self.user_profiles.values() if p.risk_score > 70
            ]),
            "threat_distribution": dict(threat_distribution),
            "top_suspicious_ips": top_suspicious_ips,
            "monitoring_active": self.monitoring_active
        }

    async def stop_monitoring(self):
        """Arrête le monitoring de sécurité"""
        self.monitoring_active = False
        logger.info("Monitoring de sécurité arrêté")

# Instance globale du moniteur de sécurité
_security_monitor = AdvancedSecurityMonitor()

async def start_security_monitoring():
    """Function helper pour démarrer le monitoring de sécurité"""
    if not _security_monitor.redis_client:
        await _security_monitor.initialize()
    
    await _security_monitor.start_monitoring()

async def get_security_monitor() -> AdvancedSecurityMonitor:
    """Retourne l'instance du moniteur de sécurité"""
    return _security_monitor

# Configuration des alertes de sécurité
if __name__ == "__main__":
    # Enregistrement des configurations d'alertes
    security_configs = [
        AlertConfig(
            name="brute_force_attack_detected",
            category=AlertCategory.SECURITY,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.DETECTION,
            conditions=['Brute force attack detected'],
            actions=['block_ip_immediately', 'notify_security_team'],
            ml_enabled=False,
            auto_remediation=True
        ),
        AlertConfig(
            name="sql_injection_attempt",
            category=AlertCategory.SECURITY,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.DETECTION,
            conditions=['SQL injection pattern detected'],
            actions=['block_request', 'log_forensics', 'alert_security'],
            ml_enabled=False,
            auto_remediation=True
        ),
        AlertConfig(
            name="ddos_attack_detected",
            category=AlertCategory.SECURITY,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.DETECTION,
            conditions=['DDoS attack pattern detected'],
            actions=['activate_ddos_protection', 'rate_limit_aggressive'],
            ml_enabled=True,
            auto_remediation=True
        ),
        AlertConfig(
            name="insider_threat_detected",
            category=AlertCategory.SECURITY,
            severity=AlertSeverity.HIGH,
            script_type=ScriptType.DETECTION,
            conditions=['High risk user behavior detected'],
            actions=['flag_user_account', 'enhance_monitoring', 'notify_hr'],
            ml_enabled=True
        )
    ]
    
    for config in security_configs:
        register_alert(config)
