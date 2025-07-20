# 🔐 Security Monitoring & Threat Detection
# ==========================================
# 
# Module de surveillance de sécurité et détection
# des menaces en temps réel pour l'enterprise.
#
# 🎖️ Expert: Lead Dev + Architecte IA + Spécialiste Sécurité Backend
#
# Développé par l'équipe d'experts enterprise
# ==========================================

"""
🔐 Enterprise Security Monitoring & Threat Detection
====================================================

Advanced security monitoring and threat detection providing:
- Real-time threat detection and analysis
- Behavioral analytics and anomaly detection
- Security event correlation and SIEM integration
- Automated incident response and mitigation
- Threat intelligence integration
- ML-powered attack pattern recognition
- Compliance monitoring and reporting
- Security metrics and dashboards
"""

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, asdict
import redis
import logging
from collections import defaultdict, deque
import geoip2.database
import geoip2.errors
import user_agents
import re
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration et logging
logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Niveaux de menace"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Types d'événements de sécurité"""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    PASSWORD_CHANGE = "password_change"
    PERMISSION_ESCALATION = "permission_escalation"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    API_ACCESS = "api_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MALWARE_DETECTION = "malware_detection"
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    CSRF_ATTEMPT = "csrf_attempt"
    DDoS_ATTACK = "ddos_attack"


class AttackVector(Enum):
    """Vecteurs d'attaque"""
    WEB_APPLICATION = "web_application"
    API_ENDPOINT = "api_endpoint"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SESSION_MANAGEMENT = "session_management"
    INPUT_VALIDATION = "input_validation"
    ENCRYPTION = "encryption"
    NETWORK = "network"
    SOCIAL_ENGINEERING = "social_engineering"


class ResponseAction(Enum):
    """Actions de réponse automatiques"""
    LOG_ONLY = "log_only"
    RATE_LIMIT = "rate_limit"
    TEMPORARY_BLOCK = "temporary_block"
    PERMANENT_BLOCK = "permanent_block"
    CHALLENGE_USER = "challenge_user"
    REQUIRE_MFA = "require_mfa"
    LOCKOUT_ACCOUNT = "lockout_account"
    ALERT_ADMIN = "alert_admin"
    QUARANTINE = "quarantine"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class SecurityEvent:
    """Événement de sécurité"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    threat_level: ThreatLevel
    attack_vector: Optional[AttackVector]
    details: Dict[str, Any]
    source: str
    location: Optional[Dict[str, str]] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())


@dataclass
class ThreatIndicator:
    """Indicateur de menace"""
    indicator_id: str
    indicator_type: str  # ip, domain, hash, pattern
    value: str
    threat_level: ThreatLevel
    category: str
    description: str
    source: str
    confidence: float
    created_at: datetime
    expires_at: Optional[datetime] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class SecurityIncident:
    """Incident de sécurité"""
    incident_id: str
    title: str
    description: str
    severity: ThreatLevel
    status: str  # open, investigating, resolved, closed
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str]
    events: List[str]  # IDs des événements
    response_actions: List[ResponseAction]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdvancedThreatDetector:
    """Détecteur avancé de menaces"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.event_retention_days = 90
        self.anomaly_detection_window = 3600  # 1 heure
        self.threat_score_threshold = 0.7
        
        # Modèles ML pour détection d'anomalies
        self.anomaly_detector = None
        self.scaler = None
        
        # GeoIP pour géolocalisation
        self.geoip_reader = None
        try:
            self.geoip_reader = geoip2.database.Reader('/usr/share/GeoIP/GeoLite2-City.mmdb')
        except Exception:
            self.logger.warning("Base de données GeoIP non disponible")
        
        # Patterns d'attaque
        self.attack_patterns = {
            EventType.SQL_INJECTION: [
                r"union\s+select",
                r"or\s+1\s*=\s*1",
                r"drop\s+table",
                r"insert\s+into",
                r"delete\s+from",
                r"update\s+.*\s+set",
                r"--\s*$",
                r"/\*.*\*/"
            ],
            EventType.XSS_ATTEMPT: [
                r"<script.*?>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe.*?>",
                r"<object.*?>",
                r"eval\s*\(",
                r"document\.cookie"
            ],
            EventType.BRUTE_FORCE: [
                # Détecté par analyse comportementale
            ]
        }
        
        # Initialiser le détecteur d'anomalies
        asyncio.create_task(self._initialize_ml_models())
    
    async def process_security_event(
        self,
        event_type: EventType,
        user_id: Optional[str],
        ip_address: str,
        user_agent: str,
        details: Dict[str, Any],
        source: str = "application"
    ) -> SecurityEvent:
        """Traite un événement de sécurité"""
        try:
            # Créer l'événement
            event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                threat_level=ThreatLevel.INFO,  # Initial, sera recalculé
                attack_vector=None,
                details=details,
                source=source
            )
            
            # Enrichir l'événement
            await self._enrich_event(event)
            
            # Analyser la menace
            threat_analysis = await self._analyze_threat(event)
            event.threat_level = threat_analysis["threat_level"]
            event.attack_vector = threat_analysis.get("attack_vector")
            
            # Stocker l'événement
            await self._store_security_event(event)
            
            # Détecter les patterns d'attaque
            await self._detect_attack_patterns(event)
            
            # Corréler avec d'autres événements
            correlations = await self._correlate_events(event)
            
            # Déclencher les réponses automatiques
            if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                await self._trigger_automated_response(event, correlations)
            
            # Mettre à jour les métriques
            await self._update_security_metrics(event)
            
            self.logger.info(f"Événement de sécurité traité: {event.event_id} ({event.threat_level.value})")
            return event
            
        except Exception as exc:
            self.logger.error(f"Erreur traitement événement sécurité: {exc}")
            raise
    
    async def detect_anomalies(
        self,
        user_id: str,
        time_window: int = 3600
    ) -> List[Dict[str, Any]]:
        """Détecte les anomalies comportementales"""
        try:
            # Récupérer l'historique des événements
            events = await self._get_user_events(user_id, time_window)
            
            if len(events) < 10:  # Pas assez de données
                return []
            
            # Extraire les features
            features = await self._extract_behavioral_features(events)
            
            # Détecter les anomalies avec ML
            anomalies = await self._detect_ml_anomalies(features)
            
            return anomalies
            
        except Exception as exc:
            self.logger.error(f"Erreur détection anomalies utilisateur {user_id}: {exc}")
            return []
    
    async def get_threat_intelligence(
        self,
        indicator: str,
        indicator_type: str = "ip"
    ) -> Optional[ThreatIndicator]:
        """Récupère les informations de threat intelligence"""
        try:
            # Chercher dans le cache local
            ti_data = await self.redis_client.get(f"threat_intel:{indicator_type}:{indicator}")
            
            if ti_data:
                ti_dict = json.loads(ti_data)
                return ThreatIndicator(**ti_dict)
            
            # Interroger les sources externes (implémentation simplifiée)
            external_ti = await self._query_external_threat_intel(indicator, indicator_type)
            
            if external_ti:
                # Mettre en cache
                await self.redis_client.setex(
                    f"threat_intel:{indicator_type}:{indicator}",
                    3600,  # 1 heure
                    json.dumps(asdict(external_ti), default=str)
                )
                
                return external_ti
            
            return None
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération threat intelligence: {exc}")
            return None
    
    async def create_security_incident(
        self,
        title: str,
        description: str,
        severity: ThreatLevel,
        event_ids: List[str],
        response_actions: List[ResponseAction] = None
    ) -> SecurityIncident:
        """Crée un incident de sécurité"""
        try:
            incident = SecurityIncident(
                incident_id=str(uuid.uuid4()),
                title=title,
                description=description,
                severity=severity,
                status="open",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                assigned_to=None,
                events=event_ids,
                response_actions=response_actions or []
            )
            
            # Stocker l'incident
            await self._store_security_incident(incident)
            
            # Notifier les équipes de sécurité
            await self._notify_security_team(incident)
            
            self.logger.warning(f"Incident de sécurité créé: {incident.incident_id} ({severity.value})")
            return incident
            
        except Exception as exc:
            self.logger.error(f"Erreur création incident sécurité: {exc}")
            raise
    
    async def get_security_dashboard_data(
        self,
        time_range: int = 86400  # 24 heures
    ) -> Dict[str, Any]:
        """Récupère les données pour le tableau de bord sécurité"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(seconds=time_range)
            
            # Métriques générales
            total_events = await self._count_events_in_range(start_time, end_time)
            events_by_type = await self._get_events_by_type(start_time, end_time)
            events_by_threat_level = await self._get_events_by_threat_level(start_time, end_time)
            
            # Top des IPs suspectes
            suspicious_ips = await self._get_top_suspicious_ips(start_time, end_time)
            
            # Incidents actifs
            active_incidents = await self._get_active_incidents()
            
            # Tendances horaires
            hourly_trends = await self._get_hourly_event_trends(start_time, end_time)
            
            return {
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "metrics": {
                    "total_events": total_events,
                    "events_by_type": events_by_type,
                    "events_by_threat_level": events_by_threat_level,
                    "active_incidents": len(active_incidents)
                },
                "threats": {
                    "suspicious_ips": suspicious_ips,
                    "active_incidents": active_incidents
                },
                "trends": {
                    "hourly": hourly_trends
                }
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération données tableau de bord: {exc}")
            return {}
    
    # Méthodes privées
    async def _enrich_event(self, event: SecurityEvent):
        """Enrichit un événement avec des métadonnées"""
        try:
            # Géolocalisation
            if self.geoip_reader:
                try:
                    response = self.geoip_reader.city(event.ip_address)
                    event.location = {
                        "country": response.country.name,
                        "country_code": response.country.iso_code,
                        "city": response.city.name,
                        "latitude": str(response.location.latitude),
                        "longitude": str(response.location.longitude)
                    }
                except (geoip2.errors.AddressNotFoundError, Exception):
                    pass
            
            # Analyse User-Agent
            if event.user_agent:
                ua = user_agents.parse(event.user_agent)
                event.details.update({
                    "browser": ua.browser.family,
                    "browser_version": ua.browser.version_string,
                    "os": ua.os.family,
                    "os_version": ua.os.version_string,
                    "device": ua.device.family,
                    "is_mobile": ua.is_mobile,
                    "is_bot": ua.is_bot
                })
            
            # Threat Intelligence
            threat_info = await self.get_threat_intelligence(event.ip_address, "ip")
            if threat_info:
                event.details["threat_intel"] = {
                    "threat_level": threat_info.threat_level.value,
                    "category": threat_info.category,
                    "confidence": threat_info.confidence,
                    "source": threat_info.source
                }
            
        except Exception as exc:
            self.logger.error(f"Erreur enrichissement événement: {exc}")
    
    async def _analyze_threat(self, event: SecurityEvent) -> Dict[str, Any]:
        """Analyse le niveau de menace d'un événement"""
        try:
            threat_score = 0.0
            attack_vector = None
            
            # Score basé sur le type d'événement
            event_scores = {
                EventType.LOGIN_FAILURE: 0.2,
                EventType.LOGIN_SUCCESS: 0.1,
                EventType.PASSWORD_CHANGE: 0.3,
                EventType.PERMISSION_ESCALATION: 0.8,
                EventType.DATA_EXPORT: 0.6,
                EventType.SUSPICIOUS_ACTIVITY: 0.7,
                EventType.MALWARE_DETECTION: 0.9,
                EventType.BRUTE_FORCE: 0.8,
                EventType.SQL_INJECTION: 0.9,
                EventType.XSS_ATTEMPT: 0.7,
                EventType.DDoS_ATTACK: 0.8
            }
            
            threat_score += event_scores.get(event.event_type, 0.1)
            
            # Score basé sur la threat intelligence
            if "threat_intel" in event.details:
                ti = event.details["threat_intel"]
                threat_score += ti["confidence"] * 0.5
                
                if ti["threat_level"] in ["high", "critical"]:
                    threat_score += 0.3
            
            # Score basé sur la géolocalisation
            if event.location:
                # Vérifier si c'est un pays à risque (liste simplifiée)
                high_risk_countries = ["CN", "RU", "KP", "IR"]
                if event.location.get("country_code") in high_risk_countries:
                    threat_score += 0.2
            
            # Score basé sur l'heure (activité nocturne suspecte)
            hour = event.timestamp.hour
            if hour < 6 or hour > 22:  # Entre 22h et 6h
                threat_score += 0.1
            
            # Score basé sur les patterns d'attaque
            for attack_type, patterns in self.attack_patterns.items():
                for pattern in patterns:
                    for detail_value in event.details.values():
                        if isinstance(detail_value, str) and re.search(pattern, detail_value, re.IGNORECASE):
                            threat_score += 0.4
                            attack_vector = self._map_event_to_attack_vector(attack_type)
                            break
            
            # Déterminer le niveau de menace
            if threat_score < 0.3:
                threat_level = ThreatLevel.LOW
            elif threat_score < 0.5:
                threat_level = ThreatLevel.MEDIUM
            elif threat_score < 0.8:
                threat_level = ThreatLevel.HIGH
            else:
                threat_level = ThreatLevel.CRITICAL
            
            return {
                "threat_level": threat_level,
                "threat_score": threat_score,
                "attack_vector": attack_vector
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur analyse menace: {exc}")
            return {
                "threat_level": ThreatLevel.LOW,
                "threat_score": 0.0,
                "attack_vector": None
            }
    
    async def _detect_attack_patterns(self, event: SecurityEvent):
        """Détecte les patterns d'attaque spécifiques"""
        try:
            # Détection de brute force
            if event.event_type == EventType.LOGIN_FAILURE:
                await self._detect_brute_force(event)
            
            # Détection d'escalade de privilèges
            if event.event_type == EventType.PERMISSION_ESCALATION:
                await self._detect_privilege_escalation(event)
            
            # Détection d'accès anormal aux données
            if event.event_type == EventType.DATA_ACCESS:
                await self._detect_abnormal_data_access(event)
            
        except Exception as exc:
            self.logger.error(f"Erreur détection patterns d'attaque: {exc}")
    
    async def _detect_brute_force(self, event: SecurityEvent):
        """Détecte les attaques par force brute"""
        try:
            # Compter les échecs de connexion récents
            window = 300  # 5 minutes
            threshold = 5
            
            recent_failures = await self._count_recent_failures(
                event.ip_address,
                event.user_id,
                window
            )
            
            if recent_failures >= threshold:
                # Créer un événement de brute force
                brute_force_event = SecurityEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.BRUTE_FORCE,
                    timestamp=datetime.utcnow(),
                    user_id=event.user_id,
                    ip_address=event.ip_address,
                    user_agent=event.user_agent,
                    threat_level=ThreatLevel.HIGH,
                    attack_vector=AttackVector.AUTHENTICATION,
                    details={
                        "failure_count": recent_failures,
                        "time_window": window,
                        "related_event": event.event_id
                    },
                    source="threat_detector"
                )
                
                await self._store_security_event(brute_force_event)
                
                # Déclencher une réponse automatique
                await self._trigger_automated_response(brute_force_event, [])
            
        except Exception as exc:
            self.logger.error(f"Erreur détection brute force: {exc}")
    
    async def _correlate_events(self, event: SecurityEvent) -> List[SecurityEvent]:
        """Corrèle un événement avec d'autres événements"""
        try:
            correlations = []
            
            # Corréler par IP
            ip_events = await self._get_events_by_ip(event.ip_address, 3600)  # 1 heure
            correlations.extend(ip_events)
            
            # Corréler par utilisateur
            if event.user_id:
                user_events = await self._get_user_events(event.user_id, 1800)  # 30 minutes
                correlations.extend(user_events)
            
            # Filtrer les doublons
            unique_correlations = []
            seen_ids = set()
            
            for corr_event in correlations:
                if corr_event.event_id not in seen_ids and corr_event.event_id != event.event_id:
                    unique_correlations.append(corr_event)
                    seen_ids.add(corr_event.event_id)
            
            return unique_correlations
            
        except Exception as exc:
            self.logger.error(f"Erreur corrélation événements: {exc}")
            return []
    
    async def _trigger_automated_response(
        self,
        event: SecurityEvent,
        correlations: List[SecurityEvent]
    ):
        """Déclenche une réponse automatique"""
        try:
            actions = []
            
            # Déterminer les actions selon le type de menace
            if event.event_type == EventType.BRUTE_FORCE:
                actions = [ResponseAction.TEMPORARY_BLOCK, ResponseAction.ALERT_ADMIN]
            elif event.event_type == EventType.SQL_INJECTION:
                actions = [ResponseAction.PERMANENT_BLOCK, ResponseAction.ALERT_ADMIN]
            elif event.threat_level == ThreatLevel.CRITICAL:
                actions = [ResponseAction.EMERGENCY_SHUTDOWN, ResponseAction.ALERT_ADMIN]
            elif event.threat_level == ThreatLevel.HIGH:
                actions = [ResponseAction.RATE_LIMIT, ResponseAction.REQUIRE_MFA, ResponseAction.ALERT_ADMIN]
            
            # Exécuter les actions
            for action in actions:
                await self._execute_response_action(action, event)
            
            # Enregistrer les actions prises
            response_log = {
                "event_id": event.event_id,
                "actions": [action.value for action in actions],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.lpush(
                "automated_responses",
                json.dumps(response_log)
            )
            
        except Exception as exc:
            self.logger.error(f"Erreur réponse automatique: {exc}")
    
    async def _execute_response_action(self, action: ResponseAction, event: SecurityEvent):
        """Exécute une action de réponse"""
        try:
            if action == ResponseAction.TEMPORARY_BLOCK:
                await self._block_ip_temporarily(event.ip_address, 3600)  # 1 heure
            
            elif action == ResponseAction.PERMANENT_BLOCK:
                await self._block_ip_permanently(event.ip_address)
            
            elif action == ResponseAction.RATE_LIMIT:
                await self._apply_rate_limit(event.ip_address, event.user_id)
            
            elif action == ResponseAction.REQUIRE_MFA:
                await self._require_mfa_for_user(event.user_id)
            
            elif action == ResponseAction.LOCKOUT_ACCOUNT:
                await self._lockout_user_account(event.user_id)
            
            elif action == ResponseAction.ALERT_ADMIN:
                await self._send_admin_alert(event)
            
            elif action == ResponseAction.EMERGENCY_SHUTDOWN:
                await self._trigger_emergency_shutdown(event)
            
            self.logger.info(f"Action de réponse exécutée: {action.value} pour {event.event_id}")
            
        except Exception as exc:
            self.logger.error(f"Erreur exécution action {action.value}: {exc}")
    
    async def _initialize_ml_models(self):
        """Initialise les modèles ML pour la détection d'anomalies"""
        try:
            # Modèle d'isolation forest pour détection d'anomalies
            self.anomaly_detector = IsolationForest(
                contamination=0.1,  # 10% d'anomalies attendues
                random_state=42,
                n_estimators=100
            )
            
            # Scaler pour normaliser les features
            self.scaler = StandardScaler()
            
            # Charger un modèle pré-entraîné s'il existe
            try:
                self.anomaly_detector = joblib.load('anomaly_detector.pkl')
                self.scaler = joblib.load('scaler.pkl')
                self.logger.info("Modèles ML chargés depuis le disque")
            except FileNotFoundError:
                self.logger.info("Nouveaux modèles ML initialisés")
            
        except Exception as exc:
            self.logger.error(f"Erreur initialisation modèles ML: {exc}")
    
    async def _extract_behavioral_features(self, events: List[SecurityEvent]) -> np.ndarray:
        """Extrait les features comportementales des événements"""
        try:
            features = []
            
            for event in events:
                feature_vector = [
                    # Features temporelles
                    event.timestamp.hour,
                    event.timestamp.weekday(),
                    
                    # Features de géolocalisation
                    float(event.location.get("latitude", 0)) if event.location else 0,
                    float(event.location.get("longitude", 0)) if event.location else 0,
                    
                    # Features d'événement
                    len(EventType.__members__) if event.event_type else 0,
                    len(ThreatLevel.__members__) if event.threat_level else 0,
                    
                    # Features de session
                    len(event.user_agent) if event.user_agent else 0,
                    1 if event.details.get("is_mobile", False) else 0,
                    1 if event.details.get("is_bot", False) else 0,
                    
                    # Features de threat intel
                    event.details.get("threat_intel", {}).get("confidence", 0),
                ]
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as exc:
            self.logger.error(f"Erreur extraction features: {exc}")
            return np.array([])
    
    async def _detect_ml_anomalies(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Détecte les anomalies avec ML"""
        try:
            if len(features) == 0 or self.anomaly_detector is None:
                return []
            
            # Normaliser les features
            features_scaled = self.scaler.fit_transform(features)
            
            # Prédire les anomalies
            anomaly_predictions = self.anomaly_detector.predict(features_scaled)
            anomaly_scores = self.anomaly_detector.decision_function(features_scaled)
            
            # Identifier les anomalies
            anomalies = []
            for i, (prediction, score) in enumerate(zip(anomaly_predictions, anomaly_scores)):
                if prediction == -1:  # Anomalie détectée
                    anomalies.append({
                        "index": i,
                        "anomaly_score": float(score),
                        "confidence": abs(float(score))
                    })
            
            return anomalies
            
        except Exception as exc:
            self.logger.error(f"Erreur détection anomalies ML: {exc}")
            return []
    
    # Méthodes utilitaires
    async def _store_security_event(self, event: SecurityEvent):
        """Stocke un événement de sécurité"""
        try:
            event_data = json.dumps(asdict(event), default=str)
            
            # Stocker avec TTL
            ttl = self.event_retention_days * 86400
            await self.redis_client.setex(f"security_event:{event.event_id}", ttl, event_data)
            
            # Ajouter aux index
            await self.redis_client.zadd(
                "events_by_time",
                {event.event_id: event.timestamp.timestamp()}
            )
            
            await self.redis_client.sadd(f"events_by_ip:{event.ip_address}", event.event_id)
            
            if event.user_id:
                await self.redis_client.sadd(f"events_by_user:{event.user_id}", event.event_id)
            
        except Exception as exc:
            self.logger.error(f"Erreur stockage événement: {exc}")
    
    async def _get_user_events(self, user_id: str, time_window: int) -> List[SecurityEvent]:
        """Récupère les événements d'un utilisateur"""
        try:
            event_ids = await self.redis_client.smembers(f"events_by_user:{user_id}")
            events = []
            
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
            
            for event_id in event_ids:
                event_id_str = event_id.decode() if isinstance(event_id, bytes) else event_id
                event_data = await self.redis_client.get(f"security_event:{event_id_str}")
                
                if event_data:
                    event_dict = json.loads(event_data)
                    event_time = datetime.fromisoformat(event_dict["timestamp"])
                    
                    if event_time > cutoff_time:
                        events.append(SecurityEvent(**event_dict))
            
            return events
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération événements utilisateur: {exc}")
            return []
    
    async def _get_events_by_ip(self, ip_address: str, time_window: int) -> List[SecurityEvent]:
        """Récupère les événements par IP"""
        try:
            event_ids = await self.redis_client.smembers(f"events_by_ip:{ip_address}")
            events = []
            
            cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
            
            for event_id in event_ids:
                event_id_str = event_id.decode() if isinstance(event_id, bytes) else event_id
                event_data = await self.redis_client.get(f"security_event:{event_id_str}")
                
                if event_data:
                    event_dict = json.loads(event_data)
                    event_time = datetime.fromisoformat(event_dict["timestamp"])
                    
                    if event_time > cutoff_time:
                        events.append(SecurityEvent(**event_dict))
            
            return events
            
        except Exception as exc:
            self.logger.error(f"Erreur récupération événements IP: {exc}")
            return []
    
    async def _count_recent_failures(self, ip_address: str, user_id: Optional[str], window: int) -> int:
        """Compte les échecs de connexion récents"""
        try:
            # Implémentation simplifiée
            # Dans un vrai système, interroger la base de données d'événements
            return 0
            
        except Exception as exc:
            self.logger.error(f"Erreur comptage échecs récents: {exc}")
            return 0
    
    async def _query_external_threat_intel(
        self,
        indicator: str,
        indicator_type: str
    ) -> Optional[ThreatIndicator]:
        """Interroge les sources externes de threat intelligence"""
        try:
            # Implémentation simplifiée
            # Dans un vrai système, interroger VirusTotal, AbuseIPDB, etc.
            return None
            
        except Exception as exc:
            self.logger.error(f"Erreur requête threat intel externe: {exc}")
            return None
    
    # Actions de réponse
    async def _block_ip_temporarily(self, ip_address: str, duration: int):
        """Bloque temporairement une IP"""
        try:
            await self.redis_client.setex(f"blocked_ip:{ip_address}", duration, "temporary")
            self.logger.info(f"IP bloquée temporairement: {ip_address} pour {duration}s")
        except Exception as exc:
            self.logger.error(f"Erreur blocage temporaire IP: {exc}")
    
    async def _block_ip_permanently(self, ip_address: str):
        """Bloque définitivement une IP"""
        try:
            await self.redis_client.set(f"blocked_ip:{ip_address}", "permanent")
            self.logger.warning(f"IP bloquée définitivement: {ip_address}")
        except Exception as exc:
            self.logger.error(f"Erreur blocage permanent IP: {exc}")
    
    async def _apply_rate_limit(self, ip_address: str, user_id: Optional[str]):
        """Applique une limitation de taux"""
        try:
            # Implémentation simplifiée
            await self.redis_client.setex(f"rate_limit:{ip_address}", 3600, "restricted")
            self.logger.info(f"Limitation de taux appliquée: {ip_address}")
        except Exception as exc:
            self.logger.error(f"Erreur application rate limit: {exc}")
    
    async def _require_mfa_for_user(self, user_id: Optional[str]):
        """Exige l'authentification multifacteur"""
        try:
            if user_id:
                await self.redis_client.setex(f"require_mfa:{user_id}", 86400, "required")
                self.logger.info(f"MFA requis pour utilisateur: {user_id}")
        except Exception as exc:
            self.logger.error(f"Erreur exigence MFA: {exc}")
    
    async def _lockout_user_account(self, user_id: Optional[str]):
        """Verrouille un compte utilisateur"""
        try:
            if user_id:
                await self.redis_client.setex(f"locked_account:{user_id}", 86400, "locked")
                self.logger.warning(f"Compte utilisateur verrouillé: {user_id}")
        except Exception as exc:
            self.logger.error(f"Erreur verrouillage compte: {exc}")
    
    async def _send_admin_alert(self, event: SecurityEvent):
        """Envoie une alerte aux administrateurs"""
        try:
            alert = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_id": event.event_id,
                "threat_level": event.threat_level.value,
                "event_type": event.event_type.value,
                "ip_address": event.ip_address,
                "user_id": event.user_id,
                "details": event.details
            }
            
            await self.redis_client.lpush("admin_alerts", json.dumps(alert))
            self.logger.critical(f"Alerte admin envoyée pour événement: {event.event_id}")
        except Exception as exc:
            self.logger.error(f"Erreur envoi alerte admin: {exc}")
    
    async def _trigger_emergency_shutdown(self, event: SecurityEvent):
        """Déclenche un arrêt d'urgence"""
        try:
            # Implémentation dépendante de l'infrastructure
            self.logger.critical(f"ARRÊT D'URGENCE DÉCLENCHÉ pour événement: {event.event_id}")
            
            # Notifier tous les services
            emergency_signal = {
                "action": "emergency_shutdown",
                "reason": event.event_type.value,
                "timestamp": datetime.utcnow().isoformat(),
                "event_id": event.event_id
            }
            
            await self.redis_client.publish("emergency_channel", json.dumps(emergency_signal))
            
        except Exception as exc:
            self.logger.error(f"Erreur arrêt d'urgence: {exc}")
    
    # Méthodes utilitaires pour le dashboard
    async def _count_events_in_range(self, start_time: datetime, end_time: datetime) -> int:
        """Compte les événements dans une plage de temps"""
        try:
            # Implémentation simplifiée
            return 0
        except Exception:
            return 0
    
    async def _get_events_by_type(self, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """Récupère les événements par type"""
        try:
            # Implémentation simplifiée
            return {}
        except Exception:
            return {}
    
    async def _get_events_by_threat_level(self, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """Récupère les événements par niveau de menace"""
        try:
            # Implémentation simplifiée
            return {}
        except Exception:
            return {}
    
    async def _get_top_suspicious_ips(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Récupère les IPs les plus suspectes"""
        try:
            # Implémentation simplifiée
            return []
        except Exception:
            return []
    
    async def _get_active_incidents(self) -> List[Dict[str, Any]]:
        """Récupère les incidents actifs"""
        try:
            # Implémentation simplifiée
            return []
        except Exception:
            return []
    
    async def _get_hourly_event_trends(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Récupère les tendances horaires des événements"""
        try:
            # Implémentation simplifiée
            return []
        except Exception:
            return []
    
    async def _store_security_incident(self, incident: SecurityIncident):
        """Stocke un incident de sécurité"""
        try:
            incident_data = json.dumps(asdict(incident), default=str)
            await self.redis_client.set(f"security_incident:{incident.incident_id}", incident_data)
            
            # Ajouter à l'index des incidents actifs
            if incident.status in ["open", "investigating"]:
                await self.redis_client.sadd("active_incidents", incident.incident_id)
                
        except Exception as exc:
            self.logger.error(f"Erreur stockage incident: {exc}")
    
    async def _notify_security_team(self, incident: SecurityIncident):
        """Notifie l'équipe de sécurité"""
        try:
            notification = {
                "incident_id": incident.incident_id,
                "title": incident.title,
                "severity": incident.severity.value,
                "timestamp": incident.created_at.isoformat()
            }
            
            await self.redis_client.lpush("security_notifications", json.dumps(notification))
            
        except Exception as exc:
            self.logger.error(f"Erreur notification équipe sécurité: {exc}")
    
    async def _update_security_metrics(self, event: SecurityEvent):
        """Met à jour les métriques de sécurité"""
        try:
            # Incrémenter les compteurs globaux
            await self.redis_client.incr("security_metrics:total_events")
            await self.redis_client.incr(f"security_metrics:events_by_type:{event.event_type.value}")
            await self.redis_client.incr(f"security_metrics:events_by_threat:{event.threat_level.value}")
            
            # Métriques par heure
            hour_key = event.timestamp.strftime("%Y-%m-%d-%H")
            await self.redis_client.incr(f"security_metrics:hourly:{hour_key}")
            
        except Exception as exc:
            self.logger.error(f"Erreur mise à jour métriques: {exc}")
    
    def _map_event_to_attack_vector(self, event_type: EventType) -> Optional[AttackVector]:
        """Mappe un type d'événement vers un vecteur d'attaque"""
        mapping = {
            EventType.SQL_INJECTION: AttackVector.DATABASE,
            EventType.XSS_ATTEMPT: AttackVector.WEB_APPLICATION,
            EventType.BRUTE_FORCE: AttackVector.AUTHENTICATION,
            EventType.PERMISSION_ESCALATION: AttackVector.AUTHORIZATION,
            EventType.CSRF_ATTEMPT: AttackVector.WEB_APPLICATION,
            EventType.DDoS_ATTACK: AttackVector.NETWORK
        }
        
        return mapping.get(event_type)


class ComplianceMonitor:
    """Moniteur de conformité"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def check_gdpr_compliance(self, user_id: str) -> Dict[str, Any]:
        """Vérifie la conformité GDPR"""
        try:
            # Vérifier le consentement
            consent_status = await self._check_user_consent(user_id)
            
            # Vérifier la conservation des données
            data_retention = await self._check_data_retention(user_id)
            
            # Vérifier les droits des utilisateurs
            user_rights = await self._check_user_rights_compliance(user_id)
            
            return {
                "consent": consent_status,
                "data_retention": data_retention,
                "user_rights": user_rights,
                "overall_compliant": all([consent_status, data_retention, user_rights])
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur vérification conformité GDPR: {exc}")
            return {"overall_compliant": False, "error": str(exc)}
    
    async def _check_user_consent(self, user_id: str) -> bool:
        """Vérifie le consentement utilisateur"""
        # Implémentation simplifiée
        return True
    
    async def _check_data_retention(self, user_id: str) -> bool:
        """Vérifie la politique de conservation des données"""
        # Implémentation simplifiée
        return True
    
    async def _check_user_rights_compliance(self, user_id: str) -> bool:
        """Vérifie le respect des droits utilisateur"""
        # Implémentation simplifiée
        return True
