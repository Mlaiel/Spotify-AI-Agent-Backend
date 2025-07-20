"""
Système de Sécurité et Détection d'Intrusion Ultra-Avancé
=========================================================

Module de sécurité sophistiqué avec détection d'anomalies par ML, 
threat intelligence, et réponse automatique aux incidents pour 
l'architecture multi-tenant du Spotify AI Agent.
"""

import logging
import hashlib
import hmac
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import ipaddress
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Niveaux de menace standardisés"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SecurityEventType(Enum):
    """Types d'événements de sécurité"""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    DATA_ACCESS_ANOMALY = "data_access_anomaly"
    CROSS_TENANT_VIOLATION = "cross_tenant_violation"
    SUSPICIOUS_API_USAGE = "suspicious_api_usage"
    POTENTIAL_DDoS = "potential_ddos"
    MALWARE_DETECTION = "malware_detection"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    COMPLIANCE_VIOLATION = "compliance_violation"

@dataclass
class SecurityEvent:
    """Structure d'événement de sécurité"""
    event_id: str
    timestamp: datetime
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    tenant_id: Optional[str]
    resource: str
    description: str
    evidence: Dict[str, Any]
    confidence_score: float
    is_false_positive: bool = False
    response_actions: List[str] = None

class AdvancedSecurityMonitor:
    """Moniteur de sécurité ultra-avancé avec ML et threat intelligence"""
    
    def __init__(self):
        self.threat_intel_db = {}
        self.known_good_ips = set()
        self.blocked_ips = set()
        self.behavioral_baselines = {}
        self.ml_anomaly_detector = None
        self.security_rules = []
        self._initialize_security_system()
    
    def _initialize_security_system(self):
        """Initialise le système de sécurité ultra-avancé"""
        logger.info("Initialisation du système de sécurité ultra-avancé")
        
        # Chargement des bases de threat intelligence
        self._load_threat_intelligence()
        
        # Initialisation du détecteur d'anomalies ML
        self._initialize_ml_detector()
        
        # Chargement des règles de sécurité
        self._load_security_rules()
        
        # Configuration des signatures d'attaque
        self._load_attack_signatures()
    
    def _load_threat_intelligence(self):
        """Charge les bases de threat intelligence"""
        # Simulation de chargement de feeds de threat intel
        self.threat_intel_db = {
            "malicious_ips": {
                "192.168.1.100": {"type": "botnet", "confidence": 0.9, "last_seen": "2024-01-15"},
                "10.0.0.50": {"type": "scanner", "confidence": 0.7, "last_seen": "2024-01-14"}
            },
            "malicious_domains": {
                "evil.example.com": {"type": "c2_server", "confidence": 0.95},
                "phishing.fake.com": {"type": "phishing", "confidence": 0.8}
            },
            "known_attack_patterns": {
                "sql_injection": [
                    r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
                    r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
                    r"((\%27)|(\')).*((\%6F)|o|(\%4F))((\%72)|r|(\%52))"
                ],
                "xss_injection": [
                    r"<[^>]*script",
                    r"javascript:",
                    r"onload\s*=",
                    r"onerror\s*="
                ]
            }
        }
    
    def _initialize_ml_detector(self):
        """Initialise le détecteur d'anomalies par Machine Learning"""
        self.ml_anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # Scaler pour normaliser les features
        self.scaler = StandardScaler()
        
        logger.info("Détecteur d'anomalies ML initialisé")
    
    def _load_security_rules(self):
        """Charge les règles de sécurité sophistiquées"""
        self.security_rules = [
            {
                "name": "Multiple Authentication Failures",
                "pattern": "auth_failure_rate > 10 per minute per ip",
                "action": "block_ip_temporary",
                "threat_level": ThreatLevel.HIGH
            },
            {
                "name": "Cross-Tenant Data Access",
                "pattern": "tenant_access_violation",
                "action": "immediate_block",
                "threat_level": ThreatLevel.CRITICAL
            },
            {
                "name": "Unusual Data Volume Access",
                "pattern": "data_access_volume > baseline * 10",
                "action": "alert_and_monitor",
                "threat_level": ThreatLevel.MEDIUM
            },
            {
                "name": "Suspicious API Usage Pattern",
                "pattern": "api_calls_pattern != user_baseline",
                "action": "increase_monitoring",
                "threat_level": ThreatLevel.LOW
            }
        ]
    
    def _load_attack_signatures(self):
        """Charge les signatures d'attaques connues"""
        self.attack_signatures = {
            "brute_force": {
                "indicators": [
                    "rapid_login_attempts",
                    "password_spraying",
                    "credential_stuffing"
                ],
                "thresholds": {
                    "attempts_per_minute": 20,
                    "unique_passwords": 50,
                    "failure_rate": 0.9
                }
            },
            "data_exfiltration": {
                "indicators": [
                    "large_data_download",
                    "bulk_api_requests",
                    "unusual_export_patterns"
                ],
                "thresholds": {
                    "data_volume_mb": 1000,
                    "api_calls_per_minute": 100,
                    "export_frequency": 5
                }
            },
            "privilege_escalation": {
                "indicators": [
                    "admin_function_access",
                    "permission_change_requests",
                    "unauthorized_resource_access"
                ],
                "thresholds": {
                    "admin_attempts": 3,
                    "permission_requests": 10,
                    "resource_violations": 1
                }
            }
        }
    
    def analyze_security_event(self, event_data: Dict[str, Any]) -> SecurityEvent:
        """
        Analyse un événement de sécurité avec ML et threat intelligence
        
        Args:
            event_data: Données de l'événement à analyser
            
        Returns:
            SecurityEvent: Événement analysé avec niveau de menace
        """
        # Extraction des features pour l'analyse
        features = self._extract_security_features(event_data)
        
        # Vérification contre threat intelligence
        threat_intel_score = self._check_threat_intelligence(event_data)
        
        # Détection d'anomalies par ML
        ml_anomaly_score = self._detect_ml_anomaly(features)
        
        # Analyse des patterns d'attaque
        attack_pattern_score = self._analyze_attack_patterns(event_data)
        
        # Vérification des règles de sécurité
        rule_violations = self._check_security_rules(event_data)
        
        # Calcul du score de confiance global
        confidence_score = self._calculate_confidence_score(
            threat_intel_score, ml_anomaly_score, attack_pattern_score, rule_violations
        )
        
        # Détermination du niveau de menace
        threat_level = self._determine_threat_level(confidence_score, rule_violations)
        
        # Classification du type d'événement
        event_type = self._classify_event_type(event_data, features)
        
        # Génération des actions de réponse
        response_actions = self._generate_response_actions(threat_level, event_type, event_data)
        
        # Création de l'événement de sécurité
        security_event = SecurityEvent(
            event_id=self._generate_event_id(event_data),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=event_data.get("source_ip", "unknown"),
            user_id=event_data.get("user_id"),
            tenant_id=event_data.get("tenant_id"),
            resource=event_data.get("resource", "unknown"),
            description=self._generate_event_description(event_type, event_data),
            evidence={
                "raw_event": event_data,
                "threat_intel_score": threat_intel_score,
                "ml_anomaly_score": ml_anomaly_score,
                "attack_pattern_score": attack_pattern_score,
                "rule_violations": rule_violations,
                "extracted_features": features
            },
            confidence_score=confidence_score,
            response_actions=response_actions
        )
        
        return security_event
    
    def _extract_security_features(self, event_data: Dict[str, Any]) -> np.ndarray:
        """Extrait les features pour l'analyse ML"""
        features = []
        
        # Features temporelles
        hour = datetime.now().hour
        features.extend([
            hour,
            1 if 22 <= hour or hour <= 6 else 0,  # Heure suspecte
            datetime.now().weekday()
        ])
        
        # Features de requête
        features.extend([
            len(event_data.get("user_agent", "")),
            event_data.get("request_size", 0),
            event_data.get("response_size", 0),
            event_data.get("request_duration", 0)
        ])
        
        # Features de géolocalisation (simulation)
        source_ip = event_data.get("source_ip", "")
        features.extend([
            self._get_geo_risk_score(source_ip),
            1 if self._is_vpn_tor_ip(source_ip) else 0,
            1 if source_ip in self.threat_intel_db.get("malicious_ips", {}) else 0
        ])
        
        # Features comportementales
        user_id = event_data.get("user_id")
        if user_id and user_id in self.behavioral_baselines:
            baseline = self.behavioral_baselines[user_id]
            features.extend([
                abs(event_data.get("api_calls_per_hour", 0) - baseline.get("avg_api_calls", 0)),
                abs(event_data.get("data_accessed_mb", 0) - baseline.get("avg_data_access", 0)),
                1 if event_data.get("new_device", False) else 0
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def _check_threat_intelligence(self, event_data: Dict[str, Any]) -> float:
        """Vérifie l'événement contre les bases de threat intelligence"""
        score = 0.0
        
        source_ip = event_data.get("source_ip", "")
        if source_ip in self.threat_intel_db.get("malicious_ips", {}):
            intel_data = self.threat_intel_db["malicious_ips"][source_ip]
            score += intel_data.get("confidence", 0.5)
        
        # Vérification des domaines suspects dans les headers
        user_agent = event_data.get("user_agent", "")
        referer = event_data.get("referer", "")
        
        for domain in self.threat_intel_db.get("malicious_domains", {}):
            if domain in user_agent or domain in referer:
                intel_data = self.threat_intel_db["malicious_domains"][domain]
                score += intel_data.get("confidence", 0.3)
        
        return min(score, 1.0)
    
    def _detect_ml_anomaly(self, features: np.ndarray) -> float:
        """Détecte les anomalies avec le modèle ML"""
        try:
            if hasattr(self.ml_anomaly_detector, 'decision_function'):
                # Normalisation des features
                features_scaled = self.scaler.transform(features)
                
                # Score d'anomalie (plus négatif = plus anormal)
                anomaly_score = self.ml_anomaly_detector.decision_function(features_scaled)[0]
                
                # Conversion en score 0-1 (plus élevé = plus anormal)
                normalized_score = max(0, (0.5 - anomaly_score) / 0.5)
                return min(normalized_score, 1.0)
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Erreur détection anomalie ML: {e}")
            return 0.0
    
    def _analyze_attack_patterns(self, event_data: Dict[str, Any]) -> float:
        """Analyse les patterns d'attaque connus"""
        max_score = 0.0
        
        request_data = event_data.get("request_data", "")
        
        # Vérification des patterns SQL injection
        for pattern in self.threat_intel_db["known_attack_patterns"]["sql_injection"]:
            if re.search(pattern, request_data, re.IGNORECASE):
                max_score = max(max_score, 0.8)
        
        # Vérification des patterns XSS
        for pattern in self.threat_intel_db["known_attack_patterns"]["xss_injection"]:
            if re.search(pattern, request_data, re.IGNORECASE):
                max_score = max(max_score, 0.7)
        
        # Vérification des signatures d'attaque avancées
        for attack_type, signature in self.attack_signatures.items():
            if self._matches_attack_signature(event_data, signature):
                max_score = max(max_score, 0.9)
        
        return max_score
    
    def _check_security_rules(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Vérifie les règles de sécurité"""
        violations = []
        
        for rule in self.security_rules:
            if self._evaluate_security_rule(event_data, rule):
                violations.append({
                    "rule_name": rule["name"],
                    "threat_level": rule["threat_level"],
                    "action": rule["action"],
                    "details": self._get_rule_violation_details(event_data, rule)
                })
        
        return violations
    
    def _calculate_confidence_score(self, threat_intel_score: float, 
                                   ml_anomaly_score: float,
                                   attack_pattern_score: float,
                                   rule_violations: List[Dict[str, Any]]) -> float:
        """Calcule le score de confiance global"""
        # Pondération des différents scores
        weights = {
            "threat_intel": 0.3,
            "ml_anomaly": 0.2,
            "attack_pattern": 0.3,
            "rule_violations": 0.2
        }
        
        # Score des violations de règles
        violation_score = min(len(rule_violations) * 0.2, 1.0)
        
        # Calcul pondéré
        confidence = (
            threat_intel_score * weights["threat_intel"] +
            ml_anomaly_score * weights["ml_anomaly"] +
            attack_pattern_score * weights["attack_pattern"] +
            violation_score * weights["rule_violations"]
        )
        
        return min(confidence, 1.0)
    
    def _determine_threat_level(self, confidence_score: float, 
                               rule_violations: List[Dict[str, Any]]) -> ThreatLevel:
        """Détermine le niveau de menace"""
        # Vérification des violations critiques
        for violation in rule_violations:
            if violation["threat_level"] == ThreatLevel.CRITICAL:
                return ThreatLevel.CRITICAL
            elif violation["threat_level"] == ThreatLevel.HIGH and confidence_score > 0.7:
                return ThreatLevel.HIGH
        
        # Basé sur le score de confiance
        if confidence_score >= 0.9:
            return ThreatLevel.CRITICAL
        elif confidence_score >= 0.7:
            return ThreatLevel.HIGH
        elif confidence_score >= 0.5:
            return ThreatLevel.MEDIUM
        elif confidence_score >= 0.3:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.LOW
    
    def _classify_event_type(self, event_data: Dict[str, Any], 
                            features: np.ndarray) -> SecurityEventType:
        """Classifie le type d'événement de sécurité"""
        # Classification basée sur les patterns dans les données
        if "login" in event_data.get("endpoint", "").lower():
            if event_data.get("status_code") in [401, 403]:
                return SecurityEventType.AUTHENTICATION_FAILURE
        
        if "admin" in event_data.get("endpoint", "").lower():
            return SecurityEventType.PRIVILEGE_ESCALATION
        
        if event_data.get("cross_tenant_access", False):
            return SecurityEventType.CROSS_TENANT_VIOLATION
        
        if event_data.get("data_volume_mb", 0) > 100:
            return SecurityEventType.DATA_EXFILTRATION
        
        # Classification par défaut basée sur les features
        return SecurityEventType.SUSPICIOUS_API_USAGE
    
    def _generate_response_actions(self, threat_level: ThreatLevel, 
                                  event_type: SecurityEventType,
                                  event_data: Dict[str, Any]) -> List[str]:
        """Génère les actions de réponse automatiques"""
        actions = []
        
        if threat_level == ThreatLevel.CRITICAL:
            actions.extend([
                "immediate_ip_block",
                "revoke_user_tokens",
                "alert_security_team",
                "create_incident_ticket"
            ])
        elif threat_level == ThreatLevel.HIGH:
            actions.extend([
                "temporary_ip_block",
                "increase_monitoring",
                "alert_security_team"
            ])
        elif threat_level == ThreatLevel.MEDIUM:
            actions.extend([
                "rate_limit_user",
                "increase_monitoring",
                "log_detailed_audit"
            ])
        
        # Actions spécifiques par type d'événement
        if event_type == SecurityEventType.CROSS_TENANT_VIOLATION:
            actions.extend([
                "isolate_tenant_data",
                "audit_permissions",
                "notify_compliance_team"
            ])
        elif event_type == SecurityEventType.DATA_EXFILTRATION:
            actions.extend([
                "block_data_access",
                "forensic_investigation",
                "notify_data_protection_officer"
            ])
        
        return list(set(actions))  # Suppression des doublons
    
    # Méthodes utilitaires
    def _generate_event_id(self, event_data: Dict[str, Any]) -> str:
        """Génère un ID unique pour l'événement"""
        data_str = json.dumps(event_data, sort_keys=True)
        timestamp = str(datetime.utcnow().timestamp())
        return hashlib.sha256((data_str + timestamp).encode()).hexdigest()[:16]
    
    def _generate_event_description(self, event_type: SecurityEventType, 
                                   event_data: Dict[str, Any]) -> str:
        """Génère une description claire de l'événement"""
        descriptions = {
            SecurityEventType.AUTHENTICATION_FAILURE: "Échec d'authentification suspect détecté",
            SecurityEventType.CROSS_TENANT_VIOLATION: "Violation d'isolation multi-tenant détectée",
            SecurityEventType.DATA_EXFILTRATION: "Tentative potentielle d'exfiltration de données",
            SecurityEventType.PRIVILEGE_ESCALATION: "Tentative d'élévation de privilèges détectée"
        }
        
        base_desc = descriptions.get(event_type, "Activité suspecte détectée")
        source_ip = event_data.get("source_ip", "IP inconnue")
        user_id = event_data.get("user_id", "utilisateur inconnu")
        
        return f"{base_desc} depuis {source_ip} pour {user_id}"
    
    def _get_geo_risk_score(self, ip: str) -> float:
        """Calcule un score de risque géographique (simulation)"""
        # Simulation basée sur des plages d'IP
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private:
                return 0.1  # IP privée = risque faible
            # Simulation de scoring géographique
            return hash(ip) % 100 / 100.0
        except:
            return 0.5
    
    def _is_vpn_tor_ip(self, ip: str) -> bool:
        """Vérifie si l'IP est un VPN/Tor (simulation)"""
        # Dans un vrai système, cela interrogerait des bases de données spécialisées
        suspicious_ranges = ["192.168.100.", "10.0.50.", "172.16.20."]
        return any(ip.startswith(range_ip) for range_ip in suspicious_ranges)
    
    def _matches_attack_signature(self, event_data: Dict[str, Any], 
                                 signature: Dict[str, Any]) -> bool:
        """Vérifie si l'événement correspond à une signature d'attaque"""
        # Implémentation simplifiée
        indicators = signature.get("indicators", [])
        thresholds = signature.get("thresholds", {})
        
        matches = 0
        for indicator in indicators:
            if indicator in str(event_data):
                matches += 1
        
        return matches >= len(indicators) * 0.6  # 60% des indicateurs
    
    def _evaluate_security_rule(self, event_data: Dict[str, Any], 
                               rule: Dict[str, Any]) -> bool:
        """Évalue une règle de sécurité"""
        # Implémentation simplifiée d'évaluation de règles
        pattern = rule.get("pattern", "")
        
        if "auth_failure_rate" in pattern:
            return event_data.get("auth_failures_per_minute", 0) > 10
        elif "tenant_access_violation" in pattern:
            return event_data.get("cross_tenant_access", False)
        elif "data_access_volume" in pattern:
            baseline = event_data.get("baseline_data_access", 10)
            return event_data.get("data_accessed_mb", 0) > baseline * 10
        
        return False
    
    def _get_rule_violation_details(self, event_data: Dict[str, Any], 
                                   rule: Dict[str, Any]) -> Dict[str, Any]:
        """Récupère les détails d'une violation de règle"""
        return {
            "rule_pattern": rule.get("pattern", ""),
            "event_values": {k: v for k, v in event_data.items() if k in rule.get("pattern", "")},
            "threshold_exceeded": True
        }

# Instance globale du moniteur de sécurité
security_monitor = AdvancedSecurityMonitor()
