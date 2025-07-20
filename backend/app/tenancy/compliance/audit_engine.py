"""
Spotify AI Agent - AuditEngine Ultra-Avancé
==========================================

Moteur d'audit en temps réel avec intelligence artificielle intégrée
pour surveillance continue, détection d'anomalies et génération automatique de preuves.

Développé par l'équipe d'experts Audit & Compliance
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from uuid import uuid4
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class AuditEventType(Enum):
    """Types d'événements d'audit"""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    USER_AUTHENTICATION = "user_authentication"
    PERMISSION_CHANGE = "permission_change"
    CONSENT_CHANGE = "consent_change"
    POLICY_VIOLATION = "policy_violation"
    SYSTEM_CONFIGURATION = "system_configuration"
    BREACH_DETECTION = "breach_detection"
    REGULATORY_ACTION = "regulatory_action"
    MUSIC_CONTENT_ACCESS = "music_content_access"
    ROYALTY_CALCULATION = "royalty_calculation"
    CROSS_BORDER_TRANSFER = "cross_border_transfer"

class AuditSeverity(Enum):
    """Niveaux de sévérité d'audit"""
    CRITICAL = 10
    HIGH = 8
    MEDIUM = 6
    LOW = 4
    INFO = 2

class ComplianceFramework(Enum):
    """Frameworks de conformité auditables"""
    GDPR = "gdpr"
    SOX = "sox"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    MUSIC_INDUSTRY = "music_industry"

@dataclass
class AuditEvent:
    """Événement d'audit immutable"""
    event_id: str
    tenant_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: str
    user_agent: str
    resource_accessed: str
    action_performed: str
    outcome: str
    compliance_frameworks: List[ComplianceFramework]
    
    # Données contextuelles
    request_payload: Dict[str, Any] = field(default_factory=dict)
    response_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Informations de géolocalisation
    country_code: Optional[str] = None
    region: Optional[str] = None
    
    # Hash d'intégrité
    integrity_hash: Optional[str] = None
    
    def __post_init__(self):
        """Calcul automatique du hash d'intégrité"""
        if self.integrity_hash is None:
            self.integrity_hash = self._calculate_integrity_hash()
    
    def _calculate_integrity_hash(self) -> str:
        """Calcul du hash d'intégrité pour prévenir la falsification"""
        event_data = {
            'event_id': self.event_id,
            'tenant_id': self.tenant_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'resource_accessed': self.resource_accessed,
            'action_performed': self.action_performed,
            'outcome': self.outcome
        }
        
        return hashlib.sha256(
            json.dumps(event_data, sort_keys=True).encode()
        ).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Vérification de l'intégrité de l'événement"""
        current_hash = self.integrity_hash
        self.integrity_hash = None
        calculated_hash = self._calculate_integrity_hash()
        self.integrity_hash = current_hash
        
        return current_hash == calculated_hash

@dataclass
class AuditTrail:
    """Piste d'audit sécurisée et immuable"""
    trail_id: str
    tenant_id: str
    framework: ComplianceFramework
    events: List[AuditEvent] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    chain_hash: Optional[str] = None
    
    def add_event(self, event: AuditEvent):
        """Ajout d'un événement avec chaînage cryptographique"""
        self.events.append(event)
        self.last_updated = datetime.utcnow()
        self._update_chain_hash()
    
    def _update_chain_hash(self):
        """Mise à jour du hash de chaînage"""
        if not self.events:
            self.chain_hash = hashlib.sha256(f"{self.trail_id}_{self.created_at}".encode()).hexdigest()
        else:
            previous_hash = self.chain_hash or ""
            last_event_hash = self.events[-1].integrity_hash
            combined = f"{previous_hash}_{last_event_hash}_{self.last_updated}"
            self.chain_hash = hashlib.sha256(combined.encode()).hexdigest()
    
    def verify_chain_integrity(self) -> bool:
        """Vérification de l'intégrité de la chaîne"""
        for event in self.events:
            if not event.verify_integrity():
                return False
        
        # Recalcul du hash de chaîne
        original_hash = self.chain_hash
        self.chain_hash = None
        self._update_chain_hash()
        result = self.chain_hash == original_hash
        self.chain_hash = original_hash
        
        return result

class AnomalyDetector:
    """
    Détecteur d'anomalies utilisant l'IA pour identifier les comportements suspects
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.logger = logging.getLogger(f"audit.anomaly.{tenant_id}")
        
        # Modèles de détection
        self._baseline_patterns = {}
        self._anomaly_thresholds = {
            'access_frequency': 3.0,      # Écart-type
            'unusual_time': 2.5,          # Accès hors heures habituelles
            'geographic_anomaly': 2.0,     # Géolocalisation inhabituelle
            'privilege_escalation': 1.0,   # Élévation de privilèges
            'bulk_operations': 2.0         # Opérations en masse
        }
        
        # Historique des comportements
        self._user_behavior_history = defaultdict(list)
        self._system_baseline = {}
    
    async def detect_anomalies(self, event: AuditEvent) -> List[Dict[str, Any]]:
        """Détection d'anomalies en temps réel"""
        
        anomalies = []
        
        # Détection d'anomalies d'accès
        access_anomaly = await self._detect_access_anomaly(event)
        if access_anomaly:
            anomalies.append(access_anomaly)
        
        # Détection d'anomalies temporelles
        temporal_anomaly = await self._detect_temporal_anomaly(event)
        if temporal_anomaly:
            anomalies.append(temporal_anomaly)
        
        # Détection d'anomalies géographiques
        geographic_anomaly = await self._detect_geographic_anomaly(event)
        if geographic_anomaly:
            anomalies.append(geographic_anomaly)
        
        # Détection d'élévation de privilèges
        privilege_anomaly = await self._detect_privilege_anomaly(event)
        if privilege_anomaly:
            anomalies.append(privilege_anomaly)
        
        # Détection d'opérations en masse
        bulk_anomaly = await self._detect_bulk_operations(event)
        if bulk_anomaly:
            anomalies.append(bulk_anomaly)
        
        # Mise à jour de l'historique
        await self._update_behavior_history(event)
        
        return anomalies
    
    async def _detect_access_anomaly(self, event: AuditEvent) -> Optional[Dict[str, Any]]:
        """Détection d'anomalies d'accès aux ressources"""
        
        if not event.user_id:
            return None
        
        # Analyse des patterns d'accès habituels
        user_history = self._user_behavior_history[event.user_id]
        
        if len(user_history) < 10:  # Pas assez d'historique
            return None
        
        # Calcul de la fréquence d'accès normale
        recent_accesses = [e for e in user_history[-100:] 
                          if e['resource_accessed'] == event.resource_accessed]
        
        if len(recent_accesses) == 0:
            # Première fois qu'il accède à cette ressource
            if event.severity.value >= AuditSeverity.HIGH.value:
                return {
                    'type': 'first_time_access',
                    'severity': 'medium',
                    'description': f"Premier accès à la ressource {event.resource_accessed}",
                    'confidence': 0.7
                }
        
        # Analyse de la fréquence
        time_intervals = []
        for i in range(1, len(recent_accesses)):
            interval = (recent_accesses[i]['timestamp'] - recent_accesses[i-1]['timestamp']).total_seconds()
            time_intervals.append(interval)
        
        if time_intervals:
            mean_interval = np.mean(time_intervals)
            std_interval = np.std(time_intervals)
            
            # Calcul de l'intervalle actuel
            last_access = max(recent_accesses, key=lambda x: x['timestamp'])
            current_interval = (event.timestamp - last_access['timestamp']).total_seconds()
            
            # Détection d'anomalie
            if std_interval > 0:
                z_score = abs(current_interval - mean_interval) / std_interval
                if z_score > self._anomaly_thresholds['access_frequency']:
                    return {
                        'type': 'unusual_access_frequency',
                        'severity': 'high' if z_score > 4.0 else 'medium',
                        'description': f"Fréquence d'accès inhabituelle (z-score: {z_score:.2f})",
                        'confidence': min(z_score / 5.0, 0.95)
                    }
        
        return None
    
    async def _detect_temporal_anomaly(self, event: AuditEvent) -> Optional[Dict[str, Any]]:
        """Détection d'anomalies temporelles"""
        
        if not event.user_id:
            return None
        
        user_history = self._user_behavior_history[event.user_id]
        
        if len(user_history) < 20:
            return None
        
        # Analyse des heures d'activité habituelles
        activity_hours = [e['timestamp'].hour for e in user_history[-100:]]
        
        if activity_hours:
            current_hour = event.timestamp.hour
            
            # Calcul de la distribution des heures
            hour_counts = np.bincount(activity_hours, minlength=24)
            hour_probabilities = hour_counts / np.sum(hour_counts)
            
            # Détection d'heure inhabituelle
            if hour_probabilities[current_hour] < 0.05:  # Moins de 5% d'activité habituelle
                return {
                    'type': 'unusual_time_access',
                    'severity': 'medium',
                    'description': f"Accès à une heure inhabituelle: {current_hour}h",
                    'confidence': 1.0 - hour_probabilities[current_hour] * 20
                }
        
        return None
    
    async def _detect_geographic_anomaly(self, event: AuditEvent) -> Optional[Dict[str, Any]]:
        """Détection d'anomalies géographiques"""
        
        if not event.user_id or not event.country_code:
            return None
        
        user_history = self._user_behavior_history[event.user_id]
        
        # Analyse des pays d'origine habituels
        recent_countries = [e.get('country_code') for e in user_history[-50:] 
                           if e.get('country_code')]
        
        if recent_countries:
            unique_countries = set(recent_countries)
            
            # Nouveau pays
            if event.country_code not in unique_countries:
                return {
                    'type': 'new_geographic_location',
                    'severity': 'high',
                    'description': f"Accès depuis un nouveau pays: {event.country_code}",
                    'confidence': 0.8
                }
            
            # Changement rapide de géolocalisation
            last_country_event = None
            for e in reversed(user_history):
                if e.get('country_code'):
                    last_country_event = e
                    break
            
            if (last_country_event and 
                last_country_event['country_code'] != event.country_code and
                (event.timestamp - last_country_event['timestamp']).total_seconds() < 3600):  # 1 heure
                
                return {
                    'type': 'rapid_geographic_change',
                    'severity': 'critical',
                    'description': f"Changement géographique rapide: {last_country_event['country_code']} -> {event.country_code}",
                    'confidence': 0.9
                }
        
        return None
    
    async def _detect_privilege_anomaly(self, event: AuditEvent) -> Optional[Dict[str, Any]]:
        """Détection d'élévation de privilèges"""
        
        if not event.user_id:
            return None
        
        # Analyse des actions habituelles de l'utilisateur
        user_history = self._user_behavior_history[event.user_id]
        recent_actions = [e['action_performed'] for e in user_history[-50:]]
        
        # Actions privilégiées
        privileged_actions = [
            'admin_access', 'user_management', 'system_configuration',
            'data_export', 'bulk_deletion', 'security_configuration'
        ]
        
        if event.action_performed in privileged_actions:
            # Vérifier si l'utilisateur fait habituellement ce type d'action
            if event.action_performed not in recent_actions:
                return {
                    'type': 'privilege_escalation',
                    'severity': 'critical',
                    'description': f"Action privilégiée inhabituelle: {event.action_performed}",
                    'confidence': 0.85
                }
        
        return None
    
    async def _detect_bulk_operations(self, event: AuditEvent) -> Optional[Dict[str, Any]]:
        """Détection d'opérations en masse suspectes"""
        
        if not event.user_id:
            return None
        
        # Compter les opérations récentes du même type
        recent_events = [e for e in self._user_behavior_history[event.user_id][-20:]
                        if (event.timestamp - e['timestamp']).total_seconds() < 300]  # 5 minutes
        
        same_action_count = len([e for e in recent_events 
                               if e['action_performed'] == event.action_performed])
        
        # Seuils pour opérations en masse
        bulk_thresholds = {
            'data_access': 50,
            'data_modification': 20,
            'data_deletion': 10,
            'download': 30
        }
        
        threshold = bulk_thresholds.get(event.action_performed, 25)
        
        if same_action_count > threshold:
            return {
                'type': 'bulk_operations',
                'severity': 'high',
                'description': f"Opérations en masse détectées: {same_action_count} {event.action_performed}",
                'confidence': 0.8
            }
        
        return None
    
    async def _update_behavior_history(self, event: AuditEvent):
        """Mise à jour de l'historique comportemental"""
        
        if event.user_id:
            event_data = {
                'timestamp': event.timestamp,
                'event_type': event.event_type.value,
                'resource_accessed': event.resource_accessed,
                'action_performed': event.action_performed,
                'country_code': event.country_code,
                'ip_address': event.ip_address
            }
            
            # Maintenir un historique limité (derniers 1000 événements)
            user_history = self._user_behavior_history[event.user_id]
            user_history.append(event_data)
            
            if len(user_history) > 1000:
                user_history.pop(0)

class ComplianceAuditor:
    """
    Auditeur de conformité spécialisé par framework
    """
    
    def __init__(self, framework: ComplianceFramework, tenant_id: str):
        self.framework = framework
        self.tenant_id = tenant_id
        self.logger = logging.getLogger(f"audit.compliance.{framework.value}.{tenant_id}")
        
        # Règles de conformité spécifiques au framework
        self._compliance_rules = self._load_framework_rules()
        self._violation_patterns = self._load_violation_patterns()
    
    def _load_framework_rules(self) -> Dict[str, Any]:
        """Chargement des règles de conformité par framework"""
        
        if self.framework == ComplianceFramework.GDPR:
            return {
                'data_access_logging': {
                    'required': True,
                    'retention_period': timedelta(days=2555),  # 7 ans
                    'fields_required': ['user_id', 'data_subject', 'legal_basis', 'purpose']
                },
                'consent_tracking': {
                    'required': True,
                    'granular_consent': True,
                    'withdrawal_process': True
                },
                'breach_notification': {
                    'authority_deadline': timedelta(hours=72),
                    'subject_notification': True,
                    'high_risk_threshold': 'medium'
                }
            }
        
        elif self.framework == ComplianceFramework.SOX:
            return {
                'financial_controls': {
                    'segregation_of_duties': True,
                    'authorization_levels': True,
                    'audit_trail_immutable': True
                },
                'change_management': {
                    'approval_required': True,
                    'testing_required': True,
                    'rollback_procedures': True
                }
            }
        
        elif self.framework == ComplianceFramework.ISO27001:
            return {
                'access_control': {
                    'least_privilege': True,
                    'regular_review': timedelta(days=90),
                    'segregation_of_duties': True
                },
                'incident_management': {
                    'detection_procedures': True,
                    'response_procedures': True,
                    'recovery_procedures': True
                }
            }
        
        elif self.framework == ComplianceFramework.PCI_DSS:
            return {
                'cardholder_data': {
                    'encryption_required': True,
                    'access_logging': True,
                    'storage_restrictions': True
                },
                'network_security': {
                    'firewall_configuration': True,
                    'intrusion_detection': True,
                    'vulnerability_scanning': True
                }
            }
        
        elif self.framework == ComplianceFramework.MUSIC_INDUSTRY:
            return {
                'copyright_compliance': {
                    'license_verification': True,
                    'usage_tracking': True,
                    'royalty_calculation': True
                },
                'content_protection': {
                    'drm_enforcement': True,
                    'geographic_restrictions': True,
                    'anti_piracy_measures': True
                }
            }
        
        return {}
    
    def _load_violation_patterns(self) -> List[Dict[str, Any]]:
        """Chargement des patterns de violation"""
        
        base_patterns = [
            {
                'name': 'unauthorized_access',
                'pattern': 'access_denied_then_successful',
                'severity': AuditSeverity.HIGH,
                'time_window': timedelta(minutes=5)
            },
            {
                'name': 'privilege_abuse',
                'pattern': 'admin_action_unusual_time',
                'severity': AuditSeverity.CRITICAL,
                'time_window': timedelta(hours=1)
            }
        ]
        
        if self.framework == ComplianceFramework.GDPR:
            base_patterns.extend([
                {
                    'name': 'consent_bypass',
                    'pattern': 'data_processing_without_consent',
                    'severity': AuditSeverity.CRITICAL,
                    'time_window': timedelta(minutes=1)
                },
                {
                    'name': 'unlawful_transfer',
                    'pattern': 'cross_border_transfer_no_safeguards',
                    'severity': AuditSeverity.HIGH,
                    'time_window': timedelta(minutes=1)
                }
            ])
        
        elif self.framework == ComplianceFramework.MUSIC_INDUSTRY:
            base_patterns.extend([
                {
                    'name': 'unlicensed_content',
                    'pattern': 'content_access_without_license',
                    'severity': AuditSeverity.CRITICAL,
                    'time_window': timedelta(seconds=30)
                },
                {
                    'name': 'geographic_violation',
                    'pattern': 'content_access_restricted_region',
                    'severity': AuditSeverity.HIGH,
                    'time_window': timedelta(seconds=30)
                }
            ])
        
        return base_patterns
    
    async def audit_compliance(self, event: AuditEvent) -> List[Dict[str, Any]]:
        """Audit de conformité pour un événement"""
        
        violations = []
        
        # Vérification des règles de conformité
        for rule_name, rule_config in self._compliance_rules.items():
            violation = await self._check_compliance_rule(event, rule_name, rule_config)
            if violation:
                violations.append(violation)
        
        # Détection de patterns de violation
        pattern_violations = await self._detect_violation_patterns(event)
        violations.extend(pattern_violations)
        
        return violations
    
    async def _check_compliance_rule(
        self,
        event: AuditEvent,
        rule_name: str,
        rule_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Vérification d'une règle de conformité spécifique"""
        
        # Exemple de vérification GDPR
        if self.framework == ComplianceFramework.GDPR and rule_name == 'data_access_logging':
            required_fields = rule_config.get('fields_required', [])
            
            missing_fields = []
            for field in required_fields:
                if field not in event.metadata:
                    missing_fields.append(field)
            
            if missing_fields:
                return {
                    'type': 'gdpr_logging_incomplete',
                    'rule': rule_name,
                    'severity': 'high',
                    'description': f"Champs manquants dans le log GDPR: {missing_fields}",
                    'framework': self.framework.value
                }
        
        # Exemple de vérification SOX
        elif self.framework == ComplianceFramework.SOX and rule_name == 'financial_controls':
            if event.event_type in [AuditEventType.DATA_MODIFICATION, AuditEventType.DATA_DELETION]:
                if not event.metadata.get('approval_id'):
                    return {
                        'type': 'sox_unauthorized_financial_change',
                        'rule': rule_name,
                        'severity': 'critical',
                        'description': "Modification financière sans approbation",
                        'framework': self.framework.value
                    }
        
        # Exemple de vérification Industrie Musicale
        elif self.framework == ComplianceFramework.MUSIC_INDUSTRY and rule_name == 'copyright_compliance':
            if event.event_type == AuditEventType.MUSIC_CONTENT_ACCESS:
                if not event.metadata.get('license_verified'):
                    return {
                        'type': 'copyright_violation',
                        'rule': rule_name,
                        'severity': 'critical',
                        'description': "Accès à du contenu musical sans licence vérifiée",
                        'framework': self.framework.value
                    }
        
        return None
    
    async def _detect_violation_patterns(self, event: AuditEvent) -> List[Dict[str, Any]]:
        """Détection de patterns de violation complexes"""
        
        violations = []
        
        # Implémentation simplifiée - dans un vrai système, 
        # cela analyserait les patterns temporels complexes
        for pattern in self._violation_patterns:
            if await self._matches_pattern(event, pattern):
                violations.append({
                    'type': f"pattern_{pattern['name']}",
                    'pattern': pattern['name'],
                    'severity': pattern['severity'].name.lower(),
                    'description': f"Pattern de violation détecté: {pattern['name']}",
                    'framework': self.framework.value
                })
        
        return violations
    
    async def _matches_pattern(self, event: AuditEvent, pattern: Dict[str, Any]) -> bool:
        """Vérification si un événement correspond à un pattern de violation"""
        
        pattern_name = pattern['name']
        
        if pattern_name == 'unauthorized_access':
            return (event.event_type == AuditEventType.DATA_ACCESS and 
                   event.outcome == 'denied')
        
        elif pattern_name == 'consent_bypass':
            return (event.event_type == AuditEventType.DATA_ACCESS and 
                   not event.metadata.get('consent_verified', False))
        
        elif pattern_name == 'unlicensed_content':
            return (event.event_type == AuditEventType.MUSIC_CONTENT_ACCESS and 
                   not event.metadata.get('license_valid', False))
        
        return False

class AuditEngine:
    """
    Moteur d'audit central ultra-avancé
    
    Fonctionnalités principales:
    - Collecte d'événements en temps réel
    - Détection d'anomalies par IA
    - Audit de conformité multi-framework
    - Génération de rapports automatisés
    - Alertes intelligentes
    """
    
    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.logger = logging.getLogger(f"audit.engine.{tenant_id}")
        
        # Composants spécialisés
        self.anomaly_detector = AnomalyDetector(tenant_id)
        self.compliance_auditors = {
            framework: ComplianceAuditor(framework, tenant_id)
            for framework in ComplianceFramework
        }
        
        # Stockage des événements et pistes d'audit
        self._audit_trails: Dict[ComplianceFramework, AuditTrail] = {}
        self._recent_events = deque(maxlen=10000)  # Buffer des événements récents
        
        # Configuration
        self._config = {
            'real_time_processing': True,
            'anomaly_detection_enabled': True,
            'compliance_checking_enabled': True,
            'alert_threshold': AuditSeverity.HIGH,
            'retention_period': timedelta(days=2555),  # 7 ans
            'batch_processing_size': 100,
            'alert_cooldown': timedelta(minutes=5)
        }
        
        # Alertes et métriques
        self._active_alerts = {}
        self._metrics = {
            'events_processed': 0,
            'anomalies_detected': 0,
            'violations_found': 0,
            'alerts_generated': 0
        }
        
        # Pool d'exécution pour traitement asynchrone
        self._executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialisation des pistes d'audit
        self._initialize_audit_trails()
        
        self.logger.info(f"AuditEngine initialisé pour tenant {tenant_id}")
    
    def _initialize_audit_trails(self):
        """Initialisation des pistes d'audit par framework"""
        
        for framework in ComplianceFramework:
            trail_id = f"{self.tenant_id}_{framework.value}_{datetime.utcnow().strftime('%Y%m%d')}"
            
            self._audit_trails[framework] = AuditTrail(
                trail_id=trail_id,
                tenant_id=self.tenant_id,
                framework=framework
            )
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        resource_accessed: str,
        action_performed: str,
        outcome: str,
        ip_address: str,
        user_agent: str = "",
        severity: AuditSeverity = AuditSeverity.INFO,
        compliance_frameworks: List[ComplianceFramework] = None,
        metadata: Dict[str, Any] = None,
        session_id: Optional[str] = None,
        country_code: Optional[str] = None,
        region: Optional[str] = None
    ) -> AuditEvent:
        """
        Enregistrement d'un événement d'audit avec traitement en temps réel
        """
        
        # Création de l'événement
        event = AuditEvent(
            event_id=str(uuid4()),
            tenant_id=self.tenant_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_accessed=resource_accessed,
            action_performed=action_performed,
            outcome=outcome,
            compliance_frameworks=compliance_frameworks or [],
            metadata=metadata or {},
            country_code=country_code,
            region=region
        )
        
        # Traitement en temps réel
        if self._config['real_time_processing']:
            await self._process_event_realtime(event)
        
        # Ajout aux pistes d'audit appropriées
        for framework in event.compliance_frameworks:
            if framework in self._audit_trails:
                self._audit_trails[framework].add_event(event)
        
        # Ajout au buffer des événements récents
        self._recent_events.append(event)
        
        # Mise à jour des métriques
        self._metrics['events_processed'] += 1
        
        self.logger.debug(f"Événement d'audit enregistré: {event.event_id}")
        
        return event
    
    async def _process_event_realtime(self, event: AuditEvent):
        """Traitement en temps réel d'un événement"""
        
        processing_tasks = []
        
        # Détection d'anomalies
        if self._config['anomaly_detection_enabled']:
            processing_tasks.append(self._process_anomaly_detection(event))
        
        # Audit de conformité
        if self._config['compliance_checking_enabled']:
            processing_tasks.append(self._process_compliance_audit(event))
        
        # Exécution en parallèle
        if processing_tasks:
            results = await asyncio.gather(*processing_tasks, return_exceptions=True)
            
            # Traitement des résultats
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Erreur lors du traitement: {result}")
                elif result:
                    await self._handle_processing_result(event, result)
    
    async def _process_anomaly_detection(self, event: AuditEvent) -> List[Dict[str, Any]]:
        """Traitement de détection d'anomalies"""
        
        try:
            anomalies = await self.anomaly_detector.detect_anomalies(event)
            
            if anomalies:
                self._metrics['anomalies_detected'] += len(anomalies)
                
                for anomaly in anomalies:
                    anomaly['event_id'] = event.event_id
                    anomaly['detected_at'] = datetime.utcnow()
                    
                    # Génération d'alerte si nécessaire
                    if anomaly.get('severity') in ['high', 'critical']:
                        await self._generate_alert('anomaly', anomaly, event)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Erreur détection d'anomalies: {e}")
            return []
    
    async def _process_compliance_audit(self, event: AuditEvent) -> List[Dict[str, Any]]:
        """Traitement d'audit de conformité"""
        
        all_violations = []
        
        try:
            # Audit par chaque framework applicable
            for framework in event.compliance_frameworks:
                if framework in self.compliance_auditors:
                    auditor = self.compliance_auditors[framework]
                    violations = await auditor.audit_compliance(event)
                    
                    for violation in violations:
                        violation['event_id'] = event.event_id
                        violation['detected_at'] = datetime.utcnow()
                    
                    all_violations.extend(violations)
            
            if all_violations:
                self._metrics['violations_found'] += len(all_violations)
                
                # Génération d'alertes pour violations critiques
                for violation in all_violations:
                    if violation.get('severity') in ['high', 'critical']:
                        await self._generate_alert('compliance_violation', violation, event)
            
            return all_violations
            
        except Exception as e:
            self.logger.error(f"Erreur audit de conformité: {e}")
            return []
    
    async def _handle_processing_result(self, event: AuditEvent, result: List[Dict[str, Any]]):
        """Traitement des résultats d'analyse"""
        
        for item in result:
            # Enrichissement avec contexte de l'événement
            item.update({
                'tenant_id': self.tenant_id,
                'source_event': {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'user_id': event.user_id,
                    'resource': event.resource_accessed
                }
            })
            
            # Stockage pour analyse ultérieure
            await self._store_analysis_result(item)
    
    async def _generate_alert(self, alert_type: str, data: Dict[str, Any], event: AuditEvent):
        """Génération d'alerte intelligente"""
        
        alert_key = f"{alert_type}_{event.user_id}_{data.get('type', 'unknown')}"
        
        # Vérification du cooldown
        if alert_key in self._active_alerts:
            last_alert = self._active_alerts[alert_key]
            if datetime.utcnow() - last_alert < self._config['alert_cooldown']:
                return  # Alerte en cooldown
        
        # Création de l'alerte
        alert = {
            'alert_id': str(uuid4()),
            'type': alert_type,
            'severity': data.get('severity', 'medium'),
            'title': self._generate_alert_title(alert_type, data),
            'description': data.get('description', 'Anomalie détectée'),
            'event_id': event.event_id,
            'user_id': event.user_id,
            'resource': event.resource_accessed,
            'timestamp': datetime.utcnow(),
            'confidence': data.get('confidence', 0.5),
            'raw_data': data,
            'tenant_id': self.tenant_id
        }
        
        # Enregistrement de l'alerte
        self._active_alerts[alert_key] = datetime.utcnow()
        self._metrics['alerts_generated'] += 1
        
        # Notification (simulée)
        await self._send_alert_notification(alert)
        
        self.logger.warning(f"Alerte générée: {alert['title']} (ID: {alert['alert_id']})")
    
    def _generate_alert_title(self, alert_type: str, data: Dict[str, Any]) -> str:
        """Génération du titre d'alerte"""
        
        severity = data.get('severity', 'medium').upper()
        
        if alert_type == 'anomaly':
            return f"[{severity}] Anomalie comportementale détectée"
        elif alert_type == 'compliance_violation':
            framework = data.get('framework', 'unknown').upper()
            return f"[{severity}] Violation de conformité {framework}"
        else:
            return f"[{severity}] Alerte de sécurité"
    
    async def _send_alert_notification(self, alert: Dict[str, Any]):
        """Envoi de notification d'alerte"""
        
        # Simulation d'envoi de notification
        # Dans un vrai système, cela enverrait des emails, SMS, webhooks, etc.
        
        if alert['severity'] in ['high', 'critical']:
            self.logger.critical(f"ALERTE CRITIQUE: {alert['title']}")
        else:
            self.logger.warning(f"ALERTE: {alert['title']}")
    
    async def _store_analysis_result(self, result: Dict[str, Any]):
        """Stockage des résultats d'analyse"""
        
        # Simulation de stockage
        # Dans un vrai système, cela irait en base de données
        pass
    
    async def generate_audit_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Génération de rapport d'audit pour un framework"""
        
        trail = self._audit_trails.get(framework)
        if not trail:
            return {'error': f"Piste d'audit non trouvée pour {framework.value}"}
        
        # Filtrage des événements par période
        filtered_events = [
            event for event in trail.events
            if start_date <= event.timestamp <= end_date
        ]
        
        # Analyse des événements
        analysis = await self._analyze_events(filtered_events, framework)
        
        # Construction du rapport
        report = {
            'framework': framework.value,
            'tenant_id': self.tenant_id,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'trail_info': {
                'trail_id': trail.trail_id,
                'total_events': len(trail.events),
                'filtered_events': len(filtered_events),
                'integrity_verified': trail.verify_chain_integrity()
            },
            'analysis': analysis,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return report
    
    async def _analyze_events(
        self,
        events: List[AuditEvent],
        framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Analyse des événements pour génération de rapport"""
        
        if not events:
            return {
                'summary': 'Aucun événement dans la période',
                'metrics': {},
                'recommendations': []
            }
        
        # Métriques de base
        event_types = defaultdict(int)
        severity_counts = defaultdict(int)
        user_activity = defaultdict(int)
        hourly_distribution = defaultdict(int)
        
        for event in events:
            event_types[event.event_type.value] += 1
            severity_counts[event.severity.name] += 1
            if event.user_id:
                user_activity[event.user_id] += 1
            hourly_distribution[event.timestamp.hour] += 1
        
        # Calculs avancés
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1])[0]
        most_active_user = max(user_activity.items(), key=lambda x: x[1])[0] if user_activity else None
        
        # Détection de patterns suspects
        suspicious_patterns = await self._detect_suspicious_patterns(events)
        
        # Recommandations
        recommendations = await self._generate_audit_recommendations(events, framework)
        
        return {
            'summary': f"Analyse de {len(events)} événements",
            'metrics': {
                'event_types': dict(event_types),
                'severity_distribution': dict(severity_counts),
                'unique_users': len(user_activity),
                'peak_activity_hour': peak_hour,
                'most_active_user': most_active_user,
                'hourly_distribution': dict(hourly_distribution)
            },
            'suspicious_patterns': suspicious_patterns,
            'recommendations': recommendations,
            'compliance_score': await self._calculate_compliance_score(events, framework)
        }
    
    async def _detect_suspicious_patterns(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Détection de patterns suspects dans les événements"""
        
        patterns = []
        
        # Pattern 1: Accès multiples refusés suivis d'un succès
        for i, event in enumerate(events[:-2]):
            if (event.outcome == 'denied' and 
                events[i+1].outcome == 'denied' and 
                events[i+2].outcome == 'success' and
                event.user_id == events[i+1].user_id == events[i+2].user_id):
                
                patterns.append({
                    'type': 'brute_force_attempt',
                    'description': 'Tentative de force brute détectée',
                    'severity': 'high',
                    'events': [event.event_id, events[i+1].event_id, events[i+2].event_id]
                })
        
        # Pattern 2: Activité inhabituelle hors heures de bureau
        off_hours_events = [e for e in events if e.timestamp.hour < 6 or e.timestamp.hour > 22]
        if len(off_hours_events) > len(events) * 0.3:  # Plus de 30% hors heures
            patterns.append({
                'type': 'off_hours_activity',
                'description': 'Activité importante détectée hors heures de bureau',
                'severity': 'medium',
                'count': len(off_hours_events)
            })
        
        return patterns
    
    async def _generate_audit_recommendations(
        self,
        events: List[AuditEvent],
        framework: ComplianceFramework
    ) -> List[str]:
        """Génération de recommandations d'audit"""
        
        recommendations = []
        
        # Analyse par framework
        if framework == ComplianceFramework.GDPR:
            consent_events = [e for e in events if 'consent' in e.metadata]
            if len(consent_events) < len(events) * 0.5:
                recommendations.append("Améliorer le tracking du consentement GDPR")
        
        elif framework == ComplianceFramework.SOX:
            financial_events = [e for e in events if 'financial' in e.resource_accessed.lower()]
            if financial_events:
                recommendations.append("Renforcer les contrôles d'accès aux données financières")
        
        # Recommandations générales
        high_severity_events = [e for e in events if e.severity.value >= AuditSeverity.HIGH.value]
        if len(high_severity_events) > 10:
            recommendations.append("Investiguer les événements de haute sévérité")
        
        unique_users = len(set(e.user_id for e in events if e.user_id))
        if unique_users > 100:
            recommendations.append("Considérer l'implémentation d'analyses comportementales avancées")
        
        return recommendations
    
    async def _calculate_compliance_score(
        self,
        events: List[AuditEvent],
        framework: ComplianceFramework
    ) -> float:
        """Calcul du score de conformité"""
        
        if not events:
            return 10.0
        
        # Facteurs de réduction du score
        violations = len([e for e in events if e.severity.value >= AuditSeverity.HIGH.value])
        failed_events = len([e for e in events if e.outcome == 'failed'])
        
        base_score = 10.0
        violation_penalty = min(violations * 0.5, 3.0)
        failure_penalty = min(failed_events * 0.1, 2.0)
        
        final_score = max(base_score - violation_penalty - failure_penalty, 0.0)
        
        return round(final_score, 2)
    
    async def get_audit_metrics(self) -> Dict[str, Any]:
        """Récupération des métriques d'audit"""
        
        return {
            'tenant_id': self.tenant_id,
            'metrics': self._metrics.copy(),
            'active_trails': len(self._audit_trails),
            'recent_events_count': len(self._recent_events),
            'config': self._config.copy(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def verify_audit_integrity(self) -> Dict[str, Any]:
        """Vérification de l'intégrité des pistes d'audit"""
        
        integrity_results = {}
        
        for framework, trail in self._audit_trails.items():
            integrity_results[framework.value] = {
                'trail_id': trail.trail_id,
                'events_count': len(trail.events),
                'integrity_verified': trail.verify_chain_integrity(),
                'last_updated': trail.last_updated.isoformat()
            }
        
        return {
            'tenant_id': self.tenant_id,
            'integrity_check': integrity_results,
            'overall_integrity': all(result['integrity_verified'] 
                                   for result in integrity_results.values()),
            'checked_at': datetime.utcnow().isoformat()
        }
