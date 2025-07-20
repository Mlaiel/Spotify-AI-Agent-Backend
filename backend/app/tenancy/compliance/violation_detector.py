"""
Spotify AI Agent - ViolationDetector Ultra-Avancé
================================================

Système de détection intelligente des violations de conformité avec
machine learning, analyse comportementale et réponse automatisée.

Développé par l'équipe d'experts Compliance Security & AI Detection
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
import re
import math
import statistics

class ViolationType(Enum):
    """Types de violations de conformité"""
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    POLICY_VIOLATION = "policy_violation"
    CONSENT_VIOLATION = "consent_violation"
    RETENTION_VIOLATION = "retention_violation"
    TRANSFER_VIOLATION = "transfer_violation"
    SECURITY_VIOLATION = "security_violation"
    LICENSING_VIOLATION = "licensing_violation"
    AUDIT_VIOLATION = "audit_violation"
    REGULATORY_VIOLATION = "regulatory_violation"

class ViolationSeverity(Enum):
    """Niveaux de gravité des violations"""
    CRITICAL = 10
    HIGH = 8
    ELEVATED = 6
    MEDIUM = 4
    LOW = 2
    INFORMATIONAL = 1

class DetectionMethod(Enum):
    """Méthodes de détection"""
    RULE_BASED = "rule_based"
    ML_ANOMALY = "ml_anomaly"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    PATTERN_MATCHING = "pattern_matching"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    HYBRID_DETECTION = "hybrid_detection"

class ResponseAction(Enum):
    """Actions de réponse automatique"""
    ALERT_ONLY = "alert_only"
    BLOCK_ACCESS = "block_access"
    QUARANTINE_DATA = "quarantine_data"
    ESCALATE_INCIDENT = "escalate_incident"
    AUTO_REMEDIATE = "auto_remediate"
    NOTIFY_AUTHORITIES = "notify_authorities"

@dataclass
class ViolationRule:
    """Règle de détection de violation"""
    rule_id: str
    name: str
    description: str
    violation_type: ViolationType
    severity: ViolationSeverity
    
    # Critères de détection
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    threshold: float = 1.0
    time_window: timedelta = field(default=timedelta(minutes=5))
    
    # Configuration
    enabled: bool = True
    detection_method: DetectionMethod = DetectionMethod.RULE_BASED
    confidence_threshold: float = 0.8
    
    # Actions automatiques
    response_actions: List[ResponseAction] = field(default_factory=list)
    auto_escalate: bool = False
    notification_required: bool = True
    
    # Métadonnées
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    framework_reference: Optional[str] = None
    
    def matches_event(self, event: Dict[str, Any]) -> Tuple[bool, float]:
        """Vérification si un événement correspond à cette règle"""
        
        if not self.enabled:
            return False, 0.0
        
        matches = 0
        total_conditions = len(self.conditions)
        
        if total_conditions == 0:
            return False, 0.0
        
        for condition in self.conditions:
            if self._evaluate_condition(condition, event):
                matches += 1
        
        confidence = matches / total_conditions
        
        return confidence >= self.confidence_threshold, confidence
    
    def _evaluate_condition(self, condition: Dict[str, Any], event: Dict[str, Any]) -> bool:
        """Évaluation d'une condition individuelle"""
        
        field = condition.get('field', '')
        operator = condition.get('operator', 'equals')
        value = condition.get('value')
        
        event_value = event.get(field)
        
        if event_value is None:
            return False
        
        if operator == 'equals':
            return event_value == value
        elif operator == 'not_equals':
            return event_value != value
        elif operator == 'greater_than':
            return float(event_value) > float(value)
        elif operator == 'less_than':
            return float(event_value) < float(value)
        elif operator == 'contains':
            return str(value).lower() in str(event_value).lower()
        elif operator == 'regex':
            return bool(re.search(str(value), str(event_value)))
        elif operator == 'in_list':
            return event_value in value if isinstance(value, list) else False
        
        return False

@dataclass
class ViolationEvent:
    """Événement de violation détecté"""
    event_id: str
    violation_type: ViolationType
    severity: ViolationSeverity
    description: str
    
    # Détection
    detected_at: datetime = field(default_factory=datetime.utcnow)
    detection_method: DetectionMethod = DetectionMethod.RULE_BASED
    confidence_score: float = 1.0
    rule_id: Optional[str] = None
    
    # Contexte
    affected_entity: Optional[str] = None
    source_system: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Données d'événement
    event_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Investigation
    status: str = "open"  # open, investigating, resolved, false_positive
    assigned_to: Optional[str] = None
    investigation_notes: List[str] = field(default_factory=list)
    
    # Réponse
    actions_taken: List[str] = field(default_factory=list)
    remediation_completed: bool = False
    resolution_time: Optional[datetime] = None
    
    # Impact
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    affected_records: int = 0
    business_impact: str = "unknown"
    
    def calculate_risk_score(self) -> float:
        """Calcul du score de risque de la violation"""
        
        base_score = self.severity.value
        confidence_factor = self.confidence_score
        
        # Facteur d'impact
        impact_factor = 1.0
        if self.affected_records > 1000:
            impact_factor = 1.5
        elif self.affected_records > 100:
            impact_factor = 1.2
        
        # Facteur temporel (plus récent = plus critique)
        time_factor = 1.0
        age_hours = (datetime.utcnow() - self.detected_at).total_seconds() / 3600
        if age_hours <= 1:
            time_factor = 1.3
        elif age_hours <= 24:
            time_factor = 1.1
        
        return base_score * confidence_factor * impact_factor * time_factor
    
    def is_critical(self) -> bool:
        """Vérification si la violation est critique"""
        return self.severity.value >= ViolationSeverity.HIGH.value
    
    def requires_immediate_action(self) -> bool:
        """Vérification si une action immédiate est requise"""
        return (
            self.severity.value >= ViolationSeverity.CRITICAL.value or
            self.violation_type in [ViolationType.DATA_BREACH, ViolationType.SECURITY_VIOLATION] or
            self.affected_records > 1000
        )

class AnomalyDetector:
    """
    Détecteur d'anomalies basé sur l'apprentissage automatique
    pour identifier les violations de comportement
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.logger = logging.getLogger(f"violation.anomaly.{tenant_id}")
        
        # Modèles de détection
        self._models = {
            'access_patterns': self._create_access_pattern_model(),
            'data_usage': self._create_data_usage_model(),
            'user_behavior': self._create_user_behavior_model(),
            'system_metrics': self._create_system_metrics_model()
        }
        
        # Historique pour apprentissage
        self._baseline_data = defaultdict(deque)
        self._anomaly_thresholds = {
            'access_frequency': 3.0,  # Écarts-types
            'data_volume': 2.5,
            'time_patterns': 2.0,
            'geographic_anomaly': 4.0
        }
        
        # Cache des patterns normaux
        self._normal_patterns = {}
        
    def _create_access_pattern_model(self) -> Dict[str, Any]:
        """Création du modèle d'analyse des patterns d'accès"""
        return {
            'type': 'isolation_forest',
            'features': [
                'access_frequency',
                'access_time_hour',
                'access_duration',
                'resources_accessed',
                'geographic_location'
            ],
            'contamination': 0.1,  # 10% d'anomalies attendues
            'training_window': timedelta(days=30),
            'min_samples': 100
        }
    
    def _create_data_usage_model(self) -> Dict[str, Any]:
        """Création du modèle d'analyse d'usage des données"""
        return {
            'type': 'one_class_svm',
            'features': [
                'data_volume_accessed',
                'data_types_accessed',
                'processing_operations',
                'retention_compliance',
                'consent_status'
            ],
            'nu': 0.05,  # Fraction d'outliers
            'gamma': 'scale'
        }
    
    def _create_user_behavior_model(self) -> Dict[str, Any]:
        """Création du modèle d'analyse comportementale"""
        return {
            'type': 'local_outlier_factor',
            'features': [
                'session_duration',
                'actions_per_session',
                'navigation_patterns',
                'preference_changes',
                'interaction_velocity'
            ],
            'n_neighbors': 20,
            'contamination': 0.1
        }
    
    def _create_system_metrics_model(self) -> Dict[str, Any]:
        """Création du modèle d'analyse des métriques système"""
        return {
            'type': 'statistical_analysis',
            'features': [
                'cpu_usage',
                'memory_usage',
                'network_traffic',
                'disk_io',
                'error_rates'
            ],
            'window_size': timedelta(minutes=15),
            'confidence_interval': 0.95
        }
    
    async def detect_anomalies(self, events: List[Dict[str, Any]]) -> List[ViolationEvent]:
        """Détection d'anomalies dans les événements"""
        
        detected_violations = []
        
        for event in events:
            # Analyse par chaque modèle
            for model_name, model_config in self._models.items():
                anomaly_result = await self._analyze_with_model(event, model_name, model_config)
                
                if anomaly_result['is_anomaly']:
                    violation = await self._create_anomaly_violation(
                        event, model_name, anomaly_result
                    )
                    detected_violations.append(violation)
        
        return detected_violations
    
    async def _analyze_with_model(
        self,
        event: Dict[str, Any],
        model_name: str,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyse d'un événement avec un modèle spécifique"""
        
        model_type = model_config['type']
        features = model_config['features']
        
        # Extraction des caractéristiques
        feature_vector = self._extract_features(event, features)
        
        if model_type == 'isolation_forest':
            return await self._isolation_forest_analysis(feature_vector, model_config)
        elif model_type == 'one_class_svm':
            return await self._svm_analysis(feature_vector, model_config)
        elif model_type == 'local_outlier_factor':
            return await self._lof_analysis(feature_vector, model_config)
        elif model_type == 'statistical_analysis':
            return await self._statistical_analysis(feature_vector, model_config)
        
        return {'is_anomaly': False, 'confidence': 0.0}
    
    def _extract_features(self, event: Dict[str, Any], feature_names: List[str]) -> List[float]:
        """Extraction des caractéristiques numériques"""
        
        features = []
        
        for feature_name in feature_names:
            if feature_name == 'access_frequency':
                # Fréquence d'accès dans la dernière heure
                features.append(self._calculate_access_frequency(event))
            
            elif feature_name == 'access_time_hour':
                # Heure d'accès (0-23)
                timestamp = event.get('timestamp', datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                features.append(float(timestamp.hour))
            
            elif feature_name == 'data_volume_accessed':
                # Volume de données accédées
                features.append(float(event.get('data_size_bytes', 0)))
            
            elif feature_name == 'session_duration':
                # Durée de session en minutes
                duration = event.get('session_duration_minutes', 0)
                features.append(float(duration))
            
            elif feature_name == 'geographic_location':
                # Distance par rapport à la localisation habituelle
                features.append(self._calculate_geographic_anomaly(event))
            
            else:
                # Valeur par défaut
                features.append(float(event.get(feature_name, 0)))
        
        return features
    
    def _calculate_access_frequency(self, event: Dict[str, Any]) -> float:
        """Calcul de la fréquence d'accès"""
        
        user_id = event.get('user_id', 'unknown')
        current_time = datetime.utcnow()
        
        # Compter les accès dans la dernière heure
        user_history = self._baseline_data.get(f'access_{user_id}', deque())
        
        recent_accesses = [
            access_time for access_time in user_history
            if (current_time - access_time).total_seconds() <= 3600
        ]
        
        return float(len(recent_accesses))
    
    def _calculate_geographic_anomaly(self, event: Dict[str, Any]) -> float:
        """Calcul de l'anomalie géographique"""
        
        current_location = event.get('location', {})
        user_id = event.get('user_id', 'unknown')
        
        # Localisation habituelle (simulation)
        normal_location = self._normal_patterns.get(f'location_{user_id}', {
            'lat': 48.8566,  # Paris par défaut
            'lon': 2.3522
        })
        
        if not current_location:
            return 0.0
        
        # Calcul de distance approximative
        lat_diff = abs(current_location.get('lat', 0) - normal_location['lat'])
        lon_diff = abs(current_location.get('lon', 0) - normal_location['lon'])
        
        distance = math.sqrt(lat_diff**2 + lon_diff**2) * 111  # Approximation en km
        
        return min(distance / 1000, 10.0)  # Normalisation
    
    async def _isolation_forest_analysis(
        self,
        features: List[float],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyse avec Isolation Forest"""
        
        # Simulation d'Isolation Forest
        # En production, utiliser scikit-learn
        
        if not features:
            return {'is_anomaly': False, 'confidence': 0.0}
        
        # Score d'anomalie simulé basé sur les valeurs extrêmes
        normalized_features = [(f - 0.5) / 0.5 for f in features]
        anomaly_score = max(abs(f) for f in normalized_features)
        
        contamination = config.get('contamination', 0.1)
        threshold = 2.0  # Seuil d'anomalie
        
        is_anomaly = anomaly_score > threshold
        confidence = min(anomaly_score / threshold, 1.0) if is_anomaly else 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'features': features
        }
    
    async def _svm_analysis(
        self,
        features: List[float],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyse avec One-Class SVM"""
        
        if not features:
            return {'is_anomaly': False, 'confidence': 0.0}
        
        # Simulation de One-Class SVM
        feature_variance = statistics.variance(features) if len(features) > 1 else 0
        mean_feature = statistics.mean(features)
        
        # Détection basée sur la variance et la moyenne
        anomaly_score = feature_variance + abs(mean_feature - 1.0)
        threshold = 1.5
        
        is_anomaly = anomaly_score > threshold
        confidence = min(anomaly_score / threshold, 1.0) if is_anomaly else 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'anomaly_score': anomaly_score
        }
    
    async def _lof_analysis(
        self,
        features: List[float],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyse avec Local Outlier Factor"""
        
        if not features:
            return {'is_anomaly': False, 'confidence': 0.0}
        
        # Simulation de LOF
        # Calcul de densité locale
        local_density = 1.0 / (1.0 + statistics.mean(features))
        
        # Score d'outlier local
        lof_score = 1.0 / local_density if local_density > 0 else 10.0
        
        threshold = 2.0
        is_anomaly = lof_score > threshold
        confidence = min(lof_score / threshold, 1.0) if is_anomaly else 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'lof_score': lof_score
        }
    
    async def _statistical_analysis(
        self,
        features: List[float],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyse statistique pour détection d'anomalies"""
        
        if len(features) < 2:
            return {'is_anomaly': False, 'confidence': 0.0}
        
        # Calculs statistiques
        mean_val = statistics.mean(features)
        std_val = statistics.stdev(features)
        
        # Détection d'outliers basée sur l'écart-type
        threshold_std = 2.0  # 2 écarts-types
        
        anomalies = []
        for feature in features:
            if std_val > 0:
                z_score = abs(feature - mean_val) / std_val
                if z_score > threshold_std:
                    anomalies.append(z_score)
        
        is_anomaly = len(anomalies) > 0
        confidence = max(anomalies) / threshold_std if anomalies else 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': min(confidence, 1.0),
            'z_scores': anomalies,
            'mean': mean_val,
            'std': std_val
        }
    
    async def _create_anomaly_violation(
        self,
        event: Dict[str, Any],
        model_name: str,
        anomaly_result: Dict[str, Any]
    ) -> ViolationEvent:
        """Création d'une violation basée sur une anomalie"""
        
        # Détermination du type de violation selon le modèle
        violation_type_mapping = {
            'access_patterns': ViolationType.UNAUTHORIZED_ACCESS,
            'data_usage': ViolationType.POLICY_VIOLATION,
            'user_behavior': ViolationType.POLICY_VIOLATION,
            'system_metrics': ViolationType.SECURITY_VIOLATION
        }
        
        violation_type = violation_type_mapping.get(model_name, ViolationType.POLICY_VIOLATION)
        
        # Détermination de la gravité selon le score de confiance
        confidence = anomaly_result.get('confidence', 0.0)
        if confidence >= 0.9:
            severity = ViolationSeverity.CRITICAL
        elif confidence >= 0.7:
            severity = ViolationSeverity.HIGH
        elif confidence >= 0.5:
            severity = ViolationSeverity.ELEVATED
        else:
            severity = ViolationSeverity.MEDIUM
        
        violation = ViolationEvent(
            event_id=str(uuid4()),
            violation_type=violation_type,
            severity=severity,
            description=f"Anomalie détectée par {model_name}: {anomaly_result}",
            detection_method=DetectionMethod.ML_ANOMALY,
            confidence_score=confidence,
            affected_entity=event.get('entity_id'),
            source_system=event.get('source_system'),
            user_id=event.get('user_id'),
            session_id=event.get('session_id'),
            event_data=event,
            metadata={
                'model_name': model_name,
                'anomaly_details': anomaly_result
            }
        )
        
        return violation

class ComplianceRuleEngine:
    """
    Moteur de règles de conformité pour détection
    basée sur des règles métier et réglementaires
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.logger = logging.getLogger(f"violation.rules.{tenant_id}")
        
        # Ensemble de règles
        self._rules: Dict[str, ViolationRule] = {}
        
        # Cache d'évaluation
        self._evaluation_cache = {}
        
        # Métriques des règles
        self._rule_metrics = defaultdict(lambda: {
            'evaluations': 0,
            'violations_detected': 0,
            'false_positives': 0,
            'accuracy': 0.0
        })
        
        # Initialisation des règles par défaut
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialisation des règles de conformité par défaut"""
        
        # Règle GDPR - Accès non autorisé aux données personnelles
        gdpr_unauthorized_access = ViolationRule(
            rule_id="gdpr_unauthorized_access",
            name="Accès Non Autorisé - Données Personnelles",
            description="Détection d'accès aux données personnelles sans autorisation appropriée",
            violation_type=ViolationType.UNAUTHORIZED_ACCESS,
            severity=ViolationSeverity.HIGH,
            conditions=[
                {'field': 'data_category', 'operator': 'in_list', 'value': ['personal_data', 'sensitive_data']},
                {'field': 'access_authorized', 'operator': 'equals', 'value': False}
            ],
            detection_method=DetectionMethod.RULE_BASED,
            response_actions=[ResponseAction.BLOCK_ACCESS, ResponseAction.ALERT_ONLY],
            framework_reference="GDPR Article 6"
        )
        self._rules[gdpr_unauthorized_access.rule_id] = gdpr_unauthorized_access
        
        # Règle - Violation de rétention des données
        retention_violation = ViolationRule(
            rule_id="data_retention_violation",
            name="Violation de Rétention des Données",
            description="Données conservées au-delà de la période de rétention autorisée",
            violation_type=ViolationType.RETENTION_VIOLATION,
            severity=ViolationSeverity.ELEVATED,
            conditions=[
                {'field': 'retention_expired', 'operator': 'equals', 'value': True},
                {'field': 'deletion_pending', 'operator': 'equals', 'value': False}
            ],
            response_actions=[ResponseAction.QUARANTINE_DATA, ResponseAction.AUTO_REMEDIATE]
        )
        self._rules[retention_violation.rule_id] = retention_violation
        
        # Règle - Violation de consentement
        consent_violation = ViolationRule(
            rule_id="consent_violation",
            name="Violation de Consentement",
            description="Traitement de données sans consentement valide",
            violation_type=ViolationType.CONSENT_VIOLATION,
            severity=ViolationSeverity.HIGH,
            conditions=[
                {'field': 'requires_consent', 'operator': 'equals', 'value': True},
                {'field': 'consent_status', 'operator': 'not_equals', 'value': 'granted'}
            ],
            response_actions=[ResponseAction.BLOCK_ACCESS, ResponseAction.ESCALATE_INCIDENT]
        )
        self._rules[consent_violation.rule_id] = consent_violation
        
        # Règle - Transfert international non autorisé
        transfer_violation = ViolationRule(
            rule_id="international_transfer_violation",
            name="Transfert International Non Autorisé",
            description="Transfert de données vers un pays sans niveau de protection adéquat",
            violation_type=ViolationType.TRANSFER_VIOLATION,
            severity=ViolationSeverity.CRITICAL,
            conditions=[
                {'field': 'data_transfer', 'operator': 'equals', 'value': True},
                {'field': 'destination_adequate_protection', 'operator': 'equals', 'value': False}
            ],
            response_actions=[ResponseAction.BLOCK_ACCESS, ResponseAction.NOTIFY_AUTHORITIES]
        )
        self._rules[transfer_violation.rule_id] = transfer_violation
        
        # Règle - Violation de licence musicale
        music_license_violation = ViolationRule(
            rule_id="music_license_violation",
            name="Violation de Licence Musicale",
            description="Lecture de contenu musical sans licence appropriée",
            violation_type=ViolationType.LICENSING_VIOLATION,
            severity=ViolationSeverity.HIGH,
            conditions=[
                {'field': 'content_type', 'operator': 'equals', 'value': 'music'},
                {'field': 'license_valid', 'operator': 'equals', 'value': False}
            ],
            response_actions=[ResponseAction.BLOCK_ACCESS, ResponseAction.ESCALATE_INCIDENT]
        )
        self._rules[music_license_violation.rule_id] = music_license_violation
        
        # Règle - Accès suspect géographique
        geo_suspicious_access = ViolationRule(
            rule_id="suspicious_geographic_access",
            name="Accès Géographique Suspect",
            description="Accès depuis une localisation inhabituelle ou à risque",
            violation_type=ViolationType.SECURITY_VIOLATION,
            severity=ViolationSeverity.MEDIUM,
            conditions=[
                {'field': 'geographic_anomaly_score', 'operator': 'greater_than', 'value': 5.0},
                {'field': 'high_risk_country', 'operator': 'equals', 'value': True}
            ],
            response_actions=[ResponseAction.ALERT_ONLY, ResponseAction.ESCALATE_INCIDENT],
            detection_method=DetectionMethod.HYBRID_DETECTION
        )
        self._rules[geo_suspicious_access.rule_id] = geo_suspicious_access
        
        # Règle - Volume d'accès anormal
        volume_anomaly = ViolationRule(
            rule_id="abnormal_access_volume",
            name="Volume d'Accès Anormal",
            description="Volume d'accès aux données anormalement élevé",
            violation_type=ViolationType.POLICY_VIOLATION,
            severity=ViolationSeverity.ELEVATED,
            conditions=[
                {'field': 'access_count_hourly', 'operator': 'greater_than', 'value': 1000},
                {'field': 'user_type', 'operator': 'not_equals', 'value': 'system'}
            ],
            response_actions=[ResponseAction.ALERT_ONLY, ResponseAction.BLOCK_ACCESS],
            time_window=timedelta(hours=1)
        )
        self._rules[volume_anomaly.rule_id] = volume_anomaly
    
    async def evaluate_event(self, event: Dict[str, Any]) -> List[ViolationEvent]:
        """Évaluation d'un événement contre toutes les règles"""
        
        violations = []
        
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            
            # Évaluation de la règle
            matches, confidence = rule.matches_event(event)
            
            # Mise à jour des métriques
            self._rule_metrics[rule.rule_id]['evaluations'] += 1
            
            if matches:
                violation = await self._create_rule_violation(event, rule, confidence)
                violations.append(violation)
                
                self._rule_metrics[rule.rule_id]['violations_detected'] += 1
                
                self.logger.warning(
                    f"Violation détectée - Règle: {rule.name}, "
                    f"Confiance: {confidence:.2f}, Event: {event.get('event_id', 'unknown')}"
                )
        
        return violations
    
    async def _create_rule_violation(
        self,
        event: Dict[str, Any],
        rule: ViolationRule,
        confidence: float
    ) -> ViolationEvent:
        """Création d'une violation basée sur une règle"""
        
        violation = ViolationEvent(
            event_id=str(uuid4()),
            violation_type=rule.violation_type,
            severity=rule.severity,
            description=f"{rule.name}: {rule.description}",
            detection_method=rule.detection_method,
            confidence_score=confidence,
            rule_id=rule.rule_id,
            affected_entity=event.get('entity_id'),
            source_system=event.get('source_system'),
            user_id=event.get('user_id'),
            session_id=event.get('session_id'),
            event_data=event,
            metadata={
                'rule_name': rule.name,
                'framework_reference': rule.framework_reference,
                'response_actions': [action.value for action in rule.response_actions]
            }
        )
        
        # Ajout des actions de réponse dans les métadonnées
        violation.actions_taken = [f"Règle {rule.rule_id} déclenchée"]
        
        return violation
    
    def add_rule(self, rule: ViolationRule) -> bool:
        """Ajout d'une nouvelle règle"""
        
        if rule.rule_id in self._rules:
            self.logger.warning(f"Règle {rule.rule_id} existe déjà, remplacement")
        
        self._rules[rule.rule_id] = rule
        self.logger.info(f"Règle ajoutée: {rule.rule_id} - {rule.name}")
        
        return True
    
    def remove_rule(self, rule_id: str) -> bool:
        """Suppression d'une règle"""
        
        if rule_id in self._rules:
            del self._rules[rule_id]
            self.logger.info(f"Règle supprimée: {rule_id}")
            return True
        
        return False
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Mise à jour d'une règle existante"""
        
        if rule_id not in self._rules:
            return False
        
        rule = self._rules[rule_id]
        
        # Application des mises à jour
        for field, value in updates.items():
            if hasattr(rule, field):
                setattr(rule, field, value)
        
        rule.last_updated = datetime.utcnow()
        
        self.logger.info(f"Règle mise à jour: {rule_id}")
        return True
    
    def get_rule_metrics(self) -> Dict[str, Any]:
        """Récupération des métriques des règles"""
        
        metrics = {}
        
        for rule_id, stats in self._rule_metrics.items():
            if stats['evaluations'] > 0:
                detection_rate = stats['violations_detected'] / stats['evaluations']
                accuracy = 1.0 - (stats['false_positives'] / stats['violations_detected']) if stats['violations_detected'] > 0 else 1.0
            else:
                detection_rate = 0.0
                accuracy = 0.0
            
            metrics[rule_id] = {
                'evaluations': stats['evaluations'],
                'violations_detected': stats['violations_detected'],
                'detection_rate': detection_rate,
                'accuracy': accuracy,
                'false_positives': stats['false_positives']
            }
        
        return metrics

class ViolationDetector:
    """
    Système central de détection des violations de conformité ultra-avancé
    
    Fonctionnalités principales:
    - Détection multi-méthodes (règles, ML, comportementale)
    - Évaluation intelligente de la gravité
    - Réponse automatisée et escalade
    - Apprentissage continu et amélioration
    - Intégration multi-framework de conformité
    """
    
    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.logger = logging.getLogger(f"violation.detector.{tenant_id}")
        
        # Composants de détection
        self.anomaly_detector = AnomalyDetector(tenant_id)
        self.rule_engine = ComplianceRuleEngine(tenant_id)
        
        # Gestionnaire de violations
        self._active_violations: Dict[str, ViolationEvent] = {}
        self._violation_history = deque(maxlen=10000)
        
        # Gestionnaire de réponses
        self._response_handlers = {
            ResponseAction.ALERT_ONLY: self._handle_alert_only,
            ResponseAction.BLOCK_ACCESS: self._handle_block_access,
            ResponseAction.QUARANTINE_DATA: self._handle_quarantine_data,
            ResponseAction.ESCALATE_INCIDENT: self._handle_escalate_incident,
            ResponseAction.AUTO_REMEDIATE: self._handle_auto_remediate,
            ResponseAction.NOTIFY_AUTHORITIES: self._handle_notify_authorities
        }
        
        # Configuration
        self._config = {
            'enable_ml_detection': True,
            'enable_rule_detection': True,
            'enable_behavioral_analysis': True,
            'auto_response_enabled': True,
            'escalation_threshold': ViolationSeverity.HIGH.value,
            'batch_processing_size': 100,
            'detection_interval': timedelta(minutes=1)
        }
        
        # Métriques globales
        self._metrics = {
            'total_events_processed': 0,
            'violations_detected': 0,
            'false_positives': 0,
            'auto_responses_triggered': 0,
            'escalations_performed': 0,
            'detection_accuracy': 0.0,
            'average_detection_time': 0.0
        }
        
        self.logger.info(f"ViolationDetector initialisé pour tenant {tenant_id}")
    
    async def process_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Traitement d'un lot d'événements pour détection de violations"""
        
        start_time = datetime.utcnow()
        
        all_violations = []
        processing_stats = {
            'events_processed': len(events),
            'violations_detected': 0,
            'detection_methods': defaultdict(int),
            'severity_distribution': defaultdict(int),
            'auto_responses': 0
        }
        
        # Traitement par lots
        batch_size = self._config['batch_processing_size']
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            batch_violations = await self._process_event_batch(batch)
            all_violations.extend(batch_violations)
        
        # Traitement des violations détectées
        for violation in all_violations:
            # Enregistrement de la violation
            self._active_violations[violation.event_id] = violation
            self._violation_history.append({
                'event_id': violation.event_id,
                'type': violation.violation_type.value,
                'severity': violation.severity.value,
                'detected_at': violation.detected_at.isoformat(),
                'confidence': violation.confidence_score
            })
            
            # Mise à jour des statistiques
            processing_stats['violations_detected'] += 1
            processing_stats['detection_methods'][violation.detection_method.value] += 1
            processing_stats['severity_distribution'][violation.severity.name] += 1
            
            # Déclenchement des réponses automatiques
            if self._config['auto_response_enabled']:
                responses = await self._trigger_automatic_responses(violation)
                processing_stats['auto_responses'] += len(responses)
        
        # Calcul du temps de traitement
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Mise à jour des métriques globales
        self._update_global_metrics(processing_stats, processing_time)
        
        # Résultat de traitement
        result = {
            'processing_summary': {
                'events_processed': processing_stats['events_processed'],
                'violations_detected': processing_stats['violations_detected'],
                'processing_time_seconds': processing_time,
                'auto_responses_triggered': processing_stats['auto_responses']
            },
            'violation_details': [
                {
                    'event_id': v.event_id,
                    'type': v.violation_type.value,
                    'severity': v.severity.name,
                    'confidence': v.confidence_score,
                    'description': v.description,
                    'requires_immediate_action': v.requires_immediate_action()
                }
                for v in all_violations
            ],
            'detection_statistics': {
                'by_method': dict(processing_stats['detection_methods']),
                'by_severity': dict(processing_stats['severity_distribution']),
                'critical_violations': len([v for v in all_violations if v.is_critical()])
            },
            'tenant_id': self.tenant_id,
            'processed_at': start_time.isoformat()
        }
        
        return result
    
    async def _process_event_batch(self, events: List[Dict[str, Any]]) -> List[ViolationEvent]:
        """Traitement d'un lot d'événements"""
        
        violations = []
        
        # Détection basée sur les règles
        if self._config['enable_rule_detection']:
            for event in events:
                rule_violations = await self.rule_engine.evaluate_event(event)
                violations.extend(rule_violations)
        
        # Détection basée sur le ML
        if self._config['enable_ml_detection']:
            ml_violations = await self.anomaly_detector.detect_anomalies(events)
            violations.extend(ml_violations)
        
        # Analyse comportementale additionnelle
        if self._config['enable_behavioral_analysis']:
            behavioral_violations = await self._analyze_behavioral_patterns(events)
            violations.extend(behavioral_violations)
        
        # Déduplication et fusion des violations similaires
        deduplicated_violations = await self._deduplicate_violations(violations)
        
        return deduplicated_violations
    
    async def _analyze_behavioral_patterns(self, events: List[Dict[str, Any]]) -> List[ViolationEvent]:
        """Analyse des patterns comportementaux"""
        
        violations = []
        
        # Groupement par utilisateur
        user_events = defaultdict(list)
        for event in events:
            user_id = event.get('user_id')
            if user_id:
                user_events[user_id].append(event)
        
        # Analyse par utilisateur
        for user_id, user_event_list in user_events.items():
            # Pattern 1: Accès en rafale
            if len(user_event_list) > 50:  # Plus de 50 événements
                violation = ViolationEvent(
                    event_id=str(uuid4()),
                    violation_type=ViolationType.POLICY_VIOLATION,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Accès en rafale détecté pour l'utilisateur {user_id}",
                    detection_method=DetectionMethod.BEHAVIORAL_ANALYSIS,
                    confidence_score=0.8,
                    user_id=user_id,
                    affected_records=len(user_event_list),
                    event_data={'pattern': 'burst_access', 'event_count': len(user_event_list)}
                )
                violations.append(violation)
            
            # Pattern 2: Accès à des données sensibles multiples
            sensitive_accesses = [
                e for e in user_event_list
                if e.get('data_category') in ['personal_data', 'sensitive_data', 'financial_data']
            ]
            
            if len(sensitive_accesses) > 10:
                violation = ViolationEvent(
                    event_id=str(uuid4()),
                    violation_type=ViolationType.UNAUTHORIZED_ACCESS,
                    severity=ViolationSeverity.HIGH,
                    description=f"Accès excessif aux données sensibles par {user_id}",
                    detection_method=DetectionMethod.BEHAVIORAL_ANALYSIS,
                    confidence_score=0.9,
                    user_id=user_id,
                    affected_records=len(sensitive_accesses),
                    event_data={'pattern': 'sensitive_data_access', 'access_count': len(sensitive_accesses)}
                )
                violations.append(violation)
        
        return violations
    
    async def _deduplicate_violations(self, violations: List[ViolationEvent]) -> List[ViolationEvent]:
        """Déduplication et fusion des violations similaires"""
        
        if not violations:
            return []
        
        # Groupement par similitude
        violation_groups = defaultdict(list)
        
        for violation in violations:
            # Clé de groupement basée sur type, utilisateur, et temps
            group_key = (
                violation.violation_type,
                violation.user_id,
                violation.detected_at.replace(minute=0, second=0, microsecond=0)  # Groupement par heure
            )
            violation_groups[group_key].append(violation)
        
        deduplicated = []
        
        for group_violations in violation_groups.values():
            if len(group_violations) == 1:
                deduplicated.append(group_violations[0])
            else:
                # Fusion des violations similaires
                merged_violation = await self._merge_violations(group_violations)
                deduplicated.append(merged_violation)
        
        return deduplicated
    
    async def _merge_violations(self, violations: List[ViolationEvent]) -> ViolationEvent:
        """Fusion de plusieurs violations similaires"""
        
        # Prendre la violation la plus sévère comme base
        base_violation = max(violations, key=lambda v: v.severity.value)
        
        # Fusion des données
        merged_violation = ViolationEvent(
            event_id=str(uuid4()),
            violation_type=base_violation.violation_type,
            severity=base_violation.severity,
            description=f"Violations multiples fusionnées: {base_violation.description}",
            detection_method=DetectionMethod.HYBRID_DETECTION,
            confidence_score=max(v.confidence_score for v in violations),
            user_id=base_violation.user_id,
            affected_entity=base_violation.affected_entity,
            source_system=base_violation.source_system,
            affected_records=sum(v.affected_records for v in violations),
            event_data={
                'merged_violations': len(violations),
                'original_violations': [v.event_id for v in violations],
                'detection_methods': list(set(v.detection_method.value for v in violations))
            }
        )
        
        return merged_violation
    
    async def _trigger_automatic_responses(self, violation: ViolationEvent) -> List[str]:
        """Déclenchement des réponses automatiques"""
        
        triggered_responses = []
        
        # Détermination des actions basées sur la gravité
        if violation.requires_immediate_action():
            # Actions critiques
            responses = [ResponseAction.ALERT_ONLY, ResponseAction.ESCALATE_INCIDENT]
            
            if violation.violation_type in [ViolationType.DATA_BREACH, ViolationType.SECURITY_VIOLATION]:
                responses.append(ResponseAction.BLOCK_ACCESS)
            
            if violation.violation_type == ViolationType.TRANSFER_VIOLATION:
                responses.append(ResponseAction.NOTIFY_AUTHORITIES)
        
        elif violation.severity.value >= ViolationSeverity.HIGH.value:
            # Actions pour violations importantes
            responses = [ResponseAction.ALERT_ONLY, ResponseAction.ESCALATE_INCIDENT]
        
        else:
            # Actions standard
            responses = [ResponseAction.ALERT_ONLY]
        
        # Exécution des réponses
        for response_action in responses:
            if response_action in self._response_handlers:
                try:
                    result = await self._response_handlers[response_action](violation)
                    triggered_responses.append(f"{response_action.value}: {result}")
                    
                    # Enregistrement de l'action
                    violation.actions_taken.append(f"Action automatique: {response_action.value}")
                    
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'exécution de {response_action.value}: {str(e)}")
        
        # Mise à jour des métriques
        self._metrics['auto_responses_triggered'] += len(triggered_responses)
        
        return triggered_responses
    
    async def _handle_alert_only(self, violation: ViolationEvent) -> str:
        """Gestionnaire d'alerte uniquement"""
        
        self.logger.warning(
            f"ALERTE VIOLATION - {violation.violation_type.value.upper()}: "
            f"{violation.description} (Confiance: {violation.confidence_score:.2f})"
        )
        
        return "Alerte envoyée"
    
    async def _handle_block_access(self, violation: ViolationEvent) -> str:
        """Gestionnaire de blocage d'accès"""
        
        self.logger.critical(
            f"BLOCAGE D'ACCÈS - Violation: {violation.event_id}, "
            f"Utilisateur: {violation.user_id}, Session: {violation.session_id}"
        )
        
        # Simulation de blocage d'accès
        # En production: appel aux systèmes de gestion d'accès
        
        return f"Accès bloqué pour utilisateur {violation.user_id}"
    
    async def _handle_quarantine_data(self, violation: ViolationEvent) -> str:
        """Gestionnaire de mise en quarantaine des données"""
        
        self.logger.warning(
            f"QUARANTAINE DONNÉES - Violation: {violation.event_id}, "
            f"Entité: {violation.affected_entity}"
        )
        
        # Simulation de quarantaine
        # En production: déplacement vers zone sécurisée
        
        return f"Données mises en quarantaine: {violation.affected_entity}"
    
    async def _handle_escalate_incident(self, violation: ViolationEvent) -> str:
        """Gestionnaire d'escalade d'incident"""
        
        self.logger.error(
            f"ESCALADE INCIDENT - Violation critique: {violation.event_id}, "
            f"Type: {violation.violation_type.value}, Gravité: {violation.severity.name}"
        )
        
        # Simulation d'escalade
        # En production: création de ticket, notification équipes
        
        self._metrics['escalations_performed'] += 1
        
        return f"Incident escalé: {violation.event_id}"
    
    async def _handle_auto_remediate(self, violation: ViolationEvent) -> str:
        """Gestionnaire de remédiation automatique"""
        
        self.logger.info(
            f"REMÉDIATION AUTO - Violation: {violation.event_id}, "
            f"Type: {violation.violation_type.value}"
        )
        
        # Remédiation selon le type de violation
        if violation.violation_type == ViolationType.RETENTION_VIOLATION:
            return "Suppression automatique des données expirées"
        elif violation.violation_type == ViolationType.CONSENT_VIOLATION:
            return "Arrêt automatique du traitement des données"
        else:
            return "Mesures correctives automatiques appliquées"
    
    async def _handle_notify_authorities(self, violation: ViolationEvent) -> str:
        """Gestionnaire de notification aux autorités"""
        
        self.logger.critical(
            f"NOTIFICATION AUTORITÉS - Violation: {violation.event_id}, "
            f"Type: {violation.violation_type.value}, Impact: {violation.affected_records} enregistrements"
        )
        
        # Simulation de notification
        # En production: notification CNIL, autorités compétentes
        
        return "Notification aux autorités de régulation envoyée"
    
    def _update_global_metrics(self, processing_stats: Dict[str, Any], processing_time: float):
        """Mise à jour des métriques globales"""
        
        self._metrics['total_events_processed'] += processing_stats['events_processed']
        self._metrics['violations_detected'] += processing_stats['violations_detected']
        
        # Calcul de la moyenne mobile du temps de traitement
        current_avg = self._metrics['average_detection_time']
        total_processed = self._metrics['total_events_processed']
        
        if total_processed > 0:
            new_avg = ((current_avg * (total_processed - processing_stats['events_processed'])) + processing_time) / total_processed
            self._metrics['average_detection_time'] = new_avg
    
    async def get_violation_by_id(self, event_id: str) -> Optional[ViolationEvent]:
        """Récupération d'une violation par ID"""
        
        return self._active_violations.get(event_id)
    
    async def list_active_violations(
        self,
        severity_filter: Optional[ViolationSeverity] = None,
        violation_type_filter: Optional[ViolationType] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Liste des violations actives avec filtrage"""
        
        violations = []
        
        for violation in self._active_violations.values():
            # Application des filtres
            if severity_filter and violation.severity != severity_filter:
                continue
            
            if violation_type_filter and violation.violation_type != violation_type_filter:
                continue
            
            violations.append({
                'event_id': violation.event_id,
                'type': violation.violation_type.value,
                'severity': violation.severity.name,
                'description': violation.description,
                'detected_at': violation.detected_at.isoformat(),
                'confidence': violation.confidence_score,
                'status': violation.status,
                'requires_immediate_action': violation.requires_immediate_action(),
                'risk_score': violation.calculate_risk_score(),
                'affected_records': violation.affected_records
            })
        
        # Tri par score de risque (plus élevé en premier)
        violations.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return violations[:limit]
    
    async def resolve_violation(self, event_id: str, resolution_notes: str) -> bool:
        """Résolution d'une violation"""
        
        if event_id not in self._active_violations:
            return False
        
        violation = self._active_violations[event_id]
        violation.status = "resolved"
        violation.resolution_time = datetime.utcnow()
        violation.investigation_notes.append(f"Résolution: {resolution_notes}")
        violation.remediation_completed = True
        
        # Suppression des violations actives
        del self._active_violations[event_id]
        
        self.logger.info(f"Violation résolue: {event_id}")
        
        return True
    
    async def get_detection_metrics(self) -> Dict[str, Any]:
        """Récupération des métriques de détection"""
        
        return {
            'tenant_id': self.tenant_id,
            'global_metrics': self._metrics.copy(),
            'active_violations_count': len(self._active_violations),
            'critical_violations_count': len([
                v for v in self._active_violations.values()
                if v.is_critical()
            ]),
            'rule_engine_metrics': self.rule_engine.get_rule_metrics(),
            'configuration': self._config.copy(),
            'violation_history_size': len(self._violation_history),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_violation_dashboard(self) -> Dict[str, Any]:
        """Tableau de bord des violations"""
        
        # Analyse des violations actives
        active_by_type = defaultdict(int)
        active_by_severity = defaultdict(int)
        
        for violation in self._active_violations.values():
            active_by_type[violation.violation_type.value] += 1
            active_by_severity[violation.severity.name] += 1
        
        # Tendances récentes
        recent_history = list(self._violation_history)[-100:]  # 100 dernières
        
        return {
            'tenant_id': self.tenant_id,
            'dashboard_timestamp': datetime.utcnow().isoformat(),
            'overview': {
                'total_active_violations': len(self._active_violations),
                'critical_violations': len([v for v in self._active_violations.values() if v.is_critical()]),
                'immediate_action_required': len([v for v in self._active_violations.values() if v.requires_immediate_action()]),
                'total_processed_events': self._metrics['total_events_processed'],
                'detection_rate': (self._metrics['violations_detected'] / self._metrics['total_events_processed']) if self._metrics['total_events_processed'] > 0 else 0
            },
            'active_violations': {
                'by_type': dict(active_by_type),
                'by_severity': dict(active_by_severity)
            },
            'recent_trends': {
                'last_24h_violations': len([h for h in recent_history if (datetime.utcnow() - datetime.fromisoformat(h['detected_at'])).total_seconds() <= 86400]),
                'trending_violation_types': list(active_by_type.keys())[:5]
            },
            'performance': {
                'average_detection_time': self._metrics['average_detection_time'],
                'auto_response_rate': (self._metrics['auto_responses_triggered'] / self._metrics['violations_detected']) if self._metrics['violations_detected'] > 0 else 0,
                'escalation_rate': (self._metrics['escalations_performed'] / self._metrics['violations_detected']) if self._metrics['violations_detected'] > 0 else 0
            }
        }
