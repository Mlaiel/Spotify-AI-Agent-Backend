"""
Gestionnaire d'Escalade Automatique - Spotify AI Agent
======================================================

Système intelligent d'escalade des incidents avec ML, rotation d'équipes
et prise de décision automatisée basée sur les SLA et patterns historiques.

Fonctionnalités:
- Escalade automatique basée sur les SLA
- Rotation intelligente des équipes d'astreinte
- Prédiction des besoins d'escalade avec ML
- Gestion des priorités dynamiques
- Intégration avec les systèmes de ticketing
- Analytics et reporting d'escalade
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import uuid
from sklearn.ensemble import RandomForestClassifier
import redis.asyncio as redis


class EscalationLevel(Enum):
    """Niveaux d'escalade"""
    L1 = "l1"  # Support de premier niveau
    L2 = "l2"  # Support technique avancé
    L3 = "l3"  # Experts/Architectes
    MANAGER = "manager"  # Management
    EXECUTIVE = "executive"  # Direction
    EMERGENCY = "emergency"  # Équipe d'urgence


class EscalationReason(Enum):
    """Raisons d'escalade"""
    SLA_BREACH = "sla_breach"
    SEVERITY_INCREASE = "severity_increase"
    NO_RESPONSE = "no_response"
    COMPLEXITY = "complexity"
    BUSINESS_IMPACT = "business_impact"
    MANUAL_REQUEST = "manual_request"
    AUTOMATED_PREDICTION = "automated_prediction"


class EscalationStatus(Enum):
    """États d'escalade"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OnCallSchedule:
    """Planning d'astreinte"""
    id: str
    team: str
    level: EscalationLevel
    primary_contact: str
    secondary_contact: str
    start_time: datetime
    end_time: datetime
    timezone: str = "UTC"
    skills: List[str] = field(default_factory=list)
    max_concurrent_incidents: int = 5
    escalation_delay: timedelta = field(default=timedelta(minutes=30))


@dataclass
class EscalationRule:
    """Règle d'escalade"""
    id: str
    name: str
    tenant_id: str = ""
    service: str = ""
    severity: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    escalation_path: List[EscalationLevel] = field(default_factory=list)
    sla_thresholds: Dict[str, timedelta] = field(default_factory=dict)
    auto_escalate: bool = True
    business_hours_only: bool = False
    required_skills: List[str] = field(default_factory=list)
    priority_boost: float = 1.0


@dataclass
class EscalationEvent:
    """Événement d'escalade"""
    id: str
    alert_id: str
    from_level: Optional[EscalationLevel]
    to_level: EscalationLevel
    reason: EscalationReason
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    status: EscalationStatus = EscalationStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    predicted_by_ml: bool = False
    escalation_rule_id: Optional[str] = None


class EscalationManager:
    """Gestionnaire d'escalade intelligent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.default_sla = timedelta(hours=config.get('default_sla_hours', 4))
        self.max_escalation_level = EscalationLevel.EXECUTIVE
        self.prediction_model_threshold = config.get('prediction_threshold', 0.8)
        
        # Stockage
        self.redis_client = None
        self.escalation_rules: Dict[str, EscalationRule] = {}
        self.oncall_schedules: Dict[str, OnCallSchedule] = {}
        self.active_escalations: Dict[str, List[EscalationEvent]] = defaultdict(list)
        self.escalation_history: deque = deque(maxlen=50000)
        
        # ML pour prédiction d'escalade
        self.escalation_predictor: Optional[RandomForestClassifier] = None
        self.prediction_features = [
            'severity_score', 'response_time', 'complexity_score',
            'business_impact_score', 'historical_escalation_rate',
            'team_capacity', 'time_of_day', 'day_of_week'
        ]
        
        # Cache de capacité des équipes
        self.team_capacity_cache: Dict[str, Dict[str, Any]] = {}
        self.capacity_cache_ttl = timedelta(minutes=5)
        
        # Métriques et statistiques
        self.escalation_stats = {
            'total_escalations': 0,
            'auto_escalations': 0,
            'successful_predictions': 0,
            'avg_escalation_time': 0.0,
            'sla_breaches': 0
        }
        
        # Callbacks personnalisés
        self.escalation_callbacks: Dict[EscalationLevel, List[Callable]] = defaultdict(list)
        
    async def initialize(self):
        """Initialisation asynchrone du gestionnaire"""
        try:
            # Connexion Redis
            self.redis_client = redis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
            
            # Chargement des règles et plannings
            await self._load_escalation_rules()
            await self._load_oncall_schedules()
            
            # Initialisation du modèle ML
            await self._initialize_prediction_model()
            
            # Démarrage des tâches de fond
            asyncio.create_task(self._monitor_sla_thresholds())
            asyncio.create_task(self._predict_escalation_needs())
            asyncio.create_task(self._update_team_capacity())
            asyncio.create_task(self._rotate_oncall_schedules())
            
            self.logger.info("EscalationManager initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def evaluate_escalation_need(self, alert: Any) -> Optional[EscalationEvent]:
        """Évaluation du besoin d'escalade pour une alerte"""
        try:
            # Vérification des règles d'escalade applicables
            applicable_rules = await self._find_applicable_rules(alert)
            
            if not applicable_rules:
                return None
            
            # Calcul du score de priorité
            priority_score = await self._calculate_priority_score(alert)
            
            # Vérification des seuils SLA
            sla_status = await self._check_sla_status(alert)
            
            # Prédiction ML si disponible
            ml_prediction = None
            if self.escalation_predictor:
                ml_prediction = await self._predict_escalation_ml(alert)
            
            # Détermination du niveau d'escalade nécessaire
            target_level = await self._determine_escalation_level(
                alert, applicable_rules, priority_score, sla_status, ml_prediction
            )
            
            if target_level:
                # Création de l'événement d'escalade
                escalation_event = EscalationEvent(
                    id=str(uuid.uuid4()),
                    alert_id=alert.id,
                    from_level=self._get_current_escalation_level(alert),
                    to_level=target_level,
                    reason=self._determine_escalation_reason(alert, sla_status),
                    predicted_by_ml=ml_prediction is not None and ml_prediction['confidence'] > 0.8,
                    escalation_rule_id=applicable_rules[0].id if applicable_rules else None
                )
                
                return escalation_event
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation d'escalade: {e}")
            return None
    
    async def execute_escalation(self, escalation_event: EscalationEvent) -> bool:
        """Exécution d'une escalade"""
        try:
            escalation_event.status = EscalationStatus.IN_PROGRESS
            
            # Recherche de la personne d'astreinte appropriée
            assignee = await self._find_oncall_assignee(
                escalation_event.to_level,
                escalation_event.alert_id
            )
            
            if not assignee:
                self.logger.warning(
                    f"Aucune personne d'astreinte trouvée pour le niveau {escalation_event.to_level.value}"
                )
                escalation_event.status = EscalationStatus.FAILED
                return False
            
            escalation_event.assigned_to = assignee['id']
            
            # Notification de l'escalade
            await self._notify_escalation(escalation_event, assignee)
            
            # Mise à jour des systèmes de ticketing si configurés
            await self._update_ticketing_systems(escalation_event)
            
            # Enregistrement de l'escalade
            self.active_escalations[escalation_event.alert_id].append(escalation_event)
            await self._persist_escalation_event(escalation_event)
            
            # Mise à jour des statistiques
            self.escalation_stats['total_escalations'] += 1
            if escalation_event.predicted_by_ml:
                self.escalation_stats['auto_escalations'] += 1
            
            # Déclenchement des callbacks
            await self._trigger_escalation_callbacks(escalation_event)
            
            escalation_event.status = EscalationStatus.COMPLETED
            escalation_event.completed_at = datetime.utcnow()
            
            self.logger.info(
                f"Escalade exécutée: alerte {escalation_event.alert_id} "
                f"vers {escalation_event.to_level.value} (assigné à {assignee['name']})"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution d'escalade: {e}")
            escalation_event.status = EscalationStatus.FAILED
            return False
    
    async def create_escalation_rule(self, rule_data: Dict[str, Any]) -> EscalationRule:
        """Création d'une règle d'escalade"""
        try:
            rule = EscalationRule(
                id=rule_data.get('id', str(uuid.uuid4())),
                name=rule_data['name'],
                tenant_id=rule_data.get('tenant_id', ''),
                service=rule_data.get('service', ''),
                severity=rule_data.get('severity', ''),
                conditions=rule_data.get('conditions', {}),
                escalation_path=[
                    EscalationLevel(level) for level in rule_data.get('escalation_path', [])
                ],
                sla_thresholds={
                    level: timedelta(minutes=minutes) 
                    for level, minutes in rule_data.get('sla_thresholds', {}).items()
                },
                auto_escalate=rule_data.get('auto_escalate', True),
                business_hours_only=rule_data.get('business_hours_only', False),
                required_skills=rule_data.get('required_skills', []),
                priority_boost=rule_data.get('priority_boost', 1.0)
            )
            
            self.escalation_rules[rule.id] = rule
            await self._persist_escalation_rule(rule)
            
            self.logger.info(f"Règle d'escalade créée: {rule.id} - {rule.name}")
            return rule
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de règle: {e}")
            raise
    
    async def add_oncall_schedule(self, schedule_data: Dict[str, Any]) -> OnCallSchedule:
        """Ajout d'un planning d'astreinte"""
        try:
            schedule = OnCallSchedule(
                id=schedule_data.get('id', str(uuid.uuid4())),
                team=schedule_data['team'],
                level=EscalationLevel(schedule_data['level']),
                primary_contact=schedule_data['primary_contact'],
                secondary_contact=schedule_data['secondary_contact'],
                start_time=datetime.fromisoformat(schedule_data['start_time']),
                end_time=datetime.fromisoformat(schedule_data['end_time']),
                timezone=schedule_data.get('timezone', 'UTC'),
                skills=schedule_data.get('skills', []),
                max_concurrent_incidents=schedule_data.get('max_concurrent_incidents', 5),
                escalation_delay=timedelta(
                    minutes=schedule_data.get('escalation_delay_minutes', 30)
                )
            )
            
            self.oncall_schedules[schedule.id] = schedule
            await self._persist_oncall_schedule(schedule)
            
            self.logger.info(f"Planning d'astreinte ajouté: {schedule.id} - {schedule.team}")
            return schedule
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout de planning: {e}")
            raise
    
    async def get_escalation_analytics(self, tenant_id: str = None) -> Dict[str, Any]:
        """Analytics des escalades"""
        try:
            # Filtrage par tenant si spécifié
            escalations = self.escalation_history
            if tenant_id:
                escalations = [
                    e for e in escalations 
                    if e.metadata.get('tenant_id') == tenant_id
                ]
            
            if not escalations:
                return {}
            
            # Calcul des métriques
            total_escalations = len(escalations)
            successful_escalations = len([e for e in escalations if e.status == EscalationStatus.COMPLETED])
            avg_escalation_time = sum(
                (e.completed_at - e.triggered_at).total_seconds() 
                for e in escalations if e.completed_at
            ) / max(1, len([e for e in escalations if e.completed_at]))
            
            # Distribution par niveau
            level_distribution = defaultdict(int)
            for escalation in escalations:
                level_distribution[escalation.to_level.value] += 1
            
            # Distribution par raison
            reason_distribution = defaultdict(int)
            for escalation in escalations:
                reason_distribution[escalation.reason.value] += 1
            
            # Tendances temporelles
            daily_counts = defaultdict(int)
            for escalation in escalations:
                day = escalation.triggered_at.date().isoformat()
                daily_counts[day] += 1
            
            # Efficacité des prédictions ML
            ml_predictions = [e for e in escalations if e.predicted_by_ml]
            ml_accuracy = len([e for e in ml_predictions if e.status == EscalationStatus.COMPLETED]) / max(1, len(ml_predictions))
            
            return {
                'period': {
                    'start': min(e.triggered_at for e in escalations).isoformat(),
                    'end': max(e.triggered_at for e in escalations).isoformat()
                },
                'summary': {
                    'total_escalations': total_escalations,
                    'successful_escalations': successful_escalations,
                    'success_rate': successful_escalations / total_escalations,
                    'avg_escalation_time_minutes': avg_escalation_time / 60,
                    'ml_predictions_count': len(ml_predictions),
                    'ml_accuracy': ml_accuracy
                },
                'distributions': {
                    'by_level': dict(level_distribution),
                    'by_reason': dict(reason_distribution),
                    'daily_counts': dict(daily_counts)
                },
                'trends': await self._calculate_escalation_trends(escalations),
                'recommendations': await self._generate_escalation_recommendations(escalations)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des analytics: {e}")
            return {}
    
    async def _find_applicable_rules(self, alert: Any) -> List[EscalationRule]:
        """Recherche des règles d'escalade applicables"""
        applicable_rules = []
        
        for rule in self.escalation_rules.values():
            # Vérification des critères de base
            if rule.tenant_id and rule.tenant_id != alert.tenant_id:
                continue
            if rule.service and rule.service != alert.service:
                continue
            if rule.severity and rule.severity != alert.severity.value:
                continue
            
            # Vérification des heures ouvrables
            if rule.business_hours_only and not self._is_business_hours():
                continue
            
            # Vérification des conditions personnalisées
            if rule.conditions:
                if not await self._evaluate_rule_conditions(rule.conditions, alert):
                    continue
            
            applicable_rules.append(rule)
        
        # Tri par priorité (priority_boost)
        applicable_rules.sort(key=lambda r: r.priority_boost, reverse=True)
        
        return applicable_rules
    
    async def _calculate_priority_score(self, alert: Any) -> float:
        """Calcul du score de priorité d'une alerte"""
        base_scores = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'info': 0.2
        }
        
        score = base_scores.get(alert.severity.value, 0.5)
        
        # Ajustements basés sur d'autres facteurs
        if hasattr(alert, 'ml_confidence') and alert.ml_confidence:
            score += 0.1 * alert.ml_confidence
        
        if hasattr(alert, 'correlation_id') and alert.correlation_id:
            score += 0.15  # Bonus pour les alertes corrélées
        
        # Ajustement basé sur l'historique du service
        service_history = await self._get_service_escalation_history(alert.service)
        if service_history and service_history.get('escalation_rate', 0) > 0.3:
            score += 0.1
        
        return min(1.0, score)
    
    async def _check_sla_status(self, alert: Any) -> Dict[str, Any]:
        """Vérification du statut SLA d'une alerte"""
        now = datetime.utcnow()
        alert_age = now - alert.timestamp
        
        # SLA par défaut basé sur la sévérité
        sla_thresholds = {
            'critical': timedelta(minutes=15),
            'high': timedelta(hours=1),
            'medium': timedelta(hours=4),
            'low': timedelta(hours=24)
        }
        
        expected_sla = sla_thresholds.get(alert.severity.value, self.default_sla)
        
        # Recherche de SLA personnalisé
        applicable_rules = await self._find_applicable_rules(alert)
        if applicable_rules and applicable_rules[0].sla_thresholds:
            custom_sla = applicable_rules[0].sla_thresholds.get(
                alert.severity.value
            )
            if custom_sla:
                expected_sla = custom_sla
        
        time_remaining = expected_sla - alert_age
        breach_percentage = alert_age.total_seconds() / expected_sla.total_seconds()
        
        return {
            'expected_sla': expected_sla,
            'alert_age': alert_age,
            'time_remaining': time_remaining,
            'breach_percentage': breach_percentage,
            'is_breached': breach_percentage >= 1.0,
            'near_breach': breach_percentage >= 0.8
        }
    
    async def _find_oncall_assignee(self, level: EscalationLevel, alert_id: str) -> Optional[Dict[str, Any]]:
        """Recherche de la personne d'astreinte appropriée"""
        now = datetime.utcnow()
        
        # Recherche des plannings actifs pour ce niveau
        active_schedules = [
            schedule for schedule in self.oncall_schedules.values()
            if (schedule.level == level and 
                schedule.start_time <= now <= schedule.end_time)
        ]
        
        if not active_schedules:
            return None
        
        # Vérification de la capacité des équipes
        for schedule in active_schedules:
            team_capacity = await self._get_team_capacity(schedule.team)
            
            if team_capacity['current_incidents'] < schedule.max_concurrent_incidents:
                # Contact principal disponible
                primary_contact = await self._get_contact_info(schedule.primary_contact)
                if primary_contact and primary_contact.get('available', True):
                    return {
                        'id': schedule.primary_contact,
                        'name': primary_contact.get('name', 'Unknown'),
                        'contact_info': primary_contact,
                        'schedule_id': schedule.id,
                        'type': 'primary'
                    }
                
                # Contact secondaire si le principal n'est pas disponible
                secondary_contact = await self._get_contact_info(schedule.secondary_contact)
                if secondary_contact and secondary_contact.get('available', True):
                    return {
                        'id': schedule.secondary_contact,
                        'name': secondary_contact.get('name', 'Unknown'),
                        'contact_info': secondary_contact,
                        'schedule_id': schedule.id,
                        'type': 'secondary'
                    }
        
        return None
    
    def register_escalation_callback(self, level: EscalationLevel, callback: Callable):
        """Enregistrement d'un callback d'escalade"""
        self.escalation_callbacks[level].append(callback)
        self.logger.info(f"Callback d'escalade enregistré pour le niveau {level.value}")
