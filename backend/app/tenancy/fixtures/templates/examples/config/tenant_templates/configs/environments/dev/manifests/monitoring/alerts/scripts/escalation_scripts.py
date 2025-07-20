"""
Scripts d'Escalade Intelligente d'Incidents
Système d'escalade automatique basé sur l'IA pour Spotify AI Agent

Fonctionnalités:
- Escalade intelligente basée sur la sévérité et l'impact business
- Routage automatique vers les équipes expertes
- Prédiction du temps de résolution par ML
- Escalade géolocalisée selon les fuseaux horaires
- Intégration avec les systèmes d'astreinte
- Analyse de l'historique pour optimisation continue
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pytz
from collections import defaultdict

from . import AlertConfig, AlertSeverity, AlertCategory, ScriptType, register_alert

logger = logging.getLogger(__name__)

class EscalationTrigger(Enum):
    """Déclencheurs d'escalade"""
    TIME_BASED = "time_based"
    SEVERITY_INCREASE = "severity_increase"
    NO_ACKNOWLEDGMENT = "no_acknowledgment"
    FAILED_RESOLUTION = "failed_resolution"
    BUSINESS_IMPACT = "business_impact"
    MANUAL_REQUEST = "manual_request"
    AI_RECOMMENDATION = "ai_recommendation"

class TeamType(Enum):
    """Types d'équipes d'intervention"""
    L1_SUPPORT = "l1_support"
    L2_ENGINEERS = "l2_engineers"
    L3_SPECIALISTS = "l3_specialists"
    SECURITY_TEAM = "security_team"
    DBA_TEAM = "dba_team"
    ML_ENGINEERS = "ml_engineers"
    ARCHITECTURE_TEAM = "architecture_team"
    EXECUTIVE_TEAM = "executive_team"
    VENDOR_SUPPORT = "vendor_support"

class EscalationStatus(Enum):
    """Status d'escalade"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CLOSED = "closed"

@dataclass
class TeamMember:
    """Membre d'une équipe d'intervention"""
    user_id: str
    name: str
    email: str
    phone: str
    timezone: str
    skills: List[str]
    availability_schedule: Dict[str, Any]
    escalation_priority: int = 1
    is_on_call: bool = False
    max_concurrent_incidents: int = 3

@dataclass
class EscalationTeam:
    """Équipe d'escalade"""
    team_id: str
    name: str
    team_type: TeamType
    members: List[TeamMember]
    specializations: List[str]
    escalation_threshold_minutes: int
    max_concurrent_incidents: int = 10
    timezone: str = "UTC"
    working_hours: Dict[str, str] = field(default_factory=dict)

@dataclass
class EscalationRule:
    """Règle d'escalade automatique"""
    rule_id: str
    name: str
    triggers: List[EscalationTrigger]
    source_teams: List[TeamType]
    target_teams: List[TeamType]
    conditions: Dict[str, Any]
    escalation_delays: List[int]  # en minutes
    business_hours_only: bool = False
    severity_levels: List[AlertSeverity] = field(default_factory=list)
    categories: List[AlertCategory] = field(default_factory=list)
    tenant_id: Optional[str] = None
    enabled: bool = True

@dataclass
class IncidentEscalation:
    """Escalade d'un incident"""
    escalation_id: str
    incident_id: str
    current_team: TeamType
    assigned_members: List[str]
    escalation_level: int
    status: EscalationStatus
    trigger: EscalationTrigger
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    business_impact_score: float = 0.0
    predicted_resolution_time: Optional[int] = None  # en minutes

class IntelligentEscalationEngine:
    """Moteur d'escalade intelligent avec IA"""
    
    def __init__(self):
        self.teams: Dict[TeamType, EscalationTeam] = {}
        self.rules: List[EscalationRule] = []
        self.active_escalations: Dict[str, IncidentEscalation] = {}
        self.escalation_history: List[IncidentEscalation] = []
        self.business_impact_weights = {
            AlertCategory.AVAILABILITY: 1.0,
            AlertCategory.SECURITY: 0.9,
            AlertCategory.PERFORMANCE: 0.7,
            AlertCategory.AUDIO_QUALITY: 0.8,
            AlertCategory.ML_MODEL: 0.6,
            AlertCategory.TENANT_ISOLATION: 0.8,
            AlertCategory.COST_OPTIMIZATION: 0.3
        }
        
        self._initialize_teams()
        self._initialize_escalation_rules()

    def _initialize_teams(self):
        """Initialise les équipes d'escalade"""
        
        # Équipe Support L1
        l1_team = EscalationTeam(
            team_id="support_l1",
            name="Support Level 1",
            team_type=TeamType.L1_SUPPORT,
            members=[
                TeamMember(
                    user_id="l1_001",
                    name="Agent Support 1",
                    email="l1-1@spotify-ai-agent.com",
                    phone="+33123456789",
                    timezone="Europe/Paris",
                    skills=["incident_triage", "basic_troubleshooting"],
                    availability_schedule={"24/7": True},
                    is_on_call=True
                ),
                TeamMember(
                    user_id="l1_002", 
                    name="Agent Support 2",
                    email="l1-2@spotify-ai-agent.com",
                    phone="+33123456790",
                    timezone="America/New_York",
                    skills=["incident_triage", "customer_communication"],
                    availability_schedule={"24/7": True}
                )
            ],
            specializations=["user_support", "incident_triage"],
            escalation_threshold_minutes=15,
            working_hours={"start": "00:00", "end": "23:59"}
        )
        
        # Équipe Ingénieurs L2
        l2_team = EscalationTeam(
            team_id="engineers_l2",
            name="Engineers Level 2",
            team_type=TeamType.L2_ENGINEERS,
            members=[
                TeamMember(
                    user_id="l2_001",
                    name="Ingénieur Backend 1", 
                    email="backend-1@spotify-ai-agent.com",
                    phone="+33123456791",
                    timezone="Europe/Paris",
                    skills=["python", "fastapi", "postgresql", "redis"],
                    availability_schedule={"weekdays": "08:00-18:00"},
                    escalation_priority=1
                ),
                TeamMember(
                    user_id="l2_002",
                    name="Ingénieur DevOps 1",
                    email="devops-1@spotify-ai-agent.com", 
                    phone="+33123456792",
                    timezone="Europe/Paris",
                    skills=["kubernetes", "docker", "monitoring", "ci_cd"],
                    availability_schedule={"weekdays": "08:00-18:00"},
                    escalation_priority=2
                )
            ],
            specializations=["backend_services", "infrastructure", "performance"],
            escalation_threshold_minutes=30,
            working_hours={"start": "08:00", "end": "18:00"}
        )
        
        # Équipe Spécialistes L3
        l3_team = EscalationTeam(
            team_id="specialists_l3",
            name="Specialists Level 3",
            team_type=TeamType.L3_SPECIALISTS,
            members=[
                TeamMember(
                    user_id="l3_001",
                    name="Architecte Principal",
                    email="architect@spotify-ai-agent.com",
                    phone="+33123456793", 
                    timezone="Europe/Paris",
                    skills=["system_architecture", "performance_optimization", "scalability"],
                    availability_schedule={"on_call": True},
                    escalation_priority=1,
                    max_concurrent_incidents=2
                ),
                TeamMember(
                    user_id="l3_002",
                    name="Expert ML/IA",
                    email="ml-expert@spotify-ai-agent.com",
                    phone="+33123456794",
                    timezone="Europe/Paris", 
                    skills=["machine_learning", "ai_models", "data_science"],
                    availability_schedule={"weekdays": "09:00-17:00"},
                    escalation_priority=2
                )
            ],
            specializations=["complex_architecture", "ml_models", "critical_incidents"],
            escalation_threshold_minutes=60,
            max_concurrent_incidents=5
        )
        
        # Équipe Sécurité
        security_team = EscalationTeam(
            team_id="security_team",
            name="Security Team",
            team_type=TeamType.SECURITY_TEAM,
            members=[
                TeamMember(
                    user_id="sec_001",
                    name="Expert Sécurité 1",
                    email="security-1@spotify-ai-agent.com",
                    phone="+33123456795",
                    timezone="Europe/Paris",
                    skills=["security_analysis", "intrusion_detection", "forensics"],
                    availability_schedule={"24/7": True},
                    is_on_call=True,
                    escalation_priority=1
                )
            ],
            specializations=["security_incidents", "data_breaches", "compliance"],
            escalation_threshold_minutes=10,
            working_hours={"start": "00:00", "end": "23:59"}
        )
        
        self.teams = {
            TeamType.L1_SUPPORT: l1_team,
            TeamType.L2_ENGINEERS: l2_team,
            TeamType.L3_SPECIALISTS: l3_team,
            TeamType.SECURITY_TEAM: security_team
        }

    def _initialize_escalation_rules(self):
        """Initialise les règles d'escalade"""
        
        # Règle d'escalade temporelle standard
        time_based_rule = EscalationRule(
            rule_id="standard_time_escalation",
            name="Escalade temporelle standard",
            triggers=[EscalationTrigger.TIME_BASED],
            source_teams=[TeamType.L1_SUPPORT],
            target_teams=[TeamType.L2_ENGINEERS],
            conditions={"no_progress": True},
            escalation_delays=[15, 30, 60],  # 15min -> L2, 30min -> L3, 60min -> Executive
            severity_levels=[AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        )
        
        # Règle d'escalade sécurité immédiate
        security_immediate_rule = EscalationRule(
            rule_id="security_immediate_escalation",
            name="Escalade sécurité immédiate",
            triggers=[EscalationTrigger.SEVERITY_INCREASE],
            source_teams=[TeamType.L1_SUPPORT, TeamType.L2_ENGINEERS],
            target_teams=[TeamType.SECURITY_TEAM],
            conditions={"category": "security"},
            escalation_delays=[0],  # Immédiat
            severity_levels=[AlertSeverity.CRITICAL],
            categories=[AlertCategory.SECURITY]
        )
        
        # Règle d'escalade ML/IA
        ml_specialist_rule = EscalationRule(
            rule_id="ml_specialist_escalation",
            name="Escalade vers experts ML",
            triggers=[EscalationTrigger.FAILED_RESOLUTION],
            source_teams=[TeamType.L2_ENGINEERS],
            target_teams=[TeamType.L3_SPECIALISTS],
            conditions={"category": "ml_model", "failed_attempts": 2},
            escalation_delays=[20],
            categories=[AlertCategory.ML_MODEL]
        )
        
        # Règle d'escalade impact business
        business_impact_rule = EscalationRule(
            rule_id="business_impact_escalation",
            name="Escalade impact business critique",
            triggers=[EscalationTrigger.BUSINESS_IMPACT],
            source_teams=[TeamType.L2_ENGINEERS, TeamType.L3_SPECIALISTS],
            target_teams=[TeamType.EXECUTIVE_TEAM],
            conditions={"business_impact_score": {"operator": ">", "threshold": 0.8}},
            escalation_delays=[30],
            severity_levels=[AlertSeverity.CRITICAL]
        )
        
        self.rules.extend([
            time_based_rule,
            security_immediate_rule,
            ml_specialist_rule,
            business_impact_rule
        ])

    async def create_escalation(self, incident_id: str, initial_team: TeamType, context: Dict[str, Any]) -> IncidentEscalation:
        """Crée une nouvelle escalade d'incident"""
        
        escalation_id = f"esc_{incident_id}_{int(datetime.utcnow().timestamp())}"
        
        # Calcul du score d'impact business
        business_impact = await self._calculate_business_impact(context)
        
        # Prédiction du temps de résolution avec IA
        predicted_time = await self._predict_resolution_time(context)
        
        # Assignation automatique des membres disponibles
        assigned_members = await self._assign_team_members(initial_team, context)
        
        escalation = IncidentEscalation(
            escalation_id=escalation_id,
            incident_id=incident_id,
            current_team=initial_team,
            assigned_members=assigned_members,
            escalation_level=1,
            status=EscalationStatus.PENDING,
            trigger=EscalationTrigger.MANUAL_REQUEST,
            created_at=datetime.utcnow(),
            business_impact_score=business_impact,
            predicted_resolution_time=predicted_time
        )
        
        self.active_escalations[escalation_id] = escalation
        
        # Planification des escalades automatiques
        await self._schedule_automatic_escalations(escalation, context)
        
        logger.info(f"Escalade créée: {escalation_id} assignée à {initial_team.value}")
        
        return escalation

    async def _calculate_business_impact(self, context: Dict[str, Any]) -> float:
        """Calcule le score d'impact business d'un incident"""
        
        base_score = 0.0
        
        # Impact basé sur la catégorie
        category = context.get('category', AlertCategory.PERFORMANCE)
        base_score += self.business_impact_weights.get(category, 0.5)
        
        # Impact basé sur la sévérité
        severity = context.get('severity', AlertSeverity.MEDIUM)
        severity_multiplier = {
            AlertSeverity.CRITICAL: 1.0,
            AlertSeverity.HIGH: 0.8,
            AlertSeverity.MEDIUM: 0.5,
            AlertSeverity.LOW: 0.2,
            AlertSeverity.INFO: 0.1
        }
        base_score *= severity_multiplier.get(severity, 0.5)
        
        # Impact basé sur les services affectés
        affected_services = context.get('affected_services', [])
        critical_services = ['api-gateway', 'authentication-service', 'audio-processing']
        
        if any(service in critical_services for service in affected_services):
            base_score *= 1.5
        
        # Impact basé sur le nombre d'utilisateurs affectés
        affected_users = context.get('affected_users', 0)
        if affected_users > 10000:
            base_score *= 1.8
        elif affected_users > 1000:
            base_score *= 1.3
        
        # Impact temporel (heures de pointe)
        current_hour = datetime.utcnow().hour
        if 8 <= current_hour <= 22:  # Heures de pointe
            base_score *= 1.2
        
        return min(base_score, 1.0)  # Normalisation à 1.0 max

    async def _predict_resolution_time(self, context: Dict[str, Any]) -> Optional[int]:
        """Prédit le temps de résolution avec ML"""
        
        try:
            # En production, utiliser un modèle ML entraîné
            # Ici, simulation basée sur l'historique et les patterns
            
            category = context.get('category', AlertCategory.PERFORMANCE)
            severity = context.get('severity', AlertSeverity.MEDIUM)
            
            # Temps de base par catégorie (en minutes)
            base_times = {
                AlertCategory.SECURITY: 45,
                AlertCategory.AVAILABILITY: 60,
                AlertCategory.PERFORMANCE: 90,
                AlertCategory.AUDIO_QUALITY: 120,
                AlertCategory.ML_MODEL: 180,
                AlertCategory.TENANT_ISOLATION: 90,
                AlertCategory.COST_OPTIMIZATION: 240
            }
            
            base_time = base_times.get(category, 120)
            
            # Ajustement par sévérité
            severity_multipliers = {
                AlertSeverity.CRITICAL: 0.7,  # Traité plus rapidement
                AlertSeverity.HIGH: 1.0,
                AlertSeverity.MEDIUM: 1.3,
                AlertSeverity.LOW: 2.0,
                AlertSeverity.INFO: 3.0
            }
            
            predicted = base_time * severity_multipliers.get(severity, 1.0)
            
            # Ajustement basé sur la complexité
            affected_services = len(context.get('affected_services', []))
            if affected_services > 3:
                predicted *= 1.4
            
            return int(predicted)
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction ML: {e}")
            return None

    async def _assign_team_members(self, team_type: TeamType, context: Dict[str, Any]) -> List[str]:
        """Assigne automatiquement les membres d'équipe disponibles"""
        
        team = self.teams.get(team_type)
        if not team:
            return []
        
        available_members = []
        current_time = datetime.utcnow()
        
        for member in team.members:
            # Vérification de la disponibilité temporelle
            if await self._is_member_available(member, current_time):
                # Vérification de la charge de travail
                current_incidents = await self._get_member_incident_count(member.user_id)
                if current_incidents < member.max_concurrent_incidents:
                    available_members.append(member)
        
        # Tri par priorité d'escalade et compétences
        available_members.sort(key=lambda m: (m.escalation_priority, -len(m.skills)))
        
        # Sélection des meilleurs candidats (max 2 pour la plupart des incidents)
        selected_count = 1 if context.get('severity') in [AlertSeverity.LOW, AlertSeverity.MEDIUM] else 2
        selected_members = available_members[:selected_count]
        
        return [member.user_id for member in selected_members]

    async def _is_member_available(self, member: TeamMember, check_time: datetime) -> bool:
        """Vérifie si un membre est disponible à un moment donné"""
        
        try:
            # Conversion au fuseau horaire du membre
            member_tz = pytz.timezone(member.timezone)
            local_time = check_time.astimezone(member_tz)
            
            # Vérification des heures de travail
            schedule = member.availability_schedule
            
            if "24/7" in schedule and schedule["24/7"]:
                return True
            
            if "on_call" in schedule and schedule["on_call"] and member.is_on_call:
                return True
            
            if "weekdays" in schedule:
                if local_time.weekday() < 5:  # Lundi à Vendredi
                    hours_range = schedule["weekdays"]
                    start_hour, end_hour = hours_range.split("-")
                    start = datetime.strptime(start_hour, "%H:%M").time()
                    end = datetime.strptime(end_hour, "%H:%M").time()
                    
                    return start <= local_time.time() <= end
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur vérification disponibilité {member.user_id}: {e}")
            return False

    async def _get_member_incident_count(self, user_id: str) -> int:
        """Retourne le nombre d'incidents actifs assignés à un membre"""
        count = 0
        for escalation in self.active_escalations.values():
            if (user_id in escalation.assigned_members and 
                escalation.status in [EscalationStatus.PENDING, EscalationStatus.IN_PROGRESS]):
                count += 1
        return count

    async def _schedule_automatic_escalations(self, escalation: IncidentEscalation, context: Dict[str, Any]):
        """Planifie les escalades automatiques futures"""
        
        applicable_rules = self._find_applicable_escalation_rules(escalation, context)
        
        for rule in applicable_rules:
            for i, delay_minutes in enumerate(rule.escalation_delays):
                if delay_minutes > 0:  # Skip immediate escalations
                    escalation_time = datetime.utcnow() + timedelta(minutes=delay_minutes)
                    
                    # En production, utiliser un scheduler comme Celery
                    logger.info(
                        f"Escalade automatique planifiée pour {escalation.escalation_id} "
                        f"dans {delay_minutes} minutes vers {rule.target_teams[0].value if rule.target_teams else 'unknown'}"
                    )

    def _find_applicable_escalation_rules(self, escalation: IncidentEscalation, context: Dict[str, Any]) -> List[EscalationRule]:
        """Trouve les règles d'escalade applicables"""
        
        applicable_rules = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Vérification du tenant
            if rule.tenant_id and rule.tenant_id != context.get('tenant_id'):
                continue
            
            # Vérification de l'équipe source
            if escalation.current_team not in rule.source_teams:
                continue
            
            # Vérification de la sévérité
            if rule.severity_levels and context.get('severity') not in rule.severity_levels:
                continue
            
            # Vérification de la catégorie
            if rule.categories and context.get('category') not in rule.categories:
                continue
            
            # Vérification des conditions spécifiques
            if self._check_rule_conditions(rule, escalation, context):
                applicable_rules.append(rule)
        
        return applicable_rules

    def _check_rule_conditions(self, rule: EscalationRule, escalation: IncidentEscalation, context: Dict[str, Any]) -> bool:
        """Vérifie les conditions spécifiques d'une règle"""
        
        for condition_key, condition_value in rule.conditions.items():
            
            if condition_key == "business_impact_score":
                if isinstance(condition_value, dict):
                    operator = condition_value.get("operator", "==")
                    threshold = condition_value.get("threshold", 0)
                    
                    if operator == ">" and escalation.business_impact_score <= threshold:
                        return False
                    elif operator == "<" and escalation.business_impact_score >= threshold:
                        return False
                    elif operator == "==" and escalation.business_impact_score != threshold:
                        return False
            
            elif condition_key == "no_progress":
                # Vérifier s'il y a eu des progrès récents
                time_since_created = datetime.utcnow() - escalation.created_at
                if time_since_created < timedelta(minutes=10):
                    return False
            
            elif condition_key == "failed_attempts":
                # Compter les tentatives de résolution échouées
                failed_count = len([h for h in escalation.escalation_history if h.get('status') == 'failed'])
                if failed_count < condition_value:
                    return False
        
        return True

    async def escalate_incident(self, escalation_id: str, trigger: EscalationTrigger, target_team: Optional[TeamType] = None) -> bool:
        """Escalade un incident vers l'équipe suivante"""
        
        escalation = self.active_escalations.get(escalation_id)
        if not escalation:
            logger.error(f"Escalade non trouvée: {escalation_id}")
            return False
        
        try:
            # Détermination de l'équipe cible
            if not target_team:
                target_team = await self._determine_next_escalation_team(escalation)
            
            if not target_team:
                logger.warning(f"Aucune équipe cible trouvée pour l'escalade {escalation_id}")
                return False
            
            # Enregistrement de l'escalade dans l'historique
            escalation_record = {
                "from_team": escalation.current_team.value,
                "to_team": target_team.value,
                "trigger": trigger.value,
                "timestamp": datetime.utcnow().isoformat(),
                "escalation_level": escalation.escalation_level + 1
            }
            escalation.escalation_history.append(escalation_record)
            
            # Mise à jour de l'escalade
            old_team = escalation.current_team
            escalation.current_team = target_team
            escalation.escalation_level += 1
            escalation.trigger = trigger
            escalation.status = EscalationStatus.ESCALATED
            
            # Assignation de nouveaux membres
            context = {"severity": AlertSeverity.HIGH}  # Context basique
            escalation.assigned_members = await self._assign_team_members(target_team, context)
            
            logger.info(
                f"Incident {escalation.incident_id} escaladé de {old_team.value} "
                f"vers {target_team.value} (Niveau {escalation.escalation_level})"
            )
            
            # Notification de l'escalade
            await self._notify_escalation(escalation, escalation_record)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'escalade {escalation_id}: {e}")
            return False

    async def _determine_next_escalation_team(self, escalation: IncidentEscalation) -> Optional[TeamType]:
        """Détermine la prochaine équipe d'escalade"""
        
        current_team = escalation.current_team
        
        # Hiérarchie d'escalade standard
        escalation_hierarchy = {
            TeamType.L1_SUPPORT: TeamType.L2_ENGINEERS,
            TeamType.L2_ENGINEERS: TeamType.L3_SPECIALISTS,
            TeamType.L3_SPECIALISTS: TeamType.ARCHITECTURE_TEAM
        }
        
        # Escalades spécialisées
        if escalation.business_impact_score > 0.8:
            return TeamType.EXECUTIVE_TEAM
        
        return escalation_hierarchy.get(current_team)

    async def _notify_escalation(self, escalation: IncidentEscalation, escalation_record: Dict[str, Any]):
        """Notifie les équipes de l'escalade"""
        
        try:
            target_team = self.teams.get(escalation.current_team)
            if not target_team:
                return
            
            # Préparation du message de notification
            message = f"""
🚨 ESCALADE D'INCIDENT 🚨

Incident: {escalation.incident_id}
Escalé de: {escalation_record['from_team']}
Escalé vers: {escalation_record['to_team']}
Niveau d'escalade: {escalation.escalation_level}
Impact business: {escalation.business_impact_score:.2%}

Membres assignés: {', '.join(escalation.assigned_members)}
Temps prédit résolution: {escalation.predicted_resolution_time or 'Non calculé'} minutes

Historique escalade:
{json.dumps(escalation.escalation_history, indent=2)}
            """
            
            # Notification des membres assignés
            for member_id in escalation.assigned_members:
                member = next(
                    (m for m in target_team.members if m.user_id == member_id),
                    None
                )
                if member:
                    logger.info(f"Notification escalade envoyée à {member.email}")
                    # En production, envoyer vraie notification
            
        except Exception as e:
            logger.error(f"Erreur lors de la notification d'escalade: {e}")

    async def acknowledge_escalation(self, escalation_id: str, member_id: str) -> bool:
        """Marque une escalade comme acquittée"""
        
        escalation = self.active_escalations.get(escalation_id)
        if not escalation:
            return False
        
        escalation.status = EscalationStatus.ACKNOWLEDGED
        escalation.acknowledged_at = datetime.utcnow()
        
        # Ajout à l'historique
        escalation.escalation_history.append({
            "action": "acknowledged",
            "member_id": member_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Escalade {escalation_id} acquittée par {member_id}")
        return True

    async def resolve_escalation(self, escalation_id: str, member_id: str, resolution_notes: str) -> bool:
        """Marque une escalade comme résolue"""
        
        escalation = self.active_escalations.get(escalation_id)
        if not escalation:
            return False
        
        escalation.status = EscalationStatus.RESOLVED
        escalation.resolved_at = datetime.utcnow()
        
        # Calcul du temps de résolution réel
        resolution_time = (escalation.resolved_at - escalation.created_at).total_seconds() / 60
        
        # Ajout à l'historique
        escalation.escalation_history.append({
            "action": "resolved",
            "member_id": member_id,
            "resolution_notes": resolution_notes,
            "resolution_time_minutes": resolution_time,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Déplacement vers l'historique
        self.escalation_history.append(escalation)
        del self.active_escalations[escalation_id]
        
        logger.info(f"Escalade {escalation_id} résolue par {member_id} en {resolution_time:.1f} minutes")
        return True

    async def get_escalation_analytics(self) -> Dict[str, Any]:
        """Retourne les analytics d'escalade"""
        
        total_escalations = len(self.escalation_history)
        if total_escalations == 0:
            return {"total_escalations": 0}
        
        # Temps de résolution moyens
        resolution_times = [
            (e.resolved_at - e.created_at).total_seconds() / 60
            for e in self.escalation_history
            if e.resolved_at
        ]
        
        avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
        
        # Escalades par équipe
        team_stats = defaultdict(int)
        for escalation in self.escalation_history:
            team_stats[escalation.current_team.value] += 1
        
        # Précision des prédictions ML
        prediction_accuracy = 0.0
        accurate_predictions = 0
        total_predictions = 0
        
        for escalation in self.escalation_history:
            if escalation.predicted_resolution_time and escalation.resolved_at:
                actual_time = (escalation.resolved_at - escalation.created_at).total_seconds() / 60
                predicted_time = escalation.predicted_resolution_time
                
                # Considéré précis si l'écart est < 20%
                if abs(actual_time - predicted_time) / predicted_time < 0.2:
                    accurate_predictions += 1
                total_predictions += 1
        
        if total_predictions > 0:
            prediction_accuracy = accurate_predictions / total_predictions
        
        return {
            "total_escalations": total_escalations,
            "active_escalations": len(self.active_escalations),
            "average_resolution_time_minutes": round(avg_resolution_time, 1),
            "team_distribution": dict(team_stats),
            "ml_prediction_accuracy": f"{prediction_accuracy:.1%}",
            "escalation_trends": await self._calculate_escalation_trends()
        }

    async def _calculate_escalation_trends(self) -> Dict[str, Any]:
        """Calcule les tendances d'escalade"""
        
        recent_escalations = [
            e for e in self.escalation_history
            if e.created_at > datetime.utcnow() - timedelta(days=7)
        ]
        
        if not recent_escalations:
            return {"trend": "no_data"}
        
        # Tendance par jour
        daily_counts = defaultdict(int)
        for escalation in recent_escalations:
            day = escalation.created_at.date()
            daily_counts[day] += 1
        
        # Calcul de la tendance (simple moyenne mobile)
        recent_days = sorted(daily_counts.keys())[-3:]
        if len(recent_days) >= 2:
            recent_avg = sum(daily_counts[day] for day in recent_days) / len(recent_days)
            older_days = sorted(daily_counts.keys())[:-3]
            if older_days:
                older_avg = sum(daily_counts[day] for day in older_days) / len(older_days)
                
                if recent_avg > older_avg * 1.2:
                    trend = "increasing"
                elif recent_avg < older_avg * 0.8:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "daily_average": round(sum(daily_counts.values()) / len(daily_counts), 1) if daily_counts else 0,
            "peak_day": max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else None
        }

# Instance globale du moteur d'escalade
_escalation_engine = IntelligentEscalationEngine()

async def create_incident_escalation(incident_id: str, initial_team: TeamType, context: Dict[str, Any]) -> IncidentEscalation:
    """Function helper pour créer une escalade d'incident"""
    return await _escalation_engine.create_escalation(incident_id, initial_team, context)

async def get_escalation_engine() -> IntelligentEscalationEngine:
    """Retourne l'instance du moteur d'escalade"""
    return _escalation_engine

# Configuration des alertes d'escalade
if __name__ == "__main__":
    # Enregistrement des configurations d'alertes
    escalation_configs = [
        AlertConfig(
            name="automatic_escalation_trigger",
            category=AlertCategory.AVAILABILITY,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.ESCALATION,
            conditions=['Incident non résolu dans le délai imparti'],
            actions=['escalate_to_next_level', 'notify_management'],
            ml_enabled=True,
            auto_remediation=False
        ),
        AlertConfig(
            name="business_impact_escalation",
            category=AlertCategory.AVAILABILITY,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.ESCALATION,
            conditions=['Impact business critique détecté'],
            actions=['escalate_to_executive_team', 'send_executive_summary'],
            ml_enabled=True
        )
    ]
    
    for config in escalation_configs:
        register_alert(config)
