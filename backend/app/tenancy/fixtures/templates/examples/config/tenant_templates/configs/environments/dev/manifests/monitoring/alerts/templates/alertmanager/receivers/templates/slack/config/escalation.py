"""
Système d'Escalade Slack Ultra-Intelligent
==========================================

Module de gestion avancée des escalades d'alertes avec support multi-niveau,
policies personnalisables, et intégration avec les systèmes de garde.

Développé par l'équipe Backend Senior sous la direction de Fahed Mlaiel.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import redis.asyncio as redis
from pydantic import BaseModel, Field, validator
import json

from . import SlackSeverity, SlackChannelType
from .utils import SlackUtils

logger = logging.getLogger(__name__)

class EscalationLevel(Enum):
    """Niveaux d'escalade."""
    LEVEL_1 = "level_1"  # Équipe technique
    LEVEL_2 = "level_2"  # Lead technique / Manager
    LEVEL_3 = "level_3"  # Management / Direction
    CRITICAL = "critical"  # Escalade critique immédiate

class EscalationStatus(Enum):
    """États d'escalade."""
    PENDING = auto()
    TRIGGERED = auto()
    ACKNOWLEDGED = auto()
    RESOLVED = auto()
    EXPIRED = auto()
    CANCELLED = auto()

class EscalationTrigger(Enum):
    """Déclencheurs d'escalade."""
    TIME_BASED = "time_based"        # Basé sur le temps
    SEVERITY_BASED = "severity_based"  # Basé sur la sévérité
    FREQUENCY_BASED = "frequency_based"  # Basé sur la fréquence
    MANUAL = "manual"                # Escalade manuelle
    CUSTOM_RULE = "custom_rule"      # Règle personnalisée

@dataclass
class EscalationTarget:
    """Cible d'escalade (personne ou canal)."""
    
    type: str  # "user", "channel", "group"
    identifier: str  # ID Slack, email, etc.
    name: str
    contact_methods: List[str] = field(default_factory=list)  # slack, email, sms, call
    timezone: str = "UTC"
    availability: Dict[str, Any] = field(default_factory=dict)  # Plages horaires
    priority: int = 1  # 1 = haute priorité
    
    def is_available_now(self) -> bool:
        """Vérifie si la cible est disponible maintenant."""
        if not self.availability:
            return True  # Toujours disponible si pas de restriction
        
        now = datetime.utcnow()
        current_hour = now.hour
        current_day = now.strftime("%A").lower()
        
        # Vérifier les heures de disponibilité
        if "hours" in self.availability:
            start_hour = self.availability["hours"].get("start", 0)
            end_hour = self.availability["hours"].get("end", 23)
            if not (start_hour <= current_hour <= end_hour):
                return False
        
        # Vérifier les jours de disponibilité
        if "days" in self.availability:
            available_days = [day.lower() for day in self.availability["days"]]
            if current_day not in available_days:
                return False
        
        return True

@dataclass
class EscalationRule:
    """Règle d'escalade."""
    
    id: str
    name: str
    description: str
    tenant_id: str
    
    # Conditions de déclenchement
    severity_levels: List[SlackSeverity] = field(default_factory=list)
    channel_types: List[SlackChannelType] = field(default_factory=list)
    alert_patterns: List[str] = field(default_factory=list)  # Regex patterns
    
    # Temporisation
    initial_delay: int = 300  # 5 minutes par défaut
    escalation_delays: List[int] = field(default_factory=lambda: [300, 900, 1800])  # 5m, 15m, 30m
    
    # Cibles par niveau
    level_targets: Dict[EscalationLevel, List[EscalationTarget]] = field(default_factory=dict)
    
    # Configuration
    max_escalations: int = 3
    repeat_notifications: bool = True
    repeat_interval: int = 1800  # 30 minutes
    auto_resolve: bool = True
    auto_resolve_timeout: int = 7200  # 2 heures
    
    # État
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches_alert(self, alert_data: Dict[str, Any], severity: SlackSeverity, channel_type: SlackChannelType) -> bool:
        """Vérifie si la règle correspond à une alerte."""
        # Vérifier la sévérité
        if self.severity_levels and severity not in self.severity_levels:
            return False
        
        # Vérifier le type de canal
        if self.channel_types and channel_type not in self.channel_types:
            return False
        
        # Vérifier les patterns d'alerte
        if self.alert_patterns:
            import re
            alert_name = alert_data.get('alertname', '')
            alert_description = alert_data.get('description', '')
            alert_text = f"{alert_name} {alert_description}"
            
            pattern_matched = False
            for pattern in self.alert_patterns:
                if re.search(pattern, alert_text, re.IGNORECASE):
                    pattern_matched = True
                    break
            
            if not pattern_matched:
                return False
        
        return True

@dataclass
class EscalationInstance:
    """Instance d'escalade active."""
    
    id: str = field(default_factory=lambda: SlackUtils.generate_id())
    rule_id: str = ""
    tenant_id: str = ""
    alert_id: str = ""
    alert_data: Dict[str, Any] = field(default_factory=dict)
    
    # État de l'escalade
    status: EscalationStatus = EscalationStatus.PENDING
    current_level: EscalationLevel = EscalationLevel.LEVEL_1
    triggered_levels: Set[EscalationLevel] = field(default_factory=set)
    
    # Temporisation
    created_at: datetime = field(default_factory=datetime.utcnow)
    next_escalation_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Historique des notifications
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    
    # Métadonnées
    acknowledged_by: Optional[str] = None
    resolution_reason: Optional[str] = None

class SlackEscalationManager:
    """
    Gestionnaire ultra-avancé des escalades Slack.
    
    Fonctionnalités:
    - Escalades multi-niveaux configurables
    - Gestion des horaires et disponibilités
    - Intégration avec systèmes de garde
    - Notifications multi-canaux (Slack, email, SMS)
    - Policies d'escalade par tenant
    - Métriques d'escalade détaillées
    - Escalades automatiques et manuelles
    - Gestion des accusations de réception
    - Auto-résolution configurable
    - Audit trail complet
    """
    
    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 webhook_manager = None,
                 template_manager = None):
        """
        Initialise le gestionnaire d'escalades.
        
        Args:
            redis_client: Client Redis pour la persistance
            webhook_manager: Gestionnaire de webhooks
            template_manager: Gestionnaire de templates
        """
        self.redis_client = redis_client
        self.webhook_manager = webhook_manager
        self.template_manager = template_manager
        
        # Configuration des règles d'escalade par tenant
        self.escalation_rules: Dict[str, List[EscalationRule]] = {}
        
        # Instances d'escalade actives
        self.active_escalations: Dict[str, EscalationInstance] = {}
        
        # Tâches d'escalade planifiées
        self.scheduled_tasks: Dict[str, asyncio.Task] = {}
        
        # Configuration par défaut
        self.default_config = {
            'initial_delay': 300,
            'escalation_delays': [300, 900, 1800],
            'max_escalations': 3,
            'repeat_interval': 1800,
            'auto_resolve_timeout': 7200
        }
        
        # Métriques
        self.metrics = {
            'escalations_created': 0,
            'escalations_triggered': 0,
            'escalations_acknowledged': 0,
            'escalations_resolved': 0,
            'escalations_expired': 0,
            'notifications_sent': 0,
            'average_resolution_time': 0,
            'false_escalations': 0
        }
        
        # Gestionnaire de tâches en cours
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        logger.info("SlackEscalationManager initialisé")
    
    async def start(self):
        """Démarre le gestionnaire d'escalades."""
        if self._running:
            return
        
        self._running = True
        
        # Charger les règles d'escalade
        await self._load_escalation_rules()
        
        # Charger les escalades actives
        await self._load_active_escalations()
        
        # Démarrer les tâches de fond
        cleanup_task = asyncio.create_task(self._cleanup_expired_escalations())
        metrics_task = asyncio.create_task(self._update_metrics_periodically())
        
        self._background_tasks.extend([cleanup_task, metrics_task])
        
        logger.info("SlackEscalationManager démarré")
    
    async def stop(self):
        """Arrête le gestionnaire d'escalades."""
        if not self._running:
            return
        
        self._running = False
        
        # Annuler toutes les tâches planifiées
        for task in self.scheduled_tasks.values():
            task.cancel()
        
        # Annuler les tâches de fond
        for task in self._background_tasks:
            task.cancel()
        
        # Attendre la fin des tâches
        if self.scheduled_tasks:
            await asyncio.gather(*self.scheduled_tasks.values(), return_exceptions=True)
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self.scheduled_tasks.clear()
        self._background_tasks.clear()
        
        logger.info("SlackEscalationManager arrêté")
    
    async def register_escalation_rule(self, rule: EscalationRule) -> bool:
        """
        Enregistre une nouvelle règle d'escalade.
        
        Args:
            rule: Règle d'escalade à enregistrer
            
        Returns:
            True si succès, False sinon
        """
        try:
            if rule.tenant_id not in self.escalation_rules:
                self.escalation_rules[rule.tenant_id] = []
            
            # Vérifier si la règle existe déjà
            existing_rule = next(
                (r for r in self.escalation_rules[rule.tenant_id] if r.id == rule.id),
                None
            )
            
            if existing_rule:
                # Mettre à jour la règle existante
                rule.updated_at = datetime.utcnow()
                self.escalation_rules[rule.tenant_id] = [
                    rule if r.id == rule.id else r
                    for r in self.escalation_rules[rule.tenant_id]
                ]
            else:
                # Ajouter la nouvelle règle
                self.escalation_rules[rule.tenant_id].append(rule)
            
            # Persister en Redis
            if self.redis_client:
                await self._persist_escalation_rules(rule.tenant_id)
            
            logger.info(f"Règle d'escalade {rule.id} enregistrée pour tenant {rule.tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur enregistrement règle escalade: {e}")
            return False
    
    async def create_escalation(self,
                              tenant_id: str,
                              alert_id: str,
                              alert_data: Dict[str, Any],
                              severity: SlackSeverity,
                              channel_type: SlackChannelType = SlackChannelType.ALERTS) -> Optional[str]:
        """
        Crée une nouvelle escalade.
        
        Args:
            tenant_id: ID du tenant
            alert_id: ID de l'alerte
            alert_data: Données de l'alerte
            severity: Sévérité de l'alerte
            channel_type: Type de canal
            
        Returns:
            ID de l'escalade créée ou None
        """
        try:
            # Trouver les règles d'escalade applicables
            applicable_rules = await self._find_applicable_rules(
                tenant_id, alert_data, severity, channel_type
            )
            
            if not applicable_rules:
                logger.debug(f"Aucune règle d'escalade applicable pour {tenant_id}/{alert_id}")
                return None
            
            # Utiliser la première règle (la plus prioritaire)
            rule = applicable_rules[0]
            
            # Créer l'instance d'escalade
            escalation = EscalationInstance(
                rule_id=rule.id,
                tenant_id=tenant_id,
                alert_id=alert_id,
                alert_data=alert_data
            )
            
            # Calculer le prochain escalade
            escalation.next_escalation_at = datetime.utcnow() + timedelta(seconds=rule.initial_delay)
            
            # Stocker l'escalade
            self.active_escalations[escalation.id] = escalation
            
            # Planifier la première escalade
            await self._schedule_escalation(escalation.id, rule.initial_delay)
            
            # Persister en Redis
            if self.redis_client:
                await self._persist_escalation_instance(escalation)
            
            self.metrics['escalations_created'] += 1
            
            logger.info(f"Escalade {escalation.id} créée pour alerte {alert_id}")
            return escalation.id
            
        except Exception as e:
            logger.error(f"Erreur création escalade: {e}")
            return None
    
    async def acknowledge_escalation(self,
                                   escalation_id: str,
                                   acknowledged_by: str,
                                   reason: Optional[str] = None) -> bool:
        """
        Marque une escalade comme accusée réception.
        
        Args:
            escalation_id: ID de l'escalade
            acknowledged_by: Utilisateur qui a accusé réception
            reason: Raison optionnelle
            
        Returns:
            True si succès, False sinon
        """
        try:
            if escalation_id not in self.active_escalations:
                logger.warning(f"Escalade {escalation_id} introuvable")
                return False
            
            escalation = self.active_escalations[escalation_id]
            
            # Mettre à jour l'état
            escalation.status = EscalationStatus.ACKNOWLEDGED
            escalation.acknowledged_at = datetime.utcnow()
            escalation.acknowledged_by = acknowledged_by
            
            # Annuler les escalades futures
            if escalation_id in self.scheduled_tasks:
                self.scheduled_tasks[escalation_id].cancel()
                del self.scheduled_tasks[escalation_id]
            
            # Envoyer notification d'accusé de réception
            await self._send_acknowledgment_notification(escalation, reason)
            
            # Persister les changements
            if self.redis_client:
                await self._persist_escalation_instance(escalation)
            
            self.metrics['escalations_acknowledged'] += 1
            
            logger.info(f"Escalade {escalation_id} accusée réception par {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur accusé réception escalade: {e}")
            return False
    
    async def resolve_escalation(self,
                               escalation_id: str,
                               resolved_by: Optional[str] = None,
                               reason: Optional[str] = None) -> bool:
        """
        Résout une escalade.
        
        Args:
            escalation_id: ID de l'escalade
            resolved_by: Utilisateur qui a résolu
            reason: Raison de résolution
            
        Returns:
            True si succès, False sinon
        """
        try:
            if escalation_id not in self.active_escalations:
                logger.warning(f"Escalade {escalation_id} introuvable")
                return False
            
            escalation = self.active_escalations[escalation_id]
            
            # Mettre à jour l'état
            escalation.status = EscalationStatus.RESOLVED
            escalation.resolved_at = datetime.utcnow()
            escalation.resolution_reason = reason
            
            # Annuler les escalades futures
            if escalation_id in self.scheduled_tasks:
                self.scheduled_tasks[escalation_id].cancel()
                del self.scheduled_tasks[escalation_id]
            
            # Envoyer notification de résolution
            await self._send_resolution_notification(escalation, resolved_by, reason)
            
            # Persister et archiver
            if self.redis_client:
                await self._archive_escalation_instance(escalation)
            
            # Retirer des escalades actives
            del self.active_escalations[escalation_id]
            
            self.metrics['escalations_resolved'] += 1
            
            logger.info(f"Escalade {escalation_id} résolue")
            return True
            
        except Exception as e:
            logger.error(f"Erreur résolution escalade: {e}")
            return False
    
    async def trigger_manual_escalation(self,
                                      tenant_id: str,
                                      alert_data: Dict[str, Any],
                                      target_level: EscalationLevel,
                                      triggered_by: str,
                                      reason: str) -> Optional[str]:
        """
        Déclenche une escalade manuelle.
        
        Args:
            tenant_id: ID du tenant
            alert_data: Données de l'alerte
            target_level: Niveau d'escalade cible
            triggered_by: Utilisateur qui déclenche
            reason: Raison de l'escalade manuelle
            
        Returns:
            ID de l'escalade créée ou None
        """
        try:
            # Créer une instance d'escalade manuelle
            escalation = EscalationInstance(
                rule_id="manual",
                tenant_id=tenant_id,
                alert_id=f"manual_{SlackUtils.generate_id()}",
                alert_data=alert_data,
                current_level=target_level,
                status=EscalationStatus.TRIGGERED
            )
            
            # Ajouter l'information de déclenchement manuel
            escalation.notifications_sent.append({
                'type': 'manual_trigger',
                'triggered_by': triggered_by,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Stocker l'escalade
            self.active_escalations[escalation.id] = escalation
            
            # Déclencher immédiatement l'escalade au niveau demandé
            await self._execute_escalation_level(escalation, target_level)
            
            # Persister en Redis
            if self.redis_client:
                await self._persist_escalation_instance(escalation)
            
            self.metrics['escalations_created'] += 1
            self.metrics['escalations_triggered'] += 1
            
            logger.info(f"Escalade manuelle {escalation.id} déclenchée par {triggered_by}")
            return escalation.id
            
        except Exception as e:
            logger.error(f"Erreur escalade manuelle: {e}")
            return None
    
    async def _find_applicable_rules(self,
                                   tenant_id: str,
                                   alert_data: Dict[str, Any],
                                   severity: SlackSeverity,
                                   channel_type: SlackChannelType) -> List[EscalationRule]:
        """Trouve les règles d'escalade applicables."""
        applicable_rules = []
        
        tenant_rules = self.escalation_rules.get(tenant_id, [])
        
        for rule in tenant_rules:
            if rule.enabled and rule.matches_alert(alert_data, severity, channel_type):
                applicable_rules.append(rule)
        
        # Trier par priorité (règles plus spécifiques en premier)
        applicable_rules.sort(key=lambda r: (
            len(r.alert_patterns),
            len(r.severity_levels),
            len(r.channel_types)
        ), reverse=True)
        
        return applicable_rules
    
    async def _schedule_escalation(self, escalation_id: str, delay_seconds: int):
        """Planifie une escalade future."""
        try:
            async def escalation_task():
                await asyncio.sleep(delay_seconds)
                await self._execute_next_escalation(escalation_id)
            
            task = asyncio.create_task(escalation_task())
            self.scheduled_tasks[escalation_id] = task
            
        except Exception as e:
            logger.error(f"Erreur planification escalade {escalation_id}: {e}")
    
    async def _execute_next_escalation(self, escalation_id: str):
        """Exécute la prochaine escalade."""
        try:
            if escalation_id not in self.active_escalations:
                return
            
            escalation = self.active_escalations[escalation_id]
            
            # Vérifier si l'escalade est toujours active
            if escalation.status in [EscalationStatus.ACKNOWLEDGED, EscalationStatus.RESOLVED]:
                return
            
            # Obtenir la règle d'escalade
            rule = await self._get_escalation_rule(escalation.tenant_id, escalation.rule_id)
            if not rule:
                logger.error(f"Règle d'escalade {escalation.rule_id} introuvable")
                return
            
            # Déterminer le niveau d'escalade
            current_level = escalation.current_level
            next_level = self._get_next_escalation_level(current_level, escalation.triggered_levels)
            
            if next_level is None:
                # Pas de niveau suivant, marquer comme expiré
                escalation.status = EscalationStatus.EXPIRED
                self.metrics['escalations_expired'] += 1
                logger.info(f"Escalade {escalation_id} expirée")
                return
            
            # Exécuter l'escalade au niveau suivant
            await self._execute_escalation_level(escalation, next_level)
            
            # Mettre à jour l'état
            escalation.current_level = next_level
            escalation.triggered_levels.add(next_level)
            escalation.status = EscalationStatus.TRIGGERED
            
            # Planifier l'escalade suivante si applicable
            if len(escalation.triggered_levels) < rule.max_escalations:
                delay_index = len(escalation.triggered_levels) - 1
                if delay_index < len(rule.escalation_delays):
                    next_delay = rule.escalation_delays[delay_index]
                    escalation.next_escalation_at = datetime.utcnow() + timedelta(seconds=next_delay)
                    await self._schedule_escalation(escalation_id, next_delay)
            
            # Persister les changements
            if self.redis_client:
                await self._persist_escalation_instance(escalation)
            
            self.metrics['escalations_triggered'] += 1
            
        except Exception as e:
            logger.error(f"Erreur exécution escalade {escalation_id}: {e}")
        finally:
            # Nettoyer la tâche planifiée
            if escalation_id in self.scheduled_tasks:
                del self.scheduled_tasks[escalation_id]
    
    def _get_next_escalation_level(self, 
                                 current_level: EscalationLevel, 
                                 triggered_levels: Set[EscalationLevel]) -> Optional[EscalationLevel]:
        """Détermine le prochain niveau d'escalade."""
        levels_order = [
            EscalationLevel.LEVEL_1,
            EscalationLevel.LEVEL_2,
            EscalationLevel.LEVEL_3,
            EscalationLevel.CRITICAL
        ]
        
        try:
            current_index = levels_order.index(current_level)
            
            # Chercher le prochain niveau non encore déclenché
            for i in range(current_index + 1, len(levels_order)):
                next_level = levels_order[i]
                if next_level not in triggered_levels:
                    return next_level
            
            return None
            
        except ValueError:
            # Niveau actuel non trouvé, commencer au niveau 1
            return EscalationLevel.LEVEL_1 if EscalationLevel.LEVEL_1 not in triggered_levels else None
    
    async def _execute_escalation_level(self, escalation: EscalationInstance, level: EscalationLevel):
        """Exécute une escalade à un niveau spécifique."""
        try:
            # Obtenir la règle d'escalade
            rule = await self._get_escalation_rule(escalation.tenant_id, escalation.rule_id)
            if not rule:
                return
            
            # Obtenir les cibles pour ce niveau
            targets = rule.level_targets.get(level, [])
            if not targets:
                logger.warning(f"Aucune cible définie pour niveau {level.value}")
                return
            
            # Filtrer les cibles disponibles
            available_targets = [target for target in targets if target.is_available_now()]
            
            if not available_targets:
                logger.warning(f"Aucune cible disponible pour niveau {level.value}")
                # Utiliser toutes les cibles si aucune n'est disponible
                available_targets = targets
            
            # Envoyer les notifications
            for target in available_targets:
                await self._send_escalation_notification(escalation, target, level)
            
            # Enregistrer les notifications envoyées
            escalation.notifications_sent.append({
                'level': level.value,
                'targets': [{'type': t.type, 'identifier': t.identifier} for t in available_targets],
                'timestamp': datetime.utcnow().isoformat()
            })
            
            self.metrics['notifications_sent'] += len(available_targets)
            
            logger.info(f"Escalade niveau {level.value} exécutée pour {escalation.id}")
            
        except Exception as e:
            logger.error(f"Erreur exécution escalade niveau {level.value}: {e}")
    
    async def _send_escalation_notification(self,
                                          escalation: EscalationInstance,
                                          target: EscalationTarget,
                                          level: EscalationLevel):
        """Envoie une notification d'escalade à une cible."""
        try:
            # Préparer les données pour le template
            template_data = {
                'escalation_id': escalation.id,
                'alert_id': escalation.alert_id,
                'escalation_level': level.value,
                'target_name': target.name,
                'tenant_id': escalation.tenant_id,
                'urgency': 'HIGH' if level in [EscalationLevel.LEVEL_3, EscalationLevel.CRITICAL] else 'MEDIUM',
                **escalation.alert_data
            }
            
            # Sélectionner le template approprié
            template_id = f"escalation_{level.value}"
            
            # Formater le message
            if self.template_manager:
                message = await self.template_manager.render_template(
                    template_id, template_data, escalation.tenant_id
                )
            else:
                message = self._create_default_escalation_message(escalation, target, level)
            
            if not message:
                logger.error(f"Impossible de formater le message d'escalade pour {target.identifier}")
                return
            
            # Envoyer selon le type de cible
            if target.type == "channel" and self.webhook_manager:
                # Obtenir l'URL du webhook pour le canal
                webhook_url = await self._get_channel_webhook_url(escalation.tenant_id, target.identifier)
                if webhook_url:
                    await self.webhook_manager.send_webhook(
                        escalation.tenant_id,
                        webhook_url,
                        message,
                        severity=SlackSeverity.HIGH,
                        priority=True
                    )
            
            elif target.type == "user":
                # Envoyer un message direct (nécessiterait l'API Slack)
                await self._send_direct_message(escalation.tenant_id, target.identifier, message)
            
            logger.debug(f"Notification d'escalade envoyée à {target.identifier}")
            
        except Exception as e:
            logger.error(f"Erreur envoi notification escalade à {target.identifier}: {e}")
    
    def _create_default_escalation_message(self,
                                         escalation: EscalationInstance,
                                         target: EscalationTarget,
                                         level: EscalationLevel) -> Dict[str, Any]:
        """Crée un message d'escalade par défaut."""
        alert_name = escalation.alert_data.get('alertname', 'Alerte Inconnue')
        severity_emoji = "🚨" if level == EscalationLevel.CRITICAL else "⚠️"
        
        message = {
            "text": f"{severity_emoji} ESCALADE {level.value.upper()} - {alert_name}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{severity_emoji} ESCALADE {level.value.upper()}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Alerte:* {alert_name}\\n*Niveau:* {level.value}\\n*Cible:* {target.name}\\n*ID Escalade:* `{escalation.id}`"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Description:* {escalation.alert_data.get('description', 'Aucune description disponible')}"
                    }
                }
            ]
        }
        
        return message
    
    async def _send_acknowledgment_notification(self, escalation: EscalationInstance, reason: Optional[str]):
        """Envoie une notification d'accusé de réception."""
        try:
            # Message d'accusé de réception
            message = {
                "text": f"✅ Escalade {escalation.id} accusée réception",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"✅ *Accusé de réception*\\n*Escalade:* {escalation.id}\\n*Par:* {escalation.acknowledged_by}\\n*Heure:* {escalation.acknowledged_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        }
                    }
                ]
            }
            
            if reason:
                message["blocks"].append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Raison:* {reason}"
                    }
                })
            
            # Envoyer aux canaux de suivi
            await self._send_to_tracking_channels(escalation.tenant_id, message)
            
        except Exception as e:
            logger.error(f"Erreur envoi notification accusé réception: {e}")
    
    async def _send_resolution_notification(self,
                                          escalation: EscalationInstance,
                                          resolved_by: Optional[str],
                                          reason: Optional[str]):
        """Envoie une notification de résolution."""
        try:
            duration = ""
            if escalation.resolved_at and escalation.created_at:
                delta = escalation.resolved_at - escalation.created_at
                duration = f" (durée: {delta})"
            
            message = {
                "text": f"✅ Escalade {escalation.id} résolue",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"✅ *Résolution d'escalade*\\n*Escalade:* {escalation.id}\\n*Heure:* {escalation.resolved_at.strftime('%Y-%m-%d %H:%M:%S UTC')}{duration}"
                        }
                    }
                ]
            }
            
            if resolved_by:
                message["blocks"][0]["text"]["text"] += f"\\n*Par:* {resolved_by}"
            
            if reason:
                message["blocks"].append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Raison:* {reason}"
                    }
                })
            
            # Envoyer aux canaux de suivi
            await self._send_to_tracking_channels(escalation.tenant_id, message)
            
        except Exception as e:
            logger.error(f"Erreur envoi notification résolution: {e}")
    
    async def _send_to_tracking_channels(self, tenant_id: str, message: Dict[str, Any]):
        """Envoie un message aux canaux de suivi des escalades."""
        try:
            # Canaux de suivi par défaut
            tracking_channels = ["#escalations", "#monitoring"]
            
            if self.webhook_manager:
                for channel in tracking_channels:
                    webhook_url = await self._get_channel_webhook_url(tenant_id, channel)
                    if webhook_url:
                        await self.webhook_manager.send_webhook(
                            tenant_id,
                            webhook_url,
                            message,
                            severity=SlackSeverity.INFO
                        )
            
        except Exception as e:
            logger.error(f"Erreur envoi aux canaux de suivi: {e}")
    
    async def _get_channel_webhook_url(self, tenant_id: str, channel_identifier: str) -> Optional[str]:
        """Récupère l'URL du webhook pour un canal."""
        # Cette méthode devrait être implémentée pour récupérer
        # les URLs de webhooks depuis la configuration
        # Pour l'instant, retourner une URL d'exemple
        return f"https://hooks.slack.com/services/EXAMPLE/{tenant_id}/{channel_identifier}"
    
    async def _send_direct_message(self, tenant_id: str, user_id: str, message: Dict[str, Any]):
        """Envoie un message direct à un utilisateur."""
        # Cette méthode nécessiterait l'API Slack pour les messages directs
        # Pour l'instant, logger l'intention
        logger.info(f"Message direct à envoyer à {user_id}: {message.get('text', '')}")
    
    async def _get_escalation_rule(self, tenant_id: str, rule_id: str) -> Optional[EscalationRule]:
        """Récupère une règle d'escalade."""
        tenant_rules = self.escalation_rules.get(tenant_id, [])
        return next((rule for rule in tenant_rules if rule.id == rule_id), None)
    
    async def _load_escalation_rules(self):
        """Charge les règles d'escalade depuis Redis."""
        try:
            if not self.redis_client:
                return
            
            # Récupérer toutes les clés de règles
            pattern = "escalation_rules:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                tenant_id = key.decode().split(':')[1]
                rules_data = await self.redis_client.hgetall(key)
                
                tenant_rules = []
                for rule_key, rule_data in rules_data.items():
                    rule_dict = json.loads(rule_data.decode())
                    rule = self._dict_to_escalation_rule(rule_dict)
                    tenant_rules.append(rule)
                
                self.escalation_rules[tenant_id] = tenant_rules
            
            logger.info(f"Règles d'escalade chargées pour {len(self.escalation_rules)} tenants")
            
        except Exception as e:
            logger.error(f"Erreur chargement règles escalade: {e}")
    
    async def _load_active_escalations(self):
        """Charge les escalades actives depuis Redis."""
        try:
            if not self.redis_client:
                return
            
            # Récupérer toutes les escalades actives
            pattern = "active_escalation:*"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                escalation_data = await self.redis_client.get(key)
                if escalation_data:
                    escalation_dict = json.loads(escalation_data.decode())
                    escalation = self._dict_to_escalation_instance(escalation_dict)
                    self.active_escalations[escalation.id] = escalation
                    
                    # Replanifier si nécessaire
                    if escalation.next_escalation_at and escalation.status == EscalationStatus.PENDING:
                        delay = max(0, (escalation.next_escalation_at - datetime.utcnow()).total_seconds())
                        if delay > 0:
                            await self._schedule_escalation(escalation.id, int(delay))
            
            logger.info(f"{len(self.active_escalations)} escalades actives chargées")
            
        except Exception as e:
            logger.error(f"Erreur chargement escalades actives: {e}")
    
    async def _persist_escalation_rules(self, tenant_id: str):
        """Persiste les règles d'escalade en Redis."""
        try:
            if not self.redis_client:
                return
            
            key = f"escalation_rules:{tenant_id}"
            rules_data = {}
            
            for rule in self.escalation_rules.get(tenant_id, []):
                rule_dict = self._escalation_rule_to_dict(rule)
                rules_data[rule.id] = json.dumps(rule_dict)
            
            await self.redis_client.hset(key, mapping=rules_data)
            await self.redis_client.expire(key, 86400)  # 24h
            
        except Exception as e:
            logger.error(f"Erreur persistance règles escalade: {e}")
    
    async def _persist_escalation_instance(self, escalation: EscalationInstance):
        """Persiste une instance d'escalade en Redis."""
        try:
            if not self.redis_client:
                return
            
            key = f"active_escalation:{escalation.id}"
            escalation_dict = self._escalation_instance_to_dict(escalation)
            
            await self.redis_client.set(key, json.dumps(escalation_dict))
            await self.redis_client.expire(key, 86400)  # 24h
            
        except Exception as e:
            logger.error(f"Erreur persistance instance escalade: {e}")
    
    async def _archive_escalation_instance(self, escalation: EscalationInstance):
        """Archive une instance d'escalade."""
        try:
            if not self.redis_client:
                return
            
            # Déplacer vers l'archive
            archive_key = f"archived_escalation:{escalation.id}"
            escalation_dict = self._escalation_instance_to_dict(escalation)
            
            await self.redis_client.set(archive_key, json.dumps(escalation_dict))
            await self.redis_client.expire(archive_key, 604800)  # 7 jours
            
            # Supprimer de l'actif
            active_key = f"active_escalation:{escalation.id}"
            await self.redis_client.delete(active_key)
            
        except Exception as e:
            logger.error(f"Erreur archivage escalade: {e}")
    
    def _escalation_rule_to_dict(self, rule: EscalationRule) -> Dict[str, Any]:
        """Convertit une règle d'escalade en dictionnaire."""
        rule_dict = {
            'id': rule.id,
            'name': rule.name,
            'description': rule.description,
            'tenant_id': rule.tenant_id,
            'severity_levels': [s.value for s in rule.severity_levels],
            'channel_types': [c.value for c in rule.channel_types],
            'alert_patterns': rule.alert_patterns,
            'initial_delay': rule.initial_delay,
            'escalation_delays': rule.escalation_delays,
            'level_targets': {},
            'max_escalations': rule.max_escalations,
            'repeat_notifications': rule.repeat_notifications,
            'repeat_interval': rule.repeat_interval,
            'auto_resolve': rule.auto_resolve,
            'auto_resolve_timeout': rule.auto_resolve_timeout,
            'enabled': rule.enabled,
            'created_at': rule.created_at.isoformat(),
            'updated_at': rule.updated_at.isoformat()
        }
        
        # Convertir les cibles par niveau
        for level, targets in rule.level_targets.items():
            rule_dict['level_targets'][level.value] = [
                {
                    'type': t.type,
                    'identifier': t.identifier,
                    'name': t.name,
                    'contact_methods': t.contact_methods,
                    'timezone': t.timezone,
                    'availability': t.availability,
                    'priority': t.priority
                }
                for t in targets
            ]
        
        return rule_dict
    
    def _dict_to_escalation_rule(self, rule_dict: Dict[str, Any]) -> EscalationRule:
        """Convertit un dictionnaire en règle d'escalade."""
        # Convertir les énums
        severity_levels = [SlackSeverity(s) for s in rule_dict.get('severity_levels', [])]
        channel_types = [SlackChannelType(c) for c in rule_dict.get('channel_types', [])]
        
        # Convertir les cibles par niveau
        level_targets = {}
        for level_str, targets_data in rule_dict.get('level_targets', {}).items():
            level = EscalationLevel(level_str)
            targets = [
                EscalationTarget(
                    type=t['type'],
                    identifier=t['identifier'],
                    name=t['name'],
                    contact_methods=t.get('contact_methods', []),
                    timezone=t.get('timezone', 'UTC'),
                    availability=t.get('availability', {}),
                    priority=t.get('priority', 1)
                )
                for t in targets_data
            ]
            level_targets[level] = targets
        
        return EscalationRule(
            id=rule_dict['id'],
            name=rule_dict['name'],
            description=rule_dict['description'],
            tenant_id=rule_dict['tenant_id'],
            severity_levels=severity_levels,
            channel_types=channel_types,
            alert_patterns=rule_dict.get('alert_patterns', []),
            initial_delay=rule_dict.get('initial_delay', 300),
            escalation_delays=rule_dict.get('escalation_delays', [300, 900, 1800]),
            level_targets=level_targets,
            max_escalations=rule_dict.get('max_escalations', 3),
            repeat_notifications=rule_dict.get('repeat_notifications', True),
            repeat_interval=rule_dict.get('repeat_interval', 1800),
            auto_resolve=rule_dict.get('auto_resolve', True),
            auto_resolve_timeout=rule_dict.get('auto_resolve_timeout', 7200),
            enabled=rule_dict.get('enabled', True),
            created_at=datetime.fromisoformat(rule_dict['created_at']),
            updated_at=datetime.fromisoformat(rule_dict['updated_at'])
        )
    
    def _escalation_instance_to_dict(self, escalation: EscalationInstance) -> Dict[str, Any]:
        """Convertit une instance d'escalade en dictionnaire."""
        return {
            'id': escalation.id,
            'rule_id': escalation.rule_id,
            'tenant_id': escalation.tenant_id,
            'alert_id': escalation.alert_id,
            'alert_data': escalation.alert_data,
            'status': escalation.status.name,
            'current_level': escalation.current_level.value,
            'triggered_levels': [level.value for level in escalation.triggered_levels],
            'created_at': escalation.created_at.isoformat(),
            'next_escalation_at': escalation.next_escalation_at.isoformat() if escalation.next_escalation_at else None,
            'acknowledged_at': escalation.acknowledged_at.isoformat() if escalation.acknowledged_at else None,
            'resolved_at': escalation.resolved_at.isoformat() if escalation.resolved_at else None,
            'notifications_sent': escalation.notifications_sent,
            'acknowledged_by': escalation.acknowledged_by,
            'resolution_reason': escalation.resolution_reason
        }
    
    def _dict_to_escalation_instance(self, escalation_dict: Dict[str, Any]) -> EscalationInstance:
        """Convertit un dictionnaire en instance d'escalade."""
        return EscalationInstance(
            id=escalation_dict['id'],
            rule_id=escalation_dict['rule_id'],
            tenant_id=escalation_dict['tenant_id'],
            alert_id=escalation_dict['alert_id'],
            alert_data=escalation_dict['alert_data'],
            status=EscalationStatus[escalation_dict['status']],
            current_level=EscalationLevel(escalation_dict['current_level']),
            triggered_levels={EscalationLevel(level) for level in escalation_dict['triggered_levels']},
            created_at=datetime.fromisoformat(escalation_dict['created_at']),
            next_escalation_at=datetime.fromisoformat(escalation_dict['next_escalation_at']) if escalation_dict['next_escalation_at'] else None,
            acknowledged_at=datetime.fromisoformat(escalation_dict['acknowledged_at']) if escalation_dict['acknowledged_at'] else None,
            resolved_at=datetime.fromisoformat(escalation_dict['resolved_at']) if escalation_dict['resolved_at'] else None,
            notifications_sent=escalation_dict['notifications_sent'],
            acknowledged_by=escalation_dict.get('acknowledged_by'),
            resolution_reason=escalation_dict.get('resolution_reason')
        )
    
    async def _cleanup_expired_escalations(self):
        """Nettoie périodiquement les escalades expirées."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                now = datetime.utcnow()
                expired_escalations = []
                
                for escalation_id, escalation in self.active_escalations.items():
                    # Vérifier l'auto-résolution
                    rule = await self._get_escalation_rule(escalation.tenant_id, escalation.rule_id)
                    if rule and rule.auto_resolve:
                        max_age = timedelta(seconds=rule.auto_resolve_timeout)
                        if now - escalation.created_at > max_age:
                            expired_escalations.append(escalation_id)
                
                # Résoudre les escalades expirées
                for escalation_id in expired_escalations:
                    await self.resolve_escalation(
                        escalation_id,
                        resolved_by="system",
                        reason="Auto-résolution par timeout"
                    )
                
                if expired_escalations:
                    logger.info(f"{len(expired_escalations)} escalades auto-résolues")
                
            except Exception as e:
                logger.error(f"Erreur nettoyage escalades expirées: {e}")
    
    async def _update_metrics_periodically(self):
        """Met à jour les métriques périodiquement."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Toutes les minutes
                
                # Calculer le temps de résolution moyen
                if self.redis_client:
                    resolved_escalations = await self._get_recent_resolved_escalations()
                    if resolved_escalations:
                        total_time = sum(
                            (escalation['resolved_at'] - escalation['created_at']).total_seconds()
                            for escalation in resolved_escalations
                        )
                        self.metrics['average_resolution_time'] = total_time / len(resolved_escalations)
                
            except Exception as e:
                logger.error(f"Erreur mise à jour métriques: {e}")
    
    async def _get_recent_resolved_escalations(self) -> List[Dict[str, Any]]:
        """Récupère les escalades résolues récemment."""
        try:
            if not self.redis_client:
                return []
            
            # Récupérer les escalades archivées des dernières 24h
            pattern = "archived_escalation:*"
            keys = await self.redis_client.keys(pattern)
            
            recent_escalations = []
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            for key in keys:
                escalation_data = await self.redis_client.get(key)
                if escalation_data:
                    escalation_dict = json.loads(escalation_data.decode())
                    if escalation_dict.get('resolved_at'):
                        resolved_at = datetime.fromisoformat(escalation_dict['resolved_at'])
                        if resolved_at > cutoff_time:
                            recent_escalations.append({
                                'created_at': datetime.fromisoformat(escalation_dict['created_at']),
                                'resolved_at': resolved_at
                            })
            
            return recent_escalations
            
        except Exception as e:
            logger.error(f"Erreur récupération escalades résolues: {e}")
            return []
    
    async def get_escalation_status(self, escalation_id: str) -> Optional[Dict[str, Any]]:
        """Récupère le statut d'une escalade."""
        try:
            if escalation_id in self.active_escalations:
                escalation = self.active_escalations[escalation_id]
                return {
                    'id': escalation.id,
                    'status': escalation.status.name,
                    'current_level': escalation.current_level.value,
                    'created_at': escalation.created_at.isoformat(),
                    'next_escalation_at': escalation.next_escalation_at.isoformat() if escalation.next_escalation_at else None,
                    'acknowledged_at': escalation.acknowledged_at.isoformat() if escalation.acknowledged_at else None,
                    'notifications_count': len(escalation.notifications_sent)
                }
            
            # Chercher dans les archives
            if self.redis_client:
                archive_key = f"archived_escalation:{escalation_id}"
                escalation_data = await self.redis_client.get(archive_key)
                if escalation_data:
                    escalation_dict = json.loads(escalation_data.decode())
                    return {
                        'id': escalation_dict['id'],
                        'status': escalation_dict['status'],
                        'resolved_at': escalation_dict.get('resolved_at'),
                        'resolution_reason': escalation_dict.get('resolution_reason')
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur récupération statut escalade {escalation_id}: {e}")
            return None
    
    async def list_active_escalations(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Liste les escalades actives."""
        try:
            escalations = []
            
            for escalation in self.active_escalations.values():
                if tenant_id is None or escalation.tenant_id == tenant_id:
                    escalations.append({
                        'id': escalation.id,
                        'tenant_id': escalation.tenant_id,
                        'alert_id': escalation.alert_id,
                        'status': escalation.status.name,
                        'current_level': escalation.current_level.value,
                        'created_at': escalation.created_at.isoformat(),
                        'next_escalation_at': escalation.next_escalation_at.isoformat() if escalation.next_escalation_at else None
                    })
            
            # Trier par date de création
            escalations.sort(key=lambda x: x['created_at'], reverse=True)
            
            return escalations
            
        except Exception as e:
            logger.error(f"Erreur listage escalades actives: {e}")
            return []
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du gestionnaire d'escalades."""
        return {
            **self.metrics,
            'active_escalations': len(self.active_escalations),
            'scheduled_tasks': len(self.scheduled_tasks),
            'registered_tenants': len(self.escalation_rules),
            'total_rules': sum(len(rules) for rules in self.escalation_rules.values()),
            'running': self._running
        }
    
    def __repr__(self) -> str:
        return f"SlackEscalationManager(active={len(self.active_escalations)}, rules={sum(len(r) for r in self.escalation_rules.values())})"
