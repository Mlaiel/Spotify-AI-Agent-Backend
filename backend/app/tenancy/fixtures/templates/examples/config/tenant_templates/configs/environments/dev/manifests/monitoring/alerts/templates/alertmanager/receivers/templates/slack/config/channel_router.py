"""
Routeur de Canaux Slack Ultra-Intelligent
=========================================

Module de routage intelligent des messages Slack basé sur la criticité,
le type d'alerte, le tenant et les règles de routage personnalisées.

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
import re
import json

from . import SlackSeverity, SlackChannelType
from .utils import SlackUtils

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Stratégies de routage disponibles."""
    SEVERITY_BASED = "severity"
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    TENANT_SPECIFIC = "tenant_specific"
    TIME_BASED = "time_based"
    CUSTOM = "custom"

class ChannelStatus(Enum):
    """États des canaux Slack."""
    ACTIVE = auto()
    INACTIVE = auto()
    MAINTENANCE = auto()
    RATE_LIMITED = auto()
    ERROR = auto()

@dataclass
class ChannelInfo:
    """Informations sur un canal Slack."""
    
    id: str
    name: str
    workspace_id: str
    is_private: bool = False
    members_count: int = 0
    status: ChannelStatus = ChannelStatus.ACTIVE
    last_message_at: Optional[datetime] = None
    rate_limit: int = 100  # messages par heure
    current_usage: int = 0
    reset_time: datetime = field(default_factory=datetime.utcnow)
    tags: Set[str] = field(default_factory=set)
    priority: int = 1  # 1 = haute priorité, 5 = basse priorité
    
    def is_available(self) -> bool:
        """Vérifie si le canal est disponible."""
        if self.status != ChannelStatus.ACTIVE:
            return False
        
        # Vérifier le rate limiting
        now = datetime.utcnow()
        if now >= self.reset_time:
            self.current_usage = 0
            self.reset_time = now + timedelta(hours=1)
        
        return self.current_usage < self.rate_limit

@dataclass
class RoutingRule:
    """Règle de routage personnalisée."""
    
    id: str
    name: str
    description: str
    conditions: Dict[str, Any]  # Conditions de déclenchement
    target_channels: List[str]  # Canaux cibles
    priority: int = 1
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches(self, context: Dict[str, Any]) -> bool:
        """Vérifie si la règle correspond au contexte."""
        try:
            for condition_key, condition_value in self.conditions.items():
                if not self._evaluate_condition(condition_key, condition_value, context):
                    return False
            return True
        except Exception as e:
            logger.error(f"Erreur évaluation règle {self.id}: {e}")
            return False
    
    def _evaluate_condition(self, key: str, condition: Any, context: Dict[str, Any]) -> bool:
        """Évalue une condition spécifique."""
        context_value = context.get(key)
        
        if isinstance(condition, dict):
            # Conditions avancées
            if "$eq" in condition:
                return context_value == condition["$eq"]
            elif "$ne" in condition:
                return context_value != condition["$ne"]
            elif "$in" in condition:
                return context_value in condition["$in"]
            elif "$regex" in condition:
                if isinstance(context_value, str):
                    return bool(re.match(condition["$regex"], context_value))
                return False
            elif "$exists" in condition:
                return (key in context) == condition["$exists"]
            elif "$gt" in condition:
                return context_value > condition["$gt"]
            elif "$lt" in condition:
                return context_value < condition["$lt"]
        else:
            # Condition simple d'égalité
            return context_value == condition
        
        return False

class SlackChannelRouter:
    """
    Routeur ultra-intelligent pour les canaux Slack.
    
    Fonctionnalités:
    - Routage basé sur la sévérité et le type d'alerte
    - Règles de routage personnalisables et dynamiques
    - Load balancing intelligent entre canaux
    - Failover automatique en cas d'indisponibilité
    - Gestion du rate limiting par canal
    - Routage tenant-aware
    - Escalade automatique pour alertes critiques
    - Métriques de routage détaillées
    - Cache Redis pour performances optimales
    """
    
    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 default_strategy: RoutingStrategy = RoutingStrategy.SEVERITY_BASED):
        """
        Initialise le routeur de canaux.
        
        Args:
            redis_client: Client Redis pour le cache
            default_strategy: Stratégie de routage par défaut
        """
        self.redis_client = redis_client
        self.default_strategy = default_strategy
        
        # Configuration des canaux par tenant
        self.tenant_channels: Dict[str, Dict[str, ChannelInfo]] = {}
        
        # Canaux globaux par sévérité
        self.global_channels: Dict[SlackSeverity, List[str]] = {
            SlackSeverity.CRITICAL: ["#critical-alerts", "#escalation"],
            SlackSeverity.HIGH: ["#high-alerts", "#monitoring"],
            SlackSeverity.MEDIUM: ["#medium-alerts", "#monitoring"],
            SlackSeverity.LOW: ["#low-alerts"],
            SlackSeverity.INFO: ["#info", "#general"]
        }
        
        # Canaux par type
        self.type_channels: Dict[SlackChannelType, List[str]] = {
            SlackChannelType.ALERTS: ["#alerts", "#monitoring"],
            SlackChannelType.MONITORING: ["#monitoring", "#ops"],
            SlackChannelType.INCIDENTS: ["#incidents", "#escalation"],
            SlackChannelType.ESCALATION: ["#escalation", "#management"],
            SlackChannelType.AUDIT: ["#audit", "#compliance"],
            SlackChannelType.SYSTEM: ["#system", "#admin"]
        }
        
        # Règles de routage personnalisées
        self.routing_rules: Dict[str, List[RoutingRule]] = {}
        
        # Canaux de fallback
        self.fallback_channels = ["#alerts", "#general"]
        
        # Round-robin state
        self._round_robin_state: Dict[str, int] = {}
        
        # Métriques
        self.metrics = {
            'routing_requests': 0,
            'successful_routes': 0,
            'failed_routes': 0,
            'fallback_used': 0,
            'rules_matched': 0,
            'rate_limited': 0,
            'escalations': 0
        }
        
        logger.info("SlackChannelRouter initialisé")
    
    async def register_tenant_channels(self, 
                                     tenant_id: str, 
                                     channels: Dict[str, ChannelInfo]) -> bool:
        """
        Enregistre les canaux d'un tenant.
        
        Args:
            tenant_id: ID du tenant
            channels: Dictionnaire des canaux
            
        Returns:
            True si succès, False sinon
        """
        try:
            self.tenant_channels[tenant_id] = channels
            
            # Persister en Redis
            if self.redis_client:
                await self._persist_tenant_channels(tenant_id, channels)
            
            logger.info(f"Canaux tenant {tenant_id} enregistrés: {len(channels)} canaux")
            return True
            
        except Exception as e:
            logger.error(f"Erreur enregistrement canaux tenant {tenant_id}: {e}")
            return False
    
    async def add_routing_rule(self, 
                             tenant_id: str, 
                             rule: RoutingRule) -> bool:
        """
        Ajoute une règle de routage pour un tenant.
        
        Args:
            tenant_id: ID du tenant
            rule: Règle de routage
            
        Returns:
            True si succès, False sinon
        """
        try:
            if tenant_id not in self.routing_rules:
                self.routing_rules[tenant_id] = []
            
            # Vérifier si la règle existe déjà
            existing_rule = next(
                (r for r in self.routing_rules[tenant_id] if r.id == rule.id),
                None
            )
            
            if existing_rule:
                # Mettre à jour la règle existante
                self.routing_rules[tenant_id] = [
                    rule if r.id == rule.id else r
                    for r in self.routing_rules[tenant_id]
                ]
            else:
                # Ajouter la nouvelle règle
                self.routing_rules[tenant_id].append(rule)
            
            # Trier par priorité
            self.routing_rules[tenant_id].sort(key=lambda r: r.priority)
            
            # Persister en Redis
            if self.redis_client:
                await self._persist_routing_rules(tenant_id)
            
            logger.info(f"Règle de routage {rule.id} ajoutée pour tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur ajout règle routage: {e}")
            return False
    
    async def route_message(self,
                          tenant_id: str,
                          severity: SlackSeverity,
                          channel_type: SlackChannelType = SlackChannelType.ALERTS,
                          context: Optional[Dict[str, Any]] = None,
                          strategy: Optional[RoutingStrategy] = None) -> List[str]:
        """
        Route un message vers les canaux appropriés.
        
        Args:
            tenant_id: ID du tenant
            severity: Sévérité du message
            channel_type: Type de canal
            context: Contexte additionnel pour le routage
            strategy: Stratégie de routage spécifique
            
        Returns:
            Liste des canaux sélectionnés
        """
        self.metrics['routing_requests'] += 1
        
        try:
            context = context or {}
            context.update({
                'tenant_id': tenant_id,
                'severity': severity.value,
                'channel_type': channel_type.value,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # 1. Vérifier les règles personnalisées
            custom_channels = await self._apply_custom_rules(tenant_id, context)
            if custom_channels:
                self.metrics['rules_matched'] += 1
                return await self._validate_and_filter_channels(custom_channels, tenant_id)
            
            # 2. Appliquer la stratégie de routage
            strategy = strategy or self.default_strategy
            channels = await self._apply_routing_strategy(
                tenant_id, severity, channel_type, context, strategy
            )
            
            # 3. Valider et filtrer les canaux
            final_channels = await self._validate_and_filter_channels(channels, tenant_id)
            
            # 4. Escalade automatique pour alertes critiques
            if severity == SlackSeverity.CRITICAL and not final_channels:
                escalation_channels = await self._escalate_critical_alert(tenant_id, context)
                final_channels.extend(escalation_channels)
                self.metrics['escalations'] += 1
            
            # 5. Fallback si aucun canal disponible
            if not final_channels:
                final_channels = self.fallback_channels.copy()
                self.metrics['fallback_used'] += 1
                logger.warning(f"Utilisation des canaux de fallback pour {tenant_id}")
            
            # 6. Mettre à jour les métriques d'utilisation
            await self._update_channel_usage(final_channels, tenant_id)
            
            self.metrics['successful_routes'] += 1
            logger.debug(f"Message routé vers {len(final_channels)} canaux pour {tenant_id}")
            
            return final_channels
            
        except Exception as e:
            self.metrics['failed_routes'] += 1
            logger.error(f"Erreur routage message: {e}")
            return self.fallback_channels.copy()
    
    async def _apply_custom_rules(self, 
                                tenant_id: str, 
                                context: Dict[str, Any]) -> List[str]:
        """Applique les règles de routage personnalisées."""
        try:
            if tenant_id not in self.routing_rules:
                return []
            
            # Évaluer les règles par ordre de priorité
            for rule in self.routing_rules[tenant_id]:
                if rule.enabled and rule.matches(context):
                    logger.debug(f"Règle {rule.id} appliquée pour {tenant_id}")
                    return rule.target_channels.copy()
            
            return []
            
        except Exception as e:
            logger.error(f"Erreur application règles personnalisées: {e}")
            return []
    
    async def _apply_routing_strategy(self,
                                    tenant_id: str,
                                    severity: SlackSeverity,
                                    channel_type: SlackChannelType,
                                    context: Dict[str, Any],
                                    strategy: RoutingStrategy) -> List[str]:
        """Applique une stratégie de routage spécifique."""
        try:
            if strategy == RoutingStrategy.SEVERITY_BASED:
                return await self._route_by_severity(tenant_id, severity)
            
            elif strategy == RoutingStrategy.ROUND_ROBIN:
                return await self._route_round_robin(tenant_id, severity, channel_type)
            
            elif strategy == RoutingStrategy.LOAD_BALANCED:
                return await self._route_load_balanced(tenant_id, severity, channel_type)
            
            elif strategy == RoutingStrategy.TENANT_SPECIFIC:
                return await self._route_tenant_specific(tenant_id, severity, channel_type)
            
            elif strategy == RoutingStrategy.TIME_BASED:
                return await self._route_time_based(tenant_id, severity, context)
            
            elif strategy == RoutingStrategy.CUSTOM:
                return await self._route_custom(tenant_id, context)
            
            else:
                # Par défaut: routage par sévérité
                return await self._route_by_severity(tenant_id, severity)
                
        except Exception as e:
            logger.error(f"Erreur application stratégie {strategy.value}: {e}")
            return []
    
    async def _route_by_severity(self, tenant_id: str, severity: SlackSeverity) -> List[str]:
        """Route basé sur la sévérité."""
        # Canaux spécifiques au tenant
        tenant_channels = self.tenant_channels.get(tenant_id, {})
        tenant_severity_channels = [
            channel.name for channel in tenant_channels.values()
            if f"severity_{severity.value}" in channel.tags
        ]
        
        if tenant_severity_channels:
            return tenant_severity_channels
        
        # Canaux globaux par sévérité
        return self.global_channels.get(severity, []).copy()
    
    async def _route_round_robin(self, 
                               tenant_id: str, 
                               severity: SlackSeverity, 
                               channel_type: SlackChannelType) -> List[str]:
        """Route en round-robin."""
        # Obtenir les canaux candidats
        candidate_channels = await self._get_candidate_channels(tenant_id, severity, channel_type)
        
        if not candidate_channels:
            return []
        
        # State key pour le round-robin
        state_key = f"{tenant_id}_{severity.value}_{channel_type.value}"
        
        # Obtenir l'index actuel
        current_index = self._round_robin_state.get(state_key, 0)
        
        # Sélectionner le canal
        selected_channel = candidate_channels[current_index % len(candidate_channels)]
        
        # Mettre à jour l'index
        self._round_robin_state[state_key] = (current_index + 1) % len(candidate_channels)
        
        return [selected_channel]
    
    async def _route_load_balanced(self, 
                                 tenant_id: str, 
                                 severity: SlackSeverity, 
                                 channel_type: SlackChannelType) -> List[str]:
        """Route basé sur la charge des canaux."""
        candidate_channels = await self._get_candidate_channels(tenant_id, severity, channel_type)
        
        if not candidate_channels:
            return []
        
        # Obtenir les informations de charge
        channel_loads = {}
        tenant_channels = self.tenant_channels.get(tenant_id, {})
        
        for channel_name in candidate_channels:
            if channel_name in tenant_channels:
                channel_info = tenant_channels[channel_name]
                load_percent = (channel_info.current_usage / channel_info.rate_limit) * 100
                channel_loads[channel_name] = load_percent
            else:
                # Canal global, charge inconnue
                channel_loads[channel_name] = 50.0  # Charge moyenne supposée
        
        # Sélectionner le canal le moins chargé
        selected_channel = min(channel_loads.items(), key=lambda x: x[1])[0]
        
        return [selected_channel]
    
    async def _route_tenant_specific(self, 
                                   tenant_id: str, 
                                   severity: SlackSeverity, 
                                   channel_type: SlackChannelType) -> List[str]:
        """Route vers les canaux spécifiques au tenant uniquement."""
        tenant_channels = self.tenant_channels.get(tenant_id, {})
        
        # Filtrer par sévérité et type
        matching_channels = []
        for channel_name, channel_info in tenant_channels.items():
            if (f"severity_{severity.value}" in channel_info.tags or
                f"type_{channel_type.value}" in channel_info.tags):
                matching_channels.append(channel_name)
        
        return matching_channels
    
    async def _route_time_based(self, 
                              tenant_id: str, 
                              severity: SlackSeverity, 
                              context: Dict[str, Any]) -> List[str]:
        """Route basé sur l'heure (heures ouvrables vs hors heures)."""
        now = datetime.utcnow()
        is_business_hours = (9 <= now.hour < 17) and (now.weekday() < 5)
        
        time_suffix = "business" if is_business_hours else "after_hours"
        
        # Chercher des canaux spécifiques à l'heure
        tenant_channels = self.tenant_channels.get(tenant_id, {})
        time_channels = [
            channel.name for channel in tenant_channels.values()
            if time_suffix in channel.tags
        ]
        
        if time_channels:
            return time_channels
        
        # Fallback vers routage par sévérité
        return await self._route_by_severity(tenant_id, severity)
    
    async def _route_custom(self, tenant_id: str, context: Dict[str, Any]) -> List[str]:
        """Route basé sur une logique personnalisée."""
        # Cette méthode peut être étendue pour des logiques métier spécifiques
        
        # Exemple: Routage basé sur l'application source
        app_name = context.get('app_name', '')
        if app_name:
            app_channels = [f"#{app_name}-alerts", f"#{app_name}-monitoring"]
            return app_channels
        
        return []
    
    async def _get_candidate_channels(self, 
                                    tenant_id: str, 
                                    severity: SlackSeverity, 
                                    channel_type: SlackChannelType) -> List[str]:
        """Obtient la liste des canaux candidats."""
        candidates = set()
        
        # Canaux spécifiques au tenant
        tenant_channels = self.tenant_channels.get(tenant_id, {})
        for channel_name, channel_info in tenant_channels.items():
            if (f"severity_{severity.value}" in channel_info.tags or
                f"type_{channel_type.value}" in channel_info.tags):
                candidates.add(channel_name)
        
        # Canaux globaux
        global_severity_channels = self.global_channels.get(severity, [])
        global_type_channels = self.type_channels.get(channel_type, [])
        
        candidates.update(global_severity_channels)
        candidates.update(global_type_channels)
        
        return list(candidates)
    
    async def _validate_and_filter_channels(self, 
                                          channels: List[str], 
                                          tenant_id: str) -> List[str]:
        """Valide et filtre les canaux selon leur disponibilité."""
        valid_channels = []
        tenant_channels = self.tenant_channels.get(tenant_id, {})
        
        for channel_name in channels:
            # Vérifier si c'est un canal tenant avec info de statut
            if channel_name in tenant_channels:
                channel_info = tenant_channels[channel_name]
                if channel_info.is_available():
                    valid_channels.append(channel_name)
                elif channel_info.status == ChannelStatus.RATE_LIMITED:
                    self.metrics['rate_limited'] += 1
                    logger.warning(f"Canal {channel_name} rate-limité pour {tenant_id}")
            else:
                # Canal global, on suppose qu'il est disponible
                valid_channels.append(channel_name)
        
        return valid_channels
    
    async def _escalate_critical_alert(self, 
                                     tenant_id: str, 
                                     context: Dict[str, Any]) -> List[str]:
        """Escalade automatique pour alertes critiques."""
        escalation_channels = []
        
        # Canaux d'escalade spécifiques au tenant
        tenant_channels = self.tenant_channels.get(tenant_id, {})
        for channel_name, channel_info in tenant_channels.items():
            if "escalation" in channel_info.tags:
                escalation_channels.append(channel_name)
        
        # Canaux d'escalade globaux
        if not escalation_channels:
            escalation_channels = self.global_channels.get(SlackSeverity.CRITICAL, []).copy()
        
        logger.info(f"Escalade critique activée pour {tenant_id}: {escalation_channels}")
        
        return escalation_channels
    
    async def _update_channel_usage(self, channels: List[str], tenant_id: str):
        """Met à jour les métriques d'utilisation des canaux."""
        tenant_channels = self.tenant_channels.get(tenant_id, {})
        
        for channel_name in channels:
            if channel_name in tenant_channels:
                channel_info = tenant_channels[channel_name]
                channel_info.current_usage += 1
                channel_info.last_message_at = datetime.utcnow()
    
    async def _persist_tenant_channels(self, tenant_id: str, channels: Dict[str, ChannelInfo]):
        """Persiste les canaux d'un tenant en Redis."""
        try:
            if not self.redis_client:
                return
            
            key = f"slack_channels:{tenant_id}"
            channels_data = {}
            
            for channel_name, channel_info in channels.items():
                channels_data[channel_name] = json.dumps({
                    'id': channel_info.id,
                    'name': channel_info.name,
                    'workspace_id': channel_info.workspace_id,
                    'is_private': channel_info.is_private,
                    'members_count': channel_info.members_count,
                    'status': channel_info.status.name,
                    'rate_limit': channel_info.rate_limit,
                    'current_usage': channel_info.current_usage,
                    'reset_time': channel_info.reset_time.isoformat(),
                    'tags': list(channel_info.tags),
                    'priority': channel_info.priority
                })
            
            await self.redis_client.hset(key, mapping=channels_data)
            await self.redis_client.expire(key, 86400)  # 24h
            
        except Exception as e:
            logger.error(f"Erreur persistance canaux tenant {tenant_id}: {e}")
    
    async def _persist_routing_rules(self, tenant_id: str):
        """Persiste les règles de routage en Redis."""
        try:
            if not self.redis_client:
                return
            
            key = f"slack_routing_rules:{tenant_id}"
            rules_data = {}
            
            for i, rule in enumerate(self.routing_rules.get(tenant_id, [])):
                rules_data[f"rule_{i}"] = json.dumps({
                    'id': rule.id,
                    'name': rule.name,
                    'description': rule.description,
                    'conditions': rule.conditions,
                    'target_channels': rule.target_channels,
                    'priority': rule.priority,
                    'enabled': rule.enabled,
                    'created_at': rule.created_at.isoformat()
                })
            
            await self.redis_client.hset(key, mapping=rules_data)
            await self.redis_client.expire(key, 86400)  # 24h
            
        except Exception as e:
            logger.error(f"Erreur persistance règles routage {tenant_id}: {e}")
    
    async def get_channel_status(self, tenant_id: str, channel_name: str) -> Optional[ChannelInfo]:
        """Récupère le statut d'un canal."""
        try:
            tenant_channels = self.tenant_channels.get(tenant_id, {})
            return tenant_channels.get(channel_name)
        except Exception as e:
            logger.error(f"Erreur récupération statut canal {channel_name}: {e}")
            return None
    
    async def update_channel_status(self, 
                                  tenant_id: str, 
                                  channel_name: str, 
                                  status: ChannelStatus) -> bool:
        """Met à jour le statut d'un canal."""
        try:
            tenant_channels = self.tenant_channels.get(tenant_id, {})
            if channel_name in tenant_channels:
                tenant_channels[channel_name].status = status
                
                # Persister la mise à jour
                if self.redis_client:
                    await self._persist_tenant_channels(tenant_id, tenant_channels)
                
                logger.info(f"Statut canal {channel_name} mis à jour: {status.name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur mise à jour statut canal: {e}")
            return False
    
    async def get_routing_suggestions(self, 
                                    tenant_id: str, 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des suggestions de routage basées sur l'historique."""
        try:
            suggestions = {
                'recommended_channels': [],
                'alternative_channels': [],
                'new_rule_suggestions': [],
                'optimization_tips': []
            }
            
            # Analyser l'historique de routage
            if self.redis_client:
                # Récupérer les métriques de routage
                routing_history = await self._get_routing_history(tenant_id)
                
                # Analyser les patterns
                patterns = self._analyze_routing_patterns(routing_history, context)
                suggestions.update(patterns)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Erreur génération suggestions routage: {e}")
            return {}
    
    async def _get_routing_history(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Récupère l'historique de routage depuis Redis."""
        try:
            if not self.redis_client:
                return []
            
            key = f"slack_routing_history:{tenant_id}"
            history_data = await self.redis_client.lrange(key, 0, -1)
            
            history = []
            for item in history_data:
                history.append(json.loads(item.decode()))
            
            return history
            
        except Exception as e:
            logger.error(f"Erreur récupération historique routage: {e}")
            return []
    
    def _analyze_routing_patterns(self, 
                                history: List[Dict[str, Any]], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse les patterns de routage pour générer des suggestions."""
        patterns = {
            'recommended_channels': [],
            'alternative_channels': [],
            'new_rule_suggestions': [],
            'optimization_tips': []
        }
        
        if not history:
            return patterns
        
        # Analyser les canaux les plus utilisés
        channel_usage = {}
        for entry in history:
            for channel in entry.get('channels', []):
                channel_usage[channel] = channel_usage.get(channel, 0) + 1
        
        # Recommandations basées sur l'usage
        if channel_usage:
            sorted_channels = sorted(channel_usage.items(), key=lambda x: x[1], reverse=True)
            patterns['recommended_channels'] = [ch[0] for ch in sorted_channels[:3]]
            patterns['alternative_channels'] = [ch[0] for ch in sorted_channels[3:6]]
        
        # Suggestions d'optimisation
        if len(set(ch for entry in history for ch in entry.get('channels', []))) > 10:
            patterns['optimization_tips'].append(
                "Considérez créer des règles de routage pour réduire le nombre de canaux utilisés"
            )
        
        return patterns
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du routeur."""
        success_rate = 0
        if self.metrics['routing_requests'] > 0:
            success_rate = self.metrics['successful_routes'] / self.metrics['routing_requests']
        
        return {
            **self.metrics,
            'success_rate': success_rate,
            'registered_tenants': len(self.tenant_channels),
            'total_channels': sum(len(channels) for channels in self.tenant_channels.values()),
            'total_routing_rules': sum(len(rules) for rules in self.routing_rules.values()),
            'round_robin_states': len(self._round_robin_state),
            'default_strategy': self.default_strategy.value
        }
    
    async def export_configuration(self, tenant_id: str) -> Dict[str, Any]:
        """Exporte la configuration complète d'un tenant."""
        try:
            config = {
                'tenant_id': tenant_id,
                'channels': {},
                'routing_rules': [],
                'exported_at': datetime.utcnow().isoformat()
            }
            
            # Exporter les canaux
            tenant_channels = self.tenant_channels.get(tenant_id, {})
            for channel_name, channel_info in tenant_channels.items():
                config['channels'][channel_name] = {
                    'id': channel_info.id,
                    'name': channel_info.name,
                    'workspace_id': channel_info.workspace_id,
                    'is_private': channel_info.is_private,
                    'rate_limit': channel_info.rate_limit,
                    'tags': list(channel_info.tags),
                    'priority': channel_info.priority
                }
            
            # Exporter les règles
            tenant_rules = self.routing_rules.get(tenant_id, [])
            for rule in tenant_rules:
                config['routing_rules'].append({
                    'id': rule.id,
                    'name': rule.name,
                    'description': rule.description,
                    'conditions': rule.conditions,
                    'target_channels': rule.target_channels,
                    'priority': rule.priority,
                    'enabled': rule.enabled
                })
            
            return config
            
        except Exception as e:
            logger.error(f"Erreur export configuration {tenant_id}: {e}")
            return {}
    
    async def import_configuration(self, config: Dict[str, Any]) -> bool:
        """Importe une configuration pour un tenant."""
        try:
            tenant_id = config['tenant_id']
            
            # Importer les canaux
            channels = {}
            for channel_name, channel_data in config.get('channels', {}).items():
                channel_info = ChannelInfo(
                    id=channel_data['id'],
                    name=channel_data['name'],
                    workspace_id=channel_data['workspace_id'],
                    is_private=channel_data.get('is_private', False),
                    rate_limit=channel_data.get('rate_limit', 100),
                    tags=set(channel_data.get('tags', [])),
                    priority=channel_data.get('priority', 1)
                )
                channels[channel_name] = channel_info
            
            await self.register_tenant_channels(tenant_id, channels)
            
            # Importer les règles
            for rule_data in config.get('routing_rules', []):
                rule = RoutingRule(
                    id=rule_data['id'],
                    name=rule_data['name'],
                    description=rule_data['description'],
                    conditions=rule_data['conditions'],
                    target_channels=rule_data['target_channels'],
                    priority=rule_data.get('priority', 1),
                    enabled=rule_data.get('enabled', True)
                )
                await self.add_routing_rule(tenant_id, rule)
            
            logger.info(f"Configuration importée pour tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur import configuration: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"SlackChannelRouter(tenants={len(self.tenant_channels)}, strategy={self.default_strategy.value})"
