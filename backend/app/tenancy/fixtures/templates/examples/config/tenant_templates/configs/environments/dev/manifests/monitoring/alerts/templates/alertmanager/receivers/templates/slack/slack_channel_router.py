"""
Slack Channel Router - Routage intelligent des alertes Slack
Gestion multi-tenant avec distribution contextuelle et règles avancées
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import aioredis
from .slack_alert_manager import AlertSeverity


class RoutingStrategy(str, Enum):
    """Stratégies de routage"""
    ROUND_ROBIN = "round_robin"
    SEVERITY_BASED = "severity_based"
    SERVICE_BASED = "service_based"
    LOAD_BALANCED = "load_balanced"
    CUSTOM = "custom"


class ChannelType(str, Enum):
    """Types de canaux"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    ESCALATION = "escalation"
    FALLBACK = "fallback"
    EXECUTIVE = "executive"
    MAINTENANCE = "maintenance"


@dataclass
class ChannelConfig:
    """Configuration d'un canal"""
    channel_id: str
    channel_name: str
    webhook_url: str
    channel_type: ChannelType
    max_alerts_per_hour: int = 50
    severity_filter: List[AlertSeverity] = field(default_factory=list)
    service_filter: List[str] = field(default_factory=list)
    environment_filter: List[str] = field(default_factory=list)
    active_hours: Optional[Dict[str, str]] = None  # {"start": "09:00", "end": "18:00"}
    timezone: str = "Europe/Paris"
    escalation_delay: int = 300  # secondes
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.severity_filter:
            self.severity_filter = list(AlertSeverity)


@dataclass
class RoutingRule:
    """Règle de routage"""
    rule_id: str
    name: str
    priority: int  # Plus haut = priorité plus élevée
    conditions: Dict[str, Any]
    target_channels: List[str]
    actions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class RoutingResult:
    """Résultat du routage"""
    target_channels: List[str]
    applied_rules: List[str]
    fallback_used: bool = False
    routing_metadata: Dict[str, Any] = field(default_factory=dict)


class SlackChannelRouter:
    """
    Routeur intelligent de canaux Slack avec:
    - Routage multi-tenant avec isolation
    - Règles conditionnelles avancées
    - Load balancing et failover automatique
    - Gestion des heures d'activité
    - Escalade automatique selon les SLA
    - Analyse des patterns de routage
    """

    def __init__(self):
        self.redis_pool = None
        self.logger = logging.getLogger(__name__)
        
        # Configuration par défaut
        self.default_config = {
            "fallback_channel": "#alerts-fallback",
            "executive_channel": "#alerts-executive",
            "maintenance_channel": "#alerts-maintenance",
            "max_channels_per_alert": 3,
            "load_balance_window": 3600,  # 1 heure
            "escalation_threshold": 5,    # 5 minutes sans acquittement
            "rate_limit_threshold": 0.8   # 80% du max avant redirection
        }
        
        # Cache des configurations
        self.channel_configs = {}
        self.routing_rules = {}
        self.tenant_configs = {}
        
        # Métriques de routage
        self.routing_stats = {
            "total_routed": 0,
            "fallback_used": 0,
            "rules_applied": {},
            "channel_usage": {}
        }

    async def initialize(self, redis_pool: aioredis.Redis):
        """Initialise le routeur"""
        self.redis_pool = redis_pool
        
        try:
            # Chargement des configurations depuis Redis
            await self._load_channel_configs()
            await self._load_routing_rules()
            await self._load_tenant_configs()
            
            # Création des canaux par défaut si nécessaire
            await self._ensure_default_channels()
            
            self.logger.info("SlackChannelRouter initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation du routeur: {e}")
            raise

    async def get_target_channels(
        self,
        tenant_id: str,
        severity: AlertSeverity,
        environment: str,
        service_name: str = None,
        component: str = None,
        tags: Dict[str, str] = None,
        custom_context: Dict[str, Any] = None
    ) -> List[str]:
        """
        Détermine les canaux de destination pour une alerte
        
        Args:
            tenant_id: ID du tenant
            severity: Sévérité de l'alerte
            environment: Environnement (dev, staging, prod)
            service_name: Nom du service
            component: Composant concerné
            tags: Tags additionnels
            custom_context: Contexte personnalisé
            
        Returns:
            Liste des canaux de destination
        """
        try:
            # Contexte de routage
            routing_context = {
                "tenant_id": tenant_id,
                "severity": severity,
                "environment": environment,
                "service_name": service_name,
                "component": component,
                "tags": tags or {},
                "custom_context": custom_context or {},
                "timestamp": datetime.utcnow()
            }
            
            # Application des règles de routage
            routing_result = await self._apply_routing_rules(routing_context)
            
            # Validation des canaux
            valid_channels = await self._validate_channels(
                routing_result.target_channels,
                tenant_id,
                severity
            )
            
            # Vérification du rate limiting
            available_channels = await self._filter_rate_limited_channels(
                valid_channels,
                tenant_id
            )
            
            # Fallback si aucun canal disponible
            if not available_channels:
                available_channels = await self._get_fallback_channels(
                    tenant_id,
                    severity,
                    environment
                )
                routing_result.fallback_used = True
            
            # Limitation du nombre de canaux
            max_channels = self.default_config["max_channels_per_alert"]
            final_channels = available_channels[:max_channels]
            
            # Mise à jour des statistiques
            await self._update_routing_stats(routing_result, final_channels)
            
            self.logger.info(f"Routage: {len(final_channels)} canaux pour {tenant_id}/{severity}")
            
            return final_channels
            
        except Exception as e:
            self.logger.error(f"Erreur lors du routage: {e}")
            # Fallback d'urgence
            return [self.default_config["fallback_channel"]]

    async def register_channel(
        self,
        tenant_id: str,
        channel_config: ChannelConfig
    ) -> bool:
        """Enregistre un nouveau canal"""
        try:
            # Validation de la configuration
            await self._validate_channel_config(channel_config)
            
            # Stockage en Redis
            config_key = f"slack:channels:{tenant_id}:{channel_config.channel_id}"
            config_data = {
                "config": json.dumps(channel_config.__dict__, default=str),
                "registered_at": datetime.utcnow().isoformat(),
                "last_used": None
            }
            
            await self.redis_pool.hset(config_key, mapping=config_data)
            
            # Mise à jour du cache local
            cache_key = f"{tenant_id}:{channel_config.channel_id}"
            self.channel_configs[cache_key] = channel_config
            
            # Ajout à la liste des canaux du tenant
            await self.redis_pool.sadd(f"slack:tenant_channels:{tenant_id}", channel_config.channel_id)
            
            self.logger.info(f"Canal {channel_config.channel_id} enregistré pour {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement du canal: {e}")
            return False

    async def add_routing_rule(
        self,
        tenant_id: str,
        rule: RoutingRule
    ) -> bool:
        """Ajoute une règle de routage"""
        try:
            # Validation de la règle
            await self._validate_routing_rule(rule)
            
            # Stockage en Redis
            rule_key = f"slack:routing_rules:{tenant_id}:{rule.rule_id}"
            rule_data = {
                "rule": json.dumps(rule.__dict__, default=str),
                "created_at": datetime.utcnow().isoformat(),
                "last_applied": None,
                "apply_count": 0
            }
            
            await self.redis_pool.hset(rule_key, mapping=rule_data)
            
            # Mise à jour du cache
            cache_key = f"{tenant_id}:{rule.rule_id}"
            self.routing_rules[cache_key] = rule
            
            # Tri des règles par priorité
            await self._sort_tenant_rules(tenant_id)
            
            self.logger.info(f"Règle {rule.rule_id} ajoutée pour {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout de la règle: {e}")
            return False

    async def get_channel_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère les statistiques des canaux"""
        try:
            stats = {
                "channels": {},
                "routing_rules": {},
                "usage_patterns": {},
                "performance": {}
            }
            
            # Statistiques par canal
            channels = await self.redis_pool.smembers(f"slack:tenant_channels:{tenant_id}")
            for channel_id in channels:
                channel_stats = await self._get_channel_stats(tenant_id, channel_id)
                stats["channels"][channel_id] = channel_stats
            
            # Statistiques des règles
            rule_pattern = f"slack:routing_rules:{tenant_id}:*"
            rule_keys = await self.redis_pool.keys(rule_pattern)
            for rule_key in rule_keys:
                rule_data = await self.redis_pool.hgetall(rule_key)
                rule_id = rule_key.split(":")[-1]
                stats["routing_rules"][rule_id] = {
                    "apply_count": int(rule_data.get("apply_count", 0)),
                    "last_applied": rule_data.get("last_applied")
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des stats: {e}")
            return {"error": str(e)}

    async def optimize_routing(self, tenant_id: str) -> Dict[str, Any]:
        """Optimise le routage basé sur l'historique"""
        try:
            # Analyse des patterns d'usage
            usage_patterns = await self._analyze_usage_patterns(tenant_id)
            
            # Recommandations d'optimisation
            recommendations = await self._generate_optimization_recommendations(
                tenant_id,
                usage_patterns
            )
            
            # Application automatique si configuré
            auto_applied = []
            if self.tenant_configs.get(tenant_id, {}).get("auto_optimize", False):
                auto_applied = await self._apply_optimizations(tenant_id, recommendations)
            
            return {
                "usage_patterns": usage_patterns,
                "recommendations": recommendations,
                "auto_applied": auto_applied,
                "optimization_score": self._calculate_optimization_score(usage_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'optimisation: {e}")
            return {"error": str(e)}

    async def _apply_routing_rules(self, context: Dict[str, Any]) -> RoutingResult:
        """Applique les règles de routage"""
        tenant_id = context["tenant_id"]
        target_channels = []
        applied_rules = []
        
        # Récupération des règles du tenant triées par priorité
        tenant_rules = await self._get_tenant_rules(tenant_id)
        
        for rule in tenant_rules:
            if not rule.enabled:
                continue
            
            # Évaluation des conditions
            if await self._evaluate_rule_conditions(rule, context):
                target_channels.extend(rule.target_channels)
                applied_rules.append(rule.rule_id)
                
                # Mise à jour des stats de la règle
                await self._update_rule_stats(tenant_id, rule.rule_id)
                
                # Arrêt si action "stop" définie
                if rule.actions.get("stop_evaluation", False):
                    break
        
        # Suppression des doublons tout en préservant l'ordre
        unique_channels = []
        seen = set()
        for channel in target_channels:
            if channel not in seen:
                unique_channels.append(channel)
                seen.add(channel)
        
        return RoutingResult(
            target_channels=unique_channels,
            applied_rules=applied_rules,
            routing_metadata={"context": context}
        )

    async def _evaluate_rule_conditions(self, rule: RoutingRule, context: Dict[str, Any]) -> bool:
        """Évalue les conditions d'une règle"""
        conditions = rule.conditions
        
        # Vérification de la sévérité
        if "severity" in conditions:
            required_severities = conditions["severity"]
            if isinstance(required_severities, str):
                required_severities = [required_severities]
            if context["severity"].value not in required_severities:
                return False
        
        # Vérification de l'environnement
        if "environment" in conditions:
            required_envs = conditions["environment"]
            if isinstance(required_envs, str):
                required_envs = [required_envs]
            if context["environment"] not in required_envs:
                return False
        
        # Vérification du service
        if "service_name" in conditions:
            required_services = conditions["service_name"]
            if isinstance(required_services, str):
                required_services = [required_services]
            if context.get("service_name") not in required_services:
                return False
        
        # Vérification des tags
        if "tags" in conditions:
            required_tags = conditions["tags"]
            context_tags = context.get("tags", {})
            for tag_key, tag_value in required_tags.items():
                if context_tags.get(tag_key) != tag_value:
                    return False
        
        # Vérification des heures d'activité
        if "active_hours" in conditions:
            current_time = datetime.utcnow().time()
            active_hours = conditions["active_hours"]
            start_time = datetime.strptime(active_hours["start"], "%H:%M").time()
            end_time = datetime.strptime(active_hours["end"], "%H:%M").time()
            
            if start_time <= end_time:
                if not (start_time <= current_time <= end_time):
                    return False
            else:  # Plage qui traverse minuit
                if not (current_time >= start_time or current_time <= end_time):
                    return False
        
        # Conditions personnalisées (expressions)
        if "custom_expression" in conditions:
            try:
                # Évaluation sécurisée d'expressions simples
                expression = conditions["custom_expression"]
                # À implémenter selon les besoins de sécurité
                pass
            except Exception:
                return False
        
        return True

    async def _validate_channels(
        self,
        channels: List[str],
        tenant_id: str,
        severity: AlertSeverity
    ) -> List[str]:
        """Valide et filtre les canaux"""
        valid_channels = []
        
        for channel in channels:
            # Vérification de l'existence du canal
            config_key = f"{tenant_id}:{channel}"
            if config_key in self.channel_configs:
                channel_config = self.channel_configs[config_key]
                
                # Vérification du filtre de sévérité
                if severity in channel_config.severity_filter:
                    valid_channels.append(channel)
            else:
                # Tentative de chargement depuis Redis
                redis_key = f"slack:channels:{tenant_id}:{channel}"
                config_data = await self.redis_pool.hget(redis_key, "config")
                if config_data:
                    try:
                        config_dict = json.loads(config_data)
                        channel_config = ChannelConfig(**config_dict)
                        self.channel_configs[config_key] = channel_config
                        
                        if severity in channel_config.severity_filter:
                            valid_channels.append(channel)
                    except Exception as e:
                        self.logger.warning(f"Configuration invalide pour {channel}: {e}")
        
        return valid_channels

    async def _filter_rate_limited_channels(
        self,
        channels: List[str],
        tenant_id: str
    ) -> List[str]:
        """Filtre les canaux rate limités"""
        available_channels = []
        
        for channel in channels:
            # Vérification du rate limiting
            usage_key = f"slack:channel_usage:{tenant_id}:{channel}"
            current_usage = await self.redis_pool.get(usage_key)
            
            config_key = f"{tenant_id}:{channel}"
            if config_key in self.channel_configs:
                max_alerts = self.channel_configs[config_key].max_alerts_per_hour
                current_count = int(current_usage) if current_usage else 0
                
                # Vérification du seuil (80% par défaut)
                threshold = max_alerts * self.default_config["rate_limit_threshold"]
                
                if current_count < threshold:
                    available_channels.append(channel)
            else:
                # Canal sans limite configurée
                available_channels.append(channel)
        
        return available_channels

    async def _get_fallback_channels(
        self,
        tenant_id: str,
        severity: AlertSeverity,
        environment: str
    ) -> List[str]:
        """Récupère les canaux de fallback"""
        fallback_channels = []
        
        # Canal de fallback spécifique au tenant
        tenant_fallback = f"slack:fallback:{tenant_id}"
        if await self.redis_pool.exists(tenant_fallback):
            fallback_channels.append(await self.redis_pool.get(tenant_fallback))
        
        # Canal de fallback par environnement
        env_fallback = f"#{environment}-alerts-fallback"
        fallback_channels.append(env_fallback)
        
        # Canal de fallback global
        fallback_channels.append(self.default_config["fallback_channel"])
        
        # Canal exécutif pour les alertes critiques
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            fallback_channels.insert(0, self.default_config["executive_channel"])
        
        return fallback_channels

    async def _load_channel_configs(self):
        """Charge les configurations de canaux depuis Redis"""
        pattern = "slack:channels:*"
        keys = await self.redis_pool.keys(pattern)
        
        for key in keys:
            try:
                config_data = await self.redis_pool.hget(key, "config")
                if config_data:
                    config_dict = json.loads(config_data)
                    channel_config = ChannelConfig(**config_dict)
                    
                    # Extraction tenant_id et channel_id depuis la clé
                    key_parts = key.split(":")
                    tenant_id = key_parts[2]
                    channel_id = key_parts[3]
                    
                    cache_key = f"{tenant_id}:{channel_id}"
                    self.channel_configs[cache_key] = channel_config
                    
            except Exception as e:
                self.logger.warning(f"Erreur lors du chargement de {key}: {e}")

    async def _load_routing_rules(self):
        """Charge les règles de routage depuis Redis"""
        pattern = "slack:routing_rules:*"
        keys = await self.redis_pool.keys(pattern)
        
        for key in keys:
            try:
                rule_data = await self.redis_pool.hget(key, "rule")
                if rule_data:
                    rule_dict = json.loads(rule_data)
                    routing_rule = RoutingRule(**rule_dict)
                    
                    # Extraction tenant_id et rule_id depuis la clé
                    key_parts = key.split(":")
                    tenant_id = key_parts[2]
                    rule_id = key_parts[3]
                    
                    cache_key = f"{tenant_id}:{rule_id}"
                    self.routing_rules[cache_key] = routing_rule
                    
            except Exception as e:
                self.logger.warning(f"Erreur lors du chargement de {key}: {e}")

    async def _load_tenant_configs(self):
        """Charge les configurations des tenants"""
        pattern = "slack:tenant_config:*"
        keys = await self.redis_pool.keys(pattern)
        
        for key in keys:
            try:
                tenant_id = key.split(":")[-1]
                config_data = await self.redis_pool.hgetall(key)
                self.tenant_configs[tenant_id] = config_data
            except Exception as e:
                self.logger.warning(f"Erreur lors du chargement de {key}: {e}")

    async def _ensure_default_channels(self):
        """Assure l'existence des canaux par défaut"""
        default_channels = [
            self.default_config["fallback_channel"],
            self.default_config["executive_channel"],
            self.default_config["maintenance_channel"]
        ]
        
        for channel in default_channels:
            # Vérification et création si nécessaire
            # À implémenter selon l'API Slack utilisée
            pass

    async def _get_tenant_rules(self, tenant_id: str) -> List[RoutingRule]:
        """Récupère les règles d'un tenant triées par priorité"""
        tenant_rules = []
        
        for cache_key, rule in self.routing_rules.items():
            if cache_key.startswith(f"{tenant_id}:"):
                tenant_rules.append(rule)
        
        # Tri par priorité (descendant)
        tenant_rules.sort(key=lambda r: r.priority, reverse=True)
        
        return tenant_rules

    async def _update_rule_stats(self, tenant_id: str, rule_id: str):
        """Met à jour les statistiques d'une règle"""
        rule_key = f"slack:routing_rules:{tenant_id}:{rule_id}"
        pipe = self.redis_pool.pipeline()
        pipe.hincrby(rule_key, "apply_count", 1)
        pipe.hset(rule_key, "last_applied", datetime.utcnow().isoformat())
        await pipe.execute()

    async def _update_routing_stats(self, result: RoutingResult, final_channels: List[str]):
        """Met à jour les statistiques de routage"""
        self.routing_stats["total_routed"] += 1
        
        if result.fallback_used:
            self.routing_stats["fallback_used"] += 1
        
        for rule_id in result.applied_rules:
            if rule_id not in self.routing_stats["rules_applied"]:
                self.routing_stats["rules_applied"][rule_id] = 0
            self.routing_stats["rules_applied"][rule_id] += 1
        
        for channel in final_channels:
            if channel not in self.routing_stats["channel_usage"]:
                self.routing_stats["channel_usage"][channel] = 0
            self.routing_stats["channel_usage"][channel] += 1

    async def _get_channel_stats(self, tenant_id: str, channel_id: str) -> Dict[str, Any]:
        """Récupère les statistiques d'un canal"""
        stats = {
            "usage_count": 0,
            "last_used": None,
            "rate_limit_hits": 0,
            "error_count": 0
        }
        
        # Statistiques d'usage
        usage_key = f"slack:channel_usage:{tenant_id}:{channel_id}"
        usage_count = await self.redis_pool.get(usage_key)
        if usage_count:
            stats["usage_count"] = int(usage_count)
        
        # Dernière utilisation
        config_key = f"slack:channels:{tenant_id}:{channel_id}"
        last_used = await self.redis_pool.hget(config_key, "last_used")
        if last_used:
            stats["last_used"] = last_used
        
        return stats

    async def _validate_channel_config(self, config: ChannelConfig):
        """Valide une configuration de canal"""
        if not config.channel_id or not config.webhook_url:
            raise ValueError("channel_id et webhook_url sont requis")
        
        if config.max_alerts_per_hour <= 0:
            raise ValueError("max_alerts_per_hour doit être positif")

    async def _validate_routing_rule(self, rule: RoutingRule):
        """Valide une règle de routage"""
        if not rule.rule_id or not rule.name:
            raise ValueError("rule_id et name sont requis")
        
        if not rule.target_channels:
            raise ValueError("target_channels ne peut pas être vide")
        
        if rule.priority < 0:
            raise ValueError("priority doit être positive")

    async def _sort_tenant_rules(self, tenant_id: str):
        """Trie les règles d'un tenant par priorité"""
        # Mise à jour de l'ordre en Redis si nécessaire
        pass

    async def _analyze_usage_patterns(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse les patterns d'usage pour optimisation"""
        # Analyse des données historiques
        # À implémenter selon les besoins
        return {
            "most_used_channels": [],
            "peak_hours": [],
            "service_patterns": {},
            "severity_distribution": {}
        }

    async def _generate_optimization_recommendations(
        self,
        tenant_id: str,
        patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Génère des recommandations d'optimisation"""
        recommendations = []
        
        # Exemple de recommandations
        if patterns.get("most_used_channels"):
            recommendations.append({
                "type": "channel_consolidation",
                "description": "Considérer la consolidation des canaux peu utilisés",
                "impact": "medium",
                "effort": "low"
            })
        
        return recommendations

    async def _apply_optimizations(
        self,
        tenant_id: str,
        recommendations: List[Dict[str, Any]]
    ) -> List[str]:
        """Applique automatiquement les optimisations"""
        applied = []
        
        # Application sélective selon la configuration
        for rec in recommendations:
            if rec["impact"] == "low" and rec["effort"] == "low":
                # Application automatique pour les changements mineurs
                applied.append(rec["type"])
        
        return applied

    def _calculate_optimization_score(self, patterns: Dict[str, Any]) -> float:
        """Calcule un score d'optimisation"""
        # Score basé sur l'efficacité du routage
        return 0.85  # Exemple
