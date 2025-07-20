"""
Slack Rate Limiter - Gestionnaire intelligent de limitation de débit
Protection anti-spam avec agrégation et analyse prédictive
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

import aioredis


class RateLimitStrategy(str, Enum):
    """Stratégies de rate limiting"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"


class AggregationStrategy(str, Enum):
    """Stratégies d'agrégation"""
    COUNT_ONLY = "count_only"
    SMART_MERGE = "smart_merge"
    SEVERITY_BASED = "severity_based"
    PATTERN_BASED = "pattern_based"


@dataclass
class RateLimitConfig:
    """Configuration de rate limiting"""
    tenant_id: str
    max_alerts_per_minute: int = 10
    max_alerts_per_hour: int = 100
    burst_threshold: int = 5
    burst_window: int = 60  # secondes
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    aggregation_strategy: AggregationStrategy = AggregationStrategy.SMART_MERGE
    whitelist_patterns: List[str] = field(default_factory=list)
    blacklist_patterns: List[str] = field(default_factory=list)
    custom_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitResult:
    """Résultat du rate limiting"""
    allowed: bool
    remaining_quota: int
    reset_time: datetime
    reason: Optional[str] = None
    suggested_delay: int = 0
    aggregated_with: Optional[str] = None


@dataclass
class AlertFingerprint:
    """Empreinte d'alerte pour déduplication"""
    tenant_id: str
    service_name: str
    component: str
    alert_type: str
    severity: str
    fingerprint_hash: str
    first_seen: datetime
    last_seen: datetime
    count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class SlackRateLimiter:
    """
    Gestionnaire intelligent de rate limiting avec:
    - Limitation multi-niveau (minute/heure/jour)
    - Détection de burst et protection
    - Agrégation intelligente des alertes similaires
    - Analyse prédictive des patterns
    - Whitelist/blacklist configurables
    - Adaptation dynamique selon la charge
    """

    def __init__(self):
        self.redis_pool = None
        self.logger = logging.getLogger(__name__)
        
        # Configuration par défaut
        self.default_config = RateLimitConfig(
            tenant_id="default",
            max_alerts_per_minute=10,
            max_alerts_per_hour=100,
            burst_threshold=5,
            burst_window=60
        )
        
        # Cache des configurations
        self.tenant_configs = {}
        
        # Métriques
        self.metrics = {
            "rate_limited_count": 0,
            "aggregated_count": 0,
            "burst_detected": 0,
            "adaptive_adjustments": 0
        }
        
        # Patterns d'agrégation
        self.aggregation_patterns = {
            "similar_errors": r"(?P<error_type>\w+)Error.*(?P<service>\w+)",
            "resource_limits": r"(?P<resource>CPU|Memory|Disk).*(?P<threshold>\d+)%",
            "network_issues": r"(?P<network_type>Connection|Timeout|DNS).*(?P<target>\w+)"
        }

    async def initialize(self, redis_pool: aioredis.Redis):
        """Initialise le rate limiter"""
        self.redis_pool = redis_pool
        
        try:
            # Chargement des configurations des tenants
            await self._load_tenant_configs()
            
            # Démarrage du worker de nettoyage
            asyncio.create_task(self._cleanup_worker())
            
            # Démarrage du worker d'adaptation
            asyncio.create_task(self._adaptive_worker())
            
            self.logger.info("SlackRateLimiter initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise

    async def check_rate_limit(
        self,
        tenant_id: str,
        alert_type: str,
        fingerprint: str,
        service_name: str = None,
        severity: str = "info",
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Vérifie si l'alerte doit être rate limitée
        
        Args:
            tenant_id: ID du tenant
            alert_type: Type d'alerte
            fingerprint: Empreinte de l'alerte
            service_name: Nom du service
            severity: Sévérité de l'alerte
            metadata: Métadonnées additionnelles
            
        Returns:
            True si l'alerte doit être bloquée
        """
        try:
            # Récupération de la configuration
            config = await self._get_tenant_config(tenant_id)
            
            # Vérification des patterns de whitelist
            if await self._is_whitelisted(alert_type, config):
                return False
            
            # Vérification des patterns de blacklist
            if await self._is_blacklisted(alert_type, config):
                self.logger.info(f"Alerte blacklistée: {alert_type}")
                return True
            
            # Bypass pour les alertes critiques
            if severity in ["critical", "emergency"]:
                # Limitation réduite pour les alertes critiques
                return await self._check_critical_rate_limit(tenant_id, config)
            
            # Vérification de déduplication/agrégation
            aggregation_result = await self._check_aggregation(
                tenant_id, alert_type, fingerprint, service_name, severity, metadata
            )
            
            if aggregation_result["should_aggregate"]:
                self.metrics["aggregated_count"] += 1
                return True
            
            # Vérification du rate limiting standard
            rate_limit_result = await self._check_standard_rate_limit(tenant_id, config)
            
            if not rate_limit_result.allowed:
                self.metrics["rate_limited_count"] += 1
                self.logger.info(f"Rate limit atteint pour {tenant_id}: {rate_limit_result.reason}")
                return True
            
            # Détection de burst
            if await self._detect_burst(tenant_id, config):
                self.metrics["burst_detected"] += 1
                return True
            
            # Mise à jour des compteurs
            await self._update_counters(tenant_id, alert_type, severity)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification du rate limit: {e}")
            # En cas d'erreur, on laisse passer pour éviter les blocages
            return False

    async def increment_duplicate_count(self, tenant_id: str, fingerprint: str):
        """Incrémente le compteur de duplicatas"""
        try:
            fingerprint_key = f"slack:fingerprint:{tenant_id}:{fingerprint}"
            
            # Mise à jour de l'empreinte
            pipe = self.redis_pool.pipeline()
            pipe.hincrby(fingerprint_key, "count", 1)
            pipe.hset(fingerprint_key, "last_seen", datetime.utcnow().isoformat())
            pipe.expire(fingerprint_key, 3600)  # Expire après 1 heure
            await pipe.execute()
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'incrémentation des duplicatas: {e}")

    async def get_rate_limit_status(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère le statut du rate limiting pour un tenant"""
        try:
            config = await self._get_tenant_config(tenant_id)
            
            # Compteurs actuels
            minute_count = await self._get_current_count(tenant_id, "minute")
            hour_count = await self._get_current_count(tenant_id, "hour")
            
            # Calcul des quotas restants
            remaining_minute = max(0, config.max_alerts_per_minute - minute_count)
            remaining_hour = max(0, config.max_alerts_per_hour - hour_count)
            
            # Statistiques d'agrégation
            aggregation_stats = await self._get_aggregation_stats(tenant_id)
            
            return {
                "tenant_id": tenant_id,
                "quotas": {
                    "minute": {
                        "limit": config.max_alerts_per_minute,
                        "used": minute_count,
                        "remaining": remaining_minute
                    },
                    "hour": {
                        "limit": config.max_alerts_per_hour,
                        "used": hour_count,
                        "remaining": remaining_hour
                    }
                },
                "aggregation": aggregation_stats,
                "burst_protection": {
                    "threshold": config.burst_threshold,
                    "window": config.burst_window,
                    "active": await self._is_burst_protection_active(tenant_id)
                },
                "metrics": self.metrics.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du statut: {e}")
            return {"error": str(e)}

    async def configure_tenant_rate_limit(
        self,
        tenant_id: str,
        config: RateLimitConfig
    ) -> bool:
        """Configure le rate limiting pour un tenant"""
        try:
            # Validation de la configuration
            await self._validate_config(config)
            
            # Stockage en Redis
            config_key = f"slack:rate_limit_config:{tenant_id}"
            config_data = {
                "config": json.dumps(config.__dict__, default=str),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_pool.hset(config_key, mapping=config_data)
            
            # Mise à jour du cache
            self.tenant_configs[tenant_id] = config
            
            self.logger.info(f"Configuration rate limit mise à jour pour {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration: {e}")
            return False

    async def analyze_alert_patterns(self, tenant_id: str) -> Dict[str, Any]:
        """Analyse les patterns d'alertes pour optimisation"""
        try:
            # Récupération des données historiques
            pattern_data = await self._get_pattern_data(tenant_id)
            
            # Analyse des tendances
            trends = await self._analyze_trends(pattern_data)
            
            # Détection d'anomalies
            anomalies = await self._detect_anomalies(pattern_data)
            
            # Recommandations d'optimisation
            recommendations = await self._generate_recommendations(trends, anomalies)
            
            return {
                "tenant_id": tenant_id,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "patterns": pattern_data,
                "trends": trends,
                "anomalies": anomalies,
                "recommendations": recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des patterns: {e}")
            return {"error": str(e)}

    async def _get_tenant_config(self, tenant_id: str) -> RateLimitConfig:
        """Récupère la configuration d'un tenant"""
        if tenant_id not in self.tenant_configs:
            # Tentative de chargement depuis Redis
            config_key = f"slack:rate_limit_config:{tenant_id}"
            config_data = await self.redis_pool.hget(config_key, "config")
            
            if config_data:
                try:
                    config_dict = json.loads(config_data)
                    config = RateLimitConfig(**config_dict)
                    self.tenant_configs[tenant_id] = config
                except Exception as e:
                    self.logger.warning(f"Configuration invalide pour {tenant_id}: {e}")
                    self.tenant_configs[tenant_id] = self.default_config
            else:
                # Utilisation de la configuration par défaut
                self.tenant_configs[tenant_id] = self.default_config
        
        return self.tenant_configs[tenant_id]

    async def _is_whitelisted(self, alert_type: str, config: RateLimitConfig) -> bool:
        """Vérifie si l'alerte est dans la whitelist"""
        for pattern in config.whitelist_patterns:
            if pattern in alert_type:
                return True
        return False

    async def _is_blacklisted(self, alert_type: str, config: RateLimitConfig) -> bool:
        """Vérifie si l'alerte est dans la blacklist"""
        for pattern in config.blacklist_patterns:
            if pattern in alert_type:
                return True
        return False

    async def _check_critical_rate_limit(self, tenant_id: str, config: RateLimitConfig) -> bool:
        """Vérifie le rate limiting pour les alertes critiques (limites réduites)"""
        # Limite spéciale pour les alertes critiques (2x la limite normale)
        critical_limit_minute = config.max_alerts_per_minute * 2
        critical_limit_hour = config.max_alerts_per_hour * 2
        
        minute_count = await self._get_current_count(tenant_id, "minute", "critical")
        hour_count = await self._get_current_count(tenant_id, "hour", "critical")
        
        return minute_count >= critical_limit_minute or hour_count >= critical_limit_hour

    async def _check_aggregation(
        self,
        tenant_id: str,
        alert_type: str,
        fingerprint: str,
        service_name: str,
        severity: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Vérifie si l'alerte doit être agrégée"""
        # Génération de l'empreinte d'agrégation
        agg_fingerprint = await self._generate_aggregation_fingerprint(
            tenant_id, alert_type, service_name, severity
        )
        
        fingerprint_key = f"slack:fingerprint:{tenant_id}:{agg_fingerprint}"
        fingerprint_data = await self.redis_pool.hgetall(fingerprint_key)
        
        if fingerprint_data:
            # Alerte similaire récente trouvée
            count = int(fingerprint_data.get("count", 0))
            first_seen = datetime.fromisoformat(fingerprint_data["first_seen"])
            
            # Vérification de l'âge
            age = datetime.utcnow() - first_seen
            
            # Agrégation si < 5 minutes et > 2 occurrences
            if age < timedelta(minutes=5) and count >= 2:
                return {
                    "should_aggregate": True,
                    "existing_fingerprint": agg_fingerprint,
                    "count": count + 1,
                    "first_seen": first_seen
                }
        else:
            # Première occurrence, création de l'empreinte
            fingerprint_data = {
                "tenant_id": tenant_id,
                "alert_type": alert_type,
                "service_name": service_name or "unknown",
                "severity": severity,
                "fingerprint_hash": agg_fingerprint,
                "first_seen": datetime.utcnow().isoformat(),
                "last_seen": datetime.utcnow().isoformat(),
                "count": 1,
                "metadata": json.dumps(metadata or {})
            }
            
            await self.redis_pool.hset(fingerprint_key, mapping=fingerprint_data)
            await self.redis_pool.expire(fingerprint_key, 3600)  # 1 heure
        
        return {"should_aggregate": False}

    async def _generate_aggregation_fingerprint(
        self,
        tenant_id: str,
        alert_type: str,
        service_name: str,
        severity: str
    ) -> str:
        """Génère une empreinte pour l'agrégation"""
        # Normalisation des données
        normalized_data = f"{tenant_id}:{alert_type}:{service_name}:{severity}"
        
        # Hash de l'empreinte
        fingerprint_hash = hashlib.sha256(normalized_data.encode()).hexdigest()[:16]
        
        return fingerprint_hash

    async def _check_standard_rate_limit(
        self,
        tenant_id: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Vérifie le rate limiting standard"""
        if config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(tenant_id, config)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(tenant_id, config)
        elif config.strategy == RateLimitStrategy.ADAPTIVE:
            return await self._check_adaptive_limit(tenant_id, config)
        else:  # FIXED_WINDOW
            return await self._check_fixed_window(tenant_id, config)

    async def _check_sliding_window(
        self,
        tenant_id: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Implémentation de la fenêtre glissante"""
        now = time.time()
        window_key = f"slack:sliding_window:{tenant_id}"
        
        # Nettoyage des entrées anciennes
        cutoff_time = now - 60  # Fenêtre d'1 minute
        await self.redis_pool.zremrangebyscore(window_key, 0, cutoff_time)
        
        # Comptage des alertes dans la fenêtre
        count = await self.redis_pool.zcard(window_key)
        
        if count >= config.max_alerts_per_minute:
            oldest_score = await self.redis_pool.zrange(window_key, 0, 0, withscores=True)
            reset_time = datetime.fromtimestamp(oldest_score[0][1] + 60) if oldest_score else datetime.utcnow()
            
            return RateLimitResult(
                allowed=False,
                remaining_quota=0,
                reset_time=reset_time,
                reason="Rate limit par minute atteint",
                suggested_delay=int(reset_time.timestamp() - now)
            )
        
        return RateLimitResult(
            allowed=True,
            remaining_quota=config.max_alerts_per_minute - count,
            reset_time=datetime.fromtimestamp(now + 60)
        )

    async def _check_token_bucket(
        self,
        tenant_id: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Implémentation du token bucket"""
        bucket_key = f"slack:token_bucket:{tenant_id}"
        now = time.time()
        
        # Récupération de l'état du bucket
        bucket_data = await self.redis_pool.hgetall(bucket_key)
        
        if bucket_data:
            tokens = float(bucket_data["tokens"])
            last_refill = float(bucket_data["last_refill"])
        else:
            tokens = float(config.max_alerts_per_minute)
            last_refill = now
        
        # Calcul du rechargement
        time_passed = now - last_refill
        tokens_to_add = time_passed * (config.max_alerts_per_minute / 60)  # tokens par seconde
        tokens = min(config.max_alerts_per_minute, tokens + tokens_to_add)
        
        if tokens >= 1:
            # Consommation d'un token
            tokens -= 1
            
            # Mise à jour du bucket
            await self.redis_pool.hset(bucket_key, mapping={
                "tokens": str(tokens),
                "last_refill": str(now)
            })
            await self.redis_pool.expire(bucket_key, 3600)
            
            return RateLimitResult(
                allowed=True,
                remaining_quota=int(tokens),
                reset_time=datetime.fromtimestamp(now + (1 - tokens) / (config.max_alerts_per_minute / 60))
            )
        else:
            return RateLimitResult(
                allowed=False,
                remaining_quota=0,
                reset_time=datetime.fromtimestamp(now + (1 - tokens) / (config.max_alerts_per_minute / 60)),
                reason="Token bucket vide",
                suggested_delay=int((1 - tokens) / (config.max_alerts_per_minute / 60))
            )

    async def _check_fixed_window(
        self,
        tenant_id: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Implémentation de la fenêtre fixe"""
        now = datetime.utcnow()
        minute_window = now.replace(second=0, microsecond=0)
        
        minute_key = f"slack:fixed_window:{tenant_id}:{minute_window.isoformat()}"
        minute_count = await self.redis_pool.get(minute_key)
        minute_count = int(minute_count) if minute_count else 0
        
        if minute_count >= config.max_alerts_per_minute:
            reset_time = minute_window + timedelta(minutes=1)
            return RateLimitResult(
                allowed=False,
                remaining_quota=0,
                reset_time=reset_time,
                reason="Limite par minute atteinte",
                suggested_delay=(reset_time - now).seconds
            )
        
        return RateLimitResult(
            allowed=True,
            remaining_quota=config.max_alerts_per_minute - minute_count,
            reset_time=minute_window + timedelta(minutes=1)
        )

    async def _check_adaptive_limit(
        self,
        tenant_id: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Implémentation adaptive basée sur la charge système"""
        # Récupération des métriques système
        system_load = await self._get_system_load()
        
        # Ajustement de la limite selon la charge
        if system_load > 0.8:
            # Réduction de 50% si charge élevée
            adjusted_limit = int(config.max_alerts_per_minute * 0.5)
            self.metrics["adaptive_adjustments"] += 1
        elif system_load > 0.6:
            # Réduction de 25% si charge modérée
            adjusted_limit = int(config.max_alerts_per_minute * 0.75)
        else:
            # Limite normale
            adjusted_limit = config.max_alerts_per_minute
        
        # Application de la limite ajustée
        minute_count = await self._get_current_count(tenant_id, "minute")
        
        if minute_count >= adjusted_limit:
            return RateLimitResult(
                allowed=False,
                remaining_quota=0,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
                reason=f"Limite adaptive atteinte (charge: {system_load:.2f})"
            )
        
        return RateLimitResult(
            allowed=True,
            remaining_quota=adjusted_limit - minute_count,
            reset_time=datetime.utcnow() + timedelta(minutes=1)
        )

    async def _detect_burst(self, tenant_id: str, config: RateLimitConfig) -> bool:
        """Détecte les bursts d'alertes"""
        burst_key = f"slack:burst:{tenant_id}"
        now = time.time()
        
        # Ajout de l'alerte actuelle
        await self.redis_pool.zadd(burst_key, {str(now): now})
        
        # Nettoyage des anciennes entrées
        cutoff_time = now - config.burst_window
        await self.redis_pool.zremrangebyscore(burst_key, 0, cutoff_time)
        
        # Comptage des alertes dans la fenêtre de burst
        burst_count = await self.redis_pool.zcard(burst_key)
        
        # Expiration de la clé
        await self.redis_pool.expire(burst_key, config.burst_window)
        
        return burst_count > config.burst_threshold

    async def _get_current_count(
        self,
        tenant_id: str,
        window: str,
        severity: str = None
    ) -> int:
        """Récupère le compteur actuel pour une fenêtre"""
        now = datetime.utcnow()
        
        if window == "minute":
            time_key = now.replace(second=0, microsecond=0).isoformat()
        elif window == "hour":
            time_key = now.replace(minute=0, second=0, microsecond=0).isoformat()
        else:
            time_key = now.date().isoformat()
        
        count_key = f"slack:count:{tenant_id}:{window}:{time_key}"
        if severity:
            count_key += f":{severity}"
        
        count = await self.redis_pool.get(count_key)
        return int(count) if count else 0

    async def _update_counters(
        self,
        tenant_id: str,
        alert_type: str,
        severity: str
    ):
        """Met à jour les compteurs d'alertes"""
        now = datetime.utcnow()
        
        # Compteurs par fenêtre de temps
        minute_key = f"slack:count:{tenant_id}:minute:{now.replace(second=0, microsecond=0).isoformat()}"
        hour_key = f"slack:count:{tenant_id}:hour:{now.replace(minute=0, second=0, microsecond=0).isoformat()}"
        
        pipe = self.redis_pool.pipeline()
        
        # Incrémentation des compteurs
        pipe.incr(minute_key)
        pipe.expire(minute_key, 120)  # 2 minutes de rétention
        
        pipe.incr(hour_key)
        pipe.expire(hour_key, 7200)  # 2 heures de rétention
        
        # Compteurs par sévérité
        if severity in ["critical", "emergency"]:
            critical_minute_key = f"{minute_key}:critical"
            critical_hour_key = f"{hour_key}:critical"
            
            pipe.incr(critical_minute_key)
            pipe.expire(critical_minute_key, 120)
            
            pipe.incr(critical_hour_key)
            pipe.expire(critical_hour_key, 7200)
        
        await pipe.execute()

    async def _get_system_load(self) -> float:
        """Récupère la charge système actuelle"""
        # Simulation - à remplacer par de vraies métriques
        import random
        return random.uniform(0.1, 0.9)

    async def _load_tenant_configs(self):
        """Charge les configurations des tenants depuis Redis"""
        pattern = "slack:rate_limit_config:*"
        keys = await self.redis_pool.keys(pattern)
        
        for key in keys:
            try:
                tenant_id = key.split(":")[-1]
                config_data = await self.redis_pool.hget(key, "config")
                
                if config_data:
                    config_dict = json.loads(config_data)
                    config = RateLimitConfig(**config_dict)
                    self.tenant_configs[tenant_id] = config
                    
            except Exception as e:
                self.logger.warning(f"Erreur lors du chargement de {key}: {e}")

    async def _get_aggregation_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère les statistiques d'agrégation"""
        pattern = f"slack:fingerprint:{tenant_id}:*"
        keys = await self.redis_pool.keys(pattern)
        
        stats = {
            "total_fingerprints": len(keys),
            "aggregated_alerts": 0,
            "top_patterns": []
        }
        
        pattern_counts = {}
        
        for key in keys:
            fingerprint_data = await self.redis_pool.hgetall(key)
            if fingerprint_data:
                count = int(fingerprint_data.get("count", 1))
                if count > 1:
                    stats["aggregated_alerts"] += count - 1
                
                alert_type = fingerprint_data.get("alert_type", "unknown")
                pattern_counts[alert_type] = pattern_counts.get(alert_type, 0) + count
        
        # Top 5 des patterns
        stats["top_patterns"] = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return stats

    async def _is_burst_protection_active(self, tenant_id: str) -> bool:
        """Vérifie si la protection burst est active"""
        burst_key = f"slack:burst:{tenant_id}"
        return await self.redis_pool.exists(burst_key)

    async def _validate_config(self, config: RateLimitConfig):
        """Valide une configuration de rate limiting"""
        if config.max_alerts_per_minute <= 0 or config.max_alerts_per_hour <= 0:
            raise ValueError("Les limites doivent être positives")
        
        if config.max_alerts_per_minute > config.max_alerts_per_hour:
            raise ValueError("La limite par minute ne peut pas dépasser celle par heure")
        
        if config.burst_threshold <= 0 or config.burst_window <= 0:
            raise ValueError("Les paramètres de burst doivent être positifs")

    async def _get_pattern_data(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère les données de patterns pour analyse"""
        # Simulation - à implémenter selon les besoins
        return {
            "hourly_distribution": {},
            "service_patterns": {},
            "severity_patterns": {},
            "aggregation_effectiveness": 0.85
        }

    async def _analyze_trends(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse les tendances des alertes"""
        return {
            "trend_direction": "stable",
            "growth_rate": 0.05,
            "seasonality_detected": False,
            "peak_hours": ["09:00", "14:00", "17:00"]
        }

    async def _detect_anomalies(self, pattern_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Détecte les anomalies dans les patterns"""
        return [
            {
                "type": "volume_spike",
                "severity": "medium",
                "description": "Augmentation inhabituelle du volume d'alertes",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]

    async def _generate_recommendations(
        self,
        trends: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Génère des recommandations d'optimisation"""
        recommendations = []
        
        if trends["growth_rate"] > 0.1:
            recommendations.append({
                "type": "increase_limits",
                "priority": "medium",
                "description": "Considérer l'augmentation des limites de rate limiting",
                "impact": "improved_availability"
            })
        
        return recommendations

    async def _cleanup_worker(self):
        """Worker de nettoyage périodique"""
        while True:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                # Nettoyage des anciennes empreintes
                cutoff_time = datetime.utcnow() - timedelta(hours=2)
                pattern = "slack:fingerprint:*"
                keys = await self.redis_pool.keys(pattern)
                
                for key in keys:
                    fingerprint_data = await self.redis_pool.hget(key, "first_seen")
                    if fingerprint_data:
                        first_seen = datetime.fromisoformat(fingerprint_data)
                        if first_seen < cutoff_time:
                            await self.redis_pool.delete(key)
                
            except Exception as e:
                self.logger.error(f"Erreur dans le cleanup worker: {e}")

    async def _adaptive_worker(self):
        """Worker d'adaptation automatique"""
        while True:
            try:
                await asyncio.sleep(600)  # Toutes les 10 minutes
                
                # Analyse et ajustement automatique pour chaque tenant
                for tenant_id in self.tenant_configs:
                    try:
                        analysis = await self.analyze_alert_patterns(tenant_id)
                        # Logique d'adaptation basée sur l'analyse
                        # À implémenter selon les besoins
                    except Exception as e:
                        self.logger.warning(f"Erreur d'adaptation pour {tenant_id}: {e}")
                
            except Exception as e:
                self.logger.error(f"Erreur dans l'adaptive worker: {e}")
