#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spotify AI Agent - Tenant Context Provider pour Alerting Multi-Tenant

Fournisseur de contexte tenant pour système d'alerting avec isolation
complète des données et personnalisation par type de tenant.

Fonctionnalités:
- Isolation stricte des données par tenant
- Contexte enrichi avec métadonnées business
- Personnalisation des alertes par tenant
- Gestion des hiérarchies de tenants (labels/artistes)
- Cache intelligent des contextes fréquents
- Validation sécurisée des accès
- Audit trail complet des actions

Architecture:
- Repository Pattern pour accès données
- Strategy Pattern pour types de tenants
- Decorator Pattern pour enrichissement
- Observer Pattern pour audit logging
- Cache multi-niveaux (L1: mémoire, L2: Redis)

Sécurité:
- Validation RBAC par tenant
- Chiffrement des données sensibles
- Rate limiting par tenant
- Logs d'audit sécurisés
- Detection d'anomalies d'accès

Utilisation:
    provider = TenantContextProvider()
    
    context = provider.get_tenant_context(
        tenant_id="universal_music_001",
        alert_type="ai_model_performance",
        user_context={"user_id": "admin_123"}
    )
"""

import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import asyncio
from functools import wraps, lru_cache
import logging

# Imports externes
import structlog
from prometheus_client import Counter, Histogram, Gauge
import redis
import jwt
from cryptography.fernet import Fernet
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, select, and_

# Imports internes
from .config import TENANT_TYPES, get_environment

# Configuration logging
logger = structlog.get_logger(__name__)

# Métriques Prometheus
tenant_context_requests_total = Counter(
    'spotify_ai_tenant_context_requests_total',
    'Nombre total de requêtes de contexte tenant',
    ['tenant_id', 'tenant_type', 'alert_type', 'status']
)

tenant_context_cache_hits = Counter(
    'spotify_ai_tenant_context_cache_hits_total',
    'Hits du cache contexte tenant',
    ['tenant_id', 'cache_level']
)

tenant_context_duration = Histogram(
    'spotify_ai_tenant_context_duration_seconds',
    'Durée de récupération du contexte tenant',
    ['tenant_id', 'source']
)

tenant_access_violations = Counter(
    'spotify_ai_tenant_access_violations_total',
    'Violations d\'accès tenant détectées',
    ['tenant_id', 'violation_type', 'user_id']
)

active_tenant_contexts = Gauge(
    'spotify_ai_active_tenant_contexts',
    'Nombre de contextes tenant actifs en cache'
)

class TenantType(Enum):
    """Types de tenants supportés."""
    ARTIST = "artist"
    LABEL = "label"
    STUDIO = "studio"
    PLATFORM = "platform"
    SYSTEM = "system"

class AccessLevel(Enum):
    """Niveaux d'accès aux données tenant."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"

class ViolationType(Enum):
    """Types de violations d'accès."""
    UNAUTHORIZED_TENANT = "unauthorized_tenant"
    INSUFFICIENT_PERMISSIONS = "insufficient_permissions"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    DATA_EXPOSURE_ATTEMPT = "data_exposure_attempt"

@dataclass
class TenantMetadata:
    """Métadonnées d'un tenant."""
    tenant_id: str
    name: str
    type: TenantType
    status: str = "active"  # active, suspended, archived
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Hiérarchie
    parent_tenant_id: Optional[str] = None
    child_tenant_ids: List[str] = field(default_factory=list)
    
    # Business context
    industry_sector: str = "music"
    region: str = "global"
    timezone: str = "UTC"
    locale: str = "en"
    
    # Configuration alerting
    alert_preferences: Dict[str, Any] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)
    escalation_policies: Dict[str, str] = field(default_factory=dict)
    
    # Limites et quotas
    quota_limits: Dict[str, int] = field(default_factory=dict)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    
    # Sécurité
    access_policies: Dict[str, List[str]] = field(default_factory=dict)
    encryption_required: bool = True
    audit_level: str = "standard"  # minimal, standard, detailed
    
    # Métadonnées business
    business_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        data = asdict(self)
        data['type'] = self.type.value
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class UserContext:
    """Contexte utilisateur pour validation RBAC."""
    user_id: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    tenant_access: Dict[str, AccessLevel] = field(default_factory=dict)
    session_id: str = ""
    ip_address: str = ""
    user_agent: str = ""
    last_activity: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TenantContext:
    """Contexte complet d'un tenant pour alerting."""
    metadata: TenantMetadata
    
    # Contexte alerting enrichi
    alert_config: Dict[str, Any] = field(default_factory=dict)
    current_alerts: List[Dict[str, Any]] = field(default_factory=list)
    alert_history: Dict[str, int] = field(default_factory=dict)
    
    # Métriques en temps réel
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Relations et dépendances
    related_tenants: List[str] = field(default_factory=list)
    service_dependencies: List[str] = field(default_factory=list)
    
    # Cache metadata
    cached_at: datetime = field(default_factory=datetime.utcnow)
    cache_ttl: int = 3600  # 1 heure
    cache_source: str = "unknown"
    
    def is_cache_valid(self) -> bool:
        """Vérifie si le cache est encore valide."""
        return datetime.utcnow() < self.cached_at + timedelta(seconds=self.cache_ttl)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        data = {
            'metadata': self.metadata.to_dict(),
            'alert_config': self.alert_config,
            'current_alerts': self.current_alerts,
            'alert_history': self.alert_history,
            'performance_metrics': self.performance_metrics,
            'resource_usage': self.resource_usage,
            'related_tenants': self.related_tenants,
            'service_dependencies': self.service_dependencies,
            'cached_at': self.cached_at.isoformat(),
            'cache_ttl': self.cache_ttl,
            'cache_source': self.cache_source
        }
        return data

class TenantSecurityManager:
    """Gestionnaire de sécurité pour accès tenant."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        # Clé de chiffrement pour données sensibles
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode())
        else:
            self.fernet = Fernet(Fernet.generate_key())
        
        # Patterns suspects
        self.suspicious_patterns = {
            'rapid_requests': {'threshold': 100, 'window': 60},  # 100 req/min
            'cross_tenant_access': {'threshold': 5, 'window': 300},  # 5 tentatives/5min
            'privilege_escalation': {'threshold': 3, 'window': 3600}  # 3 tentatives/heure
        }
        
        # Historique des accès pour détection d'anomalies
        self.access_history = {}
        self.access_lock = threading.RLock()
    
    def validate_access(
        self,
        tenant_id: str,
        user_context: UserContext,
        requested_access: AccessLevel
    ) -> Tuple[bool, Optional[ViolationType]]:
        """
        Valide l'accès d'un utilisateur à un tenant.
        
        Returns:
            (access_granted, violation_type)
        """
        try:
            # Vérification des permissions basiques
            if tenant_id not in user_context.tenant_access:
                return False, ViolationType.UNAUTHORIZED_TENANT
            
            user_access = user_context.tenant_access[tenant_id]
            
            # Hiérarchie des accès: OWNER > ADMIN > WRITE > READ
            access_hierarchy = {
                AccessLevel.READ: 0,
                AccessLevel.WRITE: 1,
                AccessLevel.ADMIN: 2,
                AccessLevel.OWNER: 3
            }
            
            if access_hierarchy[user_access] < access_hierarchy[requested_access]:
                return False, ViolationType.INSUFFICIENT_PERMISSIONS
            
            # Détection de patterns suspects
            violation = self._detect_suspicious_activity(tenant_id, user_context)
            if violation:
                return False, violation
            
            # Enregistrement de l'accès valide
            self._record_access(tenant_id, user_context.user_id)
            
            return True, None
            
        except Exception as e:
            logger.error(
                "Erreur validation accès tenant",
                tenant_id=tenant_id,
                user_id=user_context.user_id,
                error=str(e)
            )
            return False, ViolationType.SUSPICIOUS_PATTERN
    
    def _detect_suspicious_activity(
        self,
        tenant_id: str,
        user_context: UserContext
    ) -> Optional[ViolationType]:
        """Détecte les activités suspectes."""
        user_id = user_context.user_id
        now = datetime.utcnow()
        
        with self.access_lock:
            # Initialisation historique utilisateur
            if user_id not in self.access_history:
                self.access_history[user_id] = {
                    'requests': [],
                    'tenant_accesses': {},
                    'escalation_attempts': []
                }
            
            history = self.access_history[user_id]
            
            # Nettoyage de l'historique ancien
            cutoff = now - timedelta(hours=1)
            history['requests'] = [req for req in history['requests'] if req['timestamp'] > cutoff]
            
            # Détection de requêtes rapides
            recent_requests = [
                req for req in history['requests']
                if req['timestamp'] > now - timedelta(seconds=60)
            ]
            
            if len(recent_requests) > self.suspicious_patterns['rapid_requests']['threshold']:
                return ViolationType.RATE_LIMIT_EXCEEDED
            
            # Détection d'accès cross-tenant suspect
            recent_tenants = set()
            for req in history['requests']:
                if req['timestamp'] > now - timedelta(minutes=5):
                    recent_tenants.add(req['tenant_id'])
            
            if len(recent_tenants) > self.suspicious_patterns['cross_tenant_access']['threshold']:
                return ViolationType.SUSPICIOUS_PATTERN
            
            # Enregistrement de la requête actuelle
            history['requests'].append({
                'timestamp': now,
                'tenant_id': tenant_id,
                'ip_address': user_context.ip_address
            })
            
            return None
    
    def _record_access(self, tenant_id: str, user_id: str):
        """Enregistre un accès valide pour audit."""
        # Ici on pourrait envoyer vers un système d'audit externe
        logger.info(
            "Accès tenant autorisé",
            tenant_id=tenant_id,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Chiffre des données sensibles."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Déchiffre des données sensibles."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

class TenantRepository:
    """Repository pour accès aux données tenant."""
    
    def __init__(self, database_url: Optional[str] = None):
        if database_url:
            self.engine = create_engine(database_url)
            self.metadata = MetaData()
            self._init_tables()
        else:
            # Mode mock pour développement
            self.engine = None
            self._mock_data = self._generate_mock_data()
    
    def _init_tables(self):
        """Initialise les tables de base de données."""
        # Définition des tables (simplifié)
        self.tenants_table = Table(
            'tenants', self.metadata,
            # Colonnes définies ailleurs ou via migrations
            autoload=True, autoload_with=self.engine
        )
        
        self.tenant_metrics_table = Table(
            'tenant_metrics', self.metadata,
            autoload=True, autoload_with=self.engine
        )
    
    def _generate_mock_data(self) -> Dict[str, TenantMetadata]:
        """Génère des données mock pour développement."""
        return {
            "universal_music_001": TenantMetadata(
                tenant_id="universal_music_001",
                name="Universal Music Group",
                type=TenantType.LABEL,
                region="global",
                locale="en",
                business_data={
                    "artists_count": 2500,
                    "monthly_streams": 15000000000,
                    "countries": 180,
                    "genres": ["pop", "rock", "hip-hop", "classical"]
                },
                alert_preferences={
                    "critical_threshold": 5.0,
                    "auto_escalation": True,
                    "business_hours_only": False
                },
                notification_channels=["slack", "email", "pagerduty"],
                quota_limits={
                    "api_calls_per_minute": 10000,
                    "storage_gb": 50000,
                    "ai_inference_per_hour": 100000
                }
            ),
            "taylor_swift_001": TenantMetadata(
                tenant_id="taylor_swift_001",
                name="Taylor Swift",
                type=TenantType.ARTIST,
                parent_tenant_id="universal_music_001",
                region="us",
                locale="en",
                business_data={
                    "monthly_streams": 800000000,
                    "albums_count": 15,
                    "fan_base_size": 250000000,
                    "primary_genres": ["pop", "country", "folk"]
                },
                alert_preferences={
                    "critical_threshold": 2.0,
                    "auto_escalation": True,
                    "business_hours_only": False
                },
                notification_channels=["slack", "email"]
            ),
            "abbey_road_studios": TenantMetadata(
                tenant_id="abbey_road_studios",
                name="Abbey Road Studios",
                type=TenantType.STUDIO,
                region="uk",
                locale="en",
                business_data={
                    "studio_rooms": 12,
                    "recording_hours_per_month": 2000,
                    "equipment_value_usd": 5000000
                },
                alert_preferences={
                    "critical_threshold": 10.0,
                    "auto_escalation": False,
                    "business_hours_only": True
                }
            )
        }
    
    async def get_tenant_metadata(self, tenant_id: str) -> Optional[TenantMetadata]:
        """Récupère les métadonnées d'un tenant."""
        if self.engine:
            # Requête base de données réelle
            async with self.engine.connect() as conn:
                query = select(self.tenants_table).where(
                    self.tenants_table.c.tenant_id == tenant_id
                )
                result = await conn.execute(query)
                row = result.fetchone()
                
                if row:
                    return TenantMetadata(**dict(row))
                
                return None
        else:
            # Mode mock
            return self._mock_data.get(tenant_id)
    
    async def get_tenant_metrics(self, tenant_id: str) -> Dict[str, float]:
        """Récupère les métriques en temps réel d'un tenant."""
        if self.engine:
            # Requête métriques réelles
            async with self.engine.connect() as conn:
                query = select(self.tenant_metrics_table).where(
                    and_(
                        self.tenant_metrics_table.c.tenant_id == tenant_id,
                        self.tenant_metrics_table.c.timestamp > datetime.utcnow() - timedelta(minutes=5)
                    )
                )
                result = await conn.execute(query)
                
                metrics = {}
                for row in result:
                    metrics[row.metric_name] = row.metric_value
                
                return metrics
        else:
            # Métriques mock
            import random
            return {
                "api_latency_p95": random.uniform(50, 500),
                "error_rate_percent": random.uniform(0, 5),
                "ai_accuracy_score": random.uniform(85, 99),
                "cpu_usage_percent": random.uniform(20, 80),
                "memory_usage_percent": random.uniform(30, 90),
                "active_users": random.randint(100, 10000),
                "requests_per_minute": random.randint(1000, 50000)
            }
    
    async def get_related_tenants(self, tenant_id: str) -> List[str]:
        """Récupère les tenants liés (hiérarchie, partenariats)."""
        metadata = await self.get_tenant_metadata(tenant_id)
        if not metadata:
            return []
        
        related = []
        
        # Ajout du parent
        if metadata.parent_tenant_id:
            related.append(metadata.parent_tenant_id)
        
        # Ajout des enfants
        related.extend(metadata.child_tenant_ids)
        
        return related

class TenantContextProvider:
    """
    Fournisseur principal de contexte tenant pour alerting.
    
    Centralise l'accès aux informations tenant avec cache intelligent,
    validation sécurisée et enrichissement contextuel.
    """
    
    def __init__(
        self,
        repository: Optional[TenantRepository] = None,
        redis_client: Optional[redis.Redis] = None,
        encryption_key: Optional[str] = None
    ):
        self.repository = repository or TenantRepository()
        self.redis_client = redis_client
        self.security_manager = TenantSecurityManager(encryption_key)
        
        # Cache L1 (mémoire)
        self._memory_cache = {}
        self._cache_lock = threading.RLock()
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Configuration cache
        self.cache_ttl = 3600  # 1 heure
        self.max_cache_size = 1000
        
        logger.info("TenantContextProvider initialisé")
    
    async def get_tenant_context(
        self,
        tenant_id: str,
        alert_type: str = "",
        user_context: Optional[UserContext] = None,
        force_refresh: bool = False
    ) -> Optional[TenantContext]:
        """
        Récupère le contexte complet d'un tenant.
        
        Args:
            tenant_id: ID du tenant
            alert_type: Type d'alerte pour contexte spécifique
            user_context: Contexte utilisateur pour validation RBAC
            force_refresh: Force le rechargement depuis la source
            
        Returns:
            Contexte tenant enrichi ou None si non autorisé
        """
        start_time = time.time()
        
        try:
            # Validation sécurité si contexte utilisateur fourni
            if user_context:
                access_granted, violation = self.security_manager.validate_access(
                    tenant_id, user_context, AccessLevel.READ
                )
                
                if not access_granted:
                    tenant_access_violations.labels(
                        tenant_id=tenant_id,
                        violation_type=violation.value,
                        user_id=user_context.user_id
                    ).inc()
                    
                    logger.warning(
                        "Accès tenant refusé",
                        tenant_id=tenant_id,
                        user_id=user_context.user_id,
                        violation=violation.value
                    )
                    return None
            
            # Vérification cache si pas de force refresh
            if not force_refresh:
                cached_context = await self._get_from_cache(tenant_id)
                if cached_context and cached_context.is_cache_valid():
                    tenant_context_cache_hits.labels(
                        tenant_id=tenant_id,
                        cache_level="l1"
                    ).inc()
                    
                    # Mise à jour des métriques de requête
                    tenant_context_requests_total.labels(
                        tenant_id=tenant_id,
                        tenant_type=cached_context.metadata.type.value,
                        alert_type=alert_type,
                        status="cache_hit"
                    ).inc()
                    
                    return cached_context
            
            # Chargement depuis la source
            context = await self._load_from_source(tenant_id, alert_type)
            
            if context:
                # Mise en cache
                await self._store_in_cache(tenant_id, context)
                
                # Métriques
                tenant_context_requests_total.labels(
                    tenant_id=tenant_id,
                    tenant_type=context.metadata.type.value,
                    alert_type=alert_type,
                    status="loaded"
                ).inc()
                
                tenant_context_duration.labels(
                    tenant_id=tenant_id,
                    source="database"
                ).observe(time.time() - start_time)
                
                logger.info(
                    "Contexte tenant chargé",
                    tenant_id=tenant_id,
                    tenant_type=context.metadata.type.value,
                    cache_source=context.cache_source,
                    duration_ms=(time.time() - start_time) * 1000
                )
                
                return context
            else:
                tenant_context_requests_total.labels(
                    tenant_id=tenant_id,
                    tenant_type="unknown",
                    alert_type=alert_type,
                    status="not_found"
                ).inc()
                
                logger.warning("Tenant non trouvé", tenant_id=tenant_id)
                return None
                
        except Exception as e:
            tenant_context_requests_total.labels(
                tenant_id=tenant_id,
                tenant_type="unknown",
                alert_type=alert_type,
                status="error"
            ).inc()
            
            logger.error(
                "Erreur récupération contexte tenant",
                tenant_id=tenant_id,
                alert_type=alert_type,
                error=str(e)
            )
            return None
    
    async def _get_from_cache(self, tenant_id: str) -> Optional[TenantContext]:
        """Récupère depuis le cache (L1 puis L2)."""
        # Cache L1 (mémoire)
        with self._cache_lock:
            if tenant_id in self._memory_cache:
                self._cache_stats['hits'] += 1
                return self._memory_cache[tenant_id]
        
        # Cache L2 (Redis)
        if self.redis_client:
            try:
                cache_key = f"tenant_context:{tenant_id}"
                cached_data = self.redis_client.get(cache_key)
                
                if cached_data:
                    data = json.loads(cached_data)
                    
                    # Reconstruction du contexte
                    metadata = TenantMetadata(
                        tenant_id=data['metadata']['tenant_id'],
                        name=data['metadata']['name'],
                        type=TenantType(data['metadata']['type']),
                        **{k: v for k, v in data['metadata'].items() 
                           if k not in ['tenant_id', 'name', 'type']}
                    )
                    
                    context = TenantContext(
                        metadata=metadata,
                        alert_config=data['alert_config'],
                        current_alerts=data['current_alerts'],
                        alert_history=data['alert_history'],
                        performance_metrics=data['performance_metrics'],
                        resource_usage=data['resource_usage'],
                        related_tenants=data['related_tenants'],
                        service_dependencies=data['service_dependencies'],
                        cached_at=datetime.fromisoformat(data['cached_at']),
                        cache_ttl=data['cache_ttl'],
                        cache_source="redis"
                    )
                    
                    # Mise en cache L1
                    with self._cache_lock:
                        self._memory_cache[tenant_id] = context
                    
                    tenant_context_cache_hits.labels(
                        tenant_id=tenant_id,
                        cache_level="l2"
                    ).inc()
                    
                    return context
                    
            except Exception as e:
                logger.debug(f"Erreur lecture cache Redis", tenant_id=tenant_id, error=str(e))
        
        # Cache miss
        with self._cache_lock:
            self._cache_stats['misses'] += 1
        
        return None
    
    async def _load_from_source(
        self,
        tenant_id: str,
        alert_type: str
    ) -> Optional[TenantContext]:
        """Charge le contexte depuis la source de données."""
        # Chargement des métadonnées base
        metadata = await self.repository.get_tenant_metadata(tenant_id)
        if not metadata:
            return None
        
        # Enrichissement avec métriques temps réel
        performance_metrics = await self.repository.get_tenant_metrics(tenant_id)
        
        # Tenants liés
        related_tenants = await self.repository.get_related_tenants(tenant_id)
        
        # Configuration d'alerting spécifique
        alert_config = self._build_alert_config(metadata, alert_type)
        
        # Construction du contexte enrichi
        context = TenantContext(
            metadata=metadata,
            alert_config=alert_config,
            performance_metrics=performance_metrics,
            related_tenants=related_tenants,
            cache_source="database"
        )
        
        return context
    
    def _build_alert_config(
        self,
        metadata: TenantMetadata,
        alert_type: str
    ) -> Dict[str, Any]:
        """Construit la configuration d'alerting pour le tenant."""
        # Configuration de base par type de tenant
        base_config = TENANT_TYPES.get(metadata.type.value, {})
        
        # Merge avec préférences tenant
        config = {
            "channels": metadata.notification_channels or base_config.get("default_channels", ["email"]),
            "escalation_policy": metadata.escalation_policies.get(alert_type) or base_config.get("escalation_policy"),
            "thresholds": metadata.alert_preferences,
            "business_hours_only": metadata.alert_preferences.get("business_hours_only", False),
            "auto_escalation": metadata.alert_preferences.get("auto_escalation", True),
            "tenant_specific": {
                "locale": metadata.locale,
                "timezone": metadata.timezone,
                "industry_context": metadata.business_data
            }
        }
        
        return config
    
    async def _store_in_cache(self, tenant_id: str, context: TenantContext):
        """Stocke le contexte dans les caches."""
        # Cache L1 (mémoire)
        with self._cache_lock:
            # Éviction si cache plein
            if len(self._memory_cache) >= self.max_cache_size:
                self._evict_lru()
            
            self._memory_cache[tenant_id] = context
            active_tenant_contexts.set(len(self._memory_cache))
        
        # Cache L2 (Redis)
        if self.redis_client:
            try:
                cache_key = f"tenant_context:{tenant_id}"
                cache_data = json.dumps(context.to_dict(), default=str)
                
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    cache_data
                )
                
            except Exception as e:
                logger.debug(f"Erreur écriture cache Redis", tenant_id=tenant_id, error=str(e))
    
    def _evict_lru(self):
        """Éviction LRU du cache mémoire."""
        if not self._memory_cache:
            return
        
        # Trouve l'entrée la plus ancienne
        oldest_key = min(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k].cached_at
        )
        
        del self._memory_cache[oldest_key]
        self._cache_stats['evictions'] += 1
    
    def validate_tenant_context(
        self,
        tenant_id: str,
        user_context: UserContext
    ) -> bool:
        """
        Valide l'accès à un contexte tenant.
        
        Fonction publique pour validation externe.
        """
        access_granted, _ = self.security_manager.validate_access(
            tenant_id, user_context, AccessLevel.READ
        )
        return access_granted
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        with self._cache_lock:
            stats = self._cache_stats.copy()
            stats['memory_cache_size'] = len(self._memory_cache)
            stats['max_cache_size'] = self.max_cache_size
            
            if stats['hits'] + stats['misses'] > 0:
                stats['hit_rate'] = stats['hits'] / (stats['hits'] + stats['misses'])
            else:
                stats['hit_rate'] = 0.0
        
        return stats
    
    async def invalidate_cache(self, tenant_id: str):
        """Invalide le cache pour un tenant spécifique."""
        # Cache L1
        with self._cache_lock:
            if tenant_id in self._memory_cache:
                del self._memory_cache[tenant_id]
                active_tenant_contexts.set(len(self._memory_cache))
        
        # Cache L2
        if self.redis_client:
            try:
                cache_key = f"tenant_context:{tenant_id}"
                self.redis_client.delete(cache_key)
            except Exception as e:
                logger.debug(f"Erreur invalidation cache Redis", tenant_id=tenant_id, error=str(e))
        
        logger.info("Cache tenant invalidé", tenant_id=tenant_id)
    
    async def warm_cache(self, tenant_ids: List[str]):
        """Préchauffe le cache pour une liste de tenants."""
        for tenant_id in tenant_ids:
            try:
                await self.get_tenant_context(tenant_id, force_refresh=True)
            except Exception as e:
                logger.warning(f"Erreur préchauffage cache", tenant_id=tenant_id, error=str(e))
        
        logger.info(f"Cache préchauffé", tenant_count=len(tenant_ids))

# Fonctions utilitaires

def create_user_context(
    user_id: str,
    tenant_access: Dict[str, str],
    roles: List[str] = None,
    **kwargs
) -> UserContext:
    """Crée un contexte utilisateur pour validation."""
    # Conversion des niveaux d'accès string vers enum
    access_dict = {}
    for tenant_id, access_str in tenant_access.items():
        try:
            access_dict[tenant_id] = AccessLevel(access_str)
        except ValueError:
            access_dict[tenant_id] = AccessLevel.READ  # Fallback sûr
    
    return UserContext(
        user_id=user_id,
        roles=roles or [],
        tenant_access=access_dict,
        **kwargs
    )

async def get_tenant_alert_config(
    tenant_id: str,
    alert_type: str,
    provider: Optional[TenantContextProvider] = None
) -> Dict[str, Any]:
    """Récupère la configuration d'alerte pour un tenant."""
    if not provider:
        provider = TenantContextProvider()
    
    context = await provider.get_tenant_context(tenant_id, alert_type)
    
    if context:
        return context.alert_config
    
    # Configuration par défaut si tenant non trouvé
    return {
        "channels": ["email"],
        "escalation_policy": "default",
        "thresholds": {},
        "business_hours_only": False,
        "auto_escalation": False
    }

def audit_tenant_access(
    tenant_id: str,
    user_id: str,
    action: str,
    result: str,
    **metadata
):
    """Enregistre un accès tenant pour audit."""
    audit_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "tenant_id": tenant_id,
        "user_id": user_id,
        "action": action,
        "result": result,
        "metadata": metadata
    }
    
    # Ici on pourrait envoyer vers un système d'audit externe
    logger.info(
        "Audit accès tenant",
        **audit_data
    )
