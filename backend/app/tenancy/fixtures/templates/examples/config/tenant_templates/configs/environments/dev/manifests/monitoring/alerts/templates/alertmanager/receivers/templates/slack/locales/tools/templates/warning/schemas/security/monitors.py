"""
Security Monitoring System for Multi-Tenant Architecture
======================================================

Ce module implémente un système de monitoring de sécurité avancé pour
l'architecture multi-tenant du Spotify AI Agent.

Auteur: Fahed Mlaiel
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np
from collections import defaultdict, deque
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from .core import SecurityLevel, ThreatType, SecurityEvent, TenantSecurityConfig
from .schemas import ComplianceStandard

logger = logging.getLogger(__name__)


class MonitoringLevel(Enum):
    """Niveaux de monitoring"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    INTENSIVE = "intensive"
    FORENSIC = "forensic"


class AnomalyType(Enum):
    """Types d'anomalies détectées"""
    VOLUME_ANOMALY = "volume_anomaly"
    PATTERN_ANOMALY = "pattern_anomaly"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"
    DEVICE_ANOMALY = "device_anomaly"
    API_ANOMALY = "api_anomaly"


@dataclass
class SecurityMetrics:
    """Métriques de sécurité"""
    tenant_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Métriques d'accès
    successful_logins: int = 0
    failed_logins: int = 0
    active_sessions: int = 0
    unique_users: int = 0
    
    # Métriques de menaces
    threat_events: int = 0
    blocked_attempts: int = 0
    escalated_events: int = 0
    false_positives: int = 0
    
    # Métriques de conformité
    compliance_violations: int = 0
    audit_events: int = 0
    data_access_events: int = 0
    
    # Métriques de performance
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    availability: float = 100.0
    
    # Métriques réseau
    unique_ips: int = 0
    geographic_regions: int = 0
    vpn_connections: int = 0


@dataclass
class AnomalyDetection:
    """Résultat de détection d'anomalie"""
    anomaly_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    anomaly_type: AnomalyType = AnomalyType.VOLUME_ANOMALY
    severity: SecurityLevel = SecurityLevel.LOW
    confidence: float = 0.0
    
    # Détails de l'anomalie
    detected_at: datetime = field(default_factory=datetime.utcnow)
    affected_resource: str = ""
    baseline_value: float = 0.0
    current_value: float = 0.0
    deviation_percentage: float = 0.0
    
    # Contexte
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_events: List[str] = field(default_factory=list)
    impact_assessment: str = ""
    
    # Actions recommandées
    recommended_actions: List[str] = field(default_factory=list)
    auto_mitigated: bool = False


@dataclass
class ThreatIntelligence:
    """Intelligence sur les menaces"""
    threat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    threat_type: ThreatType = ThreatType.SUSPICIOUS_ACTIVITY
    source_ip: str = ""
    user_agent: str = ""
    
    # Classification
    severity: SecurityLevel = SecurityLevel.MEDIUM
    confidence: float = 0.0
    ioc_match: bool = False  # Indicator of Compromise
    
    # Géolocalisation
    country: str = ""
    region: str = ""
    is_tor: bool = False
    is_vpn: bool = False
    is_proxy: bool = False
    
    # Historique
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    occurrence_count: int = 1
    
    # Attribution
    attack_patterns: List[str] = field(default_factory=list)
    malware_family: Optional[str] = None
    campaign_id: Optional[str] = None


class SecurityMonitor:
    """
    Moniteur principal de sécurité
    """
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.monitoring_level = MonitoringLevel.ENHANCED
        self.metrics_buffer: Dict[str, List[SecurityMetrics]] = defaultdict(list)
        self.is_running = False
        self.monitoring_tasks = []
        
    async def initialize(self):
        """Initialise le moniteur de sécurité"""
        self.is_running = True
        
        # Démarrage des tâches de monitoring
        tasks = [
            self._metrics_collector(),
            self._real_time_monitor(),
            self._metrics_aggregator(),
            self._health_checker()
        ]
        
        for task_func in tasks:
            task = asyncio.create_task(task_func)
            self.monitoring_tasks.append(task)
        
        logger.info("SecurityMonitor initialized and started")
    
    async def shutdown(self):
        """Arrête le moniteur de sécurité"""
        self.is_running = False
        
        # Arrêt des tâches
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        logger.info("SecurityMonitor shutdown completed")
    
    async def _metrics_collector(self):
        """Collecte les métriques de sécurité"""
        while self.is_running:
            try:
                await self._collect_tenant_metrics()
                await asyncio.sleep(60)  # Collecte toutes les minutes
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(5)
    
    async def _real_time_monitor(self):
        """Monitoring en temps réel"""
        while self.is_running:
            try:
                await self._process_security_events()
                await asyncio.sleep(1)  # Traitement en temps réel
            except Exception as e:
                logger.error(f"Error in real-time monitor: {e}")
                await asyncio.sleep(1)
    
    async def _metrics_aggregator(self):
        """Agrégation des métriques"""
        while self.is_running:
            try:
                await self._aggregate_metrics()
                await asyncio.sleep(300)  # Agrégation toutes les 5 minutes
            except Exception as e:
                logger.error(f"Error in metrics aggregator: {e}")
                await asyncio.sleep(30)
    
    async def _health_checker(self):
        """Vérification de santé du système"""
        while self.is_running:
            try:
                await self._check_system_health()
                await asyncio.sleep(120)  # Vérification toutes les 2 minutes
            except Exception as e:
                logger.error(f"Error in health checker: {e}")
                await asyncio.sleep(60)
    
    async def _collect_tenant_metrics(self):
        """Collecte les métriques pour tous les tenants"""
        tenant_ids = await self._get_active_tenants()
        
        for tenant_id in tenant_ids:
            try:
                metrics = await self._calculate_tenant_metrics(tenant_id)
                self.metrics_buffer[tenant_id].append(metrics)
                
                # Limitation de la taille du buffer
                if len(self.metrics_buffer[tenant_id]) > 1440:  # 24h de données
                    self.metrics_buffer[tenant_id].pop(0)
                
                # Stockage en cache Redis
                await self._store_metrics_cache(tenant_id, metrics)
                
            except Exception as e:
                logger.error(f"Error collecting metrics for tenant {tenant_id}: {e}")
    
    async def _get_active_tenants(self) -> List[str]:
        """Récupère la liste des tenants actifs"""
        tenant_keys = await self.redis.keys("tenant:*:active")
        return [key.decode().split(':')[1] for key in tenant_keys]
    
    async def _calculate_tenant_metrics(self, tenant_id: str) -> SecurityMetrics:
        """Calcule les métriques pour un tenant"""
        metrics = SecurityMetrics(tenant_id=tenant_id)
        
        # Métriques d'accès
        metrics.successful_logins = await self._count_events(tenant_id, "successful_login", hours=1)
        metrics.failed_logins = await self._count_events(tenant_id, "failed_login", hours=1)
        metrics.active_sessions = await self._count_active_sessions(tenant_id)
        metrics.unique_users = await self._count_unique_users(tenant_id, hours=1)
        
        # Métriques de menaces
        metrics.threat_events = await self._count_events(tenant_id, "threat_detected", hours=1)
        metrics.blocked_attempts = await self._count_events(tenant_id, "access_blocked", hours=1)
        metrics.escalated_events = await self._count_events(tenant_id, "event_escalated", hours=1)
        
        # Métriques de conformité
        metrics.compliance_violations = await self._count_events(tenant_id, "compliance_violation", hours=1)
        metrics.audit_events = await self._count_events(tenant_id, "audit_event", hours=1)
        
        # Métriques de performance
        metrics.avg_response_time_ms = await self._calculate_avg_response_time(tenant_id)
        metrics.error_rate = await self._calculate_error_rate(tenant_id)
        
        # Métriques réseau
        metrics.unique_ips = await self._count_unique_ips(tenant_id, hours=1)
        metrics.geographic_regions = await self._count_geographic_regions(tenant_id, hours=1)
        
        return metrics
    
    async def _count_events(self, tenant_id: str, event_type: str, hours: int = 1) -> int:
        """Compte les événements d'un type donné"""
        key = f"events:{tenant_id}:{event_type}"
        
        # Utilisation d'un compteur Redis avec expiration
        since = datetime.utcnow() - timedelta(hours=hours)
        timestamp = int(since.timestamp())
        
        count = await self.redis.zcount(key, timestamp, "+inf")
        return count
    
    async def _count_active_sessions(self, tenant_id: str) -> int:
        """Compte les sessions actives"""
        pattern = f"session:{tenant_id}:*"
        keys = await self.redis.keys(pattern)
        return len(keys)
    
    async def _count_unique_users(self, tenant_id: str, hours: int = 1) -> int:
        """Compte les utilisateurs uniques"""
        key = f"unique_users:{tenant_id}"
        
        # Utilisation d'un HyperLogLog pour estimer les uniques
        count = await self.redis.pfcount(key)
        return count
    
    async def _calculate_avg_response_time(self, tenant_id: str) -> float:
        """Calcule le temps de réponse moyen"""
        key = f"response_times:{tenant_id}"
        
        times = await self.redis.lrange(key, 0, -1)
        if not times:
            return 0.0
        
        float_times = [float(t) for t in times]
        return statistics.mean(float_times)
    
    async def _calculate_error_rate(self, tenant_id: str) -> float:
        """Calcule le taux d'erreur"""
        success_key = f"requests_success:{tenant_id}"
        error_key = f"requests_error:{tenant_id}"
        
        success_count = await self.redis.get(success_key) or 0
        error_count = await self.redis.get(error_key) or 0
        
        total = int(success_count) + int(error_count)
        if total == 0:
            return 0.0
        
        return (int(error_count) / total) * 100
    
    async def _count_unique_ips(self, tenant_id: str, hours: int = 1) -> int:
        """Compte les IPs uniques"""
        key = f"unique_ips:{tenant_id}"
        count = await self.redis.pfcount(key)
        return count
    
    async def _count_geographic_regions(self, tenant_id: str, hours: int = 1) -> int:
        """Compte les régions géographiques uniques"""
        key = f"geo_regions:{tenant_id}"
        regions = await self.redis.smembers(key)
        return len(regions)
    
    async def _store_metrics_cache(self, tenant_id: str, metrics: SecurityMetrics):
        """Stocke les métriques en cache"""
        key = f"metrics:{tenant_id}:current"
        metrics_json = json.dumps(metrics.__dict__, default=str)
        await self.redis.set(key, metrics_json, ex=3600)
    
    async def _process_security_events(self):
        """Traite les événements de sécurité en temps réel"""
        # Écoute des événements depuis la queue Redis
        events_queue = "security_events_queue"
        
        event_data = await self.redis.blpop(events_queue, timeout=1)
        if not event_data:
            return
        
        try:
            event_json = event_data[1]
            event = SecurityEvent(**json.loads(event_json))
            
            # Traitement de l'événement
            await self._analyze_security_event(event)
            
        except Exception as e:
            logger.error(f"Error processing security event: {e}")
    
    async def _analyze_security_event(self, event: SecurityEvent):
        """Analyse un événement de sécurité"""
        # Mise à jour des compteurs
        await self._update_event_counters(event)
        
        # Détection d'anomalies
        await self._check_for_anomalies(event)
        
        # Évaluation de la menace
        threat_score = await self._calculate_threat_score(event)
        event.threat_score = threat_score
        
        # Escalade si nécessaire
        if threat_score > 0.8:
            await self._escalate_event(event)
    
    async def _update_event_counters(self, event: SecurityEvent):
        """Met à jour les compteurs d'événements"""
        timestamp = int(event.timestamp.timestamp())
        
        # Compteur par type d'événement
        event_key = f"events:{event.tenant_id}:{event.event_type}"
        await self.redis.zadd(event_key, {event.event_id: timestamp})
        await self.redis.expire(event_key, 86400)  # 24h
        
        # Compteur d'IPs uniques
        if event.source_ip:
            ip_key = f"unique_ips:{event.tenant_id}"
            await self.redis.pfadd(ip_key, event.source_ip)
            await self.redis.expire(ip_key, 86400)
        
        # Compteur d'utilisateurs uniques
        if event.user_id:
            user_key = f"unique_users:{event.tenant_id}"
            await self.redis.pfadd(user_key, event.user_id)
            await self.redis.expire(user_key, 86400)
    
    async def _check_for_anomalies(self, event: SecurityEvent):
        """Vérifie les anomalies pour un événement"""
        # Délégation aux détecteurs spécialisés
        anomaly_detector = AnomalyDetector(self.redis, self.db)
        await anomaly_detector.analyze_event(event)
    
    async def _calculate_threat_score(self, event: SecurityEvent) -> float:
        """Calcule le score de menace d'un événement"""
        base_score = 0.0
        
        # Score basé sur la sévérité
        severity_scores = {
            SecurityLevel.LOW: 0.1,
            SecurityLevel.MEDIUM: 0.3,
            SecurityLevel.HIGH: 0.7,
            SecurityLevel.CRITICAL: 1.0
        }
        base_score += severity_scores.get(event.severity, 0.1)
        
        # Score basé sur l'historique de l'IP
        if event.source_ip:
            ip_history_score = await self._get_ip_reputation_score(event.source_ip)
            base_score += ip_history_score * 0.3
        
        # Score basé sur l'historique de l'utilisateur
        if event.user_id:
            user_history_score = await self._get_user_risk_score(event.user_id, event.tenant_id)
            base_score += user_history_score * 0.2
        
        # Score basé sur la fréquence des événements
        frequency_score = await self._get_frequency_score(event)
        base_score += frequency_score * 0.3
        
        return min(base_score, 1.0)
    
    async def _get_ip_reputation_score(self, ip: str) -> float:
        """Récupère le score de réputation d'une IP"""
        key = f"ip_reputation:{ip}"
        score = await self.redis.get(key)
        return float(score) if score else 0.0
    
    async def _get_user_risk_score(self, user_id: str, tenant_id: str) -> float:
        """Récupère le score de risque d'un utilisateur"""
        key = f"user_risk:{tenant_id}:{user_id}"
        score = await self.redis.get(key)
        return float(score) if score else 0.0
    
    async def _get_frequency_score(self, event: SecurityEvent) -> float:
        """Calcule un score basé sur la fréquence des événements"""
        # Analyse de la fréquence des événements similaires
        recent_events = await self._get_recent_similar_events(event)
        
        if len(recent_events) > 10:  # Seuil de fréquence élevée
            return 0.8
        elif len(recent_events) > 5:
            return 0.5
        elif len(recent_events) > 2:
            return 0.3
        
        return 0.0
    
    async def _get_recent_similar_events(self, event: SecurityEvent) -> List[str]:
        """Récupère les événements similaires récents"""
        key = f"events:{event.tenant_id}:{event.event_type}"
        since = int((datetime.utcnow() - timedelta(minutes=5)).timestamp())
        
        events = await self.redis.zrangebyscore(key, since, "+inf")
        return events
    
    async def _escalate_event(self, event: SecurityEvent):
        """Escalade un événement critique"""
        event.escalated = True
        
        # Envoi vers le processeur d'alertes
        alert_data = {
            "event_id": event.event_id,
            "tenant_id": event.tenant_id,
            "severity": event.severity.value,
            "threat_score": event.threat_score,
            "escalated_at": datetime.utcnow().isoformat()
        }
        
        await self.redis.lpush("escalated_events_queue", json.dumps(alert_data))
    
    async def _aggregate_metrics(self):
        """Agrège les métriques collectées"""
        for tenant_id, metrics_list in self.metrics_buffer.items():
            if not metrics_list:
                continue
            
            try:
                # Agrégation par heure
                hourly_metrics = await self._aggregate_hourly_metrics(tenant_id, metrics_list)
                await self._store_hourly_metrics(tenant_id, hourly_metrics)
                
                # Agrégation par jour (si nécessaire)
                if len(metrics_list) >= 1440:  # 24h de données
                    daily_metrics = await self._aggregate_daily_metrics(tenant_id, metrics_list)
                    await self._store_daily_metrics(tenant_id, daily_metrics)
                
            except Exception as e:
                logger.error(f"Error aggregating metrics for tenant {tenant_id}: {e}")
    
    async def _aggregate_hourly_metrics(self, tenant_id: str, metrics_list: List[SecurityMetrics]) -> SecurityMetrics:
        """Agrège les métriques par heure"""
        if not metrics_list:
            return SecurityMetrics(tenant_id=tenant_id)
        
        # Agrégation des dernières 60 valeurs (1 heure)
        recent_metrics = metrics_list[-60:] if len(metrics_list) >= 60 else metrics_list
        
        aggregated = SecurityMetrics(tenant_id=tenant_id)
        
        # Sommes
        aggregated.successful_logins = sum(m.successful_logins for m in recent_metrics)
        aggregated.failed_logins = sum(m.failed_logins for m in recent_metrics)
        aggregated.threat_events = sum(m.threat_events for m in recent_metrics)
        aggregated.blocked_attempts = sum(m.blocked_attempts for m in recent_metrics)
        
        # Moyennes
        if recent_metrics:
            aggregated.avg_response_time_ms = statistics.mean(m.avg_response_time_ms for m in recent_metrics)
            aggregated.error_rate = statistics.mean(m.error_rate for m in recent_metrics)
            aggregated.availability = statistics.mean(m.availability for m in recent_metrics)
        
        # Maximums
        aggregated.active_sessions = max(m.active_sessions for m in recent_metrics) if recent_metrics else 0
        aggregated.unique_users = max(m.unique_users for m in recent_metrics) if recent_metrics else 0
        
        return aggregated
    
    async def _store_hourly_metrics(self, tenant_id: str, metrics: SecurityMetrics):
        """Stocke les métriques horaires"""
        hour_key = metrics.timestamp.strftime("%Y-%m-%d-%H")
        key = f"metrics_hourly:{tenant_id}:{hour_key}"
        
        metrics_json = json.dumps(metrics.__dict__, default=str)
        await self.redis.set(key, metrics_json, ex=86400 * 7)  # 7 jours
    
    async def _aggregate_daily_metrics(self, tenant_id: str, metrics_list: List[SecurityMetrics]) -> SecurityMetrics:
        """Agrège les métriques par jour"""
        # Implémentation similaire à l'agrégation horaire
        pass
    
    async def _store_daily_metrics(self, tenant_id: str, metrics: SecurityMetrics):
        """Stocke les métriques quotidiennes"""
        date_key = metrics.timestamp.strftime("%Y-%m-%d")
        key = f"metrics_daily:{tenant_id}:{date_key}"
        
        metrics_json = json.dumps(metrics.__dict__, default=str)
        await self.redis.set(key, metrics_json, ex=86400 * 30)  # 30 jours
    
    async def _check_system_health(self):
        """Vérifie la santé du système de monitoring"""
        health_status = {
            "monitoring_active": self.is_running,
            "redis_connected": await self._check_redis_health(),
            "db_connected": await self._check_db_health(),
            "metrics_collection_rate": await self._check_metrics_rate(),
            "last_health_check": datetime.utcnow().isoformat()
        }
        
        await self.redis.set("monitor_health", json.dumps(health_status), ex=300)
    
    async def _check_redis_health(self) -> bool:
        """Vérifie la santé de Redis"""
        try:
            await self.redis.ping()
            return True
        except Exception:
            return False
    
    async def _check_db_health(self) -> bool:
        """Vérifie la santé de la base de données"""
        try:
            # Test de connexion simple
            await self.db.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    async def _check_metrics_rate(self) -> float:
        """Vérifie le taux de collecte de métriques"""
        # Calcule le nombre de métriques collectées dans la dernière minute
        total_metrics = sum(len(metrics) for metrics in self.metrics_buffer.values())
        return total_metrics / 60.0  # métriques par seconde


class ThreatDetector:
    """
    Détecteur de menaces en temps réel
    """
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.threat_patterns = {}
        self.ioc_database = set()  # Indicators of Compromise
        
    async def initialize(self):
        """Initialise le détecteur de menaces"""
        await self._load_threat_patterns()
        await self._load_ioc_database()
        logger.info("ThreatDetector initialized")
    
    async def _load_threat_patterns(self):
        """Charge les patterns de menaces"""
        # Patterns de détection d'attaques
        self.threat_patterns = {
            ThreatType.BRUTE_FORCE: {
                "failed_login_threshold": 5,
                "time_window_minutes": 5,
                "confidence_threshold": 0.8
            },
            ThreatType.SQL_INJECTION: {
                "patterns": [
                    r"union\s+select",
                    r"or\s+1\s*=\s*1",
                    r"drop\s+table",
                    r"exec\s*\(",
                    r"script\s*>"
                ],
                "confidence_threshold": 0.9
            },
            ThreatType.XSS_ATTEMPT: {
                "patterns": [
                    r"<script[^>]*>",
                    r"javascript:",
                    r"onload\s*=",
                    r"onerror\s*=",
                    r"eval\s*\("
                ],
                "confidence_threshold": 0.8
            }
        }
    
    async def _load_ioc_database(self):
        """Charge la base de données des IoC"""
        # Chargement depuis des sources de threat intelligence
        # Simulation pour l'exemple
        self.ioc_database = {
            "192.168.1.100",  # IP malveillante exemple
            "malware.example.com",  # Domaine malveillant
            "bad-user-agent-v1.0"  # User-Agent suspect
        }
    
    async def detect_threats(self, event: SecurityEvent) -> List[ThreatIntelligence]:
        """Détecte les menaces dans un événement"""
        threats = []
        
        try:
            # Détection de force brute
            brute_force_threats = await self._detect_brute_force(event)
            threats.extend(brute_force_threats)
            
            # Détection d'injection SQL
            sql_injection_threats = await self._detect_sql_injection(event)
            threats.extend(sql_injection_threats)
            
            # Détection XSS
            xss_threats = await self._detect_xss_attempts(event)
            threats.extend(xss_threats)
            
            # Vérification IoC
            ioc_threats = await self._check_ioc_match(event)
            threats.extend(ioc_threats)
            
            # Détection d'anomalies de géolocalisation
            geo_threats = await self._detect_geo_anomalies(event)
            threats.extend(geo_threats)
            
        except Exception as e:
            logger.error(f"Error detecting threats: {e}")
        
        return threats
    
    async def _detect_brute_force(self, event: SecurityEvent) -> List[ThreatIntelligence]:
        """Détecte les attaques par force brute"""
        if event.event_type != "failed_login":
            return []
        
        threats = []
        pattern = self.threat_patterns[ThreatType.BRUTE_FORCE]
        
        # Compte les échecs de connexion récents
        key = f"failed_logins:{event.tenant_id}:{event.source_ip}"
        failed_count = await self.redis.incr(key)
        await self.redis.expire(key, pattern["time_window_minutes"] * 60)
        
        if failed_count >= pattern["failed_login_threshold"]:
            threat = ThreatIntelligence(
                threat_type=ThreatType.BRUTE_FORCE,
                source_ip=event.source_ip,
                user_agent=event.user_agent,
                severity=SecurityLevel.HIGH,
                confidence=pattern["confidence_threshold"],
                occurrence_count=failed_count
            )
            threats.append(threat)
        
        return threats
    
    async def _detect_sql_injection(self, event: SecurityEvent) -> List[ThreatIntelligence]:
        """Détecte les tentatives d'injection SQL"""
        import re
        
        threats = []
        pattern_config = self.threat_patterns[ThreatType.SQL_INJECTION]
        
        # Analyse du contenu de l'événement
        content_to_check = []
        
        if event.metadata:
            content_to_check.extend([
                str(event.metadata.get("query_params", "")),
                str(event.metadata.get("request_body", "")),
                str(event.metadata.get("headers", ""))
            ])
        
        for content in content_to_check:
            if not content:
                continue
            
            content_lower = content.lower()
            
            for pattern in pattern_config["patterns"]:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    threat = ThreatIntelligence(
                        threat_type=ThreatType.SQL_INJECTION,
                        source_ip=event.source_ip,
                        user_agent=event.user_agent,
                        severity=SecurityLevel.CRITICAL,
                        confidence=pattern_config["confidence_threshold"]
                    )
                    threat.attack_patterns.append(pattern)
                    threats.append(threat)
                    break
        
        return threats
    
    async def _detect_xss_attempts(self, event: SecurityEvent) -> List[ThreatIntelligence]:
        """Détecte les tentatives XSS"""
        import re
        
        threats = []
        pattern_config = self.threat_patterns[ThreatType.XSS_ATTEMPT]
        
        # Analyse similaire à SQL injection
        content_to_check = []
        
        if event.metadata:
            content_to_check.extend([
                str(event.metadata.get("query_params", "")),
                str(event.metadata.get("request_body", "")),
                str(event.metadata.get("user_input", ""))
            ])
        
        for content in content_to_check:
            if not content:
                continue
            
            for pattern in pattern_config["patterns"]:
                if re.search(pattern, content, re.IGNORECASE):
                    threat = ThreatIntelligence(
                        threat_type=ThreatType.XSS_ATTEMPT,
                        source_ip=event.source_ip,
                        user_agent=event.user_agent,
                        severity=SecurityLevel.HIGH,
                        confidence=pattern_config["confidence_threshold"]
                    )
                    threat.attack_patterns.append(pattern)
                    threats.append(threat)
                    break
        
        return threats
    
    async def _check_ioc_match(self, event: SecurityEvent) -> List[ThreatIntelligence]:
        """Vérifie la correspondance avec les IoC"""
        threats = []
        
        # Vérification IP
        if event.source_ip and event.source_ip in self.ioc_database:
            threat = ThreatIntelligence(
                threat_type=ThreatType.SUSPICIOUS_ACTIVITY,
                source_ip=event.source_ip,
                user_agent=event.user_agent,
                severity=SecurityLevel.CRITICAL,
                confidence=0.95,
                ioc_match=True
            )
            threats.append(threat)
        
        # Vérification User-Agent
        if event.user_agent and event.user_agent in self.ioc_database:
            threat = ThreatIntelligence(
                threat_type=ThreatType.SUSPICIOUS_ACTIVITY,
                source_ip=event.source_ip,
                user_agent=event.user_agent,
                severity=SecurityLevel.HIGH,
                confidence=0.85,
                ioc_match=True
            )
            threats.append(threat)
        
        return threats
    
    async def _detect_geo_anomalies(self, event: SecurityEvent) -> List[ThreatIntelligence]:
        """Détecte les anomalies géographiques"""
        if not event.source_ip or not event.user_id:
            return []
        
        threats = []
        
        try:
            # Récupération de la localisation de l'IP
            current_country = await self._get_country_from_ip(event.source_ip)
            
            # Récupération des pays habituels de l'utilisateur
            user_countries_key = f"user_countries:{event.tenant_id}:{event.user_id}"
            usual_countries = await self.redis.smembers(user_countries_key)
            usual_countries = {c.decode() for c in usual_countries}
            
            # Détection d'anomalie géographique
            if usual_countries and current_country not in usual_countries:
                # Vérification de la distance géographique
                is_far = await self._is_geographically_distant(current_country, usual_countries)
                
                if is_far:
                    threat = ThreatIntelligence(
                        threat_type=ThreatType.SUSPICIOUS_ACTIVITY,
                        source_ip=event.source_ip,
                        user_agent=event.user_agent,
                        severity=SecurityLevel.MEDIUM,
                        confidence=0.6,
                        country=current_country
                    )
                    threats.append(threat)
            
            # Mise à jour des pays de l'utilisateur
            await self.redis.sadd(user_countries_key, current_country)
            await self.redis.expire(user_countries_key, 86400 * 30)  # 30 jours
            
        except Exception as e:
            logger.error(f"Error detecting geo anomalies: {e}")
        
        return threats
    
    async def _get_country_from_ip(self, ip: str) -> str:
        """Récupère le pays depuis l'IP"""
        # Implémentation avec service de géolocalisation
        # Simulation pour l'exemple
        return "US"
    
    async def _is_geographically_distant(self, current_country: str, usual_countries: Set[str]) -> bool:
        """Vérifie si le pays actuel est géographiquement distant"""
        # Logique de calcul de distance géographique
        # Simulation pour l'exemple
        distant_pairs = {
            ("US", "RU"), ("FR", "CN"), ("DE", "BR")
        }
        
        for usual_country in usual_countries:
            if (current_country, usual_country) in distant_pairs or (usual_country, current_country) in distant_pairs:
                return True
        
        return False


class AnomalyDetector:
    """
    Détecteur d'anomalies comportementales
    """
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.baseline_window_days = 7
        self.anomaly_threshold = 2.0  # Nombre d'écarts-types
        
    async def initialize(self):
        """Initialise le détecteur d'anomalies"""
        logger.info("AnomalyDetector initialized")
    
    async def analyze_event(self, event: SecurityEvent) -> List[AnomalyDetection]:
        """Analyse un événement pour détecter des anomalies"""
        anomalies = []
        
        try:
            # Détection d'anomalies de volume
            volume_anomalies = await self._detect_volume_anomalies(event)
            anomalies.extend(volume_anomalies)
            
            # Détection d'anomalies temporelles
            temporal_anomalies = await self._detect_temporal_anomalies(event)
            anomalies.extend(temporal_anomalies)
            
            # Détection d'anomalies comportementales
            behavioral_anomalies = await self._detect_behavioral_anomalies(event)
            anomalies.extend(behavioral_anomalies)
            
        except Exception as e:
            logger.error(f"Error analyzing event for anomalies: {e}")
        
        return anomalies
    
    async def _detect_volume_anomalies(self, event: SecurityEvent) -> List[AnomalyDetection]:
        """Détecte les anomalies de volume"""
        anomalies = []
        
        # Analyse du volume d'événements par type
        current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        
        # Comptage actuel
        current_count = await self._count_events_in_hour(event.tenant_id, event.event_type, current_hour)
        
        # Calcul de la baseline
        baseline_stats = await self._calculate_baseline_stats(event.tenant_id, event.event_type)
        
        if baseline_stats and current_count > 0:
            mean = baseline_stats["mean"]
            std_dev = baseline_stats["std_dev"]
            
            # Détection d'anomalie si > threshold écarts-types
            if std_dev > 0:
                z_score = (current_count - mean) / std_dev
                
                if abs(z_score) > self.anomaly_threshold:
                    anomaly = AnomalyDetection(
                        tenant_id=event.tenant_id,
                        anomaly_type=AnomalyType.VOLUME_ANOMALY,
                        severity=SecurityLevel.MEDIUM if z_score > 0 else SecurityLevel.LOW,
                        confidence=min(abs(z_score) / self.anomaly_threshold, 1.0),
                        affected_resource=event.event_type,
                        baseline_value=mean,
                        current_value=current_count,
                        deviation_percentage=((current_count - mean) / mean) * 100 if mean > 0 else 0
                    )
                    
                    anomaly.impact_assessment = f"Volume spike detected: {current_count} vs baseline {mean:.1f}"
                    anomaly.recommended_actions = ["investigate_cause", "check_system_health"]
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _count_events_in_hour(self, tenant_id: str, event_type: str, hour: datetime) -> int:
        """Compte les événements dans une heure donnée"""
        start_timestamp = int(hour.timestamp())
        end_timestamp = int((hour + timedelta(hours=1)).timestamp())
        
        key = f"events:{tenant_id}:{event_type}"
        count = await self.redis.zcount(key, start_timestamp, end_timestamp)
        
        return count
    
    async def _calculate_baseline_stats(self, tenant_id: str, event_type: str) -> Optional[Dict[str, float]]:
        """Calcule les statistiques de baseline"""
        # Collecte des données historiques
        counts = []
        current_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        
        for i in range(24 * self.baseline_window_days):  # Données horaires sur la période
            hour = current_time - timedelta(hours=i+1)
            count = await self._count_events_in_hour(tenant_id, event_type, hour)
            counts.append(count)
        
        if len(counts) < 24:  # Données insuffisantes
            return None
        
        # Calcul des statistiques
        mean = statistics.mean(counts)
        std_dev = statistics.stdev(counts) if len(counts) > 1 else 0
        
        return {
            "mean": mean,
            "std_dev": std_dev,
            "median": statistics.median(counts),
            "sample_size": len(counts)
        }
    
    async def _detect_temporal_anomalies(self, event: SecurityEvent) -> List[AnomalyDetection]:
        """Détecte les anomalies temporelles"""
        anomalies = []
        
        # Analyse des patterns temporels inhabituels
        current_hour = event.timestamp.hour
        current_day = event.timestamp.weekday()
        
        # Récupération des patterns habituels
        user_patterns = await self._get_user_temporal_patterns(event.user_id, event.tenant_id)
        
        if user_patterns:
            # Vérification de l'heure inhabituelle
            usual_hours = user_patterns.get("usual_hours", [])
            if usual_hours and current_hour not in usual_hours:
                anomaly = AnomalyDetection(
                    tenant_id=event.tenant_id,
                    anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                    severity=SecurityLevel.LOW,
                    confidence=0.6,
                    affected_resource=f"user:{event.user_id}",
                    metadata={"unusual_hour": current_hour, "usual_hours": usual_hours}
                )
                anomaly.impact_assessment = f"Activity at unusual hour: {current_hour}"
                anomalies.append(anomaly)
        
        # Mise à jour des patterns
        await self._update_user_temporal_patterns(event.user_id, event.tenant_id, current_hour, current_day)
        
        return anomalies
    
    async def _get_user_temporal_patterns(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
        """Récupère les patterns temporels d'un utilisateur"""
        key = f"temporal_patterns:{tenant_id}:{user_id}"
        patterns_json = await self.redis.get(key)
        
        if patterns_json:
            return json.loads(patterns_json)
        
        return {}
    
    async def _update_user_temporal_patterns(self, user_id: str, tenant_id: str, hour: int, day: int):
        """Met à jour les patterns temporels d'un utilisateur"""
        key = f"temporal_patterns:{tenant_id}:{user_id}"
        patterns = await self._get_user_temporal_patterns(user_id, tenant_id)
        
        # Mise à jour des heures habituelles
        usual_hours = set(patterns.get("usual_hours", []))
        usual_hours.add(hour)
        patterns["usual_hours"] = list(usual_hours)
        
        # Mise à jour des jours habituels
        usual_days = set(patterns.get("usual_days", []))
        usual_days.add(day)
        patterns["usual_days"] = list(usual_days)
        
        patterns["last_updated"] = datetime.utcnow().isoformat()
        
        await self.redis.set(key, json.dumps(patterns), ex=86400 * 30)  # 30 jours
    
    async def _detect_behavioral_anomalies(self, event: SecurityEvent) -> List[AnomalyDetection]:
        """Détecte les anomalies comportementales"""
        anomalies = []
        
        if not event.user_id:
            return anomalies
        
        # Analyse des changements de comportement utilisateur
        user_behavior = await self._get_user_behavior_profile(event.user_id, event.tenant_id)
        
        if user_behavior:
            # Vérification des changements d'API usage
            api_anomalies = await self._detect_api_usage_anomalies(event, user_behavior)
            anomalies.extend(api_anomalies)
            
            # Vérification des changements de device/browser
            device_anomalies = await self._detect_device_anomalies(event, user_behavior)
            anomalies.extend(device_anomalies)
        
        # Mise à jour du profil comportemental
        await self._update_user_behavior_profile(event)
        
        return anomalies
    
    async def _get_user_behavior_profile(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
        """Récupère le profil comportemental d'un utilisateur"""
        key = f"behavior_profile:{tenant_id}:{user_id}"
        profile_json = await self.redis.get(key)
        
        if profile_json:
            return json.loads(profile_json)
        
        return {}
    
    async def _detect_api_usage_anomalies(self, event: SecurityEvent, user_behavior: Dict[str, Any]) -> List[AnomalyDetection]:
        """Détecte les anomalies d'usage API"""
        anomalies = []
        
        if event.action and "api_usage" in user_behavior:
            usual_apis = set(user_behavior["api_usage"].get("frequent_endpoints", []))
            current_api = event.metadata.get("api_endpoint")
            
            if current_api and current_api not in usual_apis:
                anomaly = AnomalyDetection(
                    tenant_id=event.tenant_id,
                    anomaly_type=AnomalyType.API_ANOMALY,
                    severity=SecurityLevel.LOW,
                    confidence=0.4,
                    affected_resource=f"user:{event.user_id}",
                    metadata={"new_api": current_api, "usual_apis": list(usual_apis)}
                )
                anomaly.impact_assessment = f"New API endpoint used: {current_api}"
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_device_anomalies(self, event: SecurityEvent, user_behavior: Dict[str, Any]) -> List[AnomalyDetection]:
        """Détecte les anomalies de device"""
        anomalies = []
        
        if event.user_agent and "device_info" in user_behavior:
            usual_user_agents = set(user_behavior["device_info"].get("user_agents", []))
            
            if event.user_agent not in usual_user_agents:
                anomaly = AnomalyDetection(
                    tenant_id=event.tenant_id,
                    anomaly_type=AnomalyType.DEVICE_ANOMALY,
                    severity=SecurityLevel.MEDIUM,
                    confidence=0.7,
                    affected_resource=f"user:{event.user_id}",
                    metadata={"new_user_agent": event.user_agent}
                )
                anomaly.impact_assessment = "New device/browser detected"
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _update_user_behavior_profile(self, event: SecurityEvent):
        """Met à jour le profil comportemental"""
        if not event.user_id:
            return
        
        key = f"behavior_profile:{event.tenant_id}:{event.user_id}"
        profile = await self._get_user_behavior_profile(event.user_id, event.tenant_id)
        
        # Mise à jour des API utilisées
        if "api_usage" not in profile:
            profile["api_usage"] = {"frequent_endpoints": []}
        
        api_endpoint = event.metadata.get("api_endpoint")
        if api_endpoint:
            endpoints = profile["api_usage"]["frequent_endpoints"]
            if api_endpoint not in endpoints:
                endpoints.append(api_endpoint)
                # Limitation à 50 endpoints
                if len(endpoints) > 50:
                    endpoints.pop(0)
        
        # Mise à jour des informations de device
        if "device_info" not in profile:
            profile["device_info"] = {"user_agents": []}
        
        if event.user_agent:
            user_agents = profile["device_info"]["user_agents"]
            if event.user_agent not in user_agents:
                user_agents.append(event.user_agent)
                # Limitation à 10 user agents
                if len(user_agents) > 10:
                    user_agents.pop(0)
        
        profile["last_updated"] = datetime.utcnow().isoformat()
        
        await self.redis.set(key, json.dumps(profile), ex=86400 * 60)  # 60 jours


class ComplianceMonitor:
    """
    Moniteur de conformité réglementaire
    """
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.compliance_rules = {}
        
    async def initialize(self):
        """Initialise le moniteur de conformité"""
        await self._load_compliance_rules()
        logger.info("ComplianceMonitor initialized")
    
    async def _load_compliance_rules(self):
        """Charge les règles de conformité"""
        # Implémentation du chargement des règles
        pass
    
    async def monitor_compliance(self, tenant_id: str) -> Dict[ComplianceStandard, float]:
        """Surveille la conformité pour un tenant"""
        compliance_scores = {}
        
        try:
            # Monitoring RGPD
            gdpr_score = await self._monitor_gdpr_compliance(tenant_id)
            compliance_scores[ComplianceStandard.GDPR] = gdpr_score
            
            # Monitoring SOC2
            soc2_score = await self._monitor_soc2_compliance(tenant_id)
            compliance_scores[ComplianceStandard.SOC2] = soc2_score
            
            # Monitoring ISO27001
            iso_score = await self._monitor_iso27001_compliance(tenant_id)
            compliance_scores[ComplianceStandard.ISO27001] = iso_score
            
        except Exception as e:
            logger.error(f"Error monitoring compliance for tenant {tenant_id}: {e}")
        
        return compliance_scores
    
    async def _monitor_gdpr_compliance(self, tenant_id: str) -> float:
        """Surveille la conformité RGPD"""
        score = 1.0
        
        # Vérification de la rétention des données
        retention_compliance = await self._check_data_retention(tenant_id)
        score *= retention_compliance
        
        # Vérification du consentement
        consent_compliance = await self._check_consent_management(tenant_id)
        score *= consent_compliance
        
        # Vérification des droits des utilisateurs
        rights_compliance = await self._check_user_rights(tenant_id)
        score *= rights_compliance
        
        return score
    
    async def _monitor_soc2_compliance(self, tenant_id: str) -> float:
        """Surveille la conformité SOC2"""
        score = 1.0
        
        # Vérification des contrôles d'accès
        access_compliance = await self._check_access_controls(tenant_id)
        score *= access_compliance
        
        # Vérification de l'audit logging
        audit_compliance = await self._check_audit_logging(tenant_id)
        score *= audit_compliance
        
        return score
    
    async def _monitor_iso27001_compliance(self, tenant_id: str) -> float:
        """Surveille la conformité ISO27001"""
        score = 1.0
        
        # Vérification de la gestion des risques
        risk_compliance = await self._check_risk_management(tenant_id)
        score *= risk_compliance
        
        # Vérification de la sécurité physique
        physical_compliance = await self._check_physical_security(tenant_id)
        score *= physical_compliance
        
        return score
    
    async def _check_data_retention(self, tenant_id: str) -> float:
        """Vérifie la conformité de rétention des données"""
        # Implémentation de vérification
        return 1.0
    
    async def _check_consent_management(self, tenant_id: str) -> float:
        """Vérifie la gestion du consentement"""
        # Implémentation de vérification
        return 1.0
    
    async def _check_user_rights(self, tenant_id: str) -> float:
        """Vérifie le respect des droits utilisateurs"""
        # Implémentation de vérification
        return 1.0
    
    async def _check_access_controls(self, tenant_id: str) -> float:
        """Vérifie les contrôles d'accès"""
        # Implémentation de vérification
        return 1.0
    
    async def _check_audit_logging(self, tenant_id: str) -> float:
        """Vérifie l'audit logging"""
        # Implémentation de vérification
        return 1.0
    
    async def _check_risk_management(self, tenant_id: str) -> float:
        """Vérifie la gestion des risques"""
        # Implémentation de vérification
        return 1.0
    
    async def _check_physical_security(self, tenant_id: str) -> float:
        """Vérifie la sécurité physique"""
        # Implémentation de vérification
        return 1.0
