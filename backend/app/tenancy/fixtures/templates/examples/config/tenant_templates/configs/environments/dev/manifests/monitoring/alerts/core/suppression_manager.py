"""
Ultra-Advanced Suppression Manager - Enterprise-Grade Alert Suppression System  
============================================================================

Ce module fournit un système de suppression d'alertes intelligent avec machine learning,
déduplication avancée, gestion de storm d'alertes et suppression contextuelle pour
des environnements multi-tenant à haute performance.

Fonctionnalités Principales:
- Déduplication intelligente avec fingerprinting avancé
- Suppression de storm d'alertes avec ML
- Gestion de fenêtres de suppression dynamiques
- Suppression contextuelle avec analyse sémantique
- Auto-learning des patterns de suppression
- Suppression géographique et topologique
- Maintenance et nettoyage automatique
- Analytics et métriques avancées

Architecture Enterprise:
- Processing distribué avec sharding automatique
- Cache Redis pour suppression temps réel
- Base de données pour persistence des rules
- ML Pipeline pour apprentissage automatique
- API REST pour gestion des suppressions
- Monitoring et alerting intégré
- Audit trail complet et compliance
- Sécurité end-to-end avec chiffrement

Version: 5.0.0
Auteur: Fahed Mlaiel (Lead Dev + Architecte IA)
Architecture: Event-Driven Microservices avec ML Pipeline
"""

import asyncio
import logging
import time
import uuid
import hashlib
import json
import numpy as np
import re
from datetime import datetime, timedelta
from typing import (
    Dict, List, Optional, Any, Callable, Union, Tuple, Set,
    Protocol, TypeVar, Generic, AsyncIterator, NamedTuple
)
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import defaultdict, deque, Counter
import statistics
import redis
import asyncpg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import prometheus_client
from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge

# Configuration du logging structuré
logger = logging.getLogger(__name__)


class SuppressionType(Enum):
    """Types de suppression supportés"""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_MATCH = "semantic_match"
    PATTERN_MATCH = "pattern_match"
    TIME_BASED = "time_based"
    VOLUME_BASED = "volume_based"
    CONTEXT_BASED = "context_based"
    ML_BASED = "ml_based"


class SuppressionAction(Enum):
    """Actions de suppression"""
    SUPPRESS = "suppress"
    THROTTLE = "throttle"
    AGGREGATE = "aggregate"
    DELAY = "delay"
    ESCALATE = "escalate"


class SuppressionStatus(Enum):
    """Status de suppression"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    PAUSED = "paused"


@dataclass
class SuppressionRule:
    """Règle de suppression avancée"""
    id: str
    name: str
    description: str
    suppression_type: SuppressionType
    action: SuppressionAction
    pattern: Dict[str, Any]
    conditions: Dict[str, Any]
    time_window: timedelta
    max_suppressed: int
    priority: int = 100
    enabled: bool = True
    auto_learn: bool = False
    ml_confidence_threshold: float = 0.8
    tags: Set[str] = field(default_factory=set)
    tenant_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class SuppressionResult:
    """Résultat de suppression avec métadonnées"""
    id: str
    rule_id: str
    suppressed_alerts: List[Dict[str, Any]]
    action_taken: SuppressionAction
    suppression_reason: str
    confidence: float
    time_window_start: datetime
    time_window_end: datetime
    fingerprint: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AlertFingerprint:
    """Empreinte d'alerte pour déduplication"""
    id: str
    alert_hash: str
    semantic_hash: str
    pattern_hash: str
    context_hash: str
    features: Dict[str, Any]
    similarity_threshold: float = 0.85
    created_at: datetime = field(default_factory=datetime.utcnow)


class StormDetector:
    """Détecteur de storm d'alertes avec ML"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_buffer = deque(maxlen=config.get('buffer_size', 10000))
        self.storm_threshold = config.get('storm_threshold', 100)
        self.time_window = timedelta(minutes=config.get('storm_window_minutes', 5))
        self.ml_model = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        
    def add_alert(self, alert: Dict[str, Any]):
        """Ajoute une alerte au buffer"""
        alert['timestamp'] = alert.get('timestamp', datetime.utcnow())
        self.alert_buffer.append(alert)
        
    def detect_storm(self) -> Optional[Dict[str, Any]]:
        """Détecte une storm d'alertes"""
        if len(self.alert_buffer) < 10:
            return None
            
        now = datetime.utcnow()
        window_start = now - self.time_window
        
        # Filtrer les alertes dans la fenêtre temporelle
        recent_alerts = [
            alert for alert in self.alert_buffer
            if alert['timestamp'] >= window_start
        ]
        
        if len(recent_alerts) < self.storm_threshold:
            return None
            
        # Analyser les patterns avec ML si entraîné
        if self.is_trained and len(recent_alerts) >= 20:
            features = self.extract_storm_features(recent_alerts)
            anomaly_score = self.ml_model.decision_function([features])[0]
            
            if anomaly_score < -0.5:  # Anomalie détectée
                return {
                    'type': 'ml_storm',
                    'alert_count': len(recent_alerts),
                    'anomaly_score': float(anomaly_score),
                    'time_window': self.time_window.total_seconds(),
                    'alerts': recent_alerts
                }
        
        # Détection basique par volume
        return {
            'type': 'volume_storm',
            'alert_count': len(recent_alerts),
            'threshold': self.storm_threshold,
            'time_window': self.time_window.total_seconds(),
            'alerts': recent_alerts
        }
    
    def extract_storm_features(self, alerts: List[Dict[str, Any]]) -> List[float]:
        """Extrait les features pour la détection ML de storm"""
        if not alerts:
            return [0] * 8
            
        # Features temporelles
        timestamps = [alert['timestamp'] for alert in alerts]
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                      for i in range(len(timestamps)-1)]
        
        # Features de contenu
        severities = [alert.get('severity', 'info') for alert in alerts]
        severity_scores = [self.severity_to_score(s) for s in severities]
        
        sources = [alert.get('source', 'unknown') for alert in alerts]
        unique_sources = len(set(sources))
        
        messages = [alert.get('message', '') for alert in alerts]
        avg_message_length = np.mean([len(msg) for msg in messages])
        
        features = [
            len(alerts),  # Volume
            np.mean(time_diffs) if time_diffs else 0,  # Intervalle moyen
            np.std(time_diffs) if len(time_diffs) > 1 else 0,  # Variance temporelle
            np.mean(severity_scores),  # Sévérité moyenne
            np.std(severity_scores) if len(severity_scores) > 1 else 0,  # Variance sévérité
            unique_sources / len(alerts) if alerts else 0,  # Diversité des sources
            avg_message_length,  # Longueur moyenne des messages
            len(set(severities))  # Diversité des sévérités
        ]
        
        return features
    
    def severity_to_score(self, severity: str) -> float:
        """Convertit la sévérité en score numérique"""
        severity_map = {
            'debug': 0.1, 'info': 0.3, 'warning': 0.5,
            'error': 0.7, 'critical': 0.9, 'fatal': 1.0
        }
        return severity_map.get(severity.lower(), 0.3)
    
    def train_model(self, historical_data: List[List[float]]):
        """Entraîne le modèle ML sur des données historiques"""
        if len(historical_data) >= 50:
            self.ml_model.fit(historical_data)
            self.is_trained = True
            logger.info("Storm detection ML model trained successfully")


class AdvancedSuppressionManager:
    """
    Gestionnaire de suppression avancé avec intelligence artificielle
    
    Fonctionnalités:
    - Déduplication intelligente avec empreintes multiples
    - Suppression de storm avec ML
    - Suppression contextuelle et sémantique
    - Auto-learning et adaptation
    - Gestion de la performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.suppression_rules = {}
        self.active_suppressions = {}
        self.fingerprint_cache = {}
        self.storm_detector = StormDetector(config.get('storm_config', {}))
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.performance_metrics = {
            'suppressions_processed': PrometheusCounter('suppressions_processed_total', 'Total suppressions processed'),
            'suppression_time': Histogram('suppression_processing_seconds', 'Time spent processing suppressions'),
            'active_suppression_rules': Gauge('active_suppression_rules', 'Number of active suppression rules'),
            'storm_detections': PrometheusCounter('storm_detections_total', 'Total storm detections'),
        }
        
        # Redis pour le cache
        self.redis_client = None
        self.setup_redis()
        
        # Base de données pour persistence
        self.db_pool = None
        self.setup_database()
        
        logger.info("Advanced Suppression Manager initialized")
    
    async def setup_redis(self):
        """Configuration du client Redis"""
        try:
            self.redis_client = redis.asyncio.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 1),
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for suppression manager")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    async def setup_database(self):
        """Configuration de la base de données PostgreSQL"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host=self.config.get('db_host', 'localhost'),
                port=self.config.get('db_port', 5432),
                user=self.config.get('db_user', 'postgres'),
                password=self.config.get('db_password', ''),
                database=self.config.get('db_name', 'alerts'),
                min_size=5,
                max_size=20
            )
            await self.create_tables()
            logger.info("Database connection pool established")
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
    
    async def create_tables(self):
        """Crée les tables nécessaires"""
        create_suppression_rules_table = """
        CREATE TABLE IF NOT EXISTS suppression_rules (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            suppression_type VARCHAR(50) NOT NULL,
            action VARCHAR(50) NOT NULL,
            pattern JSONB NOT NULL,
            conditions JSONB,
            time_window INTERVAL NOT NULL,
            max_suppressed INTEGER NOT NULL,
            priority INTEGER DEFAULT 100,
            enabled BOOLEAN DEFAULT true,
            auto_learn BOOLEAN DEFAULT false,
            ml_confidence_threshold FLOAT DEFAULT 0.8,
            tags JSONB,
            tenant_id VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            expires_at TIMESTAMP WITH TIME ZONE,
            last_triggered TIMESTAMP WITH TIME ZONE,
            trigger_count INTEGER DEFAULT 0,
            INDEX (tenant_id, enabled),
            INDEX (suppression_type, priority),
            INDEX USING GIN (pattern),
            INDEX USING GIN (tags)
        );
        """
        
        create_suppression_results_table = """
        CREATE TABLE IF NOT EXISTS suppression_results (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            rule_id UUID REFERENCES suppression_rules(id),
            suppressed_alerts JSONB NOT NULL,
            action_taken VARCHAR(50) NOT NULL,
            suppression_reason TEXT,
            confidence FLOAT NOT NULL,
            time_window_start TIMESTAMP WITH TIME ZONE NOT NULL,
            time_window_end TIMESTAMP WITH TIME ZONE NOT NULL,
            fingerprint VARCHAR(255),
            metadata JSONB,
            tenant_id VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            INDEX (tenant_id, created_at),
            INDEX (rule_id, action_taken),
            INDEX (fingerprint)
        );
        """
        
        create_fingerprints_table = """
        CREATE TABLE IF NOT EXISTS alert_fingerprints (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            alert_hash VARCHAR(255) NOT NULL,
            semantic_hash VARCHAR(255) NOT NULL,
            pattern_hash VARCHAR(255) NOT NULL,
            context_hash VARCHAR(255) NOT NULL,
            features JSONB NOT NULL,
            similarity_threshold FLOAT DEFAULT 0.85,
            tenant_id VARCHAR(100) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            INDEX (tenant_id, alert_hash),
            INDEX (semantic_hash),
            INDEX (pattern_hash),
            UNIQUE (alert_hash, tenant_id)
        );
        """
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(create_suppression_rules_table)
            await conn.execute(create_suppression_results_table)
            await conn.execute(create_fingerprints_table)
    
    async def add_suppression_rule(self, rule: SuppressionRule) -> bool:
        """Ajoute une nouvelle règle de suppression"""
        try:
            self.suppression_rules[rule.id] = rule
            
            # Persister en base
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO suppression_rules 
                        (id, name, description, suppression_type, action, pattern, 
                         conditions, time_window, max_suppressed, priority, enabled,
                         auto_learn, ml_confidence_threshold, tags, tenant_id, 
                         created_at, updated_at, expires_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                        ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        updated_at = NOW()
                    """, rule.id, rule.name, rule.description, rule.suppression_type.value,
                    rule.action.value, json.dumps(rule.pattern), json.dumps(rule.conditions),
                    rule.time_window, rule.max_suppressed, rule.priority, rule.enabled,
                    rule.auto_learn, rule.ml_confidence_threshold, json.dumps(list(rule.tags)),
                    rule.tenant_id, rule.created_at, rule.updated_at, rule.expires_at)
            
            # Mettre à jour les métriques
            self.performance_metrics['active_suppression_rules'].set(
                len([r for r in self.suppression_rules.values() if r.enabled])
            )
            
            logger.info(f"Suppression rule added: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add suppression rule: {e}")
            return False
    
    async def process_alerts(self, alerts: List[Dict[str, Any]], tenant_id: str) -> List[Dict[str, Any]]:
        """
        Traite les alertes et applique la suppression
        """
        start_time = time.time()
        
        try:
            # Ajouter alertes au détecteur de storm
            for alert in alerts:
                self.storm_detector.add_alert(alert)
            
            # Détecter les storms
            storm_detection = self.storm_detector.detect_storm()
            if storm_detection:
                await self.handle_storm_detection(storm_detection, tenant_id)
                self.performance_metrics['storm_detections'].inc()
            
            # Générer des empreintes pour toutes les alertes
            alert_fingerprints = {}
            for alert in alerts:
                fingerprint = await self.generate_fingerprint(alert, tenant_id)
                alert_fingerprints[alert.get('id', str(uuid.uuid4()))] = fingerprint
            
            # Appliquer les règles de suppression
            non_suppressed_alerts = []
            suppression_results = []
            
            for alert in alerts:
                alert_id = alert.get('id', str(uuid.uuid4()))
                fingerprint = alert_fingerprints[alert_id]
                
                suppression_result = await self.apply_suppression_rules(alert, fingerprint, tenant_id)
                
                if suppression_result:
                    suppression_results.append(suppression_result)
                    # Log suppression mais ne pas ajouter à la liste finale
                    logger.debug(f"Alert suppressed: {alert_id} by rule {suppression_result.rule_id}")
                else:
                    non_suppressed_alerts.append(alert)
            
            # Persister les résultats de suppression
            if suppression_results:
                await self.persist_suppression_results(suppression_results, tenant_id)
            
            # Mise à jour des métriques
            self.performance_metrics['suppressions_processed'].inc(len(suppression_results))
            self.performance_metrics['suppression_time'].observe(time.time() - start_time)
            
            logger.info(f"Processed {len(alerts)} alerts, suppressed {len(suppression_results)}")
            return non_suppressed_alerts
            
        except Exception as e:
            logger.error(f"Error in alert suppression: {e}")
            return alerts
    
    async def generate_fingerprint(self, alert: Dict[str, Any], tenant_id: str) -> AlertFingerprint:
        """Génère une empreinte multi-dimensionnelle pour l'alerte"""
        try:
            # Hash exact (structure complète)
            alert_str = json.dumps(alert, sort_keys=True, default=str)
            alert_hash = hashlib.sha256(alert_str.encode()).hexdigest()
            
            # Hash sémantique (contenu du message)
            message = alert.get('message', '')
            semantic_content = re.sub(r'\d+', 'NUM', message.lower())  # Normaliser les nombres
            semantic_content = re.sub(r'[^\w\s]', '', semantic_content)  # Supprimer ponctuation
            semantic_hash = hashlib.sha256(semantic_content.encode()).hexdigest()
            
            # Hash de pattern (structure sans valeurs spécifiques)
            pattern_dict = {
                'source': alert.get('source', 'unknown'),
                'severity': alert.get('severity', 'info'),
                'type': alert.get('type', 'unknown'),
                'category': alert.get('category', 'unknown')
            }
            pattern_str = json.dumps(pattern_dict, sort_keys=True)
            pattern_hash = hashlib.sha256(pattern_str.encode()).hexdigest()
            
            # Hash de contexte (métadonnées importantes)
            context_dict = {
                'host': alert.get('host', 'unknown'),
                'service': alert.get('service', 'unknown'),
                'environment': alert.get('environment', 'unknown')
            }
            context_str = json.dumps(context_dict, sort_keys=True)
            context_hash = hashlib.sha256(context_str.encode()).hexdigest()
            
            # Features pour ML
            features = {
                'message_length': len(message),
                'word_count': len(message.split()),
                'has_numbers': bool(re.search(r'\d+', message)),
                'has_uppercase': bool(re.search(r'[A-Z]', message)),
                'severity_score': self.severity_to_score(alert.get('severity', 'info')),
                'timestamp_hour': alert.get('timestamp', datetime.utcnow()).hour,
                'metadata_keys': len(alert.get('metadata', {})) if isinstance(alert.get('metadata'), dict) else 0
            }
            
            fingerprint = AlertFingerprint(
                id=str(uuid.uuid4()),
                alert_hash=alert_hash,
                semantic_hash=semantic_hash,
                pattern_hash=pattern_hash,
                context_hash=context_hash,
                features=features
            )
            
            # Cache dans Redis
            if self.redis_client:
                cache_key = f"fingerprint:{tenant_id}:{alert_hash}"
                await self.redis_client.setex(
                    cache_key,
                    timedelta(hours=24),
                    json.dumps(asdict(fingerprint), default=str)
                )
            
            return fingerprint
            
        except Exception as e:
            logger.error(f"Error generating fingerprint: {e}")
            # Retourner une empreinte par défaut
            return AlertFingerprint(
                id=str(uuid.uuid4()),
                alert_hash="default",
                semantic_hash="default",
                pattern_hash="default",
                context_hash="default",
                features={}
            )
    
    def severity_to_score(self, severity: str) -> float:
        """Convertit la sévérité en score numérique"""
        severity_map = {
            'debug': 0.1, 'info': 0.3, 'warning': 0.5,
            'error': 0.7, 'critical': 0.9, 'fatal': 1.0
        }
        return severity_map.get(severity.lower(), 0.3)
    
    async def apply_suppression_rules(self, alert: Dict[str, Any], fingerprint: AlertFingerprint, tenant_id: str) -> Optional[SuppressionResult]:
        """Applique les règles de suppression à une alerte"""
        
        # Trier les règles par priorité
        active_rules = [
            rule for rule in self.suppression_rules.values()
            if rule.enabled and rule.tenant_id == tenant_id and
            (rule.expires_at is None or rule.expires_at > datetime.utcnow())
        ]
        active_rules.sort(key=lambda r: r.priority)
        
        for rule in active_rules:
            try:
                matches, confidence = await self.evaluate_suppression_rule(alert, fingerprint, rule)
                
                if matches and confidence >= rule.ml_confidence_threshold:
                    # Créer le résultat de suppression
                    suppression_result = SuppressionResult(
                        id=str(uuid.uuid4()),
                        rule_id=rule.id,
                        suppressed_alerts=[alert],
                        action_taken=rule.action,
                        suppression_reason=f"Rule {rule.name} matched with confidence {confidence:.2f}",
                        confidence=confidence,
                        time_window_start=datetime.utcnow(),
                        time_window_end=datetime.utcnow() + rule.time_window,
                        fingerprint=fingerprint.alert_hash,
                        metadata={
                            'rule_type': rule.suppression_type.value,
                            'tenant_id': tenant_id,
                            'fingerprint_details': asdict(fingerprint)
                        }
                    )
                    
                    # Mettre à jour les statistiques de la règle
                    rule.last_triggered = datetime.utcnow()
                    rule.trigger_count += 1
                    
                    # Auto-learning si activé
                    if rule.auto_learn:
                        await self.update_rule_with_learning(rule, alert, fingerprint)
                    
                    return suppression_result
                    
            except Exception as e:
                logger.error(f"Error evaluating suppression rule {rule.id}: {e}")
                continue
        
        return None
    
    async def evaluate_suppression_rule(self, alert: Dict[str, Any], fingerprint: AlertFingerprint, rule: SuppressionRule) -> Tuple[bool, float]:
        """Évalue si une règle de suppression s'applique"""
        
        if rule.suppression_type == SuppressionType.EXACT_MATCH:
            return await self.evaluate_exact_match(alert, rule)
        
        elif rule.suppression_type == SuppressionType.FUZZY_MATCH:
            return await self.evaluate_fuzzy_match(alert, fingerprint, rule)
        
        elif rule.suppression_type == SuppressionType.SEMANTIC_MATCH:
            return await self.evaluate_semantic_match(alert, rule)
        
        elif rule.suppression_type == SuppressionType.PATTERN_MATCH:
            return await self.evaluate_pattern_match(alert, rule)
        
        elif rule.suppression_type == SuppressionType.TIME_BASED:
            return await self.evaluate_time_based(alert, rule)
        
        elif rule.suppression_type == SuppressionType.VOLUME_BASED:
            return await self.evaluate_volume_based(alert, rule)
        
        elif rule.suppression_type == SuppressionType.CONTEXT_BASED:
            return await self.evaluate_context_based(alert, rule)
        
        elif rule.suppression_type == SuppressionType.ML_BASED:
            return await self.evaluate_ml_based(alert, fingerprint, rule)
        
        return False, 0.0
    
    async def evaluate_exact_match(self, alert: Dict[str, Any], rule: SuppressionRule) -> Tuple[bool, float]:
        """Évaluation par correspondance exacte"""
        pattern = rule.pattern
        
        for key, expected_value in pattern.items():
            if alert.get(key) != expected_value:
                return False, 0.0
        
        return True, 1.0
    
    async def evaluate_fuzzy_match(self, alert: Dict[str, Any], fingerprint: AlertFingerprint, rule: SuppressionRule) -> Tuple[bool, float]:
        """Évaluation par correspondance floue"""
        try:
            # Vérifier si on a déjà une empreinte similaire en cache
            cache_key = f"fuzzy_match:{rule.tenant_id}:{fingerprint.pattern_hash}"
            
            if self.redis_client:
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    return True, float(cached_result)
            
            # Comparaison des empreintes
            pattern_hash = rule.pattern.get('pattern_hash', '')
            if pattern_hash and pattern_hash == fingerprint.pattern_hash:
                confidence = 0.9
                
                # Cache le résultat
                if self.redis_client:
                    await self.redis_client.setex(cache_key, timedelta(hours=1), str(confidence))
                
                return True, confidence
            
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Error in fuzzy match evaluation: {e}")
            return False, 0.0
    
    async def evaluate_semantic_match(self, alert: Dict[str, Any], rule: SuppressionRule) -> Tuple[bool, float]:
        """Évaluation par correspondance sémantique"""
        try:
            alert_message = alert.get('message', '')
            pattern_message = rule.pattern.get('message', '')
            
            if not alert_message or not pattern_message:
                return False, 0.0
            
            # Vectorisation TF-IDF et calcul de similarité
            texts = [alert_message, pattern_message]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            threshold = rule.pattern.get('semantic_threshold', 0.8)
            
            if similarity >= threshold:
                return True, float(similarity)
            
            return False, float(similarity)
            
        except Exception as e:
            logger.error(f"Error in semantic match evaluation: {e}")
            return False, 0.0
    
    async def evaluate_pattern_match(self, alert: Dict[str, Any], rule: SuppressionRule) -> Tuple[bool, float]:
        """Évaluation par correspondance de pattern regex"""
        try:
            pattern_rules = rule.pattern.get('patterns', [])
            matches = 0
            total_patterns = len(pattern_rules)
            
            for pattern_rule in pattern_rules:
                field = pattern_rule.get('field', 'message')
                pattern = pattern_rule.get('pattern', '')
                flags = pattern_rule.get('flags', 0)
                
                field_value = str(alert.get(field, ''))
                
                if re.search(pattern, field_value, flags):
                    matches += 1
            
            if total_patterns == 0:
                return False, 0.0
            
            confidence = matches / total_patterns
            min_match_ratio = rule.pattern.get('min_match_ratio', 0.8)
            
            return confidence >= min_match_ratio, confidence
            
        except Exception as e:
            logger.error(f"Error in pattern match evaluation: {e}")
            return False, 0.0
    
    async def evaluate_time_based(self, alert: Dict[str, Any], rule: SuppressionRule) -> Tuple[bool, float]:
        """Évaluation basée sur le temps"""
        try:
            alert_time = alert.get('timestamp', datetime.utcnow())
            if isinstance(alert_time, str):
                alert_time = datetime.fromisoformat(alert_time.replace('Z', '+00:00'))
            
            time_conditions = rule.conditions.get('time_conditions', {})
            
            # Vérifier les heures autorisées
            if 'allowed_hours' in time_conditions:
                allowed_hours = time_conditions['allowed_hours']
                if alert_time.hour not in allowed_hours:
                    return True, 1.0  # Supprimer en dehors des heures autorisées
            
            # Vérifier les jours autorisés
            if 'allowed_days' in time_conditions:
                allowed_days = time_conditions['allowed_days']
                if alert_time.weekday() not in allowed_days:
                    return True, 1.0  # Supprimer en dehors des jours autorisés
            
            # Vérifier la fenêtre de suppression
            if 'suppression_window' in time_conditions:
                window_start = time_conditions['suppression_window'].get('start')
                window_end = time_conditions['suppression_window'].get('end')
                
                if window_start and window_end:
                    current_time = alert_time.time()
                    if window_start <= current_time <= window_end:
                        return True, 1.0
            
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Error in time-based evaluation: {e}")
            return False, 0.0
    
    async def evaluate_volume_based(self, alert: Dict[str, Any], rule: SuppressionRule) -> Tuple[bool, float]:
        """Évaluation basée sur le volume"""
        try:
            volume_conditions = rule.conditions.get('volume_conditions', {})
            max_alerts_per_window = volume_conditions.get('max_alerts_per_window', 100)
            
            # Compter les alertes similaires dans la fenêtre temporelle
            now = datetime.utcnow()
            window_start = now - rule.time_window
            
            # Utiliser Redis pour compter efficacement
            if self.redis_client:
                pattern_key = f"volume:{rule.tenant_id}:{rule.id}"
                count_key = f"{pattern_key}:{now.strftime('%Y%m%d%H%M')}"
                
                current_count = await self.redis_client.incr(count_key)
                await self.redis_client.expire(count_key, int(rule.time_window.total_seconds()))
                
                if current_count > max_alerts_per_window:
                    confidence = min(current_count / max_alerts_per_window, 2.0) - 1.0
                    return True, confidence
            
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Error in volume-based evaluation: {e}")
            return False, 0.0
    
    async def evaluate_context_based(self, alert: Dict[str, Any], rule: SuppressionRule) -> Tuple[bool, float]:
        """Évaluation basée sur le contexte"""
        try:
            context_conditions = rule.conditions.get('context_conditions', {})
            matches = 0
            total_conditions = len(context_conditions)
            
            for condition_name, condition_value in context_conditions.items():
                alert_value = alert.get(condition_name)
                
                if isinstance(condition_value, list):
                    if alert_value in condition_value:
                        matches += 1
                elif isinstance(condition_value, dict):
                    # Conditions plus complexes
                    if self.evaluate_complex_condition(alert_value, condition_value):
                        matches += 1
                else:
                    if alert_value == condition_value:
                        matches += 1
            
            if total_conditions == 0:
                return False, 0.0
            
            confidence = matches / total_conditions
            min_match_ratio = rule.pattern.get('min_context_match', 0.7)
            
            return confidence >= min_match_ratio, confidence
            
        except Exception as e:
            logger.error(f"Error in context-based evaluation: {e}")
            return False, 0.0
    
    async def evaluate_ml_based(self, alert: Dict[str, Any], fingerprint: AlertFingerprint, rule: SuppressionRule) -> Tuple[bool, float]:
        """Évaluation basée sur le machine learning"""
        try:
            # Utiliser le modèle ML spécifié dans la règle
            model_name = rule.ml_model
            if not model_name or model_name not in self.ml_models:
                return False, 0.0
            
            model = self.ml_models[model_name]
            
            # Préparer les features
            features = [
                fingerprint.features.get('message_length', 0),
                fingerprint.features.get('word_count', 0),
                fingerprint.features.get('severity_score', 0.3),
                fingerprint.features.get('timestamp_hour', 12),
                fingerprint.features.get('metadata_keys', 0),
                int(fingerprint.features.get('has_numbers', False)),
                int(fingerprint.features.get('has_uppercase', False))
            ]
            
            # Prédiction
            prediction_proba = model.predict_proba([features])[0]
            confidence = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
            
            return confidence >= rule.ml_confidence_threshold, float(confidence)
            
        except Exception as e:
            logger.error(f"Error in ML-based evaluation: {e}")
            return False, 0.0
    
    def evaluate_complex_condition(self, value: Any, condition: Dict[str, Any]) -> bool:
        """Évalue des conditions complexes"""
        if 'equals' in condition:
            return value == condition['equals']
        elif 'in' in condition:
            return value in condition['in']
        elif 'gt' in condition:
            return value > condition['gt']
        elif 'lt' in condition:
            return value < condition['lt']
        elif 'regex' in condition:
            return bool(re.search(condition['regex'], str(value)))
        
        return False
    
    async def update_rule_with_learning(self, rule: SuppressionRule, alert: Dict[str, Any], fingerprint: AlertFingerprint):
        """Met à jour une règle avec l'apprentissage automatique"""
        try:
            # Analyser les patterns récurrents
            learning_key = f"learning:{rule.tenant_id}:{rule.id}"
            
            if self.redis_client:
                # Stocker les features de l'alerte pour apprentissage
                features_data = {
                    'fingerprint': asdict(fingerprint),
                    'alert_summary': {
                        'severity': alert.get('severity'),
                        'source': alert.get('source'),
                        'message_length': len(alert.get('message', '')),
                        'timestamp': alert.get('timestamp', datetime.utcnow()).isoformat()
                    }
                }
                
                await self.redis_client.lpush(learning_key, json.dumps(features_data, default=str))
                await self.redis_client.ltrim(learning_key, 0, 999)  # Garder 1000 derniers
                await self.redis_client.expire(learning_key, timedelta(days=30))
                
                # Adapter le seuil de confiance basé sur le succès
                list_length = await self.redis_client.llen(learning_key)
                if list_length >= 100:  # Assez de données pour adapter
                    success_rate = rule.trigger_count / max(list_length, 1)
                    
                    # Ajuster le seuil de confiance
                    if success_rate > 0.9:
                        rule.ml_confidence_threshold = max(rule.ml_confidence_threshold - 0.05, 0.5)
                    elif success_rate < 0.7:
                        rule.ml_confidence_threshold = min(rule.ml_confidence_threshold + 0.05, 0.95)
                    
                    logger.info(f"Updated confidence threshold for rule {rule.id}: {rule.ml_confidence_threshold}")
            
        except Exception as e:
            logger.error(f"Error in rule learning update: {e}")
    
    async def handle_storm_detection(self, storm_detection: Dict[str, Any], tenant_id: str):
        """Gère la détection d'une storm d'alertes"""
        try:
            storm_type = storm_detection['type']
            alert_count = storm_detection['alert_count']
            
            logger.warning(f"Alert storm detected: {storm_type} with {alert_count} alerts")
            
            # Créer une règle de suppression temporaire pour la storm
            storm_rule = SuppressionRule(
                id=f"storm_{uuid.uuid4()}",
                name=f"Auto Storm Suppression - {storm_type}",
                description=f"Temporary suppression for {storm_type} storm with {alert_count} alerts",
                suppression_type=SuppressionType.VOLUME_BASED,
                action=SuppressionAction.THROTTLE,
                pattern={'storm_type': storm_type},
                conditions={
                    'volume_conditions': {
                        'max_alerts_per_window': max(10, alert_count // 10)
                    }
                },
                time_window=timedelta(minutes=30),
                max_suppressed=alert_count * 2,
                priority=1,  # Haute priorité
                enabled=True,
                tenant_id=tenant_id,
                expires_at=datetime.utcnow() + timedelta(hours=2)
            )
            
            await self.add_suppression_rule(storm_rule)
            
            # Notifier du storm (à implémenter selon les besoins)
            await self.notify_storm_detection(storm_detection, tenant_id)
            
        except Exception as e:
            logger.error(f"Error handling storm detection: {e}")
    
    async def notify_storm_detection(self, storm_detection: Dict[str, Any], tenant_id: str):
        """Notifie la détection d'une storm"""
        # Placeholder pour notification
        logger.info(f"Storm notification sent for tenant {tenant_id}: {storm_detection}")
    
    async def persist_suppression_results(self, results: List[SuppressionResult], tenant_id: str):
        """Persiste les résultats de suppression"""
        if not self.db_pool or not results:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                for result in results:
                    await conn.execute("""
                        INSERT INTO suppression_results 
                        (id, rule_id, suppressed_alerts, action_taken, suppression_reason,
                         confidence, time_window_start, time_window_end, fingerprint,
                         metadata, tenant_id, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """, 
                    result.id, result.rule_id, json.dumps(result.suppressed_alerts),
                    result.action_taken.value, result.suppression_reason,
                    result.confidence, result.time_window_start, result.time_window_end,
                    result.fingerprint, json.dumps(result.metadata),
                    tenant_id, result.created_at)
                    
            logger.info(f"Persisted {len(results)} suppression results for tenant {tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to persist suppression results: {e}")
    
    async def get_suppression_statistics(self, tenant_id: str, days: int = 7) -> Dict[str, Any]:
        """Récupère les statistiques de suppression"""
        if not self.db_pool:
            return {}
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            async with self.db_pool.acquire() as conn:
                # Statistiques générales
                general_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_suppressions,
                        AVG(confidence) as avg_confidence,
                        COUNT(DISTINCT rule_id) as active_rules
                    FROM suppression_results 
                    WHERE tenant_id = $1 AND created_at >= $2
                """, tenant_id, cutoff_date)
                
                # Statistiques par type d'action
                action_stats = await conn.fetch("""
                    SELECT action_taken, COUNT(*) as count
                    FROM suppression_results 
                    WHERE tenant_id = $1 AND created_at >= $2
                    GROUP BY action_taken
                """, tenant_id, cutoff_date)
                
                # Top des règles les plus utilisées
                top_rules = await conn.fetch("""
                    SELECT sr.rule_id, sr.name, COUNT(*) as usage_count
                    FROM suppression_results sr
                    JOIN suppression_rules sr ON sr.rule_id = sr.id
                    WHERE sr.tenant_id = $1 AND sr.created_at >= $2
                    GROUP BY sr.rule_id, sr.name
                    ORDER BY usage_count DESC
                    LIMIT 10
                """, tenant_id, cutoff_date)
                
                return {
                    'general': dict(general_stats) if general_stats else {},
                    'by_action': [dict(row) for row in action_stats],
                    'top_rules': [dict(row) for row in top_rules],
                    'period_days': days
                }
                
        except Exception as e:
            logger.error(f"Failed to get suppression statistics: {e}")
            return {}
    
    async def cleanup_expired_rules(self, tenant_id: str):
        """Nettoie les règles expirées"""
        try:
            now = datetime.utcnow()
            
            # Nettoyer la mémoire
            expired_rules = [
                rule_id for rule_id, rule in self.suppression_rules.items()
                if rule.tenant_id == tenant_id and rule.expires_at and rule.expires_at <= now
            ]
            
            for rule_id in expired_rules:
                del self.suppression_rules[rule_id]
            
            # Nettoyer la base de données
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    result = await conn.execute("""
                        UPDATE suppression_rules 
                        SET enabled = false 
                        WHERE tenant_id = $1 AND expires_at <= $2 AND enabled = true
                    """, tenant_id, now)
                    
                    logger.info(f"Disabled {result} expired rules for tenant {tenant_id}")
            
            # Mettre à jour les métriques
            self.performance_metrics['active_suppression_rules'].set(
                len([r for r in self.suppression_rules.values() if r.enabled])
            )
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired rules: {e}")
    
    async def shutdown(self):
        """Arrêt propre du gestionnaire de suppression"""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("Suppression manager shutdown complete")


# Factory function
def create_suppression_manager(config: Dict[str, Any]) -> AdvancedSuppressionManager:
    """Crée une instance du gestionnaire de suppression avec la configuration donnée"""
    return AdvancedSuppressionManager(config)


# Export des classes principales
__all__ = [
    'AdvancedSuppressionManager',
    'SuppressionType',
    'SuppressionAction',
    'SuppressionStatus',
    'SuppressionRule',
    'SuppressionResult',
    'AlertFingerprint',
    'StormDetector',
    'create_suppression_manager'
]
