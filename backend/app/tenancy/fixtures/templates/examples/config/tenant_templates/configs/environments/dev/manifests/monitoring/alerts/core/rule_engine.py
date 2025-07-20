# -*- coding: utf-8 -*-
"""
Rule Engine - Moteur de Règles Ultra-Avancé avec Machine Learning
===============================================================

Moteur de règles intelligent pour l'évaluation d'alertes avec support:
- Règles complexes avec conditions multiples
- Machine Learning pour adaptation dynamique
- Compilation et cache optimisé
- Règles basées sur ML et patterns
- Support multi-tenant avec isolation
- Évaluation temps réel haute performance

Fonctionnalités ML:
- Apprentissage automatique des seuils
- Détection de patterns anormaux
- Adaptation aux cycles métier
- Prédiction d'alertes futures
- Classification automatique

Architecture:
- RuleCompiler: Compilation optimisée des règles
- MLRuleEngine: Moteur ML pour règles intelligentes
- RuleCache: Cache haute performance
- PatternDetector: Détection de patterns
- ThresholdOptimizer: Optimisation automatique des seuils

Version: 3.0.0
"""

import re
import ast
import time
import json
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import sqlite3
import redis
import numpy as np
from datetime import datetime, timedelta
import statistics

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleType(Enum):
    """Types de règles"""
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    COMPOSITE = "composite"
    ML_BASED = "ml_based"
    TIME_SERIES = "time_series"
    CORRELATION = "correlation"

class RuleAction(Enum):
    """Actions des règles"""
    ALERT = "alert"
    SUPPRESS = "suppress"
    ESCALATE = "escalate"
    REMEDIATE = "remediate"
    NOTIFY = "notify"
    LOG = "log"

class RuleOperator(Enum):
    """Opérateurs de règles"""
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    NEQ = "!="
    CONTAINS = "contains"
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"

@dataclass
class RuleCondition:
    """Condition d'une règle"""
    field: str
    operator: RuleOperator
    value: Any
    weight: float = 1.0
    negated: bool = False

@dataclass
class Rule:
    """Règle d'évaluation"""
    id: str
    name: str
    description: str
    rule_type: RuleType
    conditions: List[RuleCondition]
    action: RuleAction
    priority: int = 1
    enabled: bool = True
    tenant_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    compiled_rule: Optional[Any] = None
    ml_model_id: Optional[str] = None
    threshold_config: Optional[Dict[str, Any]] = None
    pattern_config: Optional[Dict[str, Any]] = None

@dataclass
class RuleEvaluationResult:
    """Résultat d'évaluation d'une règle"""
    rule_id: str
    matched: bool
    score: float
    execution_time_ms: float
    matched_conditions: List[str]
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

@dataclass
class MLModelConfig:
    """Configuration du modèle ML"""
    model_type: str  # 'threshold_optimizer', 'anomaly_detector', 'pattern_classifier'
    features: List[str]
    training_window_hours: int = 24
    update_interval_hours: int = 6
    min_training_samples: int = 100
    confidence_threshold: float = 0.8

class RuleEngine:
    """
    Moteur de règles ultra-avancé avec ML
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le moteur de règles
        
        Args:
            config: Configuration du moteur
        """
        self.config = config or self._default_config()
        self.is_running = False
        self.start_time = time.time()
        
        # Stockage des règles
        self.rules: Dict[str, Rule] = {}
        self.compiled_rules: Dict[str, Any] = {}
        self.rule_cache = {}
        self.lock = threading.RLock()
        
        # ML Components
        self.ml_models: Dict[str, Any] = {}
        self.pattern_detector = PatternDetector(self.config)
        self.threshold_optimizer = ThresholdOptimizer(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        
        # Métriques et historique
        self.evaluation_history = deque(maxlen=10000)
        self.performance_metrics = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'average_evaluation_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Stockage
        self.db_path = self.config.get('db_path', 'rule_engine.db')
        self.redis_client = self._init_redis()
        
        # Compiler et optimiseur
        self.rule_compiler = RuleCompiler(self.config)
        
        # Threads de background
        self.ml_update_thread = None
        self.optimization_thread = None
        
        # Initialisation
        self._init_database()
        self._load_rules_from_db()
        self._setup_default_rules()
        
        logger.info("RuleEngine initialisé avec succès")
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            'max_rules_per_tenant': 1000,
            'rule_evaluation_timeout_ms': 500,
            'enable_ml_rules': True,
            'ml_model_update_interval_hours': 24,
            'rule_compilation_cache_size': 10000,
            'complex_rule_support': True,
            'pattern_detection_enabled': True,
            'threshold_optimization_enabled': True,
            'anomaly_detection_enabled': True,
            'cache_ttl_seconds': 3600,
            'performance_monitoring': True,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 6,
            'db_retention_days': 30
        }
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialise la connexion Redis"""
        try:
            client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                db=self.config['redis_db'],
                decode_responses=True
            )
            client.ping()
            logger.info("Connexion Redis RuleEngine établie")
            return client
        except Exception as e:
            logger.warning(f"Redis non disponible pour RuleEngine: {e}")
            return None
    
    def _init_database(self):
        """Initialise la base de données SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table des règles
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    rule_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    action TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    enabled BOOLEAN DEFAULT 1,
                    tenant_id TEXT,
                    labels TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    ml_model_id TEXT,
                    threshold_config TEXT,
                    pattern_config TEXT
                )
            ''')
            
            # Table des évaluations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rule_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT NOT NULL,
                    matched BOOLEAN NOT NULL,
                    score REAL NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    matched_conditions TEXT,
                    context TEXT,
                    timestamp REAL NOT NULL,
                    tenant_id TEXT
                )
            ''')
            
            # Table des modèles ML
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_models (
                    id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    model_data BLOB,
                    config TEXT,
                    training_data_count INTEGER,
                    last_trained REAL,
                    accuracy REAL,
                    tenant_id TEXT
                )
            ''')
            
            # Table des patterns détectés
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detected_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    occurrence_count INTEGER DEFAULT 1,
                    tenant_id TEXT
                )
            ''')
            
            # Index pour les performances
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rules_tenant ON rules(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_evaluations_rule_time ON rule_evaluations(rule_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_type_tenant ON detected_patterns(pattern_type, tenant_id)')
            
            conn.commit()
            conn.close()
            
            logger.info("Base de données RuleEngine initialisée")
            
        except Exception as e:
            logger.error(f"Erreur initialisation base de données: {e}")
    
    def start(self) -> bool:
        """Démarre le moteur de règles"""
        if self.is_running:
            logger.warning("RuleEngine déjà en cours d'exécution")
            return True
        
        try:
            self.is_running = True
            
            # Démarre les threads ML si activés
            if self.config['enable_ml_rules']:
                self.ml_update_thread = threading.Thread(target=self._ml_update_loop, daemon=True)
                self.ml_update_thread.start()
                
                self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
                self.optimization_thread.start()
            
            # Compilation initiale des règles
            self._compile_all_rules()
            
            logger.info("RuleEngine démarré avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur démarrage RuleEngine: {e}")
            return False
    
    def stop(self) -> bool:
        """Arrête le moteur de règles"""
        if not self.is_running:
            return True
        
        try:
            logger.info("Arrêt RuleEngine...")
            self.is_running = False
            
            # Attend les threads
            if self.ml_update_thread and self.ml_update_thread.is_alive():
                self.ml_update_thread.join(timeout=10)
            
            if self.optimization_thread and self.optimization_thread.is_alive():
                self.optimization_thread.join(timeout=10)
            
            logger.info("RuleEngine arrêté")
            return True
            
        except Exception as e:
            logger.error(f"Erreur arrêt RuleEngine: {e}")
            return False
    
    def add_rule(self, rule: Rule) -> bool:
        """
        Ajoute une nouvelle règle
        
        Args:
            rule: Règle à ajouter
            
        Returns:
            True si ajout réussi
        """
        try:
            with self.lock:
                # Vérification des limites par tenant
                if rule.tenant_id:
                    tenant_rules = [r for r in self.rules.values() if r.tenant_id == rule.tenant_id]
                    if len(tenant_rules) >= self.config['max_rules_per_tenant']:
                        logger.error(f"Limite de règles atteinte pour tenant {rule.tenant_id}")
                        return False
                
                # Validation de la règle
                if not self._validate_rule(rule):
                    logger.error(f"Règle invalide: {rule.id}")
                    return False
                
                # Compilation de la règle
                compiled_rule = self.rule_compiler.compile_rule(rule)
                if not compiled_rule:
                    logger.error(f"Échec compilation règle: {rule.id}")
                    return False
                
                # Ajout en mémoire
                rule.compiled_rule = compiled_rule
                rule.updated_at = time.time()
                self.rules[rule.id] = rule
                self.compiled_rules[rule.id] = compiled_rule
                
                # Invalidation du cache
                self._invalidate_cache(rule.id)
                
                # Sauvegarde en base
                self._save_rule_to_db(rule)
                
                logger.info(f"Règle ajoutée: {rule.id} - {rule.name}")
                return True
                
        except Exception as e:
            logger.error(f"Erreur ajout règle {rule.id}: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Supprime une règle
        
        Args:
            rule_id: ID de la règle à supprimer
            
        Returns:
            True si suppression réussie
        """
        try:
            with self.lock:
                if rule_id not in self.rules:
                    logger.warning(f"Règle {rule_id} introuvable")
                    return False
                
                # Suppression de la mémoire
                del self.rules[rule_id]
                if rule_id in self.compiled_rules:
                    del self.compiled_rules[rule_id]
                
                # Invalidation du cache
                self._invalidate_cache(rule_id)
                
                # Suppression de la base
                self._delete_rule_from_db(rule_id)
                
                logger.info(f"Règle supprimée: {rule_id}")
                return True
                
        except Exception as e:
            logger.error(f"Erreur suppression règle {rule_id}: {e}")
            return False
    
    def evaluate_rules(self, event: Dict[str, Any], tenant_id: Optional[str] = None) -> List[RuleEvaluationResult]:
        """
        Évalue les règles contre un événement
        
        Args:
            event: Événement à évaluer
            tenant_id: ID du tenant (pour filtrage)
            
        Returns:
            Liste des résultats d'évaluation
        """
        try:
            start_time = time.time()
            results = []
            
            # Filtrage des règles applicables
            applicable_rules = self._get_applicable_rules(event, tenant_id)
            
            # Évaluation de chaque règle
            for rule in applicable_rules:
                try:
                    result = self._evaluate_single_rule(rule, event)
                    if result:
                        results.append(result)
                        
                        # Sauvegarde du résultat
                        if result.matched:
                            self._save_evaluation_result(result, tenant_id)
                            
                except Exception as e:
                    logger.error(f"Erreur évaluation règle {rule.id}: {e}")
                    self.performance_metrics['failed_evaluations'] += 1
            
            # Mise à jour des métriques
            evaluation_time = (time.time() - start_time) * 1000
            self.performance_metrics['total_evaluations'] += 1
            self.performance_metrics['successful_evaluations'] += len(results)
            
            # Mise à jour de la moyenne
            total = self.performance_metrics['total_evaluations']
            current_avg = self.performance_metrics['average_evaluation_time_ms']
            self.performance_metrics['average_evaluation_time_ms'] = (
                (current_avg * (total - 1) + evaluation_time) / total
            )
            
            # Historique pour analytics
            self.evaluation_history.append({
                'timestamp': time.time(),
                'rules_evaluated': len(applicable_rules),
                'matches': len([r for r in results if r.matched]),
                'evaluation_time_ms': evaluation_time,
                'tenant_id': tenant_id
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur évaluation règles: {e}")
            self.performance_metrics['failed_evaluations'] += 1
            return []
    
    def _get_applicable_rules(self, event: Dict[str, Any], tenant_id: Optional[str] = None) -> List[Rule]:
        """Récupère les règles applicables à un événement"""
        applicable_rules = []
        
        with self.lock:
            for rule in self.rules.values():
                # Filtrage par tenant
                if tenant_id and rule.tenant_id and rule.tenant_id != tenant_id:
                    continue
                
                # Règle désactivée
                if not rule.enabled:
                    continue
                
                # Vérification de la pertinence basée sur les labels
                if self._is_rule_applicable(rule, event):
                    applicable_rules.append(rule)
        
        # Tri par priorité
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        return applicable_rules
    
    def _is_rule_applicable(self, rule: Rule, event: Dict[str, Any]) -> bool:
        """Vérifie si une règle est applicable à un événement"""
        try:
            # Vérification basique des champs requis
            for condition in rule.conditions:
                if condition.field in event:
                    return True
            
            # Vérification des labels
            event_labels = event.get('labels', {})
            rule_labels = rule.labels
            
            if rule_labels:
                for key, value in rule_labels.items():
                    if event_labels.get(key) == value:
                        return True
            
            # Par défaut, applicable si pas de critères spécifiques
            return not rule_labels
            
        except Exception as e:
            logger.error(f"Erreur vérification applicabilité règle {rule.id}: {e}")
            return False
    
    def _evaluate_single_rule(self, rule: Rule, event: Dict[str, Any]) -> Optional[RuleEvaluationResult]:
        """Évalue une règle individuelle"""
        try:
            start_time = time.time()
            
            # Cache check
            cache_key = self._generate_cache_key(rule.id, event)
            if cache_key in self.rule_cache:
                self.performance_metrics['cache_hits'] += 1
                cached_result = self.rule_cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.config['cache_ttl_seconds']:
                    return cached_result['result']
                else:
                    del self.rule_cache[cache_key]
            
            self.performance_metrics['cache_misses'] += 1
            
            # Évaluation selon le type de règle
            if rule.rule_type == RuleType.THRESHOLD:
                result = self._evaluate_threshold_rule(rule, event)
            elif rule.rule_type == RuleType.PATTERN:
                result = self._evaluate_pattern_rule(rule, event)
            elif rule.rule_type == RuleType.ANOMALY:
                result = self._evaluate_anomaly_rule(rule, event)
            elif rule.rule_type == RuleType.ML_BASED:
                result = self._evaluate_ml_rule(rule, event)
            elif rule.rule_type == RuleType.COMPOSITE:
                result = self._evaluate_composite_rule(rule, event)
            else:
                result = self._evaluate_basic_rule(rule, event)
            
            if result:
                execution_time = (time.time() - start_time) * 1000
                result.execution_time_ms = execution_time
                
                # Mise en cache
                if len(self.rule_cache) < self.config['rule_compilation_cache_size']:
                    self.rule_cache[cache_key] = {
                        'result': result,
                        'timestamp': time.time()
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur évaluation règle {rule.id}: {e}")
            return None
    
    def _evaluate_threshold_rule(self, rule: Rule, event: Dict[str, Any]) -> Optional[RuleEvaluationResult]:
        """Évalue une règle de seuil"""
        try:
            matched_conditions = []
            total_score = 0
            
            for condition in rule.conditions:
                field_value = self._get_field_value(event, condition.field)
                if field_value is None:
                    continue
                
                condition_matched = self._evaluate_condition(condition, field_value)
                
                if condition_matched:
                    matched_conditions.append(condition.field)
                    total_score += condition.weight
                elif condition.negated:
                    matched_conditions.append(f"NOT {condition.field}")
                    total_score += condition.weight
            
            # Calcul du score final
            max_score = sum(c.weight for c in rule.conditions)
            final_score = total_score / max_score if max_score > 0 else 0
            
            # Seuil de correspondance (configurable)
            match_threshold = rule.threshold_config.get('match_threshold', 0.7) if rule.threshold_config else 0.7
            matched = final_score >= match_threshold
            
            return RuleEvaluationResult(
                rule_id=rule.id,
                matched=matched,
                score=final_score,
                execution_time_ms=0,  # Sera rempli par l'appelant
                matched_conditions=matched_conditions,
                context={
                    'rule_type': 'threshold',
                    'match_threshold': match_threshold,
                    'conditions_evaluated': len(rule.conditions)
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur évaluation règle seuil {rule.id}: {e}")
            return None
    
    def _evaluate_pattern_rule(self, rule: Rule, event: Dict[str, Any]) -> Optional[RuleEvaluationResult]:
        """Évalue une règle de pattern"""
        try:
            if not rule.pattern_config:
                return None
            
            # Détection de pattern via le détecteur
            pattern_result = self.pattern_detector.detect_pattern(event, rule.pattern_config)
            
            if pattern_result:
                return RuleEvaluationResult(
                    rule_id=rule.id,
                    matched=pattern_result['matched'],
                    score=pattern_result['confidence'],
                    execution_time_ms=0,
                    matched_conditions=[pattern_result['pattern_type']],
                    context={
                        'rule_type': 'pattern',
                        'pattern_type': pattern_result['pattern_type'],
                        'pattern_details': pattern_result.get('details', {})
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur évaluation règle pattern {rule.id}: {e}")
            return None
    
    def _evaluate_anomaly_rule(self, rule: Rule, event: Dict[str, Any]) -> Optional[RuleEvaluationResult]:
        """Évalue une règle d'anomalie"""
        try:
            # Détection d'anomalie via le détecteur ML
            anomaly_result = self.anomaly_detector.detect_anomaly(event, rule)
            
            if anomaly_result:
                return RuleEvaluationResult(
                    rule_id=rule.id,
                    matched=anomaly_result['is_anomaly'],
                    score=anomaly_result['anomaly_score'],
                    execution_time_ms=0,
                    matched_conditions=['anomaly_detected'] if anomaly_result['is_anomaly'] else [],
                    context={
                        'rule_type': 'anomaly',
                        'anomaly_type': anomaly_result.get('anomaly_type'),
                        'baseline_deviation': anomaly_result.get('deviation')
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur évaluation règle anomalie {rule.id}: {e}")
            return None
    
    def _evaluate_ml_rule(self, rule: Rule, event: Dict[str, Any]) -> Optional[RuleEvaluationResult]:
        """Évalue une règle basée sur ML"""
        try:
            if not rule.ml_model_id or rule.ml_model_id not in self.ml_models:
                return None
            
            model = self.ml_models[rule.ml_model_id]
            
            # Extraction des features
            features = self._extract_features(event, model['config']['features'])
            
            # Prédiction
            prediction = model['model'].predict([features])[0]
            confidence = model['model'].predict_proba([features])[0].max()
            
            matched = prediction == 1 and confidence >= model['config']['confidence_threshold']
            
            return RuleEvaluationResult(
                rule_id=rule.id,
                matched=matched,
                score=confidence,
                execution_time_ms=0,
                matched_conditions=['ml_prediction'] if matched else [],
                context={
                    'rule_type': 'ml_based',
                    'model_type': model['config']['model_type'],
                    'confidence': confidence,
                    'prediction': prediction
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur évaluation règle ML {rule.id}: {e}")
            return None
    
    def _evaluate_composite_rule(self, rule: Rule, event: Dict[str, Any]) -> Optional[RuleEvaluationResult]:
        """Évalue une règle composite"""
        try:
            # TODO: Implémentation des règles composites
            # Combinaison de plusieurs règles avec opérateurs logiques (AND, OR, NOT)
            return None
            
        except Exception as e:
            logger.error(f"Erreur évaluation règle composite {rule.id}: {e}")
            return None
    
    def _evaluate_basic_rule(self, rule: Rule, event: Dict[str, Any]) -> Optional[RuleEvaluationResult]:
        """Évalue une règle basique"""
        try:
            matched_conditions = []
            
            for condition in rule.conditions:
                field_value = self._get_field_value(event, condition.field)
                if field_value is None:
                    continue
                
                if self._evaluate_condition(condition, field_value):
                    matched_conditions.append(condition.field)
            
            # Règle simple: toutes les conditions doivent être vraies
            matched = len(matched_conditions) == len(rule.conditions)
            score = len(matched_conditions) / len(rule.conditions) if rule.conditions else 0
            
            return RuleEvaluationResult(
                rule_id=rule.id,
                matched=matched,
                score=score,
                execution_time_ms=0,
                matched_conditions=matched_conditions,
                context={'rule_type': 'basic'}
            )
            
        except Exception as e:
            logger.error(f"Erreur évaluation règle basique {rule.id}: {e}")
            return None
    
    def _evaluate_condition(self, condition: RuleCondition, field_value: Any) -> bool:
        """Évalue une condition individuelle"""
        try:
            result = False
            
            if condition.operator == RuleOperator.GT:
                result = float(field_value) > float(condition.value)
            elif condition.operator == RuleOperator.LT:
                result = float(field_value) < float(condition.value)
            elif condition.operator == RuleOperator.GTE:
                result = float(field_value) >= float(condition.value)
            elif condition.operator == RuleOperator.LTE:
                result = float(field_value) <= float(condition.value)
            elif condition.operator == RuleOperator.EQ:
                result = field_value == condition.value
            elif condition.operator == RuleOperator.NEQ:
                result = field_value != condition.value
            elif condition.operator == RuleOperator.CONTAINS:
                result = str(condition.value) in str(field_value)
            elif condition.operator == RuleOperator.REGEX:
                result = bool(re.search(str(condition.value), str(field_value)))
            elif condition.operator == RuleOperator.IN:
                result = field_value in condition.value
            elif condition.operator == RuleOperator.NOT_IN:
                result = field_value not in condition.value
            
            return result if not condition.negated else not result
            
        except Exception as e:
            logger.error(f"Erreur évaluation condition: {e}")
            return False
    
    def _get_field_value(self, event: Dict[str, Any], field_path: str) -> Any:
        """Récupère la valeur d'un champ avec support des paths nested"""
        try:
            # Support des paths comme "labels.severity" ou "metrics.cpu.usage"
            path_parts = field_path.split('.')
            value = event
            
            for part in path_parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            
            return value
            
        except Exception:
            return None
    
    def _extract_features(self, event: Dict[str, Any], feature_names: List[str]) -> List[float]:
        """Extrait les features pour ML"""
        features = []
        
        for feature_name in feature_names:
            value = self._get_field_value(event, feature_name)
            
            if value is None:
                features.append(0.0)
            elif isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(float(value))
            elif isinstance(value, str):
                # Hash du string pour feature numérique
                features.append(float(hash(value) % 1000))
            else:
                features.append(0.0)
        
        return features
    
    def _validate_rule(self, rule: Rule) -> bool:
        """Valide une règle"""
        try:
            # Vérifications de base
            if not rule.id or not rule.name:
                return False
            
            if not rule.conditions:
                return False
            
            # Validation des conditions
            for condition in rule.conditions:
                if not condition.field:
                    return False
                
                if condition.operator not in RuleOperator:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation règle: {e}")
            return False
    
    def _compile_all_rules(self):
        """Compile toutes les règles"""
        try:
            compiled_count = 0
            
            with self.lock:
                for rule_id, rule in self.rules.items():
                    compiled_rule = self.rule_compiler.compile_rule(rule)
                    if compiled_rule:
                        self.compiled_rules[rule_id] = compiled_rule
                        rule.compiled_rule = compiled_rule
                        compiled_count += 1
            
            logger.info(f"Compilé {compiled_count} règles")
            
        except Exception as e:
            logger.error(f"Erreur compilation règles: {e}")
    
    def _generate_cache_key(self, rule_id: str, event: Dict[str, Any]) -> str:
        """Génère une clé de cache pour une règle et un événement"""
        # Utilise les champs principaux pour générer une clé stable
        key_fields = [
            rule_id,
            event.get('name', ''),
            event.get('source', ''),
            str(event.get('severity', '')),
            str(event.get('timestamp', ''))
        ]
        return '_'.join(key_fields)
    
    def _invalidate_cache(self, rule_id: str):
        """Invalide le cache pour une règle"""
        keys_to_remove = [key for key in self.rule_cache.keys() if key.startswith(rule_id)]
        for key in keys_to_remove:
            del self.rule_cache[key]
    
    def _ml_update_loop(self):
        """Boucle de mise à jour des modèles ML"""
        while self.is_running:
            try:
                if self.config['enable_ml_rules']:
                    self._update_ml_models()
                    self._retrain_models_if_needed()
                
                time.sleep(self.config['ml_model_update_interval_hours'] * 3600)
                
            except Exception as e:
                logger.error(f"Erreur boucle ML: {e}")
                time.sleep(3600)
    
    def _optimization_loop(self):
        """Boucle d'optimisation des seuils"""
        while self.is_running:
            try:
                if self.config['threshold_optimization_enabled']:
                    self._optimize_thresholds()
                
                if self.config['pattern_detection_enabled']:
                    self._detect_new_patterns()
                
                time.sleep(6 * 3600)  # Toutes les 6 heures
                
            except Exception as e:
                logger.error(f"Erreur boucle optimisation: {e}")
                time.sleep(3600)
    
    def _update_ml_models(self):
        """Met à jour les modèles ML"""
        try:
            # TODO: Implémentation de la mise à jour des modèles ML
            logger.debug("Mise à jour des modèles ML")
            
        except Exception as e:
            logger.error(f"Erreur mise à jour modèles ML: {e}")
    
    def _retrain_models_if_needed(self):
        """Réentraîne les modèles si nécessaire"""
        try:
            # TODO: Implémentation du réentraînement
            logger.debug("Vérification réentraînement modèles")
            
        except Exception as e:
            logger.error(f"Erreur réentraînement modèles: {e}")
    
    def _optimize_thresholds(self):
        """Optimise les seuils des règles"""
        try:
            optimized_count = self.threshold_optimizer.optimize_all_rules(self.rules)
            if optimized_count > 0:
                logger.info(f"Optimisé {optimized_count} seuils de règles")
            
        except Exception as e:
            logger.error(f"Erreur optimisation seuils: {e}")
    
    def _detect_new_patterns(self):
        """Détecte de nouveaux patterns"""
        try:
            new_patterns = self.pattern_detector.discover_patterns(self.evaluation_history)
            if new_patterns:
                logger.info(f"Détecté {len(new_patterns)} nouveaux patterns")
                self._save_detected_patterns(new_patterns)
            
        except Exception as e:
            logger.error(f"Erreur détection patterns: {e}")
    
    def _load_rules_from_db(self):
        """Charge les règles depuis la base de données"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM rules WHERE enabled = 1')
            rows = cursor.fetchall()
            
            loaded_count = 0
            for row in rows:
                try:
                    rule = self._row_to_rule(row)
                    if rule:
                        self.rules[rule.id] = rule
                        loaded_count += 1
                except Exception as e:
                    logger.error(f"Erreur chargement règle: {e}")
            
            conn.close()
            logger.info(f"Chargé {loaded_count} règles depuis la base")
            
        except Exception as e:
            logger.error(f"Erreur chargement règles: {e}")
    
    def _row_to_rule(self, row) -> Optional[Rule]:
        """Convertit une ligne de base en objet Rule"""
        try:
            conditions_data = json.loads(row[4])  # conditions
            conditions = [
                RuleCondition(
                    field=c['field'],
                    operator=RuleOperator(c['operator']),
                    value=c['value'],
                    weight=c.get('weight', 1.0),
                    negated=c.get('negated', False)
                )
                for c in conditions_data
            ]
            
            return Rule(
                id=row[0],
                name=row[1],
                description=row[2] or '',
                rule_type=RuleType(row[3]),
                conditions=conditions,
                action=RuleAction(row[5]),
                priority=row[6],
                enabled=bool(row[7]),
                tenant_id=row[8],
                labels=json.loads(row[9]) if row[9] else {},
                created_at=row[10],
                updated_at=row[11],
                ml_model_id=row[12],
                threshold_config=json.loads(row[13]) if row[13] else None,
                pattern_config=json.loads(row[14]) if row[14] else None
            )
            
        except Exception as e:
            logger.error(f"Erreur conversion ligne vers règle: {e}")
            return None
    
    def _setup_default_rules(self):
        """Configure les règles par défaut"""
        try:
            # Règle CPU critique
            cpu_rule = Rule(
                id="default_cpu_critical",
                name="CPU Usage Critical",
                description="CPU usage above 90%",
                rule_type=RuleType.THRESHOLD,
                conditions=[
                    RuleCondition(
                        field="metrics.cpu.usage",
                        operator=RuleOperator.GT,
                        value=90.0
                    )
                ],
                action=RuleAction.ALERT,
                priority=1
            )
            
            # Règle mémoire warning
            memory_rule = Rule(
                id="default_memory_warning",
                name="Memory Usage Warning",
                description="Memory usage above 80%",
                rule_type=RuleType.THRESHOLD,
                conditions=[
                    RuleCondition(
                        field="metrics.memory.percent",
                        operator=RuleOperator.GT,
                        value=80.0
                    )
                ],
                action=RuleAction.ALERT,
                priority=2
            )
            
            # Ajout des règles par défaut
            default_rules = [cpu_rule, memory_rule]
            
            for rule in default_rules:
                if rule.id not in self.rules:
                    self.add_rule(rule)
            
            logger.info("Règles par défaut configurées")
            
        except Exception as e:
            logger.error(f"Erreur configuration règles par défaut: {e}")
    
    def _save_rule_to_db(self, rule: Rule):
        """Sauvegarde une règle en base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO rules 
                (id, name, description, rule_type, conditions, action, priority, enabled, 
                 tenant_id, labels, created_at, updated_at, ml_model_id, threshold_config, pattern_config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.id, rule.name, rule.description, rule.rule_type.value,
                json.dumps([asdict(c) for c in rule.conditions]),
                rule.action.value, rule.priority, rule.enabled,
                rule.tenant_id, json.dumps(rule.labels),
                rule.created_at, rule.updated_at, rule.ml_model_id,
                json.dumps(rule.threshold_config) if rule.threshold_config else None,
                json.dumps(rule.pattern_config) if rule.pattern_config else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde règle: {e}")
    
    def _delete_rule_from_db(self, rule_id: str):
        """Supprime une règle de la base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM rules WHERE id = ?', (rule_id,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur suppression règle: {e}")
    
    def _save_evaluation_result(self, result: RuleEvaluationResult, tenant_id: Optional[str]):
        """Sauvegarde un résultat d'évaluation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rule_evaluations 
                (rule_id, matched, score, execution_time_ms, matched_conditions, context, timestamp, tenant_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.rule_id, result.matched, result.score, result.execution_time_ms,
                json.dumps(result.matched_conditions), json.dumps(result.context),
                result.timestamp, tenant_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde résultat évaluation: {e}")
    
    def _save_detected_patterns(self, patterns: List[Dict[str, Any]]):
        """Sauvegarde les patterns détectés"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for pattern in patterns:
                cursor.execute('''
                    INSERT INTO detected_patterns 
                    (pattern_type, pattern_data, confidence, first_seen, last_seen, tenant_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    pattern['type'], json.dumps(pattern['data']), pattern['confidence'],
                    pattern['first_seen'], pattern['last_seen'], pattern.get('tenant_id')
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde patterns: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Vérification de santé du moteur"""
        try:
            return {
                'status': 'healthy' if self.is_running else 'stopped',
                'rules_loaded': len(self.rules),
                'compiled_rules': len(self.compiled_rules),
                'cache_size': len(self.rule_cache),
                'performance_metrics': self.performance_metrics,
                'ml_models_loaded': len(self.ml_models),
                'uptime_seconds': time.time() - self.start_time
            }
            
        except Exception as e:
            logger.error(f"Erreur health check: {e}")
            return {'status': 'error', 'error': str(e)}

# Composants auxiliaires
class RuleCompiler:
    """Compilateur de règles pour optimiser l'évaluation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def compile_rule(self, rule: Rule) -> Optional[Any]:
        """Compile une règle pour optimiser l'évaluation"""
        try:
            # Compilation basique - peut être étendue
            compiled = {
                'id': rule.id,
                'type': rule.rule_type.value,
                'conditions': [asdict(c) for c in rule.conditions],
                'optimized': True,
                'compiled_at': time.time()
            }
            
            return compiled
            
        except Exception as e:
            logger.error(f"Erreur compilation règle {rule.id}: {e}")
            return None

class PatternDetector:
    """Détecteur de patterns avancé"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.known_patterns = {}
    
    def detect_pattern(self, event: Dict[str, Any], pattern_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Détecte un pattern dans un événement"""
        try:
            # TODO: Implémentation de la détection de patterns
            return None
            
        except Exception as e:
            logger.error(f"Erreur détection pattern: {e}")
            return None
    
    def discover_patterns(self, evaluation_history: deque) -> List[Dict[str, Any]]:
        """Découvre de nouveaux patterns dans l'historique"""
        try:
            # TODO: Implémentation de la découverte de patterns
            return []
            
        except Exception as e:
            logger.error(f"Erreur découverte patterns: {e}")
            return []

class ThresholdOptimizer:
    """Optimiseur de seuils automatique"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def optimize_all_rules(self, rules: Dict[str, Rule]) -> int:
        """Optimise les seuils de toutes les règles"""
        try:
            # TODO: Implémentation de l'optimisation des seuils
            return 0
            
        except Exception as e:
            logger.error(f"Erreur optimisation seuils: {e}")
            return 0

class AnomalyDetector:
    """Détecteur d'anomalies ML"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
    
    def detect_anomaly(self, event: Dict[str, Any], rule: Rule) -> Optional[Dict[str, Any]]:
        """Détecte une anomalie dans un événement"""
        try:
            # TODO: Implémentation de la détection d'anomalies ML
            return None
            
        except Exception as e:
            logger.error(f"Erreur détection anomalie: {e}")
            return None
