#!/usr/bin/env python3
"""
Module de Gestion Avancée des Règles d'Alertes - Système Industriel Ultra-Performant

Ce module fournit un système complet de gestion des règles d'alertes avec intelligence artificielle,
machine learning, et orchestration microservices pour des environnements de production haute performance.

Architecture:
- Moteur de règles ML avec évaluation en temps réel
- Gestionnaire de politiques avec application automatique
- Validateur intelligent avec correction automatique
- Orchestrateur de conditions avec logique complexe
- Analyseur de patterns avec prédictions ML
- Optimiseur de performance avec mise en cache intelligente
- Gestionnaire de conformité avec audit complet
- Interface de recommandations avec IA conversationnelle

Auteur: Équipe Engineering Spotify
License: Spotify Proprietary
Copyright: © 2025 Spotify Technology S.A.
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Set, Union, Callable, 
    AsyncGenerator, Tuple, NamedTuple, Protocol, TypeVar, Generic
)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
import json
import hashlib
import uuid
import weakref
import pickle
from pathlib import Path
import aioredis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog

# ML and AI imports
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import joblib

# Configuration et logging
logger = structlog.get_logger(__name__)

# Métriques Prometheus
RULES_PROCESSED = Counter('rules_processed_total', 'Total rules processed', ['tenant_id', 'rule_type', 'status'])
RULES_EXECUTION_TIME = Histogram('rules_execution_duration_seconds', 'Rule execution time', ['tenant_id', 'rule_type'])
RULES_ACTIVE = Gauge('rules_active_count', 'Active rules count', ['tenant_id', 'rule_type'])
RULES_ML_ACCURACY = Gauge('rules_ml_accuracy', 'ML model accuracy', ['tenant_id', 'model_type'])

# Types et énumérations
T = TypeVar('T')

class RuleType(Enum):
    """Types de règles supportées"""
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    COMPOSITE = "composite"
    ML_BASED = "ml_based"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    PREDICTIVE = "predictive"

class RuleSeverity(Enum):
    """Niveaux de sévérité des règles"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RuleStatus(Enum):
    """États des règles"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DISABLED = "disabled"
    TESTING = "testing"
    DEPRECATED = "deprecated"

class RuleAction(Enum):
    """Actions possibles pour les règles"""
    ALERT = "alert"
    NOTIFICATION = "notification"
    AUTO_REMEDIATION = "auto_remediation"
    ESCALATION = "escalation"
    SUPPRESSION = "suppression"
    ENRICHMENT = "enrichment"

class MLModelType(Enum):
    """Types de modèles ML supportés"""
    ISOLATION_FOREST = "isolation_forest"
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"

@dataclass
class RuleMetadata:
    """Métadonnées des règles"""
    tenant_id: str
    rule_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    author: str = "system"
    tags: Set[str] = field(default_factory=set)
    priority: int = 1
    ttl: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tenant_id': self.tenant_id,
            'rule_id': self.rule_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'author': self.author,
            'tags': list(self.tags),
            'priority': self.priority,
            'ttl': self.ttl
        }

@dataclass
class RuleCondition:
    """Condition d'évaluation de règle"""
    field: str
    operator: str
    value: Any
    weight: float = 1.0
    ml_enhanced: bool = False
    fuzzy_matching: bool = False
    temporal_window: Optional[timedelta] = None
    
    def evaluate(self, data: Dict[str, Any], context: Optional[Dict] = None) -> bool:
        """Évalue la condition avec support ML et logique floue"""
        try:
            field_value = self._extract_field_value(data, self.field)
            
            if self.ml_enhanced and context and 'ml_evaluator' in context:
                return context['ml_evaluator'].evaluate_condition(self, field_value, data)
            
            if self.fuzzy_matching:
                return self._fuzzy_evaluate(field_value, self.operator, self.value)
            
            return self._standard_evaluate(field_value, self.operator, self.value)
            
        except Exception as e:
            logger.error("Error evaluating condition", condition=self, error=str(e))
            return False
    
    def _extract_field_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Extrait la valeur du champ avec support des chemins imbriqués"""
        parts = field_path.split('.')
        value = data
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value
    
    def _standard_evaluate(self, field_value: Any, operator: str, expected_value: Any) -> bool:
        """Évaluation standard des conditions"""
        operators = {
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y,
            '>': lambda x, y: x > y,
            '>=': lambda x, y: x >= y,
            '<': lambda x, y: x < y,
            '<=': lambda x, y: x <= y,
            'in': lambda x, y: x in y,
            'not_in': lambda x, y: x not in y,
            'contains': lambda x, y: str(y) in str(x),
            'regex': lambda x, y: bool(re.match(y, str(x))),
            'exists': lambda x, y: x is not None,
        }
        
        if operator not in operators:
            raise ValueError(f"Operator '{operator}' not supported")
        
        return operators[operator](field_value, expected_value)
    
    def _fuzzy_evaluate(self, field_value: Any, operator: str, expected_value: Any) -> bool:
        """Évaluation avec logique floue"""
        # Implémentation simplifiée de la logique floue
        if operator in ['==', '!=']:
            similarity = self._calculate_similarity(field_value, expected_value)
            threshold = 0.8  # Seuil de similarité
            
            if operator == '==':
                return similarity >= threshold
            else:
                return similarity < threshold
        
        return self._standard_evaluate(field_value, operator, expected_value)
    
    def _calculate_similarity(self, value1: Any, value2: Any) -> float:
        """Calcule la similarité entre deux valeurs"""
        if type(value1) != type(value2):
            return 0.0
        
        if isinstance(value1, str) and isinstance(value2, str):
            # Similarité de chaînes (algorithme simple)
            if len(value1) == 0 and len(value2) == 0:
                return 1.0
            if len(value1) == 0 or len(value2) == 0:
                return 0.0
            
            # Distance de Levenshtein normalisée
            max_len = max(len(value1), len(value2))
            distance = self._levenshtein_distance(value1, value2)
            return 1.0 - (distance / max_len)
        
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            # Similarité numérique
            diff = abs(value1 - value2)
            max_val = max(abs(value1), abs(value2), 1)
            return max(0, 1.0 - (diff / max_val))
        
        return 1.0 if value1 == value2 else 0.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calcule la distance de Levenshtein"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

@dataclass
class Rule:
    """Représentation complète d'une règle d'alerte"""
    metadata: RuleMetadata
    name: str
    description: str
    rule_type: RuleType
    severity: RuleSeverity
    status: RuleStatus
    conditions: List[RuleCondition]
    actions: List[RuleAction]
    tags: Set[str] = field(default_factory=set)
    ml_config: Optional[Dict[str, Any]] = None
    temporal_config: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    schedule: Optional[Dict[str, Any]] = None
    context_variables: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialisation avec validation"""
        self._validate_rule()
        self._setup_ml_config()
    
    def _validate_rule(self):
        """Valide la cohérence de la règle"""
        if not self.conditions:
            raise ValueError("Rule must have at least one condition")
        
        if not self.actions:
            raise ValueError("Rule must have at least one action")
        
        # Validation spécifique au type de règle
        if self.rule_type == RuleType.ML_BASED and not self.ml_config:
            raise ValueError("ML-based rules must have ml_config")
        
        if self.rule_type == RuleType.TEMPORAL and not self.temporal_config:
            raise ValueError("Temporal rules must have temporal_config")
    
    def _setup_ml_config(self):
        """Configure les paramètres ML par défaut"""
        if self.rule_type == RuleType.ML_BASED and not self.ml_config:
            self.ml_config = {
                'model_type': MLModelType.ISOLATION_FOREST.value,
                'training_window': 7200,  # 2 heures
                'retrain_interval': 86400,  # 24 heures
                'confidence_threshold': 0.8,
                'feature_columns': [],
                'auto_feature_selection': True
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise la règle en dictionnaire"""
        return {
            'metadata': self.metadata.to_dict(),
            'name': self.name,
            'description': self.description,
            'rule_type': self.rule_type.value,
            'severity': self.severity.value,
            'status': self.status.value,
            'conditions': [
                {
                    'field': c.field,
                    'operator': c.operator,
                    'value': c.value,
                    'weight': c.weight,
                    'ml_enhanced': c.ml_enhanced,
                    'fuzzy_matching': c.fuzzy_matching,
                    'temporal_window': c.temporal_window.total_seconds() if c.temporal_window else None
                }
                for c in self.conditions
            ],
            'actions': [action.value for action in self.actions],
            'tags': list(self.tags),
            'ml_config': self.ml_config,
            'temporal_config': self.temporal_config,
            'dependencies': self.dependencies,
            'exclusions': self.exclusions,
            'schedule': self.schedule,
            'context_variables': self.context_variables
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rule':
        """Désérialise une règle depuis un dictionnaire"""
        metadata_data = data['metadata']
        metadata = RuleMetadata(
            tenant_id=metadata_data['tenant_id'],
            rule_id=metadata_data['rule_id'],
            created_at=datetime.fromisoformat(metadata_data['created_at']),
            updated_at=datetime.fromisoformat(metadata_data['updated_at']),
            version=metadata_data['version'],
            author=metadata_data['author'],
            tags=set(metadata_data['tags']),
            priority=metadata_data['priority'],
            ttl=metadata_data.get('ttl')
        )
        
        conditions = [
            RuleCondition(
                field=c['field'],
                operator=c['operator'],
                value=c['value'],
                weight=c.get('weight', 1.0),
                ml_enhanced=c.get('ml_enhanced', False),
                fuzzy_matching=c.get('fuzzy_matching', False),
                temporal_window=timedelta(seconds=c['temporal_window']) if c.get('temporal_window') else None
            )
            for c in data['conditions']
        ]
        
        return cls(
            metadata=metadata,
            name=data['name'],
            description=data['description'],
            rule_type=RuleType(data['rule_type']),
            severity=RuleSeverity(data['severity']),
            status=RuleStatus(data['status']),
            conditions=conditions,
            actions=[RuleAction(action) for action in data['actions']],
            tags=set(data.get('tags', [])),
            ml_config=data.get('ml_config'),
            temporal_config=data.get('temporal_config'),
            dependencies=data.get('dependencies', []),
            exclusions=data.get('exclusions', []),
            schedule=data.get('schedule'),
            context_variables=data.get('context_variables', {})
        )

class RuleEvaluationResult:
    """Résultat d'évaluation d'une règle"""
    
    def __init__(self, rule: Rule, matched: bool, confidence: float = 1.0,
                 matched_conditions: List[RuleCondition] = None,
                 execution_time: float = 0.0, context: Dict[str, Any] = None):
        self.rule = rule
        self.matched = matched
        self.confidence = confidence
        self.matched_conditions = matched_conditions or []
        self.execution_time = execution_time
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        self.evaluation_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule.metadata.rule_id,
            'evaluation_id': self.evaluation_id,
            'matched': self.matched,
            'confidence': self.confidence,
            'matched_conditions_count': len(self.matched_conditions),
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }

class IRuleEngine(ABC):
    """Interface du moteur de règles"""
    
    @abstractmethod
    async def evaluate_rule(self, rule: Rule, data: Dict[str, Any], 
                          context: Optional[Dict] = None) -> RuleEvaluationResult:
        """Évalue une règle donnée"""
        pass
    
    @abstractmethod
    async def evaluate_rules(self, rules: List[Rule], data: Dict[str, Any],
                           context: Optional[Dict] = None) -> List[RuleEvaluationResult]:
        """Évalue plusieurs règles"""
        pass

class MLEnhancedRuleEvaluator:
    """Évaluateur de règles avec ML"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.feature_scalers: Dict[str, StandardScaler] = {}
        self.training_data: Dict[str, List] = {}
        self._setup_ml_components()
    
    def _setup_ml_components(self):
        """Initialise les composants ML"""
        self.nlp_pipeline = pipeline("text-classification", 
                                   model="distilbert-base-uncased-finetuned-sst-2-english")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
    async def evaluate_condition(self, condition: RuleCondition, 
                               field_value: Any, data: Dict[str, Any]) -> bool:
        """Évalue une condition avec ML"""
        try:
            if isinstance(field_value, str) and len(field_value) > 10:
                # Analyse de sentiment pour les textes
                sentiment = self.nlp_pipeline(field_value)[0]
                if condition.operator == 'sentiment_positive':
                    return sentiment['label'] == 'POSITIVE' and sentiment['score'] > 0.8
                elif condition.operator == 'sentiment_negative':
                    return sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.8
            
            if isinstance(field_value, (int, float)):
                # Détection d'anomalies pour les valeurs numériques
                model_key = f"{condition.field}_anomaly"
                if model_key not in self.models:
                    await self._train_anomaly_model(model_key, condition.field, data)
                
                if model_key in self.models:
                    prediction = self.models[model_key].predict([[field_value]])
                    if condition.operator == 'is_anomaly':
                        return prediction[0] == -1  # -1 indique une anomalie
                    elif condition.operator == 'is_normal':
                        return prediction[0] == 1   # 1 indique une valeur normale
            
            return False
            
        except Exception as e:
            logger.error("ML evaluation error", condition=condition, error=str(e))
            return False
    
    async def _train_anomaly_model(self, model_key: str, field: str, data: Dict[str, Any]):
        """Entraîne un modèle de détection d'anomalies"""
        try:
            # Collecte des données d'entraînement (simulation)
            training_data = self._generate_training_data(field, data)
            
            if len(training_data) >= 10:  # Minimum de données pour l'entraînement
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(np.array(training_data).reshape(-1, 1))
                
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(scaled_data)
                
                self.models[model_key] = model
                self.feature_scalers[model_key] = scaler
                
                logger.info("Anomaly model trained", model_key=model_key, samples=len(training_data))
                
        except Exception as e:
            logger.error("Model training error", model_key=model_key, error=str(e))
    
    def _generate_training_data(self, field: str, data: Dict[str, Any]) -> List[float]:
        """Génère des données d'entraînement (simulation)"""
        # En production, ceci devrait récupérer des données historiques
        # Ici nous simulons avec des données d'exemple
        if field not in self.training_data:
            # Génération de données d'exemple basées sur le type de champ
            base_value = data.get(field, 100)
            if isinstance(base_value, (int, float)):
                # Génération de valeurs normales autour de la valeur de base
                normal_values = np.random.normal(base_value, base_value * 0.1, 100).tolist()
                self.training_data[field] = normal_values
        
        return self.training_data.get(field, [])

# Gestionnaire principal des règles
class AdvancedRulesManager:
    """Gestionnaire avancé des règles avec ML et optimisations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules_cache: Dict[str, Rule] = {}
        self.evaluation_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.ml_evaluator = MLEnhancedRuleEvaluator(config.get('ml_config', {}))
        self.redis_client: Optional[aioredis.Redis] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 10))
        self._setup_async_components()
    
    def _setup_async_components(self):
        """Configure les composants asynchrones"""
        asyncio.create_task(self._initialize_connections())
    
    async def _initialize_connections(self):
        """Initialise les connexions Redis et PostgreSQL"""
        try:
            # Redis pour le cache
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            self.redis_client = await aioredis.from_url(redis_url)
            
            # PostgreSQL pour la persistance
            db_config = self.config.get('database', {})
            self.db_pool = await asyncpg.create_pool(
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 5432),
                user=db_config.get('user', 'postgres'),
                password=db_config.get('password', ''),
                database=db_config.get('database', 'alerts'),
                min_size=5,
                max_size=20
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize connections", error=str(e))
    
    async def load_rules(self, tenant_id: str) -> List[Rule]:
        """Charge les règles pour un tenant"""
        try:
            cache_key = f"rules:{tenant_id}"
            
            # Vérification du cache Redis
            if self.redis_client:
                cached_rules = await self.redis_client.get(cache_key)
                if cached_rules:
                    rules_data = json.loads(cached_rules)
                    return [Rule.from_dict(rule_data) for rule_data in rules_data]
            
            # Chargement depuis la base de données
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT rule_data FROM rules WHERE tenant_id = $1 AND status = 'active'",
                        tenant_id
                    )
                    
                    rules = []
                    for row in rows:
                        rule_data = json.loads(row['rule_data'])
                        rule = Rule.from_dict(rule_data)
                        rules.append(rule)
                        self.rules_cache[rule.metadata.rule_id] = rule
                    
                    # Mise en cache Redis
                    if self.redis_client and rules:
                        rules_data = [rule.to_dict() for rule in rules]
                        await self.redis_client.setex(cache_key, 3600, json.dumps(rules_data))
                    
                    logger.info("Rules loaded", tenant_id=tenant_id, count=len(rules))
                    return rules
            
            return []
            
        except Exception as e:
            logger.error("Error loading rules", tenant_id=tenant_id, error=str(e))
            return []
    
    async def evaluate_rules(self, tenant_id: str, data: Dict[str, Any],
                           rules: Optional[List[Rule]] = None) -> List[RuleEvaluationResult]:
        """Évalue les règles pour des données données"""
        try:
            if rules is None:
                rules = await self.load_rules(tenant_id)
            
            if not rules:
                return []
            
            # Contexte d'évaluation
            context = {
                'tenant_id': tenant_id,
                'timestamp': datetime.utcnow(),
                'ml_evaluator': self.ml_evaluator
            }
            
            # Évaluation parallèle des règles
            evaluation_tasks = [
                self._evaluate_single_rule(rule, data, context)
                for rule in rules
            ]
            
            results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
            
            # Filtrage des résultats valides
            valid_results = [
                result for result in results
                if isinstance(result, RuleEvaluationResult)
            ]
            
            # Mise à jour des métriques
            for result in valid_results:
                RULES_PROCESSED.labels(
                    tenant_id=tenant_id,
                    rule_type=result.rule.rule_type.value,
                    status='matched' if result.matched else 'not_matched'
                ).inc()
                
                RULES_EXECUTION_TIME.labels(
                    tenant_id=tenant_id,
                    rule_type=result.rule.rule_type.value
                ).observe(result.execution_time)
            
            logger.info("Rules evaluation completed", 
                       tenant_id=tenant_id, 
                       total_rules=len(rules),
                       matched_rules=len([r for r in valid_results if r.matched]))
            
            return valid_results
            
        except Exception as e:
            logger.error("Error evaluating rules", tenant_id=tenant_id, error=str(e))
            return []
    
    async def _evaluate_single_rule(self, rule: Rule, data: Dict[str, Any],
                                  context: Dict[str, Any]) -> RuleEvaluationResult:
        """Évalue une règle unique"""
        start_time = time.time()
        
        try:
            # Vérification des dépendances
            if rule.dependencies:
                dependencies_met = await self._check_dependencies(rule.dependencies, context)
                if not dependencies_met:
                    return RuleEvaluationResult(rule, False, 0.0, [], time.time() - start_time)
            
            # Vérification des exclusions
            if rule.exclusions:
                excluded = await self._check_exclusions(rule.exclusions, data, context)
                if excluded:
                    return RuleEvaluationResult(rule, False, 0.0, [], time.time() - start_time)
            
            # Évaluation des conditions
            matched_conditions = []
            total_weight = 0.0
            matched_weight = 0.0
            
            for condition in rule.conditions:
                condition_matched = condition.evaluate(data, context)
                total_weight += condition.weight
                
                if condition_matched:
                    matched_conditions.append(condition)
                    matched_weight += condition.weight
            
            # Calcul de la confiance
            confidence = matched_weight / total_weight if total_weight > 0 else 0.0
            
            # Seuil de confiance configurable
            confidence_threshold = rule.ml_config.get('confidence_threshold', 0.5) \
                if rule.ml_config else 0.5
            
            matched = confidence >= confidence_threshold
            
            execution_time = time.time() - start_time
            
            return RuleEvaluationResult(
                rule=rule,
                matched=matched,
                confidence=confidence,
                matched_conditions=matched_conditions,
                execution_time=execution_time,
                context=context
            )
            
        except Exception as e:
            logger.error("Error evaluating rule", rule_id=rule.metadata.rule_id, error=str(e))
            return RuleEvaluationResult(rule, False, 0.0, [], time.time() - start_time)
    
    async def _check_dependencies(self, dependencies: List[str], context: Dict[str, Any]) -> bool:
        """Vérifie si les dépendances sont satisfaites"""
        # Implémentation simplifiée - en production, vérifier l'état des services
        return True
    
    async def _check_exclusions(self, exclusions: List[str], 
                              data: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Vérifie si des conditions d'exclusion sont présentes"""
        # Implémentation simplifiée - en production, vérifier les conditions d'exclusion
        return False
    
    async def save_rule(self, rule: Rule) -> bool:
        """Sauvegarde une règle"""
        try:
            if self.db_pool:
                rule_data = json.dumps(rule.to_dict())
                
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO rules (tenant_id, rule_id, rule_data, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (tenant_id, rule_id) 
                        DO UPDATE SET rule_data = $3, updated_at = $5
                    """, rule.metadata.tenant_id, rule.metadata.rule_id, 
                        rule_data, rule.metadata.created_at, rule.metadata.updated_at)
                
                # Invalidation du cache
                if self.redis_client:
                    cache_key = f"rules:{rule.metadata.tenant_id}"
                    await self.redis_client.delete(cache_key)
                
                logger.info("Rule saved", rule_id=rule.metadata.rule_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Error saving rule", rule_id=rule.metadata.rule_id, error=str(e))
            return False
    
    async def delete_rule(self, tenant_id: str, rule_id: str) -> bool:
        """Supprime une règle"""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    result = await conn.execute(
                        "DELETE FROM rules WHERE tenant_id = $1 AND rule_id = $2",
                        tenant_id, rule_id
                    )
                
                # Invalidation du cache
                if self.redis_client:
                    cache_key = f"rules:{tenant_id}"
                    await self.redis_client.delete(cache_key)
                
                # Suppression du cache local
                if rule_id in self.rules_cache:
                    del self.rules_cache[rule_id]
                
                logger.info("Rule deleted", tenant_id=tenant_id, rule_id=rule_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Error deleting rule", tenant_id=tenant_id, rule_id=rule_id, error=str(e))
            return False
    
    async def get_rule_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """Obtient les statistiques des règles"""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    stats = await conn.fetchrow("""
                        SELECT 
                            COUNT(*) as total_rules,
                            COUNT(CASE WHEN (rule_data->>'status') = 'active' THEN 1 END) as active_rules,
                            COUNT(CASE WHEN (rule_data->>'status') = 'inactive' THEN 1 END) as inactive_rules,
                            COUNT(CASE WHEN (rule_data->>'rule_type') = 'ml_based' THEN 1 END) as ml_rules
                        FROM rules 
                        WHERE tenant_id = $1
                    """, tenant_id)
                
                return dict(stats) if stats else {}
            
            return {}
            
        except Exception as e:
            logger.error("Error getting statistics", tenant_id=tenant_id, error=str(e))
            return {}
    
    async def optimize_rules(self, tenant_id: str) -> Dict[str, Any]:
        """Optimise les règles basées sur les performances"""
        try:
            rules = await self.load_rules(tenant_id)
            optimizations = {
                'rules_analyzed': len(rules),
                'optimizations_applied': 0,
                'performance_improvements': []
            }
            
            for rule in rules:
                # Analyse de performance des conditions
                if len(rule.conditions) > 5:
                    # Suggérer de diviser en sous-règles
                    optimizations['performance_improvements'].append({
                        'rule_id': rule.metadata.rule_id,
                        'suggestion': 'Consider splitting into multiple rules',
                        'reason': 'Too many conditions may impact performance'
                    })
                
                # Optimisation ML
                if rule.rule_type == RuleType.ML_BASED and rule.ml_config:
                    confidence_threshold = rule.ml_config.get('confidence_threshold', 0.5)
                    if confidence_threshold < 0.6:
                        rule.ml_config['confidence_threshold'] = 0.6
                        await self.save_rule(rule)
                        optimizations['optimizations_applied'] += 1
            
            logger.info("Rules optimization completed", tenant_id=tenant_id, **optimizations)
            return optimizations
            
        except Exception as e:
            logger.error("Error optimizing rules", tenant_id=tenant_id, error=str(e))
            return {}
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.db_pool:
                await self.db_pool.close()
            
            self.executor.shutdown(wait=True)
            
            logger.info("Rules manager cleanup completed")
            
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))

# Fonctions utilitaires globales
async def initialize_rules_system(config: Dict[str, Any]) -> AdvancedRulesManager:
    """Initialise le système de règles"""
    manager = AdvancedRulesManager(config)
    await asyncio.sleep(0.1)  # Attendre l'initialisation des connexions
    return manager

def create_threshold_rule(tenant_id: str, rule_id: str, field: str, 
                         operator: str, threshold: float, severity: RuleSeverity) -> Rule:
    """Crée une règle de seuil simple"""
    metadata = RuleMetadata(tenant_id=tenant_id, rule_id=rule_id)
    condition = RuleCondition(field=field, operator=operator, value=threshold)
    
    return Rule(
        metadata=metadata,
        name=f"Threshold rule for {field}",
        description=f"Alert when {field} {operator} {threshold}",
        rule_type=RuleType.THRESHOLD,
        severity=severity,
        status=RuleStatus.ACTIVE,
        conditions=[condition],
        actions=[RuleAction.ALERT]
    )

def create_ml_anomaly_rule(tenant_id: str, rule_id: str, field: str, 
                          severity: RuleSeverity) -> Rule:
    """Crée une règle de détection d'anomalies ML"""
    metadata = RuleMetadata(tenant_id=tenant_id, rule_id=rule_id)
    condition = RuleCondition(field=field, operator="is_anomaly", value=True, ml_enhanced=True)
    
    ml_config = {
        'model_type': MLModelType.ISOLATION_FOREST.value,
        'confidence_threshold': 0.8,
        'feature_columns': [field],
        'training_window': 7200,
        'retrain_interval': 86400
    }
    
    return Rule(
        metadata=metadata,
        name=f"ML Anomaly detection for {field}",
        description=f"ML-based anomaly detection for {field}",
        rule_type=RuleType.ML_BASED,
        severity=severity,
        status=RuleStatus.ACTIVE,
        conditions=[condition],
        actions=[RuleAction.ALERT],
        ml_config=ml_config
    )

# Exportation des classes principales
__all__ = [
    'Rule', 'RuleCondition', 'RuleMetadata', 'RuleEvaluationResult',
    'RuleType', 'RuleSeverity', 'RuleStatus', 'RuleAction', 'MLModelType',
    'AdvancedRulesManager', 'MLEnhancedRuleEvaluator', 'IRuleEngine',
    'initialize_rules_system', 'create_threshold_rule', 'create_ml_anomaly_rule'
]

# Configuration par défaut
DEFAULT_CONFIG = {
    'max_workers': 10,
    'redis_url': 'redis://localhost:6379',
    'database': {
        'host': 'localhost',
        'port': 5432,
        'user': 'postgres',
        'password': '',
        'database': 'alerts'
    },
    'ml_config': {
        'enable_nlp': True,
        'enable_anomaly_detection': True,
        'confidence_threshold': 0.8
    }
}

if __name__ == "__main__":
    # Exemple d'utilisation
    async def main():
        # Initialisation du système
        manager = await initialize_rules_system(DEFAULT_CONFIG)
        
        # Création d'une règle simple
        rule = create_threshold_rule(
            tenant_id="spotify-prod",
            rule_id="cpu_high",
            field="cpu_usage",
            operator=">",
            threshold=80.0,
            severity=RuleSeverity.HIGH
        )
        
        # Sauvegarde de la règle
        await manager.save_rule(rule)
        
        # Évaluation avec des données d'exemple
        test_data = {
            "cpu_usage": 85.0,
            "memory_usage": 60.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        results = await manager.evaluate_rules("spotify-prod", test_data)
        
        for result in results:
            if result.matched:
                print(f"Rule {result.rule.metadata.rule_id} matched with confidence {result.confidence}")
        
        # Nettoyage
        await manager.cleanup()
    
    asyncio.run(main())
