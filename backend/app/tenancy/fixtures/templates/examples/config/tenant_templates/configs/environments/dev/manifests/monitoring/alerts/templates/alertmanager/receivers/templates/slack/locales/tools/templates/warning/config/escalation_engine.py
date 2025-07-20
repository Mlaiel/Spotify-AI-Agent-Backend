"""
Moteur d'Escalade Intelligent - Spotify AI Agent
===============================================

Système ultra-avancé d'escalade automatique pour les alertes Warning
avec Machine Learning, analyse des patterns et optimisation dynamique.

Auteur: Équipe d'experts dirigée par Fahed Mlaiel
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict, deque
import sqlite3
import hashlib

# Configuration du logging
logger = logging.getLogger(__name__)

class EscalationLevel(Enum):
    """Niveaux d'escalade disponibles."""
    NONE = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    CRITICAL = 4

class EscalationAction(Enum):
    """Actions d'escalade possibles."""
    NOTIFY_ADDITIONAL_CHANNELS = "notify_additional"
    INCREASE_FREQUENCY = "increase_frequency"
    ESCALATE_TO_MANAGER = "escalate_manager"
    TRIGGER_INCIDENT = "trigger_incident"
    AUTO_REMEDIATION = "auto_remediation"

@dataclass
class EscalationRule:
    """Règle d'escalade configurée."""
    rule_id: str
    tenant_id: str
    alert_pattern: str
    initial_delay_minutes: int
    escalation_intervals: List[int]  # minutes entre chaque escalade
    max_escalations: int
    escalation_actions: List[EscalationAction]
    conditions: Dict[str, Any]
    enabled: bool
    created_at: datetime
    updated_at: datetime

@dataclass
class EscalationEvent:
    """Événement d'escalade déclenché."""
    event_id: str
    alert_id: str
    tenant_id: str
    escalation_level: EscalationLevel
    escalation_action: EscalationAction
    trigger_time: datetime
    next_escalation_time: Optional[datetime]
    metadata: Dict[str, Any]
    completed: bool = False

@dataclass
class AlertPattern:
    """Pattern d'alerte analysé par ML."""
    pattern_id: str
    pattern_signature: str
    frequency_score: float
    severity_trend: float
    escalation_probability: float
    suggested_delay: int
    confidence_score: float
    last_updated: datetime

class MLEscalationPredictor:
    """Prédicteur ML pour optimiser les délais d'escalade."""
    
    def __init__(self):
        self.feature_history = deque(maxlen=10000)
        self.escalation_history = deque(maxlen=10000)
        self.pattern_cache = {}
        self.cache_lock = threading.RLock()
    
    def extract_features(self, alert_data: Dict[str, Any]) -> np.ndarray:
        """Extrait les caractéristiques d'une alerte pour le ML."""
        features = []
        
        # Caractéristiques temporelles
        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        features.extend([hour / 24.0, day_of_week / 6.0])
        
        # Caractéristiques de l'alerte
        severity_map = {'DEBUG': 0.2, 'INFO': 0.4, 'WARNING': 0.6, 'HIGH': 0.8, 'CRITICAL': 1.0}
        severity = severity_map.get(alert_data.get('level', 'WARNING'), 0.6)
        features.append(severity)
        
        # Fréquence des alertes similaires
        pattern_signature = self._generate_pattern_signature(alert_data)
        recent_count = self._count_recent_patterns(pattern_signature, hours=24)
        features.append(min(recent_count / 100.0, 1.0))  # Normalisation
        
        # Longueur du message
        message_length = len(alert_data.get('message', ''))
        features.append(min(message_length / 1000.0, 1.0))
        
        # Source de l'alerte
        source_hash = hash(alert_data.get('source', '')) % 1000
        features.append(source_hash / 1000.0)
        
        # Métadonnées
        metadata_count = len(alert_data.get('metadata', {}))
        features.append(min(metadata_count / 20.0, 1.0))
        
        return np.array(features)
    
    def _generate_pattern_signature(self, alert_data: Dict[str, Any]) -> str:
        """Génère une signature unique pour un pattern d'alerte."""
        components = [
            alert_data.get('level', ''),
            alert_data.get('source', ''),
            alert_data.get('message', '')[:100]  # Premiers 100 caractères
        ]
        signature_string = '|'.join(components)
        return hashlib.md5(signature_string.encode()).hexdigest()[:16]
    
    def _count_recent_patterns(self, pattern_signature: str, hours: int = 24) -> int:
        """Compte les occurrences récentes d'un pattern."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        count = 0
        
        with self.cache_lock:
            for entry in self.feature_history:
                if entry['timestamp'] > cutoff_time and entry['pattern'] == pattern_signature:
                    count += 1
        
        return count
    
    def predict_escalation_delay(self, alert_data: Dict[str, Any]) -> Tuple[int, float]:
        """Prédit le délai optimal d'escalade avec score de confiance."""
        features = self.extract_features(alert_data)
        pattern_signature = self._generate_pattern_signature(alert_data)
        
        # Recherche de patterns similaires
        with self.cache_lock:
            if pattern_signature in self.pattern_cache:
                cached_pattern = self.pattern_cache[pattern_signature]
                if (datetime.now() - cached_pattern.last_updated).seconds < 3600:
                    return cached_pattern.suggested_delay, cached_pattern.confidence_score
        
        # Calcul basé sur les caractéristiques
        base_delay = 60  # 60 minutes par défaut
        
        # Ajustement selon la sévérité
        severity = features[2]
        if severity > 0.8:  # CRITICAL
            base_delay = 5
        elif severity > 0.6:  # HIGH
            base_delay = 15
        elif severity > 0.4:  # WARNING
            base_delay = 60
        else:  # INFO/DEBUG
            base_delay = 240
        
        # Ajustement selon la fréquence
        frequency = features[3]
        if frequency > 0.5:  # Très fréquent
            base_delay = int(base_delay * 0.5)  # Escalade plus rapide
        elif frequency < 0.1:  # Rare
            base_delay = int(base_delay * 1.5)  # Escalade plus lente
        
        # Ajustement selon l'heure
        hour_factor = features[0]
        if 0.75 <= hour_factor <= 0.95:  # Heures de bureau (18h-23h)
            base_delay = int(base_delay * 1.2)
        elif hour_factor <= 0.25:  # Nuit (0h-6h)
            base_delay = int(base_delay * 0.8)
        
        # Score de confiance basé sur l'historique
        confidence = min(len(self.feature_history) / 1000.0, 1.0)
        confidence = max(confidence, 0.3)  # Confiance minimale
        
        # Mise en cache du pattern
        pattern = AlertPattern(
            pattern_id=pattern_signature,
            pattern_signature=pattern_signature,
            frequency_score=frequency,
            severity_trend=severity,
            escalation_probability=0.5,  # Valeur par défaut
            suggested_delay=base_delay,
            confidence_score=confidence,
            last_updated=datetime.now()
        )
        
        with self.cache_lock:
            self.pattern_cache[pattern_signature] = pattern
        
        return base_delay, confidence
    
    def record_escalation_outcome(self, alert_data: Dict[str, Any], 
                                actual_escalation_time: int, was_necessary: bool):
        """Enregistre le résultat d'une escalade pour l'apprentissage."""
        features = self.extract_features(alert_data)
        pattern_signature = self._generate_pattern_signature(alert_data)
        
        outcome_data = {
            'timestamp': datetime.now(),
            'pattern': pattern_signature,
            'features': features.tolist(),
            'escalation_time': actual_escalation_time,
            'was_necessary': was_necessary,
            'effectiveness_score': 1.0 if was_necessary else 0.0
        }
        
        with self.cache_lock:
            self.feature_history.append(outcome_data)
            self.escalation_history.append(outcome_data)

class EscalationScheduler:
    """Planificateur d'escalades avec support asynchrone."""
    
    def __init__(self):
        self.scheduled_escalations = {}
        self.scheduler_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.running = True
        
        # Démarrage du thread de surveillance
        self.monitor_thread = threading.Thread(target=self._monitor_escalations, daemon=True)
        self.monitor_thread.start()
    
    def schedule_escalation(self, event: EscalationEvent, callback: Callable):
        """Programme une escalade."""
        with self.scheduler_lock:
            self.scheduled_escalations[event.event_id] = {
                'event': event,
                'callback': callback,
                'scheduled_time': event.next_escalation_time
            }
        
        logger.info(f"Escalade programmée: {event.event_id} pour {event.next_escalation_time}")
    
    def cancel_escalation(self, event_id: str) -> bool:
        """Annule une escalade programmée."""
        with self.scheduler_lock:
            if event_id in self.scheduled_escalations:
                del self.scheduled_escalations[event_id]
                logger.info(f"Escalade annulée: {event_id}")
                return True
        return False
    
    def _monitor_escalations(self):
        """Thread de surveillance des escalades programmées."""
        while self.running:
            try:
                current_time = datetime.now()
                escalations_to_trigger = []
                
                with self.scheduler_lock:
                    for event_id, escalation_data in list(self.scheduled_escalations.items()):
                        if current_time >= escalation_data['scheduled_time']:
                            escalations_to_trigger.append((event_id, escalation_data))
                            del self.scheduled_escalations[event_id]
                
                # Déclencher les escalades
                for event_id, escalation_data in escalations_to_trigger:
                    self.executor.submit(
                        self._execute_escalation,
                        escalation_data['event'],
                        escalation_data['callback']
                    )
                
                time.sleep(10)  # Vérification toutes les 10 secondes
                
            except Exception as e:
                logger.error(f"Erreur monitoring escalades: {e}")
                time.sleep(30)
    
    def _execute_escalation(self, event: EscalationEvent, callback: Callable):
        """Exécute une escalade."""
        try:
            logger.info(f"Exécution escalade: {event.event_id}")
            callback(event)
            event.completed = True
        except Exception as e:
            logger.error(f"Erreur exécution escalade {event.event_id}: {e}")
    
    def get_scheduled_count(self) -> int:
        """Retourne le nombre d'escalades programmées."""
        with self.scheduler_lock:
            return len(self.scheduled_escalations)
    
    def cleanup(self):
        """Nettoie les ressources."""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.executor.shutdown(wait=True)

class EscalationEngine:
    """
    Moteur d'escalade intelligent ultra-avancé.
    
    Fonctionnalités:
    - Machine Learning pour optimisation des délais
    - Analyse des patterns d'alertes
    - Escalade adaptative selon l'historique
    - Support multi-tenant avec règles personnalisées
    - Monitoring et métriques en temps réel
    - Système de feedback pour amélioration continue
    """
    
    def __init__(self, database_path: str = None):
        """Initialise le moteur d'escalade."""
        self.database_path = database_path or ":memory:"
        self.ml_predictor = MLEscalationPredictor()
        self.scheduler = EscalationScheduler()
        
        # Base de données pour la persistance
        self._init_database()
        
        # Configuration
        self.escalation_rules = {}
        self.active_escalations = {}
        
        # Métriques
        self.metrics = {
            'escalations_triggered': 0,
            'escalations_completed': 0,
            'escalations_cancelled': 0,
            'ml_predictions': 0,
            'pattern_matches': 0,
            'rule_violations': 0
        }
        self.metrics_lock = threading.RLock()
        
        # Chargement des règles par défaut
        self._load_default_rules()
        
        logger.info("EscalationEngine initialisé avec succès")
    
    def _init_database(self):
        """Initialise la base de données SQLite."""
        self.db_connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.db_lock = threading.RLock()
        
        with self.db_lock:
            cursor = self.db_connection.cursor()
            
            # Table des événements d'escalade
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS escalation_events (
                    event_id TEXT PRIMARY KEY,
                    alert_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    escalation_level INTEGER NOT NULL,
                    escalation_action TEXT NOT NULL,
                    trigger_time TEXT NOT NULL,
                    next_escalation_time TEXT,
                    metadata TEXT,
                    completed INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table des patterns d'alertes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_signature TEXT NOT NULL,
                    frequency_score REAL,
                    severity_trend REAL,
                    escalation_probability REAL,
                    suggested_delay INTEGER,
                    confidence_score REAL,
                    last_updated TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index pour optimisation
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tenant_alert ON escalation_events(tenant_id, alert_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_signature ON alert_patterns(pattern_signature)")
            
            self.db_connection.commit()
    
    def _load_default_rules(self):
        """Charge les règles d'escalade par défaut."""
        
        # Règle par défaut pour les alertes WARNING
        default_warning_rule = EscalationRule(
            rule_id="default_warning",
            tenant_id="*",  # Applicable à tous les tenants
            alert_pattern="WARNING",
            initial_delay_minutes=60,
            escalation_intervals=[60, 120, 240],  # 1h, 2h, 4h
            max_escalations=3,
            escalation_actions=[
                EscalationAction.NOTIFY_ADDITIONAL_CHANNELS,
                EscalationAction.ESCALATE_TO_MANAGER,
                EscalationAction.TRIGGER_INCIDENT
            ],
            conditions={
                "min_frequency": 3,  # Au moins 3 alertes similaires
                "time_window_hours": 24,
                "business_hours_only": False
            },
            enabled=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Règle pour les alertes critiques
        critical_rule = EscalationRule(
            rule_id="critical_immediate",
            tenant_id="*",
            alert_pattern="CRITICAL",
            initial_delay_minutes=5,
            escalation_intervals=[5, 10, 15],  # 5min, 10min, 15min
            max_escalations=3,
            escalation_actions=[
                EscalationAction.NOTIFY_ADDITIONAL_CHANNELS,
                EscalationAction.ESCALATE_TO_MANAGER,
                EscalationAction.TRIGGER_INCIDENT,
                EscalationAction.AUTO_REMEDIATION
            ],
            conditions={
                "min_frequency": 1,
                "time_window_hours": 1,
                "business_hours_only": False
            },
            enabled=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Règle pour les alertes à haute fréquence
        high_frequency_rule = EscalationRule(
            rule_id="high_frequency_pattern",
            tenant_id="*",
            alert_pattern=".*",  # Toutes les alertes
            initial_delay_minutes=30,
            escalation_intervals=[30, 60],  # 30min, 1h
            max_escalations=2,
            escalation_actions=[
                EscalationAction.INCREASE_FREQUENCY,
                EscalationAction.AUTO_REMEDIATION
            ],
            conditions={
                "min_frequency": 10,  # 10+ alertes similaires
                "time_window_hours": 2,
                "business_hours_only": False
            },
            enabled=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.escalation_rules = {
            "default_warning": default_warning_rule,
            "critical_immediate": critical_rule,
            "high_frequency_pattern": high_frequency_rule
        }
    
    def evaluate_escalation(self, alert_data: Dict[str, Any]) -> Optional[EscalationEvent]:
        """Évalue si une alerte doit déclencher une escalade."""
        
        alert_id = alert_data.get('alert_id')
        tenant_id = alert_data.get('tenant_id')
        level = alert_data.get('level', 'WARNING')
        
        if not alert_id or not tenant_id:
            return None
        
        # Recherche de règles applicables
        applicable_rules = self._find_applicable_rules(tenant_id, level, alert_data)
        
        if not applicable_rules:
            logger.debug(f"Aucune règle d'escalade applicable pour {alert_id}")
            return None
        
        # Sélection de la meilleure règle
        best_rule = self._select_best_rule(applicable_rules, alert_data)
        
        # Prédiction ML du délai optimal
        predicted_delay, confidence = self.ml_predictor.predict_escalation_delay(alert_data)
        self._increment_metric('ml_predictions')
        
        # Ajustement du délai selon la règle et la prédiction ML
        if confidence > 0.7:
            # Haute confiance dans la prédiction ML
            escalation_delay = predicted_delay
        else:
            # Utilisation de la règle avec ajustement ML
            rule_delay = best_rule.initial_delay_minutes
            escalation_delay = int((rule_delay + predicted_delay) / 2)
        
        # Création de l'événement d'escalade
        event_id = self._generate_event_id(alert_id, tenant_id)
        next_escalation_time = datetime.now() + timedelta(minutes=escalation_delay)
        
        escalation_event = EscalationEvent(
            event_id=event_id,
            alert_id=alert_id,
            tenant_id=tenant_id,
            escalation_level=EscalationLevel.LEVEL_1,
            escalation_action=best_rule.escalation_actions[0],
            trigger_time=datetime.now(),
            next_escalation_time=next_escalation_time,
            metadata={
                'rule_id': best_rule.rule_id,
                'predicted_delay': predicted_delay,
                'ml_confidence': confidence,
                'original_alert': alert_data
            }
        )
        
        # Sauvegarde en base
        self._save_escalation_event(escalation_event)
        
        # Programmation de l'escalade
        self.scheduler.schedule_escalation(escalation_event, self._execute_escalation_action)
        
        # Mise à jour des métriques
        self._increment_metric('escalations_triggered')
        
        logger.info(f"Escalade programmée: {event_id} dans {escalation_delay} minutes")
        
        return escalation_event
    
    def _find_applicable_rules(self, tenant_id: str, level: str, 
                             alert_data: Dict[str, Any]) -> List[EscalationRule]:
        """Trouve les règles d'escalade applicables."""
        applicable_rules = []
        
        for rule in self.escalation_rules.values():
            if not rule.enabled:
                continue
            
            # Vérification du tenant
            if rule.tenant_id != "*" and rule.tenant_id != tenant_id:
                continue
            
            # Vérification du pattern d'alerte
            import re
            if not re.search(rule.alert_pattern, level, re.IGNORECASE):
                continue
            
            # Vérification des conditions
            if self._check_rule_conditions(rule, alert_data):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _check_rule_conditions(self, rule: EscalationRule, alert_data: Dict[str, Any]) -> bool:
        """Vérifie si les conditions d'une règle sont remplies."""
        
        conditions = rule.conditions
        
        # Vérification de la fréquence minimale
        min_frequency = conditions.get('min_frequency', 1)
        if min_frequency > 1:
            pattern_signature = self.ml_predictor._generate_pattern_signature(alert_data)
            time_window = conditions.get('time_window_hours', 24)
            recent_count = self.ml_predictor._count_recent_patterns(pattern_signature, time_window)
            
            if recent_count < min_frequency:
                return False
        
        # Vérification des heures de bureau
        business_hours_only = conditions.get('business_hours_only', False)
        if business_hours_only:
            current_hour = datetime.now().hour
            if current_hour < 9 or current_hour > 17:  # En dehors de 9h-17h
                return False
        
        return True
    
    def _select_best_rule(self, rules: List[EscalationRule], alert_data: Dict[str, Any]) -> EscalationRule:
        """Sélectionne la meilleure règle parmi les applicables."""
        
        if len(rules) == 1:
            return rules[0]
        
        # Scoring des règles
        rule_scores = []
        
        for rule in rules:
            score = 0
            
            # Préférence pour les règles spécifiques au tenant
            if rule.tenant_id != "*":
                score += 10
            
            # Préférence pour les patterns plus spécifiques
            if rule.alert_pattern != ".*":
                score += 5
            
            # Préférence pour les règles récemment mises à jour
            days_since_update = (datetime.now() - rule.updated_at).days
            score += max(0, 5 - days_since_update)
            
            rule_scores.append((rule, score))
        
        # Tri par score décroissant
        rule_scores.sort(key=lambda x: x[1], reverse=True)
        
        return rule_scores[0][0]
    
    def _execute_escalation_action(self, event: EscalationEvent):
        """Exécute une action d'escalade."""
        
        try:
            action = event.escalation_action
            metadata = event.metadata
            
            logger.info(f"Exécution action d'escalade: {action.value} pour {event.alert_id}")
            
            if action == EscalationAction.NOTIFY_ADDITIONAL_CHANNELS:
                self._notify_additional_channels(event)
            elif action == EscalationAction.INCREASE_FREQUENCY:
                self._increase_notification_frequency(event)
            elif action == EscalationAction.ESCALATE_TO_MANAGER:
                self._escalate_to_manager(event)
            elif action == EscalationAction.TRIGGER_INCIDENT:
                self._trigger_incident(event)
            elif action == EscalationAction.AUTO_REMEDIATION:
                self._attempt_auto_remediation(event)
            
            # Vérification de la nécessité d'une escalade supplémentaire
            self._check_further_escalation(event)
            
            self._increment_metric('escalations_completed')
            
        except Exception as e:
            logger.error(f"Erreur exécution escalade {event.event_id}: {e}")
    
    def _notify_additional_channels(self, event: EscalationEvent):
        """Notifie des canaux supplémentaires."""
        # Implémentation spécifique à votre système de notification
        logger.info(f"Notification canaux supplémentaires pour {event.alert_id}")
    
    def _increase_notification_frequency(self, event: EscalationEvent):
        """Augmente la fréquence des notifications."""
        logger.info(f"Augmentation fréquence notifications pour {event.alert_id}")
    
    def _escalate_to_manager(self, event: EscalationEvent):
        """Escalade vers le manager."""
        logger.info(f"Escalade vers manager pour {event.alert_id}")
    
    def _trigger_incident(self, event: EscalationEvent):
        """Déclenche un incident."""
        logger.info(f"Déclenchement incident pour {event.alert_id}")
    
    def _attempt_auto_remediation(self, event: EscalationEvent):
        """Tente une remédiation automatique."""
        logger.info(f"Tentative remédiation automatique pour {event.alert_id}")
    
    def _check_further_escalation(self, event: EscalationEvent):
        """Vérifie si une escalade supplémentaire est nécessaire."""
        
        # Récupération de la règle originale
        rule_id = event.metadata.get('rule_id')
        if not rule_id or rule_id not in self.escalation_rules:
            return
        
        rule = self.escalation_rules[rule_id]
        current_level = event.escalation_level.value
        
        # Vérification si d'autres escalades sont possibles
        if current_level < rule.max_escalations and current_level < len(rule.escalation_intervals):
            
            next_level = EscalationLevel(current_level + 1)
            next_action_index = min(current_level, len(rule.escalation_actions) - 1)
            next_action = rule.escalation_actions[next_action_index]
            next_delay = rule.escalation_intervals[current_level - 1]
            
            # Création du prochain événement d'escalade
            next_event = EscalationEvent(
                event_id=f"{event.event_id}_level_{next_level.value}",
                alert_id=event.alert_id,
                tenant_id=event.tenant_id,
                escalation_level=next_level,
                escalation_action=next_action,
                trigger_time=datetime.now(),
                next_escalation_time=datetime.now() + timedelta(minutes=next_delay),
                metadata=event.metadata.copy()
            )
            
            # Programmation de la prochaine escalade
            self.scheduler.schedule_escalation(next_event, self._execute_escalation_action)
            self._save_escalation_event(next_event)
            
            logger.info(f"Prochaine escalade programmée: {next_event.event_id}")
    
    def _generate_event_id(self, alert_id: str, tenant_id: str) -> str:
        """Génère un ID unique pour l'événement d'escalade."""
        timestamp = int(time.time() * 1000)
        data = f"{alert_id}:{tenant_id}:{timestamp}"
        hash_obj = hashlib.md5(data.encode())
        return f"esc_{hash_obj.hexdigest()[:12]}"
    
    def _save_escalation_event(self, event: EscalationEvent):
        """Sauvegarde un événement d'escalade en base."""
        with self.db_lock:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO escalation_events 
                (event_id, alert_id, tenant_id, escalation_level, escalation_action,
                 trigger_time, next_escalation_time, metadata, completed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.alert_id,
                event.tenant_id,
                event.escalation_level.value,
                event.escalation_action.value,
                event.trigger_time.isoformat(),
                event.next_escalation_time.isoformat() if event.next_escalation_time else None,
                json.dumps(event.metadata),
                1 if event.completed else 0
            ))
            self.db_connection.commit()
    
    def cancel_escalation(self, alert_id: str, tenant_id: str) -> bool:
        """Annule toutes les escalades pour une alerte."""
        
        # Annulation dans le scheduler
        cancelled_count = 0
        events_to_cancel = []
        
        with self.db_lock:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT event_id FROM escalation_events 
                WHERE alert_id = ? AND tenant_id = ? AND completed = 0
            """, (alert_id, tenant_id))
            
            for row in cursor.fetchall():
                events_to_cancel.append(row[0])
        
        # Annulation des événements programmés
        for event_id in events_to_cancel:
            if self.scheduler.cancel_escalation(event_id):
                cancelled_count += 1
        
        # Mise à jour en base
        if cancelled_count > 0:
            with self.db_lock:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    UPDATE escalation_events 
                    SET completed = 1 
                    WHERE alert_id = ? AND tenant_id = ? AND completed = 0
                """, (alert_id, tenant_id))
                self.db_connection.commit()
            
            self._increment_metric('escalations_cancelled')
            logger.info(f"Escalades annulées pour {alert_id}: {cancelled_count}")
        
        return cancelled_count > 0
    
    def _increment_metric(self, metric_name: str):
        """Incrémente une métrique."""
        with self.metrics_lock:
            self.metrics[metric_name] = self.metrics.get(metric_name, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du moteur."""
        with self.metrics_lock:
            metrics = self.metrics.copy()
            metrics['active_escalations'] = len(self.active_escalations)
            metrics['scheduled_escalations'] = self.scheduler.get_scheduled_count()
            metrics['ml_patterns_cached'] = len(self.ml_predictor.pattern_cache)
            return metrics
    
    def cleanup(self):
        """Nettoie les ressources."""
        self.scheduler.cleanup()
        if hasattr(self, 'db_connection'):
            self.db_connection.close()
        logger.info("EscalationEngine nettoyé avec succès")

# Factory function
def create_escalation_engine(database_path: str = None) -> EscalationEngine:
    """Factory function pour créer un moteur d'escalade."""
    return EscalationEngine(database_path)
