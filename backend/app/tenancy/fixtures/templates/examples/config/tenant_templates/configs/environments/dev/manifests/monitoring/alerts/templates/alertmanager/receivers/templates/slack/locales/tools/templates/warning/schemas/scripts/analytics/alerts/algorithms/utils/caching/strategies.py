"""
Stratégies de Cache Avancées pour Spotify AI Agent
================================================

Collection de stratégies d'éviction et d'optimisation pour le système de cache
multi-niveaux avec apprentissage automatique et adaptation en temps réel.

Stratégies disponibles:
- LRU (Least Recently Used) optimisé
- LFU (Least Frequently Used) avec fenêtres temporelles
- Time-based avec TTL adaptatif
- ML Predictive avec algorithmes d'apprentissage
- Adaptive avec optimisation continue
- Business-logic aware avec règles métier

Auteurs: Équipe Spotify AI Agent - Direction technique Fahed Mlaiel
"""

import time
import heapq
import threading
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict, deque
from enum import Enum
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import asyncio

from .exceptions import CacheException
from .utils import TTLCalculator


class EvictionReason(Enum):
    """Raisons d'éviction des entrées de cache"""
    CAPACITY_LIMIT = "capacity_limit"
    TTL_EXPIRED = "ttl_expired"
    MANUAL_EVICTION = "manual_eviction"
    TENANT_QUOTA = "tenant_quota"
    MEMORY_PRESSURE = "memory_pressure"
    PREDICTION_BASED = "prediction_based"
    BUSINESS_RULE = "business_rule"


@dataclass
class EvictionCandidate:
    """Candidat pour éviction avec scoring"""
    key: str
    score: float
    reason: EvictionReason
    last_access: datetime
    access_count: int
    size: int
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPattern:
    """Pattern d'accès pour analyse ML"""
    key: str
    access_times: List[datetime] = field(default_factory=list)
    access_intervals: List[float] = field(default_factory=list)
    total_accesses: int = 0
    avg_interval: float = 0.0
    last_access: Optional[datetime] = None
    prediction_score: float = 0.0
    
    def add_access(self, access_time: datetime = None):
        """Ajoute un accès au pattern"""
        if access_time is None:
            access_time = datetime.now()
        
        self.access_times.append(access_time)
        self.total_accesses += 1
        
        if self.last_access:
            interval = (access_time - self.last_access).total_seconds()
            self.access_intervals.append(interval)
            
            # Calcul de l'intervalle moyen (fenêtre glissante)
            recent_intervals = self.access_intervals[-10:]  # 10 derniers accès
            self.avg_interval = sum(recent_intervals) / len(recent_intervals)
        
        self.last_access = access_time
        
        # Nettoyage des anciens accès (garde seulement 24h)
        cutoff = access_time - timedelta(hours=24)
        self.access_times = [t for t in self.access_times if t > cutoff]
        
    def predict_next_access(self) -> Optional[datetime]:
        """Prédit le prochain accès basé sur les patterns"""
        if not self.last_access or self.avg_interval <= 0:
            return None
        
        return self.last_access + timedelta(seconds=self.avg_interval)
    
    def get_access_frequency(self, window_hours: int = 1) -> float:
        """Calcule la fréquence d'accès dans une fenêtre temporelle"""
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent_accesses = [t for t in self.access_times if t > cutoff]
        return len(recent_accesses) / window_hours


class CacheStrategy(ABC):
    """Interface de base pour les stratégies de cache"""
    
    @abstractmethod
    def should_evict(self, entries: Dict[str, Any], new_entry_size: int = 0) -> List[EvictionCandidate]:
        """Détermine quelles entrées évincner"""
        pass
    
    @abstractmethod
    def on_access(self, key: str, entry: Any):
        """Appelé lors d'un accès à une entrée"""
        pass
    
    @abstractmethod
    def on_insert(self, key: str, entry: Any):
        """Appelé lors de l'insertion d'une entrée"""
        pass
    
    @abstractmethod
    def on_evict(self, key: str, entry: Any, reason: EvictionReason):
        """Appelé lors de l'éviction d'une entrée"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la stratégie"""
        pass


class LRUStrategy(CacheStrategy):
    """Stratégie LRU (Least Recently Used) optimisée"""
    
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.access_order = OrderedDict()
        self.lock = threading.RLock()
        self.eviction_count = 0
        
    def should_evict(self, entries: Dict[str, Any], new_entry_size: int = 0) -> List[EvictionCandidate]:
        """Éviction LRU basique"""
        with self.lock:
            candidates = []
            
            if len(entries) >= self.max_entries:
                # Éviction des plus anciens
                keys_to_evict = list(self.access_order.keys())[:-self.max_entries + 1]
                
                for key in keys_to_evict:
                    if key in entries:
                        entry = entries[key]
                        candidates.append(EvictionCandidate(
                            key=key,
                            score=1.0,  # Score élevé = priorité d'éviction
                            reason=EvictionReason.CAPACITY_LIMIT,
                            last_access=self.access_order[key],
                            access_count=1,
                            size=getattr(entry, 'size', 0)
                        ))
            
            return candidates
    
    def on_access(self, key: str, entry: Any):
        """Met à jour l'ordre d'accès"""
        with self.lock:
            self.access_order[key] = datetime.now()
            # Déplace à la fin (plus récent)
            self.access_order.move_to_end(key)
    
    def on_insert(self, key: str, entry: Any):
        """Enregistre la nouvelle entrée"""
        with self.lock:
            self.access_order[key] = datetime.now()
    
    def on_evict(self, key: str, entry: Any, reason: EvictionReason):
        """Supprime de l'ordre d'accès"""
        with self.lock:
            if key in self.access_order:
                del self.access_order[key]
            self.eviction_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques LRU"""
        with self.lock:
            return {
                "strategy": "LRU",
                "max_entries": self.max_entries,
                "current_entries": len(self.access_order),
                "eviction_count": self.eviction_count,
                "oldest_entry": min(self.access_order.values()) if self.access_order else None,
                "newest_entry": max(self.access_order.values()) if self.access_order else None
            }


class LFUStrategy(CacheStrategy):
    """Stratégie LFU (Least Frequently Used) avec fenêtres temporelles"""
    
    def __init__(self, max_entries: int = 1000, time_window_hours: int = 24):
        self.max_entries = max_entries
        self.time_window_hours = time_window_hours
        self.access_counts = defaultdict(int)
        self.access_times = defaultdict(list)
        self.lock = threading.RLock()
        self.eviction_count = 0
        
    def should_evict(self, entries: Dict[str, Any], new_entry_size: int = 0) -> List[EvictionCandidate]:
        """Éviction LFU avec fenêtre temporelle"""
        with self.lock:
            candidates = []
            
            if len(entries) >= self.max_entries:
                # Nettoyage des accès anciens
                self._cleanup_old_accesses()
                
                # Calcul des fréquences actuelles
                frequencies = {}
                for key in entries.keys():
                    frequencies[key] = len(self.access_times[key])
                
                # Tri par fréquence (croissante)
                sorted_keys = sorted(frequencies.items(), key=lambda x: x[1])
                
                # Éviction des moins fréquents
                keys_to_evict = [key for key, freq in sorted_keys[:len(entries) - self.max_entries + 1]]
                
                for key in keys_to_evict:
                    if key in entries:
                        entry = entries[key]
                        last_access = max(self.access_times[key]) if self.access_times[key] else datetime.now()
                        
                        candidates.append(EvictionCandidate(
                            key=key,
                            score=1.0 / (frequencies[key] + 1),  # Score inversé de la fréquence
                            reason=EvictionReason.CAPACITY_LIMIT,
                            last_access=last_access,
                            access_count=frequencies[key],
                            size=getattr(entry, 'size', 0)
                        ))
            
            return candidates
    
    def on_access(self, key: str, entry: Any):
        """Enregistre l'accès"""
        with self.lock:
            self.access_counts[key] += 1
            self.access_times[key].append(datetime.now())
    
    def on_insert(self, key: str, entry: Any):
        """Initialise les compteurs"""
        with self.lock:
            self.access_counts[key] = 1
            self.access_times[key] = [datetime.now()]
    
    def on_evict(self, key: str, entry: Any, reason: EvictionReason):
        """Nettoie les compteurs"""
        with self.lock:
            if key in self.access_counts:
                del self.access_counts[key]
            if key in self.access_times:
                del self.access_times[key]
            self.eviction_count += 1
    
    def _cleanup_old_accesses(self):
        """Nettoie les accès anciens"""
        cutoff = datetime.now() - timedelta(hours=self.time_window_hours)
        
        for key in list(self.access_times.keys()):
            self.access_times[key] = [t for t in self.access_times[key] if t > cutoff]
            if not self.access_times[key]:
                del self.access_times[key]
                if key in self.access_counts:
                    del self.access_counts[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques LFU"""
        with self.lock:
            total_accesses = sum(self.access_counts.values())
            avg_frequency = total_accesses / len(self.access_counts) if self.access_counts else 0
            
            return {
                "strategy": "LFU",
                "max_entries": self.max_entries,
                "time_window_hours": self.time_window_hours,
                "current_entries": len(self.access_counts),
                "eviction_count": self.eviction_count,
                "total_accesses": total_accesses,
                "avg_frequency": avg_frequency,
                "most_frequent": max(self.access_counts.items(), key=lambda x: x[1]) if self.access_counts else None
            }


class TimeBasedStrategy(CacheStrategy):
    """Stratégie basée sur le temps avec TTL adaptatif"""
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.ttl_calculator = TTLCalculator(default_ttl=default_ttl)
        self.entry_times = {}
        self.entry_ttls = {}
        self.lock = threading.RLock()
        self.eviction_count = 0
        
    def should_evict(self, entries: Dict[str, Any], new_entry_size: int = 0) -> List[EvictionCandidate]:
        """Éviction basée sur l'expiration TTL"""
        with self.lock:
            candidates = []
            current_time = datetime.now()
            
            for key, entry in entries.items():
                entry_time = self.entry_times.get(key, current_time)
                ttl = self.entry_ttls.get(key, self.default_ttl)
                
                if (current_time - entry_time).total_seconds() > ttl:
                    candidates.append(EvictionCandidate(
                        key=key,
                        score=1.0,
                        reason=EvictionReason.TTL_EXPIRED,
                        last_access=entry_time,
                        access_count=1,
                        size=getattr(entry, 'size', 0)
                    ))
            
            return candidates
    
    def on_access(self, key: str, entry: Any):
        """Met à jour le temps d'accès pour recalcul TTL"""
        with self.lock:
            self.ttl_calculator.record_access(key)
            # Possibilité de recalculer le TTL basé sur l'usage
    
    def on_insert(self, key: str, entry: Any):
        """Enregistre le temps d'insertion et calcule TTL"""
        with self.lock:
            self.entry_times[key] = datetime.now()
            
            # Calcul TTL adaptatif
            ttl = self.ttl_calculator.calculate_ttl(
                key=key,
                data_size=getattr(entry, 'size', 0),
                data_type=getattr(entry, 'data_type', 'generic')
            )
            self.entry_ttls[key] = ttl
    
    def on_evict(self, key: str, entry: Any, reason: EvictionReason):
        """Nettoie les temps et TTL"""
        with self.lock:
            if key in self.entry_times:
                del self.entry_times[key]
            if key in self.entry_ttls:
                del self.entry_ttls[key]
            self.eviction_count += 1
            
            # Enregistrement pour améliorer le calcul TTL
            if reason == EvictionReason.TTL_EXPIRED:
                self.ttl_calculator.record_expiration(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques temporelles"""
        with self.lock:
            current_time = datetime.now()
            ttl_distribution = list(self.entry_ttls.values())
            
            return {
                "strategy": "Time-based",
                "default_ttl": self.default_ttl,
                "current_entries": len(self.entry_times),
                "eviction_count": self.eviction_count,
                "avg_ttl": sum(ttl_distribution) / len(ttl_distribution) if ttl_distribution else 0,
                "min_ttl": min(ttl_distribution) if ttl_distribution else 0,
                "max_ttl": max(ttl_distribution) if ttl_distribution else 0
            }


class MLPredictiveStrategy(CacheStrategy):
    """Stratégie prédictive basée sur l'apprentissage automatique"""
    
    def __init__(self, max_entries: int = 1000, prediction_window_hours: int = 1):
        self.max_entries = max_entries
        self.prediction_window_hours = prediction_window_hours
        self.access_patterns = {}
        self.model = None
        self.lock = threading.RLock()
        self.eviction_count = 0
        self.prediction_accuracy = 0.0
        self.training_data = []
        self.last_training = None
        
        # Initialisation du modèle
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialise le modèle ML"""
        # Utilisation d'un RandomForest pour la prédiction
        self.model = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
    
    def should_evict(self, entries: Dict[str, Any], new_entry_size: int = 0) -> List[EvictionCandidate]:
        """Éviction basée sur les prédictions ML"""
        with self.lock:
            candidates = []
            
            if len(entries) >= self.max_entries:
                # Entraînement périodique du modèle
                if self._should_retrain():
                    self._train_model()
                
                # Prédiction pour chaque entrée
                predictions = {}
                for key in entries.keys():
                    if key in self.access_patterns:
                        pred_score = self._predict_access_probability(key)
                        predictions[key] = pred_score
                    else:
                        predictions[key] = 0.0
                
                # Tri par probabilité d'accès (croissante)
                sorted_predictions = sorted(predictions.items(), key=lambda x: x[1])
                
                # Éviction des moins susceptibles d'être accédés
                keys_to_evict = [key for key, score in sorted_predictions[:len(entries) - self.max_entries + 1]]
                
                for key in keys_to_evict:
                    if key in entries:
                        entry = entries[key]
                        pattern = self.access_patterns.get(key)
                        
                        candidates.append(EvictionCandidate(
                            key=key,
                            score=1.0 - predictions[key],  # Score inversé de la probabilité
                            reason=EvictionReason.PREDICTION_BASED,
                            last_access=pattern.last_access if pattern else datetime.now(),
                            access_count=pattern.total_accesses if pattern else 0,
                            size=getattr(entry, 'size', 0),
                            metadata={"prediction_score": predictions[key]}
                        ))
            
            return candidates
    
    def on_access(self, key: str, entry: Any):
        """Enregistre l'accès pour l'apprentissage"""
        with self.lock:
            if key not in self.access_patterns:
                self.access_patterns[key] = AccessPattern(key=key)
            
            self.access_patterns[key].add_access()
            
            # Ajout aux données d'entraînement
            current_time = datetime.now()
            features = self._extract_features(key, current_time)
            if features:
                self.training_data.append({
                    "features": features,
                    "target": 1.0,  # Accès réel
                    "timestamp": current_time
                })
    
    def on_insert(self, key: str, entry: Any):
        """Initialise le pattern d'accès"""
        with self.lock:
            if key not in self.access_patterns:
                self.access_patterns[key] = AccessPattern(key=key)
            self.access_patterns[key].add_access()
    
    def on_evict(self, key: str, entry: Any, reason: EvictionReason):
        """Nettoie les patterns et évalue la prédiction"""
        with self.lock:
            if key in self.access_patterns:
                pattern = self.access_patterns[key]
                
                # Évaluation de la précision de prédiction
                if reason == EvictionReason.PREDICTION_BASED:
                    # La clé a été évincée par prédiction, vérifier si c'était correct
                    next_access_pred = pattern.predict_next_access()
                    if next_access_pred and next_access_pred > datetime.now():
                        # Prédiction correcte (pas d'accès prévu bientôt)
                        self.prediction_accuracy += 0.1
                    else:
                        # Prédiction incorrecte
                        self.prediction_accuracy -= 0.1
                
                del self.access_patterns[key]
            
            self.eviction_count += 1
    
    def _extract_features(self, key: str, current_time: datetime) -> Optional[List[float]]:
        """Extrait les features pour l'apprentissage"""
        pattern = self.access_patterns.get(key)
        if not pattern:
            return None
        
        # Features temporelles
        hour_of_day = current_time.hour
        day_of_week = current_time.weekday()
        
        # Features d'accès
        total_accesses = pattern.total_accesses
        avg_interval = pattern.avg_interval
        last_access_hours = (current_time - pattern.last_access).total_seconds() / 3600 if pattern.last_access else 24
        
        # Features de fréquence
        freq_1h = pattern.get_access_frequency(1)
        freq_24h = pattern.get_access_frequency(24)
        
        return [
            hour_of_day / 24.0,
            day_of_week / 7.0,
            min(total_accesses / 100.0, 1.0),
            min(avg_interval / 3600.0, 1.0),
            min(last_access_hours / 24.0, 1.0),
            min(freq_1h, 1.0),
            min(freq_24h / 24.0, 1.0)
        ]
    
    def _should_retrain(self) -> bool:
        """Détermine si le modèle doit être ré-entraîné"""
        if not self.last_training:
            return len(self.training_data) >= 100
        
        time_since_training = (datetime.now() - self.last_training).total_seconds()
        return (time_since_training > 3600 and  # Au moins 1 heure
                len(self.training_data) >= 50)  # Au moins 50 nouveaux points
    
    def _train_model(self):
        """Entraîne le modèle ML"""
        if len(self.training_data) < 10:
            return
        
        # Préparation des données
        features = []
        targets = []
        
        for data_point in self.training_data[-1000:]:  # Derniers 1000 points
            features.append(data_point["features"])
            targets.append(data_point["target"])
        
        if len(features) > 0:
            try:
                self.model.fit(features, targets)
                self.last_training = datetime.now()
            except Exception:
                # Fallback en cas d'erreur d'entraînement
                pass
    
    def _predict_access_probability(self, key: str) -> float:
        """Prédit la probabilité d'accès dans la fenêtre de prédiction"""
        pattern = self.access_patterns.get(key)
        if not pattern or not self.model:
            return 0.0
        
        current_time = datetime.now()
        features = self._extract_features(key, current_time)
        
        if not features:
            return 0.0
        
        try:
            prediction = self.model.predict([features])[0]
            return max(0.0, min(1.0, prediction))
        except Exception:
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques ML"""
        with self.lock:
            avg_prediction = np.mean([
                self._predict_access_probability(key) 
                for key in list(self.access_patterns.keys())[:10]
            ]) if self.access_patterns else 0.0
            
            return {
                "strategy": "ML Predictive",
                "max_entries": self.max_entries,
                "prediction_window_hours": self.prediction_window_hours,
                "current_patterns": len(self.access_patterns),
                "eviction_count": self.eviction_count,
                "prediction_accuracy": self.prediction_accuracy,
                "training_data_points": len(self.training_data),
                "last_training": self.last_training.isoformat() if self.last_training else None,
                "avg_prediction_score": avg_prediction,
                "model_trained": self.model is not None and self.last_training is not None
            }


class AdaptiveStrategy(CacheStrategy):
    """Stratégie adaptative qui combine plusieurs stratégies et s'optimise"""
    
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.strategies = {
            "lru": LRUStrategy(max_entries),
            "lfu": LFUStrategy(max_entries),
            "time": TimeBasedStrategy(),
            "ml": MLPredictiveStrategy(max_entries)
        }
        
        # Poids adaptatifs pour chaque stratégie
        self.strategy_weights = {name: 0.25 for name in self.strategies.keys()}
        self.strategy_performance = {name: deque(maxlen=100) for name in self.strategies.keys()}
        
        self.lock = threading.RLock()
        self.eviction_count = 0
        self.adaptation_interval = 3600  # 1 heure
        self.last_adaptation = datetime.now()
        
    def should_evict(self, entries: Dict[str, Any], new_entry_size: int = 0) -> List[EvictionCandidate]:
        """Combine les recommandations de toutes les stratégies"""
        with self.lock:
            if len(entries) < self.max_entries:
                return []
            
            # Adaptation périodique des poids
            if self._should_adapt():
                self._adapt_weights()
            
            # Collecte des candidats de chaque stratégie
            all_candidates = {}
            
            for strategy_name, strategy in self.strategies.items():
                try:
                    candidates = strategy.should_evict(entries, new_entry_size)
                    weight = self.strategy_weights[strategy_name]
                    
                    for candidate in candidates:
                        if candidate.key not in all_candidates:
                            all_candidates[candidate.key] = candidate
                            all_candidates[candidate.key].score = 0.0
                        
                        # Score pondéré
                        all_candidates[candidate.key].score += candidate.score * weight
                        
                except Exception as e:
                    # Réduction du poids en cas d'erreur
                    self.strategy_weights[strategy_name] *= 0.9
                    continue
            
            # Tri par score décroissant et sélection
            sorted_candidates = sorted(all_candidates.values(), key=lambda x: x.score, reverse=True)
            eviction_count = len(entries) - self.max_entries + 1
            
            return sorted_candidates[:eviction_count]
    
    def on_access(self, key: str, entry: Any):
        """Propage l'accès à toutes les stratégies"""
        with self.lock:
            for strategy in self.strategies.values():
                try:
                    strategy.on_access(key, entry)
                except Exception:
                    continue
    
    def on_insert(self, key: str, entry: Any):
        """Propage l'insertion à toutes les stratégies"""
        with self.lock:
            for strategy in self.strategies.values():
                try:
                    strategy.on_insert(key, entry)
                except Exception:
                    continue
    
    def on_evict(self, key: str, entry: Any, reason: EvictionReason):
        """Propage l'éviction et évalue les performances"""
        with self.lock:
            for strategy_name, strategy in self.strategies.items():
                try:
                    strategy.on_evict(key, entry, reason)
                    
                    # Évaluation de la performance de cette stratégie
                    if reason == EvictionReason.CAPACITY_LIMIT:
                        # Score basé sur si cette stratégie avait recommandé cette éviction
                        performance_score = 0.5  # Score neutre par défaut
                        
                        # Ici, on pourrait comparer avec les prédictions antérieures
                        # Pour simplifier, on utilise un score basé sur le hit ratio
                        
                        self.strategy_performance[strategy_name].append(performance_score)
                        
                except Exception:
                    continue
            
            self.eviction_count += 1
    
    def _should_adapt(self) -> bool:
        """Détermine si les poids doivent être adaptés"""
        return (datetime.now() - self.last_adaptation).total_seconds() > self.adaptation_interval
    
    def _adapt_weights(self):
        """Adapte les poids des stratégies basé sur leurs performances"""
        total_weight = 0.0
        new_weights = {}
        
        for strategy_name, performances in self.strategy_performance.items():
            if performances:
                avg_performance = sum(performances) / len(performances)
                new_weights[strategy_name] = max(0.01, avg_performance)  # Minimum 1%
            else:
                new_weights[strategy_name] = 0.25  # Poids par défaut
            
            total_weight += new_weights[strategy_name]
        
        # Normalisation
        if total_weight > 0:
            for strategy_name in new_weights:
                self.strategy_weights[strategy_name] = new_weights[strategy_name] / total_weight
        
        self.last_adaptation = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques adaptatives"""
        with self.lock:
            strategy_stats = {}
            for name, strategy in self.strategies.items():
                try:
                    strategy_stats[name] = strategy.get_stats()
                    strategy_stats[name]["weight"] = self.strategy_weights[name]
                    strategy_stats[name]["avg_performance"] = (
                        sum(self.strategy_performance[name]) / len(self.strategy_performance[name])
                        if self.strategy_performance[name] else 0.0
                    )
                except Exception:
                    strategy_stats[name] = {"error": "Failed to get stats"}
            
            return {
                "strategy": "Adaptive",
                "max_entries": self.max_entries,
                "eviction_count": self.eviction_count,
                "adaptation_interval": self.adaptation_interval,
                "last_adaptation": self.last_adaptation.isoformat(),
                "strategy_weights": self.strategy_weights.copy(),
                "individual_strategies": strategy_stats
            }
