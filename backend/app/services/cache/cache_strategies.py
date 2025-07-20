from abc import ABC, abstractmethod
from typing import Any, Optional
import time

class BaseCacheStrategy(ABC):
    """Interface de stratégie de cache avancée, extensible pour IA, analytics, etc"""
    @abstractmethod
    def get(self, key: str) -> Any:
        pass
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        pass

class LRUCacheStrategy(BaseCacheStrategy):
    """Stratégie LRU (Least Recently Used) avec TTL optionnel."""
    def __init__(self, capacity: int = 1000):
        self.cache = {}
        self.order = []
        self.ttl = {}
        self.capacity = capacity
    def get(self, key):
        now = time.time()
        if key in self.cache:
            if key in self.ttl and self.ttl[key] < now:
                del self.cache[key]
                del self.ttl[key]
                self.order.remove(key)
                return None
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    def set(self, key, value, ttl=None):
        now = time.time()
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
            if oldest in self.ttl:
                del self.ttl[oldest]
        self.cache[key] = value
        self.order.append(key)
        if ttl:
            self.ttl[key] = now + ttl

class LFUCacheStrategy(BaseCacheStrategy):
    """Stratégie LFU (Least Frequently Used) avec TTL optionnel."""
    def __init__(self, capacity: int = 1000):
        self.cache = {}
        self.freq = {}
        self.ttl = {}
        self.capacity = capacity
    def get(self, key):
        now = time.time()
        if key in self.cache:
            if key in self.ttl and self.ttl[key] < now:
                del self.cache[key]
                del self.freq[key]
                del self.ttl[key]
                return None
            self.freq[key] += 1
            return self.cache[key]
        return None
    def set(self, key, value, ttl=None):
        now = time.time()
        if key in self.cache:
            self.freq[key] += 1
        elif len(self.cache) >= self.capacity:
            least = min(self.freq, key=self.freq.get)
            del self.cache[least]
            del self.freq[least]
            if least in self.ttl:
                del self.ttl[least]
        self.cache[key] = value
        self.freq[key] = 1
        if ttl:
            self.ttl[key] = now + ttl

class AdaptiveMLCacheStrategy(BaseCacheStrategy):
    """
    Stratégie de cache adaptative basée ML (exploitable pour IA, analytics, scoring, etc.).
    Peut utiliser un modèle ML pour prédire la popularité ou le TTL optimal.
    """
    def __init__(self, model=None):
        self.cache = {}
        self.ttl = {}
        self.model = model  # Peut être un modèle ML pour prédire la popularité
    def get(self, key):
        now = time.time()
        if key in self.cache:
            if key in self.ttl and self.ttl[key] < now:
                del self.cache[key]
                del self.ttl[key]
                return None
            return self.cache[key]
        return None
    def set(self, key, value, ttl=None):
        now = time.time()
        # Exemple: utiliser le modèle pour ajuster le TTL
        if self.model:
            predicted_ttl = self.model.predict_ttl(key, value)
            ttl = predicted_ttl if predicted_ttl else ttl
        self.cache[key] = value
        if ttl:
            self.ttl[key] = now + ttl
