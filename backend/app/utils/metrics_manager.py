"""
Système centralisé de gestion des métriques Prometheus.
Évite les doublons et gère les métriques de façon thread-safe.
"""

import threading
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry

class MetricsManager:
    """Gestionnaire centralisé des métriques pour éviter les doublons."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._metrics: Dict[str, Any] = {}
            self._registry = CollectorRegistry()
            self._initialized = True
    
    def get_counter(self, name: str, description: str, labels: Optional[list] = None) -> Counter:
        """Obtient ou crée un Counter avec protection contre les doublons."""
        key = f"counter_{name}"
        if key not in self._metrics:
            try:
                self._metrics[key] = Counter(
                    name, description, 
                    labels or [], 
                    registry=self._registry
                )
            except ValueError as e:
                # Si la métrique existe déjà, essayons de la récupérer
                if "already registered" in str(e):
                    # Utiliser un nom légèrement différent
                    import uuid
                    unique_name = f"{name}_{uuid.uuid4().hex[:8]}"
                    self._metrics[key] = Counter(
                        unique_name, description,
                        labels or [],
                        registry=self._registry
                    )
                else:
                    raise
        return self._metrics[key]
    
    def get_or_create_counter(self, name: str, description: str, labels: Optional[list] = None) -> Counter:
        """Alias pour get_counter pour compatibilité."""
        return self.get_counter(name, description, labels)
    
    def get_histogram(self, name: str, description: str, labels: Optional[list] = None, buckets: Optional[list] = None) -> Histogram:
        """Obtient ou crée un Histogram avec protection contre les doublons."""
        key = f"histogram_{name}"
        if key not in self._metrics:
            try:
                self._metrics[key] = Histogram(
                    name, description,
                    labels or [],
                    buckets=buckets,
                    registry=self._registry
                )
            except ValueError as e:
                if "already registered" in str(e):
                    import uuid
                    unique_name = f"{name}_{uuid.uuid4().hex[:8]}"
                    self._metrics[key] = Histogram(
                        unique_name, description,
                        labels or [],
                        buckets=buckets,
                        registry=self._registry
                    )
                else:
                    raise
        return self._metrics[key]
    
    def get_gauge(self, name: str, description: str, labels: Optional[list] = None) -> Gauge:
        """Obtient ou crée un Gauge avec protection contre les doublons."""
        key = f"gauge_{name}"
        if key not in self._metrics:
            try:
                self._metrics[key] = Gauge(
                    name, description,
                    labels or [],
                    registry=self._registry
                )
            except ValueError as e:
                if "already registered" in str(e):
                    import uuid
                    unique_name = f"{name}_{uuid.uuid4().hex[:8]}"
                    self._metrics[key] = Gauge(
                        unique_name, description,
                        labels or [],
                        registry=self._registry
                    )
                else:
                    raise
        return self._metrics[key]
    
    def clear_all(self):
        """Nettoie toutes les métriques (utile pour les tests)."""
        self._metrics.clear()
        self._registry = CollectorRegistry()

# Instance globale
metrics_manager = MetricsManager()

# Fonctions utilitaires pour faciliter l'usage
def get_counter(name: str, description: str, labels: Optional[list] = None) -> Counter:
    """Fonction utilitaire pour obtenir un Counter."""
    return metrics_manager.get_counter(name, description, labels)

def get_histogram(name: str, description: str, labels: Optional[list] = None, buckets: Optional[list] = None) -> Histogram:
    """Fonction utilitaire pour obtenir un Histogram."""
    return metrics_manager.get_histogram(name, description, labels, buckets)

def get_gauge(name: str, description: str, labels: Optional[list] = None) -> Gauge:
    """Fonction utilitaire pour obtenir un Gauge."""
    return metrics_manager.get_gauge(name, description, labels)

def clear_metrics():
    """Nettoie toutes les métriques."""
    metrics_manager.clear_all()
