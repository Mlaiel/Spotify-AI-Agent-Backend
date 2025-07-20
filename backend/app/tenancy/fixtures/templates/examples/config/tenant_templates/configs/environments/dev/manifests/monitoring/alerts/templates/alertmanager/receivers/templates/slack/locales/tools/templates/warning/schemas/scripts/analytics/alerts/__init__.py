"""
Spotify AI Agent - Module d'Analytics et d'Alertes
Module avancé de surveillance et d'analyse des alertes pour l'écosystème Spotify AI Agent.

Ce module fournit:
- Analytics avancées des patterns d'alertes
- Détection d'anomalies dans les métriques
- Corrélation intelligente entre événements
- Prédiction proactive des incidents
- Optimisation automatique des seuils d'alerte
"""

from .alert_analytics_engine import AlertAnalyticsEngine
from .anomaly_detector import AnomalyDetector
from .correlation_analyzer import CorrelationAnalyzer
from .predictive_analyzer import PredictiveAnalyzer
from .threshold_optimizer import ThresholdOptimizer
from .reporting_engine import ReportingEngine

__version__ = "2.0.0"
__author__ = "Spotify AI Agent Backend Team"

__all__ = [
    'AlertAnalyticsEngine',
    'AnomalyDetector', 
    'CorrelationAnalyzer',
    'PredictiveAnalyzer',
    'ThresholdOptimizer',
    'ReportingEngine'
]

# Configuration par défaut pour l'analytics
DEFAULT_ANALYTICS_CONFIG = {
    "anomaly_detection": {
        "enabled": True,
        "sensitivity": 0.8,
        "window_size": "1h",
        "algorithms": ["isolation_forest", "statistical", "ml_based"]
    },
    "correlation": {
        "enabled": True,
        "max_correlation_window": "30m",
        "min_confidence": 0.75
    },
    "prediction": {
        "enabled": True,
        "horizon": "4h",
        "model_refresh_interval": "1d"
    },
    "threshold_optimization": {
        "enabled": True,
        "auto_adjust": True,
        "learning_rate": 0.1
    }
}
