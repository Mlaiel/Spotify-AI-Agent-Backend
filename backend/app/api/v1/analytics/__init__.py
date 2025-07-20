"""
Module Analytics pour Spotify AI Agent

Ce package expose tous les services analytiques avancés pour artistes Spotify :
- Recommandation intelligente (ML, IA)
- Analyse de tendances et prédiction
- Suivi de performance et reporting
- Intelligence marché et veille concurrentielle
- Insights audience avancés
- Prédiction de revenus

Auteur : Lead Dev, Architecte IA, ML, Data Engineer, Sécurité
"""

from .recommendation_engine import RecommendationEngine
from .trend_analyzer import TrendAnalyzer
from .performance_tracker import PerformanceTracker
from .market_intelligence import MarketIntelligence
from .audience_insights import AudienceInsights
from .revenue_predictor import RevenuePredictor

__all__ = [
    "RecommendationEngine",
    "TrendAnalyzer",
    "PerformanceTracker",
    "MarketIntelligence",
    "AudienceInsights",
    "RevenuePredictor",
]
