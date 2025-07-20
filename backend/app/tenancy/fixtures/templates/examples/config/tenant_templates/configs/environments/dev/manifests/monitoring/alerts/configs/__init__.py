"""
Module de Configuration des Alertes de Monitoring Avancé
========================================================

Ce module fournit une infrastructure complète de monitoring et d'alertes
pour l'architecture multi-tenant du Spotify AI Agent.

Fonctionnalités:
- Système d'alertes temps réel
- Métriques personnalisées par tenant
- Intégration avec Prometheus/Grafana
- Alertes intelligentes avec ML
- Escalade automatique des incidents
- Corrélation d'événements
- Tableau de bord unifié

Architecture:
- AlertManager: Gestionnaire central des alertes
- MetricsCollector: Collecte des métriques personnalisées
- RuleEngine: Moteur de règles d'alertes
- NotificationDispatcher: Distribution des notifications
- CorrelationEngine: Corrélation d'événements
- EscalationManager: Escalade automatique

Auteur: Fahed Mlaiel
Rôles: Lead Dev + Architecte IA, Développeur Backend Senior, 
       Ingénieur ML, DBA & Data Engineer, Spécialiste Sécurité Backend,
       Architecte Microservices
"""

from .alert_manager import AlertManager
from .metrics_collector import MetricsCollector
from .rule_engine import RuleEngine
from .notification_dispatcher import NotificationDispatcher
from .correlation_engine import CorrelationEngine
from .escalation_manager import EscalationManager
from .config_loader import ConfigLoader
from .dashboard_generator import DashboardGenerator

__all__ = [
    'AlertManager',
    'MetricsCollector', 
    'RuleEngine',
    'NotificationDispatcher',
    'CorrelationEngine',
    'EscalationManager',
    'ConfigLoader',
    'DashboardGenerator'
]

__version__ = "2.0.0"
__author__ = "Fahed Mlaiel"
